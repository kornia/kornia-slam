//! SLAM runtime: orchestrates tracking, mapping, and state transitions.
//!
//! The runtime flow is kept in one file so it can be read from top to bottom
//! in the same order frames move through the system.

pub mod config;

pub use crate::loop_closure::LoopClosureEvent;
#[cfg(test)]
use crate::loop_closure::pose_graph_reference_correction;
pub use config::{LoopClosingConfig, SlamConfig};

use std::sync::{Arc, Mutex};

use crate::Frame;
use crate::initialization::{
    ImuInitConfig, ImuInitNotReadyReason, ImuInitResult, ImuInitializer, InertialInitOutcome,
    TwoViewInitConfig, try_initialize_two_view,
};
use crate::loop_closure::{InertialPgoContext, LoopCloser, LoopClosingContext, LoopClosingOutcome};
use crate::map::{
    InertialAlignment, InertialAlignmentError, Keyframe, KeyframeJob, LocalMapping, Map, MapPoint,
};
use crate::place_recognition::Vocabulary;
use crate::pose_conversion::rotation_from_to;
use crate::stereo::unproject_stereo;
use crate::tracking::optical_flow::{
    FlowSurvivor, KltTracker, MapKeypointMatch, TrackSet, snap_unique,
};
use crate::tracking::pose_estimation::MapProjectionEstimator;
use crate::tracking::{
    KeyframePolicy, SystemMode, SystemState, TrackingLossRecoveryPolicy, TrackingResult,
    TrackingStatus,
};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

/// Top-level SLAM system: orchestrates tracking, mapping, and state transitions.
pub struct SlamSystem {
    // Camera model
    camera: PinholeCamera,
    // Primary pose estimator
    estimator: MapProjectionEstimator,
    // Boostrap pose estimator
    two_view_init_config: TwoViewInitConfig,
    // Keyframe insertion policy
    keyframe_policy: KeyframePolicy,
    // Recently-lost grace period policy
    tracking_loss_recovery: TrackingLossRecoveryPolicy,
    // mThDepth (metres): back-project close stereo points at each keyframe when set
    stereo_close_depth: Option<f64>,
    // Emit per-frame diagnostic logs (skip/reject reasons, growth counters)
    debug: bool,
    // Buffered debug messages produced during the most recent process_frame call;
    // drained by the caller (TUI panel or stderr).
    debug_messages: Vec<String>,
    // Map object
    map: Arc<Mutex<Map>>,
    // Serializes compound map publication and short local-BA snapshot/merge phases.
    map_publication_gate: Option<Arc<Mutex<()>>>,
    // IMU states
    imu_calib: ImuCalib,
    imu_bias: ImuBias,
    // Camera-to-body extrinsic T_BC (X_body = T_BC * X_cam). IMU deltas live in
    // the body frame, so every place that mixes them with camera poses must go
    // through this; None disables the inertial path entirely.
    imu_t_bc: Option<Pose3d>,
    pending_imu: Vec<ImuMeasurement>,
    gravity_world: Vec3F64,
    bootstrap_timestamp_sec: Option<f64>,
    last_keyframe_timestamp_sec: Option<f64>,
    // Owns the inertial-initialization window, readiness gate and the
    // VIBA0/VIBA1/VIBA2 schedule; the system only applies its results.
    inertial_init: ImuInitializer,
    local_mapping: LocalMapping,
    klt_tracker: KltTracker,
    track_set: TrackSet,
    loop_closer: LoopCloser,
    loop_closure_events: Vec<LoopClosureEvent>,
    // System state
    state: SystemState,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ImuInitApplyError {
    #[error("initialization gravity vector is zero or non-finite")]
    InvalidGravity,
    #[error(transparent)]
    Alignment(#[from] InertialAlignmentError),
}

impl SlamSystem {
    /// Creates a new SLAM system with identity pose.
    pub fn new(camera: PinholeCamera, config: SlamConfig) -> Self {
        let map = Arc::new(Mutex::new(Map::new()));
        let local_mapping =
            LocalMapping::new(config.local_mapping, Arc::clone(&map), camera.clone());
        let map_publication_gate = local_mapping.publication_gate();
        Self {
            camera,
            estimator: MapProjectionEstimator::new(config.map_projection),
            two_view_init_config: config.two_view_init,
            keyframe_policy: config.keyframe_policy,
            tracking_loss_recovery: config.tracking_loss_recovery,
            stereo_close_depth: config.stereo_close_depth_m,
            debug: config.debug,
            debug_messages: Vec::new(),
            map,
            map_publication_gate,
            local_mapping,
            state: SystemState::new(),
            imu_calib: ImuCalib {
                gyro_noise: 1.6968e-4,
                accel_noise: 2.0e-3,
                gyro_bias_noise: 1.9393e-5,
                accel_bias_noise: 3.0e-3,
            },
            imu_bias: ImuBias::default(),
            imu_t_bc: None,
            pending_imu: Vec::new(),
            gravity_world: Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE),
            bootstrap_timestamp_sec: None,
            last_keyframe_timestamp_sec: None,
            inertial_init: ImuInitializer::new(ImuInitConfig::default()),
            klt_tracker: KltTracker::default(),
            track_set: TrackSet::new(),
            loop_closer: LoopCloser::new(config.pgo),
            loop_closure_events: Vec::new(),
        }
    }

    /// The local-mapping job description for the current system state.
    fn keyframe_job(&self) -> KeyframeJob {
        KeyframeJob {
            imu_initialized: self.state.imu_initialized,
            imu_t_bc: self.imu_t_bc,
            gravity_world: self.gravity_world,
        }
    }

    /// Atomically validates and applies an inertial initialization result: the
    /// map takes the scale, gravity-aligning rotation, velocities and bias;
    /// the system then adopts the last aligned keyframe's state.
    fn apply_inertial_initialization(
        &mut self,
        init: ImuInitResult,
    ) -> Result<(), ImuInitApplyError> {
        let gravity_norm = init.gravity_world.length();
        if !gravity_norm.is_finite() || gravity_norm <= 1e-9 {
            return Err(ImuInitApplyError::InvalidGravity);
        }
        let rotation = rotation_from_to(
            init.gravity_world / gravity_norm,
            Vec3F64::new(0.0, 1.0, 0.0),
        );

        let map = Arc::clone(&self.map);
        let mut map = map.lock().unwrap();
        let last_keyframe_idx = map.apply_inertial_alignment(InertialAlignment {
            scale: init.scale,
            rotation,
            keyframe_velocities: init.keyframe_velocities,
            bias: init.bias,
        })?;

        let last_keyframe = map
            .get_keyframe(last_keyframe_idx)
            .expect("last keyframe existence was checked before mutating the map");
        self.state.velocity_world = last_keyframe.velocity_world;
        self.state.pose_world_to_cam = last_keyframe.frame.pose_world_to_cam;
        self.state.velocity = None;
        self.state.imu_initialized = true;
        self.gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        self.imu_bias = init.bias;
        Ok(())
    }

    /// Enables appearance-based loop detection with a bag-of-words vocabulary.
    /// Without it, keyframes are not indexed and no loop candidates are emitted.
    pub fn set_vocabulary(&mut self, vocabulary: Vocabulary) {
        self.loop_closer.set_vocabulary(vocabulary);
    }

    pub fn drain_loop_closure_events(&mut self) -> Vec<LoopClosureEvent> {
        std::mem::take(&mut self.loop_closure_events)
    }

    /// Enables the inertial path by providing the camera-to-body extrinsic
    /// `T_BC` (`X_body = T_BC * X_cam`). Without it, IMU samples are ignored.
    pub fn set_imu_extrinsics(&mut self, t_bc: Pose3d) {
        self.imu_t_bc = Some(t_bc);
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    pub fn process_frame(
        &mut self,
        mut frame: Frame,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        timestamp_sec: f64,
        imu_samples: Vec<ImuMeasurement>,
    ) -> TrackingResult {
        // Local-BA snapshots, merges, and their correction messages can only
        // cross this boundary between complete tracking frames.
        let publication_gate = self.map_publication_gate.clone();
        let _publication = publication_gate
            .as_ref()
            .map(|gate| gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        self.apply_local_mapping_results();
        // Fill the per-frame undistortion cache once; tracking, BA gathering,
        // growth, and fuse all read from it.
        frame.ensure_undistorted(&self.camera);
        self.pending_imu.extend(imu_samples);

        match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_step(frame, timestamp_sec),
            SystemMode::ImuInit => {
                self.inertial_init_step(frame, previous_image, current_image, timestamp_sec)
            }
            SystemMode::Tracking => {
                self.tracking_step(frame, previous_image, current_image, timestamp_sec)
            }
        }
    }

    /// Runs `f` against the live map points, holding the map lock only for the
    /// duration of the call. Avoids cloning the whole point list (descriptors
    /// included) for read-only consumers such as viz logging and summaries.
    pub fn with_map_points<R>(&self, f: impl FnOnce(&[MapPoint]) -> R) -> R {
        f(self.map.lock().unwrap().map_points())
    }

    /// Returns the index of the current reference keyframe, if tracking has one.
    pub fn current_keyframe_idx(&self) -> Option<usize> {
        self.state.current_keyframe_idx.and_then(|ki| {
            self.map
                .lock()
                .unwrap()
                .get_keyframe(ki)
                .map(|kf| kf.frame.idx)
        })
    }

    /// Returns the number of active (non-culled) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map.lock().unwrap().num_active_map_points()
    }

    /// Drain any debug messages accumulated since the last call.
    pub fn drain_debug_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_messages)
    }

    /// Toggle whether the system buffers per-frame debug messages.
    pub fn set_debug(&mut self, on: bool) {
        self.debug = on;
        if !on {
            self.debug_messages.clear();
        }
    }

    fn apply_local_mapping_results(&mut self) {
        let Some(reference_idx) = self.state.current_keyframe_idx else {
            // Still drain results so a completed worker cannot build a result backlog.
            let _ = self.local_mapping.drain_results();
            return;
        };

        for result in self.local_mapping.drain_results() {
            let Some(correction) = result
                .keyframe_corrections
                .iter()
                .find(|correction| correction.kf_idx == reference_idx)
            else {
                continue;
            };

            self.state.pose_world_to_cam = apply_reference_pose_correction(
                self.state.pose_world_to_cam,
                correction.pose_before,
                correction.pose_after,
            );
            self.state.velocity_world = correction.velocity_world;
            self.imu_bias = correction.imu_bias;
        }
    }

    fn dbg(&mut self, msg: String) {
        if self.debug {
            self.debug_messages.push(msg);
        }
    }

    fn bootstrap_step(&mut self, curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stereo frames carry metric per-keypoint depth, so we can build a
        // metric map from a single keyframe (ORB-SLAM3's StereoInitialization)
        // instead of waiting for two-view parallax.
        if curr_frame.is_stereo() {
            return self.bootstrap_stereo(curr_frame, timestamp_sec);
        }
        self.bootstrap_mono(curr_frame, timestamp_sec)
    }

    /// Single-frame metric initialization from stereo depth.
    fn bootstrap_stereo(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Build the new map in the current odometry frame (identity at start,
        // or the recovery pose after a tracking loss).
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        const MIN_STEREO_POINTS: usize = 50;
        let cam_points = unproject_stereo(&curr_frame, &self.camera);
        if cam_points.len() < MIN_STEREO_POINTS {
            self.dbg(format!(
                "[bootstrap_stereo] frame={} skip: only {} stereo points (need >= {})",
                curr_frame.idx,
                cam_points.len(),
                MIN_STEREO_POINTS,
            ));
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let pose_inv = curr_frame.pose_world_to_cam.inverse();
        let mut keyframe = Keyframe::from_frame(curr_frame);
        let curr_idx = keyframe.frame.idx;

        let mut points = Vec::with_capacity(cam_points.len());
        for (desc_idx, p_cam) in &cam_points {
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = keyframe.frame.features.descriptors[*desc_idx];
            let color = keyframe
                .frame
                .keypoint_colors
                .get(*desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        let added = self
            .map
            .lock()
            .unwrap()
            .add_triangulated_points(None, &mut keyframe, &points);
        self.map.lock().unwrap().upsert_keyframe(keyframe);

        self.dbg(format!(
            "[bootstrap_stereo] frame={curr_idx} metric map created with {added} points",
        ));

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.velocity = None;
        // The map is already metric (stereo baseline), but gravity, velocities,
        // and the gyro bias still need the inertial init before IMU prediction
        // can run; the solve there keeps scale fixed at 1.
        self.state.mode = if self.imu_t_bc.is_some() {
            self.inertial_init.begin_window(curr_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.prune_imu_before(timestamp_sec);

        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    fn bootstrap_mono(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        // Staleness guard (mirrors ORB-SLAM3's MonocularInitialization):
        // a frame with too few keypoints is neither a viable reference nor
        // a viable current frame. If we already had a reference, drop it
        // and wait for a feature-rich frame to start over.
        const MIN_KEYPOINTS_FOR_BOOTSTRAP: usize = 100;
        if curr_frame.features.keypoints_xy.len() <= MIN_KEYPOINTS_FOR_BOOTSTRAP {
            self.dbg(format!(
                "[bootstrap] frame={} skip: too few keypoints ({}, need > {})",
                curr_frame.idx,
                curr_frame.features.keypoints_xy.len(),
                MIN_KEYPOINTS_FOR_BOOTSTRAP,
            ));
            self.state.bootstrap_frame = None;
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let Some(prev_bootstrap_frame) = self.state.bootstrap_frame.take() else {
            self.dbg(format!(
                "[bootstrap] frame={} stored as reference (awaiting second frame)",
                curr_frame.idx,
            ));
            self.state.bootstrap_frame = Some(curr_frame);
            self.bootstrap_timestamp_sec = Some(timestamp_sec);
            // Samples before the reference frame can never enter an edge.
            self.prune_imu_before(timestamp_sec);
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        };

        let result = try_initialize_two_view(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &self.camera,
            &self.two_view_init_config,
        );

        let two_view_estimate = match result {
            Err(reason) => {
                self.dbg(format!(
                    "[bootstrap] frame={} (ref={}) reject: {}",
                    curr_frame.idx, prev_bootstrap_frame.idx, reason,
                ));
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            Ok(tv) => tv,
        };

        self.dbg(format!(
            "[bootstrap] frame={} accept: model={} triangulated={} inliers={}",
            curr_frame.idx,
            two_view_estimate.model_kind,
            two_view_estimate.points3d.len(),
            two_view_estimate.inliers,
        ));

        let estimated_pose = two_view_estimate.pose;
        let prev_pose_world_to_cam = curr_frame.pose_world_to_cam;
        self.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
        let prev_idx = prev_bootstrap_frame.idx;
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        self.build_initial_map(
            reference_kf,
            current_kf,
            &two_view_estimate.matches,
            &two_view_estimate.points3d,
            &two_view_estimate.inlier_indices,
            two_view_estimate.median_depth,
        );

        // Post-BA sanity gate (mirrors ORB-SLAM3's reset criteria in
        // CreateInitialMapMonocular). Discard the bootstrap if the resulting
        // map has too few valid points or a degenerate scale.
        const MIN_VALID_POINTS: usize = 50;
        let health = self.map.lock().unwrap().initial_map_health();
        if health.valid_in_both < MIN_VALID_POINTS || health.median_depth_older_kf <= 0.0 {
            self.dbg(format!(
                "[init_gate] reject: valid_in_both={} median_depth={:.3} (need >= {} and > 0)",
                health.valid_in_both, health.median_depth_older_kf, MIN_VALID_POINTS,
            ));
            self.map.lock().unwrap().clear_active();
            self.state.reset();
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        // BA inside build_initial_map may have refined KF1's pose; sync state
        // and recompute velocity from the post-BA pose.
        if let Some(kf) = self.map.lock().unwrap().get_keyframe(curr_idx) {
            self.state.pose_world_to_cam = kf.frame.pose_world_to_cam;
        }

        if let Some(prev_ts) = self.bootstrap_timestamp_sec {
            let (preint, raw_samples) = self.preintegrate_window(prev_ts, timestamp_sec);
            if preint.dt > 0.0 {
                self.map.lock().unwrap().add_imu_factor(
                    prev_idx,
                    curr_idx,
                    preint,
                    raw_samples,
                    prev_ts,
                    timestamp_sec,
                );
            }
            self.prune_imu_before(timestamp_sec);
        }

        self.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.state.pose_world_to_cam,
        ));

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        // Inertial init needs the camera-to-body extrinsic to relate IMU deltas
        // to camera poses; without it, run visual-only as before.
        self.state.mode = if self.imu_t_bc.is_some() {
            self.inertial_init.begin_window(curr_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);

        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    fn build_initial_map(
        &mut self,
        mut reference_kf: Keyframe,
        mut current_kf: Keyframe,
        matches: &[(usize, usize)],
        points3d: &[Vec3F64],
        inlier_indices: &[usize],
        median_depth: Option<f64>,
    ) -> usize {
        let depth_scale = median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
        let reference_pose_inv = reference_kf.frame.pose_world_to_cam.inverse();

        let mut triangulated = Vec::new();
        for (p_cam, &match_idx) in points3d.iter().zip(inlier_indices.iter()) {
            let Some(&(ref_desc_idx, curr_desc_idx)) = matches.get(match_idx) else {
                continue;
            };
            if ref_desc_idx >= reference_kf.map_point_by_desc_idx.len()
                || curr_desc_idx >= current_kf.map_point_by_desc_idx.len()
            {
                continue;
            }
            let descriptor = current_kf
                .frame
                .features
                .descriptors
                .get(curr_desc_idx)
                .copied()
                .or_else(|| {
                    reference_kf
                        .frame
                        .features
                        .descriptors
                        .get(ref_desc_idx)
                        .copied()
                });
            let Some(descriptor) = descriptor else {
                continue;
            };
            let color = current_kf
                .frame
                .keypoint_colors
                .get(curr_desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            let p_world = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
            triangulated.push((p_world, descriptor, color, ref_desc_idx, curr_desc_idx));
        }

        let added = self.map.lock().unwrap().add_triangulated_points(
            Some(&mut reference_kf),
            &mut current_kf,
            &triangulated,
        );

        let reference_kf_idx = reference_kf.frame.idx;
        let current_kf_idx = current_kf.frame.idx;
        self.map.lock().unwrap().upsert_keyframe(reference_kf);
        self.map.lock().unwrap().upsert_keyframe(current_kf);

        self.map.lock().unwrap().run_initial_ba(&self.camera);

        // Seed the place-recognition database with the two bootstrap keyframes so
        // a later revisit of the start can match them.
        self.register_place_recognition(reference_kf_idx);
        self.register_place_recognition(current_kf_idx);

        added
    }

    /// Preintegrates buffered IMU samples over `[t0, t1]` without consuming
    /// them: the same samples serve both per-frame pose prediction and the
    /// keyframe-to-keyframe edges. [`Self::prune_imu_before`] discards samples
    /// once no future window can need them.
    /// Preintegrates over `[t0, t1]` and also returns the raw samples used,
    /// so the caller can hand them to `Map::add_imu_factor` for later
    /// repropagation (see `PreintegratedImu::from_measurements` doc) — once
    /// this returns, `prune_imu_before` is free to drop them from
    /// `self.pending_imu`, since the edge now carries its own copy.
    fn preintegrate_window(&self, t0: f64, t1: f64) -> (PreintegratedImu, Vec<ImuMeasurement>) {
        let samples: Vec<ImuMeasurement> = self
            .pending_imu
            .iter()
            .filter(|m| m.timestamp >= t0 && m.timestamp <= t1)
            .copied()
            .collect();
        let pre =
            PreintegratedImu::from_measurements(self.imu_bias, self.imu_calib, &samples, t0, t1);
        (pre, samples)
    }

    /// Drops buffered IMU samples strictly older than `t` (typically the last
    /// keyframe timestamp: the next edge and all per-frame windows start there).
    fn prune_imu_before(&mut self, t: f64) {
        self.pending_imu.retain(|m| m.timestamp >= t);
    }

    /// Body-to-world pose `T_WB` for a world-to-camera pose, via
    /// `T_WB = T_WC ∘ T_CB`. Treats camera == body when no extrinsic is set.
    fn body_to_world(&self, pose_w2c: &Pose3d) -> Pose3d {
        let cam_to_world = pose_w2c.inverse();
        match &self.imu_t_bc {
            Some(t_bc) => cam_to_world.compose(&t_bc.inverse()),
            None => cam_to_world,
        }
    }

    /// Propagates the camera pose and body velocity through one preintegrated
    /// IMU window.
    fn predict_pose_imu(
        &self,
        pose_w2c: Pose3d,
        vel_world: Vec3F64,
        gravity_world: Vec3F64,
        preint: &PreintegratedImu,
    ) -> (Pose3d, Vec3F64) {
        let body_to_world = self.body_to_world(&pose_w2c);
        let (r_j, v_j, p_j) = preint.predict(
            &body_to_world.rotation,
            &vel_world,
            &body_to_world.translation,
            &gravity_world,
        );

        let pred_body_to_world = Pose3d::from_rt(r_j, p_j);
        let pred_cam_to_world = match &self.imu_t_bc {
            Some(t_bc) => pred_body_to_world.compose(t_bc),
            None => pred_body_to_world,
        };
        (pred_cam_to_world.inverse(), v_j)
    }

    fn inertial_init_step(
        &mut self,
        frame: Frame,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        timestamp_sec: f64,
    ) -> TrackingResult {
        let result = self.tracking_step(frame, previous_image, current_image, timestamp_sec);
        if result.status != TrackingStatus::KeyframeAccepted {
            return result;
        }

        // Drop the solve's map lock before applying its result with a new lock.
        let outcome = {
            let map = self.map.lock().unwrap();
            self.inertial_init.on_keyframe_uninitialized(
                &map,
                timestamp_sec,
                self.imu_t_bc,
                self.imu_bias,
            )
        };

        match outcome {
            InertialInitOutcome::NotDue => {}
            InertialInitOutcome::NotReady(not_ready) => {
                if not_ready.reason != ImuInitNotReadyReason::NoWindow {
                    self.dbg(not_ready.to_string());
                }
            }
            InertialInitOutcome::Attempted {
                stage,
                result: init,
            } => match init {
                Ok(init) => {
                    let label = stage.label();
                    let scale = init.scale;
                    let gravity = init.gravity_world;
                    let bg = init.bias.gyro;
                    if let Err(error) = self.apply_inertial_initialization(init) {
                        self.dbg(format!("[imu_init] {label} apply rejected: {error}"));
                        return result;
                    }
                    // Mirrors ORB-SLAM3: IMU is marked initialized (and
                    // tracking resumes) immediately after VIBA0 succeeds —
                    // VIBA1/VIBA2 refine bg/ba/scale/gravity further in the
                    // background (see try_insert_keyframe), they don't gate
                    // resuming tracking.
                    self.state.mode = SystemMode::Tracking;
                    self.state.imu_init_timestamp_sec = Some(timestamp_sec);
                    let job = self.keyframe_job();
                    if !self.local_mapping.submit(job) {
                        self.dbg("[local_mapping] worker is unavailable".into());
                    }
                    self.apply_local_mapping_results();
                    self.dbg(format!(
                        "[imu_init] {label} accepted: scale={scale:.4} gravity=({:.3},{:.3},{:.3}) \
                         gyro_bias=({:.4},{:.4},{:.4})",
                        gravity.x, gravity.y, gravity.z, bg.x, bg.y, bg.z
                    ));
                }
                Err(error) => {
                    self.dbg(format!("[imu_init] {} rejected: {error}", stage.label()));
                }
            },
        }

        result
    }

    fn tracking_step(
        &mut self,
        frame: Frame,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        timestamp_sec: f64,
    ) -> TrackingResult {
        let image_size = frame.image_size;
        let pose_before = self.state.pose_world_to_cam;
        let prev_timestamp = self.state.last_frame_timestamp_sec;

        // Local BA updates keyframe state asynchronously, so refresh cached IMU state.
        if let Some(kf_idx) = self.state.current_keyframe_idx
            && let Some(kf) = self.map.lock().unwrap().get_keyframe(kf_idx)
        {
            self.state.velocity_world = kf.velocity_world;
            self.imu_bias = kf.imu_bias;
        }

        let candidate_pose = if self.state.imu_initialized && prev_timestamp > 0.0 {
            let (preint, _) = self.preintegrate_window(prev_timestamp, timestamp_sec);
            if preint.dt > 0.0 {
                let (pred_pose, pred_vel) = self.predict_pose_imu(
                    pose_before,
                    self.state.velocity_world,
                    self.gravity_world,
                    &preint,
                );
                self.state.velocity_world = pred_vel; // propagate for next frame
                pred_pose
            } else {
                // IMU stalled, fall back to visual constant velocity
                self.state
                    .velocity
                    .map(|v| v.compose(&pose_before))
                    .unwrap_or(pose_before)
            }
        } else {
            self.state
                .velocity
                .map(|v| v.compose(&pose_before))
                .unwrap_or(pose_before)
        };

        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let search_scale = self.estimator.config().search_scale_for(currently_lost_for);

        let klt_survivors = if self.track_set.is_empty() {
            None
        } else {
            previous_image.and_then(|previous_image| {
                self.klt_tracker
                    .track(self.track_set.tracks(), previous_image, current_image)
                    .ok()
            })
        };
        let pre_seeded = klt_survivors
            .as_ref()
            .and_then(|survivors| {
                snap_unique(
                    &self.track_set,
                    survivors,
                    &frame.features.keypoints_xy,
                    3.0,
                )
                .ok()
            })
            .map(|matches| {
                matches
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                    .collect()
            });

        let result = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before,
            &self.map.lock().unwrap(),
            &self.camera,
            self.state.current_keyframe_idx,
            search_scale,
            pre_seeded,
        );

        let (mut status, matches, tracked_inliers, reject_reason) = match result {
            Ok(estimate) => {
                self.state.pose_world_to_cam = estimate.pose;

                // When IMU-initialized, velocity_world was already updated by IMU
                // preintegration (predict_pose_imu → pred_vel) before PnP ran; don't
                // overwrite it with a visual finite-difference, since at 30 fps the
                // inter-frame displacement is noise-dominated during low-translation
                // segments, which would collapse velocity to zero and permanently
                // freeze the IMU pose prediction.
                if !self.state.imu_initialized {
                    self.state.velocity = Some(Pose3d::between(&pose_before, &estimate.pose));
                }

                let track_matches: Vec<MapKeypointMatch> = estimate
                    .matches
                    .iter()
                    .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                        map_point_idx,
                        keypoint_idx,
                    })
                    .collect();
                if self
                    .track_set
                    .reconcile_from_matches(&track_matches, &frame.features.keypoints_xy)
                    .is_err()
                {
                    self.track_set = TrackSet::new();
                }

                (
                    TrackingStatus::Tracked,
                    estimate.matches,
                    estimate.inliers,
                    None,
                )
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);

                // Carry the predicted pose forward instead of freezing at
                // pose_before. state.velocity_world was already advanced by
                // predict_pose_imu above regardless of visual outcome, so
                // anchoring the next frame's prediction on a stale position
                // would desync position/rotation from velocity: every
                // subsequent frame's candidate pose would drift further from
                // reality, making the projection search miss again and
                // compounding a single bad frame into a full tracking loss.
                self.state.pose_world_to_cam = candidate_pose;
                (TrackingStatus::Skipped, Vec::new(), 0, Some(reason))
            }
        };
        if self.debug {
            let msg = match reject_reason {
                Some(reason) => format!("[track] frame={} reject: {:?}", frame.idx, reason),
                None => format!(
                    "[track] frame={} ok: matches={} inliers={}",
                    frame.idx,
                    matches.len(),
                    tracked_inliers,
                ),
            };
            self.debug_messages.push(msg);
        }

        if status == TrackingStatus::Tracked {
            // Visibility bookkeeping over the local map only (mirrors
            // ORB-SLAM3, which counts mnVisible on local-map points): full-map
            // scans here would grow with trajectory length.
            // Release this non-reentrant lock before keyframe insertion locks the map.
            {
                let mut map_guard = self.map.lock().unwrap();
                let current_kf = self
                    .state
                    .current_keyframe_idx
                    .and_then(|ki| map_guard.get_keyframe(ki));
                let local_indices = map_guard.build_local_map_point_indices(&matches, current_kf);
                let visible = map_guard.map_points_in_frustum(
                    &local_indices,
                    &self.camera,
                    &candidate_pose,
                    image_size,
                );
                map_guard.update_observation_counts(&visible, &matches);
            }

            if self.try_insert_keyframe(&frame, timestamp_sec, tracked_inliers, &matches) {
                status = TrackingStatus::KeyframeAccepted;
            }
        }

        if status == TrackingStatus::Skipped {
            let policy = &self.tracking_loss_recovery;

            let lost_since = *self.state.lost_since_sec.get_or_insert(timestamp_sec);
            let recently_lost_for = timestamp_sec - lost_since;

            let imu_confident = self.state.imu_initialized
                && self
                    .state
                    .imu_init_timestamp_sec
                    .is_some_and(|t0| timestamp_sec - t0 >= policy.min_imu_confidence_sec);
            let grace_period_sec = policy.grace_period_sec(imu_confident);
            let map_established =
                self.map.lock().unwrap().keyframes().len() > policy.min_keyframes_for_grace;

            if !map_established || recently_lost_for >= grace_period_sec {
                self.dbg(format!(
                    "[lost] frame={} giving up after {:.2}s (map_established={}): resetting",
                    frame.idx, recently_lost_for, map_established,
                ));
                self.track_set = TrackSet::new();
                self.state.reset();
                return self.bootstrap_step(frame, timestamp_sec);
            }
        } else {
            self.state.lost_since_sec = None;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        // Samples older than the last keyframe can't enter any future window
        // (the next edge and all per-frame predictions start at or after it).
        if let Some(kf_ts) = self.last_keyframe_timestamp_sec {
            self.prune_imu_before(kf_ts.min(timestamp_sec));
        }
        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        let n_ref_map_points = if let Some(ki) = self.state.current_keyframe_idx {
            let map = self.map.lock().unwrap();
            map.get_keyframe(ki)
                .map(|kf| kf.num_associated_points())
                .unwrap_or(0)
        } else {
            0
        };

        if !self.keyframe_policy.should_insert(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        // Guard: reference KF must exist before we can triangulate.
        if let Some(ki) = self.state.current_keyframe_idx {
            let map = self.map.lock().unwrap();
            if map.get_keyframe(ki).is_none() {
                return false;
            }
        } else {
            return false;
        }

        let mut curr_kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
            keypoints_undist: frame.keypoints_undist.clone(),
        });
        // Seed the new keyframe with the IMU-propagated velocity and current bias
        // estimate so that VI-BA starts from a reasonable linearisation point rather
        // than zero, which would produce huge residuals on the newest IMU edge.
        if self.state.imu_initialized {
            curr_kf.velocity_world = self.state.velocity_world;
            curr_kf.imu_bias = self.imu_bias;
        }
        for &(mp_idx, curr_idx) in matches {
            curr_kf.associate_map_point(curr_idx, mp_idx);
            self.map
                .lock()
                .unwrap()
                .register_observation(mp_idx, &curr_kf, curr_idx);
        }

        // Stereo densification: back-project this keyframe's unassociated
        // "close" stereo keypoints directly into metric map points. Mirrors
        // ORB-SLAM3's CreateNewKeyFrame, which seeds close points from stereo
        // and leaves far points to multi-view triangulation (the grow pass).
        if let Some(mthdepth) = self.stereo_close_depth
            && curr_kf.frame.is_stereo()
        {
            let n_close = self.map.lock().unwrap().add_close_stereo_points(
                &mut curr_kf,
                mthdepth,
                &self.camera,
            );
            self.dbg(format!(
                "[kf_stereo] frame={} close_points={}",
                frame.idx, n_close
            ));
        }

        // Triangulate new map points against the last MAX_COVIS_KFS keyframes,
        // not just the immediate predecessor. Mirrors ORB-SLAM3's
        // CreateNewMapPoints which uses the 30 best covisible KFs; we
        // approximate covisibility by recency until a covisibility graph is
        // available. The grow pass works against keyframes stored in the map
        // (addressed by frame index), so no keyframe clones are needed.
        const MAX_COVIS_KFS: usize = 10;
        let neighbor_kf_indices: Vec<usize> = self
            .map
            .lock()
            .unwrap()
            .keyframes()
            .iter()
            .rev()
            .take(MAX_COVIS_KFS)
            .map(|kf| kf.frame.idx)
            .collect();

        let imu_initialized = self.state.imu_initialized;
        let match_config = self.two_view_init_config.match_config;
        let triangulation_config = self.two_view_init_config.triangulation_config.clone();

        let mut total_grown = 0usize;
        for &nb_kf_idx in &neighbor_kf_indices {
            total_grown += self.map.lock().unwrap().grow_map_points_from_keyframe_pair(
                nb_kf_idx,
                &mut curr_kf,
                match_config,
                &triangulation_config,
                &self.camera,
            );
        }
        self.dbg(format!(
            "[kf] frame={} grown={} from {} neighbor kfs",
            frame.idx,
            total_grown,
            neighbor_kf_indices.len()
        ));

        self.map.lock().unwrap().upsert_keyframe(curr_kf);
        if let (Some(prev_kf_idx), Some(prev_ts)) = (
            self.state.last_keyframe_idx,
            self.last_keyframe_timestamp_sec,
        ) {
            let (preint, raw_samples) = self.preintegrate_window(prev_ts, timestamp_sec);
            if preint.dt > 0.0 {
                self.map.lock().unwrap().add_imu_factor(
                    prev_kf_idx,
                    frame.idx,
                    preint,
                    raw_samples,
                    prev_ts,
                    timestamp_sec,
                );
            }
        }

        self.last_keyframe_timestamp_sec = Some(timestamp_sec);

        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Forward SearchInNeighbors / Fuse: extend each curr_kf-observed map
        // point's observation list to neighbor KFs that don't yet observe it.
        // Run before local BA so BA sees the extra reprojection constraints.
        let n_fused = self.map.lock().unwrap().fuse_into_neighbors(
            frame.idx,
            &neighbor_kf_indices,
            &self.camera,
        );
        self.dbg(format!("[fuse] frame={} fused={}", frame.idx, n_fused));

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            // Drop the solve's map lock before applying its result with a new lock.
            let outcome = {
                let map = self.map.lock().unwrap();
                self.inertial_init.on_keyframe_initialized(
                    &map,
                    timestamp_sec,
                    self.imu_t_bc,
                    self.imu_bias,
                    self.gravity_world,
                )
            };
            self.apply_inertial_refinement(outcome);
        }

        let job = self.keyframe_job();
        if !self.local_mapping.submit(job) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        // Synchronous mode has a completed correction available immediately;
        // asynchronous mode will deliver it at a later frame boundary.
        self.apply_local_mapping_results();

        // Index this keyframe for appearance-based place recognition and surface
        // any loop candidates (no-op unless a vocabulary was provided).
        self.register_place_recognition(frame.idx);

        true
    }

    /// Applies a VIBA1/VIBA2 refinement outcome produced by the initializer.
    fn apply_inertial_refinement(&mut self, outcome: InertialInitOutcome) {
        let InertialInitOutcome::Attempted { stage, result } = outcome else {
            return;
        };
        let stage_label = stage.label();
        match result {
            Ok(init) => {
                let scale = init.scale;
                let bg = init.bias.gyro;
                match self.apply_inertial_initialization(init) {
                    Ok(()) => self.dbg(format!(
                        "[imu_init] {stage_label} accepted: scale_correction={scale:.4} gyro_bias=({:.4},{:.4},{:.4})",
                        bg.x, bg.y, bg.z
                    )),
                    Err(error) => {
                        self.dbg(format!("[imu_init] {stage_label} apply rejected: {error}"));
                    }
                }
            }
            Err(error) => {
                self.dbg(format!("[imu_init] {stage_label} rejected: {error}"));
            }
        }
    }

    /// Runs map-side loop closing, then applies its live tracking consequences.
    fn register_place_recognition(&mut self, kf_idx: usize) {
        let context = LoopClosingContext {
            reference_keyframe_idx: self.state.current_keyframe_idx,
            inertial: self.state.imu_initialized.then_some(InertialPgoContext {
                gravity_world: self.gravity_world,
            }),
        };
        let outcome = {
            let mut map = self.map.lock().unwrap();
            self.loop_closer
                .on_keyframe(&mut map, &self.camera, kf_idx, context)
        };
        self.apply_loop_closure_outcome(outcome);
    }

    fn apply_loop_closure_outcome(&mut self, outcome: LoopClosingOutcome) {
        if let Some(message) = outcome.debug_message {
            self.dbg(message);
        }
        if let Some(correction) = outcome.reference_correction {
            self.state.pose_world_to_cam = apply_reference_pose_correction(
                self.state.pose_world_to_cam,
                correction.before,
                correction.after,
            );
            self.state.velocity_world = correction.world.rotation * self.state.velocity_world;
            self.track_set = TrackSet::new();
            if self.state.imu_initialized {
                let job = self.keyframe_job();
                if !self.local_mapping.submit(job) {
                    self.dbg("[local_mapping] worker is unavailable after PGO".into());
                }
                self.apply_local_mapping_results();
            }
        }
        self.loop_closure_events.extend(outcome.events);
    }
}

fn carry_klt_survivors(track_set: &mut TrackSet, survivors: Option<Vec<FlowSurvivor>>) {
    if survivors.is_none_or(|survivors| track_set.advance(survivors).is_err()) {
        *track_set = TrackSet::new();
    }
}

/// Carries a reference-keyframe BA correction into the current tracking pose
/// while preserving the current camera's pose relative to that reference.
fn apply_reference_pose_correction(
    current_pose: Pose3d,
    reference_before: Pose3d,
    reference_after: Pose3d,
) -> Pose3d {
    let current_from_reference = Pose3d::between(&reference_before, &current_pose);
    current_from_reference.compose(&reference_after)
}

#[cfg(test)]
fn pose_graph_tracking_correction(
    current_pose: Pose3d,
    reference_kf_idx: usize,
    keyframe_indices: &[usize],
    poses_before: &[Pose3d],
    poses_after: &[Pose3d],
) -> Option<Pose3d> {
    let (reference_before, reference_after, _) = pose_graph_reference_correction(
        reference_kf_idx,
        keyframe_indices,
        poses_before,
        poses_after,
    )?;
    Some(apply_reference_pose_correction(
        current_pose,
        reference_before,
        reference_after,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ImuInitApplyError, SlamConfig, SlamSystem, apply_reference_pose_correction,
        carry_klt_survivors, pose_graph_reference_correction, pose_graph_tracking_correction,
    };
    use crate::Frame;
    use crate::initialization::{ImuInitResult, KeyframeVelocity};
    use crate::map::Keyframe;
    use crate::tracking::optical_flow::{FlowSurvivor, MapKeypointMatch, TrackSet};
    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

    fn assert_pose_close(actual: Pose3d, expected: Pose3d) {
        assert!((actual.translation - expected.translation).length() < 1e-10);
        for (actual, expected) in actual
            .rotation
            .to_cols_array()
            .iter()
            .zip(expected.rotation.to_cols_array())
        {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    fn empty_keyframe(idx: usize) -> Keyframe {
        Keyframe::from_frame(Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        })
    }

    /// The map-side alignment is covered in `map`; this checks the system
    /// state the application adopts from the last aligned keyframe.
    #[test]
    fn inertial_initialization_adopts_the_last_keyframe_state() {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut system = SlamSystem::new(camera, SlamConfig::default());
        {
            let mut map = system.map.lock().unwrap();
            map.upsert_keyframe(empty_keyframe(20));
            map.upsert_keyframe(empty_keyframe(10));
        }
        let velocity_10 = Vec3F64::new(1.0, 2.0, 3.0);
        let velocity_20 = Vec3F64::new(4.0, 5.0, 6.0);
        let result = ImuInitResult {
            scale: 1.0,
            // Already at the canonical gravity direction, so the alignment
            // rotation is identity and the velocities pass through unchanged.
            gravity_world: Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0),
            keyframe_velocities: vec![
                KeyframeVelocity {
                    keyframe_idx: 10,
                    velocity_world: velocity_10,
                },
                KeyframeVelocity {
                    keyframe_idx: 20,
                    velocity_world: velocity_20,
                },
            ],
            bias: ImuBias::default(),
        };

        system
            .apply_inertial_initialization(result)
            .expect("valid initialization should apply");

        assert!((system.state.velocity_world - velocity_20).length() < 1e-12);
        assert!(system.state.imu_initialized);
        assert!(system.state.velocity.is_none());
        assert!(
            (system.gravity_world - Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)).length() < 1e-12
        );
    }

    #[test]
    fn inertial_initialization_rejects_zero_gravity() {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut system = SlamSystem::new(camera, SlamConfig::default());
        let result = ImuInitResult {
            scale: 1.0,
            gravity_world: Vec3F64::ZERO,
            keyframe_velocities: Vec::new(),
            bias: ImuBias::default(),
        };

        assert!(matches!(
            system.apply_inertial_initialization(result),
            Err(ImuInitApplyError::InvalidGravity)
        ));
    }

    #[test]
    fn reference_pose_correction_preserves_relative_camera_pose() {
        let reference_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(-1.0, 0.5, 0.2),
        );
        let relative_pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.05, 0.2)).matrix(),
            Vec3F64::new(-0.5, 0.1, 0.3),
        );
        let current_before = relative_pose.compose(&reference_before);
        let reference_after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.1, -0.1)).matrix(),
            Vec3F64::new(-2.0, -0.3, 0.8),
        );

        let corrected =
            apply_reference_pose_correction(current_before, reference_before, reference_after);

        assert_pose_close(Pose3d::between(&reference_after, &corrected), relative_pose);
    }

    #[test]
    fn pose_graph_tracking_correction_uses_current_reference_keyframe() {
        let reference_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(-1.0, 0.5, 0.2),
        );
        let reference_after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.1, -0.1)).matrix(),
            Vec3F64::new(-2.0, -0.3, 0.8),
        );
        let relative_pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.05, 0.2)).matrix(),
            Vec3F64::new(-0.5, 0.1, 0.3),
        );
        let current_before = relative_pose.compose(&reference_before);

        let corrected = pose_graph_tracking_correction(
            current_before,
            20,
            &[10, 20],
            &[Pose3d::IDENTITY, reference_before],
            &[Pose3d::IDENTITY, reference_after],
        )
        .unwrap();

        assert_pose_close(Pose3d::between(&reference_after, &corrected), relative_pose);
        assert!(
            pose_graph_tracking_correction(
                current_before,
                99,
                &[10, 20],
                &[Pose3d::IDENTITY, reference_before],
                &[Pose3d::IDENTITY, reference_after],
            )
            .is_none()
        );
    }

    #[test]
    fn pose_graph_reference_correction_rotates_live_world_velocity() {
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.4, 0.0)).matrix();
        let reference_before = Pose3d::IDENTITY;
        let reference_after = Pose3d::new(yaw.transpose(), Vec3F64::ZERO);
        let (_, _, correction) = pose_graph_reference_correction(
            20,
            &[10, 20],
            &[Pose3d::IDENTITY, reference_before],
            &[Pose3d::IDENTITY, reference_after],
        )
        .unwrap();
        let velocity = Vec3F64::new(1.0, 0.2, -0.5);

        let corrected = correction.rotation * velocity;

        assert!((corrected - yaw * velocity).length() < 1e-10);
    }

    #[test]
    fn klt_tracks_survive_skipped_frame_and_clear_without_survivors() {
        let mut tracks = TrackSet::new();
        tracks
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[10.0, 20.0]],
            )
            .unwrap();
        let track_id = tracks.tracks()[0].id();

        carry_klt_survivors(
            &mut tracks,
            Some(vec![FlowSurvivor {
                track_id,
                pixel: [12.0, 21.0],
                error: 0.5,
            }]),
        );

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks.tracks()[0].id(), track_id);
        assert_eq!(tracks.tracks()[0].map_point_idx(), Some(42));
        assert_eq!(tracks.tracks()[0].pixel(), [12.0, 21.0]);
        assert_eq!(tracks.tracks()[0].age(), 2);

        carry_klt_survivors(&mut tracks, None);
        assert!(tracks.is_empty());
    }
    #[test]
    fn loop_closure_outcome_only_changes_tracking_after_map_correction() {
        use crate::loop_closure::{LoopClosingOutcome, LoopClosureEvent, ReferencePoseCorrection};
        use crate::map::LocalMappingMode;
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut system = SlamSystem::new(
            camera,
            SlamConfig {
                local_mapping: LocalMappingMode::Synchronous,
                ..SlamConfig::default()
            },
        );
        let before = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(-1.0, 0.0, 0.0));
        let relative = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(0.0, 0.0, 0.2));
        let live_pose = relative.compose(&before);
        let velocity = Vec3F64::new(1.0, 0.2, -0.5);
        system.state.pose_world_to_cam = live_pose;
        system.state.velocity_world = velocity;
        system
            .track_set
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[10.0, 20.0]],
            )
            .unwrap();
        system.apply_loop_closure_outcome(LoopClosingOutcome {
            events: vec![LoopClosureEvent::PgoFailed {
                query_kf_idx: 10,
                candidate_kf_idx: 0,
                reason: "rejected".into(),
            }],
            ..LoopClosingOutcome::default()
        });
        assert_eq!(system.state.pose_world_to_cam, live_pose);
        assert_eq!(system.state.velocity_world, velocity);
        assert_eq!(system.track_set.len(), 1);
        assert_eq!(system.drain_loop_closure_events().len(), 1);
        assert!(system.drain_loop_closure_events().is_empty());

        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.4, 0.0)).matrix();
        let after = Pose3d::new(yaw.transpose(), Vec3F64::new(-2.0, 0.0, 0.0));
        system.apply_loop_closure_outcome(LoopClosingOutcome {
            reference_correction: Some(ReferencePoseCorrection {
                before,
                after,
                world: after.inverse().compose(&before),
            }),
            ..LoopClosingOutcome::default()
        });
        assert_pose_close(
            Pose3d::between(&after, &system.state.pose_world_to_cam),
            relative,
        );
        assert!((system.state.velocity_world - yaw * velocity).length() < 1e-10);
        assert!(system.track_set.is_empty());
    }
}
