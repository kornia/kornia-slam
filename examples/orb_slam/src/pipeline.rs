//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::Image;
use kornia_imgproc::features::{OrbMatchConfig, hamming_distance, match_orb_descriptors};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};
use kornia_slam::Frame;
use kornia_slam::estimation::optical_flow::{
    FlowSurvivor, KltTracker, MapKeypointMatch, TrackSet, snap_unique,
};
use kornia_slam::estimation::two_view::{TwoViewInitConfig, try_initialize_two_view};
use kornia_slam::estimation::{ImuInitConfig, ImuInitializer, MapProjectionEstimator};
use kornia_slam::map::{Keyframe, KeyframeJob, LocalMapping, Map, MapPoint, ORB_SCALE_FACTOR};
use kornia_slam::stereo::unproject_stereo;
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingLossRecoveryPolicy, TrackingResult,
    TrackingStatus,
};

/// Top-level ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
pub struct Pipeline {
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
    inertial_init_start_kf_idx: Option<usize>,
    inertial_init: ImuInitializer,
    // Timestamp of the last try_initialize attempt (successful or not), so
    // retries are throttled to a fixed cadence instead of firing on every
    // single keyframe forever once `ready()` is true — with an ever-growing
    // window (start_idx never resets) and a solve that scales with window
    // size, unthrottled per-keyframe retries turn into an ever-more-expensive
    // no-op once a call starts getting rejected.
    inertial_init_last_attempt_sec: Option<f64>,
    // Timestamp the current inertial-init window started (first keyframe at
    // or after `inertial_init_start_kf_idx`). Mirrors ORB-SLAM3's `mFirstTs`
    // / `mTinit` — used to gate the VIBA1/VIBA2 progressive visual-inertial
    // BA refinement passes (mTinit>5s / mTinit>15s respectively, after the
    // initial VIBA0 solve) at LocalMapping.cc:200-228.
    imu_init_window_start_sec: Option<f64>,
    // VIBA1/VIBA2 fire at most once each, mirroring
    // Map::GetIniertialBA1()/GetIniertialBA2() latching in ORB-SLAM3.
    imu_viba1_done: bool,
    imu_viba2_done: bool,
    local_mapping: LocalMapping,
    klt_tracker: KltTracker,
    track_set: TrackSet,
    // System state
    state: SystemState,
}

impl Pipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(camera: PinholeCamera, config: PipelineConfig) -> Self {
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
            inertial_init_start_kf_idx: None,
            // Matches ORB-SLAM3's LocalMapping::InitializeIMU VIBA0 gate
            // (nMinKF=10; minTime=1.0s stereo/2.0s mono — `ready()` doubles
            // this for mono). The previous min_keyframes=30/min_time_sec=15.0
            // was effectively skipping VIBA0/VIBA1 and attempting a
            // VIBA2-strength window on the very first try.
            inertial_init: ImuInitializer::new(ImuInitConfig {
                min_keyframes: 10,
                min_time_sec: 1.0,
                min_motion: 0.05,
            }),
            inertial_init_last_attempt_sec: None,
            imu_init_window_start_sec: None,
            imu_viba1_done: false,
            imu_viba2_done: false,
            klt_tracker: KltTracker::default(),
            track_set: TrackSet::new(),
        }
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

    /// Toggle whether the pipeline buffers per-frame debug messages.
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
            self.inertial_init_start_kf_idx = Some(curr_idx);
            self.imu_init_window_start_sec = Some(timestamp_sec);
            self.imu_viba1_done = false;
            self.imu_viba2_done = false;
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

    /// Back-projects `curr_kf`'s unassociated close stereo keypoints
    /// (`z < mthdepth`) into new metric map points, associating them to the
    /// keyframe. Returns the number of points created.
    fn add_close_stereo_points(&mut self, curr_kf: &mut Keyframe, mthdepth: f64) -> usize {
        let cam_points = unproject_stereo(&curr_kf.frame, &self.camera);
        if cam_points.is_empty() {
            return 0;
        }
        let pose_inv = curr_kf.frame.pose_world_to_cam.inverse();

        let mut points = Vec::new();
        for (desc_idx, p_cam) in &cam_points {
            // Far points: leave to multi-view triangulation.
            if p_cam.z > mthdepth {
                continue;
            }
            // Skip keypoints already tied to a map point (tracked this frame).
            if curr_kf.map_point(*desc_idx).is_some() {
                continue;
            }
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = curr_kf.frame.features.descriptors[*desc_idx];
            let color = curr_kf
                .frame
                .keypoint_colors
                .get(*desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        self.map
            .lock()
            .unwrap()
            .add_triangulated_points(None, curr_kf, &points)
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
                    "[bootstrap] frame={} (ref={}) reject: {:?}",
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
            two_view_estimate.estimate.inliers,
        ));

        let estimated_pose = two_view_estimate.estimate.pose;
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
            &two_view_estimate.estimate.matches,
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
            self.inertial_init_start_kf_idx = Some(curr_idx);
            self.imu_init_window_start_sec = Some(timestamp_sec);
            self.imu_viba1_done = false;
            self.imu_viba2_done = false;
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

        self.map.lock().unwrap().upsert_keyframe(reference_kf);
        self.map.lock().unwrap().upsert_keyframe(current_kf);

        self.map.lock().unwrap().run_initial_ba(&self.camera);

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

        if result.status == TrackingStatus::KeyframeAccepted
            && let Some(start_idx) = self.inertial_init_start_kf_idx
        {
            // Snapshot the fields needed after releasing the map lock.
            let kfs: Vec<usize> = self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .filter(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.idx)
                .collect();
            let imu_time: f64 = self
                .map
                .lock()
                .unwrap()
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
                .map(|f| f.preintegrated.dt)
                .sum();
            let gate_msg = format_imu_init_gate(
                start_idx,
                kfs.first().copied(),
                kfs.last().copied(),
                kfs.len(),
                self.inertial_init.config.min_keyframes,
                imu_time,
                self.inertial_init.config.min_time_sec,
            );
            self.dbg(gate_msg);
        }

        // Throttle retries: without this, once `ready()` is true, a rejected
        // attempt keeps mode at ImuInit and never resets start_idx, so the
        // exact same (growing) window gets re-solved from scratch on every
        // single subsequent keyframe forever — an ever-more-expensive no-op
        // once a call starts failing. Re-attempt at most once every 5s of
        // new data (mirrors the VIBA1 5s cadence), not every keyframe.
        const RETRY_INTERVAL_SEC: f64 = 5.0;
        let due_for_retry = self
            .inertial_init_last_attempt_sec
            .is_none_or(|last| timestamp_sec - last >= RETRY_INTERVAL_SEC);
        let imu_init_ready = self
            .inertial_init
            .ready(&self.map.lock().unwrap(), self.inertial_init_start_kf_idx);

        if result.status == TrackingStatus::KeyframeAccepted && due_for_retry && imu_init_ready {
            let Some(start_idx) = self.inertial_init_start_kf_idx else {
                return result;
            };
            self.inertial_init_last_attempt_sec = Some(timestamp_sec);
            // VIBA0: ORB-SLAM3's first InitializeIMU call
            // (LocalMapping.cc:183-186) — heavily-regularized, mono suppresses
            // accel bias almost entirely (priorA=1e10) since a short/early
            // window can't yet observe it; stereo uses priorA=1e5.
            let is_mono = !self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .find(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.is_stereo())
                .unwrap_or(false);
            let prior_a0 = if is_mono { 1e10 } else { 1e5 };
            // Drop the solve's map lock before applying its result with a new lock.
            let init_result = self.inertial_init.try_initialize(
                &self.map.lock().unwrap(),
                self.imu_t_bc,
                self.imu_bias,
                start_idx,
                1e2,
                prior_a0,
                false,
            );
            match init_result {
                Some(init) => {
                    let scale = init.scale;
                    let gravity = init.gravity_world;
                    let bg = init.bias.gyro;
                    self.inertial_init.apply_initialization(
                        &mut self.map.lock().unwrap(),
                        &mut self.state,
                        &mut self.imu_bias,
                        &mut self.gravity_world,
                        init,
                        start_idx,
                    );
                    // Mirrors ORB-SLAM3: IMU is marked initialized (and
                    // tracking resumes) immediately after VIBA0 succeeds —
                    // VIBA1/VIBA2 refine bg/ba/scale/gravity further in the
                    // background (see try_insert_keyframe), they don't gate
                    // resuming tracking.
                    self.state.mode = SystemMode::Tracking;
                    self.state.imu_init_timestamp_sec = Some(timestamp_sec);
                    if !self.local_mapping.submit(KeyframeJob {
                        imu_initialized: true,
                        imu_t_bc: self.imu_t_bc,
                        gravity_world: self.gravity_world,
                    }) {
                        self.dbg("[local_mapping] worker is unavailable".into());
                    }
                    self.apply_local_mapping_results();
                    self.dbg(format!(
                        "[imu_init] VIBA0 accepted: scale={scale:.4} gravity=({:.3},{:.3},{:.3}) \
                         gyro_bias=({:.4},{:.4},{:.4})",
                        gravity.x, gravity.y, gravity.z, bg.x, bg.y, bg.z
                    ));
                }
                None => {
                    self.dbg("[imu_init] VIBA0 rejected: solve failed or invalid scale".into());
                }
            }
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
            let n_close = self.add_close_stereo_points(&mut curr_kf, mthdepth);
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
            total_grown += self.grow_map_points_from_keyframe_pair(
                nb_kf_idx,
                &mut curr_kf,
                match_config,
                &triangulation_config,
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
        let n_fused = self.fuse_into_neighbors(frame.idx, &neighbor_kf_indices);
        self.dbg(format!("[fuse] frame={} fused={}", frame.idx, n_fused));

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            self.refine_inertial_init(timestamp_sec);
        }

        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized,
            imu_t_bc: self.imu_t_bc,
            gravity_world: self.gravity_world,
        }) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        // Synchronous mode has a completed correction available immediately;
        // asynchronous mode will deliver it at a later frame boundary.
        self.apply_local_mapping_results();

        true
    }

    /// Kept at VIBA2 because this pipeline lacks the intervening pose-adjusting
    /// inertial BA that lets ORB-SLAM3 safely remove the prior (kornia-slam#51).
    const VIBA_PRIOR_A: f64 = 1e5;

    /// VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s): progressive re-solves with
    /// relaxed priors over the same (now-growing) window that VIBA0 used,
    /// mirroring LocalMapping.cc:200-228. Each fires at most once and refines
    /// bg/ba/scale/gravity further — tracking is already running on VIBA0's
    /// result by the time these get a chance to fire, so a rejection here
    /// just means "try again never" for that stage, not a tracking failure.
    fn refine_inertial_init(&mut self, timestamp_sec: f64) {
        let (Some(start_idx), Some(window_start_sec)) = (
            self.inertial_init_start_kf_idx,
            self.imu_init_window_start_sec,
        ) else {
            return;
        };
        let mtinit = timestamp_sec - window_start_sec;
        if mtinit >= 50.0 {
            return;
        }

        let (prior_g, prior_a, stage) = if !self.imu_viba1_done && mtinit > 5.0 {
            (1.0, Self::VIBA_PRIOR_A, "VIBA1")
        } else if self.imu_viba1_done && !self.imu_viba2_done && mtinit > 15.0 {
            (0.0, Self::VIBA_PRIOR_A, "VIBA2")
        } else {
            return;
        };

        // Drop the solve's map lock before applying its result with a new lock.
        let init_result = self.inertial_init.try_initialize(
            &self.map.lock().unwrap(),
            self.imu_t_bc,
            self.imu_bias,
            start_idx,
            prior_g,
            prior_a,
            true,
        );
        match init_result {
            Some(init) => {
                let scale = init.scale;
                let bg = init.bias.gyro;
                self.inertial_init.apply_initialization(
                    &mut self.map.lock().unwrap(),
                    &mut self.state,
                    &mut self.imu_bias,
                    &mut self.gravity_world,
                    init,
                    start_idx,
                );
                self.dbg(format!(
                    "[imu_init] {stage} accepted: scale_correction={scale:.4} gyro_bias=({:.4},{:.4},{:.4})",
                    bg.x, bg.y, bg.z
                ));
            }
            None => {
                self.dbg(format!(
                    "[imu_init] {stage} rejected: solve failed or invalid scale"
                ));
            }
        }

        if stage == "VIBA1" {
            self.imu_viba1_done = true;
        } else {
            self.imu_viba2_done = true;
        }
    }

    fn grow_map_points_from_keyframe_pair(
        &mut self,
        prev_kf_idx: usize,
        curr_kf: &mut Keyframe,
        match_config: OrbMatchConfig,
        triangulation_config: &TriangulationConfig,
    ) -> usize {
        const MIN_GROWTH_MATCHES: usize = 15;
        // 1-DOF chi-square gate at 95% for the point-to-epipolar-line distance
        // (ORB-SLAM3's CheckDistEpipolarLine), scaled per-octave below.
        const EPIPOLAR_CHI2: f64 = 3.84;

        // Read-only phase: match and triangulate against the neighbor
        // keyframe stored in the map. The shared borrow on self.map ends with
        // this block so the write phase below can mutate the map.
        let points = {
            let map_guard = self.map.lock().unwrap();
            let Some(prev_kf) = map_guard.get_keyframe(prev_kf_idx) else {
                return 0;
            };

            // Only consider features that don't already have a map point in
            // either KF. Matching the full descriptor arrays and then
            // filtering discards almost everything once the KFs are mature
            // (the best matches always land on the already-tracked features).
            let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
                .filter(|&i| prev_kf.map_point(i).is_none())
                .collect();
            let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
                .filter(|&i| curr_kf.map_point(i).is_none())
                .collect();
            if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
                return 0;
            }

            // Both keyframe poses are known, so the fundamental matrix between
            // the pair is fully determined: F = K^-T [t]x R K^-1 with (R, t)
            // the prev->curr relative pose. Filtering matches against this F
            // replaces the F-matrix RANSAC of the two-view estimator (mirrors
            // ORB-SLAM3's SearchForTriangulation).
            let rel = Pose3d::between(
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
            );
            if rel.translation.length() <= 1e-8 {
                // No baseline: epipolar geometry degenerates and triangulation
                // would reject everything anyway.
                return 0;
            }
            let t = rel.translation;
            let t_skew = Mat3F64::from_cols(
                Vec3F64::new(0.0, t.z, -t.y),
                Vec3F64::new(-t.z, 0.0, t.x),
                Vec3F64::new(t.y, -t.x, 0.0),
            );
            let camera = &self.camera;
            let k_inv = Mat3F64::from_cols(
                Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
                Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
                Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
            );
            let f_mat = k_inv.transpose() * (t_skew * rel.rotation) * k_inv;

            // Epipole of the prev camera in the curr image (projection of
            // prev's camera center). Near it every keypoint is close to every
            // epipolar line, so the chi-square gate below is uninformative
            // there: wrong matches survive and triangulate to depth-garbage
            // points that still reproject well in both views. Mirrors
            // ORB-SLAM3's epipole-proximity rejection in
            // SearchForTriangulation; RANSAC consensus used to absorb these.
            let prev_center_world = prev_kf.frame.pose_world_to_cam.inverse().translation;
            let epipole_cam = curr_kf
                .frame
                .pose_world_to_cam
                .transform_point(&prev_center_world);
            let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
                Vec2F64::new(
                    camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
                    camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
                )
            });

            // Brute-force descriptor matching over the unassociated subsets
            // (global second-best ratio test + orientation consistency live
            // inside the matcher and are essential for match quality).
            let prev_orients: Vec<f32> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.orientations[i])
                .collect();
            let prev_descs: Vec<[u8; 32]> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.descriptors[i])
                .collect();
            let curr_orients: Vec<f32> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.orientations[i])
                .collect();
            let curr_descs: Vec<[u8; 32]> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.descriptors[i])
                .collect();

            let sub_matches = match_orb_descriptors(
                &prev_orients,
                &prev_descs,
                &curr_orients,
                &curr_descs,
                match_config,
            );

            // Keep only matches consistent with the pose-derived epipolar
            // geometry: distance from the curr keypoint to its epipolar line
            // must pass the chi-square gate at the octave's detection sigma.
            let mut pair_indices: Vec<(usize, usize)> = Vec::new();
            let mut matched_prev: Vec<Vec2F64> = Vec::new();
            let mut matched_curr: Vec<Vec2F64> = Vec::new();
            for (prev_sub, curr_sub) in sub_matches {
                let (Some(&prev_idx), Some(&curr_idx)) =
                    (prev_unassoc.get(prev_sub), curr_unassoc.get(curr_sub))
                else {
                    continue;
                };
                let (Some(pu), Some(qu)) = (
                    prev_kf.frame.undistorted_xy(prev_idx, camera),
                    curr_kf.frame.undistorted_xy(curr_idx, camera),
                ) else {
                    continue;
                };
                let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
                let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);

                // Reject curr keypoints near the epipole (radius grows with
                // octave; ORB-SLAM3 uses 100 * scaleFactor^octave px^2).
                if let Some(e) = epipole_px {
                    let octave = curr_kf
                        .frame
                        .features
                        .octaves
                        .get(curr_idx)
                        .copied()
                        .unwrap_or(0);
                    let dx = q.x - e.x;
                    let dy = q.y - e.y;
                    if dx * dx + dy * dy < 100.0 * ORB_SCALE_FACTOR.powi(octave as i32) {
                        continue;
                    }
                }

                let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
                let line_norm_sq = l.x * l.x + l.y * l.y;
                if line_norm_sq <= 1e-12 {
                    continue;
                }
                let d = l.x * q.x + l.y * q.y + l.z;
                let octave = curr_kf
                    .frame
                    .features
                    .octaves
                    .get(curr_idx)
                    .copied()
                    .unwrap_or(0);
                let sigma_sq = ORB_SCALE_FACTOR.powi(2 * octave as i32);
                if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
                    continue;
                }

                pair_indices.push((prev_idx, curr_idx));
                matched_prev.push(p);
                matched_curr.push(q);
            }
            if pair_indices.len() < MIN_GROWTH_MATCHES {
                return 0;
            }

            let triangulated = match triangulate_matched_points(
                &matched_prev,
                &matched_curr,
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
                camera,
                triangulation_config,
            ) {
                Ok(pts) => pts,
                Err(_) => return 0,
            };

            let mut points = Vec::new();
            for tp in &triangulated {
                let Some(&(prev_idx, curr_idx)) = pair_indices.get(tp.pair_index) else {
                    continue;
                };
                if curr_kf.map_point(curr_idx).is_some() {
                    continue;
                }
                let color = curr_kf
                    .frame
                    .keypoint_colors
                    .get(curr_idx)
                    .copied()
                    .unwrap_or([128; 3]);
                points.push((
                    tp.position,
                    curr_kf.frame.features.descriptors[curr_idx],
                    color,
                    prev_idx,
                    curr_idx,
                ));
            }
            points
        };

        // Write phase: create the new map points; curr_kf is registered as
        // the first observer inside add_triangulated_points.
        let first_mp_idx = self.map.lock().unwrap().num_map_points();
        let added = self
            .map
            .lock()
            .unwrap()
            .add_triangulated_points(None, curr_kf, &points);

        // Register the neighbor as a second observer on each new map point.
        // This is the SearchInNeighbors-equivalent piece for the
        // triangulating pair: without it the new point would have a single
        // observation, biasing scale/normal geometry and making the cull
        // overly aggressive.
        for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().take(added).enumerate() {
            let mp_idx = first_mp_idx + i;
            self.map
                .lock()
                .unwrap()
                .register_observation_at(mp_idx, prev_kf_idx, prev_desc_idx);
            if let Some(prev_live) = self.map.lock().unwrap().get_keyframe_mut(prev_kf_idx) {
                prev_live.associate_map_point(prev_desc_idx, mp_idx);
            }
        }

        added
    }

    /// Forward Fuse pass: project each map point observed by the current KF
    /// into every neighbor KF that doesn't already observe it. If the
    /// projection lands near an unassociated keypoint with a matching
    /// descriptor, register the observation. Mirrors a subset of ORB-SLAM3's
    /// `SearchInNeighbors` (forward direction only; we don't yet do duplicate
    /// merging or the second-hop covisible expansion).
    fn fuse_into_neighbors(&mut self, curr_kf_idx: usize, neighbor_kf_indices: &[usize]) -> usize {
        const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
        const FUSE_MAX_HAMMING: u32 = 50;

        // Collect map points observed by curr_kf. We snapshot the indices
        // here so we can hold no other borrow on self.map during the loop.
        let curr_mp_indices: Vec<usize> = match self.map.lock().unwrap().get_keyframe(curr_kf_idx) {
            Some(kf) => kf
                .map_point_by_desc_idx
                .iter()
                .filter_map(|&mp| mp)
                .collect(),
            None => return 0,
        };
        if curr_mp_indices.is_empty() {
            return 0;
        }

        let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
        let mut n_fused = 0usize;

        for &nb_kf_idx in neighbor_kf_indices {
            if nb_kf_idx == curr_kf_idx {
                continue;
            }

            // Proposals: (kp_idx_in_nb_kf, mp_idx, hamming). Collected under a
            // shared borrow of the neighbor KF in the map (no clone), resolved
            // in the write phase below so a single keypoint can't be claimed
            // by two map points.
            let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
            {
                let map_guard = self.map.lock().unwrap();
                let Some(nb_kf) = map_guard.get_keyframe(nb_kf_idx) else {
                    continue;
                };

                for &mp_idx in &curr_mp_indices {
                    let mp = match map_guard.map_points().get(mp_idx) {
                        Some(mp) if !mp.culled => mp,
                        _ => continue,
                    };
                    // Skip if neighbor already observes this map point.
                    if mp.observation_kf_indices.contains(&nb_kf_idx) {
                        continue;
                    }

                    // Project into the neighbor's frame.
                    let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 0.0 {
                        continue;
                    }
                    let Ok(pixel) =
                        self.camera
                            .project_to_image(&p_cam, 0.0, nb_kf.frame.image_size)
                    else {
                        continue;
                    };
                    let u = pixel.x as f32;
                    let v = pixel.y as f32;

                    // Find the closest unassociated keypoint within the radius
                    // that matches the map point's representative descriptor.
                    let mut best_dist = u32::MAX;
                    let mut best_kp = usize::MAX;
                    for kp_idx in 0..nb_kf.frame.features.keypoints_xy.len() {
                        if nb_kf.map_point(kp_idx).is_some() {
                            continue;
                        }
                        let Some(kp) = nb_kf.frame.undistorted_xy(kp_idx, &self.camera) else {
                            continue;
                        };
                        let dx = kp[0] - u;
                        let dy = kp[1] - v;
                        if dx * dx + dy * dy > r2 {
                            continue;
                        }
                        let dist = hamming_distance(
                            &mp.descriptor,
                            &nb_kf.frame.features.descriptors[kp_idx],
                        );
                        if dist < best_dist {
                            best_dist = dist;
                            best_kp = kp_idx;
                        }
                    }

                    if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                        proposals.push((best_kp, mp_idx, best_dist));
                    }
                }
            }

            // Resolve proposals: if two map points want the same keypoint,
            // the one with the smaller Hamming distance wins. Track which
            // keypoints are already taken in this pass.
            proposals.sort_by_key(|&(_, _, dist)| dist);
            let mut taken_kp: HashSet<usize> = HashSet::new();
            for (kp_idx, mp_idx, _) in proposals {
                if taken_kp.contains(&kp_idx) {
                    continue;
                }
                // Re-check that the live neighbor KF hasn't already had this
                // keypoint claimed (e.g. by a prior iteration in this fuse
                // call associating a different mp).
                let already = self
                    .map
                    .lock()
                    .unwrap()
                    .get_keyframe(nb_kf_idx)
                    .and_then(|kf| kf.map_point(kp_idx))
                    .is_some();
                if already {
                    continue;
                }
                self.map
                    .lock()
                    .unwrap()
                    .register_observation_at(mp_idx, nb_kf_idx, kp_idx);
                if let Some(nb_live) = self.map.lock().unwrap().get_keyframe_mut(nb_kf_idx) {
                    nb_live.associate_map_point(kp_idx, mp_idx);
                }
                taken_kp.insert(kp_idx);
                n_fused += 1;
            }
        }

        n_fused
    }
}

fn carry_klt_survivors(track_set: &mut TrackSet, survivors: Option<Vec<FlowSurvivor>>) {
    if survivors.is_none_or(|survivors| track_set.advance(survivors).is_err()) {
        *track_set = TrackSet::new();
    }
}

fn format_imu_init_gate(
    start_idx: usize,
    first_idx: Option<usize>,
    last_idx: Option<usize>,
    keyframes: usize,
    min_keyframes: usize,
    imu_time: f64,
    min_time_sec: f64,
) -> String {
    format!(
        "[imu_init_gate] start_idx={start_idx} first_idx={first_idx:?} last_idx={last_idx:?} kfs={keyframes}/{min_keyframes} imu_time={imu_time:.2}/{min_time_sec:.1}s"
    )
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
mod tests {
    use super::{apply_reference_pose_correction, carry_klt_survivors, format_imu_init_gate};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_slam::estimation::optical_flow::{FlowSurvivor, MapKeypointMatch, TrackSet};

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
    fn formats_compact_imu_init_gate() {
        assert_eq!(
            format_imu_init_gate(12, Some(12), Some(32), 7, 10, 1.05, 1.0),
            "[imu_init_gate] start_idx=12 first_idx=Some(12) last_idx=Some(32) kfs=7/10 imu_time=1.05/1.0s"
        );
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
}
