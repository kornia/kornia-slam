//! SLAM runtime: orchestrates tracking, mapping, and state transitions.
//!
//! The runtime flow is kept in one file so it can be read from top to bottom
//! in the same order frames move through the system.

pub mod config;
mod inertial;

use inertial::InertialState;

pub use crate::loop_closure::LoopClosureEvent;
pub use config::{LoopClosingConfig, SlamConfig};

use std::sync::{Arc, Mutex};

use crate::Frame;
use crate::initialization::{
    ImuInitNotReadyReason, ImuInitResult, InertialInitOutcome, TwoViewInitConfig,
    try_initialize_two_view,
};
use crate::loop_closure::{InertialPgoContext, LoopCloser, LoopClosingContext, LoopClosingOutcome};
use crate::map::{
    ImuFactor, InertialAlignment, InertialAlignmentError, Keyframe, KeyframeJob, LandmarkSeed,
    LandmarkTarget, LocalMapping, Map, MapInsertion, MapMutationError, MapPoint, ObservationKey,
    ObservationLink,
};
use crate::place_recognition::Vocabulary;
use crate::pose_conversion::rotation_from_to;
use crate::sensor_rig::{ImuCalibration, SensorRig};
use crate::stereo::unproject_stereo;
use crate::tracking::{
    InertialPrediction, KeyframePolicy, RecoveryDecision, SystemMode, Tracker, TrackingInput,
    TrackingResult, TrackingStatus,
};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuMeasurement};

/// Top-level SLAM system: orchestrates tracking, mapping, and state transitions.
pub struct SlamSystem {
    // Fixed camera and IMU calibration
    rig: SensorRig,
    tracker: Tracker,
    // Boostrap pose estimator
    two_view_init_config: TwoViewInitConfig,
    // Keyframe insertion policy
    keyframe_policy: KeyframePolicy,
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
    inertial: InertialState,
    local_mapping: LocalMapping,
    loop_closer: LoopCloser,
    loop_closure_events: Vec<LoopClosureEvent>,
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
        Self::with_rig(SensorRig { camera, imu: None }, config)
    }

    /// Creates a SLAM system with fixed camera and optional IMU calibration.
    pub fn with_rig(rig: SensorRig, config: SlamConfig) -> Self {
        let map = Arc::new(Mutex::new(Map::new()));
        let local_mapping =
            LocalMapping::new(config.local_mapping, Arc::clone(&map), rig.camera.clone());
        let map_publication_gate = local_mapping.publication_gate();
        Self {
            rig,
            tracker: Tracker::new(config.map_projection, config.tracking_loss_recovery),
            two_view_init_config: config.two_view_init,
            keyframe_policy: config.keyframe_policy,
            stereo_close_depth: config.stereo_close_depth_m,
            debug: config.debug,
            debug_messages: Vec::new(),
            map,
            map_publication_gate,
            local_mapping,
            inertial: InertialState::new(),
            loop_closer: LoopCloser::new(config.pgo),
            loop_closure_events: Vec::new(),
        }
    }

    /// The local-mapping job description for the current system state.
    fn keyframe_job(&self) -> KeyframeJob {
        KeyframeJob {
            imu_initialized: self.tracker.state.imu_initialized,
            imu_t_bc: self.rig.imu.as_ref().map(|imu| imu.camera_to_body),
            gravity_world: self.inertial.gravity_world,
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
        self.tracker.adopt_inertial_alignment(
            last_keyframe.frame.pose_world_to_cam,
            last_keyframe.velocity_world,
        );
        self.inertial.gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        self.inertial.bias = init.bias;
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
        match &mut self.rig.imu {
            Some(imu) => imu.camera_to_body = t_bc,
            None => self.rig.imu = Some(ImuCalibration::new(t_bc)),
        }
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
        frame.ensure_undistorted(&self.rig.camera);
        self.inertial.buffer_samples(imu_samples);

        match self.tracker.state.mode {
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
        self.tracker.state.current_keyframe_idx.and_then(|ki| {
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
        let Some(reference_idx) = self.tracker.state.current_keyframe_idx else {
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

            self.tracker.apply_local_ba_correction(
                correction.pose_before,
                correction.pose_after,
                correction.velocity_world,
            );
            self.inertial.bias = correction.imu_bias;
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
        curr_frame.pose_world_to_cam = self.tracker.state.pose_world_to_cam;

        const MIN_STEREO_POINTS: usize = 50;
        let cam_points = unproject_stereo(&curr_frame, &self.rig.camera);
        if cam_points.len() < MIN_STEREO_POINTS {
            self.dbg(format!(
                "[bootstrap_stereo] frame={} skip: only {} stereo points (need >= {})",
                curr_frame.idx,
                cam_points.len(),
                MIN_STEREO_POINTS,
            ));
            return TrackingResult {
                pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let pose_inv = curr_frame.pose_world_to_cam.inverse();
        let keyframe = Keyframe::from_frame(curr_frame);
        let curr_idx = keyframe.frame.idx;

        let landmarks: Vec<LandmarkSeed> = cam_points
            .iter()
            .map(|(desc_idx, p_cam)| LandmarkSeed {
                position: pose_inv.transform_point(p_cam),
                color: keyframe
                    .frame
                    .keypoint_colors
                    .get(*desc_idx)
                    .copied()
                    .unwrap_or([128; 3]),
                reference: ObservationKey {
                    keyframe_idx: curr_idx,
                    feature_idx: *desc_idx,
                },
            })
            .collect();

        // The keyframe and its seeds are published together; tracker state is
        // adopted below only once that succeeded.
        let published = self.map.lock().unwrap().apply_insertion(MapInsertion {
            keyframes: vec![keyframe],
            landmarks,
            ..Default::default()
        });
        let added = match published {
            Ok(result) => result.landmark_ids.len(),
            Err(error) => {
                self.dbg(format!(
                    "[bootstrap_stereo] frame={curr_idx} publication rejected: {error}"
                ));
                return TrackingResult {
                    pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
        };

        self.dbg(format!(
            "[bootstrap_stereo] frame={curr_idx} metric map created with {added} points",
        ));

        self.tracker.state.current_keyframe_idx = Some(curr_idx);
        self.tracker.state.last_keyframe_idx = Some(curr_idx);
        self.tracker.state.velocity = None;
        // The map is already metric (stereo baseline), but gravity, velocities,
        // and the gyro bias still need the inertial init before IMU prediction
        // can run; the solve there keeps scale fixed at 1.
        self.tracker.state.mode = if self
            .rig
            .imu
            .as_ref()
            .map(|imu| imu.camera_to_body)
            .is_some()
        {
            self.inertial
                .initializer
                .begin_window(curr_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.inertial.prune_before(timestamp_sec);

        TrackingResult {
            pose_world_to_cam: self.tracker.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    fn bootstrap_mono(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.tracker.state.pose_world_to_cam;

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
            self.tracker.state.bootstrap_frame = None;
            return TrackingResult {
                pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let Some(prev_bootstrap_frame) = self.tracker.state.bootstrap_frame.take() else {
            self.dbg(format!(
                "[bootstrap] frame={} stored as reference (awaiting second frame)",
                curr_frame.idx,
            ));
            self.tracker.state.bootstrap_frame = Some(curr_frame);
            self.inertial.bootstrap_timestamp_sec = Some(timestamp_sec);
            // Samples before the reference frame can never enter an edge.
            self.inertial.prune_before(timestamp_sec);
            return TrackingResult {
                pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        };

        let result = try_initialize_two_view(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &self.rig.camera,
            &self.two_view_init_config,
        );

        let two_view_estimate = match result {
            Err(reason) => {
                self.dbg(format!(
                    "[bootstrap] frame={} (ref={}) reject: {}",
                    curr_frame.idx, prev_bootstrap_frame.idx, reason,
                ));
                self.tracker.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return TrackingResult {
                    pose_world_to_cam: self.tracker.state.pose_world_to_cam,
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
        self.tracker.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
        let prev_idx = prev_bootstrap_frame.idx;
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        // A rejected publication leaves no map to evaluate, and tracker state
        // must not advertise a keyframe pair the map does not hold.
        if let Err(error) = self.build_initial_map(
            reference_kf,
            current_kf,
            &two_view_estimate.matches,
            &two_view_estimate.points3d,
            &two_view_estimate.inlier_indices,
            two_view_estimate.median_depth,
        ) {
            self.dbg(format!(
                "[bootstrap] frame={curr_idx} publication rejected: {error}"
            ));
            self.map.lock().unwrap().clear_active();
            self.tracker.restart_bootstrap();
            return TrackingResult {
                pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        // Post-BA acceptance gate. Initialization owns the metric and its
        // thresholds; the system only acts on the verdict.
        let outcome = {
            let map = self.map.lock().unwrap();
            crate::initialization::bootstrap::evaluate_bootstrap(&map, prev_idx, curr_idx)
        };
        if let crate::initialization::bootstrap::BootstrapOutcome::Rejected { reason, quality } =
            outcome
        {
            self.dbg(format!(
                "[init_gate] reject: {reason:?} quality={quality:?}"
            ));
            self.map.lock().unwrap().clear_active();
            self.tracker.state.reset();
            return TrackingResult {
                pose_world_to_cam: self.tracker.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        // BA inside build_initial_map may have refined KF1's pose; sync state
        // and recompute velocity from the post-BA pose.
        if let Some(kf) = self.map.lock().unwrap().get_keyframe(curr_idx) {
            self.tracker.state.pose_world_to_cam = kf.frame.pose_world_to_cam;
        }

        if let Some(prev_ts) = self.inertial.bootstrap_timestamp_sec {
            let (preint, raw_samples) =
                self.inertial
                    .preintegrate_window(self.rig.imu.as_ref(), prev_ts, timestamp_sec);
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
            self.inertial.prune_before(timestamp_sec);
        }

        self.tracker.state.current_keyframe_idx = Some(curr_idx);
        self.tracker.state.last_keyframe_idx = Some(curr_idx);
        self.tracker.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.tracker.state.pose_world_to_cam,
        ));
        // Inertial init needs the camera-to-body extrinsic to relate IMU deltas
        // to camera poses; without it, run visual-only as before.
        self.tracker.state.mode = if self
            .rig
            .imu
            .as_ref()
            .map(|imu| imu.camera_to_body)
            .is_some()
        {
            self.inertial
                .initializer
                .begin_window(curr_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);

        TrackingResult {
            pose_world_to_cam: self.tracker.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    fn build_initial_map(
        &mut self,
        reference_kf: Keyframe,
        current_kf: Keyframe,
        matches: &[(usize, usize)],
        points3d: &[Vec3F64],
        inlier_indices: &[usize],
        median_depth: Option<f64>,
    ) -> Result<usize, MapMutationError> {
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

        let reference_kf_idx = reference_kf.frame.idx;
        let current_kf_idx = current_kf.frame.idx;

        // Both keyframes, the triangulated landmarks and their second observers
        // publish as one batch: the reference keyframe it points at arrives in
        // the same request.
        let mut request = MapInsertion {
            keyframes: vec![reference_kf, current_kf],
            ..Default::default()
        };
        for (position, _descriptor, color, ref_desc_idx, curr_desc_idx) in &triangulated {
            let new_index = request.landmarks.len();
            request.landmarks.push(LandmarkSeed {
                position: *position,
                color: *color,
                reference: ObservationKey {
                    keyframe_idx: current_kf_idx,
                    feature_idx: *curr_desc_idx,
                },
            });
            request.observations.push(ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: reference_kf_idx,
                    feature_idx: *ref_desc_idx,
                },
                landmark: LandmarkTarget::New(new_index),
            });
        }
        let added = self
            .map
            .lock()
            .unwrap()
            .apply_insertion(request)?
            .landmark_ids
            .len();

        self.map.lock().unwrap().run_initial_ba(&self.rig.camera);

        // Seed the place-recognition database with the two bootstrap keyframes so
        // a later revisit of the start can match them.
        self.register_place_recognition(reference_kf_idx);
        self.register_place_recognition(current_kf_idx);

        Ok(added)
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
            self.inertial.initializer.on_keyframe_uninitialized(
                &map,
                timestamp_sec,
                self.rig.imu.as_ref().map(|imu| imu.camera_to_body),
                self.inertial.bias,
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
                    self.tracker.state.mode = SystemMode::Tracking;
                    self.tracker.state.imu_init_timestamp_sec = Some(timestamp_sec);
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
        // Preserve reference-state refresh before building the IMU window.
        if let Some(kf_idx) = self.tracker.state.current_keyframe_idx
            && let Some(kf) = self.map.lock().unwrap().get_keyframe(kf_idx)
        {
            self.tracker.state.velocity_world = kf.velocity_world;
            self.inertial.bias = kf.imu_bias;
        }
        let prev_timestamp = self.tracker.state.last_frame_timestamp_sec;
        let preint = (self.tracker.state.imu_initialized && prev_timestamp > 0.0).then(|| {
            self.inertial
                .preintegrate_window(self.rig.imu.as_ref(), prev_timestamp, timestamp_sec)
                .0
        });
        let outcome = {
            let map = self.map.lock().unwrap();
            self.tracker.track(
                TrackingInput {
                    frame: &frame,
                    previous_image,
                    current_image,
                    timestamp_sec,
                    inertial: preint.as_ref().map(|preintegrated| InertialPrediction {
                        preintegrated,
                        t_bc: self.rig.imu.as_ref().map(|imu| imu.camera_to_body),
                        gravity_world: self.inertial.gravity_world,
                    }),
                },
                &map,
                &self.rig.camera,
            )
        };
        let mut status = if outcome.rejection.is_some() {
            TrackingStatus::Skipped
        } else {
            TrackingStatus::Tracked
        };
        if self.debug {
            let msg = match outcome.rejection {
                Some(reason) => format!("[track] frame={} reject: {:?}", frame.idx, reason),
                None => format!(
                    "[track] frame={} ok: matches={} inliers={}",
                    frame.idx,
                    outcome.matches.len(),
                    outcome.inliers,
                ),
            };
            self.debug_messages.push(msg);
        }
        if status == TrackingStatus::Tracked {
            // Keep visibility accounting on the predicted pose and local map.
            // Release the map borrow before publication can lock it again.
            {
                let mut map = self.map.lock().unwrap();
                let matched_ids: Vec<usize> =
                    outcome.matches.iter().map(|&(mp_idx, _)| mp_idx).collect();
                let local_indices = crate::tracking::local_map::select_local_landmarks(
                    &map,
                    &matched_ids,
                    self.tracker.state.current_keyframe_idx,
                    &Default::default(),
                );
                let visible = crate::tracking::local_map::landmarks_in_frustum(
                    &map,
                    &local_indices,
                    &self.rig.camera,
                    &outcome.candidate_pose,
                    frame.image_size,
                );
                map.update_observation_counts(&visible, &outcome.matches);
            }
            if self.try_insert_keyframe(&frame, timestamp_sec, outcome.inliers, &outcome.matches) {
                status = TrackingStatus::KeyframeAccepted;
            }
        }
        if outcome.recovery == RecoveryDecision::RestartBootstrap {
            let lost_for_sec = self
                .tracker
                .state
                .lost_since_sec
                .map_or(0.0, |since| timestamp_sec - since);
            self.dbg(format!(
                "[lost] frame={} giving up after {:.2}s: resetting",
                frame.idx, lost_for_sec,
            ));
            self.tracker.restart_bootstrap();
            return self.bootstrap_step(frame, timestamp_sec);
        }
        // Keyframe insertion may have corrected the live pose. Finalize and
        // return that state, not a pose captured before BA or loop closing.
        if status != TrackingStatus::Skipped {
            self.tracker.state.lost_since_sec = None;
        }
        self.tracker.state.last_frame_timestamp_sec = timestamp_sec;
        if let Some(kf_ts) = self.inertial.last_keyframe_timestamp_sec {
            self.inertial.prune_before(kf_ts.min(timestamp_sec));
        }
        TrackingResult {
            pose_world_to_cam: self.tracker.state.pose_world_to_cam,
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
        let n_ref_map_points = if let Some(ki) = self.tracker.state.current_keyframe_idx {
            let map = self.map.lock().unwrap();
            map.get_keyframe(ki)
                .map(|kf| kf.num_associated_points())
                .unwrap_or(0)
        } else {
            0
        };

        if !self.keyframe_policy.should_insert(
            frame.idx,
            self.tracker.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        // Guard: reference KF must exist before we can triangulate.
        if let Some(ki) = self.tracker.state.current_keyframe_idx {
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
            pose_world_to_cam: self.tracker.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
            keypoints_undist: frame.keypoints_undist.clone(),
        });
        // Seed the new keyframe with the IMU-propagated velocity and current bias
        // estimate so that VI-BA starts from a reasonable linearisation point rather
        // than zero, which would produce huge residuals on the newest IMU edge.
        if self.tracker.state.imu_initialized {
            curr_kf.velocity_world = self.tracker.state.velocity_world;
            curr_kf.imu_bias = self.inertial.bias;
        }

        // Neighbours are captured BEFORE publication: growing against a list
        // that already contains the current keyframe would triangulate it
        // against itself and drop the oldest real neighbour.
        //
        // Mirrors ORB-SLAM3's CreateNewMapPoints, which uses the 30 best
        // covisible keyframes; recency approximates covisibility until the
        // graph is available.
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

        // Core publication: the keyframe, its tracked links, the close stereo
        // seeds and the IMU edge go in as one validated batch. Tracked links are
        // recorded as claims first, so stereo seeding cannot take a feature
        // tracking already owns.
        let mut claimed: Vec<Option<usize>> = vec![None; frame.features.descriptors.len()];
        let mut core = MapInsertion::default();
        for &(mp_idx, curr_idx) in matches {
            if claimed.get(curr_idx).copied().flatten().is_some() {
                continue;
            }
            if let Some(slot) = claimed.get_mut(curr_idx) {
                *slot = Some(mp_idx);
            }
            core.observations.push(ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: frame.idx,
                    feature_idx: curr_idx,
                },
                landmark: LandmarkTarget::Existing(mp_idx),
            });
        }

        // Stereo densification: close stereo keypoints become metric landmarks
        // directly. Far points are left to the pair-growth pass, mirroring
        // ORB-SLAM3's CreateNewKeyFrame.
        if let Some(mthdepth) = self.stereo_close_depth
            && curr_kf.frame.is_stereo()
        {
            core.landmarks = crate::mapping::growth::stereo_seeds(
                &curr_kf.frame,
                &self.rig.camera,
                mthdepth,
                &claimed,
            );
        }
        let n_close = core.landmarks.len();
        core.keyframes.push(curr_kf);

        if let (Some(prev_kf_idx), Some(prev_ts)) = (
            self.tracker.state.last_keyframe_idx,
            self.inertial.last_keyframe_timestamp_sec,
        ) {
            let (preint, raw_samples) =
                self.inertial
                    .preintegrate_window(self.rig.imu.as_ref(), prev_ts, timestamp_sec);
            if preint.dt > 0.0 {
                core.imu_factors.push(ImuFactor {
                    prev_kf_idx,
                    curr_kf_idx: frame.idx,
                    preintegrated: preint,
                    raw_samples,
                    t0: prev_ts,
                    t1: timestamp_sec,
                });
            }
        }

        let published = self.map.lock().unwrap().apply_insertion(core);
        if let Err(error) = published {
            // Nothing was written, so no tracker or IMU reference may advance to
            // a keyframe the map does not hold.
            self.dbg(format!(
                "[kf] frame={} publication rejected: {error}",
                frame.idx
            ));
            return false;
        }
        if n_close > 0 {
            self.dbg(format!(
                "[kf_stereo] frame={} close_points={}",
                frame.idx, n_close
            ));
        }

        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.tracker.state.current_keyframe_idx = Some(frame.idx);
        self.tracker.state.last_keyframe_idx = Some(frame.idx);

        // Optional pair growth against the stored keyframe. Each pair is its own
        // validated batch, published before the next is prepared so a later pass
        // sees the claims the previous one took; a skipped pair does not undo the
        // accepted publication above.
        let imu_initialized = self.tracker.state.imu_initialized;
        let match_config = self.two_view_init_config.match_config;
        let triangulation_config = self.two_view_init_config.triangulation_config.clone();

        let mut total_grown = 0usize;
        for &nb_kf_idx in &neighbor_kf_indices {
            let mut map = self.map.lock().unwrap();
            let Some(request) = crate::mapping::growth::pair_growth_request(
                &map,
                nb_kf_idx,
                frame.idx,
                match_config,
                &triangulation_config,
                &self.rig.camera,
            ) else {
                continue;
            };
            let outcome = map.apply_insertion(request);
            drop(map);
            match outcome {
                Ok(result) => total_grown += result.landmark_ids.len(),
                Err(error) => self.dbg(format!(
                    "[kf] frame={} pair growth against {nb_kf_idx} rejected: {error}",
                    frame.idx
                )),
            }
        }
        self.dbg(format!(
            "[kf] frame={} grown={} from {} neighbor kfs",
            frame.idx,
            total_grown,
            neighbor_kf_indices.len()
        ));

        // Forward SearchInNeighbors: extend this keyframe's landmarks into
        // neighbours that don't yet observe them, before local BA so BA sees the
        // extra reprojection constraints.
        let (n_fused, fuse_conflicts) = {
            let mut map = self.map.lock().unwrap();
            let links = crate::mapping::growth::neighbor_fusion_links(
                &map,
                frame.idx,
                &neighbor_kf_indices,
                &self.rig.camera,
            );
            let mut fused = 0usize;
            let mut conflicts: Vec<MapMutationError> = Vec::new();
            for link in links {
                let LandmarkTarget::Existing(landmark) = link.landmark else {
                    continue;
                };
                match map.link_observation(
                    link.observation.keyframe_idx,
                    link.observation.feature_idx,
                    landmark,
                ) {
                    Ok(true) => fused += 1,
                    // Already linked: the proposal was redundant, not wrong.
                    Ok(false) => {}
                    // Proposals are resolved against a map held under this same
                    // lock, so a refusal means an invariant we believed held did
                    // not. Surface it rather than counting it as a no-op.
                    Err(error) => conflicts.push(error),
                }
            }
            (fused, conflicts)
        };
        self.dbg(format!("[fuse] frame={} fused={}", frame.idx, n_fused));
        for error in fuse_conflicts {
            self.dbg(format!("[fuse] frame={} link refused: {error}", frame.idx));
        }

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            // Drop the solve's map lock before applying its result with a new lock.
            let outcome = {
                let map = self.map.lock().unwrap();
                self.inertial.initializer.on_keyframe_initialized(
                    &map,
                    timestamp_sec,
                    self.rig.imu.as_ref().map(|imu| imu.camera_to_body),
                    self.inertial.bias,
                    self.inertial.gravity_world,
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
            reference_keyframe_idx: self.tracker.state.current_keyframe_idx,
            inertial: self
                .tracker
                .state
                .imu_initialized
                .then_some(InertialPgoContext {
                    gravity_world: self.inertial.gravity_world,
                }),
        };
        let outcome = {
            let mut map = self.map.lock().unwrap();
            self.loop_closer
                .on_keyframe(&mut map, &self.rig.camera, kf_idx, context)
        };
        self.apply_loop_closure_outcome(outcome);
    }

    fn apply_loop_closure_outcome(&mut self, outcome: LoopClosingOutcome) {
        if let Some(message) = outcome.debug_message {
            self.dbg(message);
        }
        if let Some(correction) = outcome.reference_correction {
            self.tracker.apply_loop_correction(
                correction.before,
                correction.after,
                correction.world,
            );
            if self.tracker.state.imu_initialized {
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

#[cfg(test)]
mod tests {
    use super::{ImuInitApplyError, SlamConfig, SlamSystem};
    use crate::Frame;
    use crate::initialization::{ImuInitResult, KeyframeVelocity};
    use crate::map::Keyframe;
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

    #[test]
    fn legacy_extrinsics_setter_preserves_configured_imu_noise() {
        let (_, _, camera) = crate::tracking::tests::synthetic_scene();
        let mut calibration = crate::ImuCalibration::new(Pose3d::IDENTITY);
        calibration.noise.gyro_noise = 0.123;
        calibration.noise.accel_noise = 0.456;
        let mut system = SlamSystem::with_rig(
            crate::SensorRig {
                camera,
                imu: Some(calibration),
            },
            SlamConfig::default(),
        );
        let extrinsic = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(0.1, 0.2, 0.3));
        system.set_imu_extrinsics(extrinsic);
        let imu = system.rig.imu.as_ref().unwrap();
        assert_eq!(imu.noise.gyro_noise, 0.123);
        assert_eq!(imu.noise.accel_noise, 0.456);
        assert_pose_close(imu.camera_to_body, extrinsic);
        assert_pose_close(system.keyframe_job().imu_t_bc.unwrap(), extrinsic);
    }

    #[test]
    fn legacy_constructor_enables_imu_with_historical_noise() {
        let (_, _, camera) = crate::tracking::tests::synthetic_scene();
        let mut system = SlamSystem::new(camera, SlamConfig::default());
        assert!(system.rig.imu.is_none());
        system.set_imu_extrinsics(Pose3d::IDENTITY);
        let imu = system.rig.imu.as_ref().unwrap();
        assert_eq!(imu.noise.gyro_noise, 1.6968e-4);
        assert_eq!(imu.noise.accel_noise, 2.0e-3);
        assert_eq!(imu.noise.gyro_bias_noise, 1.9393e-5);
        assert_eq!(imu.noise.accel_bias_noise, 3.0e-3);
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

        assert!((system.tracker.state.velocity_world - velocity_20).length() < 1e-12);
        assert!(system.tracker.state.imu_initialized);
        assert!(system.tracker.state.velocity.is_none());
        assert!(
            (system.inertial.gravity_world - Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)).length()
                < 1e-12
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
        system.tracker.state.pose_world_to_cam = live_pose;
        system.tracker.state.velocity_world = velocity;
        system.apply_loop_closure_outcome(LoopClosingOutcome {
            events: vec![LoopClosureEvent::PgoFailed {
                query_kf_idx: 10,
                candidate_kf_idx: 0,
                reason: "rejected".into(),
            }],
            ..LoopClosingOutcome::default()
        });
        assert_eq!(system.tracker.state.pose_world_to_cam, live_pose);
        assert_eq!(system.tracker.state.velocity_world, velocity);
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
            Pose3d::between(&after, &system.tracker.state.pose_world_to_cam),
            relative,
        );
        assert!((system.tracker.state.velocity_world - yaw * velocity).length() < 1e-10);
    }
    fn tracking_test_system(map_size: usize) -> SlamSystem {
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
                local_mapping: crate::map::LocalMappingMode::Synchronous,
                ..SlamConfig::default()
            },
        );
        for idx in 0..map_size {
            system
                .map
                .lock()
                .unwrap()
                .upsert_keyframe(empty_keyframe(idx));
        }
        system.tracker.state.mode = crate::tracking::SystemMode::Tracking;
        system.tracker.state.last_frame_timestamp_sec = 1.0;
        system.tracker.state.velocity = Some(Pose3d::new(
            SO3F64::IDENTITY.matrix(),
            Vec3F64::new(0.1, 0.0, 0.0),
        ));
        system
    }

    #[test]
    fn tracking_rejection_carries_visual_prediction_during_grace() {
        let mut system = tracking_test_system(11);
        let image = kornia_image::Image::from_size_val(
            ImageSize {
                width: 640,
                height: 480,
            },
            0u8,
        )
        .unwrap();
        for (idx, timestamp) in [(20, 2.0), (21, 2.25)] {
            let result = system.tracking_step(empty_keyframe(idx).frame, None, &image, timestamp);
            assert_eq!(result.status, crate::tracking::TrackingStatus::Skipped);
            assert_eq!(
                system.tracker.state.mode,
                crate::tracking::SystemMode::Tracking
            );
            assert_eq!(system.tracker.state.last_frame_timestamp_sec, timestamp);
            assert_eq!(system.tracker.state.lost_since_sec, Some(2.0));
        }
        assert!((system.tracker.state.pose_world_to_cam.translation.x - 0.2).abs() < 1e-12);
    }

    #[test]
    fn tracking_loss_boundaries_preserve_same_frame_bootstrap_and_map() {
        let image = kornia_image::Image::from_size_val(
            ImageSize {
                width: 640,
                height: 480,
            },
            0u8,
        )
        .unwrap();
        // Exactly the minimum map size is not established; timeout is inclusive.
        for (map_size, lost_since, timestamp, expect_reset) in [
            (10, None, 2.0, true),
            (11, None, 2.0, false),
            (11, Some(2.0), 2.499, false),
            (11, Some(2.0), 2.5, true),
        ] {
            let mut system = tracking_test_system(map_size);
            system.tracker.state.lost_since_sec = lost_since;
            let mut frame = empty_keyframe(30).frame;
            frame.features.keypoints_xy = vec![[320.0, 240.0]; 101];
            frame.features.descriptors = vec![[0; 32]; 101];
            frame.features.orientations = vec![0.0; 101];
            frame.features.octaves = vec![0; 101];
            system.tracking_step(frame, None, &image, timestamp);
            assert_eq!(system.map.lock().unwrap().keyframes().len(), map_size);
            assert!((system.tracker.state.pose_world_to_cam.translation.x - 0.1).abs() < 1e-12);
            if expect_reset {
                assert_eq!(
                    system.tracker.state.mode,
                    crate::tracking::SystemMode::Bootstrap
                );
                assert_eq!(
                    system.tracker.state.bootstrap_frame.as_ref().unwrap().idx,
                    30
                );
                assert_eq!(system.inertial.bootstrap_timestamp_sec, Some(timestamp));
                assert_eq!(system.tracker.state.last_frame_timestamp_sec, 1.0);
                assert!(system.tracker.state.velocity.is_none());
            } else {
                assert_eq!(
                    system.tracker.state.mode,
                    crate::tracking::SystemMode::Tracking
                );
                assert_eq!(system.tracker.state.last_frame_timestamp_sec, timestamp);
            }
        }
    }

    #[test]
    fn tracking_imu_confidence_boundary_selects_longer_grace() {
        let image = kornia_image::Image::from_size_val(
            ImageSize {
                width: 640,
                height: 480,
            },
            0u8,
        )
        .unwrap();
        for (initialized_at, timestamp, reset) in
            [(0.5, 2.5, false), (0.501, 2.5, true), (0.5, 3.0, true)]
        {
            let mut system = tracking_test_system(11);
            system.tracker.state.imu_initialized = true;
            system.tracker.state.imu_init_timestamp_sec = Some(initialized_at);
            system.tracker.state.lost_since_sec = Some(2.0);
            system.tracking_step(empty_keyframe(20).frame, None, &image, timestamp);
            assert_eq!(
                system.tracker.state.mode == crate::tracking::SystemMode::Bootstrap,
                reset
            );
        }
    }

    #[test]
    fn tracked_keyframe_result_uses_live_state_after_synchronous_mapping() {
        let (map, frame, camera) = crate::tracking::tests::synthetic_scene();
        let image = kornia_image::Image::from_size_val(frame.image_size, 0u8).unwrap();
        let mut system = SlamSystem::new(
            camera,
            SlamConfig {
                local_mapping: crate::map::LocalMappingMode::Synchronous,
                ..SlamConfig::default()
            },
        );
        *system.map.lock().unwrap() = map;
        system.tracker.state.mode = crate::tracking::SystemMode::Tracking;
        system.tracker.state.current_keyframe_idx = Some(0);
        system.tracker.state.last_keyframe_idx = Some(0);
        system.tracker.state.lost_since_sec = Some(0.9);
        let result = system.tracking_step(frame, None, &image, 1.0);
        assert_eq!(
            result.status,
            crate::tracking::TrackingStatus::KeyframeAccepted
        );
        assert_eq!(system.current_keyframe_idx(), Some(8));
        assert_eq!(
            result.pose_world_to_cam,
            system.tracker.state.pose_world_to_cam
        );
        assert_eq!(system.tracker.state.last_frame_timestamp_sec, 1.0);
        assert!(system.tracker.state.lost_since_sec.is_none());
        assert_eq!(system.map.lock().unwrap().keyframes().len(), 2);
    }
}
