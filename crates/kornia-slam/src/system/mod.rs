//! SLAM runtime: orchestrates tracking, mapping, and state transitions.
//!
//! The runtime flow is kept in one file so it can be read from top to bottom
//! in the same order frames move through the system.

mod config;
mod inertial;

pub use config::{LoopClosingConfig, SlamConfig};

use inertial::{AppliedInitialization, InertialState, due_for_retry, viba0_accel_bias_prior};

#[deprecated(since = "0.1.0", note = "use `SlamSystem`")]
pub type SlamPipeline = SlamSystem;

#[allow(deprecated)]
pub use config::{PgoPipelineConfig, PipelineConfig};

// Re-exported so `kornia_slam::system::*` keeps resolving the state and policy
// types; `tracking` owns them.
pub use crate::tracking::{
    KeyframePolicy, SystemMode, SystemState, TrackingLossRecoveryPolicy, TrackingResult,
    TrackingStatus,
};

use crate::tracking::local_map::{LocalMapSelectionConfig, select_local_map_points};
use crate::tracking::motion::{InertialPrediction, predict_pose};
use crate::tracking::tracker::{FrameInput, Tracker};

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::Frame;
use crate::initialization::bootstrap::{
    BootstrapDecision, MIN_KEYPOINTS_FOR_BOOTSTRAP, evaluate_bootstrap,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopCloser, LoopClosingContext};
use crate::map::{Keyframe, KeyframeJob, LocalMapping, Map, MapPoint};
use crate::mapping::growth::{fuse_into_neighbors, grow_map_points_from_keyframe_pair};
use crate::place_recognition::{KeyFrameDatabase, Vocabulary, compute_bow};
use crate::pose_conversion::apply_reference_pose_correction;
use crate::sensor_rig::{ImuCalibration, SensorRig};
use crate::stereo::unproject_stereo;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};

/// Top-level ORB-SLAM system: orchestrates tracking, mapping, and state transitions.
pub struct SlamSystem {
    // Camera model
    rig: SensorRig,
    // Primary pose estimator
    tracker: Tracker,
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
    inertial: InertialState,
    // Camera-to-body extrinsic T_BC (X_body = T_BC * X_cam). IMU deltas live in
    // the body frame, so every place that mixes them with camera poses must go
    inertial_init_start_kf_idx: Option<usize>,
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
    // Place recognition: bag-of-words vocabulary (None disables loop detection)
    // and the inverted-index keyframe database queried at each keyframe insert.
    vocabulary: Option<Vocabulary>,
    kf_database: KeyFrameDatabase,
    loop_closer: Option<LoopCloser>,
    loop_closure_events: Vec<LoopClosureEvent>,
    // System state
    state: SystemState,
}

pub use crate::loop_closure::LoopClosureEvent;

impl SlamSystem {
    /// Creates a new system with identity pose.
    pub fn new(camera: PinholeCamera, config: SlamConfig) -> Self {
        Self::with_rig(SensorRig::new(camera), config)
    }

    /// Builds a system from an explicit sensor rig, so IMU noise parameters and
    /// extrinsics can be supplied for the actual sensor.
    pub fn with_rig(rig: SensorRig, config: SlamConfig) -> Self {
        let camera = rig.camera.clone();
        let map = Arc::new(Mutex::new(Map::new()));
        let local_mapping =
            LocalMapping::new(config.local_mapping, Arc::clone(&map), camera.clone());
        let map_publication_gate = local_mapping.publication_gate();
        let loop_closer = config.pgo.map(LoopCloser::new);
        Self {
            rig,
            tracker: Tracker::new(config.map_projection),
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
            inertial: InertialState::new(),
            inertial_init_start_kf_idx: None,
            inertial_init_last_attempt_sec: None,
            imu_init_window_start_sec: None,
            imu_viba1_done: false,
            imu_viba2_done: false,
            vocabulary: None,
            kf_database: KeyFrameDatabase::new(),
            loop_closer,
            loop_closure_events: Vec::new(),
        }
    }

    /// Enables appearance-based loop detection with a bag-of-words vocabulary.
    /// Without it, keyframes are not indexed and no loop candidates are emitted.
    pub fn set_vocabulary(&mut self, vocabulary: Vocabulary) {
        self.vocabulary = Some(vocabulary);
    }

    pub fn drain_loop_closure_events(&mut self) -> Vec<LoopClosureEvent> {
        std::mem::take(&mut self.loop_closure_events)
    }

    /// Enables the inertial path by providing the camera-to-body extrinsic
    /// `T_BC` (`X_body = T_BC * X_cam`). Without it, IMU samples are ignored.
    pub fn set_imu_extrinsics(&mut self, t_bc: Pose3d) {
        self.rig.imu = Some(ImuCalibration::new(t_bc));
    }

    /// The system's fixed sensor calibration.
    pub fn rig(&self) -> &SensorRig {
        &self.rig
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
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

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
        self.state.mode = if self.rig.camera_to_body().is_some() {
            self.inertial_init_start_kf_idx = Some(curr_idx);
            self.imu_init_window_start_sec = Some(timestamp_sec);
            self.imu_viba1_done = false;
            self.imu_viba2_done = false;
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
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
        let cam_points = unproject_stereo(&curr_kf.frame, &self.rig.camera);
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

        let decision = evaluate_bootstrap(
            self.state.bootstrap_frame.as_ref(),
            &curr_frame,
            &self.rig.camera,
            &self.two_view_init_config,
        );
        let two_view_estimate = match decision {
            // A stale reference is dropped with the frame: neither is viable.
            BootstrapDecision::Unusable { keypoints } => {
                self.dbg(format!(
                    "[bootstrap] frame={} skip: too few keypoints ({}, need > {})",
                    curr_frame.idx, keypoints, MIN_KEYPOINTS_FOR_BOOTSTRAP,
                ));
                self.state.bootstrap_frame = None;
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            BootstrapDecision::StoreAsReference => {
                self.dbg(format!(
                    "[bootstrap] frame={} stored as reference (awaiting second frame)",
                    curr_frame.idx,
                ));
                self.state.bootstrap_frame = Some(curr_frame);
                self.inertial.bootstrap_timestamp_sec = Some(timestamp_sec);
                // Samples before the reference frame can never enter an edge.
                self.prune_imu_before(timestamp_sec);
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            // The reference is kept: only the second frame was unsuitable.
            BootstrapDecision::Rejected {
                reference_idx,
                reason,
            } => {
                self.dbg(format!(
                    "[bootstrap] frame={} (ref={}) reject: {:?}",
                    curr_frame.idx, reference_idx, reason,
                ));
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            BootstrapDecision::Initialized(estimate) => *estimate,
        };
        let prev_bootstrap_frame = self
            .state
            .bootstrap_frame
            .take()
            .expect("an accepted bootstrap consumes the reference it matched against");

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

        if let Some(prev_ts) = self.inertial.bootstrap_timestamp_sec {
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
        self.state.mode = if self.rig.camera_to_body().is_some() {
            self.inertial_init_start_kf_idx = Some(curr_idx);
            self.imu_init_window_start_sec = Some(timestamp_sec);
            self.imu_viba1_done = false;
            self.imu_viba2_done = false;
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);

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

        // The two-view result can name one feature twice — two triangulated
        // points landing on the same keypoint in either view. A feature holds
        // one landmark, so the first claim wins, as in pair growth; a refused
        // point reserves neither feature.
        let mut claimed_ref: HashSet<usize> = HashSet::new();
        let mut claimed_curr: HashSet<usize> = HashSet::new();
        triangulated.retain(|&(_, _, _, ref_desc_idx, curr_desc_idx)| {
            if claimed_ref.contains(&ref_desc_idx) || claimed_curr.contains(&curr_desc_idx) {
                return false;
            }
            claimed_ref.insert(ref_desc_idx);
            claimed_curr.insert(curr_desc_idx);
            true
        });

        let added = self.map.lock().unwrap().add_triangulated_points(
            Some(&mut reference_kf),
            &mut current_kf,
            &triangulated,
        );

        let reference_kf_idx = reference_kf.frame.idx;
        let current_kf_idx = current_kf.frame.idx;
        self.map.lock().unwrap().upsert_keyframe(reference_kf);
        self.map.lock().unwrap().upsert_keyframe(current_kf);

        self.map.lock().unwrap().run_initial_ba(&self.rig.camera);

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
    /// this returns, `prune_imu_before` is free to drop them from the buffer,
    /// since the edge now carries its own copy.
    fn preintegrate_window(&self, t0: f64, t1: f64) -> (PreintegratedImu, Vec<ImuMeasurement>) {
        self.inertial
            .preintegrate_window(self.rig.imu_noise(), t0, t1)
    }

    /// Drops buffered IMU samples strictly older than `t` (typically the last
    /// keyframe timestamp: the next edge and all per-frame windows start there).
    fn prune_imu_before(&mut self, t: f64) {
        self.inertial.prune_before(t);
    }

    /// Body-to-world pose `T_WB` for a world-to-camera pose, via
    /// `T_WB = T_WC ∘ T_CB`. Treats camera == body when no extrinsic is set.
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
                self.inertial.initializer.config.min_keyframes,
                imu_time,
                self.inertial.initializer.config.min_time_sec,
            );
            self.dbg(gate_msg);
        }

        let due_for_retry = due_for_retry(self.inertial_init_last_attempt_sec, timestamp_sec);
        let imu_init_ready = self
            .inertial
            .initializer
            .ready(&self.map.lock().unwrap(), self.inertial_init_start_kf_idx);

        if result.status == TrackingStatus::KeyframeAccepted && due_for_retry && imu_init_ready {
            let Some(start_idx) = self.inertial_init_start_kf_idx else {
                return result;
            };
            self.inertial_init_last_attempt_sec = Some(timestamp_sec);
            let is_mono = !self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .find(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.is_stereo())
                .unwrap_or(false);
            let prior_a0 = viba0_accel_bias_prior(is_mono);
            // Drop the solve's map lock before applying its result with a new lock.
            let init_result = self.inertial.initializer.try_initialize(
                &self.map.lock().unwrap(),
                self.rig.camera_to_body(),
                self.inertial.bias,
                start_idx,
                1e2,
                prior_a0,
                false,
            );
            match init_result {
                Some(init) => {
                    let applied = self.inertial.apply_initialization(
                        &mut self.map.lock().unwrap(),
                        &mut self.state,
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
                        imu_t_bc: self.rig.camera_to_body(),
                        gravity_world: self.inertial.gravity_world,
                    }) {
                        self.dbg("[local_mapping] worker is unavailable".into());
                    }
                    self.apply_local_mapping_results();
                    let AppliedInitialization {
                        scale,
                        gravity_world: gravity,
                        gyro_bias: bg,
                    } = applied;
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
            self.inertial.bias = kf.imu_bias;
        }

        // Preintegration is prepared here: the system owns bias and the sample
        // buffer. The motion model chooses between it and the visual model.
        let preintegrated = (self.state.imu_initialized && prev_timestamp > 0.0)
            .then(|| self.preintegrate_window(prev_timestamp, timestamp_sec).0);
        let (candidate_pose, predicted_velocity) = predict_pose(
            pose_before,
            self.state.velocity,
            self.state.velocity_world,
            preintegrated
                .as_ref()
                .map(|preintegrated| InertialPrediction {
                    preintegrated,
                    camera_to_body: self.rig.camera_to_body(),
                    gravity_world: self.inertial.gravity_world,
                }),
        );
        if let Some(velocity_world) = predicted_velocity {
            self.state.velocity_world = velocity_world; // propagate for next frame
        }

        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let result = {
            let map = self.map.lock().unwrap();
            self.tracker.estimate(
                FrameInput {
                    frame: &frame,
                    previous_image,
                    current_image,
                    candidate_pose,
                    pose_before,
                    current_keyframe_idx: self.state.current_keyframe_idx,
                    lost_for_sec: currently_lost_for,
                },
                &map,
                &self.rig.camera,
            )
        };

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

                (
                    TrackingStatus::Tracked,
                    estimate.matches,
                    estimate.inliers,
                    None,
                )
            }
            Err(reason) => {
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
                let local_indices = select_local_map_points(
                    &map_guard,
                    &matches,
                    current_kf,
                    LocalMapSelectionConfig::default(),
                );
                let visible = map_guard.map_points_in_frustum(
                    &local_indices,
                    &self.rig.camera,
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
                self.tracker.reset_tracks();
                self.state.reset();
                return self.bootstrap_step(frame, timestamp_sec);
            }
        } else {
            self.state.lost_since_sec = None;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        // Samples older than the last keyframe can't enter any future window
        // (the next edge and all per-frame predictions start at or after it).
        if let Some(kf_ts) = self.inertial.last_keyframe_timestamp_sec {
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
            curr_kf.imu_bias = self.inertial.bias;
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
            total_grown += grow_map_points_from_keyframe_pair(
                &self.map,
                &self.rig.camera,
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
            self.inertial.last_keyframe_timestamp_sec,
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

        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);

        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Forward SearchInNeighbors / Fuse: extend each curr_kf-observed map
        // point's observation list to neighbor KFs that don't yet observe it.
        // Run before local BA so BA sees the extra reprojection constraints.
        let n_fused =
            fuse_into_neighbors(&self.map, &self.rig.camera, frame.idx, &neighbor_kf_indices);
        self.dbg(format!("[fuse] frame={} fused={}", frame.idx, n_fused));

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            self.refine_inertial_init(timestamp_sec);
        }

        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized,
            imu_t_bc: self.rig.camera_to_body(),
            gravity_world: self.inertial.gravity_world,
        }) {
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
        let init_result = self.inertial.initializer.try_initialize(
            &self.map.lock().unwrap(),
            self.rig.camera_to_body(),
            self.inertial.bias,
            start_idx,
            prior_g,
            prior_a,
            true,
        );
        match init_result {
            Some(init) => {
                let scale = init.scale;
                let bg = init.bias.gyro;
                self.inertial.initializer.apply_initialization(
                    &mut self.map.lock().unwrap(),
                    &mut self.state,
                    &mut self.inertial.bias,
                    &mut self.inertial.gravity_world,
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

    /// Indexes a freshly inserted keyframe for place recognition and queries the
    /// database for appearance-based loop candidates.
    ///
    /// Mirrors ORB-SLAM3's `LoopClosing::DetectLoop`: the acceptance threshold is
    /// the lowest BoW similarity to a covisible neighbour, and the covisibility
    /// set is excluded so only a revisited place can match. The query runs before
    /// this keyframe is added, so it never matches itself.
    fn register_place_recognition(&mut self, kf_idx: usize) {
        let Some(vocabulary) = self.vocabulary.as_ref() else {
            return;
        };
        const MIN_COVIS_WEIGHT: usize = 15;
        let (bow, neighbors) = {
            let map = self.map.lock().unwrap();
            let Some(kf) = map.get_keyframe(kf_idx) else {
                return;
            };
            let bow = compute_bow(vocabulary, &kf.frame.features.descriptors);
            if bow.0.is_empty() {
                return;
            }
            let neighbors = map.covisible_keyframes(kf_idx, MIN_COVIS_WEIGHT);
            (bow, neighbors)
        };
        let candidates = self.kf_database.detect_loop_candidates(
            kf_idx,
            &bow,
            neighbors.iter().map(|&(nb_idx, _w)| nb_idx),
        );
        self.kf_database.add(kf_idx, bow);

        if let Some(best) = candidates.first().copied() {
            self.dbg(format!(
                "[loop] kf={kf_idx} matched kf={} score={:.3} shared_words={} ({} candidates)",
                best.kf_idx,
                best.score,
                best.shared_words,
                candidates.len()
            ));
        }

        let Some(loop_closer) = self.loop_closer.as_mut() else {
            return;
        };
        if loop_closer.requires_imu_initialized() && !self.state.imu_initialized {
            return;
        }
        let context = LoopClosingContext {
            pose_world_to_cam: self.state.pose_world_to_cam,
            velocity_world: self.state.velocity_world,
            current_keyframe_idx: self.state.current_keyframe_idx,
            imu_initialized: self.state.imu_initialized,
            gravity_world: self.inertial.gravity_world,
        };
        let outcome = {
            let mut map = self.map.lock().unwrap();
            loop_closer.close(&mut map, &self.rig.camera, kf_idx, &candidates, context)
        };
        if let Some((corrected_tracking_pose, corrected_velocity)) = outcome.tracking_correction {
            self.state.pose_world_to_cam = corrected_tracking_pose;
            self.state.velocity_world = corrected_velocity;
        }
        if outcome.pgo_applied {
            self.tracker.reset_tracks();
            if self.state.imu_initialized {
                if !self.local_mapping.submit(KeyframeJob {
                    imu_initialized: true,
                    imu_t_bc: self.rig.camera_to_body(),
                    gravity_world: self.inertial.gravity_world,
                }) {
                    self.dbg("[local_mapping] worker is unavailable after PGO".into());
                }
                self.apply_local_mapping_results();
            }
        }
        self.loop_closure_events.extend(outcome.events);
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

#[cfg(test)]
mod tests {
    use super::format_imu_init_gate;

    #[test]
    fn formats_compact_imu_init_gate() {
        assert_eq!(
            format_imu_init_gate(12, Some(12), Some(32), 7, 10, 1.05, 1.0),
            "[imu_init_gate] start_idx=12 first_idx=Some(12) last_idx=Some(32) kfs=7/10 imu_time=1.05/1.0s"
        );
    }
}
