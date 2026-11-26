//! SLAM runtime: its configuration, construction and processing flow.
//!
//! The processing flow is kept in this file so it can be read from top to
//! bottom in the same order frames move through the system. With the `serde`
//! feature, configurations load from versioned RON files.

mod build;
mod config;
mod counters;
mod estimator;
pub(crate) mod fan_out;
#[cfg(feature = "serde")]
mod file;
mod innovation;
mod input;
mod odometry;
mod recovery;
mod state;
mod stillness;
mod trace;
mod validation;
pub mod wheel;

pub use crate::odometry::{
    Odometry, OdometryCounters, OdometryImuNoise, OdometryMode, OdometryOutput, OdometryTuning,
    SlidingWindowConfig, VisualJacobians,
};
pub use build::BuildError;
pub use config::{
    BoundedFanOut, CameraSelection, FanOut, FlowBookkeeping, FlowPose, ForwardBackward,
    FrameVelocity, FrontendConfig, GyroGuess, InertialConfig, KltFirst, KltPrediction, KltRetry,
    KltRig, LocalBaConfig, LoopClosingExecution, LoopClosingMode, LostRelocalization,
    MAX_FLOW_THREADS, MappingConfig, OnlineCalibrationConfig, OrbFrontendConfig, OrbSlamPipeline,
    OrbTuning, PipelineDefinition, PoseInertial, PoseInertialMinInliers, SYSTEM_CONFIG_VERSION,
    SensorSelection, SnapbackFix, StereoCloseDepth, StereoDepthSigma, SystemConfig, TrackRefill,
    TrackingConfig, TrackingLossConfig,
};
pub use counters::{RunCounters, SkipKind, StillnessCounters};
#[cfg(feature = "serde")]
pub use file::LoadError;
pub use input::ProcessError;
pub use recovery::{RecoveryConfig, SeedPlacement};
pub use state::{TrackingResult, TrackingStatus};
pub use trace::{FrameTrace, TraceMapPoint, TraceTrack};
pub use validation::ConfigError;

use crate::frontend::{FrontendObservation, OrbFrontend};
use fan_out::TrackingFanOut;

use state::{SystemMode, SystemState};

use crate::tracking::{
    KeyframeAsks, KeyframeEvidence, KeyframePolicy, KeyframesWhileLost, TrackingLossRecoveryPolicy,
    median_parallax_px,
};

use crate::tracking::TrackingTimings;
use crate::tracking::flow_tracker::{
    FlowStepInput, RefillCameras, RigFlow, RigFlowCamera, RigFlowSettings,
};
use crate::tracking::local_map::{LocalMapRules, select_local_map_points};
use crate::tracking::motion::{
    InertialPrediction, body_to_world, predict_pose, propagate_keyframe_velocity,
};
use crate::tracking::optical_flow::{FlowSurvivor, LkKernel};
use crate::tracking::pose_estimation::Estimate;
use crate::tracking::pose_estimation::map_projection::MapProjectionRejectReason;
use crate::tracking::pose_gate::GateReference;
use crate::tracking::tracker::{FlowGuess, FrameInput, GateReading, Tracker};

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::Frame;
use crate::inertial::trust::{ImuTrust, ImuTrustMode};
use crate::inertial::{
    AppliedInitialization, ImuInitResult, InertialState, viba0_accel_bias_prior,
};
use crate::initialization::bootstrap::{
    self, BootstrapDecision, MIN_KEYPOINTS_FOR_BOOTSTRAP, MIN_VALID_POINTS, TooFewStereoPoints,
    evaluate_bootstrap,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{
    Correction, KeyframeGap, LoopCloser, LoopClosingContext, LoopClosingOutcome, LoopClosureEvent,
    LoopTiming,
};
use crate::mapping::bundle_adjustment::LocalBaParams;
use crate::mapping::keyframe_mapping::{
    self, KeyframeGrowthResult, KeyframeInsertion, TrackedMatches,
};
use crate::mapping::local_mapping::GrowthJob;
use crate::mapping::map::{
    InertialAlignmentError, Keyframe, Map, MapInsertion, MapMutationError, MapPoint,
    StateEnvelopeLimits,
};
use crate::mapping::session::{SessionKeyframe, SessionLandmark, snapshot_map};
use crate::mapping::{GrowthNeighbours, KeyframeGrowth, KeyframeJob, LocalMapping, SegmentId};
use crate::pose_conversion::{RIGID_TOLERANCE, angle_deg, apply_reference_pose_correction};
use build::SystemSettings;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Mat9F64, Vec3F64};
use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::{ImuMeasurement, SampleInterpolation};
use recovery::{BridgeAnchor, RunawayDetector, Segments, TiltMonitor};

/// The images of one capture, kept until its features are extracted.
#[derive(Clone, Copy)]
enum Capture<'a> {
    Mono(&'a Image<u8, 1>),
    Stereo(&'a Image<u8, 1>, &'a Image<u8, 1>),
    Rig(&'a [&'a Image<u8, 1>]),
}

/// A capture numbered as frame `idx`, not yet extracted.
#[derive(Clone, Copy)]
struct PendingFrame<'a> {
    idx: usize,
    capture: Capture<'a>,
}

/// A frame's features: extracted, or deferred until tracking needs them.
enum FrameSource<'a> {
    Ready(Box<Frame>),
    Deferred(PendingFrame<'a>),
}

impl FrameSource<'_> {
    fn idx(&self) -> usize {
        match self {
            Self::Ready(frame) => frame.idx,
            Self::Deferred(pending) => pending.idx,
        }
    }
}

/// Per-frame estimation inputs from the motion model and loss state.
struct Estimation {
    candidate_pose: Pose3d,
    pose_before: Pose3d,
    lost_for_sec: f64,
    trusted_prediction: bool,
    gate: Option<GateReference>,
}

/// What a new keyframe is to the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyframeRole {
    Tracked,
    /// Inserted at the inertial prediction while lost, as a full BA state.
    Lost,
    /// Inserted at the inertial prediction while lost and held fixed
    /// ([`KeyframesWhileLost::Anchor`]).
    Anchor,
}

/// Where a frame's starting velocity came from, for the `[vel]` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameVelocitySource {
    /// The reference keyframe's velocity.
    Keyframe,
    /// [`FrameVelocity::PropagatedFromKeyframe`].
    Propagated,
    /// Left as the previous frame's prediction wrote it.
    Carried,
}

impl FrameVelocitySource {
    fn label(self) -> &'static str {
        match self {
            Self::Keyframe => "kf",
            Self::Propagated => "prop",
            Self::Carried => "carried",
        }
    }
}

/// What tracking a deferred frame from its flow survivors alone gave.
enum FlowOnly {
    /// Too few survivors with a map point, or no flow at all.
    Unavailable,
    /// The flow-only pose was not accepted.
    Refused,
    /// The flow-only pose was accepted, but the frame is due as a keyframe,
    /// which needs ORB; the pose seeds the full estimate.
    KeyframeDue(Pose3d),
    /// Tracked without ORB, on this pseudo frame.
    Tracked(Box<Frame>, Estimate),
}

/// Locks `map`, adding the wait to `wait`.
fn lock_map<'a>(map: &'a Mutex<Map>, wait: &Cell<Duration>) -> MutexGuard<'a, Map> {
    let started = Instant::now();
    let guard = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    wait.set(wait.get() + started.elapsed());
    guard
}

/// Top-level ORB-SLAM system: orchestrates tracking, mapping, and state transitions.
pub struct SlamSystem {
    rig: SensorRig,
    tracker: Tracker,
    two_view_init_config: TwoViewInitConfig,
    keyframe_policy: KeyframePolicy,
    tracking_loss_recovery: TrackingLossRecoveryPolicy,
    keyframes_while_lost: KeyframesWhileLost,
    recovery: RecoveryConfig,
    runaway: RunawayDetector,
    tilt: TiltMonitor,
    segments: Segments,
    /// When the current segment's bootstrap finished, for the young-death count.
    segment_begun_sec: Option<f64>,
    /// A segment has lived [`counters::YOUNG_SEGMENT_SEC`]: give-ups of young
    /// segments before that are start-up resets, not deaths after a loss.
    startup_over: bool,
    /// The segment the first inertial alignment was applied in: a later
    /// segment's alignment rotates that already gravity-aligned world.
    first_aligned_segment: Option<SegmentId>,
    relocalization: LostRelocalization,
    tracking_config: TrackingConfig,
    growth_neighbours: GrowthNeighbours,
    growth_min_baseline_ratio: f64,
    keyframe_growth: KeyframeGrowth,
    // The newest keyframe whose queued growth has come back (`KeyframeGrowth::Mapping`).
    growth_done_through: Option<usize>,
    // This frame's wait for the map lock; a `Cell` so `&self` paths can add to it.
    map_wait: Cell<Duration>,
    sample_interpolation: SampleInterpolation,
    // Keyframes a map bound evicted, copied while a session is recorded.
    session_recording: bool,
    evicted_keyframes: Vec<Keyframe>,
    // `(frame idx, timestamp)` of every frame tracked while recording: a
    // keyframe holds no stamp of its own.
    session_frame_timestamps: Vec<(usize, f64)>,
    /// The current re-seed holds its translation at the last tracked
    /// position; the attitude still follows the gyro.
    reseed_held: bool,
    /// Start of the current re-seed's IMU-only stretch: the bridge anchor, or
    /// the give-up when there is none.
    reseed_coast_from_sec: f64,
    // mThDepth (metres): back-project close stereo points at each keyframe when set
    stereo_close_depth: Option<f64>,
    debug: bool,
    debug_messages: Vec<String>,
    // Under debug: when tracking first matched each landmark, by stable id.
    landmark_first_tracked_sec: HashMap<usize, f64>,
    map: Arc<Mutex<Map>>,
    // Serializes compound map publication and short local-BA snapshot/merge phases.
    map_publication_gate: Option<Arc<Mutex<()>>>,
    // IMU estimates, sample buffer and the inertial-initialization schedule
    inertial: InertialState,
    local_mapping: LocalMapping,
    local_ba: LocalBaParams,
    // Place recognition for every keyframe, and loop closing when configured.
    loop_closer: Option<LoopCloser>,
    loop_closure_events: Vec<LoopClosureEvent>,
    // How the next keyframe joins the pose graph after tracking resumed across a jump.
    pending_graph_gap: Option<KeyframeGap>,
    state: SystemState,
    frontend: OrbFrontend,
    fan_out: TrackingFanOut,
    // The last processed image, for optical-flow tracks into the next frame.
    previous_image: Option<Image<u8, 1>>,
    // Rig cameras 1 and up of the last processed capture, kept only when
    // today's kernels track every rig camera (`klt_rig: AllCameras`).
    previous_secondary: Vec<Image<u8, 1>>,
    tracking_duration: Duration,
    tracking_timings: TrackingTimings,
    trace_enabled: bool,
    frame_trace: Option<FrameTrace>,
    // Flow survivors with a map point needed to skip ORB; None: ORB every frame.
    klt_min_tracks: Option<usize>,
    // Flow search (max_level, max_iter) from a trusted prediction.
    klt_prediction: Option<(usize, usize)>,
    deferred_extraction: Duration,
    // Number given to the next frame; it becomes the keyframe's id in the map.
    next_frame_idx: usize,
    counters: RunCounters,
    last_skip: Option<SkipKind>,
    // Wheel-odometry readings and the keyframe edges built from them, when enabled.
    wheel: Option<wheel::WheelState>,
    // Card A's per-frame estimator state.
    estimator: estimator::EstimatorState,
    // The keyframe bound and the culling that enforces it first, when set.
    map_bound: Option<(usize, crate::mapping::culling::KeyframeCullingConfig)>,
    // `WindowedU8` flow: level-0 frames, shards and workers of the 1 ms
    // frontend; `None` on today's kernels.
    rig_flow: Option<RigFlow>,
    flow: build::FlowSettings,
    // The primary camera's `R_cur_prev` from this frame's preintegrated gyro
    // (`GyroGuess::RotatedRay`).
    gyro_rotation: Option<Mat3F64>,
    // The keyframe whose landmarks were last adopted (`adopt_landmarks`).
    adopted_keyframe: Option<usize>,
    // First keyframe of the re-seeded segment the own window holds; cleared
    // once it is full or a relocalization is adopted (G2).
    window_segment_start: Option<usize>,
    // The still run since the last keyframe (`keyframes.stillness`, card AB).
    stillness: stillness::StillnessState,
    // `odometry: SlidingWindow` beside the pipeline (Observe).
    odometry: Option<odometry::OdometryRunner>,
}

impl SlamSystem {
    /// Builds the system `config` describes from the source's calibrated rig,
    /// restricted to the selected sensors. Resources such as the vocabulary
    /// are loaded before any mapping worker starts.
    pub fn build(config: SystemConfig, rig: SensorRig) -> Result<Self, BuildError> {
        config.validate()?;
        let rig = config.sensors.select_rig(rig)?;
        config.validate_flow_threads(&rig)?;
        config.validate_raw_fisheye(&rig)?;
        let settings = config.settings(&rig);
        let frontend = OrbFrontend::new(config.orb_detector(), &rig)
            .with_fisheye_weight(config.fisheye_weight(&rig));
        let loop_closer = config.loop_closer(&rig)?;
        let PipelineDefinition::OrbSlam(orb) = &config.pipeline;
        let odometry =
            odometry::OdometryRunner::build(orb, &rig, settings.inertial.sample_interpolation)?;
        let fan_out = TrackingFanOut::new(settings.fan_out)?;
        let rig_flow = match settings.klt_kernel {
            LkKernel::WindowedU8 => Some(RigFlow::new(
                rig_flow_settings(&settings),
                rig_flow_cameras(&rig, &frontend, settings.frontend_flow.rig),
            )?),
            LkKernel::F32 | LkKernel::FixedPointU8 => None,
        };
        let mut system = Self::assemble(rig, settings, frontend, loop_closer, rig_flow, fan_out);
        system.odometry = odometry;
        Ok(system)
    }

    fn assemble(
        rig: SensorRig,
        settings: SystemSettings,
        frontend: OrbFrontend,
        loop_closer: Option<LoopCloser>,
        rig_flow: Option<RigFlow>,
        fan_out: TrackingFanOut,
    ) -> Self {
        let camera = rig.camera.clone();
        let flow = settings.frontend_flow;
        let mut tracker = Tracker::new(settings.map_projection)
            .with_lk_kernel(settings.klt_kernel)
            .with_lk_window(flow.window)
            .with_track_refill(flow.refill)
            .with_pose_gate(settings.pose_gate)
            .with_local_map_rules(settings.tracking.local_map)
            .with_relocalization_solver(settings.relocalization.solver);
        if flow.rig == KltRig::AllCameras
            && rig.is_multi_camera()
            && let Some(geometry) = frontend.rig_geometry()
        {
            tracker = tracker.with_rig_flow(Arc::clone(geometry), flow.threads);
        }
        let map = Arc::new(Mutex::new(Map::new()));
        let local_mapping = LocalMapping::with_fan_out(
            settings.local_mapping,
            Arc::clone(&map),
            camera.clone(),
            settings.fan_out,
        );
        let map_publication_gate = local_mapping.publication_gate();
        Self {
            rig,
            tracker,
            two_view_init_config: settings.two_view_init,
            keyframe_policy: settings.keyframe_policy,
            tracking_loss_recovery: settings.tracking_loss_recovery,
            keyframes_while_lost: settings.keyframes_while_lost,
            recovery: settings.recovery,
            runaway: RunawayDetector::new(
                recovery::RUNAWAY_WINDOW_SEC,
                recovery::MAX_COAST_SPEED_MPS,
            ),
            tilt: TiltMonitor::default(),
            segments: Segments::default(),
            segment_begun_sec: None,
            startup_over: false,
            first_aligned_segment: None,
            relocalization: settings.relocalization,
            tracking_config: settings.tracking,
            growth_neighbours: settings.growth_neighbours,
            growth_min_baseline_ratio: settings.growth_min_baseline_ratio,
            keyframe_growth: settings.keyframe_growth,
            growth_done_through: None,
            map_wait: Cell::new(Duration::ZERO),
            sample_interpolation: settings.inertial.sample_interpolation,
            session_recording: false,
            evicted_keyframes: Vec::new(),
            session_frame_timestamps: Vec::new(),
            reseed_held: false,
            reseed_coast_from_sec: 0.0,
            stereo_close_depth: settings.stereo_close_depth_m,
            debug: false,
            debug_messages: Vec::new(),
            landmark_first_tracked_sec: HashMap::new(),
            map,
            map_publication_gate,
            local_mapping,
            local_ba: settings.local_ba,
            state: SystemState::new(),
            inertial: InertialState::with_sample_interpolation(
                settings.inertial.sample_interpolation,
            ),
            loop_closer,
            loop_closure_events: Vec::new(),
            pending_graph_gap: None,
            frontend,
            fan_out,
            previous_image: None,
            previous_secondary: Vec::new(),
            tracking_duration: Duration::ZERO,
            tracking_timings: TrackingTimings::default(),
            trace_enabled: false,
            frame_trace: None,
            klt_min_tracks: settings.klt_min_tracks,
            klt_prediction: settings.klt_prediction,
            deferred_extraction: Duration::ZERO,
            next_frame_idx: 0,
            counters: RunCounters::default(),
            last_skip: None,
            wheel: None,
            estimator: estimator::EstimatorState::default(),
            map_bound: None,
            rig_flow,
            flow,
            gyro_rotation: None,
            window_segment_start: None,
            stillness: stillness::StillnessState::default(),
            adopted_keyframe: None,
            odometry: None,
        }
    }

    /// Holds the map to `max_keyframes` after each keyframe: redundant keyframes are culled first,
    /// then the oldest evicted (see [`crate::mapping::culling::bound_keyframes`]). Unbounded until
    /// called; a bounded map trades loops past the bound for a per-frame cost that stops growing.
    pub fn set_map_bound(
        &mut self,
        max_keyframes: usize,
        culling: crate::mapping::culling::KeyframeCullingConfig,
    ) {
        self.map_bound = Some((max_keyframes, culling));
    }

    /// Records a session for an offline pass: each keyframe a map bound removes
    /// is copied first, so a dump holds every keyframe made. The online map
    /// behaves as without it.
    pub fn set_session_recording(&mut self, on: bool) {
        self.session_recording = on;
    }

    /// Keyframes a map bound removed while recording, oldest first.
    pub fn evicted_keyframes(&self) -> &[Keyframe] {
        &self.evicted_keyframes
    }

    /// The map segment keyframe `kf_idx` was seeded in.
    pub fn segment_of(&self, kf_idx: usize) -> SegmentId {
        self.segments.of(kf_idx)
    }

    /// The recorded session's keyframes (live, then evicted) and live
    /// landmarks; a keyframe tracked before recording began is stamped NaN.
    pub fn session_snapshot(&self) -> (Vec<SessionKeyframe>, Vec<SessionLandmark>) {
        let stamps = &self.session_frame_timestamps;
        let timestamp_of = |idx: usize| {
            stamps
                .binary_search_by_key(&idx, |&(i, _)| i)
                .map_or(f64::NAN, |slot| stamps[slot].1)
        };
        let map = lock_map(&self.map, &self.map_wait);
        snapshot_map(
            &map,
            &self.rig,
            &self.evicted_keyframes,
            |kf| self.segment_of(kf),
            timestamp_of,
        )
    }

    /// Gravity in the world frame (m/s²): `+Y` once the inertial
    /// initialization has aligned the map.
    pub fn gravity_world(&self) -> Vec3F64 {
        self.inertial.gravity_world
    }

    /// Runs inside a frame, which already holds the map publication gate.
    fn enforce_map_bound(&mut self) {
        let Some((max_keyframes, culling)) = self.map_bound.clone() else {
            return;
        };
        let report = {
            let mut map = lock_map(&self.map, &self.map_wait);
            // Only a map above the bound loses keyframes, so only then is a copy owed.
            let before = (self.session_recording && map.keyframes().len() > max_keyframes)
                .then(|| map.keyframes().to_vec());
            let wheel_factors = self.wheel.as_mut().map(|w| w.factors_mut());
            let report = crate::mapping::culling::bound_keyframes(
                &mut map,
                max_keyframes,
                &culling,
                wheel_factors,
            );
            if let Some(before) = before {
                let removed: std::collections::HashSet<usize> = report.removed().collect();
                self.evicted_keyframes.extend(
                    before
                        .into_iter()
                        .filter(|kf| removed.contains(&kf.frame.idx)),
                );
            }
            report
        };
        if report.culled.is_empty() && report.evicted.is_empty() {
            return;
        }
        if let Some(loop_closer) = self.loop_closer.as_mut() {
            loop_closer.forget_keyframes(report.removed());
        }
        self.dbg(format!(
            "[map_bound] culled={} evicted={} (bound {max_keyframes})",
            report.culled.len(),
            report.evicted.len()
        ));
    }

    /// Adds wheel odometry to local BA: each keyframe interval the readings cover becomes a
    /// planar [`kornia_factors::wheel_odometry::WheelOdomFactor`]. Off until called, and with no
    /// effect on a system that is never given readings.
    pub fn enable_wheel_odometry(&mut self, config: wheel::WheelOdomConfig) {
        self.wheel = Some(wheel::WheelState::new(config));
    }

    /// Feeds wheel readings, stamped in nanoseconds on the same origin as the frame timestamps
    /// (`timestamp_sec * 1e9`). Returns how many were kept; ignored while wheels are off.
    pub fn process_wheel_odometry(
        &mut self,
        samples: impl IntoIterator<Item = wheel::WheelSample>,
    ) -> usize {
        self.wheel.as_mut().map_or(0, |w| w.push(samples))
    }

    /// The base-to-camera transform (`X_cam = t_cb * X_base`) from `stamp_ns` on, for a camera
    /// whose mount moves (a tilting head). Ignored while wheels are off.
    pub fn push_camera_mount(
        &mut self,
        stamp_ns: u64,
        t_cb: Pose3d,
    ) -> Result<(), wheel::WheelError> {
        match self.wheel.as_mut() {
            Some(w) => w.push_mount(stamp_ns, t_cb),
            None => Ok(()),
        }
    }

    fn wheel_factors(&self) -> Vec<kornia_factors::wheel_odometry::WheelOdomFactor> {
        self.wheel
            .as_ref()
            .map_or_else(Vec::new, |w| w.factors().to_vec())
    }

    /// Closes the wheel interval ending at keyframe `to_idx`, or queues it until the readings
    /// cover it. `prev` is the previous keyframe and its timestamp.
    fn note_wheel_keyframe(&mut self, prev: Option<(usize, f64)>, to_idx: usize, to_sec: f64) {
        let Some(w) = self.wheel.as_mut() else {
            return;
        };
        let to_ns = |t: f64| (t.max(0.0) * 1e9).round() as u64;
        let added = w.on_keyframe(prev.map(|(i, t)| (i, to_ns(t))), to_idx, to_ns(to_sec));
        for (f, path_m) in added {
            self.dbg(format!(
                "[wheel] edge {}->{} path={path_m:.3}m",
                f.from_idx, f.to_idx
            ));
        }
    }

    /// Tallies of frame outcomes and local-mapping work since [`Self::build`].
    pub fn run_counters(&self) -> RunCounters {
        let (klt_errors, snap_errors) = self.tracker.flow_error_counts();
        let (flow, refill_spawns) = self.tracker.flow_counts();
        let mapping = self.local_mapping.counts();
        RunCounters {
            klt_errors,
            snap_errors,
            ba_failed: mapping.ba_failed,
            vi_ba_failed: mapping.vi_ba_failed,
            prior_fallbacks: mapping.prior_fallbacks,
            ba_updates_rejected: mapping.ba_updates_rejected,
            ba_solves: mapping.ba_solves,
            ba_solve_ns: mapping.ba_solve_ns,
            large_window_solves: mapping.large_window_solves,
            vi_ba_base_solves: mapping.vi_ba_base_solves,
            vi_ba_base_iterations: mapping.vi_ba_base_iterations,
            vi_ba_iterations_max: mapping.vi_ba_iterations_max,
            vi_ba_rejected_steps: mapping.vi_ba_rejected_steps,
            vi_ba_solves_3plus_rejects: mapping.vi_ba_solves_3plus_rejects,
            ba_stale_keyframes: mapping.ba_stale_keyframes,
            ba_propagated_keyframes: mapping.ba_propagated_keyframes,
            ba_propagations_stopped_at_bridge: mapping.ba_propagations_stopped_at_bridge,
            erase_tested: mapping.erase_tested,
            erase_unlinked: mapping.erase_unlinked,
            erase_tested_newest: mapping.erase_tested_newest,
            erase_unlinked_newest: mapping.erase_unlinked_newest,
            erase_landmarks_retired: mapping.erase_landmarks_retired,
            erase_stale_links: mapping.erase_stale_links,
            flow_reconcile_clears: self.tracker.reconcile_clears(),
            flow_tracks_tried: flow.tracks_tried,
            flow_fb_rejects: flow.fb_rejects,
            flow_retries: flow.retries,
            flow_template_rebuilds: flow.template_rebuilds,
            flow_refill_spawns: refill_spawns,
            stillness: self.stillness.counters,
            ..self.counters
        }
    }

    /// Why the latest processed frame was skipped; `None` when it was tracked.
    pub fn last_skip(&self) -> Option<SkipKind> {
        self.last_skip
    }

    pub fn drain_loop_closure_events(&mut self) -> Vec<LoopClosureEvent> {
        std::mem::take(&mut self.loop_closure_events)
    }

    /// Where loop closing has moved the map relative to the world tracking drifted into;
    /// identity without loop closing or before the first correction.
    pub fn map_correction(&self) -> Correction {
        self.loop_closer.as_ref().map_or(
            Correction {
                map_from_odometry: Pose3d::IDENTITY,
            },
            LoopCloser::map_correction,
        )
    }

    /// Advances by one with every loop correction written to the map, so a
    /// consumer publishing odometry as `T_map * map_correction()` re-anchors
    /// when it changes; 0 without loop closing or before the first correction.
    pub fn correction_epoch(&self) -> u64 {
        self.loop_closer
            .as_ref()
            .map_or(0, LoopCloser::correction_epoch)
    }

    /// Whether a background loop solve is running or waiting for its
    /// writeback at a frame boundary.
    pub fn loop_solve_in_flight(&self) -> bool {
        self.loop_closer
            .as_ref()
            .is_some_and(LoopCloser::solve_in_flight)
    }

    /// The system's sensor calibration, restricted to the selected sensors.
    pub fn rig(&self) -> &SensorRig {
        &self.rig
    }

    /// Buffers IMU samples for the frames to come, in the body frame. Call it
    /// as samples arrive; each frame uses the samples up to its own
    /// timestamp. The caller must feed every sample up to an image's
    /// timestamp before passing that image: a frame processed ahead of its
    /// samples predicts from stale ones, and its keyframe's IMU edge stays
    /// short for good. Samples with a non-finite timestamp or more than 30 s
    /// ahead of the latest image are ignored, and so are all samples when the
    /// rig has no IMU.
    pub fn process_imu(&mut self, samples: &[ImuMeasurement]) {
        if self.rig.imu.is_none() {
            return;
        }
        self.inertial.insert_buffer_samples(samples);
        if let Some(odometry) = self.odometry.as_mut() {
            odometry.push_imu(samples);
        }
    }

    /// The sliding-window odometry's estimate of the last stereo pair, when
    /// `pipeline.odometry` is `SlidingWindow` and that frame got one.
    pub fn odometry_output(&self) -> Option<&OdometryOutput> {
        self.odometry.as_ref().and_then(|o| o.output())
    }

    /// The sliding-window odometry's run totals, when it runs.
    pub fn odometry_counters(&self) -> Option<&OdometryCounters> {
        self.odometry.as_ref().map(|o| o.counters())
    }

    /// Extracts features from one image captured at `timestamp_sec` (on the
    /// IMU samples' clock) and tracks it. The system numbers its frames
    /// itself, in call order. For a rig with a fisheye model the image is the
    /// raw fisheye image (see [`SensorRig`]).
    ///
    /// # Errors
    ///
    /// Invalid input, a stereo rig (use [`Self::process_stereo`]) or a
    /// feature-extraction failure. Invalid input leaves the system state,
    /// including image history, unchanged; so does an extraction failure,
    /// except one deferred by optical-flow-first tracking (see
    /// [`KltFirst`]), which comes after the motion prediction. IMU samples
    /// already buffered are kept.
    pub fn process_image(
        &mut self,
        timestamp_sec: f64,
        image: &Image<u8, 1>,
    ) -> Result<TrackingResult, ProcessError> {
        input::validate_image(timestamp_sec, image)?;
        if self.rig.stereo_baseline_m.is_some() {
            return Err(ProcessError::StereoRequired);
        }
        if self.rig.is_multi_camera() {
            return Err(ProcessError::RigRequired);
        }
        self.track_capture(Capture::Mono(image), image, timestamp_sec)
    }

    /// Like [`Self::process_image`] for a rectified stereo pair; a mono
    /// pipeline rejects it (use [`Self::process_image`]).
    pub fn process_stereo(
        &mut self,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<TrackingResult, ProcessError> {
        input::validate_stereo_pair(timestamp_sec, left, right)?;
        if self.rig.is_multi_camera() {
            return Err(ProcessError::RigRequired);
        }
        if self.rig.stereo_baseline_m.is_none() {
            return Err(ProcessError::MonoPipeline);
        }
        let result = self.track_capture(Capture::Stereo(left, right), left, timestamp_sec);
        // After the pipeline's frame, so `track_frame` and its timings are
        // untouched; it reads only its own state and these borrows.
        if let Some(odometry) = self.odometry.as_mut() {
            odometry.observe(timestamp_sec, left, right);
        }
        result
    }

    /// Like [`Self::process_image`] for one capture of a multi-camera rig:
    /// `images[c]` comes from rig camera `c` (see [`SensorRig::cameras`]),
    /// all taken at `timestamp_sec`. Other pipelines reject it.
    pub fn process_rig(
        &mut self,
        timestamp_sec: f64,
        images: &[&Image<u8, 1>],
    ) -> Result<TrackingResult, ProcessError> {
        if !self.rig.is_multi_camera() {
            return Err(ProcessError::NotARig);
        }
        input::validate_rig_images(timestamp_sec, images, &self.rig.cameras)?;
        let Some(&primary) = images.first() else {
            return Err(ProcessError::NotARig);
        };
        self.track_capture(Capture::Rig(images), primary, timestamp_sec)
    }

    /// Extracts the capture's features now, or defers extraction to tracking
    /// when optical flow may carry the frame alone.
    fn track_capture(
        &mut self,
        capture: Capture<'_>,
        image: &Image<u8, 1>,
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let pending = PendingFrame {
            idx: self.next_frame_idx,
            capture,
        };
        let primary_only = [image];
        let images: &[&Image<u8, 1>] = match capture {
            Capture::Rig(images) => images,
            Capture::Mono(_) | Capture::Stereo(..) => &primary_only,
        };
        let deferrable = self.klt_min_tracks.is_some()
            && self.state.mode != SystemMode::Bootstrap
            && (self.previous_image.is_some()
                || self.rig_flow.as_ref().is_some_and(RigFlow::has_frames));
        let source = if deferrable {
            FrameSource::Deferred(pending)
        } else {
            FrameSource::Ready(Box::new(self.extract(pending)?))
        };
        self.track_frame(source, images, timestamp_sec)
    }

    /// ORB features of a capture, undistorted, with the stereo report.
    fn extract(&mut self, pending: PendingFrame<'_>) -> Result<Frame, ProcessError> {
        let frontend = &mut self.frontend;
        let mut frame = self
            .fan_out
            .run(|| match pending.capture {
                Capture::Mono(image) => frontend.prepare_image(pending.idx, image),
                Capture::Stereo(left, right) => {
                    frontend.prepare_stereo_pair(pending.idx, left, right)
                }
                Capture::Rig(images) => frontend.prepare_rig(pending.idx, images),
            })
            .map_err(ProcessError::Frontend)?;
        self.report_stereo(&frame);
        // Fill the per-frame undistortion cache once; tracking, BA gathering,
        // growth, and fuse all read from it.
        frame.ensure_undistorted(&self.rig.camera);
        Ok(frame)
    }

    /// The frame `source` describes, extracting it if it was deferred; the
    /// extraction time is kept out of `tracking_duration`.
    fn resolve(&mut self, source: FrameSource<'_>) -> Result<Frame, ProcessError> {
        match source {
            FrameSource::Ready(frame) => Ok(*frame),
            FrameSource::Deferred(pending) => {
                let frame = self.extract(pending)?;
                self.deferred_extraction += self.frontend.observation().duration;
                Ok(frame)
            }
        }
    }

    /// Takes a prepared or deferred frame as the next one and tracks it;
    /// `images[0]` is the primary camera's image.
    fn track_frame(
        &mut self,
        source: FrameSource<'_>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let Some(&image) = images.first() else {
            return Err(ProcessError::NotARig);
        };
        self.next_frame_idx += 1;
        if self.session_recording {
            self.session_frame_timestamps
                .push((self.next_frame_idx - 1, timestamp_sec));
        }
        self.inertial.latest_image_timestamp_sec = Some(timestamp_sec);
        // Taken out for the call so tracking can borrow it alongside `self`.
        let previous_image = self.previous_image.take();
        self.tracking_timings = TrackingTimings::default();
        self.frame_trace = None;
        self.deferred_extraction = Duration::ZERO;
        self.map_wait.set(Duration::ZERO);
        let tracking_start = Instant::now();
        self.last_skip = None;
        self.prepare_flow(images, self.next_frame_idx as u64 - 1);
        let result = self.process_prepared(source, previous_image.as_ref(), images, timestamp_sec);
        self.tracking_duration = tracking_start
            .elapsed()
            .saturating_sub(self.deferred_extraction);
        self.tracking_timings.map_wait = self.map_wait.get();
        // An error leaves the frame uncounted, as before #184 made extraction fallible here.
        if let Ok(r) = &result {
            if r.status != TrackingStatus::Skipped {
                self.last_skip = None;
            }
            self.counters.record_frame(r.status, self.last_skip);
            if r.pose_world_to_cam.rotation_defect() > RIGID_TOLERANCE {
                self.counters.nonrigid_poses += 1;
            }
        }
        // `WindowedU8` keeps its own level-0 copies.
        if self.rig_flow.is_none() {
            self.retain_previous_image(image, previous_image);
        }
        if self.tracker.flows_rig() {
            self.retain_secondary_images(&images[1..]);
        }
        result
    }

    /// Prepares the `WindowedU8` flow frames of capture `frame_idx`. The
    /// tracks' pixels refer to the previous capture: when prepare fails, or
    /// the previous frame is not capture `frame_idx - 1` (a failed prepare
    /// left a fresh pair of this capture), they are cleared, so no template is
    /// recut at stale pixels in another image (R-D4).
    fn prepare_flow(&mut self, images: &[&Image<u8, 1>], frame_idx: u64) {
        let Some(rig_flow) = self.rig_flow.as_mut() else {
            return;
        };
        match rig_flow.prepare(images, frame_idx, &mut self.tracking_timings) {
            Ok(()) => {
                let follows = rig_flow
                    .previous_frame(0)
                    .is_some_and(|previous| previous.frame_id() + 1 == frame_idx);
                if !follows {
                    self.tracker.reset_tracks();
                }
            }
            Err(error) => {
                self.tracker.reset_tracks();
                self.dbg(format!(
                    "[flow] frame={frame_idx} frames not prepared: {error}"
                ));
            }
        }
    }

    /// Time the latest successfully processed frame spent in tracking and
    /// mapping on the calling thread. It excludes feature extraction, whether
    /// it ran before tracking or was deferred into it (see
    /// [`Self::frontend_observation`]), and all work outside `process_*`.
    pub fn tracking_duration(&self) -> Duration {
        self.tracking_duration
    }

    /// Per-stage times of the latest successfully processed frame's
    /// tracking; zero for stages it did not run (all of them in bootstrap).
    pub fn tracking_timings(&self) -> &TrackingTimings {
        &self.tracking_timings
    }

    /// Records a [`FrameTrace`] for every tracked frame from now on. Off
    /// after [`SlamSystem::build`]; reading state only, it changes no result.
    pub fn set_trace(&mut self, on: bool) {
        self.trace_enabled = on;
        if !on {
            self.frame_trace = None;
        }
    }

    /// The latest frame's trace, when tracing is on and the frame went
    /// through frame-to-map tracking.
    pub fn frame_trace(&self) -> Option<&FrameTrace> {
        self.frame_trace.as_ref()
    }

    /// The frontend output of the latest successfully processed frame.
    pub fn frontend_observation(&self) -> &FrontendObservation {
        self.frontend.observation()
    }

    /// Copies `image` into the previous-image buffer, reusing its allocation.
    fn retain_previous_image(&mut self, image: &Image<u8, 1>, buffer: Option<Image<u8, 1>>) {
        self.previous_image = match buffer {
            Some(mut buffer) if buffer.size() == image.size() => {
                buffer.as_slice_mut().copy_from_slice(image.as_slice());
                Some(buffer)
            }
            _ => Some(image.clone()),
        };
    }

    /// Copies rig cameras 1 and up into their previous-image buffers.
    fn retain_secondary_images(&mut self, images: &[&Image<u8, 1>]) {
        self.previous_secondary.truncate(images.len());
        for (camera, image) in images.iter().enumerate() {
            match self.previous_secondary.get_mut(camera) {
                Some(buffer) if buffer.size() == image.size() => {
                    buffer.as_slice_mut().copy_from_slice(image.as_slice());
                }
                Some(buffer) => *buffer = (*image).clone(),
                None => self.previous_secondary.push((*image).clone()),
            }
        }
    }

    /// Map reuse: how many of the frame's matches are landmarks tracking
    /// first matched over 10 s ago.
    fn report_landmark_age(&mut self, frame_idx: usize, matches: &[(usize, usize)], now_sec: f64) {
        const OLD_SEC: f64 = 10.0;
        let map = lock_map(&self.map, &self.map_wait);
        let mut old = 0;
        for &(mp_idx, _) in matches {
            if map.map_points().get(mp_idx).is_none() {
                continue;
            }
            // Landmark slots are append-only until a map reset.
            let first = *self
                .landmark_first_tracked_sec
                .entry(mp_idx)
                .or_insert(now_sec);
            if now_sec - first > OLD_SEC {
                old += 1;
            }
        }
        drop(map);
        self.dbg(format!(
            "[age] frame={frame_idx} matches={} older_10s={old}",
            matches.len()
        ));
    }

    fn report_stereo(&mut self, frame: &Frame) {
        if !self.debug {
            return;
        }
        if frame.is_rig() {
            let mut depths: Vec<f64> = frame
                .rig
                .stereo_points
                .iter()
                .flatten()
                .map(|p| p.z)
                .collect();
            depths.sort_by(f64::total_cmp);
            let median = depths.get(depths.len() / 2).copied().unwrap_or(0.0);
            let per_camera: Vec<usize> = frame.rig.camera_ranges.iter().map(|r| r.len()).collect();
            self.dbg(format!(
                "[overlap] frame={} features={per_camera:?} triangulated={} median_depth={median:.3}m",
                frame.idx,
                depths.len()
            ));
            return;
        }
        let Some(matched) = self.frontend.observation().stereo_matched else {
            return;
        };
        let mut depths: Vec<f32> = frame.depth.iter().copied().filter(|&d| d > 0.0).collect();
        let median = if depths.is_empty() {
            0.0
        } else {
            depths.sort_by(|a, b| a.total_cmp(b));
            depths[depths.len() / 2]
        };
        self.dbg(format!(
            "[stereo] frame={} matched={matched} median_depth={median:.3}m",
            frame.idx
        ));
    }

    /// Tracks one frame of prepared or deferred features.
    fn process_prepared(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        // Local-BA snapshots, merges, and their correction messages can only
        // cross this boundary between complete tracking frames.
        let publication_gate = self.map_publication_gate.clone();
        let _publication = publication_gate
            .as_ref()
            .map(|gate| gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        self.apply_local_mapping_results();
        self.apply_loop_closing_results();
        self.adopt_landmarks();

        match self.state.mode {
            SystemMode::Bootstrap => {
                let frame = self.resolve(source)?;
                Ok(self.bootstrap_step(frame, timestamp_sec))
            }
            SystemMode::ImuInit => {
                self.inertial_init_step(source, previous_image, images, timestamp_sec)
            }
            SystemMode::Tracking => {
                self.tracking_step(source, previous_image, images, timestamp_sec)
            }
        }
    }

    /// Runs `f` against the live map points, holding the map lock only for the
    /// duration of the call. Avoids cloning the whole point list (descriptors
    /// included) for read-only consumers such as viz logging and summaries.
    pub fn with_map_points<R>(&self, f: impl FnOnce(&[MapPoint]) -> R) -> R {
        f(lock_map(&self.map, &self.map_wait).map_points())
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

    /// `T_cw` of the keyframe numbered `frame_idx`, while the map holds it.
    pub fn keyframe_pose(&self, frame_idx: usize) -> Option<Pose3d> {
        self.map
            .lock()
            .unwrap()
            .get_keyframe(frame_idx)
            .map(|kf| kf.frame.pose_world_to_cam)
    }

    /// Returns the number of active (non-culled) map points.
    pub fn num_active_map_points(&self) -> usize {
        lock_map(&self.map, &self.map_wait).num_active_map_points()
    }

    /// Drain any debug messages accumulated since the last call.
    pub fn drain_debug_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_messages)
    }

    /// Toggles per-frame diagnostics: bootstrap skip and reject reasons,
    /// tracking reject reasons, keyframe growth and fuse counters. Off after
    /// [`SlamSystem::build`].
    pub fn set_debug(&mut self, on: bool) {
        self.debug = on;
        if !on {
            self.debug_messages.clear();
        }
    }

    fn apply_local_mapping_results(&mut self) {
        self.apply_growth_results();
        let Some(reference_idx) = self.state.current_keyframe_idx else {
            // Still drain results so a completed worker cannot build a result backlog.
            let _ = self.local_mapping.drain_results();
            return;
        };

        let results = self.local_mapping.drain_results();
        if !results.is_empty() {
            self.refresh_track_positions();
        }
        let inertial_idx = if self.inertial_state_from_last_keyframe() {
            self.state.last_keyframe_idx
        } else {
            Some(reference_idx)
        };
        for result in results {
            if let Some(marginal) = result.newest_marginal {
                self.estimator.newest_marginal = Some(marginal);
            }
            // A propagated keyframe was not solved: the shift counter stays the solve's.
            if let Some(newest) = result
                .keyframe_corrections
                .iter()
                .filter(|c| !c.propagated)
                .max_by_key(|c| c.kf_idx)
            {
                self.estimator.map_changed = true;
                self.note_ba_shift(newest.kf_idx, &newest.pose_before, &newest.pose_after);
            }
            for correction in &result.keyframe_corrections {
                if self.note_pose("ba_keyframe", correction.kf_idx, &correction.pose_after) {
                    self.counters.nonrigid_keyframes += 1;
                }
            }
            let correction_of = |kf_idx: usize| {
                result
                    .keyframe_corrections
                    .iter()
                    .find(|correction| correction.kf_idx == kf_idx)
            };
            if let Some(correction) = correction_of(reference_idx) {
                self.state.pose_world_to_cam = apply_reference_pose_correction(
                    self.state.pose_world_to_cam,
                    correction.pose_before,
                    correction.pose_after,
                );
                let pose = self.state.pose_world_to_cam;
                self.note_pose(
                    "ba_correction",
                    self.next_frame_idx.saturating_sub(1),
                    &pose,
                );
            }
            if let Some(correction) = inertial_idx.and_then(correction_of) {
                if self.debug {
                    self.dbg(format!(
                        "[ba_vel] kf={} dv={:.4} dp={:.4}",
                        correction.kf_idx,
                        (correction.velocity_world - correction.velocity_before).length(),
                        (recovery::camera_centre(&correction.pose_after)
                            - recovery::camera_centre(&correction.pose_before))
                        .length(),
                    ));
                }
                self.state.velocity_world = correction.velocity_world;
                self.inertial.bias = correction.imu_bias;
            }
        }
    }

    /// Reports the queued growth that has finished. Growth moves no landmark,
    /// so the tracks' positions stay valid.
    fn apply_growth_results(&mut self) {
        for outcome in self.local_mapping.drain_growth() {
            let lag = self.next_frame_idx.saturating_sub(outcome.keyframe_idx + 1) as u64;
            self.counters.growth_lag_frames_max = self.counters.growth_lag_frames_max.max(lag);
            self.growth_done_through = self.growth_done_through.max(Some(outcome.keyframe_idx));
            self.report_growth(outcome.keyframe_idx, &outcome.result);
        }
    }

    /// Rewrites the tracks' stored map-point positions after a map change,
    /// under one map lock; only `WindowedU8` shards read them.
    fn refresh_track_positions(&mut self) {
        if self.rig_flow.is_none() {
            return;
        }
        let map = lock_map(&self.map, &self.map_wait);
        self.counters.flow_position_refreshes += self.tracker.refresh_positions(&map);
    }

    /// Spawns tracks on the landmarks the latest keyframe created, on the
    /// frame after its insertion, when `grow_keyframe` has made them; a later
    /// mapping result only moves them (`adopt_landmarks`). Tracks refer to
    /// the previous frame here: a landmark of that very frame starts at its
    /// keypoint; an older keyframe's is tracked into it from the kept
    /// keyframe frames, starting at its projection.
    fn adopt_landmarks(&mut self) {
        let (true, Some(kf_idx)) = (self.flow.adopt_landmarks, self.state.last_keyframe_idx) else {
            return;
        };
        if self.adopted_keyframe == Some(kf_idx) {
            return;
        }
        // Its new landmarks exist only once its queued growth is back.
        if self.keyframe_growth == KeyframeGrowth::Mapping
            && self.growth_done_through < Some(kf_idx)
        {
            return;
        }
        self.adopted_keyframe = Some(kf_idx);
        let previous_idx = self.next_frame_idx.checked_sub(2);
        let pose = self.state.pose_world_to_cam;
        let cameras = self.rig_flow.as_ref().map_or(1, RigFlow::cameras);
        let landmarks: Vec<([f32; 2], usize, usize, Vec3F64)> = {
            let map = lock_map(&self.map, &self.map_wait);
            let Some(kf) = map.get_keyframe(kf_idx) else {
                return;
            };
            kf.map_point_by_desc_idx
                .iter()
                .enumerate()
                .filter_map(|(desc_idx, mp_idx)| {
                    let mp_idx = (*mp_idx)?;
                    let mp = map
                        .map_points()
                        .get(mp_idx)
                        .filter(|mp| !mp.culled && mp.keyframe_idx == kf_idx)?;
                    let camera = kf.frame.camera_of(desc_idx);
                    let pixel = *kf.frame.features.keypoints_xy.get(desc_idx)?;
                    (camera < cameras).then_some((pixel, camera, mp_idx, mp.position))
                })
                .collect()
        };
        for (pixel, camera, mp_idx, position) in landmarks {
            let pixel = if previous_idx == Some(kf_idx) {
                Some(pixel)
            } else {
                self.carry_landmark(kf_idx, camera, pixel, &pose, &position)
            };
            if let Some(pixel) = pixel {
                self.tracker.adopt(pixel, camera, mp_idx, position);
            }
        }
        self.tracker.thin_tracks();
    }

    /// A kept keyframe landmark's pixel in the previous frame (`WindowedU8`
    /// only), starting at its projection at that frame's pose.
    fn carry_landmark(
        &mut self,
        kf_idx: usize,
        camera: usize,
        pixel: [f32; 2],
        pose_world_to_primary: &Pose3d,
        position: &Vec3F64,
    ) -> Option<[f32; 2]> {
        let (model, pose) = match camera {
            0 => (&self.rig.camera, *pose_world_to_primary),
            c => {
                let view = self.frontend.rig_geometry()?.view(c)?;
                (&view.camera, view.pose(pose_world_to_primary))
            }
        };
        let start =
            crate::tracking::tracker::project_raw(model, &pose.transform_point(position), pixel)
                .unwrap_or(pixel);
        self.rig_flow
            .as_mut()?
            .adopt(kf_idx as u64, camera, pixel, start)
    }

    fn dbg(&mut self, msg: String) {
        if self.debug {
            self.debug_messages.push(msg);
        }
    }

    /// Logs a `[rigid]` line when `pose` is not rigid; returns whether it is not.
    fn note_pose(&mut self, site: &'static str, idx: usize, pose: &Pose3d) -> bool {
        let defect = pose.rotation_defect();
        let nonrigid = defect > RIGID_TOLERANCE;
        if nonrigid {
            self.dbg(format!("[rigid] site={site} idx={idx} defect={defect:.1e}"));
        }
        nonrigid
    }

    fn bootstrap_step(&mut self, curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stereo frames carry metric per-keypoint depth, so we can build a
        // metric map from a single keyframe (ORB-SLAM3's StereoInitialization)
        // instead of waiting for two-view parallax.
        let mut carried = false;
        let result = if curr_frame.is_metric() {
            let reseeding = self.state.reseeding();
            if reseeding {
                carried = self.coast_while_reseeding(curr_frame.idx, timestamp_sec);
                if self.recovery.relocalize_when_lost {
                    // No segment is tracked while re-seeding, so no own window applies.
                    self.tracker.set_window_start(None);
                    if self.relocalize(&curr_frame, timestamp_sec, true) {
                        self.resume_after_lost_relocalization(timestamp_sec);
                        return self.frame_result(TrackingStatus::Tracked);
                    }
                }
            }
            let result = self.bootstrap_stereo(curr_frame, timestamp_sec);
            if reseeding && result.status == TrackingStatus::Skipped {
                self.counters.reseed_stall_frames += 1;
            }
            result
        } else {
            self.bootstrap_mono(curr_frame, timestamp_sec)
        };
        if result.status == TrackingStatus::Skipped {
            self.last_skip = Some(if carried {
                SkipKind::Reseeding
            } else {
                SkipKind::Bootstrap
            });
        }
        result
    }

    /// Tracks on from a relocalization found while re-seeding. The pose jumped
    /// to the map's, so neither the bridge nor an IMU edge from the last
    /// keyframe holds across it: the next keyframe starts a new chain.
    fn resume_after_lost_relocalization(&mut self, timestamp_sec: f64) {
        self.state.mode = SystemMode::Tracking;
        self.state.last_keyframe_idx = None;
        self.state.bridge_anchor = None;
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.runaway.clear();
        self.pending_graph_gap = self
            .state
            .current_keyframe_idx
            .map(|found_kf| KeyframeGap::Relocalized { found_kf });
    }

    /// Adds this frame's pose over `[t0, t1]` to the runaway check, which
    /// settles coast trust whatever the recovery knobs (R-C1), and to the tilt
    /// check, which only a kept inertial state reads.
    fn record_frame_pose(&mut self, t0: f64, t1: f64) {
        let pose = self.state.pose_world_to_cam;
        self.runaway.record(t1, recovery::camera_centre(&pose));
        self.record_tilt(&pose, t0, t1);
    }

    /// Marks this frame's coast trusted when the IMU is confident and the
    /// coasted pose is not a runaway. The runaway check is fed on every
    /// frame, so this holds without `keep_inertial` too (R-C1).
    fn settle_coast_trust(&mut self, imu_confident: bool) {
        if let Some(SkipKind::Coasting { inertial, .. }) = self.last_skip {
            let runaway = self.runaway.runaway();
            self.counters.coast_runaway_frames += u64::from(runaway);
            self.last_skip = Some(SkipKind::Coasting {
                inertial,
                trusted: imu_confident && !runaway,
            });
        }
    }

    /// Carries the pose over a frame of a stalled re-seed on the kept inertial
    /// state, so the segment starts where the camera is. A runaway, or a
    /// stall past `max_bridge_sec`, holds the translation at rest instead and
    /// drops the bridge; the attitude keeps following the gyro either way.
    /// Returns whether the pose was carried on an inertial prediction.
    fn coast_while_reseeding(&mut self, frame_idx: usize, timestamp_sec: f64) -> bool {
        let pose_before = self.state.pose_world_to_cam;
        let prev_timestamp = self.state.last_frame_timestamp_sec;
        let (coasted, inertial) =
            self.predict_frame_pose(pose_before, prev_timestamp, timestamp_sec);
        self.record_tilt(&coasted, prev_timestamp, timestamp_sec);
        self.state.last_frame_timestamp_sec = timestamp_sec;
        if !self.reseed_held {
            self.runaway
                .record(timestamp_sec, recovery::camera_centre(&coasted));
            let starved = timestamp_sec - self.reseed_coast_from_sec > recovery::MAX_BRIDGE_SEC;
            if !self.runaway.runaway() && !starved {
                self.state.pose_world_to_cam = coasted;
                return inertial.is_some();
            }
            self.dbg(if starved {
                format!(
                    "[recover] frame={frame_idx} re-seed starved past {:.1}s: translation held",
                    recovery::MAX_BRIDGE_SEC
                )
            } else {
                format!(
                    "[recover] frame={frame_idx} runaway at {:.1} m/s while re-seeding: held at rest",
                    self.runaway.speed_mps().unwrap_or(f64::NAN)
                )
            });
            self.runaway.clear();
            self.reseed_held = true;
            self.counters.reseeds_held += 1;
            self.state.bridge_anchor = None;
        }
        self.state.velocity_world = Vec3F64::ZERO;
        self.state.pose_world_to_cam = recovery::held_at(&coasted, &pose_before);
        false
    }

    /// Adds the IMU window `[t0, t1]` of a frame at `pose_world_to_cam` to the
    /// tilt check; only while the inertial state may be kept.
    fn record_tilt(&mut self, pose_world_to_cam: &Pose3d, t0: f64, t1: f64) {
        if !(self.recovery.keep_inertial && self.state.imu_initialized && t0 > 0.0) {
            return;
        }
        let Some(stats) = self.inertial.window_stats(t0, t1) else {
            return;
        };
        let body = body_to_world(pose_world_to_cam, self.rig.camera_to_body());
        self.tilt
            .record(t1, &stats, self.inertial.bias.accel, &body.rotation);
    }

    /// Single-frame metric initialization from stereo depth.
    fn bootstrap_stereo(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // The map will start at this frame or a later one, so older samples
        // can never enter an edge; a failed attempt (too few stereo points,
        // as in the dark) must not keep them, or the buffer grows for as
        // long as bootstrap keeps failing. A pending bridge keeps its own.
        let bridge_anchor = self.live_bridge_anchor(timestamp_sec);
        self.prune_imu_before(bridge_anchor.map_or(timestamp_sec, |anchor| anchor.timestamp_sec));
        // Build the new map in the current odometry frame (identity at start,
        // or the recovery pose after a tracking loss).
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;
        let curr_idx = curr_frame.idx;
        let mut keyframe = Keyframe::from_frame(curr_frame);
        if self.state.reseeding() {
            keyframe.velocity_world = self.state.velocity_world;
            keyframe.imu_bias = self.inertial.bias;
            keyframe.segment_start = true;
        }

        // A kept inertial state only needs a few points to start the segment;
        // keyframes at the inertial poses grow it.
        let min_points = if self.state.reseeding() {
            self.recovery.min_seed_points
        } else {
            self.recovery.min_cold_start_points
        };
        let insertion = match bootstrap::stereo_initial_map(keyframe, &self.rig.camera, min_points)
        {
            Ok(insertion) => insertion,
            Err(TooFewStereoPoints { found }) => {
                self.dbg(format!(
                    "[bootstrap_stereo] frame={curr_idx} skip: only {found} stereo points \
                     (need >= {min_points})"
                ));
                return self.frame_result(TrackingStatus::Skipped);
            }
        };

        // The keyframe and its seeds are published together; tracker state is
        // adopted below only once that succeeded.
        let published = lock_map(&self.map, &self.map_wait).apply_insertion(insertion);
        let added = match published {
            Ok(result) => result.landmark_ids.len(),
            Err(error) => {
                self.dbg(format!(
                    "[bootstrap_stereo] frame={curr_idx} publication rejected: {error}"
                ));
                return self.frame_result(TrackingStatus::Skipped);
            }
        };

        self.dbg(format!(
            "[bootstrap_stereo] frame={curr_idx} metric map created with {added} points \
             pose_defect={:.1e}",
            self.state.pose_world_to_cam.rotation_defect(),
        ));
        let bridged =
            bridge_anchor.and_then(|anchor| self.publish_bridge(anchor, curr_idx, timestamp_sec));
        if self.state.reseeding() {
            let gap = bridged.unwrap_or_else(|| self.unbridged_reseed_gap(timestamp_sec));
            self.pending_graph_gap = None;
            self.note_graph_gap(curr_idx, gap);
        }
        self.state.bridge_anchor = None;

        self.state.velocity = None;
        // The map is already metric (stereo baseline), but gravity, velocities,
        // and the gyro bias still need the inertial init before IMU prediction
        // can run; the solve there keeps scale fixed at 1.
        self.finish_bootstrap(curr_idx, timestamp_sec);
        self.prune_imu_before(timestamp_sec);

        self.frame_result(TrackingStatus::KeyframeAccepted)
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
                // No reference is held: the next one is a later frame.
                self.prune_imu_before(timestamp_sec);
                return self.frame_result(TrackingStatus::Skipped);
            }
            BootstrapDecision::StoreAsReference => {
                self.dbg(format!(
                    "[bootstrap] frame={} stored as reference (awaiting second frame)",
                    curr_frame.idx,
                ));
                self.store_bootstrap_reference(curr_frame, timestamp_sec);
                return self.frame_result(TrackingStatus::Skipped);
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
                // A reference that keeps failing to match has lost the scene
                // (ORB-SLAM3 replaces it on the first such frame; waiting a
                // little keeps the better-conditioned early pairs).
                if self
                    .state
                    .bootstrap_reference_lost(&reason, MAX_LOW_MATCH_REJECTIONS)
                {
                    self.dbg(format!(
                        "[bootstrap] frame={} replaces reference {reference_idx}",
                        curr_frame.idx
                    ));
                    self.store_bootstrap_reference(curr_frame, timestamp_sec);
                }
                return self.frame_result(TrackingStatus::Skipped);
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
        self.note_pose("bootstrap", curr_frame.idx, &estimated_pose);

        // Promote to Keyframes
        let prev_idx = prev_bootstrap_frame.idx;
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        // A rejected publication writes nothing, so there is no map to
        // evaluate; treat it like a failed health gate.
        let insertion =
            bootstrap::two_view_initial_map(reference_kf, current_kf, &two_view_estimate);
        if let Err(error) = self.publish_initial_map(insertion, [prev_idx, curr_idx]) {
            self.dbg(format!(
                "[bootstrap] frame={curr_idx} publication rejected: {error}"
            ));
            self.state.reset();
            return self.frame_result(TrackingStatus::Skipped);
        }

        // The gate runs after initial BA, which can collapse a badly
        // conditioned pair.
        let health =
            crate::initialization::initial_map_health(&lock_map(&self.map, &self.map_wait));
        if health.is_degenerate() {
            self.dbg(format!(
                "[init_gate] reject: valid_in_both={} median_depth={:.3} (need >= {} and > 0)",
                health.valid_in_both, health.median_depth_older_kf, MIN_VALID_POINTS,
            ));
            lock_map(&self.map, &self.map_wait).clear_active();
            self.local_mapping.clear_growth();
            if let Some(w) = self.wheel.as_mut() {
                w.reset_map();
            }
            self.state.reset();
            return self.frame_result(TrackingStatus::Skipped);
        }

        // Initial BA in publish_initial_map may have refined KF1's pose; sync state
        // and recompute velocity from the post-BA pose.
        if let Some(kf) = lock_map(&self.map, &self.map_wait).get_keyframe(curr_idx) {
            self.state.pose_world_to_cam = kf.frame.pose_world_to_cam;
        }
        let pose = self.state.pose_world_to_cam;
        self.note_pose("bootstrap", curr_idx, &pose);

        if let Some(prev_ts) = self.inertial.bootstrap_timestamp_sec {
            if let Some(edge) = self.inertial.keyframe_edge(
                self.rig.imu_noise(),
                prev_idx,
                curr_idx,
                prev_ts,
                timestamp_sec,
            ) {
                // Through the validated path: endpoints must exist, the edge
                // must be new, and the interval finite and ordered.
                let published = lock_map(&self.map, &self.map_wait).apply_insertion(MapInsertion {
                    imu_factors: vec![edge],
                    ..Default::default()
                });
                if let Err(error) = published {
                    self.dbg(format!(
                        "[bootstrap] frame={curr_idx} imu edge rejected: {error}"
                    ));
                }
            }
            self.prune_imu_before(timestamp_sec);
        }

        self.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.state.pose_world_to_cam,
        ));

        self.finish_bootstrap(curr_idx, timestamp_sec);

        self.frame_result(TrackingStatus::KeyframeAccepted)
    }

    fn store_bootstrap_reference(&mut self, frame: Frame, timestamp_sec: f64) {
        self.state.bootstrap_frame = Some(frame);
        self.state.bootstrap_low_match_rejections = 0;
        self.inertial.bootstrap_timestamp_sec = Some(timestamp_sec);
        // Samples before the reference frame can never enter an edge.
        self.prune_imu_before(timestamp_sec);
    }

    /// Publishes the two-view map, refines it with initial BA, and offers both
    /// keyframes to place recognition so a later revisit of the start can
    /// match them.
    fn publish_initial_map(
        &mut self,
        insertion: MapInsertion,
        keyframes: [usize; 2],
    ) -> Result<(), MapMutationError> {
        lock_map(&self.map, &self.map_wait).apply_insertion(insertion)?;
        let refined = crate::mapping::bundle_adjustment::run_initial_ba(
            &mut lock_map(&self.map, &self.map_wait),
            &self.rig.camera,
            self.local_ba.depth_sigma,
        );
        if let Err(error) = refined {
            self.dbg(format!("[init_ba] {error}; keeping the unrefined map"));
        }
        for kf_idx in keyframes {
            self.register_place_recognition(kf_idx);
        }
        Ok(())
    }

    /// Leaves bootstrap with `kf_idx` as the reference keyframe. Inertial
    /// initialization follows when the camera-to-body extrinsic is known, since
    /// it relates IMU deltas to camera poses; otherwise tracking is visual-only.
    /// A segment re-seeded with the inertial state kept tracks inertially at once.
    fn finish_bootstrap(&mut self, kf_idx: usize, timestamp_sec: f64) {
        self.segments.begin(kf_idx);
        self.segment_begun_sec = Some(timestamp_sec);
        self.window_segment_start = (self.segments.latest() > 0).then_some(kf_idx);
        if self.segments_reported() {
            self.dbg(format!(
                "[recover] segment {} begins at kf={kf_idx}",
                self.segments.latest()
            ));
        }
        self.state.current_keyframe_idx = Some(kf_idx);
        self.state.last_keyframe_idx = Some(kf_idx);
        self.state.mode = if self.state.imu_initialized {
            SystemMode::Tracking
        } else if self.rig.camera_to_body().is_some() {
            self.inertial.schedule.start(kf_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
    }

    fn frame_result(&self, status: TrackingStatus) -> TrackingResult {
        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    /// Drops buffered IMU samples strictly older than `t` (typically the last
    /// keyframe timestamp: the next edge and all per-frame windows start there).
    fn prune_imu_before(&mut self, t: f64) {
        self.inertial.prune_before(t);
    }

    /// Samples older than the last keyframe can't enter any future window
    /// (the next edge and all per-frame predictions start at or after it),
    /// except those an open still run has yet to judge.
    fn prune_imu_after_frame(&mut self, timestamp_sec: f64) {
        if let Some(kf_ts) = self.inertial.last_keyframe_timestamp_sec {
            let floor = kf_ts.min(timestamp_sec);
            let floor = self
                .stillness
                .unjudged_from()
                .map_or(floor, |from| floor.min(from));
            self.prune_imu_before(floor);
        }
    }

    /// Tracks the frame, then attempts VIBA0 on accepted keyframes once the
    /// window is ready and a retry is due.
    fn inertial_init_step(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let result = self.tracking_step(source, previous_image, images, timestamp_sec)?;

        if result.status == TrackingStatus::KeyframeAccepted
            && let Some(start_idx) = self.inertial.schedule.start_kf_idx()
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

        let due_for_retry = self.inertial.schedule.retry_due(timestamp_sec);
        let imu_init_ready = self.inertial.initializer.ready(
            &lock_map(&self.map, &self.map_wait),
            self.inertial.schedule.start_kf_idx(),
        );

        if result.status == TrackingStatus::KeyframeAccepted && due_for_retry && imu_init_ready {
            let Some(start_idx) = self.inertial.schedule.start_kf_idx() else {
                return Ok(result);
            };
            self.inertial.schedule.record_attempt(timestamp_sec);
            let is_mono = !self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .find(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.is_metric())
                .unwrap_or(false);
            let prior_a0 = viba0_accel_bias_prior(is_mono);
            // Drop the solve's map lock before applying its result with a new lock.
            let init_result = self.inertial.initializer.try_initialize(
                &lock_map(&self.map, &self.map_wait),
                self.rig.camera_to_body(),
                self.inertial.bias,
                start_idx,
                1e2,
                prior_a0,
                false,
            );
            match init_result {
                Some(init) => {
                    let applied = self.apply_inertial_initialization(init, start_idx);
                    match applied {
                        Ok(applied) => self.resume_tracking_after_viba0(timestamp_sec, applied),
                        // Staying in ImuInit lets the retry throttle re-solve later.
                        Err(error) => {
                            self.dbg(format!("[imu_init] VIBA0 alignment refused: {error}"));
                        }
                    }
                }
                None => {
                    self.dbg("[imu_init] VIBA0 rejected: solve failed or invalid scale".into());
                }
            }
        }

        Ok(result)
    }

    /// Applies an inertial initialization to the map and resumes tracking from
    /// the aligned state; a refused alignment changes nothing.
    fn apply_inertial_initialization(
        &mut self,
        init: ImuInitResult,
        start_kf_idx: usize,
    ) -> Result<AppliedInitialization, InertialAlignmentError> {
        let applied = self.inertial.apply_initialization(
            &mut lock_map(&self.map, &self.map_wait),
            init,
            start_kf_idx,
        )?;
        let pose_before = self.state.pose_world_to_cam;
        self.state.adopt_inertial_initialization(applied.aligned);
        let pose = self.state.pose_world_to_cam;
        self.count_later_alignment(&pose_before, &pose);
        self.note_pose("imu_init", self.next_frame_idx.saturating_sub(1), &pose);
        // The alignment rescaled and rotated the landmarks under the tracks (T3).
        self.refresh_track_positions();
        Ok(applied)
    }

    /// Card AC (G46): an alignment in a later segment than the first aligned
    /// one moves the already gravity-aligned older segments with it, so its
    /// size is counted. Segments reset before any alignment (a start-up
    /// reset) hold no aligned world, so their successor's alignment is not.
    fn count_later_alignment(&mut self, before: &Pose3d, after: &Pose3d) {
        let segment = self.segments.latest();
        if *self.first_aligned_segment.get_or_insert(segment) == segment {
            return;
        }
        let jump_m = (recovery::camera_centre(after) - recovery::camera_centre(before)).length();
        self.counters.later_alignments += 1;
        counters::raise_milli_max(
            &mut self.counters.later_alignment_mdeg_max,
            angle_deg(before, after),
        );
        counters::raise_milli_max(&mut self.counters.later_alignment_jump_mm_max, jump_m);
    }

    fn count_young_death(&mut self, lost_since_sec: f64) {
        let young = self
            .segment_begun_sec
            .is_some_and(|begun| lost_since_sec - begun < counters::YOUNG_SEGMENT_SEC);
        if young && self.startup_over {
            self.counters.young_segment_deaths += 1;
        } else if young {
            self.counters.startup_segment_deaths += 1;
        }
        self.startup_over |= !young;
    }

    fn resume_tracking_after_viba0(&mut self, timestamp_sec: f64, applied: AppliedInitialization) {
        // Mirrors ORB-SLAM3: IMU is marked initialized (and tracking resumes)
        // immediately after VIBA0 succeeds — VIBA1/VIBA2 refine
        // bg/ba/scale/gravity further in the background (see
        // try_insert_keyframe), they don't gate resuming tracking.
        self.state.mode = SystemMode::Tracking;
        self.state.imu_init_timestamp_sec = Some(timestamp_sec);
        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized: true,
            imu_t_bc: self.rig.camera_to_body(),
            gravity_world: self.inertial.gravity_world,
            local_ba: self.job_local_ba(),
            wheel_factors: self.wheel_factors(),
            tracked_inliers: 0,
        }) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        self.apply_local_mapping_results();
        let AppliedInitialization {
            scale,
            gravity_world: gravity,
            gyro_bias: bg,
            ..
        } = applied;
        self.dbg(format!(
            "[imu_init] VIBA0 accepted: scale={scale:.4} gravity=({:.3},{:.3},{:.3}) \
             gyro_bias=({:.4},{:.4},{:.4})",
            gravity.x, gravity.y, gravity.z, bg.x, bg.y, bg.z
        ));
    }

    /// Local BA updates keyframe state asynchronously, so the cached IMU
    /// state is refreshed from the reference keyframe. A reference other than
    /// the last keyframe (an anchor, or one a knob of
    /// [`Self::inertial_state_from_last_keyframe`] chose) holds an older
    /// velocity than the propagated one, which stays, with the last
    /// keyframe's bias (E1, G3).
    fn refresh_reference_inertial_state(&mut self) -> FrameVelocitySource {
        let Some(kf_idx) = self.state.current_keyframe_idx else {
            return FrameVelocitySource::Carried;
        };
        let map = lock_map(&self.map, &self.map_wait);
        let Some(kf) = map.get_keyframe(kf_idx) else {
            return FrameVelocitySource::Carried;
        };
        let decoupled = Some(kf_idx) != self.state.last_keyframe_idx
            && (kf.anchor || self.inertial_state_from_last_keyframe());
        if !decoupled {
            // The propagation reads `inertial.bias`, so it must be the
            // keyframe's first.
            self.inertial.bias = kf.imu_bias;
            return match self.propagated_frame_velocity(kf, kf_idx) {
                Some(velocity_world) => {
                    self.state.velocity_world = velocity_world;
                    FrameVelocitySource::Propagated
                }
                None => {
                    self.state.velocity_world = kf.velocity_world;
                    FrameVelocitySource::Keyframe
                }
            };
        }
        if let Some(last) = self
            .state
            .last_keyframe_idx
            .and_then(|ki| map.get_keyframe(ki))
        {
            self.inertial.bias = last.imu_bias;
        }
        FrameVelocitySource::Carried
    }

    /// Under [`FrameVelocity::PropagatedFromKeyframe`], the velocity at the
    /// previous frame: the reference keyframe's propagated over
    /// `[t_kf, t_prev]` (ORB-SLAM3 `Tracking.cc:1746-1758`). `None` keeps the
    /// keyframe's: the knob off, a pose-inertial mode, no IMU, or a reference
    /// that is not the last keyframe, whose samples are already pruned.
    fn propagated_frame_velocity(&self, kf: &Keyframe, kf_idx: usize) -> Option<Vec3F64> {
        if self.tracking_config.frame_velocity != FrameVelocity::PropagatedFromKeyframe
            || self.tracking_config.pose_inertial != PoseInertial::Off
            || !self.state.imu_initialized
            || Some(kf_idx) != self.state.last_keyframe_idx
        {
            return None;
        }
        let t_kf = self.inertial.last_keyframe_timestamp_sec?;
        let t_prev = self.state.last_frame_timestamp_sec;
        if t_prev <= t_kf {
            return None;
        }
        let (preintegrated, _) =
            self.inertial
                .preintegrate_window(self.rig.imu_noise(), t_kf, t_prev);
        (preintegrated.dt > 0.0).then(|| {
            propagate_keyframe_velocity(
                kf.frame.pose_world_to_cam,
                kf.velocity_world,
                self.inertial.gravity_world,
                &preintegrated,
                self.rig.camera_to_body(),
            )
        })
    }

    /// `[vel]`: the velocity this frame's prediction starts from, for comparison with GT velocity.
    fn report_frame_velocity(&mut self, frame_idx: usize, source: FrameVelocitySource) {
        let Some(kf_idx) = self.state.current_keyframe_idx else {
            return;
        };
        let ref_lost = self
            .map
            .lock()
            .unwrap()
            .get_keyframe(kf_idx)
            .is_some_and(|kf| kf.inserted_while_lost);
        let t_prev = self.state.last_frame_timestamp_sec;
        let age_s = match self.inertial.last_keyframe_timestamp_sec {
            Some(t_kf) if Some(kf_idx) == self.state.last_keyframe_idx => t_prev - t_kf,
            _ => f64::NAN,
        };
        let v = self.state.velocity_world;
        self.dbg(format!(
            "[vel] frame={frame_idx} ref_kf={kf_idx} ref_lost={} age_s={age_s:.4} \
             v=({:.4},{:.4},{:.4}) t={t_prev:.9} src={}",
            u8::from(ref_lost),
            v.x,
            v.y,
            v.z,
            source.label(),
        ));
    }

    /// Whether a knob lets the tracking reference differ from the last
    /// keyframe: A8's most-voted reference, or a relocalization while lost or
    /// after a give-up. The frame's velocity and bias then come from the last
    /// keyframe, ORB-SLAM3's `mpLastKeyFrame`; off, from the reference as before.
    fn inertial_state_from_last_keyframe(&self) -> bool {
        self.tracking_config.local_map == LocalMapRules::OrbSlam3
            || self.relocalization.while_lost
            || self.recovery.relocalize_when_lost
    }

    fn tracking_step(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let current_image = images[0];
        let frame_idx = source.idx();
        let pose_before = self.state.pose_world_to_cam;
        let prev_timestamp = self.state.last_frame_timestamp_sec;

        let velocity_source = if self.frame_state_carried(frame_idx) {
            FrameVelocitySource::Carried
        } else {
            self.refresh_reference_inertial_state()
        };
        if self.debug {
            self.report_frame_velocity(frame_idx, velocity_source);
        }
        self.update_seeded_window();
        self.advance_stillness();

        let (candidate_pose, predicted_velocity) =
            self.predict_frame(frame_idx, pose_before, prev_timestamp, timestamp_sec);
        self.update_imu_trust(frame_idx, timestamp_sec, &candidate_pose);

        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let trusted_prediction = self.imu_confident(timestamp_sec);
        // The outer prediction: a KeyframeDue flow pose shadows
        // `candidate_pose` below and must not become the reference.
        let gate = self.pose_gate_reference(
            candidate_pose,
            predicted_velocity.filter(|_| trusted_prediction),
            timestamp_sec - prev_timestamp,
            currently_lost_for,
        );
        if self.trace_enabled {
            let map = lock_map(&self.map, &self.map_wait);
            self.frame_trace = Some(FrameTrace::begin(
                source.idx(),
                timestamp_sec,
                candidate_pose,
                trusted_prediction,
                &map,
                self.state
                    .current_keyframe_idx
                    .and_then(|ki| map.get_keyframe(ki)),
                self.tracker.tracks(),
                self.tracker.local_map_config(),
            ));
        }
        let survivors = self.flow(previous_image, images, &candidate_pose, trusted_prediction);
        let estimation = Estimation {
            candidate_pose,
            pose_before,
            lost_for_sec: currently_lost_for,
            trusted_prediction,
            gate,
        };
        let flow_only = match &source {
            FrameSource::Deferred(pending) => self.track_flow_only(
                pending.idx,
                current_image,
                survivors.as_deref(),
                &estimation,
                timestamp_sec,
            ),
            FrameSource::Ready(_) => FlowOnly::Unavailable,
        };
        if matches!(source, FrameSource::Deferred(_)) {
            self.count_flow_decision(&flow_only);
        }
        let flow_tracked = matches!(flow_only, FlowOnly::Tracked(..));
        let flow_due = matches!(flow_only, FlowOnly::KeyframeDue(_));
        let (frame, result) = match flow_only {
            FlowOnly::Tracked(frame, estimate) => (*frame, Ok(estimate)),
            outcome => {
                let frame = self.resolve(source)?;
                let candidate_pose = match outcome {
                    FlowOnly::KeyframeDue(pose) => pose,
                    _ => candidate_pose,
                };
                let map = lock_map(&self.map, &self.map_wait);
                let result = self.tracker.estimate(
                    FrameInput {
                        frame: &frame,
                        candidate_pose,
                        pose_before,
                        current_keyframe_idx: self.state.current_keyframe_idx,
                        lost_for_sec: currently_lost_for,
                        trusted_prediction,
                        gate,
                    },
                    survivors,
                    &map,
                    &self.rig.camera,
                    &mut self.tracking_timings,
                );
                drop(map);
                (frame, result)
            }
        };

        self.tracker.spawn_refills();

        let result = match result {
            Ok(mut estimate) => self
                .refine_pose_inertial(&frame, &mut estimate, prev_timestamp, timestamp_sec)
                .then_some(estimate)
                .ok_or(MapProjectionRejectReason::LowPnpInliers),
            Err(reason) => Err(reason),
        }
        .and_then(|estimate| {
            self.judge_pose_innovation(&frame, &estimate, gate.as_ref(), timestamp_sec)
                .map(|()| estimate)
        });
        // S0-C11: an accepted flow pose thrown away for a due keyframe.
        self.counters.flow_due_orb_rejected += u64::from(flow_due && result.is_err());
        let (mut status, (matches, withheld), tracked_inliers, reject_reason) = match result {
            Ok(estimate) => {
                self.note_tracked_step(&candidate_pose, &estimate.pose);
                self.state.pose_world_to_cam = estimate.pose;
                self.note_pose("tracked", frame_idx, &estimate.pose);
                // ORB-SLAM3 tracks against the most-voted keyframe
                // (`Tracking.cc:3602-3606`); `None` under the default rules.
                if let Some(kf_idx) = estimate.most_voted_keyframe {
                    self.state.current_keyframe_idx = Some(kf_idx);
                }

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
                    (estimate.matches, estimate.withheld),
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
                self.note_pose("coast", frame_idx, &candidate_pose);
                // `trusted` is settled in the lost branch below, once the
                // runaway check has seen this pose.
                self.last_skip = Some(SkipKind::Coasting {
                    inertial: predicted_velocity.is_some(),
                    trusted: false,
                });
                (
                    TrackingStatus::Skipped,
                    (Vec::new(), Vec::new()),
                    0,
                    Some(reason),
                )
            }
        };
        match reject_reason {
            Some(MapProjectionRejectReason::RotationGap) => {
                self.counters.gate_rejects_rotation += 1
            }
            Some(MapProjectionRejectReason::TranslationGap) => {
                self.counters.gate_rejects_translation += 1;
            }
            _ => {}
        }
        if self.debug
            && let Some(gate) = &gate
            && let Some(reading) = self.tracker.last_gap()
        {
            self.report_pose_gap(frame.idx, gate, &reading);
        }
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
            if status == TrackingStatus::Tracked {
                self.report_landmark_age(frame.idx, &matches, timestamp_sec);
            }
        }

        // ORB-SLAM3 counts no visibility on a frame without a projection
        // search (`Tracking.cc:3357,3379`).
        let count_visibility =
            !(flow_tracked && self.flow.bookkeeping == FlowBookkeeping::KeyframesOnly);
        if status == TrackingStatus::Tracked && count_visibility {
            // Visibility bookkeeping over the local map only (mirrors
            // ORB-SLAM3, which counts mnVisible on local-map points): full-map
            // scans here would grow with trajectory length.
            // Release this non-reentrant lock before keyframe insertion locks the map.
            {
                let local_map_config = self.tracker.local_map_config();
                let mut map_guard = lock_map(&self.map, &self.map_wait);
                let current_kf = self
                    .state
                    .current_keyframe_idx
                    .and_then(|ki| map_guard.get_keyframe(ki));
                let selection =
                    select_local_map_points(&map_guard, &matches, current_kf, local_map_config);
                if local_map_config.window_start_kf.is_some() {
                    self.counters.seeded_window_frames += 1;
                    self.counters.seeded_window_dropped_points +=
                        selection.dropped_by_window as u64;
                }
                let local_indices = selection.points;
                let visible = crate::tracking::local_map::landmarks_in_frame_frustum(
                    &map_guard,
                    &local_indices,
                    &self.rig.camera,
                    &candidate_pose,
                    &frame,
                );
                map_guard.update_observation_counts(&visible, &matches);
            }
        }
        // A flow frame built without the map has no descriptors; its
        // keyframe decision ran in `track_flow_only`.
        if status == TrackingStatus::Tracked && !(flow_tracked && self.flow_frames_lock_free()) {
            let insert_start = Instant::now();
            let inserted = self.try_insert_keyframe(
                &frame,
                timestamp_sec,
                tracked_inliers,
                TrackedMatches {
                    linked: &matches,
                    withheld: &withheld,
                },
            );
            self.tracking_timings.keyframe_insert += insert_start.elapsed();
            if inserted {
                status = TrackingStatus::KeyframeAccepted;
            }
        }

        // With a settled IMU the system coasts instead (below): a relocalized
        // pose taken from an older part of a drifted map would jump away from
        // the inertial state, which ORB-SLAM3 also avoids in inertial mode.
        if status == TrackingStatus::Skipped && self.relocalization_due(timestamp_sec) {
            let only_while_lost =
                self.relocalization.while_lost && self.imu_confident(timestamp_sec);
            // The own window would hold out the older segment a candidate
            // comes from; it is put back when nothing is found.
            let window = self.seeded_window_start();
            self.tracker.set_window_start(None);
            if self.relocalize(&frame, timestamp_sec, false) {
                status = TrackingStatus::Tracked;
                if only_while_lost {
                    self.counters.relocalizations_while_lost += 1;
                }
            } else {
                self.tracker.set_window_start(window);
            }
        }
        if let Some(trace) = self.frame_trace.as_mut() {
            trace.pose = self.state.pose_world_to_cam;
            trace.tracked = status != TrackingStatus::Skipped;
            trace.keyframe = status == TrackingStatus::KeyframeAccepted;
        }
        self.record_frame_pose(prev_timestamp, timestamp_sec);
        if status == TrackingStatus::Skipped {
            let lost_since = self
                .state
                .mark_lost(timestamp_sec, pose_before, prev_timestamp);
            let recently_lost_for = timestamp_sec - lost_since;

            let imu_confident = self.imu_confident(timestamp_sec);
            self.settle_coast_trust(imu_confident);
            let policy = &self.tracking_loss_recovery;
            let grace_period_sec = policy.grace_period_sec(imu_confident);
            let map_established = lock_map(&self.map, &self.map_wait).keyframes().len()
                > policy.min_keyframes_for_grace;

            if gives_up(map_established, recently_lost_for, grace_period_sec) {
                self.dbg(format!(
                    "[lost] frame={} giving up after {:.2}s (map_established={}): resetting",
                    frame.idx, recently_lost_for, map_established,
                ));
                self.counters.give_ups += 1;
                self.count_young_death(lost_since);
                self.tracker.reset_tracks();
                if self.give_up(&frame, recently_lost_for) {
                    // This frame's coast is already in the seed.
                    self.state.last_frame_timestamp_sec = timestamp_sec;
                }
                return Ok(self.bootstrap_step(frame, timestamp_sec));
            }
            let role = match self.keyframes_while_lost {
                KeyframesWhileLost::Enabled => Some(KeyframeRole::Lost),
                KeyframesWhileLost::Anchor => Some(KeyframeRole::Anchor),
                KeyframesWhileLost::Disabled => None,
            };
            let due = imu_confident && self.keyframe_due_while_lost(timestamp_sec);
            let withheld = self.withhold_lost_keyframe(frame.idx, timestamp_sec, due);
            let due = due && !withheld;
            if due && role.is_some() && self.recovery.keep_inertial && self.runaway.runaway() {
                self.dbg(format!(
                    "[lost] frame={} runaway at {:.1} m/s: no keyframe at the coasted pose",
                    frame.idx,
                    self.runaway.speed_mps().unwrap_or(f64::NAN)
                ));
            } else if due
                && let Some(role) = role
                && self.insert_keyframe(&frame, timestamp_sec, &[], 0, role)
            {
                // Only without `keep_inertial` does a runaway get this far (S0-C1b).
                self.counters.lost_keyframes_at_runaway += u64::from(self.runaway.runaway());
                let idx = frame.idx;
                self.dbg(match role {
                    KeyframeRole::Anchor => {
                        format!("[anchor] frame={idx} inserted at the inertial prediction")
                    }
                    _ => format!("[lost] frame={idx} keyframe inserted at the inertial prediction"),
                });
            }
        } else {
            self.state.mark_tracked();
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        self.prune_imu_after_frame(timestamp_sec);
        Ok(self.frame_result(status))
    }

    /// This frame's optical-flow survivors from the previous capture, started
    /// at the predicted pose's pixels when the prediction is trusted.
    fn flow(
        &mut self,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        candidate_pose: &Pose3d,
        trusted_prediction: bool,
    ) -> Option<Vec<FlowSurvivor>> {
        if let Some(rig_flow) = self.rig_flow.as_mut() {
            let frame_idx = self.next_frame_idx - 1;
            let input = FlowStepInput {
                frame_idx: frame_idx as u64,
                pose_world_to_primary: (trusted_prediction && self.klt_prediction.is_some())
                    .then_some(*candidate_pose),
                primary_rotation: self.gyro_rotation,
            };
            let survivors =
                self.tracker
                    .flow_windowed(rig_flow, &input, &mut self.tracking_timings);
            if self.debug {
                let threads = rig_flow.threads();
                let shard_tracks = rig_flow.output().shard_tracks;
                for (shard, tracks) in shard_tracks.iter().take(threads).enumerate() {
                    self.debug_messages.push(format!(
                        "[flow] frame={frame_idx} shard={shard} tracks={tracks}"
                    ));
                }
            }
            return survivors;
        }
        let map = lock_map(&self.map, &self.map_wait);
        let guess =
            self.klt_prediction
                .filter(|_| trusted_prediction)
                .map(|(max_level, max_iter)| FlowGuess {
                    map: &map,
                    pose: candidate_pose,
                    camera: &self.rig.camera,
                    max_level,
                    max_iter,
                    gyro_rotation: self.gyro_rotation,
                });
        // The pyramidal LK forks per track; under `FanOut::Bounded` it forks
        // on the bounded pool like the frame's ORB.
        let (tracker, timings) = (&mut self.tracker, &mut self.tracking_timings);
        if tracker.flows_rig() {
            let mut previous: Vec<&Image<u8, 1>> = Vec::with_capacity(images.len());
            previous.extend(previous_image);
            previous.extend(self.previous_secondary.iter());
            return self
                .fan_out
                .run(|| tracker.flow_cameras(&previous, images, guess, timings));
        }
        if guess.is_none() {
            // Today's unpredicted flow reads no map.
            drop(map);
            return self
                .fan_out
                .run(|| tracker.flow(previous_image, images[0], None, timings));
        }
        self.fan_out
            .run(|| tracker.flow(previous_image, images[0], guess, timings))
    }

    /// What this frame's tracked pose is checked against: the inertial
    /// `prediction`, only when the IMU produced one (`velocity`) and is
    /// trusted, and only when the gate or the debug log needs it.
    fn pose_gate_reference(
        &self,
        prediction: Pose3d,
        velocity: Option<Vec3F64>,
        dt_sec: f64,
        lost_for_sec: f64,
    ) -> Option<GateReference> {
        let gate = self.tracker.pose_gate();
        (velocity.is_some() && (self.debug || gate.innovation().is_some())).then(|| GateReference {
            prediction,
            dt_sec,
            lost: self
                .state
                .pose_before_loss
                .filter(|_| self.state.lost_since_sec.is_some())
                .map(|before_loss| (before_loss, lost_for_sec + dt_sec)),
        })
    }

    /// One `[predgap]` line: the accepted pose against the inertial prediction.
    fn report_pose_gap(&mut self, idx: usize, gate: &GateReference, reading: &GateReading) {
        let GateReading {
            gap,
            inliers,
            matches,
        } = reading;
        let lost = match gap.recover_mps {
            Some(speed) => format!("1 recover_mps={speed:.3}"),
            None => "0".into(),
        };
        self.dbg(format!(
            "[predgap] frame={idx} dt={:.4} rot={:.3} trans={:.4} step={:.4} step_rot={:.3} \
             inl={inliers} m={matches} lost={lost} trusted=1",
            gate.dt_sec, gap.rot_deg, gap.trans_m, gap.step_m, gap.step_rot_deg,
        ));
    }

    /// Tracks a deferred frame from its flow survivors alone when they carry
    /// it and no keyframe is due; ORB then never runs for this frame. With
    /// `WindowedU8` and `FlowPose::AnalyticGn` the frame and its pose come
    /// from the tracks' stored positions, without the map lock.
    fn track_flow_only(
        &mut self,
        idx: usize,
        current_image: &Image<u8, 1>,
        survivors: Option<&[FlowSurvivor]>,
        estimation: &Estimation,
        timestamp_sec: f64,
    ) -> FlowOnly {
        let (Some(min_tracks), Some(survivors)) = (self.klt_min_tracks, survivors) else {
            return FlowOnly::Unavailable;
        };
        let analytic = match self.flow.pose {
            FlowPose::AnalyticGn { max_iter } => Some(max_iter),
            FlowPose::RansacPnp => None,
        };
        let mut map_guard = None;
        let mut points = Vec::new();
        let (frame, seeded) = match self.rig_flow.as_ref().filter(|_| analytic.is_some()) {
            Some(rig_flow) => {
                let Some((frame, seeded, positions)) = self.tracker.flow_frame_windowed(
                    idx,
                    rig_flow.output(),
                    current_image.size(),
                    min_tracks,
                ) else {
                    return FlowOnly::Unavailable;
                };
                points = positions;
                (frame, seeded)
            }
            None => {
                let waited = Instant::now();
                let map = lock_map(&self.map, &self.map_wait);
                self.tracking_timings.flow_map_wait += waited.elapsed();
                let Some((mut frame, seeded)) =
                    self.tracker
                        .flow_frame(idx, survivors, &map, current_image.size(), min_tracks)
                else {
                    return FlowOnly::Unavailable;
                };
                frame.ensure_undistorted(&self.rig.camera);
                if analytic.is_some() {
                    points = seeded
                        .iter()
                        .map(|&(mp_idx, _)| map.map_points()[mp_idx].position)
                        .collect();
                }
                map_guard = Some(map);
                (frame, seeded)
            }
        };
        let n_seeded = seeded.len();
        let current_keyframe_idx = self.state.current_keyframe_idx;

        let started = Instant::now();
        let solved = analytic.and_then(|max_iter| {
            self.tracker.estimate_flow_analytic(
                flow_input(&frame, estimation, current_keyframe_idx),
                seeded.clone(),
                &points,
                &self.rig.camera,
                max_iter,
            )
        });
        self.tracking_timings.flow_pose += started.elapsed();
        let fell_back = analytic.is_some() && solved.is_none();
        self.counters.flow_pose_fallbacks += u64::from(fell_back);
        let (result, secondary_inliers) = match solved {
            Some(result) => match result {
                Ok((estimate, secondary)) => (Ok(estimate), secondary),
                Err(reason) => (Err(reason), 0),
            },
            None => {
                let map = match map_guard.take() {
                    Some(map) => map,
                    None => {
                        let waited = Instant::now();
                        let map = lock_map(&self.map, &self.map_wait);
                        self.tracking_timings.flow_map_wait += waited.elapsed();
                        map
                    }
                };
                let result = self.tracker.estimate_flow_only(
                    flow_input(&frame, estimation, current_keyframe_idx),
                    seeded,
                    &map,
                    &self.rig.camera,
                    &mut self.tracking_timings,
                );
                let threshold_px = self
                    .tracker
                    .map_projection_config()
                    .pnp
                    .final_reproj_threshold_px;
                let secondary = match &result {
                    Ok(estimate) if frame.is_rig() => secondary_inliers(
                        &frame,
                        &estimate.matches,
                        &map,
                        &estimate.pose,
                        threshold_px,
                    ),
                    _ => 0,
                };
                (result, secondary)
            }
        };
        drop(map_guard);
        let Ok(estimate) = result else {
            self.dbg(format!("[klt-first] frame={idx} refused seeded={n_seeded}"));
            return FlowOnly::Refused;
        };

        let started = Instant::now();
        let (evidence, map_wait) = self.timed_keyframe_evidence(
            &frame,
            estimate.inliers,
            &estimate.matches,
            timestamp_sec,
        );
        self.tracking_timings.flow_map_wait += map_wait;
        if self.keyframe_due(&evidence, &frame, &estimate.matches, timestamp_sec) {
            self.tracking_timings.flow_bookkeeping += started.elapsed();
            self.dbg(format!(
                "[klt-first] frame={idx} keyframe due seeded={n_seeded} inliers={}",
                estimate.inliers
            ));
            return FlowOnly::KeyframeDue(estimate.pose);
        }
        self.tracker.accept_flow_only(&estimate, &frame, survivors);
        self.frontend.observe_flow(&frame.features.keypoints_xy);
        self.tracking_timings.flow_bookkeeping += started.elapsed();
        self.counters.flow_frames += 1;
        self.counters.flow_seeds += n_seeded as u64;
        self.counters.flow_inliers += estimate.inliers as u64;
        self.counters.flow_secondary_inliers += secondary_inliers as u64;
        if self.flow.rig == KltRig::AllCameras || self.rig_flow.is_some() {
            self.dbg(format!(
                "[klt-first] frame={idx} tracked seeded={n_seeded} inliers={} secondary={secondary_inliers} fallback={}",
                estimate.inliers,
                u8::from(fell_back),
            ));
        }
        FlowOnly::Tracked(Box::new(frame), estimate)
    }

    /// Counts what a deferred frame's flow-only attempt gave (card Z);
    /// `flow_frames` counts the tracked ones where they are accepted.
    fn count_flow_decision(&mut self, flow_only: &FlowOnly) {
        let counter = match flow_only {
            FlowOnly::Unavailable => &mut self.counters.flow_unavailable,
            FlowOnly::Refused => &mut self.counters.flow_refused,
            FlowOnly::KeyframeDue(_) => &mut self.counters.flow_keyframe_due,
            FlowOnly::Tracked(..) => return,
        };
        *counter += 1;
    }

    /// Whether a frame tracked from flow alone was built without the map,
    /// so it carries no descriptors and cannot become a keyframe.
    fn flow_frames_lock_free(&self) -> bool {
        self.rig_flow.is_some() && matches!(self.flow.pose, FlowPose::AnalyticGn { .. })
    }

    /// This frame's predicted pose and velocity, from the origin
    /// [`Self::prediction_origin`] picks.
    fn predict_frame(
        &mut self,
        frame_idx: usize,
        pose_before: Pose3d,
        prev_timestamp: f64,
        timestamp_sec: f64,
    ) -> (Pose3d, Option<Vec3F64>) {
        let (prediction_from, prediction_sec) =
            self.prediction_origin(frame_idx, pose_before, prev_timestamp);
        let (candidate_pose, predicted_velocity) =
            self.predict_frame_pose(prediction_from, prediction_sec, timestamp_sec);
        // A ray seen in the previous camera, in the current one: R_cur_prev
        // over this frame only, whatever window the prediction integrated
        // (card E step 9, T5).
        self.gyro_rotation = (self.flow.gyro_guess == GyroGuess::RotatedRay
            && predicted_velocity.is_some())
        .then(|| candidate_pose.rotation * pose_before.rotation.transpose());
        (candidate_pose, predicted_velocity)
    }

    /// The motion model's pose at `timestamp_sec` from `pose_before` at
    /// `prev_timestamp`, advancing the world velocity when the IMU predicted;
    /// that velocity is returned with it.
    fn predict_frame_pose(
        &mut self,
        pose_before: Pose3d,
        prev_timestamp: f64,
        timestamp_sec: f64,
    ) -> (Pose3d, Option<Vec3F64>) {
        // Preintegration is prepared here: the system owns bias and the sample
        // buffer. The motion model chooses between it and the visual model.
        let preintegrated = (self.state.imu_initialized && prev_timestamp > 0.0).then(|| {
            let (preintegrated, samples) = self.inertial.preintegrate_window(
                self.rig.imu_noise(),
                prev_timestamp,
                timestamp_sec,
            );
            self.note_tail_hold(&samples, timestamp_sec);
            preintegrated
        });
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
        (candidate_pose, predicted_velocity)
    }

    /// The local-BA settings of a job submitted now: the run's, with the
    /// inertial schedule's state.
    fn job_local_ba(&self) -> LocalBaParams {
        LocalBaParams {
            recently_initialized: self.inertial.schedule.recently_initialized(),
            newest_marginal: matches!(
                self.tracking_config.imu_trust,
                ImuTrustMode::Covariance { .. }
            ) || self.tracker.pose_gate().innovation().is_some(),
            ..self.local_ba
        }
    }

    /// Counts a `Midpoint` window that held its last sample to `t1`: no sample
    /// after the frame had arrived yet.
    fn note_tail_hold(&mut self, samples: &[ImuMeasurement], t1: f64) {
        if self.sample_interpolation == SampleInterpolation::Midpoint
            && !samples.iter().any(|m| m.timestamp >= t1)
        {
            self.counters.imu_tail_holds += 1;
        }
    }

    /// What the keyframe policy is shown for `frame`: the parallax is taken
    /// against the last keyframe at the tracked pose `state.pose_world_to_cam`,
    /// not at the frame's own (pre-tracking) pose.
    fn keyframe_evidence(
        &self,
        frame: &Frame,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> KeyframeEvidence {
        self.timed_keyframe_evidence(frame, tracked_inliers, matches, timestamp_sec)
            .0
    }

    /// [`Self::keyframe_evidence`] and how long it waited for the map lock,
    /// which a flow frame's `flow_map_wait` includes (T7).
    fn timed_keyframe_evidence(
        &self,
        frame: &Frame,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> (KeyframeEvidence, Duration) {
        let waited = Instant::now();
        let map = lock_map(&self.map, &self.map_wait);
        let map_wait = waited.elapsed();
        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);
        // Measured only when the policy reads it: it walks every match.
        let median_parallax_px = self
            .keyframe_policy
            .min_parallax_px
            .zip(self.state.last_keyframe_idx)
            .and_then(|(_, last)| {
                median_parallax_px(
                    &map,
                    last,
                    frame,
                    &self.state.pose_world_to_cam,
                    matches,
                    &self.rig.camera,
                )
            });
        drop(map);
        let evidence = KeyframeEvidence {
            curr_idx: frame.idx,
            last_keyframe_idx: self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
            median_parallax_px,
            ..KeyframeEvidence::default()
        };
        let (evidence, orb_slam3_wait) =
            self.with_orb_slam3_evidence(evidence, frame, matches, timestamp_sec);
        (evidence, map_wait + orb_slam3_wait)
    }

    /// The policy's decision, or `max_interval_sec` elapsed since the last
    /// keyframe's timestamp (a bound ask); then the still-device gate.
    fn keyframe_due(
        &mut self,
        evidence: &KeyframeEvidence,
        frame: &Frame,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> bool {
        let interval_elapsed = evidence
            .last_keyframe_idx
            .zip(self.inertial.last_keyframe_timestamp_sec)
            .is_some_and(|(last_idx, last_ts)| {
                self.keyframe_policy.interval_elapsed(
                    evidence.curr_idx.saturating_sub(last_idx),
                    timestamp_sec - last_ts,
                )
            });
        let asks = self.keyframe_policy.asks(evidence);
        let asks = KeyframeAsks {
            bound: asks.bound || interval_elapsed,
            ..asks
        };
        if self.keyframe_policy.stillness.observes() {
            self.still_gated_decision(asks, frame, matches, timestamp_sec)
        } else {
            asks.any()
        }
    }

    /// Whether the IMU prediction is trusted enough to coast on.
    fn imu_confident(&self, timestamp_sec: f64) -> bool {
        self.imu_trust(timestamp_sec).trusted
    }

    /// How far the inertial prediction at `timestamp_sec` can be trusted.
    fn imu_trust(&self, timestamp_sec: f64) -> ImuTrust {
        match self.tracking_config.imu_trust {
            ImuTrustMode::TimeSinceInit => ImuTrust::by_time(self.trusted_by_time(timestamp_sec)),
            ImuTrustMode::Covariance { search_sigmas } => {
                self.cached_covariance_trust(timestamp_sec, search_sigmas)
            }
        }
    }

    /// Whether the IMU has been initialized for `min_imu_confidence_sec`.
    fn trusted_by_time(&self, timestamp_sec: f64) -> bool {
        let min_confidence_sec = self.tracking_loss_recovery.min_imu_confidence_sec;
        self.state.imu_initialized
            && self
                .state
                .imu_init_timestamp_sec
                .is_some_and(|t0| timestamp_sec - t0 >= min_confidence_sec)
    }

    /// Whether a lost frame tries relocalization: while the IMU is not
    /// trusted, or on every lost frame with `relocalization.while_lost`.
    fn relocalization_due(&self, timestamp_sec: f64) -> bool {
        !self.imu_confident(timestamp_sec) || self.relocalization.while_lost
    }

    /// First keyframe of a re-seeded segment while it tracks and grows only
    /// against its own keyframes (`reseed.own_window_keyframes`).
    fn seeded_window_start(&self) -> Option<usize> {
        self.tracker.local_map_config().window_start_kf
    }

    /// Sets the own window for this frame: from the second segment on, while
    /// the newest segment holds fewer than `own_window_keyframes` keyframes.
    fn update_seeded_window(&mut self) {
        let own = self.recovery.own_window_keyframes;
        let window = match self.window_segment_start {
            Some(start) if own > 0 => {
                let held = self
                    .map
                    .lock()
                    .unwrap()
                    .keyframes()
                    .iter()
                    .filter(|kf| kf.frame.idx >= start)
                    .count();
                (held < own).then_some(start)
            }
            _ => None,
        };
        if window.is_none() {
            self.window_segment_start = None;
        }
        self.tracker.set_window_start(window);
    }

    /// Whether segment debug lines print: only with a knob that reads the
    /// segments, so the logs of a run without them stay comparable.
    fn segments_reported(&self) -> bool {
        self.recovery.relocalize_when_lost || self.recovery.own_window_keyframes > 0
    }

    /// Ends the segment tracking was lost in. A metric, inertially initialized
    /// map keeps its inertial state, and the next segment starts at the coasted
    /// pose when the coast `lost_for_sec` long is plausible (by its average and
    /// over the runaway window), with an IMU bridge from the last keyframe;
    /// otherwise at the last tracked position, at rest. Either way the seed
    /// takes the gyro's attitude, levelled when the measured specific force
    /// is tilted past `max_tilt_deg`. Under [`SeedPlacement::Held`] the seed
    /// is always held, and only after VIBA2. Returns whether the inertial
    /// state was kept.
    fn give_up(&mut self, frame: &Frame, lost_for_sec: f64) -> bool {
        let coasted = self.state.pose_world_to_cam;
        let before = self.state.pose_before_loss.unwrap_or(coasted);
        counters::raise_milli_max(
            &mut self.counters.coast_mm_at_give_up_max,
            (recovery::camera_centre(&coasted) - recovery::camera_centre(&before)).length(),
        );
        let held = self.recovery.seed == SeedPlacement::Held;
        let keep = self.recovery.keep_inertial && self.state.imu_initialized && frame.is_metric();
        // Under `Held`, gravity is kept only once VIBA2 has refined it.
        let refused = keep && held && self.inertial.schedule.recently_initialized();
        self.counters.keeps_refused_before_viba2 += u64::from(refused);
        if !keep || refused {
            self.runaway.clear();
            self.state.reset_after_loss();
            return false;
        }
        let give_up_sec = self.state.lost_since_sec.unwrap_or(0.0) + lost_for_sec;
        // The last tracked pose is a frame older than the loss (G7).
        let coast_sec = self
            .state
            .pose_before_loss_sec
            .map_or(lost_for_sec, |before_sec| give_up_sec - before_sec);
        let plausible = !held
            && !self.runaway.runaway()
            && recovery::coast_is_plausible(
                &before,
                &coasted,
                coast_sec,
                recovery::MAX_COAST_SPEED_MPS,
            );
        if !plausible {
            self.runaway.clear();
        }
        // The attitude is the gyro's in both cases: only the translation of
        // an implausible coast is untrusted.
        let (mut seed, mut velocity_world, mut anchor) = if plausible {
            let anchor = self
                .state
                .last_keyframe_idx
                .zip(self.inertial.last_keyframe_timestamp_sec)
                .map(|(kf_idx, timestamp_sec)| BridgeAnchor {
                    kf_idx,
                    timestamp_sec,
                });
            (coasted, self.state.velocity_world, anchor)
        } else {
            (recovery::held_at(&coasted, &before), Vec3F64::ZERO, None)
        };
        let gravity_world = self.inertial.gravity_world;
        let measured_up = self.tilt.measured_up();
        let tilt = measured_up.map_or(f64::NAN, |up| recovery::tilt_deg(up, gravity_world));
        let relevel = tilt > recovery::MAX_TILT_DEG;
        if let Some(up) = measured_up.filter(|_| relevel) {
            seed = recovery::leveled(&seed, up, gravity_world);
            velocity_world = Vec3F64::ZERO;
            anchor = None;
            self.tilt.clear();
            self.counters.tilt_relevels += 1;
        }
        let limits = StateEnvelopeLimits::default();
        let bias = &mut self.inertial.bias;
        let mut bias_reset = Vec::new();
        if bias.accel.length() > limits.max_accel_bias {
            bias_reset.push(format!("ba={:.3}", bias.accel.length()));
            bias.accel = Vec3F64::ZERO;
        }
        if bias.gyro.length() > limits.max_gyro_bias {
            bias_reset.push(format!("bg={:.3}", bias.gyro.length()));
            bias.gyro = Vec3F64::ZERO;
        }
        self.counters.bias_resets += bias_reset.len() as u64;
        let travelled =
            (recovery::camera_centre(&coasted) - recovery::camera_centre(&before)).length();
        self.dbg(format!(
            "[recover] frame={} inertial state kept; seed at the {} pose (coast {travelled:.2} m in {lost_for_sec:.2} s), bridge from {:?} tilt={tilt:.1}{}{}",
            frame.idx,
            if plausible { "coasted" } else { "last tracked" },
            anchor.map(|anchor| anchor.kf_idx),
            if relevel { " levelled" } else { "" },
            if bias_reset.is_empty() {
                String::new()
            } else {
                format!(" bias reset {}", bias_reset.join(" "))
            },
        ));
        self.counters.reseeds += 1;
        counters::raise_milli_max(
            &mut self.counters.seed_rotation_mdeg_max,
            angle_deg(&before, &seed),
        );
        self.reseed_held = !plausible;
        if self.reseed_held {
            self.counters.reseeds_held += 1;
        }
        self.counters.seeds_gyro_held += u64::from(held);
        self.reseed_coast_from_sec = anchor.map_or(give_up_sec, |anchor| anchor.timestamp_sec);
        // No refinement re-aligns the world after a re-seed.
        self.inertial.schedule.close();
        self.state
            .reset_after_loss_keeping_inertial(seed, velocity_world, anchor);
        true
    }

    /// The pending bridge anchor, dropped once the gap outgrows
    /// `max_bridge_sec`: preintegration over a longer coast says nothing.
    fn live_bridge_anchor(&mut self, timestamp_sec: f64) -> Option<BridgeAnchor> {
        let max_gap = recovery::MAX_BRIDGE_SEC;
        if self
            .state
            .bridge_anchor
            .is_some_and(|anchor| timestamp_sec - anchor.timestamp_sec > max_gap)
        {
            self.state.bridge_anchor = None;
        }
        self.state.bridge_anchor
    }

    /// Publishes the IMU edge from the last keyframe before a give-up to the
    /// next segment's first, with the covariance inflated for the coast, and
    /// returns the pose-graph gap at its position and rotation sigmas.
    fn publish_bridge(
        &mut self,
        anchor: BridgeAnchor,
        kf_idx: usize,
        timestamp_sec: f64,
    ) -> Option<KeyframeGap> {
        let noise =
            recovery::inflated_noise(self.rig.imu_noise(), recovery::BRIDGE_COVARIANCE_SCALE);
        let edge = self.inertial.keyframe_edge(
            noise,
            anchor.kf_idx,
            kf_idx,
            anchor.timestamp_sec,
            timestamp_sec,
        )?;
        let gap = bridge_gap(&edge.preintegrated.covariance);
        let published = lock_map(&self.map, &self.map_wait).apply_insertion(MapInsertion {
            imu_factors: vec![edge],
            ..Default::default()
        });
        let (message, gap) = match published {
            Ok(_) => (
                format!(
                    "[recover] frame={kf_idx} bridged from kf={} over {:.2}s",
                    anchor.kf_idx,
                    timestamp_sec - anchor.timestamp_sec
                ),
                Some(gap),
            ),
            Err(error) => (
                format!("[recover] frame={kf_idx} bridge rejected: {error}"),
                None,
            ),
        };
        self.dbg(message);
        gap
    }

    /// A re-seed without a bridge (held, levelled or starved) is placed only to
    /// within the coast speed over the time since the last tracked keyframe.
    fn unbridged_reseed_gap(&self, timestamp_sec: f64) -> KeyframeGap {
        let gap_sec = self
            .inertial
            .last_keyframe_timestamp_sec
            .map_or(0.0, |t| (timestamp_sec - t).max(0.0));
        KeyframeGap::Reseeded {
            sigma_m: recovery::MAX_COAST_SPEED_MPS * gap_sec,
            sigma_rad: 0.0,
        }
    }

    fn note_graph_gap(&mut self, kf_idx: usize, gap: KeyframeGap) {
        if let Some(loop_closer) = self.loop_closer.as_mut() {
            loop_closer.note_keyframe_gap(kf_idx, gap);
        }
    }

    /// Recovers the pose against the map from place-recognition candidates
    /// and resumes tracking from it. Needs the loop closer's vocabulary and
    /// keyframe database. `lost`: after a give-up (see
    /// [`recovery::relocalization_adopted`]).
    fn relocalize(&mut self, frame: &Frame, timestamp_sec: f64, lost: bool) -> bool {
        const MAX_CANDIDATES: usize = 5;
        let Some(loop_closer) = self.loop_closer.as_ref() else {
            if lost || self.relocalization.while_lost {
                self.counters.relocalization_without_vocabulary += 1;
            }
            return false;
        };
        self.counters.relocalization_attempts += 1;
        let started = Instant::now();
        // A pose far from where the motion model put the camera belongs to a
        // part of the map that has drifted from here; adopting it would
        // teleport the trajectory.
        let predicted_position = self.state.pose_world_to_cam.inverse().translation;
        let max_jump_m = self.relocalization.max_jump_m;
        let segments = &self.segments;
        let mut refused = Vec::new();
        let found = {
            let map = lock_map(&self.map, &self.map_wait);
            let candidates = loop_closer.relocalization_candidates(
                &frame.features.descriptors,
                &map,
                MAX_CANDIDATES,
            );
            self.tracker
                .relocalize(frame, &candidates, &map, &self.rig.camera, |found| {
                    let jump =
                        (found.estimate.pose.inverse().translation - predicted_position).length();
                    let adopted = recovery::relocalization_adopted(
                        segments,
                        found.keyframe_idx,
                        jump,
                        max_jump_m,
                        lost,
                    );
                    if !adopted {
                        refused.push((found.keyframe_idx, jump));
                    }
                    adopted
                })
        };
        self.counters.relocalization_ns += started.elapsed().as_nanos() as u64;
        self.counters.relocalization_refused_jump += refused.len() as u64;
        for (keyframe_idx, jump) in refused {
            self.dbg(format!(
                "[reloc] frame={} refused kf={keyframe_idx} jump={jump:.2}m",
                frame.idx
            ));
        }
        let Some(found) = found else {
            return false;
        };
        self.counters.relocalizations += 1;
        let segment = self.segments.of(found.keyframe_idx);
        if lost && segment != self.segments.latest() {
            self.counters.relocalizations_older_segment += 1;
        }
        self.dbg(if self.segments_reported() {
            format!(
                "[reloc] frame={} against kf={} (segment {segment}) inliers={}",
                frame.idx, found.keyframe_idx, found.estimate.inliers
            )
        } else {
            format!(
                "[reloc] frame={} against kf={} inliers={}",
                frame.idx, found.keyframe_idx, found.estimate.inliers
            )
        });
        self.adopt_relocalization(
            frame.idx,
            timestamp_sec,
            found.keyframe_idx,
            found.estimate.pose,
            lost,
        );
        true
    }

    /// Tracks on from a relocalized `pose` against keyframe `kf_idx`. After
    /// a give-up into an older segment, the keyframes from this frame on
    /// belong to that segment, and no own window holds it out (G2).
    fn adopt_relocalization(
        &mut self,
        frame_idx: usize,
        timestamp_sec: f64,
        kf_idx: usize,
        pose: Pose3d,
        lost: bool,
    ) {
        self.state.velocity_world = recovery::velocity_across_jump(
            &self.state.pose_world_to_cam,
            &pose,
            self.state.velocity_world,
        );
        self.state.pose_world_to_cam = pose;
        self.note_pose("reloc", frame_idx, &pose);
        self.state.current_keyframe_idx = Some(kf_idx);
        self.state.velocity = None;
        self.estimator.last_relocalization_frame = Some(frame_idx);
        self.estimator.last_relocalization_sec = Some(timestamp_sec);
        self.estimator.frame_prior = None;
        let segment = self.segments.of(kf_idx);
        if lost && segment != self.segments.latest() {
            self.segments.resume(frame_idx, segment);
        }
        self.window_segment_start = None;
    }

    /// Whether to extend the map from this frame while coasting on the IMU:
    /// the policy says the last keyframe is old enough. A stereo keyframe
    /// creates landmarks from its depth; a monocular one through triangulation
    /// against its neighbours at the inertial poses.
    fn keyframe_due_while_lost(&self, timestamp_sec: f64) -> bool {
        [
            self.inertial.last_keyframe_timestamp_sec,
            self.state.last_anchor_timestamp_sec,
        ]
        .into_iter()
        .flatten()
        .reduce(f64::max)
        .is_some_and(|t| self.tracking_loss_recovery.keyframe_due(timestamp_sec - t))
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        tracked_inliers: usize,
        tracked: TrackedMatches<'_>,
    ) -> bool {
        let matches = tracked.linked;
        let evidence = self.keyframe_evidence(frame, tracked_inliers, matches, timestamp_sec);
        if !self.keyframe_due(&evidence, frame, matches, timestamp_sec) {
            return false;
        }
        self.insert_tracked_keyframe(
            frame,
            timestamp_sec,
            tracked,
            tracked_inliers,
            KeyframeRole::Tracked,
        )
    }

    /// Publishes `frame` as a keyframe at the current tracking pose and runs
    /// the mapping work it earns; an anchor earns growth only.
    /// `tracked_inliers`: the frame's PnP inliers, 0 when not tracked.
    fn insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        matches: &[(usize, usize)],
        tracked_inliers: usize,
        role: KeyframeRole,
    ) -> bool {
        let tracked = TrackedMatches {
            linked: matches,
            withheld: &[],
        };
        self.insert_tracked_keyframe(frame, timestamp_sec, tracked, tracked_inliers, role)
    }

    /// [`Self::insert_keyframe`], with withheld matches whose features the
    /// keyframe claims without linking them.
    fn insert_tracked_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        tracked: TrackedMatches<'_>,
        tracked_inliers: usize,
        role: KeyframeRole,
    ) -> bool {
        // Guard: reference KF must exist before we can triangulate.
        if let Some(ki) = self.state.current_keyframe_idx {
            let map = lock_map(&self.map, &self.map_wait);
            if map.get_keyframe(ki).is_none() {
                return false;
            }
        } else {
            return false;
        }

        let mut curr_kf = Keyframe::from_frame(Frame {
            pose_world_to_cam: self.state.pose_world_to_cam,
            ..frame.clone()
        });
        let anchor = role == KeyframeRole::Anchor;
        curr_kf.inserted_while_lost = role != KeyframeRole::Tracked;
        curr_kf.anchor = anchor;
        // Seed the new keyframe with the IMU-propagated velocity and current bias
        // estimate so that VI-BA starts from a reasonable linearisation point rather
        // than zero, which would produce huge residuals on the newest IMU edge.
        if self.state.imu_initialized {
            curr_kf.velocity_world = self.state.velocity_world;
            curr_kf.imu_bias = self.inertial.bias;
        }
        let imu_initialized = self.state.imu_initialized;
        let is_stereo = curr_kf.frame.is_stereo();
        let prev_keyframe = self
            .state
            .last_keyframe_idx
            .zip(self.inertial.last_keyframe_timestamp_sec);

        // Neighbours are captured BEFORE publication: growing against a list
        // that already contains the current keyframe would triangulate it
        // against itself and drop the oldest real neighbour. Selection policy
        // belongs to mapping.
        let growth_plan = {
            let map = lock_map(&self.map, &self.map_wait);
            match self.seeded_window_start() {
                Some(window_start_kf) => keyframe_mapping::prepare_seeded_window_growth(
                    &map,
                    frame.idx,
                    self.growth_neighbours,
                    window_start_kf,
                ),
                None => keyframe_mapping::prepare_keyframe_growth(
                    &map,
                    frame.idx,
                    self.growth_neighbours,
                ),
            }
        };

        let imu_edge = match (
            self.state.last_keyframe_idx,
            self.inertial.last_keyframe_timestamp_sec,
        ) {
            (Some(prev_kf_idx), Some(prev_ts)) if !anchor => self.inertial.keyframe_edge(
                self.rig.imu_noise(),
                prev_kf_idx,
                frame.idx,
                prev_ts,
                timestamp_sec,
            ),
            _ => None,
        };
        if let Some(edge) = &imu_edge {
            self.note_tail_hold(&edge.raw_samples, timestamp_sec);
        }
        let KeyframeInsertion {
            insertion,
            close_points: n_close,
        } = keyframe_mapping::keyframe_insertion(
            curr_kf,
            tracked,
            self.stereo_close_depth,
            &self.rig.camera,
            imu_edge,
        );

        let published = lock_map(&self.map, &self.map_wait).apply_insertion(insertion);
        if let Err(error) = published {
            // Nothing was written, so no tracker or IMU reference may advance to
            // a keyframe the map does not hold.
            self.dbg(format!(
                "[kf] frame={} publication rejected: {error}",
                frame.idx
            ));
            return false;
        }
        if let Some(gap) = self.pending_graph_gap.take() {
            self.note_graph_gap(frame.idx, gap);
        }
        if self.stereo_close_depth.is_some() && is_stereo {
            self.dbg(format!(
                "[kf_stereo] frame={} close_points={}",
                frame.idx, n_close
            ));
        }

        self.state.current_keyframe_idx = Some(frame.idx);
        if anchor {
            // The IMU chain, its sample buffer and the next keyframe's edge
            // stay at the last tracked keyframe.
            self.state.last_anchor_timestamp_sec = Some(timestamp_sec);
        } else {
            self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
            self.state.last_keyframe_idx = Some(frame.idx);
            self.note_wheel_keyframe(prev_keyframe, frame.idx, timestamp_sec);
            // Only the last keyframe's landmarks are adopted, so an anchor
            // keeps its frames (T9).
            if self.flow.adopt_landmarks
                && let Some(rig_flow) = self.rig_flow.as_mut()
                && let Err(error) = rig_flow.keep_keyframe()
            {
                self.dbg(format!(
                    "[flow] frame={} keyframe frames not kept: {error}",
                    frame.idx
                ));
            }
        }

        // Optional mapping work the accepted keyframe earns: grow against the
        // captured neighbours, then forward SearchInNeighbors / Fuse, before
        // local BA so BA sees the extra constraints. Mapping owns the sequence
        // and its lock boundaries; the messages stay here.
        let growth_start = Instant::now();
        match self.keyframe_growth {
            KeyframeGrowth::Tracking => {
                let growth = keyframe_mapping::grow_keyframe(
                    &self.map,
                    growth_plan,
                    &self.rig.camera,
                    self.two_view_init_config.match_config,
                    &self.two_view_init_config.triangulation_config,
                    self.growth_min_baseline_ratio,
                );
                self.report_growth(frame.idx, &growth);
            }
            KeyframeGrowth::Mapping => {
                let queued = self.local_mapping.submit_growth(GrowthJob {
                    plan: growth_plan,
                    match_config: self.two_view_init_config.match_config,
                    triangulation_config: self.two_view_init_config.triangulation_config.clone(),
                    min_baseline_ratio: self.growth_min_baseline_ratio,
                });
                self.counters.growth_queue_max = self.counters.growth_queue_max.max(queued as u64);
                // Synchronous mapping has grown it already, in today's order.
                self.apply_growth_results();
            }
        }
        self.tracking_timings.keyframe_growth += growth_start.elapsed();
        if anchor {
            self.enforce_map_bound();
            self.refresh_track_positions();
            return true;
        }

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            self.refine_inertial_init(timestamp_sec);
        }

        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized,
            imu_t_bc: self.rig.camera_to_body(),
            gravity_world: self.inertial.gravity_world,
            local_ba: self.job_local_ba(),
            wheel_factors: self.wheel_factors(),
            tracked_inliers,
        }) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        // Synchronous mode has a completed correction available immediately;
        // asynchronous mode will deliver it at a later frame boundary.
        self.apply_local_mapping_results();

        let recognition_start = Instant::now();
        self.register_place_recognition(frame.idx);
        self.tracking_timings.loop_recognition += recognition_start.elapsed();
        self.enforce_map_bound();
        // Growth, fusion and an inertial refinement moved landmarks; an
        // asynchronous result refreshes only when it arrives (T3).
        self.refresh_track_positions();

        true
    }

    fn report_growth(&mut self, frame_idx: usize, growth: &KeyframeGrowthResult) {
        if !self.debug {
            return;
        }
        for failure in &growth.pair_failures {
            self.dbg(format!(
                "[kf] frame={frame_idx} pair growth against {} rejected: {}",
                failure.neighbor_keyframe_idx, failure.error
            ));
        }
        self.dbg(format!(
            "[kf] frame={frame_idx} grown={} from {} neighbor kfs",
            growth.landmarks_added, growth.neighbor_count
        ));
        self.dbg(format!(
            "[fuse] frame={frame_idx} fused={}",
            growth.observations_added
        ));
        for error in &growth.fusion_failures {
            self.dbg(format!("[fuse] frame={frame_idx} link refused: {error}"));
        }
    }

    /// VIBA1/VIBA2: progressive re-solves with relaxed priors over the same
    /// (now-growing) window VIBA0 used, mirroring LocalMapping.cc:200-228.
    /// Tracking already runs on VIBA0's result, so a rejection here only means
    /// that stage never runs, not a tracking failure.
    fn refine_inertial_init(&mut self, timestamp_sec: f64) {
        let (Some(start_idx), Some(refinement)) = (
            self.inertial.schedule.start_kf_idx(),
            self.inertial.schedule.due_refinement(timestamp_sec),
        ) else {
            return;
        };
        let stage = refinement.name();

        // Drop the solve's map lock before applying its result with a new lock.
        let init_result = self.inertial.initializer.try_initialize(
            &lock_map(&self.map, &self.map_wait),
            self.rig.camera_to_body(),
            self.inertial.bias,
            start_idx,
            refinement.prior_g(),
            refinement.prior_a(),
            true,
        );
        match init_result {
            Some(init) => {
                let applied = self.apply_inertial_initialization(init, start_idx);
                match applied {
                    Ok(AppliedInitialization {
                        scale,
                        gyro_bias: bg,
                        ..
                    }) => self.dbg(format!(
                        "[imu_init] {stage} accepted: scale_correction={scale:.4} gyro_bias=({:.4},{:.4},{:.4})",
                        bg.x, bg.y, bg.z
                    )),
                    Err(error) => {
                        self.dbg(format!("[imu_init] {stage} alignment refused: {error}"));
                    }
                }
            }
            None => {
                self.dbg(format!(
                    "[imu_init] {stage} rejected: solve failed or invalid scale"
                ));
            }
        }

        self.inertial.schedule.complete(refinement);
    }

    fn loop_closing_context(&self) -> LoopClosingContext {
        LoopClosingContext {
            pose_world_to_cam: self.state.pose_world_to_cam,
            velocity_world: self.state.velocity_world,
            current_keyframe_idx: self.state.current_keyframe_idx,
            imu_initialized: self.state.imu_initialized,
            gravity_world: self.inertial.gravity_world,
        }
    }

    /// Offers a freshly inserted keyframe to loop closing, when enabled, and
    /// applies whatever the closer reports back.
    fn register_place_recognition(&mut self, kf_idx: usize) {
        let context = self.loop_closing_context();
        let Some(loop_closer) = self.loop_closer.as_mut() else {
            return;
        };
        let started = Instant::now();
        let outcome = {
            let mut map = lock_map(&self.map, &self.map_wait);
            let outcome = loop_closer.on_keyframe(&mut map, &self.rig.camera, kf_idx, context);
            if self.rig_flow.is_some() && outcome.tracking_correction.is_some() {
                self.counters.flow_position_refreshes += self.tracker.refresh_positions(&map);
            }
            outcome
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let graph_nodes = loop_closer.graph_nodes();
        if let Some(message) = &outcome.debug_message {
            self.dbg(message.clone());
        }
        if !outcome.events.is_empty() || outcome.queued {
            self.dbg(format!(
                "[loop-time] kf={kf_idx} on_keyframe_ms={elapsed_ms:.1} graph_nodes={graph_nodes} {}",
                format_loop_timing(&outcome.timing)
            ));
        }
        self.adopt_loop_outcome(outcome);
    }

    /// Writes back a background loop solve that has finished, at the frame
    /// boundary, and submits the loops accepted meanwhile. Runs after local
    /// mapping's results, under the publication gate.
    fn apply_loop_closing_results(&mut self) {
        let context = self.loop_closing_context();
        let Some(loop_closer) = self
            .loop_closer
            .as_mut()
            .filter(|closer| closer.runs_in_background())
        else {
            return;
        };
        let outcome = {
            let mut map = lock_map(&self.map, &self.map_wait);
            let outcome = loop_closer.on_frame(&mut map, &self.rig.camera, context);
            if self.rig_flow.is_some() && outcome.tracking_correction.is_some() {
                self.counters.flow_position_refreshes += self.tracker.refresh_positions(&map);
            }
            outcome
        };
        let graph_nodes = loop_closer.graph_nodes();
        if let Some(latency) = outcome.written_back {
            self.dbg(format!(
                "[loop-apply] frame={} applied={} latency_ms={:.1} graph_nodes={graph_nodes} {}",
                self.next_frame_idx - 1,
                outcome.pgo_applied,
                latency.as_secs_f64() * 1000.0,
                format_loop_timing(&outcome.timing)
            ));
        }
        self.adopt_loop_outcome(outcome);
    }

    /// Takes a loop closer's report into the tracking state: the corrected
    /// pose and velocity, a fresh start for optical flow and local BA after a
    /// correction, and the events.
    fn adopt_loop_outcome(&mut self, outcome: LoopClosingOutcome) {
        for message in outcome.gate_messages {
            self.dbg(message);
        }
        if let Some((corrected_tracking_pose, corrected_velocity)) = outcome.tracking_correction {
            self.state.pose_world_to_cam = corrected_tracking_pose;
            self.note_pose(
                "loop",
                self.next_frame_idx.saturating_sub(1),
                &corrected_tracking_pose,
            );
            self.state.velocity_world = corrected_velocity;
        }
        self.counters.pgo_capped += u64::from(outcome.pgo_applied && outcome.pgo_capped);
        if outcome.pgo_applied {
            self.tracker.reset_tracks();
            if self.state.imu_initialized {
                if !self.local_mapping.submit(KeyframeJob {
                    imu_initialized: true,
                    imu_t_bc: self.rig.camera_to_body(),
                    gravity_world: self.inertial.gravity_world,
                    local_ba: self.job_local_ba(),
                    wheel_factors: self.wheel_factors(),
                    tracked_inliers: 0,
                }) {
                    self.dbg("[local_mapping] worker is unavailable after PGO".into());
                }
                self.apply_local_mapping_results();
            }
        }
        self.loop_closure_events.extend(outcome.events);
    }
}

/// The tracker's view of a flow frame `frame`.
fn flow_input<'a>(
    frame: &'a Frame,
    estimation: &Estimation,
    current_keyframe_idx: Option<usize>,
) -> FrameInput<'a> {
    FrameInput {
        frame,
        candidate_pose: estimation.candidate_pose,
        pose_before: estimation.pose_before,
        current_keyframe_idx,
        lost_for_sec: estimation.lost_for_sec,
        trusted_prediction: estimation.trusted_prediction,
        gate: estimation.gate,
    }
}

/// How many of `matches` are rig cameras 1 and up within `threshold_px` of
/// their map point at `pose`.
fn secondary_inliers(
    frame: &Frame,
    matches: &[(usize, usize)],
    map: &Map,
    pose: &Pose3d,
    threshold_px: f64,
) -> usize {
    let threshold_sq = threshold_px * threshold_px;
    matches
        .iter()
        .filter(|&&(mp_idx, kp_idx)| {
            let (Some(view), Some(mp), Some(pixel)) = (
                frame.secondary_view(kp_idx),
                map.map_points().get(mp_idx),
                frame.keypoints_undist.get(kp_idx),
            ) else {
                return false;
            };
            view.camera
                .reprojection_error_sq_world(
                    &view.pose(pose),
                    &mp.position,
                    pixel[0] as f64,
                    pixel[1] as f64,
                )
                .is_some_and(|error_sq| error_sq <= threshold_sq)
        })
        .count()
}

/// The `WindowedU8` flow knobs of `settings`; without a prediction search LK
/// runs kornia-imgproc's default pyramid, as today's kernels do.
fn rig_flow_settings(settings: &SystemSettings) -> RigFlowSettings {
    let flow = settings.frontend_flow;
    let default = kornia_imgproc::optical_flow_pyr_lk::PyrLKParams::default();
    let unpredicted = (default.max_level, default.max_iter);
    RigFlowSettings {
        threads: flow.threads,
        window: flow.window,
        predicted: settings.klt_prediction.unwrap_or(unpredicted),
        unpredicted,
        epsilon: default.epsilon,
        retry: flow.retry,
        forward_backward: flow.forward_backward,
        refill: flow.refill,
        gyro_guess: flow.gyro_guess,
        refill_cameras: RefillCameras::RoundRobin,
    }
}

/// The cameras flow tracks: every rig camera under `klt_rig: AllCameras`,
/// the primary otherwise.
fn rig_flow_cameras(
    rig: &SensorRig,
    frontend: &OrbFrontend,
    klt_rig: KltRig,
) -> Vec<RigFlowCamera> {
    match (klt_rig, frontend.rig_geometry()) {
        (KltRig::AllCameras, Some(geometry)) => geometry
            .views()
            .iter()
            .enumerate()
            .map(|(camera, view)| RigFlowCamera {
                camera: view.camera.clone(),
                camera_from_primary: (camera > 0).then_some(view.camera_from_primary),
            })
            .collect(),
        _ => vec![RigFlowCamera {
            camera: rig.camera.clone(),
            camera_from_primary: None,
        }],
    }
}

/// Whether a lost map is reset now: an established one after its grace
/// period, a young one at once (card I).
fn gives_up(map_established: bool, lost_for_sec: f64, grace_period_sec: f64) -> bool {
    !map_established || lost_for_sec >= grace_period_sec
}

/// A bridged re-seed's gap: the one-sigma bounds of the preintegrated position
/// and rotation blocks (navigation order rot, vel, pos), largest axis.
fn bridge_gap(covariance: &Mat9F64) -> KeyframeGap {
    let sigma = |base: usize| {
        covariance
            .fixed_view::<3, 3>(base, base)
            .into_owned()
            .symmetric_eigenvalues()
            .max()
            .max(0.0)
            .sqrt()
    };
    KeyframeGap::Reseeded {
        sigma_m: sigma(6),
        sigma_rad: sigma(0),
    }
}

fn format_loop_timing(t: &LoopTiming) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    format!(
        "graph_edges={} iterations={} detect_ms={:.1} verify_ms={:.1} solve_ms={:.1} \
         writeback_ms={:.1} geometry_ms={:.1} fuse_ms={:.1} landmarks_refreshed={} \
         fuse_capture_ms={:.1} fuse_plan_ms={:.1} fusion_ops={} fusion_stale={}",
        t.graph_edges,
        t.iterations,
        ms(t.detect),
        ms(t.verify),
        ms(t.solve),
        ms(t.writeback),
        ms(t.geometry),
        ms(t.fuse),
        t.landmarks_refreshed,
        ms(t.fuse_capture),
        ms(t.fuse_plan),
        t.fusion_ops,
        t.fusion_stale,
    )
}

/// Low-match two-view rejections a monocular bootstrap reference survives.
const MAX_LOW_MATCH_REJECTIONS: usize = 10;

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
mod tests;
