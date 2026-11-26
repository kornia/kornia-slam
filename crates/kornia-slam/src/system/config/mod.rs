//! Declarative system configuration: the sensors the system consumes and the
//! pipeline it runs on them.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::inertial::VibaWindow;
use crate::inertial::still::RUN_CHUNK_SEC;
use crate::inertial::trust::ImuTrustMode;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopClosingConfig, LoopGraphInformation, LoopGraphSettings};
use crate::mapping::bundle_adjustment::LargeWindow;
use crate::mapping::dense_llt::DenseParallelism;
use crate::mapping::global_vi_ba::{
    ClockOffsetPrior, FocalSlicePrior, GyroIntrinsicsPrior, RotationSlicePrior,
};
use crate::mapping::map::LocalBaMerge;
use crate::mapping::marginalization::BoundaryPrior;
use crate::mapping::outlier_erase::OutlierErase;
use crate::mapping::visual_information::{FisheyeWeight, VisualInformation};
use crate::mapping::{GrowthNeighbours, KeyframeGrowth, LocalMappingMode};
use crate::odometry::{Odometry, OdometryImuNoise, OdometryTuning, VisualJacobians};
use crate::tracking::local_map::LocalMapRules;
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::pose_estimation::map_projection::{KeyframeObservations, MapProjectionConfig};
use crate::tracking::pose_gate::PoseGate;
use crate::tracking::relocalization::RelocalizationSolver;
use crate::tracking::{KeyframePolicy, KeyframesWhileLost, TrackingLossRecoveryPolicy};
use kornia_sensors::imu::SampleInterpolation;

use super::build::BOUNDARY_IMU_INFO_SCALE;
use super::recovery::RecoveryConfig;
use super::stillness::{HEARTBEAT_SETTLED_SEC, HEARTBEAT_UNSETTLED_SEC};

/// Schema version of [`SystemConfig`] this crate reads and writes.
pub const SYSTEM_CONFIG_VERSION: u32 = 1;

/// Declarative definition of a SLAM system: the sensors it consumes and the
/// pipeline stages, optional branches and settings it runs.
///
/// This is the complete configuration of a [`SlamSystem`](crate::SlamSystem).
/// Settings that follow from the sensor rig, such as the loop-correction IMU
/// gate, are derived when the system is built.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SystemConfig {
    pub version: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sensors: SensorSelection,
    #[cfg_attr(feature = "serde", serde(default))]
    pub pipeline: PipelineDefinition,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            version: SYSTEM_CONFIG_VERSION,
            sensors: SensorSelection::default(),
            pipeline: PipelineDefinition::default(),
        }
    }
}

/// Sensors the pipeline consumes. The source supplies their data and calibration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct SensorSelection {
    pub cameras: CameraSelection,
    pub imu: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CameraSelection {
    #[default]
    Mono,
    /// A rectified pair; the right image only adds depth to left-image features.
    Stereo,
    /// Every camera of the source's multi-camera rig, each tracked with its
    /// own model and extrinsic (ORB-SLAM3's non-rectified stereo).
    Rig,
}

/// Graph family and its stage settings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PipelineDefinition {
    OrbSlam(OrbSlamPipeline),
}

impl Default for PipelineDefinition {
    fn default() -> Self {
        Self::OrbSlam(OrbSlamPipeline::default())
    }
}

/// Feature-based ORB path over a persistent map. Tracking and local mapping
/// are always present; loop closing is optional.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OrbSlamPipeline {
    pub frontend: FrontendConfig,
    pub keyframes: KeyframePolicy,
    pub mapping: MappingConfig,
    pub loop_closing: LoopClosingMode,
    pub tracking_loss: TrackingLossConfig,
    pub tracking: TrackingConfig,
    pub inertial: InertialConfig,
    pub fan_out: FanOut,
    /// Where each frame's pose comes from; `MapTracking` by default.
    pub odometry: Odometry,
    /// Algorithm tuning, set from Rust only: configuration files always use the
    /// defaults.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub tuning: OrbTuning,
}

/// The rayon pools the tracking thread's ORB extraction and the mapping and
/// loop-closing workers fork into.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FanOut {
    /// rayon's process-wide pool for all of them.
    #[default]
    Global,
    /// Extraction on a private pool, and each worker forks only into itself:
    /// rayon's idle workers yield 32 times before sleeping, which on 2 cores
    /// takes time from the BA thread.
    Bounded(BoundedFanOut),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct BoundedFanOut {
    /// Threads of the extraction pool, at least 1.
    pub tracking_threads: usize,
}

impl Default for BoundedFanOut {
    fn default() -> Self {
        Self {
            tracking_threads: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrontendConfig {
    Orb(OrbFrontendConfig),
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self::Orb(OrbFrontendConfig::default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OrbFrontendConfig {
    /// Keypoints extracted per image, within [`OrbFrontendConfig::N_KEYPOINTS_RANGE`].
    pub n_keypoints: usize,
    /// Stereo keypoints closer than this are back-projected directly into map
    /// points at each keyframe. Unused without stereo cameras.
    pub stereo_close_depth: StereoCloseDepth,
    /// Whether frames between keyframes are tracked by optical flow alone,
    /// extracting ORB only when the flow cannot carry the frame.
    pub klt_first: KltFirst,
    /// Optical-flow search when the inertial prediction is trusted.
    pub klt_prediction: KltPrediction,
    /// The pyramidal LK implementation optical flow runs on.
    pub klt_kernel: LkKernel,
    /// Threads the flow step runs on, the tracking thread included: 1, the
    /// rig's camera count, or one more (see `FlowPool`).
    pub flow_threads: usize,
    /// Which rig cameras optical flow tracks.
    pub klt_rig: KltRig,
    /// LK window side, odd; the f32 kernel supports 21 only.
    pub klt_window: usize,
    /// A second, coarser search for the tracks that failed at the first.
    pub klt_retry: KltRetry,
    pub forward_backward: ForwardBackward,
    pub track_refill: TrackRefill,
    /// Tracks start from the landmarks each new keyframe creates.
    pub adopt_landmarks: bool,
    pub gyro_guess: GyroGuess,
    /// How a frame tracked from flow alone gets its pose.
    pub flow_pose: FlowPose,
    pub flow_bookkeeping: FlowBookkeeping,
    pub keyframe_orb: KeyframeOrb,
    /// Weights and search radius of raw fisheye rig views; `Uniform` on a
    /// rig without raw images whatever is set.
    pub fisheye_weight: FisheyeWeight,
}

/// Most flow threads a frontend runs, the tracking thread included.
pub const MAX_FLOW_THREADS: usize = 4;

/// Which rig cameras optical flow tracks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRig {
    /// The primary camera; a rig extracts ORB on every frame.
    #[default]
    PrimaryOnly,
    /// Every rig camera, so a flow-only rig frame keeps each camera's rays.
    AllCameras,
}

/// The search for tracks that failed at the predicted level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRetry {
    #[default]
    Off,
    /// Re-run the failed tracks over levels `0..=max_level` after the join
    /// (VINS-Fusion's retry, `feature_tracker.cpp:118-133`).
    CoarseLevel { max_level: usize },
}

/// Round-trip check: each track is solved back into the previous frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForwardBackward {
    #[default]
    Off,
    /// Drop a track whose round trip exceeds `max_round_trip_px`, checked on
    /// frames whose index is a multiple of `every_n_frames`.
    Enabled {
        max_round_trip_px: f32,
        max_iter: usize,
        every_n_frames: usize,
    },
}

/// Shi-Tomasi refill of a camera's tracks, one camera per frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrackRefill {
    #[default]
    Off,
    /// Refill to `per_camera` tracks when fewer than `refill_below` of them
    /// are left, `min_spacing_px` apart and `border_px` from the edge.
    Enabled {
        per_camera: usize,
        min_spacing_px: f32,
        refill_below: f32,
        border_px: usize,
    },
}

/// Start pixel of a track with no trusted map-point prediction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GyroGuess {
    /// The previous pixel.
    #[default]
    Off,
    /// The previous ray rotated by the preintegrated gyro rotation.
    RotatedRay,
}

/// Pose of a frame tracked from flow alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowPose {
    /// Epipolar filter, then RANSAC PnP.
    #[default]
    RansacPnp,
    /// Gauss-Newton from the IMU prediction; RANSAC PnP below `min_inliers`.
    AnalyticGn { max_iter: usize },
}

/// When tracking counts landmark visibility on flow frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowBookkeeping {
    #[default]
    EveryFrame,
    /// Only on frames that run ORB, as ORB-SLAM3 (`Tracking.cc:3357,3379`).
    KeyframesOnly,
}

/// Where a flow frame that is due as a keyframe extracts its ORB features.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframeOrb {
    #[default]
    Tracking,
    /// On the mapping thread, in keyframe order (`LocalMapping.cc:284-287`).
    Mapping,
}

/// Optical flow from a trusted inertial prediction: each track starts at
/// its map point projected at the predicted pose, and LK searches pyramid
/// levels `0..=max_level` with `max_iter` iterations each instead of the
/// full pyramid. Raw fisheye rigs keep the full search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltPrediction {
    Disabled,
    Enabled { max_level: usize, max_iter: usize },
}

impl Default for KltPrediction {
    fn default() -> Self {
        Self::Enabled {
            max_level: 1,
            max_iter: 10,
        }
    }
}

impl KltPrediction {
    /// `(max_level, max_iter)`, or `None` when disabled.
    pub fn search(self) -> Option<(usize, usize)> {
        match self {
            Self::Disabled => None,
            Self::Enabled {
                max_level,
                max_iter,
            } => Some((max_level, max_iter)),
        }
    }
}

/// Optical-flow-first tracking: a frame is tracked from its flow survivors
/// alone, without ORB, unless a keyframe is due, the flow-only pose is
/// refused, or fewer than `min_tracks` survivors carry a map point.
/// Multi-camera rigs extract ORB on every frame regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltFirst {
    /// ORB on every frame.
    Disabled,
    Enabled {
        min_tracks: usize,
    },
}

impl Default for KltFirst {
    fn default() -> Self {
        Self::Enabled { min_tracks: 80 }
    }
}

impl KltFirst {
    /// The survivor floor, or `None` when disabled.
    pub fn min_tracks(self) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Enabled { min_tracks } => Some(min_tracks),
        }
    }
}

impl OrbFrontendConfig {
    /// Accepted keypoint budgets. Large fisheye images need about 3000 to bootstrap.
    pub const N_KEYPOINTS_RANGE: std::ops::RangeInclusive<usize> = 100..=10_000;
}

impl Default for OrbFrontendConfig {
    fn default() -> Self {
        Self {
            n_keypoints: 1000,
            stereo_close_depth: StereoCloseDepth::default(),
            klt_first: KltFirst::default(),
            klt_prediction: KltPrediction::default(),
            klt_kernel: LkKernel::FixedPointU8,
            flow_threads: 1,
            klt_rig: KltRig::default(),
            klt_window: 21,
            klt_retry: KltRetry::default(),
            forward_backward: ForwardBackward::default(),
            track_refill: TrackRefill::default(),
            adopt_landmarks: false,
            gyro_guess: GyroGuess::default(),
            flow_pose: FlowPose::default(),
            flow_bookkeeping: FlowBookkeeping::default(),
            keyframe_orb: KeyframeOrb::default(),
            fisheye_weight: FisheyeWeight::default(),
        }
    }
}

/// Near/far depth threshold for stereo keypoints (ORB-SLAM3's `ThDepth`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StereoCloseDepth {
    /// A multiple of the stereo baseline; ORB-SLAM3 uses about 35 for EuRoC.
    Baselines(f64),
    Metres(f64),
    /// No direct back-projection; stereo points only enter by triangulation.
    Disabled,
}

impl Default for StereoCloseDepth {
    fn default() -> Self {
        Self::Baselines(35.0)
    }
}

impl StereoCloseDepth {
    /// Threshold in metres for a rig with the given stereo baseline.
    pub fn metres(self, baseline_m: f64) -> Option<f64> {
        match self {
            Self::Baselines(baselines) => Some(baselines * baseline_m),
            Self::Metres(metres) => Some(metres),
            Self::Disabled => None,
        }
    }
}

/// What the system does while tracking is lost.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct TrackingLossConfig {
    /// How frames coasted on the IMU extend the map.
    pub keyframes_while_lost: KeyframesWhileLost,
    /// How the map is re-seeded after a give-up.
    pub reseed: RecoveryConfig,
    /// Whether a tracked pose that disagrees with the inertial prediction is
    /// rejected; a reject coasts, as any other.
    pub pose_gate: PoseGate,
    pub relocalization: LostRelocalization,
}

/// When relocalization runs while tracking is lost, and how.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct LostRelocalization {
    /// Relocalize on every lost frame, also while the IMU is confident;
    /// otherwise only while it is not. Needs loop closing's vocabulary.
    pub while_lost: bool,
    pub solver: RelocalizationSolver,
    /// Farthest a relocalized pose may lie from the coasted one, metres,
    /// within the segment being tracked.
    pub max_jump_m: f64,
}

impl Default for LostRelocalization {
    fn default() -> Self {
        Self {
            while_lost: false,
            solver: RelocalizationSolver::PrimaryEpnp,
            max_jump_m: 0.5,
        }
    }
}

/// Per-frame tracking estimators. The defaults are the behaviour before these
/// settings existed; the alternatives are ORB-SLAM3's, kept selectable.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct TrackingConfig {
    pub pose_inertial: PoseInertial,
    /// How far the inertial prediction is trusted.
    pub imu_trust: ImuTrustMode,
    pub local_map: LocalMapRules,
    pub snapback_fix: SnapbackFix,
    /// The analytic pose refine adopts its last LM step only when the step does
    /// not raise the cost; `false` keeps the generic solver's rule.
    pub refine_monotone: bool,
    pub frame_velocity: FrameVelocity,
    /// The inlier count a pose-inertial solve must keep for its frame to track.
    pub pose_inertial_min_inliers: PoseInertialMinInliers,
}

/// A pose-inertial solve per frame (the idea of PR #195, credit Astik):
/// ORB-SLAM3's `PoseInertialOptimizationLast{Frame,KeyFrame}`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoseInertial {
    /// Visual-only PnP; the IMU only seeds and vetoes it.
    #[default]
    Off,
    /// `PoseInertialOptimizationLastKeyFrame` on every frame: the last
    /// keyframe held fixed.
    LastKeyframe,
    /// ORB-SLAM3's choice per frame: `...LastFrame` under the previous frame's
    /// 15×15 prior; `...LastKeyFrame` on the first frame after the map changed
    /// (`Tracking.cc:2970-2992`).
    OrbSlam3,
}

/// Inlier bar after the pose-inertial solve. Read only with a
/// [`PoseInertial`] mode other than `Off`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoseInertialMinInliers {
    /// The visual PnP bar, `pnp.min_inliers`.
    #[default]
    Visual,
    /// The bar with a trusted inertial prediction, `min_inliers_with_prediction`:
    /// ORB-SLAM3 `TrackLocalMap` fails an IMU-stereo frame below 15 after the
    /// pose-inertial solve.
    WithPrediction,
}

/// The velocity a frame's prediction starts from when the reference is the
/// last keyframe. Read only with [`PoseInertial::Off`]; the pose-inertial
/// modes carry their own frame state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameVelocity {
    /// The keyframe's velocity, at the keyframe's timestamp, every frame.
    #[default]
    KeyframeVelocity,
    /// The keyframe's velocity propagated on the IMU over `[t_kf, t_prev]`
    /// with the keyframe's bias, so a one-frame prediction from the previous
    /// frame starts at that frame's velocity (ORB-SLAM3 `Tracking.cc:1746-1758`).
    PropagatedFromKeyframe,
}

/// What follows a local-BA write-back that moved the reference keyframe.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SnapbackFix {
    #[default]
    Off,
    /// On the first frame after a write-back, a `LastKeyframe` pose-inertial
    /// solve anchored at the corrected keyframe (`Tracking.cc:2982-2991`).
    AnchorNextFrame,
}

/// How IMU samples are integrated, and what the VIBA solves cover.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct InertialConfig {
    pub sample_interpolation: SampleInterpolation,
    /// Keyframes the VIBA solves cover (card J4).
    pub viba_window: VibaWindow,
    /// Multiple of the median edge velocity cost above which a VIBA edge is
    /// dropped and the solve repeated (card J1); `0.0` is off.
    pub viba_velocity_gate: f64,
}

/// Local mapping (keyframe insertion, culling and local BA / VI-BA).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct MappingConfig {
    /// `Synchronous` maps on the tracking thread before each frame returns,
    /// which makes runs replay identically.
    pub execution: LocalMappingMode,
    pub local_ba: LocalBaConfig,
    /// Which tracked matches a new keyframe links; ORB-SLAM3 keeps only the
    /// pose inliers.
    pub keyframe_observations: KeyframeObservations,
    /// Which keyframes a new keyframe triangulates and fuses against.
    pub growth_neighbours: GrowthNeighbours,
    /// Smallest baseline, as a fraction of the new keyframe's median scene depth, from which a keyframe pair grows map points (card K); `0.0` is off.
    pub growth_min_baseline_ratio: f64,
    /// The thread keyframe growth and forward fusion run on.
    pub keyframe_growth: KeyframeGrowth,
}

/// The local BA / VI-BA window and its error model. The defaults are the
/// behaviour before these settings existed; the alternatives are
/// ORB-SLAM3's choices, kept selectable for ablation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct LocalBaConfig {
    /// Newest keyframes optimized; older ones enter as fixed context.
    /// ORB-SLAM3's `LocalInertialBA` uses 10.
    pub active_keyframes: usize,
    /// Levenberg-Marquardt iteration cap of one VI-BA solve. ORB-SLAM3's
    /// `LocalInertialBA` runs 10.
    pub max_iterations: usize,
    /// Levenberg-Marquardt damping a VI-BA solve starts from, outside
    /// `large_window`. Basalt starts every solve at 1e-4 (`vio_config.cpp:78`).
    pub initial_lambda: f64,
    /// Whether that scale also hits the edge's bias random-walk rows
    /// (ORB-SLAM3 leaves them at full weight).
    pub boundary_scales_bias_rows: bool,
    /// The scaled boundary edge, or a marginal prior in its place.
    pub boundary: BoundaryPrior,
    pub stereo_depth_sigma: StereoDepthSigma,
    /// Reprojection weight by keypoint octave, in VI-BA and in the tracking
    /// PnP gates; with `Disparity` it also scales the stereo depth sigma.
    pub visual_information: VisualInformation,
    /// Whether a merge also corrects the keyframes inserted during the solve.
    pub merge: LocalBaMerge,
    pub large_window: LargeWindow,
    /// Calibration states VI-BA estimates online; needs the IMU.
    pub online_calibration: OnlineCalibrationConfig,
    /// Threads of VI-BA's dense factor and solves.
    pub dense_parallelism: DenseParallelism,
    /// Unlink the observations a VI-BA solution rejects (ORB-SLAM3
    /// `LocalInertialBA`); VI-BA only, needs the IMU.
    pub outlier_erase: OutlierErase,
}

/// Calibration states the local VI-BA estimates online; each is off when `None`.
/// Priors: `value_*`/`sigma*` to the factory value, `step_*` to the previous solve.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OnlineCalibrationConfig {
    /// Camera 0: the rotation of the IMU-camera extrinsic, inside the IMU
    /// residual (VINS-Fusion); camera `k ≥ 1`: of its extrinsic to camera 0.
    pub camera_rotation: Option<RotationSlicePrior>,
    pub camera_focal: Option<FocalSlicePrior>,
    pub gyro_intrinsics: Option<GyroIntrinsicsPrior>,
    pub clock_offset: Option<ClockOffsetPrior>,
}

impl OnlineCalibrationConfig {
    /// Whether any state is on.
    pub fn is_enabled(&self) -> bool {
        self.camera_rotation.is_some()
            || self.camera_focal.is_some()
            || self.gyro_intrinsics.is_some()
            || self.clock_offset.is_some()
    }
}

impl Default for LocalBaConfig {
    fn default() -> Self {
        Self {
            active_keyframes: 3,
            max_iterations: 50,
            initial_lambda: 1.0,
            boundary_scales_bias_rows: true,
            boundary: BoundaryPrior::ScaledEdge,
            stereo_depth_sigma: StereoDepthSigma::default(),
            visual_information: VisualInformation::default(),
            merge: LocalBaMerge::SnapshotOnly,
            large_window: LargeWindow::Off,
            online_calibration: OnlineCalibrationConfig::default(),
            dense_parallelism: DenseParallelism::Global,
            outlier_erase: OutlierErase::Off,
        }
    }
}

/// Standard deviation of a stereo depth measurement in BA.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StereoDepthSigma {
    /// A fraction of the measured depth.
    RelativeDepth(f64),
    /// Disparity noise of `px` pixels propagated to depth, `z² · px / (f · b)`.
    Disparity { px: f64 },
}

impl Default for StereoDepthSigma {
    fn default() -> Self {
        Self::RelativeDepth(0.05)
    }
}

/// Loop closing: place recognition over every keyframe, and verification,
/// fusion and pose-graph correction of accepted loops.
///
/// `vocabulary` is a `.bin` from `convert_orbvoc` or a DBoW2 `ORBvoc.txt`,
/// selected by extension. A relative path in a file is resolved against that
/// file's directory.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum LoopClosingMode {
    /// No vocabulary is loaded and no recognition or correction work runs.
    #[default]
    Disabled,
    /// Needs metric input: stereo cameras or the IMU. With the IMU selected,
    /// corrections wait for inertial initialization.
    Enabled {
        vocabulary: PathBuf,
        /// The pose graph accepted loops solve; the default is the snapshot `w * I` solve.
        #[cfg_attr(feature = "serde", serde(default))]
        graph: LoopGraphSettings,
        #[cfg_attr(feature = "serde", serde(default))]
        execution: LoopClosingExecution,
    },
}

/// Where an accepted loop's pose-graph solve runs and when it is written back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoopClosingExecution {
    /// `Synchronous` with synchronous local mapping, else `Asynchronous`.
    FollowMapping,
    /// Solved and written back inside the keyframe that accepted the loop. The default: the
    /// background path accepts fewer loops than this one on the same input, for reasons not yet
    /// explained, so it stays opt-in.
    #[default]
    Synchronous,
    /// Solved inline, written back through the asynchronous path `frames`
    /// frames later (at least one): reproducible runs of the background path.
    Deferred { frames: usize },
    /// Solved on a background thread and written back at the first frame
    /// boundary after it ends.
    Asynchronous,
}

impl LoopClosingExecution {
    /// The execution `FollowMapping` stands for under `mapping`.
    pub fn resolve(self, mapping: LocalMappingMode) -> Self {
        match (self, mapping) {
            (Self::FollowMapping, LocalMappingMode::Synchronous) => Self::Synchronous,
            (Self::FollowMapping, LocalMappingMode::Asynchronous) => Self::Asynchronous,
            (execution, _) => execution,
        }
    }
}

impl LoopClosingMode {
    pub fn vocabulary(&self) -> Option<&Path> {
        match self {
            Self::Disabled => None,
            Self::Enabled { vocabulary, .. } => Some(vocabulary),
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    pub fn graph(&self) -> Option<LoopGraphSettings> {
        match self {
            Self::Disabled => None,
            Self::Enabled { graph, .. } => Some(*graph),
        }
    }
}

impl SystemConfig {
    /// Resolves relative resource paths, such as the vocabulary, against
    /// `base_dir`, usually the directory of the file the configuration came
    /// from. Absolute paths are kept.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        let PipelineDefinition::OrbSlam(orb) = &mut self.pipeline;
        if let LoopClosingMode::Enabled { vocabulary, .. } = &mut orb.loop_closing
            && vocabulary.is_relative()
        {
            *vocabulary = base_dir.join(&*vocabulary);
        }
    }
}

/// Algorithm thresholds of the ORB pipeline that configuration files do not expose.
#[derive(Debug, Clone, Default)]
pub struct OrbTuning {
    /// Monocular two-view map initialization, and the matching and
    /// triangulation settings that keyframe growth shares with it.
    pub initialization: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub loss_recovery: TrackingLossRecoveryPolicy,
    /// Used when loop closing is enabled.
    pub loop_correction: LoopClosingConfig,
    /// Used by `odometry: SlidingWindow`.
    pub odometry: OdometryTuning,
}

/// One line per enabled stage with its resolved settings.
impl fmt::Display for SystemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        let FrontendConfig::Orb(frontend) = orb.frontend;
        writeln!(f, "OrbSlam pipeline (config version {})", self.version)?;
        writeln!(f, "  frontend: ORB, {} keypoints", frontend.n_keypoints)?;
        if frontend.klt_kernel == LkKernel::FixedPointU8 {
            writeln!(f, "  klt kernel: fixed-point u8")?;
        }
        if let Some(min_tracks) = frontend.klt_first.min_tracks() {
            writeln!(
                f,
                "  klt-first: ORB only when a keyframe is due or fewer than {min_tracks} tracks survive"
            )?;
        }
        if let Some((max_level, max_iter)) = frontend.klt_prediction.search() {
            writeln!(
                f,
                "  klt prediction: levels 0..={max_level}, {max_iter} iterations when trusted"
            )?;
        }
        write_flow_knobs(f, &frontend)?;
        if self.sensors.cameras == CameraSelection::Stereo {
            match frontend.stereo_close_depth {
                StereoCloseDepth::Baselines(n) => writeln!(
                    f,
                    "  stereo depth: rectified pair, close within {n} baselines"
                )?,
                StereoCloseDepth::Metres(m) => {
                    writeln!(f, "  stereo depth: rectified pair, close within {m} m")?
                }
                StereoCloseDepth::Disabled => writeln!(f, "  stereo depth: rectified pair")?,
            }
        }
        if self.sensors.cameras == CameraSelection::Rig {
            writeln!(f, "  cameras: multi-camera rig")?;
        }
        if self.sensors.imu {
            writeln!(f, "  IMU integration")?;
        }
        writeln!(f, "  tracking")?;
        if let Some(window) = orb.odometry.sliding_window() {
            write!(
                f,
                "  odometry: sliding window, {:?}, IMU noise ",
                window.mode
            )?;
            match window.imu_noise {
                OdometryImuNoise::Rig => write!(f, "Rig")?,
                OdometryImuNoise::Explicit(n) => write!(
                    f,
                    "explicit (gyro {}, accel {}, gyro walk {}, accel walk {})",
                    n.gyro_noise, n.accel_noise, n.gyro_bias_noise, n.accel_bias_noise
                )?,
            }
            if window.visual_jacobians != VisualJacobians::default() {
                write!(f, ", visual Jacobians {:?}", window.visual_jacobians)?;
            }
            writeln!(f)?;
        }
        if let FanOut::Bounded(bounded) = orb.fan_out {
            writeln!(
                f,
                "  fan-out: bounded, {} extraction threads, workers fork inline",
                bounded.tracking_threads
            )?;
        }
        match orb.tracking_loss.pose_gate {
            PoseGate::Enabled(limits) => writeln!(
                f,
                "  pose gate: {} deg, {} m/s from the prediction, {} m/s while lost",
                limits.max_rotation_gap_deg,
                limits.max_translation_gap_mps,
                limits.max_recover_speed_mps
            )?,
            PoseGate::Innovation(gate) => {
                writeln!(f, "  pose gate: innovation at 99.9 %, {:?}", gate.mode)?
            }
            PoseGate::Disabled => {}
        }
        let k = &orb.keyframes;
        write!(f, "  keyframes: every {}..=", k.min_frames_between)?;
        if k.max_frames_between == usize::MAX {
            write!(f, "unbounded")?;
        } else {
            write!(f, "{}", k.max_frames_between)?;
        }
        write!(f, " frames, ref ratio {}", k.ref_ratio)?;
        if let Some(px) = k.min_parallax_px {
            write!(f, ", parallax {px} px")?;
        }
        match k.max_interval_sec {
            Some(secs) => writeln!(f, ", at most {secs} s apart")?,
            None => writeln!(f)?,
        }
        if k.stillness.observes() {
            writeln!(
                f,
                "  stillness: {:?} (99.9 %, {} s IMU chunks, heartbeat {} s settled, {} s before)",
                k.stillness, RUN_CHUNK_SEC, HEARTBEAT_SETTLED_SEC, HEARTBEAT_UNSETTLED_SEC
            )?;
        }
        let execution = match orb.mapping.execution {
            LocalMappingMode::Synchronous => "synchronous",
            LocalMappingMode::Asynchronous => "asynchronous",
        };
        writeln!(f, "  local mapping: {execution}")?;
        if orb.mapping.keyframe_observations != KeyframeObservations::default() {
            writeln!(
                f,
                "  keyframe observations: {:?}",
                orb.mapping.keyframe_observations
            )?;
        }
        if orb.mapping.growth_min_baseline_ratio > 0.0 {
            writeln!(
                f,
                "  growth: pairs under baseline/depth {} skipped",
                orb.mapping.growth_min_baseline_ratio
            )?;
        }
        if orb.mapping.keyframe_growth != KeyframeGrowth::default() {
            writeln!(f, "  keyframe growth: {:?}", orb.mapping.keyframe_growth)?;
        }
        if orb.tracking.frame_velocity != FrameVelocity::default() {
            writeln!(f, "  frame velocity: {:?}", orb.tracking.frame_velocity)?;
        }
        if orb.tracking.pose_inertial_min_inliers != PoseInertialMinInliers::default() {
            writeln!(
                f,
                "  pose-inertial min inliers: {:?}",
                orb.tracking.pose_inertial_min_inliers
            )?;
        }
        let ba = &orb.mapping.local_ba;
        if *ba != LocalBaConfig::default() {
            writeln!(
                f,
                "  local BA: {} active keyframes, {} iterations from lambda {}, \
                 boundary IMU scale {} (bias rows {}), \
                 boundary {:?}, IMU Huber AllEdges, depth sigma {:?}, visual information {:?}, \
                 state envelope Off, lost keyframes Free, visual kernel L2",
                ba.active_keyframes,
                ba.max_iterations,
                ba.initial_lambda,
                BOUNDARY_IMU_INFO_SCALE,
                if ba.boundary_scales_bias_rows {
                    "scaled"
                } else {
                    "full"
                },
                ba.boundary,
                ba.stereo_depth_sigma,
                ba.visual_information,
            )?;
        }
        if ba.merge != LocalBaMerge::default() {
            writeln!(f, "  local BA merge: {:?}", ba.merge)?;
        }
        if ba.dense_parallelism != DenseParallelism::default() {
            writeln!(
                f,
                "  local BA dense parallelism: {:?}",
                ba.dense_parallelism
            )?;
        }
        if ba.online_calibration.is_enabled() {
            writeln!(f, "  online calibration: {:?}", ba.online_calibration)?;
        }
        if ba.outlier_erase != OutlierErase::default() {
            writeln!(f, "  local BA outlier erase: {:?}", ba.outlier_erase)?;
        }
        if let LoopClosingMode::Enabled {
            vocabulary,
            execution,
            graph,
        } = &orb.loop_closing
        {
            let information = match graph.information {
                LoopGraphInformation::ScalarWeights => "w*I",
                LoopGraphInformation::Drift { .. } => "drift",
            };
            let kept = if graph.persistent {
                "persistent"
            } else {
                "snapshot"
            };
            let capped = if graph.accept_capped_solves {
                ", capped solves accepted"
            } else {
                ""
            };
            write!(
                f,
                "  loop closing: {} (pose graph: {information}, {kept}{capped})",
                vocabulary.display()
            )?;
            match execution {
                LoopClosingExecution::FollowMapping => writeln!(f, ", following mapping")?,
                LoopClosingExecution::Synchronous => writeln!(f)?,
                LoopClosingExecution::Deferred { frames } => {
                    writeln!(f, ", deferred by {frames} frames")?
                }
                LoopClosingExecution::Asynchronous => writeln!(f, ", asynchronous")?,
            }
        }
        Ok(())
    }
}

/// One line per 1 ms frontend knob that is not at its default.
fn write_flow_knobs(f: &mut fmt::Formatter<'_>, frontend: &OrbFrontendConfig) -> fmt::Result {
    let default = OrbFrontendConfig::default();
    if frontend.klt_kernel == LkKernel::WindowedU8 {
        writeln!(f, "  klt kernel: window-only fixed-point u8")?;
    }
    if frontend.flow_threads != default.flow_threads {
        writeln!(f, "  flow threads: {}", frontend.flow_threads)?;
    }
    if frontend.klt_rig != default.klt_rig {
        writeln!(f, "  klt rig: {:?}", frontend.klt_rig)?;
    }
    if frontend.klt_window != default.klt_window {
        writeln!(f, "  klt window: {}", frontend.klt_window)?;
    }
    if frontend.klt_retry != default.klt_retry {
        writeln!(f, "  klt retry: {:?}", frontend.klt_retry)?;
    }
    if frontend.forward_backward != default.forward_backward {
        writeln!(f, "  forward-backward: {:?}", frontend.forward_backward)?;
    }
    if frontend.track_refill != default.track_refill {
        writeln!(f, "  track refill: {:?}", frontend.track_refill)?;
    }
    if frontend.adopt_landmarks {
        writeln!(f, "  flow adopts new landmarks")?;
    }
    if frontend.gyro_guess != default.gyro_guess {
        writeln!(f, "  gyro guess: {:?}", frontend.gyro_guess)?;
    }
    if frontend.flow_pose != default.flow_pose {
        writeln!(f, "  flow pose: {:?}", frontend.flow_pose)?;
    }
    if frontend.flow_bookkeeping != default.flow_bookkeeping {
        writeln!(f, "  flow bookkeeping: {:?}", frontend.flow_bookkeeping)?;
    }
    if frontend.keyframe_orb != default.keyframe_orb {
        writeln!(f, "  keyframe ORB: {:?}", frontend.keyframe_orb)?;
    }
    if frontend.fisheye_weight != default.fisheye_weight {
        writeln!(f, "  fisheye weight: {:?}", frontend.fisheye_weight)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
