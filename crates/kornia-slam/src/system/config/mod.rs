//! Declarative system configuration: the sensors the system consumes and the
//! pipeline it runs on them.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopClosingConfig, LoopGraphInformation, LoopGraphSettings};
use crate::mapping::LocalMappingMode;
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
use crate::tracking::{KeyframePolicy, TrackingLossRecoveryPolicy};

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
    /// Algorithm tuning, set from Rust only: configuration files always use the
    /// defaults.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub tuning: OrbTuning,
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
    /// Threads the `WindowedU8` flow step runs on, the tracking thread
    /// included: 1, the rig's camera count, or one more.
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
}

/// Most flow threads a frontend runs, the tracking thread included.
pub const MAX_FLOW_THREADS: usize = 4;

/// Largest LK window side: three times OpenCV's and VINS-Fusion's 21. The
/// fixed-point pyramid pads every level by the window, so an unbounded one
/// allocates without bound.
pub const MAX_KLT_WINDOW: usize = 63;

/// Which rig cameras optical flow tracks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRig {
    /// The primary camera; a multi-camera rig extracts ORB on every frame.
    #[default]
    PrimaryOnly,
    /// Every rig camera, so a flow-only rig frame keeps each camera's rays;
    /// needs `sensors.cameras: Rig`.
    AllCameras,
}

/// The search for `WindowedU8` tracks that failed at the first level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRetry {
    #[default]
    Off,
    /// Re-run the failed tracks over levels `0..=max_level` after the join
    /// (VINS-Fusion's retry, `feature_tracker.cpp:118-133`).
    CoarseLevel { max_level: usize },
}

/// Round-trip check of `WindowedU8` flow: each track is solved back into the
/// previous frame.
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

/// Shi-Tomasi refill of a camera's `WindowedU8` tracks, one camera per frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrackRefill {
    #[default]
    Off,
    /// Refill to `per_camera` tracks when fewer than `refill_below` of them
    /// are left, `min_spacing_px` apart and `border_px` from the edge. It
    /// runs inside a flow step, so a rig with no track left waits for ORB.
    Enabled {
        per_camera: usize,
        min_spacing_px: f32,
        refill_below: f32,
        border_px: usize,
    },
}

/// Start pixel of a track with no map-point prediction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GyroGuess {
    /// The previous pixel.
    #[default]
    Off,
    /// The previous ray rotated by the preintegrated gyro rotation. The
    /// full-pyramid kernels read it on the trusted predicted search only, so
    /// they need `klt_prediction`.
    RotatedRay,
}

/// Pose of a frame tracked from flow alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowPose {
    /// Epipolar filter, then RANSAC PnP.
    #[default]
    RansacPnp,
    /// Gauss-Newton from the prediction, no RANSAC; RANSAC PnP below
    /// `min_inliers`. On `WindowedU8` it reads the tracks' stored landmark
    /// positions, not the map.
    AnalyticGn { max_iter: usize },
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
            klt_kernel: LkKernel::default(),
            flow_threads: 1,
            klt_rig: KltRig::default(),
            klt_window: 21,
            klt_retry: KltRetry::default(),
            forward_backward: ForwardBackward::default(),
            track_refill: TrackRefill::default(),
            adopt_landmarks: false,
            gyro_guess: GyroGuess::default(),
            flow_pose: FlowPose::default(),
        }
    }
}

/// Optical-flow-first tracking: a frame is tracked from its flow survivors
/// alone, without ORB, unless a keyframe is due, the flow-only pose is
/// refused, or fewer than `min_tracks` survivors carry a map point.
/// A multi-camera rig extracts ORB on every frame unless `klt_rig` is
/// `AllCameras`; a raw fisheye primary does regardless.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltFirst {
    /// ORB on every frame.
    #[default]
    Disabled,
    Enabled {
        min_tracks: usize,
    },
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

/// Optical flow from a trusted inertial prediction: each track starts at
/// its map point projected at the predicted pose, and LK searches pyramid
/// levels `0..=max_level` with `max_iter` iterations each instead of the
/// full pyramid. Raw fisheye primaries keep the full search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltPrediction {
    #[default]
    Disabled,
    Enabled {
        max_level: usize,
        max_iter: usize,
    },
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

/// Local mapping (keyframe insertion, culling and local BA / VI-BA).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct MappingConfig {
    /// `Synchronous` maps on the tracking thread before each frame returns,
    /// which makes runs replay identically.
    pub execution: LocalMappingMode,
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
        /// The pose graph accepted loops solve; the default is the `w * I` solve.
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
    /// background modes are opt-in.
    #[default]
    Synchronous,
    /// Solved inline, written back through the asynchronous path `frames` frames later (at
    /// least one): reproducible runs of the background path. It differs from `Synchronous` in
    /// three ways: the camera follows the correction most of its local landmarks take, not its
    /// reference keyframe's; fusion is planned on the copy of the loop neighbourhoods taken at
    /// submit and each operation is checked against the live map at writeback; and keyframes
    /// created during the solve are not fused.
    AsynchronousReplay { frames: usize },
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
}

/// One line per enabled stage with its resolved settings.
impl fmt::Display for SystemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        let FrontendConfig::Orb(frontend) = orb.frontend;
        writeln!(f, "OrbSlam pipeline (config version {})", self.version)?;
        writeln!(f, "  frontend: ORB, {} keypoints", frontend.n_keypoints)?;
        match frontend.klt_kernel {
            LkKernel::F32 => {}
            LkKernel::FixedPointU8 => writeln!(f, "  klt kernel: fixed-point u8")?,
            LkKernel::WindowedU8 => writeln!(f, "  klt kernel: window-only fixed-point u8")?,
        }
        match frontend.klt_first.min_tracks() {
            Some(_)
                if self.sensors.cameras == CameraSelection::Rig
                    && frontend.klt_rig != KltRig::AllCameras =>
            {
                writeln!(f, "  klt-first: inactive on a rig, ORB on every frame")?;
            }
            Some(min_tracks) => writeln!(
                f,
                "  klt-first: ORB when a keyframe is due, the flow pose is refused or fewer than {min_tracks} tracks survive"
            )?,
            None => {}
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
        let execution = match orb.mapping.execution {
            LocalMappingMode::Synchronous => "synchronous",
            LocalMappingMode::Asynchronous => "asynchronous",
        };
        writeln!(f, "  local mapping: {execution}")?;
        if let LoopClosingMode::Enabled {
            vocabulary,
            graph,
            execution,
        } = &orb.loop_closing
        {
            let information = match graph.information {
                LoopGraphInformation::ScalarWeights => "w*I",
                LoopGraphInformation::Drift { .. } => "drift",
            };
            write!(
                f,
                "  loop closing: {} (pose graph: {information})",
                vocabulary.display()
            )?;
            match execution {
                LoopClosingExecution::FollowMapping => writeln!(f, ", following mapping")?,
                LoopClosingExecution::Synchronous => writeln!(f)?,
                LoopClosingExecution::AsynchronousReplay { frames } => {
                    writeln!(f, ", asynchronous replay after {frames} frames")?
                }
                LoopClosingExecution::Asynchronous => writeln!(f, ", asynchronous")?,
            }
        }
        Ok(())
    }
}

/// One line per optical-flow knob that is not at its default.
fn write_flow_knobs(f: &mut fmt::Formatter<'_>, frontend: &OrbFrontendConfig) -> fmt::Result {
    let default = OrbFrontendConfig::default();
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
    Ok(())
}

#[cfg(test)]
mod tests;
