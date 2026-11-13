//! Declarative system configuration: the sensors the system consumes and the
//! pipeline it runs on them.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopClosingConfig, LoopGraphInformation, LoopGraphSettings};
use crate::mapping::LocalMappingMode;
use crate::mapping::bundle_adjustment::{LostKeyframes, VisualKernel};
use crate::mapping::map::StateEnvelope;
use crate::mapping::marginalization::BoundaryPrior;
use crate::mapping::vi_ba_schur::ImuHuber;
use crate::mapping::visual_information::VisualInformation;
use crate::tracking::pose_estimation::map_projection::{KeyframeObservations, MapProjectionConfig};
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
    /// Information scale of the IMU edge from the fixed past into the window.
    pub boundary_imu_info_scale: f64,
    /// Whether that scale also hits the edge's bias random-walk rows
    /// (ORB-SLAM3 leaves them at full weight).
    pub boundary_scales_bias_rows: bool,
    /// The scaled boundary edge, or a marginal prior in its place.
    pub boundary: BoundaryPrior,
    pub imu_huber: ImuHuber,
    pub stereo_depth_sigma: StereoDepthSigma,
    /// Reprojection weight by keypoint octave, in VI-BA and in the tracking
    /// PnP gates; with `Disparity` it also scales the stereo depth sigma.
    pub visual_information: VisualInformation,
    /// Refuse a VI-BA result whose velocity or bias leaves these bounds.
    pub state_envelope: StateEnvelope,
    pub lost_keyframes: LostKeyframes,
    /// The visual-only local BA's kernel; VI-BA is always robust.
    pub visual_kernel: VisualKernel,
}

impl Default for LocalBaConfig {
    fn default() -> Self {
        Self {
            active_keyframes: 3,
            boundary_imu_info_scale: 1e-2,
            boundary_scales_bias_rows: true,
            boundary: BoundaryPrior::ScaledEdge,
            imu_huber: ImuHuber::AllEdges,
            stereo_depth_sigma: StereoDepthSigma::default(),
            visual_information: VisualInformation::default(),
            state_envelope: StateEnvelope::Off,
            lost_keyframes: LostKeyframes::Free,
            visual_kernel: VisualKernel::L2,
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
}

/// One line per enabled stage with its resolved settings.
impl fmt::Display for SystemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        let FrontendConfig::Orb(frontend) = orb.frontend;
        writeln!(f, "OrbSlam pipeline (config version {})", self.version)?;
        writeln!(f, "  frontend: ORB, {} keypoints", frontend.n_keypoints)?;
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
        if orb.mapping.keyframe_observations != KeyframeObservations::default() {
            writeln!(
                f,
                "  keyframe observations: {:?}",
                orb.mapping.keyframe_observations
            )?;
        }
        let ba = &orb.mapping.local_ba;
        if *ba != LocalBaConfig::default() {
            writeln!(
                f,
                "  local BA: {} active keyframes, boundary IMU scale {} (bias rows {}), \
                 boundary {:?}, IMU Huber {:?}, depth sigma {:?}, visual information {:?}, \
                 state envelope {:?}, lost keyframes {:?}, visual kernel {:?}",
                ba.active_keyframes,
                ba.boundary_imu_info_scale,
                if ba.boundary_scales_bias_rows {
                    "scaled"
                } else {
                    "full"
                },
                ba.boundary,
                ba.imu_huber,
                ba.stereo_depth_sigma,
                ba.visual_information,
                ba.state_envelope,
                ba.lost_keyframes,
                ba.visual_kernel,
            )?;
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
            write!(
                f,
                "  loop closing: {} (pose graph: {information}, {kept})",
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

#[cfg(test)]
mod tests;
