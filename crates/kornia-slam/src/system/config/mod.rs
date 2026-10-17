//! Declarative system configuration: the sensors the system consumes and the
//! pipeline it runs on them.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopClosingConfig;
use crate::mapping::LocalMappingMode;
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
    /// Optional AprilTag metric anchor stage: observes a known tag in
    /// keyframes and applies one post-hoc Sim3 (or gravity-preserving)
    /// correction. Disabled leaves the map untouched.
    #[cfg(feature = "apriltag")]
    pub apriltag_anchor: AprilTagAnchorMode,
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum LoopClosingMode {
    /// No vocabulary is loaded and no recognition or correction work runs.
    #[default]
    Disabled,
    /// Needs metric input: stereo cameras or the IMU. With the IMU selected,
    /// corrections wait for inertial initialization.
    Enabled { vocabulary: PathBuf },
}

impl LoopClosingMode {
    pub fn vocabulary(&self) -> Option<&Path> {
        match self {
            Self::Disabled => None,
            Self::Enabled { vocabulary } => Some(vocabulary),
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

/// Optional AprilTag metric-anchor stage, enabled from a run file (or in Rust).
#[cfg(feature = "apriltag")]
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum AprilTagAnchorMode {
    /// No tag detection: the map keeps its arbitrary monocular scale.
    #[default]
    Disabled,
    /// Observe this tag and apply one post-hoc metric correction.
    Enabled {
        /// Decode only this tag id.
        tag_id: u16,
        /// Physical tag size in metres (edge length of the black square).
        tag_size_m: f64,
        /// Tag family, by the detector's canonical name (`"tag36_h11"`, ...);
        /// parsed by `kornia-apriltag`'s `FromStr`, so the crate stays the one
        /// source of truth for which families exist.
        family: String,
    },
}

#[cfg(feature = "apriltag")]
impl AprilTagAnchorMode {
    /// Resolves the stage into its anchor configuration, or `None` when
    /// disabled or the family name is unknown (`validate` rejects the latter
    /// before the system is built). Detector tuning (decision margin, Hamming
    /// distance, robust weights) stays Rust-side, the `OrbTuning` precedent; a
    /// run file sets only the id, size and family.
    pub(crate) fn config(&self) -> Option<crate::apriltag_anchor::AprilTagAnchorConfig> {
        match self {
            Self::Disabled => None,
            Self::Enabled {
                tag_id,
                tag_size_m,
                family,
            } => Some(crate::apriltag_anchor::AprilTagAnchorConfig {
                tag_id: *tag_id,
                tag_size_m: *tag_size_m,
                family: family.parse().ok()?,
                ..Default::default()
            }),
        }
    }
}

impl SystemConfig {
    /// Resolves relative resource paths, such as the vocabulary, against
    /// `base_dir`, usually the directory of the file the configuration came
    /// from. Absolute paths are kept.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        let PipelineDefinition::OrbSlam(orb) = &mut self.pipeline;
        if let LoopClosingMode::Enabled { vocabulary } = &mut orb.loop_closing
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
        if let Some(vocabulary) = orb.loop_closing.vocabulary() {
            writeln!(f, "  loop closing: {}", vocabulary.display())?;
        }
        #[cfg(feature = "apriltag")]
        if let AprilTagAnchorMode::Enabled {
            tag_id,
            tag_size_m,
            family,
        } = &orb.apriltag_anchor
        {
            writeln!(
                f,
                "  apriltag anchor: id={tag_id} size={tag_size_m} m family={family}"
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
