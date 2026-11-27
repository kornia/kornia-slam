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
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrontendConfig {
    Orb(OrbFrontendConfig),
    /// CLIDD features exported offline, for a system without loop closing;
    /// build it with
    /// [`SlamSystem::build_precomputed`](crate::SlamSystem::build_precomputed).
    Precomputed(PrecomputedFrontendConfig),
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

/// Features read from a file instead of extracted; see
/// `scripts/export_clidd_features.py` for the format. Relative paths in a
/// file are resolved against that file's directory.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct PrecomputedFrontendConfig {
    /// Features of the images the system is given: the rectified left view
    /// with stereo cameras.
    pub path: PathBuf,
    /// Features of the rectified right view, looked up by the left image's
    /// timestamp. Required with stereo cameras, unused without.
    #[cfg_attr(feature = "serde", serde(default))]
    pub right_path: Option<PathBuf>,
    /// As [`OrbFrontendConfig::stereo_close_depth`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub stereo_close_depth: StereoCloseDepth,
}

impl PrecomputedFrontendConfig {
    /// Monocular features from `path`.
    pub fn mono(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            right_path: None,
            stereo_close_depth: StereoCloseDepth::default(),
        }
    }

    /// Rectified stereo features, `left` and `right`.
    pub fn stereo(left: impl Into<PathBuf>, right: impl Into<PathBuf>) -> Self {
        Self {
            right_path: Some(right.into()),
            ..Self::mono(left)
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
        if let FrontendConfig::Precomputed(precomputed) = &mut orb.frontend {
            for path in std::iter::once(&mut precomputed.path).chain(&mut precomputed.right_path) {
                if path.is_relative() {
                    *path = base_dir.join(&*path);
                }
            }
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
        writeln!(f, "OrbSlam pipeline (config version {})", self.version)?;
        match &orb.frontend {
            FrontendConfig::Orb(frontend) => {
                writeln!(f, "  frontend: ORB, {} keypoints", frontend.n_keypoints)?;
                if self.sensors.cameras == CameraSelection::Stereo {
                    write_stereo_depth(f, frontend.stereo_close_depth)?;
                }
            }
            FrontendConfig::Precomputed(precomputed) => {
                writeln!(
                    f,
                    "  frontend: precomputed CLIDD from {}",
                    precomputed.path.display()
                )?;
                if self.sensors.cameras == CameraSelection::Stereo {
                    if let Some(right) = &precomputed.right_path {
                        writeln!(f, "  right view: {}", right.display())?;
                    }
                    write_stereo_depth(f, precomputed.stereo_close_depth)?;
                }
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
        Ok(())
    }
}

fn write_stereo_depth(f: &mut fmt::Formatter<'_>, close: StereoCloseDepth) -> fmt::Result {
    match close {
        StereoCloseDepth::Baselines(n) => writeln!(
            f,
            "  stereo depth: rectified pair, close within {n} baselines"
        ),
        StereoCloseDepth::Metres(m) => {
            writeln!(f, "  stereo depth: rectified pair, close within {m} m")
        }
        StereoCloseDepth::Disabled => writeln!(f, "  stereo depth: rectified pair"),
    }
}

#[cfg(test)]
mod tests;
