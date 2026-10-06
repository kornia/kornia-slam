use std::path::{Path, PathBuf};

use crate::tracking::KeyframePolicy;

/// Schema version of [`PipelineConfig`] this crate reads and writes.
pub const PIPELINE_CONFIG_VERSION: u32 = 1;

/// Declarative definition of a SLAM pipeline: the sensors it consumes and the
/// stages, optional branches and settings it runs.
///
/// This is the public file contract. It maps onto the internal runtime
/// configuration rather than mirroring it, so internal renames do not break
/// existing files.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct PipelineConfig {
    pub version: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sensors: SensorSelection,
    #[cfg_attr(feature = "serde", serde(default))]
    pub pipeline: PipelineDefinition,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            version: PIPELINE_CONFIG_VERSION,
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
#[derive(Debug, Clone, PartialEq)]
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
/// are always present; loop closing adds optional branches.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OrbSlamPipeline {
    pub frontend: FrontendConfig,
    pub keyframes: KeyframeConfig,
    pub mapping: MappingConfig,
    pub loop_closing: LoopClosingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrontendConfig {
    Orb(OrbFrontendConfig),
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self::Orb(OrbFrontendConfig::default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OrbFrontendConfig {
    /// Keypoints extracted per image, within [`OrbFrontendConfig::N_KEYPOINTS_RANGE`].
    pub n_keypoints: usize,
}

impl OrbFrontendConfig {
    /// Accepted keypoint budgets. Large fisheye images need about 3000 to bootstrap.
    pub const N_KEYPOINTS_RANGE: std::ops::RangeInclusive<usize> = 100..=10_000;
}

impl Default for OrbFrontendConfig {
    fn default() -> Self {
        Self { n_keypoints: 1000 }
    }
}

/// Keyframe insertion policy, in frames since the last keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct KeyframeConfig {
    pub min_frames_between: usize,
    /// A keyframe is forced once this many frames have passed.
    pub max_frames_between: usize,
    /// Tracked inliers below this fraction of the reference keyframe's map
    /// points trigger a keyframe; in `[0, 1]`.
    pub ref_ratio: f64,
}

impl Default for KeyframeConfig {
    fn default() -> Self {
        let policy = KeyframePolicy::default();
        Self {
            min_frames_between: policy.min_frames_between,
            max_frames_between: policy.max_frames_between,
            ref_ratio: policy.ref_ratio,
        }
    }
}

/// Local mapping (keyframe insertion, culling and local BA / VI-BA).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct MappingConfig {
    pub execution: MappingExecution,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MappingExecution {
    /// On the tracking thread, before the frame result is returned.
    Synchronous,
    /// On a background worker.
    #[default]
    Asynchronous,
}

/// Optional loop-closing branches.
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
    /// Place recognition only; no correction path is constructed.
    DetectOnly { vocabulary: PathBuf },
    /// Recognition plus verification, fusion and pose-graph correction.
    DetectAndCorrect { vocabulary: PathBuf },
}

impl LoopClosingMode {
    pub fn vocabulary(&self) -> Option<&Path> {
        match self {
            Self::Disabled => None,
            Self::DetectOnly { vocabulary } | Self::DetectAndCorrect { vocabulary } => {
                Some(vocabulary)
            }
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn vocabulary_mut(&mut self) -> Option<&mut PathBuf> {
        match self {
            Self::Disabled => None,
            Self::DetectOnly { vocabulary } | Self::DetectAndCorrect { vocabulary } => {
                Some(vocabulary)
            }
        }
    }
}
