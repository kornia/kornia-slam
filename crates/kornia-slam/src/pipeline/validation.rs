use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, KeyframeConfig, LoopClosingMode, OrbFrontendConfig,
    OrbSlamPipeline, PIPELINE_CONFIG_VERSION, PipelineConfig, PipelineDefinition, SensorSelection,
};

/// A pipeline configuration that cannot be built or does not fit its source.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    #[error("unsupported pipeline config version {found}; this build reads version {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("ORB n_keypoints {value} is outside the supported range {min}..={max}")]
    KeypointsOutOfRange {
        value: usize,
        min: usize,
        max: usize,
    },
    #[error(
        "keyframe gaps must be positive with min_frames_between ({min}) <= max_frames_between ({max})"
    )]
    InvalidKeyframeGaps { min: usize, max: usize },
    #[error("keyframe ref_ratio {0} must be finite and in [0, 1]")]
    InvalidRefRatio(f64),
    #[error("loop closing vocabulary path is empty")]
    EmptyVocabularyPath,
    #[error("DetectAndCorrect needs metric input: enable stereo cameras or the IMU")]
    CorrectionWithoutMetricScale,
    #[error("configuration requests {0}, but the source does not provide it")]
    MissingSensor(&'static str),
}

impl PipelineConfig {
    /// Checks the definition on its own, before any source or resource is opened.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != PIPELINE_CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: self.version,
                supported: PIPELINE_CONFIG_VERSION,
            });
        }
        match &self.pipeline {
            PipelineDefinition::OrbSlam(orb) => orb.validate(&self.sensors),
        }
    }
}

impl OrbSlamPipeline {
    fn validate(&self, sensors: &SensorSelection) -> Result<(), ConfigError> {
        match &self.frontend {
            FrontendConfig::Orb(orb) => orb.validate()?,
        }
        self.keyframes.validate()?;
        validate_loop_closing(&self.loop_closing, sensors)
    }
}

impl OrbFrontendConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        let range = Self::N_KEYPOINTS_RANGE;
        if range.contains(&self.n_keypoints) {
            Ok(())
        } else {
            Err(ConfigError::KeypointsOutOfRange {
                value: self.n_keypoints,
                min: *range.start(),
                max: *range.end(),
            })
        }
    }
}

impl KeyframeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        let (min, max) = (self.min_frames_between, self.max_frames_between);
        if min == 0 || min > max {
            return Err(ConfigError::InvalidKeyframeGaps { min, max });
        }
        if !(0.0..=1.0).contains(&self.ref_ratio) {
            return Err(ConfigError::InvalidRefRatio(self.ref_ratio));
        }
        Ok(())
    }
}

fn validate_loop_closing(
    mode: &LoopClosingMode,
    sensors: &SensorSelection,
) -> Result<(), ConfigError> {
    if mode
        .vocabulary()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(ConfigError::EmptyVocabularyPath);
    }
    // Correcting a visual-only monocular map would apply scale-ambiguous
    // corrections; matches the app's previous `--apply-pgo` rule.
    let metric = sensors.cameras == CameraSelection::Stereo || sensors.imu;
    if matches!(mode, LoopClosingMode::DetectAndCorrect { .. }) && !metric {
        return Err(ConfigError::CorrectionWithoutMetricScale);
    }
    Ok(())
}

impl SensorSelection {
    /// Checks that the source's rig provides every selected sensor.
    pub fn validate_rig(&self, rig: &SensorRig) -> Result<(), ConfigError> {
        if self.cameras == CameraSelection::Stereo && rig.stereo_baseline_m.is_none() {
            return Err(ConfigError::MissingSensor("stereo cameras"));
        }
        if self.imu && rig.imu.is_none() {
            return Err(ConfigError::MissingSensor("an IMU"));
        }
        Ok(())
    }
}
