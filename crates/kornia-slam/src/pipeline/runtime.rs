use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, KeyframeConfig, LoopClosingMode, MappingExecution,
    OrbSlamPipeline, PipelineConfig, PipelineDefinition, SensorSelection,
};
use super::validation::ConfigError;
use crate::loop_closure::LoopClosingConfig;
use crate::loop_closure::place_recognition::{Vocabulary, VocabularyLoadError, load_vocabulary};
use crate::mapping::LocalMappingMode;
use crate::system::SlamConfig;
use crate::tracking::KeyframePolicy;

/// Stereo points closer than this many baselines are back-projected directly
/// at each keyframe (ORB-SLAM3's `ThDepth`, ~35 for EuRoC).
const STEREO_CLOSE_DEPTH_BASELINES: f64 = 35.0;

impl SensorSelection {
    /// The source's rig restricted to the selected sensors.
    pub fn select_rig(&self, mut rig: SensorRig) -> Result<SensorRig, ConfigError> {
        self.validate_rig(&rig)?;
        if self.cameras == CameraSelection::Mono {
            rig.stereo_baseline_m = None;
        }
        if !self.imu {
            rig.imu = None;
        }
        Ok(rig)
    }
}

impl PipelineConfig {
    /// Runtime settings for a rig returned by [`SensorSelection::select_rig`].
    ///
    /// Stereo close depth and the loop-correction IMU gate are derived from
    /// the rig rather than stored in the configuration.
    pub fn slam_config(&self, rig: &SensorRig) -> SlamConfig {
        let orb = self.orb();
        let pgo = matches!(orb.loop_closing, LoopClosingMode::DetectAndCorrect { .. }).then(|| {
            LoopClosingConfig {
                require_imu_initialized: rig.imu.is_some(),
                ..LoopClosingConfig::default()
            }
        });
        SlamConfig {
            keyframe_policy: orb.keyframes.into(),
            local_mapping: orb.mapping.execution.into(),
            stereo_close_depth_m: rig
                .stereo_baseline_m
                .map(|baseline| baseline * STEREO_CLOSE_DEPTH_BASELINES),
            pgo,
            ..SlamConfig::default()
        }
    }

    /// Feature extractor settings for the configured frontend.
    pub fn orb_detector(&self) -> OrbDetector {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        OrbDetector {
            n_keypoints: frontend.n_keypoints,
            ..OrbDetector::default()
        }
    }

    /// Loads the vocabulary when a loop-closing branch is enabled; `None` when disabled.
    pub fn load_vocabulary(&self) -> Result<Option<Vocabulary>, VocabularyLoadError> {
        self.orb()
            .loop_closing
            .vocabulary()
            .map(load_vocabulary)
            .transpose()
    }

    fn orb(&self) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        orb
    }
}

impl From<KeyframeConfig> for KeyframePolicy {
    fn from(config: KeyframeConfig) -> Self {
        Self {
            min_frames_between: config.min_frames_between,
            max_frames_between: config.max_frames_between,
            ref_ratio: config.ref_ratio,
        }
    }
}

impl From<MappingExecution> for LocalMappingMode {
    fn from(execution: MappingExecution) -> Self {
        match execution {
            MappingExecution::Synchronous => Self::Synchronous,
            MappingExecution::Asynchronous => Self::Asynchronous,
        }
    }
}
