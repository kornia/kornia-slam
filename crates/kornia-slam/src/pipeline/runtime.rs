use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, OrbSlamPipeline, PipelineConfig, PipelineDefinition,
    SensorSelection,
};
use super::validation::ConfigError;
use crate::loop_closure::place_recognition::{Vocabulary, VocabularyLoadError, load_vocabulary};
use crate::system::SlamConfig;

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
    /// Stereo close depth is resolved against the rig's baseline.
    pub fn slam_config(&self, rig: &SensorRig) -> SlamConfig {
        let orb = self.orb();
        let FrontendConfig::Orb(frontend) = orb.frontend;
        SlamConfig {
            two_view_init: orb.initialization.clone(),
            map_projection: orb.tracking.map_projection.clone(),
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: orb.tracking.loss_recovery,
            local_mapping: orb.mapping.execution,
            stereo_close_depth_m: rig
                .stereo_baseline_m
                .and_then(|baseline| frontend.stereo_close_depth.metres(baseline)),
            debug: false,
            pgo: orb.loop_closing.correction().cloned(),
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
