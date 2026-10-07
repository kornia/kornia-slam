use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::validation::ConfigError;
use super::{
    CameraSelection, FrontendConfig, OrbSlamPipeline, PipelineConfig, PipelineDefinition,
    SensorSelection,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopClosingConfig;
use crate::loop_closure::place_recognition::{Vocabulary, VocabularyLoadError, load_vocabulary};
use crate::mapping::LocalMappingMode;
use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
use crate::tracking::{KeyframePolicy, TrackingLossRecoveryPolicy};

/// A [`SlamSystem`](crate::SlamSystem) that cannot be built from its
/// configuration and rig.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Vocabulary(#[from] VocabularyLoadError),
}

/// Settings a [`SlamSystem`](crate::SlamSystem) is assembled from, resolved
/// from a pipeline configuration against the selected rig.
pub(crate) struct SystemSettings {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub local_mapping: LocalMappingMode,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points.
    pub stereo_close_depth_m: Option<f64>,
    /// Verified loop closure and live pose-graph correction.
    pub pgo: Option<LoopClosingConfig>,
}

impl SensorSelection {
    /// The source's rig restricted to the selected sensors.
    pub(crate) fn select_rig(&self, mut rig: SensorRig) -> Result<SensorRig, ConfigError> {
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
    /// System settings for a rig returned by [`SensorSelection::select_rig`].
    ///
    /// Stereo close depth is resolved against the rig's baseline.
    pub(crate) fn settings(&self, rig: &SensorRig) -> SystemSettings {
        let orb = self.orb();
        let FrontendConfig::Orb(frontend) = orb.frontend;
        let tuning = &orb.tuning;
        SystemSettings {
            two_view_init: tuning.initialization.clone(),
            map_projection: tuning.map_projection.clone(),
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: tuning.loss_recovery,
            local_mapping: orb.mapping.execution,
            stereo_close_depth_m: rig
                .stereo_baseline_m
                .and_then(|baseline| frontend.stereo_close_depth.metres(baseline)),
            pgo: orb
                .loop_closing
                .corrects()
                .then(|| tuning.loop_correction.clone()),
        }
    }

    /// Feature extractor settings for the configured frontend.
    pub(crate) fn orb_detector(&self) -> OrbDetector {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        OrbDetector {
            n_keypoints: frontend.n_keypoints,
            ..OrbDetector::default()
        }
    }

    /// Loads the vocabulary when a loop-closing branch is enabled; `None` when disabled.
    pub(crate) fn load_vocabulary(&self) -> Result<Option<Vocabulary>, VocabularyLoadError> {
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
