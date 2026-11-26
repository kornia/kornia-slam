use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, OrbSlamPipeline, PipelineDefinition, SensorSelection,
    SystemConfig,
};
use super::validation::ConfigError;
use crate::features::{Descriptor, RBrief};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopCloser;
use crate::loop_closure::place_recognition::{VocabularyLoadError, load_vocabulary};
use crate::mapping::LocalMappingMode;
use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
use crate::tracking::relocalization::RelocalizationConfig;
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
/// from a system configuration against the selected rig.
pub(crate) struct SystemSettings<D: Descriptor> {
    pub matching: MatchingSettings<D>,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub local_mapping: LocalMappingMode,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points.
    pub stereo_close_depth_m: Option<f64>,
}

/// The descriptor-dependent settings of bootstrap, tracking and relocalization.
pub(crate) struct MatchingSettings<D: Descriptor> {
    pub two_view_init: TwoViewInitConfig<D>,
    pub map_projection: MapProjectionConfig<D>,
    pub relocalization: RelocalizationConfig<D>,
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

impl SystemConfig {
    /// System settings for a rig returned by [`SensorSelection::select_rig`].
    ///
    /// Stereo close depth is resolved against the rig's baseline.
    pub(crate) fn settings<D: Descriptor>(
        &self,
        rig: &SensorRig,
        matching: MatchingSettings<D>,
    ) -> SystemSettings<D> {
        let orb = self.orb();
        let FrontendConfig::Orb(frontend) = orb.frontend;
        SystemSettings {
            matching,
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: orb.tuning.loss_recovery,
            local_mapping: orb.mapping.execution,
            stereo_close_depth_m: rig
                .stereo_baseline_m
                .and_then(|baseline| frontend.stereo_close_depth.metres(baseline)),
        }
    }

    /// The ORB pipeline's matching settings, from its tuning.
    pub(crate) fn orb_matching(&self) -> MatchingSettings<RBrief> {
        let tuning = &self.orb().tuning;
        MatchingSettings {
            two_view_init: tuning.initialization.clone(),
            map_projection: tuning.map_projection.clone(),
            relocalization: RelocalizationConfig::default(),
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

    /// The loop closer, with its vocabulary loaded, when loop closing is
    /// enabled. With the IMU selected, correction waits for inertial
    /// initialization.
    pub(crate) fn loop_closer(
        &self,
        rig: &SensorRig,
    ) -> Result<Option<LoopCloser>, VocabularyLoadError> {
        let orb = self.orb();
        orb.loop_closing
            .vocabulary()
            .map(|path| {
                Ok(LoopCloser::new(
                    load_vocabulary(path)?,
                    orb.tuning.loop_correction.clone(),
                    rig.imu.is_some(),
                ))
            })
            .transpose()
    }

    fn orb(&self) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        orb
    }
}
