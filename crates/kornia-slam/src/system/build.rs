use std::path::Path;

use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, OrbFrontendConfig, OrbSlamPipeline, PipelineDefinition,
    SensorSelection, SystemConfig,
};
use super::validation::ConfigError;
use crate::features::{Clidd, Descriptor, RBrief};
use crate::frontend::FeatureFileError;
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
    #[error(transparent)]
    FeatureFile(#[from] FeatureFileError),
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

impl Default for MatchingSettings<Clidd> {
    fn default() -> Self {
        Self {
            two_view_init: TwoViewInitConfig::default(),
            map_projection: MapProjectionConfig::default(),
            relocalization: RelocalizationConfig::default(),
        }
    }
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
        let stereo_close_depth = match &orb.frontend {
            FrontendConfig::Orb(frontend) => Some(frontend.stereo_close_depth),
            FrontendConfig::Precomputed(_) => None,
        };
        SystemSettings {
            matching,
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: orb.tuning.loss_recovery,
            local_mapping: orb.mapping.execution,
            stereo_close_depth_m: rig
                .stereo_baseline_m
                .zip(stereo_close_depth)
                .and_then(|(baseline, close)| close.metres(baseline)),
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

    /// Feature extractor settings for the configured ORB frontend.
    pub(crate) fn orb_detector(&self) -> Result<OrbDetector, ConfigError> {
        let FrontendConfig::Orb(OrbFrontendConfig { n_keypoints, .. }) = self.orb().frontend else {
            return Err(ConfigError::FrontendMismatch {
                configured: "precomputed",
                system: "ORB",
            });
        };
        Ok(OrbDetector {
            n_keypoints,
            ..OrbDetector::default()
        })
    }

    /// The feature file of the configured precomputed frontend.
    pub(crate) fn precomputed_features(&self) -> Result<&Path, ConfigError> {
        match &self.orb().frontend {
            FrontendConfig::Precomputed(precomputed) => Ok(&precomputed.path),
            FrontendConfig::Orb(_) => Err(ConfigError::FrontendMismatch {
                configured: "ORB",
                system: "precomputed CLIDD",
            }),
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
