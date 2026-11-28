use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, OrbSlamPipeline, PipelineDefinition, SensorSelection,
    SystemConfig,
};
use super::validation::ConfigError;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopCloser;
use crate::loop_closure::place_recognition::{VocabularyLoadError, load_vocabulary};
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
/// from a system configuration against the selected rig.
pub(crate) struct SystemSettings {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub local_mapping: LocalMappingMode,
    /// Metric baseline of the rectified pair under [`CameraSelection::Stereo`];
    /// `None` for every other selection, whatever the rig could offer.
    pub stereo_baseline_m: Option<f64>,
    /// Every camera of the rig is tracked; `true` only under
    /// [`CameraSelection::Rig`].
    pub multi_camera: bool,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points.
    pub stereo_close_depth_m: Option<f64>,
}

impl SensorSelection {
    /// The source's rig restricted to the selected sensors.
    pub(crate) fn select_rig(&self, mut rig: SensorRig) -> Result<SensorRig, ConfigError> {
        self.validate_rig(&rig)?;
        if self.cameras == CameraSelection::Mono {
            rig = rig.truncated_to_primary();
        }
        if !self.imu {
            rig.imu = None;
        }
        Ok(rig)
    }
}

/// Distance between the primary rig camera and camera 1, the baseline of
/// their overlap triangulation.
fn rig_baseline_m(rig: &SensorRig) -> Option<f64> {
    rig.cameras()
        .get(1)
        .map(|camera| camera.primary_from_camera.translation.length())
}

impl SystemConfig {
    /// System settings for a rig returned by [`SensorSelection::select_rig`].
    ///
    /// Stereo close depth is resolved against the rig's baseline.
    pub(crate) fn settings(&self, rig: &SensorRig) -> SystemSettings {
        let orb = self.orb();
        let stereo_baseline_m = match self.sensors.cameras {
            CameraSelection::Stereo => rig.rectified_stereo_baseline_m(),
            CameraSelection::Mono | CameraSelection::Rig => None,
        };
        let multi_camera = self.sensors.cameras == CameraSelection::Rig;
        let close_depth_baseline_m = if multi_camera {
            rig_baseline_m(rig)
        } else {
            stereo_baseline_m
        };
        let FrontendConfig::Orb(frontend) = orb.frontend;
        let tuning = &orb.tuning;
        SystemSettings {
            two_view_init: tuning.initialization.clone(),
            map_projection: tuning.map_projection.clone(),
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: tuning.loss_recovery,
            local_mapping: orb.mapping.execution,
            stereo_baseline_m,
            multi_camera,
            stereo_close_depth_m: close_depth_baseline_m
                .and_then(|baseline| frontend.stereo_close_depth.metres(baseline)),
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
