use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, LocalBaConfig, LoopClosingExecution, LoopClosingMode,
    OrbSlamPipeline, PipelineDefinition, SensorSelection, StereoDepthSigma, SystemConfig,
};
use super::validation::ConfigError;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::place_recognition::{VocabularyLoadError, load_vocabulary};
use crate::loop_closure::{LoopCloser, LoopSolver};
use crate::mapping::LocalMappingMode;
use crate::mapping::bundle_adjustment::{DepthSigma, LocalBaParams};
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
    #[error("could not start the loop-closing worker: {0}")]
    LoopWorker(#[from] std::io::Error),
}

/// Settings a [`SlamSystem`](crate::SlamSystem) is assembled from, resolved
/// from a system configuration against the selected rig.
pub(crate) struct SystemSettings {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub local_mapping: LocalMappingMode,
    pub local_ba: LocalBaParams,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points.
    pub stereo_close_depth_m: Option<f64>,
}

impl SensorSelection {
    /// The source's rig restricted to the selected sensors.
    pub(crate) fn select_rig(&self, mut rig: SensorRig) -> Result<SensorRig, ConfigError> {
        self.validate_rig(&rig)?;
        if self.cameras != CameraSelection::Stereo {
            rig.stereo_baseline_m = None;
        }
        if self.cameras != CameraSelection::Rig {
            rig.cameras.clear();
        }
        if !self.imu {
            rig.imu = None;
        }
        Ok(rig)
    }
}

/// Distance between the primary rig camera and camera 1, the baseline of
/// their overlap triangulation; `None` outside a multi-camera rig.
fn rig_baseline_m(rig: &SensorRig) -> Option<f64> {
    rig.is_multi_camera()
        .then(|| rig.cameras.get(1))
        .flatten()
        .map(|camera| camera.primary_from_camera.translation.length())
}

fn stereo_baseline_m(rig: &SensorRig) -> Option<f64> {
    rig.stereo_baseline_m.or_else(|| rig_baseline_m(rig))
}

impl LocalBaConfig {
    /// Solver parameters for a rig whose stereo depth comes from focal length
    /// times baseline `focal_baseline` (pixel·metres).
    pub(crate) fn params(&self, focal_baseline: Option<f64>) -> LocalBaParams {
        let depth_sigma = match self.stereo_depth_sigma {
            StereoDepthSigma::RelativeDepth(fraction) => DepthSigma::Relative(fraction as f32),
            // Without a baseline there is no stereo depth to weight.
            StereoDepthSigma::Disparity { px } => DepthSigma::Disparity {
                px_over_focal_baseline: focal_baseline.map_or(0.0, |fb| (px / fb) as f32),
            },
        };
        LocalBaParams {
            active_keyframes: self.active_keyframes,
            boundary_imu_info_scale: self.boundary_imu_info_scale,
            boundary_scales_bias_rows: self.boundary_scales_bias_rows,
            imu_huber: self.imu_huber,
            depth_sigma,
        }
    }
}

impl SystemConfig {
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
            local_ba: orb
                .mapping
                .local_ba
                .params(stereo_baseline_m(rig).map(|baseline| rig.camera.fx * baseline)),
            stereo_close_depth_m: stereo_baseline_m(rig)
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

    /// The loop closer, with its vocabulary loaded and its solver started, when
    /// loop closing is enabled. With the IMU selected, correction waits for
    /// inertial initialization.
    pub(crate) fn loop_closer(&self, rig: &SensorRig) -> Result<Option<LoopCloser>, BuildError> {
        let orb = self.orb();
        let LoopClosingMode::Enabled {
            vocabulary,
            execution,
        } = &orb.loop_closing
        else {
            return Ok(None);
        };
        let solver = match execution.resolve(orb.mapping.execution) {
            LoopClosingExecution::Deferred { frames } => LoopSolver::deferred(frames),
            LoopClosingExecution::Asynchronous => LoopSolver::asynchronous()?,
            LoopClosingExecution::Synchronous | LoopClosingExecution::FollowMapping => {
                LoopSolver::Synchronous
            }
        };
        let closer = LoopCloser::new(
            load_vocabulary(vocabulary)?,
            orb.tuning.loop_correction.clone(),
            rig.imu.is_some(),
        );
        Ok(Some(closer.with_solver(solver)))
    }

    fn orb(&self) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        orb
    }
}
