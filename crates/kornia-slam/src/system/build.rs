use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FlowPose, ForwardBackward, FrontendConfig, GyroGuess, KltRetry, KltRig,
    OrbFrontendConfig, OrbSlamPipeline, PipelineDefinition, SensorSelection, SystemConfig,
    TrackRefill,
};
use super::validation::ConfigError;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopCloser;
use crate::loop_closure::place_recognition::{VocabularyLoadError, load_vocabulary};
use crate::mapping::LocalMappingMode;
use crate::tracking::flow_pool::FlowPoolError;
use crate::tracking::optical_flow::LkKernel;
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
    #[error(transparent)]
    FlowPool(#[from] FlowPoolError),
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
    /// Survivors with a map point a frame needs to be tracked by optical flow
    /// alone; `None` without `klt_first`, on a multi-camera rig unless flow
    /// tracks every camera (a primary-only flow frame would drop the other
    /// cameras), and on a raw fisheye primary, whose flow pixels are raw
    /// while its frame's keypoints are virtual-pinhole ones.
    pub klt_min_tracks: Option<usize>,
    pub klt_kernel: LkKernel,
    /// `(max_level, max_iter)` of the predicted flow search; `None` when
    /// disabled or on a raw fisheye primary, whose raw pixels the pinhole
    /// prediction cannot place.
    pub klt_prediction: Option<(usize, usize)>,
    pub flow: FlowSettings,
}

/// The optical-flow knobs the tracking path reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlowSettings {
    pub threads: usize,
    pub rig: KltRig,
    pub window: usize,
    pub retry: KltRetry,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    pub adopt_landmarks: bool,
    pub gyro_guess: GyroGuess,
    pub pose: FlowPose,
}

impl FlowSettings {
    fn new(frontend: &OrbFrontendConfig) -> Self {
        Self {
            threads: frontend.flow_threads,
            rig: frontend.klt_rig,
            window: frontend.klt_window,
            retry: frontend.klt_retry,
            forward_backward: frontend.forward_backward,
            refill: frontend.track_refill,
            adopt_landmarks: frontend.adopt_landmarks,
            gyro_guess: frontend.gyro_guess,
            pose: frontend.flow_pose,
        }
    }
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
            klt_min_tracks: frontend
                .klt_first
                .min_tracks()
                .filter(|_| {
                    (!multi_camera || frontend.klt_rig == KltRig::AllCameras)
                        && !rig.primary().is_fisheye()
                }),
            klt_kernel: frontend.klt_kernel,
            klt_prediction: frontend
                .klt_prediction
                .search()
                .filter(|_| !rig.primary().is_fisheye()),
            flow: FlowSettings::new(&frontend),
        }
    }

    /// Checks `frontend.flow_threads` against a rig returned by
    /// [`SensorSelection::select_rig`].
    pub(crate) fn validate_flow_threads(&self, rig: &SensorRig) -> Result<(), ConfigError> {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        let cameras = match self.sensors.cameras {
            CameraSelection::Rig => rig.cameras().len(),
            CameraSelection::Mono | CameraSelection::Stereo => 1,
        };
        frontend.validate_flow_threads(cameras)
    }

    /// Rejects `frontend.klt_first` with `klt_rig: AllCameras` on raw
    /// fisheye rig cameras: flowing them reads raw pixels as pinhole pixels.
    /// On the primary alone `klt_first` stays inert on a raw primary.
    pub(crate) fn validate_raw_fisheye(&self, rig: &SensorRig) -> Result<(), ConfigError> {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        let raw = rig.cameras().iter().any(|camera| camera.is_fisheye());
        let flowed = frontend.klt_rig == KltRig::AllCameras;
        if raw && flowed && frontend.klt_first.min_tracks().is_some() {
            return Err(ConfigError::FrontendKnobNeeds {
                knob: "klt_first",
                needs: "pinhole rig images",
            });
        }
        Ok(())
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
