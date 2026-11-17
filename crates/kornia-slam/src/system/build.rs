use kornia_imgproc::features::OrbDetector;
use kornia_sensors::SensorRig;

use super::config::{
    AnchorGuard, CameraSelection, FlowBookkeeping, FlowPose, ForwardBackward, FrontendConfig,
    GyroGuess, InertialConfig, KeyframeOrb, KltRetry, KltRig, LocalBaConfig, LoopClosingExecution,
    LoopClosingMode, LostRelocalization, OrbFrontendConfig, OrbSlamPipeline, PipelineDefinition,
    SensorSelection, StereoDepthSigma, StereoPyramid, SystemConfig, TrackRefill, TrackingConfig,
};
use super::recovery::RecoveryConfig;
use super::validation::ConfigError;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::place_recognition::{VocabularyLoadError, load_vocabulary};
use crate::loop_closure::{LoopCloser, LoopClosingConfig, LoopSolver};
use crate::mapping::bundle_adjustment::{DepthSigma, LocalBaParams};
use crate::mapping::{GrowthNeighbours, LocalMappingMode};
use crate::tracking::flow_pool::FlowPoolError;
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
use crate::tracking::pose_gate::PoseGate;
use crate::tracking::{KeyframePolicy, KeyframesWhileLost, TrackingLossRecoveryPolicy};

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
    pub keyframes_while_lost: KeyframesWhileLost,
    pub recovery: RecoveryConfig,
    pub pose_gate: PoseGate,
    pub local_mapping: LocalMappingMode,
    pub local_ba: LocalBaParams,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points.
    pub stereo_close_depth_m: Option<f64>,
    /// Flow survivors with a map point needed to track without ORB; `None`
    /// extracts ORB on every frame.
    pub klt_min_tracks: Option<usize>,
    /// `(max_level, max_iter)` of the optical-flow search from a trusted
    /// prediction; `None` keeps the full search.
    pub klt_prediction: Option<(usize, usize)>,
    pub klt_kernel: LkKernel,
    pub frontend_flow: FlowSettings,
    pub tracking: TrackingConfig,
    pub inertial: InertialConfig,
    pub relocalization: LostRelocalization,
    pub growth_neighbours: GrowthNeighbours,
}

/// The 1 ms frontend knobs the tracking path reads.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // skeleton: lane T reads the rest
pub(crate) struct FlowSettings {
    pub threads: usize,
    pub first_core: Option<usize>,
    pub rig: KltRig,
    pub window: usize,
    pub retry: KltRetry,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    pub adopt_landmarks: bool,
    pub gyro_guess: GyroGuess,
    pub anchor_guard: AnchorGuard,
    pub pose: FlowPose,
    pub bookkeeping: FlowBookkeeping,
    pub keyframe_orb: KeyframeOrb,
}

impl FlowSettings {
    fn new(frontend: &OrbFrontendConfig) -> Self {
        Self {
            threads: frontend.flow_threads,
            first_core: frontend.flow_first_core,
            rig: frontend.klt_rig,
            window: frontend.klt_window,
            retry: frontend.klt_retry,
            forward_backward: frontend.forward_backward,
            refill: frontend.track_refill,
            adopt_landmarks: frontend.adopt_landmarks,
            gyro_guess: frontend.gyro_guess,
            anchor_guard: frontend.anchor_guard,
            pose: frontend.flow_pose,
            bookkeeping: frontend.flow_bookkeeping,
            keyframe_orb: frontend.keyframe_orb,
        }
    }

    /// The first knob set away from today's behaviour that no tracking path
    /// reads yet. Knobs that need `WindowedU8` stop inside its flow path.
    pub(crate) fn unbuilt_knob(&self) -> Option<&'static str> {
        let today = Self::new(&OrbFrontendConfig::default());
        [
            ("klt_window", self.window != today.window),
            ("adopt_landmarks", self.adopt_landmarks),
            ("gyro_guess", self.gyro_guess != today.gyro_guess),
            ("flow_pose", self.pose != today.pose),
            ("flow_bookkeeping", self.bookkeeping != today.bookkeeping),
            ("keyframe_orb", self.keyframe_orb != today.keyframe_orb),
        ]
        .into_iter()
        .find_map(|(knob, set)| set.then_some(knob))
    }
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
            max_iterations: self.max_iterations,
            boundary_scales_bias_rows: self.boundary_scales_bias_rows,
            boundary: self.boundary,
            imu_huber: self.imu_huber,
            depth_sigma,
            visual_information: self.visual_information,
            state_envelope: self.state_envelope,
            lost_keyframes: self.lost_keyframes,
            visual_kernel: self.visual_kernel,
            large_window: self.large_window,
            online_calibration: self.online_calibration,
            initial_lambda: LocalBaParams::default().initial_lambda,
            recently_initialized: false,
            newest_marginal: false,
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
            map_projection: MapProjectionConfig {
                keyframe_observations: orb.mapping.keyframe_observations,
                ..tuning
                    .map_projection
                    .with_visual_information(orb.mapping.local_ba.visual_information)
            },
            keyframe_policy: orb.keyframes,
            tracking_loss_recovery: tuning.loss_recovery,
            keyframes_while_lost: orb.tracking_loss.keyframes_while_lost,
            recovery: orb.tracking_loss.reseed,
            pose_gate: orb.tracking_loss.pose_gate,
            local_mapping: orb.mapping.execution,
            local_ba: orb
                .mapping
                .local_ba
                .params(stereo_baseline_m(rig).map(|baseline| rig.camera.fx * baseline)),
            stereo_close_depth_m: stereo_baseline_m(rig)
                .and_then(|baseline| frontend.stereo_close_depth.metres(baseline)),
            // A primary-only flow frame drops the other cameras, whose
            // correspondences are what carry a rig through hard segments.
            klt_min_tracks: frontend
                .klt_first
                .min_tracks()
                .filter(|_| frontend.klt_rig == KltRig::AllCameras || !rig.is_multi_camera()),
            klt_kernel: frontend.klt_kernel,
            frontend_flow: FlowSettings::new(&frontend),
            klt_prediction: frontend
                .klt_prediction
                .search()
                .filter(|_| rig.fisheye.is_none()),
            tracking: orb.tracking,
            inertial: orb.inertial,
            relocalization: orb.tracking_loss.relocalization,
            growth_neighbours: orb.mapping.growth_neighbours,
        }
    }

    /// Keypoints for every image but the primary.
    pub(crate) fn secondary_n_keypoints(&self) -> usize {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        frontend
            .secondary_n_keypoints
            .unwrap_or(frontend.n_keypoints)
    }

    /// The pyramid the stereo and overlap matching read.
    pub(crate) fn stereo_pyramid(&self) -> StereoPyramid {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        frontend.stereo_pyramid
    }

    /// Checks `frontend.flow_threads` against a rig returned by
    /// [`SensorSelection::select_rig`].
    pub(crate) fn validate_flow_threads(&self, rig: &SensorRig) -> Result<(), ConfigError> {
        let FrontendConfig::Orb(frontend) = self.orb().frontend;
        frontend.validate_flow_threads(rig.cameras.len())
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
            graph,
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
        let config = LoopClosingConfig {
            graph: *graph,
            ..orb.tuning.loop_correction.clone()
        };
        let closer = LoopCloser::new(load_vocabulary(vocabulary)?, config, rig.imu.is_some());
        Ok(Some(closer.with_solver(solver)))
    }

    fn orb(&self) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        orb
    }
}
