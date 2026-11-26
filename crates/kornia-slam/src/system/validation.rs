use kornia_imgproc::features::OrbMatchConfig;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FanOut, FlowPose, ForwardBackward, FrameVelocity, FrontendConfig, KeyframeOrb,
    KltFirst, KltRetry, KltRig, LocalBaConfig, LoopClosingMode, MAX_FLOW_THREADS,
    OnlineCalibrationConfig, OrbFrontendConfig, OrbSlamPipeline, OrbTuning, PipelineDefinition,
    PoseInertial, PoseInertialMinInliers, SYSTEM_CONFIG_VERSION, SensorSelection, SnapbackFix,
    StereoCloseDepth, StereoDepthSigma, SystemConfig, TrackRefill, TrackingConfig,
    TrackingLossConfig,
};
use super::recovery::{RecoveryConfig, SeedPlacement};
use crate::inertial::MAX_BUFFERED_SEC;
use crate::inertial::trust::ImuTrustMode;
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{
    LoopClosingConfig, LoopGraphInformation, LoopOdometryGate, LoopTemporalConsistency,
    OdometryRandomWalk,
};
use crate::mapping::GrowthNeighbours;
use crate::mapping::bundle_adjustment::LargeWindow;
use crate::mapping::global_vi_ba::OfflineBaConfig;
use crate::mapping::marginalization::BoundaryPrior;
use crate::mapping::outlier_erase::OutlierErase;
use crate::odometry::{Odometry, OdometryImuNoise, OdometryMode};
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::pose_estimation::map_projection::ProjectionMatchConfig;
use crate::tracking::pose_gate::PoseGate;
use crate::tracking::relocalization::RelocalizationSolver;
use crate::tracking::{KeyframePolicy, KeyframeRules};

/// A system configuration that cannot be built or does not fit its source.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    #[error("unsupported system config version {found}; this build reads version {supported}")]
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
    #[error("{setting} is {value}, but must be {requirement}")]
    InvalidSetting {
        setting: String,
        value: f64,
        requirement: &'static str,
    },
    #[error(
        "an inertial system needs a keyframe bound: keyframes.max_frames_between is unbounded, \
         so set keyframes.max_interval_sec (ORB-SLAM3 forces one every 0.5 s)"
    )]
    UnboundedInertialKeyframes,
    #[error("loop closing vocabulary path is empty")]
    EmptyVocabularyPath,
    #[error("loop closing needs metric input: enable stereo cameras or the IMU")]
    LoopClosingWithoutMetricScale,
    #[error("configuration requests {0}, but the source does not provide it")]
    MissingSensor(&'static str),
    #[error("stereo cameras need rectified images; the source supplies raw fisheye images")]
    FisheyeStereo,
    #[error("raw fisheye rig cameras need the `Rig` camera selection")]
    RawFisheyeOutsideRig,
    #[error(
        "mapping.local_ba.large_window active_keyframes ({large}) is below \
         mapping.local_ba.active_keyframes ({active})"
    )]
    LargeWindowNotLarger { active: usize, large: usize },
    #[error("{0} needs the IMU: select sensors.imu")]
    InertialKnobWithoutImu(&'static str),
    #[error(
        "tracking_loss.reseed.seed Held places a re-seed that keeps the inertial state: \
         set tracking_loss.reseed.keep_inertial"
    )]
    HeldSeedWithoutKeptInertialState,
    #[error(
        "tracking.snapback_fix AnchorNextFrame is part of tracking.pose_inertial: set one of them"
    )]
    SnapbackFixWithPoseInertial,
    #[error(
        "tracking.frame_velocity PropagatedFromKeyframe has no effect with tracking.pose_inertial: \
         the pose-inertial solve carries its own velocity; set one of them"
    )]
    FrameVelocityWithPoseInertial,
    #[error(
        "tracking.pose_inertial_min_inliers WithPrediction has no effect with tracking.pose_inertial \
         Off: it is the bar after the pose-inertial solve"
    )]
    PoseInertialMinInliersWithoutSolve,
    #[error(
        "tracking.pose_inertial needs descriptors on every frame it can reject, but with \
         frontend.klt_first, klt_kernel WindowedU8 and flow_pose AnalyticGn a flow frame has \
         none: set flow_pose RansacPnp or pose_inertial Off"
    )]
    PoseInertialWithLockFreeFlow,
    #[error("`{0}` needs loop closing's vocabulary: enable `pipeline.loop_closing`")]
    RelocalizationWithoutVocabulary(&'static str),
    #[error("odometry SlidingWindow needs {needs}")]
    OdometryNeeds { needs: &'static str },
    #[error("frontend.{knob} needs {needs}")]
    FrontendKnobNeeds {
        knob: &'static str,
        needs: &'static str,
    },
    #[error(
        "frontend.flow_threads {threads} has no shard layout for {cameras} cameras: \
         use 1, the camera count or one more"
    )]
    FlowThreadsForRig { threads: usize, cameras: usize },
}

impl SystemConfig {
    /// Checks the definition on its own, before any source or resource is opened.
    pub fn validate(&self) -> Result<(), ConfigError> {
        Self::check_version(self.version)?;
        match &self.pipeline {
            PipelineDefinition::OrbSlam(orb) => orb.validate(&self.sensors),
        }
    }

    /// Checks a schema version on its own, so a file written for another
    /// version can be reported as such before its fields are parsed.
    pub fn check_version(version: u32) -> Result<(), ConfigError> {
        if version == SYSTEM_CONFIG_VERSION {
            Ok(())
        } else {
            Err(ConfigError::UnsupportedVersion {
                found: version,
                supported: SYSTEM_CONFIG_VERSION,
            })
        }
    }
}

impl OrbSlamPipeline {
    fn validate(&self, sensors: &SensorSelection) -> Result<(), ConfigError> {
        match &self.frontend {
            FrontendConfig::Orb(orb) => {
                orb.validate()?;
                validate_flow(orb, sensors)?;
            }
        }
        validate_keyframes(&self.keyframes, sensors.imu)?;
        validate_local_ba(&self.mapping.local_ba)?;
        if self.mapping.local_ba.online_calibration.is_enabled() && !sensors.imu {
            return Err(ConfigError::InertialKnobWithoutImu(
                "mapping.local_ba.online_calibration",
            ));
        }
        if self.mapping.local_ba.outlier_erase.is_enabled() && !sensors.imu {
            return Err(ConfigError::InertialKnobWithoutImu(
                "mapping.local_ba.outlier_erase",
            ));
        }
        validate_loop_closing(&self.loop_closing, sensors)?;
        validate_recovery(&self.tracking_loss.reseed, sensors.imu)?;
        validate_pose_gate(&self.tracking_loss.pose_gate, sensors.imu)?;
        validate_lost_relocalization(&self.tracking_loss, &self.loop_closing)?;
        validate_estimator(&self.tracking, sensors.imu)?;
        match &self.frontend {
            FrontendConfig::Orb(orb) => validate_lock_free_flow(orb, &self.tracking)?,
        }
        check(
            "inertial.viba_velocity_gate",
            self.inertial.viba_velocity_gate,
            NON_NEGATIVE,
        )?;
        check(
            "mapping.growth_min_baseline_ratio",
            self.mapping.growth_min_baseline_ratio,
            NON_NEGATIVE,
        )?;
        if let GrowthNeighbours::OrbSlam3 { covisible } = self.mapping.growth_neighbours {
            check(
                "mapping.growth_neighbours.covisible",
                covisible as f64,
                AT_LEAST_ONE,
            )?;
        }
        if let FanOut::Bounded(bounded) = self.fan_out {
            check(
                "fan_out.tracking_threads",
                bounded.tracking_threads as f64,
                AT_LEAST_ONE,
            )?;
        }
        validate_odometry(&self.odometry, sensors)?;
        validate_tuning(&self.tuning, self.loop_closing.is_enabled())
    }
}

/// A kept inertial state needs the IMU (G12).
fn validate_recovery(recovery: &RecoveryConfig, imu: bool) -> Result<(), ConfigError> {
    let held = recovery.seed == SeedPlacement::Held;
    if held && !imu {
        return Err(ConfigError::InertialKnobWithoutImu(
            "tracking_loss.reseed.seed",
        ));
    }
    if held && !recovery.keep_inertial {
        return Err(ConfigError::HeldSeedWithoutKeptInertialState);
    }
    if recovery.keep_inertial && !imu {
        return Err(ConfigError::InertialKnobWithoutImu(
            "tracking_loss.reseed.keep_inertial",
        ));
    }
    let at = |field| format!("tracking_loss.reseed.{field}");
    check(
        at("max_coast_speed_mps"),
        recovery.max_coast_speed_mps,
        POSITIVE,
    )?;
    check(
        at("runaway_window_sec"),
        recovery.runaway_window_sec,
        POSITIVE,
    )?;
    check(
        at("bridge_covariance_scale"),
        recovery.bridge_covariance_scale,
        AT_LEAST_ONE,
    )?;
    check(
        at("max_bridge_sec"),
        recovery.max_bridge_sec,
        WITHIN_IMU_BUFFER,
    )?;
    check(
        at("min_seed_points"),
        recovery.min_seed_points as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("min_cold_start_points"),
        recovery.min_cold_start_points as f64,
        AT_LEAST_ONE,
    )?;
    check(at("max_tilt_deg"), recovery.max_tilt_deg, POSITIVE)
}

/// Each relocalization knob is inert without the loop closer's vocabulary.
fn validate_lost_relocalization(
    tracking_loss: &TrackingLossConfig,
    loop_closing: &LoopClosingMode,
) -> Result<(), ConfigError> {
    let relocalization = &tracking_loss.relocalization;
    check(
        "tracking_loss.relocalization.max_jump_m",
        relocalization.max_jump_m,
        POSITIVE,
    )?;
    if loop_closing.is_enabled() {
        return Ok(());
    }
    let inert = if relocalization.while_lost {
        Some("tracking_loss.relocalization.while_lost")
    } else if relocalization.solver != RelocalizationSolver::PrimaryEpnp {
        Some("tracking_loss.relocalization.solver")
    } else if tracking_loss.reseed.relocalize_when_lost {
        Some("tracking_loss.reseed.relocalize_when_lost")
    } else {
        None
    };
    inert.map_or(Ok(()), |knob| {
        Err(ConfigError::RelocalizationWithoutVocabulary(knob))
    })
}

/// The per-frame estimator knobs that read the IMU need it, and the snap-back
/// fix is already part of the pose-inertial solve.
fn validate_estimator(tracking: &TrackingConfig, imu: bool) -> Result<(), ConfigError> {
    if let ImuTrustMode::Covariance { search_sigmas } = tracking.imu_trust {
        check("tracking.imu_trust.search_sigmas", search_sigmas, POSITIVE)?;
    }
    if !imu {
        if tracking.pose_inertial != PoseInertial::Off {
            return Err(ConfigError::InertialKnobWithoutImu(
                "tracking.pose_inertial",
            ));
        }
        if tracking.imu_trust != ImuTrustMode::TimeSinceInit {
            return Err(ConfigError::InertialKnobWithoutImu("tracking.imu_trust"));
        }
        if tracking.snapback_fix != SnapbackFix::Off {
            return Err(ConfigError::InertialKnobWithoutImu("tracking.snapback_fix"));
        }
        if tracking.frame_velocity != FrameVelocity::KeyframeVelocity {
            return Err(ConfigError::InertialKnobWithoutImu(
                "tracking.frame_velocity",
            ));
        }
    }
    if tracking.snapback_fix != SnapbackFix::Off && tracking.pose_inertial != PoseInertial::Off {
        return Err(ConfigError::SnapbackFixWithPoseInertial);
    }
    // The pose-inertial solve carries its own velocity (R-C8).
    if tracking.frame_velocity != FrameVelocity::KeyframeVelocity
        && tracking.pose_inertial != PoseInertial::Off
    {
        return Err(ConfigError::FrameVelocityWithPoseInertial);
    }
    if tracking.pose_inertial_min_inliers != PoseInertialMinInliers::Visual
        && tracking.pose_inertial == PoseInertial::Off
    {
        return Err(ConfigError::PoseInertialMinInliersWithoutSolve);
    }
    Ok(())
}

/// A flow frame built without the map has no descriptors, so it must not
/// reach the lost path, which a late pose-inertial reject sends it to (R-D3).
fn validate_lock_free_flow(
    frontend: &OrbFrontendConfig,
    tracking: &TrackingConfig,
) -> Result<(), ConfigError> {
    let lock_free = frontend.klt_first != KltFirst::Disabled
        && frontend.klt_kernel == LkKernel::WindowedU8
        && matches!(frontend.flow_pose, FlowPose::AnalyticGn { .. });
    if lock_free && tracking.pose_inertial != PoseInertial::Off {
        return Err(ConfigError::PoseInertialWithLockFreeFlow);
    }
    Ok(())
}

/// The innovation gate checks against the propagated IMU covariance (card AC-2b).
fn validate_pose_gate(gate: &PoseGate, imu: bool) -> Result<(), ConfigError> {
    if *gate != PoseGate::Disabled && !imu {
        return Err(ConfigError::InertialKnobWithoutImu(
            "tracking_loss.pose_gate",
        ));
    }
    Ok(())
}

fn validate_tuning(tuning: &OrbTuning, loop_closing: bool) -> Result<(), ConfigError> {
    validate_initialization(&tuning.initialization)?;
    validate_tracking(tuning)?;
    if loop_closing {
        validate_correction(&tuning.loop_correction)?;
    }
    Ok(())
}

impl OrbFrontendConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        let range = Self::N_KEYPOINTS_RANGE;
        if !range.contains(&self.n_keypoints) {
            return Err(ConfigError::KeypointsOutOfRange {
                value: self.n_keypoints,
                min: *range.start(),
                max: *range.end(),
            });
        }
        if let Some((_, max_iter)) = self.klt_prediction.search() {
            check(
                "frontend.klt_prediction.max_iter",
                max_iter as f64,
                POSITIVE,
            )?;
        }
        if let Some(min_tracks) = self.klt_first.min_tracks() {
            check("frontend.klt_first.min_tracks", min_tracks as f64, POSITIVE)?;
        }
        match self.stereo_close_depth {
            StereoCloseDepth::Baselines(value) | StereoCloseDepth::Metres(value) => {
                check("frontend.stereo_close_depth", value, POSITIVE)
            }
            StereoCloseDepth::Disabled => Ok(()),
        }
    }

    /// Checks `flow_threads` against the camera count of the selected rig:
    /// shards exist for 1 thread, one per camera, or one more (plan section 3).
    pub(crate) fn validate_flow_threads(&self, cameras: usize) -> Result<(), ConfigError> {
        let threads = self.flow_threads;
        if threads == 1 || threads == cameras || threads == cameras + 1 {
            Ok(())
        } else {
            Err(ConfigError::FlowThreadsForRig { threads, cameras })
        }
    }
}

/// The 1 ms frontend knobs: ranges, and the knobs each one needs.
fn validate_flow(
    frontend: &OrbFrontendConfig,
    sensors: &SensorSelection,
) -> Result<(), ConfigError> {
    let at = |field| format!("frontend.{field}");
    let needs = |knob: &'static str, requirement: &'static str| -> Result<(), ConfigError> {
        Err(ConfigError::FrontendKnobNeeds {
            knob,
            needs: requirement,
        })
    };
    check(
        at("flow_threads"),
        frontend.flow_threads as f64,
        Requirement("in 1..=MAX_FLOW_THREADS", |v| {
            (1.0..=MAX_FLOW_THREADS as f64).contains(&v)
        }),
    )?;
    check(
        at("klt_window"),
        frontend.klt_window as f64,
        Requirement("odd and >= 3", |v| v >= 3.0 && v % 2.0 == 1.0),
    )?;
    if let KltRetry::CoarseLevel { max_level } = frontend.klt_retry {
        check(at("klt_retry.max_level"), max_level as f64, AT_LEAST_ONE)?;
    }
    if let ForwardBackward::Enabled {
        max_round_trip_px,
        max_iter,
        every_n_frames,
    } = frontend.forward_backward
    {
        check(
            at("forward_backward.max_round_trip_px"),
            max_round_trip_px.into(),
            POSITIVE,
        )?;
        check(
            at("forward_backward.max_iter"),
            max_iter as f64,
            AT_LEAST_ONE,
        )?;
        check(
            at("forward_backward.every_n_frames"),
            every_n_frames as f64,
            AT_LEAST_ONE,
        )?;
    }
    if let TrackRefill::Enabled {
        per_camera,
        min_spacing_px,
        refill_below,
        ..
    } = frontend.track_refill
    {
        check(
            at("track_refill.per_camera"),
            per_camera as f64,
            AT_LEAST_ONE,
        )?;
        check(
            at("track_refill.min_spacing_px"),
            min_spacing_px.into(),
            POSITIVE,
        )?;
        check(
            at("track_refill.refill_below"),
            refill_below.into(),
            OPEN_UNIT_UPPER,
        )?;
    }
    if let FlowPose::AnalyticGn { max_iter } = frontend.flow_pose {
        check(at("flow_pose.max_iter"), max_iter as f64, AT_LEAST_ONE)?;
    }
    let threaded = frontend.flow_threads > 1;
    if threaded
        && (frontend.klt_rig != KltRig::AllCameras || sensors.cameras != CameraSelection::Rig)
    {
        return needs("flow_threads", "klt_rig AllCameras on a rig");
    }
    // kornia-imgproc's f32 LK supports this window only (`optical_flow_pyr_lk.rs:39-44`).
    if frontend.klt_window != 21 && frontend.klt_kernel == LkKernel::F32 {
        return needs(
            "klt_window",
            "klt_kernel FixedPointU8 or WindowedU8 unless it is 21",
        );
    }
    // These run on WindowedU8's per-track template cache.
    if frontend.klt_kernel != LkKernel::WindowedU8 {
        let windowed = [
            ("klt_retry", frontend.klt_retry != KltRetry::Off),
            (
                "forward_backward",
                frontend.forward_backward != ForwardBackward::Off,
            ),
            ("track_refill", frontend.track_refill != TrackRefill::Off),
            ("flow_threads", threaded),
        ];
        if let Some((knob, _)) = windowed.into_iter().find(|&(_, on)| on) {
            return needs(knob, "klt_kernel WindowedU8");
        }
    }
    // Only a flow frame defers its keyframe ORB.
    if frontend.keyframe_orb == KeyframeOrb::Mapping && frontend.klt_first == KltFirst::Disabled {
        return needs("keyframe_orb", "klt_first Enabled");
    }
    // `SlamSystem::build` has no tracking path for it yet (R-C6).
    if frontend.keyframe_orb == KeyframeOrb::Mapping {
        return needs("keyframe_orb", UNBUILT);
    }
    Ok(())
}

/// The sliding window runs on a rectified pair and the IMU; `Drive` has no
/// tracking path yet (rung 2x-6).
fn validate_odometry(odometry: &Odometry, sensors: &SensorSelection) -> Result<(), ConfigError> {
    let Some(window) = odometry.sliding_window() else {
        return Ok(());
    };
    if !sensors.imu {
        return Err(ConfigError::InertialKnobWithoutImu("odometry"));
    }
    if sensors.cameras != CameraSelection::Stereo {
        return Err(ConfigError::OdometryNeeds {
            needs: "sensors.cameras: Stereo (a rectified pair)",
        });
    }
    if window.mode == OdometryMode::Drive {
        return Err(ConfigError::OdometryNeeds { needs: UNBUILT });
    }
    if let OdometryImuNoise::Explicit(noise) = window.imu_noise {
        for (field, value) in [
            ("gyro_noise", noise.gyro_noise),
            ("accel_noise", noise.accel_noise),
            ("gyro_bias_noise", noise.gyro_bias_noise),
            ("accel_bias_noise", noise.accel_bias_noise),
        ] {
            check(format!("odometry.imu_noise.{field}"), value, POSITIVE)?;
        }
    }
    Ok(())
}

/// What a knob that parses but no tracking path reads yet needs.
pub(super) const UNBUILT: &str = "a tracking path, which is not built yet (keep the default)";

/// Every online calibration sigma, as the offline ones.
fn validate_online_calibration(config: &OnlineCalibrationConfig) -> Result<(), ConfigError> {
    let at = |field| format!("mapping.local_ba.online_calibration.{field}");
    if let Some(prior) = config.camera_rotation {
        check(
            at("camera_rotation.step_sigma_deg"),
            prior.step_sigma_deg,
            POSITIVE,
        )?;
        check(
            at("camera_rotation.value_sigma_deg"),
            prior.value_sigma_deg,
            POSITIVE,
        )?;
    }
    if let Some(prior) = config.camera_focal {
        check(at("camera_focal.step_sigma"), prior.step_sigma, POSITIVE)?;
        check(at("camera_focal.value_sigma"), prior.value_sigma, POSITIVE)?;
    }
    if let Some(prior) = config.gyro_intrinsics {
        check(at("gyro_intrinsics.sigma"), prior.sigma, POSITIVE)?;
    }
    if let Some(prior) = config.clock_offset {
        check(at("clock_offset.sigma_ms"), prior.sigma_ms, POSITIVE)?;
    }
    Ok(())
}

fn validate_keyframes(policy: &KeyframePolicy, imu: bool) -> Result<(), ConfigError> {
    let (min, max) = (policy.min_frames_between, policy.max_frames_between);
    if min == 0 || min > max {
        return Err(ConfigError::InvalidKeyframeGaps { min, max });
    }
    if !(0.0..=1.0).contains(&policy.ref_ratio) {
        return Err(ConfigError::InvalidRefRatio(policy.ref_ratio));
    }
    if let Some(px) = policy.min_parallax_px {
        check("keyframes.min_parallax_px", px, POSITIVE)?;
    }
    if let Some(secs) = policy.max_interval_sec {
        check("keyframes.max_interval_sec", secs, POSITIVE)?;
    }
    if let KeyframeRules::OrbSlam3 { max_frames } = policy.rules {
        check(
            "keyframes.rules.max_frames",
            max_frames as f64,
            AT_LEAST_ONE,
        )?;
    }
    // IMU samples are kept until the next keyframe: a parked rig with no
    // bound would buffer them, and grow the next IMU edge, without limit.
    if imu && max == usize::MAX && policy.max_interval_sec.is_none() {
        return Err(ConfigError::UnboundedInertialKeyframes);
    }
    Ok(())
}

fn validate_local_ba(config: &LocalBaConfig) -> Result<(), ConfigError> {
    let at = |field| format!("mapping.local_ba.{field}");
    check(
        at("active_keyframes"),
        config.active_keyframes as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("max_iterations"),
        config.max_iterations as f64,
        AT_LEAST_ONE,
    )?;
    check(at("initial_lambda"), config.initial_lambda, POSITIVE)?;
    if let BoundaryPrior::Marginal(sigma) = config.boundary {
        check(at("boundary"), sigma.frozen_pose_sigma_m, POSITIVE)?;
        check(at("boundary"), sigma.frozen_pose_sigma_rad, POSITIVE)?;
    }
    if let LargeWindow::Enabled {
        active_keyframes,
        max_iterations,
        initial_lambda,
        ..
    } = config.large_window
    {
        if active_keyframes < config.active_keyframes {
            return Err(ConfigError::LargeWindowNotLarger {
                active: config.active_keyframes,
                large: active_keyframes,
            });
        }
        check(
            at("large_window.max_iterations"),
            max_iterations as f64,
            AT_LEAST_ONE,
        )?;
        check(at("large_window.initial_lambda"), initial_lambda, POSITIVE)?;
    }
    match config.stereo_depth_sigma {
        StereoDepthSigma::RelativeDepth(value) | StereoDepthSigma::Disparity { px: value } => {
            check(at("stereo_depth_sigma"), value, POSITIVE)?;
        }
    }
    if let OutlierErase::ChiSquare(rule) = config.outlier_erase {
        check(at("outlier_erase"), rule.reprojection_chi2, POSITIVE)?;
        check(at("outlier_erase"), rule.stereo_chi2, POSITIVE)?;
        check(at("outlier_erase"), rule.close_factor, AT_LEAST_ONE)?;
        check(at("outlier_erase"), rule.close_range_m, NON_NEGATIVE)?;
    }
    validate_online_calibration(&config.online_calibration)
}

fn validate_initialization(config: &TwoViewInitConfig) -> Result<(), ConfigError> {
    let section = "tuning.initialization";
    validate_orb_match(section, "match_config", &config.match_config)?;
    let t = &config.triangulation_config;
    let at = |field| format!("{section}.triangulation_config.{field}");
    check(at("min_parallax_deg"), t.min_parallax_deg, NON_NEGATIVE)?;
    check(at("max_midpoint_gap"), t.max_midpoint_gap, POSITIVE)?;
    check(
        at("max_reprojection_error"),
        t.max_reprojection_error,
        POSITIVE,
    )?;
    check(
        at("cheirality_ambiguity_max"),
        t.cheirality_ambiguity_max,
        UNIT_INTERVAL,
    )
}

fn validate_tracking(tuning: &OrbTuning) -> Result<(), ConfigError> {
    let section = "tuning.map_projection";
    let at = |field| format!("{section}.{field}");
    let projection = &tuning.map_projection;
    validate_orb_match(section, "match_config", &projection.match_config)?;
    let pnp = &projection.pnp;
    check(
        at("pnp.prior_reproj_threshold_px"),
        pnp.prior_reproj_threshold_px,
        POSITIVE,
    )?;
    check(
        at("pnp.final_reproj_threshold_px"),
        pnp.final_reproj_threshold_px,
        POSITIVE,
    )?;
    check(
        at("pnp.robust_scale_sq"),
        pnp.robust_scale_sq.into(),
        POSITIVE,
    )?;
    validate_projection(at("projection"), &projection.projection)?;
    validate_projection(at("local_projection"), &projection.local_projection)?;
    check(
        at("lost_search_scale"),
        projection.lost_search_scale.into(),
        AT_LEAST_ONE,
    )?;
    check(
        at("search_widen_per_sec"),
        projection.search_widen_per_sec.into(),
        NON_NEGATIVE,
    )?;
    check(
        at("min_inliers_with_prediction"),
        projection.min_inliers_with_prediction as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("geometric_filter_threshold_px"),
        projection.geometric_filter_threshold_px,
        POSITIVE,
    )?;

    let recovery = &tuning.loss_recovery;
    let at = |field| format!("tuning.loss_recovery.{field}");
    check(
        at("timeout_imu_sec"),
        recovery.timeout_imu_sec,
        NON_NEGATIVE,
    )?;
    check(
        at("timeout_visual_sec"),
        recovery.timeout_visual_sec,
        NON_NEGATIVE,
    )?;
    check(
        at("min_imu_confidence_sec"),
        recovery.min_imu_confidence_sec,
        NON_NEGATIVE,
    )?;
    check(
        at("keyframe_interval_while_lost_sec"),
        recovery.keyframe_interval_while_lost_sec,
        POSITIVE,
    )
}

fn validate_projection(section: String, config: &ProjectionMatchConfig) -> Result<(), ConfigError> {
    check(
        format!("{section}.min_depth"),
        config.min_depth,
        NON_NEGATIVE,
    )?;
    check(
        format!("{section}.search_radius"),
        config.search_radius.into(),
        POSITIVE,
    )
}

fn validate_orb_match(
    section: &str,
    field: &str,
    config: &OrbMatchConfig,
) -> Result<(), ConfigError> {
    let at = |name| format!("{section}.{field}.{name}");
    check(at("nn_ratio"), config.nn_ratio.into(), OPEN_UNIT_UPPER)?;
    if config.check_orientation {
        check(at("histo_length"), config.histo_length as f64, AT_LEAST_ONE)?;
    }
    Ok(())
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
    // corrections.
    let metric = sensors.cameras != CameraSelection::Mono || sensors.imu;
    if mode.is_enabled() && !metric {
        return Err(ConfigError::LoopClosingWithoutMetricScale);
    }
    match mode.graph().map(|graph| graph.information) {
        Some(LoopGraphInformation::Drift {
            odometry,
            loop_sigma_m,
            loop_sigma_rad,
            loop_cauchy_scale,
        }) => {
            validate_drift_information(&odometry, [loop_sigma_m, loop_sigma_rad, loop_cauchy_scale])
        }
        Some(LoopGraphInformation::ScalarWeights) | None => Ok(()),
    }
}

fn validate_drift_information(
    odometry: &OdometryRandomWalk,
    [loop_sigma_m, loop_sigma_rad, loop_cauchy_scale]: [f64; 3],
) -> Result<(), ConfigError> {
    let at = |field| format!("pipeline.loop_closing.graph.information.{field}");
    check(
        at("odometry.translation_m2_per_m"),
        odometry.translation_m2_per_m,
        NON_NEGATIVE,
    )?;
    check(
        at("odometry.rotation_rad2_per_m"),
        odometry.rotation_rad2_per_m,
        NON_NEGATIVE,
    )?;
    check(at("odometry.floor_m"), odometry.floor_m, POSITIVE)?;
    check(at("odometry.floor_rad"), odometry.floor_rad, POSITIVE)?;
    check(at("loop_sigma_m"), loop_sigma_m, POSITIVE)?;
    check(at("loop_sigma_rad"), loop_sigma_rad, POSITIVE)?;
    check(at("loop_cauchy_scale"), loop_cauchy_scale, POSITIVE)
}

fn validate_correction(config: &LoopClosingConfig) -> Result<(), ConfigError> {
    let section = "tuning.loop_correction";
    let at = |field| format!("{section}.{field}");
    let v = &config.verification;
    validate_orb_match(section, "verification.orb_match", &v.orb_match)?;
    check(
        at("verification.min_inlier_ratio"),
        v.min_inlier_ratio.into(),
        UNIT_INTERVAL,
    )?;
    check(
        at("verification.max_reprojection_rmse_px"),
        v.max_reprojection_rmse_px.into(),
        POSITIVE,
    )?;
    check(
        at("verification.coverage_rows"),
        v.coverage_rows as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("verification.coverage_cols"),
        v.coverage_cols as f64,
        AT_LEAST_ONE,
    )?;
    let ransac = &v.pnp_ransac;
    check(
        at("verification.pnp_ransac.max_iterations"),
        ransac.max_iterations as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("verification.pnp_ransac.reproj_threshold_px"),
        ransac.reproj_threshold_px.into(),
        POSITIVE,
    )?;
    check(
        at("verification.pnp_ransac.confidence"),
        ransac.confidence.into(),
        OPEN_UNIT,
    )?;
    check(
        at("episode.min_consistent_edges"),
        config.episode.min_consistent_edges as f64,
        AT_LEAST_ONE,
    )?;
    if let LoopOdometryGate::Enabled { drift, k_sigma } = config.odometry_gate {
        check(at("odometry_gate.k_sigma"), k_sigma, POSITIVE)?;
        check(
            at("odometry_gate.drift.translation_per_m"),
            drift.translation_per_m,
            NON_NEGATIVE,
        )?;
        check(
            at("odometry_gate.drift.translation_floor_m"),
            drift.translation_floor_m,
            POSITIVE,
        )?;
        check(
            at("odometry_gate.drift.yaw_per_m_rad"),
            drift.yaw_per_m_rad,
            NON_NEGATIVE,
        )?;
        check(
            at("odometry_gate.drift.yaw_floor_rad"),
            drift.yaw_floor_rad,
            POSITIVE,
        )?;
    }
    if let LoopTemporalConsistency::Enabled {
        min_detections,
        max_translation_m,
        max_yaw_rad,
    } = config.temporal_consistency
    {
        check(
            at("temporal_consistency.min_detections"),
            min_detections as f64,
            AT_LEAST_ONE,
        )?;
        check(
            at("temporal_consistency.max_translation_m"),
            max_translation_m,
            POSITIVE,
        )?;
        check(
            at("temporal_consistency.max_yaw_rad"),
            max_yaw_rad,
            POSITIVE,
        )?;
    }
    let fusion = &config.fusion;
    check(
        at("fusion.search_radius_px"),
        fusion.search_radius_px.into(),
        POSITIVE,
    )?;
    check(
        at("fusion.max_reprojection_error_px"),
        fusion.max_reprojection_error_px,
        POSITIVE,
    )?;
    let pgo = &config.optimizer;
    check(
        at("optimizer.loop_edge_weight"),
        pgo.loop_edge_weight.into(),
        POSITIVE,
    )?;
    check(
        at("optimizer.max_iterations"),
        pgo.max_iterations as f64,
        AT_LEAST_ONE,
    )?;
    check(
        at("optimizer.cost_tolerance"),
        pgo.cost_tolerance.into(),
        NON_NEGATIVE,
    )?;
    check(
        at("optimizer.gradient_tolerance"),
        pgo.gradient_tolerance.into(),
        NON_NEGATIVE,
    )?;
    check(
        at("optimizer.initial_lambda"),
        pgo.initial_lambda.into(),
        POSITIVE,
    )
}

impl OfflineBaConfig {
    /// Checks the offline pass's settings on their own.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let at = |field| format!("offline_ba.{field}");
        check(
            at("relative_cost_tolerance"),
            self.relative_cost_tolerance,
            POSITIVE,
        )?;
        check(at("outlier_px"), self.outlier_px, POSITIVE)?;
        check(at("tight_outlier_px"), self.tight_outlier_px, POSITIVE)?;
        if self.tight_outlier_px > self.outlier_px {
            return Err(ConfigError::InvalidSetting {
                setting: at("tight_outlier_px"),
                value: self.tight_outlier_px,
                requirement: "<= offline_ba.outlier_px",
            });
        }
        check(
            at("tighten_from_iteration"),
            self.tighten_from_iteration as f64,
            AT_LEAST_ONE,
        )?;
        check(
            at("landmark_damping_floor"),
            self.landmark_damping_floor,
            OPEN_UNIT_UPPER,
        )?;
        let calibration = &self.calibration;
        let at = |field| format!("offline_ba.calibration.{field}");
        check(
            at("time_slices"),
            calibration.time_slices as f64,
            Requirement("in 1..=64", |v| (1.0..=64.0).contains(&v)),
        )?;
        if let Some(prior) = calibration.camera_rotation {
            check(
                at("camera_rotation.step_sigma_deg"),
                prior.step_sigma_deg,
                POSITIVE,
            )?;
            check(
                at("camera_rotation.value_sigma_deg"),
                prior.value_sigma_deg,
                POSITIVE,
            )?;
        }
        if let Some(prior) = calibration.camera_focal {
            check(at("camera_focal.step_sigma"), prior.step_sigma, POSITIVE)?;
            check(at("camera_focal.value_sigma"), prior.value_sigma, POSITIVE)?;
        }
        if let Some(prior) = calibration.gyro_intrinsics {
            check(at("gyro_intrinsics.sigma"), prior.sigma, POSITIVE)?;
        }
        if let Some(prior) = calibration.clock_offset {
            check(at("clock_offset.sigma_ms"), prior.sigma_ms, POSITIVE)?;
        }
        Ok(())
    }
}

/// A predicate on a finite value, with its description for error messages.
struct Requirement(&'static str, fn(f64) -> bool);

const POSITIVE: Requirement = Requirement("finite and > 0", |v| v > 0.0);
const NON_NEGATIVE: Requirement = Requirement("finite and >= 0", |v| v >= 0.0);
const AT_LEAST_ONE: Requirement = Requirement("finite and >= 1", |v| v >= 1.0);
const UNIT_INTERVAL: Requirement = Requirement("in [0, 1]", |v| (0.0..=1.0).contains(&v));
const OPEN_UNIT: Requirement = Requirement("in (0, 1)", |v| v > 0.0 && v < 1.0);
const OPEN_UNIT_UPPER: Requirement = Requirement("in (0, 1]", |v| v > 0.0 && v <= 1.0);
/// A span the IMU buffer still holds whole: a bridge edge integrated over a
/// longer one would come from a truncated buffer (R-C8).
const WITHIN_IMU_BUFFER: Requirement =
    Requirement("in (0, 30] s, the span of the IMU buffer", |v| {
        v > 0.0 && v <= MAX_BUFFERED_SEC
    });

fn check(
    setting: impl Into<String>,
    value: f64,
    requirement: Requirement,
) -> Result<(), ConfigError> {
    let Requirement(description, holds) = requirement;
    if value.is_finite() && holds(value) {
        Ok(())
    } else {
        Err(ConfigError::InvalidSetting {
            setting: setting.into(),
            value,
            requirement: description,
        })
    }
}

impl SensorSelection {
    /// Checks that the source's rig provides every selected sensor.
    pub fn validate_rig(&self, rig: &SensorRig) -> Result<(), ConfigError> {
        if self.cameras == CameraSelection::Stereo {
            if rig.stereo_baseline_m.is_none() {
                return Err(ConfigError::MissingSensor("stereo cameras"));
            }
            if rig.fisheye.is_some() {
                return Err(ConfigError::FisheyeStereo);
            }
        }
        if self.cameras == CameraSelection::Rig {
            if !rig.is_multi_camera() {
                return Err(ConfigError::MissingSensor("a multi-camera rig"));
            }
            if rig.fisheye.is_some() {
                return Err(ConfigError::FisheyeStereo);
            }
        }
        // Any other selection would run the raw images as pinhole ones.
        if self.cameras != CameraSelection::Rig
            && rig.cameras.iter().any(|camera| camera.raw_model.is_some())
        {
            return Err(ConfigError::RawFisheyeOutsideRig);
        }
        if self.imu && rig.imu.is_none() {
            return Err(ConfigError::MissingSensor("an IMU"));
        }
        Ok(())
    }
}
