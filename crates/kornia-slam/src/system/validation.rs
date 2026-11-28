use kornia_imgproc::features::OrbMatchConfig;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FlowPose, ForwardBackward, FrontendConfig, GyroGuess, KltRetry, KltRig,
    LoopClosingMode, MAX_FLOW_THREADS, MAX_KLT_WINDOW, OrbFrontendConfig, OrbSlamPipeline,
    OrbTuning, PipelineDefinition, SYSTEM_CONFIG_VERSION, SensorSelection, StereoCloseDepth,
    SystemConfig, TrackRefill,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopClosingConfig;
use crate::tracking::KeyframePolicy;
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::pose_estimation::map_projection::ProjectionMatchConfig;

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
    #[error("frontend.{knob} needs {needs}")]
    FrontendKnobNeeds {
        knob: &'static str,
        needs: &'static str,
    },
    #[error("frontend.flow_threads {threads} needs 1, {cameras} or {} threads on this rig", cameras + 1)]
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
        validate_loop_closing(&self.loop_closing, sensors)?;
        validate_tuning(&self.tuning, self.loop_closing.is_enabled())
    }
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
}

impl OrbFrontendConfig {
    /// `flow_threads` against a rig of `cameras` cameras: one shard per
    /// thread needs 1, one thread per camera, or one more.
    pub(crate) fn validate_flow_threads(&self, cameras: usize) -> Result<(), ConfigError> {
        let threads = self.flow_threads;
        if threads == 1 || threads == cameras || threads == cameras + 1 {
            Ok(())
        } else {
            Err(ConfigError::FlowThreadsForRig { threads, cameras })
        }
    }
}

/// The optical-flow knobs: ranges, and the knobs each one needs.
fn validate_flow(
    frontend: &OrbFrontendConfig,
    sensors: &SensorSelection,
) -> Result<(), ConfigError> {
    let at = |field| format!("frontend.{field}");
    let needs = |knob: &'static str, needs: &'static str| -> Result<(), ConfigError> {
        Err(ConfigError::FrontendKnobNeeds { knob, needs })
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
        Requirement("odd and in 3..=MAX_KLT_WINDOW", |v| {
            (3.0..=MAX_KLT_WINDOW as f64).contains(&v) && v % 2.0 == 1.0
        }),
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
    // Only a rig frontend has the other cameras' geometry; elsewhere the
    // knob would silently track the primary alone.
    if frontend.klt_rig == KltRig::AllCameras && sensors.cameras != CameraSelection::Rig {
        return needs("klt_rig", "sensors.cameras Rig");
    }
    // The full-pyramid kernels start from a guess only on the predicted
    // search, so without one the gyro would never be read.
    if frontend.gyro_guess == GyroGuess::RotatedRay
        && frontend.klt_kernel != LkKernel::WindowedU8
        && frontend.klt_prediction.search().is_none()
    {
        return needs("gyro_guess", "klt_prediction or klt_kernel WindowedU8");
    }
    // kornia-imgproc's f32 LK supports this window only.
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
    // IMU samples are kept until the next keyframe: a parked rig with no
    // bound would buffer them, and grow the next IMU edge, without limit.
    if imu && max == usize::MAX && policy.max_interval_sec.is_none() {
        return Err(ConfigError::UnboundedInertialKeyframes);
    }
    Ok(())
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
    Ok(())
}

fn validate_correction(config: &LoopClosingConfig) -> Result<(), ConfigError> {
    let section = "tuning.loop_correction";
    let at = |field| format!("{section}.{field}");
    let v = &config.verification;
    validate_orb_match(
        section,
        "verification.descriptor_match",
        &v.descriptor_match,
    )?;
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

/// A predicate on a finite value, with its description for error messages.
struct Requirement(&'static str, fn(f64) -> bool);

const POSITIVE: Requirement = Requirement("finite and > 0", |v| v > 0.0);
const NON_NEGATIVE: Requirement = Requirement("finite and >= 0", |v| v >= 0.0);
const AT_LEAST_ONE: Requirement = Requirement("finite and >= 1", |v| v >= 1.0);
const UNIT_INTERVAL: Requirement = Requirement("in [0, 1]", |v| (0.0..=1.0).contains(&v));
const OPEN_UNIT: Requirement = Requirement("in (0, 1)", |v| v > 0.0 && v < 1.0);
const OPEN_UNIT_UPPER: Requirement = Requirement("in (0, 1]", |v| v > 0.0 && v <= 1.0);

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
        if self.cameras == CameraSelection::Stereo && rig.rectified_stereo_baseline_m().is_none() {
            return Err(ConfigError::MissingSensor("stereo cameras"));
        }
        if self.cameras == CameraSelection::Rig && rig.cameras().len() < 2 {
            return Err(ConfigError::MissingSensor("a multi-camera rig"));
        }
        if self.imu && rig.imu.is_none() {
            return Err(ConfigError::MissingSensor("an IMU"));
        }
        Ok(())
    }
}
