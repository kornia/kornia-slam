use kornia_imgproc::features::OrbMatchConfig;
use kornia_sensors::SensorRig;

use super::config::{
    CameraSelection, FrontendConfig, LoopClosingMode, OrbFrontendConfig, OrbSlamPipeline,
    OrbTuning, PIPELINE_CONFIG_VERSION, PipelineConfig, PipelineDefinition, SensorSelection,
    StereoCloseDepth,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::LoopClosingConfig;
use crate::tracking::KeyframePolicy;
use crate::tracking::pose_estimation::map_projection::ProjectionMatchConfig;

/// A pipeline configuration that cannot be built or does not fit its source.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    #[error("unsupported pipeline config version {found}; this build reads version {supported}")]
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
    #[error("loop closing vocabulary path is empty")]
    EmptyVocabularyPath,
    #[error("loop closing needs metric input: enable stereo cameras or the IMU")]
    LoopClosingWithoutMetricScale,
    #[error("configuration requests {0}, but the source does not provide it")]
    MissingSensor(&'static str),
    #[error("stereo cameras need rectified images; the source supplies raw fisheye images")]
    FisheyeStereo,
}

impl PipelineConfig {
    /// Checks the definition on its own, before any source or resource is opened.
    pub fn validate(&self) -> Result<(), ConfigError> {
        Self::check_version(self.version)?;
        match &self.pipeline {
            PipelineDefinition::OrbSlam(orb) => orb.validate(&self.sensors),
        }
    }
}

impl PipelineConfig {
    /// Checks a schema version on its own, so a file written for another
    /// version can be reported as such before its fields are parsed.
    pub fn check_version(version: u32) -> Result<(), ConfigError> {
        if version == PIPELINE_CONFIG_VERSION {
            Ok(())
        } else {
            Err(ConfigError::UnsupportedVersion {
                found: version,
                supported: PIPELINE_CONFIG_VERSION,
            })
        }
    }
}

impl OrbSlamPipeline {
    fn validate(&self, sensors: &SensorSelection) -> Result<(), ConfigError> {
        match &self.frontend {
            FrontendConfig::Orb(orb) => orb.validate()?,
        }
        validate_keyframes(&self.keyframes)?;
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
        match self.stereo_close_depth {
            StereoCloseDepth::Baselines(value) | StereoCloseDepth::Metres(value) => {
                check("frontend.stereo_close_depth", value, POSITIVE)
            }
            StereoCloseDepth::Disabled => Ok(()),
        }
    }
}

fn validate_keyframes(policy: &KeyframePolicy) -> Result<(), ConfigError> {
    let (min, max) = (policy.min_frames_between, policy.max_frames_between);
    if min == 0 || min > max {
        return Err(ConfigError::InvalidKeyframeGaps { min, max });
    }
    if !(0.0..=1.0).contains(&policy.ref_ratio) {
        return Err(ConfigError::InvalidRefRatio(policy.ref_ratio));
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
        at("search_widen_per_sec"),
        projection.search_widen_per_sec.into(),
        NON_NEGATIVE,
    )?;
    check(
        at("max_search_scale"),
        projection.max_search_scale.into(),
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
    let metric = sensors.cameras == CameraSelection::Stereo || sensors.imu;
    if mode.is_enabled() && !metric {
        return Err(ConfigError::LoopClosingWithoutMetricScale);
    }
    Ok(())
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
        if self.cameras == CameraSelection::Stereo {
            if rig.stereo_baseline_m.is_none() {
                return Err(ConfigError::MissingSensor("stereo cameras"));
            }
            if rig.fisheye.is_some() {
                return Err(ConfigError::FisheyeStereo);
            }
        }
        if self.imu && rig.imu.is_none() {
            return Err(ConfigError::MissingSensor("an IMU"));
        }
        Ok(())
    }
}
