//! Offline global visual-inertial BA over a recorded session
//! ([`super::session::SessionDump`]), with optional calibration states:
//! camera drift, gyro intrinsics and clock. Problem shape after ORB-SLAM3's
//! `FullInertialBA` (`Optimizer.cc:392-811`); landmarks are eliminated by the
//! Schur complement and the reduced system is a faer sparse Cholesky in f64.

use kornia_3d::pose::Pose3d;
use kornia_sensors::imu::GyroIntrinsics;

use super::session::SessionDump;
use super::visual_information::VisualInformation;
use crate::system::{ConfigError, StereoDepthSigma};

/// The offline pass's settings: the run file's `offline_ba` section. Only the
/// `offline-ba` subcommand reads them; every calibration state and the segment
/// bridge are off by default.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OfflineBaConfig {
    /// LM iteration cap; `0` writes the start back out (the round trip).
    pub max_iterations: usize,
    /// Stop once an accepted step lowers the cost by less than this fraction.
    pub relative_cost_tolerance: f64,
    /// Observations above this reprojection error (px) are dropped.
    pub outlier_px: f64,
    /// The tighter threshold (px) used from `tighten_from_iteration` on.
    pub tight_outlier_px: f64,
    pub tighten_from_iteration: usize,
    /// Landmark damping floor, a fraction of its largest Hessian diagonal entry.
    pub landmark_damping_floor: f64,
    pub imu_kernel: OfflineImuKernel,
    /// Join consecutive segments with an IMU edge over a gap of at most 10 s.
    pub bridge_segments: bool,
    pub calibration: OfflineCalibrationConfig,
}

impl Default for OfflineBaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 60,
            relative_cost_tolerance: 1e-7,
            outlier_px: 10.0,
            tight_outlier_px: 3.0,
            tighten_from_iteration: 8,
            landmark_damping_floor: 0.01,
            imu_kernel: OfflineImuKernel::None,
            bridge_segments: false,
            calibration: OfflineCalibrationConfig::default(),
        }
    }
}

/// Robust kernel on the inertial edges of the offline solve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OfflineImuKernel {
    #[default]
    None,
    /// ORB-SLAM3 `FullInertialBA`: Huber δ √16.92 (`Optimizer.cc:540-542`).
    Huber,
}

/// Calibration states the offline solve estimates; each is off when `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OfflineCalibrationConfig {
    /// Time slices per recording for the per-camera drift states.
    pub time_slices: usize,
    pub camera_rotation: Option<RotationSlicePrior>,
    pub camera_focal: Option<FocalSlicePrior>,
    pub gyro_intrinsics: Option<GyroIntrinsicsPrior>,
    pub clock_offset: Option<ClockOffsetPrior>,
}

impl Default for OfflineCalibrationConfig {
    fn default() -> Self {
        Self {
            time_slices: 8,
            camera_rotation: None,
            camera_focal: None,
            gyro_intrinsics: None,
            clock_offset: None,
        }
    }
}

/// Per-camera IMU-camera rotation per slice, a random walk between slices.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RotationSlicePrior {
    pub step_sigma_deg: f64,
    pub value_sigma_deg: f64,
}

/// Per-camera focal scale per slice, relative to the calibrated focal.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FocalSlicePrior {
    pub step_sigma: f64,
    pub value_sigma: f64,
}

/// Symmetric gyro correction `ω ← (I + S) ω`; `sigma` on every entry of `S`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GyroIntrinsicsPrior {
    pub sigma: f64,
}

/// One IMU-camera time offset per recording (VINS-Fusion's `td`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClockOffsetPrior {
    pub sigma_ms: f64,
}

/// The calibration states' solved values; empty for a state that was off.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalibrationEstimate {
    /// Per camera, per slice: rotation offset, axis-angle (rad).
    pub camera_rotation: Vec<Vec<[f64; 3]>>,
    /// Per camera, per slice: focal scale minus one.
    pub camera_focal: Vec<Vec<f64>>,
    pub gyro_intrinsics: Option<GyroIntrinsics>,
    pub clock_offset_sec: Option<f64>,
}

/// Counters of one offline solve, printed as `[offline_ba]` lines.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OfflineBaReport {
    pub frames: usize,
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub dropped_outlier: usize,
    pub dropped_tight_outlier: usize,
    pub components: usize,
    pub bridged: usize,
    pub covisible_median: usize,
    pub nnz_reduced: usize,
    pub nnz_factor: usize,
}

pub struct OfflineBaResult {
    /// Per dump keyframe, in dump order: the solved `T_bw`.
    pub body_poses: Vec<Pose3d>,
    pub calibration: CalibrationEstimate,
    pub report: OfflineBaReport,
}

#[derive(Debug, thiserror::Error)]
pub enum OfflineBaError {
    #[error("the session holds no IMU samples")]
    NoImu,
    #[error("session dump version {0} is not the version this build reads")]
    VersionMismatch(u32),
    #[error("reduced-system Cholesky failed: {0}")]
    CholeskyFailed(String),
    #[error(transparent)]
    InvalidConfig(#[from] ConfigError),
}

/// Solves one global VI-BA over `session`, weighting observations as the
/// online VI-BA does (`visual`, `stereo_depth_sigma`).
pub fn offline_visual_inertial_ba(
    _session: &SessionDump,
    _visual: &VisualInformation,
    _stereo_depth_sigma: StereoDepthSigma,
    _config: &OfflineBaConfig,
) -> Result<OfflineBaResult, OfflineBaError> {
    todo!("lane D: offline global VI-BA")
}
