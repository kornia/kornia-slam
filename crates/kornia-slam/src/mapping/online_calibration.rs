//! Online calibration in the local VI-BA (card G): the estimate carried from
//! one solve to the next, and what a calibrated solve reads.

use crate::mapping::global_vi_ba::CalibrationEstimate;
use crate::mapping::map::ImuFactor as MapImuFactor;
use crate::system::OnlineCalibrationConfig;

/// Oldest free keyframe speed (m/s) below which the states are held (VINS-Fusion
/// `estimator.cpp:1042`).
pub(crate) const GATE_MIN_SPEED_MPS: f64 = 0.2;

/// The calibration estimate carried from one local VI-BA to the next, relative to the
/// factory calibration; `Default` is the factory calibration.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OnlineCalibrationState {
    /// Lane C: the values in `Layout::online` order.
    values: Vec<f64>,
}

impl OnlineCalibrationState {
    /// The solved values in the offline report's shape (one slice per camera);
    /// `None` before any ungated solve.
    pub(crate) fn estimate(&self) -> Option<CalibrationEstimate> {
        todo!("lane C: online calibration estimate")
    }
}

/// What one calibrated solve reads besides `ViBaParams`.
pub(crate) struct OnlineCalibrationInput<'a> {
    pub config: OnlineCalibrationConfig,
    pub start: &'a OnlineCalibrationState,
    /// The map edge of each entry of `imu_edges`, in the same order (raw samples,
    /// `t0`, `t1`, interpolation, bias walk).
    pub map_edges: Vec<&'a MapImuFactor>,
    /// `LocalBaParams::active_keyframes` of this solve, for the "window full" gate.
    pub active_keyframes: usize,
}
