//! Online calibration in the local VI-BA (card G): the estimate carried from
//! one solve to the next, the gate that holds it, and what a calibrated solve
//! reads.

use std::fmt::Write as _;

use crate::mapping::calibration_states::{Calibration, Layout};
#[cfg(test)]
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
    /// The values in `Layout::online(&config, cameras)` order.
    values: Vec<f64>,
    config: OnlineCalibrationConfig,
    /// Cameras the values cover: the most any solve has seen, so a window
    /// without the secondary cameras keeps their estimate.
    cameras: usize,
    /// Ungated solves the values come from.
    solves: usize,
}

impl OnlineCalibrationState {
    /// The solved values in the offline report's shape (one slice per camera);
    /// `None` before any ungated solve; the calibration tests read it.
    #[cfg(test)]
    pub(crate) fn estimate(&self) -> Option<CalibrationEstimate> {
        (self.solves > 0).then(|| self.calibration().estimate(&self.layout()))
    }

    fn layout(&self) -> Layout {
        Layout::online(&self.config, self.cameras)
    }

    #[cfg(test)]
    fn calibration(&self) -> Calibration {
        Calibration {
            values: self.values.clone(),
        }
    }

    /// The carried values laid out in `layout`: factory (0) for a state the
    /// estimate does not hold.
    pub(in crate::mapping) fn start(&self, layout: &Layout) -> Calibration {
        let mut start = Calibration::zero(layout);
        let own = self.layout();
        if own.len == self.values.len() {
            transfer(&own, &self.values, layout, &mut start.values);
        }
        start
    }

    /// The state after a solve over `layout` returned `solved`; a gated solve
    /// leaves the estimate unchanged.
    pub(in crate::mapping) fn advanced(
        &self,
        config: &OnlineCalibrationConfig,
        layout: &Layout,
        solved: &Calibration,
        gated: bool,
    ) -> Self {
        if gated {
            return self.clone();
        }
        let cameras = self.cameras.max(layout.cameras());
        let next_layout = Layout::online(config, cameras);
        let mut values = vec![0.0; next_layout.len];
        let own = self.layout();
        if self.config == *config && own.len == self.values.len() {
            transfer(&own, &self.values, &next_layout, &mut values);
        }
        transfer(layout, &solved.values, &next_layout, &mut values);
        Self {
            values,
            config: *config,
            cameras,
            solves: self.solves + 1,
        }
    }
}

/// Copies every state both one-slice layouts hold from `values` (in `from`)
/// into `out` (in `to`).
fn transfer(from: &Layout, values: &[f64], to: &Layout, out: &mut [f64]) {
    for camera in 0..from.cameras().min(to.cameras()) {
        if let (Some(a), Some(b)) = (from.rotation(camera, 0), to.rotation(camera, 0)) {
            out[b..b + 3].copy_from_slice(&values[a..a + 3]);
        }
        if let (Some(a), Some(b)) = (from.focal(camera, 0), to.focal(camera, 0)) {
            out[b] = values[a];
        }
    }
    if let (Some(a), Some(b)) = (from.gyro(), to.gyro()) {
        out[b..b + 6].copy_from_slice(&values[a..a + 6]);
    }
    if let (Some(a), Some(b)) = (from.clock(), to.clock()) {
        out[b] = values[a];
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

/// Whether a solve estimates the states (card G, C5): the oldest free
/// keyframe moves at [`GATE_MIN_SPEED_MPS`] or faster and the window holds
/// `active_keyframes` free keyframes (VINS-Fusion `estimator.cpp:1029-1043`).
/// Otherwise the carried estimate is applied as constants.
pub(crate) fn gate_open(
    oldest_free_speed: Option<f64>,
    free_keyframes: usize,
    active_keyframes: usize,
) -> bool {
    free_keyframes >= active_keyframes
        && oldest_free_speed.is_some_and(|speed| speed >= GATE_MIN_SPEED_MPS)
}

/// The figures of one calibrated solve that its `[online_calib]` line prints.
pub(in crate::mapping) struct SolveLog<'a> {
    pub(in crate::mapping) gated: bool,
    pub(in crate::mapping) speed: Option<f64>,
    pub(in crate::mapping) free: usize,
    pub(in crate::mapping) layout: &'a Layout,
    pub(in crate::mapping) values: &'a Calibration,
    /// The newest free keyframe's accel-bias norm, for the absorption check.
    pub(in crate::mapping) ba_norm: f64,
    pub(in crate::mapping) edges_without_samples: usize,
    pub(in crate::mapping) reintegrated: usize,
}

impl SolveLog<'_> {
    /// The `[online_calib]` line (card G, C7); only the states that are on.
    pub(in crate::mapping) fn line(&self, solve: usize) -> String {
        let (layout, values) = (self.layout, self.values);
        let mut line = format!(
            "[online_calib] solve={solve} gated={} speed={:.3} free={}",
            u8::from(self.gated),
            self.speed.unwrap_or(f64::NAN),
            self.free,
        );
        if let Some(at) = layout.clock() {
            let _ = write!(line, " td_ms={:.4}", values.values[at] * 1e3);
        }
        for camera in 0..layout.cameras() {
            if let Some(at) = layout.rotation(camera, 0) {
                let deg = values.vec3(at).to_array().map(f64::to_degrees);
                let _ = write!(
                    line,
                    " cam{camera}_rot_deg=({:.4},{:.4},{:.4})",
                    deg[0], deg[1], deg[2]
                );
            }
        }
        for camera in 0..layout.cameras() {
            if let Some(at) = layout.focal(camera, 0) {
                let _ = write!(
                    line,
                    " cam{camera}_focal_pct={:.4}",
                    values.values[at] * 1e2
                );
            }
        }
        if layout.gyro().is_some() {
            let gyro = values.gyro(layout);
            let (s, k) = (gyro.scale * 1e2, gyro.shear * 1e3);
            let _ = write!(
                line,
                " gyro_scale_pct=({:.4},{:.4},{:.4}) gyro_shear_mrad=({:.4},{:.4},{:.4})",
                s.x, s.y, s.z, k.x, k.y, k.z
            );
        }
        let _ = write!(
            line,
            " ba_norm={:.5} edges_without_samples={} reintegrated={}",
            self.ba_norm, self.edges_without_samples, self.reintegrated
        );
        line
    }
}
