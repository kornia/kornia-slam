//! IMU edges of the offline solve: one per consecutive keyframe pair of a
//! segment, preintegrated from the raw stream, and across a segment boundary
//! only with `bridge_segments` (ORB-SLAM3 `FullInertialBA`,
//! `Optimizer.cc:532-563`: `EdgeInertial` plus the bias random walk).

use kornia_algebra::{Mat15F64, Vec3F64, Vec15F64};
use kornia_factors::inertial::NavState;
use kornia_factors::inertial::preintegration::{
    information, residual_and_jacobians, weight_and_cost,
};
use kornia_sensors::imu::{GyroIntrinsics, ImuBias, ImuMeasurement, PreintegratedImu};

use super::OfflineImuKernel;
use super::calibration::Layout;
use super::problem::{Problem, State};

/// VINS-Fusion builds no IMU factor over more than 10 s (`estimator.cpp:1057-1058`).
const MAX_EDGE_GAP_SEC: f64 = 10.0;
/// ORB-SLAM3 `FullInertialBA`: Huber δ = √16.92 on every inertial edge
/// (`Optimizer.cc:540-542`).
const IMU_HUBER_CHI2: f64 = 16.92;
/// Central-difference steps of the gyro-intrinsics and clock columns: the
/// clock step is a quarter of the 1 kHz sample spacing.
const GYRO_STEP: f64 = 1e-6;
const CLOCK_STEP_SEC: f64 = 0.25e-3;

#[derive(Debug, Clone, Copy)]
pub(super) struct ImuEdge {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) t0: f64,
    pub(super) t1: f64,
    /// Joins two segments.
    pub(super) bridged: bool,
}

/// The edges of `problem`, and the longest gap (s) at a segment boundary.
pub(super) fn build_edges(problem: &Problem, bridge_segments: bool) -> (Vec<ImuEdge>, f64) {
    let mut edges = Vec::new();
    let mut gap_max_s: f64 = 0.0;
    for from in 0..problem.timestamps.len().saturating_sub(1) {
        let to = from + 1;
        let (t0, t1) = (problem.timestamps[from], problem.timestamps[to]);
        let bridged = problem.segments[from] != problem.segments[to];
        if bridged {
            gap_max_s = gap_max_s.max(t1 - t0);
        }
        let samples =
            problem.imu.partition_point(|s| s.t <= t1) - problem.imu.partition_point(|s| s.t < t0);
        if (bridged && !bridge_segments) || t1 - t0 > MAX_EDGE_GAP_SEC || samples < 2 {
            continue;
        }
        edges.push(ImuEdge {
            from,
            to,
            t0,
            t1,
            bridged,
        });
    }
    (edges, gap_max_s)
}

/// One edge linearized at a state.
pub(super) struct ImuLinearization {
    pub(super) residual: Vec15F64,
    pub(super) j_from: Mat15F64,
    pub(super) j_to: Mat15F64,
    /// Information, its navigation block scaled by the kernel weight.
    pub(super) omega: Mat15F64,
    /// The residual's derivative along each IMU calibration column.
    pub(super) j_calibration: Vec<(usize, Vec15F64)>,
}

/// How the edges are integrated and weighted.
pub(super) struct ImuModel<'p, 'a> {
    problem: &'p Problem<'a>,
    kernel: OfflineImuKernel,
    reintegrate: bool,
}

impl<'p, 'a> ImuModel<'p, 'a> {
    pub(super) fn new(problem: &'p Problem<'a>, kernel: OfflineImuKernel, layout: &Layout) -> Self {
        Self {
            problem,
            kernel,
            reintegrate: layout.reintegrates(),
        }
    }

    /// Whether each state gets its own integrations, or the start's are kept
    /// and corrected to first order in the bias (`FullInertialBA` after
    /// `SetNewBias`, `Optimizer.cc:491`).
    pub(super) fn reintegrates(&self) -> bool {
        self.reintegrate
    }

    fn chi2_threshold(&self) -> f64 {
        match self.kernel {
            OfflineImuKernel::None => 0.0,
            OfflineImuKernel::Huber => IMU_HUBER_CHI2,
        }
    }

    /// `edge` integrated from the raw stream at `bias`, with the gyro
    /// correction and the clock offset `td`: the camera interval
    /// `[t0, t1]` is `[t0 - td, t1 - td]` on the IMU clock.
    fn integrate(
        &self,
        edge: &ImuEdge,
        bias: ImuBias,
        gyro: &GyroIntrinsics,
        td: f64,
    ) -> PreintegratedImu {
        let samples = &self.problem.imu;
        let (t0, t1) = (edge.t0 - td, edge.t1 - td);
        let lo = samples.partition_point(|s| s.t < t0);
        let hi = samples.partition_point(|s| s.t <= t1).max(lo);
        let measurements: Vec<ImuMeasurement> = samples[lo..hi]
            .iter()
            .map(|s| ImuMeasurement {
                timestamp: s.t,
                gyro: gyro.apply(Vec3F64::from_array(s.gyro)),
                accel: Vec3F64::from_array(s.accel),
            })
            .collect();
        PreintegratedImu::from_measurements(bias, self.problem.noise, &measurements, t0, t1)
    }

    /// Every edge integrated at `state`'s biases and calibration.
    pub(super) fn integrate_all(
        &self,
        edges: &[ImuEdge],
        state: &State,
        layout: &Layout,
    ) -> Vec<PreintegratedImu> {
        let gyro = state.calibration.gyro(layout);
        let td = state.calibration.clock_offset(layout);
        edges
            .iter()
            .map(|edge| self.integrate(edge, state.keyframes[edge.from].bias, &gyro, td))
            .collect()
    }

    fn residual(
        &self,
        edge: &ImuEdge,
        pim: &PreintegratedImu,
        state: &State,
    ) -> (Vec15F64, Mat15F64, Mat15F64) {
        let nav = |k: usize| {
            let kf = &state.keyframes[k];
            NavState {
                pose_cw: &kf.pose_bw,
                velocity: kf.velocity,
                bias: kf.bias,
            }
        };
        // `T_bw` is the state, so no extrinsic enters the residual.
        residual_and_jacobians(
            nav(edge.from),
            nav(edge.to),
            pim,
            &self.problem.gravity,
            None,
        )
    }

    pub(super) fn cost(&self, edge: &ImuEdge, pim: &PreintegratedImu, state: &State) -> f64 {
        let (residual, _, _) = self.residual(edge, pim, state);
        weight_and_cost(
            &residual,
            &information(pim, 1.0).omega,
            self.chi2_threshold(),
        )
        .1
    }

    pub(super) fn linearize(
        &self,
        edge: &ImuEdge,
        pim: &PreintegratedImu,
        state: &State,
        layout: &Layout,
    ) -> ImuLinearization {
        let (residual, j_from, j_to) = self.residual(edge, pim, state);
        let mut omega = information(pim, 1.0).omega;
        let (w, _) = weight_and_cost(&residual, &omega, self.chi2_threshold());
        for r in 0..9 {
            for c in 0..9 {
                omega[(r, c)] *= w;
            }
        }
        ImuLinearization {
            residual,
            j_from,
            j_to,
            omega,
            j_calibration: self.calibration_jacobian(edge, state, layout),
        }
    }

    /// Central differences of the re-integrated residual along the gyro and
    /// clock columns; an analytic `∂Δ/∂S` would have to enter
    /// `PreintegratedImu::integrate`, every online user's hot path.
    fn calibration_jacobian(
        &self,
        edge: &ImuEdge,
        state: &State,
        layout: &Layout,
    ) -> Vec<(usize, Vec15F64)> {
        let mut columns: Vec<(usize, f64)> = Vec::new();
        if let Some(at) = layout.gyro() {
            columns.extend((at..at + 6).map(|c| (c, GYRO_STEP)));
        }
        if let Some(at) = layout.clock() {
            columns.push((at, CLOCK_STEP_SEC));
        }
        columns
            .into_iter()
            .map(|(column, h)| {
                let at = |step: f64| {
                    let mut calibration = state.calibration.clone();
                    calibration.values[column] += step;
                    let pim = self.integrate(
                        edge,
                        state.keyframes[edge.from].bias,
                        &calibration.gyro(layout),
                        calibration.clock_offset(layout),
                    );
                    self.residual(edge, &pim, state).0
                };
                (column, (at(h) - at(-h)) / (2.0 * h))
            })
            .collect()
    }

    /// Mean whitened rotation residual `√(r_Rᵀ Σ_RR⁻¹ r_R)` over the edges
    /// `keep` selects; NaN without one.
    pub(super) fn mean_rotation_whitened(
        &self,
        edges: &[ImuEdge],
        pims: &[PreintegratedImu],
        state: &State,
        keep: impl Fn(&ImuEdge) -> bool,
    ) -> f64 {
        let values: Vec<f64> = edges
            .iter()
            .zip(pims)
            .filter(|(edge, _)| keep(edge))
            .filter_map(|(edge, pim)| {
                let residual = self.residual(edge, pim, state).0;
                let r = residual.fixed_rows::<3>(0).into_owned();
                let rotation_information = pim
                    .covariance
                    .fixed_view::<3, 3>(0, 0)
                    .into_owned()
                    .try_inverse()?;
                Some(r.dot(&(rotation_information * r)).max(0.0).sqrt())
            })
            .collect();
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
}
