//! Offline global visual-inertial BA over a recorded session
//! ([`super::session::SessionDump`]), with optional calibration states:
//! camera drift, gyro intrinsics and clock. Problem shape after ORB-SLAM3's
//! `FullInertialBA` (`Optimizer.cc:392-811`); landmarks are eliminated by the
//! Schur complement and the reduced system is a faer sparse Cholesky in f64.

mod imu_edges;
mod problem;
mod reduced;
mod report;

use std::collections::HashMap;
use std::time::Instant;

use kornia_3d::pose::Pose3d;
use kornia_sensors::imu::{GyroIntrinsics, PreintegratedImu, SampleInterpolation};

use super::session::SessionDump;
use super::visual_information::VisualInformation;
use crate::mapping::calibration_states::{Layout, prior_rows};
use crate::system::{ConfigError, StereoDepthSigma};
use imu_edges::{ImuEdge, ImuModel, build_edges};
use problem::{GaugeAnchor, Problem, State};
use reduced::{Linearization, Pattern};

/// Levenberg-Marquardt damping: start, bounds and factor.
const LAMBDA_START: f64 = 1e-3;
const LAMBDA_MIN: f64 = 1e-9;
const LAMBDA_MAX: f64 = 1e10;
const LAMBDA_FACTOR: f64 = 10.0;

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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct OfflineBaReport {
    /// Tracked frames the output re-anchors.
    pub frames: usize,
    /// Accepted LM steps.
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    /// Observations dropped at `outlier_px`, and at `tight_outlier_px`.
    pub dropped_outlier: usize,
    pub dropped_tight_outlier: usize,
    /// Connected components: keyframes joined by an IMU edge or a landmark.
    pub components: usize,
    /// IMU edges across a segment boundary.
    pub bridged: usize,
    /// Longest gap (s) at a segment boundary.
    pub gap_max_s: f64,
    pub covisible_median: usize,
    pub nnz_reduced: usize,
    pub nnz_factor: usize,
    /// Wall time of each LM iteration, damping trials included (ms).
    pub iteration_ms: Vec<f64>,
    /// Mean whitened IMU rotation residual at the start and the end, and of
    /// the bridged edges at the end.
    pub imu_rot_whitened0: f64,
    pub imu_rot_whitened: f64,
    pub bridge_rot_whitened: f64,
    /// Per camera, per slice: rotation offset (°) and focal scale (%).
    pub camera_rotation_deg: Vec<Vec<f64>>,
    pub camera_focal_pct: Vec<Vec<f64>>,
    pub gyro_scale_pct: Option<[f64; 3]>,
    /// Shear xy, xz, yz (mrad).
    pub gyro_shear_mrad: Option<[f64; 3]>,
    /// The residual clock offset on top of the one the dump already applied.
    pub td_ms: Option<f64>,
    /// The offset to apply next: the dump's `imu_time_offset_ns` plus `td_ms`,
    /// both with the sign of an offset added to the IMU stamps.
    pub td_total_ms: Option<f64>,
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

/// Solves one global VI-BA over `session`, weighting each observation by its
/// octave's pixel sigma (`visual`) and its stereo depth by
/// `stereo_depth_sigma`, and integrating the IMU as the run did
/// (`interpolation`). Unlike the online VI-BA it applies no `fisheye_weight`,
/// and it reads no keyframe's `anchor`, `evicted` or `inserted_while_lost`
/// flag. Runs on one thread, so the result depends on the dump alone: faer's
/// global parallelism is sequential during the solve and restored after it.
pub fn offline_visual_inertial_ba(
    session: &SessionDump,
    visual: &VisualInformation,
    stereo_depth_sigma: StereoDepthSigma,
    interpolation: SampleInterpolation,
    config: &OfflineBaConfig,
) -> Result<OfflineBaResult, OfflineBaError> {
    config.validate()?;
    let _sequential = SequentialFaer::enter();
    let layout = Layout::new(&config.calibration, session.rig.cameras.len());
    let (problem, start) = Problem::new(session, *visual, stereo_depth_sigma, &layout)?;
    let (edges, gap_max_s) = build_edges(&problem, config.bridge_segments);
    let imu = ImuModel::new(&problem, config.imu_kernel, &layout, interpolation);
    let slots = keyframe_slots(session);
    let mut report = OfflineBaReport {
        frames: session
            .frames
            .iter()
            .filter(|f| slots.contains_key(&f.reference_kf))
            .count(),
        bridged: edges.iter().filter(|e| e.bridged).count(),
        gap_max_s,
        ..OfflineBaReport::default()
    };
    let solves = config.max_iterations > 0 && !start.keyframes.is_empty();
    if solves {
        report.dropped_outlier = problem.drop_outliers(&start, &layout, config.outlier_px);
    }
    let imu_pairs: Vec<(usize, usize)> = edges.iter().map(|e| (e.from, e.to)).collect();
    let observers = problem.observers();
    let gauge = problem.gauge_anchors(&imu_pairs, &observers, &start);
    report.components = gauge.len();

    let solve = Solve {
        problem: &problem,
        imu: &imu,
        layout: &layout,
        edges: &edges,
        gauge: &gauge,
        config,
    };
    let pims = imu.integrate_all(&edges, &start, &layout);
    report.imu_rot_whitened0 = imu.mean_rotation_whitened(&edges, &pims, &start, |_| true);
    report.initial_cost = solve.cost(&start, &pims);
    report.final_cost = report.initial_cost;
    let (state, pims) = if solves {
        let pattern = Pattern::new(start.keyframes.len(), &imu_pairs, &observers)?;
        report.nnz_reduced = pattern.nnz();
        report.nnz_factor = pattern.nnz_factor();
        report.covisible_median = pattern.covisible_median();
        solve.run(&pattern, start, pims, &mut report)?
    } else {
        (start, pims)
    };
    report.imu_rot_whitened = imu.mean_rotation_whitened(&edges, &pims, &state, |_| true);
    report.bridge_rot_whitened = imu.mean_rotation_whitened(&edges, &pims, &state, |e| e.bridged);
    let calibration = state.calibration.estimate(&layout);
    report.record_calibration(&calibration);
    report.td_total_ms = report
        .td_ms
        .map(|td| session.rig.imu_time_offset_ns as f64 * 1e-6 + td);
    Ok(OfflineBaResult {
        body_poses: problem.dump_body_poses(session, &state),
        calibration,
        report,
    })
}

/// Each tracked frame of `session` re-anchored on its keyframe's solved
/// `T_bw` (`body_poses`, in dump order): `T_bw(frame) = T_bc ·
/// T_frame_from_kf · T_cb · T_bw(kf)`, with the factory extrinsic the online
/// relative motion was measured with. In recording order; a frame whose
/// keyframe has no pose is left out.
pub fn reanchor_frames(session: &SessionDump, body_poses: &[Pose3d]) -> Vec<(f64, Pose3d)> {
    let body_from_primary = session.rig.camera_to_body.to_pose();
    let primary_from_body = body_from_primary.inverse();
    let slots = keyframe_slots(session);
    session
        .frames
        .iter()
        .filter_map(|frame| {
            let pose_bw = body_poses.get(*slots.get(&frame.reference_kf)?)?;
            let body_from_kf_body = body_from_primary
                .compose(&frame.frame_from_kf.to_pose())
                .compose(&primary_from_body);
            Some((frame.timestamp_sec, body_from_kf_body.compose(pose_bw)))
        })
        .collect()
}

/// Holds faer's global parallelism at `Par::Seq` and restores the previous
/// value on drop, on the error paths too.
struct SequentialFaer(faer::Par);

impl SequentialFaer {
    fn enter() -> Self {
        let previous = faer::get_global_parallelism();
        faer::set_global_parallelism(faer::Par::Seq);
        Self(previous)
    }
}

impl Drop for SequentialFaer {
    fn drop(&mut self) {
        faer::set_global_parallelism(self.0);
    }
}

/// Dump position of every keyframe by its id.
fn keyframe_slots(session: &SessionDump) -> HashMap<u64, usize> {
    session
        .keyframes
        .iter()
        .enumerate()
        .map(|(slot, kf)| (kf.kf_idx, slot))
        .collect()
}

/// What one solve reads and never changes.
struct Solve<'s, 'a> {
    problem: &'s Problem<'a>,
    imu: &'s ImuModel<'s, 'a>,
    layout: &'s Layout,
    edges: &'s [ImuEdge],
    gauge: &'s [GaugeAnchor],
    config: &'s OfflineBaConfig,
}

/// An accepted step: the new state, its integrations when each state gets
/// its own, and its cost.
type Accepted = (State, Option<Vec<PreintegratedImu>>, f64);

impl Solve<'_, '_> {
    fn cost(&self, state: &State, pims: &[PreintegratedImu]) -> f64 {
        let visual: f64 = (0..self.problem.observations.len())
            .filter(|&o| self.problem.observations[o].active.get())
            .map(|o| self.problem.observation_cost(o, state, self.layout))
            .sum();
        let inertial: f64 = self
            .edges
            .iter()
            .zip(pims)
            .map(|(edge, pim)| self.imu.cost(edge, pim, state))
            .sum();
        let priors: f64 = prior_rows(self.layout, &self.config.calibration, &state.calibration)
            .iter()
            .map(|row| 0.5 * row.residual * row.residual)
            .sum();
        visual + inertial + priors + self.problem.gauge_cost(self.gauge, state)
    }

    /// Levenberg-Marquardt from `state`: observations past `outlier_px` were
    /// dropped before, those past `tight_outlier_px` go from
    /// `tighten_from_iteration` on (visloc's two stages), or once the solve
    /// stops before then: the tight stage always runs (D8).
    fn run(
        &self,
        pattern: &Pattern,
        mut state: State,
        mut pims: Vec<PreintegratedImu>,
        report: &mut OfflineBaReport,
    ) -> Result<(State, Vec<PreintegratedImu>), OfflineBaError> {
        let config = self.config;
        let mut cost = report.initial_cost;
        let mut lambda = LAMBDA_START;
        let mut tightened = false;
        let mut tighten = |state: &State, pims: &[PreintegratedImu], cost: &mut f64| {
            let dropped = self
                .problem
                .drop_outliers(state, self.layout, config.tight_outlier_px);
            if dropped > 0 {
                report.dropped_tight_outlier += dropped;
                *cost = self.cost(state, pims);
            }
        };
        for iteration in 1..=config.max_iterations {
            let started = Instant::now();
            if iteration >= config.tighten_from_iteration {
                tighten(&state, &pims, &mut cost);
                tightened = true;
            }
            let imu: Vec<_> = self
                .edges
                .iter()
                .zip(&pims)
                .map(|(edge, pim)| self.imu.linearize(edge, pim, &state, self.layout))
                .collect();
            let priors = prior_rows(self.layout, &config.calibration, &state.calibration);
            let linearization = Linearization {
                problem: self.problem,
                state: &state,
                layout: self.layout,
                edges: self.edges,
                imu: &imu,
                gauge: self.gauge,
                priors: &priors,
                landmark_damping_floor: config.landmark_damping_floor,
            };
            let step = self.damped_step(&linearization, pattern, &pims, cost, &mut lambda);
            report
                .iteration_ms
                .push(started.elapsed().as_secs_f64() * 1e3);
            let converged = match step {
                Ok(Some((candidate, candidate_pims, candidate_cost))) => {
                    report.iterations = iteration;
                    let decrease = (cost - candidate_cost) / cost.max(f64::MIN_POSITIVE);
                    state = candidate;
                    if let Some(candidate_pims) = candidate_pims {
                        pims = candidate_pims;
                    }
                    cost = candidate_cost;
                    decrease < config.relative_cost_tolerance
                }
                Ok(None) => true,
                Err(error) if iteration == 1 => return Err(error),
                Err(_) => break,
            };
            if converged {
                if tightened {
                    break;
                }
                tighten(&state, &pims, &mut cost);
                tightened = true;
                lambda = LAMBDA_START;
            }
        }
        report.final_cost = cost;
        Ok((state, pims))
    }

    /// Raises the damping until a step lowers the cost; `Ok(None)` when none
    /// did, the error when no damping gave a system that factorised.
    fn damped_step(
        &self,
        linearization: &Linearization<'_>,
        pattern: &Pattern,
        pims: &[PreintegratedImu],
        cost: f64,
        lambda: &mut f64,
    ) -> Result<Option<Accepted>, OfflineBaError> {
        let mut factorised = false;
        let mut last_error = None;
        while *lambda <= LAMBDA_MAX {
            match linearization.assemble(pattern, *lambda).solve(pattern) {
                Ok((dx, dc)) => {
                    factorised = true;
                    let dl = linearization.landmark_steps(*lambda, &dx, &dc);
                    let candidate = linearization.state.retracted(&dx, &dl, &dc);
                    let candidate_pims = self
                        .imu
                        .reintegrates()
                        .then(|| self.imu.integrate_all(self.edges, &candidate, self.layout));
                    let candidate_cost =
                        self.cost(&candidate, candidate_pims.as_deref().unwrap_or(pims));
                    // A NaN cost is refused too.
                    if candidate_cost < cost {
                        *lambda = (*lambda / LAMBDA_FACTOR).max(LAMBDA_MIN);
                        return Ok(Some((candidate, candidate_pims, candidate_cost)));
                    }
                }
                Err(error) => last_error = Some(error),
            }
            *lambda *= LAMBDA_FACTOR;
        }
        match (factorised, last_error) {
            (false, Some(error)) => Err(error),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests;
