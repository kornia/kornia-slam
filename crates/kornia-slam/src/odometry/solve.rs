//! Levenberg-Marquardt over the window, landmarks eliminated by their 3×3
//! Schur complement, the reduced system factored by a dense Cholesky on the
//! calling thread (Basalt `sqrt_keypoint_vio.cpp:1012-1396`).

use faer::{Mat, Par};

use super::landmarks::{LandmarkId, LandmarkStore};
use super::linearize::{Linearization, VarKind, cost, linearize};
use super::state::{FrameId, POSE_DOF, PoseWithLin, STATE_DOF, StateWithLin};
use super::window::Window;
use crate::mapping::dense_llt::DenseLlt;
use kornia_algebra::SVecF64;
use kornia_factors::reprojection::AnchoredLandmark;

/// How one solve went.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SolveReport {
    /// Steps tried, accepted or rejected (at most `lm_iterations + 1`).
    pub attempts: usize,
    pub rejected: usize,
    pub converged: bool,
    /// The damped system would not factor: the states keep the prediction.
    pub failed: bool,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub landmarks_fixed: usize,
    /// Visual rows per camera and their Huber cost, at the final values.
    pub rows: [usize; 2],
    pub vision_cost: f64,
}

/// `λ · max(1/3, 1 − (2q − 1)³)`, at least `lambda_min`, after a step of
/// quality `q` is accepted (`:1350-1355`).
pub(crate) fn lambda_after_accept(lambda: f64, quality: f64, lambda_min: f64) -> f64 {
    (lambda * (1.0f64 / 3.0).max(1.0 - (2.0 * quality - 1.0).powi(3))).max(lambda_min)
}

/// `(λ · ν, 2ν)` after a reject (`:1382-1396`).
pub(crate) fn lambda_after_reject(lambda: f64, vee: f64) -> (f64, f64) {
    (lambda * vee, 2.0 * vee)
}

/// Buffers a solve reuses from frame to frame.
#[derive(Default)]
pub(crate) struct SolveScratch {
    lin: Linearization,
    damped: Option<Mat<f64>>,
    llt: Option<(usize, DenseLlt)>,
    increment: Vec<f64>,
    backup: Snapshot,
    start: Snapshot,
}

/// A copy of everything a step moves.
#[derive(Clone, Default)]
struct Snapshot {
    states: Vec<(FrameId, StateWithLin)>,
    poses: Vec<(FrameId, PoseWithLin)>,
    landmarks: Vec<(LandmarkId, AnchoredLandmark)>,
}

impl Snapshot {
    fn save(&mut self, window: &Window) {
        self.states.clear();
        self.states
            .extend(window.states.iter().map(|(k, v)| (*k, *v)));
        self.poses.clear();
        self.poses
            .extend(window.poses.iter().map(|(k, v)| (*k, *v)));
        self.landmarks.clear();
        self.landmarks
            .extend(window.landmarks.iter().map(|(id, lm)| (*id, lm.value)));
    }

    fn restore(&self, window: &mut Window) {
        for (id, state) in &self.states {
            window.states.insert(*id, *state);
        }
        for (id, pose) in &self.poses {
            window.poses.insert(*id, *pose);
        }
        for (id, value) in &self.landmarks {
            if let Some(lm) = window.landmarks.get_mut(*id) {
                lm.value = *value;
            }
        }
    }
}

/// Damping `max(λ · diag(H), λ_min)` on the reduced system, undamped
/// landmarks (`:1212-1214`).
fn damp(h: &Mat<f64>, lambda: f64, lambda_min: f64, out: &mut Mat<f64>) {
    out.copy_from(h);
    for i in 0..h.nrows() {
        out[(i, i)] += (lambda * h[(i, i)]).max(lambda_min);
    }
}

/// Applies the pose step `inc` (in `lin.order`) and each landmark's
/// back-substituted step.
fn apply_step(window: &mut Window, lin: &Linearization, inc: &[f64]) {
    for entry in lin.order.entries() {
        let o = entry.offset;
        match entry.kind {
            VarKind::State => {
                if let Some(state) = window.states.get_mut(&entry.id) {
                    state.apply_increment(&SVecF64::<STATE_DOF>::from_fn(|r, _| inc[o + r]));
                }
            }
            VarKind::Pose => {
                if let Some(pose) = window.poses.get_mut(&entry.id) {
                    pose.apply_increment(&SVecF64::<POSE_DOF>::from_fn(|r, _| inc[o + r]));
                }
            }
        }
    }
    for (id, block) in &lin.blocks[..lin.used_blocks] {
        apply_landmark_step(&mut window.landmarks, *id, &block.back_substitute(inc));
    }
}

fn apply_landmark_step(store: &mut LandmarkStore, id: LandmarkId, step: &SVecF64<3>) {
    if let Some(lm) = store.get_mut(id) {
        lm.value.bearing[0] += step[0];
        lm.value.bearing[1] += step[1];
        lm.value.inverse_distance += step[2];
    }
}

/// The damped step `−H_d⁻¹ b` into `scratch.increment`, raising λ as on a
/// reject up to 3 times when the factor fails or the step is not finite
/// (`:1207-1227`); `false` when it never succeeds.
fn damped_step(
    scratch: &mut SolveScratch,
    lambda: &mut f64,
    vee: &mut f64,
    lambda_min: f64,
) -> bool {
    let n = scratch.lin.system.dim();
    let damped = scratch.damped.get_or_insert_with(|| Mat::zeros(n, n));
    if damped.nrows() != n {
        *damped = Mat::zeros(n, n);
    }
    if scratch.llt.as_ref().is_none_or(|(dim, _)| *dim != n) {
        scratch.llt = Some((n, DenseLlt::new(n, Par::Seq)));
    }
    let Some((_, llt)) = scratch.llt.as_mut() else {
        return false;
    };
    for _ in 0..=3 {
        damp(&scratch.lin.system.h, *lambda, lambda_min, damped);
        if llt.factor(damped).is_ok() {
            let x = llt.solve_column(&scratch.lin.system.b);
            if x.iter().all(|v| v.is_finite()) {
                scratch.increment.clear();
                scratch.increment.extend(x.iter().map(|v| -v));
                return true;
            }
        }
        (*lambda, *vee) = lambda_after_reject(*lambda, *vee);
    }
    false
}

/// `l = −bᵀδ − ½ δᵀHδ + ½ Σ b_lᵀ H_ll⁻¹ b_l` with the undamped `H`
/// (`imu_block.hpp:210-222`).
fn model_decrease(lin: &Linearization, inc: &[f64]) -> f64 {
    let n = inc.len();
    let h = &lin.system.h;
    let quadratic: f64 = (0..n)
        .map(|r| inc[r] * (0..n).map(|c| h[(r, c)] * inc[c]).sum::<f64>())
        .sum();
    let linear: f64 = (0..n).map(|r| lin.system.b[r] * inc[r]).sum();
    -linear - 0.5 * quadratic + lin.landmark_model
}

/// The LM loop of the design card section 5.
pub(crate) fn optimize(window: &mut Window, scratch: &mut SolveScratch) -> SolveReport {
    let tuning = window.tuning;
    let mut report = SolveReport::default();
    let mut lambda = tuning.lm_lambda_initial;
    let mut vee = 2.0;
    scratch.start.save(window);
    if linearize(window, &mut scratch.lin).is_err() {
        report.failed = true;
        return report;
    }
    let mut current = scratch.lin.cost.total();
    report.rows = scratch.lin.cost.rows;
    report.vision_cost = scratch.lin.cost.vision;
    report.initial_cost = current;
    report.landmarks_fixed = scratch.lin.fixed_landmarks;
    let mut iteration = 0;
    while iteration <= tuning.lm_iterations {
        if !damped_step(scratch, &mut lambda, &mut vee, tuning.lm_lambda_min) {
            scratch.start.restore(window);
            report.failed = true;
            report.final_cost = report.initial_cost;
            return report;
        }
        let max_step = scratch.increment.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let decrease = model_decrease(&scratch.lin, &scratch.increment);
        scratch.backup.save(window);
        let increment = std::mem::take(&mut scratch.increment);
        apply_step(window, &scratch.lin, &increment);
        scratch.increment = increment;
        let evaluated = cost(window);
        let next = evaluated.total();
        let actual = current - next;
        let quality = actual / decrease;
        report.attempts += 1;
        iteration += 1;
        if decrease > 0.0 && quality > 0.0 && next.is_finite() {
            lambda = lambda_after_accept(lambda, quality, tuning.lm_lambda_min);
            vee = 2.0;
            current = next;
            report.rows = evaluated.rows;
            report.vision_cost = evaluated.vision;
            if (actual > 0.0 && actual < 1e-6) || max_step < 1e-4 {
                report.converged = true;
                break;
            }
            if iteration > tuning.lm_iterations {
                break;
            }
            if linearize(window, &mut scratch.lin).is_err() {
                break;
            }
            report.landmarks_fixed = report.landmarks_fixed.max(scratch.lin.fixed_landmarks);
        } else {
            scratch.backup.restore(window);
            report.rejected += 1;
            (lambda, vee) = lambda_after_reject(lambda, vee);
            if lambda > tuning.lm_lambda_max || max_step < 1e-4 {
                report.converged = max_step < 1e-4;
                break;
            }
        }
    }
    report.final_cost = current;
    report
}
