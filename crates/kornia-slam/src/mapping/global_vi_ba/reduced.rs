//! The reduced system of the offline solve. Landmarks are eliminated by the
//! Schur complement (ORB-SLAM3 marginalises them, `Optimizer.cc:606`); the
//! keyframe states form `M = A - B C⁻¹ Bᵀ`, a faer sparse lower triangle
//! whose block pattern and symbolic Cholesky are built once and whose values
//! are refilled in place each iteration. The dense calibration border is
//! solved by a second Schur complement on that factor, so the factor never
//! sees a dense row.

// Fixed-size block loops read as the math they implement.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

use faer::prelude::Solve;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, SymmetricOrdering, factorize_symbolic_cholesky,
};
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use faer::{Mat, Side};
use kornia_algebra::{Mat15F64, SMatF64, SVecF64, Vec3F64};

use super::OfflineBaError;
use super::calibration::{Layout, PriorRow};
use super::imu_edges::{ImuEdge, ImuLinearization};
use super::problem::{GaugeAnchor, ObservationRows, Problem, State};

/// DoF of a keyframe state `[ρ ω v b_g b_a]`, and of its pose part.
const STATE_DOF: usize = 15;
const POSE_DOF: usize = 6;
/// Absolute floor of a damped diagonal entry, as the online VI-BA's.
const MIN_DAMPING_FLOOR: f64 = 1e-6;

/// An off-diagonal block of one block column: a full 15×15 IMU block, or a
/// 6×6 pose block of two keyframes that share only landmarks. `prefix_*`
/// are the rows of the earlier blocks of the column in a pose column
/// (`< 6`), and in a column past the pose.
#[derive(Debug, Clone, Copy)]
struct OffBlock {
    row: usize,
    full: bool,
    prefix_pose: usize,
    prefix_rest: usize,
}

/// The fixed sparsity of `M`: per block column, the diagonal block's lower
/// triangle, then the blocks below it by block row.
pub(super) struct Pattern {
    col_ptr: Vec<usize>,
    columns: Vec<Vec<OffBlock>>,
    symbolic: SymbolicSparseColMat<usize>,
    llt: SymbolicLlt<usize>,
    nnz_factor: usize,
}

impl Pattern {
    /// The pattern of `keyframes` states joined by `imu_pairs` and by the
    /// landmarks that `observers` lists (each one's keyframes, ascending).
    pub(super) fn new(
        keyframes: usize,
        imu_pairs: &[(usize, usize)],
        observers: &[Vec<usize>],
    ) -> Result<Self, OfflineBaError> {
        let mut blocks: Vec<BTreeMap<usize, bool>> = vec![BTreeMap::new(); keyframes];
        for &(a, b) in imu_pairs {
            if a != b {
                blocks[a.min(b)].insert(a.max(b), true);
            }
        }
        for kfs in observers {
            for (i, &hi) in kfs.iter().enumerate() {
                for &lo in &kfs[..i] {
                    blocks[lo].entry(hi).or_insert(false);
                }
            }
        }
        let columns: Vec<Vec<OffBlock>> = blocks
            .iter()
            .map(|column| {
                let (mut prefix_pose, mut prefix_rest) = (0, 0);
                column
                    .iter()
                    .map(|(&row, &full)| {
                        let block = OffBlock {
                            row,
                            full,
                            prefix_pose,
                            prefix_rest,
                        };
                        prefix_pose += if full { STATE_DOF } else { POSE_DOF };
                        prefix_rest += if full { STATE_DOF } else { 0 };
                        block
                    })
                    .collect()
            })
            .collect();

        let n = keyframes * STATE_DOF;
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for (b, column) in columns.iter().enumerate() {
            for lc in 0..STATE_DOF {
                row_idx.extend((lc..STATE_DOF).map(|lr| b * STATE_DOF + lr));
                for block in column {
                    row_idx.extend((0..block_rows(block, lc)).map(|lr| block.row * STATE_DOF + lr));
                }
                col_ptr.push(row_idx.len());
            }
        }
        let symbolic = SymbolicSparseColMat::new_checked(n, n, col_ptr.clone(), None, row_idx);
        let llt = SymbolicLlt::try_new(symbolic.as_ref(), Side::Lower)
            .map_err(|e| OfflineBaError::CholeskyFailed(format!("{e:?}")))?;
        // `SymbolicLlt` keeps the factor's size private; a second analysis reads it.
        let nnz_factor = factorize_symbolic_cholesky(
            symbolic.as_ref(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_or(0, |s| s.len_val());
        Ok(Self {
            col_ptr,
            columns,
            symbolic,
            llt,
            nnz_factor,
        })
    }

    pub(super) fn nnz(&self) -> usize {
        self.col_ptr.last().copied().unwrap_or(0)
    }

    pub(super) fn nnz_factor(&self) -> usize {
        self.nnz_factor
    }

    /// Median count of later keyframes each keyframe shares a block with.
    pub(super) fn covisible_median(&self) -> usize {
        let mut counts: Vec<usize> = self.columns.iter().map(Vec::len).collect();
        counts.sort_unstable();
        counts.get(counts.len() / 2).copied().unwrap_or(0)
    }

    fn block(&self, row: usize, col: usize) -> Option<&OffBlock> {
        let column = &self.columns[col];
        column
            .binary_search_by_key(&row, |b| b.row)
            .ok()
            .map(|at| &column[at])
    }

    fn diagonal_slot(&self, k: usize, lr: usize, lc: usize) -> usize {
        self.col_ptr[k * STATE_DOF + lc] + (lr - lc)
    }

    fn off_slot(&self, col: usize, block: &OffBlock, lr: usize, lc: usize) -> usize {
        let prefix = if lc < POSE_DOF {
            block.prefix_pose
        } else {
            block.prefix_rest
        };
        self.col_ptr[col * STATE_DOF + lc] + (STATE_DOF - lc) + prefix + lr
    }

    /// Adds the symmetric `h` to keyframe `k`'s diagonal block, its leading
    /// `D×D` part when `D` is 6.
    fn add_diagonal<const D: usize>(&self, values: &mut [f64], k: usize, h: &SMatF64<D, D>) {
        for lc in 0..D {
            for lr in lc..D {
                values[self.diagonal_slot(k, lr, lc)] += h[(lr, lc)];
            }
        }
    }

    /// Adds `h = ∂²/∂x_row ∂x_col` (leading `D×D` part for `D` = 6) to the
    /// block of keyframes `row` and `col`.
    fn add_block<const D: usize>(
        &self,
        values: &mut [f64],
        row: usize,
        col: usize,
        h: &SMatF64<D, D>,
    ) {
        if row == col {
            self.add_diagonal(values, row, h);
            return;
        }
        if row < col {
            self.add_block(values, col, row, &h.transpose());
            return;
        }
        let Some(block) = self.block(row, col).copied() else {
            return;
        };
        if D > POSE_DOF && !block.full {
            return;
        }
        for lc in 0..D {
            for lr in 0..D {
                values[self.off_slot(col, &block, lr, lc)] += h[(lr, lc)];
            }
        }
    }
}

/// Rows of `block` in local column `lc` of its block column.
fn block_rows(block: &OffBlock, lc: usize) -> usize {
    match (block.full, lc < POSE_DOF) {
        (true, _) => STATE_DOF,
        (false, true) => POSE_DOF,
        (false, false) => 0,
    }
}

/// Everything one landmark contributes, eliminated: its damped `C⁻¹`, its
/// gradient part `b_l = -J_lᵀ r`, and its couplings `W = Jᵀ J_l` to the
/// observing keyframes' poses and to calibration columns.
struct LandmarkBlock {
    c_inv: SMatF64<3, 3>,
    rhs: SVecF64<3>,
    keyframes: Vec<(usize, SMatF64<6, 3>)>,
    calibration: Vec<(usize, SVecF64<3>)>,
}

/// Inputs of one assembly that stay fixed over an iteration's damping trials.
pub(super) struct Linearization<'s> {
    pub(super) problem: &'s Problem<'s>,
    pub(super) state: &'s State,
    pub(super) layout: &'s Layout,
    pub(super) edges: &'s [ImuEdge],
    pub(super) imu: &'s [ImuLinearization],
    pub(super) gauge: &'s [GaugeAnchor],
    pub(super) priors: &'s [PriorRow],
    pub(super) landmark_damping_floor: f64,
}

impl Linearization<'_> {
    /// The landmark's block at damping `lambda`; `on_rows` sees each active
    /// observation's rows. `None` when no active observation remains or its
    /// damped `C` does not invert, which holds the landmark fixed.
    fn landmark(
        &self,
        landmark: usize,
        lambda: f64,
        mut on_rows: impl FnMut(usize, &ObservationRows),
    ) -> Option<LandmarkBlock> {
        let mut c = SMatF64::<3, 3>::zeros();
        let mut rhs = SVecF64::<3>::zeros();
        let mut keyframes: Vec<(usize, SMatF64<6, 3>)> = Vec::new();
        let mut calibration: Vec<(usize, SVecF64<3>)> = Vec::new();
        for &o in &self.problem.by_landmark[landmark] {
            let obs = &self.problem.observations[o];
            if !obs.active.get() {
                continue;
            }
            let rows = self
                .problem
                .linearize_observation(o, self.state, self.layout);
            let w = rows.j_pose.transpose() * rows.j_point;
            match keyframes.iter_mut().find(|(k, _)| *k == obs.keyframe) {
                Some((_, acc)) => *acc += w,
                None => keyframes.push((obs.keyframe, w)),
            }
            c += rows.j_point.transpose() * rows.j_point;
            rhs -= rows.j_point.transpose() * rows.residual;
            for (column, jc) in rows.calibration() {
                let wc = rows.j_point.transpose() * jc;
                match calibration.iter_mut().find(|(at, _)| *at == column) {
                    Some((_, acc)) => *acc += wc,
                    None => calibration.push((column, wc)),
                }
            }
            on_rows(obs.keyframe, &rows);
        }
        if keyframes.is_empty() {
            return None;
        }
        // visloc's landmark damping floor: a fraction of the largest diagonal entry.
        let largest = (0..3).map(|d| c[(d, d)]).fold(0.0, f64::max);
        let floor = (self.landmark_damping_floor * largest).max(MIN_DAMPING_FLOOR);
        for d in 0..3 {
            c[(d, d)] += lambda * c[(d, d)].max(floor);
        }
        let c_inv = c
            .try_inverse()
            .filter(|m| m.iter().all(|v| v.is_finite()))?;
        Some(LandmarkBlock {
            c_inv,
            rhs,
            keyframes,
            calibration,
        })
    }

    /// The reduced system at damping `lambda`.
    pub(super) fn assemble(&self, pattern: &Pattern, lambda: f64) -> System {
        let n = self.state.keyframes.len() * STATE_DOF;
        let nc = self.layout.len;
        let mut sys = System {
            values: vec![0.0; pattern.nnz()],
            rhs: vec![0.0; n],
            border: Mat::zeros(n, nc),
            border_diag: Mat::zeros(nc, nc),
            border_rhs: vec![0.0; nc],
        };
        // Diagonal of the Hessian before elimination: the damping scale.
        let mut direct = vec![0.0; n];
        let mut direct_border = vec![0.0; nc];

        for (edge, lin) in self.edges.iter().zip(self.imu) {
            sys.add_imu_edge(pattern, edge, lin, &mut direct, &mut direct_border);
        }
        for anchor in self.gauge {
            let (k, rows) = self.problem.gauge_rows(anchor, self.state);
            let mut h = SMatF64::<6, 6>::zeros();
            for (r, j) in rows {
                let j = SVecF64::<6>::from_column_slice(&j);
                h += j * j.transpose();
                for d in 0..POSE_DOF {
                    sys.rhs[k * STATE_DOF + d] -= j[d] * r;
                    direct[k * STATE_DOF + d] += j[d] * j[d];
                }
            }
            pattern.add_diagonal(&mut sys.values, k, &h);
        }
        for row in self.priors {
            for &(a, wa) in &row.terms {
                sys.border_rhs[a] -= wa * row.residual;
                for &(b, wb) in &row.terms {
                    sys.border_diag[(a, b)] += wa * wb;
                }
                direct_border[a] += wa * wa;
            }
        }
        for landmark in 0..self.problem.by_landmark.len() {
            let block = self.landmark(landmark, lambda, |k, rows| {
                sys.add_observation(pattern, k, rows, &mut direct, &mut direct_border);
            });
            if let Some(block) = block {
                sys.eliminate(pattern, &block);
            }
        }

        for (col, d) in direct.iter().enumerate() {
            sys.values[pattern.col_ptr[col]] += lambda * d.max(MIN_DAMPING_FLOOR);
        }
        for (col, d) in direct_border.iter().enumerate() {
            sys.border_diag[(col, col)] += lambda * d.max(MIN_DAMPING_FLOOR);
        }
        sys
    }

    /// Landmark steps from the keyframe and calibration steps, at the
    /// damping the system was assembled with.
    pub(super) fn landmark_steps(&self, lambda: f64, dx: &[f64], dc: &[f64]) -> Vec<Vec3F64> {
        (0..self.problem.by_landmark.len())
            .map(|landmark| {
                let Some(block) = self.landmark(landmark, lambda, |_, _| {}) else {
                    return Vec3F64::ZERO;
                };
                let mut rhs = block.rhs;
                for (k, w) in &block.keyframes {
                    let start = k * STATE_DOF;
                    let d_pose = SVecF64::<6>::from_column_slice(&dx[start..start + POSE_DOF]);
                    rhs -= w.transpose() * d_pose;
                }
                for (column, wc) in &block.calibration {
                    rhs -= wc * dc[*column];
                }
                let step = block.c_inv * rhs;
                Vec3F64::new(step[0], step[1], step[2])
            })
            .collect()
    }
}

/// `[M E; Eᵀ D] [δx; δc] = [b; b_c]`, `b` the negative gradient.
pub(super) struct System {
    values: Vec<f64>,
    rhs: Vec<f64>,
    border: Mat<f64>,
    border_diag: Mat<f64>,
    border_rhs: Vec<f64>,
}

impl System {
    fn add_imu_edge(
        &mut self,
        pattern: &Pattern,
        edge: &ImuEdge,
        lin: &ImuLinearization,
        direct: &mut [f64],
        direct_border: &mut [f64],
    ) {
        let jt_o_from = lin.j_from.transpose() * lin.omega;
        let jt_o_to = lin.j_to.transpose() * lin.omega;
        let h_from: Mat15F64 = jt_o_from * lin.j_from;
        let h_to: Mat15F64 = jt_o_to * lin.j_to;
        pattern.add_diagonal(&mut self.values, edge.from, &h_from);
        pattern.add_diagonal(&mut self.values, edge.to, &h_to);
        pattern.add_block(
            &mut self.values,
            edge.to,
            edge.from,
            &(jt_o_to * lin.j_from),
        );
        let g_from = jt_o_from * lin.residual;
        let g_to = jt_o_to * lin.residual;
        for d in 0..STATE_DOF {
            self.rhs[edge.from * STATE_DOF + d] -= g_from[d];
            self.rhs[edge.to * STATE_DOF + d] -= g_to[d];
            direct[edge.from * STATE_DOF + d] += h_from[(d, d)];
            direct[edge.to * STATE_DOF + d] += h_to[(d, d)];
        }
        for (a, ja) in &lin.j_calibration {
            let o_ja = lin.omega * ja;
            let e_from = lin.j_from.transpose() * o_ja;
            let e_to = lin.j_to.transpose() * o_ja;
            for d in 0..STATE_DOF {
                self.border[(edge.from * STATE_DOF + d, *a)] += e_from[d];
                self.border[(edge.to * STATE_DOF + d, *a)] += e_to[d];
            }
            self.border_rhs[*a] -= o_ja.dot(&lin.residual);
            for (b, jb) in &lin.j_calibration {
                self.border_diag[(*a, *b)] += o_ja.dot(jb);
            }
            direct_border[*a] += o_ja.dot(ja);
        }
    }

    /// The direct (pre-elimination) terms of one observation of keyframe `k`.
    fn add_observation(
        &mut self,
        pattern: &Pattern,
        k: usize,
        rows: &ObservationRows,
        direct: &mut [f64],
        direct_border: &mut [f64],
    ) {
        let h = rows.j_pose.transpose() * rows.j_pose;
        pattern.add_diagonal(&mut self.values, k, &h);
        let g = rows.j_pose.transpose() * rows.residual;
        for d in 0..POSE_DOF {
            self.rhs[k * STATE_DOF + d] -= g[d];
            direct[k * STATE_DOF + d] += h[(d, d)];
        }
        for (a, ja) in rows.calibration() {
            let e = rows.j_pose.transpose() * ja;
            for d in 0..POSE_DOF {
                self.border[(k * STATE_DOF + d, a)] += e[d];
            }
            self.border_rhs[a] -= ja.dot(&rows.residual);
            for (b, jb) in rows.calibration() {
                self.border_diag[(a, b)] += ja.dot(jb);
            }
            direct_border[a] += ja.dot(ja);
        }
    }

    /// Subtracts `W C⁻¹ Wᵀ` and `W C⁻¹ b_l` of an eliminated landmark.
    fn eliminate(&mut self, pattern: &Pattern, block: &LandmarkBlock) {
        for (a, (ka, wa)) in block.keyframes.iter().enumerate() {
            let wa_c = wa * block.c_inv;
            let g = wa_c * block.rhs;
            for d in 0..POSE_DOF {
                self.rhs[ka * STATE_DOF + d] -= g[d];
            }
            for (kb, wb) in &block.keyframes[..=a] {
                pattern.add_block(&mut self.values, *ka, *kb, &(-(wa_c * wb.transpose())));
            }
            for (column, wc) in &block.calibration {
                let e = wa_c * wc;
                for d in 0..POSE_DOF {
                    self.border[(ka * STATE_DOF + d, *column)] -= e[d];
                }
            }
        }
        for (a, wa) in &block.calibration {
            let v = block.c_inv * wa;
            self.border_rhs[*a] -= v.dot(&block.rhs);
            for (b, wb) in &block.calibration {
                self.border_diag[(*a, *b)] -= v.dot(wb);
            }
        }
    }

    /// The keyframe and calibration steps.
    pub(super) fn solve(&self, pattern: &Pattern) -> Result<(Vec<f64>, Vec<f64>), OfflineBaError> {
        let failed = |what: String| OfflineBaError::CholeskyFailed(what);
        if !self.values.iter().all(|v| v.is_finite()) {
            return Err(failed("non-finite reduced system".into()));
        }
        let matrix = SparseColMatRef::new(pattern.symbolic.as_ref(), self.values.as_slice());
        let factor = Llt::try_new_with_symbolic(pattern.llt.clone(), matrix, Side::Lower)
            .map_err(|e| failed(format!("{e:?}")))?;
        let n = self.rhs.len();
        let mut z = Mat::from_fn(n, 1, |r, _| self.rhs[r]);
        factor.solve_in_place(&mut z);
        let nc = self.border_rhs.len();
        if nc == 0 {
            return finite((0..n).map(|r| z[(r, 0)]).collect(), Vec::new());
        }
        // `S = D - Eᵀ M⁻¹ E` on the border, then `δx = M⁻¹ b - M⁻¹ E δc`.
        let mut y = self.border.clone();
        factor.solve_in_place(&mut y);
        let s = Mat::from_fn(nc, nc, |i, j| {
            self.border_diag[(i, j)] - (0..n).map(|r| self.border[(r, i)] * y[(r, j)]).sum::<f64>()
        });
        let rc = Mat::from_fn(nc, 1, |i, _| {
            self.border_rhs[i] - (0..n).map(|r| self.border[(r, i)] * z[(r, 0)]).sum::<f64>()
        });
        let border = s
            .llt(Side::Lower)
            .map_err(|e| failed(format!("calibration border: {e:?}")))?;
        let dc_col = border.solve(&rc);
        let dc: Vec<f64> = (0..nc).map(|i| dc_col[(i, 0)]).collect();
        let dx = (0..n)
            .map(|r| z[(r, 0)] - (0..nc).map(|j| y[(r, j)] * dc[j]).sum::<f64>())
            .collect();
        finite(dx, dc)
    }
}

fn finite(dx: Vec<f64>, dc: Vec<f64>) -> Result<(Vec<f64>, Vec<f64>), OfflineBaError> {
    if dx.iter().chain(&dc).all(|v| v.is_finite()) {
        Ok((dx, dc))
    } else {
        Err(OfflineBaError::CholeskyFailed("non-finite step".into()))
    }
}
