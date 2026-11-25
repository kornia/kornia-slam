//! The estimator's tests (design card section 9).

mod frontend;
mod linearize;
mod marginalize;
mod window;

use std::collections::BTreeMap;

use faer::Mat;
use kornia_algebra::{Mat15F64, SMatF64, SVecF64, Vec15F64};

use super::linearize::{LandmarkBlock, Order};
use super::marginalize::MargPrior;
use super::state::{FrameId, STATE_DOF};
use super::window::{ImuEdge, Window};

/// A variable of the ledger: a frame's 15-DOF tangent (a pose uses its
/// first 6) or one absorbed landmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum LedgerVar {
    Frame(FrameId),
    Landmark(usize),
}

/// One absorbed factor as the quadratic `½ δᵀHδ + gᵀδ` in the variables'
/// first-estimate tangents.
pub(crate) struct LedgerEntry {
    vars: Vec<(LedgerVar, usize)>,
    h: Mat<f64>,
    g: Vec<f64>,
}

/// T2: every factor the window absorbs into its prior, with the Jacobians
/// it used, and the priors it produced.
#[derive(Default)]
pub(crate) struct Ledger {
    entries: Vec<LedgerEntry>,
    landmarks: usize,
    pub priors: usize,
}

impl Ledger {
    pub(crate) fn start(&mut self, id: FrameId, prior: &MargPrior) {
        self.entries.clear();
        self.entries.push(LedgerEntry {
            vars: vec![(LedgerVar::Frame(id), STATE_DOF)],
            h: prior.h.clone(),
            g: prior.b.clone(),
        });
    }

    /// Adds absorbed entries; each landmark gets its own variable.
    pub(crate) fn extend(&mut self, entries: Vec<LedgerEntry>) {
        for mut entry in entries {
            for (var, _) in &mut entry.vars {
                if *var == LedgerVar::Landmark(usize::MAX) {
                    *var = LedgerVar::Landmark(self.landmarks);
                    self.landmarks += 1;
                }
            }
            self.entries.push(entry);
        }
    }

    pub(crate) fn check_point(&mut self, _prior: &MargPrior) {
        self.priors += 1;
    }

    /// The one-shot Schur complement of every entry onto `prior`'s
    /// variables, against `prior`: the largest relative difference of `H`
    /// and of `b`.
    pub(crate) fn compare(&self, prior: &MargPrior) -> (f64, f64) {
        let mut index: BTreeMap<LedgerVar, usize> = BTreeMap::new();
        let mut n = 0;
        for entry in &self.entries {
            for (var, _) in &entry.vars {
                if !index.contains_key(var) {
                    index.insert(*var, n);
                    n += var_dim(var);
                }
            }
        }
        let mut h = Mat::<f64>::zeros(n, n);
        let mut b = vec![0.0; n];
        for entry in &self.entries {
            let idx: Vec<usize> = entry
                .vars
                .iter()
                .flat_map(|(var, dim)| {
                    let o = index[var];
                    (0..*dim).map(move |d| o + d)
                })
                .collect();
            for (r, &ir) in idx.iter().enumerate() {
                b[ir] += entry.g[r];
                for (c, &ic) in idx.iter().enumerate() {
                    h[(ir, ic)] += entry.h[(r, c)];
                }
            }
        }
        let keep: Vec<usize> = prior
            .order
            .iter()
            .flat_map(|e| (0..e.kind.dim()).map(move |d| (e.id, d)))
            .map(|(id, d)| index[&LedgerVar::Frame(id)] + d)
            .collect();
        let marg: Vec<usize> = (0..n).filter(|i| !keep.contains(i)).collect();
        let Ok(schur) = crate::mapping::marginalization::schur_complement(&h, &b, &keep, &marg)
        else {
            return (f64::INFINITY, f64::INFINITY);
        };
        let m = keep.len();
        let h_scale = (0..m).map(|i| prior.h[(i, i)].abs()).fold(0.0, f64::max);
        let b_scale = prior
            .b
            .iter()
            .fold(0.0f64, |a, v| a.max(v.abs()))
            .max(1e-300);
        let mut dh = 0.0f64;
        let mut db = 0.0f64;
        for r in 0..m {
            for c in 0..m {
                dh = dh.max((schur.h[(r, c)] - prior.h[(r, c)]).abs() / h_scale);
            }
            db = db.max((schur.b[r] - prior.b[r]).abs() / b_scale);
        }
        (dh, db)
    }
}

fn var_dim(var: &LedgerVar) -> usize {
    match var {
        LedgerVar::Frame(_) => STATE_DOF,
        LedgerVar::Landmark(_) => 3,
    }
}

impl Window {
    /// The current delta of frame `id`'s 15-DOF tangent (zero while free;
    /// a pose fills the first 6).
    fn ledger_delta(&self, id: FrameId) -> Vec15F64 {
        let mut d = Vec15F64::zeros();
        if let Some(s) = self.states.get(&id) {
            d.copy_from(s.delta());
        } else if let Some(p) = self.poses.get(&id) {
            d.fixed_rows_mut::<6>(0).copy_from(p.delta());
        }
        d
    }

    pub(crate) fn ledger_imu(
        &self,
        from: FrameId,
        edge: &ImuEdge,
        residual: &Vec15F64,
        j_i: &Mat15F64,
        j_j: &Mat15F64,
    ) -> Option<LedgerEntry> {
        self.ledger.as_ref()?;
        let mut j = SMatF64::<15, 30>::zeros();
        j.fixed_view_mut::<15, 15>(0, 0).copy_from(j_i);
        j.fixed_view_mut::<15, 15>(0, 15).copy_from(j_j);
        let mut delta = SVecF64::<30>::zeros();
        delta
            .fixed_rows_mut::<15>(0)
            .copy_from(&self.ledger_delta(from));
        delta
            .fixed_rows_mut::<15>(15)
            .copy_from(&self.ledger_delta(edge.to));
        let jt_omega = j.transpose() * edge.omega;
        let h = jt_omega * j;
        let g = jt_omega * (residual - j * delta);
        let entry = LedgerEntry {
            vars: vec![
                (LedgerVar::Frame(from), STATE_DOF),
                (LedgerVar::Frame(edge.to), STATE_DOF),
            ],
            h: Mat::from_fn(30, 30, |r, c| h[(r, c)]),
            g: g.iter().copied().collect(),
        };
        Some(entry)
    }

    pub(crate) fn ledger_landmark(
        &self,
        order: &Order,
        block: &LandmarkBlock,
    ) -> Option<LedgerEntry> {
        self.ledger.as_ref()?;
        let frame_of = |offset: usize| {
            order
                .entries()
                .iter()
                .find(|e| e.offset == offset)
                .map(|e| e.id)
        };
        let mut frames: Vec<FrameId> = Vec::new();
        for row in &block.recorded {
            for offset in [row.0, row.1].into_iter().flatten() {
                if let Some(id) = frame_of(offset)
                    && !frames.contains(&id)
                {
                    frames.push(id);
                }
            }
        }
        let n = 6 * frames.len() + 3;
        let mut h = Mat::<f64>::zeros(n, n);
        let mut g = vec![0.0; n];
        for (target, host, d_pose, d_landmark, residual, weight) in &block.recorded {
            let mut j = vec![[0.0f64; 2]; n];
            let mut r = *residual;
            for (offset, sign) in [(*target, 1.0), (*host, -1.0)] {
                let Some(id) = offset.and_then(frame_of) else {
                    continue;
                };
                let slot = frames.iter().position(|f| *f == id).unwrap_or(0);
                let delta = self.ledger_delta(id);
                for k in 0..6 {
                    for row in 0..2 {
                        j[6 * slot + k][row] += sign * d_pose[(row, k)];
                        r[row] -= sign * d_pose[(row, k)] * delta[k];
                    }
                }
            }
            for k in 0..3 {
                for row in 0..2 {
                    j[6 * frames.len() + k][row] = d_landmark[(row, k)];
                }
            }
            for a in 0..n {
                g[a] += weight * (j[a][0] * r[0] + j[a][1] * r[1]);
                for c in 0..n {
                    h[(a, c)] += weight * (j[a][0] * j[c][0] + j[a][1] * j[c][1]);
                }
            }
        }
        let mut vars: Vec<(LedgerVar, usize)> =
            frames.iter().map(|f| (LedgerVar::Frame(*f), 6)).collect();
        vars.push((LedgerVar::Landmark(usize::MAX), 3));
        Some(LedgerEntry { vars, h, g })
    }
}
