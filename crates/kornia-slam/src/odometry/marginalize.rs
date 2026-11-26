//! The dense marginal prior on the window's oldest variables, built with
//! first-estimate Jacobians and kept as `½ δᵀHδ + bᵀδ` in `δ = x ⊟ lin`
//! (Basalt `sqrt_keypoint_vio.cpp:860-984`, `ba_base.cpp:408-476`).

use faer::Mat;

use super::linearize::{DenseSystem, Order, VarKind};
use super::state::{FrameId, POSE_DOF, STATE_DOF};
use crate::mapping::marginalization::schur_complement;

/// One variable of the prior: a frame's pose or state, at `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PriorEntry {
    pub id: FrameId,
    pub kind: VarKind,
    pub offset: usize,
}

/// `½ δᵀHδ + bᵀδ` over `order`; the linearization points live in the
/// window's variables.
#[derive(Debug, Clone)]
pub(crate) struct MargPrior {
    pub order: Vec<PriorEntry>,
    pub h: Mat<f64>,
    pub b: Vec<f64>,
}

impl MargPrior {
    /// A zero prior on the 15-DOF state `id`.
    pub(crate) fn zero(id: FrameId) -> Self {
        Self {
            order: vec![PriorEntry {
                id,
                kind: VarKind::State,
                offset: 0,
            }],
            h: Mat::zeros(STATE_DOF, STATE_DOF),
            b: vec![0.0; STATE_DOF],
        }
    }

    /// The first state's prior (`sqrt_keypoint_vio.cpp:76-110`): `weight`
    /// on the body position, `[−I₃ 0]` at the world origin, and on the yaw
    /// row `ĝᵀω`; the bias sigmas on the biases; nothing on roll, pitch or
    /// velocity.
    pub(crate) fn initial(
        id: FrameId,
        gravity_dir: [f64; 3],
        weight: f64,
        gyro_bias_sigma: f64,
        accel_bias_sigma: f64,
    ) -> Self {
        let mut prior = Self::zero(id);
        for d in 0..3 {
            prior.h[(d, d)] = weight;
            prior.h[(9 + d, 9 + d)] = gyro_bias_sigma.powi(-2);
            prior.h[(12 + d, 12 + d)] = accel_bias_sigma.powi(-2);
            for e in 0..3 {
                prior.h[(3 + d, 3 + e)] = weight * gravity_dir[d] * gravity_dir[e];
            }
        }
        prior
    }

    pub(crate) fn dim(&self) -> usize {
        self.b.len()
    }

    /// `½ δᵀHδ + bᵀδ` at `delta`, in the prior's order.
    pub(crate) fn cost(&self, delta: &[f64]) -> f64 {
        let n = self.dim();
        (0..n)
            .map(|r| {
                let hd: f64 = (0..n).map(|c| self.h[(r, c)] * delta[c]).sum();
                delta[r] * (0.5 * hd + self.b[r])
            })
            .sum()
    }

    /// Adds `H` and the gradient `b + Hδ` to `system`, each prior variable
    /// at its offset in `order`; returns the prior's cost at `delta`.
    pub(crate) fn accumulate(
        &self,
        delta: &[f64],
        order: &Order,
        system: &mut DenseSystem,
    ) -> Result<f64, MissingVariable> {
        let offsets = self.system_offsets(order)?;
        let n = self.dim();
        for r in 0..n {
            let hd: f64 = (0..n).map(|c| self.h[(r, c)] * delta[c]).sum();
            system.b[offsets[r]] += self.b[r] + hd;
            for c in 0..n {
                system.h[(offsets[r], offsets[c])] += self.h[(r, c)];
            }
        }
        Ok(self.cost(delta))
    }

    /// The system index of each prior dimension.
    fn system_offsets(&self, order: &Order) -> Result<Vec<usize>, MissingVariable> {
        let mut offsets = Vec::with_capacity(self.dim());
        for entry in &self.order {
            let found = order.find(entry.id).ok_or(MissingVariable(entry.id))?;
            if found.kind.dim() < entry.kind.dim() {
                return Err(MissingVariable(entry.id));
            }
            offsets.extend((0..entry.kind.dim()).map(|d| found.offset + d));
        }
        Ok(offsets)
    }

    /// `b −= Hδ`: moves the gradient at the current values to the
    /// linearization points, so the prior reads `δ = x ⊟ lin`
    /// (`sqrt_keypoint_vio.cpp:959-984`).
    pub(crate) fn recentre(&mut self, delta: &[f64]) {
        let n = self.dim();
        for r in 0..n {
            self.b[r] -= (0..n).map(|c| self.h[(r, c)] * delta[c]).sum::<f64>();
        }
    }
}

/// A prior variable is not in the system it is added to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MissingVariable(pub FrameId);

/// What a marginalization does with one variable of its order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fate {
    Keep,
    /// Of a state: its pose stays, velocity and biases leave.
    KeepPose,
    Drop,
}

/// The prior left when `system` (over `order`) loses what `fate` drops, in
/// the order's sequence; `b` is still the gradient at the current values
/// (call [`MargPrior::recentre`] next). The flag says `H_mm` was
/// pseudo-inverted.
pub(crate) fn marginalize_system(
    system: &DenseSystem,
    order: &Order,
    fate: impl Fn(FrameId) -> Fate,
) -> Result<(MargPrior, bool), crate::mapping::marginalization::MarginalizationError> {
    let mut keep = Vec::new();
    let mut marg = Vec::new();
    let mut prior_order = Vec::new();
    for entry in order.entries() {
        let dim = entry.kind.dim();
        let kept = match (fate(entry.id), entry.kind) {
            (Fate::Keep, kind) => Some(kind),
            (Fate::KeepPose, _) => Some(VarKind::Pose),
            (Fate::Drop, _) => None,
        };
        let kept_dim = kept.map_or(0, VarKind::dim);
        if let Some(kind) = kept {
            prior_order.push(PriorEntry {
                id: entry.id,
                kind,
                offset: keep.len(),
            });
        }
        keep.extend((0..kept_dim).map(|d| entry.offset + d));
        marg.extend((kept_dim..dim).map(|d| entry.offset + d));
    }
    let schur = schur_complement(&system.h, &system.b, &keep, &marg)?;
    Ok((
        MargPrior {
            order: prior_order,
            h: schur.h,
            b: schur.b,
        },
        schur.pseudo_inverse,
    ))
}

/// Per direction `x, y, z, roll, pitch, yaw`: `nᵀHn / (‖n‖² ‖H‖_F)` of the
/// prior on the world-transform tangent `n` (Basalt `sqrt_ba_base.cpp:43-228`).
/// A world translation `e_k` is `ρ = −e_k` on every pose; a world rotation
/// about axis `a` through `m`, the centroid of the positions, is
/// `ρ = a × m`, `ω = −a`, `δv = a × v` on every state. `positions` and
/// `velocities` are in the prior's order (`None` velocity for a pose).
pub(crate) fn nullspace_residual(
    prior: &MargPrior,
    centroid: [f64; 3],
    velocities: &[Option<[f64; 3]>],
) -> [f64; 6] {
    let n = prior.dim();
    let frobenius = (0..n)
        .flat_map(|c| (0..n).map(move |r| (r, c)))
        .map(|(r, c)| prior.h[(r, c)].powi(2))
        .sum::<f64>()
        .sqrt();
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    std::array::from_fn(|direction| {
        let mut v = vec![0.0; n];
        let axis: [f64; 3] = std::array::from_fn(|k| f64::from(u8::from(k == direction % 3)));
        for (entry, velocity) in prior.order.iter().zip(velocities) {
            let o = entry.offset;
            if direction < 3 {
                v[o + direction] = -1.0;
                continue;
            }
            let rho = cross(axis, centroid);
            for k in 0..3 {
                v[o + k] = rho[k];
                v[o + 3 + k] = -axis[k];
            }
            if let (VarKind::State, Some(velocity)) = (entry.kind, velocity) {
                let dv = cross(axis, *velocity);
                v[o + POSE_DOF..o + POSE_DOF + 3].copy_from_slice(&dv);
            }
        }
        let norm2: f64 = v.iter().map(|x| x * x).sum();
        let quadratic: f64 = (0..n)
            .map(|r| v[r] * (0..n).map(|c| prior.h[(r, c)] * v[c]).sum::<f64>())
            .sum();
        if norm2 == 0.0 || frobenius == 0.0 {
            0.0
        } else {
            quadratic.abs() / (norm2 * frobenius)
        }
    })
}
