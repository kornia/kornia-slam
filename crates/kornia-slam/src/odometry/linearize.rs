//! The window's linear system: the variable order, the dense reduced
//! system and the per-landmark blocks eliminated into it by a 3×3 Schur
//! complement (Basalt `landmark_block_abs_dynamic.hpp`).

use faer::Mat;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{HuberKernel, RobustKernel};
use kornia_algebra::{Mat15F64, SMatF64, SVecF64, Vec15F64};
use kornia_factors::inertial::preintegration::residual_and_jacobians;
use kornia_factors::reprojection::{anchored_residual, anchored_residual_and_jacobians};

use super::config::VisualJacobians;
use super::landmarks::{Landmark, LandmarkId};
use super::marginalize::MissingVariable;
use super::state::{FrameId, POSE_DOF, STATE_DOF};
use super::window::{ImuEdge, Window};

/// Rows of points closer than this to the target camera are skipped, m.
const MIN_DEPTH_M: f64 = 1e-3;

/// What a frame contributes to the order: its 6-DOF pose or its 15-DOF state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VarKind {
    Pose,
    State,
}

impl VarKind {
    pub(crate) fn dim(self) -> usize {
        match self {
            Self::Pose => POSE_DOF,
            Self::State => STATE_DOF,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrderEntry {
    pub id: FrameId,
    pub kind: VarKind,
    pub offset: usize,
}

/// Variable order of a system: keyframe poses by time, then frame states by
/// time (`sqrt_keypoint_vio.cpp:1033-1055`).
#[derive(Debug, Clone, Default)]
pub(crate) struct Order {
    entries: Vec<OrderEntry>,
    dim: usize,
}

impl Order {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.dim = 0;
    }

    pub(crate) fn push(&mut self, id: FrameId, kind: VarKind) {
        self.entries.push(OrderEntry {
            id,
            kind,
            offset: self.dim,
        });
        self.dim += kind.dim();
    }

    pub(crate) fn entries(&self) -> &[OrderEntry] {
        &self.entries
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn find(&self, id: FrameId) -> Option<&OrderEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// Offset of `id`'s pose block, when it is in the order.
    pub(crate) fn pose_offset(&self, id: FrameId) -> Option<usize> {
        self.find(id).map(|entry| entry.offset)
    }
}

/// A dense information-form system `½ xᵀHx + bᵀx`, `b` the cost gradient.
#[derive(Debug, Clone)]
pub(crate) struct DenseSystem {
    pub h: Mat<f64>,
    pub b: Vec<f64>,
}

impl Default for DenseSystem {
    fn default() -> Self {
        Self {
            h: Mat::zeros(0, 0),
            b: Vec::new(),
        }
    }
}

impl DenseSystem {
    /// Zeroes the system and sizes it to `dim`, reusing its buffers when
    /// they fit.
    pub(crate) fn reset(&mut self, dim: usize) {
        if self.h.nrows() != dim {
            self.h = Mat::zeros(dim, dim);
        } else {
            self.h.fill(0.0);
        }
        self.b.clear();
        self.b.resize(dim, 0.0);
    }

    pub(crate) fn dim(&self) -> usize {
        self.b.len()
    }

    /// `H[r.., c..] += block`.
    pub(crate) fn add_block<const R: usize, const C: usize>(
        &mut self,
        row: usize,
        col: usize,
        block: &SMatF64<R, C>,
    ) {
        for c in 0..C {
            for r in 0..R {
                self.h[(row + r, col + c)] += block[(r, c)];
            }
        }
    }

    pub(crate) fn add_gradient<const R: usize>(&mut self, row: usize, g: &SVecF64<R>) {
        for r in 0..R {
            self.b[row + r] += g[r];
        }
    }

    /// One factor `r + J_a δ_a + J_b δ_b` on the two variables at offsets
    /// `a` and `b` (either may be absent), under the information `omega`.
    pub(crate) fn add_pair<const N: usize, const DA: usize, const DB: usize>(
        &mut self,
        a: Option<(usize, &SMatF64<N, DA>)>,
        b: Option<(usize, &SMatF64<N, DB>)>,
        omega: &SMatF64<N, N>,
        residual: &SVecF64<N>,
    ) {
        let omega_r = omega * residual;
        if let Some((oa, ja)) = a {
            let ja_omega = ja.transpose() * omega;
            self.add_block(oa, oa, &(ja_omega * ja));
            self.add_gradient(oa, &(ja.transpose() * omega_r));
            if let Some((ob, jb)) = b {
                let cross = ja_omega * jb;
                self.add_block(oa, ob, &cross);
                self.add_block(ob, oa, &cross.transpose());
            }
        }
        if let Some((ob, jb)) = b {
            self.add_block(ob, ob, &(jb.transpose() * omega * jb));
            self.add_gradient(ob, &(jb.transpose() * omega_r));
        }
    }
}

/// The rows of one landmark, gathered by pose block, and its elimination
/// into the pose system.
#[derive(Debug, Clone, Default)]
pub(crate) struct LandmarkBlock {
    /// Offsets of the pose blocks the rows touch.
    frames: Vec<usize>,
    /// `H_pp` blocks, `frames.len()²`, row-major.
    h_pp: Vec<SMatF64<6, 6>>,
    h_pl: Vec<SMatF64<6, 3>>,
    b_p: Vec<SVecF64<6>>,
    h_ll: SMatF64<3, 3>,
    b_l: SVecF64<3>,
    /// `H_ll⁻¹` once eliminated; `None` when the landmark is held fixed.
    h_ll_inv: Option<SMatF64<3, 3>>,
    rows: usize,
    /// Every row as added: target and host offsets, Jacobians, residual and
    /// weight (T2's ledger).
    #[cfg(test)]
    pub recorded: Vec<RecordedRow>,
}

#[cfg(test)]
pub(crate) type RecordedRow = (
    Option<usize>,
    Option<usize>,
    SMatF64<2, 6>,
    SMatF64<2, 3>,
    SVecF64<2>,
    f64,
);

impl LandmarkBlock {
    pub(crate) fn clear(&mut self) {
        self.frames.clear();
        self.h_pp.clear();
        self.h_pl.clear();
        self.b_p.clear();
        self.h_ll = SMatF64::zeros();
        self.b_l = SVecF64::zeros();
        self.h_ll_inv = None;
        self.rows = 0;
        #[cfg(test)]
        self.recorded.clear();
    }

    #[cfg(test)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Starts the landmark's rows over the pose blocks at `offsets`
    /// (duplicates ignored); every row's poses must be among them.
    pub(crate) fn begin(&mut self, offsets: impl IntoIterator<Item = usize>) {
        self.clear();
        for offset in offsets {
            if !self.frames.contains(&offset) {
                self.frames.push(offset);
            }
        }
        let m = self.frames.len();
        self.h_pp.resize(m * m, SMatF64::zeros());
        self.h_pl.resize(m, SMatF64::zeros());
        self.b_p.resize(m, SVecF64::zeros());
    }

    fn slot(&self, offset: usize) -> Option<usize> {
        self.frames.iter().position(|&o| o == offset)
    }

    /// One 2-dof row `r + J_t δ_t − J_t δ_h + J_l δ_l` under weight `w`:
    /// `target` and `host` are the pose offsets, absent for a frame outside
    /// the order or for a row whose two poses are one variable.
    pub(crate) fn add_row(
        &mut self,
        target: Option<usize>,
        host: Option<usize>,
        d_pose: &SMatF64<2, 6>,
        d_landmark: &SMatF64<2, 3>,
        residual: &SVecF64<2>,
        weight: f64,
    ) {
        self.rows += 1;
        #[cfg(test)]
        self.recorded
            .push((target, host, *d_pose, *d_landmark, *residual, weight));
        let mut slots: [(usize, f64); 2] = [(0, 0.0); 2];
        let mut n = 0;
        for (offset, sign) in [(target, 1.0), (host, -1.0)] {
            if let Some(slot) = offset.and_then(|o| self.slot(o)) {
                slots[n] = (slot, sign);
                n += 1;
            }
        }
        let m = self.frames.len();
        let jtw = d_pose.transpose() * weight;
        let jtj = jtw * d_pose;
        let jtl = jtw * d_landmark;
        let jtr = jtw * residual;
        for &(a, sa) in &slots[..n] {
            for &(b, sb) in &slots[..n] {
                self.h_pp[a * m + b] += jtj * (sa * sb);
            }
            self.h_pl[a] += jtl * sa;
            self.b_p[a] += jtr * sa;
        }
        let ltw = d_landmark.transpose() * weight;
        self.h_ll += ltw * d_landmark;
        self.b_l += ltw * residual;
    }

    /// `H_ll⁻¹`, after a relative ridge of 1e-9 when `H_ll` is not
    /// positive definite; `None` when it still is not.
    fn landmark_inverse(&self) -> Option<SMatF64<3, 3>> {
        let ridge = 1e-9 * self.h_ll.diagonal().max();
        self.h_ll.cholesky().map(|c| c.inverse()).or_else(|| {
            (self.h_ll + SMatF64::<3, 3>::identity() * ridge)
                .cholesky()
                .map(|c| c.inverse())
        })
    }

    /// Eliminates the landmark into `system`; returns `½ b_lᵀ H_ll⁻¹ b_l`,
    /// its share of the model decrease, or `None` when the landmark is held
    /// fixed because [`Self::landmark_inverse`] fails: its pose rows then
    /// enter unreduced (design card section 5).
    pub(crate) fn eliminate_into(&mut self, system: &mut DenseSystem) -> Option<f64> {
        self.h_ll_inv = self.landmark_inverse();
        self.apply(system);
        self.h_ll_inv
            .map(|inv| 0.5 * self.b_l.dot(&(inv * self.b_l)))
    }

    /// As [`Self::eliminate_into`] for a marginalization, where a landmark
    /// that cannot be eliminated is dropped without contributing; returns
    /// whether it entered.
    pub(crate) fn eliminate_reduced_into(&mut self, system: &mut DenseSystem) -> bool {
        self.h_ll_inv = self.landmark_inverse();
        if self.h_ll_inv.is_some() {
            self.apply(system);
        }
        self.h_ll_inv.is_some()
    }

    fn apply(&self, system: &mut DenseSystem) {
        let m = self.frames.len();
        for a in 0..m {
            let g_a = self.h_ll_inv.map(|inv| self.h_pl[a] * inv);
            let mut b_a = self.b_p[a];
            if let Some(g_a) = g_a {
                b_a -= g_a * self.b_l;
            }
            system.add_gradient(self.frames[a], &b_a);
            for b in 0..m {
                let mut block = self.h_pp[a * m + b];
                if let Some(g_a) = g_a {
                    block -= g_a * self.h_pl[b].transpose();
                }
                system.add_block(self.frames[a], self.frames[b], &block);
            }
        }
    }

    /// `(H_pl H_ll⁻¹ b_l)[i]` at system index `i`: what elimination took
    /// from the gradient there.
    #[cfg(test)]
    pub(crate) fn schur_gradient(&self, i: usize) -> f64 {
        let Some(inv) = self.h_ll_inv else {
            return 0.0;
        };
        self.frames
            .iter()
            .enumerate()
            .filter(|(_, o)| (**o..**o + 6).contains(&i))
            .map(|(a, o)| (self.h_pl[a] * inv * self.b_l)[i - o])
            .sum()
    }

    /// `δ_l = −H_ll⁻¹ (b_l + H_lp δ_p)` for the pose step `inc`; zero when
    /// the landmark is fixed.
    pub(crate) fn back_substitute(&self, inc: &[f64]) -> SVecF64<3> {
        let Some(inv) = self.h_ll_inv else {
            return SVecF64::zeros();
        };
        let mut rhs = self.b_l;
        for (a, &offset) in self.frames.iter().enumerate() {
            let step = SVecF64::<6>::from_fn(|r, _| inc[offset + r]);
            rhs += self.h_pl[a].transpose() * step;
        }
        -(inv * rhs)
    }
}

/// Cost of the window: Huber vision, IMU with its bias walk, and the prior
/// at the current deltas (`sqrt_keypoint_vio.cpp:1269-1291`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Cost {
    pub vision: f64,
    pub imu: f64,
    pub prior: f64,
    /// Visual rows used, per camera, and rows skipped for cheirality.
    pub rows: [usize; 2],
    pub skipped: usize,
}

impl Cost {
    pub(crate) fn total(&self) -> f64 {
        self.vision + self.imu + self.prior
    }
}

/// `w = ρ'(s) / σ²` and `½ ρ(s)` of a pixel residual under Huber, `s` the
/// squared residual in units of `σ` and the knee `(huber_px / σ)²`.
pub(crate) fn huber(residual: &SVecF64<2>, sigma: f64, huber_px: f64) -> (f64, f64) {
    let s = residual.norm_squared() / (sigma * sigma);
    let knee = (huber_px / sigma).powi(2);
    (
        HuberKernel.weight(s, knee) / (sigma * sigma),
        0.5 * HuberKernel.rho(s, knee),
    )
}

/// Which frames a landmark's rows may use.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Rows<'a> {
    /// Every window frame (optimization).
    All,
    /// Only frames in this order (marginalization, M8).
    In(&'a Order),
}

/// The rows of landmark `lm` into `block`, pose blocks at their offsets in
/// `order`: residual at the current values, Jacobians at the first-estimate
/// poses (design card section 3). A row whose poses are one variable (host
/// and stereo rows) has no pose columns. Returns the rows' cost, or `None`
/// when the host is not in the order.
pub(crate) fn landmark_rows(
    window: &Window,
    lm: &Landmark,
    order: &Order,
    rows: Rows<'_>,
    block: &mut LandmarkBlock,
    cost: &mut Cost,
) -> Option<()> {
    let host_offset = order.pose_offset(lm.host)?;
    let host = window.frame_pose(lm.host)?;
    let current = window.tuning.visual_jacobians == VisualJacobians::Current;
    let jacobian_pose = |id: FrameId| {
        if current {
            window.frame_pose(id)
        } else {
            window.frame_jacobian_pose(id)
        }
    };
    let host_lin = jacobian_pose(lm.host)?;
    let allowed = |frame: FrameId| match rows {
        Rows::All => true,
        Rows::In(o) => o.find(frame).is_some(),
    };
    block.begin(
        lm.observations
            .keys()
            .filter(|(frame, _)| allowed(*frame))
            .filter_map(|(frame, _)| order.pose_offset(*frame)),
    );
    let tuning = &window.tuning;
    for (&(frame, camera), &pixel) in &lm.observations {
        if !allowed(frame) {
            continue;
        }
        let Some(view) = window.cameras.get(usize::from(camera)) else {
            continue;
        };
        let (Some(target), Some(target_lin)) = (window.frame_pose(frame), jacobian_pose(frame))
        else {
            continue;
        };
        let to_camera = |pose: &Pose3d| view.camera_from_primary.compose(pose);
        let residual = anchored_residual(
            &host,
            &to_camera(&target),
            &lm.value,
            pixel,
            &view.camera,
            MIN_DEPTH_M,
        );
        let jacobians = anchored_residual_and_jacobians(
            &host_lin,
            &to_camera(&target_lin),
            &lm.value,
            pixel,
            &view.camera,
            MIN_DEPTH_M,
        );
        let (Some(residual), Some(row)) = (residual, jacobians) else {
            cost.skipped += 1;
            continue;
        };
        let (weight, row_cost) = huber(&residual, tuning.pixel_sigma, tuning.huber_px);
        cost.vision += row_cost;
        cost.rows[usize::from(camera).min(1)] += 1;
        let (t, h) = if frame == lm.host {
            (None, None)
        } else {
            (order.pose_offset(frame), Some(host_offset))
        };
        block.add_row(t, h, &row.d_target, &row.d_landmark, &residual, weight);
    }
    Some(())
}

/// The cost of landmark `lm`'s rows at the current values.
fn landmark_cost(window: &Window, lm: &Landmark, cost: &mut Cost) {
    let Some(host) = window.frame_pose(lm.host) else {
        return;
    };
    let tuning = &window.tuning;
    for (&(frame, camera), &pixel) in &lm.observations {
        let (Some(view), Some(target)) = (
            window.cameras.get(usize::from(camera)),
            window.frame_pose(frame),
        ) else {
            continue;
        };
        let target = view.camera_from_primary.compose(&target);
        match anchored_residual(&host, &target, &lm.value, pixel, &view.camera, MIN_DEPTH_M) {
            Some(residual) => {
                cost.vision += huber(&residual, tuning.pixel_sigma, tuning.huber_px).1;
                cost.rows[usize::from(camera).min(1)] += 1;
            }
            None => cost.skipped += 1,
        }
    }
}

/// The IMU edge from `from` into `system`: Jacobians from one call with
/// every state at its linearization point, the residual from a second call
/// at the current values when either is linearized (`imu_block.hpp:44-54`).
/// Returns `(cost, residual, J_from, J_to)`; `None` when a state is missing.
pub(crate) fn imu_edge(
    window: &Window,
    from: FrameId,
    edge: &ImuEdge,
) -> Option<(f64, Vec15F64, Mat15F64, Mat15F64)> {
    let (i, j) = (window.states.get(&from)?, window.states.get(&edge.to)?);
    let (li, lj) = (i.jacobian_point(), j.jacobian_point());
    let (mut residual, j_i, j_j) = residual_and_jacobians(
        li.nav_state(),
        lj.nav_state(),
        &edge.pim,
        &window.gravity,
        Some(&window.t_bc),
    );
    if i.is_linearized() || j.is_linearized() {
        let (vi, vj) = (i.value(), j.value());
        residual = residual_and_jacobians(
            vi.nav_state(),
            vj.nav_state(),
            &edge.pim,
            &window.gravity,
            Some(&window.t_bc),
        )
        .0;
    }
    let cost = 0.5 * residual.dot(&(edge.omega * residual));
    Some((cost, residual, j_i, j_j))
}

/// Everything one linearization leaves for the solve.
#[derive(Debug, Default)]
pub(crate) struct Linearization {
    pub order: Order,
    pub system: DenseSystem,
    pub blocks: Vec<(LandmarkId, LandmarkBlock)>,
    pub used_blocks: usize,
    /// `½ Σ b_lᵀ H_ll⁻¹ b_l`, the landmarks' share of the model decrease.
    pub landmark_model: f64,
    pub fixed_landmarks: usize,
    pub cost: Cost,
}

/// Linearizes the whole window at its current values (`linearize` of the
/// design card); the cost is `lin.cost`.
pub(crate) fn linearize(window: &Window, lin: &mut Linearization) -> Result<(), MissingVariable> {
    window.order(&mut lin.order);
    lin.system.reset(lin.order.dim());
    lin.cost = Cost::default();
    lin.landmark_model = 0.0;
    lin.fixed_landmarks = 0;
    lin.used_blocks = 0;
    lin.cost.prior = window.prior.accumulate(
        &window.prior_delta(&window.prior),
        &lin.order,
        &mut lin.system,
    )?;
    for (&from, edge) in &window.imu {
        let (Some((cost, residual, j_i, j_j)), Some(oi), Some(oj)) = (
            imu_edge(window, from, edge),
            lin.order.pose_offset(from),
            lin.order.pose_offset(edge.to),
        ) else {
            continue;
        };
        lin.cost.imu += cost;
        lin.system
            .add_pair(Some((oi, &j_i)), Some((oj, &j_j)), &edge.omega, &residual);
    }
    for (&id, lm) in window.landmarks.iter() {
        if lin.used_blocks == lin.blocks.len() {
            lin.blocks.push((id, LandmarkBlock::default()));
        }
        let (slot_id, block) = &mut lin.blocks[lin.used_blocks];
        *slot_id = id;
        if landmark_rows(window, lm, &lin.order, Rows::All, block, &mut lin.cost).is_none() {
            continue;
        }
        match block.eliminate_into(&mut lin.system) {
            Some(model) => lin.landmark_model += model,
            None => lin.fixed_landmarks += 1,
        }
        lin.used_blocks += 1;
    }
    Ok(())
}

/// The window's cost at its current values (`cost` of the design card).
pub(crate) fn cost(window: &Window) -> Cost {
    let mut cost = Cost {
        prior: window.prior.cost(&window.prior_delta(&window.prior)),
        ..Cost::default()
    };
    for (&from, edge) in &window.imu {
        if let Some((c, ..)) = imu_edge(window, from, edge) {
            cost.imu += c;
        }
    }
    for (_, lm) in window.landmarks.iter() {
        landmark_cost(window, lm, &mut cost);
    }
    cost
}
