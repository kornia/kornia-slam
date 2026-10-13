//! Wheel-odometry factor for visual-inertial BA.
//!
//! A ground robot's wheels measure the planar motion of its base between two
//! keyframes. The factor compares that motion with the one the keyframe poses
//! imply, through the base-to-camera extrinsic of each keyframe (a moving
//! head makes it time-varying):
//!
//! ```text
//!     T_wb = T_cw⁻¹ · T_cb                 (base in world)
//!     r    = Log( ΔT⁻¹ · T_wb_i⁻¹ · T_wb_j )  ∈ ℝ⁶, tangent [ρ; ω]
//! ```
//!
//! The solver perturbs a keyframe as `T_cw ← T_cw · Exp(ξ)`, so
//! `T_wb⁻¹ ← T_wb⁻¹ · Exp(ξ)` too, and with `A = ΔT⁻¹ · T_cb_i⁻¹ · T_cw_i`
//!
//! ```text
//!     ∂r/∂ξ_i = J_l⁻¹(r) · Ad(A),    ∂r/∂ξ_j = −∂r/∂ξ_i.
//! ```
//!
//! kornia-algebra's SE(3) is f32-only, which cannot reach the 1e-6 Jacobian
//! check, so the few f64 SE(3) pieces live here.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

/// 6×6 row-major block over the pose tangent `[ρ; ω]`.
pub(crate) type Mat6 = [[f64; 6]; 6];

/// Wheel-odometry motion of the base between two keyframes.
///
/// The base frame is planar: x forward, y left, z up (REP-103). The wheels do
/// not observe z, roll or pitch, so the lift to SE(3) holds them at zero with
/// the stiff `z_sigma` / `tilt_sigma`; a floor that is not flat is what those
/// two absorb.
///
/// `#[non_exhaustive]`: build it with [`Self::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WheelOdomMeasurement {
    /// `(dx, dy, dθ)`: base j in base i, metres and radians.
    pub delta: [f64; 3],
    /// Covariance of `delta`, row-major (m², m·rad, rad²).
    pub covariance: [[f64; 3]; 3],
    /// Out-of-plane translation sigma, metres.
    pub z_sigma: f64,
    /// Roll and pitch sigma, radians.
    pub tilt_sigma: f64,
}

impl WheelOdomMeasurement {
    /// Out-of-plane sigmas used by [`Self::new`]: 1 cm and 0.01 rad (0.57°).
    /// Unmeasured; tune them on the robot.
    pub const DEFAULT_Z_SIGMA: f64 = 0.01;
    pub const DEFAULT_TILT_SIGMA: f64 = 0.01;

    /// A measurement with the default out-of-plane sigmas.
    pub fn new(delta: [f64; 3], covariance: [[f64; 3]; 3]) -> Self {
        Self {
            delta,
            covariance,
            z_sigma: Self::DEFAULT_Z_SIGMA,
            tilt_sigma: Self::DEFAULT_TILT_SIGMA,
        }
    }

    /// `self` with the out-of-plane sigmas `z_sigma` (metres) and
    /// `tilt_sigma` (radians).
    pub fn with_out_of_plane_sigmas(self, z_sigma: f64, tilt_sigma: f64) -> Self {
        Self {
            z_sigma,
            tilt_sigma,
            ..self
        }
    }

    /// The planar delta as a transform taking base-j coordinates to base-i.
    pub fn delta_pose(&self) -> Pose3d {
        let [dx, dy, dtheta] = self.delta;
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, dtheta)).matrix(),
            Vec3F64::new(dx, dy, 0.0),
        )
    }

    /// Information over the residual tangent `[x, y, z, roll, pitch, yaw]`.
    ///
    /// `covariance` is over `delta`, whose translation is in base-i axes, but
    /// the residual `Log(ΔT⁻¹ · T_ij)` carries the translation error in the
    /// predicted base-j axes: `r ≈ J δz` with `J = blockdiag(R(dθ)ᵀ, 1)`. The
    /// planar block is therefore `J⁻ᵀ Σ⁻¹ J⁻¹`, so that `rᵀ Ω r` is the
    /// Mahalanobis distance of `δz` under `covariance` whatever the turn.
    ///
    /// `None` if `delta` or the covariance is not finite, the covariance is
    /// not symmetric positive definite, or a sigma is not finite and positive.
    pub fn information(&self) -> Option<Mat6> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !(positive(self.z_sigma) && positive(self.tilt_sigma)) {
            return None;
        }
        if !self
            .delta
            .iter()
            .chain(self.covariance.iter().flatten())
            .all(|v| v.is_finite())
        {
            return None;
        }
        let c = &self.covariance;
        // det > 0 alone passes diag(-a, -a, a), whose negative information
        // makes LM grow the residual. Sylvester needs symmetry, so check it.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let tol = 1e-9 * (c[i][i] * c[j][j]).abs().sqrt();
                if (c[i][j] - c[j][i]).abs() > tol {
                    return None;
                }
            }
        }
        if !(c[0][0] > 0.0 && c[0][0] * c[1][1] - c[0][1] * c[1][0] > 0.0) {
            return None;
        }
        let cov = Mat3F64::from_cols_array(&[
            c[0][0], c[1][0], c[2][0], c[0][1], c[1][1], c[2][1], c[0][2], c[1][2], c[2][2],
        ]);
        let det = cov.determinant();
        if !(det.is_finite() && det > 0.0) {
            return None;
        }
        // J⁻¹ = blockdiag(R(dθ), 1) rotates a base-j residual back into the
        // base-i axes the covariance is written in.
        let (sin, cos) = self.delta[2].sin_cos();
        let j_inv = Mat3F64::from_cols_array(&[cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0]);
        let planar = (j_inv.transpose() * cov.inverse() * j_inv).to_cols_array();
        // delta rows (x, y, θ) sit at tangent indices (0, 1, 5).
        const AXES: [usize; 3] = [0, 1, 5];
        let mut info = [[0.0; 6]; 6];
        for (col, &ci) in AXES.iter().enumerate() {
            for (row, &ri) in AXES.iter().enumerate() {
                info[ri][ci] = planar[col * 3 + row];
            }
        }
        info[2][2] = 1.0 / (self.z_sigma * self.z_sigma);
        let tilt_info = 1.0 / (self.tilt_sigma * self.tilt_sigma);
        info[3][3] = tilt_info;
        info[4][4] = tilt_info;
        Some(info)
    }
}

impl WheelOdomMeasurement {
    /// The motion `self` then `next` (base k in base i from i→j and j→k).
    ///
    /// The planar covariance propagates to first order through the SE(2)
    /// composition, `Jᵢ Σᵢ Jᵢᵀ + Jⱼ Σⱼ Jⱼᵀ` (the two intervals' wheel noise is
    /// independent); the out-of-plane sigmas add in quadrature.
    pub fn compose(&self, next: &Self) -> Self {
        let [x1, y1, th1] = self.delta;
        let [x2, y2, th2] = next.delta;
        let (s, c) = th1.sin_cos();
        let delta = [
            x1 + c * x2 - s * y2,
            y1 + s * x2 + c * y2,
            (th1 + th2).sin().atan2((th1 + th2).cos()),
        ];
        let j_first = [
            [1.0, 0.0, -s * x2 - c * y2],
            [0.0, 1.0, c * x2 - s * y2],
            [0.0, 0.0, 1.0],
        ];
        let j_next = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let a = congruence(&j_first, &self.covariance);
        let b = congruence(&j_next, &next.covariance);
        let mut covariance = [[0.0; 3]; 3];
        for (r, row) in covariance.iter_mut().enumerate() {
            for (col, x) in row.iter_mut().enumerate() {
                // Symmetrized: `information` refuses an asymmetric covariance.
                *x = 0.5 * (a[r][col] + a[col][r] + b[r][col] + b[col][r]);
            }
        }
        Self {
            delta,
            covariance,
            z_sigma: self.z_sigma.hypot(next.z_sigma),
            tilt_sigma: self.tilt_sigma.hypot(next.tilt_sigma),
        }
    }
}

/// `J Σ Jᵀ` for row-major 3×3 matrices.
fn congruence(j: &[[f64; 3]; 3], sigma: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = (0..3)
                .flat_map(|k| (0..3).map(move |l| (k, l)))
                .map(|(k, l)| j[r][k] * sigma[k][l] * j[c][l])
                .sum();
        }
    }
    out
}

/// A wheel-odometry constraint between keyframes `from_idx` and `to_idx`.
///
/// `#[non_exhaustive]`: build it with [`Self::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WheelOdomFactor {
    pub from_idx: usize,
    pub to_idx: usize,
    pub measurement: WheelOdomMeasurement,
    /// Base-to-camera transform at `from_idx`: `X_cam = T_cb · X_base`.
    pub t_cb_from: Pose3d,
    /// Base-to-camera transform at `to_idx`.
    pub t_cb_to: Pose3d,
    /// Huber threshold on the 6-DOF chi² (12.59 is the 95 % quantile); `None`
    /// keeps the factor quadratic.
    pub huber_chi2: Option<f64>,
}

impl WheelOdomFactor {
    /// A factor between keyframes `from_idx` and `to_idx` with the
    /// base-to-camera extrinsic at each end and an optional Huber threshold.
    pub fn new(
        from_idx: usize,
        to_idx: usize,
        measurement: WheelOdomMeasurement,
        t_cb_from: Pose3d,
        t_cb_to: Pose3d,
        huber_chi2: Option<f64>,
    ) -> Self {
        Self {
            from_idx,
            to_idx,
            measurement,
            t_cb_from,
            t_cb_to,
            huber_chi2,
        }
    }

    /// Whether the factor can enter a solve: it links two distinct
    /// keyframes, the measurement has an information matrix
    /// ([`WheelOdomMeasurement::information`]), both extrinsics are finite and
    /// the Huber threshold, if any, is finite and positive.
    /// [`super::bundle_adjustment::run_local_inertial_ba`] skips a factor that
    /// is not; `visual_inertial_bundle_adjust` refuses it.
    pub fn is_valid(&self) -> bool {
        self.from_idx != self.to_idx
            && self.huber_valid()
            && self.extrinsics_finite()
            && self.measurement.information().is_some()
    }

    pub(crate) fn huber_valid(&self) -> bool {
        self.huber_chi2.is_none_or(|k| k.is_finite() && k > 0.0)
    }

    pub(crate) fn extrinsics_finite(&self) -> bool {
        [&self.t_cb_from, &self.t_cb_to].iter().all(|t| {
            t.rotation.to_cols_array().iter().all(|v| v.is_finite())
                && [t.translation.x, t.translation.y, t.translation.z]
                    .iter()
                    .all(|v| v.is_finite())
        })
    }

    /// The factor from `self.from_idx` to `next.to_idx`, replacing the two
    /// when the keyframe between them is removed; `None` unless `next` starts
    /// where `self` ends.
    ///
    /// Either Huber threshold makes the composite robust, at the tighter of
    /// the two.
    pub fn compose(&self, next: &Self) -> Option<Self> {
        if self.to_idx != next.from_idx {
            return None;
        }
        let huber_chi2 = match (self.huber_chi2, next.huber_chi2) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Some(Self {
            from_idx: self.from_idx,
            to_idx: next.to_idx,
            measurement: self.measurement.compose(&next.measurement),
            t_cb_from: self.t_cb_from,
            t_cb_to: next.t_cb_to,
            huber_chi2,
        })
    }
}

/// Residual, and its Jacobian wrt keyframe i's pose; keyframe j's is the negation.
///
/// `pose_cw_*` are the world→camera keyframe poses.
pub(crate) fn residual_and_jacobian(
    pose_cw_i: &Pose3d,
    pose_cw_j: &Pose3d,
    factor: &WheelOdomFactor,
) -> ([f64; 6], Mat6) {
    relative_pose_residual(
        pose_cw_i,
        pose_cw_j,
        &factor.t_cb_from,
        &factor.t_cb_to,
        &factor.measurement.delta_pose(),
    )
}

/// `r = Log(Δ⁻¹ · T_cb_from⁻¹ · T_cw_i · T_cw_j⁻¹ · T_cb_to)` and its
/// Jacobian wrt keyframe i's pose, `J_l⁻¹(r) · Ad(A)` with
/// `A = Δ⁻¹ · T_cb_from⁻¹ · T_cw_i`; keyframe j's is the negation. With
/// identity extrinsics and delta it is the zero-relative-pose residual
/// `Log(T_cw_i · T_cw_j⁻¹)` of [`super::stationary`].
pub(crate) fn relative_pose_residual(
    pose_cw_i: &Pose3d,
    pose_cw_j: &Pose3d,
    t_cb_from: &Pose3d,
    t_cb_to: &Pose3d,
    delta: &Pose3d,
) -> ([f64; 6], Mat6) {
    let t_bw_i = t_cb_from.inverse().compose(pose_cw_i);
    let t_wb_j = pose_cw_j.inverse().compose(t_cb_to);
    let a = delta.inverse().compose(&t_bw_i);
    let r = se3_log(&a.compose(&t_wb_j));
    (
        r,
        mat6_mul(&se3_left_jacobian_inverse(&r), &se3_adjoint(&a)),
    )
}

/// The robust weight and the Huber cost `½ρ(rᵀΩr)` of one residual
/// ([`super::robust`]).
pub(crate) fn robust_weight_and_cost(
    r: &[f64; 6],
    info: &Mat6,
    huber_chi2: Option<f64>,
) -> (f64, f64) {
    super::robust::huber_weight_and_cost(quadratic_form(info, r), huber_chi2)
}

pub(crate) fn quadratic_form(m: &Mat6, v: &[f64; 6]) -> f64 {
    (0..6)
        .map(|a| (0..6).map(|b| v[a] * m[a][b] * v[b]).sum::<f64>())
        .sum()
}

fn mat6_mul(a: &Mat6, b: &Mat6) -> Mat6 {
    let mut out = [[0.0; 6]; 6];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = (0..6).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

fn set_block(m: &mut Mat6, row: usize, col: usize, block: &Mat3F64) {
    let cols = block.to_cols_array();
    for c in 0..3 {
        for r in 0..3 {
            m[row + r][col + c] = cols[c * 3 + r];
        }
    }
}

/// `[ρ; ω]` with `T = Exp(ρ, ω)`, translation `J_l(ω)·ρ`.
fn se3_log(t: &Pose3d) -> [f64; 6] {
    let omega = SO3F64::from_matrix(&t.rotation).log();
    let rho = SO3F64::left_jacobian(omega).inverse() * t.translation;
    [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z]
}

/// `Ad(T) = [[R, t^R], [0, R]]` for the `[ρ; ω]` tangent order.
fn se3_adjoint(t: &Pose3d) -> Mat6 {
    let mut ad = [[0.0; 6]; 6];
    set_block(&mut ad, 0, 0, &t.rotation);
    set_block(&mut ad, 0, 3, &(SO3F64::hat(t.translation) * t.rotation));
    set_block(&mut ad, 3, 3, &t.rotation);
    ad
}

/// Inverse of the SE(3) left Jacobian, `[[J⁻¹, −J⁻¹QJ⁻¹], [0, J⁻¹]]`, with
/// `J` the SO(3) left Jacobian and `Q` from Barfoot, *State Estimation for
/// Robotics*, eq. 7.86. The f32 `SE3F32::left_jacobian` is not used: its last
/// Q coefficient divides `(θ − sin θ)/θ³` by θ⁵ again.
fn se3_left_jacobian_inverse(xi: &[f64; 6]) -> Mat6 {
    let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
    let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
    let j_inv = SO3F64::left_jacobian(omega).inverse();
    let q = barfoot_q(rho, omega);
    let mut out = [[0.0; 6]; 6];
    set_block(&mut out, 0, 0, &j_inv);
    set_block(&mut out, 0, 3, &(j_inv * q * j_inv * -1.0));
    set_block(&mut out, 3, 3, &j_inv);
    out
}

fn barfoot_q(rho: Vec3F64, omega: Vec3F64) -> Mat3F64 {
    let theta2 = omega.dot(omega);
    // Below 1e-2 rad the closed forms lose digits to cancellation; the
    // two-term series is exact to ~θ⁴.
    let (c1, c2, c3) = if theta2 < 1e-4 {
        (
            1.0 / 6.0 - theta2 / 120.0,
            1.0 / 24.0 - theta2 / 720.0,
            1.0 / 120.0 - theta2 / 2520.0,
        )
    } else {
        let theta = theta2.sqrt();
        let (s, c) = theta.sin_cos();
        (
            (theta - s) / (theta2 * theta),
            (theta2 + 2.0 * c - 2.0) / (2.0 * theta2 * theta2),
            (2.0 * theta - 3.0 * s + theta * c) / (2.0 * theta2 * theta2 * theta),
        )
    };
    let p = SO3F64::hat(rho);
    let w = SO3F64::hat(omega);
    let wp = w * p;
    let pw = p * w;
    let wpw = wp * w;
    p * 0.5 + (wp + pw + wpw) * c1 + (w * wp + pw * w - wpw * 3.0) * c2 + (wpw * w + w * wpw) * c3
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform samples in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn vec3(&mut self, scale: f64) -> Vec3F64 {
            Vec3F64::new(self.next(), self.next(), self.next()) * scale
        }
        fn pose(&mut self, t_scale: f64, r_scale: f64) -> Pose3d {
            Pose3d::new(SO3F64::exp(self.vec3(r_scale)).matrix(), self.vec3(t_scale))
        }
    }

    fn se3_exp(xi: &[f64; 6]) -> Pose3d {
        let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
        let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
        Pose3d::new(
            SO3F64::exp(omega).matrix(),
            SO3F64::left_jacobian(omega) * rho,
        )
    }

    fn perturbed(pose: &Pose3d, k: usize, eps: f64) -> Pose3d {
        let mut xi = [0.0; 6];
        xi[k] = eps;
        pose.compose(&se3_exp(&xi))
    }

    fn factor(rng: &mut Lcg, t_cb_from: Pose3d, t_cb_to: Pose3d) -> WheelOdomFactor {
        WheelOdomFactor {
            from_idx: 0,
            to_idx: 1,
            measurement: WheelOdomMeasurement::new(
                [rng.next(), rng.next(), rng.next() * 1.5],
                [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
            ),
            t_cb_from,
            t_cb_to,
            huber_chi2: None,
        }
    }

    /// Worst |analytic − central difference| over both Jacobians.
    fn jacobian_error(pose_i: &Pose3d, pose_j: &Pose3d, f: &WheelOdomFactor) -> f64 {
        const EPS: f64 = 1e-6;
        let (_, j_i) = residual_and_jacobian(pose_i, pose_j, f);
        let mut worst = 0.0f64;
        for k in 0..6 {
            let d_i = |e| residual_and_jacobian(&perturbed(pose_i, k, e), pose_j, f).0;
            let d_j = |e| residual_and_jacobian(pose_i, &perturbed(pose_j, k, e), f).0;
            let (pi, mi, pj, mj) = (d_i(EPS), d_i(-EPS), d_j(EPS), d_j(-EPS));
            for (row, analytic) in j_i.iter().enumerate() {
                let num_i = (pi[row] - mi[row]) / (2.0 * EPS);
                let num_j = (pj[row] - mj[row]) / (2.0 * EPS);
                worst = worst
                    .max((analytic[k] - num_i).abs())
                    .max((-analytic[k] - num_j).abs());
            }
        }
        worst
    }

    /// The stationary factor's zero-relative-pose residual: identity
    /// extrinsics and delta, `r = Log(T_i · T_j⁻¹)`, `∂r/∂ξ_i = J_l⁻¹(r)·Ad(T_i)`.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn identity_relative_pose_jacobians_match_central_differences() {
        const EPS: f64 = 1e-6;
        let id = Pose3d::IDENTITY;
        let mut rng = Lcg(23);
        for k in 0..60 {
            let pose_i = rng.pose(3.0, 1.0);
            // Half far apart, half a few mm / mrad apart (the series branch).
            let pose_j = if k % 2 == 0 {
                rng.pose(3.0, 1.0)
            } else {
                pose_i.compose(&se3_exp(&std::array::from_fn(|_| rng.next() * 3e-3)))
            };
            let res = |a: &Pose3d, b: &Pose3d| relative_pose_residual(a, b, &id, &id, &id);
            let (r, j_i) = res(&pose_i, &pose_j);
            let direct = se3_log(&pose_i.compose(&pose_j.inverse()));
            for (a, b) in r.iter().zip(direct) {
                assert!((a - b).abs() < 1e-12);
            }
            let ad = se3_adjoint(&pose_i);
            let expected = mat6_mul(&se3_left_jacobian_inverse(&r), &ad);
            assert_eq!(j_i, expected);
            for c in 0..6 {
                let (pi, mi) = (
                    res(&perturbed(&pose_i, c, EPS), &pose_j).0,
                    res(&perturbed(&pose_i, c, -EPS), &pose_j).0,
                );
                let (pj, mj) = (
                    res(&pose_i, &perturbed(&pose_j, c, EPS)).0,
                    res(&pose_i, &perturbed(&pose_j, c, -EPS)).0,
                );
                for row in 0..6 {
                    let num_i = (pi[row] - mi[row]) / (2.0 * EPS);
                    let num_j = (pj[row] - mj[row]) / (2.0 * EPS);
                    assert!((j_i[row][c] - num_i).abs() <= 1e-6, "i ({row},{c})");
                    assert!((-j_i[row][c] - num_j).abs() <= 1e-6, "j ({row},{c})");
                }
            }
        }
    }

    #[test]
    fn jacobians_match_central_differences() {
        let mut rng = Lcg(7);
        for _ in 0..50 {
            let f = random_factor(&mut rng);
            let (pose_i, pose_j) = (rng.pose(3.0, 1.2), rng.pose(3.0, 1.2));
            let err = jacobian_error(&pose_i, &pose_j, &f);
            assert!(err <= 1e-6, "large residual: {err:e}");
        }
    }

    fn random_factor(rng: &mut Lcg) -> WheelOdomFactor {
        let (t_cb_from, t_cb_to) = (rng.pose(0.5, 1.0), rng.pose(0.5, 1.0));
        factor(rng, t_cb_from, t_cb_to)
    }

    /// Near a solution the residual is small and `barfoot_q` takes its series.
    #[test]
    fn jacobians_match_central_differences_near_the_solution() {
        let mut rng = Lcg(11);
        for _ in 0..50 {
            let f = random_factor(&mut rng);
            let pose_i = rng.pose(3.0, 1.2);
            // T_wb_j = T_wb_i · ΔT · small, solved for T_cw_j.
            let t_wb_i = pose_i.inverse().compose(&f.t_cb_from);
            let small = se3_exp(&std::array::from_fn(|_| rng.next() * 1e-3));
            let t_wb_j = t_wb_i.compose(&f.measurement.delta_pose()).compose(&small);
            let pose_j = f.t_cb_to.compose(&t_wb_j.inverse());
            let (r, _) = residual_and_jacobian(&pose_i, &pose_j, &f);
            assert!(r[3..].iter().map(|x| x * x).sum::<f64>() < 1e-4);
            let err = jacobian_error(&pose_i, &pose_j, &f);
            assert!(err <= 1e-6, "small residual: {err:e}");
        }
    }

    /// `Log(Exp(δ)·Exp(ξ)) ≈ ξ + J_l⁻¹(ξ)·δ` on both sides of the series
    /// switch, with a long ρ so the small-angle Q terms are visible.
    #[test]
    fn left_jacobian_inverse_matches_central_differences() {
        const EPS: f64 = 1e-6;
        let mut rng = Lcg(5);
        for theta in [5e-3, 9e-3, 0.5, 2.5] {
            let rho = rng.vec3(10.0);
            let axis = rng.vec3(1.0).normalize();
            let xi = [
                rho.x,
                rho.y,
                rho.z,
                axis.x * theta,
                axis.y * theta,
                axis.z * theta,
            ];
            let t = se3_exp(&xi);
            let j_inv = se3_left_jacobian_inverse(&xi);
            for k in 0..6 {
                let log_at = |e: f64| {
                    let mut d = [0.0; 6];
                    d[k] = e;
                    se3_log(&se3_exp(&d).compose(&t))
                };
                let (p, m) = (log_at(EPS), log_at(-EPS));
                for row in 0..6 {
                    let num = (p[row] - m[row]) / (2.0 * EPS);
                    assert!(
                        (j_inv[row][k] - num).abs() <= 1e-6,
                        "θ={theta} ({row},{k}): {} vs {num}",
                        j_inv[row][k]
                    );
                }
            }
        }
    }

    #[test]
    fn consistent_poses_give_a_zero_residual() {
        let mut rng = Lcg(3);
        let f = random_factor(&mut rng);
        let pose_i = rng.pose(2.0, 1.0);
        let t_wb_j = pose_i
            .inverse()
            .compose(&f.t_cb_from)
            .compose(&f.measurement.delta_pose());
        let pose_j = f.t_cb_to.compose(&t_wb_j.inverse());
        let (r, _) = residual_and_jacobian(&pose_i, &pose_j, &f);
        assert!(r.iter().all(|x| x.abs() < 1e-12), "{r:?}");
    }

    #[test]
    fn information_places_the_planar_block_and_rejects_bad_covariance() {
        let mut m = WheelOdomMeasurement::new(
            [0.0; 3],
            [[4e-4, 1e-4, 0.0], [1e-4, 1e-4, 0.0], [0.0, 0.0, 1e-2]],
        );
        let info = m.information().unwrap();
        let det = 4e-4 * 1e-4 - 1e-8;
        assert!((info[0][0] - 1e-4 / det).abs() < 1e-6);
        assert!((info[0][1] + 1e-4 / det).abs() < 1e-6);
        assert!((info[5][5] - 100.0).abs() < 1e-9);
        assert_eq!(info[2][2], 1e4);
        assert_eq!(info[3][3], 1e4);
        assert_eq!(info[0][5], 0.0);
        m.covariance[2][2] = 0.0;
        assert!(m.information().is_none());
        m.covariance[2][2] = 1e-2;
        // Indefinite with a positive determinant.
        let good = m.covariance;
        m.covariance = [[-1e-4, 0.0, 0.0], [0.0, -1e-4, 0.0], [0.0, 0.0, 1e-4]];
        assert!(m.information().is_none());
        m.covariance = good;
        m.covariance[0][1] = 2e-4;
        assert!(m.information().is_none());
        m.covariance = good;
        m.tilt_sigma = 0.0;
        assert!(m.information().is_none());
    }

    /// `rᵀΩr` must be the Mahalanobis distance of the planar error `δz`
    /// under `covariance` (written in base-i axes) even after a turn, where
    /// the residual's translation sits in base-j axes. The covariance is
    /// tight forward, loose sideways, with the y–θ correlation `compose`
    /// creates; at 90° an unrotated information swaps the two axes.
    #[test]
    fn chi2_is_the_planar_mahalanobis_distance_after_a_turn() {
        let covariance = [[1e-6, 0.0, 0.0], [0.0, 1e-2, 2e-4], [0.0, 2e-4, 1e-4]];
        let sigma_inv = Mat3F64::from_cols_array(&[
            covariance[0][0],
            covariance[1][0],
            covariance[2][0],
            covariance[0][1],
            covariance[1][1],
            covariance[2][1],
            covariance[0][2],
            covariance[1][2],
            covariance[2][2],
        ])
        .inverse();
        for dtheta in [std::f64::consts::FRAC_PI_2, 0.7, -2.5, 0.0] {
            let delta = [0.3, 0.1, dtheta];
            let factor = WheelOdomFactor {
                from_idx: 0,
                to_idx: 1,
                measurement: WheelOdomMeasurement::new(delta, covariance),
                t_cb_from: Pose3d::IDENTITY,
                t_cb_to: Pose3d::IDENTITY,
                huber_chi2: None,
            };
            let info = factor.measurement.information().unwrap();
            for dz in [
                [1e-4, 0.0, 0.0],
                [0.0, 1e-4, 0.0],
                [0.0, 0.0, 1e-4],
                [3e-5, -7e-5, 2e-5],
            ] {
                // Base j in base i is the measurement moved by δz.
                let truth = WheelOdomMeasurement::new(
                    [delta[0] + dz[0], delta[1] + dz[1], delta[2] + dz[2]],
                    covariance,
                )
                .delta_pose();
                let (r, _) = residual_and_jacobian(&Pose3d::IDENTITY, &truth.inverse(), &factor);
                let chi2 = quadratic_form(&info, &r);
                let dz_v = Vec3F64::new(dz[0], dz[1], dz[2]);
                let expected = dz_v.dot(sigma_inv * dz_v);
                assert!(
                    (chi2 - expected).abs() <= 1e-3 * expected,
                    "dθ={dtheta} δz={dz:?}: chi2 {chi2} vs Mahalanobis {expected}"
                );
            }
        }
    }

    #[test]
    fn a_non_finite_factor_is_invalid() {
        let good = WheelOdomFactor {
            from_idx: 0,
            to_idx: 1,
            measurement: WheelOdomMeasurement::new(
                [0.1, 0.0, 0.1],
                [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
            ),
            t_cb_from: Pose3d::IDENTITY,
            t_cb_to: Pose3d::IDENTITY,
            huber_chi2: None,
        };
        assert!(good.is_valid());
        let mut f = good;
        f.measurement.delta[0] = f64::NAN;
        assert!(f.measurement.information().is_none());
        assert!(!f.is_valid());
        let mut f = good;
        f.t_cb_from.translation.y = f64::NAN;
        assert!(!f.is_valid());
        let mut f = good;
        f.t_cb_to.rotation = Mat3F64::from_cols_array(&[f64::INFINITY; 9]);
        assert!(!f.is_valid());
        let mut f = good;
        f.measurement.covariance = [[0.0; 3]; 3];
        assert!(!f.is_valid());
        let mut f = good;
        f.measurement.covariance[0][2] = f64::NAN;
        f.measurement.covariance[2][0] = f64::NAN;
        assert!(!f.is_valid());
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut f = good;
            f.measurement.z_sigma = sigma;
            assert!(!f.is_valid(), "z_sigma {sigma}");
            let mut f = good;
            f.measurement.tilt_sigma = sigma;
            assert!(!f.is_valid(), "tilt_sigma {sigma}");
        }
        for k in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut f = good;
            f.huber_chi2 = Some(k);
            assert!(!f.is_valid(), "huber {k}");
        }
        let mut f = good;
        f.huber_chi2 = Some(12.59);
        assert!(f.is_valid());
        let mut f = good;
        f.to_idx = f.from_idx;
        assert!(!f.is_valid(), "a self-loop");
    }

    #[test]
    fn huber_downweights_only_past_the_threshold() {
        let info = WheelOdomMeasurement::new(
            [0.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .information()
        .unwrap();
        let r = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(robust_weight_and_cost(&r, &info, None), (1.0, 8.0));
        assert_eq!(robust_weight_and_cost(&r, &info, Some(25.0)), (1.0, 8.0));
        // ½ρ(16) with k = 4: √64 − 2.
        assert_eq!(robust_weight_and_cost(&r, &info, Some(4.0)), (0.5, 6.0));
    }

    /// The robust cost's gradient in `r` is the weighted `w Ω r` the normal
    /// equations use, inside and past the Huber threshold.
    ///
    /// Mutation: the old `½ w rᵀΩr` cost turned this red past the threshold.
    #[test]
    fn robust_cost_gradient_is_the_weighted_gradient() {
        let mut rng = Lcg(11);
        let info = random_factor(&mut rng).measurement.information().unwrap();
        for scale in [1e-3, 0.05, 1.0] {
            let r: [f64; 6] = std::array::from_fn(|_| rng.next() * scale);
            for k in [None, Some(12.59), Some(1e-3)] {
                let (w, _) = robust_weight_and_cost(&r, &info, k);
                for a in 0..6 {
                    let h = 1e-7 * scale.max(1e-3);
                    let (mut rp, mut rm) = (r, r);
                    rp[a] += h;
                    rm[a] -= h;
                    let numeric = (robust_weight_and_cost(&rp, &info, k).1
                        - robust_weight_and_cost(&rm, &info, k).1)
                        / (2.0 * h);
                    let analytic = w * (0..6).map(|b| info[a][b] * r[b]).sum::<f64>();
                    assert!(
                        (numeric - analytic).abs() <= 1e-5 * analytic.abs().max(1.0),
                        "scale {scale} k {k:?} row {a}: {numeric} vs {analytic}"
                    );
                }
            }
        }
    }

    #[test]
    fn composition_matches_the_chained_motion() {
        let mut rng = Lcg(7);
        for _ in 0..20 {
            let mut first = random_factor(&mut rng);
            let mut next = random_factor(&mut rng);
            next.from_idx = first.to_idx;
            next.to_idx = first.to_idx + 1;
            // One keyframe, one extrinsic; the composite never reads it.
            next.t_cb_from = first.t_cb_to;
            first.huber_chi2 = Some(12.59);
            let joined = first.compose(&next).unwrap();
            assert_eq!(
                (joined.from_idx, joined.to_idx),
                (first.from_idx, next.to_idx)
            );
            assert_eq!(joined.t_cb_from, first.t_cb_from);
            assert_eq!(joined.t_cb_to, next.t_cb_to);
            assert_eq!(joined.huber_chi2, Some(12.59));
            assert!(joined.measurement.information().is_some());

            // Poses consistent with both halves are consistent with the composite.
            let pose_i = rng.pose(1.0, 0.5);
            let chain = |pose: &Pose3d, f: &WheelOdomFactor| {
                let t_wb = pose.inverse().compose(&f.t_cb_from);
                f.t_cb_to
                    .compose(&t_wb.compose(&f.measurement.delta_pose()).inverse())
            };
            let pose_k = chain(&chain(&pose_i, &first), &next);
            let (r, _) = residual_and_jacobian(&pose_i, &pose_k, &joined);
            assert!(r.iter().all(|x| x.abs() < 1e-9), "{r:?}");
        }
        let mut rng = Lcg(8);
        let (a, b) = (random_factor(&mut rng), random_factor(&mut rng));
        assert!(a.compose(&WheelOdomFactor { from_idx: 5, ..b }).is_none());
    }

    #[test]
    fn composed_covariance_is_the_linearized_propagation() {
        let first = WheelOdomMeasurement::new(
            [0.3, -0.1, 0.7],
            [[2e-4, 1e-5, 0.0], [1e-5, 1e-4, 2e-5], [0.0, 2e-5, 3e-4]],
        );
        let next = WheelOdomMeasurement::new(
            [0.2, 0.15, -0.4],
            [[1e-4, 0.0, 1e-5], [0.0, 4e-4, 0.0], [1e-5, 0.0, 2e-4]],
        );
        // Central differences of the composed delta wrt each input delta.
        const EPS: f64 = 1e-6;
        let jacobian = |which: usize| {
            let mut j = [[0.0; 3]; 3];
            for k in 0..3 {
                let shifted = |sign: f64| {
                    let (mut a, mut b) = (first, next);
                    let d = if which == 0 {
                        &mut a.delta
                    } else {
                        &mut b.delta
                    };
                    d[k] += sign * EPS;
                    a.compose(&b).delta
                };
                let (plus, minus) = (shifted(1.0), shifted(-1.0));
                for r in 0..3 {
                    j[r][k] = (plus[r] - minus[r]) / (2.0 * EPS);
                }
            }
            j
        };
        let a = congruence(&jacobian(0), &first.covariance);
        let b = congruence(&jacobian(1), &next.covariance);
        let joined = first.compose(&next);
        for r in 0..3 {
            for c in 0..3 {
                let want = a[r][c] + b[r][c];
                assert!((joined.covariance[r][c] - want).abs() < 1e-12, "[{r}][{c}]");
            }
        }
        assert!((joined.z_sigma - 0.01 * 2f64.sqrt()).abs() < 1e-15);
        assert!((joined.tilt_sigma - 0.01 * 2f64.sqrt()).abs() < 1e-15);
    }
}
