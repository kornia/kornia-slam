//! The f64 SE(3) pieces the factors share, over the tangent `[ρ; ω]` with
//! `T = Exp(ρ, ω)` perturbed on the right, `T ← T · Exp(ξ)`.
//!
//! kornia-algebra's SE(3) is f32-only, which cannot reach the 1e-6 Jacobian
//! check, so these live here until it gains an f64 SE(3).

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

/// 6×6 row-major block over the pose tangent `[ρ; ω]`.
pub type Mat6 = [[f64; 6]; 6];

/// `r = Log(Δ⁻¹ · T_cb_from⁻¹ · T_cw_i · T_cw_j⁻¹ · T_cb_to)` and its
/// Jacobian wrt keyframe i's pose, `J_l⁻¹(r) · Ad(A)` with
/// `A = Δ⁻¹ · T_cb_from⁻¹ · T_cw_i`; keyframe j's is the negation. With
/// identity extrinsics and delta it is the zero-relative-pose residual
/// `Log(T_cw_i · T_cw_j⁻¹)` of [`crate::stationary`].
pub fn relative_pose_residual(
    pose_cw_i: &Pose3d,
    pose_cw_j: &Pose3d,
    t_cb_from: &Pose3d,
    t_cb_to: &Pose3d,
    delta: &Pose3d,
) -> ([f64; 6], Mat6) {
    let t_bw_i = t_cb_from.inverse().compose(pose_cw_i);
    let t_wb_j = pose_cw_j.inverse().compose(t_cb_to);
    let a = delta.inverse().compose(&t_bw_i);
    let r = log(&a.compose(&t_wb_j));
    (r, mat6_mul(&left_jacobian_inverse(&r), &adjoint(&a)))
}

/// `vᵀ M v`.
pub fn quadratic_form(m: &Mat6, v: &[f64; 6]) -> f64 {
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
fn log(t: &Pose3d) -> [f64; 6] {
    let omega = SO3F64::from_matrix(&t.rotation).log();
    let rho = SO3F64::left_jacobian(omega).inverse() * t.translation;
    [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z]
}

/// `Ad(T) = [[R, t^R], [0, R]]` for the `[ρ; ω]` tangent order.
fn adjoint(t: &Pose3d) -> Mat6 {
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
fn left_jacobian_inverse(xi: &[f64; 6]) -> Mat6 {
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
    use crate::testing::{Lcg, exp, perturbed};

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
                pose_i.compose(&exp(&std::array::from_fn(|_| rng.next() * 3e-3)))
            };
            let res = |a: &Pose3d, b: &Pose3d| relative_pose_residual(a, b, &id, &id, &id);
            let (r, j_i) = res(&pose_i, &pose_j);
            let direct = log(&pose_i.compose(&pose_j.inverse()));
            for (a, b) in r.iter().zip(direct) {
                assert!((a - b).abs() < 1e-12);
            }
            let ad = adjoint(&pose_i);
            let expected = mat6_mul(&left_jacobian_inverse(&r), &ad);
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
            let t = exp(&xi);
            let j_inv = left_jacobian_inverse(&xi);
            for k in 0..6 {
                let log_at = |e: f64| {
                    let mut d = [0.0; 6];
                    d[k] = e;
                    log(&exp(&d).compose(&t))
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
}
