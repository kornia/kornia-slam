//! The f64 SE(3) residual the pose-pair factors share, over the tangent
//! `[ρ; ω]` with `T = Exp(ρ, ω)` perturbed on the right, `T ← T · Exp(ξ)`.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat6F64, SE3F64, SO3F64, Vec6F64};

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
) -> (Vec6F64, Mat6F64) {
    let t_bw_i = t_cb_from.inverse().compose(pose_cw_i);
    let t_wb_j = pose_cw_j.inverse().compose(t_cb_to);
    let a = delta.inverse().compose(&t_bw_i);
    let (rho, omega) = to_se3(&a.compose(&t_wb_j)).log();
    let r = Vec6F64::new(rho.x, rho.y, rho.z, omega.x, omega.y, omega.z);
    let jacobian = SE3F64::left_jacobian_inverse(rho, omega) * to_se3(&a).adjoint();
    (r, jacobian)
}

fn to_se3(pose: &Pose3d) -> SE3F64 {
    SE3F64::new(SO3F64::from_matrix(&pose.rotation), pose.translation)
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
                pose_i.compose(&exp(&Vec6F64::from_fn(|_, _| rng.next() * 3e-3)))
            };
            let res = |a: &Pose3d, b: &Pose3d| relative_pose_residual(a, b, &id, &id, &id);
            let (r, j_i) = res(&pose_i, &pose_j);
            let (rho, omega) = to_se3(&pose_i.compose(&pose_j.inverse())).log();
            let direct = [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z];
            for (a, b) in r.iter().zip(direct) {
                assert!((a - b).abs() < 1e-12);
            }
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
                    assert!((j_i[(row, c)] - num_i).abs() <= 1e-6, "i ({row},{c})");
                    assert!((-j_i[(row, c)] - num_j).abs() <= 1e-6, "j ({row},{c})");
                }
            }
        }
    }
}
