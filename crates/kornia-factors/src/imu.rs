//! Preintegrated IMU factor between two keyframe states.
//!
//! Each state carries 15 DOF, `[ρ(3) | ω(3) | v(3) | b_g(3) | b_a(3)]`, with
//! the pose `T_cw` perturbed on the right. The residual follows Forster et al.
//! (TRO 2017) in the body frame `T_bw = T_bc · T_cw`:
//!
//! ```text
//!     r_R  = Log(ΔR(b_i)ᵀ · R_bw,i · R_bw,jᵀ)
//!     r_v  = R_bw,i · (v_j − v_i − g·Δt) − Δv(b_i)
//!     r_p  = R_bw,i · (p_j − p_i − v_i·Δt − ½g·Δt²) − Δp(b_i)
//!     r_bg = b_g,j − b_g,i
//!     r_ba = b_a,j − b_a,i
//! ```
//!
//! with `p = t_wb` the body origin in the world and `ΔR, Δv, Δp` corrected to
//! first order for the bias at `i`. The 15×15 matrices are row-major.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::PreintegratedImu;

use crate::robust::huber_weight_and_cost;
use crate::state::NavState;

/// A row-major 15×15 matrix.
pub type Mat15 = [f64; 225];

/// An IMU constraint between keyframes `from_idx` and `to_idx`, usually
/// consecutive; any pair works as long as `preintegrated` covers the interval.
///
/// `#[non_exhaustive]`: build it with [`Self::new`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ImuFactor {
    pub from_idx: usize,
    pub to_idx: usize,
    /// Preintegrated measurements between the two keyframes.
    pub preintegrated: PreintegratedImu,
}

impl ImuFactor {
    /// A factor between keyframes `from_idx` and `to_idx`.
    pub fn new(from_idx: usize, to_idx: usize, preintegrated: PreintegratedImu) -> Self {
        Self {
            from_idx,
            to_idx,
            preintegrated,
        }
    }
}

/// An IMU factor's information matrix.
#[derive(Debug, Clone, Copy)]
pub struct ImuInformation {
    pub omega: Mat15,
    /// The 9×9 navigation covariance was singular, so the navigation block
    /// holds the inverse of its diagonal instead of its full inverse.
    pub diagonal_fallback: bool,
}

/// `(R_bw, t_wb)` of a world-to-camera pose, with the camera-to-body
/// extrinsic `T_bc` (`X_body = T_bc · X_cam`) when given: `R_bw` rotates
/// world vectors into the body frame, `t_wb` is the body origin in the world.
pub fn body_frame(pose_cw: &Pose3d, t_bc: Option<&Pose3d>) -> (Mat3F64, Vec3F64) {
    let r_cw = &pose_cw.rotation;
    let t_cw = &pose_cw.translation;
    match t_bc {
        None => {
            let twb = -(Mat3F64(*r_cw.transpose()) * *t_cw);
            (*r_cw, twb)
        }
        Some(t_bc) => {
            let r_bc = &t_bc.rotation;
            let r_bw = mat3_mul(r_bc, r_cw);
            let t_bw = *r_bc * *t_cw + t_bc.translation;
            let twb = -(Mat3F64(*r_bw.transpose()) * t_bw);
            (r_bw, twb)
        }
    }
}

/// The residual `[r_R, r_v, r_p, r_bg, r_ba]` between states `i` and `j`
/// and its Jacobians with respect to each state's 15 DOF.
///
/// A right perturbation of `T_cw` is the same right perturbation of `T_bw`
/// (`T_bc` is fixed), so the pose columns need no extrinsic term.
pub fn residual_and_jacobians(
    i: NavState<'_>,
    j: NavState<'_>,
    pim: &PreintegratedImu,
    gravity: &Vec3F64,
    t_bc: Option<&Pose3d>,
) -> ([f64; 15], Mat15, Mat15) {
    let dt = pim.dt;

    let (r_bw_i, twb_i) = body_frame(i.pose_cw, t_bc);
    let (r_bw_j, twb_j) = body_frame(j.pose_cw, t_bc);
    let r_wb_j = Mat3F64(*r_bw_j.transpose());

    let v_i = &i.velocity;
    let v_j = &j.velocity;

    let dr_corrected = pim.delta_rotation_with_bias(&i.bias);
    let dv_corrected = pim.delta_velocity_with_bias(&i.bias);
    let dp_corrected = pim.delta_position_with_bias(&i.bias);

    let dr_t = Mat3F64(*dr_corrected.transpose());
    let lhs_r = Mat3F64(dr_t.mul_mat3(&mat3_mul(&r_bw_i, &r_wb_j)));
    let r_rot = SO3F64::from_matrix(&lhs_r).log();

    let dv_world = *v_j - *v_i - *gravity * dt;
    let r_vel = r_bw_i * dv_world - dv_corrected;

    let dp_world = twb_j - twb_i - *v_i * dt - *gravity * (0.5 * dt * dt);
    let r_pos = r_bw_i * dp_world - dp_corrected;

    let r_bg = j.bias.gyro - i.bias.gyro;
    let r_ba = j.bias.accel - i.bias.accel;

    let mut residual = [0.0f64; 15];
    residual[0..3].copy_from_slice(&[r_rot.x, r_rot.y, r_rot.z]);
    residual[3..6].copy_from_slice(&[r_vel.x, r_vel.y, r_vel.z]);
    residual[6..9].copy_from_slice(&[r_pos.x, r_pos.y, r_pos.z]);
    residual[9..12].copy_from_slice(&[r_bg.x, r_bg.y, r_bg.z]);
    residual[12..15].copy_from_slice(&[r_ba.x, r_ba.y, r_ba.z]);

    // Under T_cw ← T_cw · Exp([ρ; ω]): R_bw ← R_bw · Exp(ω) and
    // t_wb ← t_wb − ρ + [t_wb]× · ω; velocity and biases are additive.
    let mut j_i = [0.0f64; 225];
    let mut j_j = [0.0f64; 225];

    let jr_inv = SO3F64::right_jacobian(r_rot).inverse();

    let jr_inv_rbwj = mat3_mul(&jr_inv, &r_bw_j);
    set_block_15x15(&mut j_i, 0, 3, &jr_inv_rbwj);
    // eR = ΔRᵀ · R_bw,i · R_wb,j = lhs_r.
    let er_t = Mat3F64(*lhs_r.transpose());
    set_block_15x15(
        &mut j_i,
        0,
        9,
        &mat3_neg(mat3_mul(
            &mat3_mul(&jr_inv, &er_t),
            &pim.d_rotation_d_bias_gyro,
        )),
    );
    set_block_15x15(&mut j_j, 0, 3, &mat3_neg(jr_inv_rbwj));

    set_block_15x15(
        &mut j_i,
        3,
        3,
        &mat3_neg(mat3_mul(&r_bw_i, &SO3F64::hat(dv_world))),
    );
    let neg_r_bw_i = mat3_neg(r_bw_i);
    set_block_15x15(&mut j_i, 3, 6, &neg_r_bw_i);
    set_block_15x15(&mut j_i, 3, 9, &mat3_neg(pim.d_velocity_d_bias_gyro));
    set_block_15x15(&mut j_i, 3, 12, &mat3_neg(pim.d_velocity_d_bias_accel));
    set_block_15x15(&mut j_j, 3, 6, &r_bw_i);

    set_block_15x15(&mut j_i, 6, 0, &r_bw_i);
    // ω_i rotates R_bw,i and moves t_wb,i; together −R_bw,i · [t_wb,j − v_i·Δt − ½g·Δt²]×.
    let dp_omega_world = twb_j - *v_i * dt - *gravity * (0.5 * dt * dt);
    set_block_15x15(
        &mut j_i,
        6,
        3,
        &mat3_neg(mat3_mul(&r_bw_i, &SO3F64::hat(dp_omega_world))),
    );
    set_block_15x15(&mut j_i, 6, 6, &mat3_scalar_f64(&r_bw_i, -dt));
    set_block_15x15(&mut j_i, 6, 9, &mat3_neg(pim.d_position_d_bias_gyro));
    set_block_15x15(&mut j_i, 6, 12, &mat3_neg(pim.d_position_d_bias_accel));
    set_block_15x15(&mut j_j, 6, 0, &neg_r_bw_i);
    set_block_15x15(&mut j_j, 6, 3, &mat3_mul(&r_bw_i, &SO3F64::hat(twb_j)));

    set_block_15x15(&mut j_i, 9, 9, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 9, 9, &Mat3F64::IDENTITY);
    set_block_15x15(&mut j_i, 12, 12, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 12, 12, &Mat3F64::IDENTITY);

    (residual, j_i, j_j)
}

/// Huber weight and cost of one IMU factor. The 9-dim navigation block
/// (rows/cols 0–8) is robustified on its Mahalanobis chi²; the bias random
/// walk (9–14) stays quadratic, as ORB-SLAM3 keeps it a separate edge.
/// `omega_raw` is block-diagonal between the two (see [`information`]) and
/// symmetric. The caller scales only the navigation block of omega by the
/// returned weight.
pub fn weight_and_cost(res: &[f64; 15], omega_raw: &Mat15, chi2_threshold: f64) -> (f64, f64) {
    let block_chi2 = |range: std::ops::Range<usize>| {
        let mut chi2 = 0.0f64;
        for i in range.clone() {
            for j in range.clone() {
                chi2 += res[i] * omega_raw[idx15(i, j)] * res[j];
            }
        }
        chi2
    };
    let (hw, nav_cost) = huber_weight_and_cost(block_chi2(0..9), Some(chi2_threshold));
    (hw, nav_cost + 0.5 * block_chi2(9..15))
}

/// `weight` times the full inverse of the 9×9 navigation covariance (its
/// diagonal inverse if singular) and the diagonal bias random walk.
pub fn information(pim: &PreintegratedImu, weight: f64) -> ImuInformation {
    let mut omega = [0.0f64; 225];

    // Preintegration correlates rotation, velocity and position errors (gyro
    // drives all three, accel the last two), so the diagonal alone is wrong.
    let nav_inverse = invert_9x9_f64(&pim.covariance);
    if let Some(nav_inv) = &nav_inverse {
        for r in 0..9 {
            for c in 0..9 {
                // `nav_inv` is column-major, like the covariance it inverts.
                omega[idx15(r, c)] = weight * nav_inv[c * 9 + r];
            }
        }
    } else {
        for i in 0..9 {
            let var = pim.covariance[i * 10];
            omega[idx15(i, i)] = weight * if var > 1e-20 { 1.0 / var } else { 1e6 };
        }
    }

    for i in 0..6 {
        let var = pim.bias_covariance[i * 7];
        omega[idx15(9 + i, 9 + i)] = weight * if var > 1e-20 { 1.0 / var } else { 1e6 };
    }
    ImuInformation {
        omega,
        diagonal_fallback: nav_inverse.is_none(),
    }
}

/// Inverts a column-major 9×9 matrix by Gauss-Jordan elimination with
/// partial pivoting.
fn invert_9x9_f64(cm: &[f64; 81]) -> Option<[f64; 81]> {
    let mut a = [0.0f64; 81];
    let mut inv = [0.0f64; 81];
    for r in 0..9 {
        for c in 0..9 {
            a[r * 9 + c] = cm[c * 9 + r];
        }
        inv[r * 9 + r] = 1.0;
    }
    for col in 0..9 {
        let mut max_val = a[col * 9 + col].abs();
        let mut max_row = col;
        for row in col + 1..9 {
            let v = a[row * 9 + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-20 {
            return None;
        }
        if max_row != col {
            for c in 0..9 {
                a.swap(col * 9 + c, max_row * 9 + c);
                inv.swap(col * 9 + c, max_row * 9 + c);
            }
        }
        let pivot = a[col * 9 + col];
        for c in 0..9 {
            a[col * 9 + c] /= pivot;
            inv[col * 9 + c] /= pivot;
        }
        for row in 0..9 {
            if row == col {
                continue;
            }
            let factor = a[row * 9 + col];
            for c in 0..9 {
                let da = a[col * 9 + c];
                let di = inv[col * 9 + c];
                a[row * 9 + c] -= factor * da;
                inv[row * 9 + c] -= factor * di;
            }
        }
    }
    let mut result = [0.0f64; 81];
    for r in 0..9 {
        for c in 0..9 {
            result[c * 9 + r] = inv[r * 9 + c];
        }
    }
    Some(result)
}

#[inline]
const fn idx15(row: usize, col: usize) -> usize {
    row * 15 + col
}

#[inline]
fn set_block_15x15(m: &mut Mat15, row: usize, col: usize, block: &Mat3F64) {
    let cols = block.to_cols_array();
    for c in 0..3 {
        for r in 0..3 {
            m[idx15(row + r, col + c)] = cols[c * 3 + r];
        }
    }
}

#[inline]
fn mat3_mul(a: &Mat3F64, b: &Mat3F64) -> Mat3F64 {
    Mat3F64(a.mul_mat3(b))
}

#[inline]
fn mat3_neg(a: Mat3F64) -> Mat3F64 {
    mat3_scalar_f64(&a, -1.0)
}

#[inline]
fn mat3_scalar_f64(m: &Mat3F64, s: f64) -> Mat3F64 {
    Mat3F64(m.mul_scalar(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Lcg, perturbed};
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement};

    fn preintegrated(rng: &mut Lcg) -> PreintegratedImu {
        let calib = ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3.0e-3,
        };
        let bias = ImuBias {
            gyro: rng.vec3(0.01),
            accel: rng.vec3(0.05),
        };
        let mut pim = PreintegratedImu::new(bias, calib);
        for k in 0..40 {
            let measurement = ImuMeasurement {
                timestamp: k as f64 * 0.005,
                gyro: rng.vec3(0.8),
                accel: rng.vec3(2.0) + Vec3F64::new(0.0, 0.0, 9.81),
            };
            pim.integrate(&measurement, 0.005);
        }
        pim
    }

    /// Every column of both Jacobians matches central differences of the
    /// residual: poses perturbed on the right, velocity and biases additively,
    /// with the bias at `i` away from the linearization bias.
    #[test]
    fn jacobians_match_central_differences() {
        let mut rng = Lcg(7);
        let pim = preintegrated(&mut rng);
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let t_bc = rng.pose(0.1, 0.3);
        let poses = [rng.pose(2.0, 1.0), rng.pose(2.0, 1.0)];
        let velocities = [rng.vec3(1.0), rng.vec3(1.0)];
        let biases = [
            ImuBias {
                gyro: pim.bias.gyro + rng.vec3(1e-3),
                accel: pim.bias.accel + rng.vec3(1e-2),
            },
            ImuBias {
                gyro: rng.vec3(0.01),
                accel: rng.vec3(0.05),
            },
        ];
        let eval = |poses: &[Pose3d; 2], velocities: &[Vec3F64; 2], biases: &[ImuBias; 2]| {
            let state = |k: usize| NavState {
                pose_cw: &poses[k],
                velocity: velocities[k],
                bias: biases[k],
            };
            residual_and_jacobians(state(0), state(1), &pim, &gravity, Some(&t_bc))
        };
        let (_, j_i, j_j) = eval(&poses, &velocities, &biases);
        let unit = |k: usize, eps: f64| {
            let mut e = [0.0; 3];
            e[k] = eps;
            Vec3F64::new(e[0], e[1], e[2])
        };

        let h = 1e-6;
        for (side, jacobian) in [(0, &j_i), (1, &j_j)] {
            for col in 0..15 {
                let shifted = |eps: f64| {
                    let (mut p, mut v, mut b) = (poses, velocities, biases);
                    match col {
                        0..6 => p[side] = perturbed(&p[side], col, eps),
                        6..9 => v[side] += unit(col - 6, eps),
                        9..12 => b[side].gyro += unit(col - 9, eps),
                        _ => b[side].accel += unit(col - 12, eps),
                    }
                    eval(&p, &v, &b).0
                };
                let (up, down) = (shifted(h), shifted(-h));
                for row in 0..15 {
                    let numeric = (up[row] - down[row]) / (2.0 * h);
                    let analytic = jacobian[idx15(row, col)];
                    // ΔR's bias correction is first order: its Jacobian holds
                    // at the linearization bias, to O(δb_g) away from it.
                    let tol = if (9..12).contains(&col) { 1e-4 } else { 1e-6 };
                    assert!(
                        (numeric - analytic).abs() < tol * analytic.abs().max(1.0),
                        "state {side}, row {row}, col {col}: numeric {numeric} vs analytic {analytic}"
                    );
                }
            }
        }
    }

    /// The information matrix inverts the navigation covariance:
    /// `Ω_nav · Σ_nav = weight · I`.
    #[test]
    fn information_inverts_navigation_covariance() {
        let pim = preintegrated(&mut Lcg(3));
        let info = information(&pim, 2.0);
        assert!(!info.diagonal_fallback);
        for r in 0..9 {
            for c in 0..9 {
                let product: f64 = (0..9)
                    .map(|k| info.omega[idx15(r, k)] * pim.covariance[c * 9 + k])
                    .sum();
                let expected = if r == c { 2.0 } else { 0.0 };
                assert!((product - expected).abs() < 1e-6, "({r}, {c}): {product}");
            }
        }
    }

    /// The cost has the gradient the normal equations use: `w Ω r` on the
    /// Huber-robustified nav rows 0-8, `Ω r` on the quadratic bias rows
    /// 9-14, inside and past the threshold.
    #[test]
    fn cost_gradient_is_the_weighted_gradient() {
        let mut omega = [0.0f64; 225];
        for i in 0..9 {
            for j in 0..9 {
                omega[i * 15 + j] = if i == j {
                    4.0 + i as f64
                } else {
                    0.3 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        for i in 9..15 {
            omega[i * 15 + i] = 50.0 + 10.0 * i as f64;
        }
        let k = 16.9;
        for scale in [0.05, 3.0] {
            let res: [f64; 15] = std::array::from_fn(|i| {
                scale * (0.4 + 0.1 * i as f64) * if i % 2 == 0 { 1.0 } else { -1.0 }
            });
            let (w, _) = weight_and_cost(&res, &omega, k);
            assert!(
                if scale > 1.0 { w < 1.0 } else { w == 1.0 },
                "{scale}: w = {w}"
            );
            for d in 0..15 {
                let h = 1e-6;
                let (mut up, mut down) = (res, res);
                up[d] += h;
                down[d] -= h;
                let numeric = (weight_and_cost(&up, &omega, k).1
                    - weight_and_cost(&down, &omega, k).1)
                    / (2.0 * h);
                let omega_r: f64 = (0..15).map(|m| omega[d * 15 + m] * res[m]).sum();
                let analytic = if d < 9 { w * omega_r } else { omega_r };
                assert!(
                    (numeric - analytic).abs() <= 1e-5 * analytic.abs().max(1.0),
                    "scale {scale}, row {d}: {numeric} vs {analytic}"
                );
            }
        }
    }
}
