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
//! first order for the bias at `i`.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Mat15F64, SO3F64, Vec3F64, Vec15F64};
use kornia_sensors::imu::PreintegratedImu;

use crate::inertial::NavState;
use kornia_3d::ransac::{HuberKernel, RobustKernel};

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
    pub omega: Mat15F64,
    /// The 9×9 navigation covariance was not positive definite, so the
    /// navigation block holds the inverse of its diagonal instead of its full
    /// inverse.
    pub diagonal_fallback: bool,
}

/// `(R_bw, t_wb)` of a world-to-camera pose, with the camera-to-body
/// extrinsic `T_bc` (`X_body = T_bc · X_cam`) when given: `R_bw` rotates
/// world vectors into the body frame, `t_wb` is the body origin in the world.
pub fn body_frame(pose_cw: &Pose3d, t_bc: Option<&Pose3d>) -> (Mat3F64, Vec3F64) {
    let pose_bw = t_bc.map_or(*pose_cw, |t_bc| t_bc.compose(pose_cw));
    (pose_bw.rotation, pose_bw.inverse().translation)
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
) -> (Vec15F64, Mat15F64, Mat15F64) {
    let dt = pim.dt;

    let (r_bw_i, twb_i) = body_frame(i.pose_cw, t_bc);
    let (r_bw_j, twb_j) = body_frame(j.pose_cw, t_bc);
    let r_wb_j = r_bw_j.transpose();

    let v_i = &i.velocity;
    let v_j = &j.velocity;

    let dr_corrected = pim.delta_rotation_with_bias(&i.bias);
    let dv_corrected = pim.delta_velocity_with_bias(&i.bias);
    let dp_corrected = pim.delta_position_with_bias(&i.bias);

    let dr_t = dr_corrected.transpose();
    let lhs_r = dr_t * (r_bw_i * r_wb_j);
    let r_rot = SO3F64::from_matrix(&lhs_r).log();

    let dv_world = *v_j - *v_i - *gravity * dt;
    let r_vel = r_bw_i * dv_world - dv_corrected;

    let dp_world = twb_j - twb_i - *v_i * dt - *gravity * (0.5 * dt * dt);
    let r_pos = r_bw_i * dp_world - dp_corrected;

    let r_bg = j.bias.gyro - i.bias.gyro;
    let r_ba = j.bias.accel - i.bias.accel;

    let mut residual = Vec15F64::zeros();
    for (k, r) in [r_rot, r_vel, r_pos, r_bg, r_ba].into_iter().enumerate() {
        residual.fixed_rows_mut::<3>(3 * k).copy_from(&r.into());
    }

    // Under T_cw ← T_cw · Exp([ρ; ω]): R_bw ← R_bw · Exp(ω) and
    // t_wb ← t_wb − ρ + [t_wb]× · ω; velocity and biases are additive.
    let mut j_i = Mat15F64::zeros();
    let mut j_j = Mat15F64::zeros();

    let jr_inv = SO3F64::right_jacobian(r_rot).inverse();

    let jr_inv_rbwj = jr_inv * r_bw_j;
    j_i.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&jr_inv_rbwj.into());
    // eR = ΔRᵀ · R_bw,i · R_wb,j = lhs_r.
    let er_t = lhs_r.transpose();
    j_i.fixed_view_mut::<3, 3>(0, 9)
        .copy_from(&(jr_inv * er_t * pim.d_rotation_d_bias_gyro * -1.0).into());
    j_j.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(jr_inv_rbwj * -1.0).into());

    j_i.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(r_bw_i * SO3F64::hat(dv_world) * -1.0).into());
    let neg_r_bw_i = r_bw_i * -1.0;
    j_i.fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&neg_r_bw_i.into());
    j_i.fixed_view_mut::<3, 3>(3, 9)
        .copy_from(&(pim.d_velocity_d_bias_gyro * -1.0).into());
    j_i.fixed_view_mut::<3, 3>(3, 12)
        .copy_from(&(pim.d_velocity_d_bias_accel * -1.0).into());
    j_j.fixed_view_mut::<3, 3>(3, 6).copy_from(&r_bw_i.into());

    j_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_bw_i.into());
    // ω_i rotates R_bw,i and moves t_wb,i; together −R_bw,i · [t_wb,j − v_i·Δt − ½g·Δt²]×.
    let dp_omega_world = twb_j - *v_i * dt - *gravity * (0.5 * dt * dt);
    j_i.fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(r_bw_i * SO3F64::hat(dp_omega_world) * -1.0).into());
    j_i.fixed_view_mut::<3, 3>(6, 6)
        .copy_from(&(r_bw_i * -dt).into());
    j_i.fixed_view_mut::<3, 3>(6, 9)
        .copy_from(&(pim.d_position_d_bias_gyro * -1.0).into());
    j_i.fixed_view_mut::<3, 3>(6, 12)
        .copy_from(&(pim.d_position_d_bias_accel * -1.0).into());
    j_j.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&neg_r_bw_i.into());
    j_j.fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(r_bw_i * SO3F64::hat(twb_j)).into());

    for k in [9, 12] {
        j_i.fixed_view_mut::<3, 3>(k, k).fill_diagonal(-1.0);
        j_j.fixed_view_mut::<3, 3>(k, k).fill_diagonal(1.0);
    }

    (residual, j_i, j_j)
}

/// Huber weight and cost of one IMU factor. The 9-dim navigation block
/// (rows/cols 0–8) is robustified on its Mahalanobis chi²; the bias random
/// walk (9–14) stays quadratic, as ORB-SLAM3 keeps it a separate edge.
/// `omega_raw` is block-diagonal between the two (see [`information`]) and
/// symmetric. The caller scales only the navigation block of omega by the
/// returned weight.
pub fn weight_and_cost(res: &Vec15F64, omega_raw: &Mat15F64, chi2_threshold: f64) -> (f64, f64) {
    let nav = res.fixed_rows::<9>(0);
    let bias = res.fixed_rows::<6>(9);
    let nav_chi2 = nav.dot(&(omega_raw.fixed_view::<9, 9>(0, 0) * nav));
    let bias_chi2 = bias.dot(&(omega_raw.fixed_view::<6, 6>(9, 9) * bias));
    let (hw, nav_cost) = if chi2_threshold > 0.0 {
        (
            HuberKernel.weight(nav_chi2, chi2_threshold),
            0.5 * HuberKernel.rho(nav_chi2, chi2_threshold),
        )
    } else {
        (1.0, 0.5 * nav_chi2)
    };
    (hw, nav_cost + 0.5 * bias_chi2)
}

/// `weight` times the full inverse of the 9×9 navigation covariance (its
/// diagonal inverse if not positive definite) and the diagonal bias random
/// walk.
pub fn information(pim: &PreintegratedImu, weight: f64) -> ImuInformation {
    let mut omega = Mat15F64::zeros();

    // Preintegration correlates rotation, velocity and position errors (gyro
    // drives all three, accel the last two), so the diagonal alone is wrong.
    let nav_covariance = pim.covariance;
    let nav_inverse = nav_covariance.cholesky().map(|c| c.inverse());
    let mut nav = omega.fixed_view_mut::<9, 9>(0, 0);
    match &nav_inverse {
        Some(nav_inv) => nav.copy_from(&(nav_inv * weight)),
        None => {
            nav.set_diagonal(&nav_covariance.map_diagonal(|var| weight * inverse_variance(var)))
        }
    }

    for i in 0..6 {
        omega[(9 + i, 9 + i)] = weight * inverse_variance(pim.bias_covariance[(i, i)]);
    }
    ImuInformation {
        omega,
        diagonal_fallback: nav_inverse.is_none(),
    }
}

/// Covariance of state `j` given state `i`'s, through the edge's linearisation:
/// `J_j⁻¹ (J_i Σ_i J_iᵀ + Ω⁻¹) J_j⁻ᵀ`, layout `[ρ ω v b_g b_a]`. `None` if `J_j`
/// is singular.
///
/// The residual `r(x_i, x_j)` is zero up to its noise `n ~ N(0, Ω⁻¹)`, so to
/// first order `J_i δx_i + J_j δx_j = n`. `Ω⁻¹` is the preintegration's own
/// navigation and bias-walk covariance, block diagonal as in [`information`].
pub fn propagate_state_covariance(
    state_i: NavState<'_>,
    covariance_i: &Mat15F64,
    state_j: NavState<'_>,
    preintegrated: &PreintegratedImu,
    gravity: &Vec3F64,
    t_bc: Option<&Pose3d>,
) -> Option<Mat15F64> {
    let (_, j_i, j_j) = residual_and_jacobians(state_i, state_j, preintegrated, gravity, t_bc);
    let j_j_inverse = j_j.try_inverse()?;
    let mut noise = Mat15F64::zeros();
    noise
        .fixed_view_mut::<9, 9>(0, 0)
        .copy_from(&preintegrated.covariance);
    noise
        .fixed_view_mut::<6, 6>(9, 9)
        .copy_from(&preintegrated.bias_covariance);
    let propagated =
        j_j_inverse * (j_i * covariance_i * j_i.transpose() + noise) * j_j_inverse.transpose();
    let symmetric = (propagated + propagated.transpose()) * 0.5;
    symmetric.iter().all(|v| v.is_finite()).then_some(symmetric)
}

/// `1 / var`, or 1e6 for a (near-)zero variance.
fn inverse_variance(var: f64) -> f64 {
    if var > 1e-20 { 1.0 / var } else { 1e6 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Lcg, perturbed};
    use kornia_algebra::{SMatF64, Vec6F64};
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
                    let analytic = jacobian[(row, col)];
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

    /// Card A T1: a 5 s constant-velocity coast with EuRoC noise. The
    /// propagated body-position spread matches a Monte Carlo of the start
    /// state and the sensor noise within 10 %.
    #[test]
    fn propagated_position_spread_matches_monte_carlo() {
        let calib = ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3.0e-3,
        };
        let (dt, steps) = (0.005, 1000);
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let velocity = Vec3F64::new(1.0, 0.0, 0.0);
        let level = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81),
        };
        let mut nominal = PreintegratedImu::new(ImuBias::default(), calib);
        for _ in 0..steps {
            nominal.integrate(&level, dt);
        }
        let span = nominal.dt;
        let pose_i = Pose3d::IDENTITY;
        let position_j = velocity * span;
        let pose_j = Pose3d::new(Mat3F64::IDENTITY, position_j * -1.0);
        let sigma_sq = [1e-6, 1e-6, 1e-4, 1e-8, 1e-4];
        let mut covariance_i = Mat15F64::zeros();
        for k in 0..15 {
            covariance_i[(k, k)] = sigma_sq[k / 3];
        }
        let state = |pose_cw| NavState {
            pose_cw,
            velocity,
            bias: ImuBias::default(),
        };
        let covariance_j = propagate_state_covariance(
            state(&pose_i),
            &covariance_i,
            state(&pose_j),
            &nominal,
            &gravity,
            None,
        )
        .expect("an invertible edge");
        // t_wb ← t_wb − ρ + [t_wb]× ω under the right perturbation.
        let mut to_position = SMatF64::<3, 6>::zeros();
        to_position.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(-1.0);
        to_position
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SO3F64::hat(position_j).into());
        let pose_block: SMatF64<6, 6> = covariance_j.fixed_view::<6, 6>(0, 0).into_owned();
        let predicted = (to_position * pose_block * to_position.transpose()).trace();

        let mut rng = Lcg(11);
        let mut gaussian = || {
            let u1 = 0.5 * (rng.next() + 1.0);
            let u2 = 0.5 * (rng.next() + 1.0);
            (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let samples = 2000;
        let mut spread = 0.0;
        for _ in 0..samples {
            let mut draw = |k: usize| sigma_sq[k].sqrt() * gaussian();
            let rho = Vec3F64::new(draw(0), draw(0), draw(0));
            let omega = Vec3F64::new(draw(1), draw(1), draw(1));
            let start_velocity = velocity + Vec3F64::new(draw(2), draw(2), draw(2));
            let bias = ImuBias {
                gyro: Vec3F64::new(draw(3), draw(3), draw(3)),
                accel: Vec3F64::new(draw(4), draw(4), draw(4)),
            };
            let start = pose_i.compose(&crate::testing::exp(&Vec6F64::new(
                rho.x, rho.y, rho.z, omega.x, omega.y, omega.z,
            )));
            let (sigma_g, sigma_a) = (calib.gyro_noise / dt.sqrt(), calib.accel_noise / dt.sqrt());
            let mut pim = PreintegratedImu::new(bias, calib);
            for _ in 0..steps {
                let noisy = ImuMeasurement {
                    timestamp: 0.0,
                    gyro: Vec3F64::new(gaussian(), gaussian(), gaussian()) * sigma_g,
                    accel: level.accel + Vec3F64::new(gaussian(), gaussian(), gaussian()) * sigma_a,
                };
                pim.integrate(&noisy, dt);
            }
            let r_wb = start.rotation.transpose();
            let p_i = start.inverse().translation;
            let (_, _, p_j) = pim.predict(&r_wb, &start_velocity, &p_i, &gravity);
            let error = p_j - position_j;
            spread += error.dot(error);
        }
        let measured = spread / samples as f64;
        assert!(
            (predicted.sqrt() / measured.sqrt() - 1.0).abs() < 0.1,
            "predicted {:.4} m, Monte Carlo {:.4} m",
            predicted.sqrt(),
            measured.sqrt()
        );
    }

    /// E11: card A T1 on a rotated, turning rig with a camera-to-body
    /// extrinsic and an anisotropic start covariance. The propagated camera
    /// pose block, rotation included, matches a Monte Carlo of the start state
    /// and the sensor noise; T1's identity frames and position trace could not
    /// see a frame or sign error.
    #[test]
    fn propagated_camera_pose_spread_matches_monte_carlo_on_a_rotated_rig() {
        let calib = ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3.0e-3,
        };
        let (dt, steps) = (0.005, 200);
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let reading = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::new(0.1, -0.2, 0.3),
            accel: Vec3F64::new(0.4, 0.3, 9.7),
        };
        let mut nominal = PreintegratedImu::new(ImuBias::default(), calib);
        for _ in 0..steps {
            nominal.integrate(&reading, dt);
        }
        let t_bc = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(0.05, -0.02, 0.1),
        );
        let camera_of = |r_wb: &Mat3F64, p_wb: &Vec3F64| {
            let body_from_world = Pose3d::new(r_wb.transpose(), r_wb.transpose() * *p_wb * -1.0);
            t_bc.inverse().compose(&body_from_world)
        };
        let body_of = |pose_cw: &Pose3d| {
            let world_from_body = t_bc.compose(pose_cw).inverse();
            (world_from_body.rotation, world_from_body.translation)
        };
        let pose_i = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.4, 0.2, -1.1)).matrix(),
            Vec3F64::new(1.0, -2.0, 0.5),
        );
        let velocity = Vec3F64::new(0.8, -0.3, 0.1);
        let (r_wb_i, p_i) = body_of(&pose_i);
        let (r_wb_j, velocity_j, p_j) = nominal.predict(&r_wb_i, &velocity, &p_i, &gravity);
        let pose_j = camera_of(&r_wb_j, &p_j);
        // Milliradian attitude doubt dominates: through the specific force it
        // moves velocity and position along world directions a frame error
        // would turn.
        let sigma_sq = [
            1e-7, 4e-7, 0.5e-7, 4e-6, 1e-5, 2e-6, 1e-4, 0.3e-4, 2e-4, 1e-8, 0.3e-8, 2e-8, 1e-4,
            3e-4, 0.5e-4,
        ];
        let mut covariance_i = Mat15F64::zeros();
        for (k, variance) in sigma_sq.iter().enumerate() {
            covariance_i[(k, k)] = *variance;
        }
        let state = |pose_cw, velocity| NavState {
            pose_cw,
            velocity,
            bias: ImuBias::default(),
        };
        let covariance_j = propagate_state_covariance(
            state(&pose_i, velocity),
            &covariance_i,
            state(&pose_j, velocity_j),
            &nominal,
            &gravity,
            Some(&t_bc),
        )
        .expect("an invertible edge");

        let mut rng = Lcg(17);
        let mut gaussian = || {
            let u1 = 0.5 * (rng.next() + 1.0);
            let u2 = 0.5 * (rng.next() + 1.0);
            (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let samples = 1500;
        let mut measured = SMatF64::<9, 9>::zeros();
        for _ in 0..samples {
            let draw: Vec<f64> = sigma_sq.iter().map(|v| v.sqrt() * gaussian()).collect();
            let start = pose_i.compose(&crate::testing::exp(&Vec6F64::from_column_slice(
                &draw[..6],
            )));
            let start_velocity = velocity + Vec3F64::new(draw[6], draw[7], draw[8]);
            let bias = ImuBias {
                gyro: Vec3F64::new(draw[9], draw[10], draw[11]),
                accel: Vec3F64::new(draw[12], draw[13], draw[14]),
            };
            let (sigma_g, sigma_a) = (calib.gyro_noise / dt.sqrt(), calib.accel_noise / dt.sqrt());
            let mut pim = PreintegratedImu::new(bias, calib);
            for _ in 0..steps {
                let noisy = ImuMeasurement {
                    timestamp: 0.0,
                    gyro: reading.gyro + Vec3F64::new(gaussian(), gaussian(), gaussian()) * sigma_g,
                    accel: reading.accel
                        + Vec3F64::new(gaussian(), gaussian(), gaussian()) * sigma_a,
                };
                pim.integrate(&noisy, dt);
            }
            let (r_wb, p_wb) = body_of(&start);
            let (r, v, p) = pim.predict(&r_wb, &start_velocity, &p_wb, &gravity);
            let error = pose_j.inverse().compose(&camera_of(&r, &p));
            let omega = SO3F64::from_matrix(&error.rotation).log();
            let (rho, dv) = (error.translation, v - velocity_j);
            let delta = SMatF64::<9, 1>::from_column_slice(&[
                rho.x, rho.y, rho.z, omega.x, omega.y, omega.z, dv.x, dv.y, dv.z,
            ]);
            measured += delta * delta.transpose() / samples as f64;
        }
        let predicted: SMatF64<9, 9> = covariance_j.fixed_view::<9, 9>(0, 0).into_owned();
        for (name, base) in [("translation", 0), ("rotation", 3), ("velocity", 6)] {
            let block = |m: &SMatF64<9, 9>| m.fixed_view::<3, 3>(base, base).into_owned();
            let difference =
                (block(&measured) - block(&predicted)).norm() / block(&predicted).norm();
            assert!(
                difference < 0.15,
                "{name} block off by {difference:.3}: predicted {}, Monte Carlo {}",
                block(&predicted),
                block(&measured)
            );
        }
        // The cross terms carry the signs: whitened by the prediction, the
        // Monte Carlo covariance is the identity up to sampling noise (about
        // 0.03 an entry).
        let l_inverse = predicted
            .cholesky()
            .expect("a positive definite prediction")
            .l()
            .try_inverse()
            .expect("an invertible factor");
        let whitened = l_inverse * measured * l_inverse.transpose();
        let worst = (whitened - SMatF64::<9, 9>::identity()).abs().max();
        assert!(worst < 0.15, "whitened Monte Carlo off by {worst:.3}");
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
                    .map(|k| info.omega[(r, k)] * pim.covariance[(k, c)])
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
        let mut omega = Mat15F64::zeros();
        for i in 0..9 {
            for j in 0..9 {
                omega[(i, j)] = if i == j {
                    4.0 + i as f64
                } else {
                    0.3 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        for i in 9..15 {
            omega[(i, i)] = 50.0 + 10.0 * i as f64;
        }
        let k = 16.9;
        for scale in [0.05, 3.0] {
            let res = Vec15F64::from_fn(|i, _| {
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
                let omega_r = (omega * res)[d];
                let analytic = if d < 9 { w * omega_r } else { omega_r };
                assert!(
                    (numeric - analytic).abs() <= 1e-5 * analytic.abs().max(1.0),
                    "scale {scale}, row {d}: {numeric} vs {analytic}"
                );
            }
        }
    }
}
