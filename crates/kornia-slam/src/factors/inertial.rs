//! IMU preintegration factor (Forster et al. / ORB-SLAM3 formulation).
//!
//! Residual (9D) = [r_R(3), r_v(3), r_p(3)]:
//!   r_R = Log(ΔR_corr^T · R_i · R_j^T)
//!   r_v = R_i · (v_j − v_i − g·Δt) − Δv_corr
//!   r_p = R_i · (p_j − p_i − v_i·Δt − ½g·Δt²) − Δp_corr
//!
//! Pre-whitened by the Cholesky factor of the information matrix (inverse of
//! the 9×9 navigation covariance from preintegration).

use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};
use kornia_sensors::imu::PreintegratedImu;

/// Total tangent-space dimension across all five variables.
const TOTAL_DIM: usize = 24;

/// IMU preintegration factor connecting two consecutive keyframe states.
///
/// Variables:
///   0: pose_i  (SE3, params = [qw, qx, qy, qz, tx, ty, tz], tangent dim = 6)
///   1: vel_i   (R³,  params = [vx, vy, vz], tangent dim = 3)
///   2: pose_j  (SE3, params = [qw, qx, qy, qz, tx, ty, tz], tangent dim = 6)
///   3: vel_j   (R³,  params = [vx, vy, vz], tangent dim = 3)
///   4: bias    (R⁶,  params = [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z], tangent dim = 6)
pub struct InertialFactor {
    delta_rotation: [f32; 9], // 3×3 column-major
    delta_velocity: [f32; 3],
    delta_position: [f32; 3],
    dt: f32,
    gravity: [f32; 3],
    bias_gyro_lin: [f32; 3],
    bias_accel_lin: [f32; 3],
    j_r_gyro: [f32; 9],  // 3×3 column-major
    j_v_gyro: [f32; 9],
    j_v_accel: [f32; 9],
    j_p_gyro: [f32; 9],
    j_p_accel: [f32; 9],
    /// Upper-triangular sqrt of information matrix (9×9, row-major).
    sqrt_info: [f32; 81],
}

impl InertialFactor {
    /// Construct from a `PreintegratedImu` and the world-frame gravity vector.
    pub fn new(pre: &PreintegratedImu, gravity: [f32; 3]) -> Self {
        let convert_mat3 = |m: &kornia_algebra::Mat3F64| -> [f32; 9] {
            let c = m.to_cols_array();
            std::array::from_fn(|i| c[i] as f32)
        };
        let convert_vec3 = |v: &kornia_algebra::Vec3F64| -> [f32; 3] {
            [v.x as f32, v.y as f32, v.z as f32]
        };

        let sqrt_vec = super::sqrt_info_from_cov(&pre.covariance, 9);
        let mut sqrt_info = [0.0f32; 81];
        sqrt_info.copy_from_slice(&sqrt_vec);

        Self {
            delta_rotation: convert_mat3(&pre.delta_rotation),
            delta_velocity: convert_vec3(&pre.delta_velocity),
            delta_position: convert_vec3(&pre.delta_position),
            dt: pre.dt as f32,
            gravity,
            bias_gyro_lin: convert_vec3(&pre.bias.gyro),
            bias_accel_lin: convert_vec3(&pre.bias.accel),
            j_r_gyro: convert_mat3(&pre.j_r_gyro),
            j_v_gyro: convert_mat3(&pre.j_v_gyro),
            j_v_accel: convert_mat3(&pre.j_v_accel),
            j_p_gyro: convert_mat3(&pre.j_p_gyro),
            j_p_accel: convert_mat3(&pre.j_p_accel),
            sqrt_info,
        }
    }
}

impl Factor for InertialFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 5 {
            return Err(FactorError::DimensionMismatch {
                expected: 5,
                actual: params.len(),
            });
        }

        let pose_i = params[0]; // [qw,qx,qy,qz,tx,ty,tz]
        let vel_i = params[1];  // [vx,vy,vz]
        let pose_j = params[2];
        let vel_j = params[3];
        let bias = params[4];   // [bg(3), ba(3)]

        // ── Extract rotations and world positions ────────────────────────
        let r_i = quat_to_rot(pose_i);
        let t_i = [pose_i[4], pose_i[5], pose_i[6]];
        let r_it = m3t(&r_i);
        let p_i = m3v3(&r_it, &v3_neg(&t_i)); // p = -R^T·t

        let r_j = quat_to_rot(pose_j);
        let t_j = [pose_j[4], pose_j[5], pose_j[6]];
        let r_jt = m3t(&r_j);
        let p_j = m3v3(&r_jt, &v3_neg(&t_j));

        let vi = [vel_i[0], vel_i[1], vel_i[2]];
        let vj = [vel_j[0], vel_j[1], vel_j[2]];

        // ── Bias correction ──────────────────────────────────────────────
        let db_g = [
            bias[0] - self.bias_gyro_lin[0],
            bias[1] - self.bias_gyro_lin[1],
            bias[2] - self.bias_gyro_lin[2],
        ];
        let db_a = [
            bias[3] - self.bias_accel_lin[0],
            bias[4] - self.bias_accel_lin[1],
            bias[5] - self.bias_accel_lin[2],
        ];

        // ΔR_corr = ΔR · Exp(J_R_gyro · δb_g)
        let dr_corr = m3m3(&self.delta_rotation, &so3_exp(&m3v3(&self.j_r_gyro, &db_g)));
        // Δv_corr = Δv + J_v_gyro·δb_g + J_v_accel·δb_a
        let dv_corr = v3_add(
            &self.delta_velocity,
            &v3_add(&m3v3(&self.j_v_gyro, &db_g), &m3v3(&self.j_v_accel, &db_a)),
        );
        // Δp_corr = Δp + J_p_gyro·δb_g + J_p_accel·δb_a
        let dp_corr = v3_add(
            &self.delta_position,
            &v3_add(&m3v3(&self.j_p_gyro, &db_g), &m3v3(&self.j_p_accel, &db_a)),
        );

        // ── Residual ─────────────────────────────────────────────────────
        // r_R = Log(ΔR_corr^T · R_i · R_j^T)
        let dr_corr_t = m3t(&dr_corr);
        let r_rot = so3_log(&m3m3(&dr_corr_t, &m3m3(&r_i, &r_jt)));

        // r_v = R_i · (v_j − v_i − g·dt) − Δv_corr
        let alpha = v3_sub(&v3_sub(&vj, &vi), &v3_scale(&self.gravity, self.dt));
        let r_vel = v3_sub(&m3v3(&r_i, &alpha), &dv_corr);

        // r_p = R_i · (p_j − p_i − v_i·dt − ½g·dt²) − Δp_corr
        let beta = v3_sub(
            &v3_sub(&v3_sub(&p_j, &p_i), &v3_scale(&vi, self.dt)),
            &v3_scale(&self.gravity, 0.5 * self.dt * self.dt),
        );
        let r_pos = v3_sub(&m3v3(&r_i, &beta), &dp_corr);

        let mut residual = vec![
            r_rot[0], r_rot[1], r_rot[2],
            r_vel[0], r_vel[1], r_vel[2],
            r_pos[0], r_pos[1], r_pos[2],
        ];

        // ── Jacobian (9 × 24, row-major) ─────────────────────────────────
        let jacobian = if compute_jacobian {
            let mut jac = vec![0.0f32; 9 * TOTAL_DIM];

            let jr_inv = inv_right_jacobian(&r_rot);
            let jr_inv_rj = m3m3(&jr_inv, &r_j);

            // --- r_R rows (0-2) ---
            // d/d(ω_i) = J_r^{-1}·R_j  (cols 3-5)
            set_block(&mut jac, 0, 3, &jr_inv_rj);
            // d/d(ω_j) = −J_r^{-1}·R_j  (cols 12-14)
            set_block(&mut jac, 0, 12, &m3_neg(&jr_inv_rj));
            // d/d(b_g) = −J_r^{-1}·J_R_gyro  (cols 18-20)
            set_block(&mut jac, 0, 18, &m3_neg(&m3m3(&jr_inv, &self.j_r_gyro)));

            // --- r_v rows (3-5) ---
            // d/d(ω_i) = −R_i·[α]×  (cols 3-5)
            set_block(&mut jac, 3, 3, &m3_neg(&m3m3(&r_i, &hat(&alpha))));
            // d/d(v_i) = −R_i  (cols 6-8)
            set_block(&mut jac, 3, 6, &m3_neg(&r_i));
            // d/d(v_j) = R_i  (cols 15-17)
            set_block(&mut jac, 3, 15, &r_i);
            // d/d(b_g) = −J_v_gyro  (cols 18-20)
            set_block(&mut jac, 3, 18, &m3_neg(&self.j_v_gyro));
            // d/d(b_a) = −J_v_accel  (cols 21-23)
            set_block(&mut jac, 3, 21, &m3_neg(&self.j_v_accel));

            // --- r_p rows (6-8) ---
            // d/d(υ_i) = R_i  (cols 0-2)
            set_block(&mut jac, 6, 0, &r_i);
            // d/d(ω_i) = −R_i·[p_j − v_i·dt − ½g·dt²]×  (cols 3-5)
            let gamma = v3_sub(
                &v3_sub(&p_j, &v3_scale(&vi, self.dt)),
                &v3_scale(&self.gravity, 0.5 * self.dt * self.dt),
            );
            set_block(&mut jac, 6, 3, &m3_neg(&m3m3(&r_i, &hat(&gamma))));
            // d/d(v_i) = −R_i·dt  (cols 6-8)
            set_block(&mut jac, 6, 6, &m3_scale(&r_i, -self.dt));
            // d/d(υ_j) = −R_i  (cols 9-11)
            set_block(&mut jac, 6, 9, &m3_neg(&r_i));
            // d/d(ω_j) = R_i·[p_j]×  (cols 12-14)
            set_block(&mut jac, 6, 12, &m3m3(&r_i, &hat(&p_j)));
            // d/d(b_g) = −J_p_gyro  (cols 18-20)
            set_block(&mut jac, 6, 18, &m3_neg(&self.j_p_gyro));
            // d/d(b_a) = −J_p_accel  (cols 21-23)
            set_block(&mut jac, 6, 21, &m3_neg(&self.j_p_accel));

            super::whiten_matrix(&mut residual, Some(jac.as_mut_slice()), &self.sqrt_info, 9);
            Some(jac)
        } else {
            super::whiten_matrix(&mut residual, None, &self.sqrt_info, 9);
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, TOTAL_DIM))
    }

    fn residual_dim(&self) -> usize {
        9
    }

    fn num_variables(&self) -> usize {
        5
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        match idx {
            0 => 6, // pose_i  (SE3)
            1 => 3, // vel_i   (R³)
            2 => 6, // pose_j  (SE3)
            3 => 3, // vel_j   (R³)
            4 => 6, // bias    (R⁶)
            _ => panic!("InertialFactor has 5 variables"),
        }
    }
}

// ── f32 SO(3) and matrix helpers ─────────────────────────────────────────

/// Column-major 3×3 index.
#[inline]
fn idx3(row: usize, col: usize) -> usize {
    row + col * 3
}

/// Quaternion [qw,qx,qy,qz,...] → 3×3 rotation matrix (column-major).
fn quat_to_rot(q: &[f32]) -> [f32; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y),
    ]
}

/// 3×3 × 3-vec (column-major matrix).
fn m3v3(m: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

/// 3×3 × 3×3 multiply (both column-major).
fn m3m3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for c in 0..3 {
        for r in 0..3 {
            out[idx3(r, c)] = a[idx3(r, 0)] * b[idx3(0, c)]
                + a[idx3(r, 1)] * b[idx3(1, c)]
                + a[idx3(r, 2)] * b[idx3(2, c)];
        }
    }
    out
}

/// 3×3 transpose (column-major).
fn m3t(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn m3_scale(m: &[f32; 9], s: f32) -> [f32; 9] {
    std::array::from_fn(|i| m[i] * s)
}

fn m3_neg(m: &[f32; 9]) -> [f32; 9] {
    std::array::from_fn(|i| -m[i])
}

fn v3_add(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn v3_sub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v3_scale(v: &[f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn v3_neg(v: &[f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

/// Skew-symmetric matrix [v]× (column-major).
fn hat(v: &[f32; 3]) -> [f32; 9] {
    [
         0.0,  v[2], -v[1],
        -v[2],  0.0,  v[0],
         v[1], -v[0],  0.0,
    ]
}

const ID3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// SO(3) exponential map (Rodrigues).
fn so3_exp(omega: &[f32; 3]) -> [f32; 9] {
    let theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let theta = theta_sq.sqrt();
    let w = hat(omega);
    let w2 = m3m3(&w, &w);

    if theta < 1e-6 {
        std::array::from_fn(|i| ID3[i] + w[i])
    } else {
        let a = theta.sin() / theta;
        let b = (1.0 - theta.cos()) / theta_sq;
        std::array::from_fn(|i| ID3[i] + a * w[i] + b * w2[i])
    }
}

/// SO(3) logarithmic map: rotation matrix → axis-angle vector.
fn so3_log(r: &[f32; 9]) -> [f32; 3] {
    let trace = r[0] + r[4] + r[8];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    let vee = [
        r[idx3(2, 1)] - r[idx3(1, 2)],
        r[idx3(0, 2)] - r[idx3(2, 0)],
        r[idx3(1, 0)] - r[idx3(0, 1)],
    ];

    if theta < 1e-7 {
        [vee[0] * 0.5, vee[1] * 0.5, vee[2] * 0.5]
    } else if (std::f32::consts::PI - theta).abs() < 1e-5 {
        // Near π: use column of (R + I) with largest diagonal entry
        let d0 = r[0] + 1.0;
        let d1 = r[4] + 1.0;
        let d2 = r[8] + 1.0;
        let v = if d0 >= d1 && d0 >= d2 {
            [d0, r[idx3(1, 0)], r[idx3(2, 0)]]
        } else if d1 >= d2 {
            [r[idx3(0, 1)], d1, r[idx3(2, 1)]]
        } else {
            [r[idx3(0, 2)], r[idx3(1, 2)], d2]
        };
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let s = theta / n;
        [v[0] * s, v[1] * s, v[2] * s]
    } else {
        let s = theta / (2.0 * theta.sin());
        [vee[0] * s, vee[1] * s, vee[2] * s]
    }
}

/// Inverse right Jacobian of SO(3).
///
/// J_r⁻¹(ω) = I + ½[ω]× + (1/θ² − (1+cosθ)/(2θ sinθ))·[ω]×²
fn inv_right_jacobian(omega: &[f32; 3]) -> [f32; 9] {
    let theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let theta = theta_sq.sqrt();
    let w = hat(omega);
    let w2 = m3m3(&w, &w);

    if theta < 1e-6 {
        std::array::from_fn(|i| ID3[i] + 0.5 * w[i])
    } else {
        let coeff = 1.0 / theta_sq - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
        std::array::from_fn(|i| ID3[i] + 0.5 * w[i] + coeff * w2[i])
    }
}

/// Write a 3×3 column-major block into a 9×24 row-major Jacobian.
fn set_block(jac: &mut [f32], row_off: usize, col_off: usize, block: &[f32; 9]) {
    for r in 0..3 {
        for c in 0..3 {
            jac[(row_off + r) * TOTAL_DIM + col_off + c] = block[idx3(r, c)];
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

    fn default_calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        }
    }

    fn make_stationary_pre() -> PreintegratedImu {
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81),
        };
        for _ in 0..10 {
            pre.integrate(&m, 0.01);
        }
        pre
    }

    fn identity_pose() -> [f32; 7] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn zero_residual_stationary() {
        let pre = make_stationary_pre();
        let factor = InertialFactor::new(&pre, [0.0, 0.0, -9.81]);

        let pose = identity_pose();
        let vel = [0.0f32; 3];
        let bias = [0.0f32; 6];

        let result = factor
            .linearize(&[&pose, &vel, &pose, &vel, &bias], false)
            .unwrap();

        for i in 0..9 {
            assert!(
                result.residual[i].abs() < 0.01,
                "residual[{i}] = {} (expected ~0)",
                result.residual[i]
            );
        }
    }

    // ── Numerical Jacobian check ─────────────────────────────────────────

    /// Rotation matrix → quaternion [qw,qx,qy,qz] (Shepperd's method).
    fn rot_to_quat(r: &[f32; 9]) -> [f32; 4] {
        let trace = r[0] + r[4] + r[8];
        if trace > 0.0 {
            let s = (1.0 + trace).sqrt() * 2.0;
            [s / 4.0, (r[idx3(2, 1)] - r[idx3(1, 2)]) / s,
             (r[idx3(0, 2)] - r[idx3(2, 0)]) / s, (r[idx3(1, 0)] - r[idx3(0, 1)]) / s]
        } else if r[0] > r[4] && r[0] > r[8] {
            let s = (1.0 + r[0] - r[4] - r[8]).sqrt() * 2.0;
            [(r[idx3(2, 1)] - r[idx3(1, 2)]) / s, s / 4.0,
             (r[idx3(0, 1)] + r[idx3(1, 0)]) / s, (r[idx3(0, 2)] + r[idx3(2, 0)]) / s]
        } else if r[4] > r[8] {
            let s = (1.0 + r[4] - r[0] - r[8]).sqrt() * 2.0;
            [(r[idx3(0, 2)] - r[idx3(2, 0)]) / s, (r[idx3(0, 1)] + r[idx3(1, 0)]) / s,
             s / 4.0, (r[idx3(1, 2)] + r[idx3(2, 1)]) / s]
        } else {
            let s = (1.0 + r[8] - r[0] - r[4]).sqrt() * 2.0;
            [(r[idx3(1, 0)] - r[idx3(0, 1)]) / s, (r[idx3(0, 2)] + r[idx3(2, 0)]) / s,
             (r[idx3(1, 2)] + r[idx3(2, 1)]) / s, s / 4.0]
        }
    }

    /// Right-perturb an SE3 pose by tangent vector [δυ(3), δω(3)].
    fn perturb_se3(pose: &[f32], delta: &[f32; 6]) -> Vec<f32> {
        let r = quat_to_rot(pose);
        let t = [pose[4], pose[5], pose[6]];
        let dv = [delta[0], delta[1], delta[2]];
        let dw = [delta[3], delta[4], delta[5]];

        let r_new = m3m3(&r, &so3_exp(&dw));
        let q = rot_to_quat(&r_new);
        let r_dv = m3v3(&r, &dv);

        vec![q[0], q[1], q[2], q[3],
             t[0] + r_dv[0], t[1] + r_dv[1], t[2] + r_dv[2]]
    }

    #[test]
    fn jacobian_numerical_check() {
        let mut pre = make_stationary_pre();
        // Identity covariance → sqrt_info = I → no whitening distortion.
        pre.covariance = [0.0; 81];
        for i in 0..9 {
            pre.covariance[i + i * 9] = 1.0;
        }

        let factor = InertialFactor::new(&pre, [0.0, 0.0, -9.81]);

        let pose_i = identity_pose();
        let vel_i = [0.0f32; 3];
        let pose_j = identity_pose();
        let vel_j = [0.5f32, 0.0, 0.0]; // offset → non-zero residual
        let bias = [0.0f32; 6];

        let params: Vec<&[f32]> = vec![&pose_i, &vel_i, &pose_j, &vel_j, &bias];

        let result = factor.linearize(&params, true).unwrap();
        let jac = result.jacobian.unwrap();

        let eps = 1e-3f32;
        let tangent_dims = [6, 3, 6, 3, 6];
        let mut col_offset = 0;

        for (var_idx, &dim) in tangent_dims.iter().enumerate() {
            for d in 0..dim {
                let mut delta_pos = vec![0.0f32; dim];
                delta_pos[d] = eps;
                let mut delta_neg = vec![0.0f32; dim];
                delta_neg[d] = -eps;

                let (perturbed_plus, perturbed_minus) = match var_idx {
                    0 | 2 => {
                        let dp: [f32; 6] = std::array::from_fn(|i| delta_pos[i]);
                        let dn: [f32; 6] = std::array::from_fn(|i| delta_neg[i]);
                        (perturb_se3(params[var_idx], &dp), perturb_se3(params[var_idx], &dn))
                    }
                    _ => {
                        let pp: Vec<f32> = params[var_idx].iter().enumerate()
                            .map(|(i, &v)| v + if i == d { eps } else { 0.0 }).collect();
                        let pm: Vec<f32> = params[var_idx].iter().enumerate()
                            .map(|(i, &v)| v + if i == d { -eps } else { 0.0 }).collect();
                        (pp, pm)
                    }
                };

                let mut pp = params.iter().map(|s| s.to_vec()).collect::<Vec<_>>();
                pp[var_idx] = perturbed_plus;
                let pp_refs: Vec<&[f32]> = pp.iter().map(|v| v.as_slice()).collect();

                let mut pm = params.iter().map(|s| s.to_vec()).collect::<Vec<_>>();
                pm[var_idx] = perturbed_minus;
                let pm_refs: Vec<&[f32]> = pm.iter().map(|v| v.as_slice()).collect();

                let r_plus = factor.linearize(&pp_refs, false).unwrap().residual;
                let r_minus = factor.linearize(&pm_refs, false).unwrap().residual;

                for row in 0..9 {
                    let numerical = (r_plus[row] - r_minus[row]) / (2.0 * eps);
                    let analytical = jac[row * TOTAL_DIM + col_offset + d];
                    assert!(
                        (numerical - analytical).abs() < 5e-2,
                        "jac[{row},{}]: numerical={numerical:.4}, analytical={analytical:.4}",
                        col_offset + d
                    );
                }
            }
            col_offset += dim;
        }
    }
}
