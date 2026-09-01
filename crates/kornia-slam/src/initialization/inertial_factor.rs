//! `Factor` adapter wiring `PreintegratedImu` into `kornia_algebra::optim`'s
//! generic Levenberg-Marquardt solver, for the ORB-SLAM3-style joint
//! inertial-initialization optimization (`Optimizer::InertialOptimization`,
//! `docs/imu_init_guide.md`).
//!
//! Variable order for every edge added via `Problem::add_factor` MUST be
//! `[v_i, v_j, bg, ba, gdir, scale]` — this matches `variable_local_dim`
//! below and the Jacobian column layout inside `linearize`.
//!
//!   cols:  v_i(0..3)  v_j(3..6)  bg(6..9)  ba(9..12)  gdir(12..15)  scale(15..16)
//!   rows:  er(0..3)   ev(3..6)   ep(6..9)
//!
//! Keyframe poses are NOT `Problem` variables (the optimizer has no "fixed
//! vertex" concept — see the previous discussion) — they're baked into
//! `KfConst` as plain constant data, exactly matching ORB-SLAM3's
//! `VP->setFixed(true)`.

use kornia_3d::pose::Pose3d;
use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};
use kornia_algebra::{DMatF64, DVecF64, Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, PreintegratedImu};

// ─────────────────────────────────────────────────────────────────────────────
// Baked (fixed) per-keyframe body-frame data
// ─────────────────────────────────────────────────────────────────────────────

/// World-frame body/IMU pose of a keyframe, fixed for the duration of this
/// optimization (poses are never variables here).
#[derive(Debug, Clone, Copy)]
pub struct KfConst {
    /// `R_bw`: rotates world-frame vectors into the body frame.
    pub r_bw: Mat3F64,
    /// `t_wb`: body/IMU origin position in world coordinates.
    pub t_wb: Vec3F64,
}

impl KfConst {
    /// Mirrors `vi_ba_schur::body_frame` (`crates/kornia-slam/src/vi_ba_schur.rs:364-383`):
    /// `R_bw = R_bc · R_cw`, `t_wb = -R_bw^T · (R_bc·t_cw + t_bc)`.
    pub fn new(pose_world_to_cam: &Pose3d, imu_t_bc: &Pose3d) -> Self {
        let r_bc = imu_t_bc.rotation;
        let r_cw = pose_world_to_cam.rotation;
        let r_bw = r_bc * r_cw;
        let t_bw = r_bc * pose_world_to_cam.translation + imu_t_bc.translation;
        let t_wb = -(r_bw.transpose() * t_bw);
        Self { r_bw, t_wb }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The factor
// ─────────────────────────────────────────────────────────────────────────────

/// Joint inertial-initialization edge: the port of `EdgeInertialGS`
/// (`G2oTypes.cc:548-718`), minus the two pose blocks (poses are fixed
/// constants here, not optimized).
///
/// Connects 6 `Problem` variables in this order: `v_i(3), v_j(3), bg(3),
/// ba(3), gdir(SO3, 3-dim tangent), scale(1)`.
pub struct InertialInitFactor {
    kf_i: KfConst,
    kf_j: KfConst,
    pim: PreintegratedImu,
    /// Cholesky factor `L` of the 9×9 information matrix (`L Lᵀ = C⁻¹`),
    /// used to whiten the raw residual/Jacobian before returning them.
    sqrt_info: DMatF64,
    /// `false` (stereo) pins scale at 1.0: the live "scale" variable value is
    /// ignored and its Jacobian column is left at zero, so it never receives
    /// gradient from any edge and stays at its seed value under LM — the
    /// `Problem`/`LevenbergMarquardt` API has no "fixed variable" concept
    /// (unlike g2o's `VS->setFixed(!bMono)`), so this reproduces the same
    /// effect from inside the factor instead.
    is_mono: bool,
    /// `true` for the ORB-SLAM3 `ScaleRefinement()` mode (`Optimizer::
    /// InertialOptimization(Map*, Rwg, scale)`, Optimizer.cc:3389): poses,
    /// velocities, and bias are ALL fixed there (`VV->setFixed(true)`,
    /// `VG/VA->setFixed(true)`) — only gravity direction and scale are free.
    /// Same zero-Jacobian-column trick as `is_mono` above, extended to the
    /// v_i/v_j/bg/ba columns.
    fixed_bias_vel: bool,
}

impl InertialInitFactor {
    pub fn new(kf_i: KfConst, kf_j: KfConst, pim: PreintegratedImu, is_mono: bool) -> Self {
        let sqrt_info = sqrt_information_9x9(&pim.covariance);
        Self {
            kf_i,
            kf_j,
            pim,
            sqrt_info,
            is_mono,
            fixed_bias_vel: false,
        }
    }

    /// See `ScaleRefinement` doc on `fixed_bias_vel`.
    pub fn with_fixed_bias_vel(mut self, fixed: bool) -> Self {
        self.fixed_bias_vel = fixed;
        self
    }
}

impl Factor for InertialInitFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 6 {
            return Err(FactorError::DimensionMismatch {
                expected: 6,
                actual: params.len(),
            });
        }
        let v_i = vec3_from_f32(params[0])?;
        let v_j = vec3_from_f32(params[1])?;
        let bg = vec3_from_f32(params[2])?;
        let ba = vec3_from_f32(params[3])?;
        let gdir_arr = quat_array_from_f32(params[4])?;
        if params[5].is_empty() {
            return Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        // Stereo: scale is metric already (s=1), regardless of whatever the
        // "scale" variable currently holds — see `is_mono` doc comment.
        let scale = if self.is_mono {
            params[5][0] as f64
        } else {
            1.0
        };

        let dt = self.pim.dt;
        let r_bw_i = self.kf_i.r_bw;
        let r_wb_j = self.kf_j.r_bw.transpose();

        // Bias-corrected preintegrated deltas (first-order correction around
        // the preintegration's own linearization bias — same helper the rest
        // of imu_init.rs / vi_ba_schur.rs already use).
        let bias = ImuBias {
            gyro: bg,
            accel: ba,
        };
        let d_rot = self.pim.delta_rotation_with_bias(&bias);
        let d_vel = self.pim.delta_velocity_with_bias(&bias);
        let d_pos = self.pim.delta_position_with_bias(&bias);

        // Gravity: g = Rwg · gI, ORB-SLAM3's own internal convention gI=(0,0,-G).
        // Converted to kornia's +Y convention only once, at the very end in
        // `apply_initialization` — never mix conventions mid-solve.
        let rwg = SO3F64::from_array(gdir_arr).matrix();
        let g_i = Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
        let g = rwg * g_i;

        // ── residual ─────────────────────────────────────────────────────
        let e_r_mat = d_rot.transpose() * r_bw_i * r_wb_j;
        let er = SO3F64::from_matrix(&e_r_mat).log();

        let dv_world = scale * (v_j - v_i) - g * dt;
        let ev = r_bw_i * dv_world - d_vel;

        let dp_world = scale * (self.kf_j.t_wb - self.kf_i.t_wb - v_i * dt) - g * (0.5 * dt * dt);
        let ep = r_bw_i * dp_world - d_pos;

        let residual =
            DVecF64::from_vec(vec![er.x, er.y, er.z, ev.x, ev.y, ev.z, ep.x, ep.y, ep.z]);

        // ── Jacobian (9 x 16), zero unless set below ────────────────────
        let jacobian = if compute_jacobian {
            let mut jac = DMatF64::zeros(9, 16);

            let jr_inv = SO3F64::right_jacobian(er).inverse();

            // `fixed_bias_vel` (ScaleRefinement mode): v_i/v_j/bg/ba columns
            // (0..12) stay at zero, pinning them at their seed values under
            // LM exactly like `is_mono=false` already does for the scale
            // column below.
            if !self.fixed_bias_vel {
                // ∂er/∂bg   (cols 6..9)
                let d_er_d_bg =
                    (jr_inv * e_r_mat.transpose() * self.pim.d_rotation_d_bias_gyro) * -1.0;
                set_block3(&mut jac, 0, 6, d_er_d_bg);

                // ∂ev/∂v_i, ∂ev/∂v_j   (cols 0..3, 3..6)
                set_block3(&mut jac, 3, 0, r_bw_i * -scale);
                set_block3(&mut jac, 3, 3, r_bw_i * scale);
                // ∂ev/∂bg, ∂ev/∂ba    (cols 6..9, 9..12)
                set_block3(&mut jac, 3, 6, self.pim.d_velocity_d_bias_gyro * -1.0);
                set_block3(&mut jac, 3, 9, self.pim.d_velocity_d_bias_accel * -1.0);

                // ∂ep/∂v_i            (cols 0..3)
                set_block3(&mut jac, 6, 0, r_bw_i * (-scale * dt));
                // ∂ep/∂bg, ∂ep/∂ba    (cols 6..9, 9..12)
                set_block3(&mut jac, 6, 6, self.pim.d_position_d_bias_gyro * -1.0);
                set_block3(&mut jac, 6, 9, self.pim.d_position_d_bias_accel * -1.0);
            }

            // gravity-direction block (cols 12..15) — 3-DOF SO3 tangent;
            // the column along gI itself is exactly zero (gauge freedom),
            // see the derivation in the previous message.
            let d_g_d_theta = (rwg * SO3F64::hat(g_i)) * -1.0;
            set_block3(&mut jac, 3, 12, (r_bw_i * d_g_d_theta) * dt * -1.0);
            set_block3(
                &mut jac,
                6,
                12,
                (r_bw_i * d_g_d_theta) * (0.5 * dt * dt) * -1.0,
            );

            // scale column (col 15) — left at zero for stereo, pinning it
            // at its seed value of 1.0 under LM (see `is_mono` doc comment).
            if self.is_mono {
                let d_ev_d_s = r_bw_i * (v_j - v_i);
                let d_ep_d_s = r_bw_i * (self.kf_j.t_wb - self.kf_i.t_wb - v_i * dt);
                set_col3(&mut jac, 3, 15, d_ev_d_s);
                set_col3(&mut jac, 6, 15, d_ep_d_s);
            }

            Some(jac)
        } else {
            None
        };

        // ── whiten by the preintegration's information matrix ───────────
        let residual_w = &self.sqrt_info * &residual;
        let jacobian_w = jacobian.map(|j| &self.sqrt_info * j);

        let residual_f32: Vec<f32> = residual_w.iter().map(|&x| x as f32).collect();
        let jacobian_f32 = jacobian_w.map(|j| {
            let mut out = vec![0.0f32; 9 * 16];
            for r in 0..9 {
                for c in 0..16 {
                    out[r * 16 + c] = j[(r, c)] as f32;
                }
            }
            out
        });

        Ok(LinearizationResult::new(residual_f32, jacobian_f32, 16))
    }

    fn residual_dim(&self) -> usize {
        9
    }

    fn num_variables(&self) -> usize {
        6
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        [3, 3, 3, 3, 3, 1][idx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bias prior factor  (EdgePriorGyro / EdgePriorAcc, G2oTypes.h:768-815)
// ─────────────────────────────────────────────────────────────────────────────

/// `r = sqrt(weight) · x` — a zero-target prior with an explicit information
/// weight. `optim::PriorFactor` is unweighted (implicit weight 1), so this
/// exists to reproduce `priorG`/`priorA` from the VIBA0/1/2 schedule.
pub struct WeightedZeroPrior {
    pub sqrt_weight: f64,
}

impl Factor for WeightedZeroPrior {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 1 || params[0].len() != 3 {
            return Err(FactorError::DimensionMismatch {
                expected: 3,
                actual: params.first().map(|p| p.len()).unwrap_or(0),
            });
        }
        let w = self.sqrt_weight as f32;
        let residual = vec![params[0][0] * w, params[0][1] * w, params[0][2] * w];
        let jacobian = if compute_jacobian {
            Some(vec![w, 0.0, 0.0, 0.0, w, 0.0, 0.0, 0.0, w])
        } else {
            None
        };
        Ok(LinearizationResult::new(residual, jacobian, 3))
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        3
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

fn vec3_from_f32(s: &[f32]) -> FactorResult<Vec3F64> {
    if s.len() < 3 {
        return Err(FactorError::DimensionMismatch {
            expected: 3,
            actual: s.len(),
        });
    }
    Ok(Vec3F64::new(s[0] as f64, s[1] as f64, s[2] as f64))
}

fn quat_array_from_f32(s: &[f32]) -> FactorResult<[f64; 4]> {
    if s.len() < 4 {
        return Err(FactorError::DimensionMismatch {
            expected: 4,
            actual: s.len(),
        });
    }
    Ok([s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64])
}

/// Writes a `Mat3F64` (column-major) into a 3x3 block of a dense Jacobian
/// at `(row0, col0)`.
fn set_block3(jac: &mut DMatF64, row0: usize, col0: usize, m: Mat3F64) {
    let a = m.to_cols_array(); // column-major: a[c*3+r]
    for c in 0..3 {
        for r in 0..3 {
            jac[(row0 + r, col0 + c)] = a[c * 3 + r];
        }
    }
}

/// Writes a `Vec3F64` into a single Jacobian column at `(row0, col0)`.
fn set_col3(jac: &mut DMatF64, row0: usize, col0: usize, v: Vec3F64) {
    jac[(row0, col0)] = v.x;
    jac[(row0 + 1, col0)] = v.y;
    jac[(row0 + 2, col0)] = v.z;
}

/// Cholesky factor `L` (`L Lᵀ = C⁻¹`) of a 9×9 navigation covariance block,
/// used to whiten the raw residual/Jacobian to a maximum-likelihood weighting
/// (mirrors `EdgeInertialGS`'s constructor, `G2oTypes.cc:604-611`, minus the
/// eigenvalue-clamping step — that needs `nalgebra::SymmetricEigen`, a new
/// direct dependency; the plain-inverse-then-Cholesky path below is simpler
/// and only falls back to unweighted (identity) in the rare case the
/// covariance is numerically non-PSD. Revisit if that fallback ever triggers
/// in practice).
fn sqrt_information_9x9(cov_col_major: &[f64; 81]) -> DMatF64 {
    let cov = DMatF64::from_column_slice(9, 9, cov_col_major);
    let cov_sym = (cov.clone() + cov.transpose()) * 0.5;
    let info = cov_sym
        .try_inverse()
        .unwrap_or_else(|| DMatF64::identity(9, 9));
    let info_sym = (info.clone() + info.transpose()) * 0.5;
    match info_sym.cholesky() {
        Some(c) => c.l(),
        None => DMatF64::identity(9, 9),
    }
}
