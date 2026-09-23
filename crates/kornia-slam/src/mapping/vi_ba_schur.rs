//! Schur-complement bundle adjustment with dense reduced camera system.
//!
//! The standard bipartite-Schur trick from Triggs et al. (1999). Each LM
//! iteration builds the Hessian in BLOCK form
//!
//! ```text
//!     H = [ A   B  ]    A = 6P × 6P pose blocks (block-diagonal),
//!         [ Bᵀ  C  ]    C = 3N × 3N point blocks (BLOCK-DIAGONAL),
//!                       B = 6P × 3N pose-point cross terms (sparse).
//! ```
//!
//! Block-diagonal C means C⁻¹ is cheap (per-3×3 invert). The **reduced
//! camera system**
//!
//! ```text
//!     M = A − B C⁻¹ Bᵀ           (dense 6P × 6P)
//!     m = g_pose − B C⁻¹ g_point
//! ```
//!
//! is solved with `faer`'s dense Cholesky on the small matrix; points are
//! recovered by back-substitution. For our SLAM problem (~170 poses ×
//! ~3000 points × ~15000 observations) the reduced system is just
//! 1020 × 1020 — Ceres's `DENSE_SCHUR` is exactly this regime.
//!
//! No sparse-matrix dependency is needed because the only "large" object
//! the Schur trick has to manipulate (B, 6P × 3N) is never materialised:
//! we walk observations and accumulate per-point contributions into M
//! directly.
//!
//! Jacobian conventions match [`kornia_3d::ba::ReprojFactor`]:
//!
//!   * Pose tangent layout `[ρ; ω]` (upsilon then omega), 6-dim.
//!   * Point parameters are the 3-dim world coordinates.
//!   * z is clamped to `MIN_Z` to handle mid-iteration cheirality flips.
//!
//! Currently supports: Huber-robustified reprojection and IMU-nav
//! residuals, fixed-pose anchors, fixed-point gauge (motion-only BA).
//! Full LM-with-backtracking is still TODO.

use faer::Mat;
use faer::prelude::Solve;
use kornia_algebra::{Mat3F64, SE3F32, SO3F64, Vec3AF32, Vec3F64};
use thiserror::Error;

use crate::pose_conversion::{pose_to_se3, se3_to_pose};

use kornia_3d::ba::{BaError, BaObservation};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{HuberKernel, RobustKernel};
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

const MIN_Z: f32 = 1e-3;

/// Errors specific to the Schur BA driver. Wraps existing [`BaError`].
#[derive(Debug, Error)]
pub enum SchurBaError {
    /// Linear system is rank-deficient / Cholesky failed.
    #[error("Reduced camera Cholesky failed (likely rank-deficient): {0}")]
    CholeskyFailed(String),
    /// No free variables after applying anchors.
    #[error("All variables are fixed — nothing to optimise")]
    NoFreeVariables,
    /// Other BA setup error.
    #[error(transparent)]
    Ba(#[from] BaError),
}

/// Errors specific to VI-BA.
#[derive(Debug, Error)]
pub enum ViBaError {
    #[error("Reduced system Cholesky failed (rank-deficient): {0}")]
    CholeskyFailed(String),
    #[error("All keyframe variables are fixed — nothing to optimise")]
    NoFreeVariables,
    #[error("IMU edge keyframe indices out of range: {0} → {1}")]
    ImuFactorOutOfRange(usize, usize),
    #[error(transparent)]
    Ba(#[from] BaError),
}

/// A single keyframe state for VI-BA.
///
/// Holds the 6-DOF pose, 3-DOF world-frame velocity, and 6-DOF IMU bias.
/// Together these form the 15-DOF state vector that VI-BA optimises.
#[derive(Debug, Clone)]
pub struct ViBaKeyframe {
    /// World→camera pose. Same convention as [`Pose3d`] in schur_ba.
    pub pose: Pose3d,
    /// Velocity in world frame (m/s).
    pub velocity: Vec3F64,
    /// IMU bias estimate at this keyframe.
    pub bias: ImuBias,
    /// If true, pose+velocity+bias are held fixed (used for the first keyframe
    /// as the gauge anchor, or for marginalised frames).
    pub fixed: bool,
}

/// An IMU factor connecting two consecutive keyframes.
///
/// `from_idx` and `to_idx` must be consecutive keyframe indices (to_idx =
/// from_idx + 1 in a window), though non-consecutive indices work as long as
/// `preintegrated` covers the interval.
#[derive(Debug, Clone)]
pub struct ImuFactor {
    pub from_idx: usize,
    pub to_idx: usize,
    /// Preintegrated measurements between the two keyframes.
    pub preintegrated: PreintegratedImu,
}

/// Parameters for VI-BA.
#[derive(Debug, Clone)]
pub struct ViBaParams {
    pub max_iterations: usize,
    /// Levenberg-Marquardt initial damping.
    pub initial_lambda: f64,
    /// Relative cost-decrease threshold for convergence.
    pub cost_tolerance: f64,
    /// Gravity vector in world frame, e.g. [0, 0, -9.81].
    pub gravity: Vec3F64,
    /// Camera-to-body (IMU) extrinsic: `X_body = T_BC * X_cam`.
    /// `None` means camera frame == body frame (identity T_BC).
    pub imu_t_bc: Option<Pose3d>,
    /// Global scale applied to every IMU information matrix.
    /// Use 1.0 normally; lower (e.g. 0.01) to debug without IMU dominance.
    pub imu_weight: f64,
    /// Chi-square threshold for Huber robustification of IMU nav residuals
    /// (9-DOF: rotation + velocity + position). Edges with chi2 > this value
    /// are down-weighted by sqrt(huber_imu_chi2 / chi2). The 9-DOF chi-square
    /// at 95 % confidence is 16.92. At sensor-noise level the nav chi2 is
    /// typically ~10–15, so the Huber kernel is inactive once converged
    /// (weight = 1.0, full metric constraint) but strongly suppresses the
    /// large residuals that arise from rough post-init velocity estimates.
    pub huber_imu_chi2: f64,
    /// Squared-pixel Huber scale for reprojection residuals.
    pub huber_reproj_px_sq: f64,
    /// Squared-sigma Huber scale for stereo depth residuals.
    pub huber_depth_sigma_sq: f64,
    /// Scale applied to the IMU edge whose "from" endpoint is a permanently-fixed
    /// keyframe (the one boundary edge linking the active window to the frozen
    /// past). Full-strength coupling to a value that can never be corrected again
    /// lets local estimation noise/systematic error accumulate directionally
    /// across successive windows. ORB-SLAM3 applies the same 1e-2 downweight to
    /// this edge (`Optimizer::LocalInertialBA`, "to avoid accumulating error due
    /// to fixing variables").
    pub boundary_imu_info_scale: f64,
    /// Information weight (1/sigma²) of a zero-mean Gaussian prior pulling each
    /// free keyframe's accel bias toward zero, applied every iteration
    /// independent of Huber down-weighting (unlike the IMU-edge terms, this
    /// unary factor is never scaled by `huber_imu_weight`, so it stays a solid
    /// floor even in windows where a large nav residual would otherwise
    /// suppress the bias-random-walk constraint too).
    ///
    /// A single IMU edge cannot distinguish a constant accel-bias offset from
    /// a gravity-magnitude/direction error, since gravity is held fixed for
    /// the duration of this call (only corrected periodically elsewhere).
    /// Without this prior, a windowed local BA has nowhere else to put
    /// leftover gravity error and will explain it away as bias — set to 0.0
    /// to disable and reproduce the old behaviour.
    pub accel_bias_prior_weight: f64,
}

impl Default for ViBaParams {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            initial_lambda: 1.0,
            cost_tolerance: 1e-6,
            gravity: Vec3F64::new(0.0, 9.81, 0.0), // OpenCV Y-down: gravity = +Y
            imu_t_bc: None,
            imu_weight: 1.0,
            huber_imu_chi2: 16.9, // 9-DOF chi-square at 95 %
            huber_reproj_px_sq: 25.0,
            huber_depth_sigma_sq: 9.0,
            boundary_imu_info_scale: 1e-2,
            // sigma = 0.1 m/s²: comfortably above real MEMS accel-bias
            // magnitude, so it's a floor that only bites once the
            // bias-random-walk edges have been Huber-suppressed or a value
            // is drifting well past a physically plausible range — not a
            // constraint that fights legitimate, well-observed bias updates
            // (those edges carry ~1e9 information when nav residuals are
            // healthy, per the LM-damping note below).
            accel_bias_prior_weight: 1e2,
        }
    }
}

/// Output of VI-BA.
#[derive(Debug, Clone)]
pub struct ViBaResult {
    pub keyframes: Vec<ViBaKeyframe>,
    pub points: Vec<Vec3F64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_cost: f64,
}

// ── Reprojection residual/Jacobian ────────────────────────────────────────
// Reprojection factor algebra shared with kornia_3d's visual-only BA; kept
// here in f32 for the Schur assembly below.

/// Computes (residual, J_pose 2×6, J_point 2×3) at the current state, matching
/// `kornia_3d::ba::ReprojFactor`. Jacobian layout (row-major flat):
///   J_pose[0..6]:  [du/dρ, du/dω]   J_pose[6..12]: [dv/dρ, dv/dω]
///   J_point[0..3]: [du/dxyz]        J_point[3..6]: [dv/dxyz]
fn residual_and_jacobians(
    pose: &SE3F32,
    point_w: &Vec3F64,
    pixel: [f32; 2],
    camera: &PinholeCamera,
) -> ([f32; 2], [f32; 12], [f32; 6]) {
    let fx = camera.fx as f32;
    let fy = camera.fy as f32;
    let cx = camera.cx as f32;
    let cy = camera.cy as f32;

    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
    let z = if pc.z.abs() < MIN_Z {
        if pc.z >= 0.0 { MIN_Z } else { -MIN_Z }
    } else {
        pc.z
    };
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    let u = fx * pc.x * inv_z + cx;
    let v = fy * pc.y * inv_z + cy;
    let r = [u - pixel[0], v - pixel[1]];

    // J_proj row coefficients (∂[u; v] / ∂[X_c]).
    let a0 = fx * inv_z;
    let a2 = -fx * pc.x * inv_z2;
    let b1 = fy * inv_z;
    let b2 = -fy * pc.y * inv_z2;

    // Rotation matrix elements (R: world→cam).
    let rm = pose.r.matrix();
    let r00 = rm.col(0).x;
    let r01 = rm.col(1).x;
    let r02 = rm.col(2).x;
    let r10 = rm.col(0).y;
    let r11 = rm.col(1).y;
    let r12 = rm.col(2).y;
    let r20 = rm.col(0).z;
    let r21 = rm.col(1).z;
    let r22 = rm.col(2).z;

    let (px, py, pz) = (pw.x, pw.y, pw.z);

    // S = -R · skew(p_w) — for the omega part.
    let s00 = -pz * r01 + py * r02;
    let s10 = -pz * r11 + py * r12;
    let s20 = -pz * r21 + py * r22;

    let s01 = pz * r00 - px * r02;
    let s11 = pz * r10 - px * r12;
    let s21 = pz * r20 - px * r22;

    let s02 = -py * r00 + px * r01;
    let s12 = -py * r10 + px * r11;
    let s22 = -py * r20 + px * r21;

    // J_pt = J_proj · R (3 cols).
    let jpt_00 = a0 * r00 + a2 * r20;
    let jpt_01 = a0 * r01 + a2 * r21;
    let jpt_02 = a0 * r02 + a2 * r22;
    let jpt_10 = b1 * r10 + b2 * r20;
    let jpt_11 = b1 * r11 + b2 * r21;
    let jpt_12 = b1 * r12 + b2 * r22;

    // J_omega = J_proj · S (3 cols).
    let jom_00 = a0 * s00 + a2 * s20;
    let jom_01 = a0 * s01 + a2 * s21;
    let jom_02 = a0 * s02 + a2 * s22;
    let jom_10 = b1 * s10 + b2 * s20;
    let jom_11 = b1 * s11 + b2 * s21;
    let jom_12 = b1 * s12 + b2 * s22;

    // Layout J_pose 2×6 row-major: [ρ(3) | ω(3)] per row.
    let j_pose: [f32; 12] = [
        jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10, jom_11,
        jom_12,
    ];
    // J_point 2×3 row-major.
    let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];

    (r, j_pose, j_point)
}

/// Depth (stereo Z) residual and Jacobian, in f64, w.r.t. pose (SE3 tangent
/// `[ρ|ω]`) and world point. Mirrors the rotation-row-2 algebra in
/// [`residual_and_jacobians`] (see there for the `S = -R·skew(p_w)`
/// derivation) applied to the scalar predicted-depth observation
/// `BaObservation::depth_meas` carries for stereo keyframes.
///
/// `visual_inertial_bundle_adjust` previously ignored `depth_meas` entirely
/// (unlike the plain-visual `kornia_3d::ba_schur` BA, which
/// honours it), so stereo+IMU local BA had nothing anchoring absolute scale
/// beyond the IMU factor's gravity magnitude in a 3-keyframe window — while
/// stereo-only BA (which does use this term) stayed scale-consistent. That
/// asymmetry is the reason stereo+IMU accumulated ~25% scale drift where
/// stereo-only and mono+IMU (which never had depth to anchor with anyway)
/// did not.
fn depth_residual_and_jacobian(
    pose: &SE3F32,
    point_w: &Vec3F64,
    d_meas: f64,
    sigma: f64,
) -> (f64, [f64; 6], [f64; 3]) {
    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
    let z_pred = if pc.z.abs() < MIN_Z {
        if pc.z >= 0.0 { MIN_Z } else { -MIN_Z }
    } else {
        pc.z
    } as f64;

    let inv_sigma = 1.0 / sigma;
    let r_z = (z_pred - d_meas) * inv_sigma;

    let rm = pose.r.matrix();
    let r20 = rm.col(0).z as f64;
    let r21 = rm.col(1).z as f64;
    let r22 = rm.col(2).z as f64;
    let (px, py, pz) = (point_w.x, point_w.y, point_w.z);
    // Row 2 of S = -R · skew(p_w).
    let s20 = -pz * r21 + py * r22;
    let s21 = pz * r20 - px * r22;
    let s22 = -py * r20 + px * r21;

    let j_pose = [
        0.0,
        0.0,
        inv_sigma,
        s20 * inv_sigma,
        s21 * inv_sigma,
        s22 * inv_sigma,
    ];
    let j_point = [r20 * inv_sigma, r21 * inv_sigma, r22 * inv_sigma];
    (r_z, j_pose, j_point)
}

/// Converts a camera-frame world-to-cam pose to body-frame world-to-body rotation and
/// world-frame body position, applying the camera-to-body extrinsic T_BC when provided.
///
/// Returns `(R_bw, twb)` where `R_bw` maps world vectors into the body frame and
/// `twb` is the body origin in world coordinates.
fn body_frame(cam_pose: &Pose3d, t_bc: Option<&Pose3d>) -> (Mat3F64, Vec3F64) {
    let r_cw = &cam_pose.rotation;
    let t_cw = &cam_pose.translation;
    match t_bc {
        None => {
            let twb = -(Mat3F64(*r_cw.transpose()) * *t_cw);
            (*r_cw, twb)
        }
        Some(t_bc) => {
            let r_bc = &t_bc.rotation;
            // T_bw = T_bc · T_cw:  R_bw = R_bc · R_cw
            let r_bw = mat3_mul(r_bc, r_cw);
            // t_bw = R_bc · t_cw + t_bc
            let t_bw = *r_bc * *t_cw + t_bc.translation;
            // twb = -R_bw^T · t_bw
            let twb = -(Mat3F64(*r_bw.transpose()) * t_bw);
            (r_bw, twb)
        }
    }
}

/// Computes the 15-DOF IMU residual and its Jacobians wrt the two keyframe states.
///
/// Returns:
///   residual: [f64; 15]  — [r_R(3), r_v(3), r_p(3), r_bg(3), r_ba(3)]
///   J_i: [f64; 15*15]   — Jacobian wrt state_i (row-major, 15 rows × 15 cols)
///   J_j: [f64; 15*15]   — Jacobian wrt state_j
///
/// State layout (per keyframe): [ρ(6) | v(3) | bg(3) | ba(3)]
///   ρ = [upsilon(3), omega(3)] — SE3 tangent, matching ReprojFactor convention
fn imu_residual_and_jacobians(
    kf_i: &ViBaKeyframe,
    kf_j: &ViBaKeyframe,
    pim: &PreintegratedImu,
    gravity: &Vec3F64,
    imu_t_bc: Option<&Pose3d>,
) -> ([f64; 15], [f64; 225], [f64; 225]) {
    let dt = pim.dt;

    // kf.pose = T_cw (world→cam). IMU dynamics live in the body frame.
    // If T_BC is provided (X_body = T_BC · X_cam):
    //   T_bw = T_bc · T_cw  →  R_bw = R_bc · R_cw,  t_bw = R_bc · t_cw + t_bc
    // Otherwise camera == body and R_bw = R_cw.
    // World-frame body position: twb = -R_bw^T · t_bw.
    //
    // Key property: right-SE3 perturbation of T_cw = same right-SE3 perturbation
    // of T_bw (since T_bc is fixed), so Jacobian column layout is unchanged.
    let (r_bw_i, twb_i) = body_frame(&kf_i.pose, imu_t_bc);
    let (r_bw_j, twb_j) = body_frame(&kf_j.pose, imu_t_bc);
    let r_wb_j = Mat3F64(*r_bw_j.transpose()); // R_wb_j = R_bw_j^T (body→world rotation at j)

    let v_i = &kf_i.velocity;
    let v_j = &kf_j.velocity;

    // Bias-corrected preintegrated measurements.
    let dr_corrected = pim.delta_rotation_with_bias(&kf_i.bias);
    let dv_corrected = pim.delta_velocity_with_bias(&kf_i.bias);
    let dp_corrected = pim.delta_position_with_bias(&kf_i.bias);

    // r_R = Log(ΔR^T · R_bw_i · R_bw_j^T)
    let dr_t = Mat3F64(*dr_corrected.transpose());
    let lhs_r = Mat3F64(dr_t.mul_mat3(&mat3_mul(&r_bw_i, &r_wb_j)));
    let r_rot = SO3F64::from_matrix(&lhs_r).log();

    // r_v = R_bw_i · (v_j - v_i - g·dt) - Δv
    let dv_world = *v_j - *v_i - *gravity * dt;
    let r_vel = r_bw_i * dv_world - dv_corrected;

    // r_p = R_bw_i · (twb_j - twb_i - v_i·dt - ½g·dt²) - Δp
    let dp_world = twb_j - twb_i - *v_i * dt - *gravity * (0.5 * dt * dt);
    let r_pos = r_bw_i * dp_world - dp_corrected;

    // Bias random-walk residuals.
    let r_bg = kf_j.bias.gyro - kf_i.bias.gyro;
    let r_ba = kf_j.bias.accel - kf_i.bias.accel;

    let mut residual = [0.0f64; 15];
    residual[0..3].copy_from_slice(&[r_rot.x, r_rot.y, r_rot.z]);
    residual[3..6].copy_from_slice(&[r_vel.x, r_vel.y, r_vel.z]);
    residual[6..9].copy_from_slice(&[r_pos.x, r_pos.y, r_pos.z]);
    residual[9..12].copy_from_slice(&[r_bg.x, r_bg.y, r_bg.z]);
    residual[12..15].copy_from_slice(&[r_ba.x, r_ba.y, r_ba.z]);

    // Jacobians
    // State layout: [upsilon(0-2) | ω(3-5) | v(6-8) | bg(9-11) | ba(12-14)]
    // Right-SE3 perturbation: T_cw_new = T_cw · Exp([upsilon, ω])
    //   R_cw_new ≈ R_cw · (I + [ω]×)
    //   twb_new  ≈ twb - upsilon + [twb]× · ω
    // Velocity / bias DOFs: additive.

    let mut j_i = [0.0f64; 225];
    let mut j_j = [0.0f64; 225];

    let jr_inv = SO3F64::right_jacobian(r_rot).inverse();

    // ── rotation residual ────────────────────────────────────────────────
    // ∂r_R/∂ω_i = +J_r^{-1} · R_bw_j
    let jr_inv_rbwj = mat3_mul(&jr_inv, &r_bw_j);
    set_block_15x15(&mut j_i, 0, 3, &jr_inv_rbwj);

    // ∂r_R/∂bg_i = -J_r^{-1} · eR^T · JRg  (eR = ΔR^T · R_bw_i · R_wb_j = lhs_r)
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

    // ∂r_R/∂ω_j = -J_r^{-1} · R_bw_j
    set_block_15x15(&mut j_j, 0, 3, &mat3_neg(jr_inv_rbwj));

    // ── velocity residual ────────────────────────────────────────────────
    // ∂r_v/∂ω_i = -R_bw_i · [dv_world]×
    set_block_15x15(
        &mut j_i,
        3,
        3,
        &mat3_neg(mat3_mul(&r_bw_i, &SO3F64::hat(dv_world))),
    );

    let neg_r_bw_i = mat3_neg(r_bw_i);
    set_block_15x15(&mut j_i, 3, 6, &neg_r_bw_i); // ∂r_v/∂v_i = -R_bw_i
    set_block_15x15(&mut j_i, 3, 9, &mat3_neg(pim.d_velocity_d_bias_gyro)); // ∂r_v/∂bg_i
    set_block_15x15(&mut j_i, 3, 12, &mat3_neg(pim.d_velocity_d_bias_accel)); // ∂r_v/∂ba_i
    set_block_15x15(&mut j_j, 3, 6, &r_bw_i); // ∂r_v/∂v_j = +R_bw_i

    // ── position residual ────────────────────────────────────────────────
    // Under upsilon_i: twb_i shifts by -upsilon → r_p gains R_bw_i · upsilon
    set_block_15x15(&mut j_i, 6, 0, &r_bw_i); // ∂r_p/∂ρ_i = +R_bw_i

    // Under ω_i: R_bw_i rotates and twb_i shifts. Combined: -R_bw_i · [twb_j - v_i·dt - ½g·dt²]×
    let dp_omega_world = twb_j - *v_i * dt - *gravity * (0.5 * dt * dt);
    set_block_15x15(
        &mut j_i,
        6,
        3,
        &mat3_neg(mat3_mul(&r_bw_i, &SO3F64::hat(dp_omega_world))),
    );

    set_block_15x15(&mut j_i, 6, 6, &mat3_scalar_f64(&r_bw_i, -dt)); // ∂r_p/∂v_i = -R_bw_i·dt
    set_block_15x15(&mut j_i, 6, 9, &mat3_neg(pim.d_position_d_bias_gyro)); // ∂r_p/∂bg_i
    set_block_15x15(&mut j_i, 6, 12, &mat3_neg(pim.d_position_d_bias_accel)); // ∂r_p/∂ba_i

    // Under upsilon_j: twb_j shifts by -upsilon → r_p loses R_bw_i · upsilon
    set_block_15x15(&mut j_j, 6, 0, &neg_r_bw_i); // ∂r_p/∂ρ_j = -R_bw_i

    // Under ω_j: twb_j_new = twb_j + [twb_j]× · ω → r_p gains R_bw_i · [twb_j]× · ω
    set_block_15x15(&mut j_j, 6, 3, &mat3_mul(&r_bw_i, &SO3F64::hat(twb_j))); // ∂r_p/∂ω_j

    // ── bias random-walk residuals ───────────────────────────────────────
    set_block_15x15(&mut j_i, 9, 9, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 9, 9, &Mat3F64::IDENTITY);
    set_block_15x15(&mut j_i, 12, 12, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 12, 12, &Mat3F64::IDENTITY);

    (residual, j_i, j_j)
}

/// Accumulate J_a^T · Ω · J_b into m_mat[row_base:, col_base:] (15×15 block).
/// If `update_rhs`, also subtract J_a^T · Ω · r from m_vec[row_base:].
#[allow(clippy::too_many_arguments)]
fn accum_jt_omega_j(
    m_mat: &mut Mat<f64>,
    m_vec: &mut [f64],
    row_base: usize,
    col_base: usize,
    j_a: &[f64; 225],   // 15×15 row-major
    j_b: &[f64; 225],   // 15×15 row-major
    omega: &[f64; 225], // 15×15 row-major information matrix
    res: &[f64; 15],
    update_rhs: bool,
) {
    // omega_jb[r,c] = Σ_k omega[r,k] · j_b[k,c]
    let mut omega_jb = [0.0f64; 225];
    for r in 0..15 {
        for c in 0..15 {
            let mut s = 0.0f64;
            for k in 0..15 {
                s += omega[r * 15 + k] * j_b[k * 15 + c];
            }
            omega_jb[r * 15 + c] = s;
        }
    }

    // H[row_base+i, col_base+j] += Σ_k j_a[k,i] · omega_jb[k,j]
    for i in 0..15 {
        for j in 0..15 {
            let mut s = 0.0f64;
            for k in 0..15 {
                s += j_a[k * 15 + i] * omega_jb[k * 15 + j];
            }
            m_mat[(row_base + i, col_base + j)] += s;
        }
    }

    // g[row_base+i] -= Σ_k j_a[k,i] · (Σ_m omega[k,m] · res[m])
    if update_rhs {
        let mut omega_r = [0.0f64; 15];
        for k in 0..15 {
            for m in 0..15 {
                omega_r[k] += omega[k * 15 + m] * res[m];
            }
        }
        for i in 0..15 {
            let mut s = 0.0f64;
            for k in 0..15 {
                s += j_a[k * 15 + i] * omega_r[k];
            }
            m_vec[row_base + i] -= s;
        }
    }
}

/// Huber weight for one IMU edge based on the 9-dim nav residual's Mahalanobis distance.
///
/// chi2_nav = r_nav^T · Ω_nav · r_nav  (rows/cols 0–8 of the 15-dim system).
/// Returns 1.0 when chi2_nav ≤ threshold (inlier), sqrt(threshold / chi2_nav) otherwise.
/// The caller multiplies the full 15×15 omega by this weight before accumulating.
fn huber_imu_weight(res: &[f64; 15], omega: &[f64; 225], chi2_threshold: f64) -> f64 {
    // omega is 15×15 column-major: element (row r, col c) at index c*15+r.
    let mut chi2 = 0.0f64;
    for i in 0..9 {
        for j in 0..9 {
            chi2 += res[i] * omega[j * 15 + i] * res[j];
        }
    }
    if chi2 <= chi2_threshold || chi2 < 1e-12 {
        1.0
    } else {
        (chi2_threshold / chi2).sqrt()
    }
}

fn visual_huber_weight(chi2: f64, chi2_threshold: f64) -> f64 {
    HuberKernel.weight(chi2, chi2_threshold)
}

/// Diagonal information matrix from preintegrated IMU covariances.
/// Uses diagonal approximation (off-diagonal terms ignored).
/// Replace with full 15×15 inversion once FD tests pass.
fn imu_information_matrix(pim: &PreintegratedImu, weight: f64) -> [f64; 225] {
    let mut omega = [0.0f64; 225];

    // Full 9×9 nav block inverse (pim.covariance is column-major).
    // Diagonal-only inversion is incorrect because preintegration correlates
    // rotation, velocity, and position errors (gyro drives all three; accel
    // drives velocity and position).
    if let Some(nav_inv) = invert_9x9_f64(&pim.covariance) {
        for c in 0..9 {
            for r in 0..9 {
                omega[c * 15 + r] = weight * nav_inv[c * 9 + r];
            }
        }
    } else {
        // Degenerate covariance — fall back to diagonal with a log warning.
        eprintln!("[vi_ba] WARNING: 9×9 IMU covariance singular, using diagonal fallback");
        for i in 0..9 {
            let var = pim.covariance[i * 10]; // diagonal: col-major index i*9+i
            omega[i * 15 + i] = weight * if var > 1e-20 { 1.0 / var } else { 1e6 };
        }
    }

    // Bias random-walk block (6×6): uncorrelated, diagonal is exact.
    for i in 0..6 {
        let var = pim.bias_covariance[i * 7]; // diagonal: col-major index i*6+i
        omega[(9 + i) * 15 + (9 + i)] = weight * if var > 1e-20 { 1.0 / var } else { 1e6 };
    }
    omega
}

/// Invert a column-major 9×9 matrix via Gauss-Jordan elimination with partial pivoting.
fn invert_9x9_f64(cm: &[f64; 81]) -> Option<[f64; 81]> {
    // Work in row-major internally.
    let mut a = [0.0f64; 81];
    let mut inv = [0.0f64; 81];
    for r in 0..9 {
        for c in 0..9 {
            a[r * 9 + c] = cm[c * 9 + r]; // column-major → row-major
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
    // Row-major result → column-major output.
    let mut result = [0.0f64; 81];
    for r in 0..9 {
        for c in 0..9 {
            result[c * 9 + r] = inv[r * 9 + c];
        }
    }
    Some(result)
}

/// f64 version of invert_3x3 (the existing one uses f32).
fn invert_3x3_f64(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv,
        (m[2] * m[7] - m[1] * m[8]) * inv,
        (m[1] * m[5] - m[2] * m[4]) * inv,
        (m[5] * m[6] - m[3] * m[8]) * inv,
        (m[0] * m[8] - m[2] * m[6]) * inv,
        (m[2] * m[3] - m[0] * m[5]) * inv,
        (m[3] * m[7] - m[4] * m[6]) * inv,
        (m[1] * m[6] - m[0] * m[7]) * inv,
        (m[0] * m[4] - m[1] * m[3]) * inv,
    ])
}

/// f64 version of matmul_6x3_3x3.
fn matmul_6x3_3x3_f64(a: &[f64; 18], b: &[f64; 9]) -> [f64; 18] {
    let mut out = [0.0f64; 18];
    for i in 0..6 {
        for k in 0..3 {
            let mut s = 0.0f64;
            for r in 0..3 {
                s += a[i * 3 + r] * b[r * 3 + k];
            }
            out[i * 3 + k] = s;
        }
    }
    out
}

#[inline]
fn set_block_15x15(m: &mut [f64; 225], row: usize, col: usize, block: &Mat3F64) {
    let cols = block.to_cols_array(); // column-major from glam
    for c in 0..3 {
        for r in 0..3 {
            m[(row + r) * 15 + (col + c)] = cols[c * 3 + r];
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

/// Visual-inertial bundle adjustment via dense Schur-complement reduction.
///
/// Jointly optimises keyframe poses, velocities, and IMU biases (15 DOF each)
/// together with 3D landmark positions, connected by both reprojection factors
/// and IMU preintegration factors.
///
/// The Schur trick marginalises points exactly as in pure BA. After point
/// elimination, IMU edge normal equations are added directly to the reduced
/// 15P × 15P camera system before Cholesky solve. This matches Ceres's
/// DENSE_SCHUR strategy as used in ORB-SLAM3.
///
/// # Gauge freedom
/// Fix at least one keyframe (`fixed = true`) to anchor the metric scale and
/// world frame. The first keyframe is the natural choice.
pub fn visual_inertial_bundle_adjust(
    keyframes: &[ViBaKeyframe],
    points: &[Vec3F64],
    observations: &[BaObservation],
    imu_edges: &[ImuFactor],
    camera: &PinholeCamera,
    params: &ViBaParams,
) -> Result<ViBaResult, ViBaError> {
    const KF_DOF: usize = 15; // keyframe dimensions

    let n_kf = keyframes.len();
    let n_pts = points.len();

    // ── Index maps: free keyframes and free points ────────────────────────
    // A keyframe is free if it is not marked fixed AND appears in at least
    // one observation or IMU edge (otherwise it has no gradient, and we
    // skip it to avoid rank deficiency).

    let mut kf_touched = vec![false; n_kf];
    for obs in observations {
        if obs.pose_idx < n_kf {
            kf_touched[obs.pose_idx] = true;
        }
    }

    for edge in imu_edges {
        if edge.from_idx < n_kf {
            kf_touched[edge.from_idx] = true;
        }
        if edge.to_idx < n_kf {
            kf_touched[edge.to_idx] = true;
        }
    }

    let kf_local: Vec<i64> = {
        let mut v = vec![-1i64; n_kf];
        let mut next = 0i64;
        for i in 0..n_kf {
            if kf_touched[i] && !keyframes[i].fixed {
                v[i] = next;
                next += 1;
            }
        }
        v
    };

    let point_local: Vec<i64> = {
        let mut v = vec![-1i64; n_pts];
        let mut next = 0i64;
        for obs in observations {
            if obs.point_idx < n_pts && !obs.fixed_point && v[obs.point_idx] < 0 {
                v[obs.point_idx] = next;
                next += 1;
            }
        }
        v
    };

    let n_free_kf = kf_local.iter().filter(|&&x| x >= 0).count();
    let n_free_pts = point_local.iter().filter(|&&x| x >= 0).count();

    if n_free_kf == 0 {
        return Err(ViBaError::NoFreeVariables);
    }

    // Validate IMU edge indices.
    for edge in imu_edges {
        if edge.from_idx >= n_kf || edge.to_idx >= n_kf {
            return Err(ViBaError::ImuFactorOutOfRange(edge.from_idx, edge.to_idx));
        }
    }

    // ── Mutable state (f64 throughout — IMU needs the precision) ─────────
    let mut kfs: Vec<ViBaKeyframe> = keyframes.to_vec();
    let mut xyz: Vec<Vec3F64> = points.to_vec();
    // SE3F32 mirrors for the reprojection Jacobian (matches residual_and_jacobians).
    let mut se3s: Vec<SE3F32> = kfs.iter().map(|kf| pose_to_se3(&kf.pose)).collect();

    let mut lambda = params.initial_lambda;
    let mut iters_done = 0usize;
    let mut converged = false;
    let mut final_cost = f64::MAX;

    for _iter in 0..params.max_iterations {
        iters_done += 1;
        let is_first_iter = iters_done == 1;

        // ── 1. Schur blocks for visual residuals ─────────────────────────
        // These are exactly the a_blocks / c_blocks / b_by_point / g_pose /
        // g_point from bundle_adjust_schur_with_priors, but a_blocks are
        // 6×6 (pose only) and we scatter them into the 15×15 keyframe slot
        // when building m_mat.

        // a_blocks[k]: 6×6 visual Hessian for free keyframe k (pose DOF only).
        let mut a_blocks = vec![[0.0f64; 36]; n_free_kf];
        // c_blocks[j]: 3×3 point Hessian block.
        let mut c_blocks = vec![[0.0f64; 9]; n_free_pts];
        // g_pose[k]: 6-vector RHS for free keyframe k (pose DOF).
        let mut g_pose = vec![[0.0f64; 6]; n_free_kf];
        // g_point[j]: 3-vector RHS for free point j.
        let mut g_point = vec![[0.0f64; 3]; n_free_pts];
        // B cross-terms grouped by free point index.
        let mut b_by_point: Vec<Vec<(usize, [f64; 18])>> = vec![Vec::new(); n_free_pts];

        let mut cost_vis = 0.0f64;

        for obs in observations {
            if obs.pose_idx >= n_kf || obs.point_idx >= n_pts {
                continue;
            }

            let pose = &se3s[obs.pose_idx];
            let point = &xyz[obs.point_idx];
            // residual_and_jacobians is f32 internally; cast results to f64.
            let (r_f32, jp_f32, jx_f32) = residual_and_jacobians(pose, point, obs.pixel, camera);

            let r_raw = [r_f32[0] as f64, r_f32[1] as f64];
            let jp_raw: [f64; 12] = std::array::from_fn(|i| jp_f32[i] as f64);
            let jx_raw: [f64; 6] = std::array::from_fn(|i| jx_f32[i] as f64);

            let chi2_vis = r_raw[0] * r_raw[0] + r_raw[1] * r_raw[1];
            let hw_vis = visual_huber_weight(chi2_vis, params.huber_reproj_px_sq);
            cost_vis += 0.5 * hw_vis * chi2_vis;
            let sqrt_hw_vis = hw_vis.sqrt();
            let r = [r_raw[0] * sqrt_hw_vis, r_raw[1] * sqrt_hw_vis];
            let jp: [f64; 12] = std::array::from_fn(|i| jp_raw[i] * sqrt_hw_vis);
            let jx: [f64; 6] = std::array::from_fn(|i| jx_raw[i] * sqrt_hw_vis);

            let pli = kf_local[obs.pose_idx];
            let xli = point_local[obs.point_idx];

            // Accumulate A block (6×6).
            if pli >= 0 {
                let p = pli as usize;
                let ab = &mut a_blocks[p];
                for i in 0..6 {
                    for k in 0..6 {
                        ab[i * 6 + k] += jp[i] * jp[k] + jp[6 + i] * jp[6 + k];
                    }
                }
                let gp = &mut g_pose[p];
                for i in 0..6 {
                    gp[i] -= jp[i] * r[0] + jp[6 + i] * r[1];
                }
            }

            // Accumulate C block (3×3).
            if xli >= 0 {
                let x = xli as usize;
                let cb = &mut c_blocks[x];
                for i in 0..3 {
                    for k in 0..3 {
                        cb[i * 3 + k] += jx[i] * jx[k] + jx[3 + i] * jx[3 + k];
                    }
                }
                let gx = &mut g_point[x];
                for i in 0..3 {
                    gx[i] -= jx[i] * r[0] + jx[3 + i] * r[1];
                }
            }

            // Accumulate B block (6×3) for Schur.
            if pli >= 0 && xli >= 0 {
                let mut b = [0.0f64; 18];
                for i in 0..6 {
                    for k in 0..3 {
                        b[i * 3 + k] += jp[i] * jx[k] + jp[6 + i] * jx[3 + k];
                    }
                }
                b_by_point[xli as usize].push((pli as usize, b));
            }

            // Depth residual (stereo metric anchor) — see
            // depth_residual_and_jacobian for why this matters here.
            if let Some(d_meas) = obs.depth_meas {
                let sigma = (obs.depth_sigma as f64).max(1e-6);
                let (r_z_raw, jpd_raw, jxd_raw) =
                    depth_residual_and_jacobian(pose, point, d_meas as f64, sigma);
                let chi2_depth = r_z_raw * r_z_raw;
                let hw_depth = visual_huber_weight(chi2_depth, params.huber_depth_sigma_sq);
                cost_vis += 0.5 * hw_depth * chi2_depth;
                let sqrt_hw_depth = hw_depth.sqrt();
                let r_z = r_z_raw * sqrt_hw_depth;
                let jpd: [f64; 6] = std::array::from_fn(|i| jpd_raw[i] * sqrt_hw_depth);
                let jxd: [f64; 3] = std::array::from_fn(|i| jxd_raw[i] * sqrt_hw_depth);

                if pli >= 0 {
                    let p = pli as usize;
                    let ab = &mut a_blocks[p];
                    for i in 0..6 {
                        for k in 0..6 {
                            ab[i * 6 + k] += jpd[i] * jpd[k];
                        }
                    }
                    let gp = &mut g_pose[p];
                    for i in 0..6 {
                        gp[i] -= jpd[i] * r_z;
                    }
                }
                if xli >= 0 {
                    let x = xli as usize;
                    let cb = &mut c_blocks[x];
                    for i in 0..3 {
                        for k in 0..3 {
                            cb[i * 3 + k] += jxd[i] * jxd[k];
                        }
                    }
                    let gx = &mut g_point[x];
                    for i in 0..3 {
                        gx[i] -= jxd[i] * r_z;
                    }
                }
                if pli >= 0 && xli >= 0 {
                    let mut b = [0.0f64; 18];
                    for i in 0..6 {
                        for k in 0..3 {
                            b[i * 3 + k] += jpd[i] * jxd[k];
                        }
                    }
                    b_by_point[xli as usize].push((pli as usize, b));
                }
            }
        }

        // ── 2. Invert C blocks, build Schur-reduced system ───────────────
        // dim is 15*n_free_kf. Visual terms only touch the first 6 DOF of
        // each 15-DOF block; IMU terms fill the rest.
        let dim = n_free_kf * KF_DOF;
        let mut m_mat = Mat::<f64>::zeros(dim, dim);
        let mut m_vec = vec![0.0f64; dim];

        // Scatter A blocks into the upper-left 6×6 of each 15×15 slot.
        for (k, ab) in a_blocks.iter().enumerate() {
            let base = k * KF_DOF;
            for i in 0..6 {
                for j in 0..6 {
                    m_mat[(base + i, base + j)] = ab[i * 6 + j];
                }
            }
            m_vec[base..(6 + base)].copy_from_slice(&g_pose[k]);
        }

        // Schur point elimination: M -= B C⁻¹ Bᵀ,  m -= B C⁻¹ g_point.
        // Point back-substitution only needs the first 6 elements of each
        // keyframe's delta (pose DOF), so the B blocks are 6×3 as before.

        let c_inv: Vec<Option<[f64; 9]>> = c_blocks.iter().map(invert_3x3_f64).collect();

        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(ci) = c_inv[j] else {
                continue;
            };

            // Pre-multiply: BC_inv[i] = B[i,j] · C⁻¹  (6×3).
            let bc: Vec<(usize, [f64; 18])> = b_for_j
                .iter()
                .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3_f64(b, &ci)))
                .collect();

            // RHS correction: m[i] -= BC_inv[i] · g_point[j].
            for (i_loc, bc_block) in &bc {
                let base = i_loc * KF_DOF;
                for r in 0..6 {
                    let mut s = 0.0f64;
                    for k in 0..3 {
                        s += bc_block[r * 3 + k] * g_point[j][k];
                    }
                    m_vec[base + r] -= s;
                }
            }
            // LHS correction: M[i1,i2] -= BC_inv[i1] · B[i2,j]ᵀ.
            for (i1_loc, bc1) in &bc {
                for (i2_loc, b2) in b_for_j.iter() {
                    let row0 = i1_loc * KF_DOF;
                    let col0 = i2_loc * KF_DOF;
                    for r in 0..6 {
                        for c in 0..6 {
                            let mut s = 0.0f64;
                            for k in 0..3 {
                                s += bc1[r * 3 + k] * b2[c * 3 + k];
                            }
                            m_mat[(row0 + r, col0 + c)] -= s;
                        }
                    }
                }
            }
        }

        // ── 3. IMU normal equations ───────────────────────────────────────
        // Each edge contributes J_i^T Ω J_i, J_j^T Ω J_j (diagonal 15×15
        // blocks) and J_i^T Ω J_j, J_j^T Ω J_i (off-diagonal 15×15 blocks).
        let mut cost_imu = 0.0f64;

        for edge in imu_edges {
            let i = edge.from_idx;
            let j = edge.to_idx;
            let li = kf_local[i];
            let lj = kf_local[j];
            if li < 0 && lj < 0 {
                continue;
            }

            let (res, ji, jj) = imu_residual_and_jacobians(
                &kfs[i],
                &kfs[j],
                &edge.preintegrated,
                &params.gravity,
                params.imu_t_bc.as_ref(),
            );

            let mut omega_raw = imu_information_matrix(&edge.preintegrated, params.imu_weight);
            // Boundary edge (from a permanently-fixed KF into the active window):
            // downweight so this window's estimate can't accumulate error onto a
            // value that will never be revisited. See `boundary_imu_info_scale` doc.
            if li < 0 && lj >= 0 {
                for x in omega_raw.iter_mut() {
                    *x *= params.boundary_imu_info_scale;
                }
            }
            // Huber robustification: down-weight edges whose nav-block Mahalanobis
            // chi2 exceeds the threshold. This handles large residuals from rough
            // post-init velocity estimates without a fixed weight compromise.
            //
            // Only the nav 9×9 sub-block (rows/cols 0-8) is scaled by `hw` — the
            // bias-random-walk 6×6 sub-block (rows/cols 9-14) is a separate,
            // uncorrelated measurement (see `imu_information_matrix`) that ORB-SLAM3
            // robustifies independently (EdgeInertial vs EdgeGyroRW/EdgeAccRW are
            // distinct g2o edges). Scaling it by the *nav* residual's outlier-ness
            // was silencing the one thing meant to resist a bad nav window explaining
            // itself away through bias, in exactly the windows where that matters.
            let hw = huber_imu_weight(&res, &omega_raw, params.huber_imu_chi2);
            let omega: [f64; 225] = {
                let mut o = omega_raw;
                for c in 0..9 {
                    for r in 0..9 {
                        o[c * 15 + r] *= hw;
                    }
                }
                o
            };

            if is_first_iter {
                let nr = (res[0] * res[0] + res[1] * res[1] + res[2] * res[2]).sqrt();
                let nv = (res[3] * res[3] + res[4] * res[4] + res[5] * res[5]).sqrt();
                let np = (res[6] * res[6] + res[7] * res[7] + res[8] * res[8]).sqrt();
                let nbg = (res[9] * res[9] + res[10] * res[10] + res[11] * res[11]).sqrt();
                let nba = (res[12] * res[12] + res[13] * res[13] + res[14] * res[14]).sqrt();
                let om_min = omega_raw
                    .iter()
                    .cloned()
                    .filter(|&x| x > 0.0)
                    .fold(f64::MAX, f64::min);
                let om_max = omega_raw.iter().cloned().fold(f64::MIN, f64::max);
                eprintln!(
                    "[vi_ba] edge {i}→{j}  dt={:.3}s  |r_R|={nr:.4e}  |r_v|={nv:.4e}  \
                        |r_p|={np:.4e}  |r_bg|={nbg:.4e}  |r_ba|={nba:.4e}  \
                        Ω∈[{om_min:.2e},{om_max:.2e}]  hw={hw:.4}",
                    edge.preintegrated.dt,
                );
            }

            // Cost: ½ rᵀ Ω r.
            let mut omega_r = [0.0f64; 15];
            for r in 0..15 {
                for k in 0..15 {
                    omega_r[r] += omega[r * 15 + k] * res[k];
                }
            }
            for k in 0..15 {
                cost_imu += 0.5 * res[k] * omega_r[k];
            }

            // Accumulate H_ii, g_i.
            if li >= 0 {
                let base_i = li as usize * KF_DOF;
                accum_jt_omega_j(
                    &mut m_mat, &mut m_vec, base_i, base_i, &ji, &ji, &omega, &res, true,
                );
            }
            // Accumulate H_jj, g_j.
            if lj >= 0 {
                let base_j = lj as usize * KF_DOF;
                accum_jt_omega_j(
                    &mut m_mat, &mut m_vec, base_j, base_j, &jj, &jj, &omega, &res, true,
                );
            }
            // Off-diagonal H_ij and H_ji (symmetric pair).
            if li >= 0 && lj >= 0 {
                let base_i = li as usize * KF_DOF;
                let base_j = lj as usize * KF_DOF;
                accum_jt_omega_j(
                    &mut m_mat, &mut m_vec, base_i, base_j, &ji, &jj, &omega, &res, false,
                );
                accum_jt_omega_j(
                    &mut m_mat, &mut m_vec, base_j, base_i, &jj, &ji, &omega, &res, false,
                );
            }
        }

        // ── 3b. Accel-bias magnitude prior (unary, per free keyframe) ────────
        // See ViBaParams::accel_bias_prior_weight doc. Deliberately not scaled
        // by any Huber weight — this is the one bias regularizer that must
        // hold even when nav residuals are bad.
        let mut cost_bias_prior = 0.0f64;
        if params.accel_bias_prior_weight > 0.0 {
            let w = params.accel_bias_prior_weight;
            for (i, kf) in kfs.iter().enumerate() {
                let li = kf_local[i];
                if li < 0 {
                    continue;
                }
                let base = li as usize * KF_DOF;
                let ba = [kf.bias.accel.x, kf.bias.accel.y, kf.bias.accel.z];
                for d in 0..3 {
                    m_mat[(base + 12 + d, base + 12 + d)] += w;
                    m_vec[base + 12 + d] -= w * ba[d];
                    cost_bias_prior += 0.5 * w * ba[d] * ba[d];
                }
            }
        }

        let cost = (cost_vis + cost_imu + cost_bias_prior) as f32;

        // ── 4. LM damping on full 15P system, then Cholesky ──────────────
        // Damping goes on AFTER IMU terms so every diagonal entry sees it.
        // Scale it by each diagonal's own magnitude (the classic Marquardt
        // variant) rather than adding a bare `lambda`: this system mixes
        // pixel-space visual curvature (~1e2-1e4) with IMU information
        // entries spanning several more orders of magnitude — bias blocks
        // in particular land around ~1e9-1e10 here, since
        // imu_information_matrix inverts a bias-random-walk covariance
        // that's tiny over a single ~0.1-0.4s keyframe interval. A fixed
        // additive lambda is negligible next to a ~1e9 diagonal but can
        // dominate a ~1e2 one, so no single value can regularize both
        // regimes — this was leaving every VI-BA call unable to converge
        // within the iteration budget (observed 0/2319 conv=true across a
        // full EuRoC run). Scaling by the diagonal itself keeps damping
        // proportionate regardless of a DOF's native units; the floor
        // guards DOFs an untouched free keyframe leaves at exactly zero
        // (e.g. velocity/bias when it has no IMU edge in this window).
        const MIN_DAMPING_FLOOR: f64 = 1e-6;
        for d in 0..dim {
            let diag = m_mat[(d, d)].max(MIN_DAMPING_FLOOR);
            m_mat[(d, d)] += lambda * diag;
        }

        // Symmetrize (should already be symmetric to roundoff).
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (m_mat[(i, j)] + m_mat[(j, i)]);
                m_mat[(i, j)] = avg;
                m_mat[(j, i)] = avg;
            }
        }

        let chol = match m_mat.llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(e) => {
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Err(ViBaError::CholeskyFailed(format!("{e:?}")));
                }
                continue;
            }
        };

        let m_col = Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]);
        let d_full_col = chol.solve(&m_col);
        let d_full: Vec<f64> = (0..dim).map(|i| d_full_col[(i, 0)]).collect();

        // ── 5. Point back-substitution (uses pose-DOF slice only) ────────
        let mut d_point = vec![[0.0f64; 3]; n_free_pts];
        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(ci) = c_inv[j] else {
                continue;
            };
            let mut rhs = g_point[j];
            for (i_loc, b_block) in b_for_j {
                // Only the first 6 elements of d_full for this keyframe (pose).
                let base = i_loc * KF_DOF;
                let dp6: [f64; 6] = std::array::from_fn(|k| d_full[base + k]);
                // rhs -= B[i,j]ᵀ · δ_pose[i]  (B is 6×3, so Bᵀ is 3×6).
                for c in 0..3 {
                    let mut s = 0.0f64;
                    for r in 0..6 {
                        s += b_block[r * 3 + c] * dp6[r];
                    }
                    rhs[c] -= s;
                }
            }
            // δ_x = C⁻¹ · rhs.
            for r in 0..3 {
                d_point[j][r] =
                    ci[r * 3] * rhs[0] + ci[r * 3 + 1] * rhs[1] + ci[r * 3 + 2] * rhs[2];
            }
        }

        // ── 6. Trial retraction ───────────────────────────────────────────
        let mut kfs_trial = kfs.clone();
        let mut se3s_trial = se3s.clone();

        for i in 0..n_kf {
            let li = kf_local[i];
            if li < 0 {
                continue;
            }
            let base = li as usize * KF_DOF;

            // Pose delta: first 6 elements [ρ(3)|ω(3)].
            let pose_delta: [f32; 6] = std::array::from_fn(|k| d_full[base + k] as f32);
            let new_se3 = se3s[i].retract(&pose_delta);
            se3s_trial[i] = new_se3;
            kfs_trial[i].pose = se3_to_pose(&new_se3);

            // Velocity: additive.
            kfs_trial[i].velocity.x += d_full[base + 6];
            kfs_trial[i].velocity.y += d_full[base + 7];
            kfs_trial[i].velocity.z += d_full[base + 8];

            // Bias: additive.
            kfs_trial[i].bias.gyro.x += d_full[base + 9];
            kfs_trial[i].bias.gyro.y += d_full[base + 10];
            kfs_trial[i].bias.gyro.z += d_full[base + 11];
            kfs_trial[i].bias.accel.x += d_full[base + 12];
            kfs_trial[i].bias.accel.y += d_full[base + 13];
            kfs_trial[i].bias.accel.z += d_full[base + 14];
        }

        let mut xyz_trial = xyz.clone();
        for i in 0..n_pts {
            let xli = point_local[i];
            if xli < 0 {
                continue;
            }
            let dp = d_point[xli as usize];
            xyz_trial[i].x += dp[0];
            xyz_trial[i].y += dp[1];
            xyz_trial[i].z += dp[2];
        }

        // ── 7. Trial cost (visual + IMU) ─────────────────────────────────
        let mut new_cost_vis = 0.0f64;
        for obs in observations {
            if obs.pose_idx >= n_kf || obs.point_idx >= n_pts {
                continue;
            }
            let (r_f32, _, _) = residual_and_jacobians(
                &se3s_trial[obs.pose_idx],
                &xyz_trial[obs.point_idx],
                obs.pixel,
                camera,
            );
            let chi2_vis = (r_f32[0] * r_f32[0] + r_f32[1] * r_f32[1]) as f64;
            let hw_vis = visual_huber_weight(chi2_vis, params.huber_reproj_px_sq);
            new_cost_vis += 0.5 * hw_vis * chi2_vis;

            if let Some(d_meas) = obs.depth_meas {
                let sigma = (obs.depth_sigma as f64).max(1e-6);
                let (r_z, _, _) = depth_residual_and_jacobian(
                    &se3s_trial[obs.pose_idx],
                    &xyz_trial[obs.point_idx],
                    d_meas as f64,
                    sigma,
                );
                let chi2_depth = r_z * r_z;
                let hw_depth = visual_huber_weight(chi2_depth, params.huber_depth_sigma_sq);
                new_cost_vis += 0.5 * hw_depth * chi2_depth;
            }
        }
        let mut new_cost_imu = 0.0f64;
        for edge in imu_edges {
            let i = edge.from_idx;
            let j = edge.to_idx;
            if kf_local[i] < 0 && kf_local[j] < 0 {
                continue;
            }
            let (res, _, _) = imu_residual_and_jacobians(
                &kfs_trial[i],
                &kfs_trial[j],
                &edge.preintegrated,
                &params.gravity,
                params.imu_t_bc.as_ref(),
            );
            let mut omega_raw = imu_information_matrix(&edge.preintegrated, params.imu_weight);
            if kf_local[i] < 0 && kf_local[j] >= 0 {
                for x in omega_raw.iter_mut() {
                    *x *= params.boundary_imu_info_scale;
                }
            }
            let hw = huber_imu_weight(&res, &omega_raw, params.huber_imu_chi2);
            // Mirrors the block-split in the main Hessian build: `hw` only applies to
            // the nav 9×9 sub-block, the bias-random-walk 6×6 sub-block stays at full
            // weight. Kept as separate accumulators since `omega_raw` is block-diagonal
            // (no nav/bias cross terms — see `imu_information_matrix`).
            let mut omega_r_nav = [0.0f64; 9];
            for r in 0..9 {
                for k in 0..9 {
                    omega_r_nav[r] += omega_raw[r * 15 + k] * res[k];
                }
            }
            let mut nav_cost = 0.0f64;
            for k in 0..9 {
                nav_cost += 0.5 * res[k] * omega_r_nav[k];
            }
            let mut omega_r_bias = [0.0f64; 6];
            for r in 0..6 {
                for k in 0..6 {
                    omega_r_bias[r] += omega_raw[(9 + r) * 15 + (9 + k)] * res[9 + k];
                }
            }
            let mut bias_cost = 0.0f64;
            for k in 0..6 {
                bias_cost += 0.5 * res[9 + k] * omega_r_bias[k];
            }
            new_cost_imu += hw * nav_cost + bias_cost;
        }
        let mut new_cost_bias_prior = 0.0f64;
        if params.accel_bias_prior_weight > 0.0 {
            let w = params.accel_bias_prior_weight;
            for (i, kf) in kfs_trial.iter().enumerate() {
                if kf_local[i] < 0 {
                    continue;
                }
                let ba = &kf.bias.accel;
                new_cost_bias_prior += 0.5 * w * (ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
            }
        }
        let new_cost = (new_cost_vis + new_cost_imu + new_cost_bias_prior) as f32;

        // ── 8. LM accept / reject ─────────────────────────────────────────
        if new_cost < cost {
            let rel = if cost > 1e-12 {
                (cost - new_cost) as f64 / cost as f64
            } else {
                0.0
            };
            kfs = kfs_trial;
            se3s = se3s_trial;
            xyz = xyz_trial;
            final_cost = new_cost as f64;
            lambda = (lambda / 3.0).max(1e-8);
            if rel < params.cost_tolerance {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }

    // ── Post-convergence residual diagnostic ──────────────────────────────
    for edge in imu_edges {
        let i = edge.from_idx;
        let j = edge.to_idx;
        if kf_local[i] < 0 && kf_local[j] < 0 {
            continue;
        }
        let (res, _, _) = imu_residual_and_jacobians(
            &kfs[i],
            &kfs[j],
            &edge.preintegrated,
            &params.gravity,
            params.imu_t_bc.as_ref(),
        );
        let nr = (res[0] * res[0] + res[1] * res[1] + res[2] * res[2]).sqrt();
        let nv = (res[3] * res[3] + res[4] * res[4] + res[5] * res[5]).sqrt();
        let np = (res[6] * res[6] + res[7] * res[7] + res[8] * res[8]).sqrt();
        let nbg = (res[9] * res[9] + res[10] * res[10] + res[11] * res[11]).sqrt();
        let nba = (res[12] * res[12] + res[13] * res[13] + res[14] * res[14]).sqrt();
        eprintln!(
            "[vi_ba] FINAL edge {i}→{j}  dt={:.3}s  |r_R|={nr:.4e}  |r_v|={nv:.4e}  \
                |r_p|={np:.4e}  |r_bg|={nbg:.4e}  |r_ba|={nba:.4e}  \
                (iters={iters_done} conv={converged})",
            edge.preintegrated.dt,
        );
    }

    // ── Pack output ───────────────────────────────────────────────────────
    // Fixed keyframes get their original state back; free keyframes get
    // the optimised state.
    let out_kfs: Vec<ViBaKeyframe> = (0..n_kf)
        .map(|i| {
            if kf_local[i] >= 0 {
                kfs[i].clone()
            } else {
                keyframes[i].clone()
            }
        })
        .collect();
    let out_pts: Vec<Vec3F64> = (0..n_pts)
        .map(|i| {
            if point_local[i] >= 0 {
                xyz[i]
            } else {
                points[i]
            }
        })
        .collect();

    Ok(ViBaResult {
        keyframes: out_kfs,
        points: out_pts,
        iterations: iters_done,
        converged,
        final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::{ViBaParams, visual_huber_weight};

    #[test]
    fn visual_huber_defaults_and_weights() {
        let params = ViBaParams::default();
        assert_eq!(params.huber_reproj_px_sq, 25.0);
        assert_eq!(params.huber_depth_sigma_sq, 9.0);
        assert_eq!(visual_huber_weight(4.0, 25.0), 1.0);
        assert_eq!(visual_huber_weight(100.0, 25.0), 0.5);
    }
}
