use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, QuatF64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

use crate::map::{Keyframe, Map};
use crate::system::SystemState;

// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

fn vec_axis(v: Vec3F64, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => unreachable!("axis must be 0..3"),
    }
}

/// Gauss-Jordan elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || a.iter().any(|row| row.len() != n) || b.len() != n {
        return None;
    }
    for i in 0..n {
        // partial pivot
        let pivot = (i..n).max_by(|&r1, &r2| a[r1][i].abs().partial_cmp(&a[r2][i].abs()).unwrap())?;
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);

        let diag = a[i][i];
        for v in a[i].iter_mut().skip(i) { *v /= diag; }
        b[i] /= diag;

        for r in 0..n {
            if r == i { continue; }
            let f = a[r][i];
            if f == 0.0 { continue; }
            let pivot_row = a[i].clone();
            for (dst, src) in a[r].iter_mut().skip(i).zip(pivot_row.iter().skip(i)) {
                *dst -= f * src;
            }
            b[r] -= f * b[i];
        }
    }
    Some(b)
}

/// Normal-equations least squares: solves (AᵀA) x = Aᵀb.
fn solve_least_squares(rows: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    if rows.is_empty() || rows.len() != rhs.len() { return None; }
    let cols = rows[0].len();
    if cols == 0 { return None; }

    let mut ata = vec![vec![0.0f64; cols]; cols];
    let mut atb = vec![0.0f64; cols];
    for (row, &b) in rows.iter().zip(rhs) {
        for i in 0..cols {
            atb[i] += row[i] * b;
            for j in 0..cols { ata[i][j] += row[i] * row[j]; }
        }
    }
    solve_linear_system(ata, atb)
}

/// Rotation that takes unit vector `from` to unit vector `to`.
fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
    let from = from.normalize();
    let to   = to.normalize();
    let dot  = from.dot(to).clamp(-1.0, 1.0);
    let cross = from.cross(to);

    // Anti-parallel: pick an arbitrary perpendicular axis.
    if dot < -1.0 + 1e-9 {
        let perp = if from.x.abs() < 0.9 { Vec3F64::new(1.0, 0.0, 0.0) }
                   else                   { Vec3F64::new(0.0, 1.0, 0.0) };
        let axis = from.cross(perp).normalize();
        return SO3F64::from_quaternion(QuatF64::from_array([axis.x, axis.y, axis.z, 0.0]));
    }

    let w = ((1.0 + dot) / 2.0).sqrt();
    let s = 1.0 / (2.0 * w);
    SO3F64::from_quaternion(QuatF64::from_array([cross.x*s, cross.y*s, cross.z*s, w]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitConfig {
    pub min_keyframes: usize,
    pub min_time_sec:  f64,
    pub min_motion:    f64,
}

#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale:             f64,
    pub gravity_world:     Vec3F64,
    pub velocities_world:  Vec<Vec3F64>,
    pub bias:              ImuBias,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal solve result
// ─────────────────────────────────────────────────────────────────────────────

struct SolveResult {
    scale:            f64,
    gravity_world:    Vec3F64,   // world-frame gravity vector (magnitude ≈ G after step 1)
    velocities_world: Vec<Vec3F64>,
    accel_bias:       Vec3F64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ImuInitializer
// ─────────────────────────────────────────────────────────────────────────────

pub struct ImuInitializer {
    pub config: ImuInitConfig,
}

impl ImuInitializer {
    pub fn new(config: ImuInitConfig) -> Self { Self { config } }

    // ── readiness gate ────────────────────────────────────────────────────────

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        let Some(start_idx) = start_idx else { return false; };

        let kfs: Vec<&Keyframe> = map.keyframes().iter()
            .filter(|kf| kf.frame.idx >= start_idx).collect();
        if kfs.len() < self.config.min_keyframes { return false; }

        let imu_time: f64 = map.imu_factors().iter()
            .filter(|f| f.curr_kf_idx >= start_idx)
            .map(|f| f.preintegrated.dt).sum();
        if imu_time < self.config.min_time_sec { return false; }

        let t0 = kfs.first().unwrap().frame.pose_world_to_cam.inverse().translation;
        let t1 = kfs.last() .unwrap().frame.pose_world_to_cam.inverse().translation;
        (t1 - t0).length() >= self.config.min_motion
    }

    // ── main entry point ──────────────────────────────────────────────────────

    pub fn try_initialize(
        &self,
        map:       &Map,
        imu_t_bc:  Option<Pose3d>,
        imu_bias:  ImuBias,
        start_idx: usize,
    ) -> Option<ImuInitResult> {
        let imu_t_bc = imu_t_bc?;

        // Sorted keyframes in the initialization window.
        let mut keyframes: Vec<&Keyframe> = map.keyframes().iter()
            .filter(|kf| kf.frame.idx >= start_idx).collect();
        keyframes.sort_by_key(|kf| kf.frame.idx);
        let n = keyframes.len();
        if n < self.config.min_keyframes { return None; }

        // Local index map: frame_idx → position in `keyframes`.
        let frame_to_local: HashMap<usize, usize> = keyframes.iter()
            .enumerate().map(|(i, kf)| (kf.frame.idx, i)).collect();

        // ── Step 1: gyroscope bias ────────────────────────────────────────────
        let mut bias = imu_bias;
        if let Some(bg) = self.estimate_gyro_bias(map, &keyframes, &frame_to_local, imu_t_bc) {
            bias.gyro = bg;
        }
        println!("[imu_init] gyro_bias = ({:.5}, {:.5}, {:.5})",
            bias.gyro.x, bias.gyro.y, bias.gyro.z);

        let is_mono = !keyframes.first().map(|kf| kf.frame.is_stereo()).unwrap_or(false);

        // ── Step 2: scale + gravity (no accel bias, free gravity direction) ───
        let step2 = self.solve_linear(
            map, &keyframes, &frame_to_local, imu_t_bc, &bias,
            None,       // no gravity prior yet
            false,      // don't fix gravity magnitude yet
            is_mono,
        )?;
        println!("[imu_init] step2 scale={:.4}  g=({:.3},{:.3},{:.3})  |g|={:.3}",
            step2.scale,
            step2.gravity_world.x, step2.gravity_world.y, step2.gravity_world.z,
            step2.gravity_world.length());

        // ── Step 3: refine scale + gravity direction + accel bias ─────────────
        // We iterate twice as the paper suggests; each pass tightens the gravity
        // direction constraint and estimates the accel bias.
        let mut gravity_dir = step2.gravity_world.normalize();
        let mut accel_bias  = Vec3F64::ZERO;
        let mut step3: Option<SolveResult> = None;

        for iter in 0..2 {
            let result = self.solve_linear(
                map, &keyframes, &frame_to_local, imu_t_bc,
                &ImuBias { gyro: bias.gyro, accel: accel_bias },
                Some(gravity_dir),  // constrain gravity direction, solve 2-DOF perturbation
                true,               // fix gravity magnitude to G
                is_mono,
            )?;
            gravity_dir = result.gravity_world.normalize();
            accel_bias  = result.accel_bias;
            println!("[imu_init] step3 iter={} scale={:.4}  g=({:.3},{:.3},{:.3})  \
                      ba=({:.4},{:.4},{:.4})",
                iter, result.scale,
                result.gravity_world.x, result.gravity_world.y, result.gravity_world.z,
                accel_bias.x, accel_bias.y, accel_bias.z);
            step3 = Some(result);
        }
        let step3 = step3?;

        // Sanity checks before applying.
        let g_mag = step3.gravity_world.length();
        if !step3.scale.is_finite() || step3.scale <= 1e-6 || step3.scale > 100.0 {
            println!("[imu_init] rejected: bad scale {:.4}", step3.scale);
            return None;
        }
        if (g_mag - GRAVITY_MAGNITUDE).abs() > 2.0 {
            println!("[imu_init] rejected: |g|={:.3} too far from {:.3}", g_mag, GRAVITY_MAGNITUDE);
            return None;
        }
        if accel_bias.length() > 1.0 {
            println!("[imu_init] rejected: |ba|={:.3} too large", accel_bias.length());
            return None;
        }

        bias.accel = accel_bias;

        let v_first = step3.velocities_world.first().copied().unwrap_or(Vec3F64::ZERO);
        let v_last = step3.velocities_world.last().copied().unwrap_or(Vec3F64::ZERO);
        println!(
            "[imu_init] accepted  scale={:.4}  |g|={:.4}  \
             ba=({:.6},{:.6},{:.6})  \
             v_first=({:.4},{:.4},{:.4}) |{:.4}|  \
             v_last=({:.4},{:.4},{:.4}) |{:.4}|",
            step3.scale, g_mag,
            accel_bias.x, accel_bias.y, accel_bias.z,
            v_first.x, v_first.y, v_first.z, v_first.length(),
            v_last.x, v_last.y, v_last.z, v_last.length(),
        );

        Some(ImuInitResult {
            scale:            step3.scale,
            gravity_world:    gravity_dir * GRAVITY_MAGNITUDE,
            velocities_world: step3.velocities_world,
            bias,
        })
    }

    // ── apply to map & state ──────────────────────────────────────────────────

    pub fn apply_initialization(
        &self,
        map:           &mut Map,
        state:         &mut SystemState,
        imu_bias:      &mut ImuBias,
        gravity_world: &mut Vec3F64,
        init:          ImuInitResult,
        start_idx:     usize,
    ) {
        println!("[imu_init] applying: scale={:.4}  g=({:.3},{:.3},{:.3})",
            init.scale,
            init.gravity_world.x, init.gravity_world.y, init.gravity_world.z);

        // 1. Bring the monocular map to metric scale.
        map.scale_world(init.scale);

        // 2. Rotate world so that gravity aligns with +Y (OpenCV convention).
        let g_norm = init.gravity_world / init.gravity_world.length();
        let rwg    = rotation_from_to(g_norm, Vec3F64::new(0.0, 1.0, 0.0));
        map.rotate_world(&rwg);

        // 3. Assign velocities and biases to every keyframe in the window.
        let mut vel_iter = init.velocities_world.into_iter();
        for kf in map.keyframes_mut().iter_mut()
            .filter(|kf| kf.frame.idx >= start_idx)
        {
            if let Some(v) = vel_iter.next() {
                kf.velocity_world = rwg * v;
                kf.imu_bias       = init.bias;
            }
        }

        // 4. Update the tracker state from the last initialized keyframe.
        if let Some(last_kf) = map.keyframes().iter()
            .rfind(|kf| kf.frame.idx >= start_idx)
        {
            state.velocity_world   = last_kf.velocity_world;
            state.pose_world_to_cam = last_kf.frame.pose_world_to_cam;
        }

        state.velocity      = None;
        state.imu_initialized = true;
        *gravity_world      = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        *imu_bias           = init.bias;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 1: Gyroscope bias estimation  (paper §IV-A, Eq. 9)
    // ─────────────────────────────────────────────────────────────────────────

    fn estimate_gyro_bias(
        &self,
        map:            &Map,
        keyframes:      &[&Keyframe],
        frame_to_local: &HashMap<usize, usize>,
        imu_t_bc:       Pose3d,
    ) -> Option<Vec3F64> {
        // R_CB : rotation that takes body vectors to camera frame.
        let r_cb = imu_t_bc.inverse().rotation;
        let mut gyro_bias = Vec3F64::ZERO;

        for _ in 0..3 {
            let mut h = vec![vec![0.0f64; 3]; 3];
            let mut b = vec![0.0f64; 3];
            let mut n = 0usize;

            for factor in map.imu_factors() {
                let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else { continue; };
                let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else { continue; };
                if factor.preintegrated.dt <= 0.0 { continue; }

                // R_WB at keyframes i and j.
                let r_wb_i = keyframes[i].frame.pose_world_to_cam.inverse().rotation * r_cb;
                let r_wb_j = keyframes[j].frame.pose_world_to_cam.inverse().rotation * r_cb;

                // Relative orientation from vision: R_BiBj = R_WBi^T * R_WBj
                let r_vis = Mat3F64(*r_wb_i.transpose()) * r_wb_j;

                // Preintegrated rotation corrected for current gyro bias estimate.
                let d_rot = factor.preintegrated.delta_rotation_with_bias(
                    &ImuBias { gyro: gyro_bias, accel: Vec3F64::ZERO },
                );

                // Residual: Log( ΔR^T * R_vis )   (should be zero at optimum)
                let resid = SO3F64::from_matrix(
                    &(Mat3F64(*d_rot.transpose()) * r_vis)
                ).log();
                let r = [resid.x, resid.y, resid.z];

                // J_Δp_bg : Jacobian of preintegrated rotation w.r.t. gyro bias.
                let jac = factor.preintegrated.d_rotation_d_bias_gyro.to_cols_array();

                // Accumulate normal equations: (J^T J) Δbg = J^T r
                for row in 0..3 {
                    for col in 0..3 {
                        for k in 0..3 { h[row][col] += jac[row*3+k] * jac[col*3+k]; }
                    }
                    for k in 0..3 { b[row] += jac[row*3+k] * r[k]; }
                }
                n += 1;
            }

            if n < 2 { return None; }

            let delta = solve_linear_system(h, b)?;
            let step  = Vec3F64::new(delta[0], delta[1], delta[2]);
            gyro_bias += step;
            if step.length() < 1e-8 { break; }
        }

        if !gyro_bias.length().is_finite() || gyro_bias.length() > 0.5 {
            return None;
        }
        Some(gyro_bias)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unified linear solve  (Steps 2 and 3 of the paper)
    //
    // The system is built directly from the preintegration equations (paper
    // Eq. 3 / 11) *without* the triplet velocity-elimination trick.  We solve
    // for all velocities together with scale (optional) and gravity.
    //
    // Unknowns layout  (column order in each row vector):
    //   [v₀ₓ v₀ᵧ v₀z | v₁ₓ … | … | vₙₓ vₙᵧ vₙz | g₀ g₁ (g₂?) | (s?)]
    //
    //   gravity block:
    //     gravity_prior = None  → free 3-vector  (Step 2)
    //     gravity_prior = Some  → 2-DOF tangent perturbation  (Step 3)
    //
    //   scale column:
    //     solve_scale = true  → one extra column for −visual_dp   (monocular)
    //     solve_scale = false → no scale column (stereo, s=1 known)
    //
    //   accel_bias block:
    //     fix_gravity_magnitude = true  → 3 extra columns for J_ba  (Step 3)
    //     fix_gravity_magnitude = false → no accel-bias columns      (Step 2)
    //
    // Two equation types per IMU factor (i → j):
    //   Position:  v_i * dt + 0.5*g*dt² + (−s*visual) = Rwb_i*Δp − lever_dp
    //   Velocity:  v_j − v_i − g*dt = Rwb_i*Δv  [+ accel-bias correction]
    // ─────────────────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn solve_linear(
        &self,
        map:                   &Map,
        keyframes:             &[&Keyframe],
        frame_to_local:        &HashMap<usize, usize>,
        imu_t_bc:              Pose3d,
        bias:                  &ImuBias,
        gravity_prior:         Option<Vec3F64>, // unit vector; None → free solve
        fix_gravity_magnitude: bool,            // true → Step 3 (also solves accel bias)
        solve_scale:           bool,
    ) -> Option<SolveResult> {

        let t_cb  = imu_t_bc.inverse();
        let r_cb  = t_cb.rotation;           // R_CB
        let lever = t_cb.translation;        // ^C p_B  (body origin in camera frame)
        let n     = keyframes.len();

        // ── Column layout ─────────────────────────────────────────────────────
        //   velocities : 3*n columns   cols 0 … 3n-1
        //   gravity    : 2 or 3 cols   cols 3n … 3n+ng-1
        //   accel_bias : 3 cols        cols 3n+ng … 3n+ng+2   (only Step 3)
        //   scale      : 1 col         cols 3n+ng(+3)?        (only monocular)

        let n_g_cols = if gravity_prior.is_some() { 2usize } else { 3usize };
        let n_ba_cols = if fix_gravity_magnitude  { 3usize } else { 0usize };
        let n_s_cols  = if solve_scale            { 1usize } else { 0usize };

        let vel_base    = 0usize;
        let g_base      = 3 * n;
        let ba_base     = g_base  + n_g_cols;
        let s_col       = ba_base + n_ba_cols;  // only valid when solve_scale
        let total_cols  = s_col   + n_s_cols;

        let vel_col = |kf: usize, ax: usize| vel_base + 3*kf + ax;

        // ── Gravity parameterization (Step 3 only) ────────────────────────────
        // When a gravity_prior direction d̂ is given we write
        //   g = G*d̂ + B * δ       where B = [b1 | b2] is a tangent basis
        // The two free scalars δ = [δ1, δ2]^T are the unknowns.
        // After solving, we reconstruct:  g_new = G*d̂ + b1*δ1 + b2*δ2
        // then normalise to get the updated gravity direction.

        let (r_wi_opt, tb1_opt, tb2_opt) = match gravity_prior {
            None => (None, None, None),
            Some(dir) => {
                // Tangent basis orthogonal to dir.
                let pick = if dir.x.abs() < 0.9 { Vec3F64::new(1.0,0.0,0.0) }
                           else                  { Vec3F64::new(0.0,1.0,0.0) };
                let b1 = dir.cross(pick).normalize();
                let b2 = dir.cross(b1).normalize();
                // R_WI : rotation from reference-inertial (+Y) to gravity_prior direction.
                let r_wi = rotation_from_to(Vec3F64::new(0.0,1.0,0.0), dir);
                (Some(r_wi), Some(b1), Some(b2))
            }
        };

        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs:  Vec<f64>      = Vec::new();

        for factor in map.imu_factors() {
            let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else { continue; };
            let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else { continue; };
            let dt = factor.preintegrated.dt;
            if dt <= 0.0 { continue; }

            let cam_i = keyframes[i].frame.pose_world_to_cam.inverse();
            let cam_j = keyframes[j].frame.pose_world_to_cam.inverse();
            let r_wb_i = cam_i.rotation * r_cb;  // R_WB at keyframe i

            // Preintegrated quantities rotated to world frame.
            let dp_world = r_wb_i * factor.preintegrated.delta_position_with_bias(bias);
            let dv_world = r_wb_i * factor.preintegrated.delta_velocity_with_bias(bias);

            // Lever-arm change in world frame:  (R_WCj − R_WCi) * ^C p_B
            let lever_dp = cam_j.rotation * lever - cam_i.rotation * lever;

            // Visual displacement (unscaled monocular camera-centre difference).
            let visual_dp = cam_j.translation - cam_i.translation;

            // Accel-bias Jacobians (in body frame, will be rotated to world).
            let jp = factor.preintegrated.d_position_d_bias_accel; // 3×3
            let jv = factor.preintegrated.d_velocity_d_bias_accel; // 3×3

            // ── POSITION EQUATION (3 scalar rows) ────────────────────────────
            // From Eq. 11:
            //   s*(p_j − p_i) = v_i*dt + 0.5*g*dt² + R_WBi*Δp + lever_dp
            // Move knowns to RHS, unknowns to LHS:
            //   v_i*dt − s*visual_dp + 0.5*dt²*g = dp_world + lever_dp
            // Gravity column contribution depends on parameterisation.

            for ax in 0..3usize {
                let mut row = vec![0.0f64; total_cols];

                // velocity_i coefficient
                row[vel_col(i, ax)] = dt;

                // scale coefficient (monocular only)
                if solve_scale {
                    row[s_col] = -vec_axis(visual_dp, ax);
                }

                // RHS: dp_world + lever_dp  [+ visual_dp when stereo, s=1]
                let mut b_val = vec_axis(dp_world + lever_dp, ax);
                if !solve_scale {
                    b_val -= vec_axis(visual_dp, ax); // s=1 known → move to RHS
                }

                // Gravity column(s)
                match gravity_prior {
                    None => {
                        // Step 2: free 3-vector g.
                        // Equation: v_i*dt + 0.5*g*dt² − s*visual = dp_world + lever_dp
                        // Coefficient of g[ax] = 0.5*dt²
                        row[g_base + ax] = 0.5 * dt * dt;
                    }
                    Some(g_dir) => {
                        // Step 3: g = G*g_dir + b1*δ1 + b2*δ2
                        // Equation: v_i*dt + 0.5*(G*g_dir + B*δ)*dt² − s*visual = dp_world + lever_dp
                        // Move zeroth-order term G*g_dir to RHS:
                        //   v_i*dt + 0.5*B*δ*dt² − s*visual = dp_world + lever_dp − 0.5*G*g_dir*dt²
                        let g0 = g_dir * GRAVITY_MAGNITUDE;
                        b_val -= 0.5 * dt * dt * vec_axis(g0, ax);
                        // Tangent perturbation columns
                        row[g_base + 0] = 0.5 * dt * dt * vec_axis(tb1_opt.unwrap(), ax);
                        row[g_base + 1] = 0.5 * dt * dt * vec_axis(tb2_opt.unwrap(), ax);
                    }
                }

                // Accel-bias columns (Step 3 only)
                // R_WBi * J_Δp_ba * ba  →  each column is R_WBi applied to one column of J
                if fix_gravity_magnitude {
                    let rwb_jp = r_wb_i * jp; // 3×3 world-frame Jacobian
                    for k in 0..3usize {
                        // column k of rwb_jp is the world-frame sensitivity to ba[k]
                        let col_k = mat3_col(rwb_jp, k);
                        row[ba_base + k] = -vec_axis(col_k, ax); // moves to LHS
                    }
                    // The bias correction is already inside dp_world via
                    // delta_position_with_bias, so no separate RHS term needed.
                }

                rows.push(row);
                rhs.push(b_val);
            }

            // ── VELOCITY EQUATION (3 scalar rows) ────────────────────────────
            // From Eq. 3:
            //   v_j = v_i + g*dt + R_WBi*Δv
            // →  v_j − v_i − g*dt = dv_world

            for ax in 0..3usize {
                let mut row = vec![0.0f64; total_cols];

                row[vel_col(j, ax)] =  1.0;
                row[vel_col(i, ax)] = -1.0;

                let mut b_val = vec_axis(dv_world, ax);

                match gravity_prior {
                    None => {
                        // Step 2: g is a free 3-vector unknown.
                        // Equation: v_j − v_i − g*dt = dv_world
                        // → coefficient of g[ax] = −dt
                        row[g_base + ax] = -dt;
                    }
                    Some(g_dir) => {
                        // Step 3: g = G*g_dir + b1*δ1 + b2*δ2
                        // Equation: v_j − v_i − g*dt = dv_world
                        // Expand: v_j − v_i − (G*g_dir + b1*δ1 + b2*δ2)*dt = dv_world
                        // Move the known zeroth-order term to RHS:
                        //   v_j − v_i − b1*δ1*dt − b2*δ2*dt = dv_world + G*g_dir*dt
                        let g0 = g_dir * GRAVITY_MAGNITUDE;
                        b_val += dt * vec_axis(g0, ax);  // zeroth-order gravity → RHS
                        row[g_base + 0] = -dt * vec_axis(tb1_opt.unwrap(), ax);
                        row[g_base + 1] = -dt * vec_axis(tb2_opt.unwrap(), ax);
                    }
                }

                // Accel-bias columns for velocity equation.
                if fix_gravity_magnitude {
                    let rwb_jv = r_wb_i * jv;
                    for k in 0..3usize {
                        let col_k = mat3_col(rwb_jv, k);
                        row[ba_base + k] = -vec_axis(col_k, ax);
                    }
                }

                rows.push(row);
                rhs.push(b_val);
            }
        }

        if rows.len() < total_cols { return None; }

        let sol = solve_least_squares(&rows, &rhs)?;

        // ── Extract velocities ────────────────────────────────────────────────
        let velocities: Vec<Vec3F64> = (0..n).map(|i| Vec3F64::new(
            sol[vel_col(i, 0)], sol[vel_col(i, 1)], sol[vel_col(i, 2)],
        )).collect();

        // ── Extract gravity ───────────────────────────────────────────────────
        let gravity_world = match gravity_prior {
            None => {
                // Free solve: gravity vector is directly in solution.
                Vec3F64::new(sol[g_base], sol[g_base+1], sol[g_base+2])
            }
            Some(g_dir) => {
                // Tangent update: g_new = G*g_dir + b1*δ1 + b2*δ2, then re-normalise to G.
                let delta = tb1_opt.unwrap() * sol[g_base]
                          + tb2_opt.unwrap() * sol[g_base+1];
                let g_new = g_dir * GRAVITY_MAGNITUDE + delta;
                // Renormalise so downstream code always sees |g| ≈ G.
                g_new.normalize() * GRAVITY_MAGNITUDE
            }
        };

        if !gravity_world.length().is_finite() || gravity_world.length() < 1e-3 {
            return None;
        }

        // ── Extract accel bias ────────────────────────────────────────────────
        let accel_bias = if fix_gravity_magnitude {
            Vec3F64::new(sol[ba_base], sol[ba_base+1], sol[ba_base+2])
        } else {
            Vec3F64::ZERO
        };

        // ── Extract scale ─────────────────────────────────────────────────────
        let scale = if solve_scale { sol[s_col] } else { 1.0 };

        Some(SolveResult { scale, gravity_world, velocities_world: velocities, accel_bias })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract column k of a 3×3 matrix stored in column-major order.
// ─────────────────────────────────────────────────────────────────────────────

fn mat3_col(m: Mat3F64, col: usize) -> Vec3F64 {
    let a = m.to_cols_array(); // column-major: a[0..3]=col0, a[3..6]=col1, a[6..9]=col2
    match col {
        0 => Vec3F64::new(a[0], a[1], a[2]),
        1 => Vec3F64::new(a[3], a[4], a[5]),
        2 => Vec3F64::new(a[6], a[7], a[8]),
        _ => unreachable!(),
    }
}