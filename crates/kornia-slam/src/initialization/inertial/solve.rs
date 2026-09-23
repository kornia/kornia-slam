//! The inertial-only optimization behind [`ImuInitializer::try_initialize`]:
//! building the problem over the window's keyframes, solving it, and checking
//! the solution.

use super::*;

/// Pack a `Vec3F64` into the `Vec<f32>` value layout `kornia_algebra::optim`
/// variables use.
fn vec3_to_f32(v: Vec3F64) -> Vec<f32> {
    vec![v.x as f32, v.y as f32, v.z as f32]
}

/// Read the first three components of an optimizer variable back into a `Vec3F64`.
fn vec3_from_var(values: &[f32]) -> Vec3F64 {
    Vec3F64::new(values[0] as f64, values[1] as f64, values[2] as f64)
}

impl ImuInitializer {
    // ── main entry point ──────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub fn inertial_optimizer(
        &self,
        map: &Map,
        frame_to_local: &HashMap<usize, usize>,
        keyframes: Vec<&Keyframe>,
        rwg: Mat3F64,
        seed_velocities: &[Vec3F64],
        scale_init: f64,
        bg: Vec3F64,
        ba: Vec3F64,
        is_mono: bool,
        prior_g: f64,
        prior_a: f64,
        imu_t_bc: Pose3d,
    ) -> Option<(Mat3F64, f64, Vec3F64, Vec3F64, Vec<Vec3F64>)> {
        let n = keyframes.len();
        let mut problem = Problem::new();

        for (local_idx, vel) in seed_velocities.iter().enumerate().take(n) {
            let name = format!("v{}", local_idx);
            problem
                .add_variable(Variable::euclidean(&name, 3), vec3_to_f32(*vel))
                .ok()?;
        }
        problem
            .add_variable(Variable::euclidean("bg", 3), vec3_to_f32(bg))
            .ok()?;
        problem
            .add_variable(Variable::euclidean("ba", 3), vec3_to_f32(ba))
            .ok()?;
        let q = SO3F64::from_matrix(&rwg).to_array();
        let g_f32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        problem
            .add_variable(
                Variable::new("gdir", VariableType::SO3, g_f32.clone()),
                g_f32,
            )
            .ok()?;
        problem
            .add_variable(Variable::euclidean("scale", 1), vec![scale_init as f32])
            .ok()?;
        let kf_const: Vec<KfConst> = keyframes
            .iter()
            .map(|kf| KfConst::new(&kf.frame.pose_world_to_cam, &imu_t_bc))
            .collect();

        let mut edge_count = 0;
        for factor in map.imu_factors() {
            // Factors crossing the window boundary (e.g. the mono bootstrap's
            // reference-keyframe -> first-in-window-keyframe factor, whose
            // prev_kf_idx sits before start_idx) are skipped, not fatal —
            // using `?` here previously made every call silently return None
            // as soon as any such factor existed, which mono always has and
            // stereo never does.
            let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else {
                continue;
            };
            let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else {
                continue;
            };

            if factor.preintegrated.dt <= 0.0 {
                continue;
            }

            let f = InertialInitFactor::new(
                kf_const[i],
                kf_const[j],
                factor.preintegrated.clone(),
                is_mono,
            );
            problem
                .add_factor(
                    Box::new(f),
                    vec![
                        format!("v{i}"),
                        format!("v{j}"),
                        "bg".to_string(),
                        "ba".to_string(),
                        "gdir".to_string(),
                        "scale".to_string(),
                    ],
                )
                .ok()?;
            edge_count += 1;
        }

        if edge_count == 0 {
            return None;
        }

        problem
            .add_factor(
                Box::new(WeightedZeroPrior {
                    sqrt_weight: prior_g.sqrt(),
                }),
                vec!["bg".to_string()],
            )
            .ok()?;
        problem
            .add_factor(
                Box::new(WeightedZeroPrior {
                    sqrt_weight: prior_a.sqrt(),
                }),
                vec!["ba".to_string()],
            )
            .ok()?;

        let lm = LevenbergMarquardt {
            max_iterations: 200,
            lambda_init: if prior_g != 0.0 { 1e3 } else { 1e-3 },
            ..Default::default()
        };

        let result = lm.optimize(&mut problem).ok()?;
        eprintln!(
            "[inertial_optimizer] {:?} after {} iters, final_cost={:.6}",
            result.termination_reason, result.iterations, result.final_cost
        );
        let vars = problem.get_variables(); // &HashMap<String, Variable>

        let bg_out = vec3_from_var(&vars["bg"].values);
        let ba_out = vec3_from_var(&vars["ba"].values);
        let gdir_v = &vars["gdir"].values;
        let rwg_out = SO3F64::from_array([
            gdir_v[0] as f64,
            gdir_v[1] as f64,
            gdir_v[2] as f64,
            gdir_v[3] as f64,
        ])
        .matrix();
        let scale_out = if is_mono {
            vars["scale"].values[0] as f64
        } else {
            1.0
        };

        let velocities_out: Vec<Vec3F64> = (0..n)
            .map(|i| vec3_from_var(&vars[&format!("v{i}")].values))
            .collect();

        Some((rwg_out, scale_out, bg_out, ba_out, velocities_out))
    }

    /// Mirrors ORB-SLAM3's `LocalMapping::InitializeIMU(priorG, priorA, bFIBA)`:
    /// a *single* joint LM solve per call. The pipeline is responsible for the
    /// progressive VIBA0 (immediate) / VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s)
    /// re-triggering schedule with progressively relaxed priors — this
    /// function does not chain multiple passes internally.
    ///
    /// `already_initialized` selects the same branch ORB-SLAM3 does at
    /// LocalMapping.cc:1226 (`!isImuInitialized()`): on the first (VIBA0)
    /// call, Rwg and per-keyframe velocities are derived from scratch via the
    /// visual trajectory (finite-difference velocity + gravity-direction
    /// accumulation); on later refinement calls the map is already
    /// gravity-aligned and metric (VIBA0's result was applied), so Rwg seeds
    /// as identity and velocities seed from each keyframe's current
    /// IMU-propagated estimate instead of being re-derived visually.
    #[allow(clippy::too_many_arguments)]
    pub fn try_initialize(
        &self,
        map: &Map,
        imu_t_bc: Option<Pose3d>,
        imu_bias: ImuBias,
        start_idx: usize,
        prior_g: f64,
        prior_a: f64,
        already_initialized: bool,
    ) -> Option<ImuInitResult> {
        let imu_t_bc = imu_t_bc?;

        let mut keyframes: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        keyframes.sort_by_key(|kf| kf.frame.idx);
        let n = keyframes.len();
        if n < self.config.min_keyframes {
            return None;
        }

        let frame_to_local: HashMap<usize, usize> = keyframes
            .iter()
            .enumerate()
            .map(|(i, kf)| (kf.frame.idx, i))
            .collect();

        let is_mono = window_is_mono(&keyframes);

        let (velocities, rwg): (Vec<Vec3F64>, Mat3F64) = if already_initialized {
            let vels = keyframes.iter().map(|kf| kf.velocity_world).collect();
            // NOT Identity: ORB-SLAM3 can seed Identity here because its
            // ApplyScaledRotation rotates the map so gravity lands back on
            // its own internal reference gI=(0,0,-1) — so "already aligned"
            // means "already at gI". `apply_initialization` here instead
            // rotates the map to kornia-slam's own canonical (0,+G,0), a
            // fixed ~125° offset from gI. Seeding Identity would tell this
            // solve gravity is still at gI when it is actually at (0,+G,0),
            // which is not a small perturbation and would fight convergence.
            let gi_to_canonical =
                rotation_from_to(Vec3F64::new(0.0, 0.0, -1.0), Vec3F64::new(0.0, 1.0, 0.0))
                    .matrix();
            (vels, gi_to_canonical)
        } else {
            let t_cb = imu_t_bc.inverse();
            let r_cb = t_cb.rotation;
            let lever = t_cb.translation;

            let mut velocities: Vec<Vec3F64> = vec![Vec3F64::ZERO; n];
            let mut dir_g = Vec3F64::ZERO;

            for factor in map.imu_factors() {
                let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else {
                    continue;
                };
                let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else {
                    continue;
                };
                let dt = factor.preintegrated.dt;
                if dt <= 0.0 {
                    continue;
                }

                let cam_i = keyframes[i].frame.pose_world_to_cam.inverse();
                let cam_j = keyframes[j].frame.pose_world_to_cam.inverse();
                let r_wb_i = cam_i.rotation * r_cb;

                dir_g -= r_wb_i * factor.preintegrated.delta_velocity_with_bias(&imu_bias);

                let p_wb_i = cam_i.translation + cam_i.rotation * lever;
                let p_wb_j = cam_j.translation + cam_j.rotation * lever;
                let vel = (p_wb_j - p_wb_i) / dt;
                velocities[i] = vel;
                velocities[j] = vel;
            }

            let rwg = if dir_g.length() > 1e-9 {
                rotation_from_to(Vec3F64::new(0.0, 0.0, -1.0), dir_g.normalize()).matrix()
            } else {
                Mat3F64::IDENTITY
            };
            (velocities, rwg)
        };

        let (rwg_out, scale_out, bg_out, ba_out, velocities_out) = self.inertial_optimizer(
            map,
            &frame_to_local,
            keyframes,
            rwg,
            &velocities,
            1.0, // scale_init — ORB-SLAM3 always seeds mScale=1.0 before InertialOptimization
            imu_bias.gyro,
            imu_bias.accel,
            is_mono,
            prior_g,
            prior_a,
            imu_t_bc,
        )?;

        // ── Sanity check — mirrors ORB-SLAM3's *only* gate on this result,
        // `if (mScale<1e-1) { bInitializing=false; return; }` (LocalMapping.cc
        // ~1271). There is no bg/ba magnitude gate in ORB-SLAM3: it trusts the
        // prior-regularized joint solve and refines further on the next VIBA
        // pass. A hard |bg|>0.05 reject was added here previously and is what
        // broke real-data initialization — it rejected results ORB-SLAM3
        // would have accepted and simply refined at VIBA1/VIBA2.
        if !scale_out.is_finite() || scale_out < 0.1 {
            eprintln!("[imu_init] rejected: bad scale {:.4}", scale_out);
            return None;
        }
        if !bg_out.length().is_finite() || !ba_out.length().is_finite() {
            eprintln!("[imu_init] rejected: non-finite bias");
            return None;
        }
        // Reject a diverged refinement instead of replacing the last valid bias.
        const MAX_PLAUSIBLE_ACCEL_BIAS: f64 = 1.0; // m/s^2
        if ba_out.length() > MAX_PLAUSIBLE_ACCEL_BIAS {
            eprintln!(
                "[imu_init] rejected: implausible accel bias |ba|={:.3} > {MAX_PLAUSIBLE_ACCEL_BIAS} m/s^2 ({:.3},{:.3},{:.3})",
                ba_out.length(),
                ba_out.x,
                ba_out.y,
                ba_out.z,
            );
            return None;
        }
        // Reconstruct the PHYSICAL gravity vector using the *same* gI the factor
        // used internally — do not reuse rwg_out assuming any other convention.
        let gravity_world = rwg_out * Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);

        // NOTE: a hard gravity-misalignment gate was tried here and reverted.
        // VIBA0's own bootstrap (the `!already_initialized` branch above) is a
        // crude, unweighted-sum estimate that does NOT reliably converge with a
        // larger window — on V101 it got *worse* the longer `ready()` retried it
        // (23°  →  27° over the full sequence), because gating VIBA0 on
        // misalignment prevents `state.imu_initialized` from ever becoming true,
        // which starves VIBA1/VIBA2 (`refine_inertial_init`, gated on that same
        // flag) of the chance to run at all. VIBA1/VIBA2 are what actually fix
        // VIBA0's roughness — confirmed on the same V101 sequence pre-gate:
        // VIBA0 23.0° → VIBA1 1.13° → VIBA2 0.78°, all within the first 15s of
        // IMU time. Trust that chain; don't gate its entry point.
        eprintln!(
            "[imu_init] accepted  scale={:.4}  gravity=({:.3},{:.3},{:.3})  bg=({:.5},{:.5},{:.5})  ba=({:.6},{:.6},{:.6})",
            scale_out,
            gravity_world.x,
            gravity_world.y,
            gravity_world.z,
            bg_out.x,
            bg_out.y,
            bg_out.z,
            ba_out.x,
            ba_out.y,
            ba_out.z,
        );

        Some(ImuInitResult {
            scale: scale_out,
            gravity_world,
            velocities_world: velocities_out,
            bias: ImuBias {
                gyro: bg_out,
                accel: ba_out,
            },
        })
    }
}
