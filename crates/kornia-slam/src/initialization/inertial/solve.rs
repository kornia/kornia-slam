//! The single joint LM solve: seeding, problem construction, optimization and
//! the acceptance gates on the result.

use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::optim::{LevenbergMarquardt, Problem, Variable, VariableType};
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

use super::factor::{InertialInitFactor, KfConst, WeightedZeroPrior};
use super::{
    BiasPriors, ImuInitRejectReason, ImuInitResult, ImuInitializer, InertialInitRequest,
    KeyframeVelocity, RwgSeed, window_is_mono,
};
use crate::map::{Keyframe, Map};
use crate::pose_conversion::rotation_from_to;

struct InertialSolution {
    rwg: Mat3F64,
    scale: f64,
    gyro_bias: Vec3F64,
    accel_bias: Vec3F64,
    velocities: Vec<Vec3F64>,
}

/// Initial guess handed to the optimizer.
struct InertialSeed {
    rwg: Mat3F64,
    velocities: Vec<Vec3F64>,
}

/// Everything the problem builder needs besides the map itself.
struct OptimizerSetup<'a, 'k> {
    frame_to_local: &'a HashMap<usize, usize>,
    keyframes: &'a [&'k Keyframe],
    seed: &'a InertialSeed,
    bias: ImuBias,
    priors: BiasPriors,
    is_mono: bool,
    imu_t_bc: Pose3d,
}

/// Pack a `Vec3F64` into the `Vec<f32>` value layout `kornia_algebra::optim`
/// variables use.
fn vec3_to_f32(v: Vec3F64) -> Vec<f32> {
    vec![v.x as f32, v.y as f32, v.z as f32]
}

/// Read the first three components of an optimizer variable back into a `Vec3F64`.
fn vec3_from_var(values: &[f32]) -> Vec3F64 {
    Vec3F64::new(values[0] as f64, values[1] as f64, values[2] as f64)
}

/// Wrap any optimizer error into the typed rejection the caller logs.
fn optimizer_failed(error: impl std::fmt::Display) -> ImuInitRejectReason {
    ImuInitRejectReason::OptimizerFailed(error.to_string())
}

/// Refinement seed: the map is already gravity-aligned and metric, so take the
/// velocities the keyframes already carry and derive Rwg from the current
/// gravity estimate.
fn seed_from_current_gravity(keyframes: &[&Keyframe], gravity_world: Vec3F64) -> InertialSeed {
    let velocities = keyframes.iter().map(|kf| kf.velocity_world).collect();
    // NOT Identity: ORB-SLAM3 can seed Identity here because its
    // ApplyScaledRotation rotates the map so gravity lands back on
    // its own internal reference gI=(0,0,-1) — so "already aligned"
    // means "already at gI". The system application here instead
    // rotates the map to kornia-slam's own canonical (0,+G,0), a
    // fixed ~125° offset from gI. Seeding Identity would tell this
    // solve gravity is still at gI when it is actually at (0,+G,0),
    // which is not a small perturbation and would fight convergence.
    let gi_to_current =
        rotation_from_to(Vec3F64::new(0.0, 0.0, -1.0), gravity_world.normalize()).matrix();
    InertialSeed {
        rwg: gi_to_current,
        velocities,
    }
}

/// First (VIBA0) seed: finite-difference velocities from the visual trajectory
/// plus a gravity direction accumulated from the preintegrated velocity deltas.
fn seed_from_visual_trajectory(
    map: &Map,
    frame_to_local: &HashMap<usize, usize>,
    keyframes: &[&Keyframe],
    imu_t_bc: Pose3d,
    imu_bias: ImuBias,
) -> InertialSeed {
    let n = keyframes.len();
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
    InertialSeed { rwg, velocities }
}

/// Builds the joint velocity/bias/gravity/scale problem over the window.
fn build_problem(map: &Map, setup: OptimizerSetup<'_, '_>) -> Result<Problem, ImuInitRejectReason> {
    let OptimizerSetup {
        frame_to_local,
        keyframes,
        seed,
        bias,
        priors,
        is_mono,
        imu_t_bc,
    } = setup;
    let (bg, ba) = (bias.gyro, bias.accel);
    let (prior_g, prior_a) = (priors.gyro, priors.accel);
    let n = keyframes.len();
    let mut problem = Problem::new();

    for (local_idx, vel) in seed.velocities.iter().enumerate().take(n) {
        let name = format!("v{}", local_idx);
        problem
            .add_variable(Variable::euclidean(&name, 3), vec3_to_f32(*vel))
            .map_err(optimizer_failed)?;
    }
    problem
        .add_variable(Variable::euclidean("bg", 3), vec3_to_f32(bg))
        .map_err(optimizer_failed)?;
    problem
        .add_variable(Variable::euclidean("ba", 3), vec3_to_f32(ba))
        .map_err(optimizer_failed)?;
    let q = SO3F64::from_matrix(&seed.rwg).to_array();
    let g_f32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
    problem
        .add_variable(
            Variable::new("gdir", VariableType::SO3, g_f32.clone()),
            g_f32,
        )
        .map_err(optimizer_failed)?;
    // scale_init — ORB-SLAM3 always seeds mScale=1.0 before InertialOptimization
    let scale_init = 1.0_f64;
    problem
        .add_variable(Variable::euclidean("scale", 1), vec![scale_init as f32])
        .map_err(optimizer_failed)?;
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
            .map_err(optimizer_failed)?;
        edge_count += 1;
    }

    if edge_count == 0 {
        return Err(ImuInitRejectReason::NoUsableFactors);
    }

    problem
        .add_factor(
            Box::new(WeightedZeroPrior {
                sqrt_weight: prior_g.sqrt(),
            }),
            vec!["bg".to_string()],
        )
        .map_err(optimizer_failed)?;
    problem
        .add_factor(
            Box::new(WeightedZeroPrior {
                sqrt_weight: prior_a.sqrt(),
            }),
            vec!["ba".to_string()],
        )
        .map_err(optimizer_failed)?;

    Ok(problem)
}

/// Runs LM on the built problem and reads the variables back.
fn optimize(
    problem: &mut Problem,
    priors: BiasPriors,
    n: usize,
    is_mono: bool,
) -> Result<InertialSolution, ImuInitRejectReason> {
    let prior_g = priors.gyro;
    let lm = LevenbergMarquardt {
        max_iterations: 200,
        lambda_init: if prior_g != 0.0 { 1e3 } else { 1e-3 },
        ..Default::default()
    };

    lm.optimize(problem).map_err(optimizer_failed)?;
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

    Ok(InertialSolution {
        rwg: rwg_out,
        scale: scale_out,
        gyro_bias: bg_out,
        accel_bias: ba_out,
        velocities: velocities_out,
    })
}

// ── Sanity check — mirrors ORB-SLAM3's *only* gate on this result,
// `if (mScale<1e-1) { bInitializing=false; return; }` (LocalMapping.cc
// ~1271). There is no bg/ba magnitude gate in ORB-SLAM3: it trusts the
// prior-regularized joint solve and refines further on the next VIBA
// pass. A hard |bg|>0.05 reject was added here previously and is what
// broke real-data initialization — it rejected results ORB-SLAM3
// would have accepted and simply refined at VIBA1/VIBA2.
fn check_solution(
    solution: &InertialSolution,
    max_accel_bias: f64,
) -> Result<(), ImuInitRejectReason> {
    if !solution.scale.is_finite() || solution.scale < 0.1 {
        return Err(ImuInitRejectReason::InvalidScale(solution.scale));
    }
    if !solution.gyro_bias.length().is_finite() || !solution.accel_bias.length().is_finite() {
        return Err(ImuInitRejectReason::NonFiniteBias);
    }
    // Reject a diverged refinement instead of replacing the last valid bias.
    if solution.accel_bias.length() > max_accel_bias {
        return Err(ImuInitRejectReason::ImplausibleAccelBias {
            actual: solution.accel_bias.length(),
            maximum: max_accel_bias,
        });
    }
    Ok(())
}

impl ImuInitializer {
    /// Mirrors ORB-SLAM3's `LocalMapping::InitializeIMU(priorG, priorA, bFIBA)`:
    /// a *single* joint LM solve per call. The system is responsible for the
    /// progressive VIBA0 (immediate) / VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s)
    /// re-triggering schedule with progressively relaxed priors — this
    /// function does not chain multiple passes internally.
    ///
    /// [`RwgSeed`] selects the same branch ORB-SLAM3 does at
    /// LocalMapping.cc:1226 (`!isImuInitialized()`): on the first (VIBA0)
    /// call, Rwg and per-keyframe velocities are derived from scratch via the
    /// visual trajectory (finite-difference velocity + gravity-direction
    /// accumulation); on later refinement calls the map is already
    /// gravity-aligned and metric (VIBA0's result was applied), so Rwg seeds
    /// from the current gravity estimate and velocities seed from each
    /// keyframe's current IMU-propagated estimate instead of being re-derived
    /// visually.
    pub fn try_initialize(
        &self,
        map: &Map,
        request: &InertialInitRequest,
    ) -> Result<ImuInitResult, ImuInitRejectReason> {
        self.solve(map, request, None)
    }

    /// Shared body of [`Self::try_initialize`]. `prior_override` exists only so
    /// tests can sweep the bias priors; production always resolves them from
    /// the request's stage through [`BiasPriors::for_stage`].
    pub(super) fn solve(
        &self,
        map: &Map,
        request: &InertialInitRequest,
        prior_override: Option<BiasPriors>,
    ) -> Result<ImuInitResult, ImuInitRejectReason> {
        self.config.validate()?;
        let imu_t_bc = request.imu_t_bc;
        let imu_bias = request.bias;
        let start_idx = request.start_kf_idx;

        let mut keyframes: Vec<&Keyframe> = map.keyframes_from(start_idx).collect();
        keyframes.sort_by_key(|kf| kf.frame.idx);
        let n = keyframes.len();
        if n < self.config.min_keyframes {
            return Err(ImuInitRejectReason::InsufficientKeyframes {
                found: n,
                required: self.config.min_keyframes,
            });
        }

        let frame_to_local: HashMap<usize, usize> = keyframes
            .iter()
            .enumerate()
            .map(|(i, kf)| (kf.frame.idx, i))
            .collect();

        let is_mono = window_is_mono(&keyframes);
        let priors =
            prior_override.unwrap_or_else(|| BiasPriors::for_stage(request.stage, is_mono));

        let seed = match request.seed {
            RwgSeed::FromCurrentGravity(gravity_world) => {
                seed_from_current_gravity(&keyframes, gravity_world)
            }
            RwgSeed::FromVisualTrajectory => {
                seed_from_visual_trajectory(map, &frame_to_local, &keyframes, imu_t_bc, imu_bias)
            }
        };

        let mut problem = build_problem(
            map,
            OptimizerSetup {
                frame_to_local: &frame_to_local,
                keyframes: &keyframes,
                seed: &seed,
                bias: imu_bias,
                priors,
                is_mono,
                imu_t_bc,
            },
        )?;
        let solution = optimize(&mut problem, priors, n, is_mono)?;

        check_solution(&solution, self.config.max_accel_bias)?;

        // Reconstruct the PHYSICAL gravity vector using the *same* gI the factor
        // used internally — do not reuse rwg_out assuming any other convention.
        let gravity_world = solution.rwg * Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);

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
        let keyframe_velocities = keyframes
            .iter()
            .zip(solution.velocities)
            .map(|(keyframe, velocity_world)| KeyframeVelocity {
                keyframe_idx: keyframe.frame.idx,
                velocity_world,
            })
            .collect();

        Ok(ImuInitResult {
            scale: solution.scale,
            gravity_world,
            keyframe_velocities,
            bias: ImuBias {
                gyro: solution.gyro_bias,
                accel: solution.accel_bias,
            },
        })
    }
}
