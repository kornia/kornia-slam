//! R-A11: one local VI-BA window solved with the `pr180-window10` budget
//! (10 active keyframes, 7 iterations from λ 1e-4, the sequential dense
//! solve), on the head-mounted walk, mono and with depth rows, and once with
//! a second rig camera whose observations go through `rig_views`.

use super::*;
use crate::mapping::bundle_adjustment::{LargeWindow, LocalBaParams, STEREO_DEPTH_REL_SIGMA};
use crate::mapping::synthetic_walk::{BodyState, Gauss, HEAD_WALK_IMU_DT, HeadWalk};
use crate::pose_conversion::angle_deg;
use kornia_algebra::Mat3F64;
use kornia_sensors::imu::ImuCalib;

const FIXED: usize = 6;
const ACTIVE: usize = 10;
/// 0.24 s between keyframes, in IMU samples.
const KF_STEPS: usize = 240;
/// The window starts 1 s into the walk, mid-stride.
const START_STEPS: usize = 1000;
const POINTS_PER_ACTIVE: usize = 60;
const WIDTH: f64 = 640.0;
const HEIGHT: f64 = 480.0;
/// LaMAria's IMU noise densities and bias walks.
const CALIB: ImuCalib = ImuCalib {
    gyro_noise: 1.7e-4,
    accel_noise: 7.8e-4,
    gyro_bias_noise: 1.9e-5,
    accel_bias_noise: 3.0e-3,
};

fn gravity() -> Vec3F64 {
    Vec3F64::new(0.0, 9.81, 0.0)
}

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 400.0,
        fy: 400.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// `X_body = T_bc · X_cam`, off the identity in rotation and lever arm.
fn t_bc() -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.02, -0.03, 0.01)).matrix(),
        Vec3F64::new(0.04, -0.02, 0.01),
    )
}

/// The second rig camera, `T_CP`: 40° to the side and 10 cm along x, as a
/// diverging head-mounted pair.
fn side_camera() -> RigObservationCamera {
    RigObservationCamera {
        camera: camera(),
        camera_from_primary: Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 40f64.to_radians(), 0.0)).matrix(),
            Vec3F64::new(-0.1, 0.0, 0.0),
        ),
    }
}

/// The `pr180-window10` local-BA settings, through `large()` as a
/// 0-inlier-gated job takes them, mapped onto [`ViBaParams`] as
/// `run_local_inertial_ba_inner` does.
fn preset_params() -> ViBaParams {
    let local = LocalBaParams {
        active_keyframes: ACTIVE,
        dense_parallelism: DenseParallelism::Sequential,
        large_window: LargeWindow::Enabled {
            min_tracked_inliers: 0,
            active_keyframes: ACTIVE,
            max_iterations: 7,
            initial_lambda: 1e-4,
        },
        ..LocalBaParams::default()
    }
    .large();
    assert_eq!((local.max_iterations, local.initial_lambda), (7, 1e-4));
    ViBaParams {
        imu_t_bc: Some(t_bc()),
        gravity: gravity(),
        max_iterations: local.max_iterations,
        initial_lambda: local.initial_lambda,
        boundary_imu_info_scale: local.boundary_imu_info_scale,
        boundary_scales_bias_rows: local.boundary_scales_bias_rows,
        boundary: local.boundary,
        imu_huber: local.imu_huber,
        imu_recently_initialized: local.recently_initialized,
        newest_marginal: local.newest_marginal,
        dense_parallelism: local.dense_parallelism,
        ..ViBaParams::default()
    }
}

fn pose_cw(state: &BodyState) -> Pose3d {
    t_bc()
        .inverse()
        .compose(&Pose3d::new(state.r_wb, state.position).inverse())
}

/// `T_cw` of a camera at `centre` with world-from-camera rotation `r_wc`.
fn pose_at(r_wc: Mat3F64, centre: Vec3F64) -> Pose3d {
    let r_cw = r_wc.transpose();
    Pose3d::new(r_cw, -(r_cw * centre))
}

fn centre(pose: &Pose3d) -> Vec3F64 {
    pose.inverse().translation
}

fn project(pose: &Pose3d, x: &Vec3F64) -> Option<(f64, f64, f64)> {
    let cam = camera();
    let c = pose.transform_point(x);
    if c.z < 0.3 {
        return None;
    }
    let (u, v) = (cam.fx * c.x / c.z + cam.cx, cam.fy * c.y / c.z + cam.cy);
    ((0.0..WIDTH).contains(&u) && (0.0..HEIGHT).contains(&v)).then_some((u, v, c.z))
}

struct Window {
    truth: Vec<ViBaKeyframe>,
    points: Vec<Vec3F64>,
    /// The active keyframe each landmark was placed from.
    home: Vec<usize>,
    observations: Vec<BaObservation>,
    imu_edges: Vec<ImuFactor>,
    params: ViBaParams,
}

/// `FIXED` fixed and `ACTIVE` active keyframes of the walk, exact pixels.
/// Each landmark is placed from a random camera of its home keyframe and
/// observed by every camera of the keyframes within two of it that sees it.
/// With `rig`, the side camera places and observes landmarks too.
fn build(depth: bool, rig: bool, seed: u64) -> Window {
    let n = FIXED + ACTIVE;
    let step_of = |k: usize| START_STEPS + k * KF_STEPS;
    let walk = HeadWalk {
        origin: Vec3F64::ZERO,
        accel_bias: Vec3F64::ZERO,
        gyro_bias: Vec3F64::ZERO,
    };
    let (states, readings) = walk.simulate(step_of(n - 1) + 1, gravity());
    let stamp = |i: usize| i as f64 * HEAD_WALK_IMU_DT;
    let bias = ImuBias::default();
    let truth: Vec<ViBaKeyframe> = (0..n)
        .map(|k| ViBaKeyframe {
            pose: pose_cw(&states[step_of(k)]),
            velocity: states[step_of(k)].velocity,
            bias,
            fixed: k < FIXED,
            inertial_fixed: false,
        })
        .collect();
    let imu_edges: Vec<ImuFactor> = (FIXED - 1..n - 1)
        .map(|k| {
            let (s0, s1) = (step_of(k), step_of(k + 1));
            ImuFactor::new(
                k,
                k + 1,
                PreintegratedImu::from_measurements(
                    bias,
                    CALIB,
                    &readings[s0..s1],
                    stamp(s0),
                    stamp(s1),
                ),
            )
        })
        .collect();

    let mut params = preset_params();
    let cameras = if rig { 2 } else { 1 };
    if rig {
        params.rig_views = vec![side_camera()];
    }
    let view_pose = |c: usize, pose: &Pose3d| match c {
        0 => *pose,
        _ => side_camera().camera_from_primary.compose(pose),
    };
    let cam = camera();
    let mut rng = Gauss(0x9e5e_7000 ^ seed);
    let (mut points, mut home, mut observations, mut views) = (vec![], vec![], vec![], vec![]);
    for h in FIXED..n {
        for _ in 0..POINTS_PER_ACTIVE {
            let c_home = ((rng.uniform() * cameras as f64) as usize).min(cameras - 1);
            let u = 30.0 + 580.0 * rng.uniform();
            let v = 30.0 + 420.0 * rng.uniform();
            let z = 2.0 + 6.0 * rng.uniform();
            let ray = Vec3F64::new((u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z);
            let x = view_pose(c_home, &truth[h].pose).untransform_point(&ray);
            let mut seen = Vec::new();
            let near = h.saturating_sub(2)..(h + 3).min(n);
            for (k, kf) in truth.iter().enumerate().take(near.end).skip(near.start) {
                for c in 0..cameras {
                    if let Some(p) = project(&view_pose(c, &kf.pose), &x) {
                        seen.push((k, c, p));
                    }
                }
            }
            if seen.len() < 2 {
                continue;
            }
            let point_idx = points.len();
            for (k, c, (u, v, z)) in seen {
                let (depth_meas, depth_sigma) = if depth && z < 6.0 {
                    (Some(z as f32), STEREO_DEPTH_REL_SIGMA * z as f32)
                } else {
                    (None, 1.0)
                };
                observations.push(BaObservation {
                    pose_idx: k,
                    point_idx,
                    pixel: [u as f32, v as f32],
                    fixed_pose: k < FIXED,
                    fixed_point: false,
                    depth_meas,
                    depth_sigma,
                });
                views.push(c as u8);
            }
            points.push(x);
            home.push(h);
        }
    }
    if rig {
        params.observation_views = views;
    }
    Window {
        truth,
        points,
        home,
        observations,
        imu_edges,
        params,
    }
}

/// The active keyframes off truth (per-axis sigmas giving 2 cm, 0.2° and
/// 5 cm/s RMS) and every landmark 3 % off along its home ray.
fn noisy_start(w: &Window, seed: u64) -> (Vec<ViBaKeyframe>, Vec<Vec3F64>) {
    let mut rng = Gauss(0xa11c_e000 ^ seed);
    let axis = 3f64.sqrt();
    let kfs = w
        .truth
        .iter()
        .map(|kf| {
            if kf.fixed {
                return kf.clone();
            }
            let r_wc = SO3F64::exp(rng.vec3(0.2f64.to_radians() / axis)).matrix()
                * kf.pose.rotation.transpose();
            ViBaKeyframe {
                pose: pose_at(r_wc, centre(&kf.pose) + rng.vec3(0.02 / axis)),
                velocity: kf.velocity + rng.vec3(0.05 / axis),
                ..kf.clone()
            }
        })
        .collect();
    let points = w
        .points
        .iter()
        .zip(&w.home)
        .map(|(x, &h)| {
            let c = centre(&w.truth[h].pose);
            c + (*x - c) * (1.0 + 0.03 * rng.normal())
        })
        .collect();
    (kfs, points)
}

/// Worst centre (m), rotation (°) and velocity (m/s) error of the active
/// keyframes against truth.
fn errors(w: &Window, result: &ViBaResult) -> (f64, f64, f64) {
    result.keyframes.iter().zip(&w.truth).skip(FIXED).fold(
        (0.0, 0.0, 0.0),
        |(c, r, v), (kf, truth)| {
            (
                c.max((centre(&kf.pose) - centre(&truth.pose)).length()),
                r.max(angle_deg(&kf.pose, &truth.pose)),
                v.max((kf.velocity - truth.velocity).length()),
            )
        },
    )
}

fn solve_noisy(w: &Window, seed: u64) -> ViBaResult {
    let (kfs, points) = noisy_start(w, seed);
    visual_inertial_bundle_adjust(
        &kfs,
        &points,
        &w.observations,
        &w.imu_edges,
        &[],
        &[],
        &camera(),
        &w.params,
    )
    .expect("the window solves")
}

fn assert_converged(label: &str, w: &Window, r: &ViBaResult) {
    let (centre_m, rotation_deg, velocity) = errors(w, r);
    println!(
        "{label}: iters={} rejects={} conv={} centre={:.3} mm rot={rotation_deg:.4} deg vel={:.3} mm/s",
        r.iterations,
        r.rejected_steps,
        r.converged,
        centre_m * 1e3,
        velocity * 1e3,
    );
    assert!(r.iterations <= 7, "{label}: {} iterations", r.iterations);
    assert!(r.rejected_steps <= r.iterations, "{label}");
    assert!(r.final_cost < r.initial_cost, "{label}: cost did not fall");
    assert!(
        centre_m < 2e-3 && rotation_deg < 0.01 && velocity < 5e-3,
        "{label}: centre {:.3} mm, rotation {rotation_deg:.4} deg, velocity {:.3} mm/s",
        centre_m * 1e3,
        velocity * 1e3
    );
}

/// From a noisy start the 7-iteration budget at λ 1e-4 brings the window
/// back to truth, as W3 does with 50 iterations at λ 1.
#[test]
fn the_preset_budget_solves_the_window_walk() {
    for depth in [false, true] {
        let w = build(depth, false, 1);
        let r = solve_noisy(&w, 1);
        assert_converged(if depth { "depth" } else { "mono" }, &w, &r);
    }
}

/// The same window with a diverging second camera: its observations enter
/// through `rig_views` and the solve still returns to truth.
#[test]
fn a_rig_window_solves_end_to_end() {
    let w = build(false, true, 2);
    let side = w
        .params
        .observation_views
        .iter()
        .filter(|&&c| c == 1)
        .count();
    assert!(
        side * 4 >= w.observations.len(),
        "{side} of {} observations from the side camera",
        w.observations.len()
    );
    let r = solve_noisy(&w, 2);
    assert_converged("rig", &w, &r);
}
