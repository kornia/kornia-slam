//! Card Q, W1, W3-W5: one local VI-BA window, shaped like the R_01 log, on a
//! head-mounted walk whose IMU stream is exact for the `Hold` integrator.
//! Each test runs 3 and 10 active keyframes, mono and with depth rows, so a
//! fault that only a long window shows is told from one in the solver.

use super::*;
use crate::mapping::bundle_adjustment::{LocalBaParams, STEREO_DEPTH_REL_SIGMA};
use crate::mapping::synthetic_walk::{BodyState, Gauss, HEAD_WALK_IMU_DT, HeadWalk};
use crate::pose_conversion::angle_deg;
use kornia_algebra::Mat3F64;
use kornia_sensors::imu::ImuCalib;

fn gravity() -> Vec3F64 {
    Vec3F64::new(0.0, 9.81, 0.0)
}
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

#[derive(Clone, Copy, Debug)]
struct Shape {
    active: usize,
    fixed: usize,
    /// Share of the window's landmarks some fixed keyframe observes.
    anchored: f64,
    depth: bool,
}

/// The log's two shapes: 3 active, 88 % anchored (base) and 10 active,
/// 28 % anchored (M1), each mono and with depth rows.
fn shapes() -> Vec<Shape> {
    let mut out = Vec::new();
    for (active, anchored) in [(3, 0.88), (10, 0.28)] {
        for depth in [false, true] {
            out.push(Shape {
                active,
                fixed: 12,
                anchored,
                depth,
            });
        }
    }
    out
}

struct Window {
    shape: Shape,
    truth: Vec<ViBaKeyframe>,
    points: Vec<Vec3F64>,
    /// The active keyframe each landmark was placed from.
    home: Vec<usize>,
    anchored: Vec<bool>,
    observations: Vec<BaObservation>,
    imu_edges: Vec<ImuFactor>,
    times: Vec<f64>,
    accel_rms: f64,
    params: ViBaParams,
}

/// The local-BA mapping of [`LocalBaParams`] defaults onto [`ViBaParams`],
/// as `run_local_inertial_ba_inner` does.
fn production_params() -> ViBaParams {
    let local = LocalBaParams::default();
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
        newest_marginal: true,
        ..ViBaParams::default()
    }
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

/// Observers of a landmark placed from active keyframe `home`: the active
/// keyframes around it (five, or all of a shorter window), and with
/// `anchored` three of those plus up to two fixed keyframes that see it.
fn observers(
    shape: Shape,
    home: usize,
    anchored: bool,
    sees: &dyn Fn(usize) -> bool,
    rng: &mut Gauss,
) -> Vec<usize> {
    let n = shape.fixed + shape.active;
    let span = if anchored { 3 } else { 5 }.min(shape.active);
    let lo = home.saturating_sub(span / 2).max(shape.fixed).min(n - span);
    let mut out: Vec<usize> = (lo..lo + span).filter(|&k| sees(k)).collect();
    if anchored {
        let mut fixed: Vec<usize> = (0..shape.fixed).filter(|&k| sees(k)).collect();
        for _ in 0..2 {
            if fixed.is_empty() {
                break;
            }
            let pick = ((rng.uniform() * fixed.len() as f64) as usize).min(fixed.len() - 1);
            out.push(fixed.swap_remove(pick));
        }
    }
    out.sort_unstable();
    out
}

fn build(shape: Shape, walk: &HeadWalk, pixel_sigma: f64, seed: u64) -> Window {
    let n = shape.fixed + shape.active;
    let step_of = |k: usize| START_STEPS + k * KF_STEPS;
    let (states, readings) = walk.simulate(step_of(n - 1) + 1, gravity());
    let stamp = |i: usize| i as f64 * HEAD_WALK_IMU_DT;
    let bias = ImuBias {
        gyro: walk.gyro_bias,
        accel: walk.accel_bias,
    };
    let truth: Vec<ViBaKeyframe> = (0..n)
        .map(|k| ViBaKeyframe {
            pose: pose_cw(&states[step_of(k)]),
            velocity: states[step_of(k)].velocity,
            bias,
            fixed: k < shape.fixed,
            inertial_fixed: false,
        })
        .collect();
    let imu_edges: Vec<ImuFactor> = (shape.fixed - 1..n - 1)
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
    let span = step_of(shape.fixed - 1)..step_of(n - 1);
    let accel_rms = (span
        .clone()
        .map(|i| {
            let a = (states[i + 1].velocity - states[i].velocity) * (1.0 / HEAD_WALK_IMU_DT);
            a.dot(a)
        })
        .sum::<f64>()
        / span.len() as f64)
        .sqrt();

    let cam = camera();
    let mut rng = Gauss(0x5eed_0000 ^ seed);
    let (mut points, mut home, mut anchored, mut observations) = (vec![], vec![], vec![], vec![]);
    for h in shape.fixed..n {
        for _ in 0..POINTS_PER_ACTIVE {
            let want_anchor = rng.uniform() < shape.anchored;
            let (x, obs) = (0..1000)
                .find_map(|_| {
                    let (u, v) = (30.0 + 580.0 * rng.uniform(), 30.0 + 420.0 * rng.uniform());
                    let z = 2.0 + 6.0 * rng.uniform();
                    let ray = Vec3F64::new((u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z);
                    let x = truth[h].pose.untransform_point(&ray);
                    let sees = |k: usize| project(&truth[k].pose, &x).is_some();
                    let obs = observers(shape, h, want_anchor, &sees, &mut rng);
                    let has_fixed = obs.iter().any(|&k| k < shape.fixed);
                    (obs.len() >= 2 && has_fixed == want_anchor).then_some((x, obs))
                })
                .expect("a placeable landmark");
            let point_idx = points.len();
            for &k in &obs {
                let Some((u, v, z)) = project(&truth[k].pose, &x) else {
                    continue;
                };
                let (du, dv) = (rng.normal() * pixel_sigma, rng.normal() * pixel_sigma);
                // Depth noise of 1 % alongside pixel noise, none without.
                let dz = if pixel_sigma > 0.0 {
                    0.01 * z * rng.normal()
                } else {
                    0.0
                };
                let (depth_meas, depth_sigma) = if shape.depth && z < 6.0 {
                    (Some((z + dz) as f32), STEREO_DEPTH_REL_SIGMA * z as f32)
                } else {
                    (None, 1.0)
                };
                observations.push(BaObservation {
                    pose_idx: k,
                    point_idx,
                    pixel: [(u + du) as f32, (v + dv) as f32],
                    fixed_pose: k < shape.fixed,
                    fixed_point: false,
                    depth_meas,
                    depth_sigma,
                });
            }
            points.push(x);
            home.push(h);
            anchored.push(want_anchor);
        }
    }
    Window {
        shape,
        truth,
        points,
        home,
        anchored,
        observations,
        imu_edges,
        times: (0..n).map(|k| stamp(step_of(k))).collect(),
        accel_rms,
        params: production_params(),
    }
}

/// The fixture is the log's window, and its IMU edges are exact at truth.
fn check_fixture(w: &Window) {
    let s = w.shape;
    let share = w.anchored.iter().filter(|&&a| a).count() as f64 / w.points.len() as f64;
    assert!(
        (share - s.anchored).abs() < 0.06,
        "{s:?}: anchored share {share:.3}"
    );
    let per_point = w.observations.len() as f64 / w.points.len() as f64;
    assert!(
        (3.5..=6.5).contains(&per_point),
        "{s:?}: {per_point:.2} observations per point"
    );
    let observing_fixed = (0..s.fixed)
        .filter(|&k| w.observations.iter().any(|o| o.pose_idx == k))
        .count();
    assert!(
        (9..=27).contains(&observing_fixed),
        "{s:?}: {observing_fixed} observing fixed keyframes"
    );
    assert!(
        w.accel_rms >= 0.5,
        "{s:?}: acceleration RMS {:.3} m/s^2",
        w.accel_rms
    );
    let t_bc = t_bc();
    for e in &w.imu_edges {
        let (res, _, _) = preintegration::residual_and_jacobians(
            w.truth[e.from_idx].nav_state(),
            w.truth[e.to_idx].nav_state(),
            &e.preintegrated,
            &gravity(),
            Some(&t_bc),
        );
        let omega = preintegration::information(&e.preintegrated, 1.0).omega;
        let nav = res.fixed_rows::<9>(0);
        let chi2 = nav.dot(&(omega.fixed_view::<9, 9>(0, 0) * nav));
        assert!(
            chi2 < 1e-6,
            "{s:?}: edge {}->{} chi2 {chi2:.3e} at truth",
            e.from_idx,
            e.to_idx
        );
    }
}

fn solve(w: &Window, kfs: &[ViBaKeyframe], points: &[Vec3F64]) -> ViBaResult {
    visual_inertial_bundle_adjust(
        kfs,
        points,
        &w.observations,
        &w.imu_edges,
        &[],
        &[],
        &camera(),
        &w.params,
    )
    .expect("the window solves")
}

/// Worst error of the active keyframes against truth.
#[derive(Debug, Default, Clone, Copy)]
struct Errors {
    centre_m: f64,
    rotation_deg: f64,
    velocity: f64,
    accel_bias: f64,
    gyro_bias: f64,
}

fn errors(w: &Window, result: &ViBaResult) -> Errors {
    let mut e = Errors::default();
    for (kf, truth) in result.keyframes.iter().zip(&w.truth).skip(w.shape.fixed) {
        e.centre_m = e
            .centre_m
            .max((centre(&kf.pose) - centre(&truth.pose)).length());
        e.rotation_deg = e.rotation_deg.max(angle_deg(&kf.pose, &truth.pose));
        e.velocity = e.velocity.max((kf.velocity - truth.velocity).length());
        e.accel_bias = e
            .accel_bias
            .max((kf.bias.accel - truth.bias.accel).length());
        e.gyro_bias = e.gyro_bias.max((kf.bias.gyro - truth.bias.gyro).length());
    }
    e
}

/// The newest camera centre's 1-sigma from the solver's own marginal:
/// under `T_cw ← T_cw · Exp([ρ; ω])` the centre moves by `−ρ + [c]× ω`.
fn newest_centre_sigma(result: &ViBaResult) -> f64 {
    let (Some(cov), Some(newest)) = (result.newest_marginal, result.keyframes.last()) else {
        return f64::NAN;
    };
    let c = centre(&newest.pose);
    let hat = SO3F64::hat(c);
    let mut j = SMatF64::<3, 6>::zeros();
    for r in 0..3 {
        j[(r, r)] = -1.0;
        for k in 0..3 {
            j[(r, 3 + k)] = hat.col(k)[r];
        }
    }
    let block = cov.fixed_view::<6, 6>(0, 0).into_owned();
    let sigma = j * block * j.transpose();
    (0..3).map(|d| sigma[(d, d)]).sum::<f64>().sqrt()
}

fn walk(origin: Vec3F64) -> HeadWalk {
    HeadWalk {
        origin,
        accel_bias: Vec3F64::ZERO,
        gyro_bias: Vec3F64::ZERO,
    }
}

/// Starting state: the active keyframes off truth (per-axis sigmas giving
/// 2 cm, 0.2° and 5 cm/s RMS) and every landmark 3 % off along its home ray.
fn noisy_start(w: &Window, seed: u64) -> (Vec<ViBaKeyframe>, Vec<Vec3F64>) {
    let mut rng = Gauss(0xa11c_e000 ^ seed);
    let axis = 3f64.sqrt();
    let kfs: Vec<ViBaKeyframe> = w
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

fn label(s: Shape) -> String {
    format!(
        "active={:<2} {}",
        s.active,
        if s.depth { "depth" } else { "mono " }
    )
}

/// W1: started at truth, the window stays there.
#[test]
fn w1_a_window_started_at_truth_stays() {
    let mut failed = Vec::new();
    for shape in shapes() {
        let w = build(shape, &walk(Vec3F64::ZERO), 0.0, 1);
        check_fixture(&w);
        let r = solve(&w, &w.truth, &w.points);
        let e = errors(&w, &r);
        println!("W1 {} iters={} {e:?}", label(shape), r.iterations);
        if e.centre_m >= 1e-4
            || e.rotation_deg >= 1e-4
            || e.velocity >= 1e-3
            || e.accel_bias.max(e.gyro_bias) >= 1e-4
        {
            failed.push(label(shape));
        }
    }
    assert!(failed.is_empty(), "W1 moved off truth: {failed:?}");
}

/// W3: from a noisy start the window converges; with 0.5 px pixel noise the
/// newest keyframe lands within 3 sigma of its own marginal.
#[test]
fn w3_a_noisy_start_converges() {
    let mut failed = Vec::new();
    for shape in shapes() {
        let w = build(shape, &walk(Vec3F64::ZERO), 0.0, 3);
        check_fixture(&w);
        let (kfs, points) = noisy_start(&w, 3);
        let r = solve(&w, &kfs, &points);
        let e = errors(&w, &r);
        println!(
            "W3 exact {} iters={} conv={} {e:?}",
            label(shape),
            r.iterations,
            r.converged
        );
        if e.centre_m >= 2e-3 || e.rotation_deg >= 0.01 || e.velocity >= 5e-3 {
            failed.push(format!("exact {}", label(shape)));
        }

        let w = build(shape, &walk(Vec3F64::ZERO), 0.5, 3);
        let (kfs, points) = noisy_start(&w, 3);
        let r = solve(&w, &kfs, &points);
        let newest = w.truth.len() - 1;
        let err = (centre(&r.keyframes[newest].pose) - centre(&w.truth[newest].pose)).length();
        let sigma = newest_centre_sigma(&r);
        println!(
            "W3 0.5px {} iters={} newest err={:.2} mm sigma={:.2} mm ratio={:.2}",
            label(shape),
            r.iterations,
            err * 1e3,
            sigma * 1e3,
            err / sigma
        );
        if err.is_nan() || sigma.is_nan() || err > 3.0 * sigma {
            failed.push(format!(
                "0.5px {}: {:.2} mm vs sigma {:.2} mm",
                label(shape),
                err * 1e3,
                sigma * 1e3
            ));
        }
    }
    assert!(failed.is_empty(), "W3: {failed:?}");
}

/// W4: started slid along the interior chain's null space (a velocity
/// offset u with positions moved by u (t - t_boundary), and 2° of yaw about
/// the boundary camera), with the free landmarks moved alongside their home
/// keyframe, the window returns to truth.
#[test]
fn w4_a_window_slid_along_its_null_space_returns() {
    let mut failed = Vec::new();
    let u = Vec3F64::new(0.5, 0.0, 0.2);
    let yaw = SO3F64::exp(gravity().normalize() * 2f64.to_radians()).matrix();
    for shape in shapes() {
        let w = build(shape, &walk(Vec3F64::ZERO), 0.0, 4);
        check_fixture(&w);
        let b = shape.fixed - 1;
        let (c_b, t_b) = (centre(&w.truth[b].pose), w.times[b]);
        let slide = |x: Vec3F64, t: f64| c_b + yaw * (x - c_b) + u * (t - t_b);
        let kfs: Vec<ViBaKeyframe> = w
            .truth
            .iter()
            .zip(&w.times)
            .map(|(kf, &t)| {
                if kf.fixed {
                    return kf.clone();
                }
                ViBaKeyframe {
                    pose: pose_at(
                        yaw * kf.pose.rotation.transpose(),
                        slide(centre(&kf.pose), t),
                    ),
                    velocity: yaw * kf.velocity + u,
                    ..kf.clone()
                }
            })
            .collect();
        let points: Vec<Vec3F64> = w
            .points
            .iter()
            .zip(&w.home)
            .zip(&w.anchored)
            .map(|((x, &h), &a)| if a { *x } else { slide(*x, w.times[h]) })
            .collect();
        let r = solve(&w, &kfs, &points);
        let e = errors(&w, &r);
        println!(
            "W4 {} iters={} conv={} {e:?}",
            label(shape),
            r.iterations,
            r.converged
        );
        if e.centre_m >= 5e-3 {
            failed.push(format!("{}: {:.2} mm", label(shape), e.centre_m * 1e3));
        }
    }
    assert!(failed.is_empty(), "W4 did not return: {failed:?}");
}

/// A change applied to each fixed keyframe.
type Warp<'a> = &'a dyn Fn(&ViBaKeyframe) -> ViBaKeyframe;

/// W5 (characterisation): the fixed context tilted 1° about the boundary
/// camera, or scaled 0.9 about it, against a window at truth.
#[test]
fn w5_a_window_against_a_tilted_or_scaled_fixed_context() {
    let tilt = SO3F64::exp(Vec3F64::new(1f64.to_radians(), 0.0, 0.0)).matrix();
    for shape in shapes() {
        let w = build(shape, &walk(Vec3F64::ZERO), 0.0, 5);
        check_fixture(&w);
        let c_b = centre(&w.truth[shape.fixed - 1].pose);
        let cases: [(&str, Warp<'_>); 2] = [
            ("tilt 1deg", &|kf| ViBaKeyframe {
                pose: pose_at(
                    tilt * kf.pose.rotation.transpose(),
                    c_b + tilt * (centre(&kf.pose) - c_b),
                ),
                velocity: tilt * kf.velocity,
                ..kf.clone()
            }),
            ("scale 0.9", &|kf| ViBaKeyframe {
                pose: pose_at(
                    kf.pose.rotation.transpose(),
                    c_b + (centre(&kf.pose) - c_b) * 0.9,
                ),
                velocity: kf.velocity * 0.9,
                ..kf.clone()
            }),
        ];
        for (name, warp) in cases {
            let kfs: Vec<ViBaKeyframe> = w
                .truth
                .iter()
                .map(|kf| if kf.fixed { warp(kf) } else { kf.clone() })
                .collect();
            let r = solve(&w, &kfs, &w.points);
            let newest = w.truth.len() - 1;
            let (before, after) = (&w.truth[newest].pose, &r.keyframes[newest].pose);
            let g = |p: &Pose3d| (p.rotation * gravity()).normalize();
            let tilt_deg = g(before).dot(g(after)).clamp(-1.0, 1.0).acos().to_degrees();
            println!(
                "W5 {} {name}: newest shift={:.2} cm tilt={tilt_deg:.3} deg iters={} cost {:.3e} -> {:.3e} {:?}",
                label(shape),
                (centre(after) - centre(before)).length() * 1e2,
                r.iterations,
                r.initial_cost,
                r.final_cost,
                errors(&w, &r),
            );
        }
    }
}
