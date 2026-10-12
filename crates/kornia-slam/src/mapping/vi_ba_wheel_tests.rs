use super::*;
use crate::mapping::wheel_odom::WheelOdomMeasurement;

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 200.0,
        fy: 200.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// Optical camera (z forward, y down) 0.1 m ahead of and 0.6 m above a
/// REP-103 base, pitched down by `pitch` radians.
fn t_cb(pitch: f64) -> Pose3d {
    let r_cb = Mat3F64::from_cols_array(&[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
    let tilt = SO3F64::exp(Vec3F64::new(pitch, 0.0, 0.0)).matrix();
    let r = tilt * r_cb;
    Pose3d::new(r, -(r * Vec3F64::new(0.1, 0.0, 0.6)))
}

fn base(x: f64, y: f64, yaw: f64) -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.0, 0.0, yaw)).matrix(),
        Vec3F64::new(x, y, 0.0),
    )
}

fn camera_pose(t_wb: &Pose3d, t_cb: &Pose3d) -> Pose3d {
    t_cb.compose(&t_wb.inverse())
}

fn base_pose(pose_cw: &Pose3d, t_cb: &Pose3d) -> Pose3d {
    pose_cw.inverse().compose(t_cb)
}

fn keyframe(pose: Pose3d, fixed: bool) -> ViBaKeyframe {
    ViBaKeyframe {
        pose,
        velocity: Vec3F64::ZERO,
        bias: ImuBias::default(),
        fixed,
    }
}

fn planar_delta(t_wb_i: &Pose3d, t_wb_j: &Pose3d) -> [f64; 3] {
    let d = t_wb_i.inverse().compose(t_wb_j);
    let yaw = d.rotation.x_axis.y.atan2(d.rotation.x_axis.x);
    [d.translation.x, d.translation.y, yaw]
}

fn wheel_edge(i: usize, j: usize, delta: [f64; 3], sigma: f64, t_cb: Pose3d) -> WheelOdomFactor {
    let var = sigma * sigma;
    WheelOdomFactor {
        from_idx: i,
        to_idx: j,
        measurement: WheelOdomMeasurement::new(
            delta,
            [[var, 0.0, 0.0], [0.0, var, 0.0], [0.0, 0.0, var]],
        ),
        t_cb_from: t_cb,
        t_cb_to: t_cb,
        huber_chi2: None,
    }
}

fn solve(
    kfs: &[ViBaKeyframe],
    points: &[Vec3F64],
    obs: &[BaObservation],
    wheel: &[WheelOdomFactor],
) -> ViBaResult {
    visual_inertial_bundle_adjust(
        kfs,
        points,
        obs,
        &[],
        wheel,
        &camera(),
        &ViBaParams {
            max_iterations: 50,
            ..ViBaParams::default()
        },
    )
    .unwrap()
}

fn assert_pose_near(a: &Pose3d, b: &Pose3d, tol: f64) {
    let d = a.inverse().compose(b);
    let rot = SO3F64::from_matrix(&d.rotation).log().length();
    assert!(
        d.translation.length() < tol && rot < tol,
        "pose off by {:.2e} m / {rot:.2e} rad",
        d.translation.length()
    );
}

#[test]
fn single_wheel_edge_recovers_the_delta_and_respects_the_fixed_end() {
    // A tilt that differs between the two keyframes, as a moving head gives.
    let (cb_i, cb_j) = (t_cb(0.4), t_cb(0.1));
    let (wb_i, wb_j) = (base(1.0, -0.5, 0.3), base(1.4, -0.2, 0.7));
    let mut edge = wheel_edge(0, 1, planar_delta(&wb_i, &wb_j), 0.01, cb_i);
    edge.t_cb_to = cb_j;
    let truth = [camera_pose(&wb_i, &cb_i), camera_pose(&wb_j, &cb_j)];
    // 0.3 m, a lift off the floor and 0.25 rad of yaw and roll away.
    let wrong_base = |wb: &Pose3d| {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.0, 0.25)).matrix() * wb.rotation,
            wb.translation + Vec3F64::new(0.3, -0.1, 0.2),
        )
    };

    for fixed in [0, 1] {
        let kfs: Vec<_> = (0..2)
            .map(|k| {
                let pose = if k == fixed {
                    truth[k]
                } else {
                    let wb = if k == 0 { &wb_i } else { &wb_j };
                    camera_pose(&wrong_base(wb), if k == 0 { &cb_i } else { &cb_j })
                };
                keyframe(pose, k == fixed)
            })
            .collect();
        let out = solve(&kfs, &[], &[], &[edge]);
        assert_eq!(out.keyframes[fixed].pose, truth[fixed]);
        let free = 1 - fixed;
        // f32 retraction bounds the precision.
        assert_pose_near(&out.keyframes[free].pose, &truth[free], 1e-4);
        let solved_delta = planar_delta(
            &base_pose(&out.keyframes[0].pose, &cb_i),
            &base_pose(&out.keyframes[1].pose, &cb_j),
        );
        for (got, want) in solved_delta.iter().zip(edge.measurement.delta) {
            assert!((got - want).abs() < 1e-4, "{solved_delta:?}");
        }
    }
}

#[test]
fn wheel_edge_errors_are_reported() {
    let kfs = [
        keyframe(Pose3d::IDENTITY, true),
        keyframe(Pose3d::IDENTITY, false),
    ];
    let run = |edge: WheelOdomFactor| {
        visual_inertial_bundle_adjust(
            &kfs,
            &[],
            &[],
            &[],
            &[edge],
            &camera(),
            &ViBaParams::default(),
        )
    };
    let edge = wheel_edge(0, 1, [0.1, 0.0, 0.0], 0.01, t_cb(0.0));
    assert!(matches!(
        run(WheelOdomFactor { to_idx: 2, ..edge }),
        Err(ViBaError::WheelFactorOutOfRange(0, 2))
    ));
    let mut singular = edge;
    singular.measurement.covariance[1][1] = 0.0;
    assert!(matches!(
        run(singular),
        Err(ViBaError::InvalidWheelCovariance(0))
    ));
    // A NaN delta or extrinsic is named, not left to stall LM on a NaN cost.
    let mut nan_delta = edge;
    nan_delta.measurement.delta[2] = f64::NAN;
    assert!(matches!(
        run(nan_delta),
        Err(ViBaError::NonFiniteWheelFactor(0))
    ));
    let mut nan_extrinsic = edge;
    nan_extrinsic.t_cb_to.translation.x = f64::INFINITY;
    assert!(matches!(
        run(nan_extrinsic),
        Err(ViBaError::NonFiniteWheelFactor(0))
    ));
}

/// Deterministic N(0, 1) samples.
struct Gauss(u64);
impl Gauss {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn vec3(&mut self, sigma: f64) -> Vec3F64 {
        Vec3F64::new(self.next(), self.next(), self.next()) * sigma
    }
}

struct Scene {
    truth: Vec<Pose3d>,
    initial: Vec<ViBaKeyframe>,
    points: Vec<Vec3F64>,
    observations: Vec<BaObservation>,
    wheel: Vec<WheelOdomFactor>,
}

/// A base driving a 6-keyframe left arc past a sparse wall of landmarks,
/// seen with 3 px and 3 % depth noise; wheels good to 5 mm / 5 mrad per step.
fn planar_scene(seed: u64) -> Scene {
    let mut rng = Gauss(seed);
    let cb = t_cb(0.15);
    let mut bases = vec![base(0.0, 0.0, 0.0)];
    for k in 1..6 {
        let prev = bases[k - 1];
        let yaw = 0.1 * k as f64;
        let step = prev.rotation * Vec3F64::new(0.3, 0.0, 0.0);
        bases.push(base(
            prev.translation.x + step.x,
            prev.translation.y + step.y,
            yaw,
        ));
    }
    let truth: Vec<Pose3d> = bases.iter().map(|wb| camera_pose(wb, &cb)).collect();

    let truth_points: Vec<Vec3F64> = (0..14)
        .map(|_| {
            Vec3F64::new(
                5.0 + 1.5 * rng.next(),
                1.0 + 1.5 * rng.next(),
                0.6 + 0.4 * rng.next(),
            )
        })
        .collect();
    let mut observations = Vec::new();
    for (pose_idx, pose) in truth.iter().enumerate() {
        for (point_idx, p) in truth_points.iter().enumerate() {
            let pc = pose.transform_point(p);
            let (u, v) = (200.0 * pc.x / pc.z + 320.0, 200.0 * pc.y / pc.z + 240.0);
            if pc.z < 0.5 || !(0.0..640.0).contains(&u) || !(0.0..480.0).contains(&v) {
                continue;
            }
            observations.push(BaObservation {
                pose_idx,
                point_idx,
                pixel: [(u + 3.0 * rng.next()) as f32, (v + 3.0 * rng.next()) as f32],
                fixed_pose: pose_idx == 0,
                fixed_point: false,
                depth_meas: Some((pc.z * (1.0 + 0.03 * rng.next())) as f32),
                depth_sigma: (0.03 * pc.z) as f32,
            });
        }
    }

    let initial = truth
        .iter()
        .enumerate()
        .map(|(k, pose)| {
            if k == 0 {
                return keyframe(*pose, true);
            }
            let noisy = Pose3d::new(
                pose.rotation * SO3F64::exp(rng.vec3(0.02)).matrix(),
                pose.translation + rng.vec3(0.05),
            );
            keyframe(noisy, false)
        })
        .collect();
    let points = truth_points.iter().map(|p| *p + rng.vec3(0.1)).collect();

    let wheel = (0..5)
        .map(|k| {
            let [dx, dy, dth] = planar_delta(&bases[k], &bases[k + 1]);
            let noisy = [
                dx + 0.005 * rng.next(),
                dy + 0.005 * rng.next(),
                dth + 0.005 * rng.next(),
            ];
            wheel_edge(k, k + 1, noisy, 0.005, cb)
        })
        .collect();

    Scene {
        truth,
        initial,
        points,
        observations,
        wheel,
    }
}

fn rms_position_error(out: &ViBaResult, truth: &[Pose3d]) -> f64 {
    let sq: f64 = out
        .keyframes
        .iter()
        .zip(truth)
        .map(|(kf, t)| {
            let c = kf.pose.inverse().translation;
            (c - t.inverse().translation).length().powi(2)
        })
        .sum();
    (sq / truth.len() as f64).sqrt()
}

#[test]
fn wheel_edges_reduce_pose_error_on_a_noisy_planar_trajectory() {
    let mut vision_total = 0.0;
    let mut fused_total = 0.0;
    for seed in 1..=5 {
        let s = planar_scene(seed);
        let vision = solve(&s.initial, &s.points, &s.observations, &[]);
        let fused = solve(&s.initial, &s.points, &s.observations, &s.wheel);
        let (e_vision, e_fused) = (
            rms_position_error(&vision, &s.truth),
            rms_position_error(&fused, &s.truth),
        );
        eprintln!("seed {seed}: vision {e_vision:.4} m, vision+wheel {e_fused:.4} m");
        assert!(e_fused < e_vision, "seed {seed}");
        vision_total += e_vision;
        fused_total += e_fused;
    }
    assert!(
        fused_total < 0.5 * vision_total,
        "vision {vision_total:.4} vs fused {fused_total:.4}"
    );
}

#[test]
fn final_cost_counts_conflicting_wheel_edges() {
    let cb = t_cb(0.2);
    let edges = [
        wheel_edge(0, 1, [0.4, 0.0, 0.1], 0.01, cb),
        wheel_edge(0, 1, [0.5, 0.0, 0.1], 0.01, cb),
    ];
    let kfs = [
        keyframe(camera_pose(&base(0.0, 0.0, 0.0), &cb), true),
        keyframe(camera_pose(&base(0.2, 0.1, 0.0), &cb), false),
    ];
    let out = solve(&kfs, &[], &[], &edges);
    let solved_delta = planar_delta(
        &base_pose(&out.keyframes[0].pose, &cb),
        &base_pose(&out.keyframes[1].pose, &cb),
    );
    assert!((solved_delta[0] - 0.45).abs() < 1e-4, "{solved_delta:?}");
    // Each edge is 5 cm off at 1 cm sigma: ½ · 25 apiece.
    assert!((out.final_cost - 25.0).abs() < 1e-2, "{}", out.final_cost);
}
