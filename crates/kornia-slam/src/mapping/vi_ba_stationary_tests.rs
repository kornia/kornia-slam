use super::*;
use kornia_factors::stationary::{StationaryFactor, StationaryGyro};
use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

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

fn calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.7e-4,
        accel_noise: 2e-3,
        gyro_bias_noise: 2e-5,
        accel_bias_noise: 3e-3,
    }
}

/// Deterministic uniform samples in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn vec3(&mut self, scale: f64) -> Vec3F64 {
        Vec3F64::new(self.next(), self.next(), self.next()) * scale
    }
    fn pose(&mut self, t_scale: f64, r_scale: f64) -> Pose3d {
        Pose3d::new(SO3F64::exp(self.vec3(r_scale)).matrix(), self.vec3(t_scale))
    }
}

fn se3_exp(xi: &[f64]) -> Pose3d {
    let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
    let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
    Pose3d::new(
        SO3F64::exp(omega).matrix(),
        SO3F64::left_jacobian(omega) * rho,
    )
}

/// Applies a 15-DOF tangent step the way the solver does (pose
/// right-perturbed, the rest additive), in f64.
fn retract(kf: &ViBaKeyframe, d: &[f64]) -> ViBaKeyframe {
    let mut out = kf.clone();
    out.pose = kf.pose.compose(&se3_exp(&d[0..6]));
    out.velocity += Vec3F64::new(d[6], d[7], d[8]);
    out.bias.gyro += Vec3F64::new(d[9], d[10], d[11]);
    out.bias.accel += Vec3F64::new(d[12], d[13], d[14]);
    out
}

fn random_keyframe(rng: &mut Lcg, near: Option<&Pose3d>) -> ViBaKeyframe {
    let pose = match near {
        Some(p) => p.compose(&se3_exp(&std::array::from_fn::<f64, 6, _>(|_| {
            rng.next() * 0.01
        }))),
        None => rng.pose(2.0, 1.0),
    };
    ViBaKeyframe {
        pose,
        velocity: rng.vec3(0.02),
        bias: ImuBias {
            gyro: rng.vec3(1e-3),
            accel: rng.vec3(1e-2),
        },
        fixed: false,
    }
}

fn factor(gyro: bool, huber: Option<f64>) -> StationaryFactor {
    let mut f = StationaryFactor::new(0, 1, 30.0, calib());
    f.huber_chi2 = huber;
    if gyro {
        f.gyro = Some(StationaryGyro::new(
            Vec3F64::new(0.003, -0.001, 0.0005),
            25.0,
        ));
    }
    f
}

/// The stacked 21-row residual and its per-row information (robust weight
/// folded into the 12 pose and velocity rows).
fn stacked(edge: &StationaryFactor, kfs: &[ViBaKeyframe]) -> (Vec<f64>, Vec<f64>) {
    let info = edge.information();
    let e = stationary_eval(edge, &info, &kfs[0], &kfs[1]);
    let mut r: Vec<f64> = e.pose.to_vec();
    let mut w: Vec<f64> = (0..6).map(|k| e.weight * info.pose[k][k]).collect();
    for v in [e.velocity_i, e.velocity_j] {
        r.extend([v.x, v.y, v.z]);
        w.extend([e.weight * info.velocity; 3]);
    }
    for (b, i) in [
        (e.gyro_bias, info.gyro_bias),
        (e.accel_bias, info.accel_bias),
    ] {
        r.extend([b.x, b.y, b.z]);
        w.extend([i; 3]);
    }
    if let (Some(g), Some((_, i))) = (e.gyro_mean, info.gyro_mean) {
        r.extend([g.x, g.y, g.z]);
        w.extend([i; 3]);
    }
    (r, w)
}

/// `accumulate_stationary_edges`' normal equations against `JᵀWΩJ` and
/// `−JᵀWΩr` from a central-difference Jacobian of the residual over both
/// 15-DOF tangents (the robust weight held at the linearization point, as
/// IRLS does), its cost against the Huber objective `½ρ(chi²_nav) + ½ chi²_bias`
/// ([`crate::mapping::robust`]), and its gradient against a central difference
/// of that cost (so cost and weight are one objective). Each entry is judged
/// against its Cauchy–Schwarz scale (`√(H_aa H_bb)`, `√H_aa · √(rᵀWΩr)`), so
/// the stiff bias rows cannot hide an error in the pose rows. Returns the
/// worst mismatch in units of 1e-6 of that scale.
fn normal_equation_error(edge: &StationaryFactor, kfs: &[ViBaKeyframe]) -> f64 {
    const EPS: f64 = 1e-6;
    let infos = stationary_information(std::slice::from_ref(edge), 2).unwrap();
    let mut m_mat = Mat::<f64>::zeros(30, 30);
    let mut m_vec = vec![0.0; 30];
    let cost = accumulate_stationary_edges(
        &mut m_mat,
        &mut m_vec,
        kfs,
        &[0, 1],
        std::slice::from_ref(edge),
        &infos,
    );
    assert_eq!(
        cost,
        stationary_cost(kfs, &[0, 1], std::slice::from_ref(edge), &infos)
    );

    let (r0, w) = stacked(edge, kfs);
    let rows = r0.len();
    let chi2: f64 = r0.iter().zip(&w).map(|(r, w)| w * r * r).sum();
    // The first 12 rows (pose, velocities) carry the robust weight.
    let weight = stationary_eval(edge, &infos[0], &kfs[0], &kfs[1]).weight;
    let weighted =
        |range: std::ops::Range<usize>| -> f64 { range.map(|k| w[k] * r0[k] * r0[k]).sum() };
    let expected_cost =
        kornia_factors::robust::huber_weight_and_cost(weighted(0..12) / weight, edge.huber_chi2).1
            + 0.5 * weighted(12..rows);
    let mut worst = (cost - expected_cost).abs() / (1e-6 * chi2.max(1e-12));
    let mut jac = vec![[0.0f64; 30]; rows];
    for col in 0..30 {
        let shifted = |s: f64| {
            let mut d = [0.0; 15];
            d[col % 15] = s;
            let mut moved = kfs.to_vec();
            moved[col / 15] = retract(&kfs[col / 15], &d);
            stacked(edge, &moved).0
        };
        let (p, m) = (shifted(EPS), shifted(-EPS));
        for row in 0..rows {
            jac[row][col] = (p[row] - m[row]) / (2.0 * EPS);
        }
    }
    let h = |a: usize, b: usize| -> f64 { (0..rows).map(|k| jac[k][a] * w[k] * jac[k][b]).sum() };
    let diag: Vec<f64> = (0..30).map(|a| h(a, a)).collect();
    let cost_at = |col: usize, s: f64| {
        let mut d = [0.0; 15];
        d[col % 15] = s;
        let mut moved = kfs.to_vec();
        moved[col / 15] = retract(&kfs[col / 15], &d);
        stationary_cost(&moved, &[0, 1], std::slice::from_ref(edge), &infos)
    };
    for a in 0..30 {
        let g: f64 = (0..rows).map(|k| jac[k][a] * w[k] * r0[k]).sum();
        let tol = 1e-6 * (diag[a] * chi2).sqrt() + 1e-12;
        worst = worst.max((m_vec[a] + g).abs() / tol);
        // The weighted gradient is the cost's own: d cost / d x_a = −m_a.
        // Its central difference carries an O(ε²) error, judged 100× looser.
        let numeric = (cost_at(a, EPS) - cost_at(a, -EPS)) / (2.0 * EPS);
        worst = worst.max((m_vec[a] + numeric).abs() / (100.0 * tol));
        for b in 0..30 {
            let tol = 1e-6 * (diag[a] * diag[b]).sqrt() + 1e-12;
            worst = worst.max((m_mat[(a, b)] - h(a, b)).abs() / tol);
        }
    }
    worst
}

#[test]
fn normal_equations_match_numeric_jacobians() {
    let mut rng = Lcg(17);
    for gyro in [false, true] {
        for huber in [None, Some(StationaryFactor::DEFAULT_HUBER_CHI2)] {
            for _ in 0..10 {
                // Far apart: Huber active and a large pose residual.
                let i = random_keyframe(&mut rng, None);
                let j = random_keyframe(&mut rng, None);
                let err = normal_equation_error(&factor(gyro, huber), &[i, j]);
                assert!(err < 1.0, "far, gyro={gyro} huber={huber:?}: {err:e}");
                // Close: Huber inactive, the series branch of the Jacobian.
                let i = random_keyframe(&mut rng, None);
                let j = random_keyframe(&mut rng, Some(&i.pose));
                let err = normal_equation_error(&factor(gyro, huber), &[i, j]);
                assert!(err < 1.0, "near, gyro={gyro} huber={huber:?}: {err:e}");
            }
        }
    }
}

#[test]
fn huber_weights_only_pose_and_velocity_rows() {
    let mut rng = Lcg(3);
    let i = random_keyframe(&mut rng, None);
    let j = random_keyframe(&mut rng, None);
    let f = factor(true, Some(1.0));
    let e = stationary_eval(&f, &f.information(), &i, &j);
    assert!(e.weight < 1e-2, "{}", e.weight);
    let quadratic = factor(true, None);
    let q = stationary_eval(&quadratic, &quadratic.information(), &i, &j);
    assert_eq!(q.weight, 1.0);
    // The bias part of the cost is the same with and without Huber.
    let info = f.information();
    let bias = |e: &kornia_factors::stationary::StationaryResidual| {
        let g = e.gyro_mean.unwrap();
        0.5 * (info.gyro_bias * e.gyro_bias.dot(e.gyro_bias)
            + info.accel_bias * e.accel_bias.dot(e.accel_bias)
            + info.gyro_mean.unwrap().1 * g.dot(g))
    };
    // The nav part is the Huber objective ½ρ of the quadratic one's chi².
    let chi2_nav = 2.0 * (q.cost - bias(&q));
    let (w, huber_nav) = kornia_factors::robust::huber_weight_and_cost(chi2_nav, Some(1.0));
    assert_eq!(w, e.weight);
    assert!(((e.cost - bias(&e)) - huber_nav).abs() < 1e-9 * huber_nav);
    assert_eq!(bias(&e), bias(&q));
}

fn shared_points() -> Vec<Vec3F64> {
    (0..12)
        .map(|k| {
            Vec3F64::new(
                (k % 4) as f64 * 0.6 - 0.9,
                (k / 4) as f64 * 0.5 - 0.5,
                4.0 + (k % 3) as f64 * 0.8,
            )
        })
        .collect()
}

fn observations(poses: &[(usize, Pose3d, bool)], points: &[Vec3F64]) -> Vec<BaObservation> {
    let cam = camera();
    let mut obs = Vec::new();
    for &(pose_idx, pose, fixed) in poses {
        for (point_idx, p) in points.iter().enumerate() {
            let pc = pose.transform_point(p);
            obs.push(BaObservation {
                pose_idx,
                point_idx,
                pixel: [
                    (cam.fx * pc.x / pc.z + cam.cx) as f32,
                    (cam.fy * pc.y / pc.z + cam.cy) as f32,
                ],
                fixed_pose: fixed,
                // Single-view landmarks: held, so the edge is what moves.
                fixed_point: true,
                depth_meas: None,
                depth_sigma: 1.0,
            });
        }
    }
    obs
}

fn params() -> ViBaParams {
    ViBaParams {
        max_iterations: 50,
        ..ViBaParams::default()
    }
}

fn parked_bias() -> ImuBias {
    ImuBias {
        gyro: Vec3F64::new(0.003, -0.001, 0.0005),
        accel: Vec3F64::ZERO,
    }
}

fn pose_gap(a: &Pose3d, b: &Pose3d) -> (f64, f64) {
    let d = a.inverse().compose(b);
    (
        d.translation.length(),
        SO3F64::from_matrix(&d.rotation).log().length(),
    )
}

/// A parked pair: keyframe 0 fixed at the truth, keyframe 1 started 5 mm,
/// 0.5° and 5 cm/s off with a gyro bias 1e-3 off, linked only by a
/// stationary edge. The landmarks are where 0 sees them; they are free.
#[test]
fn a_stationary_edge_pulls_a_parked_keyframe_onto_its_neighbour() {
    let truth = Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.05)).matrix(),
        Vec3F64::new(0.3, -0.1, 0.2),
    );
    let points = shared_points();
    let off = truth.compose(&se3_exp(&[0.005, 0.0, 0.0, 0.0, 0.0087, 0.0]));
    let kfs = [
        ViBaKeyframe {
            pose: truth,
            velocity: Vec3F64::ZERO,
            bias: parked_bias(),
            fixed: true,
        },
        ViBaKeyframe {
            pose: off,
            velocity: Vec3F64::new(0.05, 0.0, -0.03),
            bias: ImuBias {
                gyro: parked_bias().gyro + Vec3F64::new(1e-3, 0.0, -1e-3),
                accel: Vec3F64::ZERO,
            },
            fixed: false,
        },
    ];
    // The landmarks as keyframe 0 sees them; keyframe 1 sees them only
    // through its (wrong) pose, with no observations of its own: the edge
    // alone has to bring it back.
    let obs = observations(&[(0, truth, true)], &points);
    let mut edge = factor(true, Some(StationaryFactor::DEFAULT_HUBER_CHI2));
    edge.gyro = Some(StationaryGyro::new(parked_bias().gyro, 25.0));
    let out =
        visual_inertial_bundle_adjust(&kfs, &points, &obs, &[], &[], &[edge], &camera(), &params())
            .unwrap();
    let (dt, dr) = pose_gap(&out.keyframes[1].pose, &truth);
    assert!(dt < 1e-4 && dr < 1e-4, "{dt:e} m, {dr:e} rad");
    assert!(
        out.keyframes[1].velocity.length() < 1e-3,
        "{:?}",
        out.keyframes[1].velocity
    );
    let bg = out.keyframes[1].bias.gyro - parked_bias().gyro;
    assert!(bg.length() < 1e-5, "{bg:?}");
    assert_eq!(out.keyframes[0].pose, truth, "the fixed end stays");

    // With observations of its own (consistent with the truth) it converges
    // the same way.
    let obs = observations(&[(0, truth, true), (1, truth, false)], &points);
    let out =
        visual_inertial_bundle_adjust(&kfs, &points, &obs, &[], &[], &[edge], &camera(), &params())
            .unwrap();
    let (dt, dr) = pose_gap(&out.keyframes[1].pose, &truth);
    assert!(dt < 1e-4 && dr < 1e-4, "{dt:e} m, {dr:e} rad");
}

/// Stationary chain fixed → free → free next to a moving pair linked by an
/// IMU edge, in one Schur solve.
#[test]
fn stationary_and_imu_edges_solve_together_around_fixed_keyframes() {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let parked = Pose3d::IDENTITY;
    // The moving pair: 0.5 m/s along +x for 0.5 s, no rotation.
    let v = Vec3F64::new(0.5, 0.0, 0.0);
    let pose_at = |x: f64| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-x, 0.0, 0.0));
    let samples: Vec<ImuMeasurement> = (0..=100)
        .map(|k| ImuMeasurement {
            timestamp: k as f64 * 0.005,
            gyro: Vec3F64::ZERO,
            accel: -gravity,
        })
        .collect();
    let pim = PreintegratedImu::from_measurements(ImuBias::default(), calib(), &samples, 0.0, 0.5);
    let truth = [parked, parked, parked, pose_at(1.0), pose_at(1.25)];
    let bias = ImuBias::default();
    let mut kfs: Vec<ViBaKeyframe> = truth
        .iter()
        .enumerate()
        .map(|(k, pose)| ViBaKeyframe {
            pose: *pose,
            velocity: if k >= 3 { v } else { Vec3F64::ZERO },
            bias,
            fixed: k == 0,
        })
        .collect();
    kfs[1].pose = parked.compose(&se3_exp(&[0.004, -0.002, 0.0, 0.003, 0.0, 0.0]));
    kfs[2].pose = parked.compose(&se3_exp(&[-0.003, 0.0, 0.004, 0.0, -0.004, 0.0]));
    kfs[2].velocity = Vec3F64::new(0.0, 0.02, 0.0);
    kfs[4].pose = pose_at(1.27);
    let points = shared_points();
    let obs = observations(
        &[
            (0, truth[0], true),
            (3, truth[3], false),
            (4, truth[4], false),
        ],
        &points,
    );
    let stationary = [
        StationaryFactor::new(0, 1, 30.0, calib()),
        StationaryFactor::new(1, 2, 30.0, calib()),
    ];
    let imu = [ImuFactor {
        from_idx: 3,
        to_idx: 4,
        preintegrated: pim,
    }];
    let p = ViBaParams {
        gravity,
        ..params()
    };
    let out =
        visual_inertial_bundle_adjust(&kfs, &points, &obs, &imu, &[], &stationary, &camera(), &p)
            .unwrap();
    for k in [1, 2] {
        let (dt, dr) = pose_gap(&out.keyframes[k].pose, &parked);
        assert!(dt < 1e-4 && dr < 1e-4, "kf {k}: {dt:e} m, {dr:e} rad");
        assert!(out.keyframes[k].velocity.length() < 1e-3);
    }
    let (dt, _) = pose_gap(&out.keyframes[4].pose, &truth[4]);
    assert!(dt < 2e-3, "moving kf: {dt:e} m");

    // An edge between two fixed keyframes is skipped: bit-identical to none.
    let mut all_fixed = kfs.clone();
    all_fixed[1].fixed = true;
    let none =
        visual_inertial_bundle_adjust(&all_fixed, &points, &obs, &imu, &[], &[], &camera(), &p)
            .unwrap();
    let skipped = visual_inertial_bundle_adjust(
        &all_fixed,
        &points,
        &obs,
        &imu,
        &[],
        &stationary[..1],
        &camera(),
        &p,
    )
    .unwrap();
    assert_eq!(
        format!("{:?}", none.keyframes),
        format!("{:?}", skipped.keyframes)
    );
    assert_eq!(
        format!("{:?}", none.points),
        format!("{:?}", skipped.points)
    );
    assert_eq!(none.final_cost.to_bits(), skipped.final_cost.to_bits());
}

#[test]
fn stationary_edge_errors_are_reported() {
    let kfs = [
        ViBaKeyframe {
            pose: Pose3d::IDENTITY,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
            fixed: true,
        },
        ViBaKeyframe {
            pose: Pose3d::IDENTITY,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
            fixed: false,
        },
    ];
    let run = |edge: StationaryFactor| {
        visual_inertial_bundle_adjust(&kfs, &[], &[], &[], &[], &[edge], &camera(), &params())
    };
    let edge = StationaryFactor::new(0, 1, 10.0, calib());
    assert!(run(edge).is_ok());
    let linking = |from_idx, to_idx| {
        let mut e = edge;
        (e.from_idx, e.to_idx) = (from_idx, to_idx);
        e
    };
    let mut no_span = edge;
    no_span.span_s = 0.0;
    assert!(matches!(
        run(linking(0, 2)),
        Err(ViBaError::StationaryFactorOutOfRange(0, 2))
    ));
    assert!(matches!(
        run(no_span),
        Err(ViBaError::InvalidStationaryFactor(0))
    ));
    // A self-loop is refused, not accumulated.
    assert!(matches!(
        run(linking(1, 1)),
        Err(ViBaError::InvalidStationaryFactor(0))
    ));
    // A stationary edge alone makes its free keyframe solvable.
    assert!(!matches!(run(edge), Err(ViBaError::NoFreeVariables)));
}
