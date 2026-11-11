use super::tests::{AXIAL_MISMATCH_RAD, EDGE_S, spin_edge, spin_pose, spin_vision};
use super::*;
use crate::mapping::marginalization::{FrozenPoseSigma, marginalize_frozen};
use kornia_algebra::Mat3F64;
use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

/// The fixture of `boundary_edge_keeps_the_bias_walk_at_full_strength`.
fn bias_walk_solve(params: &ViBaParams) -> ViBaResult {
    let camera = PinholeCamera {
        fx: 4000.0,
        fy: 4000.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    let keyframes: Vec<ViBaKeyframe> = (0..4)
        .map(|k| ViBaKeyframe {
            pose: spin_pose(k as f64 * EDGE_S),
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
            fixed: k == 0,
        })
        .collect();
    let imu_edges = [
        spin_edge(0, 0.0, params.gravity),
        spin_edge(1, AXIAL_MISMATCH_RAD, params.gravity),
        spin_edge(2, AXIAL_MISMATCH_RAD, params.gravity),
    ];
    let (points, observations) = spin_vision(&keyframes, &camera);
    visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &observations,
        &imu_edges,
        &[],
        &[],
        &camera,
        params,
    )
    .unwrap()
}

/// `Marginal` without a prior is the `ScaledEdge` solve, bit for bit, and
/// says it fell back.
#[test]
fn marginal_without_a_prior_is_the_scaled_edge_solve() {
    let scaled = ViBaParams {
        boundary_scales_bias_rows: false,
        ..ViBaParams::default()
    };
    let marginal = ViBaParams {
        boundary: BoundaryPrior::Marginal(FrozenPoseSigma::default()),
        ..scaled.clone()
    };
    let (a, b) = (bias_walk_solve(&scaled), bias_walk_solve(&marginal));
    assert_eq!(format!("{:?}", a.keyframes), format!("{:?}", b.keyframes));
    assert_eq!(format!("{:?}", a.points), format!("{:?}", b.points));
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.final_cost.to_bits(), b.final_cost.to_bits());
    assert!(!a.fallback && b.fallback);
    assert_eq!(b.prior_residual_final, None);
}

const SPEED_MPS: f64 = 1.0;
const MAP_SCALE: f64 = 0.7;
const CHAIN_EDGES: usize = 20;

fn euroc_calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.7e-4,
        accel_noise: 2e-3,
        gyro_bias_noise: 2e-5,
        accel_bias_noise: 3e-3,
    }
}

/// Level camera at world `x` moving along x at `speed`, as `T_cw`.
fn level_keyframe(x: f64, speed: f64, fixed: bool) -> ViBaKeyframe {
    ViBaKeyframe {
        pose: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-x, 0.0, 0.0)),
        velocity: Vec3F64::new(speed, 0.0, 0.0),
        bias: ImuBias::default(),
        fixed,
    }
}

/// A metric edge of constant velocity at a level attitude.
fn cruise_edge(k: usize, gravity: Vec3F64) -> ImuFactor {
    let mut preintegrated = PreintegratedImu::new(ImuBias::default(), euroc_calib());
    let n = 40;
    let sample = ImuMeasurement {
        timestamp: 0.0,
        gyro: Vec3F64::ZERO,
        accel: gravity * -1.0,
    };
    for _ in 0..n {
        preintegrated.integrate(&sample, EDGE_S / n as f64);
    }
    ImuFactor {
        from_idx: k,
        to_idx: k + 1,
        preintegrated,
    }
}

/// Fixed landmarks 5 and 15 m ahead of each free keyframe, imaged exactly:
/// vision pins the poses where `keyframes` puts them, to a few mm.
fn cruise_vision(
    keyframes: &[ViBaKeyframe],
    camera: &PinholeCamera,
) -> (Vec<Vec3F64>, Vec<BaObservation>) {
    let (mut points, mut observations) = (Vec::new(), Vec::new());
    for (pose_idx, kf) in keyframes.iter().enumerate().filter(|(_, kf)| !kf.fixed) {
        let t_wc = kf.pose.inverse();
        for ix in -3..=3 {
            for iy in -3..=3 {
                let z = if (ix + iy) % 2 == 0 { 5.0 } else { 15.0 };
                let p_cam = Vec3F64::new(ix as f64 * 0.2 * z, iy as f64 * 0.2 * z, z);
                observations.push(BaObservation {
                    pose_idx,
                    point_idx: points.len(),
                    pixel: [
                        (camera.fx * p_cam.x / z + camera.cx) as f32,
                        (camera.fy * p_cam.y / z + camera.cy) as f32,
                    ],
                    fixed_pose: false,
                    fixed_point: true,
                    depth_meas: None,
                    depth_sigma: 1.0,
                });
                points.push(t_wc.transform_point(&p_cam));
            }
        }
    }
    (points, observations)
}

/// A window whose map is 0.7× scale (vision says 0.7 m/s) against metric
/// IMU edges of 1 m/s cruise. The scaled boundary edge lets the velocity
/// follow the map; a prior marginalized from a metric chain holds it.
///
/// The chain is 20 edges, where the prior's velocity sigma has settled
/// (4.1e-3 m/s at 2 mm). After 5 edges it is still 1.7e-2 m/s, and the
/// window's oldest velocity ends at 0.89 m/s, accel bias 0.18 m/s²
/// (2026-11-11).
#[test]
fn marginal_prior_holds_the_metric_velocity_against_a_scaled_map() {
    let camera = PinholeCamera {
        fx: 200.0,
        fy: 200.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    let gravity = ViBaParams::default().gravity;
    // Metric chain 0..=5; the window's oldest active keyframe is chain 5,
    // and the map is scaled about it.
    let chain: Vec<ViBaKeyframe> = (0..=CHAIN_EDGES)
        .map(|k| level_keyframe(k as f64 * EDGE_S * SPEED_MPS, SPEED_MPS, false))
        .collect();
    let anchor = CHAIN_EDGES as f64 * EDGE_S * SPEED_MPS;
    let scaled_speed = MAP_SCALE * SPEED_MPS;
    let window: Vec<ViBaKeyframe> = (-1..=2)
        .map(|k| {
            let x = anchor + k as f64 * EDGE_S * scaled_speed;
            level_keyframe(x, scaled_speed, k < 0)
        })
        .collect();
    let imu_edges: Vec<ImuFactor> = (0..3).map(|k| cruise_edge(k, gravity)).collect();
    let (points, observations) = cruise_vision(&window, &camera);
    let solve = |params: &ViBaParams| {
        visual_inertial_bundle_adjust(
            &window,
            &points,
            &observations,
            &imu_edges,
            &[],
            &[],
            &camera,
            params,
        )
        .unwrap()
    };

    let scaled = solve(&ViBaParams::default());
    for kf in &scaled.keyframes[1..] {
        let ratio = kf.velocity.x / scaled_speed;
        assert!((ratio - 1.0).abs() < 0.15, "ScaledEdge v {}", kf.velocity.x);
    }

    let sigma = FrozenPoseSigma::default();
    let base = ViBaParams {
        imu_huber: ImuHuber::Off,
        ..ViBaParams::default()
    };
    let mut prior = None;
    for k in 0..CHAIN_EDGES {
        let edge = cruise_edge(k, gravity);
        prior = Some(
            marginalize_frozen(
                prior.as_ref(),
                &chain[k],
                &chain[k + 1],
                &edge,
                &base,
                sigma,
            )
            .unwrap(),
        );
    }
    let marginal = solve(&ViBaParams {
        boundary: BoundaryPrior::Marginal(sigma),
        prior,
        ..base
    });
    assert!(!marginal.fallback);
    for kf in &marginal.keyframes[1..] {
        let ratio = kf.velocity.x / SPEED_MPS;
        assert!((ratio - 1.0).abs() < 0.10, "Marginal v {}", kf.velocity.x);
    }
}
