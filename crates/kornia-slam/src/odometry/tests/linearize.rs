//! T6 and the linearization: the system's gradient is the cost's derivative,
//! a stereo pair triangulates its point, and the LM loop's rules.

use std::collections::BTreeMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, SVecF64, Vec3F64};
use kornia_factors::inertial::preintegration::information;
use kornia_factors::reprojection::AnchoredLandmark;
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

use crate::odometry::config::{OdometryTuning, VisualJacobians};
use crate::odometry::landmarks::{Landmark, LandmarkStore, triangulate_from_host};
use crate::odometry::linearize::{
    Cost, LandmarkBlock, Linearization, Order, Rows, VarKind, cost, huber, landmark_rows, linearize,
};
use crate::odometry::solve::{SolveScratch, lambda_after_accept, lambda_after_reject, optimize};
use crate::odometry::state::{STATE_DOF, StateWithLin};
use crate::odometry::window::{ImuEdge, StereoView, Window};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

const BASELINE: f64 = 0.11;

fn camera() -> kornia_3d::camera::PinholeCamera {
    kornia_3d::camera::PinholeCamera {
        fx: 435.0,
        fy: 435.0,
        cx: 376.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

fn views() -> [StereoView; 2] {
    [
        StereoView {
            camera: camera(),
            camera_from_primary: Pose3d::IDENTITY,
        },
        StereoView {
            camera: camera(),
            camera_from_primary: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-BASELINE, 0.0, 0.0)),
        },
    ]
}

fn project(view: &StereoView, pose_cw: &Pose3d, point: &Vec3F64) -> [f64; 2] {
    let p = view
        .camera_from_primary
        .compose(pose_cw)
        .transform_point(point);
    let c = &view.camera;
    [c.fx * p.x / p.z + c.cx, c.fy * p.y / p.z + c.cy]
}

/// Two states 0.1 s apart under a constant reading, both perturbed off the
/// truth, the second free; landmarks hosted in the first, seen by both
/// cameras of both frames with a little pixel error.
fn two_state_window() -> Window {
    let t_bc = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.03, -0.01, 0.02));
    let gravity = Vec3F64::new(0.0, 0.0, -9.81);
    let mut window = Window::new(&OdometryTuning::default(), views(), t_bc, gravity);
    // The camera looks along world +x, z up.
    let r_wc = Mat3F64::from_cols(
        Vec3F64::new(0.0, -1.0, 0.0),
        Vec3F64::new(0.0, 0.0, -1.0),
        Vec3F64::new(1.0, 0.0, 0.0),
    );
    let pose0 = Pose3d::new(r_wc, Vec3F64::ZERO).inverse();
    let pose1 = Pose3d::new(
        r_wc * SO3F64::exp(Vec3F64::new(0.0, 0.02, 0.0)).matrix(),
        Vec3F64::new(0.12, 0.01, 0.0),
    )
    .inverse();
    let state = |pose_cw: Pose3d| FrameNavState {
        pose_cw,
        velocity: Vec3F64::new(1.2, 0.1, 0.0),
        bias: ImuBias {
            gyro: Vec3F64::new(1e-3, -2e-3, 5e-4),
            accel: Vec3F64::new(0.02, -0.01, 0.03),
        },
    };
    window.initialize_at(0, state(pose0));
    let calib = ImuCalib {
        gyro_noise: 1.7e-4,
        accel_noise: 2e-3,
        gyro_bias_noise: 1.9e-5,
        accel_bias_noise: 3e-3,
    };
    let samples: Vec<ImuMeasurement> = (1..=20)
        .map(|k| ImuMeasurement {
            timestamp: k as f64 * 0.005,
            gyro: Vec3F64::new(0.01, 0.2, -0.02),
            accel: Vec3F64::new(0.3, 9.7, 0.4),
        })
        .collect();
    let pim = PreintegratedImu::from_measurements(ImuBias::default(), calib, &samples, 0.0, 0.1);
    let omega = information(&pim, 1.0).omega;
    let id1 = 100_000_000;
    window.states.insert(id1, StateWithLin::new(state(pose1)));
    window.imu.insert(
        0,
        ImuEdge {
            to: id1,
            pim,
            omega,
        },
    );
    let views = views();
    for k in 0..12u64 {
        let point = Vec3F64::new(
            3.0 + 0.3 * k as f64,
            -1.0 + 0.2 * k as f64,
            -0.6 + 0.1 * (k % 4) as f64,
        );
        let mut observations = BTreeMap::new();
        for (frame, pose) in [(0, pose0), (id1, pose1)] {
            for camera in 0..2u8 {
                let mut pixel = project(&views[usize::from(camera)], &pose, &point);
                pixel[0] += 0.3 * ((k + u64::from(camera)) % 3) as f64 - 0.3;
                pixel[1] -= 0.2 * (k % 2) as f64;
                observations.insert((frame, camera), pixel);
            }
        }
        let value = AnchoredLandmark::from_host_point(&pose0.transform_point(&point)).unwrap();
        window.landmarks.insert(
            k,
            Landmark {
                host: 0,
                value,
                observations,
            },
        );
    }
    window
}

/// The gradient `linearize` puts in the system is the derivative of `cost`
/// along every dimension of the free state: the Jacobians, their signs and
/// the Huber weights agree with the cost the solve accepts on.
#[test]
fn the_system_gradient_is_the_cost_derivative() {
    let window = two_state_window();
    let mut lin = Linearization::default();
    linearize(&window, &mut lin).unwrap();
    let entry = *lin
        .order
        .entries()
        .iter()
        .find(|e| e.id == 100_000_000 && e.kind == VarKind::State)
        .unwrap();
    let h = 1e-6;
    for d in 0..STATE_DOF {
        let at = |step: f64| {
            let mut moved = two_state_window();
            let mut inc = SVecF64::<STATE_DOF>::zeros();
            inc[d] = step;
            moved
                .states
                .get_mut(&100_000_000)
                .unwrap()
                .apply_increment(&inc);
            cost(&moved).total()
        };
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        // The landmarks are eliminated: the reduced gradient differs from
        // the cost's by `H_pl H_ll⁻¹ b_l`.
        let analytic = gradient_without_landmark_elimination(&window, entry.offset + d);
        assert!(
            (analytic - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
            "dof {d}: system {analytic:.6e}, cost derivative {numeric:.6e}"
        );
    }
    assert!(lin.cost.rows[0] > 0 && lin.cost.rows[1] > 0);
}

/// The full gradient at index `i` with every landmark held: the reduced
/// system plus `H_pl H_ll⁻¹ b_l` back.
fn gradient_without_landmark_elimination(window: &Window, i: usize) -> f64 {
    let mut lin = Linearization::default();
    linearize(window, &mut lin).unwrap();
    lin.system.b[i]
        + lin.blocks[..lin.used_blocks]
            .iter()
            .map(|(_, b)| b.schur_gradient(i))
            .sum::<f64>()
}

/// A stereo pair triangulates its own point: the right camera's extrinsic
/// has the sign the rows use.
#[test]
fn a_stereo_pair_triangulates_its_point() {
    let views = views();
    let pose = Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.1, 0.2, -0.1)).matrix(),
        Vec3F64::new(0.3, -0.2, 1.0),
    );
    let point = pose
        .inverse()
        .transform_point(&Vec3F64::new(0.4, -0.3, 4.0));
    let right = views[1].camera_from_primary.compose(&pose);
    let lm = triangulate_from_host(
        &pose,
        project(&views[0], &pose, &point),
        [(right, project(&views[1], &pose, &point))],
        &camera(),
        0.05,
        3.0,
    )
    .unwrap();
    let back = lm.bearing_vector() * (1.0 / lm.inverse_distance);
    assert!(
        (back - pose.transform_point(&point)).length() < 1e-6,
        "{back:?}"
    );
    // Too close a baseline is skipped.
    assert!(
        triangulate_from_host(
            &pose,
            [376.0, 240.0],
            [(pose, [380.0, 240.0])],
            &camera(),
            0.05,
            3.0
        )
        .is_none()
    );
}

/// The store drops the observations of removed frames, then landmarks left
/// with one; a landmark the frame does not see is lost.
#[test]
fn the_landmark_store_follows_removed_frames() {
    let mut store = LandmarkStore::default();
    let value = AnchoredLandmark {
        bearing: [0.0, 0.0],
        inverse_distance: 0.2,
    };
    for (id, frames) in [(1u64, vec![10, 20, 30]), (2, vec![10, 20])] {
        let observations = frames.iter().map(|&f| ((f, 0u8), [1.0, 2.0])).collect();
        store.insert(
            id,
            Landmark {
                host: 10,
                value,
                observations,
            },
        );
    }
    assert_eq!(store.lost_in(30), vec![2]);
    store.remove_frames(&[10]);
    assert_eq!(store.len(), 1);
    assert!(store.contains(1) && !store.contains(2));
    assert_eq!(store.get(1).map(|l| l.observations.len()), Some(2));
    store.remove(1);
    assert_eq!(store.len(), 0);
}

/// T6: Huber weight at twice the knee is a half, and the damping updates.
#[test]
fn t6_huber_weight_and_damping() {
    let t = OdometryTuning::default();
    let (w, _) = huber(&SVecF64::<2>::new(2.0, 0.0), t.pixel_sigma, t.huber_px);
    assert!((w * t.pixel_sigma * t.pixel_sigma - 0.5).abs() < 1e-15);
    let (w, c) = huber(&SVecF64::<2>::new(0.6, 0.0), t.pixel_sigma, t.huber_px);
    assert!((w * t.pixel_sigma * t.pixel_sigma - 1.0).abs() < 1e-15);
    assert!((c - 0.5 * 0.36 / 0.25).abs() < 1e-15);
    assert!((lambda_after_accept(3e-4, 1.0, 1e-6) - 1e-4).abs() < 1e-18);
    assert!((lambda_after_accept(3e-4, 0.5, 1e-6) - 3e-4).abs() < 1e-18);
    assert_eq!(lambda_after_accept(1e-6, 1.0, 1e-6), 1e-6);
    assert_eq!(lambda_after_reject(1e-4, 2.0), (2e-4, 4.0));
}

/// T6: LM lowers the cost of the perturbed window within its 8 attempts.
#[test]
fn t6_the_solve_lowers_the_cost_within_its_budget() {
    let mut window = two_state_window();
    let report = optimize(&mut window, &mut SolveScratch::default());
    assert!(!report.failed);
    assert!(report.attempts >= 1 && report.attempts <= 8, "{report:?}");
    assert!(report.final_cost < report.initial_cost, "{report:?}");
    assert!((cost(&window).total() - report.final_cost).abs() <= 1e-9 * report.final_cost.max(1.0));
}

/// T6: a system that never factors leaves the states at the prediction.
#[test]
fn t6_a_failed_factorization_keeps_the_prediction() {
    let mut window = two_state_window();
    for i in 0..window.prior.dim() {
        window.prior.h[(i, i)] -= 1e12;
    }
    let before: Vec<_> = window.states.values().map(|s| s.value().pose_cw).collect();
    let report = optimize(&mut window, &mut SolveScratch::default());
    assert!(report.failed);
    let after: Vec<_> = window.states.values().map(|s| s.value().pose_cw).collect();
    assert_eq!(before, after);
}

/// M8: a marginalization reads only the rows of frames in its order; here
/// the host frame's two rows, which have no pose columns.
#[test]
fn rows_outside_the_order_are_left_out() {
    let window = two_state_window();
    let mut order = Order::default();
    order.push(0, VarKind::State);
    let lm = window.landmarks.get(3).unwrap();
    let (mut block, mut c) = (LandmarkBlock::default(), Cost::default());
    landmark_rows(&window, lm, &order, Rows::In(&order), &mut block, &mut c).unwrap();
    assert_eq!(block.rows(), 2);
    assert_eq!(c.rows, [1, 1]);
    let (mut all, mut c) = (LandmarkBlock::default(), Cost::default());
    let mut full = Order::default();
    window.order(&mut full);
    landmark_rows(&window, lm, &full, Rows::All, &mut all, &mut c).unwrap();
    assert_eq!(all.rows(), 4);
}

/// A/B 1 (2x-5 K1): under `VisualJacobians::Current` a pose in the prior with
/// a delta gives the visual rows of the same pose linearized afresh at its
/// value; under `FirstEstimate` it does not.
#[test]
fn current_visual_jacobians_read_the_current_pose() {
    let blocks = |window: &Window| {
        let mut order = Order::default();
        window.order(&mut order);
        let lm = window.landmarks.get(3).unwrap();
        let (mut block, mut c) = (LandmarkBlock::default(), Cost::default());
        landmark_rows(window, lm, &order, Rows::All, &mut block, &mut c).unwrap();
        block.recorded.clone()
    };
    let step = SVecF64::<STATE_DOF>::from_fn(|r, _| if r < 6 { 2e-2 } else { 0.0 });
    for (jacobians, same) in [
        (VisualJacobians::Current, true),
        (VisualJacobians::FirstEstimate, false),
    ] {
        // The host state 0 is in the prior: moved, it keeps its first estimate.
        let mut moved = two_state_window();
        moved.tuning.visual_jacobians = jacobians;
        moved.states.get_mut(&0).unwrap().apply_increment(&step);
        // The same value linearized afresh.
        let mut fresh = two_state_window();
        fresh.tuning.visual_jacobians = jacobians;
        let value = moved.states[&0].value();
        fresh.states.insert(0, StateWithLin::new(value));
        fresh.states.get_mut(&0).unwrap().set_linearized();
        let (a, b) = (blocks(&moved), blocks(&fresh));
        assert_eq!(a.len(), b.len());
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x.2 - y.2).abs().max().max((x.3 - y.3).abs().max()))
            .fold(0.0, f64::max);
        if same {
            assert!(max < 1e-9, "{jacobians:?}: {max}");
        } else {
            assert!(max > 1e-3, "{jacobians:?}: {max}");
        }
    }
}
