//! Perturbation-recovery tests driven by the measurement simulator.
//!
//! Each test generates ground truth, displaces the estimate by a known amount,
//! runs the **production** optimizer, and asserts the estimate returns. Nothing
//! is mocked: the code under test is `bundle_adjust_schur` and
//! `visual_inertial_bundle_adjust` as shipped.
//!
//! This is not the acceptance gate — real sequences remain that. These are the
//! instrument for answering "which component is wrong?" once a real sequence has
//! said that something is.

#![cfg(feature = "sim")]

use kornia_3d::ba::BaParams;
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::imu::ImuBias;
use kornia_slam::sim::{
    ArcConfig, ImuSimConfig, LandmarkConfig, ObservationConfig, Perturbation, Scene, SceneConfig,
    SimRng, Trajectory,
};
use kornia_slam::vi_ba_schur::{ViBaParams, visual_inertial_bundle_adjust};

/// Translation and rotation error of `estimate` relative to `truth`.
fn pose_error(estimate: &Pose3d, truth: &Pose3d) -> (f64, f64) {
    let dt = (estimate.translation - truth.translation).length();
    let dr = (SO3F64::from_matrix(&truth.rotation).inverse()
        * SO3F64::from_matrix(&estimate.rotation))
    .log()
    .length();
    (dt, dr)
}

/// Worst-case pose error across all frames.
fn max_pose_error(estimates: &[Pose3d], truth: &[Pose3d]) -> (f64, f64) {
    estimates
        .iter()
        .zip(truth.iter())
        .map(|(e, t)| pose_error(e, t))
        .fold((0.0f64, 0.0f64), |acc, e| (acc.0.max(e.0), acc.1.max(e.1)))
}

/// A visual scene with the given pixel noise, sized for good co-visibility.
fn visual_scene(pixel_noise_std: f64, seed: u64) -> Scene {
    let trajectory = Trajectory::arc(&ArcConfig::default()).expect("trajectory");
    let config = SceneConfig {
        n_keyframes: 10,
        landmarks: LandmarkConfig {
            count: 600,
            ..Default::default()
        },
        observations: ObservationConfig {
            pixel_noise_std,
            ..Default::default()
        },
        min_observations: 3,
        seed,
        ..Default::default()
    };
    Scene::build(&trajectory, &config).expect("scene")
}

/// Phase 2 baseline: with exact measurements the optimizer must return to
/// ground truth.
///
/// Two poses are fixed rather than one. Monocular BA has a 7-DOF gauge freedom
/// — six from the rigid transform plus **scale** — and fixing a single pose
/// leaves scale free, so the reconstruction could shrink or grow uniformly and
/// still be a perfect optimum. Fixing two pins the baseline and makes "returned
/// to ground truth" a well-posed claim.
#[test]
fn visual_ba_recovers_poses_from_noiseless_observations() {
    let scene = visual_scene(0.0, 7);
    let truth_poses = scene.camera_poses.clone();

    let mut visual = scene.visual.clone();
    visual.fix_poses(&[0, 1]);

    let mut poses = truth_poses.clone();
    let mut points = visual.points.clone();
    let fixed = vec![true, true]
        .into_iter()
        .chain(std::iter::repeat_n(false, poses.len() - 2))
        .collect::<Vec<bool>>();

    let perturbation = Perturbation {
        translation_std: 0.05,
        rotation_std: 0.01,
        point_std: 0.10,
        ..Default::default()
    };
    let mut rng = SimRng::new(1000);
    perturbation.apply_to_poses(&mut poses, &fixed, &mut rng);
    perturbation.apply_to_points(&mut points, &mut rng);

    let (start_t, start_r) = max_pose_error(&poses, &truth_poses);
    assert!(
        start_t > 0.02,
        "perturbation was too small to be a real test: {start_t} m"
    );

    let params = BaParams {
        max_iterations: 30,
        ..Default::default()
    };
    let result = bundle_adjust_schur(
        &poses,
        &points,
        &visual.observations,
        &scene.camera,
        &params,
    )
    .expect("bundle adjustment");

    let (end_t, end_r) = max_pose_error(&result.poses, &truth_poses);
    println!(
        "visual noiseless: translation {start_t:.4} -> {end_t:.6} m, rotation {start_r:.4} -> {end_r:.6} rad"
    );

    // The floor here is not machine precision. `bundle_adjust_schur` converts
    // poses to `SE3F32` internally, so ~1e-7 relative precision on a metre-scale
    // translation is the best attainable regardless of how exact the input is.
    // Measured: 2e-6 m / 3e-7 rad. The bounds keep ~50x margin over that so
    // the test is not brittle, while staying far tighter than the perturbation
    // it started from — a vacuous bound here would silently accept a solver
    // that had stopped working.
    assert!(end_t < 1e-4, "translation error {end_t} m after recovery");
    assert!(end_r < 1e-5, "rotation error {end_r} rad after recovery");
    assert!(
        end_t < start_t / 100.0,
        "optimizer barely improved the estimate"
    );
}

/// Phase 2 with noise: the recovered error must be commensurate with the
/// injected pixel noise, not with the size of the perturbation.
///
/// The tolerance is derived rather than picked. A 1 px measurement error at
/// focal length `f` subtends `1/f` radians; over a mean landmark depth `d` the
/// induced position uncertainty is about `d/f` per observation, shrinking as
/// `1/√n` with the number of observations. For `f ≈ 458`, `d ≈ 7 m` and tens of
/// observations per keyframe, that is a few centimetres — which is the order the
/// bound below encodes.
#[test]
fn visual_ba_error_scales_with_pixel_noise() {
    let mut previous = 0.0;
    // Bounds bracket the measured errors (0 m, 0.008 m, 0.040 m) with ~2x margin.
    for (noise, bound) in [(0.0, 1e-4), (0.5, 0.02), (2.0, 0.08)] {
        let scene = visual_scene(noise, 11);
        let truth_poses = scene.camera_poses.clone();

        let mut visual = scene.visual.clone();
        visual.fix_poses(&[0, 1]);

        let mut poses = truth_poses.clone();
        let mut points = visual.points.clone();
        let fixed = vec![true, true]
            .into_iter()
            .chain(std::iter::repeat_n(false, poses.len() - 2))
            .collect::<Vec<bool>>();

        let perturbation = Perturbation {
            translation_std: 0.03,
            rotation_std: 0.005,
            point_std: 0.05,
            ..Default::default()
        };
        let mut rng = SimRng::new(2000);
        perturbation.apply_to_poses(&mut poses, &fixed, &mut rng);
        perturbation.apply_to_points(&mut points, &mut rng);

        let params = BaParams {
            max_iterations: 30,
            ..Default::default()
        };
        let result = bundle_adjust_schur(
            &poses,
            &points,
            &visual.observations,
            &scene.camera,
            &params,
        )
        .expect("bundle adjustment");

        let (end_t, _) = max_pose_error(&result.poses, &truth_poses);
        println!("visual noise {noise} px: translation error {end_t:.5} m");

        assert!(
            end_t < bound,
            "at {noise} px noise, translation error {end_t} m exceeded {bound} m"
        );
        // Monotonicity: more measurement noise must not produce a better
        // estimate. This is the property a sensitivity sweep rests on.
        assert!(
            end_t >= previous * 0.5,
            "error fell from {previous} to {end_t} as noise increased — suspicious"
        );
        previous = end_t;
    }
}

/// An inertial scene carrying a known bias.
fn inertial_scene(bias: ImuBias, seed: u64) -> Scene {
    let trajectory = Trajectory::arc(&ArcConfig {
        climb: 0.4,
        ..Default::default()
    })
    .expect("trajectory");
    let config = SceneConfig {
        n_keyframes: 10,
        landmarks: LandmarkConfig {
            count: 600,
            ..Default::default()
        },
        observations: ObservationConfig {
            pixel_noise_std: 0.0,
            ..Default::default()
        },
        min_observations: 3,
        imu: Some(ImuSimConfig {
            bias,
            add_noise: false,
            ..Default::default()
        }),
        seed,
        ..Default::default()
    };
    Scene::build(&trajectory, &config).expect("scene")
}

/// VI-BA parameters suited to a controlled experiment.
fn vi_ba_params() -> ViBaParams {
    ViBaParams {
        max_iterations: 40,
        // The accel-bias prior exists to absorb gravity error in a windowed
        // solve on real data (issue #51). Here gravity is exactly right by
        // construction, so the prior only fights the quantity under test —
        // disabling it is what makes "did the bias converge to the injected
        // value?" a clean question rather than a tug-of-war.
        accel_bias_prior_weight: 0.0,
        ..Default::default()
    }
}

/// Phase 3, the milestone: inject a known constant accel bias, start the
/// estimate at zero, and assert VI-BA converges to the injected value.
///
/// This is the accel-bias blowup from issue #51 expressed as an assertion.
#[test]
fn vi_ba_recovers_injected_accel_bias() {
    let true_bias = ImuBias {
        gyro: Vec3F64::new(0.004, -0.003, 0.002),
        accel: Vec3F64::new(0.05, -0.04, 0.03),
    };
    let scene = inertial_scene(true_bias, 21);

    let mut keyframes = scene.ground_truth_keyframes();
    // The oldest keyframe anchors the gauge, as in ORB-SLAM3's local inertial
    // BA. Its 15 DOF — pose, velocity and bias — are all held at ground truth.
    keyframes[0].fixed = true;

    // Every free keyframe starts with zero bias: the estimator is told nothing
    // about the injected value.
    for kf in keyframes.iter_mut().skip(1) {
        kf.bias = ImuBias::default();
    }

    // Factors are linearized at the estimator's guess (zero), not at truth.
    let imu_edges = scene.imu_factors(ImuBias::default()).expect("imu factors");

    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &scene.visual.points,
        &scene.visual.observations,
        &imu_edges,
        &scene.camera,
        &vi_ba_params(),
    )
    .expect("vi-ba");

    let mut worst_accel = 0.0f64;
    let mut worst_gyro = 0.0f64;
    for kf in result.keyframes.iter().skip(1) {
        worst_accel = worst_accel.max((kf.bias.accel - true_bias.accel).length());
        worst_gyro = worst_gyro.max((kf.bias.gyro - true_bias.gyro).length());
    }

    let initial_accel_error = true_bias.accel.length();
    let initial_gyro_error = true_bias.gyro.length();
    println!(
        "accel bias error {initial_accel_error:.5} -> {worst_accel:.5} m/s^2, \
         gyro bias error {initial_gyro_error:.5} -> {worst_gyro:.5} rad/s"
    );

    assert!(
        // Measured: recovers to 0.7 % of the injected magnitude.
        worst_accel < initial_accel_error * 0.05,
        "accel bias did not converge: {worst_accel} vs injected {initial_accel_error}"
    );
    assert!(
        worst_gyro < initial_gyro_error * 0.05,
        "gyro bias did not converge: {worst_gyro} vs injected {initial_gyro_error}"
    );
}

/// VI-BA must also recover the trajectory itself from a perturbed start.
///
/// Unlike the visual-only case a single fixed keyframe suffices: the IMU makes
/// scale observable, which is the whole reason for fusing it.
#[test]
fn vi_ba_recovers_perturbed_poses_and_velocities() {
    let scene = inertial_scene(ImuBias::default(), 23);
    let truth_poses = scene.camera_poses.clone();
    let truth_velocities: Vec<Vec3F64> = scene.states.iter().map(|s| s.velocity).collect();

    let mut keyframes = scene.ground_truth_keyframes();
    keyframes[0].fixed = true;

    let perturbation = Perturbation {
        translation_std: 0.05,
        rotation_std: 0.01,
        velocity_std: 0.05,
        ..Default::default()
    };
    let mut rng = SimRng::new(3000);
    perturbation.apply_to_keyframes(&mut keyframes, &mut rng);

    let mut points = scene.visual.points.clone();
    Perturbation {
        point_std: 0.05,
        ..Default::default()
    }
    .apply_to_points(&mut points, &mut rng);

    let start_poses: Vec<Pose3d> = keyframes.iter().map(|k| k.pose).collect();
    let (start_t, _) = max_pose_error(&start_poses, &truth_poses);

    let imu_edges = scene.imu_factors(ImuBias::default()).expect("imu factors");
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &scene.visual.observations,
        &imu_edges,
        &scene.camera,
        &vi_ba_params(),
    )
    .expect("vi-ba");

    let end_poses: Vec<Pose3d> = result.keyframes.iter().map(|k| k.pose).collect();
    let (end_t, end_r) = max_pose_error(&end_poses, &truth_poses);

    let worst_velocity = result
        .keyframes
        .iter()
        .zip(truth_velocities.iter())
        .map(|(kf, v)| (kf.velocity - *v).length())
        .fold(0.0f64, f64::max);

    println!(
        "vi-ba poses: translation {start_t:.4} -> {end_t:.5} m, rotation {end_r:.5} rad, \
         velocity error {worst_velocity:.5} m/s"
    );

    // Measured: 2e-4 m, 3e-7 rad, 1.2e-4 m/s.
    assert!(end_t < 0.002, "translation error {end_t} m after recovery");
    assert!(end_r < 0.001, "rotation error {end_r} rad after recovery");
    assert!(
        worst_velocity < 0.002,
        "velocity error {worst_velocity} m/s after recovery"
    );
}

/// Scale is the quantity monocular BA cannot see and the IMU can. Starting from
/// a uniformly rescaled reconstruction, VI-BA must pull the scale back.
///
/// This is the ±10 % scale wobble from PR #46 posed as a controlled question:
/// given exact measurements and no frontend, can the optimizer recover scale at
/// all? A failure here localizes the problem to the back end; a pass says to
/// look elsewhere.
#[test]
fn vi_ba_recovers_scale_from_a_rescaled_start() {
    let scene = inertial_scene(ImuBias::default(), 29);
    let truth_poses = scene.camera_poses.clone();

    let scale = 1.10;
    let mut keyframes = scene.ground_truth_keyframes();
    keyframes[0].fixed = true;

    // Scale the world about the anchor keyframe's camera centre, so the fixed
    // frame stays consistent and only the *relative* geometry is wrong.
    let origin = truth_poses[0].inverse().translation;
    let rescale = |p: Vec3F64| origin + (p - origin) * scale;

    for kf in keyframes.iter_mut().skip(1) {
        let centre = kf.pose.inverse().translation;
        let moved = rescale(centre);
        // Rebuild T_cw with the same rotation and the moved camera centre.
        kf.pose = Pose3d::new(kf.pose.rotation, kf.pose.rotation * -moved);
        kf.velocity *= scale;
    }
    let points: Vec<Vec3F64> = scene.visual.points.iter().map(|p| rescale(*p)).collect();

    let start_poses: Vec<Pose3d> = keyframes.iter().map(|k| k.pose).collect();
    let (start_t, _) = max_pose_error(&start_poses, &truth_poses);
    assert!(
        start_t > 0.1,
        "rescaling did not displace anything: {start_t}"
    );

    let imu_edges = scene.imu_factors(ImuBias::default()).expect("imu factors");
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &scene.visual.observations,
        &imu_edges,
        &scene.camera,
        &vi_ba_params(),
    )
    .expect("vi-ba");

    let end_poses: Vec<Pose3d> = result.keyframes.iter().map(|k| k.pose).collect();
    let (end_t, _) = max_pose_error(&end_poses, &truth_poses);
    println!("vi-ba scale {scale}: translation {start_t:.4} -> {end_t:.5} m");

    assert!(
        // Measured: 0.32 m -> 2.4e-4 m, a factor of ~1300.
        end_t < start_t / 100.0,
        "scale not recovered: {start_t} -> {end_t} m"
    );
}
