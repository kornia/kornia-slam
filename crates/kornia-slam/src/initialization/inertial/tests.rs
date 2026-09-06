use super::*;
use crate::frame::Frame;
use crate::map::{InertialAlignment, Keyframe, Map};
use crate::pose_conversion::rotation_from_to;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuCalib, ImuMeasurement};

/// Test-side stand-in for `SlamSystem::apply_inertial_initialization`: the
/// map-side alignment plus the two pieces of system state these tests read
/// back afterwards (the bias estimate and the canonical gravity direction).
fn apply_viba(map: &mut Map, bias: &mut ImuBias, gravity_world: &mut Vec3F64, init: ImuInitResult) {
    let rotation = rotation_from_to(init.gravity_world.normalize(), Vec3F64::new(0.0, 1.0, 0.0));
    map.apply_inertial_alignment(InertialAlignment {
        scale: init.scale,
        rotation,
        keyframe_velocities: init.keyframe_velocities,
        bias: init.bias,
    })
    .expect("VIBA0 result should apply");
    *bias = init.bias;
    *gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
}

fn request(stage: InertialStage, seed: RwgSeed, bias: ImuBias) -> InertialInitRequest {
    InertialInitRequest {
        start_kf_idx: 0,
        imu_t_bc: Pose3d::IDENTITY,
        bias,
        stage,
        seed,
    }
}

const RADIUS: f64 = 2.0;
const OMEGA: f64 = 0.5;
// Weak rotation makes acceleration bias less observable.
const WEAK_YAW_RATE: f64 = 0.2;

/// True trajectory: circular translation *plus* the body yawing in sync
/// with it (like a vehicle banking into the turn) — needed so `R_wb(t)`
/// actually varies across keyframes. Without body rotation, a constant
/// accel-bias offset and a wrong scale/gravity estimate are genuinely
/// indistinguishable from the data (classic VIO init degeneracy); gyro
/// bias stays observable either way since it only needs rotation-only
/// consistency, which is why `bg` recovered correctly even with the
/// bug below.
fn circular_trajectory(t: f64, omega: f64) -> (Vec3F64, Vec3F64, Vec3F64, Mat3F64) {
    let theta = omega * t;
    let (s, c) = theta.sin_cos();
    let p = Vec3F64::new(RADIUS * c, 0.0, RADIUS * s);
    let v = Vec3F64::new(-RADIUS * omega * s, 0.0, RADIUS * omega * c);
    let a = Vec3F64::new(
        -RADIUS * omega * omega * c,
        0.0,
        -RADIUS * omega * omega * s,
    );
    // Rotation about world Y keeps angular velocity (0, omega, 0) in either frame.
    let r_wb = SO3F64::exp(Vec3F64::new(0.0, theta, 0.0)).matrix();
    (p, v, a, r_wb)
}

/// Builds the *corrupted* vision-frame keyframe pose: true trajectory
/// scaled by `s_true` and rotated by an arbitrary, non-gravity-aligned
/// `r_arb` — models what a monocular front-end actually hands the
/// initializer (unknown scale, unknown relation to gravity).
fn synth_pose_world_to_cam(
    r_arb: Mat3F64,
    s_true: f64,
    p_true: Vec3F64,
    r_wb_true: Mat3F64,
) -> Pose3d {
    let vision_rotation = r_arb * r_wb_true;
    let vision_position = (r_arb * p_true) * s_true;
    let cam_to_world = Pose3d::new(vision_rotation, vision_position);
    cam_to_world.inverse()
}

fn synth_frame(idx: usize, pose_world_to_cam: Pose3d) -> Frame {
    Frame {
        idx,
        features: OrbFeatures {
            keypoints_xy: vec![],
            orientations: vec![],
            descriptors: vec![],
            octaves: vec![],
        },
        pose_world_to_cam,
        image_size: ImageSize {
            width: 640,
            height: 480,
        },
        keypoint_colors: vec![],
        u_right: vec![],
        depth: vec![], // empty => is_stereo() == false => monocular path
        keypoints_undist: vec![],
    }
}

/// Integrates the TRUE (metric, gravity-aligned) trajectory's IMU signal
/// between two keyframe timestamps, with a known bias baked into the raw
/// measurement — `PreintegratedImu` is seeded with zero reference bias,
/// so it integrates the biased signal uncorrected, exactly like a real
/// biased sensor (`PreintegratedImu::integrate` subtracts `self.bias`,
/// which is zero here — see `kornia-sensors/src/imu.rs`).
#[allow(clippy::too_many_arguments)]
fn integrate_true_imu(
    t0: f64,
    t1: f64,
    imu_rate_hz: f64,
    omega: f64,
    gravity_true: Vec3F64,
    bias_gyro_true: Vec3F64,
    bias_accel_true: Vec3F64,
    calib: ImuCalib,
) -> kornia_sensors::imu::PreintegratedImu {
    let mut pim = kornia_sensors::imu::PreintegratedImu::new(ImuBias::default(), calib);
    let dt = 1.0 / imu_rate_hz;
    let mut t = t0;
    while t < t1 - 1e-9 {
        let (_, _, a_true, r_wb_true) = circular_trajectory(t + 0.5 * dt, omega);
        let gyro_meas = Vec3F64::new(0.0, omega, 0.0) + bias_gyro_true; // true body-frame ω + bias
        let accel_meas = r_wb_true.transpose() * (a_true - gravity_true) + bias_accel_true; // specific force in body frame + bias
        pim.integrate(
            &ImuMeasurement {
                timestamp: t,
                gyro: gyro_meas,
                accel: accel_meas,
            },
            dt,
        );
        t += dt;
    }
    pim
}

fn synth_map(s_true: f64, r_arb: Mat3F64) -> Map {
    synth_map_with_calib(
        s_true,
        r_arb,
        ImuCalib {
            gyro_noise: 1e-3,
            accel_noise: 1e-2,
            gyro_bias_noise: 1e-5,
            accel_bias_noise: 1e-4,
        },
        OMEGA,
    )
}

fn synth_map_with_calib(s_true: f64, r_arb: Mat3F64, calib: ImuCalib, omega: f64) -> Map {
    let bias_gyro_true = Vec3F64::new(0.01, -0.02, 0.005);
    let bias_accel_true = Vec3F64::new(0.05, -0.03, 0.02);
    let gravity_true = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
    let n_keyframes = 30;
    let kf_dt = 0.2;
    let imu_rate = 200.0;

    let mut map = Map::new();
    for k in 0..n_keyframes {
        let t = k as f64 * kf_dt;
        let (p_true, _, _, r_wb_true) = circular_trajectory(t, omega);
        let pose = synth_pose_world_to_cam(r_arb, s_true, p_true, r_wb_true);
        map.upsert_keyframe(Keyframe::from_frame(synth_frame(k, pose)));
        if k > 0 {
            let pim = integrate_true_imu(
                t - kf_dt,
                t,
                imu_rate,
                omega,
                gravity_true,
                bias_gyro_true,
                bias_accel_true,
                calib,
            );
            map.add_imu_factor(k - 1, k, pim, Vec::new(), t - kf_dt, t);
        }
    }
    map
}

#[test]
fn formats_compact_imu_init_gate() {
    let not_ready = ImuInitNotReady {
        start_kf_idx: 12,
        first_kf_idx: Some(12),
        last_kf_idx: Some(32),
        keyframes: 7,
        min_keyframes: 10,
        imu_time_sec: 1.05,
        min_time_sec: 1.0,
        motion: 0.0,
        min_motion: 0.05,
        reason: ImuInitNotReadyReason::Keyframes,
    };

    assert_eq!(
        not_ready.to_string(),
        "[imu_init_gate] start_idx=12 first_idx=Some(12) last_idx=Some(32) kfs=7/10 imu_time=1.05/1.0s"
    );
}

#[test]
fn readiness_reports_the_gate_that_is_short() {
    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.1,
        ..ImuInitConfig::default()
    });

    assert_eq!(
        initializer.readiness(&Map::new(), None).unwrap_err().reason,
        ImuInitNotReadyReason::NoWindow
    );

    let mut map = Map::new();
    map.upsert_keyframe(Keyframe::from_frame(synth_frame(0, Pose3d::IDENTITY)));
    let not_ready = initializer.readiness(&map, Some(0)).unwrap_err();
    assert_eq!(not_ready.reason, ImuInitNotReadyReason::Keyframes);
    assert_eq!(not_ready.keyframes, 1);
    // Monocular window: the stereo threshold is doubled.
    assert!((not_ready.min_time_sec - 2.0).abs() < 1e-12);

    let full = synth_map(1.0, Mat3F64::IDENTITY);
    assert!(initializer.readiness(&full, Some(0)).is_ok());
}

/// The extrinsics check now lives at the caller (the request carries a
/// non-optional `imu_t_bc`); the variant stays part of the public reject
/// enum so consumers keep matching on it.
#[test]
fn missing_extrinsics_has_a_typed_rejection() {
    assert_eq!(
        ImuInitRejectReason::MissingExtrinsics.to_string(),
        "camera-to-body IMU extrinsics are missing"
    );
}

#[test]
fn stage_priors_follow_orb_slam3() {
    assert_eq!(
        BiasPriors::for_stage(InertialStage::Viba0, true),
        BiasPriors {
            gyro: 1e2,
            accel: 1e10
        }
    );
    assert_eq!(
        BiasPriors::for_stage(InertialStage::Viba0, false),
        BiasPriors {
            gyro: 1e2,
            accel: 1e5
        }
    );
    assert_eq!(
        BiasPriors::for_stage(InertialStage::Viba1, true),
        BiasPriors {
            gyro: 1.0,
            accel: VIBA_PRIOR_A
        }
    );
    // VIBA2 relaxes only the gyro prior (kornia-slam#51).
    assert_eq!(
        BiasPriors::for_stage(InertialStage::Viba2, true),
        BiasPriors {
            gyro: 0.0,
            accel: VIBA_PRIOR_A
        }
    );
}

#[test]
fn invalid_configuration_has_a_typed_rejection() {
    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 1,
        min_time_sec: 0.0,
        min_motion: 0.0,
        ..ImuInitConfig::default()
    });

    let result = initializer.try_initialize(
        &Map::new(),
        &request(
            InertialStage::Viba0,
            RwgSeed::FromVisualTrajectory,
            ImuBias::default(),
        ),
    );

    assert!(matches!(result, Err(ImuInitRejectReason::InvalidConfig(_))));
}

#[test]
fn insufficient_keyframes_reports_found_and_required_counts() {
    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 3,
        min_time_sec: 0.0,
        min_motion: 0.0,
        ..ImuInitConfig::default()
    });
    let mut map = Map::new();
    map.upsert_keyframe(Keyframe::from_frame(synth_frame(7, Pose3d::IDENTITY)));

    let result = initializer.try_initialize(
        &map,
        &request(
            InertialStage::Viba0,
            RwgSeed::FromVisualTrajectory,
            ImuBias::default(),
        ),
    );

    assert!(matches!(
        result,
        Err(ImuInitRejectReason::InsufficientKeyframes {
            found: 1,
            required: 3
        })
    ));
}

/// Checks that recovered scale remains inversely proportional to vision-map scale.
#[test]
fn viba0_scale_is_invariant_to_vision_map_scale() {
    let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix();
    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.1,
        ..ImuInitConfig::default()
    });

    let mut products = Vec::new();
    for s_true in [0.25, 0.5, 1.0, 2.0, 4.0] {
        let map = synth_map(s_true, r_arb);
        let init = initializer
            .try_initialize(
                &map,
                &request(
                    InertialStage::Viba0,
                    RwgSeed::FromVisualTrajectory,
                    ImuBias::default(),
                ),
            )
            .expect("VIBA0 should solve on clean synthetic data at any map scale");
        println!(
            "[scale-invariance] s_true={s_true:<5} s_recovered={:<10.5} product={:.5}",
            init.scale,
            init.scale * s_true
        );
        products.push(init.scale * s_true);
    }

    let mean = products.iter().sum::<f64>() / products.len() as f64;
    let max_rel_dev = products
        .iter()
        .map(|p| (p - mean).abs() / mean)
        .fold(0.0f64, f64::max);
    assert!(
        max_rel_dev < 0.02,
        "VIBA0 scale does not track map size: products {products:?} (mean {mean:.5}, \
         max deviation {:.1}%)",
        max_rel_dev * 100.0,
    );
}

/// Ensures the VIBA2 prior keeps acceleration bias bounded when fixed poses
/// disagree with the IMU (kornia-slam#51).
#[test]
fn viba2_accel_bias_stays_bounded_under_pose_inconsistency() {
    let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix();
    let bias_accel_true = Vec3F64::new(0.05, -0.03, 0.02);
    // Realistic noise makes the synthetic pose inconsistency representative.
    let euroc_calib = ImuCalib {
        gyro_noise: 1.6968e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9393e-5,
        accel_bias_noise: 3.0e-3,
    };
    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.1,
        ..ImuInitConfig::default()
    });

    // Add deterministic millimetre-scale front-end noise.
    let mut map = synth_map_with_calib(0.5, r_arb, euroc_calib, WEAK_YAW_RATE);
    for (k, kf) in map.keyframes_mut().iter_mut().enumerate() {
        let f = k as f64;
        let jitter = Vec3F64::new(
            (3.7 * f).sin(),
            (5.1 * f + 1.3).sin(),
            (2.3 * f + 0.7).sin(),
        ) * 2e-3;
        let cam_to_world = kf.frame.pose_world_to_cam.inverse();
        kf.frame.pose_world_to_cam =
            Pose3d::new(cam_to_world.rotation, cam_to_world.translation + jitter).inverse();
    }

    let viba0 = initializer
        .try_initialize(
            &map,
            &request(
                InertialStage::Viba0,
                RwgSeed::FromVisualTrajectory,
                ImuBias::default(),
            ),
        )
        .expect("VIBA0 should solve");
    let mut bias = ImuBias::default();
    let mut gravity_world = Vec3F64::ZERO;
    apply_viba(&mut map, &mut bias, &mut gravity_world, viba0);

    let solve_viba2 = |prior_a: f64| -> Vec3F64 {
        initializer
            .solve(
                &map.clone(),
                &request(
                    InertialStage::Viba2,
                    RwgSeed::FromCurrentGravity(gravity_world),
                    bias,
                ),
                Some(BiasPriors {
                    gyro: 0.0,
                    accel: prior_a,
                }),
            )
            .map(|init| init.bias.accel)
            // Map a plausibility-gate rejection to its threshold for comparison.
            .unwrap_or(Vec3F64::new(0.0, 1.0, 0.0))
    };

    let ba_with_prior = solve_viba2(VIBA_PRIOR_A);
    let ba_no_prior = solve_viba2(0.0);
    let err_with_prior = (ba_with_prior - bias_accel_true).length();
    let err_no_prior = (ba_no_prior - bias_accel_true).length();
    println!(
        "[viba2] prior_a=1e5 ba=({:+.4},{:+.4},{:+.4}) err={err_with_prior:.4}\n\
         [viba2] prior_a=0   ba=({:+.4},{:+.4},{:+.4}) err={err_no_prior:.4}",
        ba_with_prior.x,
        ba_with_prior.y,
        ba_with_prior.z,
        ba_no_prior.x,
        ba_no_prior.y,
        ba_no_prior.z,
    );

    assert!(
        err_with_prior < 0.1,
        "accel bias should stay near truth with the prior retained: \
         got {ba_with_prior:?}, want ~{bias_accel_true:?}"
    );
    assert!(
        err_no_prior > err_with_prior,
        "dropping the accel-bias prior is what lets the bias run away; if this \
         ever stops holding, the poses have become IMU-consistent (e.g. a \
         FullInertialBA equivalent landed) and `VIBA_PRIOR_A` can be revisited. \
         with prior: {ba_with_prior:?} ({err_with_prior:.4}), \
         without: {ba_no_prior:?} ({err_no_prior:.4})"
    );
}

#[test]
fn recovers_scale_bias_gravity_from_synthetic_trajectory() {
    let s_true = 0.5; // vision map is half true metric size
    let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix(); // arbitrary, non-gravity-aligned
    let bias_gyro_true = Vec3F64::new(0.01, -0.02, 0.005);
    let bias_accel_true = Vec3F64::new(0.05, -0.03, 0.02);
    let gravity_true = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0); // kornia's Y-down convention

    let calib = ImuCalib {
        gyro_noise: 1e-3,
        accel_noise: 1e-2,
        gyro_bias_noise: 1e-5,
        accel_bias_noise: 1e-4,
    };

    let n_keyframes = 30;
    let kf_dt = 0.2; // 5 Hz keyframes -> 5.8s window, ~166° of yaw excitation
    let imu_rate = 200.0; // Hz

    let mut map = Map::new();
    for k in 0..n_keyframes {
        let t = k as f64 * kf_dt;
        let (p_true, _, _, r_wb_true) = circular_trajectory(t, OMEGA);
        let pose = synth_pose_world_to_cam(r_arb, s_true, p_true, r_wb_true);
        map.upsert_keyframe(Keyframe::from_frame(synth_frame(k, pose)));

        if k > 0 {
            let pim = integrate_true_imu(
                t - kf_dt,
                t,
                imu_rate,
                OMEGA,
                gravity_true,
                bias_gyro_true,
                bias_accel_true,
                calib,
            );
            map.add_imu_factor(k - 1, k, pim, Vec::new(), t - kf_dt, t);
        }
    }

    let initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.1,
        ..ImuInitConfig::default()
    });

    // `try_initialize` now does exactly one joint solve per call (mirrors
    // ORB-SLAM3's `InitializeIMU`/`InertialOptimization` one-call-per-
    // invocation structure). Reproduce the VIBA0 -> apply -> VIBA1
    // (loosened-prior refinement) schedule explicitly here: VIBA0 alone
    // deliberately suppresses accel bias with a huge prior_a to avoid the
    // scale/accel-bias degeneracy on a short/early window, which also
    // biases scale by several percent even on clean synthetic data.
    let viba0 = initializer
        .try_initialize(
            &map,
            &request(
                InertialStage::Viba0,
                RwgSeed::FromVisualTrajectory,
                ImuBias::default(),
            ),
        )
        .expect("VIBA0 pass should recover a rough solution from clean synthetic data");

    // Capture VIBA0's scale/gravity before it's consumed by application
    // (needed below to reconstruct what frame the second call's raw output —
    // scale correction, velocities — is expressed relative to).
    let viba0_scale = viba0.scale;
    let rwg_viba0 =
        rotation_from_to(viba0.gravity_world.normalize(), Vec3F64::new(0.0, 1.0, 0.0)).matrix();

    let mut bias = ImuBias::default();
    let mut gravity_world = Vec3F64::ZERO;
    apply_viba(&mut map, &mut bias, &mut gravity_world, viba0);

    let result = initializer
        .solve(
            &map,
            &request(
                InertialStage::Viba2,
                RwgSeed::FromCurrentGravity(gravity_world),
                bias,
            ),
            // The un-regularized refinement this assertion set was written
            // against; production VIBA2 keeps `VIBA_PRIOR_A` (see
            // `BiasPriors::for_stage`).
            Some(BiasPriors {
                gyro: 0.0,
                accel: 0.0,
            }),
        )
        .expect("VIBA1 (loosened-prior) pass should refine to the true solution");

    // `result.scale` is only the *residual* correction on top of what
    // VIBA0 already applied to the map (system application composes
    // scale/rotation multiplicatively, mirroring ORB-SLAM3's
    // ApplyScaledRotation) — compare the cumulative effect, not the raw
    // second-call output in isolation.
    let total_scale = viba0_scale * result.scale;
    assert!(
        (total_scale - 1.0 / s_true).abs() < 0.05,
        "cumulative scale: got {:.4} (viba0={:.4} * refine={:.4}), want {:.4}",
        total_scale,
        viba0_scale,
        result.scale,
        1.0 / s_true,
    );
    assert!(
        (result.bias.gyro - bias_gyro_true).length() < 0.01,
        "gyro bias: got {:?}, want {:?}",
        result.bias.gyro,
        bias_gyro_true,
    );
    assert!(
        (result.bias.accel - bias_accel_true).length() < 0.01,
        "accel bias: got {:?}, want {:?}",
        result.bias.accel,
        bias_accel_true,
    );

    // System application already rotated the map so gravity sits at
    // kornia-slam's own canonical (0,+G,0) — the second call's residual
    // gravity-direction correction should converge there too, not to the
    // original (pre-rotation) r_arb-relative direction.
    assert!(
        result
            .gravity_world
            .normalize()
            .dot(Vec3F64::new(0.0, 1.0, 0.0))
            > 0.99,
        "gravity direction: got {:?}, want ~(0,1,0)",
        result.gravity_world.normalize(),
    );

    // Velocity is expressed in the frame at the time of this second
    // solve: after VIBA0's position scale/rotation was applied to the
    // map, but before this call's own residual scale correction (that
    // correction is only reflected in `result.scale`, applied to the map
    // by subsequent system application — not folded back into the
    // returned keyframe velocities here).
    for k in 0..n_keyframes {
        let t = k as f64 * kf_dt;
        let (_, v_true, _, _) = circular_trajectory(t, OMEGA);
        let expected_v = rwg_viba0 * ((r_arb * v_true) * s_true * viba0_scale);
        let assignment = result
            .keyframe_velocities
            .iter()
            .find(|assignment| assignment.keyframe_idx == k)
            .expect("every initialized keyframe should have a velocity");
        let err = (assignment.velocity_world - expected_v).length();
        assert!(
            err < 0.15,
            "velocity[{k}]: got {:?}, want {:?}",
            assignment.velocity_world,
            expected_v
        );
    }
}
