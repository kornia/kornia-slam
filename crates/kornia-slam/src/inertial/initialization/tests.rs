use super::*;
use crate::frame::Frame;
use crate::mapping::map::{ImuFactor, MapInsertion};
use kornia_algebra::Mat3F64;
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;
use kornia_sensors::imu::{ImuCalib, ImuMeasurement, SampleInterpolation};

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
        rig: Default::default(),
    }
}

/// Integrates the TRUE (metric, gravity-aligned) trajectory's IMU signal
/// between two keyframe timestamps, with a known bias baked into the raw
/// measurement — `PreintegratedImu` is seeded with zero reference bias,
/// so it integrates the biased signal uncorrected, exactly like a real
/// biased sensor (`PreintegratedImu::integrate` subtracts `self.bias`,
/// which is zero here — see `imu.rs:140-143`).
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

fn initializer(viba_window: VibaWindow, viba_velocity_gate: f64) -> ImuInitializer {
    ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.1,
        viba_window,
        viba_velocity_gate,
    })
}

fn synth_map_with_calib(s_true: f64, r_arb: Mat3F64, calib: ImuCalib, omega: f64) -> Map {
    synth_map_filtered(s_true, r_arb, calib, omega, &|_| true, &|_, _| true)
}

/// [`synth_map_with_calib`] with only the kept keyframes, and only the edges
/// `keep_edge` admits between two kept ones.
fn synth_map_filtered(
    s_true: f64,
    r_arb: Mat3F64,
    calib: ImuCalib,
    omega: f64,
    keep_keyframe: &dyn Fn(usize) -> bool,
    keep_edge: &dyn Fn(usize, usize) -> bool,
) -> Map {
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
        if keep_keyframe(k) {
            map.insert_keyframe(Keyframe::from_frame(synth_frame(k, pose)))
                .unwrap();
        }
        if k > 0 && keep_keyframe(k - 1) && keep_keyframe(k) && keep_edge(k - 1, k) {
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
            map.apply_insertion(MapInsertion {
                imu_factors: vec![ImuFactor {
                    prev_kf_idx: k - 1,
                    curr_kf_idx: k,
                    preintegrated: pim,
                    raw_samples: Vec::new(),
                    t0: t - kf_dt,
                    t1: t,
                    bias_walk_extra_s: 0.0,
                    interpolation: SampleInterpolation::Hold,
                }],
                ..Default::default()
            })
            .unwrap();
        }
    }
    map
}

/// Checks that recovered scale remains inversely proportional to vision-map scale.
#[test]
fn viba0_scale_is_invariant_to_vision_map_scale() {
    let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix();
    let initializer = initializer(VibaWindow::All, 0.0);

    let mut products = Vec::new();
    for s_true in [0.25, 0.5, 1.0, 2.0, 4.0] {
        let map = synth_map(s_true, r_arb);
        let init = initializer
            .try_initialize(
                &map,
                Some(Pose3d::IDENTITY),
                ImuBias::default(),
                0,
                1e2,
                1e10,
                false,
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
    let initializer = initializer(VibaWindow::All, 0.0);

    // Add deterministic millimetre-scale front-end noise.
    let mut map = synth_map_with_calib(0.5, r_arb, euroc_calib, WEAK_YAW_RATE);
    map.edit_keyframes_for_test(|k, kf| {
        let f = k as f64;
        let jitter = Vec3F64::new(
            (3.7 * f).sin(),
            (5.1 * f + 1.3).sin(),
            (2.3 * f + 0.7).sin(),
        ) * 2e-3;
        let cam_to_world = kf.frame.pose_world_to_cam.inverse();
        kf.frame.pose_world_to_cam =
            Pose3d::new(cam_to_world.rotation, cam_to_world.translation + jitter).inverse();
    });

    let viba0 = initializer
        .try_initialize(
            &map,
            Some(Pose3d::IDENTITY),
            ImuBias::default(),
            0,
            1e2,
            1e10,
            false,
        )
        .expect("VIBA0 should solve");
    let mut bias = ImuBias::default();
    let mut gravity_world = Vec3F64::ZERO;
    initializer
        .apply_initialization(&mut map, &mut bias, &mut gravity_world, viba0, 0)
        .expect("VIBA0 alignment should be accepted");

    let solve_viba2 = |prior_a: f64| -> Vec3F64 {
        initializer
            .try_initialize(
                &map.clone(),
                Some(Pose3d::IDENTITY),
                bias,
                0,
                0.0,
                prior_a,
                true,
            )
            .map(|init| init.bias.accel)
            // Map a plausibility-gate rejection to its threshold for comparison.
            .unwrap_or(Vec3F64::new(0.0, 1.0, 0.0))
    };

    let ba_with_prior = solve_viba2(1e5);
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
        map.insert_keyframe(Keyframe::from_frame(synth_frame(k, pose)))
            .unwrap();

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
            map.apply_insertion(MapInsertion {
                imu_factors: vec![ImuFactor {
                    prev_kf_idx: k - 1,
                    curr_kf_idx: k,
                    preintegrated: pim,
                    raw_samples: Vec::new(),
                    t0: t - kf_dt,
                    t1: t,
                    bias_walk_extra_s: 0.0,
                    interpolation: SampleInterpolation::Hold,
                }],
                ..Default::default()
            })
            .unwrap();
        }
    }

    let initializer = initializer(VibaWindow::All, 0.0);

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
            Some(Pose3d::IDENTITY),
            ImuBias::default(),
            0,
            1e2,
            1e10,
            false,
        )
        .expect("VIBA0 pass should recover a rough solution from clean synthetic data");

    // Capture VIBA0's scale/gravity before it's consumed by apply_initialization
    // (needed below to reconstruct what frame the second call's raw output —
    // scale correction, velocities — is expressed relative to).
    let viba0_scale = viba0.scale;
    let rwg_viba0 =
        rotation_from_to(viba0.gravity_world.normalize(), Vec3F64::new(0.0, 1.0, 0.0)).matrix();

    let mut bias = ImuBias::default();
    let mut gravity_world = Vec3F64::ZERO;
    initializer
        .apply_initialization(&mut map, &mut bias, &mut gravity_world, viba0, 0)
        .expect("VIBA0 alignment should be accepted");

    let result = initializer
        .try_initialize(&map, Some(Pose3d::IDENTITY), bias, 0, 0.0, 0.0, true)
        .expect("VIBA1 (loosened-prior) pass should refine to the true solution");

    // `result.scale` is only the *residual* correction on top of what
    // VIBA0 already applied to the map (apply_initialization composes
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

    // apply_initialization already rotated the map so gravity sits at
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
    // by a subsequent apply_initialization — not folded back into
    // `result.keyframe_velocities` here).
    for (k, kv) in result.keyframe_velocities[..n_keyframes].iter().enumerate() {
        assert_eq!(kv.keyframe_idx, k);
        let velocity = kv.velocity_world;
        let t = k as f64 * kf_dt;
        let (_, v_true, _, _) = circular_trajectory(t, OMEGA);
        let expected_v = rwg_viba0 * ((r_arb * v_true) * s_true * viba0_scale);
        let err = (velocity - expected_v).length();
        assert!(
            err < 0.15,
            "velocity[{k}]: got {velocity:?}, want {expected_v:?}",
        );
    }
}

/// Anchors carry no IMU edge, so the initialization window skips them: a
/// map with one gives the solve it gives without it.
#[test]
fn an_anchor_leaves_the_initialization_unchanged() {
    let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix();
    let initializer = initializer(VibaWindow::All, 0.0);
    let solve = |map: &Map| {
        initializer
            .try_initialize(
                map,
                Some(Pose3d::IDENTITY),
                ImuBias::default(),
                0,
                1e2,
                1e10,
                false,
            )
            .expect("VIBA0 solves the synthetic window")
    };
    let map = synth_map(1.0, r_arb);
    let mut with_anchor = map.clone();
    let mut anchor = Keyframe::from_frame(synth_frame(1000, Pose3d::IDENTITY));
    anchor.anchor = true;
    anchor.velocity_world = Vec3F64::new(40.0, 0.0, 0.0);
    with_anchor.insert_keyframe(anchor).unwrap();
    assert!(initializer.ready(&with_anchor, Some(0)) == initializer.ready(&map, Some(0)));
    assert_eq!(
        format!("{:?}", solve(&with_anchor)),
        format!("{:?}", solve(&map))
    );
}

fn calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1e-3,
        accel_noise: 1e-2,
        gyro_bias_noise: 1e-5,
        accel_bias_noise: 1e-4,
    }
}

fn r_arb() -> Mat3F64 {
    SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix()
}

fn viba0(initializer: &ImuInitializer, map: &Map) -> Option<ImuInitResult> {
    initializer.try_initialize(
        map,
        Some(Pose3d::IDENTITY),
        ImuBias::default(),
        0,
        1e2,
        1e10,
        false,
    )
}

/// Moves keyframe `idx`'s camera centre by `delta`, keeping its rotation.
fn shift_centre(map: &mut Map, idx: usize, delta: Vec3F64) {
    map.edit_keyframes_for_test(|_, kf| {
        if kf.frame.idx == idx {
            let cam_to_world = kf.frame.pose_world_to_cam.inverse();
            kf.frame.pose_world_to_cam =
                Pose3d::new(cam_to_world.rotation, cam_to_world.translation + delta).inverse();
        }
    });
}

/// Marks the keyframes `lost` admits as inserted while lost, with a sentinel
/// velocity per keyframe.
fn mark_lost(map: &mut Map, lost: impl Fn(usize) -> bool) {
    map.edit_keyframes_for_test(|_, kf| {
        if lost(kf.frame.idx) {
            kf.inserted_while_lost = true;
            kf.velocity_world = sentinel(kf.frame.idx);
        }
    });
}

fn sentinel(idx: usize) -> Vec3F64 {
    Vec3F64::new(7.0, 8.0, 9.0 + idx as f64)
}

const CORRUPT_KF: usize = 15;
fn corrupt_shift() -> Vec3F64 {
    Vec3F64::new(0.05, 0.0, 0.03)
}

/// With the defaults, or with knobs that change nothing on this map, the
/// solve is today's to the last bit.
#[test]
fn the_default_window_and_gate_leave_the_solve_unchanged() {
    let mut map = synth_map(0.5, r_arb());
    shift_centre(&mut map, CORRUPT_KF, corrupt_shift());
    let solve = |window, gate| format!("{:?}", viba0(&initializer(window, gate), &map).unwrap());
    let default = solve(VibaWindow::All, 0.0);
    assert_eq!(solve(VibaWindow::VisualOnly, 0.0), default);
    assert_eq!(solve(VibaWindow::All, 1e300), default);
}

#[test]
fn visual_only_solves_as_if_the_lost_keyframe_were_absent() {
    let mut lost = synth_map(0.5, r_arb());
    mark_lost(&mut lost, |k| k == 12);
    let absent = synth_map_filtered(0.5, r_arb(), calib(), OMEGA, &|k| k != 12, &|_, _| true);
    let visual_only = initializer(VibaWindow::VisualOnly, 0.0);
    let all = initializer(VibaWindow::All, 0.0);

    let init = viba0(&visual_only, &lost).unwrap();
    assert_eq!(
        format!("{init:?}"),
        format!("{:?}", viba0(&all, &absent).unwrap())
    );
    assert_eq!(
        init.keyframe_velocities
            .iter()
            .map(|kv| kv.keyframe_idx)
            .collect::<Vec<_>>(),
        (0..30).filter(|&k| k != 12).collect::<Vec<_>>()
    );
    assert_eq!(
        visual_only.ready(&lost, Some(0)),
        all.ready(&absent, Some(0))
    );
}

/// Lost keyframes keep their velocity and bias, and tracking still resumes at
/// the newest window keyframe, lost or not.
#[test]
fn visual_only_writes_velocities_by_keyframe_and_resumes_at_the_newest() {
    let mut map = synth_map(0.5, r_arb());
    mark_lost(&mut map, |k| k == 12 || k == 29);
    let pose_12 = map.get_keyframe(12).unwrap().frame.pose_world_to_cam;
    let initializer = initializer(VibaWindow::VisualOnly, 0.0);
    let init = viba0(&initializer, &map).unwrap();
    let listed = init.keyframe_velocities.clone();
    let rotation = rotation_from_to(
        init.gravity_world / init.gravity_world.length(),
        Vec3F64::new(0.0, 1.0, 0.0),
    );

    let mut bias = ImuBias::default();
    let mut gravity_world = Vec3F64::ZERO;
    let resume = initializer
        .apply_initialization(&mut map, &mut bias, &mut gravity_world, init, 0)
        .unwrap()
        .unwrap();

    assert_eq!(listed.len(), 28);
    for kv in &listed {
        assert!(kv.keyframe_idx != 12 && kv.keyframe_idx != 29);
        let written = map.get_keyframe(kv.keyframe_idx).unwrap().velocity_world;
        assert!(
            (written - rotation * kv.velocity_world).length() < 1e-9,
            "keyframe {}",
            kv.keyframe_idx
        );
    }
    for k in [12, 29] {
        assert_eq!(map.get_keyframe(k).unwrap().velocity_world, sentinel(k));
    }
    assert!(
        format!(
            "{:?}",
            map.get_keyframe(12).unwrap().frame.pose_world_to_cam
        ) != format!("{pose_12:?}")
    );
    let newest = map.get_keyframe(29).unwrap();
    assert_eq!(
        format!("{:?}", resume.pose_world_to_cam),
        format!("{:?}", newest.frame.pose_world_to_cam)
    );
    assert_eq!(resume.velocity_world, sentinel(29));
}

#[test]
fn visual_only_does_not_count_lost_keyframes() {
    let mut map = synth_map(0.5, r_arb());
    mark_lost(&mut map, |k| k >= 9);
    let visual_only = initializer(VibaWindow::VisualOnly, 0.0);
    assert!(!visual_only.ready(&map, Some(0)));
    assert!(viba0(&visual_only, &map).is_none());
    assert!(initializer(VibaWindow::All, 0.0).ready(&map, Some(0)));
}

fn gravity_angle(init: &ImuInitResult) -> f64 {
    let want = r_arb() * Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
    init.gravity_world
        .normalize()
        .dot(want.normalize())
        .clamp(-1.0, 1.0)
        .acos()
}

#[test]
fn the_velocity_gate_drops_the_edges_of_a_corrupted_pose() {
    let mut map = synth_map(1.0, r_arb());
    shift_centre(&mut map, CORRUPT_KF, corrupt_shift());
    let gated = viba0(&initializer(VibaWindow::All, 10.0), &map).unwrap();
    let ungated = viba0(&initializer(VibaWindow::All, 0.0), &map).unwrap();
    assert!(
        gated.gated_edges.contains(&(14, 15)),
        "{:?}",
        gated.gated_edges
    );
    assert!(
        gated.gated_edges.contains(&(15, 16)),
        "{:?}",
        gated.gated_edges
    );
    assert!(gated.gated_edges.len() <= 7);
    assert!(
        gravity_angle(&gated) < gravity_angle(&ungated),
        "gated {} vs ungated {}",
        gravity_angle(&gated),
        gravity_angle(&ungated)
    );

    // VIBA0 seeds velocities and gravity from every window edge, so the
    // gated equality is checked on a refinement, whose seed is the map's.
    let aligned = |mut map: Map| {
        let (mut bias, mut gravity_world) = (ImuBias::default(), Vec3F64::ZERO);
        initializer(VibaWindow::All, 0.0)
            .apply_initialization(&mut map, &mut bias, &mut gravity_world, ungated.clone(), 0)
            .unwrap();
        (map, bias)
    };
    let refine = |gate, (map, bias): &(Map, ImuBias)| {
        initializer(VibaWindow::All, gate)
            .try_initialize(map, Some(Pose3d::IDENTITY), *bias, 0, 0.0, 0.0, true)
            .unwrap()
    };
    let gated = refine(10.0, &aligned(map));
    assert!(
        gated.gated_edges.contains(&(14, 15)),
        "{:?}",
        gated.gated_edges
    );
    assert!(
        gated.gated_edges.contains(&(15, 16)),
        "{:?}",
        gated.gated_edges
    );
    let gated_pairs = gated.gated_edges.clone();
    let mut without = synth_map_filtered(1.0, r_arb(), calib(), OMEGA, &|_| true, &|a, b| {
        !gated_pairs.contains(&(a, b))
    });
    shift_centre(&mut without, CORRUPT_KF, corrupt_shift());
    let expected = refine(0.0, &aligned(without));
    let solved = |init: &ImuInitResult| {
        format!(
            "{:?}",
            (
                init.scale,
                init.gravity_world,
                init.bias,
                &init.keyframe_velocities
            )
        )
    };
    assert_eq!(solved(&gated), solved(&expected));
}
