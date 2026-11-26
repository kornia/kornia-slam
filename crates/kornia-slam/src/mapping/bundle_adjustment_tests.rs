use super::*;
use crate::mapping::map::tests::test_frame;
use crate::mapping::map::{BaOutcome, ImuFactor, LandmarkSeed, MapInsertion, ObservationKey};
use kornia_algebra::{Mat3F64, SO3F64};
use kornia_sensors::imu::{
    ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu, SampleInterpolation,
};

const KF_IDS: [usize; 5] = [80, 10, 90, 30, 120];

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

/// Noncontiguous, unsorted frame IDs, a retired landmark slot, and an older
/// landmark outside the active window make index/selection mistakes observable.
fn fixture() -> Map {
    let points: Vec<_> = (0..12)
        .map(|i| {
            Vec3F64::new(
                (i % 4) as f64 * 0.4 - 0.6,
                (i / 4) as f64 * 0.3 - 0.3,
                4.0 + (i % 3) as f64 * 0.5,
            )
        })
        .collect();
    let mut map = Map::new();
    for (slot, id) in KF_IDS.into_iter().enumerate() {
        let mut frame = test_frame(id, vec![[0; 32]; 14]);
        let tx = -0.2 * slot as f64;
        for (feature, point) in points.iter().enumerate() {
            frame.features.keypoints_xy[feature] = [
                (200.0 * (point.x + tx) / point.z + 320.0) as f32,
                (200.0 * point.y / point.z + 240.0) as f32,
            ];
        }
        frame.pose_world_to_cam.translation.x = tx;
        if slot >= 2 {
            frame.pose_world_to_cam.translation.y = 0.03;
        }
        map.insert_keyframe(Keyframe::from_frame(frame)).unwrap();
    }
    let seed = |feature_idx, position| LandmarkSeed {
        position,
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx: KF_IDS[0],
            feature_idx,
        },
    };
    let retired = map
        .insert_landmark(seed(13, Vec3F64::new(0.0, 0.0, 5.0)))
        .unwrap();
    map.remove_landmark(retired).unwrap();
    for (feature, point) in points.into_iter().enumerate() {
        let id = map
            .insert_landmark(seed(feature, point + Vec3F64::new(0.02, -0.01, 0.04)))
            .unwrap();
        for kf in &KF_IDS[1..] {
            map.link_observation(*kf, feature, id).unwrap();
        }
    }
    map.insert_landmark(seed(12, Vec3F64::new(1.0, 1.0, 7.0)))
        .unwrap();
    map
}

#[test]
fn local_visual_ba_preserves_fixed_entities_and_solves_outside_the_live_map() {
    let mut map = fixture();
    let before = map.clone();
    let update = run_local_ba(map.ba_snapshot(&local_window(&map)), &camera());
    assert_eq!(
        map.state_fingerprint_for_test(),
        before.state_fingerprint_for_test()
    );
    let result = map.apply_ba_update(update).unwrap();
    assert!(
        !result.keyframe_corrections.is_empty(),
        "the solver actually moved poses"
    );
    assert!(result.map_points_updated > 0);
    for id in &KF_IDS[..2] {
        assert_eq!(
            map.get_keyframe(*id).unwrap().frame.pose_world_to_cam,
            before.get_keyframe(*id).unwrap().frame.pose_world_to_cam
        );
    }
    assert!(
        result
            .keyframe_corrections
            .iter()
            .all(|c| KF_IDS[2..].contains(&c.kf_idx))
    );
    assert_eq!(
        map.map_points()[0].position,
        before.map_points()[0].position
    );
    assert!(map.map_points()[0].culled);
    assert_eq!(
        map.map_points().last().unwrap().position,
        before.map_points().last().unwrap().position
    );
}

#[test]
fn local_window_holds_active_keyframes_their_landmarks_and_fixed_context() {
    let mut map = Map::new();
    for id in [50, 40, 30, 20, 10] {
        map.insert_keyframe(Keyframe::from_frame(test_frame(id, vec![[0; 32]; 2])))
            .unwrap();
    }
    let seed = |keyframe_idx| LandmarkSeed {
        position: Vec3F64::new(0.0, 0.0, 5.0),
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx,
            feature_idx: 0,
        },
    };
    map.insert_landmark(seed(50)).unwrap();
    let retired = map.insert_landmark(seed(20)).unwrap();
    map.remove_landmark(retired).unwrap();
    let active = map.insert_landmark(seed(30)).unwrap();
    map.link_observation(10, 1, active).unwrap();
    map.apply_insertion(MapInsertion {
        imu_factors: vec![imu_edge(40, 30), imu_edge(50, 40)],
        ..Default::default()
    })
    .unwrap();

    let window = local_window(&map);

    // Slot 0 observes only an older landmark and its IMU edge has no active
    // endpoint; slot 1 is in only as the IMU neighbour of the oldest active one.
    assert_eq!(window.keyframe_slots, vec![1, 2, 3, 4]);
    assert_eq!(window.landmark_ids, vec![active]);
    let snapshot = map.ba_snapshot(&window);
    assert_eq!(snapshot.landmark_ids(), &[active]);
    assert_eq!(snapshot.imu_factors().len(), 1);
}

/// Behaviour-preserving: solving the local window must give exactly what
/// solving a capture of the whole map gave before the window existed.
#[test]
fn local_window_solves_match_a_full_map_capture() {
    let mut map = fixture();
    map.edit_keyframes_for_test(|_, kf| kf.imu_bias.accel.x = 0.05);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![
            imu_edge(KF_IDS[0], KF_IDS[1]),
            imu_edge(KF_IDS[2], KF_IDS[3]),
        ],
        ..Default::default()
    })
    .unwrap();

    let merged = |update| {
        let mut map = map.clone();
        map.apply_ba_update(update).unwrap();
        map.state_fingerprint_for_test()
    };
    assert_eq!(
        merged(run_local_ba(
            map.ba_snapshot(&local_window(&map)),
            &camera()
        )),
        merged(run_local_ba(map.full_ba_snapshot(), &camera())),
    );
    let inertial =
        |snapshot| run_local_inertial_ba(snapshot, &camera(), None, Vec3F64::ZERO, &[], &[]);
    assert_eq!(
        merged(inertial(map.ba_snapshot(&local_window(&map)))),
        merged(inertial(map.full_ba_snapshot())),
    );
}

#[test]
fn initial_ba_uses_the_last_inserted_pair_and_keeps_the_older_pose_fixed() {
    let mut map = fixture();
    let before = map.clone();
    assert_eq!(
        run_initial_ba(&mut map, &camera(), LocalBaParams::default().depth_sigma).unwrap(),
        InitialBaOutcome::Refined
    );
    for id in &KF_IDS[..4] {
        assert_eq!(
            map.get_keyframe(*id).unwrap().frame.pose_world_to_cam,
            before.get_keyframe(*id).unwrap().frame.pose_world_to_cam
        );
    }
    assert_ne!(
        map.get_keyframe(KF_IDS[4]).unwrap().frame.pose_world_to_cam,
        before
            .get_keyframe(KF_IDS[4])
            .unwrap()
            .frame
            .pose_world_to_cam
    );
    assert_eq!(
        map.map_points().last().unwrap().position,
        before.map_points().last().unwrap().position
    );
}

#[test]
fn skipped_visual_and_inertial_solves_leave_the_map_unchanged() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(test_frame(19, vec![[0; 32]])))
        .unwrap();
    let before = map.state_fingerprint_for_test();
    assert_eq!(
        run_initial_ba(&mut map, &camera(), LocalBaParams::default().depth_sigma).unwrap(),
        InitialBaOutcome::Skipped
    );
    let visual = run_local_ba(map.ba_snapshot(&local_window(&map)), &camera());
    assert_eq!(map.apply_ba_update(visual).unwrap().map_points_updated, 0);
    let inertial = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
    );
    assert_eq!(map.apply_ba_update(inertial).unwrap().map_points_updated, 0);
    assert_eq!(map.state_fingerprint_for_test(), before);
}

#[test]
fn refused_initial_ba_writeback_is_an_error_and_leaves_the_map_unchanged() {
    let mut map = fixture();
    // Visual BA carries velocity through unchanged, so a corrupt one in the
    // bootstrap pair reaches writeback and makes the whole update unpublishable.
    map.edit_keyframes_for_test(|slot, kf| {
        if slot == KF_IDS.len() - 1 {
            kf.velocity_world.x = f64::NAN;
        }
    });
    let before = map.state_fingerprint_for_test();
    let error =
        run_initial_ba(&mut map, &camera(), LocalBaParams::default().depth_sigma).unwrap_err();
    assert!(
        matches!(
            error,
            InitialBaError::Writeback(BaUpdateError::NonFiniteKeyframe { .. })
        ),
        "{error:?}"
    );
    assert_eq!(map.state_fingerprint_for_test(), before);
}

fn imu_edge(prev: usize, curr: usize) -> ImuFactor {
    let calib = ImuCalib {
        gyro_noise: 1e-4,
        accel_noise: 1e-3,
        gyro_bias_noise: 1e-5,
        accel_bias_noise: 1e-3,
    };
    let samples = vec![
        ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        },
        ImuMeasurement {
            timestamp: 0.1,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        },
    ];
    ImuFactor {
        prev_kf_idx: prev,
        curr_kf_idx: curr,
        preintegrated: PreintegratedImu::from_measurements(
            ImuBias::default(),
            calib,
            &samples,
            0.0,
            0.1,
        ),
        raw_samples: samples,
        t0: 0.0,
        t1: 0.1,
        bias_walk_extra_s: 0.0,
        interpolation: SampleInterpolation::Hold,
    }
}

#[test]
fn inertial_ba_repropagates_active_edges_and_preserves_new_live_edges() {
    let mut map = fixture();
    map.edit_keyframes_for_test(|_, kf| kf.imu_bias.accel.x = 0.05);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![
            imu_edge(KF_IDS[0], KF_IDS[1]),
            imu_edge(KF_IDS[1], KF_IDS[2]),
        ],
        ..Default::default()
    })
    .unwrap();
    let before = map.state_fingerprint_for_test();
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
    );
    assert_eq!(map.state_fingerprint_for_test(), before);
    // A boundary edge has one fixed endpoint and must still be repropagated.
    assert_eq!(update.imu_preintegrations[0].bias.accel.x, 0.0);
    assert_eq!(update.imu_preintegrations[1].bias.accel.x, 0.05);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![imu_edge(KF_IDS[3], KF_IDS[4])],
        ..Default::default()
    })
    .unwrap();
    let newer = format!("{:?}", map.imu_factors()[2]);
    map.apply_ba_update(update).unwrap();
    assert_eq!(map.imu_factors()[1].preintegrated.bias.accel.x, 0.05);
    assert_eq!(format!("{:?}", map.imu_factors()[2]), newer);
}

#[test]
fn failed_inertial_solve_still_returns_repropagated_measurements() {
    let mut map = fixture();
    map.edit_keyframes_for_test(|_, kf| {
        kf.imu_bias.accel.x = 0.05;
        kf.frame.ensure_undistorted(&camera());
    });
    map.apply_insertion(MapInsertion {
        imu_factors: vec![imu_edge(KF_IDS[1], KF_IDS[2])],
        ..Default::default()
    })
    .unwrap();
    let before = map.clone();
    // Keep the normal observation count but make the numerical system invalid.
    // The failure occurs after repropagation, so discarding the whole update
    // on solver error would lose the refreshed measurement.
    let mut invalid_camera = camera();
    invalid_camera.fx = f64::NAN;
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &invalid_camera,
        None,
        Vec3F64::ZERO,
        &[],
        &[],
    );
    assert_eq!(update.imu_preintegrations[0].bias.accel.x, 0.05);
    let result = map.apply_ba_update(update).unwrap();
    assert!(result.keyframe_corrections.is_empty());
    assert_eq!(result.map_points_updated, 0);
    for (live, old) in map.keyframes().iter().zip(before.keyframes()) {
        assert_eq!(live.frame.pose_world_to_cam, old.frame.pose_world_to_cam);
        assert_eq!(live.velocity_world, old.velocity_world);
        assert_eq!(live.imu_bias.accel, old.imu_bias.accel);
        assert_eq!(live.imu_bias.gyro, old.imu_bias.gyro);
    }
    assert_eq!(map.imu_factors()[0].preintegrated.bias.accel.x, 0.05);
}
#[test]
fn stereo_depth_uses_proportional_sigma_with_floor() {
    let stereo_depth_obs = |kf: &Keyframe, desc_idx: usize, depth_sigma, pixel_sigma| {
        let observation = with_stereo_depth(
            BaObservation::default(),
            kf,
            desc_idx,
            depth_sigma,
            pixel_sigma,
        );
        (observation.depth_meas, observation.depth_sigma)
    };
    let mut frame = test_frame(0, vec![[0u8; 32], [1u8; 32]]);
    frame.depth = vec![10.0, -1.0];
    frame.u_right = vec![5.0, -1.0];
    let kf = Keyframe::from_frame(frame);
    let relative = LocalBaParams::default().depth_sigma;

    // Valid depth: sigma = 0.05 * 10 = 0.5.
    let (d, s) = stereo_depth_obs(&kf, 0, relative, 1.0);
    assert_eq!(d, Some(10.0));
    assert!((s - 0.5).abs() < 1e-6);

    // Sentinel depth: no measurement.
    let (d1, s1) = stereo_depth_obs(&kf, 1, relative, 1.0);
    assert_eq!(d1, None);
    assert!((s1 - 1.0).abs() < 1e-9);

    // Very near depth clamps to the sigma floor.
    let mut near = test_frame(1, vec![[0u8; 32]]);
    near.depth = vec![0.1]; // 0.05 * 0.1 = 0.005 < floor
    let kf_near = Keyframe::from_frame(near);
    let (_, s2) = stereo_depth_obs(&kf_near, 0, relative, 1.0);
    assert!((s2 - STEREO_DEPTH_MIN_SIGMA).abs() < 1e-9);
}

/// The default is the old `STEREO_DEPTH_REL_SIGMA · z` bit for bit, as a
/// file's `RelativeDepth(0.05)` resolves to it; disparity sigma grows as z².
#[test]
fn stereo_depth_sigma_models() {
    assert_eq!(
        LocalBaParams::default().depth_sigma,
        DepthSigma::Relative(0.05_f64 as f32)
    );
    for z in [0.3_f32, 2.0, 7.5, 40.0] {
        assert_eq!(
            DepthSigma::Relative(STEREO_DEPTH_REL_SIGMA).sigma(z),
            (STEREO_DEPTH_REL_SIGMA * z).max(STEREO_DEPTH_MIN_SIGMA)
        );
    }
    // EuRoC: f·b = 435 px · 0.11 m; 1 px of disparity at 5 m is 0.52 m.
    let disparity = DepthSigma::Disparity {
        px_over_focal_baseline: 1.0 / (435.0 * 0.11),
    };
    assert!((disparity.sigma(5.0) - 25.0 / 47.85).abs() < 1e-5);
    assert_eq!(disparity.sigma(0.5), STEREO_DEPTH_MIN_SIGMA);
    // An octave-2 keypoint (1.44 px) has 1.44 times the disparity noise; a
    // relative sigma ignores the octave.
    assert_eq!(disparity.sigma_at(5.0, 1.44), disparity.sigma(5.0) * 1.44);
    assert_eq!(
        DepthSigma::Relative(0.05).sigma_at(5.0, 1.44),
        DepthSigma::Relative(0.05).sigma(5.0)
    );
}

#[test]
fn inertial_ba_maps_wheel_factors_by_frame_id_and_drops_inactive_ones() {
    use kornia_factors::wheel_odometry::{self, WheelOdomFactor, WheelOdomMeasurement};
    use kornia_sensors::wheel::PlanarMotion;
    let map = fixture();
    let solve = |wheel: &[WheelOdomFactor]| {
        let update = run_local_inertial_ba(
            map.ba_snapshot(&local_window(&map)),
            &camera(),
            None,
            Vec3F64::ZERO,
            wheel,
            &[],
        );
        let mut map = map.clone();
        map.apply_ba_update(update).unwrap();
        map
    };
    let pose = |map: &Map, id| map.get_keyframe(id).unwrap().frame.pose_world_to_cam;
    // Identity extrinsic: the camera is the base. The fixture moves 0.2 m
    // per keyframe along camera -x; the wheels claim 0.1 m more.
    let measurement = WheelOdomMeasurement::new(PlanarMotion {
        delta: [-0.3, 0.0, 0.0],
        covariance: Mat3F64::from_diagonal(Vec3F64::new(1e-4, 1e-4, 1e-4)),
    });
    let factor = |from_idx, to_idx| {
        WheelOdomFactor::new(
            from_idx,
            to_idx,
            measurement,
            Pose3d::IDENTITY,
            Pose3d::IDENTITY,
            None,
        )
    };
    let baseline = solve(&[]).state_fingerprint_for_test();
    assert_eq!(
        solve(&[factor(999, KF_IDS[4]), factor(KF_IDS[0], KF_IDS[1])]).state_fingerprint_for_test(),
        baseline
    );

    let edge = factor(KF_IDS[3], KF_IDS[4]);
    let residual = |map: &Map| {
        let (r, _) = wheel_odometry::residual_and_jacobian(
            &pose(map, KF_IDS[3]),
            &pose(map, KF_IDS[4]),
            &edge,
        );
        r.iter().map(|x| x * x).sum::<f64>().sqrt()
    };
    let fused = solve(&[edge]);
    let visual = solve(&[]);
    assert!(
        residual(&fused) < 0.5 * residual(&visual),
        "{} vs {}",
        residual(&fused),
        residual(&visual)
    );
}

/// A keyframe linked to the active window only by a caller-held factor
/// (a stationary edge) and seeing none of its landmarks enters as fixed
/// context through `local_window_with_neighbors`; `local_window` is unchanged.
#[test]
fn caller_held_links_pull_their_neighbour_into_the_window() {
    let mut map = Map::new();
    for id in [50, 40, 30, 20, 10] {
        map.insert_keyframe(Keyframe::from_frame(test_frame(id, vec![[0; 32]; 2])))
            .unwrap();
    }
    let active = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 30,
                feature_idx: 0,
            },
        })
        .unwrap();
    map.link_observation(10, 1, active).unwrap();
    let plain = local_window(&map);
    assert_eq!(plain.keyframe_slots, vec![2, 3, 4]);
    assert_eq!(
        local_window_with_neighbors(&map, []).keyframe_slots,
        plain.keyframe_slots
    );
    // 40 → 30 links the oldest active keyframe back; 50 → 40 has no
    // active endpoint and pulls nothing.
    let linked = local_window_with_neighbors(&map, [(40, 30), (50, 40)]);
    assert_eq!(linked.keyframe_slots, vec![1, 2, 3, 4]);
    assert_eq!(linked.landmark_ids, plain.landmark_ids);
}

/// Repropagating an edge whose bias link also spans elided still time keeps
/// that walk: the samples alone would give back the walk over 0.1 s.
///
/// Mutation: rebuilding with `from_measurements` instead of `repreintegrate`
/// turned the widened-covariance assert red.
#[test]
fn inertial_ba_repropagation_keeps_the_elided_bias_walk() {
    let mut map = fixture();
    map.edit_keyframes_for_test(|_, kf| kf.imu_bias.accel.x = 0.05);
    let mut edge = imu_edge(KF_IDS[1], KF_IDS[2]);
    edge.bias_walk_extra_s = 3600.0;
    edge.preintegrated.widen_bias_walk(3600.0);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![edge],
        ..Default::default()
    })
    .unwrap();
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
    );
    let repropagated = &update.imu_preintegrations[0];
    assert_eq!(repropagated.bias.accel.x, 0.05, "repropagated at all");
    let walk = |density: f64| density * density * (3600.0 + 0.1);
    let calib = repropagated.calib;
    for i in 0..3 {
        let g = repropagated.bias_covariance[(i, i)];
        let a = repropagated.bias_covariance[(i + 3, i + 3)];
        assert!((g - walk(calib.gyro_bias_noise)).abs() < 1e-3 * g, "{g}");
        assert!((a - walk(calib.accel_bias_noise)).abs() < 1e-3 * a, "{a}");
    }
}

/// The write-back guard keeps a landmark the solve left behind an observer,
/// or flung (more than tenfold farther from its nearest observer, and beyond
/// 50 m), at its old position; ordinary refinements, near or far, pass.
///
/// Mutations (2026-10-13, each in place): dropping the `> WRITE_BACK_FAR_M`
/// half refused the 5 cm → 2 m refinement; dropping the tenfold-growth half
/// refused the 300 m → 400 m one; dropping the behind check wrote back the
/// landmark at z = −1 m. Dropping `is_finite` stays green: a NaN position
/// gives no finite distance (`f64::min` skips NaN, leaving ∞), which the
/// flung test already refuses; the check is kept as a second line.
#[test]
fn write_back_guard_refuses_landmarks_behind_or_flung() {
    let kf = |x: f64| ViBaKeyframe {
        pose: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-x, 0.0, 0.0)),
        velocity: Vec3F64::ZERO,
        bias: ImuBias::default(),
        fixed: false,
        inertial_fixed: false,
    };
    let keyframes = [kf(0.0), kf(0.5)];
    let at = |z: f64| Vec3F64::new(0.1, 0.0, z);
    // (before, after, written back?)
    let cases = [
        (at(2.0), at(2.1), true),
        (at(0.05), at(2.0), true),      // ~18× but within 50 m
        (at(300.0), at(400.0), true),   // beyond 50 m but not 10×
        (at(2.0), at(1e7), false),      // flung along the ray
        (at(2.0), at(-1.0), false),     // behind both observers
        (at(2.0), at(f64::NAN), false), // not finite
    ];
    let before: Vec<Vec3F64> = cases.iter().map(|c| c.0).collect();
    let after: Vec<Vec3F64> = cases.iter().map(|c| c.1).collect();
    let mut observations: Vec<BaObservation> = (0..cases.len())
        .flat_map(|j| {
            (0..2).map(move |k| BaObservation {
                pose_idx: k,
                point_idx: j,
                pixel: [0.0, 0.0],
                fixed_pose: false,
                fixed_point: false,
                depth_meas: None,
                depth_sigma: 1.0,
            })
        })
        .collect();
    let keep = landmarks_to_write_back(
        &observations,
        |_| None,
        (&keyframes, &before),
        (&keyframes, &after),
    );
    let want: Vec<bool> = cases.iter().map(|c| c.2).collect();
    assert_eq!(keep, want);

    // Behind only one observer is enough; unobserved landmarks pass.
    let mut moved = keyframes.clone();
    moved[1].pose = Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.0, std::f64::consts::PI, 0.0)).matrix(),
        Vec3F64::ZERO,
    );
    observations.retain(|o| o.point_idx == 0);
    let keep = landmarks_to_write_back(
        &observations,
        |_| None,
        (&keyframes, &before),
        (&moved, &before),
    );
    assert_eq!(keep, [false, true, true, true, true, true]);
}

/// `run_local_inertial_ba` reports what its solve did.
#[test]
fn local_inertial_ba_reports_its_solve() {
    let map = fixture();
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
    );
    let stats = update.solve_stats().expect("the solver ran");
    assert!(stats.iterations > 0);
    assert!(stats.initial_cost.is_finite() && stats.final_cost <= stats.initial_cost);
    assert_eq!(stats.points_rejected, 0);
    assert!(
        run_local_ba(map.ba_snapshot(&local_window(&map)), &camera())
            .solve_stats()
            .is_none()
    );
}

/// `max_iterations` caps the VI-BA solve.
#[test]
fn local_inertial_ba_honours_its_iteration_cap() {
    let map = fixture();
    let update = run_local_inertial_ba_with(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
        &LocalBaParams {
            max_iterations: 1,
            ..LocalBaParams::default()
        },
    );
    assert_eq!(update.solve_stats().expect("the solver ran").iterations, 1);
}

/// The run counters tell an early exit from a solve: a window with fewer
/// than `MIN_OBSERVATIONS` observations never reaches the solver.
#[test]
fn a_window_under_the_observation_minimum_reports_the_solver_did_not_run() {
    let mut map = Map::new();
    for id in [10, 20] {
        map.insert_keyframe(Keyframe::from_frame(test_frame(id, vec![[0; 32]; 2])))
            .unwrap();
    }
    let landmark = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 10,
                feature_idx: 0,
            },
        })
        .unwrap();
    map.link_observation(20, 0, landmark).unwrap();

    let visual = run_local_ba(map.ba_snapshot(&local_window(&map)), &camera());
    assert_eq!(visual.outcome, BaOutcome::default());
    assert_eq!(visual.outcome.solve, BaSolve::NotRun);
    let inertial = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        Some(Pose3d::IDENTITY),
        Vec3F64::new(0.0, 0.0, -9.81),
        &[],
        &[],
    );
    assert_eq!(inertial.outcome.solve, BaSolve::NotRun);

    let map = fixture();
    let solved = run_local_ba(map.ba_snapshot(&local_window(&map)), &camera());
    assert_eq!(solved.outcome.solve, BaSolve::Solved);
}

#[test]
fn active_start_counts_only_keyframes_that_are_not_anchors() {
    let mut map = fixture();
    let keyframes = map.keyframes();
    assert_eq!(active_start(keyframes, 3), 2);
    assert_eq!(active_start(keyframes, 0), 5);
    assert_eq!(active_start(keyframes, 9), 0);

    map.edit_keyframes_for_test(|slot, kf| kf.anchor = slot >= 3);
    let keyframes = map.keyframes();
    assert_eq!(active_start(keyframes, 1), 2);
    assert_eq!(active_start(keyframes, 3), 0);
    map.edit_keyframes_for_test(|slot, kf| kf.anchor = slot == 2);
    assert_eq!(active_start(map.keyframes(), 2), 3);
    assert_eq!(active_start(map.keyframes(), 3), 1);
}

/// An anchor is never active: it enters the window only as fixed context of
/// the landmarks the active keyframes observe, and its own landmarks stay out.
#[test]
fn anchors_enter_the_local_window_only_as_context() {
    let mut map = Map::new();
    for id in [50, 40, 30, 20, 10] {
        map.insert_keyframe(Keyframe::from_frame(test_frame(id, vec![[0; 32]; 2])))
            .unwrap();
    }
    map.edit_keyframes_for_test(|_, kf| kf.anchor = [30, 10].contains(&kf.frame.idx));
    let seed = |keyframe_idx| LandmarkSeed {
        position: Vec3F64::new(0.0, 0.0, 5.0),
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx,
            feature_idx: 0,
        },
    };
    let shared = map.insert_landmark(seed(30)).unwrap();
    map.link_observation(20, 1, shared).unwrap();
    map.insert_landmark(seed(10)).unwrap();

    let window = local_window(&map);
    // Active: 20, 40, 50 (the three newest non-anchors). 30 shares a landmark
    // with 20; 10 observes only its own.
    assert_eq!(window.keyframe_slots, vec![0, 1, 2, 3]);
    assert_eq!(window.landmark_ids, vec![shared]);
}

/// A window [tracked, anchor, anchor, tracked] behind fixed context, the
/// tracked pair bridged by one IMU edge, solves and leaves both anchors bit
/// for bit as they were.
#[test]
fn anchors_inside_the_window_stay_fixed_through_both_solves() {
    let mut map = fixture();
    map.edit_keyframes_for_test(|slot, kf| {
        kf.anchor = slot == 2 || slot == 3;
        kf.velocity_world = Vec3F64::new(0.1 * slot as f64, 0.0, 0.0);
        kf.imu_bias.accel.x = 0.01 * slot as f64;
    });
    map.apply_insertion(MapInsertion {
        imu_factors: vec![imu_edge(KF_IDS[1], KF_IDS[4])],
        ..Default::default()
    })
    .unwrap();
    let params = LocalBaParams {
        active_keyframes: 2,
        ..LocalBaParams::default()
    };
    let window = local_window_of(&map, params.active_keyframes, []);
    assert_eq!(window.keyframe_slots, vec![0, 1, 2, 3, 4]);
    let anchors = |map: &Map| {
        KF_IDS[2..4]
            .iter()
            .map(|id| format!("{:?}", map.get_keyframe(*id).unwrap()))
            .collect::<Vec<_>>()
    };
    let before = anchors(&map);

    let visual = run_local_ba_with(map.ba_snapshot(&window), &camera(), &params);
    let mut solved = map.clone();
    let result = solved.apply_ba_update(visual).unwrap();
    assert!(!result.keyframe_corrections.is_empty());
    assert!(
        result
            .keyframe_corrections
            .iter()
            .all(|c| [KF_IDS[1], KF_IDS[4]].contains(&c.kf_idx))
    );
    assert_eq!(anchors(&solved), before);

    let inertial = run_local_inertial_ba_with(
        map.ba_snapshot(&window),
        &camera(),
        None,
        Vec3F64::ZERO,
        &[],
        &[],
        &params,
    );
    assert!(inertial.solve_stats().is_some(), "the solver ran");
    let result = map.apply_ba_update(inertial).unwrap();
    assert!(
        result
            .keyframe_corrections
            .iter()
            .all(|c| [KF_IDS[1], KF_IDS[4]].contains(&c.kf_idx))
    );
    assert_eq!(anchors(&map), before);
}

/// ORB-SLAM3's `bLarge` is strict (`inliers > 100`), and only the window's
/// size, iteration cap and initial damping change with it.
#[test]
fn the_large_window_applies_above_its_inlier_threshold() {
    let params = LocalBaParams {
        active_keyframes: 10,
        large_window: LargeWindow::Enabled {
            min_tracked_inliers: 100,
            active_keyframes: 25,
            max_iterations: 4,
            initial_lambda: 1e-2,
        },
        ..LocalBaParams::default()
    };
    assert!(!params.large_window_applies(100));
    assert!(params.large_window_applies(101));
    assert!(!LocalBaParams::default().large_window_applies(10_000));

    let large = params.large();
    assert_eq!(
        (
            large.active_keyframes,
            large.max_iterations,
            large.initial_lambda
        ),
        (25, 4, 1e-2)
    );
    assert_eq!(
        LocalBaParams {
            active_keyframes: 10,
            max_iterations: params.max_iterations,
            initial_lambda: params.initial_lambda,
            ..large
        },
        params
    );
    assert_eq!(LocalBaParams::default().large(), LocalBaParams::default());
}

const CRUISE_EDGE_SEC: f64 = 0.1;
const CRUISE_SPEED: f64 = 2.0;

fn cruise_points() -> Vec<Vec3F64> {
    (0..12)
        .map(|i| {
            Vec3F64::new(
                (i % 4) as f64 * 0.4 - 0.6,
                (i / 4) as f64 * 0.3 - 0.3,
                4.0 + (i % 3) as f64 * 0.5,
            )
        })
        .collect()
}

/// Cruise keyframe `id` imaging `points` exactly on its first features, of
/// `features` in all.
fn cruise_keyframe(id: usize, points: &[Vec3F64], features: usize) -> Keyframe {
    let mut frame = test_frame(id, vec![[0; 32]; features]);
    let x = CRUISE_SPEED * CRUISE_EDGE_SEC * id as f64;
    for (feature, point) in points.iter().enumerate() {
        frame.features.keypoints_xy[feature] = [
            (200.0 * (point.x - x) / point.z + 320.0) as f32,
            (200.0 * point.y / point.z + 240.0) as f32,
        ];
    }
    frame.pose_world_to_cam.translation.x = -x;
    let mut keyframe = Keyframe::from_frame(frame);
    keyframe.velocity_world = Vec3F64::new(CRUISE_SPEED, 0.0, 0.0);
    keyframe
}

/// The IMU edge from cruise keyframe `k` to `k + 1`.
fn cruise_edge(k: usize, gravity: Vec3F64, theta: Vec3F64) -> ImuFactor {
    let calib = ImuCalib {
        gyro_noise: 1.7e-4,
        accel_noise: 2e-3,
        gyro_bias_noise: 2e-5,
        accel_bias_noise: 3e-3,
    };
    let accel = SO3F64::exp(theta * -1.0) * (gravity * -1.0);
    let t0 = CRUISE_EDGE_SEC * k as f64;
    let samples: Vec<ImuMeasurement> = (0..=20)
        .map(|s| ImuMeasurement {
            timestamp: t0 + CRUISE_EDGE_SEC * s as f64 / 20.0,
            gyro: Vec3F64::ZERO,
            accel,
        })
        .collect();
    ImuFactor {
        prev_kf_idx: k,
        curr_kf_idx: k + 1,
        preintegrated: PreintegratedImu::from_measurements(
            ImuBias::default(),
            calib,
            &samples,
            t0,
            t0 + CRUISE_EDGE_SEC,
        ),
        raw_samples: samples,
        t0,
        t1: t0 + CRUISE_EDGE_SEC,
        bias_walk_extra_s: 0.0,
        interpolation: SampleInterpolation::Hold,
    }
}

/// `count` level keyframes cruising along x at 2 m/s, 0.1 s apart, the
/// landmarks of `fixture` imaged exactly, and IMU edges measured by a body
/// whose true extrinsic is the factory one (identity) turned by the
/// primary rotation offset `theta`: `T_bc = Exp(−θ)`.
fn cruise_map(gravity: Vec3F64, theta: Vec3F64, count: usize) -> Map {
    let points = cruise_points();
    let mut map = Map::new();
    for id in 0..count {
        map.insert_keyframe(cruise_keyframe(id, &points, 12))
            .unwrap();
    }
    for (feature, point) in points.into_iter().enumerate() {
        let id = map
            .insert_landmark(LandmarkSeed {
                position: point,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: feature,
                },
            })
            .unwrap();
        for kf in 1..count {
            map.link_observation(kf, feature, id).unwrap();
        }
    }
    map.apply_insertion(MapInsertion {
        imu_factors: (0..count - 1)
            .map(|k| cruise_edge(k, gravity, theta))
            .collect(),
        ..Default::default()
    })
    .unwrap();
    map
}

/// D1: with the primary rotation state on, the solve reads the extrinsic
/// `T_bc · Exp(−θ)`, so the boundary prior must be formed there too. The
/// data are exact at θ = 0.2°, so that prior's gradient at the solved state
/// is round-off: 9.5e-3 (2026-11-18). Red on the old code: formed at the
/// factory extrinsic, the velocity residual `g·θ·dt` made it 42.7.
#[test]
fn the_boundary_prior_is_formed_at_the_calibrated_extrinsic() {
    use crate::mapping::calibration_states::{Calibration, Layout};
    use crate::mapping::global_vi_ba::RotationSlicePrior;

    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let theta = Vec3F64::new(0.2f64.to_radians(), 0.0, 0.0);
    let map = cruise_map(gravity, theta, 5);
    let config = OnlineCalibrationConfig {
        camera_rotation: Some(RotationSlicePrior {
            step_sigma_deg: 0.05,
            value_sigma_deg: 1.0,
        }),
        ..OnlineCalibrationConfig::default()
    };
    let start = OnlineCalibrationState::default().advanced(
        &config,
        &Layout::online(&config, 1),
        &Calibration {
            values: theta.to_array().to_vec(),
        },
        false,
    );
    let params = LocalBaParams {
        boundary: BoundaryPrior::Marginal(FrozenPoseSigma::default()),
        online_calibration: config,
        ..LocalBaParams::default()
    };
    let update = run_local_inertial_ba_calibrated(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        gravity,
        &[],
        &[],
        &params,
        &start,
    );
    let prior = update.next_prior.expect("the fallback prior");
    let worst = prior.rhs.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(worst < 1.0, "prior gradient {worst:.3e}");
}

/// E8: a window as large as the map still fixes its oldest keyframe
/// (ORB-SLAM3 `Nd = min(KFs − 2, maxOpt)`), so the newest marginal is the
/// gauge-fixed one and the oldest pose stays put. Red on the old code
/// (2026-11-18): nothing was fixed, only damping held the gauge.
#[test]
fn a_window_as_large_as_the_map_fixes_its_oldest_keyframe() {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let mut map = cruise_map(gravity, Vec3F64::ZERO, 5);
    map.edit_keyframes_for_test(|_, kf| kf.frame.pose_world_to_cam.translation.y += 0.01);
    let oldest = map.keyframes()[0].frame.pose_world_to_cam;
    let params = LocalBaParams {
        active_keyframes: 5,
        newest_marginal: true,
        ..LocalBaParams::default()
    };
    let update = run_local_inertial_ba_with(
        map.ba_snapshot(&local_window_of(&map, params.active_keyframes, [])),
        &camera(),
        None,
        gravity,
        &[],
        &[],
        &params,
    );
    assert_eq!(update.keyframes[0].pose_world_to_cam, oldest);
    let (_, marginal) = update.newest_marginal.expect("the newest marginal");
    let sigma_m = (0..3).map(|d| marginal[(d, d)]).sum::<f64>().sqrt();
    assert!(sigma_m < 0.05, "newest position sigma {sigma_m:.3e} m");
}

/// One Marginal solve of `map` with `active_keyframes`, applied.
fn marginal_job(map: &mut Map, gravity: Vec3F64, active_keyframes: usize) -> BaUpdate {
    let params = LocalBaParams {
        active_keyframes,
        boundary: BoundaryPrior::Marginal(FrozenPoseSigma::default()),
        ..LocalBaParams::default()
    };
    run_local_inertial_ba_with(
        map.ba_snapshot(&local_window_of(map, active_keyframes, [])),
        &camera(),
        None,
        gravity,
        &[],
        &[],
        &params,
    )
}

/// D4: a stored prior older than the window's boundary (a job skipped, or a
/// smaller window after a large one) is chained onto it over the fixed
/// keyframes between. Red on the old code (2026-11-18): the solve fell back
/// and a fresh, rank-deficient prior replaced the chain.
#[test]
fn an_older_prior_is_chained_onto_the_boundary() {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let mut map = cruise_map(gravity, Vec3F64::ZERO, 7);
    // Active 2..=6: the prior lands on keyframe 2.
    let first = marginal_job(&mut map, gravity, 5);
    map.apply_ba_update(first).unwrap();
    let stored = map.full_ba_snapshot();
    assert_eq!(stored.marginal_prior().map(|p| p.frame_idx), Some(2));
    // Active 4..=6: boundary 3 -> 4, two keyframes past the prior.
    let update = marginal_job(&mut map, gravity, 3);
    assert!(!update.outcome.prior_fallback);
    assert_eq!(update.next_prior.as_ref().map(|p| p.frame_idx), Some(4));
}

/// D4: a window whose boundary is older than the stored prior falls back to
/// the scaled edge and leaves the map's prior alone. Red on the old code
/// (2026-11-18): it replaced the chain with a fresh prior.
#[test]
fn a_boundary_older_than_the_prior_keeps_the_prior() {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let mut map = cruise_map(gravity, Vec3F64::ZERO, 7);
    // Active 4..=6: the prior lands on keyframe 4.
    let first = marginal_job(&mut map, gravity, 3);
    map.apply_ba_update(first).unwrap();
    let stored = map.full_ba_snapshot();
    assert_eq!(stored.marginal_prior().map(|p| p.frame_idx), Some(4));
    // Active 2..=6: boundary 1 -> 2, older than the prior.
    let update = marginal_job(&mut map, gravity, 5);
    assert!(update.outcome.prior_fallback);
    assert!(update.next_prior.is_none());
}

/// One keyframe whose camera 0 is raw fisheye, seeing landmark 0 at 70° on
/// its feature 0 (octave 2) and landmark 1 on axis on feature 1.
fn raw_fisheye_window() -> (Keyframe, WindowPoints) {
    let rig = crate::rig::tests::raw_two_camera_rig(40.0);
    let view = rig.view(0).unwrap().clone();
    let (s, c) = 70f64.to_radians().sin_cos();
    let mut frame = test_frame(0, vec![[0; 32]; 2]);
    frame.features.keypoints_xy = [Vec3F64::new(s, 0.0, c), Vec3F64::new(0.0, 0.0, 1.0)]
        .map(|p| view.project_raw(p).unwrap())
        .to_vec();
    frame.features.octaves = vec![2, 0];
    frame.rig = crate::frame::RigFeatures {
        geometry: Some(std::sync::Arc::new(rig)),
        camera_ranges: vec![0..2, 2..2],
        ..Default::default()
    };
    let mut keyframe = Keyframe::from_frame(frame);
    keyframe.associate_map_point(0, 0);
    keyframe.associate_map_point(1, 1);
    let window = WindowPoints {
        global: vec![0, 1],
        slots: vec![0, 1],
        local_of: HashMap::from([(0, 0), (1, 1)]),
        positions: vec![Vec3F64::new(s, 0.0, c) * 4.0, Vec3F64::new(0.0, 0.0, 4.0)],
    };
    (keyframe, window)
}

#[test]
fn raw_fisheye_magnification_scales_the_local_ba_sigma() {
    let (keyframe, window) = raw_fisheye_window();
    let sigmas = |information, weight| {
        window
            .observations_with_views(
                [(0, &keyframe, false)],
                &camera(),
                true,
                DepthSigma::Relative(STEREO_DEPTH_REL_SIGMA),
                information,
                weight,
            )
            .sigma_px
    };
    assert!(sigmas(VisualInformation::Unit, FisheyeWeight::Uniform).is_empty());

    let view = keyframe.frame.raw_view(0).unwrap();
    let pixel = keyframe.frame.undistorted_xy(0, &camera()).unwrap();
    let scale = FisheyeWeight::Magnification.sigma_scale(Some(view), pixel);
    assert!((scale - 4.479).abs() < 0.01, "{scale}");
    let unit = sigmas(VisualInformation::Unit, FisheyeWeight::Magnification);
    assert_eq!(unit.len(), 2);
    assert_eq!(unit[0], scale);
    assert!((unit[1] - 1.0).abs() < 1e-6, "{}", unit[1]);
    let octave = sigmas(
        VisualInformation::OctaveScaled,
        FisheyeWeight::Magnification,
    );
    assert_eq!(
        octave[0],
        VisualInformation::OctaveScaled.sigma_px(2) * scale
    );
    assert_eq!(
        sigmas(VisualInformation::OctaveScaled, FisheyeWeight::Uniform),
        vec![VisualInformation::OctaveScaled.sigma_px(2), 1.0]
    );
}

/// Card R: a cruise map solved over keyframes 0..=4, its newest 2 cm off in
/// y so the solve moves it, and keyframes 5 and 6 inserted during that solve
/// tracked from it, at a seed 0.3 m/s fast, with one landmark of their own
/// (referenced by 5).
/// Returns the live map, the solve, and that landmark.
fn stale_cruise(merge: LocalBaMerge, anchor_five: bool) -> (Map, BaUpdate, usize) {
    stale_cruise_with(merge, anchor_five, false)
}

/// `stale_cruise`, with keyframe 5 a re-seeded segment's first keyframe
/// when `segment_start_five`; the 4 -> 5 edge is then its bridge.
fn stale_cruise_with(
    merge: LocalBaMerge,
    anchor_five: bool,
    segment_start_five: bool,
) -> (Map, BaUpdate, usize) {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let mut map = cruise_map(gravity, Vec3F64::ZERO, 5);
    map.edit_keyframes_for_test(|_, kf| {
        if kf.frame.idx == 4 {
            kf.frame.pose_world_to_cam.translation.y += 0.02;
        }
    });
    let params = LocalBaParams {
        merge,
        ..LocalBaParams::default()
    };
    let update = run_local_inertial_ba_with(
        map.ba_snapshot(&local_window_of(&map, params.active_keyframes, [])),
        &camera(),
        None,
        gravity,
        &[],
        &[],
        &params,
    );
    let points = cruise_points();
    let landmarks = map.map_points().len();
    for id in [5, 6] {
        let mut kf = cruise_keyframe(id, &points, 13);
        kf.frame.pose_world_to_cam.translation.y += 0.02;
        kf.velocity_world = Vec3F64::new(CRUISE_SPEED + 0.3, 0.0, 0.0);
        kf.anchor = anchor_five && id == 5;
        kf.segment_start = segment_start_five && id == 5;
        map.insert_keyframe(kf).unwrap();
        for landmark in 0..landmarks {
            map.link_observation(id, landmark, landmark).unwrap();
        }
    }
    let own = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.9, 0.05, 4.2),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 5,
                feature_idx: 12,
            },
        })
        .unwrap();
    map.link_observation(6, 12, own).unwrap();
    map.apply_insertion(MapInsertion {
        imu_factors: vec![
            cruise_edge(4, gravity, Vec3F64::ZERO),
            cruise_edge(5, gravity, Vec3F64::ZERO),
        ],
        ..Default::default()
    })
    .unwrap();
    (map, update, own)
}

/// Pose, velocity and bias `[b_g b_a]` of live keyframe `idx`.
fn live_state(map: &Map, idx: usize) -> (Pose3d, Vec3F64, [Vec3F64; 2]) {
    let kf = map.get_keyframe(idx).unwrap();
    let bias = [kf.imu_bias.gyro, kf.imu_bias.accel];
    (kf.frame.pose_world_to_cam, kf.velocity_world, bias)
}

fn assert_pose_near(a: &Pose3d, b: &Pose3d, tolerance: f64) {
    let rotation = a
        .rotation
        .to_cols_array()
        .iter()
        .zip(b.rotation.to_cols_array())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let translation = (a.translation - b.translation).length();
    assert!(
        rotation < tolerance && translation < tolerance,
        "rotation {rotation:.3e} translation {translation:.3e}"
    );
}

/// Knob off, the keyframes inserted during the solve are left exactly as
/// they were, and the merge counts them.
#[test]
fn a_snapshot_only_merge_leaves_newer_keyframes_untouched() {
    let (mut map, update, own) = stale_cruise(LocalBaMerge::SnapshotOnly, false);
    let before = [live_state(&map, 5), live_state(&map, 6)];
    let landmark = map.map_points()[own].position;
    let result = map.apply_ba_update(update).unwrap();
    assert_eq!(result.newer_keyframes, 2);
    assert_eq!(result.propagated_keyframes, 0);
    assert!(result.keyframe_corrections.iter().all(|c| !c.propagated));
    assert_eq!([live_state(&map, 5), live_state(&map, 6)], before);
    assert_eq!(map.map_points()[own].position, landmark);
}

/// Knob on, the correction of keyframe 4 is carried to 5 and 6: their
/// tracked pose relative to 4 is kept, the crossing edge starts with no
/// velocity residual, the bias is 4's, and 5's own landmark moves with it.
#[test]
fn a_propagating_merge_carries_the_correction_to_newer_keyframes() {
    let gravity = Vec3F64::new(0.0, 9.81, 0.0);
    let (mut map, update, own) = stale_cruise(LocalBaMerge::PropagateToNewer, false);
    let (four_before, ..) = live_state(&map, 4);
    let (five_before, ..) = live_state(&map, 5);
    let landmark_in_five = five_before.transform_point(&map.map_points()[own].position);
    let result = map.apply_ba_update(update).unwrap();

    let (four, v4, b4) = live_state(&map, 4);
    let (five, v5, b5) = live_state(&map, 5);
    assert!((four.translation - four_before.translation).length() > 1e-4);
    assert_pose_near(
        &Pose3d::between(&four, &five),
        &Pose3d::between(&four_before, &five_before),
        1e-12,
    );
    let edge = &map
        .imu_factors()
        .iter()
        .find(|f| f.prev_kf_idx == 4 && f.curr_kf_idx == 5)
        .unwrap()
        .preintegrated;
    let (res, _, _) = preintegration::residual_and_jacobians(
        kornia_factors::inertial::NavState {
            pose_cw: &four,
            velocity: v4,
            bias: ImuBias {
                gyro: b4[0],
                accel: b4[1],
            },
        },
        kornia_factors::inertial::NavState {
            pose_cw: &five,
            velocity: v5,
            bias: ImuBias {
                gyro: b5[0],
                accel: b5[1],
            },
        },
        edge,
        &gravity,
        None,
    );
    let r_v = res.fixed_rows::<3>(3).norm();
    assert!(r_v < 1e-9, "crossing r_v {r_v:.3e} m/s");
    assert_eq!(b5, b4);
    assert_eq!(live_state(&map, 6).2, b4);
    let moved = five.transform_point(&map.map_points()[own].position);
    assert!((moved - landmark_in_five).length() < 1e-9);
    assert_eq!(result.propagated_keyframes, 2);
    let carried: Vec<usize> = result
        .keyframe_corrections
        .iter()
        .filter(|c| c.propagated)
        .map(|c| c.kf_idx)
        .collect();
    assert_eq!(carried, [5, 6]);
}

/// An anchor among the newer keyframes stops the propagation before it.
#[test]
fn a_propagating_merge_does_not_cross_an_anchor() {
    let (mut map, update, _) = stale_cruise(LocalBaMerge::PropagateToNewer, true);
    let before = [live_state(&map, 5), live_state(&map, 6)];
    let result = map.apply_ba_update(update).unwrap();
    assert_eq!(result.propagated_keyframes, 0);
    assert_eq!([live_state(&map, 5), live_state(&map, 6)], before);
}

/// Card V (E-2): a bridged re-seed's first keyframe stops the propagation,
/// so neither it nor anything after it takes the old window's correction.
#[test]
fn a_propagating_merge_stops_at_a_bridged_segment_start() {
    let (mut map, update, _) = stale_cruise_with(LocalBaMerge::PropagateToNewer, false, true);
    assert!(
        map.imu_factors()
            .iter()
            .any(|f| f.prev_kf_idx == 4 && f.curr_kf_idx == 5),
        "the bridge edge reaches 5"
    );
    let before = [live_state(&map, 5), live_state(&map, 6)];
    let result = map.apply_ba_update(update).unwrap();
    assert_eq!(result.newer_keyframes, 2);
    assert_eq!(result.propagated_keyframes, 0);
    assert!(result.stopped_at_bridge);
    assert_eq!([live_state(&map, 5), live_state(&map, 6)], before);

    let (mut plain, update, _) = stale_cruise(LocalBaMerge::PropagateToNewer, false);
    assert!(!plain.apply_ba_update(update).unwrap().stopped_at_bridge);
}

/// A snapshot from another world frame is still refused wholesale.
#[test]
fn a_propagating_merge_still_rejects_another_world_epoch() {
    let (mut map, update, _) = stale_cruise(LocalBaMerge::PropagateToNewer, false);
    map.advance_world_epoch_for_test();
    let before = [
        live_state(&map, 4),
        live_state(&map, 5),
        live_state(&map, 6),
    ];
    assert!(matches!(
        map.apply_ba_update(update),
        Err(BaUpdateError::ObsoleteWorldFrame)
    ));
    assert_eq!(
        [
            live_state(&map, 4),
            live_state(&map, 5),
            live_state(&map, 6)
        ],
        before
    );
}

fn erase_gravity() -> Vec3F64 {
    Vec3F64::new(0.0, 9.81, 0.0)
}

/// Five cruise keyframes whose feature 0 is unlinked and feature `f + 1`
/// images landmark `f`; on the newest keyframe, landmark 5's feature is
/// displaced down by `displace_px` and found at `octave`. Returns the map and
/// landmark 5's id.
fn erase_cruise_map(displace_px: f32, octave: u8) -> (Map, usize) {
    const COUNT: usize = 5;
    let gravity = erase_gravity();
    let points = cruise_points();
    let imaged: Vec<Vec3F64> = std::iter::once(points[0])
        .chain(points.iter().copied())
        .collect();
    let mut map = Map::new();
    for id in 0..COUNT {
        let mut keyframe = cruise_keyframe(id, &imaged, imaged.len());
        if id == COUNT - 1 {
            keyframe.frame.features.keypoints_xy[6][1] += displace_px;
            keyframe.frame.features.octaves[6] = octave;
        }
        map.insert_keyframe(keyframe).unwrap();
    }
    let mut ids = Vec::new();
    for (landmark, point) in points.into_iter().enumerate() {
        let id = map
            .insert_landmark(LandmarkSeed {
                position: point,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: landmark + 1,
                },
            })
            .unwrap();
        for kf in 1..COUNT {
            map.link_observation(kf, landmark + 1, id).unwrap();
        }
        ids.push(id);
    }
    map.apply_insertion(MapInsertion {
        imu_factors: (0..COUNT - 1)
            .map(|k| cruise_edge(k, gravity, Vec3F64::ZERO))
            .collect(),
        ..Default::default()
    })
    .unwrap();
    (map, ids[5])
}

/// One solve of [`erase_cruise_map`]: its update and landmark 5's id.
fn erase_cruise(
    displace_px: f32,
    octave: u8,
    information: VisualInformation,
    outlier_erase: OutlierErase,
) -> (BaUpdate, usize) {
    let (map, landmark) = erase_cruise_map(displace_px, octave);
    let params = LocalBaParams {
        visual_information: information,
        outlier_erase,
        ..LocalBaParams::default()
    };
    let update = run_local_inertial_ba_with(
        map.ba_snapshot(&local_window_of(&map, params.active_keyframes, [])),
        &camera(),
        None,
        erase_gravity(),
        &[],
        &[],
        &params,
    );
    (update, landmark)
}

fn chi_square_erase() -> OutlierErase {
    OutlierErase::ChiSquare(Default::default())
}

/// R11b: the displaced observation is the only one listed, under its own
/// keyframe feature, though an unlinked feature precedes it.
#[test]
fn the_erase_lists_exactly_the_displaced_observation() {
    let (update, landmark) = erase_cruise(20.0, 0, VisualInformation::Unit, chi_square_erase());
    assert_eq!(update.outcome.solve, BaSolve::Solved);
    assert_eq!(
        update.outlier_links,
        [OutlierLink {
            key: ObservationKey {
                keyframe_idx: 4,
                feature_idx: 6,
            },
            landmark,
            newest: true,
        }]
    );
    // Every window row of the 12 landmarks: 5 keyframes, the oldest
    // context ones included.
    assert_eq!(update.erase_tested.tested, 60);
    assert_eq!(update.erase_tested.tested_newest, 12);
}

#[test]
fn with_the_erase_off_nothing_is_listed() {
    let (update, _) = erase_cruise(20.0, 0, VisualInformation::Unit, OutlierErase::Off);
    assert_eq!(update.outcome.solve, BaSolve::Solved);
    assert!(update.outlier_links.is_empty());
    assert_eq!(update.erase_tested, Default::default());
}

/// The test is on the whitened residual: 6 px at octave 7 (sigma 3.6 px) is
/// an inlier under `OctaveScaled`, an outlier at unit sigma.
#[test]
fn the_erase_tests_the_residual_whitened_by_the_solve_sigma() {
    let (scaled, _) = erase_cruise(6.0, 7, VisualInformation::OctaveScaled, chi_square_erase());
    assert_eq!(scaled.outcome.solve, BaSolve::Solved);
    assert!(scaled.erase_tested.tested > 0);
    assert!(
        scaled.outlier_links.is_empty(),
        "{:?}",
        scaled.outlier_links
    );
    let (unit, _) = erase_cruise(6.0, 7, VisualInformation::Unit, chi_square_erase());
    assert_eq!(unit.outlier_links.len(), 1, "{:?}", unit.outlier_links);
}

/// Through the synchronous coordinator: the erase follows the accepted merge
/// and is counted; off, the links are those of the plain merge and cull.
#[test]
fn a_synchronous_job_erases_the_displaced_link_and_counts_it() {
    use crate::mapping::{KeyframeJob, LocalMapping, LocalMappingMode};
    use std::sync::{Arc, Mutex};

    let run = |outlier_erase| {
        let (map, landmark) = erase_cruise_map(20.0, 0);
        let map = Arc::new(Mutex::new(map));
        let mapping = LocalMapping::new(LocalMappingMode::Synchronous, Arc::clone(&map), camera());
        assert!(mapping.submit(KeyframeJob {
            imu_initialized: true,
            imu_t_bc: None,
            gravity_world: erase_gravity(),
            local_ba: LocalBaParams {
                outlier_erase,
                ..LocalBaParams::default()
            },
            wheel_factors: Vec::new(),
            tracked_inliers: 0,
        }));
        let (results, counts) = (mapping.drain_results(), mapping.counts());
        drop(mapping);
        let map = Arc::try_unwrap(map).ok().unwrap().into_inner().unwrap();
        (map, landmark, results, counts)
    };

    let (map, landmark, results, counts) = run(chi_square_erase());
    assert_eq!(map.get_keyframe(4).unwrap().map_point(6), None);
    assert!(!map.map_points()[landmark].culled, "four views left");
    assert_eq!(map.map_points()[landmark].observations().len(), 4);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].erase.unlinked, 1);
    assert_eq!(
        (counts.erase_unlinked, counts.erase_unlinked_newest),
        (1, 1)
    );
    assert!(counts.erase_tested > 0);

    let (off, _, results, counts) = run(OutlierErase::Off);
    assert_eq!(results[0].erase, Default::default());
    assert_eq!((counts.erase_tested, counts.erase_unlinked), (0, 0));
    // Off is the plain merge and cull, link for link.
    let (mut plain, _) = erase_cruise_map(20.0, 0);
    let update = run_local_inertial_ba_with(
        plain.ba_snapshot(&local_window_of(&plain, 3, [])),
        &camera(),
        None,
        erase_gravity(),
        &[],
        &[],
        &LocalBaParams::default(),
    );
    plain.apply_ba_update(update).unwrap();
    crate::mapping::culling::cull_landmarks(&mut plain);
    for (a, b) in off.keyframes().iter().zip(plain.keyframes()) {
        assert_eq!(a.map_point_by_desc_idx, b.map_point_by_desc_idx);
    }
    assert_eq!(off.keyframes().len(), plain.keyframes().len());
}
