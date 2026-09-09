use super::*;
use crate::map::tests::test_frame;
use crate::map::{ImuFactor, LandmarkSeed, MapInsertion, ObservationKey};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement};

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
    let update = run_local_ba(map.ba_snapshot(), &camera());
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
fn initial_ba_uses_the_last_inserted_pair_and_keeps_the_older_pose_fixed() {
    let mut map = fixture();
    let before = map.clone();
    assert!(run_initial_ba(&mut map, &camera()));
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
    assert!(!run_initial_ba(&mut map, &camera()));
    let visual = run_local_ba(map.ba_snapshot(), &camera());
    assert_eq!(map.apply_ba_update(visual).unwrap().map_points_updated, 0);
    let inertial = run_local_inertial_ba(map.ba_snapshot(), &camera(), None, Vec3F64::ZERO);
    assert_eq!(map.apply_ba_update(inertial).unwrap().map_points_updated, 0);
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
    let update = run_local_inertial_ba(map.ba_snapshot(), &camera(), None, Vec3F64::ZERO);
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
    map.edit_keyframes_for_test(|_, kf| kf.imu_bias.accel.x = 0.05);
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
    let update = run_local_inertial_ba(map.ba_snapshot(), &invalid_camera, None, Vec3F64::ZERO);
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
fn stereo_depth_obs_uses_proportional_sigma_with_floor() {
    let mut frame = test_frame(0, vec![[0u8; 32], [1u8; 32]]);
    frame.depth = vec![10.0, -1.0];
    frame.u_right = vec![5.0, -1.0];
    let kf = Keyframe::from_frame(frame);

    // Valid depth: sigma = 0.05 * 10 = 0.5.
    let (d, s) = stereo_depth_obs(&kf, 0);
    assert_eq!(d, Some(10.0));
    assert!((s - 0.5).abs() < 1e-6);

    // Sentinel depth: no measurement.
    let (d1, s1) = stereo_depth_obs(&kf, 1);
    assert_eq!(d1, None);
    assert!((s1 - 1.0).abs() < 1e-9);

    // Very near depth clamps to the sigma floor.
    let mut near = test_frame(1, vec![[0u8; 32]]);
    near.depth = vec![0.1]; // 0.05 * 0.1 = 0.005 < floor
    let kf_near = Keyframe::from_frame(near);
    let (_, s2) = stereo_depth_obs(&kf_near, 0);
    assert!((s2 - STEREO_DEPTH_MIN_SIGMA).abs() < 1e-9);
}
