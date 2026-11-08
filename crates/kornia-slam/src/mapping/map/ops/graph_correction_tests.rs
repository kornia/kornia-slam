use super::PoseGraphCorrectionError;
use crate::mapping::map::tests::{assert_map_consistent, test_frame_with_pose};
use crate::mapping::map::{BaUpdateError, BaWindow, Keyframe, LandmarkSeed, Map, ObservationKey};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

fn pose(rotation: [f64; 3], translation: [f64; 3]) -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(rotation[0], rotation[1], rotation[2])).matrix(),
        Vec3F64::new(translation[0], translation[1], translation[2]),
    )
}

fn assert_pose_close(actual: Pose3d, expected: Pose3d) {
    assert!((actual.translation - expected.translation).length() < 1e-9);
    for (a, e) in actual
        .rotation
        .to_cols_array()
        .iter()
        .zip(expected.rotation.to_cols_array())
    {
        assert!((a - e).abs() < 1e-9);
    }
}

/// Inserts keyframe `idx` at `pose` with a velocity and one landmark it references, seen also by
/// every keyframe already in the map. Returns the landmark id.
fn add_keyframe(map: &mut Map, idx: usize, pose: Pose3d) -> usize {
    let mut keyframe =
        Keyframe::from_frame(test_frame_with_pose(idx, vec![[idx as u8; 32]; 4], pose));
    keyframe.velocity_world = Vec3F64::new(0.5, 0.1 * idx as f64, -0.2);
    map.insert_keyframe(keyframe).unwrap();
    let position = pose
        .inverse()
        .transform_point(&Vec3F64::new(0.2, -0.1, 4.0));
    let landmark = map
        .insert_landmark(LandmarkSeed {
            position,
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: idx,
                feature_idx: 0,
            },
        })
        .unwrap();
    let others: Vec<usize> = map
        .keyframes()
        .iter()
        .map(|kf| kf.frame.idx)
        .filter(|&kf| kf != idx)
        .collect();
    for kf in others {
        map.link_observation(kf, 1 + idx, landmark).unwrap();
    }
    landmark
}

fn three_keyframe_map() -> Map {
    let mut map = Map::new();
    add_keyframe(&mut map, 0, Pose3d::IDENTITY);
    add_keyframe(&mut map, 1, pose([0.0, 0.1, 0.0], [-1.0, 0.0, 0.1]));
    add_keyframe(&mut map, 2, pose([0.02, 0.2, -0.01], [-2.0, 0.1, 0.3]));
    map
}

/// The snapshot of `map` and a solve that moves every keyframe but the first by a growing
/// yaw and shift.
fn snapshot_and_solve(map: &Map) -> (Vec<usize>, Vec<Pose3d>, Vec<Pose3d>) {
    let ids: Vec<usize> = map.keyframes().iter().map(|kf| kf.frame.idx).collect();
    let before: Vec<Pose3d> = map
        .keyframes()
        .iter()
        .map(|kf| kf.frame.pose_world_to_cam)
        .collect();
    let after = before
        .iter()
        .enumerate()
        .map(|(node, b)| {
            let k = node as f64;
            b.compose(&pose([0.0, 0.03 * k, 0.0], [0.05 * k, 0.0, -0.02 * k]))
        })
        .collect();
    (ids, before, after)
}

/// (a) On a map unchanged since the snapshot, the writeback is the synchronous one, bit for bit.
#[test]
fn unchanged_map_matches_the_pose_graph_correction() {
    let mut map = three_keyframe_map();
    let (ids, before, after) = snapshot_and_solve(&map);
    let mut reference = map.clone();
    reference
        .apply_pose_graph_correction(&ids, &before, &after)
        .unwrap();

    let result = map
        .apply_graph_correction(map.world_epoch, &ids, &before, &after)
        .unwrap();

    assert_eq!(
        map.state_fingerprint_for_test(),
        reference.state_fingerprint_for_test()
    );
    let changes: Vec<_> = result
        .keyframes
        .iter()
        .map(|change| (change.kf_idx, change.before, change.after))
        .collect();
    let expected: Vec<_> = ids
        .iter()
        .zip(&before)
        .zip(&after)
        .map(|((&kf, &b), &a)| (kf, b, a))
        .collect();
    assert_eq!(changes, expected);
    assert_map_consistent(&map);
}

/// (b) A keyframe created during the solve follows the last snapshot keyframe: their relative
/// pose, and its landmark in its own camera, are kept.
#[test]
fn a_keyframe_created_during_the_solve_follows_the_frontier() {
    let mut map = Map::new();
    add_keyframe(&mut map, 0, Pose3d::IDENTITY);
    add_keyframe(&mut map, 1, pose([0.0, 0.1, 0.0], [-1.0, 0.0, 0.1]));
    let (ids, before, after) = snapshot_and_solve(&map);
    let epoch = map.world_epoch;
    let landmark = add_keyframe(&mut map, 2, pose([0.02, 0.2, -0.01], [-2.0, 0.1, 0.3]));
    let new_before = map.get_keyframe(2).unwrap().frame.pose_world_to_cam;
    let in_camera = new_before.transform_point(&map.map_points()[landmark].position);

    map.apply_graph_correction(epoch, &ids, &before, &after)
        .unwrap();

    let frontier_after = map.get_keyframe(1).unwrap().frame.pose_world_to_cam;
    let new_after = map.get_keyframe(2).unwrap().frame.pose_world_to_cam;
    assert_eq!(frontier_after, after[1]);
    assert_pose_close(
        Pose3d::between(&frontier_after, &new_after),
        Pose3d::between(&before[1], &new_before),
    );
    let moved = new_after.transform_point(&map.map_points()[landmark].position);
    assert!((moved - in_camera).length() < 1e-9);
}

/// (c) A keyframe bundle adjustment moved during the solve keeps that refinement on top of the
/// solved pose, and its landmark stays where its camera sees it.
#[test]
fn a_keyframe_moved_during_the_solve_keeps_its_refinement() {
    let mut map = three_keyframe_map();
    let (ids, before, after) = snapshot_and_solve(&map);
    let refinement = pose([0.01, -0.005, 0.0], [0.02, 0.0, -0.01]);
    map.set_keyframe_pose_for_test(1, refinement.compose(&before[1]));
    let live = map.get_keyframe(1).unwrap().frame.pose_world_to_cam;
    let landmark = map.keyframes()[1].map_point(0).unwrap();
    let in_camera = live.transform_point(&map.map_points()[landmark].position);

    map.apply_graph_correction(map.world_epoch, &ids, &before, &after)
        .unwrap();

    let corrected = map.get_keyframe(1).unwrap().frame.pose_world_to_cam;
    assert_pose_close(corrected, refinement.compose(&after[1]));
    let moved = corrected.transform_point(&map.map_points()[landmark].position);
    assert!((moved - in_camera).length() < 1e-9);
    assert_eq!(
        map.get_keyframe(2).unwrap().frame.pose_world_to_cam,
        after[2]
    );
}

/// (d) A solve from another world is refused without a write, and an accepted writeback makes a
/// bundle adjustment captured before it stale.
#[test]
fn the_world_epoch_guards_both_directions() {
    let mut map = three_keyframe_map();
    let (ids, before, after) = snapshot_and_solve(&map);
    let untouched = map.state_fingerprint_for_test();

    let error = map
        .apply_graph_correction(map.world_epoch + 1, &ids, &before, &after)
        .unwrap_err();
    assert!(matches!(
        error,
        PoseGraphCorrectionError::ObsoleteWorldFrame
    ));
    assert_eq!(map.state_fingerprint_for_test(), untouched);

    let local_ba = map.ba_snapshot(&BaWindow {
        keyframe_slots: vec![0, 1, 2],
        landmark_ids: vec![0, 1, 2],
    });
    map.apply_graph_correction(map.world_epoch, &ids, &before, &after)
        .unwrap();
    assert_eq!(
        map.apply_ba_update(local_ba.into_update()).unwrap_err(),
        BaUpdateError::ObsoleteWorldFrame
    );
}

/// (e) Snapshot keyframes the map dropped are skipped, and culled landmarks stay where they are.
#[test]
fn removed_keyframes_and_culled_landmarks_are_skipped() {
    let mut map = three_keyframe_map();
    let (ids, before, after) = snapshot_and_solve(&map);
    map.remove_keyframe(2).unwrap();
    let culled = map.keyframes()[0].map_point(0).unwrap();
    map.remove_landmark(culled).unwrap();
    let culled_position = map.map_points()[culled].position;

    let result = map
        .apply_graph_correction(map.world_epoch, &ids, &before, &after)
        .unwrap();

    assert_eq!(
        result
            .keyframes
            .iter()
            .map(|change| change.kf_idx)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        map.get_keyframe(1).unwrap().frame.pose_world_to_cam,
        after[1]
    );
    assert_eq!(map.map_points()[culled].position, culled_position);
    assert_map_consistent(&map);
}
