use super::KeyframeRemovalError;
use crate::mapping::map::tests::{assert_map_consistent, test_frame};
use crate::mapping::map::{
    ImuFactor, Keyframe, LandmarkSeed, Map, MapInsertion, ObservationKey, ops::BaWindow,
};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

fn calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1e-4,
        accel_noise: 1e-3,
        gyro_bias_noise: 1e-5,
        accel_bias_noise: 1e-3,
    }
}

/// 200 Hz of a turning, accelerating body over `[t0, t1]`, both ends
/// included, drawn from one stream so adjacent windows share stamps exactly.
fn samples(t0: f64, t1: f64) -> Vec<ImuMeasurement> {
    let tick = |t: f64| (t * 200.0).round() as usize;
    (tick(t0)..=tick(t1))
        .map(|i| {
            let t = i as f64 * 0.005;
            ImuMeasurement {
                timestamp: t,
                gyro: Vec3F64::new(0.1, -0.3 + t, 0.2),
                accel: Vec3F64::new(0.5 * t, 0.2, 9.81),
            }
        })
        .collect()
}

fn edge(prev: usize, curr: usize, t0: f64, t1: f64) -> ImuFactor {
    let raw = samples(t0, t1);
    ImuFactor {
        prev_kf_idx: prev,
        curr_kf_idx: curr,
        preintegrated: PreintegratedImu::from_measurements(
            ImuBias::default(),
            calib(),
            &raw,
            t0,
            t1,
        ),
        raw_samples: raw,
        t0,
        t1,
        bias_walk_extra_s: 0.0,
    }
}

fn seed(kf: usize, feature: usize) -> LandmarkSeed {
    LandmarkSeed {
        position: Vec3F64::new(0.1 * feature as f64, 0.0, 4.0),
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx: kf,
            feature_idx: feature,
        },
    }
}

/// Keyframes 10, 20, 30, 40. Landmark 0 is seen only by 20; landmark 1 is
/// anchored by 20 and seen by 30 and 40; landmark 2 is anchored by 10 and seen
/// by 20; landmark 3 was seen by 20 and 30 and then retired.
fn landmark_map() -> Map {
    let mut map = Map::new();
    for id in [10, 20, 30, 40] {
        let descriptors = (0..4).map(|f| [(id + f) as u8; 32]).collect();
        map.insert_keyframe(Keyframe::from_frame(test_frame(id, descriptors)))
            .unwrap();
    }
    map.insert_landmark(seed(20, 0)).unwrap();
    let anchored = map.insert_landmark(seed(20, 1)).unwrap();
    map.link_observation(30, 1, anchored).unwrap();
    map.link_observation(40, 1, anchored).unwrap();
    let older = map.insert_landmark(seed(10, 2)).unwrap();
    map.link_observation(20, 2, older).unwrap();
    let retired = map.insert_landmark(seed(20, 3)).unwrap();
    map.link_observation(30, 3, retired).unwrap();
    map.remove_landmark(retired).unwrap();
    map
}

#[test]
fn removal_unlinks_retires_and_promotes_references() {
    let mut map = landmark_map();
    let removal = map.remove_keyframe(20).unwrap();

    assert!(map.get_keyframe(20).is_none());
    assert_eq!(
        map.keyframes()
            .iter()
            .map(|kf| kf.frame.idx)
            .collect::<Vec<_>>(),
        vec![10, 30, 40]
    );
    assert_eq!(removal.observations_removed, 4);
    assert_eq!(removal.landmarks_retired, 1);
    assert_eq!(removal.merged_imu_edge, None);
    assert!(map.map_points()[0].culled, "its only observer went");
    let promoted = &map.map_points()[1];
    assert!(!promoted.culled);
    assert_eq!(promoted.keyframe_idx, 30, "smallest remaining observer");
    assert!(!map.map_points()[2].is_observed_by(20));
    assert_eq!(map.map_points()[2].keyframe_idx, 10);
    assert!(
        !map.map_points()[3].is_observed_by(20),
        "a retired landmark's stale record goes too"
    );
    assert!(map.map_points()[3].is_observed_by(30));
    assert_map_consistent(&map);
}

#[test]
fn an_unknown_keyframe_is_refused_without_a_write() {
    let mut map = landmark_map();
    let before = map.state_fingerprint_for_test();
    assert_eq!(
        map.remove_keyframe(99).unwrap_err(),
        KeyframeRemovalError::UnknownKeyframe(99)
    );
    assert_eq!(map.state_fingerprint_for_test(), before);
}

fn inertial_map() -> Map {
    let mut map = landmark_map();
    map.edit_keyframes_for_test(|slot, kf| kf.imu_bias.gyro.x = 0.01 * (slot + 1) as f64);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![
            edge(10, 20, 0.0, 0.2),
            edge(20, 30, 0.2, 0.4),
            edge(30, 40, 0.4, 0.6),
        ],
        ..Default::default()
    })
    .unwrap();
    map
}

#[test]
fn the_imu_chain_is_merged_into_the_edge_that_skipping_the_keyframe_would_give() {
    let mut map = inertial_map();
    let (first, second) = (map.imu_factors()[0].clone(), map.imu_factors()[1].clone());
    let removal = map.remove_keyframe(20).unwrap();
    assert_eq!(removal.merged_imu_edge, Some((10, 30)));
    assert_map_consistent(&map);

    let edges: Vec<_> = map
        .imu_factors()
        .iter()
        .map(|f| (f.prev_kf_idx, f.curr_kf_idx))
        .collect();
    assert_eq!(
        edges,
        vec![(10, 30), (30, 40)],
        "position of the earlier edge"
    );
    let merged = &map.imu_factors()[0];
    assert_eq!((merged.t0, merged.t1), (0.0, 0.4));
    assert_eq!(merged.raw_samples.len(), samples(0.0, 0.4).len());

    // Re-preintegrated over the whole interval at keyframe 10's bias.
    let bias = map.get_keyframe(10).unwrap().imu_bias;
    let direct = PreintegratedImu::from_measurements(bias, calib(), &samples(0.0, 0.4), 0.0, 0.4);
    assert_eq!(format!("{:?}", merged.preintegrated), format!("{direct:?}"));

    // And it composes the two deltas it replaced (both were linearized at zero
    // bias, so compare against a zero-bias merge).
    let at_zero = PreintegratedImu::from_measurements(
        ImuBias::default(),
        calib(),
        &samples(0.0, 0.4),
        0.0,
        0.4,
    );
    let (a, b) = (&first.preintegrated, &second.preintegrated);
    let rotation = Mat3F64(a.delta_rotation.mul_mat3(&b.delta_rotation));
    let velocity = a.delta_velocity + a.delta_rotation * b.delta_velocity;
    let position = a.delta_position + a.delta_velocity * b.dt + a.delta_rotation * b.delta_position;
    let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-9);
    assert!(close(
        &rotation.to_cols_array(),
        &at_zero.delta_rotation.to_cols_array()
    ));
    assert!(close(
        &velocity.to_array(),
        &at_zero.delta_velocity.to_array()
    ));
    assert!(close(
        &position.to_array(),
        &at_zero.delta_position.to_array()
    ));
    assert!((at_zero.dt - (a.dt + b.dt)).abs() < 1e-12);
}

#[test]
fn an_end_of_the_imu_chain_is_refused_without_a_write() {
    let mut map = inertial_map();
    let before = map.state_fingerprint_for_test();
    for (kf, incoming, outgoing) in [(10, 0, 1), (40, 1, 0)] {
        assert_eq!(
            map.remove_keyframe(kf).unwrap_err(),
            KeyframeRemovalError::UnmergeableImuChain {
                keyframe_idx: kf,
                incoming,
                outgoing,
            }
        );
    }
    assert_eq!(map.state_fingerprint_for_test(), before);
}

/// Removes keyframe 20 from a map holding `edges`, which must be refused
/// with the map untouched; returns the refusal.
fn refusal_with(edges: Vec<ImuFactor>) -> KeyframeRemovalError {
    let mut map = landmark_map();
    map.apply_insertion(MapInsertion {
        imu_factors: edges,
        ..Default::default()
    })
    .unwrap();
    let before = map.state_fingerprint_for_test();
    let error = map.remove_keyframe(20).unwrap_err();
    assert_eq!(map.state_fingerprint_for_test(), before);
    error
}

#[test]
fn unmergeable_edges_are_refused_without_a_write() {
    assert_eq!(
        refusal_with(vec![edge(10, 20, 0.0, 0.2), edge(20, 30, 0.25, 0.4)]),
        KeyframeRemovalError::ImuIntervalGap {
            keyframe_idx: 20,
            t1: 0.2,
            t0: 0.25,
        }
    );
    let mut bare = edge(20, 30, 0.2, 0.4);
    bare.raw_samples.clear();
    assert_eq!(
        refusal_with(vec![edge(10, 20, 0.0, 0.2), bare]),
        KeyframeRemovalError::MissingImuSamples(20)
    );
    assert_eq!(
        refusal_with(vec![
            edge(10, 20, 0.0, 0.2),
            edge(20, 30, 0.2, 0.4),
            edge(10, 30, 0.0, 0.4),
        ]),
        KeyframeRemovalError::DuplicateImuFactor { prev: 10, next: 30 }
    );
}

#[test]
fn a_snapshot_captured_before_the_removal_still_merges_without_resurrecting_it() {
    let mut map = inertial_map();
    let snapshot = map.ba_snapshot(&BaWindow {
        keyframe_slots: (0..4).collect(),
        landmark_ids: (0..4).collect(),
    });
    let mut update = snapshot.into_update();
    for kf in &mut update.keyframes {
        kf.pose_world_to_cam.translation.z += 0.01;
    }
    for preintegrated in &mut update.imu_preintegrations {
        preintegrated.dt += 1.0;
    }
    map.remove_keyframe(20).unwrap();
    let merged = format!("{:?}", map.imu_factors()[0]);

    let result = map.apply_ba_update(update).unwrap();
    assert!(map.get_keyframe(20).is_none());
    assert_eq!(
        result
            .keyframe_corrections
            .iter()
            .map(|c| c.kf_idx)
            .collect::<Vec<_>>(),
        vec![10, 30, 40]
    );
    assert_eq!(
        format!("{:?}", map.imu_factors()[0]),
        merged,
        "the merged edge is not overwritten by a replaced edge's result"
    );
    assert!((map.imu_factors()[1].preintegrated.dt - 1.2).abs() < 1e-9);
    assert_map_consistent(&map);
}

fn gyro_walk(preintegrated: &PreintegratedImu) -> f64 {
    preintegrated.bias_covariance[0]
}

/// A merge re-preintegrates from the samples, which alone give back only the
/// walk over `[t0, t1]`; the still time either edge spans outside them
/// carries over.
///
/// Mutation: dropping `bias_walk_extra_s` from the merged edge turned it red.
#[test]
fn a_merge_keeps_the_bias_walk_over_elided_still_time() {
    let mut map = landmark_map();
    let mut tail = edge(10, 20, 0.0, 0.2);
    tail.bias_walk_extra_s = 3600.0;
    tail.preintegrated.widen_bias_walk(3600.0);
    map.apply_insertion(MapInsertion {
        imu_factors: vec![tail, edge(20, 30, 0.2, 0.4)],
        ..Default::default()
    })
    .unwrap();
    map.remove_keyframe(20).unwrap();
    let merged = &map.imu_factors()[0];
    assert_eq!(merged.bias_walk_extra_s, 3600.0);
    let expected = calib().gyro_bias_noise.powi(2) * (3600.0 + 0.4);
    let got = gyro_walk(&merged.preintegrated);
    assert!(
        (got - expected).abs() < 1e-6 * expected,
        "{got} vs {expected}"
    );
}

/// A keyframe at the end of a stationary chain hands its one IMU edge to the
/// still neighbour, on either side: same interval, samples and deltas, the
/// bias walk widened by the still span. Landmarks are repaired as in any
/// removal.
///
/// Mutation: not widening the moved edge turned the covariance asserts red.
#[test]
fn a_still_boundary_keyframe_moves_its_imu_edge_onto_the_still_neighbour() {
    use super::{StillNeighbour, StillSide};
    for (side, removed, neighbour, edges, moved_pair) in [
        (
            StillSide::Before,
            20,
            10,
            vec![edge(20, 30, 0.2, 0.4), edge(30, 40, 0.4, 0.6)],
            (10, 30),
        ),
        (
            StillSide::After,
            30,
            40,
            vec![edge(10, 20, 0.0, 0.2), edge(20, 30, 0.2, 0.4)],
            (20, 40),
        ),
    ] {
        let mut map = landmark_map();
        map.apply_insertion(MapInsertion {
            imu_factors: edges,
            ..Default::default()
        })
        .unwrap();
        let position = map
            .imu_factors()
            .iter()
            .position(|f| f.prev_kf_idx == removed || f.curr_kf_idx == removed)
            .unwrap();
        let original = map.imu_factors()[position].clone();
        let removal = map
            .remove_keyframe_beside_still(
                removed,
                StillNeighbour {
                    keyframe_idx: neighbour,
                    side,
                    span_s: 50.0,
                },
            )
            .unwrap();
        assert_eq!(removal.merged_imu_edge, Some(moved_pair), "{side:?}");
        assert!(map.get_keyframe(removed).is_none());
        assert_map_consistent(&map);
        let moved = &map.imu_factors()[position];
        assert_eq!((moved.prev_kf_idx, moved.curr_kf_idx), moved_pair);
        assert_eq!((moved.t0, moved.t1), (original.t0, original.t1));
        assert_eq!(moved.raw_samples.len(), original.raw_samples.len());
        assert_eq!(moved.bias_walk_extra_s, 50.0);
        let (a, b) = (&moved.preintegrated, &original.preintegrated);
        assert_eq!(a.delta_position, b.delta_position);
        let widened = calib().gyro_bias_noise.powi(2) * 50.0;
        assert!((gyro_walk(a) - gyro_walk(b) - widened).abs() < 1e-6 * widened);
        let accel = calib().accel_bias_noise.powi(2) * 50.0;
        assert!((a.bias_covariance[35] - b.bias_covariance[35] - accel).abs() < 1e-6 * accel);
    }
}

/// The move is refused, with the map untouched, when the keyframe has an IMU
/// edge towards the still neighbour, none away from it, or the neighbour is
/// unknown, already holds an edge on that side, or the span is not positive.
#[test]
fn a_still_boundary_move_that_would_break_the_chain_is_refused_without_a_write() {
    use super::{StillNeighbour, StillSide};
    let still = |keyframe_idx, side, span_s| StillNeighbour {
        keyframe_idx,
        side,
        span_s,
    };
    let cases: Vec<(Vec<ImuFactor>, StillNeighbour, KeyframeRemovalError)> = vec![
        (
            vec![edge(10, 20, 0.0, 0.2), edge(20, 30, 0.2, 0.4)],
            still(10, StillSide::Before, 5.0),
            KeyframeRemovalError::UnmergeableImuChain {
                keyframe_idx: 20,
                incoming: 1,
                outgoing: 1,
            },
        ),
        (
            vec![edge(30, 40, 0.4, 0.6)],
            still(10, StillSide::Before, 5.0),
            KeyframeRemovalError::UnmergeableImuChain {
                keyframe_idx: 20,
                incoming: 0,
                outgoing: 0,
            },
        ),
        (
            vec![edge(20, 30, 0.2, 0.4), edge(10, 40, 0.0, 0.6)],
            still(10, StillSide::Before, 5.0),
            KeyframeRemovalError::InvalidStillNeighbour {
                keyframe_idx: 20,
                neighbour: 10,
            },
        ),
        (
            vec![edge(20, 30, 0.2, 0.4)],
            still(99, StillSide::Before, 5.0),
            KeyframeRemovalError::InvalidStillNeighbour {
                keyframe_idx: 20,
                neighbour: 99,
            },
        ),
        (
            vec![edge(20, 30, 0.2, 0.4)],
            still(10, StillSide::Before, 0.0),
            KeyframeRemovalError::InvalidStillNeighbour {
                keyframe_idx: 20,
                neighbour: 10,
            },
        ),
        (
            vec![edge(20, 30, 0.2, 0.4)],
            still(30, StillSide::After, 5.0),
            KeyframeRemovalError::UnmergeableImuChain {
                keyframe_idx: 20,
                incoming: 0,
                outgoing: 1,
            },
        ),
    ];
    for (edges, still, expected) in cases {
        let mut map = landmark_map();
        map.apply_insertion(MapInsertion {
            imu_factors: edges,
            ..Default::default()
        })
        .unwrap();
        let before = map.state_fingerprint_for_test();
        assert_eq!(
            map.remove_keyframe_beside_still(20, still).unwrap_err(),
            expected
        );
        assert_eq!(map.state_fingerprint_for_test(), before);
    }
}

#[test]
fn evicting_the_chain_start_drops_its_imu_edge() {
    let mut map = inertial_map();
    let removal = map.evict_keyframe(10).unwrap();
    assert_eq!(removal.dropped_imu_edge, Some((10, 20)));
    assert_eq!(removal.merged_imu_edge, None);
    let edges: Vec<_> = map
        .imu_factors()
        .iter()
        .map(|f| (f.prev_kf_idx, f.curr_kf_idx))
        .collect();
    assert_eq!(edges, vec![(20, 30), (30, 40)]);
    assert!(!map.map_points()[2].is_observed_by(10));
    assert_eq!(map.map_points()[2].keyframe_idx, 20, "reference adopted");
    assert_map_consistent(&map);
}

#[test]
fn eviction_inside_the_imu_chain_is_refused_without_a_write() {
    let mut map = inertial_map();
    let before = map.state_fingerprint_for_test();
    assert_eq!(
        map.evict_keyframe(20).unwrap_err(),
        KeyframeRemovalError::NotImuChainStart {
            keyframe_idx: 20,
            incoming: 1,
            outgoing: 1,
        }
    );
    assert_eq!(map.state_fingerprint_for_test(), before);
}
