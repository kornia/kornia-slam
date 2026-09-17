use super::{
    ImuFactor, LandmarkSeed, LandmarkTarget, MapInsertion, MapMutationError, ObservationLink,
};
use crate::map::{
    Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR, ObservationKey,
    tests::{test_frame, test_frame_with_pose},
};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::PreintegratedImu;
use std::collections::HashSet;

#[test]
fn a_stored_keyframe_is_never_silently_replaced() {
    let mut map = Map::new();

    map.insert_keyframe(Keyframe::from_frame(test_frame(
        10,
        vec![[0u8; 32], [1u8; 32]],
    )))
    .unwrap();
    assert_eq!(map.keyframes().len(), 1);

    // `upsert_keyframe` used to overwrite here, discarding the stored
    // keyframe's associations along with it. Insertion refuses instead.
    assert_eq!(
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, vec![[2u8; 32]])))
            .unwrap_err(),
        MapMutationError::DuplicateKeyframe(10)
    );

    assert_eq!(map.keyframes().len(), 1);
    assert_eq!(
        map.get_keyframe(10)
            .expect("expected keyframe with idx 10")
            .frame
            .features
            .descriptors
            .len(),
        2,
        "the stored keyframe is intact"
    );
}

#[test]
fn landmark_ids_are_assigned_in_insertion_order() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();

    let first_idx = map.insert_landmark(seed(0, 0, 1.0)).unwrap();
    let second_idx = map.insert_landmark(seed(0, 1, 1.0)).unwrap();

    assert_eq!(first_idx, 0);
    assert_eq!(second_idx, 1);
    assert_eq!(map.num_map_points(), 2);
    assert_map_consistent(&map);
}

#[test]
fn merge_map_points_redirects_associations_and_deduplicates_observers() {
    let mut map = Map::new();
    for idx in 0..3 {
        map.insert_keyframe(Keyframe::from_frame(test_frame(
            idx,
            vec![[idx as u8; 32], [10 + idx as u8; 32]],
        )))
        .unwrap();
    }

    // Survivor seen by KF0 and KF1; duplicate seen by KF2 and, through a
    // different feature, KF1 — so the merge must both redirect and resolve
    // a shared keyframe.
    let survivor = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.link_observation(1, 0, survivor).unwrap();
    map.set_tracking_stats_for_test(survivor, 7, 5);

    let replaced = map.insert_landmark(seed(2, 0, 5.01)).unwrap();
    map.link_observation(1, 1, replaced).unwrap();
    map.set_tracking_stats_for_test(replaced, 4, 3);
    assert_map_consistent(&map);

    let result = map.merge_map_points(survivor, replaced).unwrap();

    assert_eq!(result.survivor, survivor);
    assert_eq!(result.replaced, replaced);
    assert_eq!(result.redirected_associations, 1);
    assert!(map.map_points()[replaced].culled);
    assert_eq!(map.map_points()[survivor].n_visible, 11);
    assert_eq!(map.map_points()[survivor].n_found, 8);
    assert_eq!(
        map.map_points()[survivor]
            .observer_keyframes()
            .collect::<HashSet<_>>(),
        HashSet::from([0, 1, 2])
    );
    assert_eq!(map.get_keyframe(2).unwrap().map_point(0), Some(survivor));
    assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
    assert_eq!(map.get_keyframe(1).unwrap().map_point(1), None);
    for keyframe in map.keyframes() {
        assert_eq!(
            keyframe
                .map_point_by_desc_idx
                .iter()
                .filter(|&&point| point == Some(survivor))
                .count(),
            1
        );
    }
}

#[test]
fn merge_map_points_rejects_invalid_or_culled_inputs() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32], [1; 32]])))
        .unwrap();
    let first = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let second = map.insert_landmark(seed(0, 1, 5.0)).unwrap();

    assert!(map.merge_map_points(first, first).is_none());
    assert!(map.merge_map_points(first, usize::MAX).is_none());
    // Retired canonically, so its association goes with it — marking the
    // flag alone would leave a link to a retired landmark, a state the
    // invariant forbids.
    map.remove_landmark(second).unwrap();
    assert!(map.merge_map_points(first, second).is_none());
    assert_map_consistent(&map);
}

// ── Scale-invariance state (T1: deterministic, cross-checked vs ORB-SLAM3
//    formulas; ORB-SLAM3 itself not needed since these are closed-form). ──

/// T1b: after `update_map_point_geometry`,
///   max_distance == dist_to_ref_kf * scaleFactor^reference_octave
///   max_distance / min_distance == scaleFactor^(n_levels - 1)
#[test]
fn scale_geometry_distance_invariants() {
    let mut map = Map::new();
    // Reference keyframe 0 at the world origin (identity pose => camera
    // center at origin), its keypoint detected at octave 2.
    let mut kf = Keyframe::from_frame(test_frame(0, vec![[0u8; 32]]));
    kf.frame.features.octaves = vec![2];
    map.insert_keyframe(kf).unwrap();

    // Point referenced to KF 0, world (0,0,5). The octave comes from the
    // referenced feature now, as production does.
    let mp_idx = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

    let mp = &map.map_points()[mp_idx];
    let expected_max = 5.0 * ORB_SCALE_FACTOR.powi(2);
    assert!((mp.max_distance - expected_max).abs() < 1e-9);
    assert!(
        (mp.max_distance / mp.min_distance - ORB_SCALE_FACTOR.powi(ORB_N_LEVELS as i32 - 1)).abs()
            < 1e-9
    );
    // Margined bounds carry the 0.8 / 1.2 factors.
    assert!((mp.min_distance_invariance() - 0.8 * mp.min_distance).abs() < 1e-12);
    assert!((mp.max_distance_invariance() - 1.2 * mp.max_distance).abs() < 1e-12);
}

/// T1c: mean viewing direction is the average of unit (point - cam_center)
/// over observing keyframes; ‖normal‖ <= 1, and for a single forward-facing
/// observation it's exactly (point - cam_center) normalized.
#[test]
fn mean_viewing_direction_averages_observations() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
        .unwrap();

    let mp_idx = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

    // Single observation from the origin looking at (0,0,5): normal = +z.
    let n0 = map.map_points()[mp_idx].mean_viewing_direction;
    assert!(n0.x.abs() < 1e-9 && n0.y.abs() < 1e-9 && (n0.z - 1.0).abs() < 1e-9);

    // Add a second keyframe translated along +x by 1 (world->cam pose has
    // translation -1 along x, so camera center is at (1,0,0)).
    let kf1 = Keyframe::from_frame(test_frame_with_pose(
        1,
        vec![[0u8; 32]],
        Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-1.0, 0.0, 0.0),
        ),
    ));
    map.insert_keyframe(kf1).unwrap();
    map.link_observation(1, 0, mp_idx).unwrap();
    map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

    let n = map.map_points()[mp_idx].mean_viewing_direction;
    // Average of (0,0,1) and (point-(1,0,0)) normalized = (-1,0,5)/sqrt(26).
    let d2 = Vec3F64::new(-1.0, 0.0, 5.0);
    let d2n = d2 / d2.length();
    let expected = Vec3F64::new(
        (n0.x + d2n.x) / 2.0,
        (n0.y + d2n.y) / 2.0,
        (n0.z + d2n.z) / 2.0,
    );
    assert!((n.x - expected.x).abs() < 1e-9);
    assert!((n.y - expected.y).abs() < 1e-9);
    assert!((n.z - expected.z).abs() < 1e-9);
    assert!(n.length() <= 1.0 + 1e-12);
}

// ── canonical mutation API ───────────────────────────────────────────

fn detached(idx: usize, n: usize) -> Keyframe {
    Keyframe::from_frame(test_frame(idx, vec![[idx as u8; 32]; n]))
}

fn seed(kf: usize, feature: usize, z: f64) -> LandmarkSeed {
    LandmarkSeed {
        position: Vec3F64::new(0.0, 0.0, z),
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx: kf,
            feature_idx: feature,
        },
    }
}

/// Every link is mirrored on both sides, and no feature or landmark is
/// claimed twice within a keyframe.
/// The full documented invariant, not a subset. Every rule here is one the
/// module doc promises, so a fixture that violates one is a broken fixture
/// rather than a tolerated shape.
fn assert_map_consistent(map: &Map) {
    // Keyframe side: each association is mirrored by a record, targets an
    // active landmark, and no landmark is claimed twice in one keyframe.
    for kf in map.keyframes() {
        let mut seen: HashSet<usize> = HashSet::new();
        for (feature, slot) in kf.map_point_by_desc_idx.iter().enumerate() {
            let Some(mp_idx) = *slot else { continue };
            let mp = &map.map_points()[mp_idx];
            assert!(!mp.culled, "kf {} links retired {mp_idx}", kf.frame.idx);
            assert!(
                mp.observations()
                    .iter()
                    .any(|o| o.key.keyframe_idx == kf.frame.idx && o.key.feature_idx == feature),
                "kf {} feature {feature} -> {mp_idx} has no matching record",
                kf.frame.idx
            );
            assert!(
                seen.insert(mp_idx),
                "landmark {mp_idx} claimed twice in one keyframe"
            );
        }
    }

    for (mp_idx, mp) in map.map_points().iter().enumerate() {
        if mp.culled {
            assert!(
                mp.observations().is_empty(),
                "retired {mp_idx} kept records"
            );
            assert_eq!(
                (mp.mean_viewing_direction, mp.min_distance, mp.max_distance),
                (Vec3F64::ZERO, 0.0, 0.0),
                "retired {mp_idx} kept derived geometry"
            );
            continue;
        }

        // An active landmark is observed, and its reference is one of those
        // observations rather than a dangling id.
        assert!(
            !mp.observations().is_empty(),
            "active {mp_idx} has no observations"
        );
        assert!(
            mp.is_observed_by(mp.keyframe_idx),
            "active {mp_idx} references kf {} without observing it",
            mp.keyframe_idx
        );

        let mut kfs: HashSet<usize> = HashSet::new();
        for obs in mp.observations() {
            assert!(kfs.insert(obs.key.keyframe_idx), "duplicate observer");
            let kf = map
                .get_keyframe(obs.key.keyframe_idx)
                .expect("observer exists");
            assert_eq!(kf.map_point(obs.key.feature_idx), Some(mp_idx));

            // The referenced feature exists in both arrays, and the record
            // carries that feature's own descriptor.
            assert!(
                obs.key.feature_idx < kf.frame.features.descriptors.len()
                    && obs.key.feature_idx < kf.frame.features.keypoints_xy.len(),
                "record on kf {} feature {} is outside the feature arrays",
                obs.key.keyframe_idx,
                obs.key.feature_idx
            );
            assert_eq!(
                obs.descriptor, kf.frame.features.descriptors[obs.key.feature_idx],
                "record on kf {} feature {} carries a foreign descriptor",
                obs.key.keyframe_idx, obs.key.feature_idx
            );

            // The reference octave agrees with its feature, with the same
            // fallback insertion uses when octave data is absent.
            if obs.key.keyframe_idx == mp.keyframe_idx {
                let expected = kf
                    .frame
                    .features
                    .octaves
                    .get(obs.key.feature_idx)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(
                    mp.reference_octave, expected,
                    "landmark {mp_idx} reference octave disagrees with its feature"
                );
            }
        }
    }

    // IMU edges connect stored keyframes, and each directed edge is unique.
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for factor in map.imu_factors() {
        assert!(
            map.get_keyframe(factor.prev_kf_idx).is_some()
                && map.get_keyframe(factor.curr_kf_idx).is_some(),
            "imu edge {} -> {} has a missing endpoint",
            factor.prev_kf_idx,
            factor.curr_kf_idx
        );
        assert!(
            edges.insert((factor.prev_kf_idx, factor.curr_kf_idx)),
            "duplicate imu edge {} -> {}",
            factor.prev_kf_idx,
            factor.curr_kf_idx
        );
    }
}

#[test]
fn insert_keyframe_rejects_a_duplicate_id() {
    let mut map = Map::new();
    assert_eq!(map.insert_keyframe(detached(7, 2)).unwrap(), 7);
    assert_eq!(
        map.insert_keyframe(detached(7, 2)).unwrap_err(),
        MapMutationError::DuplicateKeyframe(7)
    );
    assert_eq!(map.keyframes().len(), 1);
}

#[test]
fn insert_keyframe_requires_empty_associations() {
    let mut map = Map::new();
    let mut kf = detached(1, 2);
    kf.associate_map_point(0, 3);
    assert_eq!(
        map.insert_keyframe(kf).unwrap_err(),
        MapMutationError::MalformedKeyframe(1)
    );
}

#[test]
fn landmark_ids_are_append_only_and_carry_a_reference_link() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 3)).unwrap();
    let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let b = map.insert_landmark(seed(0, 1, 6.0)).unwrap();
    assert_eq!((a, b), (0, 1));
    assert_eq!(map.map_points()[a].observations().len(), 1);
    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(a));

    map.remove_landmark(a).unwrap();
    let c = map.insert_landmark(seed(0, 2, 7.0)).unwrap();
    assert_eq!(c, 2, "a retired slot is never reused");
    assert_map_consistent(&map);
}

#[test]
fn duplicate_link_is_a_noop() {
    let mut map = Map::new();
    map.insert_keyframe(detached(10, 1)).unwrap();
    let point = map.insert_landmark(seed(10, 0, 5.0)).unwrap();
    assert!(!map.link_observation(10, 0, point).unwrap());
    assert_eq!(map.get_keyframe(10).unwrap().map_point(0), Some(point));
    assert_eq!(map.map_points()[point].observations().len(), 1);
    assert_map_consistent(&map);
}

#[test]
fn conflicting_links_are_refused() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 3)).unwrap();
    let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let b = map.insert_landmark(seed(0, 1, 6.0)).unwrap();

    // Feature 0 already holds `a`.
    assert_eq!(
        map.link_observation(0, 0, b).unwrap_err(),
        MapMutationError::FeatureOccupied {
            keyframe_idx: 0,
            feature_idx: 0,
            holder: a
        }
    );
    // `a` is already seen by keyframe 0 through another feature.
    assert_eq!(
        map.link_observation(0, 2, a).unwrap_err(),
        MapMutationError::DuplicateObservation {
            landmark: a,
            keyframe_idx: 0
        }
    );
    assert_map_consistent(&map);
}

#[test]
fn unknown_and_retired_inputs_are_errors() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 1)).unwrap();
    let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.insert_keyframe(detached(1, 1)).unwrap();

    assert_eq!(
        map.link_observation(9, 0, a).unwrap_err(),
        MapMutationError::UnknownKeyframe(9)
    );
    assert_eq!(
        map.link_observation(1, 5, a).unwrap_err(),
        MapMutationError::InvalidFeature {
            keyframe_idx: 1,
            feature_idx: 5
        }
    );
    assert_eq!(
        map.link_observation(1, 0, 99).unwrap_err(),
        MapMutationError::UnknownLandmark(99)
    );
    map.remove_landmark(a).unwrap();
    assert_eq!(
        map.link_observation(1, 0, a).unwrap_err(),
        MapMutationError::RetiredLandmark(a)
    );
}

#[test]
fn a_two_keyframe_batch_publishes_atomically() {
    let mut map = Map::new();
    let result = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2), detached(1, 2)],
            landmarks: vec![seed(0, 0, 5.0), seed(0, 1, 6.0)],
            observations: vec![
                ObservationLink {
                    observation: ObservationKey {
                        keyframe_idx: 1,
                        feature_idx: 0,
                    },
                    landmark: LandmarkTarget::New(0),
                },
                ObservationLink {
                    observation: ObservationKey {
                        keyframe_idx: 1,
                        feature_idx: 1,
                    },
                    landmark: LandmarkTarget::New(1),
                },
            ],
            imu_factors: Vec::new(),
        })
        .expect("valid batch");

    assert_eq!(result.keyframe_ids, vec![0, 1]);
    assert_eq!(result.landmark_ids, vec![0, 1]);
    assert_eq!(result.observations_added, 4);
    for mp in map.map_points() {
        assert_eq!(mp.observations().len(), 2);
        // Geometry is valid on return, not at some later refresh.
        assert!(mp.max_distance > 0.0);
    }
    assert_map_consistent(&map);
}

#[test]
fn an_invalid_claim_at_the_end_leaves_the_map_untouched() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();
    let existing = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let before = map.state_fingerprint_for_test();

    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(1, 2)],
            landmarks: vec![seed(1, 0, 7.0)],
            observations: vec![ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
                // Last claim in the request: feature 0 of keyframe 0 is
                // already held by `existing`, so this conflicts.
                landmark: LandmarkTarget::New(0),
            }],
            imu_factors: Vec::new(),
        })
        .unwrap_err();
    assert_eq!(
        err,
        MapMutationError::FeatureOccupied {
            keyframe_idx: 0,
            feature_idx: 0,
            holder: existing
        }
    );
    assert_eq!(
        map.state_fingerprint_for_test(),
        before,
        "a rejected batch changed stored state"
    );
    assert_map_consistent(&map);
}

#[test]
fn an_out_of_range_new_target_is_refused() {
    let mut map = Map::new();
    let before = map.state_fingerprint_for_test();
    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2)],
            landmarks: vec![seed(0, 0, 5.0)],
            observations: vec![ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 1,
                },
                landmark: LandmarkTarget::New(4),
            }],
            imu_factors: Vec::new(),
        })
        .unwrap_err();
    assert_eq!(err, MapMutationError::InvalidNewLandmark(4));
    assert_eq!(map.state_fingerprint_for_test(), before);
}

#[test]
fn two_new_claims_on_one_feature_conflict() {
    let mut map = Map::new();
    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2)],
            landmarks: vec![seed(0, 0, 5.0), seed(0, 0, 6.0)],
            observations: Vec::new(),
            imu_factors: Vec::new(),
        })
        .unwrap_err();
    assert!(matches!(err, MapMutationError::FeatureOccupied { .. }));
    assert!(map.keyframes().is_empty());
}

#[test]
fn unlinking_the_last_observation_retires_the_landmark() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();
    map.insert_keyframe(detached(1, 2)).unwrap();
    let point = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.link_observation(1, 0, point).unwrap();

    assert_eq!(map.unlink_observation(0, 0).unwrap(), Some(point));
    assert!(!map.map_points()[point].culled);
    assert_eq!(
        map.map_points()[point].keyframe_idx,
        1,
        "reference moved to the surviving observation"
    );

    assert_eq!(map.unlink_observation(1, 0).unwrap(), Some(point));
    assert!(map.map_points()[point].culled);
    assert!(map.map_points()[point].observations().is_empty());
    assert_eq!(map.map_points()[point].max_distance, 0.0);
    assert_map_consistent(&map);
}

#[test]
fn unlinking_an_empty_slot_is_a_noop() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();
    assert_eq!(map.unlink_observation(0, 1).unwrap(), None);
    assert_eq!(
        map.unlink_observation(0, 9).unwrap_err(),
        MapMutationError::InvalidFeature {
            keyframe_idx: 0,
            feature_idx: 9
        }
    );
}

#[test]
fn removing_a_landmark_clears_every_slot_and_repeats_are_noops() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();
    map.insert_keyframe(detached(1, 2)).unwrap();
    let point = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let other = map.insert_landmark(seed(0, 1, 6.0)).unwrap();
    map.link_observation(1, 0, point).unwrap();

    assert!(map.remove_landmark(point).unwrap());
    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
    assert_eq!(map.get_keyframe(1).unwrap().map_point(0), None);
    assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(other));
    assert!(!map.remove_landmark(point).unwrap());
    assert_eq!(
        map.remove_landmark(99).unwrap_err(),
        MapMutationError::UnknownLandmark(99)
    );
    assert_map_consistent(&map);
}

#[test]
fn an_invalid_imu_edge_rejects_the_batch() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 1)).unwrap();
    let factor = |prev, curr, t0: f64, t1: f64| ImuFactor {
        prev_kf_idx: prev,
        curr_kf_idx: curr,
        preintegrated: PreintegratedImu::new(Default::default(), test_calib()),
        raw_samples: Vec::new(),
        t0,
        t1,
    };
    let mut request = MapInsertion {
        keyframes: vec![detached(1, 1)],
        imu_factors: vec![factor(0, 0, 0.0, 1.0)],
        ..Default::default()
    };
    assert_eq!(
        map.apply_insertion(request).unwrap_err(),
        MapMutationError::SelfImuFactor(0)
    );
    request = MapInsertion {
        keyframes: vec![detached(1, 1)],
        imu_factors: vec![factor(0, 9, 0.0, 1.0)],
        ..Default::default()
    };
    assert_eq!(
        map.apply_insertion(request).unwrap_err(),
        MapMutationError::UnknownKeyframe(9)
    );
    request = MapInsertion {
        keyframes: vec![detached(1, 1)],
        imu_factors: vec![factor(0, 1, 1.0, 1.0)],
        ..Default::default()
    };
    assert!(matches!(
        map.apply_insertion(request).unwrap_err(),
        MapMutationError::InvalidImuInterval { .. }
    ));
    assert!(
        map.get_keyframe(1).is_none(),
        "batch left no keyframe behind"
    );
}

fn test_calib() -> kornia_sensors::imu::ImuCalib {
    kornia_sensors::imu::ImuCalib {
        gyro_noise: 1.0e-4,
        accel_noise: 1.0e-3,
        gyro_bias_noise: 1.0e-5,
        accel_bias_noise: 1.0e-3,
    }
}

#[test]
fn merge_redirects_disjoint_observers_and_keeps_counters() {
    let mut map = Map::new();
    for idx in 0..2 {
        map.insert_keyframe(detached(idx, 2)).unwrap();
    }
    let survivor = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let duplicate = map.insert_landmark(seed(1, 0, 5.01)).unwrap();
    map.set_tracking_stats_for_test(survivor, 7, 5);
    map.set_tracking_stats_for_test(duplicate, 4, 3);

    let result = map.merge_map_points(survivor, duplicate).unwrap();

    assert_eq!(result.survivor, survivor);
    assert_eq!(result.redirected_associations, 1);
    assert_eq!(map.map_points()[survivor].n_visible, 11);
    assert_eq!(map.map_points()[survivor].n_found, 8);
    assert!(map.map_points()[duplicate].culled);
    // The duplicate's observer now points at the survivor, on the same
    // feature, with a matching record.
    assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
    assert!(map.map_points()[survivor].is_observed_by(1));
    assert_map_consistent(&map);
}

/// Both landmarks seen in one keyframe through different features: the
/// survivor keeps its own feature, the duplicate's is released.
#[test]
fn merge_keeps_the_survivors_feature_in_a_shared_keyframe() {
    let mut map = Map::new();
    map.insert_keyframe(detached(0, 2)).unwrap();
    let survivor = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let duplicate = map.insert_landmark(seed(0, 1, 5.01)).unwrap();
    let survivor_descriptor = map.map_points()[survivor].descriptor;

    map.merge_map_points(survivor, duplicate).unwrap();

    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(survivor));
    assert_eq!(map.get_keyframe(0).unwrap().map_point(1), None);
    assert_eq!(map.map_points()[survivor].observations().len(), 1);
    assert_eq!(
        map.map_points()[survivor].observations()[0].key.feature_idx,
        0
    );
    assert_eq!(
        map.map_points()[survivor].descriptor,
        survivor_descriptor,
        "kept its retained feature's contribution"
    );
    assert_map_consistent(&map);
}

/// The weaker landmark is the one retired, and the reference follows the
/// survivor's remaining observations.
#[test]
fn merge_picks_the_better_supported_survivor() {
    let mut map = Map::new();
    for idx in 0..3 {
        map.insert_keyframe(detached(idx, 2)).unwrap();
    }
    let weak = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    let strong = map.insert_landmark(seed(1, 0, 5.01)).unwrap();
    map.link_observation(2, 0, strong).unwrap();

    // Named weakest-first; support decides, not argument order.
    let result = map.merge_map_points(weak, strong).unwrap();

    assert_eq!(result.survivor, strong);
    assert_eq!(result.replaced, weak);
    assert!(map.map_points()[weak].culled);
    assert_eq!(map.map_points()[strong].observations().len(), 3);
    assert_map_consistent(&map);
}

// ── review findings R1–R4 ────────────────────────────────────────────

/// R1: unlinking an observation that is not the reference must leave the
/// reference keyframe and its octave alone.
#[test]
fn unlinking_a_non_reference_observation_keeps_the_reference() {
    let mut map = Map::new();
    for idx in [10usize, 20, 30] {
        let mut kf = detached(idx, 2);
        // Distinct octaves make a wrongly-adopted reference visible.
        kf.frame.features.octaves = vec![(idx / 10) as u8, 0];
        map.insert_keyframe(kf).unwrap();
    }
    let point = map.insert_landmark(seed(20, 0, 5.0)).unwrap();
    map.link_observation(10, 0, point).unwrap();
    map.link_observation(30, 0, point).unwrap();
    assert_eq!(map.map_points()[point].keyframe_idx, 20);
    assert_eq!(map.map_points()[point].reference_octave, 2);

    map.unlink_observation(30, 0).unwrap();

    assert_eq!(
        map.map_points()[point].keyframe_idx,
        20,
        "the reference observation was not the one removed"
    );
    assert_eq!(map.map_points()[point].reference_octave, 2);
    assert_eq!(map.map_points()[point].observations().len(), 2);
    assert_map_consistent(&map);
}

/// R1: removing the reference itself promotes the smallest remaining
/// (keyframe, feature) and adopts its octave.
#[test]
fn unlinking_the_reference_promotes_the_smallest_remaining() {
    let mut map = Map::new();
    for idx in [10usize, 20, 30] {
        let mut kf = detached(idx, 2);
        kf.frame.features.octaves = vec![(idx / 10) as u8, 0];
        map.insert_keyframe(kf).unwrap();
    }
    let point = map.insert_landmark(seed(20, 0, 5.0)).unwrap();
    map.link_observation(10, 0, point).unwrap();
    map.link_observation(30, 0, point).unwrap();

    map.unlink_observation(20, 0).unwrap();

    assert_eq!(map.map_points()[point].keyframe_idx, 10);
    assert_eq!(map.map_points()[point].reference_octave, 1);
    assert_map_consistent(&map);
}

/// R2: a duplicate directed IMU edge inside one request rejects the batch,
/// leaving nothing behind — it would otherwise double-count integrated
/// duration in the initialization readiness gate.
#[test]
fn a_duplicate_imu_edge_within_one_batch_is_refused() {
    let mut map = Map::new();
    map.insert_keyframe(detached(10, 1)).unwrap();
    let edge = |prev, curr| ImuFactor {
        prev_kf_idx: prev,
        curr_kf_idx: curr,
        preintegrated: PreintegratedImu::new(Default::default(), test_calib()),
        raw_samples: Vec::new(),
        t0: 0.0,
        t1: 1.0,
    };
    let before = map.state_fingerprint_for_test();

    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(11, 1)],
            landmarks: vec![seed(10, 0, 5.0)],
            imu_factors: vec![edge(10, 11), edge(10, 11)],
            ..Default::default()
        })
        .unwrap_err();

    assert_eq!(
        err,
        MapMutationError::DuplicateImuFactor { prev: 10, curr: 11 }
    );
    assert_eq!(
        map.state_fingerprint_for_test(),
        before,
        "the whole request was refused, factors included"
    );
}

/// R3: proposing the same link twice in one request is a no-op, not a
/// conflict — including a seed's implicit reference restated explicitly.
#[test]
fn identical_claims_within_a_batch_are_noops() {
    let mut map = Map::new();
    let link = |kf, feature, target| ObservationLink {
        observation: ObservationKey {
            keyframe_idx: kf,
            feature_idx: feature,
        },
        landmark: target,
    };

    let result = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2), detached(1, 2)],
            landmarks: vec![seed(0, 0, 5.0)],
            observations: vec![
                // The seed already implies this link.
                link(0, 0, LandmarkTarget::New(0)),
                link(1, 0, LandmarkTarget::New(0)),
                // And the same second link, restated.
                link(1, 0, LandmarkTarget::New(0)),
            ],
            ..Default::default()
        })
        .expect("identical repeats are not conflicts");

    let point = result.landmark_ids[0];
    assert_eq!(
        result.observations_added, 2,
        "each real link counted exactly once"
    );
    assert_eq!(map.map_points()[point].observations().len(), 2);
    assert_map_consistent(&map);

    // A different landmark wanting a claimed feature is still a conflict.
    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(2, 2)],
            landmarks: vec![seed(2, 0, 6.0)],
            observations: vec![link(0, 0, LandmarkTarget::New(0))],
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, MapMutationError::FeatureOccupied { .. }));
}

/// R4: a feature with a descriptor but no keypoint carries no image
/// measurement, so it cannot anchor an observation.
#[test]
fn a_feature_without_a_keypoint_is_refused() {
    let mut map = Map::new();
    let mut kf = detached(0, 1);
    kf.frame.features.keypoints_xy.clear();
    map.insert_keyframe(kf).unwrap();

    assert_eq!(
        map.insert_landmark(seed(0, 0, 5.0)).unwrap_err(),
        MapMutationError::InvalidFeature {
            keyframe_idx: 0,
            feature_idx: 0
        }
    );
    assert_eq!(map.num_map_points(), 0);

    let mut detached_kf = detached(1, 1);
    detached_kf.frame.features.keypoints_xy.clear();
    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached_kf],
            landmarks: vec![seed(1, 0, 5.0)],
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(
        err,
        MapMutationError::InvalidFeature {
            keyframe_idx: 1,
            feature_idx: 0
        }
    );
    assert!(map.get_keyframe(1).is_none());
}

/// R4: absent octave data keeps its existing fallback to 0.
#[test]
fn a_valid_feature_without_octave_data_keeps_the_fallback() {
    let mut map = Map::new();
    let mut kf = detached(0, 1);
    kf.frame.features.octaves.clear();
    map.insert_keyframe(kf).unwrap();

    let point = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    assert_eq!(map.map_points()[point].reference_octave, 0);
    assert_map_consistent(&map);
}

/// The representative descriptor must be right after a batch, which is the
/// only place finalization now happens: `add_observation` records without
/// selecting. Removing the finalize call breaks this and nothing else.
#[test]
fn a_batch_finalizes_the_representative_descriptor() {
    // Four descriptors spaced four bits apart along a line: medians of the
    // pairwise distances are 8, 4, 4, 8, so the second wins on the
    // first-of-equals tie-break.
    let d0 = [0u8; 32];
    let mut d4 = [0u8; 32];
    d4[0] = 0b0000_1111;
    let mut d8 = [0u8; 32];
    d8[0] = 0b1111_1111;
    let mut d12 = [0u8; 32];
    d12[0] = 0b1111_1111;
    d12[1] = 0b0000_1111;

    let mut map = Map::new();
    let mut request = MapInsertion::default();
    for (idx, descriptor) in [d0, d4, d8, d12].into_iter().enumerate() {
        let mut kf = detached(idx, 1);
        kf.frame.features.descriptors = vec![descriptor];
        request.keyframes.push(kf);
        if idx == 0 {
            request.landmarks.push(seed(0, 0, 5.0));
        } else {
            request.observations.push(ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: idx,
                    feature_idx: 0,
                },
                landmark: LandmarkTarget::New(0),
            });
        }
    }
    let landmark = map
        .apply_insertion(request)
        .expect("valid batch")
        .landmark_ids[0];

    assert_eq!(map.map_points()[landmark].observations().len(), 4);
    assert_eq!(
        map.map_points()[landmark].descriptor,
        d4,
        "the batch selected a representative over all four records"
    );
    assert_map_consistent(&map);
}

/// The single-operation path finalizes too: two records fall back to the
/// first, and a third can move the winner.
#[test]
fn a_single_link_finalizes_the_representative_descriptor() {
    let d0 = [0u8; 32];
    let mut d8 = [0u8; 32];
    d8[0] = 0b1111_1111;
    let mut d12 = [0u8; 32];
    d12[0] = 0b1111_1111;
    d12[1] = 0b0000_1111;

    let mut map = Map::new();
    for (idx, descriptor) in [d0, d12, d8].into_iter().enumerate() {
        let mut kf = detached(idx, 1);
        kf.frame.features.descriptors = vec![descriptor];
        map.insert_keyframe(kf).unwrap();
    }
    let landmark = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
    map.link_observation(1, 0, landmark).unwrap();
    assert_eq!(
        map.map_points()[landmark].descriptor,
        d0,
        "two records take the first"
    );

    map.link_observation(2, 0, landmark).unwrap();
    assert_eq!(
        map.map_points()[landmark].descriptor,
        d12,
        "a third record moves the median winner off the first"
    );
    assert_map_consistent(&map);
}

/// Two proposals naming one feature is a conflict, so a caller preparing a
/// batch from geometry must resolve it rather than hand the map a request
/// that refuses everything. Monocular bootstrap hit exactly this: the
/// two-view solve can triangulate two points onto one keypoint, and the
/// whole bootstrap was being rejected over it.
#[test]
fn two_landmarks_naming_one_feature_refuse_the_whole_batch() {
    let mut map = Map::new();
    let err = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2)],
            landmarks: vec![seed(0, 0, 5.0), seed(0, 0, 6.0)],
            ..Default::default()
        })
        .unwrap_err();

    assert!(matches!(err, MapMutationError::FeatureOccupied { .. }));
    assert!(
        map.keyframes().is_empty(),
        "the keyframe went back with the rest of the request"
    );

    // Resolved by the caller — first proposal keeps the feature — the same
    // geometry publishes cleanly.
    let result = map
        .apply_insertion(MapInsertion {
            keyframes: vec![detached(0, 2)],
            landmarks: vec![seed(0, 0, 5.0), seed(0, 1, 6.0)],
            ..Default::default()
        })
        .expect("distinct features publish");
    assert_eq!(result.landmark_ids.len(), 2);
    assert_map_consistent(&map);
}
