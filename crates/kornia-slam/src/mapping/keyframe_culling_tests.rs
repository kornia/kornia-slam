use super::*;
use crate::mapping::bundle_adjustment::{local_window, run_local_ba, run_local_inertial_ba};
use crate::mapping::map::tests::{assert_map_consistent, test_frame_with_pose};
use crate::mapping::map::{ImuFactor, MapInsertion};
use crate::mapping::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
use crate::mapping::wheel_odom::WheelOdomMeasurement;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Mat3F64;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

/// Frame ids of the keyframe fixture: non-contiguous so ids and slots differ.
const IDS: [usize; 8] = [3, 7, 11, 15, 19, 23, 27, 31];
/// Points every keyframe sees, then points only `LONER` sees.
const SHARED: usize = 30;
const OWN: usize = 6;
const LONER: usize = 15;

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 300.0,
        fy: 300.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

fn point(i: usize) -> Vec3F64 {
    Vec3F64::new(
        (i % 6) as f64 * 0.4 - 1.0,
        (i / 6 % 5) as f64 * 0.3 - 0.6,
        4.0 + (i % 4) as f64 * 0.7,
    )
}

/// A camera translating 0.1 m per keyframe; every keyframe sees the shared
/// points at octave 0, `LONER` also sees its own. Landmarks start a few cm
/// off the truth so BA has something to do.
fn culling_fixture() -> Map {
    let mut map = Map::new();
    for (slot, id) in IDS.into_iter().enumerate() {
        let pose = Pose3d::new(
            Mat3F64::IDENTITY,
            Vec3F64::new(-0.1 * slot as f64, 0.01 * (slot % 2) as f64, 0.0),
        );
        let mut frame = test_frame_with_pose(id, vec![[id as u8; 32]; SHARED + OWN], pose);
        for i in 0..SHARED + OWN {
            let p = camera()
                .project_to_pixel(&pose.transform_point(&point(i)), 1e-6)
                .unwrap();
            frame.features.keypoints_xy[i] = [p.x as f32, p.y as f32];
        }
        map.insert_keyframe(Keyframe::from_frame(frame)).unwrap();
    }
    for i in 0..SHARED + OWN {
        let reference = if i < SHARED { IDS[0] } else { LONER };
        let id = map
            .insert_landmark(LandmarkSeed {
                position: point(i) + Vec3F64::new(0.03, -0.02, 0.05),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: reference,
                    feature_idx: i,
                },
            })
            .unwrap();
        if i < SHARED {
            for kf in &IDS[1..] {
                map.link_observation(*kf, i, id).unwrap();
            }
        }
    }
    map
}

fn enabled() -> KeyframeCullingConfig {
    KeyframeCullingConfig {
        min_keyframes: 0,
        ..KeyframeCullingConfig::default()
    }
}

fn ids(map: &Map) -> Vec<usize> {
    map.keyframes().iter().map(|kf| kf.frame.idx).collect()
}

#[test]
fn redundant_keyframes_go_and_the_anchor_newest_and_informative_stay() {
    let mut map = culling_fixture();
    let report = cull_redundant_keyframes(&mut map, &enabled(), None);

    // Candidates are 7..=19 (23, 27 and 31 are the protected newest), judged
    // oldest first. 15 holds points nobody else sees, so it stays.
    assert_eq!(report.removed, vec![7, 11, 19]);
    assert_eq!(ids(&map), vec![3, 15, 23, 27, 31]);
    assert_eq!(report.landmarks_retired, 0);
    assert_eq!(report.kept_for_edges, 0);
    assert_map_consistent(&map);
    for mp in map.map_points() {
        assert!(!mp.culled);
        assert!(map.get_keyframe(mp.keyframe_idx).is_some());
    }
    // A second pass has only 15 to judge, and it still holds its own points.
    let again = cull_redundant_keyframes(&mut map, &enabled(), None);
    assert!(again.removed.is_empty());
}

#[test]
fn visual_ba_runs_on_the_culled_map() {
    let mut map = culling_fixture();
    cull_redundant_keyframes(&mut map, &enabled(), None);
    let window = local_window(&map);
    assert_eq!(window.keyframe_slots, vec![0, 1, 2, 3, 4]);
    let update = run_local_ba(map.ba_snapshot(&window), &camera());
    let result = map.apply_ba_update(update).unwrap();
    assert!(!result.keyframe_corrections.is_empty());
    assert!(result.map_points_updated > 0);
    assert_map_consistent(&map);
}

#[test]
fn the_defaults_leave_a_small_map_alone() {
    let mut map = culling_fixture();
    let before = map.state_fingerprint_for_test();
    let report = cull_redundant_keyframes(&mut map, &KeyframeCullingConfig::default(), None);
    assert!(report.removed.is_empty());
    assert_eq!(map.state_fingerprint_for_test(), before);
}

#[test]
fn bounds_cap_the_removals() {
    let mut map = culling_fixture();
    let capped = KeyframeCullingConfig {
        max_removals: 2,
        ..enabled()
    };
    assert_eq!(
        cull_redundant_keyframes(&mut map, &capped, None).removed,
        vec![7, 11]
    );

    let mut map = culling_fixture();
    let floor = KeyframeCullingConfig {
        min_keyframes: 6,
        ..enabled()
    };
    assert_eq!(
        cull_redundant_keyframes(&mut map, &floor, None).removed,
        vec![7, 11]
    );
    assert_eq!(map.keyframes().len(), 6);

    // protect_newest 0 still keeps the newest keyframe.
    let mut map = culling_fixture();
    let unprotected = KeyframeCullingConfig {
        protect_newest: 0,
        redundant_ratio: 0.0,
        min_other_observers: 0,
        ..enabled()
    };
    cull_redundant_keyframes(&mut map, &unprotected, None);
    assert_eq!(ids(&map), vec![3, 31]);
}

#[test]
fn observers_at_a_coarser_scale_do_not_count() {
    let mut map = culling_fixture();
    // Every keyframe but 7 sees the shared points two octaves coarser.
    map.edit_keyframes_for_test(|slot, kf| {
        if slot != 1 {
            kf.frame.features.octaves.iter_mut().for_each(|o| *o = 2);
        }
    });
    let strict = cull_redundant_keyframes(&mut map.clone(), &enabled(), None);
    assert!(!strict.removed.contains(&7), "{:?}", strict.removed);
    let loose = KeyframeCullingConfig {
        scale_level_tolerance: 2,
        ..enabled()
    };
    assert!(
        cull_redundant_keyframes(&mut map, &loose, None)
            .removed
            .contains(&7)
    );
}

fn calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1e-4,
        accel_noise: 1e-3,
        gyro_bias_noise: 1e-5,
        accel_bias_noise: 1e-3,
    }
}

/// Edge between consecutive fixture keyframes, 0.25 s each, from 200 Hz
/// samples of a body at rest under gravity.
fn imu_edge(slot: usize) -> ImuFactor {
    let (t0, t1) = (0.25 * slot as f64, 0.25 * (slot + 1) as f64);
    let raw: Vec<ImuMeasurement> = (0..=50)
        .map(|i| ImuMeasurement {
            timestamp: t0 + i as f64 * 0.005,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, -9.81, 0.0),
        })
        .collect();
    ImuFactor {
        prev_kf_idx: IDS[slot],
        curr_kf_idx: IDS[slot + 1],
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
    }
}

fn inertial_fixture() -> Map {
    let mut map = culling_fixture();
    map.apply_insertion(MapInsertion {
        imu_factors: (0..IDS.len() - 1).map(imu_edge).collect(),
        ..Default::default()
    })
    .unwrap();
    map
}

fn imu_chain(map: &Map) -> Vec<(usize, usize)> {
    map.imu_factors()
        .iter()
        .map(|f| (f.prev_kf_idx, f.curr_kf_idx))
        .collect()
}

#[test]
fn the_imu_chain_is_merged_and_inertial_ba_still_runs() {
    let mut map = inertial_fixture();
    let report = cull_redundant_keyframes(&mut map, &enabled(), None);
    assert_eq!(report.removed, vec![7, 11, 19]);
    assert_eq!(report.imu_edges_merged, 3);
    assert_eq!(imu_chain(&map), vec![(3, 15), (15, 23), (23, 27), (27, 31)]);
    let merged = &map.imu_factors()[0];
    assert_eq!((merged.t0, merged.t1), (0.0, 0.75));
    assert!((merged.preintegrated.dt - 0.75).abs() < 1e-9);
    assert_map_consistent(&map);

    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::new(0.0, 9.81, 0.0),
        &[],
    );
    assert_eq!(update.snapshot.imu_factors().len(), 4);
    map.apply_ba_update(update).unwrap();
    assert_map_consistent(&map);
}

#[test]
fn inertial_keyframes_stay_when_merging_is_off_or_too_long() {
    let mut map = inertial_fixture();
    let before = map.state_fingerprint_for_test();
    let off = KeyframeCullingConfig {
        merge_imu_edges: false,
        ..enabled()
    };
    let report = cull_redundant_keyframes(&mut map, &off, None);
    assert!(report.removed.is_empty());
    assert_eq!(report.kept_for_edges, 3);
    assert_eq!(map.state_fingerprint_for_test(), before);

    // Removing 7 merges 0.5 s; 11 would then need 0.75 s, over the limit.
    let short = KeyframeCullingConfig {
        max_merged_imu_span_sec: 0.6,
        ..enabled()
    };
    let report = cull_redundant_keyframes(&mut map, &short, None);
    assert_eq!(report.removed, vec![7, 19]);
    assert_eq!(report.kept_for_edges, 1);
    assert_eq!(imu_chain(&map)[..2], [(3, 11), (11, 15)]);
    assert_map_consistent(&map);
}

#[test]
fn a_keyframe_without_imu_edges_is_culled_even_with_merging_off() {
    let mut map = culling_fixture();
    let off = KeyframeCullingConfig {
        merge_imu_edges: false,
        ..enabled()
    };
    assert_eq!(
        cull_redundant_keyframes(&mut map, &off, None).removed,
        vec![7, 11, 19]
    );
}

fn wheel(slot: usize) -> WheelOdomFactor {
    WheelOdomFactor {
        from_idx: IDS[slot],
        to_idx: IDS[slot + 1],
        measurement: WheelOdomMeasurement::new(
            [0.1, 0.0, 0.02],
            [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        ),
        t_cb_from: Pose3d::IDENTITY,
        t_cb_to: Pose3d::IDENTITY,
        huber_chi2: None,
    }
}

#[test]
fn wheel_factors_are_composed_across_removed_keyframes() {
    let mut map = inertial_fixture();
    let mut factors: Vec<WheelOdomFactor> = (0..IDS.len() - 1).map(wheel).collect();
    let report = cull_redundant_keyframes(&mut map, &enabled(), Some(&mut factors));
    assert_eq!(report.removed, vec![7, 11, 19]);
    assert_eq!(report.wheel_edges_merged, 3);
    let chain: Vec<_> = factors.iter().map(|f| (f.from_idx, f.to_idx)).collect();
    assert_eq!(chain, vec![(3, 15), (15, 23), (23, 27), (27, 31)]);
    let expected = (0..3)
        .map(|s| wheel(s).measurement)
        .reduce(|a, b| a.compose(&b))
        .unwrap();
    assert_eq!(factors[0].measurement, expected);

    // The composed factor still enters (and moves) inertial BA.
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(&map)),
        &camera(),
        None,
        Vec3F64::new(0.0, 9.81, 0.0),
        &factors,
    );
    map.apply_ba_update(update).unwrap();
    assert_map_consistent(&map);
}

#[test]
fn a_keyframe_whose_wheel_chain_is_cut_is_kept() {
    let mut map = inertial_fixture();
    // The factor out of 11 was dropped for slip: 11 has one side only.
    let mut factors: Vec<WheelOdomFactor> =
        (0..IDS.len() - 1).filter(|&s| s != 2).map(wheel).collect();
    let report = cull_redundant_keyframes(&mut map, &enabled(), Some(&mut factors));
    assert_eq!(report.removed, vec![7, 19]);
    assert_eq!(report.kept_for_edges, 1);
    assert!(map.get_keyframe(11).is_some());
    for f in &factors {
        assert!(map.get_keyframe(f.from_idx).is_some(), "{f:?}");
        assert!(map.get_keyframe(f.to_idx).is_some(), "{f:?}");
    }
    assert_map_consistent(&map);
}
