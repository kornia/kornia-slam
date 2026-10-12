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
    culling_fixture_with_depth(|_, _| None)
}

/// [`culling_fixture`] with stereo depth: keyframe `id` is stereo when
/// `depth(id, feature)` gives `Some` for any feature (`-1` elsewhere).
fn culling_fixture_with_depth(depth: impl Fn(usize, usize) -> Option<f32>) -> Map {
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
        if (0..SHARED + OWN).any(|i| depth(id, i).is_some()) {
            frame.depth = (0..SHARED + OWN)
                .map(|i| depth(id, i).unwrap_or(-1.0))
                .collect();
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

/// ORB-SLAM3's stereo rule counts only close stereo points: `LONER`'s 30
/// shared and 6 own landmarks score 30/36 under the plain count, but with the
/// close depth set its far or mono-only points drop out of the count.
#[test]
fn the_stereo_close_depth_rule_counts_only_close_stereo_points() {
    let config = |ratio, close| KeyframeCullingConfig {
        redundant_ratio: ratio,
        stereo_close_depth_m: close,
        ..enabled()
    };
    // Shared points close, own points mono-only: 30/30 redundant.
    let shared_close = |id, i| (id == LONER).then_some(if i < SHARED { 2.0 } else { -1.0 });
    let mut map = culling_fixture_with_depth(shared_close);
    assert!(
        !cull_redundant_keyframes(&mut map, &config(0.9, None), None)
            .removed
            .contains(&LONER)
    );
    let mut map = culling_fixture_with_depth(shared_close);
    assert!(
        cull_redundant_keyframes(&mut map, &config(0.9, Some(10.0)), None)
            .removed
            .contains(&LONER)
    );

    // The review's case: shared points far (only 3+ others see them), the
    // own points close and seen by nobody else. 30/36 clears a 0.8 ratio,
    // 0/6 does not.
    let shared_far = |id, i| (id == LONER).then_some(if i < SHARED { 50.0 } else { 2.0 });
    let mut map = culling_fixture_with_depth(shared_far);
    assert!(
        cull_redundant_keyframes(&mut map, &config(0.8, None), None)
            .removed
            .contains(&LONER)
    );
    let mut map = culling_fixture_with_depth(shared_far);
    assert!(
        !cull_redundant_keyframes(&mut map, &config(0.8, Some(10.0)), None)
            .removed
            .contains(&LONER)
    );
    assert_map_consistent(&map);
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

#[test]
fn default_tolerance_counts_observers_one_level_coarser() {
    let mut map = culling_fixture();
    // Every keyframe but 7 sees the shared points one octave coarser.
    map.edit_keyframes_for_test(|slot, kf| {
        if slot != 1 {
            kf.frame.features.octaves.iter_mut().for_each(|o| *o = 1);
        }
    });
    let default = cull_redundant_keyframes(&mut map.clone(), &enabled(), None);
    assert!(default.removed.contains(&7), "{:?}", default.removed);
    let paper = KeyframeCullingConfig {
        scale_level_tolerance: 0,
        ..enabled()
    };
    assert!(
        !cull_redundant_keyframes(&mut map, &paper, None)
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

    // A composed factor still enters inertial BA. (3, 15) joins two fixed
    // context keyframes and is dropped by design; (15, 23), composed across
    // 19, reaches the active window, and dropping it alone moves the
    // solution (its 0.04 rad of turn disagrees with the straight visual track).
    let with = solved_poses(&map, &factors);
    assert_ne!(with, unsolved_poses(&map), "the solve must run");
    assert_eq!((factors[1].from_idx, factors[1].to_idx), (15, 23));
    let without_composed: Vec<WheelOdomFactor> = factors
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != 1)
        .map(|(_, f)| *f)
        .collect();
    let gap = max_pose_gap(&with, &solved_poses(&map, &without_composed));
    assert!(
        gap > 1e-6,
        "the composed factor did not reach the solver: {gap}"
    );

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

/// Window poses after one local inertial BA with `factors`.
fn solved_poses(map: &Map, factors: &[WheelOdomFactor]) -> Vec<Pose3d> {
    let update = run_local_inertial_ba(
        map.ba_snapshot(&local_window(map)),
        &camera(),
        None,
        Vec3F64::new(0.0, 9.81, 0.0),
        factors,
    );
    update
        .keyframes
        .iter()
        .map(|kf| kf.pose_world_to_cam)
        .collect()
}

/// Window poses as captured, what a failed solve returns.
fn unsolved_poses(map: &Map) -> Vec<Pose3d> {
    map.ba_snapshot(&local_window(map))
        .into_update()
        .keyframes
        .iter()
        .map(|kf| kf.pose_world_to_cam)
        .collect()
}

/// Largest translation difference between two pose lists, metres.
fn max_pose_gap(a: &[Pose3d], b: &[Pose3d]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            let d = p.translation - q.translation;
            d.dot(d).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn an_invalid_wheel_factor_is_skipped_not_fatal_to_inertial_ba() {
    let map = inertial_fixture();
    let valid: Vec<WheelOdomFactor> = (0..IDS.len() - 1).map(wheel).collect();
    let reference = solved_poses(&map, &valid);
    assert_ne!(reference, unsolved_poses(&map), "the solve must run");

    // A parked driver's all-zero covariance, a NaN delta from a zero-length
    // tick interval, and a NaN extrinsic from an unread joint.
    let mut zero_cov = wheel(2);
    zero_cov.measurement.covariance = [[0.0; 3]; 3];
    let mut nan_delta = wheel(3);
    nan_delta.measurement.delta[0] = f64::NAN;
    let mut nan_extrinsic = wheel(4);
    nan_extrinsic.t_cb_to.translation.z = f64::NAN;
    let mut with_bad = valid.clone();
    with_bad.extend([zero_cov, nan_delta, nan_extrinsic]);

    // Each is dropped; the rest solve exactly as without them.
    assert_eq!(solved_poses(&map, &with_bad), reference);
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
