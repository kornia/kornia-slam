use super::*;
use crate::mapping::map::tests::test_frame;
use crate::mapping::map::{Keyframe, LandmarkSeed};
use kornia_algebra::Vec3F64;

fn erase_on() -> OutlierErase {
    OutlierErase::ChiSquare(ChiSquareErase::default())
}

fn mono(chi2: f64, range_m: f64) -> ObservationResidual {
    ObservationResidual {
        reprojection_chi2: chi2,
        depth_chi2: None,
        depth_m: range_m,
        range_m,
    }
}

fn stereo(chi2: f64, depth_chi2: f64, range_m: f64) -> ObservationResidual {
    ObservationResidual {
        depth_chi2: Some(depth_chi2),
        ..mono(chi2, range_m)
    }
}

#[test]
fn the_rule_is_orb_slam3s_local_inertial_ba_test() {
    let rule = erase_on();
    let cases = [
        ("mono far", mono(5.9, 20.0), false),
        ("mono far", mono(6.1, 20.0), true),
        ("mono close", mono(8.5, 5.0), false),
        ("mono close", mono(9.1, 5.0), true),
        ("stereo far", stereo(5.0, 2.7, 20.0), false),
        ("stereo far", stereo(5.0, 2.9, 20.0), true),
        ("stereo close", stereo(5.0, 2.7, 5.0), false),
        ("stereo close", stereo(5.0, 2.9, 5.0), true),
        ("behind", mono(0.1, -5.0), true),
        ("on the camera plane", mono(0.1, 0.0), true),
    ];
    for (what, residual, erased) in cases {
        assert_eq!(rule.is_outlier(&residual), erased, "{what}: {residual:?}");
    }
    assert!(!OutlierErase::Off.is_outlier(&mono(1e6, 20.0)));
    assert!(!OutlierErase::Off.is_outlier(&mono(0.1, -5.0)));
}

/// Keyframes 0..`count` of `features` features each, the first two with
/// stereo depth when `stereo`; one landmark on feature 0 of every keyframe.
fn linked_map(count: usize, features: usize, stereo: bool) -> (Map, usize) {
    let mut map = Map::new();
    for idx in 0..count {
        let mut frame = test_frame(idx, vec![[idx as u8; 32]; features]);
        if stereo {
            frame.depth = vec![5.0; features];
        }
        map.insert_keyframe(Keyframe::from_frame(frame)).unwrap();
    }
    let landmark = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
        })
        .unwrap();
    for idx in 1..count {
        map.link_observation(idx, 0, landmark).unwrap();
    }
    (map, landmark)
}

fn link(keyframe_idx: usize, feature_idx: usize, landmark: usize) -> OutlierLink {
    OutlierLink {
        key: ObservationKey {
            keyframe_idx,
            feature_idx,
        },
        landmark,
        newest: false,
    }
}

#[test]
fn erase_unlinks_a_live_link_and_keeps_a_landmark_with_three_views() {
    let (mut map, landmark) = linked_map(4, 2, false);
    let report = erase(
        &mut map,
        &[OutlierLink {
            newest: true,
            ..link(3, 0, landmark)
        }],
    );
    assert_eq!(
        report,
        EraseReport {
            unlinked: 1,
            unlinked_newest: 1,
            ..EraseReport::default()
        }
    );
    assert_eq!(map.get_keyframe(3).unwrap().map_point(0), None);
    assert!(!map.map_points()[landmark].culled);
    assert_eq!(map.map_points()[landmark].observations().len(), 3);
}

#[test]
fn erase_retires_a_landmark_left_at_weight_two() {
    // Mono: three views, one erased, two left.
    let (mut map, landmark) = linked_map(3, 2, false);
    let report = erase(&mut map, &[link(2, 0, landmark)]);
    assert_eq!((report.unlinked, report.landmarks_retired), (1, 1));
    assert!(map.map_points()[landmark].culled);
    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);

    // Stereo: two views of weight 2 each, one erased, weight 2 left.
    let (mut map, landmark) = linked_map(2, 2, true);
    let report = erase(&mut map, &[link(1, 0, landmark)]);
    assert_eq!((report.unlinked, report.landmarks_retired), (1, 1));
    assert!(map.map_points()[landmark].culled);

    // Stereo: three views, one erased, weight 4 left.
    let (mut map, landmark) = linked_map(3, 2, true);
    let report = erase(&mut map, &[link(2, 0, landmark)]);
    assert_eq!((report.unlinked, report.landmarks_retired), (1, 0));
    assert!(!map.map_points()[landmark].culled);
}

#[test]
fn erase_skips_links_the_live_map_no_longer_holds() {
    let (mut map, landmark) = linked_map(4, 2, false);
    let other = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(1.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: 1,
            },
        })
        .unwrap();
    // Re-linked since the snapshot: feature 0 of keyframe 3 now holds `other`.
    map.unlink_observation(3, 0).unwrap();
    map.link_observation(3, 0, other).unwrap();
    let before = map.map_points()[landmark].observations().len();
    let report = erase(
        &mut map,
        &[
            link(3, 0, landmark),
            link(99, 0, landmark),
            link(0, 7, landmark),
        ],
    );
    assert_eq!(
        report,
        EraseReport {
            stale_links: 3,
            ..EraseReport::default()
        }
    );
    assert_eq!(map.get_keyframe(3).unwrap().map_point(0), Some(other));
    assert_eq!(map.map_points()[landmark].observations().len(), before);

    // Culled since the snapshot.
    map.remove_landmark(other).unwrap();
    let report = erase(&mut map, &[link(3, 0, other)]);
    assert_eq!(report.stale_links, 1);
    assert_eq!(report.unlinked, 0);
}

#[test]
fn select_outliers_counts_tested_rows_and_skips_guarded_landmarks() {
    let rule = erase_on();
    let (inlier, outlier) = (mono(1.0, 20.0), mono(50.0, 20.0));
    let candidate = |feature, residual, newest, guarded| EraseCandidate {
        link: OutlierLink {
            newest,
            ..link(0, feature, feature)
        },
        fixed_pose: false,
        residual,
        guarded,
    };
    let selection = select_outliers(
        &rule,
        [
            candidate(0, Some(&inlier), true, false),
            candidate(1, Some(&outlier), true, false),
            candidate(2, Some(&outlier), false, true),
            candidate(3, None, false, false),
            candidate(4, Some(&outlier), false, false),
        ],
    );
    assert_eq!(
        selection
            .links
            .iter()
            .map(|l| l.key.feature_idx)
            .collect::<Vec<_>>(),
        [1, 4]
    );
    assert_eq!(
        selection.tested,
        ErasedTested {
            tested: 3,
            tested_newest: 2
        }
    );
    assert_eq!((selection.erased_newest, selection.guard_skipped), (1, 1));
}

#[test]
fn an_empty_erase_reads_nothing() {
    let mut map = Map::new();
    assert_eq!(erase(&mut map, &[]), EraseReport::default());
}
