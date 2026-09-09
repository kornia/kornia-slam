//! Acceptance of a freshly bootstrapped two-keyframe map.
//!
//! The metric and its thresholds live together here because they are one
//! decision: the numbers are only meaningful against the cutoffs they were
//! chosen for. The system coordinates cleanup and the mode transition, but does
//! not reinterpret the fields.

use crate::map::Map;
use std::collections::HashSet;

/// Minimum landmarks visible in front of both bootstrap keyframes.
///
/// ORB-SLAM3's `CreateInitialMapMonocular` reset criterion is
/// `medianDepth < 0 || TrackedMapPoints(1) < 50`.
const MIN_POINTS_IN_FRONT_OF_BOTH: usize = 50;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BootstrapQuality {
    /// Landmarks observed by either keyframe that sit in front of both.
    pub points_in_front_of_both: usize,
    /// Median depth of those landmarks in the reference camera.
    pub median_depth_reference: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BootstrapReject {
    /// One of the named keyframes is not in the map.
    MissingKeyframe(usize),
    /// Reference and current are the same keyframe; there is no baseline.
    DegeneratePair(usize),
    TooFewPoints {
        found: usize,
        required: usize,
    },
    /// Non-positive median depth: the map is behind the reference camera or
    /// has collapsed.
    DegenerateDepth(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BootstrapOutcome {
    Accepted(BootstrapQuality),
    /// Quality is carried when it could be measured, so a caller can log why a
    /// valid pair still failed rather than only that it did.
    Rejected {
        reason: BootstrapReject,
        quality: Option<BootstrapQuality>,
    },
}

/// Measures the bootstrap pair and applies the acceptance thresholds.
///
/// The pair is named explicitly rather than taken as the last two keyframes, so
/// a keyframe inserted after the solve cannot silently change what is judged.
///
/// The population is the *union* of landmarks associated with either keyframe —
/// not the intersection of observers — counted when depth is positive in both
/// cameras. Median depth is measured in the reference camera.
pub(crate) fn evaluate_bootstrap(
    map: &Map,
    reference_kf_idx: usize,
    current_kf_idx: usize,
) -> BootstrapOutcome {
    if reference_kf_idx == current_kf_idx {
        return BootstrapOutcome::Rejected {
            reason: BootstrapReject::DegeneratePair(reference_kf_idx),
            quality: None,
        };
    }
    let (Some(reference), Some(current)) = (
        map.get_keyframe(reference_kf_idx),
        map.get_keyframe(current_kf_idx),
    ) else {
        let missing = if map.get_keyframe(reference_kf_idx).is_none() {
            reference_kf_idx
        } else {
            current_kf_idx
        };
        return BootstrapOutcome::Rejected {
            reason: BootstrapReject::MissingKeyframe(missing),
            quality: None,
        };
    };

    let pose_reference = reference.frame.pose_world_to_cam;
    let pose_current = current.frame.pose_world_to_cam;

    let observed: HashSet<usize> = reference
        .map_point_by_desc_idx
        .iter()
        .chain(current.map_point_by_desc_idx.iter())
        .flatten()
        .copied()
        .collect();

    let mut depths_reference: Vec<f64> = Vec::with_capacity(observed.len());
    for idx in observed {
        let Some(mp) = map.map_points().get(idx) else {
            continue;
        };
        if mp.culled {
            continue;
        }
        let z_reference = pose_reference.transform_point(&mp.position).z;
        let z_current = pose_current.transform_point(&mp.position).z;
        if z_reference > 0.0 && z_current > 0.0 {
            depths_reference.push(z_reference);
        }
    }

    let points_in_front_of_both = depths_reference.len();
    let median_depth_reference = if depths_reference.is_empty() {
        0.0
    } else {
        let mid = depths_reference.len() / 2;
        depths_reference.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        depths_reference[mid]
    };
    let quality = BootstrapQuality {
        points_in_front_of_both,
        median_depth_reference,
    };

    if points_in_front_of_both < MIN_POINTS_IN_FRONT_OF_BOTH {
        return BootstrapOutcome::Rejected {
            reason: BootstrapReject::TooFewPoints {
                found: points_in_front_of_both,
                required: MIN_POINTS_IN_FRONT_OF_BOTH,
            },
            quality: Some(quality),
        };
    }
    if median_depth_reference <= 0.0 {
        return BootstrapOutcome::Rejected {
            reason: BootstrapReject::DegenerateDepth(median_depth_reference),
            quality: Some(quality),
        };
    }
    BootstrapOutcome::Accepted(quality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::map::{
        Keyframe, LandmarkSeed, LandmarkTarget, MapInsertion, ObservationKey, ObservationLink,
    };
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn frame(idx: usize, n_desc: usize, translation: Vec3F64) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n_desc).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n_desc],
                descriptors: vec![[0u8; 32]; n_desc],
                octaves: vec![0; n_desc],
            },
            pose_world_to_cam: Pose3d::new(Mat3F64::IDENTITY, translation),
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n_desc],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    /// Two keyframes at the origin and slightly offset, with `n` landmarks five
    /// metres ahead of both, each associated to a distinct feature slot.
    fn bootstrap_map(n: usize) -> Map {
        let mut map = Map::new();
        let mut request = MapInsertion {
            keyframes: vec![
                Keyframe::from_frame(frame(0, n, Vec3F64::ZERO)),
                Keyframe::from_frame(frame(1, n, Vec3F64::new(-0.1, 0.0, 0.0))),
            ],
            ..Default::default()
        };
        for i in 0..n {
            request.landmarks.push(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0 + i as f64 * 0.01),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: i,
                },
            });
            request.observations.push(ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: 1,
                    feature_idx: i,
                },
                landmark: LandmarkTarget::New(i),
            });
        }
        map.apply_insertion(request).expect("valid fixture");
        map
    }

    #[test]
    fn accepts_a_healthy_pair_and_measures_depth_in_the_reference_camera() {
        let map = bootstrap_map(60);
        match evaluate_bootstrap(&map, 0, 1) {
            BootstrapOutcome::Accepted(q) => {
                assert_eq!(q.points_in_front_of_both, 60);
                assert!((q.median_depth_reference - 5.30).abs() < 0.02);
            }
            other => panic!("expected acceptance, got {other:?}"),
        }
    }

    #[test]
    fn judges_the_named_pair_despite_a_later_keyframe() {
        let mut map = bootstrap_map(60);
        // A third keyframe arrives with no associations. Evaluating the last two
        // would now see zero points; naming the pair keeps the verdict stable.
        map.insert_keyframe(Keyframe::from_frame(frame(
            2,
            0,
            Vec3F64::new(-0.2, 0.0, 0.0),
        )))
        .unwrap();
        assert!(matches!(
            evaluate_bootstrap(&map, 0, 1),
            BootstrapOutcome::Accepted(_)
        ));
    }

    #[test]
    fn counts_the_union_of_observers_not_the_intersection() {
        let mut map = bootstrap_map(60);
        // Drop the association from the current keyframe only. The landmark is
        // still in front of both cameras, so the union keeps counting it.
        map.unlink_observation(1, 0).unwrap();
        match evaluate_bootstrap(&map, 0, 1) {
            BootstrapOutcome::Accepted(q) => assert_eq!(q.points_in_front_of_both, 60),
            other => panic!("expected acceptance, got {other:?}"),
        }
    }

    #[test]
    fn rejects_just_below_the_point_threshold() {
        let map = bootstrap_map(49);
        match evaluate_bootstrap(&map, 0, 1) {
            BootstrapOutcome::Rejected {
                reason: BootstrapReject::TooFewPoints { found, required },
                quality: Some(q),
            } => {
                assert_eq!((found, required), (49, 50));
                assert!(q.median_depth_reference > 0.0);
            }
            other => panic!("expected too-few-points rejection, got {other:?}"),
        }
        assert!(matches!(
            evaluate_bootstrap(&bootstrap_map(50), 0, 1),
            BootstrapOutcome::Accepted(_)
        ));
    }

    #[test]
    fn a_valid_pair_with_no_usable_points_is_distinct_from_a_missing_one() {
        let map = bootstrap_map(0);
        assert!(matches!(
            evaluate_bootstrap(&map, 0, 1),
            BootstrapOutcome::Rejected {
                reason: BootstrapReject::TooFewPoints { found: 0, .. },
                ..
            }
        ));
        assert!(matches!(
            evaluate_bootstrap(&map, 0, 99),
            BootstrapOutcome::Rejected {
                reason: BootstrapReject::MissingKeyframe(99),
                quality: None,
            }
        ));
        assert!(matches!(
            evaluate_bootstrap(&map, 0, 0),
            BootstrapOutcome::Rejected {
                reason: BootstrapReject::DegeneratePair(0),
                quality: None,
            }
        ));
    }

    #[test]
    fn points_behind_either_camera_do_not_count() {
        let mut map = bootstrap_map(60);
        // Push the current keyframe past the landmarks so they fall behind it.
        map.set_keyframe_pose_for_test(
            1,
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -20.0)),
        );
        match evaluate_bootstrap(&map, 0, 1) {
            BootstrapOutcome::Rejected {
                reason: BootstrapReject::TooFewPoints { found: 0, .. },
                ..
            } => {}
            other => panic!("expected zero usable points, got {other:?}"),
        }
    }
}
