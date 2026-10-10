//! Landmark culling policy.
//!
//! Selection lives here because the thresholds are a mapping judgement about
//! which landmarks have earned their place. Removal itself is the map's
//! canonical operation, so this selects ids and delegates — there is no second
//! cleanup loop, and the map never calls back into mapping.

use crate::mapping::map::Map;
use std::collections::HashSet;

/// Minimum times a landmark must have been in view before its match rate is
/// judged at all.
const MIN_OBSERVATIONS: u32 = 5;
/// Match rate below which a sufficiently-observed landmark is dropped.
const MIN_FOUND_RATIO: f64 = 0.20;

/// Ids of the landmarks that fail the culling policy.
///
/// Two criteria: a poor found ratio once a landmark has been seen
/// enough times to judge, and a landmark sitting behind its reference keyframe.
fn select_for_culling(map: &Map) -> Vec<usize> {
    let mut selected: HashSet<usize> = HashSet::new();

    for (idx, mp) in map.map_points().iter().enumerate() {
        if mp.culled {
            continue;
        }

        // Criterion 1: Low found ratio once a landmark has been observed enough times to judge.
        if mp.n_visible >= MIN_OBSERVATIONS && mp.found_ratio() < MIN_FOUND_RATIO {
            selected.insert(idx);
            continue;
        }

        // Criterion 2: Landmark sitting behind its reference keyframe.
        // A landmark's scale geometry and initial coordinate frame are anchored
        // to its reference keyframe. Checking depth relative to its reference
        // keyframe prevents over-culling points that are valid in front of other
        // keyframes as the camera moves past them or during local pose adjustments.
        if let Some(ref_kf) = map.get_keyframe(mp.keyframe_idx) {
            let p_cam = ref_kf.frame.pose_world_to_cam.transform_point(&mp.position);
            if p_cam.z <= 1e-8 {
                selected.insert(idx);
            }
        }
    }

    let mut selected: Vec<usize> = selected.into_iter().collect();
    selected.sort_unstable();
    selected
}

/// Culls landmarks that fail the policy. Returns how many were retired.
///
/// Held under exclusive access so selection and removal see one state.
pub fn cull_landmarks(map: &mut Map) -> usize {
    let mut retired = 0usize;
    for idx in select_for_culling(map) {
        if map.remove_landmark(idx).unwrap_or(false) {
            retired += 1;
        }
    }
    retired
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    #[test]
    fn culls_low_found_ratio_and_clears_its_associations() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        let seed = |feature| LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: feature,
            },
        };
        let doomed = map.insert_landmark(seed(0)).unwrap();
        let kept = map.insert_landmark(seed(1)).unwrap();

        map.set_tracking_stats_for_test(doomed, 10, 1);
        map.set_tracking_stats_for_test(kept, 10, 5);

        assert_eq!(cull_landmarks(&mut map), 1);
        assert!(map.map_points()[doomed].culled);
        assert!(!map.map_points()[kept].culled);
        // Removal cleared the association; no second cleanup pass needed.
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(kept));
        // Retirement is logical: the landmark keeps its records.
        assert!(map.map_points()[doomed].is_observed_by(0));
    }

    #[test]
    fn spares_a_landmark_with_too_few_observations_to_judge() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        let idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        map.set_tracking_stats_for_test(idx, 4, 0);

        assert_eq!(cull_landmarks(&mut map), 0);
        assert!(!map.map_points()[idx].culled);
    }

    #[test]
    fn spares_landmark_that_falls_behind_a_different_keyframe() {
        let mut map = Map::new();
        // KF0 at origin (0,0,0) facing +Z
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        // Point created at (0,0,5) relative to KF0
        let idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();

        // KF1 at (0,0,10) facing +Z (so point at Z=5 is behind KF1 at Z_cam = -5)
        let mut kf1_frame = test_frame(1, vec![[0u8; 32]]);
        kf1_frame.pose_world_to_cam =
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -10.0));
        map.insert_keyframe(Keyframe::from_frame(kf1_frame))
            .unwrap();
        map.link_observation(1, 0, idx).unwrap();

        // Under the fix, the point is in front of its reference KF0 (Z=5 > 0),
        // so it must NOT be culled even though it is behind KF1.
        assert_eq!(cull_landmarks(&mut map), 0);
        assert!(!map.map_points()[idx].culled);
    }

    #[test]
    fn culls_landmark_if_behind_reference_keyframe() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        let idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, -2.0), // Behind KF0
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();

        assert_eq!(cull_landmarks(&mut map), 1);
        assert!(map.map_points()[idx].culled);
    }
}
