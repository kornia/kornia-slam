//! Landmark culling policy.
//!
//! Redundant-keyframe culling is owned by [`crate::mapping::keyframes`].

use crate::mapping::map::Map;
use std::collections::HashSet;

// Compatibility for the established public path. New callers should use
// `mapping::keyframes`.
pub use crate::mapping::keyframes::{
    KeyframeCullReport, KeyframeCullingConfig, cull_redundant_keyframes,
};

/// Minimum times a landmark must have been in view before its match rate is
/// judged at all.
const MIN_OBSERVATIONS: u32 = 5;
/// Match rate below which a sufficiently-observed landmark is dropped.
const MIN_FOUND_RATIO: f64 = 0.20;
/// Distance from its nearest observing keyframe beyond which a landmark is
/// dropped, metres. Far past anything a camera triangulates or a stereo rig
/// measures, so only a landmark a solve flung out along its ray (to 1e7 m and
/// more, from a window that could not observe its depth) reaches it.
const MAX_LANDMARK_DISTANCE_M: f64 = 1e4;

/// Ids of the landmarks that fail the culling policy.
///
/// A poor found ratio once a landmark has been seen enough times to judge,
/// any landmark sitting behind a keyframe that observes it, and any farther
/// than [`MAX_LANDMARK_DISTANCE_M`] from every keyframe observing it (the map
/// holds only finite positions).
fn select_for_culling(map: &Map) -> Vec<usize> {
    let mut selected: HashSet<usize> = HashSet::new();

    for (idx, mp) in map.map_points().iter().enumerate() {
        if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
            continue;
        }
        if mp.found_ratio() < MIN_FOUND_RATIO {
            selected.insert(idx);
        }
    }

    // Map point indices are dense, so a slot per point beats a hash map on large maps.
    let mut nearest = vec![f64::INFINITY; map.map_points().len()];
    for kf in map.keyframes() {
        let centre = kf.frame.pose_world_to_cam.inverse().translation;
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if let Some(mp) = map.map_points().get(*mp_idx)
                && !mp.culled
            {
                let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                if p_cam.z <= 1e-8 {
                    selected.insert(*mp_idx);
                }
                let d = (mp.position - centre).length();
                nearest[*mp_idx] = nearest[*mp_idx].min(d);
            }
        }
    }
    selected.extend(
        nearest
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d.is_finite() && d > MAX_LANDMARK_DISTANCE_M)
            .map(|(idx, _)| idx),
    );

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
    use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
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
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]; 2])))
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
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(kept));
        assert!(map.map_points()[doomed].is_observed_by(0));
    }

    #[test]
    fn spares_a_landmark_with_too_few_observations_to_judge() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])))
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
    fn culls_a_landmark_flung_absurdly_far() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]; 3])))
            .unwrap();
        let seed = |feature, z| LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, z),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: feature,
            },
        };
        let near = map.insert_landmark(seed(0, 5.0)).unwrap();
        let far = map.insert_landmark(seed(1, 5e3)).unwrap();
        let flung = map.insert_landmark(seed(2, 1e7)).unwrap();
        assert_eq!(cull_landmarks(&mut map), 1);
        assert!(!map.map_points()[near].culled);
        assert!(!map.map_points()[far].culled);
        assert!(map.map_points()[flung].culled);
    }
}
