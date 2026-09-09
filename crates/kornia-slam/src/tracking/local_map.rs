//! Local-map selection and projection filtering.
//!
//! Which landmarks tracking searches each frame is a tracking policy, not a
//! property of the map: the vote limits, the covisibility threshold and the
//! all-active fallback exist to bound per-frame cost. The map supplies raw
//! covisibility; the choices below are ours.

use crate::map::Map;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use std::collections::{HashMap, HashSet};

/// Applies a consumer's minimum-weight policy to raw covisibility.
///
/// Mirrors ORB-SLAM3's `KeyFrame::UpdateConnections`: links below `min_weight`
/// are dropped, but if none reach it the single strongest link is kept so an
/// under-connected keyframe is never orphaned. This fallback runs before any
/// neighbour limit a caller applies afterwards.
pub(crate) fn covisible_above_weight(
    connections: Vec<(usize, usize)>,
    min_weight: usize,
) -> Vec<(usize, usize)> {
    let strongest = connections.first().copied();
    let mut kept: Vec<(usize, usize)> = connections
        .into_iter()
        .filter(|&(_, w)| w >= min_weight)
        .collect();
    if kept.is_empty() {
        kept.extend(strongest);
    }
    kept
}

/// Bounds on the local-map search. Defaults match ORB-SLAM3's
/// `UpdateLocalKeyFrames` as this system has been running it.
#[derive(Debug, Clone, Copy)]
pub struct LocalMapSelectionConfig {
    /// Most-voted keyframes to admit.
    pub max_voted_keyframes: usize,
    /// Covisibility neighbours to admit during the no-vote expansion.
    pub max_covis_neighbors: usize,
    /// Minimum shared landmarks for a covisibility neighbour to count.
    pub min_covis_weight: usize,
    /// Below this many selected landmarks, fall back to the whole active map.
    pub min_selected_before_fallback: usize,
}

impl Default for LocalMapSelectionConfig {
    fn default() -> Self {
        Self {
            max_voted_keyframes: 10,
            max_covis_neighbors: 10,
            min_covis_weight: 15,
            min_selected_before_fallback: 4,
        }
    }
}

/// Landmarks worth projecting into the current frame.
///
/// Keyframes are voted for by the landmarks already matched, mirroring
/// ORB-SLAM3's `UpdateLocalKeyFrames`. The vote map already encodes
/// covisibility with the current frame, so the covisibility scan — the
/// expensive part, O(keyframe points × observations) per seed — runs only when
/// there are no votes at all, at the very start of tracking.
pub fn select_local_landmarks(
    map: &Map,
    matched_landmark_ids: &[usize],
    reference_keyframe_id: Option<usize>,
    config: &LocalMapSelectionConfig,
) -> Vec<usize> {
    let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
    for &mp_idx in matched_landmark_ids {
        if let Some(mp) = map.map_points().get(mp_idx) {
            for obs_kf in mp.observer_keyframes() {
                *keyframe_votes.entry(obs_kf).or_insert(0) += 1;
            }
        }
    }

    let mut voted_kfs: Vec<(usize, usize)> = keyframe_votes.into_iter().collect();
    voted_kfs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

    let mut local_kf_indices: HashSet<usize> = HashSet::new();
    if let Some(reference_kf_idx) = reference_keyframe_id {
        local_kf_indices.insert(reference_kf_idx);
        if voted_kfs.is_empty() {
            for (nb_idx, _) in covisible_above_weight(
                map.covisible_keyframes(reference_kf_idx),
                config.min_covis_weight,
            )
            .into_iter()
            .take(config.max_covis_neighbors)
            {
                local_kf_indices.insert(nb_idx);
            }
        }
    }
    for (kf_idx, _) in voted_kfs.into_iter().take(config.max_voted_keyframes) {
        local_kf_indices.insert(kf_idx);
    }

    let n_points = map.map_points().len();
    let mut mp_indices: HashSet<usize> = HashSet::new();
    for &mp_idx in matched_landmark_ids {
        if mp_idx < n_points {
            mp_indices.insert(mp_idx);
        }
    }
    for kf in map.keyframes() {
        if !local_kf_indices.contains(&kf.frame.idx) {
            continue;
        }
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if *mp_idx < n_points {
                mp_indices.insert(*mp_idx);
            }
        }
    }

    mp_indices.retain(|&idx| !map.map_points()[idx].culled);

    let mut selected: Vec<usize> = mp_indices.into_iter().collect();
    selected.sort_unstable();

    if selected.len() < config.min_selected_before_fallback && n_points >= 4 {
        selected = (0..n_points)
            .filter(|&idx| !map.map_points()[idx].culled)
            .collect();
    }
    selected
}

/// The subset of `candidates` that are active and project inside the image.
///
/// A projection test only: it says nothing about occlusion, descriptor
/// distance, or the scale and viewing-angle gates the matcher applies later.
pub fn landmarks_in_frustum(
    map: &Map,
    candidates: &[usize],
    camera: &PinholeCamera,
    pose_world_to_cam: &Pose3d,
    image_size: ImageSize,
) -> HashSet<usize> {
    let mut visible = HashSet::new();
    for &mp_idx in candidates {
        let Some(mp) = map.map_points().get(mp_idx) else {
            continue;
        };
        if mp.culled {
            continue;
        }
        let p_cam = pose_world_to_cam.transform_point(&mp.position);
        if camera.project_to_image(&p_cam, 0.0, image_size).is_ok() {
            visible.insert(mp_idx);
        }
    }
    visible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
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

    /// KF0 shares two landmarks with KF1 and one with KF2.
    fn covis_map() -> Map {
        let mut map = Map::new();
        for idx in 0..3 {
            map.insert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32]; 3],
            )))
            .unwrap();
        }
        for (slot, observers) in [(0usize, vec![0usize, 1]), (1, vec![0, 1]), (2, vec![0, 2])] {
            let mut it = observers.into_iter();
            let first = it.next().unwrap();
            let mp = map
                .insert_landmark(LandmarkSeed {
                    position: Vec3F64::new(0.0, 0.0, 1.0),
                    color: [0; 3],
                    reference: ObservationKey {
                        keyframe_idx: first,
                        feature_idx: slot,
                    },
                })
                .unwrap();
            for kf in it {
                map.link_observation(kf, slot, mp).unwrap();
            }
        }
        map
    }

    #[test]
    fn a_threshold_above_every_weight_keeps_the_strongest_link() {
        let raw = covis_map().covisible_keyframes(0);
        assert_eq!(raw, vec![(1, 2), (2, 1)]);
        // Nothing clears 5, so connectivity is preserved by the fallback.
        assert_eq!(covisible_above_weight(raw.clone(), 5), vec![(1, 2)]);
        assert_eq!(covisible_above_weight(raw.clone(), 2), vec![(1, 2)]);
        assert_eq!(covisible_above_weight(raw, 1), vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn selection_falls_back_to_the_whole_active_map_when_too_few() {
        let mut map = covis_map();
        // The fallback only engages once the map holds at least four
        // landmarks; below that there is nothing better to offer.
        assert_eq!(map.num_map_points(), 3);
        assert!(
            select_local_landmarks(&map, &[], None, &LocalMapSelectionConfig::default()).is_empty()
        );

        map.insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 2.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 2,
                feature_idx: 1,
            },
        })
        .unwrap();

        // No matches and no reference would select nothing, so every live
        // landmark is offered instead.
        let selected = select_local_landmarks(&map, &[], None, &LocalMapSelectionConfig::default());
        assert_eq!(selected.len(), 4);
        assert_eq!(selected.len(), map.num_map_points());
    }

    /// Votes must actually discriminate: a keyframe observing nothing matched
    /// contributes nothing. The earlier version of this test used a map where
    /// every landmark was reachable from the reference, so it passed whatever
    /// selection did.
    #[test]
    fn votes_admit_only_keyframes_observing_something_matched() {
        let mut map = Map::new();
        // KF1 carries four features, so the landmarks its votes pull in clear
        // the all-active fallback threshold without help.
        for (idx, slots) in [(0usize, 1usize), (1, 4), (2, 1), (3, 1)] {
            map.insert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32]; slots],
            )))
            .unwrap();
        }
        let seed = |kf: usize, feature: usize| LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: kf,
                feature_idx: feature,
            },
        };
        // Matched, and the only landmark KF0 and KF1 share.
        let shared = map.insert_landmark(seed(0, 0)).unwrap();
        map.link_observation(1, 0, shared).unwrap();
        // Also on KF1, so its votes should pull these in.
        let voter_a = map.insert_landmark(seed(1, 1)).unwrap();
        let voter_b = map.insert_landmark(seed(1, 2)).unwrap();
        let voter_c = map.insert_landmark(seed(1, 3)).unwrap();
        // On keyframes observing nothing matched.
        let unrelated_a = map.insert_landmark(seed(2, 0)).unwrap();
        let unrelated_b = map.insert_landmark(seed(3, 0)).unwrap();

        let selected =
            select_local_landmarks(&map, &[shared], None, &LocalMapSelectionConfig::default());

        assert_eq!(
            selected,
            vec![shared, voter_a, voter_b, voter_c],
            "only the voting keyframes' landmarks, sorted"
        );
        assert!(!selected.contains(&unrelated_a));
        assert!(!selected.contains(&unrelated_b));
    }

    /// A retired landmark is never offered, even when its keyframe votes.
    #[test]
    fn selection_filters_retired_landmarks() {
        let mut map = covis_map();
        map.remove_landmark(0).unwrap();
        let selected =
            select_local_landmarks(&map, &[0], Some(0), &LocalMapSelectionConfig::default());
        assert!(!selected.contains(&0));
        assert!(selected.iter().all(|&i| !map.map_points()[i].culled));
    }
}
