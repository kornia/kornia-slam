//! Local-map selection and projection filtering.
//!
//! Which landmarks tracking searches each frame is a tracking policy, not a
//! property of the map: the vote limits, the covisibility threshold and the
//! all-active fallback exist to bound per-frame cost. The map supplies raw
//! covisibility; the choices below are ours.

use crate::map::{Map, covisible_above_weight};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use std::collections::{HashMap, HashSet};

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
