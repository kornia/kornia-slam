//! Local-map selection for tracking.
//!
//! Which landmarks tracking searches each frame is a tracking policy, not a
//! property of the map: the vote limits, the covisibility threshold and the
//! all-active fallback exist to bound per-frame cost. The map supplies the raw
//! observations and covisibility; the choices below are tracking's.

use std::collections::{HashMap, HashSet};

use crate::mapping::{Keyframe, Map};

/// Bounds on the local-map search. The defaults are the values this system has
/// been running, lifted out of the function body unchanged.
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

/// Builds the local map for tracking: indices of non-culled map points
/// observed by the current keyframe, the keyframes owning the tracked points,
/// and their covisibility neighbors.
pub fn select_local_map_points(
    map: &Map,
    tracked_matches: &[(usize, usize)],
    current_keyframe: Option<&Keyframe>,
    config: LocalMapSelectionConfig,
) -> Vec<usize> {
    // Vote over every keyframe observing each tracked point (ORB-SLAM3's
    // UpdateLocalKeyFrames). The vote map already encodes covisibility
    // with the current frame, so no per-seed covisibility recomputation
    // is needed — that scan is O(KF points x observations) per seed and
    // dominated per-frame tracking cost.
    let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
    for &(mp_idx, _) in tracked_matches {
        if let Some(mp) = map.map_points().get(mp_idx) {
            for obs_kf in mp.observer_keyframes() {
                *keyframe_votes.entry(obs_kf).or_insert(0) += 1;
            }
        }
    }

    let mut voted_kfs: Vec<(usize, usize)> = keyframe_votes.into_iter().collect();
    voted_kfs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

    // Local keyframes: the current KF plus the best-covisible KFs from
    // the votes. Only when there are no votes (start of tracking, before
    // any matches exist) fall back to one covisibility expansion around
    // the current KF — that scan is the expensive part, so it must not
    // run on every build.
    let mut local_kf_indices: HashSet<usize> = HashSet::new();
    if let Some(kf) = current_keyframe {
        local_kf_indices.insert(kf.frame.idx);
        if voted_kfs.is_empty() {
            for (nb_idx, _) in map
                .covisible_keyframes(kf.frame.idx, config.min_covis_weight)
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

    let mut mp_indices: HashSet<usize> = HashSet::new();
    for &(mp_idx, _) in tracked_matches {
        if mp_idx < map.map_points().len() {
            mp_indices.insert(mp_idx);
        }
    }
    for kf in map.keyframes() {
        if !local_kf_indices.contains(&kf.frame.idx) {
            continue;
        }
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if *mp_idx < map.map_points().len() {
                mp_indices.insert(*mp_idx);
            }
        }
    }

    mp_indices.retain(|&idx| !map.map_points()[idx].culled);

    let mut global_indices: Vec<usize> = mp_indices.into_iter().collect();
    global_indices.sort_unstable();

    if global_indices.len() < config.min_selected_before_fallback
        && map.map_points().len() >= config.min_selected_before_fallback
    {
        global_indices = (0..map.map_points().len())
            .filter(|&idx| !map.map_points()[idx].culled)
            .collect();
    }

    global_indices
}
