//! Local-map selection for tracking.
//!
//! Which landmarks tracking searches each frame is a tracking policy, not a
//! property of the map: the vote limits, the covisibility threshold and the
//! all-active fallback exist to bound per-frame cost. The map supplies the raw
//! observations and covisibility; the choices below are tracking's.

use std::collections::{HashMap, HashSet};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;

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
    /// Which keyframes the selection admits.
    pub rules: LocalMapRules,
    /// Keep only points observed by a keyframe at or after this one: a
    /// re-seeded segment tracking its own window. `None` keeps every point.
    pub window_start_kf: Option<usize>,
}

/// How tracking picks its local keyframes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LocalMapRules {
    /// The current keyframe plus the 10 most-voted keyframes.
    #[default]
    Kornia,
    /// ORB-SLAM3's inertial `Tracking::UpdateLocalKeyFrames`
    /// (`Tracking.cc:3457-3606`), without children and parent: this map has
    /// no spanning tree.
    OrbSlam3,
}

/// One local-map selection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalMapSelection {
    /// Indices of the selected map points, ascending.
    pub points: Vec<usize>,
    /// Points the window removed; 0 when `window_start_kf` is `None`.
    pub dropped_by_window: usize,
    /// The most-voted keyframe; `None` without votes or under
    /// [`LocalMapRules::Kornia`].
    pub most_voted_keyframe: Option<usize>,
}

impl Default for LocalMapSelectionConfig {
    fn default() -> Self {
        Self {
            max_voted_keyframes: 10,
            max_covis_neighbors: 10,
            min_covis_weight: 15,
            min_selected_before_fallback: 4,
            rules: LocalMapRules::Kornia,
            window_start_kf: None,
        }
    }
}

/// Builds the local map for tracking: the rule picks keyframes and points,
/// whole-map fallback included, then the window, when set, filters them.
pub fn select_local_map_points(
    map: &Map,
    tracked_matches: &[(usize, usize)],
    current_keyframe: Option<&Keyframe>,
    config: LocalMapSelectionConfig,
) -> LocalMapSelection {
    let mut points = match config.rules {
        LocalMapRules::Kornia => {
            select_voted_points(map, tracked_matches, current_keyframe, config)
        }
        LocalMapRules::OrbSlam3 => todo!("lane A: ORB-SLAM3 local keyframes"),
    };
    let dropped_by_window = config
        .window_start_kf
        .map_or(0, |start| filter_to_window(map, &mut points, start));
    LocalMapSelection {
        points,
        dropped_by_window,
        most_voted_keyframe: None,
    }
}

/// Keeps the points of `points` observed by a keyframe at or after
/// `window_start_kf`; returns how many were removed.
pub(crate) fn filter_to_window(
    map: &Map,
    points: &mut Vec<usize>,
    window_start_kf: usize,
) -> usize {
    let before = points.len();
    points.retain(|&idx| {
        map.map_points()
            .get(idx)
            .is_some_and(|mp| mp.observer_keyframes().any(|kf| kf >= window_start_kf))
    });
    before - points.len()
}

/// Indices of non-culled map points observed by the current keyframe, the
/// keyframes owning the tracked points, and their covisibility neighbors.
fn select_voted_points(
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
                .covisible_keyframes_min_weight(kf.frame.idx, config.min_covis_weight)
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

/// The subset of `candidates` that are active and project inside the image.
///
/// A projection test only: it says nothing about occlusion, descriptor
/// distance, or the scale and viewing-angle gates the matcher applies later.
/// [`landmarks_in_frustum`] of every camera of `frame`'s rig, or of the
/// single camera outside one.
pub fn landmarks_in_frame_frustum(
    map: &Map,
    candidates: &[usize],
    camera: &PinholeCamera,
    pose_world_to_cam: &Pose3d,
    frame: &crate::Frame,
) -> HashSet<usize> {
    let mut visible =
        landmarks_in_frustum(map, candidates, camera, pose_world_to_cam, frame.image_size);
    if let Some(rig) = frame.rig.geometry.as_ref().filter(|_| frame.is_rig()) {
        for view in rig.views().iter().skip(1) {
            visible.extend(landmarks_in_frustum(
                map,
                candidates,
                &view.camera,
                &view.pose(pose_world_to_cam),
                view.size,
            ));
        }
    }
    visible
}

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
    use crate::Frame;
    use crate::mapping::map::{LandmarkSeed, ObservationKey};
    use kornia_imgproc::features::OrbFeatures;

    fn keyframe(idx: usize, features: usize) -> Keyframe {
        Keyframe::from_frame(Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: vec![[320.0, 240.0]; features],
                orientations: vec![0.0; features],
                descriptors: (0..features).map(|i| [i as u8; 32]).collect(),
                octaves: vec![0; features],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; features],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
            rig: Default::default(),
        })
    }

    /// Keyframes 0, 5 and 10. Landmarks 0 and 3 are seen by keyframe 0 only,
    /// 1 by 0 and 5, 2 by 10 and 4 by 5.
    fn segmented_map() -> Map {
        let mut map = Map::new();
        for idx in [0, 5, 10] {
            map.insert_keyframe(keyframe(idx, 4)).unwrap();
        }
        for (keyframe_idx, feature_idx) in [(0, 0), (0, 1), (10, 0), (0, 2), (5, 1)] {
            map.insert_landmark(LandmarkSeed {
                position: kornia_algebra::Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx,
                    feature_idx,
                },
            })
            .unwrap();
        }
        map.link_observation(5, 0, 1).unwrap();
        map
    }

    #[test]
    fn the_window_keeps_points_seen_inside_it() {
        let map = segmented_map();
        let mut points = vec![0, 1, 2, 3, 4];
        assert_eq!(filter_to_window(&map, &mut points, 5), 2);
        assert_eq!(points, [1, 2, 4]);
        let mut points = vec![0, 1, 2, 3, 4];
        assert_eq!(filter_to_window(&map, &mut points, 0), 0);
        assert_eq!(filter_to_window(&map, &mut points, 11), 5);
        assert!(points.is_empty());
    }

    #[test]
    fn the_window_filters_the_whole_map_fallback_too() {
        let map = segmented_map();
        let config = LocalMapSelectionConfig::default();
        let open = select_local_map_points(&map, &[], None, config);
        assert_eq!(
            open.points,
            [0, 1, 2, 3, 4],
            "fewer than 4 selected: whole map"
        );
        assert_eq!(open.dropped_by_window, 0);
        assert_eq!(
            open.points,
            select_voted_points(&map, &[], None, config),
            "no window, no change"
        );

        let windowed = select_local_map_points(
            &map,
            &[],
            None,
            LocalMapSelectionConfig {
                window_start_kf: Some(5),
                ..config
            },
        );
        assert_eq!(windowed.points, [1, 2, 4]);
        assert_eq!(windowed.dropped_by_window, 2);
    }
}
