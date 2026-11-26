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
use kornia_sensors::Fisheye624Camera;

use crate::mapping::{Keyframe, Map};
use crate::rig::project_in_view;

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
    let (mut points, most_voted_keyframe) = match config.rules {
        LocalMapRules::Kornia => (
            select_voted_points(map, tracked_matches, current_keyframe, config),
            None,
        ),
        LocalMapRules::OrbSlam3 => {
            select_orb_slam3_points(map, tracked_matches, current_keyframe, config)
        }
    };
    let dropped_by_window = config
        .window_start_kf
        .map_or(0, |start| filter_to_window(map, &mut points, start));
    LocalMapSelection {
        points,
        dropped_by_window,
        most_voted_keyframe,
    }
}

/// ORB-SLAM3's local keyframes stop growing past 80 (`Tracking.cc:3535`).
const ORB_SLAM3_MAX_LOCAL_KEYFRAMES: usize = 80;
/// Best covisible keyframes each local keyframe offers one neighbour from.
const ORB_SLAM3_NEIGHBOUR_CANDIDATES: usize = 10;
/// ORB-SLAM3 links keyframes sharing at least 15 landmarks
/// (`KeyFrame::UpdateConnections`).
const ORB_SLAM3_MIN_COVISIBILITY_WEIGHT: usize = 15;
/// The most recent keyframes added for the IMU (`Tracking.cc:3585-3600`).
const ORB_SLAM3_TEMPORAL_KEYFRAMES: usize = 20;

/// ORB-SLAM3's inertial `UpdateLocalKeyFrames` (`Tracking.cc:3457-3606`):
/// every keyframe a tracked point votes for, each one's first new best
/// covisible neighbour, then the most recent keyframes. Children and parent
/// are left out: this map has no spanning tree. Returns the points and the
/// most-voted keyframe, ORB-SLAM3's next reference keyframe.
fn select_orb_slam3_points(
    map: &Map,
    tracked_matches: &[(usize, usize)],
    current_keyframe: Option<&Keyframe>,
    config: LocalMapSelectionConfig,
) -> (Vec<usize>, Option<usize>) {
    let mut votes: HashMap<usize, usize> = HashMap::new();
    for &(mp_idx, _) in tracked_matches {
        if let Some(mp) = map.map_points().get(mp_idx).filter(|mp| !mp.culled) {
            for obs_kf in mp.observer_keyframes() {
                *votes.entry(obs_kf).or_insert(0) += 1;
            }
        }
    }
    let mut voted: Vec<(usize, usize)> = votes
        .into_iter()
        .filter(|&(kf_idx, _)| map.get_keyframe(kf_idx).is_some())
        .collect();
    voted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    let most_voted_keyframe = voted.first().map(|&(kf_idx, _)| kf_idx);

    let mut local: Vec<usize> = voted.into_iter().map(|(kf_idx, _)| kf_idx).collect();
    if local.is_empty()
        && let Some(kf) = current_keyframe
    {
        local.push(kf.frame.idx);
    }
    let mut admitted: HashSet<usize> = local.iter().copied().collect();
    for i in 0..local.len() {
        if local.len() > ORB_SLAM3_MAX_LOCAL_KEYFRAMES {
            break;
        }
        let neighbour = map
            .covisible_keyframes_min_weight(local[i], ORB_SLAM3_MIN_COVISIBILITY_WEIGHT)
            .into_iter()
            .take(ORB_SLAM3_NEIGHBOUR_CANDIDATES)
            .map(|(kf_idx, _)| kf_idx)
            .find(|kf_idx| !admitted.contains(kf_idx));
        if let Some(neighbour) = neighbour {
            admitted.insert(neighbour);
            local.push(neighbour);
        }
    }
    if local.len() < ORB_SLAM3_MAX_LOCAL_KEYFRAMES {
        for kf in map
            .keyframes()
            .iter()
            .rev()
            .take(ORB_SLAM3_TEMPORAL_KEYFRAMES)
        {
            admitted.insert(kf.frame.idx);
        }
    }

    (
        points_of_keyframes(map, tracked_matches, &admitted, config),
        most_voted_keyframe,
    )
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

    points_of_keyframes(map, tracked_matches, &local_kf_indices, config)
}

/// The non-culled points that `local_kf_indices` hold or `tracked_matches`
/// name, ascending; every active point when fewer than
/// `min_selected_before_fallback` remain.
fn points_of_keyframes(
    map: &Map,
    tracked_matches: &[(usize, usize)],
    local_kf_indices: &HashSet<usize>,
    config: LocalMapSelectionConfig,
) -> Vec<usize> {
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
    let rig = frame.rig.geometry.as_ref().filter(|_| frame.is_rig());
    let primary_raw = rig
        .and_then(|rig| rig.view(0))
        .and_then(|view| view.raw_model.as_ref());
    let mut visible = landmarks_in_frustum(
        map,
        candidates,
        camera,
        primary_raw,
        pose_world_to_cam,
        frame.image_size,
    );
    if let Some(rig) = rig {
        for view in rig.views().iter().skip(1) {
            visible.extend(landmarks_in_frustum(
                map,
                candidates,
                &view.camera,
                view.raw_model.as_ref(),
                &view.pose(pose_world_to_cam),
                view.size,
            ));
        }
    }
    visible
}

/// `raw` is the camera's raw fisheye model, when its images are raw (see
/// `rig::project_in_view`).
pub fn landmarks_in_frustum(
    map: &Map,
    candidates: &[usize],
    camera: &PinholeCamera,
    raw: Option<&Fisheye624Camera>,
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
        if project_in_view(camera, raw, image_size, &p_cam, 0.0).is_some() {
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
    fn raw_fisheye_frustum_counts_wide_points() {
        let mut map = Map::new();
        map.insert_keyframe(keyframe(0, 2)).unwrap();
        let (s, c) = 70f64.to_radians().sin_cos();
        for (feature_idx, position) in [
            kornia_algebra::Vec3F64::new(0.0, 0.0, 5.0),
            kornia_algebra::Vec3F64::new(s, 0.0, c) * 5.0,
        ]
        .into_iter()
        .enumerate()
        {
            map.insert_landmark(LandmarkSeed {
                position,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx,
                },
            })
            .unwrap();
        }
        let raw = crate::rig::tests::raw_model();
        let camera = PinholeCamera {
            fx: raw.f,
            fy: raw.f,
            cx: raw.cx,
            cy: raw.cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let visible =
            |raw| landmarks_in_frustum(&map, &[0, 1], &camera, raw, &Pose3d::IDENTITY, size);
        assert_eq!(visible(None), HashSet::from([0]));
        assert_eq!(visible(Some(&raw)), HashSet::from([0, 1]));
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
