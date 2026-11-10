//! Derived landmark geometry and the dependency rules for refreshing it.
//!
//! A landmark's distance bounds come from its reference camera centre and
//! octave; its mean viewing direction averages over *every* observing camera
//! centre. Both are therefore invalidated by a moved camera, not only by a
//! moved landmark, and corrections own that refresh: no solver hands the map a
//! list of landmarks to fix up.
//!
//! Collection always reads the live observation records rather than an
//! optimization's captured ones. A landmark linked while a solve was running
//! depends on a camera centre that solve moved, even though its own position
//! must never be written back from that stale capture.

use crate::mapping::map::{Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_algebra::Vec3F64;
use std::collections::{HashMap, HashSet};

impl Map {
    /// Recomputes the mean viewing direction and scale-invariance distance
    /// bounds for one map point from its observing keyframes (ORB-SLAM3's
    /// `MapPoint::UpdateNormalAndDepth`). No-op if the point is culled or its
    /// reference keyframe is missing.
    ///
    /// Private to `map` and its descendants: callers reach it through the
    /// corrections and mutations that know which landmarks became stale.
    pub(in crate::mapping::map) fn update_map_point_geometry(
        &mut self,
        mp_idx: usize,
        scale_factor: f64,
        n_levels: usize,
    ) {
        let keyframes = &self.keyframes;
        let centre = |kf_idx: usize, camera: u8| {
            keyframes
                .iter()
                .find(|kf| kf.frame.idx == kf_idx)
                .map(|kf| camera_centre(kf, camera))
        };
        let Some(geometry) = self
            .map_points
            .get(mp_idx)
            .and_then(|mp| refreshed_geometry(mp, centre, scale_factor, n_levels))
        else {
            return;
        };
        geometry.write(&mut self.map_points[mp_idx]);
    }

    /// Live landmark ids observed by any keyframe in `keyframe_indices`.
    ///
    /// Retired landmarks are excluded: they hold no observations and keep their
    /// derived geometry zeroed, and a refresh must never revive them.
    pub(in crate::mapping::map) fn landmarks_observed_by(
        &self,
        keyframe_indices: &HashSet<usize>,
    ) -> Vec<usize> {
        if keyframe_indices.is_empty() {
            return Vec::new();
        }
        self.map_points
            .iter()
            .enumerate()
            .filter(|(_, mp)| !mp.culled)
            .filter(|(_, mp)| {
                mp.observer_keyframes()
                    .any(|kf_idx| keyframe_indices.contains(&kf_idx))
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Every live landmark id, for a correction that moves the whole world and
    /// therefore every camera centre at once.
    pub(in crate::mapping::map) fn live_landmarks(&self) -> Vec<usize> {
        self.map_points
            .iter()
            .enumerate()
            .filter(|(_, mp)| !mp.culled)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Refreshes derived geometry once per affected landmark, in ascending id
    /// order so a correction's finalization is deterministic regardless of how
    /// its dependency sets were assembled. Returns how many were refreshed.
    ///
    /// Camera centres are tabulated once: looking each observer up in the
    /// keyframe list made a whole-map refresh O(landmarks x observers x keyframes).
    pub(in crate::mapping::map) fn finalize_landmark_geometry(
        &mut self,
        landmark_ids: impl IntoIterator<Item = usize>,
    ) -> usize {
        let mut ids: Vec<usize> = landmark_ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        let centres: HashMap<usize, &Keyframe> =
            self.keyframes.iter().map(|kf| (kf.frame.idx, kf)).collect();
        let centre =
            |kf_idx: usize, camera: u8| centres.get(&kf_idx).map(|kf| camera_centre(kf, camera));
        for &idx in &ids {
            let Some(geometry) = self
                .map_points
                .get(idx)
                .and_then(|mp| refreshed_geometry(mp, centre, ORB_SCALE_FACTOR, ORB_N_LEVELS))
            else {
                continue;
            };
            geometry.write(&mut self.map_points[idx]);
        }
        ids.len()
    }
}

/// Centre of rig camera `camera` of the keyframe; the primary camera's outside a rig.
fn camera_centre(keyframe: &Keyframe, camera: u8) -> Vec3F64 {
    let pose = &keyframe.frame.pose_world_to_cam;
    match keyframe
        .frame
        .rig
        .geometry
        .as_ref()
        .and_then(|rig| rig.view(camera as usize))
        .filter(|_| camera > 0)
    {
        Some(view) => view.pose(pose).inverse().translation,
        None => pose.inverse().translation,
    }
}

/// A landmark's derived geometry; the viewing direction is `None` when no
/// observer centre was usable, which keeps the stored one.
struct LandmarkGeometry {
    mean_viewing_direction: Option<Vec3F64>,
    max_distance: f64,
    min_distance: f64,
}

impl LandmarkGeometry {
    fn write(self, mp: &mut MapPoint) {
        if let Some(direction) = self.mean_viewing_direction {
            mp.mean_viewing_direction = direction;
        }
        mp.max_distance = self.max_distance;
        mp.min_distance = self.min_distance;
    }
}

/// `None` if the point is culled or its reference keyframe has no centre.
fn refreshed_geometry(
    mp: &MapPoint,
    centre: impl Fn(usize, u8) -> Option<Vec3F64>,
    scale_factor: f64,
    n_levels: usize,
) -> Option<LandmarkGeometry> {
    if mp.culled {
        return None;
    }
    let position = mp.position;
    let mut normal = Vec3F64::ZERO;
    let mut n = 0usize;
    for observation in mp.observations() {
        if let Some(cam_center) = centre(observation.key.keyframe_idx, observation.camera) {
            let dir = position - cam_center;
            let len = dir.length();
            if len > 1e-9 {
                normal += dir / len;
                n += 1;
            }
        }
    }

    let ref_camera = mp
        .observations()
        .iter()
        .find(|o| o.key.keyframe_idx == mp.keyframe_idx)
        .map_or(0, |o| o.camera);
    let ref_center = centre(mp.keyframe_idx, ref_camera)?;
    let dist = (position - ref_center).length();
    let level_scale = scale_factor.powi(mp.reference_octave as i32);
    let max_dist = dist * level_scale;
    let min_dist = if n_levels > 0 {
        max_dist / scale_factor.powi(n_levels as i32 - 1)
    } else {
        max_dist
    };
    Some(LandmarkGeometry {
        mean_viewing_direction: (n > 0).then(|| normal / n as f64),
        max_distance: max_dist,
        min_distance: min_dist,
    })
}
