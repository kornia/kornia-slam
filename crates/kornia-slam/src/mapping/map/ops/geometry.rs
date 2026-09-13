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

use crate::map::{Map, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_algebra::Vec3F64;
use std::collections::HashSet;

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
        let (position, kf_indices, ref_kf_idx, ref_octave) = {
            let Some(mp) = self.map_points.get(mp_idx) else {
                return;
            };
            if mp.culled {
                return;
            }
            (
                mp.position,
                mp.observer_keyframes().collect::<Vec<_>>(),
                mp.keyframe_idx,
                mp.reference_octave,
            )
        };

        let mut normal = Vec3F64::ZERO;
        let mut n = 0usize;
        for kf_idx in &kf_indices {
            if let Some(kf) = self.get_keyframe(*kf_idx) {
                let cam_center = kf.frame.pose_world_to_cam.inverse().translation;
                let dir = position - cam_center;
                let len = dir.length();
                if len > 1e-9 {
                    normal += dir / len;
                    n += 1;
                }
            }
        }

        let Some(ref_kf) = self.get_keyframe(ref_kf_idx) else {
            return;
        };
        let ref_center = ref_kf.frame.pose_world_to_cam.inverse().translation;
        let dist = (position - ref_center).length();
        let level_scale = scale_factor.powi(ref_octave as i32);
        let max_dist = dist * level_scale;
        let min_dist = if n_levels > 0 {
            max_dist / scale_factor.powi(n_levels as i32 - 1)
        } else {
            max_dist
        };

        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            if n > 0 {
                mp.mean_viewing_direction = normal / n as f64;
            }
            mp.max_distance = max_dist;
            mp.min_distance = min_dist;
        }
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
    /// its dependency sets were assembled.
    pub(in crate::mapping::map) fn finalize_landmark_geometry(
        &mut self,
        landmark_ids: impl IntoIterator<Item = usize>,
    ) {
        let mut ids: Vec<usize> = landmark_ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        for idx in ids {
            self.update_map_point_geometry(idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
    }
}
