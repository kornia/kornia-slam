//! Mutating the map: inserting keyframes and points, and recording observations.

use crate::map::ObservationKey;
use crate::map::{
    ImuFactor, Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR, TriangulatedPoint,
};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuMeasurement;
use kornia_sensors::imu::PreintegratedImu;
use std::collections::HashSet;

/// Result of replacing a duplicate landmark with a surviving landmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapPointMergeResult {
    pub survivor: usize,
    pub replaced: usize,
    pub redirected_associations: usize,
}

impl Map {
    /// Inserts or replaces a keyframe by frame index.
    pub fn upsert_keyframe(&mut self, keyframe: Keyframe) {
        if let Some(pos) = self
            .keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe.frame.idx)
        {
            self.keyframes[pos] = keyframe;
        } else {
            self.keyframes.push(keyframe);
        }
    }
    /// Inserts triangulated 3D points as map points and associates them to
    /// keyframes.
    ///
    /// `curr_kf` becomes the reference keyframe for each new point's scale
    /// geometry (matches ORB-SLAM3, which creates points referenced to the
    /// newer keyframe). If `prev_kf` is provided, its observation is also
    /// recorded. Mean viewing direction and scale-invariance bounds are
    /// computed for every new point.
    pub fn add_triangulated_points(
        &mut self,
        prev_kf: Option<&mut Keyframe>,
        curr_kf: &mut Keyframe,
        points: &[TriangulatedPoint],
    ) -> usize {
        let first_mp_idx = self.map_points.len();
        let curr_kf_idx = curr_kf.frame.idx;
        for (i, &(position, descriptor, color, _, curr_desc_idx)) in points.iter().enumerate() {
            let octave = curr_kf
                .frame
                .features
                .octaves
                .get(curr_desc_idx)
                .copied()
                .unwrap_or(0);
            let desc = curr_kf
                .frame
                .features
                .descriptors
                .get(curr_desc_idx)
                .copied()
                .unwrap_or(descriptor);
            self.push_map_point(MapPoint::new(
                position,
                desc,
                octave,
                color,
                curr_kf_idx,
                curr_desc_idx,
            ));
            curr_kf.associate_map_point(curr_desc_idx, first_mp_idx + i);
        }
        if let Some(prev) = prev_kf {
            let prev_kf_idx = prev.frame.idx;
            for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().enumerate() {
                prev.associate_map_point(prev_desc_idx, first_mp_idx + i);
                // Feed the prev KF's descriptor as a second observation so the
                // representative descriptor is computed from both viewpoints.
                if let Some(&prev_desc) = prev.frame.features.descriptors.get(prev_desc_idx)
                    && let Some(mp) = self.map_points.get_mut(first_mp_idx + i)
                {
                    mp.add_observation(
                        ObservationKey {
                            keyframe_idx: prev_kf_idx,
                            feature_idx: prev_desc_idx,
                        },
                        prev_desc,
                    );
                }
            }
        }
        for i in 0..points.len() {
            self.update_map_point_geometry(first_mp_idx + i, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
        points.len()
    }
    /// Records that an existing map point was observed at `desc_idx` in
    /// `keyframe`, pushing the descriptor into the map point's observation
    /// list, refreshing the representative descriptor, and recomputing the
    /// scale geometry.
    pub fn register_observation(&mut self, mp_idx: usize, keyframe: &Keyframe, desc_idx: usize) {
        let Some(&descriptor) = keyframe.frame.features.descriptors.get(desc_idx) else {
            return;
        };
        let key = ObservationKey {
            keyframe_idx: keyframe.frame.idx,
            feature_idx: desc_idx,
        };
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation(key, descriptor);
        }
        self.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
    }
    /// [`Map::register_observation`] for a keyframe already stored in the
    /// map, addressed by frame index. Lets callers register observations
    /// while only holding `&mut Map` (no borrowed `Keyframe` clone needed).
    pub fn register_observation_at(&mut self, mp_idx: usize, kf_idx: usize, desc_idx: usize) {
        let Some(descriptor) = self
            .get_keyframe(kf_idx)
            .and_then(|kf| kf.frame.features.descriptors.get(desc_idx))
            .copied()
        else {
            return;
        };
        let key = ObservationKey {
            keyframe_idx: kf_idx,
            feature_idx: desc_idx,
        };
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation(key, descriptor);
        }
        self.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
    }
    /// Recomputes the mean viewing direction and scale-invariance distance
    /// bounds for one map point from its observing keyframes (ORB-SLAM3's
    /// `MapPoint::UpdateNormalAndDepth`). No-op if the point is culled or its
    /// reference keyframe is missing.
    pub fn update_map_point_geometry(&mut self, mp_idx: usize, scale_factor: f64, n_levels: usize) {
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
    /// Appends a map point and returns its index.
    pub fn push_map_point(&mut self, map_point: MapPoint) -> usize {
        let idx = self.map_points.len();
        self.map_points.push(map_point);
        idx
    }
    /// Returns a mutable reference to all map points.
    pub fn map_points_mut(&mut self) -> &mut Vec<MapPoint> {
        &mut self.map_points
    }
    /// Replaces a duplicate map point and redirects its keyframe associations.
    ///
    /// The point with stronger observation support survives. Associations are
    /// deduplicated so a keyframe references the survivor from at most one slot.
    pub fn merge_map_points(&mut self, first: usize, second: usize) -> Option<MapPointMergeResult> {
        if first == second {
            return None;
        }
        let first_point = self.map_points.get(first)?;
        let second_point = self.map_points.get(second)?;
        if first_point.culled || second_point.culled {
            return None;
        }

        let support = |point: &MapPoint| point.observer_keyframes().collect::<HashSet<_>>().len();
        let first_support = support(first_point);
        let second_support = support(second_point);
        let second_is_stronger = second_support > first_support
            || (second_support == first_support && second_point.n_found > first_point.n_found)
            || (second_support == first_support
                && second_point.n_found == first_point.n_found
                && second < first);
        let (survivor, replaced) = if second_is_stronger {
            (second, first)
        } else {
            (first, second)
        };

        let mut redirected_associations = 0;
        for keyframe in &mut self.keyframes {
            let survivor_slots: Vec<_> = keyframe
                .map_point_by_desc_idx
                .iter()
                .enumerate()
                .filter_map(|(slot, &point)| (point == Some(survivor)).then_some(slot))
                .collect();
            let replaced_slots: Vec<_> = keyframe
                .map_point_by_desc_idx
                .iter()
                .enumerate()
                .filter_map(|(slot, &point)| (point == Some(replaced)).then_some(slot))
                .collect();

            let mut keep_survivor = survivor_slots.first().copied();
            if keep_survivor.is_none()
                && let Some(&slot) = replaced_slots.first()
            {
                keyframe.map_point_by_desc_idx[slot] = Some(survivor);
                keep_survivor = Some(slot);
                redirected_associations += 1;
            }
            for slot in survivor_slots.into_iter().chain(replaced_slots) {
                if Some(slot) != keep_survivor {
                    keyframe.map_point_by_desc_idx[slot] = None;
                }
            }
        }

        let replaced_observations: Vec<_> = self.map_points[replaced]
            .observations()
            .iter()
            .map(|observation| (observation.key, observation.descriptor))
            .collect();
        let replaced_visible = self.map_points[replaced].n_visible;
        let replaced_found = self.map_points[replaced].n_found;
        for (key, descriptor) in replaced_observations {
            // add_observation refuses a keyframe already linked, so the
            // survivor keeps its own record where both observed the same frame.
            self.map_points[survivor].add_observation(key, descriptor);
        }
        self.map_points[survivor].n_visible = self.map_points[survivor]
            .n_visible
            .saturating_add(replaced_visible);
        self.map_points[survivor].n_found = self.map_points[survivor]
            .n_found
            .saturating_add(replaced_found);
        self.map_points[replaced].mark_culled();
        self.update_map_point_geometry(survivor, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        Some(MapPointMergeResult {
            survivor,
            replaced,
            redirected_associations,
        })
    }
    /// Update `n_visible` and `n_found` counters for map points.
    pub fn update_observation_counts(
        &mut self,
        visible: &HashSet<usize>,
        matched: &[(usize, usize)],
    ) {
        let matched_set: HashSet<usize> = matched.iter().map(|&(mp_idx, _)| mp_idx).collect();

        for &mp_idx in visible {
            if let Some(mp) = self.map_points.get_mut(mp_idx) {
                mp.n_visible = mp.n_visible.saturating_add(1);
                if matched_set.contains(&mp_idx) {
                    mp.n_found = mp.n_found.saturating_add(1);
                }
            }
        }
    }
    /// Records preintegrated IMU measurements between two consecutive keyframes.
    /// `raw_samples` (covering `[t0, t1]`) are retained for repropagation —
    /// see `ImuFactor::raw_samples`.
    pub fn add_imu_factor(
        &mut self,
        prev_kf_idx: usize,
        curr_kf_idx: usize,
        preintegrated: PreintegratedImu,
        raw_samples: Vec<ImuMeasurement>,
        t0: f64,
        t1: f64,
    ) {
        self.imu_factors.push(ImuFactor {
            prev_kf_idx,
            curr_kf_idx,
            preintegrated,
            raw_samples,
            t0,
            t1,
        });
    }
    /// Wipes all keyframes and map points. Used to discard a failed bootstrap.
    pub fn clear_active(&mut self) {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        self.keyframes.clear();
        self.map_points.clear();
        self.imu_factors.clear();
    }
}
