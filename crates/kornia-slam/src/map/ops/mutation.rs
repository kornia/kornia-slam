//! Accepted changes to map contents and relationships.
//!
//! Behaviour-preserving relocation: the existing split between keyframe
//! association and point-side registration is unchanged here.

use crate::map::{ImuFactor, Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};
use std::collections::HashSet;

/// A triangulated point ready for map insertion: (position, descriptor, color, prev_desc_idx, curr_desc_idx).
pub type TriangulatedPoint = (Vec3F64, [u8; 32], [u8; 3], usize, usize);

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

    /// Appends a map point and returns its index.
    pub fn push_map_point(&mut self, map_point: MapPoint) -> usize {
        let idx = self.map_points.len();
        self.map_points.push(map_point);
        idx
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
            self.push_map_point(MapPoint::new(position, desc, octave, color, curr_kf_idx));
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
                    mp.add_observation_descriptor(prev_kf_idx, prev_desc);
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
        let kf_idx = keyframe.frame.idx;
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation_descriptor(kf_idx, descriptor);
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
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation_descriptor(kf_idx, descriptor);
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
                mp.observation_kf_indices.clone(),
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

        let support = |point: &MapPoint| {
            point
                .observation_kf_indices
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
        };
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
            .observation_kf_indices
            .iter()
            .copied()
            .zip(
                self.map_points[replaced]
                    .observed_descriptors
                    .iter()
                    .copied(),
            )
            .collect();
        let replaced_visible = self.map_points[replaced].n_visible;
        let replaced_found = self.map_points[replaced].n_found;
        for (keyframe_idx, descriptor) in replaced_observations {
            if !self.map_points[survivor]
                .observation_kf_indices
                .contains(&keyframe_idx)
            {
                self.map_points[survivor].add_observation_descriptor(keyframe_idx, descriptor);
            }
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
}

#[cfg(test)]
mod tests {
    use crate::map::{
        Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR,
        tests::{test_frame, test_frame_with_pose},
    };
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use std::collections::HashSet;

    #[test]
    fn upsert_keyframe_replaces_existing_idx() {
        let mut map = Map::new();

        map.upsert_keyframe(Keyframe::from_frame(test_frame(
            10,
            vec![[0u8; 32], [1u8; 32]],
        )));
        assert_eq!(map.keyframes().len(), 1);

        map.upsert_keyframe(Keyframe::from_frame(test_frame(10, vec![[2u8; 32]])));

        assert_eq!(map.keyframes().len(), 1);
        assert_eq!(
            map.get_keyframe(10)
                .expect("expected keyframe with idx 10")
                .frame
                .features
                .descriptors
                .len(),
            1
        );
    }

    #[test]
    fn push_map_point_returns_sequential_index() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 1.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 1.0),
            [1u8; 32],
            0,
            [0; 3],
            0,
        ));

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.num_map_points(), 2);
    }

    #[test]
    fn merge_map_points_redirects_associations_and_deduplicates_observers() {
        let mut map = Map::new();
        for idx in 0..3 {
            map.upsert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32], [10 + idx as u8; 32]],
            )));
        }

        let survivor = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
        ));
        map.map_points_mut()[survivor].add_observation_descriptor(1, [1; 32]);
        map.map_points_mut()[survivor].n_visible = 7;
        map.map_points_mut()[survivor].n_found = 5;
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, survivor);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(0, survivor);

        let replaced = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.01, 0.0, 5.0),
            [2; 32],
            0,
            [0; 3],
            2,
        ));
        map.map_points_mut()[replaced].add_observation_descriptor(1, [11; 32]);
        map.map_points_mut()[replaced].n_visible = 4;
        map.map_points_mut()[replaced].n_found = 3;
        map.get_keyframe_mut(2)
            .unwrap()
            .associate_map_point(0, replaced);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(1, replaced);

        let result = map.merge_map_points(survivor, replaced).unwrap();

        assert_eq!(result.survivor, survivor);
        assert_eq!(result.replaced, replaced);
        assert_eq!(result.redirected_associations, 1);
        assert!(map.map_points()[replaced].culled);
        assert_eq!(map.map_points()[survivor].n_visible, 11);
        assert_eq!(map.map_points()[survivor].n_found, 8);
        assert_eq!(
            map.map_points()[survivor]
                .observation_kf_indices
                .iter()
                .copied()
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2])
        );
        assert_eq!(map.get_keyframe(2).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(1), None);
        for keyframe in map.keyframes() {
            assert_eq!(
                keyframe
                    .map_point_by_desc_idx
                    .iter()
                    .filter(|&&point| point == Some(survivor))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn merge_map_points_rejects_invalid_or_culled_inputs() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])));
        let first = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
        ));
        let second = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [1; 32],
            0,
            [0; 3],
            0,
        ));

        assert!(map.merge_map_points(first, first).is_none());
        assert!(map.merge_map_points(first, usize::MAX).is_none());
        map.map_points_mut()[second].mark_culled();
        assert!(map.merge_map_points(first, second).is_none());
    }

    // ── Scale-invariance state (T1: deterministic, cross-checked vs ORB-SLAM3
    //    formulas; ORB-SLAM3 itself not needed since these are closed-form). ──

    /// T1b: after `update_map_point_geometry`,
    ///   max_distance == dist_to_ref_kf * scaleFactor^reference_octave
    ///   max_distance / min_distance == scaleFactor^(n_levels - 1)
    #[test]
    fn scale_geometry_distance_invariants() {
        let mut map = Map::new();
        // Reference keyframe 0 at the world origin (identity pose => camera
        // center at origin).
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        // Point referenced to KF 0, keypoint detected at octave 2, world (0,0,5).
        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            2,
            [0; 3],
            0,
        ));
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let mp = &map.map_points()[mp_idx];
        let expected_max = 5.0 * ORB_SCALE_FACTOR.powi(2);
        assert!((mp.max_distance - expected_max).abs() < 1e-9);
        assert!(
            (mp.max_distance / mp.min_distance - ORB_SCALE_FACTOR.powi(ORB_N_LEVELS as i32 - 1))
                .abs()
                < 1e-9
        );
        // Margined bounds carry the 0.8 / 1.2 factors.
        assert!((mp.min_distance_invariance() - 0.8 * mp.min_distance).abs() < 1e-12);
        assert!((mp.max_distance_invariance() - 1.2 * mp.max_distance).abs() < 1e-12);
    }

    /// T1c: mean viewing direction is the average of unit (point - cam_center)
    /// over observing keyframes; ‖normal‖ <= 1, and for a single forward-facing
    /// observation it's exactly (point - cam_center) normalized.
    #[test]
    fn mean_viewing_direction_averages_observations() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        // Single observation from the origin looking at (0,0,5): normal = +z.
        let n0 = map.map_points()[mp_idx].mean_viewing_direction;
        assert!(n0.x.abs() < 1e-9 && n0.y.abs() < 1e-9 && (n0.z - 1.0).abs() < 1e-9);

        // Add a second keyframe translated along +x by 1 (world->cam pose has
        // translation -1 along x, so camera center is at (1,0,0)).
        let kf1 = Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![[0u8; 32]],
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-1.0, 0.0, 0.0),
            ),
        ));
        map.register_observation(mp_idx, &kf1, 0);
        map.upsert_keyframe(kf1);
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let n = map.map_points()[mp_idx].mean_viewing_direction;
        // Average of (0,0,1) and (point-(1,0,0)) normalized = (-1,0,5)/sqrt(26).
        let d2 = Vec3F64::new(-1.0, 0.0, 5.0);
        let d2n = d2 / d2.length();
        let expected = Vec3F64::new(
            (n0.x + d2n.x) / 2.0,
            (n0.y + d2n.y) / 2.0,
            (n0.z + d2n.z) / 2.0,
        );
        assert!((n.x - expected.x).abs() < 1e-9);
        assert!((n.y - expected.y).abs() < 1e-9);
        assert!((n.z - expected.z).abs() < 1e-9);
        assert!(n.length() <= 1.0 + 1e-12);
    }
}
