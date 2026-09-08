//! Read-only views and measurements derived from map state.

use crate::map::{Keyframe, Map};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use std::collections::{HashMap, HashSet};

/// Quality metrics for a freshly-bootstrapped 2-keyframe map.
///
/// Used as the gate for accepting a bootstrap result. Mirrors
/// ORB-SLAM3's reset criteria in `CreateInitialMapMonocular`:
/// `medianDepth < 0 || TrackedMapPoints(1) < 50`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitialMapHealth {
    /// Number of map points with positive depth in both bootstrap KFs.
    pub valid_in_both: usize,
    /// Median depth of valid points in the older KF's frame.
    pub median_depth_older_kf: f64,
}

impl Map {
    /// Health metrics for the just-bootstrapped pair of keyframes.
    ///
    /// Inspects the last two keyframes in insertion order and reports how
    /// many associated map points still have positive depth in both KFs,
    /// plus the median depth in the older KF's frame. Used to decide whether
    /// a freshly-bootstrapped map is safe to commit.
    pub fn initial_map_health(&self) -> InitialMapHealth {
        let n = self.keyframes.len();
        if n < 2 {
            return InitialMapHealth::default();
        }
        let kf_older = &self.keyframes[n - 2];
        let kf_newer = &self.keyframes[n - 1];
        let pose_older = kf_older.frame.pose_world_to_cam;
        let pose_newer = kf_newer.frame.pose_world_to_cam;

        // Collect MPs observed by either KF, dedup.
        let mut seen: HashSet<usize> = HashSet::new();
        for mp_idx in kf_older
            .map_point_by_desc_idx
            .iter()
            .chain(kf_newer.map_point_by_desc_idx.iter())
            .flatten()
        {
            seen.insert(*mp_idx);
        }

        let mut depths_older: Vec<f64> = Vec::with_capacity(seen.len());
        let mut valid_in_both = 0usize;
        for idx in seen {
            let Some(mp) = self.map_points.get(idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            let z_older = pose_older.transform_point(&mp.position).z;
            let z_newer = pose_newer.transform_point(&mp.position).z;
            if z_older > 0.0 && z_newer > 0.0 {
                valid_in_both += 1;
                depths_older.push(z_older);
            }
        }

        let median_depth = if depths_older.is_empty() {
            0.0
        } else {
            let mid = depths_older.len() / 2;
            depths_older.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            depths_older[mid]
        };

        InitialMapHealth {
            valid_in_both,
            median_depth_older_kf: median_depth,
        }
    }

    /// Returns the subset of `candidates` (map-point indices) that are
    /// non-culled and project inside the image frustum.
    pub fn map_points_in_frustum(
        &self,
        candidates: &[usize],
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
    ) -> HashSet<usize> {
        let mut visible = HashSet::new();
        for &mp_idx in candidates {
            let Some(mp) = self.map_points.get(mp_idx) else {
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

    /// Covisibility neighbors of keyframe `kf_idx`: other keyframes that share
    /// observed map points with it, as `(frame_idx, weight)` where `weight` is
    /// the number of map points both observe, sorted by descending weight.
    ///
    /// Derived on demand by inverting `MapPoint::observation_kf_indices` — no
    /// cached graph state, so it stays correct across culls and fuses. Mirrors
    /// ORB-SLAM3's `KeyFrame::UpdateConnections`: links below `min_weight` are
    /// dropped, but if none reach the threshold the single strongest link is
    /// kept so the graph stays connected.
    pub fn covisible_keyframes(&self, kf_idx: usize, min_weight: usize) -> Vec<(usize, usize)> {
        let Some(kf) = self.get_keyframe(kf_idx) else {
            return Vec::new();
        };

        let mut weights: HashMap<usize, usize> = HashMap::new();
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            let Some(mp) = self.map_points.get(*mp_idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            for &obs_kf in &mp.observation_kf_indices {
                if obs_kf != kf_idx {
                    *weights.entry(obs_kf).or_insert(0) += 1;
                }
            }
        }

        let mut connections: Vec<(usize, usize)> = weights.into_iter().collect();
        // Descending weight; break ties by descending frame index for determinism.
        connections.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

        let strongest = connections.first().copied();
        connections.retain(|&(_, w)| w >= min_weight);
        if connections.is_empty() {
            // Keep the best link so an under-connected KF is never orphaned.
            connections.extend(strongest);
        }
        connections
    }

    /// Builds the local map for tracking: indices of non-culled map points
    /// observed by the current keyframe, the keyframes owning the tracked
    /// points, and their covisibility neighbors.
    pub fn build_local_map_point_indices(
        &self,
        tracked_matches: &[(usize, usize)],
        current_keyframe: Option<&Keyframe>,
    ) -> Vec<usize> {
        const MAX_VOTED_KEYFRAMES: usize = 10;
        const MAX_COVIS_NEIGHBORS: usize = 10;
        const MIN_COVIS_WEIGHT: usize = 15;

        // Vote over every keyframe observing each tracked point (ORB-SLAM3's
        // UpdateLocalKeyFrames). The vote map already encodes covisibility
        // with the current frame, so no per-seed covisibility recomputation
        // is needed — that scan is O(KF points x observations) per seed and
        // dominated per-frame tracking cost.
        let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
        for &(mp_idx, _) in tracked_matches {
            if let Some(mp) = self.map_points.get(mp_idx) {
                for &obs_kf in &mp.observation_kf_indices {
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
                for (nb_idx, _) in self
                    .covisible_keyframes(kf.frame.idx, MIN_COVIS_WEIGHT)
                    .into_iter()
                    .take(MAX_COVIS_NEIGHBORS)
                {
                    local_kf_indices.insert(nb_idx);
                }
            }
        }
        for (kf_idx, _) in voted_kfs.into_iter().take(MAX_VOTED_KEYFRAMES) {
            local_kf_indices.insert(kf_idx);
        }

        let mut mp_indices: HashSet<usize> = HashSet::new();
        for &(mp_idx, _) in tracked_matches {
            if mp_idx < self.map_points.len() {
                mp_indices.insert(mp_idx);
            }
        }
        for kf in &self.keyframes {
            if !local_kf_indices.contains(&kf.frame.idx) {
                continue;
            }
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if *mp_idx < self.map_points.len() {
                    mp_indices.insert(*mp_idx);
                }
            }
        }

        mp_indices.retain(|&idx| !self.map_points[idx].culled);

        let mut global_indices: Vec<usize> = mp_indices.into_iter().collect();
        global_indices.sort_unstable();

        if global_indices.len() < 4 && self.map_points.len() >= 4 {
            global_indices = (0..self.map_points.len())
                .filter(|&idx| !self.map_points[idx].culled)
                .collect();
        }

        global_indices
    }
}

#[cfg(test)]
mod tests {
    use crate::map::{Keyframe, Map, MapPoint, tests::test_frame};
    use kornia_algebra::Vec3F64;
    use std::collections::HashSet;

    #[test]
    fn covisible_keyframes_weights_by_shared_points() {
        // Three keyframes; map points observed by overlapping subsets:
        //   A: KF0, KF1, KF2   B: KF0, KF1   C: KF0, KF2
        // From KF0's view: KF1 shares {A,B}=2, KF2 shares {A,C}=2.
        let mut map = Map::new();
        for idx in 0..3 {
            // Four descriptor slots so KF0 can hold its three observations.
            map.upsert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                (0..4).map(|d| [(idx * 10 + d) as u8; 32]).collect(),
            )));
        }

        // (observers, name) -> push a map point and associate desc slot 0/1.
        let specs: [(&[usize], usize); 3] = [(&[0, 1, 2], 0), (&[0, 1], 1), (&[0, 2], 1)];
        for (observers, _) in specs {
            let mp_idx = map.push_map_point(MapPoint::new(
                Vec3F64::new(0.0, 0.0, 1.0),
                [0u8; 32],
                0,
                [0; 3],
                observers[0],
            ));
            for (slot, &kf_idx) in observers.iter().enumerate() {
                if slot > 0
                    && let Some(mp) = map.map_points.get_mut(mp_idx)
                {
                    mp.add_observation_descriptor(kf_idx, [0u8; 32]);
                }
                // Associate a free descriptor slot in that keyframe.
                let kf = map.get_keyframe_mut(kf_idx).unwrap();
                let free = kf.map_point_by_desc_idx.iter().position(|s| s.is_none());
                kf.associate_map_point(free.unwrap_or(0), mp_idx);
            }
        }

        // min_weight 1 keeps both neighbors; sorted by descending weight.
        let covis = map.covisible_keyframes(0, 1);
        assert_eq!(covis.len(), 2);
        assert!(covis.iter().all(|&(_, w)| w == 2));
        assert_eq!(
            covis.iter().map(|&(k, _)| k).collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );

        // High threshold drops everything but the strongest link is retained.
        let fallback = map.covisible_keyframes(0, 99);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].1, 2);

        // Unknown keyframe yields no connections.
        assert!(map.covisible_keyframes(999, 1).is_empty());
    }
}
