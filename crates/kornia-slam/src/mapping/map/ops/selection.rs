//! Read-only selection over the map: accessors, frustum and covisibility queries.

use crate::map::{ImuFactor, Keyframe, Map, MapPoint};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use std::collections::HashMap;
use std::collections::HashSet;

impl Map {
    /// Returns all keyframes.
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }
    /// Returns mutable access to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut [Keyframe] {
        &mut self.keyframes
    }
    /// Returns all keyframe-to-keyframe IMU factors in insertion order.
    pub fn imu_factors(&self) -> &[ImuFactor] {
        &self.imu_factors
    }
    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }
    /// Returns the number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }
    /// Returns the number of non-culled (active) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map_points.iter().filter(|mp| !mp.culled).count()
    }
    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }
    /// Mutable version of [`Map::get_keyframe`]. Needed when a triangulation
    /// or fusion pass produces a new observation that must be recorded on the
    /// live keyframe in the map (not a clone).
    pub fn get_keyframe_mut(&mut self, idx: usize) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|kf| kf.frame.idx == idx)
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
