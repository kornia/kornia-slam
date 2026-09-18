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
}
