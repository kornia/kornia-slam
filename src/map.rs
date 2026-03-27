//! Map: keyframes, map points, local map selection, and culling.
//!
//! ```text
//!    +--------+
//!    | Frame  |
//!    +--------+
//!         |
//!         v
//!    +----------------------+
//!    | Keyframe             |
//!    | frame + desc -> mp   |
//!    +----------------------+
//!         |
//!         v
//!    +----------------------+
//!    | Map                  |
//!    | keyframes + points   |
//!    +----------------------+
//!
//!    ops:
//!      * upsert_keyframe
//!      * push_map_point
//!      * build_local_map_points
//!      * cull
//!      * run_local_ba
//! ```

use std::collections::{HashMap, HashSet};

use kornia_3d::ba::{self, BaObservation, BaParams};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

use crate::frame::Frame;

/// A frame promoted into the map, with descriptor-to-map-point associations.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub frame: Frame,
    /// For each descriptor index in `frame.features`, associated map-point index.
    pub map_point_by_desc_idx: Vec<Option<usize>>,
}

impl Keyframe {
    /// Creates a keyframe from a frame, with empty map-point associations.
    pub fn from_frame(frame: Frame) -> Self {
        let map_point_by_desc_idx = vec![None; frame.features.descriptors.len()];
        Self {
            frame,
            map_point_by_desc_idx,
        }
    }

    /// Associates a descriptor slot with a persistent map point.
    pub fn associate_map_point(&mut self, desc_idx: usize, mp_idx: usize) {
        if let Some(slot) = self.map_point_by_desc_idx.get_mut(desc_idx) {
            *slot = Some(mp_idx);
        }
    }

    /// Clears the map-point association for a descriptor slot.
    pub fn clear_map_point(&mut self, desc_idx: usize) {
        if let Some(slot) = self.map_point_by_desc_idx.get_mut(desc_idx) {
            *slot = None;
        }
    }

    /// Returns the associated map-point index for a descriptor slot.
    pub fn map_point(&self, desc_idx: usize) -> Option<usize> {
        self.map_point_by_desc_idx.get(desc_idx).copied().flatten()
    }

    /// Counts how many descriptor slots currently reference a map point.
    pub fn num_associated_points(&self) -> usize {
        self.map_point_by_desc_idx
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }
}

/// A triangulated point ready for map insertion: (position, descriptor, color, prev_desc_idx, curr_desc_idx).
pub type TriangulatedPoint = (Vec3F64, [u8; 32], [u8; 3], usize, usize);

/// A persistent 3D landmark in the map.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// 3D position in world frame.
    pub position: Vec3F64,
    /// ORB descriptor used for projection-guided matching.
    pub descriptor: [u8; 32],
    /// Pixel color sampled at the keypoint that created this point.
    pub color: [u8; 3],
    /// Index of the keyframe that first observed this point.
    pub keyframe_idx: usize,
    /// Number of frames where this point was in the camera frustum.
    pub n_visible: u32,
    /// Number of frames where this point was successfully matched.
    pub n_found: u32,
    /// Whether this point has been culled (logically deleted).
    pub culled: bool,
}

impl MapPoint {
    /// Creates a fresh active map point.
    pub fn new(
        position: Vec3F64,
        descriptor: [u8; 32],
        color: [u8; 3],
        keyframe_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            color,
            keyframe_idx,
            n_visible: 1,
            n_found: 1,
            culled: false,
        }
    }

    /// Marks the point as logically deleted.
    pub fn mark_culled(&mut self) {
        self.culled = true;
    }

    /// Returns the tracking success ratio for this point.
    pub fn found_ratio(&self) -> f64 {
        if self.n_visible == 0 {
            return 0.0;
        }
        self.n_found as f64 / self.n_visible as f64
    }
}

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
}

impl Map {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all keyframes.
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Returns the number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }

    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }

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

    /// Inserts triangulated 3D points as map points and associates them to keyframes.
    ///
    /// For each entry, creates a `MapPoint` and associates it with `curr_kf`.
    /// If `prev_kf` is provided, associates it there too.
    /// Inserts triangulated 3D points as map points and associates them to keyframes.
    ///
    /// For each entry, creates a `MapPoint` and associates it with `curr_kf`.
    /// If `prev_kf` is provided, associates it there too.
    pub fn add_triangulated_points(
        &mut self,
        prev_kf: Option<&mut Keyframe>,
        curr_kf: &mut Keyframe,
        points: &[TriangulatedPoint],
        keyframe_idx: usize,
    ) -> usize {
        let first_mp_idx = self.map_points.len();
        for (i, &(position, descriptor, color, _, curr_desc_idx)) in points.iter().enumerate() {
            self.push_map_point(MapPoint::new(position, descriptor, color, keyframe_idx));
            curr_kf.associate_map_point(curr_desc_idx, first_mp_idx + i);
        }
        if let Some(prev) = prev_kf {
            for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().enumerate() {
                prev.associate_map_point(prev_desc_idx, first_mp_idx + i);
            }
        }
        points.len()
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

    /// Returns a mutable reference to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut Vec<Keyframe> {
        &mut self.keyframes
    }

    /// Returns indices of non-culled map points that project inside the image frustum.
    pub fn map_points_in_frustum(
        &self,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
    ) -> HashSet<usize> {
        let mut visible = HashSet::new();
        for (mp_idx, mp) in self.map_points.iter().enumerate() {
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

    /// Builds a local map of visible points from nearby keyframes.
    pub fn build_local_map_points(
        &self,
        tracked_matches: &[(usize, usize)],
        current_keyframe: Option<&Keyframe>,
    ) -> (Vec<MapPoint>, Vec<usize>) {
        const MAX_VOTED_KEYFRAMES: usize = 10;
        const MAX_RECENT_KEYFRAMES: usize = 10;

        let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
        for &(mp_idx, _) in tracked_matches {
            if let Some(mp) = self.map_points.get(mp_idx) {
                *keyframe_votes.entry(mp.keyframe_idx).or_insert(0) += 1;
            }
        }

        let mut voted_kfs: Vec<(usize, usize)> = keyframe_votes.into_iter().collect();
        voted_kfs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

        let mut local_kf_indices: HashSet<usize> = HashSet::new();
        if let Some(kf) = current_keyframe {
            local_kf_indices.insert(kf.frame.idx);
        }
        for (kf_idx, _) in voted_kfs.into_iter().take(MAX_VOTED_KEYFRAMES) {
            local_kf_indices.insert(kf_idx);
        }
        for kf in self.keyframes.iter().rev().take(MAX_RECENT_KEYFRAMES) {
            local_kf_indices.insert(kf.frame.idx);
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

        let mut global_indices: Vec<usize> = mp_indices.into_iter().collect();
        global_indices.sort_unstable();

        if global_indices.len() < 4 && self.map_points.len() >= 4 {
            global_indices = (0..self.map_points.len()).collect();
        }

        let local_map_points: Vec<MapPoint> = global_indices
            .iter()
            .filter_map(|&idx| self.map_points.get(idx).filter(|mp| !mp.culled).cloned())
            .collect();
        (local_map_points, global_indices)
    }

    /// Cull map points with poor observation ratios or that project behind cameras.
    pub fn cull(&mut self) {
        const MIN_OBSERVATIONS: u32 = 5;
        const MIN_FOUND_RATIO: f64 = 0.20;

        let mut n_culled = 0usize;

        for mp in self.map_points.iter_mut() {
            if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
                continue;
            }
            if mp.found_ratio() < MIN_FOUND_RATIO {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        let mut behind_camera: Vec<usize> = Vec::new();
        for kf in &self.keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 1e-8 {
                        behind_camera.push(*mp_idx);
                    }
                }
            }
        }

        for mp_idx in &behind_camera {
            if let Some(mp) = self.map_points.get_mut(*mp_idx)
                && !mp.culled
            {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        if n_culled > 0 {
            let culled_set: HashSet<usize> = self
                .map_points
                .iter()
                .enumerate()
                .filter(|(_, mp)| mp.culled)
                .map(|(i, _)| i)
                .collect();

            for kf in &mut self.keyframes {
                for desc_idx in 0..kf.map_point_by_desc_idx.len() {
                    if let Some(mp_idx) = kf.map_point(desc_idx)
                        && culled_set.contains(&mp_idx)
                    {
                        kf.clear_map_point(desc_idx);
                    }
                }
            }
        }
    }

    /// Run local bundle adjustment over recent keyframes and their observed map points.
    ///
    /// Collects the last N active keyframes, gathers observations (undistorting keypoints
    /// via camera), calls `kornia_3d::ba::bundle_adjust`, and writes back optimized poses
    /// and point positions.
    pub fn run_local_ba(&mut self, camera: &PinholeCamera) {
        const MAX_ACTIVE_KFS: usize = 3;
        const MIN_OBSERVATIONS: usize = 8;

        let n_kfs = self.keyframes.len();
        if n_kfs < 2 {
            return;
        }

        let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

        let mut mp_set: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes[active_start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    mp_set.insert(*mp_idx);
                }
            }
        }
        if mp_set.is_empty() {
            return;
        }

        let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
        mp_global_indices.sort_unstable();

        let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let points: Vec<Vec3F64> = mp_global_indices
            .iter()
            .map(|&idx| self.map_points[idx].position)
            .collect();

        let poses: Vec<Pose3d> = self
            .keyframes
            .iter()
            .map(|kf| kf.frame.pose_world_to_cam)
            .collect();

        let mut observations = Vec::new();
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
            let is_fixed = kf_idx < active_start;
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                if let Some(mp_idx) = mp_opt {
                    let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                        continue;
                    };
                    if let Some(kp) = kf.frame.features.keypoints_xy.get(desc_idx) {
                        let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                        observations.push(BaObservation {
                            pose_idx: kf_idx,
                            point_idx,
                            pixel: [p.x as f32, p.y as f32],
                            fixed_pose: is_fixed,
                        });
                    }
                }
            }
        }

        if observations.len() < MIN_OBSERVATIONS {
            return;
        }

        let ba_result =
            match ba::bundle_adjust(&poses, &points, &observations, camera, &BaParams::default()) {
                Ok(r) => r,
                Err(_) => return,
            };

        for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
            if kf_idx >= active_start {
                self.keyframes[kf_idx].frame.pose_world_to_cam = *pose;
            }
        }

        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = ba_result.points[local_idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
        }
    }

    #[test]
    fn keyframe_from_frame_initializes_map_point_slots() {
        let keyframe = Keyframe::from_frame(test_frame(7, vec![[0u8; 32], [1u8; 32], [2u8; 32]]));

        assert_eq!(keyframe.frame.idx, 7);
        assert_eq!(keyframe.map_point_by_desc_idx.len(), 3);
        assert!(
            keyframe
                .map_point_by_desc_idx
                .iter()
                .all(|slot| slot.is_none())
        );
    }

    #[test]
    fn keyframe_association_helpers_work() {
        let mut keyframe = Keyframe::from_frame(test_frame(1, vec![[0u8; 32], [1u8; 32]]));

        keyframe.associate_map_point(1, 42);
        assert_eq!(keyframe.map_point(1), Some(42));
        assert_eq!(keyframe.num_associated_points(), 1);

        keyframe.clear_map_point(1);
        assert_eq!(keyframe.map_point(1), None);
        assert_eq!(keyframe.num_associated_points(), 0);
    }

    #[test]
    fn map_point_new_sets_active_defaults() {
        let mp = MapPoint::new(Vec3F64::new(1.0, 2.0, 3.0), [9u8; 32], [0; 3], 5);

        assert_eq!(mp.position, Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(mp.descriptor, [9u8; 32]);
        assert_eq!(mp.keyframe_idx, 5);
        assert_eq!(mp.n_visible, 1);
        assert_eq!(mp.n_found, 1);
        assert!(!mp.culled);
    }

    #[test]
    fn map_point_tracking_helpers_work() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], [0; 3], 0);
        mp.n_visible = 10;
        mp.n_found = 4;

        assert!((mp.found_ratio() - 0.4).abs() < 1e-9);
        mp.mark_culled();
        assert!(mp.culled);
    }

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
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 1.0),
            [1u8; 32],
            [0; 3],
            0,
        ));

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.num_map_points(), 2);
    }

    #[test]
    fn cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            [0; 3],
            0,
        ));
        map.map_points_mut()[first_idx].n_visible = 10;
        map.map_points_mut()[first_idx].n_found = 1;
        map.map_points_mut()[second_idx].n_visible = 10;
        map.map_points_mut()[second_idx].n_found = 5;

        map.cull();

        assert!(map.map_points()[first_idx].culled);
        assert!(!map.map_points()[second_idx].culled);
    }
}
