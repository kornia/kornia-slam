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

use crate::frame::Frame;
use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

/// A frame promoted into the map, with descriptor-to-map-point associations.
#[derive(Debug, Clone)]
pub struct ImuEdge {
    pub prev_kf_idx: usize,
    pub curr_kf_idx: usize,
    pub preintegrated: PreintegratedImu,
}
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub frame: Frame,
    /// For each descriptor index in `frame.features`, associated map-point index.
    pub map_point_by_desc_idx: Vec<Option<usize>>,
    /// Metric linear velocity in world frame, initialized by visual-inertial bootstrap.
    pub velocity_world: Vec3F64,
    /// IMU bias estimate associated with this keyframe.
    pub imu_bias: ImuBias,
}

impl Keyframe {
    /// Creates a keyframe from a frame, with empty map-point associations.
    pub fn from_frame(frame: Frame) -> Self {
        let map_point_by_desc_idx = vec![None; frame.features.descriptors.len()];
        Self {
            frame,
            map_point_by_desc_idx,
            velocity_world: Vec3F64::ZERO,
            imu_bias: ImuBias::default(),
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

/// Relative standard deviation of a stereo depth measurement, as a fraction of
/// the measured depth (used to weight the BA depth residual). Depth-proportional
/// so far points—where disparity is least reliable—are downweighted.
pub const STEREO_DEPTH_REL_SIGMA: f32 = 0.05;
/// Floor on the stereo depth sigma (metres) to avoid over-trusting very near
/// points.
pub const STEREO_DEPTH_MIN_SIGMA: f32 = 0.02;

/// ORB pyramid scale factor between adjacent levels (matches the kornia-imgproc
/// ORB extractor default `downscale = 1.2`, which equals ORB-SLAM3's default).
pub const ORB_SCALE_FACTOR: f64 = 1.2;
/// Number of ORB pyramid levels (matches `OrbDetector` default `n_scales = 8`).
pub const ORB_N_LEVELS: usize = 8;

/// A persistent 3D landmark in the map.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// 3D position in world frame.
    pub position: Vec3F64,
    /// Representative ORB descriptor used for projection-guided matching.
    /// Recomputed as the descriptor with minimum median Hamming distance to
    /// all observations whenever a new observation is added (mirrors
    /// ORB-SLAM3's `MapPoint::ComputeDistinctiveDescriptors`).
    pub descriptor: [u8; 32],
    /// All descriptors of this point across observing keyframes. Parallel to
    /// `observation_kf_indices`. Used to recompute `descriptor`.
    pub observed_descriptors: Vec<[u8; 32]>,
    /// Frame indices (`Keyframe::frame.idx`) of the keyframes observing this
    /// point, parallel to `observed_descriptors`. Used to recompute the
    /// viewing direction.
    pub observation_kf_indices: Vec<usize>,
    /// Pyramid octave of this point's keypoint in the reference keyframe
    /// (`keyframe_idx`). Drives the scale-invariance distance bounds.
    pub reference_octave: u8,
    /// Mean unit viewing direction (world -> point), averaged over observing
    /// keyframes (ORB-SLAM3's `mNormalVector`). Zero until computed.
    pub mean_viewing_direction: Vec3F64,
    /// Minimum / maximum camera-to-point distance at which the point is
    /// expected to be matchable (ORB-SLAM3's `mfMinDistance` / `mfMaxDistance`,
    /// the raw values before the 0.8 / 1.2 query margins). Zero until computed.
    pub min_distance: f64,
    pub max_distance: f64,
    /// Pixel color sampled at the keypoint that created this point.
    pub color: [u8; 3],
    /// Frame index of the reference keyframe for this point's scale geometry.
    pub keyframe_idx: usize,
    /// Number of frames where this point was in the camera frustum.
    pub n_visible: u32,
    /// Number of frames where this point was successfully matched.
    pub n_found: u32,
    /// Whether this point has been culled (logically deleted).
    pub culled: bool,
}

impl MapPoint {
    /// Creates a fresh active map point with one observed descriptor from
    /// keyframe `keyframe_idx`, detected at pyramid octave `octave`.
    pub fn new(
        position: Vec3F64,
        descriptor: [u8; 32],
        octave: u8,
        color: [u8; 3],
        keyframe_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            observed_descriptors: vec![descriptor],
            observation_kf_indices: vec![keyframe_idx],
            reference_octave: octave,
            mean_viewing_direction: Vec3F64::ZERO,
            min_distance: 0.0,
            max_distance: 0.0,
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

    /// Records a new observed descriptor from keyframe `kf_idx` and refreshes
    /// the representative.
    ///
    /// The representative is the observed descriptor with minimum median
    /// Hamming distance to all others (ORB-SLAM3's
    /// `ComputeDistinctiveDescriptors`). With <=2 observations the choice is
    /// trivial; with >=3 we run the O(n^2) pairwise distance scan.
    pub fn add_observation_descriptor(&mut self, kf_idx: usize, descriptor: [u8; 32]) {
        self.observed_descriptors.push(descriptor);
        self.observation_kf_indices.push(kf_idx);
        self.recompute_representative_descriptor();
    }

    fn recompute_representative_descriptor(&mut self) {
        let n = self.observed_descriptors.len();
        match n {
            0 => {}
            1 => self.descriptor = self.observed_descriptors[0],
            2 => self.descriptor = self.observed_descriptors[0],
            _ => {
                let mut dist_buf = vec![0u32; n];
                let mut best_idx = 0usize;
                let mut best_median = u32::MAX;
                for i in 0..n {
                    for (slot, other) in dist_buf.iter_mut().zip(self.observed_descriptors.iter()) {
                        *slot = hamming_distance(&self.observed_descriptors[i], other);
                    }
                    let mid = n / 2;
                    dist_buf.select_nth_unstable(mid);
                    let median = dist_buf[mid];
                    if median < best_median {
                        best_median = median;
                        best_idx = i;
                    }
                }
                self.descriptor = self.observed_descriptors[best_idx];
            }
        }
    }

    /// Predicts the ORB pyramid level the point is expected to be observable
    /// at given the current camera-to-point distance (ORB-SLAM3's
    /// `MapPoint::PredictScale`). Returns 0 when the scale geometry is unset.
    pub fn predict_scale(&self, distance: f64, scale_factor: f64, n_levels: usize) -> usize {
        if distance <= 0.0 || self.max_distance <= 0.0 || n_levels == 0 {
            return 0;
        }
        let ratio = self.max_distance / distance;
        let level = (ratio.ln() / scale_factor.ln()).ceil();
        if level.is_nan() || level < 0.0 {
            0
        } else {
            (level as usize).min(n_levels - 1)
        }
    }

    /// Lower bound of the matchable distance range, with ORB-SLAM3's 0.8 margin.
    pub fn min_distance_invariance(&self) -> f64 {
        0.8 * self.min_distance
    }

    /// Upper bound of the matchable distance range, with ORB-SLAM3's 1.2 margin.
    pub fn max_distance_invariance(&self) -> f64 {
        1.2 * self.max_distance
    }
}

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

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
    imu_edges: Vec<ImuEdge>,
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

    /// Returns mutable access to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut [Keyframe] {
        &mut self.keyframes
    }

    pub fn add_imu_edge(
        &mut self,
        prev_kf_idx: usize,
        curr_kf_idx: usize,
        preintegrated: PreintegratedImu,
    ) {
        self.imu_edges.push(ImuEdge {
            prev_kf_idx,
            curr_kf_idx,
            preintegrated,
        });
    }

    pub fn imu_edges(&self) -> &[ImuEdge] {
        &self.imu_edges
    }

    /// Applies a metric scale to camera centers and map points.
    pub fn scale_world(&mut self, scale: f64) {
        for kf in &mut self.keyframes {
            let mut cam_to_world = kf.frame.pose_world_to_cam.inverse();
            cam_to_world.translation *= scale;
            kf.frame.pose_world_to_cam = cam_to_world.inverse();
        }

        for mp in &mut self.map_points {
            mp.position *= scale;
        }
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

    /// Wipes all keyframes and map points. Used to discard a failed bootstrap.
    pub fn clear_active(&mut self) {
        self.keyframes.clear();
        self.map_points.clear();
        self.imu_edges.clear();
    }

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

    /// Run a 2-keyframe bundle adjustment over the bootstrap pair.
    ///
    /// Operates on the two most recently inserted keyframes (the bootstrap
    /// pair). Optimizes the newer KF's pose and all map points observed by
    /// either KF; the older KF is held fixed as the gauge anchor. Mirrors
    /// ORB-SLAM3's `GlobalBundleAdjustemnt(map, 20)` in
    /// `CreateInitialMapMonocular`.
    ///
    /// Returns `true` if BA ran and wrote back optimized values; `false` if
    /// there were too few observations or the optimizer errored (map left
    /// untouched in that case).
    pub fn run_initial_ba(&mut self, camera: &PinholeCamera) -> bool {
        const MAX_ITERS: usize = 5;
        const HUBER_SCALE_SQ: f32 = 5.991;
        const MIN_OBSERVATIONS: usize = 8;

        let n = self.keyframes.len();
        if n < 2 {
            return false;
        }

        // Last two keyframes; older is gauge-fixed.
        let kf_indices = [n - 2, n - 1];

        let mut mp_set: HashSet<usize> = HashSet::new();
        for &kf_idx in &kf_indices {
            for mp_idx in self.keyframes[kf_idx]
                .map_point_by_desc_idx
                .iter()
                .flatten()
            {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    mp_set.insert(*mp_idx);
                }
            }
        }
        if mp_set.is_empty() {
            return false;
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

        let poses: Vec<Pose3d> = kf_indices
            .iter()
            .map(|&i| self.keyframes[i].frame.pose_world_to_cam)
            .collect();

        let mut observations = Vec::new();
        for (pose_idx, &kf_idx) in kf_indices.iter().enumerate() {
            let is_fixed = pose_idx == 0;
            let kf = &self.keyframes[kf_idx];
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                if let Some(mp_idx) = mp_opt {
                    let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                        continue;
                    };
                    if let Some(kp) = kf.frame.features.keypoints_xy.get(desc_idx) {
                        let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                        let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                        observations.push(BaObservation {
                            pose_idx,
                            point_idx,
                            pixel: [p.x as f32, p.y as f32],
                            fixed_pose: is_fixed,
                            fixed_point: false,
                            depth_meas,
                            depth_sigma,
                        });
                    }
                }
            }
        }

        if observations.len() < MIN_OBSERVATIONS {
            return false;
        }

        let (sq_err_before, depth_before, kf1_t_before) =
            initial_ba_diagnostics(&poses, &points, &observations, camera);

        let params = BaParams {
            max_iterations: MAX_ITERS,
            // Two-view monocular BA has a 1-DOF scale gauge (only KF0 is
            // fixed). Bump LM damping so the augmented normal equations stay
            // well-conditioned even though H is rank-deficient.
            initial_lambda: 1.0,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: HUBER_SCALE_SQ,
            ..BaParams::default()
        };

        let ba_result = match bundle_adjust_schur(&poses, &points, &observations, camera, &params) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[init_ba] bundle_adjust failed: {e}");
                return false;
            }
        };

        let (sq_err_after, depth_after, kf1_t_after) =
            initial_ba_diagnostics(&ba_result.poses, &ba_result.points, &observations, camera);

        eprintln!(
            "[init_ba] before: reproj_rms={:.3}px median_depth={:.3} kf1_t_norm={:.3} obs={}",
            sq_err_before.sqrt(),
            depth_before,
            kf1_t_before,
            observations.len()
        );
        eprintln!(
            "[init_ba] after:  reproj_rms={:.3}px median_depth={:.3} kf1_t_norm={:.3} iters={} converged={}",
            sq_err_after.sqrt(),
            depth_after,
            kf1_t_after,
            ba_result.iterations,
            ba_result.converged
        );

        self.keyframes[kf_indices[1]].frame.pose_world_to_cam = ba_result.poses[1];
        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = ba_result.points[local_idx];
            }
        }
        // Positions (and KF1's pose) moved: refresh scale geometry.
        for &global_idx in &mp_global_indices {
            self.update_map_point_geometry(global_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }

        // Diagnostics: sample one point's scale state for a sanity check.
        if let Some(&sample) = mp_global_indices.first()
            && let Some(mp) = self.map_points.get(sample)
        {
            let normal_len = mp.mean_viewing_direction.length();
            let predicted = mp.predict_scale(depth_after.max(1e-6), ORB_SCALE_FACTOR, ORB_N_LEVELS);
            eprintln!(
                "[init_ba] scale_state[mp{sample}]: min_dist={:.3} max_dist={:.3} normal_len={:.3} predict_scale@median={predicted}",
                mp.min_distance, mp.max_distance, normal_len,
            );
        }

        true
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
                        let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                        observations.push(BaObservation {
                            pose_idx: kf_idx,
                            point_idx,
                            pixel: [p.x as f32, p.y as f32],
                            fixed_pose: is_fixed,
                            fixed_point: false,
                            depth_meas,
                            depth_sigma,
                        });
                    }
                }
            }
        }

        if observations.len() < MIN_OBSERVATIONS {
            return;
        }

        let ba_result =
            match bundle_adjust_schur(&poses, &points, &observations, camera, &BaParams::default())
            {
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
        for &global_idx in &mp_global_indices {
            self.update_map_point_geometry(global_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
    }
}

/// Depth measurement + sigma for a BA observation at `desc_idx` of `kf`.
///
/// Returns `(Some(z), sigma)` when the keyframe's keypoint has a valid stereo
/// depth (anchoring the BA's metric scale), else `(None, 1.0)` for a pure
/// reprojection observation.
fn stereo_depth_obs(kf: &Keyframe, desc_idx: usize) -> (Option<f32>, f32) {
    match kf.frame.stereo_depth(desc_idx) {
        Some(z) => (
            Some(z),
            (STEREO_DEPTH_REL_SIGMA * z).max(STEREO_DEPTH_MIN_SIGMA),
        ),
        None => (None, 1.0),
    }
}

/// Returns (mean_sq_reproj_error, median_depth_in_kf0_frame, kf1_translation_norm).
fn initial_ba_diagnostics(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
) -> (f64, f64, f64) {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for obs in observations {
        if let (Some(pose), Some(pt)) = (poses.get(obs.pose_idx), points.get(obs.point_idx))
            && let Some(err_sq) = camera.reprojection_error_sq_world(
                pose,
                pt,
                obs.pixel[0] as f64,
                obs.pixel[1] as f64,
            )
        {
            sum_sq += err_sq;
            count += 1;
        }
    }
    let mean_sq = if count > 0 {
        sum_sq / count as f64
    } else {
        0.0
    };

    let median_depth = match poses.first() {
        Some(kf0) => {
            let mut depths: Vec<f64> = points
                .iter()
                .map(|p| kf0.transform_point(p).z)
                .filter(|&z| z > 0.0)
                .collect();
            if depths.is_empty() {
                0.0
            } else {
                let mid = depths.len() / 2;
                depths.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                depths[mid]
            }
        }
        None => 0.0,
    };

    let kf1_t_norm = poses.get(1).map(|p| p.translation.length()).unwrap_or(0.0);

    (mean_sq, median_depth, kf1_t_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        test_frame_with_pose(idx, descriptors, Pose3d::IDENTITY)
    }

    fn test_frame_with_pose(idx: usize, descriptors: Vec<[u8; 32]>, pose: Pose3d) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
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
        let mp = MapPoint::new(Vec3F64::new(1.0, 2.0, 3.0), [9u8; 32], 0, [0; 3], 5);

        assert_eq!(mp.position, Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(mp.descriptor, [9u8; 32]);
        assert_eq!(mp.keyframe_idx, 5);
        assert_eq!(mp.n_visible, 1);
        assert_eq!(mp.n_found, 1);
        assert!(!mp.culled);
    }

    #[test]
    fn map_point_tracking_helpers_work() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0);
        mp.n_visible = 10;
        mp.n_found = 4;

        assert!((mp.found_ratio() - 0.4).abs() < 1e-9);
        mp.mark_culled();
        assert!(mp.culled);
    }

    #[test]
    fn stereo_depth_obs_uses_proportional_sigma_with_floor() {
        let mut frame = test_frame(0, vec![[0u8; 32], [1u8; 32]]);
        frame.depth = vec![10.0, -1.0];
        frame.u_right = vec![5.0, -1.0];
        let kf = Keyframe::from_frame(frame);

        // Valid depth: sigma = 0.05 * 10 = 0.5.
        let (d, s) = stereo_depth_obs(&kf, 0);
        assert_eq!(d, Some(10.0));
        assert!((s - 0.5).abs() < 1e-6);

        // Sentinel depth: no measurement.
        let (d1, s1) = stereo_depth_obs(&kf, 1);
        assert_eq!(d1, None);
        assert!((s1 - 1.0).abs() < 1e-9);

        // Very near depth clamps to the sigma floor.
        let mut near = test_frame(1, vec![[0u8; 32]]);
        near.depth = vec![0.1]; // 0.05 * 0.1 = 0.005 < floor
        let kf_near = Keyframe::from_frame(near);
        let (_, s2) = stereo_depth_obs(&kf_near, 0);
        assert!((s2 - STEREO_DEPTH_MIN_SIGMA).abs() < 1e-9);
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
    fn cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            0,
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

    // ── Scale-invariance state (T1: deterministic, cross-checked vs ORB-SLAM3
    //    formulas; ORB-SLAM3 itself not needed since these are closed-form). ──

    /// T1a: `predict_scale` matches ORB-SLAM3's
    /// `nScale = clamp(ceil(log(maxDist/dist) / log(scaleFactor)), 0, nLevels-1)`.
    #[test]
    fn predict_scale_matches_orbslam3_closed_form() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0);
        mp.max_distance = 10.0;
        let sf = ORB_SCALE_FACTOR;
        let n = ORB_N_LEVELS;

        for &dist in &[40.0_f64, 12.0, 10.0, 8.0, 3.7, 0.9, 0.01] {
            let want = {
                let ratio = mp.max_distance / dist;
                let lvl = (ratio.ln() / sf.ln()).ceil();
                if lvl.is_nan() || lvl < 0.0 {
                    0
                } else {
                    (lvl as usize).min(n - 1)
                }
            };
            assert_eq!(mp.predict_scale(dist, sf, n), want, "dist={dist}");
        }

        // Degenerate inputs return level 0.
        assert_eq!(mp.predict_scale(0.0, sf, n), 0);
        let mut unset = MapPoint::new(Vec3F64::ZERO, [0u8; 32], 0, [0; 3], 0);
        unset.max_distance = 0.0;
        assert_eq!(unset.predict_scale(5.0, sf, n), 0);
    }

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
