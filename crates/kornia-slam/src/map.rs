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
use kornia_sensors::imu::{ImuBias, PreintegratedImu};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_algebra::optim::{LevenbergMarquardt, Problem, Variable, VariableType};
use kornia_image::ImageSize;

use crate::factors::inertial::InertialFactor;
use crate::factors::bias_random_walk::BiasRandomWalkFactor;
use crate::factors::reprojection::ReprojectionFactor;
use crate::frame::Frame;

/// A frame promoted into the map, with descriptor-to-map-point associations.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub frame: Frame,
    /// For each descriptor index in `frame.features`, associated map-point index.
    pub map_point_by_desc_idx: Vec<Option<usize>>,
    /// Preintegrated IMU measurements from the previous keyframe to this one.
    /// `None` for the first keyframe or when no IMU data is available.
    pub preintegrated_imu: Option<PreintegratedImu>,
}

impl Keyframe {
    /// Creates a keyframe from a frame, with empty map-point associations.
    pub fn from_frame(frame: Frame) -> Self {
        let map_point_by_desc_idx = vec![None; frame.features.descriptors.len()];
        Self {
            frame,
            map_point_by_desc_idx,
            preintegrated_imu: None,
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

/// Result returned by `run_local_ba_inertial`.
///
/// Contains optimized velocities and biases for each active keyframe,
/// so the caller can update its system state.
pub struct InertialBaResult {
    /// Optimized velocity for the most recent (active) keyframe.
    pub latest_velocity: Vec3F64,
    /// Optimized bias for the most recent (active) keyframe.
    pub latest_bias: ImuBias,
}

impl Map {
    /// Run local bundle adjustment with inertial factors.
    ///
    /// This extends the visual-only BA by adding:
    ///   - Velocity (R³) and bias (R⁶) variables per keyframe
    ///   - `InertialFactor` between consecutive keyframes
    ///   - `BiasRandomWalkFactor` between consecutive keyframe biases
    ///   - `ReprojectionFactor` for each map point observation
    ///
    /// Variables before the active window are fixed (pose, velocity, bias).
    /// Returns `None` if there aren't enough keyframes with IMU data.
    pub fn run_local_ba_inertial(
        &mut self,
        camera: &PinholeCamera,
        gravity: &[f32; 3],
        current_velocity: &Vec3F64,
        current_bias: &ImuBias,
    ) -> Option<InertialBaResult> {
        const MAX_ACTIVE_KFS: usize = 5;
        const MIN_OBSERVATIONS: usize = 8;

        let n_kfs = self.keyframes.len();
        if n_kfs < 2 {
            return None;
        }

        let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

        // Check that active keyframes have IMU preintegration
        let has_imu = self.keyframes[active_start..]
            .iter()
            .skip(1) // first KF in window won't have preintegration to its predecessor
            .all(|kf| kf.preintegrated_imu.is_some());
        if !has_imu {
            return None;
        }

        // ── Build the optimization problem ──────────────────────────────

        let mut problem = Problem::new();

        // Collect map points observed by active keyframes
        let mut mp_set: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes[active_start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx) {
                    if !mp.culled {
                        mp_set.insert(*mp_idx);
                    }
                }
            }
        }
        if mp_set.is_empty() {
            return None;
        }

        let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
        mp_global_indices.sort_unstable();
        let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // ── Add variables ───────────────────────────────────────────────

        // Pose, velocity, and bias variables for each keyframe
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
            let is_fixed = kf_idx < active_start;
            let pose_name = format!("pose_{kf_idx}");
            let vel_name = format!("vel_{kf_idx}");
            let bias_name = format!("bias_{kf_idx}");

            // Pose (SE3): [qw, qx, qy, qz, tx, ty, tz]
            let se3_params = pose_to_se3_params(&kf.frame.pose_world_to_cam);
            if !is_fixed {
                problem
                    .add_variable(
                        Variable::new(pose_name, VariableType::SE3, vec![0.0; 7]),
                        se3_params,
                    )
                    .ok()?;
            }

            // Velocity (R³)
            let vel_init = if kf_idx == n_kfs - 1 {
                // Most recent keyframe uses current velocity
                vec![
                    current_velocity.x as f32,
                    current_velocity.y as f32,
                    current_velocity.z as f32,
                ]
            } else {
                // For intermediate keyframes, initialize to zero
                // (the optimizer will find the right values)
                vec![0.0; 3]
            };
            if !is_fixed {
                problem
                    .add_variable(Variable::euclidean(vel_name, 3), vel_init)
                    .ok()?;
            }

            // Bias (R⁶): [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
            let bias_init = if kf_idx == n_kfs - 1 {
                vec![
                    current_bias.gyro.x as f32,
                    current_bias.gyro.y as f32,
                    current_bias.gyro.z as f32,
                    current_bias.accel.x as f32,
                    current_bias.accel.y as f32,
                    current_bias.accel.z as f32,
                ]
            } else {
                vec![0.0; 6]
            };
            if !is_fixed {
                problem
                    .add_variable(Variable::euclidean(bias_name, 6), bias_init)
                    .ok()?;
            }
        }

        // Fixed variables: add as separate variables with fixed values
        // For fixed keyframes, we embed their values directly in the factors
        // by using the same parameter values. However, the Problem API needs
        // all referenced variables to exist, so add fixed ones too.
        for kf_idx in 0..active_start {
            let kf = &self.keyframes[kf_idx];
            let pose_name = format!("pose_{kf_idx}");
            let vel_name = format!("vel_{kf_idx}");
            let bias_name = format!("bias_{kf_idx}");

            let se3_params = pose_to_se3_params(&kf.frame.pose_world_to_cam);
            problem
                .add_variable(
                    Variable::new(pose_name, VariableType::SE3, vec![0.0; 7]),
                    se3_params,
                )
                .ok()?;

            problem
                .add_variable(Variable::euclidean(vel_name, 3), vec![0.0; 3])
                .ok()?;

            problem
                .add_variable(Variable::euclidean(bias_name, 6), vec![0.0; 6])
                .ok()?;
        }

        // Point variables
        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            let pt = &self.map_points[global_idx].position;
            let var = Variable::euclidean(format!("pt_{local_idx}"), 3);
            let init = vec![pt.x as f32, pt.y as f32, pt.z as f32];
            problem.add_variable(var, init).ok()?;
        }

        // ── Add factors ─────────────────────────────────────────────────

        let intrinsics = [
            camera.fx as f32,
            camera.fy as f32,
            camera.cx as f32,
            camera.cy as f32,
        ];

        // Reprojection factors
        let mut n_obs = 0usize;
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
            let pose_name = format!("pose_{kf_idx}");
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                if let Some(mp_idx) = mp_opt {
                    let Some(&local_pt_idx) = mp_global_to_local.get(mp_idx) else {
                        continue;
                    };
                    if let Some(kp) = kf.frame.features.keypoints_xy.get(desc_idx) {
                        let scale = kf
                            .frame
                            .features
                            .scales
                            .get(desc_idx)
                            .copied()
                            .unwrap_or(1.0);
                        let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                        let factor = Box::new(ReprojectionFactor::new(
                            [p.x as f32, p.y as f32],
                            intrinsics,
                            scale,
                        ));
                        let pt_name = format!("pt_{local_pt_idx}");
                        problem
                            .add_factor(factor, vec![pt_name, pose_name.clone()])
                            .ok()?;
                        n_obs += 1;
                    }
                }
            }
        }

        if n_obs < MIN_OBSERVATIONS {
            return None;
        }

        // Inertial + bias random walk factors between consecutive keyframes
        for kf_idx in 1..n_kfs {
            let Some(ref pre) = self.keyframes[kf_idx].preintegrated_imu else {
                continue;
            };

            let prev_idx = kf_idx - 1;
            let pose_i = format!("pose_{prev_idx}");
            let vel_i = format!("vel_{prev_idx}");
            let pose_j = format!("pose_{kf_idx}");
            let vel_j = format!("vel_{kf_idx}");
            let bias_i = format!("bias_{prev_idx}");
            let bias_j = format!("bias_{kf_idx}");

            // Inertial factor: connects pose_i, vel_i, pose_j, vel_j, bias_i
            let inertial = Box::new(InertialFactor::new(pre, *gravity));
            problem
                .add_factor(
                    inertial,
                    vec![
                        pose_i,
                        vel_i,
                        pose_j,
                        vel_j,
                        bias_i.clone(),
                    ],
                )
                .ok()?;

            // Bias random walk: connects bias_i, bias_j
            let bias_rw = Box::new(BiasRandomWalkFactor::new(pre));
            problem
                .add_factor(bias_rw, vec![bias_i, bias_j])
                .ok()?;
        }

        // ── Optimize ────────────────────────────────────────────────────

        let optimizer = LevenbergMarquardt {
            lambda_init: 1.0,
            lambda_max: 1e10,
            lambda_factor: 10.0,
            max_iterations: 10,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-8,
        };

        if optimizer.optimize(&mut problem).is_err() {
            return None;
        }

        // ── Read back results ───────────────────────────────────────────

        let vars = problem.get_variables();

        // Update active keyframe poses
        for kf_idx in active_start..n_kfs {
            let pose_name = format!("pose_{kf_idx}");
            if let Some(var) = vars.get(&pose_name) {
                self.keyframes[kf_idx].frame.pose_world_to_cam =
                    se3_params_to_pose(&var.values);
            }
        }

        // Update map points
        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            let pt_name = format!("pt_{local_idx}");
            if let Some(var) = vars.get(&pt_name) {
                if let Some(mp) = self.map_points.get_mut(global_idx) {
                    mp.position = Vec3F64::new(
                        var.values[0] as f64,
                        var.values[1] as f64,
                        var.values[2] as f64,
                    );
                }
            }
        }

        // Read latest velocity and bias
        let last_kf_idx = n_kfs - 1;
        let vel_name = format!("vel_{last_kf_idx}");
        let bias_name = format!("bias_{last_kf_idx}");

        let latest_velocity = vars.get(&vel_name).map(|v| {
            Vec3F64::new(v.values[0] as f64, v.values[1] as f64, v.values[2] as f64)
        }).unwrap_or(*current_velocity);

        let latest_bias = vars.get(&bias_name).map(|v| {
            ImuBias {
                gyro: Vec3F64::new(v.values[0] as f64, v.values[1] as f64, v.values[2] as f64),
                accel: Vec3F64::new(v.values[3] as f64, v.values[4] as f64, v.values[5] as f64),
            }
        }).unwrap_or(*current_bias);

        Some(InertialBaResult {
            latest_velocity,
            latest_bias,
        })
    }
}

/// Convert a `Pose3d` (rotation matrix + translation) to SE3 parameters [qw, qx, qy, qz, tx, ty, tz].
fn pose_to_se3_params(pose: &Pose3d) -> Vec<f32> {
    let r = &pose.rotation;
    // Rotation matrix to quaternion (Shepperd's method)
    let trace = r.col(0).x + r.col(1).y + r.col(2).z;
    let (qw, qx, qy, qz) = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (r.col(1).z - r.col(2).y) * s;
        let y = (r.col(2).x - r.col(0).z) * s;
        let z = (r.col(0).y - r.col(1).x) * s;
        (w, x, y, z)
    } else if r.col(0).x > r.col(1).y && r.col(0).x > r.col(2).z {
        let s = 2.0 * (1.0 + r.col(0).x - r.col(1).y - r.col(2).z).sqrt();
        let w = (r.col(1).z - r.col(2).y) / s;
        let x = 0.25 * s;
        let y = (r.col(1).x + r.col(0).y) / s;
        let z = (r.col(2).x + r.col(0).z) / s;
        (w, x, y, z)
    } else if r.col(1).y > r.col(2).z {
        let s = 2.0 * (1.0 + r.col(1).y - r.col(0).x - r.col(2).z).sqrt();
        let w = (r.col(2).x - r.col(0).z) / s;
        let x = (r.col(1).x + r.col(0).y) / s;
        let y = 0.25 * s;
        let z = (r.col(2).y + r.col(1).z) / s;
        (w, x, y, z)
    } else {
        let s = 2.0 * (1.0 + r.col(2).z - r.col(0).x - r.col(1).y).sqrt();
        let w = (r.col(0).y - r.col(1).x) / s;
        let x = (r.col(2).x + r.col(0).z) / s;
        let y = (r.col(2).y + r.col(1).z) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    };
    vec![
        qw as f32, qx as f32, qy as f32, qz as f32,
        pose.translation.x as f32,
        pose.translation.y as f32,
        pose.translation.z as f32,
    ]
}

/// Convert SE3 parameters [qw, qx, qy, qz, tx, ty, tz] back to a `Pose3d`.
fn se3_params_to_pose(params: &[f32]) -> Pose3d {
    let (qw, qx, qy, qz) = (params[0] as f64, params[1] as f64, params[2] as f64, params[3] as f64);
    // Normalize quaternion
    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    let (qw, qx, qy, qz) = (qw / norm, qx / norm, qy / norm, qz / norm);
    // Quaternion to rotation matrix
    let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let r01 = 2.0 * (qx * qy - qw * qz);
    let r02 = 2.0 * (qx * qz + qw * qy);
    let r10 = 2.0 * (qx * qy + qw * qz);
    let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let r12 = 2.0 * (qy * qz - qw * qx);
    let r20 = 2.0 * (qx * qz - qw * qy);
    let r21 = 2.0 * (qy * qz + qw * qx);
    let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
    Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(r00, r10, r20),
            Vec3F64::new(r01, r11, r21),
            Vec3F64::new(r02, r12, r22),
        ),
        Vec3F64::new(params[4] as f64, params[5] as f64, params[6] as f64),
    )
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
            timestamp: 0.0,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                scales: vec![1.0; n],
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
