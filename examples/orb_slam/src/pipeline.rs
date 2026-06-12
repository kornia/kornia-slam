//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use std::collections::HashSet;

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{
    TriangulationConfig, TwoViewEstimator, TwoViewModel, triangulate_matched_points,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbMatchConfig, hamming_distance, match_orb_descriptors};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};
use kornia_slam::Frame;
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::two_view::{TwoViewInitConfig, try_initialize_two_view};
use kornia_slam::map::{Keyframe, Map, MapPoint};
use kornia_slam::stereo::unproject_stereo;
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus,
};
use kornia_algebra::{Mat4F64, Mat3F64, SO3F64};

struct InertialInitConfig {
    min_keyframes: usize,
    min_time_sec: f64,
    min_motion: f64,
}

struct ImuInitResult {
    scale: f64,
    gravity_world: Vec3F64,
    velocities_world: Vec<Vec3F64>,
    bias: ImuBias,
}

#[inline]
fn vec3_cross(
    a: &Vec3F64,
    b: &Vec3F64,
) -> Vec3F64 {
    Vec3F64::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

#[inline]
fn vec3_dot(
    a: &Vec3F64,
    b: &Vec3F64,
) -> f64 {
    a.x * b.x
    + a.y * b.y
    + a.z * b.z
}

#[inline]
fn vec3_normalize(
    v: &Vec3F64,
) -> Vec3F64 {
    let n = v.length();

    if n < 1e-12 {
        return *v;
    }

    *v / n
}

/// Top-level ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
pub struct Pipeline {
    // Camera model
    camera: PinholeCamera,
    // Primary pose estimator
    estimator: MapProjectionEstimator,
    // Boostrap pose estimator
    two_view_init_config: TwoViewInitConfig,
    // Keyframe insertion policy
    keyframe_policy: KeyframePolicy,
    // Enable local bundle adjustment after keyframe insertion
    enable_local_ba: bool,
    // mThDepth (metres): back-project close stereo points at each keyframe when set
    stereo_close_depth: Option<f64>,
    // Emit per-frame diagnostic logs (skip/reject reasons, growth counters)
    debug: bool,
    // Buffered debug messages produced during the most recent process_frame call;
    // drained by the caller (TUI panel or stderr).
    debug_messages: Vec<String>,
    // Map object
    map: Map,
    // IMU states
    imu_calib: ImuCalib,
    imu_bias: ImuBias,
    pending_imu: Vec<ImuMeasurement>,
    gravity_world: Vec3F64,
    bootstrap_timestamp_sec: Option<f64>,
    last_keyframe_timestamp_sec: Option<f64>,
    inertial_init_start_kf_idx: Option<usize>,
    inertial_init_config: InertialInitConfig,
    t_bc: Mat4F64,   // body <- left camera
    t_cb: Mat4F64,   // left camera <- body
    // System state
    state: SystemState,
}

impl Pipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(camera: PinholeCamera, config: PipelineConfig, T_bc: Mat4F64) -> Self {
        Self {
            camera,
            estimator: MapProjectionEstimator::new(config.map_projection),
            two_view_init_config: config.two_view_init,
            keyframe_policy: config.keyframe_policy,
            enable_local_ba: config.enable_local_ba,
            stereo_close_depth: config.stereo_close_depth_m,
            debug: config.debug,
            debug_messages: Vec::new(),
            map: Map::new(),
            state: SystemState::new(),
            imu_calib: ImuCalib {
                gyro_noise: 1.6968e-4,
                accel_noise: 2.0e-3,
                gyro_bias_noise: 1.9393e-5,
                accel_bias_noise: 3.0e-3,
            },
            imu_bias: ImuBias::default(),
            pending_imu: Vec::new(),
            gravity_world: Vec3F64::new(0.0, 0.0, -9.81),
            bootstrap_timestamp_sec: None,
            last_keyframe_timestamp_sec: None,
            inertial_init_start_kf_idx: None,
            inertial_init_config: InertialInitConfig {
                min_keyframes: 30,
                min_time_sec: 1.0,
                min_motion: 0.05,
            },
            t_bc: T_bc,
            t_cb: T_bc.inverse(),
        }
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    pub fn process_frame(
        &mut self,
        frame: Frame,
        timestamp_sec: f64,
        imu_samples: Vec<ImuMeasurement>,
    ) -> TrackingResult {
        self.pending_imu.extend(imu_samples);

        match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_step(frame, timestamp_sec),
            SystemMode::InertialInit => self.inertial_init_step(frame, timestamp_sec),
            SystemMode::Tracking => self.tracking_step(frame, timestamp_sec),
        }
    }

    /// Returns all persistent map points.
    pub fn map_points(&self) -> &[MapPoint] {
        self.map.map_points()
    }

    /// Returns the index of the current reference keyframe, if tracking has one.
    pub fn current_keyframe_idx(&self) -> Option<usize> {
        self.state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki).map(|kf| kf.frame.idx))
    }

    /// Returns the number of active (non-culled) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map.num_active_map_points()
    }

    /// Drain any debug messages accumulated since the last call.
    pub fn drain_debug_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_messages)
    }

    /// Toggle whether the pipeline buffers per-frame debug messages.
    pub fn set_debug(&mut self, on: bool) {
        self.debug = on;
        if !on {
            self.debug_messages.clear();
        }
    }

    fn dbg(&mut self, msg: String) {
        if self.debug {
            self.debug_messages.push(msg);
        }
    }

    fn bootstrap_step(&mut self, curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stereo frames carry metric per-keypoint depth, so we can build a
        // metric map from a single keyframe (ORB-SLAM3's StereoInitialization)
        // instead of waiting for two-view parallax.
        if curr_frame.is_stereo() {
            return self.bootstrap_stereo(curr_frame);
        }
        self.bootstrap_mono(curr_frame, timestamp_sec)
    }

    /// Single-frame metric initialization from stereo depth.
    fn bootstrap_stereo(&mut self, mut curr_frame: Frame) -> TrackingResult {
        // Build the new map in the current odometry frame (identity at start,
        // or the recovery pose after a tracking loss).
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        const MIN_STEREO_POINTS: usize = 50;
        let cam_points = unproject_stereo(&curr_frame, &self.camera);
        if cam_points.len() < MIN_STEREO_POINTS {
            self.dbg(format!(
                "[bootstrap_stereo] frame={} skip: only {} stereo points (need >= {})",
                curr_frame.idx,
                cam_points.len(),
                MIN_STEREO_POINTS,
            ));
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let pose_inv = curr_frame.pose_world_to_cam.inverse();
        let mut keyframe = Keyframe::from_frame(curr_frame);
        let curr_idx = keyframe.frame.idx;

        let mut points = Vec::with_capacity(cam_points.len());
        for (desc_idx, p_cam) in &cam_points {
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = keyframe.frame.features.descriptors[*desc_idx];
            let color = keyframe
                .frame
                .keypoint_colors
                .get(*desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        let added = self
            .map
            .add_triangulated_points(None, &mut keyframe, &points);
        self.map.upsert_keyframe(keyframe);

        self.dbg(format!(
            "[bootstrap_stereo] frame={curr_idx} metric map created with {added} points",
        ));

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.velocity = None;
        self.state.mode = SystemMode::Tracking;

        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    /// Back-projects `curr_kf`'s unassociated close stereo keypoints
    /// (`z < mthdepth`) into new metric map points, associating them to the
    /// keyframe. Returns the number of points created.
    fn add_close_stereo_points(&mut self, curr_kf: &mut Keyframe, mthdepth: f64) -> usize {
        let cam_points = unproject_stereo(&curr_kf.frame, &self.camera);
        if cam_points.is_empty() {
            return 0;
        }
        let pose_inv = curr_kf.frame.pose_world_to_cam.inverse();

        let mut points = Vec::new();
        for (desc_idx, p_cam) in &cam_points {
            // Far points: leave to multi-view triangulation.
            if p_cam.z > mthdepth {
                continue;
            }
            // Skip keypoints already tied to a map point (tracked this frame).
            if curr_kf.map_point(*desc_idx).is_some() {
                continue;
            }
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = curr_kf.frame.features.descriptors[*desc_idx];
            let color = curr_kf
                .frame
                .keypoint_colors
                .get(*desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        self.map.add_triangulated_points(None, curr_kf, &points)
    }

    fn bootstrap_mono(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        // Staleness guard (mirrors ORB-SLAM3's MonocularInitialization):
        // a frame with too few keypoints is neither a viable reference nor
        // a viable current frame. If we already had a reference, drop it
        // and wait for a feature-rich frame to start over.
        const MIN_KEYPOINTS_FOR_BOOTSTRAP: usize = 100;
        if curr_frame.features.keypoints_xy.len() <= MIN_KEYPOINTS_FOR_BOOTSTRAP {
            self.dbg(format!(
                "[bootstrap] frame={} skip: too few keypoints ({}, need > {})",
                curr_frame.idx,
                curr_frame.features.keypoints_xy.len(),
                MIN_KEYPOINTS_FOR_BOOTSTRAP,
            ));
            self.state.bootstrap_frame = None;
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        let Some(prev_bootstrap_frame) = self.state.bootstrap_frame.take() else {
            self.dbg(format!(
                "[bootstrap] frame={} stored as reference (awaiting second frame)",
                curr_frame.idx,
            ));
            self.state.bootstrap_frame = Some(curr_frame);
            self.bootstrap_timestamp_sec = Some(timestamp_sec);
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        };

        let result = try_initialize_two_view(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &self.camera,
            &self.two_view_init_config,
        );

        let two_view_estimate = match result {
            Err(reason) => {
                self.dbg(format!(
                    "[bootstrap] frame={} (ref={}) reject: {:?}",
                    curr_frame.idx, prev_bootstrap_frame.idx, reason,
                ));
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            Ok(tv) => tv,
        };

        self.dbg(format!(
            "[bootstrap] frame={} accept: model={} triangulated={} inliers={}",
            curr_frame.idx,
            two_view_estimate.model_kind,
            two_view_estimate.points3d.len(),
            two_view_estimate.estimate.inliers,
        ));

        let estimated_pose = two_view_estimate.estimate.pose;
        let prev_pose_world_to_cam = curr_frame.pose_world_to_cam;
        self.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
        let prev_idx = prev_bootstrap_frame.idx;
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        self.build_initial_map(
            reference_kf,
            current_kf,
            &two_view_estimate.estimate.matches,
            &two_view_estimate.points3d,
            &two_view_estimate.inlier_indices,
            two_view_estimate.median_depth,
        );

        // Post-BA sanity gate (mirrors ORB-SLAM3's reset criteria in
        // CreateInitialMapMonocular). Discard the bootstrap if the resulting
        // map has too few valid points or a degenerate scale.
        const MIN_VALID_POINTS: usize = 50;
        let health = self.map.initial_map_health();
        if health.valid_in_both < MIN_VALID_POINTS || health.median_depth_older_kf <= 0.0 {
            self.dbg(format!(
                "[init_gate] reject: valid_in_both={} median_depth={:.3} (need >= {} and > 0)",
                health.valid_in_both, health.median_depth_older_kf, MIN_VALID_POINTS,
            ));
            self.map.clear_active();
            self.state.reset();
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

        // BA inside build_initial_map may have refined KF1's pose; sync state
        // and recompute velocity from the post-BA pose.
        if let Some(kf) = self.map.get_keyframe(curr_idx) {
            self.state.pose_world_to_cam = kf.frame.pose_world_to_cam;
        }

        if let Some(prev_ts) = self.bootstrap_timestamp_sec {
            let (preint,imu_samples) = self.preintegrate_pending_imu(prev_ts, timestamp_sec);
            if preint.dt > 0.0 {
                self.map.add_imu_edge(prev_idx, curr_idx, preint, imu_samples);
            }
        }

        self.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.state.pose_world_to_cam,
        ));

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.mode = SystemMode::InertialInit;
        self.inertial_init_start_kf_idx = Some(curr_idx);
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);

        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status: TrackingStatus::KeyframeAccepted,
        }
    }

    fn build_initial_map(
        &mut self,
        mut reference_kf: Keyframe,
        mut current_kf: Keyframe,
        matches: &[(usize, usize)],
        points3d: &[Vec3F64],
        inlier_indices: &[usize],
        median_depth: Option<f64>,
    ) -> usize {
        let depth_scale = median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
        let reference_pose_inv = reference_kf.frame.pose_world_to_cam.inverse();

        let mut triangulated = Vec::new();
        for (p_cam, &match_idx) in points3d.iter().zip(inlier_indices.iter()) {
            let Some(&(ref_desc_idx, curr_desc_idx)) = matches.get(match_idx) else {
                continue;
            };
            if ref_desc_idx >= reference_kf.map_point_by_desc_idx.len()
                || curr_desc_idx >= current_kf.map_point_by_desc_idx.len()
            {
                continue;
            }
            let descriptor = current_kf
                .frame
                .features
                .descriptors
                .get(curr_desc_idx)
                .copied()
                .or_else(|| {
                    reference_kf
                        .frame
                        .features
                        .descriptors
                        .get(ref_desc_idx)
                        .copied()
                });
            let Some(descriptor) = descriptor else {
                continue;
            };
            let color = current_kf
                .frame
                .keypoint_colors
                .get(curr_desc_idx)
                .copied()
                .unwrap_or([128; 3]);
            let p_world = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
            triangulated.push((p_world, descriptor, color, ref_desc_idx, curr_desc_idx));
        }

        let added = self.map.add_triangulated_points(
            Some(&mut reference_kf),
            &mut current_kf,
            &triangulated,
        );

        self.map.upsert_keyframe(reference_kf);
        self.map.upsert_keyframe(current_kf);

        self.map.run_initial_ba(&self.camera);

        added
    }

    fn preintegrate_pending_imu(&mut self, t0: f64, t1: f64) -> (PreintegratedImu, Vec<ImuMeasurement>) {
        let mut pre = PreintegratedImu::new(self.imu_bias, self.imu_calib);

        self.pending_imu
            .sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));

        let samples: Vec<ImuMeasurement> = self
            .pending_imu
            .iter()
            .copied()
            .filter(|m| m.timestamp >= t0 && m.timestamp <= t1)
            .collect();

        if samples.is_empty() {
            return (pre, samples);
        }

        let mut last_t = t0;

        for sample in &samples {
            let dt = sample.timestamp - last_t;
            if dt > 0.0 {
                pre.integrate(sample, dt);
                last_t = sample.timestamp;
            }
        }

        if last_t < t1 {
            if let Some(last_sample) = samples.last() {
                pre.integrate(last_sample, t1 - last_t);
            }
        }

        self.pending_imu.retain(|m| m.timestamp > t1);

        (pre, samples)
    }

    fn inertial_init_ready(&self) -> bool {
        let Some(start_idx) = self.inertial_init_start_kf_idx else {
            return false;
        };

        let init_kfs: Vec<&Keyframe> = self
            .map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        if init_kfs.len() < self.inertial_init_config.min_keyframes {
            return false;
        }

        let imu_time: f64 = self
            .map
            .imu_edges()
            .iter()
            .filter(|edge| edge.curr_kf_idx >= start_idx)
            .map(|edge| edge.preintegrated.dt)
            .sum();
        if imu_time < self.inertial_init_config.min_time_sec {
            return false;
        }

        let Some(first) = init_kfs.first() else {
            return false;
        };
        let Some(last) = init_kfs.last() else {
            return false;
        };
        let first_center = first.frame.pose_world_to_cam.inverse().translation;
        let last_center = last.frame.pose_world_to_cam.inverse().translation;
        (last_center - first_center).length() >= self.inertial_init_config.min_motion
    }

    fn estimate_gyro_bias(&mut self) -> Vec3F64 {

        let mut sum = Vec3F64::ZERO;
        let mut total_dt = 0.0;

        for edge in self.map.imu_edges() {

            let Some(kf_i) =
                self.map.get_keyframe(edge.prev_kf_idx)
            else {
                continue;
            };

            let Some(kf_j) =
                self.map.get_keyframe(edge.curr_kf_idx)
            else {
                continue;
            };

            let cam_i_world =
                kf_i.frame.pose_world_to_cam.inverse();

            let cam_j_world =
                kf_j.frame.pose_world_to_cam.inverse();

            let t_cb_cols = self.t_cb.to_cols_array();
            let r_cb = Mat3F64::from_cols(
                Vec3F64::new(
                    t_cb_cols[0],
                    t_cb_cols[1],
                    t_cb_cols[2],
                ),
                Vec3F64::new(
                    t_cb_cols[4],
                    t_cb_cols[5],
                    t_cb_cols[6],
                ),
                Vec3F64::new(
                    t_cb_cols[8],
                    t_cb_cols[9],
                    t_cb_cols[10],
                ),
            );

            let r_i_wb =
                cam_i_world.rotation * r_cb;

            let r_j_wb =
                cam_j_world.rotation * r_cb;

            let r_vis =
                SO3F64::from_matrix(
                    &(r_i_wb.inverse() * r_j_wb)
                );

            let r_imu =
                SO3F64::from_matrix(
                    &edge.preintegrated.delta_rotation
                );

            let e = r_imu.rminus(&r_vis);

            sum += e;

            total_dt += edge.preintegrated.dt;
        }

        if total_dt < 1e-6 {
            return Vec3F64::ZERO;
        }
        // self.imu_bias.gyro = sum / total_dt;
        sum / total_dt
    }

    fn reintegrate_all_edges(
        &mut self,
        gyro_bias: Vec3F64,
        accel_bias: Vec3F64,
    )
    {
        for edge in self.map.imu_edges_mut() {

            let bias =
                ImuBias {
                    gyro: gyro_bias,
                    accel: accel_bias,
                };

            let mut pre =
                PreintegratedImu::new(
                    bias,
                    self.imu_calib,
                );

            if edge.imu_measurements.is_empty() {
                continue;
            }

            let mut last_t =
                edge.imu_measurements[0].timestamp;

            for m in &edge.imu_measurements {

                let dt =
                    m.timestamp - last_t;

                if dt > 0.0 {
                    pre.integrate(m, dt);
                }

                last_t = m.timestamp;
            }

            edge.preintegrated = pre;
        }
    }

    fn solve_scale_gravity(
        &self,
        keyframes: &[&Keyframe],
    ) -> Option<(f64, Vec3F64)>
    {
        if keyframes.len() < 3 {
            return None;
        }

        let t_cb_cols = self.t_cb.to_cols_array();

        let p_cb = Vec3F64::new(
            t_cb_cols[12],
            t_cb_cols[13],
            t_cb_cols[14],
        );

        let r_cb = Mat3F64::from_cols(
            Vec3F64::new(
                t_cb_cols[0],
                t_cb_cols[1],
                t_cb_cols[2],
            ),
            Vec3F64::new(
                t_cb_cols[4],
                t_cb_cols[5],
                t_cb_cols[6],
            ),
            Vec3F64::new(
                t_cb_cols[8],
                t_cb_cols[9],
                t_cb_cols[10],
            ),
        );

        let mut edge_map = std::collections::HashMap::new();
        for edge in self.map.imu_edges() {
            edge_map.insert(
                (edge.prev_kf_idx, edge.curr_kf_idx),
                edge,
            );
        }

        // Build matrices A and B for A * [s; gx; gy; gz] = B
        // Following equation (12) and (13) from the paper
        let mut a_rows: Vec<Vec<f64>> = Vec::new();
        let mut b_vals: Vec<f64> = Vec::new();

        for i in 0..(keyframes.len() - 2) {
            let kf1 = keyframes[i];
            let kf2 = keyframes[i + 1];
            let kf3 = keyframes[i + 2];

            let edge12 = match edge_map.get(&(kf1.frame.idx, kf2.frame.idx)) {
                Some(e) => *e,
                None => continue,
            };

            let edge23 = match edge_map.get(&(kf2.frame.idx, kf3.frame.idx)) {
                Some(e) => *e,
                None => continue,
            };

            let dt12 = edge12.preintegrated.dt;
            let dt23 = edge23.preintegrated.dt;

            if dt12 <= 1e-6 || dt23 <= 1e-6 {
                continue;
            }

            let t1 = kf1.frame.pose_world_to_cam.inverse();
            let t2 = kf2.frame.pose_world_to_cam.inverse();
            let t3 = kf3.frame.pose_world_to_cam.inverse();

            let p1 = t1.translation;  // p_C^1 in world frame (scaled)
            let p2 = t2.translation;  // p_C^2 in world frame (scaled)
            let p3 = t3.translation;  // p_C^3 in world frame (scaled)

            let r1_wc = t1.rotation;
            let r2_wc = t2.rotation;
            let r3_wc = t3.rotation;

            let r1_wb = r1_wc * r_cb;
            let r2_wb = r2_wc * r_cb;
            // let r3_wb = r3_wc * r_cb;

            let dp12 = edge12.preintegrated.delta_position;
            let dv12 = edge12.preintegrated.delta_velocity;
            let dp23 = edge23.preintegrated.delta_position;

            // Compute λ(i) from equation (13)
            let lambda = (p2 - p1) * dt23 - (p3 - p2) * dt12;

            // Compute β(i) from equation (13)
            let beta = 0.5 * (dt12 * dt12 * dt23 + dt23 * dt23 * dt12);

            // Compute γ(i) from equation (13)
            let gamma = (r2_wc * p_cb - r1_wc * p_cb) * dt23
                - (r3_wc * p_cb - r2_wc * p_cb) * dt12
                + (r2_wb * dp23) * dt12
                + (r1_wb * dv12) * dt12 * dt23
                - (r1_wb * dp12) * dt23;

            // Each triple gives 3 equations (x, y, z)
            // Equation: s * λ + β * g = γ
            // Rearranged: λ * s + β * gx = γ.x (for x component)
            // etc.
            
            // x equation: λ.x * s + β * gx = γ.x
            a_rows.push(vec![lambda.x, beta, 0.0, 0.0]);
            b_vals.push(gamma.x);

            // y equation: λ.y * s + β * gy = γ.y
            a_rows.push(vec![lambda.y, 0.0, beta, 0.0]);
            b_vals.push(gamma.y);

            // z equation: λ.z * s + β * gz = γ.z
            a_rows.push(vec![lambda.z, 0.0, 0.0, beta]);
            b_vals.push(gamma.z);
        }

        if a_rows.len() < 4 {
            println!("[scale_gravity] insufficient equations: {}", a_rows.len());
            return None;
        }

        // Solve least squares: A * x = B
        let solution = solve_least_squares(&a_rows, &b_vals)?;
        
        let scale = solution[0];
        let gravity = Vec3F64::new(solution[1], solution[2], solution[3]);
        let gravity_norm = gravity.length();

        println!("[scale_gravity] scale={:.6}, gravity=({:.6}, {:.6}, {:.6}), |g|={:.6}", 
                scale, gravity.x, gravity.y, gravity.z, gravity_norm);

        // The issue: your gravity vector should have magnitude ~9.81
        // If it doesn't, you need to constrain the solution
        if gravity_norm < 1e-6 || !gravity_norm.is_finite() {
            println!("[scale_gravity] invalid gravity magnitude");
            return None;
        }

        // Normalize gravity to standard magnitude (optional, but helps)
        let standard_g = 9.81;
        let gravity_corrected = gravity * (standard_g / gravity_norm);
        
        println!("[scale_gravity] corrected gravity magnitude: {}", gravity_corrected.length());

        Some((scale, gravity_corrected))
    }

    fn solve_velocities(
        &self,
        scale: f64,
        gravity: &Vec3F64,
        keyframes: &[&Keyframe],
    ) -> Option<Vec<Vec3F64>>
    {
        let n = keyframes.len();

        if n < 2 {
            return None;
        }

        let mut velocities =
            vec![Vec3F64::ZERO; n];

        let mut edge_map =
            std::collections::HashMap::new();

        for edge in self.map.imu_edges() {
            edge_map.insert(
                (edge.prev_kf_idx, edge.curr_kf_idx),
                edge,
            );
        }

        let t_cb_cols =
            self.t_cb.to_cols_array();

        let p_cb =
            Vec3F64::new(
                t_cb_cols[12],
                t_cb_cols[13],
                t_cb_cols[14],
            );

        let r_cb =
            Mat3F64::from_cols(
                Vec3F64::new(
                    t_cb_cols[0],
                    t_cb_cols[1],
                    t_cb_cols[2],
                ),
                Vec3F64::new(
                    t_cb_cols[4],
                    t_cb_cols[5],
                    t_cb_cols[6],
                ),
                Vec3F64::new(
                    t_cb_cols[8],
                    t_cb_cols[9],
                    t_cb_cols[10],
                ),
            );

        for i in 0..(n - 1) {

            let kf_i = keyframes[i];
            let kf_j = keyframes[i + 1];

            let edge =
                edge_map.get(
                    &(kf_i.frame.idx,
                    kf_j.frame.idx)
                )?;

            let dt =
                edge.preintegrated.dt;

            if dt <= 1e-6 {
                continue;
            }

            let T_i =
                kf_i.frame.pose_world_to_cam.inverse();

            let T_j =
                kf_j.frame.pose_world_to_cam.inverse();

            let p_i =
                T_i.translation;

            let p_j =
                T_j.translation;

            let r_wc_i =
                T_i.rotation;

            let r_wc_j =
                T_j.rotation;

            let r_wb_i =
                r_wc_i * r_cb;

            velocities[i] =
                (
                    scale * (p_j - p_i)

                    +

                    (r_wc_j * p_cb
                    -
                    r_wc_i * p_cb)

                    -

                    *gravity
                    * (0.5 * dt * dt)

                    -

                    r_wb_i
                    * edge.preintegrated.delta_position

                ) / dt;
        }

        //
        // last velocity
        //
        {
            let i = n - 2;

            let edge =
                edge_map.get(
                    &(keyframes[i].frame.idx,
                    keyframes[i+1].frame.idx)
                )?;

            let T_i =
                keyframes[i]
                .frame
                .pose_world_to_cam
                .inverse();

            let r_wb_i =
                T_i.rotation * r_cb;

            velocities[n - 1] =
                velocities[n - 2]
                +
                *gravity
                * edge.preintegrated.dt
                +
                r_wb_i
                * edge.preintegrated.delta_velocity;
        }

        Some(velocities)
    }

    fn try_initialize_imu(
        &mut self,
    ) -> Option<ImuInitResult>
    {
        let start_idx =
            self.inertial_init_start_kf_idx?;

        //
        // Store ids instead of references.
        //
        let keyframe_ids: Vec<usize> =
            self.map
                .keyframes()
                .iter()
                .filter(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.idx)
                .collect();

        if keyframe_ids.len()
            < self.inertial_init_config.min_keyframes
        {
            return None;
        }

        //
        // 1. Gyro bias
        //
        let gyro_bias =
            self.estimate_gyro_bias();

        println!(
            "[imu-init] gyro bias {:?}",
            gyro_bias
        );

        //
        // 2. Reintegrate using gyro bias
        //
        // self.reintegrate_all_edges(
        //     Vec3F64::ZERO,
        //     Vec3F64::ZERO,
        // );

        //
        // Reacquire keyframe refs AFTER reintegration.
        //
        let keyframes: Vec<&Keyframe> =
            keyframe_ids
                .iter()
                .filter_map(|idx|
                    self.map.get_keyframe(*idx)
                )
                .collect();

        //
        // 3. Scale + gravity
        //
        println!(
            "[scale_gravity] keyframes={}",
            keyframes.len()
        );
        let (scale, gravity_world) =
            self.solve_scale_gravity(
                &keyframes,
            )?;

        println!(
            "[imu-init] scale={} gravity={:?} |g|={}",
            scale,
            gravity_world,
            gravity_world.length(),
        );

        //
        // 4. Accelerometer bias
        //
        // Placeholder until Jacobian-based solve exists.
        //
        let accel_bias =
            Vec3F64::ZERO;

        //
        // 5. Reintegrate again using both biases.
        //
        self.reintegrate_all_edges(
            gyro_bias,
            accel_bias,
        );

        //
        // Reacquire keyframe refs again because map
        // was mutably borrowed.
        //
        let keyframes: Vec<&Keyframe> =
            keyframe_ids
                .iter()
                .filter_map(|idx|
                    self.map.get_keyframe(*idx)
                )
                .collect();

        //
        // 6. Velocities
        //
        let velocities_world =
            self.solve_velocities(
                scale,
                &gravity_world,
                &keyframes,
            )?;
        
        println!(
            "gravity = {:?}, |g|={}",
            gravity_world,
            gravity_world.length()
        );

        Some(
            ImuInitResult {
                scale,
                gravity_world,
                velocities_world,

                bias: ImuBias {
                    gyro: gyro_bias,
                    accel: accel_bias,
                },
            }
        )
    }

    fn apply_imu_initialization(
        &mut self,
        init: ImuInitResult,
    ) 
    {
        let Some(start_idx) =
            self.inertial_init_start_kf_idx
        else {
            return;
        };

        //
        // 1. scale
        //
        self.map.scale_world(
            init.scale
        );

        //
        // 2. compute gravity alignment
        //

        let g_est =
    vec3_normalize(
        &init.gravity_world
    );

        let g_target =
            Vec3F64::new(
                0.0,
                0.0,
                -1.0,
            );

        let axis =
            vec3_cross(
                &g_est,
                &g_target,
            );

        let dot =
            vec3_dot(
                &g_est,
                &g_target,
            )
            .clamp(-1.0, 1.0);

        let angle =
            dot.acos();

        let axis_norm =
            axis.length();

        let r_align =
            if axis_norm > 1e-8 {

                let angle =
                    g_est
                        .dot(g_target)
                        .clamp(-1.0, 1.0)
                        .acos();

                SO3F64::exp(
                    axis.normalize()
                    * angle
                )
            }
            else {
                // exp(0) == identity
                SO3F64::exp(Vec3F64::ZERO)
            };

        //
        // 3. rotate keyframes
        //
        let mut velocity_iter =
            init.velocities_world.into_iter();

        for kf in self
            .map
            .keyframes_mut()
            .iter_mut()
            .filter(|kf|
                kf.frame.idx >= start_idx
            )
        {
            let mut T_wc =
                kf.frame
                    .pose_world_to_cam
                    .inverse();

            T_wc.translation =
                r_align.matrix()
                * T_wc.translation;

            T_wc.rotation =
                r_align.matrix()
                * T_wc.rotation;

            kf.frame.pose_world_to_cam =
                T_wc.inverse();

            if let Some(v) =
                velocity_iter.next()
            {
                kf.velocity_world =
                    r_align.matrix() * v;
            }

            kf.imu_bias =
                init.bias;
        }

        //
        // 4. rotate map points
        //
        for mp in self.map.map_points_mut()
        {
            mp.position =
                r_align.matrix()
                * mp.position;
        }

        //
        // 5. state velocity
        //
        if let Some(last_kf) =
            self.map
                .keyframes()
                .iter()
                .filter(|kf|
                    kf.frame.idx >= start_idx
                )
                .last()
        {
            self.state.velocity_world =
                last_kf.velocity_world;
        }

        //
        // 6. final state
        //
        self.state.imu_initialized =
            true;

        self.gravity_world =
            Vec3F64::new(
                0.0,
                0.0,
                -9.81,
            );

        self.imu_bias =
            init.bias;
    }

    fn inertial_init_step(&mut self, frame: Frame, timestamp_sec: f64) -> TrackingResult {
        let result = self.tracking_step(frame, timestamp_sec);

        if result.status == TrackingStatus::KeyframeAccepted && self.inertial_init_ready() {
            match self.try_initialize_imu() {
                Some(init) => {
                    let scale = init.scale;
                    let gravity = init.gravity_world;
                    self.apply_imu_initialization(init);
                    self.state.mode = SystemMode::Tracking;
                    self.dbg(format!(
                        "[imu_init] accepted: scale={scale:.4} gravity=({:.3},{:.3},{:.3})",
                        gravity.x, gravity.y, gravity.z
                    ));
                }
                None => {
                    self.dbg("[imu_init] rejected: solve failed or invalid scale/gravity".into());
                }
            }
        }

        result
    }

    fn predict_pose_imu(&mut self, pose_w2c: Pose3d, vel_world: Vec3F64, gravity_world: Vec3F64, preint: &PreintegratedImu,) -> (Pose3d, Vec3F64) {
        let dt = preint.dt;
        
        // Camera center and rotation in world frame
        let cam_to_world = pose_w2c.inverse();
        let p_i = cam_to_world.translation;     // camera center
        let r_i = cam_to_world.rotation;        // world←camera rotation

        // Predicted camera center via IMU kinematics:
        // p_j = p_i + v_i*dt + 0.5*g*dt² + R_i * Δp_imu
        let p_j = p_i 
            + vel_world * dt 
            + gravity_world * 0.5 * dt * dt
            + r_i * preint.delta_position;

        // Predicted rotation: R_j = R_i * ΔR_imu
        let r_j = r_i * preint.delta_rotation;

        // Predicted velocity: v_j = v_i + g*dt + R_i * Δv_imu
        let v_j = vel_world 
            + gravity_world * dt 
            + r_i * preint.delta_velocity;

        // Convert back to world-to-camera convention
        let pred_pose = Pose3d::from_rt(r_j, p_j).inverse();
        (pred_pose, v_j)
    }

    fn tracking_step(&mut self, frame: Frame, timestamp_sec: f64) -> TrackingResult {
        let image_size = frame.image_size;
        let pose_before = self.state.pose_world_to_cam;
        let prev_timestamp = self.state.last_frame_timestamp_sec;

        let candidate_pose = if self.state.imu_initialized 
            && prev_timestamp > 0.0 
        {
            let (preint, imu_measurements) = self.preintegrate_pending_imu(
                prev_timestamp, 
                timestamp_sec
            );
            if preint.dt > 0.0 {
                let (pred_pose, pred_vel) = self.predict_pose_imu(
                    pose_before,
                    self.state.velocity_world,
                    self.gravity_world,
                    &preint,
                );
                self.state.velocity_world = pred_vel; // propagate for next frame
                pred_pose
            } else {
                // IMU stalled, fall back to visual constant velocity
                self.state.velocity
                    .map(|v| v.compose(&pose_before))
                    .unwrap_or(pose_before)
            }
        } else {
            self.state.velocity
                .map(|v| v.compose(&pose_before))
                .unwrap_or(pose_before)
        };
        
        let result = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before,
            &self.map,
            &self.camera,
            self.state.current_keyframe_idx,
        );

        let (mut status, matches, tracked_inliers, reject_reason) = match result {
            Ok(estimate) => {
                self.state.pose_world_to_cam = estimate.pose;

                if self.state.imu_initialized {
                    let cam_before = pose_before.inverse().translation;
                    let cam_after  = estimate.pose.inverse().translation;

                    let dt =
                        timestamp_sec - prev_timestamp;

                    if dt > 1e-6 {
                        self.state.velocity_world =
                            (cam_after - cam_before) / dt;
                    }
                } else {
                    self.state.velocity =
                        Some(Pose3d::between(
                            &pose_before,
                            &estimate.pose,
                        ));
                }

                (
                    TrackingStatus::Tracked,
                    estimate.matches,
                    estimate.inliers,
                    None,
                )
            }
            Err(reason) => (TrackingStatus::Skipped, Vec::new(), 0, Some(reason)),
        };
        if self.debug {
            let msg = match reject_reason {
                Some(reason) => format!("[track] frame={} reject: {:?}", frame.idx, reason),
                None => format!(
                    "[track] frame={} ok: matches={} inliers={}",
                    frame.idx,
                    matches.len(),
                    tracked_inliers,
                ),
            };
            self.debug_messages.push(msg);
        }

        if status == TrackingStatus::Tracked {
            let visible = self
                .map
                .map_points_in_frustum(&self.camera, &candidate_pose, image_size);
            self.map.update_observation_counts(&visible, &matches);

            if self.try_insert_keyframe(&frame, timestamp_sec, tracked_inliers, &matches) {
                status = TrackingStatus::KeyframeAccepted;
            }
        }

        if status == TrackingStatus::Skipped {
            self.state.consecutive_failures += 1;
            if self.state.consecutive_failures >= self.state.max_consecutive_failures {
                self.state.reset();
                return self.bootstrap_step(frame, timestamp_sec);
            }
        } else {
            self.state.consecutive_failures = 0;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);

        if !self.keyframe_policy.should_insert(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        // Guard: reference KF must exist before we can triangulate.
        if self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .is_none()
        {
            return false;
        }

        let mut curr_kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
        });
        for &(mp_idx, curr_idx) in matches {
            curr_kf.associate_map_point(curr_idx, mp_idx);
            self.map.register_observation(mp_idx, &curr_kf, curr_idx);
        }

        // Stereo densification: back-project this keyframe's unassociated
        // "close" stereo keypoints directly into metric map points. Mirrors
        // ORB-SLAM3's CreateNewKeyFrame, which seeds close points from stereo
        // and leaves far points to multi-view triangulation (the grow pass).
        if let Some(mthdepth) = self.stereo_close_depth
            && curr_kf.frame.is_stereo()
        {
            let n_close = self.add_close_stereo_points(&mut curr_kf, mthdepth);
            self.dbg(format!(
                "[kf_stereo] frame={} close_points={}",
                frame.idx, n_close
            ));
        }

        // Triangulate new map points against the last MAX_COVIS_KFS keyframes,
        // not just the immediate predecessor. Mirrors ORB-SLAM3's
        // CreateNewMapPoints which uses the 30 best covisible KFs; we
        // approximate covisibility by recency until a covisibility graph is
        // available. Cloning upfront releases the shared borrow on self.map
        // so grow_map_points_from_keyframe_pair can take &mut self.
        const MAX_COVIS_KFS: usize = 10;
        let neighbor_kfs: Vec<Keyframe> = self
            .map
            .keyframes()
            .iter()
            .rev()
            .take(MAX_COVIS_KFS)
            .cloned()
            .collect();

        let enable_local_ba = self.enable_local_ba;
        let match_config = self.two_view_init_config.match_config;
        let triangulation_config = self.two_view_init_config.triangulation_config.clone();

        let mut total_grown = 0usize;
        for neighbor_kf in &neighbor_kfs {
            total_grown += self.grow_map_points_from_keyframe_pair(
                neighbor_kf,
                &mut curr_kf,
                match_config,
                &triangulation_config,
            );
        }
        self.dbg(format!(
            "[kf] frame={} grown={} from {} neighbor kfs",
            frame.idx,
            total_grown,
            neighbor_kfs.len()
        ));

        self.map.upsert_keyframe(curr_kf);
        if let Some(prev_kf_idx) = self.state.last_keyframe_idx {
            if let Some(prev_ts) = self.last_keyframe_timestamp_sec {
                let (preint, imu_measurements) = self.preintegrate_pending_imu(prev_ts, timestamp_sec);

                if preint.dt > 0.0 {
                    self.map.add_imu_edge(prev_kf_idx, frame.idx, preint, imu_measurements);
                }
            }
        }

        self.last_keyframe_timestamp_sec = Some(timestamp_sec);

        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Forward SearchInNeighbors / Fuse: extend each curr_kf-observed map
        // point's observation list to neighbor KFs that don't yet observe it.
        // Run before local BA so BA sees the extra reprojection constraints.
        let neighbor_kf_indices: Vec<usize> = neighbor_kfs.iter().map(|kf| kf.frame.idx).collect();
        let n_fused = self.fuse_into_neighbors(frame.idx, &neighbor_kf_indices);
        self.dbg(format!("[fuse] frame={} fused={}", frame.idx, n_fused));

        if enable_local_ba {
            self.map.run_local_ba(&self.camera);
            if let Some(newest_kf) = self.map.keyframes().last() {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
        }

        self.map.cull();
        true
    }

    fn grow_map_points_from_keyframe_pair(
        &mut self,
        prev_kf: &Keyframe,
        curr_kf: &mut Keyframe,
        match_config: OrbMatchConfig,
        triangulation_config: &TriangulationConfig,
    ) -> usize {
        const MIN_GROWTH_INLIERS: usize = 15;

        // Only consider features that don't already have a map point in either
        // KF. Matching the full descriptor arrays and then filtering discards
        // almost everything once the KFs are mature (the best matches always
        // land on the already-tracked features). Instead, build sub-arrays of
        // unassociated features from each side and match only those.
        let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
            .filter(|&i| prev_kf.map_point(i).is_none())
            .collect();
        let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
            .filter(|&i| curr_kf.map_point(i).is_none())
            .collect();

        // Need at least 8 unassociated features on each side for F-matrix RANSAC.
        if prev_unassoc.len() < 8 || curr_unassoc.len() < 8 {
            return 0;
        }

        let prev_orients: Vec<f32> = prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.orientations[i])
            .collect();
        let prev_descs: Vec<[u8; 32]> = prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.descriptors[i])
            .collect();
        let curr_orients: Vec<f32> = curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.orientations[i])
            .collect();
        let curr_descs: Vec<[u8; 32]> = curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.descriptors[i])
            .collect();

        let sub_matches = match_orb_descriptors(
            &prev_orients,
            &prev_descs,
            &curr_orients,
            &curr_descs,
            match_config,
        );

        // Map sub-array indices back to original feature indices.
        let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(sub_matches.len());
        for (prev_sub, curr_sub) in sub_matches {
            let Some(&prev_idx) = prev_unassoc.get(prev_sub) else {
                continue;
            };
            let Some(&curr_idx) = curr_unassoc.get(curr_sub) else {
                continue;
            };
            if prev_idx >= prev_kf.frame.features.keypoints_xy.len()
                || curr_idx >= curr_kf.frame.features.keypoints_xy.len()
            {
                continue;
            }
            pair_indices.push((prev_idx, curr_idx));
        }

        let camera = &self.camera;
        let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
            &prev_kf.frame.features.keypoints_xy,
            &curr_kf.frame.features.keypoints_xy,
            &pair_indices,
        );
        if pair_indices.len() < 8 {
            return 0;
        }

        let k = camera.intrinsic_matrix();
        let estimator = TwoViewEstimator::builder()
            .triangulation(triangulation_config.clone())
            .build();
        let two_view = match estimator.estimate(&prev_pts, &curr_pts, &k, &k) {
            Ok(tv) if matches!(tv.model, TwoViewModel::Fundamental(_)) => tv,
            _ => return 0,
        };
        if two_view.inlier_indices.len() < MIN_GROWTH_INLIERS {
            return 0;
        }

        // Collect inlier undistorted points for triangulation.
        let inlier_prev: Vec<_> = two_view
            .inlier_indices
            .iter()
            .map(|&i| prev_pts[i])
            .collect();
        let inlier_curr: Vec<_> = two_view
            .inlier_indices
            .iter()
            .map(|&i| curr_pts[i])
            .collect();

        let triangulated = match triangulate_matched_points(
            &inlier_prev,
            &inlier_curr,
            &prev_kf.frame.pose_world_to_cam,
            &curr_kf.frame.pose_world_to_cam,
            camera,
            triangulation_config,
        ) {
            Ok(pts) => pts,
            Err(_) => return 0,
        };

        let mut points = Vec::new();
        let mut used_curr = HashSet::new();
        for tp in &triangulated {
            let inlier_idx = two_view.inlier_indices[tp.pair_index];
            let Some(&(prev_idx, curr_idx)) = pair_indices.get(inlier_idx) else {
                continue;
            };
            if curr_kf.map_point(curr_idx).is_some() || !used_curr.insert(curr_idx) {
                continue;
            }
            let color = curr_kf
                .frame
                .keypoint_colors
                .get(curr_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((
                tp.position,
                curr_kf.frame.features.descriptors[curr_idx],
                color,
                prev_idx,
                curr_idx,
            ));
        }

        // Create the new map points; curr_kf is registered as the first
        // observer inside add_triangulated_points.
        let prev_kf_idx = prev_kf.frame.idx;
        let first_mp_idx = self.map.num_map_points();
        let added = self.map.add_triangulated_points(None, curr_kf, &points);

        // Register the neighbor (`prev_kf`) as a second observer on each new
        // map point. This is the SearchInNeighbors-equivalent piece for the
        // triangulating pair: without it the new point would have a single
        // observation, biasing scale/normal geometry and making the cull
        // overly aggressive. The neighbor KF in the map (not the clone) gets
        // its desc slot pointed at the new map point.
        for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().take(added).enumerate() {
            let mp_idx = first_mp_idx + i;
            self.map
                .register_observation(mp_idx, prev_kf, prev_desc_idx);
            if let Some(prev_live) = self.map.get_keyframe_mut(prev_kf_idx) {
                prev_live.associate_map_point(prev_desc_idx, mp_idx);
            }
        }

        added
    }

    /// Forward Fuse pass: project each map point observed by the current KF
    /// into every neighbor KF that doesn't already observe it. If the
    /// projection lands near an unassociated keypoint with a matching
    /// descriptor, register the observation. Mirrors a subset of ORB-SLAM3's
    /// `SearchInNeighbors` (forward direction only; we don't yet do duplicate
    /// merging or the second-hop covisible expansion).
    fn fuse_into_neighbors(&mut self, curr_kf_idx: usize, neighbor_kf_indices: &[usize]) -> usize {
        const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
        const FUSE_MAX_HAMMING: u32 = 50;

        // Collect map points observed by curr_kf. We snapshot the indices
        // here so we can hold no other borrow on self.map during the loop.
        let curr_mp_indices: Vec<usize> = match self.map.get_keyframe(curr_kf_idx) {
            Some(kf) => kf
                .map_point_by_desc_idx
                .iter()
                .filter_map(|&mp| mp)
                .collect(),
            None => return 0,
        };
        if curr_mp_indices.is_empty() {
            return 0;
        }

        let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
        let mut n_fused = 0usize;

        for &nb_kf_idx in neighbor_kf_indices {
            if nb_kf_idx == curr_kf_idx {
                continue;
            }
            // Clone the neighbor KF for the read-only inner loop; we need
            // to take `&mut self` later to register observations.
            let nb_kf = match self.map.get_keyframe(nb_kf_idx) {
                Some(kf) => kf.clone(),
                None => continue,
            };

            // Undistort neighbor keypoints once for projection comparison.
            let nb_kp_undist: Vec<[f32; 2]> = nb_kf
                .frame
                .features
                .keypoints_xy
                .iter()
                .map(|kp| {
                    let p = self.camera.undistort(kp[0] as f64, kp[1] as f64);
                    [p.x as f32, p.y as f32]
                })
                .collect();

            // Proposals: (kp_idx_in_nb_kf, mp_idx). Resolved at the end so
            // a single keypoint can't be claimed by two map points.
            let mut proposals: Vec<(usize, usize, u32)> = Vec::new();

            for &mp_idx in &curr_mp_indices {
                let mp = match self.map.map_points().get(mp_idx) {
                    Some(mp) if !mp.culled => mp,
                    _ => continue,
                };
                // Skip if neighbor already observes this map point.
                if mp.observation_kf_indices.contains(&nb_kf_idx) {
                    continue;
                }

                // Project into the neighbor's frame.
                let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
                if p_cam.z <= 0.0 {
                    continue;
                }
                let Ok(pixel) = self
                    .camera
                    .project_to_image(&p_cam, 0.0, nb_kf.frame.image_size)
                else {
                    continue;
                };
                let u = pixel.x as f32;
                let v = pixel.y as f32;

                // Find the closest unassociated keypoint within the radius
                // that matches the map point's representative descriptor.
                let mut best_dist = u32::MAX;
                let mut best_kp = usize::MAX;
                for (kp_idx, kp) in nb_kp_undist.iter().enumerate() {
                    if nb_kf.map_point(kp_idx).is_some() {
                        continue;
                    }
                    let dx = kp[0] - u;
                    let dy = kp[1] - v;
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let dist =
                        hamming_distance(&mp.descriptor, &nb_kf.frame.features.descriptors[kp_idx]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_kp = kp_idx;
                    }
                }

                if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                    proposals.push((best_kp, mp_idx, best_dist));
                }
            }

            // Resolve proposals: if two map points want the same keypoint,
            // the one with the smaller Hamming distance wins. Track which
            // keypoints are already taken in this pass.
            proposals.sort_by_key(|&(_, _, dist)| dist);
            let mut taken_kp: HashSet<usize> = HashSet::new();
            for (kp_idx, mp_idx, _) in proposals {
                if taken_kp.contains(&kp_idx) {
                    continue;
                }
                // Re-check that the live neighbor KF hasn't already had this
                // keypoint claimed (e.g. by a prior iteration in this fuse
                // call associating a different mp).
                let already = self
                    .map
                    .get_keyframe(nb_kf_idx)
                    .and_then(|kf| kf.map_point(kp_idx))
                    .is_some();
                if already {
                    continue;
                }
                self.map.register_observation(mp_idx, &nb_kf, kp_idx);
                if let Some(nb_live) = self.map.get_keyframe_mut(nb_kf_idx) {
                    nb_live.associate_map_point(kp_idx, mp_idx);
                }
                taken_kp.insert(kp_idx);
                n_fused += 1;
            }
        }

        n_fused
    }
}

fn vec_axis(v: Vec3F64, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => unreachable!("Vec3F64 has three axes"),
    }
}

fn solve_least_squares(rows: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rows.first()?.len();
    if rows.len() != rhs.len() {
        return None;
    }

    let mut normal = vec![vec![0.0; n]; n];
    let mut normal_rhs = vec![0.0; n];

    for (row, &b) in rows.iter().zip(rhs.iter()) {
        if row.len() != n {
            return None;
        }
        for i in 0..n {
            normal_rhs[i] += row[i] * b;
            for j in 0..n {
                normal[i][j] += row[i] * row[j];
            }
        }
    }

    solve_linear_system(normal, normal_rhs)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs < 1e-9 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}
