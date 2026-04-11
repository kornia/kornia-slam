//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use std::collections::HashSet;

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{TwoViewConfig, TwoViewModel, triangulate_matched_points, two_view_estimate};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};
use kornia_slam::Frame;
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::two_view::{TwoViewInitConfig, try_initialize_two_view};
use kornia_slam::map::{Keyframe, Map, MapPoint};
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus,
};

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
    // Map object
    map: Map,
    // System state
    state: SystemState,
}

impl Pipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(camera: PinholeCamera, config: PipelineConfig) -> Self {
        Self {
            camera,
            estimator: MapProjectionEstimator::new(config.map_projection),
            two_view_init_config: config.two_view_init,
            keyframe_policy: config.keyframe_policy,
            enable_local_ba: config.enable_local_ba,
            map: Map::new(),
            state: SystemState::new(),
        }
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    pub fn process_frame(&mut self, frame: Frame) -> TrackingResult {
        match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_step(frame),
            SystemMode::Tracking => self.tracking_step(frame),
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

    /// Returns the total number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map.map_points().len()
    }

    fn bootstrap_step(&mut self, mut curr_frame: Frame) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let Some(prev_bootstrap_frame) = self.state.bootstrap_frame.take() else {
            self.state.bootstrap_frame = Some(curr_frame);
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
            Err(_) => {
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            Ok(tv) => tv,
        };

        let estimated_pose = two_view_estimate.estimate.pose;
        self.state.velocity = Some(Pose3d::between(
            &curr_frame.pose_world_to_cam,
            &estimated_pose,
        ));
        self.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
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
        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.mode = SystemMode::Tracking;

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

        let ref_idx = reference_kf.frame.idx;
        let added = self.map.add_triangulated_points(
            Some(&mut reference_kf),
            &mut current_kf,
            &triangulated,
            ref_idx,
        );

        self.map.upsert_keyframe(reference_kf);
        self.map.upsert_keyframe(current_kf);
        added
    }

    fn tracking_step(&mut self, frame: Frame) -> TrackingResult {
        let pose_before_tracking = self.state.pose_world_to_cam;
        let image_size = frame.image_size;

        let candidate_pose = if let Some(vel) = self.state.velocity {
            vel.compose(&self.state.pose_world_to_cam)
        } else {
            self.state.pose_world_to_cam
        };

        let result = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before_tracking,
            &self.map,
            &self.camera,
            self.state.current_keyframe_idx,
        );

        let (mut status, matches, tracked_inliers) = match result {
            Ok(estimate) => {
                self.state.velocity = Some(Pose3d::between(&pose_before_tracking, &estimate.pose));
                self.state.pose_world_to_cam = estimate.pose;
                (TrackingStatus::Tracked, estimate.matches, estimate.inliers)
            }
            Err(_) => (TrackingStatus::Skipped, Vec::new(), 0),
        };

        if status == TrackingStatus::Tracked {
            let visible = self
                .map
                .map_points_in_frustum(&self.camera, &candidate_pose, image_size);
            self.map.update_observation_counts(&visible, &matches);

            if self.try_insert_keyframe(&frame, tracked_inliers, &matches) {
                status = TrackingStatus::KeyframeAccepted;
            }
        }

        if status == TrackingStatus::Skipped {
            self.state.consecutive_failures += 1;
            if self.state.consecutive_failures >= self.state.max_consecutive_failures {
                self.state.reset();
                return self.bootstrap_step(frame);
            }
        } else {
            self.state.consecutive_failures = 0;
        }

        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
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

        let Some(prev_kf) = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .cloned()
        else {
            return false;
        };

        let mut curr_kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            timestamp: frame.timestamp,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
        });
        for &(mp_idx, curr_idx) in matches {
            curr_kf.associate_map_point(curr_idx, mp_idx);
        }

        let enable_local_ba = self.enable_local_ba;
        let match_config = self.two_view_init_config.match_config;
        let estimation_config = self.two_view_init_config.estimation_config.clone();
        self.grow_map_points_from_keyframe_pair(
            &prev_kf,
            &mut curr_kf,
            match_config,
            &estimation_config,
        );

        self.map.upsert_keyframe(curr_kf);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

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
        two_view_config: &TwoViewConfig,
    ) -> usize {
        const MIN_GROWTH_MATCHES: usize = 20;
        const MIN_GROWTH_INLIERS: usize = 15;

        let camera = &self.camera;
        let triangulation_config = &two_view_config.triangulation;
        let matches = match_orb_descriptors(
            &prev_kf.frame.features.orientations,
            &prev_kf.frame.features.descriptors,
            &curr_kf.frame.features.orientations,
            &curr_kf.frame.features.descriptors,
            match_config,
        );
        if matches.len() < MIN_GROWTH_MATCHES {
            return 0;
        }

        let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(matches.len());
        for (prev_idx, curr_idx) in matches {
            if prev_idx >= prev_kf.frame.features.keypoints_xy.len()
                || curr_idx >= curr_kf.frame.features.keypoints_xy.len()
            {
                continue;
            }
            if curr_kf.map_point(curr_idx).is_some() {
                continue;
            }
            if prev_kf.map_point(prev_idx).is_some() {
                continue;
            }
            pair_indices.push((prev_idx, curr_idx));
        }

        let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
            &prev_kf.frame.features.keypoints_xy,
            &curr_kf.frame.features.keypoints_xy,
            &pair_indices,
        );
        if pair_indices.len() < 8 {
            return 0;
        }

        let k = camera.intrinsic_matrix();
        let two_view = match two_view_estimate(&prev_pts, &curr_pts, &k, &k, two_view_config) {
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

        let kf_idx = curr_kf.frame.idx;
        self.map
            .add_triangulated_points(None, curr_kf, &points, kf_idx)
    }
}
