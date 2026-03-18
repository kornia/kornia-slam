//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{
    TwoViewConfig, TwoViewModel, triangulate_matched_points, two_view_estimate,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::two_view::{TwoViewInitConfig, try_initialize_two_view};
use kornia_slam::map::{Keyframe, Map, MapPoint};
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus,
};
use kornia_slam::{Frame, OrbFeatures};

/// Top-level ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
pub struct Pipeline {
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
            estimator: MapProjectionEstimator::new(camera, config.map_projection),
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
            self.estimator.camera(),
            &self.two_view_init_config,
        );

        let tv = match result {
            Err(_) => {
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                };
            }
            Ok(tv) => tv,
        };

        let motion_increment =
            Pose3d::between(&curr_frame.pose_world_to_cam, &tv.estimate.pose);
        self.state.velocity = Some(motion_increment);
        self.state.pose_world_to_cam = tv.estimate.pose;
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let curr_idx = curr_frame.idx;
        self.build_initial_map(
            prev_bootstrap_frame,
            curr_frame,
            &tv.estimate.matches,
            &tv.points3d,
            &tv.inlier_indices,
            tv.median_depth,
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
        reference_frame: Frame,
        current_frame: Frame,
        matches: &[(usize, usize)],
        points3d: &[Vec3F64],
        inlier_indices: &[usize],
        median_depth: Option<f64>,
    ) -> usize {
        let mut reference_kf = Keyframe::from_frame(reference_frame);
        let mut current_kf = Keyframe::from_frame(current_frame);
        let depth_scale = median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
        let reference_pose_inv = reference_kf.frame.pose_world_to_cam.inverse();
        let mut added = 0usize;

        for (p_cam, &match_idx) in points3d.iter().zip(inlier_indices.iter()) {
            let Some(&(reference_desc_idx, current_desc_idx)) = matches.get(match_idx) else {
                continue;
            };
            if reference_desc_idx >= reference_kf.map_point_by_desc_idx.len()
                || current_desc_idx >= current_kf.map_point_by_desc_idx.len()
            {
                continue;
            }

            let descriptor = current_kf
                .frame
                .features
                .descriptors
                .get(current_desc_idx)
                .copied()
                .or_else(|| {
                    reference_kf
                        .frame
                        .features
                        .descriptors
                        .get(reference_desc_idx)
                        .copied()
                });
            let Some(descriptor) = descriptor else {
                continue;
            };

            let p_world = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
            let mp_idx =
                self.map
                    .push_map_point(MapPoint::new(p_world, descriptor, reference_kf.frame.idx));
            reference_kf.associate_map_point(reference_desc_idx, mp_idx);
            current_kf.associate_map_point(current_desc_idx, mp_idx);
            added += 1;
        }

        let ref_idx = reference_kf.frame.idx;
        let cur_idx = current_kf.frame.idx;
        self.map.upsert_keyframe(reference_kf);
        self.map.upsert_keyframe(current_kf);
        self.map.update_covisibility(ref_idx);
        self.map.update_covisibility(cur_idx);
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
            self.state.current_keyframe_idx,
        );

        let (mut status, matches, tracked_inliers) = match result {
            Ok(estimate) => {
                self.state.velocity =
                    Some(Pose3d::between(&pose_before_tracking, &estimate.pose));
                self.state.pose_world_to_cam = estimate.pose;
                (TrackingStatus::Tracked, estimate.matches, estimate.inliers)
            }
            Err(_) => (TrackingStatus::Skipped, Vec::new(), 0),
        };

        if status == TrackingStatus::Tracked {
            let visible =
                self.map
                    .map_points_in_frustum(self.estimator.camera(), &candidate_pose, image_size);
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

        let mut curr_kf_map_assoc = vec![None; frame.features.descriptors.len()];
        for &(mp_idx, curr_idx) in matches {
            if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
                *slot = Some(mp_idx);
            }
        }

        let enable_local_ba = self.enable_local_ba;
        let pose_world_to_cam = self.state.pose_world_to_cam;
        let match_config = self.two_view_init_config.match_config;
        let estimation_config = self.two_view_init_config.estimation_config.clone();
        self.grow_map_points_from_keyframe_pair(
            frame.idx,
            &prev_kf,
            &frame.features,
            &mut curr_kf_map_assoc,
            &pose_world_to_cam,
            match_config,
            &estimation_config,
        );

        let mut kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
        });
        kf.map_point_by_desc_idx = curr_kf_map_assoc;
        self.map.upsert_keyframe(kf);
        self.map.update_covisibility(frame.idx);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        if enable_local_ba {
            self.map.optimize(self.estimator.camera());
            if let Some(newest_kf) = self.map.keyframes().last() {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
        }

        self.map.cull();
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn grow_map_points_from_keyframe_pair(
        &mut self,
        curr_kf_idx: usize,
        prev_kf: &Keyframe,
        curr_features: &OrbFeatures,
        curr_kf_map_assoc: &mut [Option<usize>],
        pose_world_to_cam: &Pose3d,
        match_config: OrbMatchConfig,
        two_view_config: &TwoViewConfig,
    ) -> usize {
        const MIN_GROWTH_MATCHES: usize = 20;
        const MIN_GROWTH_INLIERS: usize = 15;

        let camera = self.estimator.camera();
        let triangulation_config = &two_view_config.triangulation;
        let matches = match_orb_descriptors(
            &prev_kf.frame.features.orientations,
            &prev_kf.frame.features.descriptors,
            &curr_features.orientations,
            &curr_features.descriptors,
            match_config,
        );
        if matches.len() < MIN_GROWTH_MATCHES {
            return 0;
        }

        let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(matches.len());
        for (prev_idx, curr_idx) in matches {
            if prev_idx >= prev_kf.frame.features.keypoints_xy.len()
                || curr_idx >= curr_features.keypoints_xy.len()
            {
                continue;
            }
            if curr_kf_map_assoc.get(curr_idx).is_some_and(|m| m.is_some()) {
                continue;
            }
            if prev_kf.map_point(prev_idx).is_some() {
                continue;
            }
            pair_indices.push((prev_idx, curr_idx));
        }

        let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
            &prev_kf.frame.features.keypoints_xy,
            &curr_features.keypoints_xy,
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

        let triangulated = triangulate_matched_points(
            &inlier_prev,
            &inlier_curr,
            &prev_kf.frame.pose_world_to_cam,
            pose_world_to_cam,
            camera,
            triangulation_config,
        );

        let mut n_added = 0usize;
        for tp in &triangulated {
            let inlier_idx = two_view.inlier_indices[tp.pair_index];
            let Some(&(_prev_idx, curr_idx)) = pair_indices.get(inlier_idx) else {
                continue;
            };
            if curr_kf_map_assoc.get(curr_idx).is_some_and(|m: &Option<usize>| m.is_some()) {
                continue;
            }

            let mp_idx = self.map.push_map_point(MapPoint::new(
                tp.position,
                curr_features.descriptors[curr_idx],
                curr_kf_idx,
            ));

            if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
                *slot = Some(mp_idx);
                n_added += 1;
            }
        }

        n_added
    }
}
