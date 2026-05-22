//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! DUPLICATED from `examples/orb_slam/src/pipeline.rs` (see PIPELINE_SYNC.md).
//! Do not edit independently without considering whether the example's copy
//! should change too.

use std::collections::HashSet;

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{
    triangulate_matched_points, TriangulationConfig, TwoViewEstimator, TwoViewModel,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{hamming_distance, match_orb_descriptors, OrbMatchConfig};
use kornia_slam::estimation::two_view::{try_initialize_two_view, TwoViewInitConfig};
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::map::{Keyframe, Map, MapPoint};
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus,
};
use kornia_slam::Frame;

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

    /// Returns the number of active (non-culled) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map.num_active_map_points()
    }

    fn bootstrap_step(&mut self, mut curr_frame: Frame) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        // Staleness guard (mirrors ORB-SLAM3's MonocularInitialization):
        // a frame with too few keypoints is neither a viable reference nor
        // a viable current frame. If we already had a reference, drop it
        // and wait for a feature-rich frame to start over.
        const MIN_KEYPOINTS_FOR_BOOTSTRAP: usize = 100;
        if curr_frame.features.keypoints_xy.len() <= MIN_KEYPOINTS_FOR_BOOTSTRAP {
            self.state.bootstrap_frame = None;
            return TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::Skipped,
            };
        }

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

        eprintln!(
            "[bootstrap] model={} triangulated={} inliers={}",
            two_view_estimate.model_kind,
            two_view_estimate.points3d.len(),
            two_view_estimate.estimate.inliers,
        );

        let estimated_pose = two_view_estimate.estimate.pose;
        let prev_pose_world_to_cam = curr_frame.pose_world_to_cam;
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

        // Post-BA sanity gate (mirrors ORB-SLAM3's reset criteria in
        // CreateInitialMapMonocular). Discard the bootstrap if the resulting
        // map has too few valid points or a degenerate scale.
        const MIN_VALID_POINTS: usize = 50;
        let health = self.map.initial_map_health();
        if health.valid_in_both < MIN_VALID_POINTS || health.median_depth_older_kf <= 0.0 {
            eprintln!(
                "[init_gate] reject: valid_in_both={} median_depth={:.3} (need >= {} and > 0)",
                health.valid_in_both, health.median_depth_older_kf, MIN_VALID_POINTS,
            );
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
        self.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.state.pose_world_to_cam,
        ));

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
        eprintln!(
            "[track] frame={} status={:?} matches={} inliers={}",
            frame.idx,
            status,
            matches.len(),
            tracked_inliers,
        );

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
        });
        for &(mp_idx, curr_idx) in matches {
            curr_kf.associate_map_point(curr_idx, mp_idx);
            self.map.register_observation(mp_idx, &curr_kf, curr_idx);
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
        eprintln!(
            "[kf] frame={} grown={} from {} neighbor kfs",
            frame.idx,
            total_grown,
            neighbor_kfs.len()
        );

        self.map.upsert_keyframe(curr_kf);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Forward SearchInNeighbors / Fuse: extend each curr_kf-observed map
        // point's observation list to neighbor KFs that don't yet observe it.
        // Run before local BA so BA sees the extra reprojection constraints.
        let neighbor_kf_indices: Vec<usize> = neighbor_kfs.iter().map(|kf| kf.frame.idx).collect();
        let n_fused = self.fuse_into_neighbors(frame.idx, &neighbor_kf_indices);
        eprintln!("[fuse] frame={} fused={}", frame.idx, n_fused);

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
