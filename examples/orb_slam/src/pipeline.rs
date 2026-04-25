//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use std::collections::HashSet;
use std::sync::Arc;

use crate::config::PipelineConfig;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{TwoViewConfig, TwoViewModel, triangulate_matched_points, two_view_estimate};
use kornia_algebra::Vec3F64;
use kornia_bow::Vocabulary;
use kornia_bow::metric::Hamming;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};
use kornia_slam::Frame;
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::map_projection::MapProjectionReport;
use kornia_slam::estimation::triangulation_search::{
    TriangulationSearchConfig, search_for_triangulation,
};
use kornia_slam::estimation::two_view::{
    TwoViewInitConfig, TwoViewInitReport, try_initialize_two_view_with_report,
};
use kornia_slam::map::{FuseMode, Keyframe, Map, MapPoint};
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
    // Experimental SearchInNeighbors-style fusion. Kept disabled by default
    // until its association heuristics are stable over longer runs.
    enable_neighbor_fusion: bool,
    // Synchronous approximation of ORB-SLAM3's LocalMapping busy/idle gate.
    mapping_cooldown_frames: usize,
    mapping_busy_until_frame: Option<usize>,
    // Keep using the high-capacity initial ORB detector until this frame index.
    initial_orb_until_frame: Option<usize>,
    // ORB-SLAM3 keeps the initial detector active for a short window after map init.
    initial_orb_window_frames: usize,
    // Map object
    map: Map,
    // System state
    state: SystemState,
    // ORB-SLAM3 vocabulary for BoW-accelerated matching (optional).
    vocabulary: Option<Arc<Vocabulary<10, Hamming<4>>>>,
    // Tree level at which direct-index groups are collected (ORB-SLAM3 uses 2).
    direct_index_level: usize,
}

/// Non-behavioral trace data produced while processing one frame.
#[derive(Debug, Clone)]
pub struct PipelineTrace {
    pub mode_before: SystemMode,
    pub mode_after: SystemMode,
    pub tracked_inliers: usize,
    pub init: Option<PipelineInitTrace>,
    pub tracking: Option<MapProjectionReport>,
}

impl PipelineTrace {
    fn new(mode: SystemMode) -> Self {
        Self {
            mode_before: mode,
            mode_after: mode,
            tracked_inliers: 0,
            init: None,
            tracking: None,
        }
    }
}

/// Trace data for a bootstrap two-view attempt.
#[derive(Debug, Clone)]
pub struct PipelineInitTrace {
    pub frame_idx_ref: usize,
    pub frame_idx_cur: usize,
    pub feature_count_ref: usize,
    pub feature_count_cur: usize,
    pub report: TwoViewInitReport,
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
            enable_neighbor_fusion: config.enable_neighbor_fusion,
            mapping_cooldown_frames: config.mapping_cooldown_frames,
            mapping_busy_until_frame: None,
            initial_orb_until_frame: None,
            initial_orb_window_frames: config.initial_orb_window_frames,
            map: Map::new(),
            state: SystemState::new(),
            vocabulary: None,
            direct_index_level: 2,
        }
    }

    /// Attaches an ORB-SLAM3 vocabulary so BoW/DirectIndex are computed for new keyframes.
    pub fn with_vocabulary(
        mut self,
        vocabulary: Arc<Vocabulary<10, Hamming<4>>>,
        direct_index_level: usize,
    ) -> Self {
        self.vocabulary = Some(vocabulary);
        self.direct_index_level = direct_index_level;
        self
    }

    fn attach_bow(&self, kf: &mut Keyframe) {
        if let Some(vocab) = self.vocabulary.as_ref() {
            if let Err(e) = kf.compute_bow(vocab, self.direct_index_level) {
                eprintln!(
                    "[pipeline] warn: compute_bow failed for kf idx {}: {:?}",
                    kf.frame.idx, e
                );
            }
        }
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    #[allow(dead_code)]
    pub fn process_frame(&mut self, frame: Frame) -> TrackingResult {
        self.process_frame_with_trace(frame).0
    }

    /// Processes one frame and returns trace counters alongside the tracking result.
    pub fn process_frame_with_trace(&mut self, frame: Frame) -> (TrackingResult, PipelineTrace) {
        let mode_before = self.state.mode;
        let (result, mut trace) = match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_step(frame),
            SystemMode::Tracking => self.tracking_step(frame),
        };
        trace.mode_before = mode_before;
        trace.mode_after = self.state.mode;
        (result, trace)
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    #[allow(dead_code)]
    pub fn process_frame_without_trace(&mut self, frame: Frame) -> TrackingResult {
        match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_step(frame),
            SystemMode::Tracking => self.tracking_step(frame),
        }
        .0
    }

    /// Returns true when the caller should use the high-capacity initial ORB detector.
    pub fn use_initial_orb_extractor(&self, next_frame_idx: usize) -> bool {
        self.state.mode == SystemMode::Bootstrap
            || self
                .initial_orb_until_frame
                .is_some_and(|until| next_frame_idx < until)
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

    /// Returns the number of active, non-culled persistent map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map.num_active_map_points()
    }

    /// Returns the total number of keyframes.
    pub fn num_keyframes(&self) -> usize {
        self.map.keyframes().len()
    }

    fn bootstrap_step(&mut self, mut curr_frame: Frame) -> (TrackingResult, PipelineTrace) {
        let mut trace = PipelineTrace::new(SystemMode::Bootstrap);
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let Some(prev_bootstrap_frame) = self.state.bootstrap_frame.take() else {
            self.state.bootstrap_frame = Some(curr_frame);
            return (
                TrackingResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: TrackingStatus::Skipped,
                },
                trace,
            );
        };

        let (result, init_report) = try_initialize_two_view_with_report(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &self.camera,
            &self.two_view_init_config,
        );
        trace.init = Some(PipelineInitTrace {
            frame_idx_ref: prev_bootstrap_frame.idx,
            frame_idx_cur: curr_frame.idx,
            feature_count_ref: prev_bootstrap_frame.features.descriptors.len(),
            feature_count_cur: curr_frame.features.descriptors.len(),
            report: init_report,
        });

        let two_view_estimate = match result {
            Err(_) => {
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                return (
                    TrackingResult {
                        pose_world_to_cam: self.state.pose_world_to_cam,
                        status: TrackingStatus::Skipped,
                    },
                    trace,
                );
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
        let mut reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let mut current_kf = Keyframe::from_frame(curr_frame);
        self.attach_bow(&mut reference_kf);
        self.attach_bow(&mut current_kf);
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
        self.initial_orb_until_frame = Some(curr_idx + self.initial_orb_window_frames);
        trace.tracked_inliers = two_view_estimate.estimate.inliers;

        (
            TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: TrackingStatus::KeyframeAccepted,
            },
            trace,
        )
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

    fn tracking_step(&mut self, frame: Frame) -> (TrackingResult, PipelineTrace) {
        let mut trace = PipelineTrace::new(SystemMode::Tracking);
        let pose_before_tracking = self.state.pose_world_to_cam;
        let image_size = frame.image_size;

        let candidate_pose = if let Some(vel) = self.state.velocity {
            vel.compose(&self.state.pose_world_to_cam)
        } else {
            self.state.pose_world_to_cam
        };

        let (result, tracking_report) = self.estimator.estimate_pose_with_report(
            &frame,
            &candidate_pose,
            &pose_before_tracking,
            &self.map,
            &self.camera,
            self.state.current_keyframe_idx,
        );
        if result.is_err() {
            eprintln!(
                "[diag] frame {} TRACK_FAIL: proj_matches={} proj_pnp_in={} ref_matches={} ref_corr={} local_matches={} local_pnp_in={} reason={:?}",
                frame.idx,
                tracking_report.projection_matches,
                tracking_report.projection_pnp_inliers,
                tracking_report.reference_matches,
                tracking_report.reference_correspondences,
                tracking_report.local_projection_matches,
                tracking_report.local_pnp_inliers,
                tracking_report.reject_reason,
            );
        }
        trace.tracking = Some(tracking_report);

        let (mut status, matches, tracked_inliers) = match result {
            Ok(estimate) => {
                self.state.velocity = Some(Pose3d::between(&pose_before_tracking, &estimate.pose));
                self.state.pose_world_to_cam = estimate.pose;
                (TrackingStatus::Tracked, estimate.matches, estimate.inliers)
            }
            Err(_) => (TrackingStatus::Skipped, Vec::new(), 0),
        };
        trace.tracked_inliers = tracked_inliers;

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
                self.initial_orb_until_frame = None;
                let (result, mut reset_trace) = self.bootstrap_step(frame);
                reset_trace.tracking = trace.tracking;
                return (result, reset_trace);
            }
        } else {
            self.state.consecutive_failures = 0;
        }

        (
            TrackingResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status,
            },
            trace,
        )
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        // ORB-SLAM3 NeedNewKeyFrame uses TrackedMapPoints(nMinObs) of the
        // reference KF — points observed by >=3 KFs once the map has grown.
        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .map(|ki| {
                let min_obs = if self.map.keyframes().len() <= 2 {
                    2
                } else {
                    3
                };
                self.map.tracked_map_points(ki, min_obs)
            })
            .unwrap_or(0);

        let mapping_busy = self
            .mapping_busy_until_frame
            .is_some_and(|until| frame.idx < until);
        let forced_by_max_gap = self.state.last_keyframe_idx.is_some_and(|last| {
            frame.idx.saturating_sub(last) >= self.keyframe_policy.max_frames_between
        });
        if mapping_busy && !forced_by_max_gap {
            return false;
        }

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
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
        });
        self.attach_bow(&mut curr_kf);
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

        if self.enable_neighbor_fusion {
            let fused = self.search_in_neighbors(frame.idx);
            if fused > 0 {
                eprintln!("[pipeline] SearchInNeighbors fused {} observations", fused);
            }
        }

        let culled = self.map.map_point_culling();
        if culled > 0 {
            eprintln!("[pipeline] map point culling removed {} points", culled);
        }

        if enable_local_ba {
            self.map.run_local_ba(&self.camera);
            if let Some(newest_kf) = self.map.keyframes().last() {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
        }

        let culled_keyframes = self.map.keyframe_culling();
        if culled_keyframes > 0 {
            eprintln!(
                "[pipeline] keyframe culling removed {} keyframes",
                culled_keyframes
            );
        }

        self.map.cull();
        if self.mapping_cooldown_frames > 0 {
            self.mapping_busy_until_frame = Some(frame.idx + self.mapping_cooldown_frames);
        }
        true
    }

    fn search_in_neighbors(&mut self, curr_kf_idx: usize) -> usize {
        // ORB-SLAM3 `LocalMapping::SearchInNeighbors` calls `Fuse(..)` once per
        // target KF (forward) and once for the union of target-KF MPs into
        // currKF (backward), with NO cap and full Replace semantics. Module 6
        // replay parity (`parity/rust/src/bin/parity_local_mapping.rs`) found
        // that:
        //   - `FuseMode::AddOnly` causes Rust to never cull loser MPs in
        //     conflicts, so they keep getting projected into later target KFs
        //     (~75% of replay mismatches). `ReplaceWeaker` drops this.
        //   - `MAX_FUSED_PER_INSERT = 120` was only "safe" because of
        //     `AddOnly`; with proper Replace, ORB-SLAM3 has no such cap.
        // The two combined cut replay diff by ~30% on the smoke set.
        const FIRST_ORDER_NEIGHBORS: usize = 30;
        const SECOND_ORDER_NEIGHBORS: usize = 20;
        const MAX_TARGET_KEYFRAMES: usize = 50;

        let mut target_kfs = Vec::new();
        let mut seen = HashSet::new();
        let push_target = |kf_idx: usize, targets: &mut Vec<usize>, seen: &mut HashSet<usize>| {
            if kf_idx != curr_kf_idx && targets.len() < MAX_TARGET_KEYFRAMES && seen.insert(kf_idx)
            {
                targets.push(kf_idx);
            }
        };

        for (neighbor_idx, _) in self
            .map
            .covisibility_neighbors(curr_kf_idx, FIRST_ORDER_NEIGHBORS)
        {
            push_target(neighbor_idx, &mut target_kfs, &mut seen);
        }

        let first_order = target_kfs.clone();
        for kf_idx in first_order {
            if target_kfs.len() >= MAX_TARGET_KEYFRAMES {
                break;
            }
            for (second_idx, _) in self
                .map
                .covisibility_neighbors(kf_idx, SECOND_ORDER_NEIGHBORS)
            {
                push_target(second_idx, &mut target_kfs, &mut seen);
                if target_kfs.len() >= MAX_TARGET_KEYFRAMES {
                    break;
                }
            }
        }

        if target_kfs.is_empty() {
            return 0;
        }

        let current_map_points = self.map.keyframe_map_point_indices(curr_kf_idx);
        let mut fused = 0usize;
        for target_kf_idx in target_kfs.iter().copied() {
            fused += self
                .map
                .fuse_projected_map_points_into_keyframe_limited_with_mode(
                    target_kf_idx,
                    &current_map_points,
                    &self.camera,
                    usize::MAX,
                    FuseMode::ReplaceWeaker,
                );
        }

        // Match ORB-SLAM3 backward pass: union of target KFs' MPs in first-seen
        // order across target KFs (each KF's MPs in keypoint-index order via
        // `keyframe_map_point_indices`). Sequential Replace makes order
        // matter.
        let mut fuse_candidates: Vec<usize> = Vec::new();
        let mut seen_candidates = HashSet::new();
        for target_kf_idx in target_kfs.iter().copied() {
            for mp_idx in self.map.keyframe_map_point_indices(target_kf_idx) {
                if seen_candidates.insert(mp_idx) {
                    fuse_candidates.push(mp_idx);
                }
            }
        }
        fused += self
            .map
            .fuse_projected_map_points_into_keyframe_limited_with_mode(
                curr_kf_idx,
                &fuse_candidates,
                &self.camera,
                usize::MAX,
                FuseMode::ReplaceWeaker,
            );

        fused
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

    /// Grows new map points from BoW-matched pairs against the most recent
    /// keyframes. Port of ORB-SLAM3's `LocalMapping::CreateNewMapPoints`.
    ///
    /// Unlike [`grow_map_points_from_keyframe_pair`] this skips brute-force
    /// descriptor matching and a second two-view RANSAC, trusting the existing
    /// poses + the BoW/epipolar filter in [`search_for_triangulation`].
    fn grow_map_points_bow_against_recent(
        &mut self,
        curr_kf: &mut Keyframe,
        two_view_config: &TwoViewConfig,
        num_neighbors: usize,
    ) -> usize {
        if self.vocabulary.is_none() || curr_kf.direct_index.is_none() {
            return 0;
        }
        let search_config = TriangulationSearchConfig::default();
        let triangulation_config = &two_view_config.triangulation;
        let camera = self.camera.clone();

        // Snapshot the neighbor keyframes so we can pass an immutable borrow of
        // each through the search + triangulation loop while holding `&mut self`.
        let neighbor_kfs: Vec<Keyframe> = self
            .map
            .keyframes()
            .iter()
            .rev()
            .filter(|kf| kf.frame.idx != curr_kf.frame.idx)
            .filter(|kf| kf.direct_index.is_some())
            .take(num_neighbors)
            .cloned()
            .collect();

        let mut total_added = 0usize;
        for neighbor in neighbor_kfs.iter() {
            let pairs = search_for_triangulation(neighbor, curr_kf, &camera, &search_config);
            eprintln!(
                "[pipeline]   bow pairs kf{} <-> kf{}: {} (neighbor desc={}, curr desc={})",
                neighbor.frame.idx,
                curr_kf.frame.idx,
                pairs.len(),
                neighbor.frame.features.descriptors.len(),
                curr_kf.frame.features.descriptors.len(),
            );
            if pairs.is_empty() {
                continue;
            }

            let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
                &neighbor.frame.features.keypoints_xy,
                &curr_kf.frame.features.keypoints_xy,
                &pairs,
            );

            let triangulated = match triangulate_matched_points(
                &prev_pts,
                &curr_pts,
                &neighbor.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
                &camera,
                triangulation_config,
            ) {
                Ok(pts) => pts,
                Err(_) => continue,
            };
            if triangulated.is_empty() {
                continue;
            }

            let mut points = Vec::new();
            let mut used_curr: HashSet<usize> = HashSet::new();
            for tp in &triangulated {
                let Some(&(prev_idx, curr_idx)) = pairs.get(tp.pair_index) else {
                    continue;
                };
                if curr_kf.map_point(curr_idx).is_some() {
                    continue;
                }
                if neighbor.map_point(prev_idx).is_some() {
                    // Would only happen if the neighbor acquired a map point at this
                    // feature since BoW was computed — skip to avoid duplicates.
                    continue;
                }
                if !used_curr.insert(curr_idx) {
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

            if points.is_empty() {
                continue;
            }

            let kf_idx = curr_kf.frame.idx;
            // Back-associate on the neighbor too: look up its mutable reference
            // via frame idx after insertion.
            let neighbor_idx = neighbor.frame.idx;
            let added = self
                .map
                .add_triangulated_points(None, curr_kf, &points, kf_idx);
            if added > 0 {
                if let Some(mp_start) = self.map.map_points().len().checked_sub(added) {
                    for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().enumerate() {
                        self.map
                            .associate_keyframe_map_point(neighbor_idx, prev_desc_idx, mp_start + i);
                    }
                }
            }
            total_added += added;
        }
        total_added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 0.00001762,
        }
    }

    #[test]
    fn initial_orb_extractor_window_switches_back_to_tracking_detector() {
        let mut pipeline = Pipeline::new(test_camera(), PipelineConfig::default());
        assert!(pipeline.use_initial_orb_extractor(0));

        pipeline.state.mode = SystemMode::Tracking;
        pipeline.initial_orb_until_frame = Some(22);

        assert!(pipeline.use_initial_orb_extractor(21));
        assert!(!pipeline.use_initial_orb_extractor(22));
    }
}
