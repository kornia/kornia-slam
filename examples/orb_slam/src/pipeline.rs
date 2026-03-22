//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This example keeps the runtime flow in one file so it can be read from top
//! to bottom in the same order frames move through the system.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};

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
    // Camera intrinsics (shared across subsystems)
    camera: PinholeCamera,
    // Primary pose estimator
    estimator: MapProjectionEstimator,
    // Boostrap pose estimator
    two_view_init_config: TwoViewInitConfig,
    // Keyframe insertion policy
    keyframe_policy: KeyframePolicy,
    // Shared map object
    map: Arc<RwLock<Map>>,
    // System state
    state: SystemState,
    // Dedicated mapping worker
    mapping_worker: MappingWorker,
    // Mapping updates consumed by tracking
    mapping_updates: Receiver<MappingUpdate>,
}

impl Pipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(camera: PinholeCamera, config: PipelineConfig) -> Self {
        let map = Arc::new(RwLock::new(Map::new()));
        let (mapping_updates, mapping_worker) = MappingWorker::spawn(
            Arc::clone(&map),
            camera.clone(),
            config.two_view_init.match_config,
            config.two_view_init.estimation_config.clone(),
            config.enable_local_ba,
            config.mapping_queue_capacity,
        );

        Self {
            estimator: MapProjectionEstimator::new(camera.clone(), config.map_projection),
            camera,
            two_view_init_config: config.two_view_init,
            keyframe_policy: config.keyframe_policy,
            map,
            state: SystemState::new(),
            mapping_worker,
            mapping_updates,
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
    pub fn map_points(&self) -> Vec<MapPoint> {
        self.map
            .read()
            .expect("map read lock poisoned")
            .map_points()
            .to_vec()
    }

    /// Returns the index of the current reference keyframe, if tracking has one.
    pub fn current_keyframe_idx(&self) -> Option<usize> {
        self.state
            .current_keyframe_idx
            .and_then(|ki| {
                self.map
                    .read()
                    .expect("map read lock poisoned")
                    .get_keyframe(ki)
                    .map(|kf| kf.frame.idx)
            })
    }

    /// Returns the total number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map
            .read()
            .expect("map read lock poisoned")
            .map_points()
            .len()
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
        self.state.velocity = Some(Pose3d::between(&curr_frame.pose_world_to_cam, &estimated_pose));
        self.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        self.mapping_worker.clear_pending();
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
        let cam_to_world_transform = reference_kf.frame.pose_world_to_cam.inverse();
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

            let p_world = cam_to_world_transform.transform_point(&(*p_cam / depth_scale));
            let mp_idx =
                self.map
                    .write()
                    .expect("map write lock poisoned")
                    .push_map_point(MapPoint::new(p_world, descriptor, reference_kf.frame.idx));
            reference_kf.associate_map_point(reference_desc_idx, mp_idx);
            current_kf.associate_map_point(current_desc_idx, mp_idx);
            added += 1;
        }

        let mut map = self.map.write().expect("map write lock poisoned");
        map.upsert_keyframe(reference_kf);
        map.upsert_keyframe(current_kf);
        added
    }

    fn tracking_step(&mut self, frame: Frame) -> TrackingResult {
        self.apply_mapping_updates();
        let pose_before_tracking = self.state.pose_world_to_cam;
        let image_size = frame.image_size;

        let candidate_pose = if let Some(vel) = self.state.velocity {
            vel.compose(&self.state.pose_world_to_cam)
        } else {
            self.state.pose_world_to_cam
        };

        let result = {
            let map = self.map.read().expect("map read lock poisoned");
            self.estimator.estimate_pose(
                &frame,
                &candidate_pose,
                &pose_before_tracking,
                &map,
                self.state.current_keyframe_idx,
            )
        };

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
                    .read()
                    .expect("map read lock poisoned")
                    .map_points_in_frustum(&self.camera, &candidate_pose, image_size);
            self.map
                .write()
                .expect("map write lock poisoned")
                .update_observation_counts(&visible, &matches);

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
        let n_ref_map_points = {
            let map = self.map.read().expect("map read lock poisoned");
            self.state
                .current_keyframe_idx
                .and_then(|ki| map.get_keyframe(ki))
                .map(|kf| kf.num_associated_points())
                .unwrap_or(0)
        };

        if !self.keyframe_policy.should_insert(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        let Some(prev_kf_idx) = self.state.current_keyframe_idx else {
            return false;
        };

        let mut curr_kf_map_assoc = vec![None; frame.features.descriptors.len()];
        for &(mp_idx, curr_idx) in matches {
            if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
                *slot = Some(mp_idx);
            }
        }

        let mut kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
        });
        kf.map_point_by_desc_idx = curr_kf_map_assoc;
        self.map
            .write()
            .expect("map write lock poisoned")
            .upsert_keyframe(kf);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        self.mapping_worker.enqueue(MappingTask {
            previous_keyframe_idx: prev_kf_idx,
            current_keyframe_idx: frame.idx,
        });
        true
    }

    fn apply_mapping_updates(&mut self) {
        while let Ok(update) = self.mapping_updates.try_recv() {
            if self.state.current_keyframe_idx == Some(update.current_keyframe_idx) {
                self.state.pose_world_to_cam = update.pose_world_to_cam;
                self.state.velocity = None;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn grow_map_points_from_keyframe_pair(
        map: &mut Map,
        camera: &PinholeCamera,
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

            let mp_idx = map.push_map_point(MapPoint::new(
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

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.mapping_worker.shutdown();
    }
}

#[derive(Debug, Clone, Copy)]
struct MappingTask {
    previous_keyframe_idx: usize,
    current_keyframe_idx: usize,
}

#[derive(Debug, Clone, Copy)]
struct MappingUpdate {
    current_keyframe_idx: usize,
    pose_world_to_cam: Pose3d,
}

#[derive(Default)]
struct MappingQueueState {
    pending: VecDeque<MappingTask>,
    shutdown: bool,
}

struct MappingQueue {
    capacity: usize,
    state: Mutex<MappingQueueState>,
    ready: Condvar,
}

impl MappingQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            state: Mutex::new(MappingQueueState::default()),
            ready: Condvar::new(),
        }
    }

    fn push(&self, task: MappingTask) {
        let mut state = self.state.lock().expect("mapping queue lock poisoned");
        if state.pending.len() == self.capacity {
            state.pending.pop_front();
        }
        state.pending.push_back(task);
        self.ready.notify_one();
    }

    fn pop(&self) -> Option<MappingTask> {
        let mut state = self.state.lock().expect("mapping queue lock poisoned");
        loop {
            if let Some(task) = state.pending.pop_front() {
                return Some(task);
            }
            if state.shutdown {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .expect("mapping queue wait poisoned");
        }
    }

    fn clear(&self) {
        self.state
            .lock()
            .expect("mapping queue lock poisoned")
            .pending
            .clear();
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().expect("mapping queue lock poisoned");
        state.shutdown = true;
        self.ready.notify_all();
    }
}

struct MappingWorker {
    queue: Arc<MappingQueue>,
    thread: Option<JoinHandle<()>>,
}

impl MappingWorker {
    fn spawn(
        map: Arc<RwLock<Map>>,
        camera: PinholeCamera,
        match_config: OrbMatchConfig,
        two_view_config: TwoViewConfig,
        enable_local_ba: bool,
        queue_capacity: usize,
    ) -> (Receiver<MappingUpdate>, Self) {
        let queue = Arc::new(MappingQueue::new(queue_capacity));
        let queue_for_thread = Arc::clone(&queue);
        let (updates_tx, updates_rx) = mpsc::channel();

        let thread = thread::Builder::new()
            .name("mapping".to_owned())
            .spawn(move || {
                while let Some(task) = queue_for_thread.pop() {
                    run_mapping_task(
                        &map,
                        &camera,
                        match_config,
                        &two_view_config,
                        enable_local_ba,
                        task,
                        &updates_tx,
                    );
                }
            })
            .expect("failed to spawn mapping thread");

        (
            updates_rx,
            Self {
                queue,
                thread: Some(thread),
            },
        )
    }

    fn enqueue(&self, task: MappingTask) {
        self.queue.push(task);
    }

    fn clear_pending(&self) {
        self.queue.clear();
    }

    fn shutdown(&mut self) {
        self.queue.shutdown();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run_mapping_task(
    map: &Arc<RwLock<Map>>,
    camera: &PinholeCamera,
    match_config: OrbMatchConfig,
    two_view_config: &TwoViewConfig,
    enable_local_ba: bool,
    task: MappingTask,
    updates_tx: &Sender<MappingUpdate>,
) {
    let mut working_map = map.read().expect("map read lock poisoned").clone();
    let base_map_point_count = working_map.map_points().len();

    let Some(prev_kf) = working_map.get_keyframe(task.previous_keyframe_idx).cloned() else {
        return;
    };
    let Some(curr_kf) = working_map.get_keyframe(task.current_keyframe_idx).cloned() else {
        return;
    };

    let mut curr_kf_map_assoc = curr_kf.map_point_by_desc_idx.clone();
    Pipeline::grow_map_points_from_keyframe_pair(
        &mut working_map,
        camera,
        task.current_keyframe_idx,
        &prev_kf,
        &curr_kf.frame.features,
        &mut curr_kf_map_assoc,
        &curr_kf.frame.pose_world_to_cam,
        match_config,
        two_view_config,
    );

    let mut updated_kf = curr_kf;
    updated_kf.map_point_by_desc_idx = curr_kf_map_assoc;
    working_map.upsert_keyframe(updated_kf);

    if enable_local_ba {
        working_map.optimize(camera);
    }
    working_map.cull();

    let optimized_pose = working_map
        .get_keyframe(task.current_keyframe_idx)
        .map(|kf| kf.frame.pose_world_to_cam);

    let mut shared_map = map.write().expect("map write lock poisoned");
    merge_mapping_result(&mut shared_map, &working_map, base_map_point_count);

    if let Some(pose_world_to_cam) = optimized_pose {
        let _ = updates_tx.send(MappingUpdate {
            current_keyframe_idx: task.current_keyframe_idx,
            pose_world_to_cam,
        });
    }
}

fn merge_mapping_result(shared_map: &mut Map, working_map: &Map, base_map_point_count: usize) {
    for (mp_idx, updated_mp) in working_map.map_points().iter().enumerate() {
        if mp_idx < base_map_point_count {
            if let Some(shared_mp) = shared_map.map_points_mut().get_mut(mp_idx) {
                shared_mp.position = updated_mp.position;
                shared_mp.culled = updated_mp.culled;
            }
            continue;
        }

        if mp_idx >= shared_map.map_points().len() {
            shared_map.push_map_point(updated_mp.clone());
        } else if let Some(shared_mp) = shared_map.map_points_mut().get_mut(mp_idx) {
            *shared_mp = updated_mp.clone();
        }
    }

    for working_kf in working_map.keyframes() {
        let Some(shared_kf) = shared_map
            .keyframes_mut()
            .iter_mut()
            .find(|kf| kf.frame.idx == working_kf.frame.idx)
        else {
            continue;
        };
        shared_kf.frame.pose_world_to_cam = working_kf.frame.pose_world_to_cam;
        shared_kf.map_point_by_desc_idx = working_kf.map_point_by_desc_idx.clone();
    }
}
