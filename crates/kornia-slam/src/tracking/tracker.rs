//! Frame-to-map pose estimation and the optical-flow track continuity that
//! seeds it.
//!
//! [`Tracker`] owns the estimator and the track set. The system owns the map,
//! the runtime state and the mode transitions, and supplies the predicted pose
//! each frame; the tracker reports an estimate or a rejection and keeps the
//! track set consistent with whichever happened.

use std::collections::HashMap;
use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::OrbFeatures;

use crate::Frame;
use crate::frame::RigFeatures;
use crate::mapping::Map;
use crate::pose_conversion::{mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64};
use crate::rig::RigGeometry;
use crate::system::TrackRefill;
use crate::tracking::flow_tracker::{
    FlowOutput, FlowStepCounts, FlowStepInput, RigFlow, rotated_ray_pixel,
};
use crate::tracking::local_map::{LocalMapRules, LocalMapSelectionConfig};
use crate::tracking::optical_flow::{
    FlowPrediction, FlowSurvivor, KltPyramidCache, KltTracker, LkKernel, MapKeypointMatch, Track,
    TrackId, TrackSet, carry_klt_survivors, snap_unique,
};
use crate::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use crate::tracking::pose_estimation::rig_pnp::{RigObservation, solve_rig_pnp};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};
use crate::tracking::pose_gate::{GateReference, GateVerdict, PoseGap, PoseGate};
use crate::tracking::relocalization::{
    Relocalization, RelocalizationConfig, RelocalizationSolver, relocalize,
};
use crate::tracking::timings::TrackingTimings;
use crate::upstream::kornia_3d::refine_analytic::refine_pose_analytic;

/// How far from an adopted landmark a refill track may be to take its map
/// point: the keypoint snap radius.
const ADOPTION_RADIUS_PX: f32 = 3.0;

/// A flow frame, its `(keypoint, map point)` seeds and each seed's world point.
pub(crate) type FlowFrame = (Frame, Vec<(usize, usize)>, Vec<kornia_algebra::Vec3F64>);

/// Everything the tracker needs about the current frame that it does not own.
pub(crate) struct FrameInput<'a> {
    pub frame: &'a Frame,
    /// Pose predicted by the motion model, used to seed the projection search.
    pub candidate_pose: Pose3d,
    /// Pose at the end of the previous frame.
    pub pose_before: Pose3d,
    pub current_keyframe_idx: Option<usize>,
    /// How long tracking has been failing, which widens the search.
    pub lost_for_sec: f64,
    /// Whether `candidate_pose` is an inertial prediction trustworthy enough
    /// to veto a weakly supported visual estimate.
    pub trusted_prediction: bool,
    /// What an accepted pose is measured against; `None` without a trusted
    /// inertial prediction.
    pub gate: Option<GateReference>,
}

/// A tracked pose measured against its [`GateReference`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GateReading {
    pub gap: PoseGap,
    pub inliers: usize,
    pub matches: usize,
    /// `None` when the gate is disabled.
    pub verdict: Option<GateVerdict>,
}

/// A predicted pose to start optical flow from, and the shallower search a
/// good start allows.
pub(crate) struct FlowGuess<'a> {
    pub map: &'a Map,
    /// Predicted world-to-camera pose of the current frame.
    pub pose: &'a Pose3d,
    pub camera: &'a PinholeCamera,
    pub max_level: usize,
    pub max_iter: usize,
    /// The primary camera's `R_cur_prev` from the gyro, for tracks without a
    /// map point (`GyroGuess::RotatedRay`).
    pub gyro_rotation: Option<kornia_algebra::Mat3F64>,
}

pub(crate) struct Tracker {
    estimator: MapProjectionEstimator,
    klt_tracker: KltTracker,
    klt_cache: KltPyramidCache,
    track_set: TrackSet,
    // Optical-flow failures dropped to "no seeds" instead of failing the frame.
    klt_errors: u64,
    snap_errors: u64,
    pose_gate: PoseGate,
    last_gap: Option<GateReading>,
    /// Projection search scale set from the IMU trust; `None` keeps
    /// [`MapProjectionConfig::search_scale_for`].
    search_scale_override: Option<f32>,
    relocalization: RelocalizationConfig,
    /// The rig whose every camera optical flow tracks (`klt_rig:
    /// AllCameras`); `None` tracks the primary camera only.
    rig: Option<Arc<RigGeometry>>,
    /// Pyramid caches of rig cameras 1 and up on the per-camera kernels.
    secondary_klt_caches: Vec<KltPyramidCache>,
    /// Track sets cleared by a duplicate map point in reconciliation.
    reconcile_clears: u64,
    /// `WindowedU8` flow tallies so far.
    flow_counts: FlowStepCounts,
    /// Refill corners of the latest flow step, spawned once the frame's
    /// tracks are settled.
    pending_spawns: Vec<(usize, [f32; 2])>,
    refill_spawns: u64,
    /// `(per_camera, min_spacing_px)` of `track_refill`, which also bounds
    /// the tracks ORB and adoption add on `WindowedU8`.
    track_cap: Option<(usize, f32)>,
}

impl Tracker {
    pub(crate) fn new(config: MapProjectionConfig) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(config),
            klt_tracker: KltTracker::default(),
            klt_cache: KltPyramidCache::new(),
            track_set: TrackSet::new(),
            klt_errors: 0,
            snap_errors: 0,
            pose_gate: PoseGate::Disabled,
            last_gap: None,
            search_scale_override: None,
            relocalization: RelocalizationConfig::default(),
            rig: None,
            secondary_klt_caches: Vec::new(),
            reconcile_clears: 0,
            flow_counts: FlowStepCounts::default(),
            pending_spawns: Vec::new(),
            refill_spawns: 0,
            track_cap: None,
        }
    }

    /// Caps the tracks per camera at `refill`'s budget and spacing.
    pub(crate) fn with_track_refill(mut self, refill: TrackRefill) -> Self {
        self.track_cap = match refill {
            TrackRefill::Off => None,
            TrackRefill::Enabled {
                per_camera,
                min_spacing_px,
                ..
            } => Some((per_camera, min_spacing_px)),
        };
        self
    }

    /// Tracks every camera of `rig` by optical flow, spreading new tracks
    /// over `flow_threads` shards.
    pub(crate) fn with_rig_flow(mut self, rig: Arc<RigGeometry>, flow_threads: usize) -> Self {
        let cameras = rig.views().len();
        self.secondary_klt_caches = (1..cameras).map(|_| KltPyramidCache::new()).collect();
        self.track_set = TrackSet::with_flow_threads(flow_threads, cameras);
        if self.windowed() {
            self.track_set = std::mem::take(&mut self.track_set).with_monotonic_ids();
        }
        self.rig = Some(rig);
        self
    }

    /// Whether flow runs on `WindowedU8` shards, which cache templates by
    /// track identity and predict from each track's stored position.
    fn windowed(&self) -> bool {
        self.klt_tracker.kernel() == LkKernel::WindowedU8
    }

    /// `WindowedU8` flow tallies so far, and the tracks refill spawned.
    pub(crate) fn flow_counts(&self) -> (FlowStepCounts, u64) {
        (self.flow_counts, self.refill_spawns)
    }

    /// Runs optical flow with `window`-pixel LK windows.
    pub(crate) fn with_lk_window(mut self, window: usize) -> Self {
        self.klt_tracker = self.klt_tracker.with_window(window);
        self
    }

    /// Whether today's kernels track every rig camera, each from its own
    /// previous image.
    pub(crate) fn flows_rig(&self) -> bool {
        self.rig.is_some() && self.klt_tracker.kernel() != LkKernel::WindowedU8
    }

    /// Track sets cleared by a duplicate map point in reconciliation so far.
    pub(crate) fn reconcile_clears(&self) -> u64 {
        self.reconcile_clears
    }

    /// Draws relocalization hypotheses with `solver`.
    pub(crate) fn with_relocalization_solver(mut self, solver: RelocalizationSolver) -> Self {
        self.relocalization.solver = solver;
        self
    }

    pub(crate) fn with_local_map_rules(mut self, rules: LocalMapRules) -> Self {
        self.estimator.set_local_map_rules(rules);
        self
    }

    /// Restricts the local map to keyframes at or after `window_start_kf`.
    pub(crate) fn set_window_start(&mut self, window_start_kf: Option<usize>) {
        self.estimator.set_window_start(window_start_kf);
    }

    /// The local-map selection the estimator searches with.
    pub(crate) fn local_map_config(&self) -> LocalMapSelectionConfig {
        self.estimator.local_map_config()
    }

    pub(crate) fn set_search_scale_override(&mut self, search_scale: Option<f32>) {
        self.search_scale_override = search_scale;
    }

    /// The map-projection settings tracking searches with.
    pub(crate) fn map_projection_config(&self) -> &MapProjectionConfig {
        self.estimator.config()
    }

    fn search_scale(&self, input: &FrameInput<'_>) -> f32 {
        self.search_scale_override.unwrap_or_else(|| {
            self.estimator
                .config()
                .search_scale_for(input.lost_for_sec, input.trusted_prediction)
        })
    }

    /// Rejects an accepted pose that disagrees with the inertial prediction.
    pub(crate) fn with_pose_gate(mut self, gate: PoseGate) -> Self {
        self.pose_gate = gate;
        self
    }

    pub(crate) fn pose_gate(&self) -> &PoseGate {
        &self.pose_gate
    }

    /// The measurement of the latest estimate that passed the inlier bar.
    pub(crate) fn last_gap(&self) -> Option<GateReading> {
        self.last_gap
    }

    /// Optical-flow tracking and keypoint-snap failures so far, in that order.
    pub(crate) fn flow_error_counts(&self) -> (u64, u64) {
        (self.klt_errors, self.snap_errors)
    }

    /// Runs optical flow on `kernel`.
    pub(crate) fn with_lk_kernel(mut self, kernel: LkKernel) -> Self {
        self.klt_tracker = self.klt_tracker.with_kernel(kernel);
        if self.windowed() {
            self.track_set = std::mem::take(&mut self.track_set).with_monotonic_ids();
        }
        self
    }

    /// Recovers the pose against one of the candidate keyframes, taking the
    /// first that `accept` agrees to; only then are the optical-flow tracks,
    /// which belonged to the lost pose, dropped.
    pub(crate) fn relocalize(
        &mut self,
        frame: &Frame,
        candidates: &[usize],
        map: &Map,
        camera: &PinholeCamera,
        accept: impl FnMut(&Relocalization) -> bool,
    ) -> Option<Relocalization> {
        let found = relocalize(
            frame,
            candidates,
            map,
            camera,
            &self.estimator,
            &self.relocalization,
            accept,
        );
        if found.is_some() {
            self.reset_tracks();
        }
        found
    }

    /// The optical-flow tracks carried into the next frame.
    pub(crate) fn tracks(&self) -> &[Track] {
        self.track_set.tracks()
    }

    /// Drops optical-flow continuity. Used when the map's world frame changes
    /// under the tracks, which invalidates them.
    pub(crate) fn reset_tracks(&mut self) {
        self.track_set.clear();
    }

    /// Follows the tracks from `previous` into `current` by optical flow,
    /// starting from `guess`'s predicted pixels when given; `None` without
    /// tracks, a previous image, or a flow solution.
    pub(crate) fn flow(
        &mut self,
        previous: Option<&Image<u8, 1>>,
        current: &Image<u8, 1>,
        guess: Option<FlowGuess<'_>>,
        timings: &mut TrackingTimings,
    ) -> Option<Vec<FlowSurvivor>> {
        if self.track_set.is_empty() {
            return None;
        }
        let initial_pixels = guess.as_ref().map(|guess| self.predicted_pixels(guess));
        let prediction =
            guess
                .as_ref()
                .zip(initial_pixels.as_deref())
                .map(|(guess, initial_pixels)| FlowPrediction {
                    initial_pixels,
                    max_level: guess.max_level,
                    max_iter: guess.max_iter,
                });
        let tracked = self.klt_tracker.track_cached(
            &mut self.klt_cache,
            self.track_set.tracks(),
            prediction,
            previous?,
            current,
            timings,
        );
        self.klt_errors += u64::from(tracked.is_err());
        tracked.ok()
    }

    /// [`Self::flow`] on every rig camera, each from its own previous image
    /// with today's kernels: `previous[c]` and `current[c]` are camera `c`'s.
    /// Survivors come out in camera order.
    pub(crate) fn flow_cameras(
        &mut self,
        previous: &[&Image<u8, 1>],
        current: &[&Image<u8, 1>],
        guess: Option<FlowGuess<'_>>,
        timings: &mut TrackingTimings,
    ) -> Option<Vec<FlowSurvivor>> {
        if self.track_set.is_empty() || previous.len() < current.len() {
            return None;
        }
        let rig = self.rig.clone()?;
        let mut survivors = Vec::with_capacity(self.track_set.len());
        for (camera, (&previous, &current)) in previous.iter().zip(current).enumerate() {
            let tracks: Vec<Track> = self
                .track_set
                .tracks()
                .iter()
                .filter(|track| track.camera() == camera)
                .copied()
                .collect();
            if tracks.is_empty() {
                continue;
            }
            let view = rig.view(camera)?;
            let initial_pixels = guess.as_ref().map(|guess| {
                // R_cur_prev(c) = R_cp R_cur_prev(0) R_pc.
                let r_cp = view.camera_from_primary.rotation;
                let rotation = guess
                    .gyro_rotation
                    .map(|rotation| r_cp * rotation * r_cp.transpose());
                predicted_pixels(
                    &tracks,
                    guess.map,
                    &view.pose(guess.pose),
                    &view.camera,
                    rotation,
                )
            });
            let prediction =
                guess
                    .as_ref()
                    .zip(initial_pixels.as_deref())
                    .map(|(guess, initial_pixels)| FlowPrediction {
                        initial_pixels,
                        max_level: guess.max_level,
                        max_iter: guess.max_iter,
                    });
            let cache = match camera {
                0 => &mut self.klt_cache,
                c => self.secondary_klt_caches.get_mut(c - 1)?,
            };
            let tracked = self
                .klt_tracker
                .track_cached(cache, &tracks, prediction, previous, current, timings);
            self.klt_errors += u64::from(tracked.is_err());
            survivors.extend(tracked.ok()?);
        }
        Some(survivors)
    }

    /// [`Self::flow`] on `WindowedU8` shards: every camera's tracks from
    /// `flow`'s previous frames into its current ones.
    pub(crate) fn flow_windowed(
        &mut self,
        flow: &mut RigFlow,
        input: &FlowStepInput,
        timings: &mut TrackingTimings,
    ) -> Option<Vec<FlowSurvivor>> {
        self.pending_spawns.clear();
        if self.track_set.is_empty() {
            return None;
        }
        match flow.step(&self.track_set, input, timings) {
            Ok(output) => {
                let counts = &mut self.flow_counts;
                counts.tracks_tried += output.counts.tracks_tried;
                counts.fb_rejects += output.counts.fb_rejects;
                counts.retries += output.counts.retries;
                counts.template_rebuilds += output.counts.template_rebuilds;
                self.pending_spawns.extend_from_slice(&output.spawned);
                Some(output.survivors.clone())
            }
            Err(_) => {
                self.klt_errors += 1;
                None
            }
        }
    }

    /// Spawns the latest flow step's refill corners, once this frame's
    /// tracks are settled; they join the map when a keyframe's landmark is
    /// adopted next to them.
    pub(crate) fn spawn_refills(&mut self) {
        for &(camera, pixel) in &self.pending_spawns {
            self.track_set.spawn_on(pixel, camera);
        }
        self.refill_spawns += self.pending_spawns.len() as u64;
        self.pending_spawns.clear();
    }

    /// Gives map point `map_point_idx` at `position`, seen at `pixel` in
    /// `camera`, to the nearest refill track, or spawns a track on it
    /// (`adopt_landmarks`).
    pub(crate) fn adopt(
        &mut self,
        pixel: [f32; 2],
        camera: usize,
        map_point_idx: usize,
        position: kornia_algebra::Vec3F64,
    ) {
        if self.track_set.follows(map_point_idx)
            || self.track_set.associate_nearest(
                pixel,
                camera,
                ADOPTION_RADIUS_PX,
                map_point_idx,
                position,
            )
        {
            return;
        }
        self.track_set
            .spawn_associated(pixel, camera, map_point_idx, position);
    }

    /// Drops the youngest tracks of a camera past the refill budget or
    /// closer than its spacing to an older one (`WindowedU8` only).
    pub(crate) fn thin_tracks(&mut self) {
        if let (true, Some((per_camera, min_spacing_px))) = (self.windowed(), self.track_cap) {
            self.track_set.thin(per_camera, min_spacing_px);
        }
    }

    /// Rewrites every track's map-point position after a map change; returns
    /// how many it rewrote. Only `WindowedU8` shards read them.
    pub(crate) fn refresh_positions(&mut self, map: &Map) -> u64 {
        if !self.windowed() {
            return 0;
        }
        self.track_set.refresh_positions(map)
    }

    /// Where each track should appear at `guess.pose`; see [`predicted_pixels`].
    fn predicted_pixels(&self, guess: &FlowGuess<'_>) -> Vec<[f32; 2]> {
        predicted_pixels(
            self.track_set.tracks(),
            guess.map,
            guess.pose,
            guess.camera,
            guess.gyro_rotation,
        )
    }

    /// A frame made of the survivors that carry a map point: each keypoint
    /// is the survivor's pixel with its map point's descriptor. With rig flow
    /// the keypoints are grouped by camera into the frame's camera ranges, so
    /// the pose solve sees every camera's rays; otherwise primary camera only.
    /// Returns it with the `(map point, keypoint)` seeds, or `None` when fewer
    /// than `min_tracks` survivors carry a map point.
    pub(crate) fn flow_frame(
        &self,
        idx: usize,
        survivors: &[FlowSurvivor],
        map: &Map,
        image_size: ImageSize,
        min_tracks: usize,
    ) -> Option<(Frame, Vec<(usize, usize)>)> {
        let tracks: HashMap<TrackId, (usize, usize)> = self
            .track_set
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id(), (track.map_point_idx()?, track.camera()))))
            .collect();
        let mut features = OrbFeatures {
            keypoints_xy: Vec::with_capacity(survivors.len()),
            orientations: Vec::with_capacity(survivors.len()),
            descriptors: Vec::with_capacity(survivors.len()),
            octaves: Vec::with_capacity(survivors.len()),
        };
        let cameras = self.rig.as_ref().map_or(1, |rig| rig.views().len());
        let mut camera_ranges = Vec::with_capacity(cameras);
        let mut seeded = Vec::with_capacity(survivors.len());
        for camera in 0..cameras {
            let begin = features.keypoints_xy.len();
            for survivor in survivors {
                let Some(&(mp_idx, _)) = tracks
                    .get(&survivor.track_id)
                    .filter(|&&(_, track_camera)| track_camera == camera)
                else {
                    continue;
                };
                let Some(mp) = map.map_points().get(mp_idx).filter(|mp| !mp.culled) else {
                    continue;
                };
                seeded.push((mp_idx, features.keypoints_xy.len()));
                features.keypoints_xy.push(survivor.pixel);
                features.orientations.push(0.0);
                features.descriptors.push(mp.descriptor);
                features.octaves.push(0);
            }
            camera_ranges.push(begin..features.keypoints_xy.len());
        }
        if seeded.len() < min_tracks {
            return None;
        }
        let n = seeded.len();
        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
            rig: self.flow_rig_features(camera_ranges),
        };
        Some((frame, seeded))
    }

    /// The rig part of a flow frame: every camera's range under rig flow,
    /// nothing otherwise (a primary-only frame is mono to the solver).
    fn flow_rig_features(&self, camera_ranges: Vec<std::ops::Range<usize>>) -> RigFeatures {
        match &self.rig {
            Some(rig) => RigFeatures {
                geometry: Some(Arc::clone(rig)),
                camera_ranges,
                left_to_right: Vec::new(),
                stereo_points: Vec::new(),
            },
            None => RigFeatures::default(),
        }
    }

    /// [`Self::flow_frame`] from a `WindowedU8` step's `output` and each
    /// track's stored position, without the map: the seeds' world points
    /// come back with the frame, whose keypoints are already undistorted
    /// and carry no descriptors (an ORB or keyframe path never sees it).
    pub(crate) fn flow_frame_windowed(
        &self,
        idx: usize,
        output: &FlowOutput,
        image_size: ImageSize,
        min_tracks: usize,
    ) -> Option<FlowFrame> {
        let tracks: HashMap<TrackId, (usize, kornia_algebra::Vec3F64)> = self
            .track_set
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id(), (track.map_point_idx()?, track.position()?))))
            .collect();
        let n = output.survivors.len();
        let mut features = OrbFeatures {
            keypoints_xy: Vec::with_capacity(n),
            orientations: Vec::with_capacity(n),
            descriptors: Vec::with_capacity(n),
            octaves: Vec::with_capacity(n),
        };
        let mut keypoints_undist = Vec::with_capacity(n);
        let mut points = Vec::with_capacity(n);
        let mut seeded = Vec::with_capacity(n);
        let cameras = self.rig.as_ref().map_or(1, |rig| rig.views().len());
        let mut camera_ranges: Vec<std::ops::Range<usize>> = (0..cameras).map(|_| 0..0).collect();
        // The output is in camera order already.
        for ((survivor, &camera), undistorted) in output
            .survivors
            .iter()
            .zip(&output.cameras)
            .zip(&output.undistorted)
        {
            let (Some(&(mp_idx, position)), Some(range)) = (
                tracks.get(&survivor.track_id),
                camera_ranges.get_mut(camera),
            ) else {
                continue;
            };
            let kp_idx = features.keypoints_xy.len();
            if (*range).is_empty() {
                *range = kp_idx..kp_idx;
            }
            range.end = kp_idx + 1;
            seeded.push((mp_idx, kp_idx));
            points.push(position);
            features.keypoints_xy.push(survivor.pixel);
            features.orientations.push(0.0);
            features.descriptors.push([0; 32]);
            features.octaves.push(0);
            keypoints_undist.push(*undistorted);
        }
        if seeded.len() < min_tracks {
            return None;
        }
        // Empty cameras sit where their neighbours end, so the ranges tile.
        let mut end = 0;
        for range in &mut camera_ranges {
            if (*range).is_empty() {
                *range = end..end;
            }
            end = range.end;
        }
        let n = seeded.len();
        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist,
            rig: self.flow_rig_features(camera_ranges),
        };
        Some((frame, seeded, points))
    }

    /// The pose of a flow frame by Gauss-Newton from the prediction over its
    /// seeds, no RANSAC (`FlowPose::AnalyticGn`): `solve_rig_pnp` over every
    /// camera on a rig, `refine_pose_analytic` on one camera. `points[k]` is
    /// seed `k`'s world point. Returns the estimate with its inliers from
    /// cameras 1 and up, or `None` below `min_inliers`, where RANSAC PnP
    /// takes over.
    pub(crate) fn estimate_flow_analytic(
        &mut self,
        input: FrameInput<'_>,
        seeded: Vec<(usize, usize)>,
        points: &[kornia_algebra::Vec3F64],
        camera: &PinholeCamera,
        max_iter: usize,
    ) -> Option<Result<(Estimate, usize), MapProjectionRejectReason>> {
        let frame = input.frame;
        let mut config = self.estimator.config().pnp.clone();
        config.lm_max_iterations = max_iter;
        // No RANSAC here: a Huber delta at the 2-dof chi-square 95 % level
        // (ORB-SLAM3's 2.45 px) and one outlier round bound what wrong
        // depths and moving objects can pull.
        config.robust_scale_sq = 5.991;
        config.outlier_rounds = 2;
        let mut observations = Vec::with_capacity(seeded.len());
        let mut pairs = Vec::with_capacity(seeded.len());
        for (&(mp_idx, kp_idx), &point_world) in seeded.iter().zip(points) {
            let Some(&pixel) = frame
                .keypoints_undist
                .get(kp_idx)
                .filter(|p| p[0].is_finite() && p[1].is_finite())
            else {
                continue;
            };
            let view = frame.secondary_view(kp_idx);
            observations.push(RigObservation {
                point_world,
                pixel: [pixel[0] as f64, pixel[1] as f64],
                camera: view.map_or(camera, |view| &view.camera),
                camera_from_primary: view.map(|view| &view.camera_from_primary),
                // Level-0 tracks: octave 0's unit information.
                sigma_px: 1.0,
            });
            pairs.push((mp_idx, kp_idx));
        }
        let start = input.candidate_pose;
        let pose = if frame.is_rig() {
            solve_rig_pnp(&observations, &start, &config)?.0
        } else {
            refine_mono(&observations, &start, &config, max_iter)?
        };
        let final_sq = config.final_reproj_threshold_px.powi(2);
        let inlier = |o: &RigObservation<'_>| {
            let p_primary = pose.transform_point(&o.point_world);
            let p_camera = o
                .camera_from_primary
                .map_or(p_primary, |t| t.transform_point(&p_primary));
            o.camera
                .reprojection_error_sq_cam(&p_camera, o.pixel[0], o.pixel[1])
                .is_some_and(|e| e <= final_sq)
        };
        // Only inliers go on as matches, so the outliers' tracks end here.
        let matches: Vec<(usize, usize)> = observations
            .iter()
            .zip(&pairs)
            .filter(|(o, _)| inlier(o))
            .map(|(_, &pair)| pair)
            .collect();
        let inliers = matches.len();
        if inliers < config.min_inliers {
            return None;
        }
        let secondary = observations
            .iter()
            .filter(|o| o.camera_from_primary.is_some() && inlier(o))
            .count();
        let estimate = Estimate {
            pose,
            matches,
            inliers,
            withheld: Vec::new(),
            most_voted_keyframe: None,
        };
        self.last_gap = None;
        Some(
            self.accept(&input, estimate)
                .map(|estimate| (estimate, secondary)),
        )
    }

    /// The pose of a [`Self::flow_frame`] from its seeds alone: consistency
    /// filter and PnP, no projection search or refinement. The track set is
    /// left untouched; [`Self::accept_flow_only`] commits it.
    pub(crate) fn estimate_flow_only(
        &mut self,
        input: FrameInput<'_>,
        seeded: Vec<(usize, usize)>,
        map: &Map,
        camera: &PinholeCamera,
        timings: &mut TrackingTimings,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let search_scale = self.search_scale(&input);
        let estimate = self.estimator.estimate_from_seeds(
            input.frame,
            &input.candidate_pose,
            map,
            camera,
            input.current_keyframe_idx,
            search_scale,
            seeded,
            timings,
        );
        self.last_gap = None;
        estimate.and_then(|estimate| self.accept(&input, estimate))
    }

    /// Today's inlier bar, then the measurement against the gate reference.
    fn accept(
        &mut self,
        input: &FrameInput<'_>,
        estimate: Estimate,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        // With a trusted inertial prediction the threshold drops to
        // ORB-SLAM3's inertial value; a solution below it is too weakly
        // supported to override the prediction and would teleport the pose.
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        if input.trusted_prediction && estimate.inliers < min_inliers {
            debug_eprintln!(
                "[proj] frame={} trusted_gate inliers={}",
                input.frame.idx,
                estimate.inliers
            );
            return Err(MapProjectionRejectReason::LowPnpInliers);
        }
        if let Some(reference) = &input.gate {
            let gap = PoseGap::measure(reference, &input.pose_before, &estimate.pose);
            let verdict = self
                .pose_gate
                .is_enabled()
                .then(|| self.pose_gate.judge(reference, &gap, &estimate.pose));
            self.last_gap = Some(GateReading {
                gap,
                inliers: estimate.inliers,
                matches: estimate.matches.len(),
                verdict,
            });
            match verdict {
                Some(GateVerdict::RotationGap) => {
                    return Err(MapProjectionRejectReason::RotationGap);
                }
                Some(GateVerdict::TranslationGap) => {
                    return Err(MapProjectionRejectReason::TranslationGap);
                }
                Some(GateVerdict::Accepted) | None => {}
            }
        }
        Ok(estimate)
    }

    /// Advances the tracks onto an accepted flow-only estimate of `frame`.
    /// On `WindowedU8` the tracks without a map point, refill's, are kept
    /// too, so they live until a keyframe can give them one.
    pub(crate) fn accept_flow_only(
        &mut self,
        estimate: &Estimate,
        frame: &Frame,
        survivors: &[FlowSurvivor],
    ) {
        if !self.windowed() {
            self.reconcile(&estimate.matches, frame);
            return;
        }
        let matched: std::collections::HashSet<usize> =
            estimate.matches.iter().map(|&(mp_idx, _)| mp_idx).collect();
        let keep: HashMap<TrackId, bool> = self
            .track_set
            .tracks()
            .iter()
            .map(|track| {
                let kept = track
                    .map_point_idx()
                    .is_none_or(|mp_idx| matched.contains(&mp_idx));
                (track.id(), kept)
            })
            .collect();
        let advanced: Vec<FlowSurvivor> = survivors
            .iter()
            .filter(|survivor| keep.get(&survivor.track_id) == Some(&true))
            .copied()
            .collect();
        if self.track_set.advance(advanced).is_err() {
            self.track_set.clear();
        }
    }

    /// Re-associates the tracks with `frame`'s accepted matches: the primary
    /// camera's, or every camera's under rig flow. Returns whether the set
    /// survived; it is cleared otherwise.
    fn reconcile(&mut self, matches: &[(usize, usize)], frame: &Frame) -> bool {
        let keypoints = match self.rig {
            Some(_) => &frame.features.keypoints_xy[..],
            None => &frame.features.keypoints_xy[frame.camera_range(0)],
        };
        let track_matches: Vec<MapKeypointMatch> = matches
            .iter()
            .filter(|&&(_, keypoint_idx)| keypoint_idx < keypoints.len())
            .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                map_point_idx,
                keypoint_idx,
            })
            .collect();
        let reconciled = match self.rig {
            Some(_) => self
                .track_set
                .reconcile_in_cameras(&track_matches, keypoints, |kp| frame.camera_of(kp)),
            None => self
                .track_set
                .reconcile_from_matches(&track_matches, keypoints),
        };
        if reconciled.is_err() {
            self.reconcile_clears += 1;
            self.track_set.clear();
        }
        reconciled.is_ok()
    }

    /// The flow survivors snapped to `frame`'s ORB keypoints of their own
    /// camera, as `(map point, keypoint)` seeds.
    fn snap_survivors(
        &mut self,
        survivors: &[FlowSurvivor],
        frame: &Frame,
    ) -> Option<Vec<(usize, usize)>> {
        let cameras = match self.rig {
            Some(_) => frame.rig.camera_ranges.len().max(1),
            None => 1,
        };
        let camera_of: HashMap<TrackId, usize> = self
            .track_set
            .tracks()
            .iter()
            .map(|track| (track.id(), track.camera()))
            .collect();
        let mut seeds = Vec::new();
        for camera in 0..cameras {
            let range = frame.camera_range(camera);
            let in_camera: Vec<FlowSurvivor> = if cameras == 1 {
                survivors.to_vec()
            } else {
                survivors
                    .iter()
                    .filter(|s| camera_of.get(&s.track_id) == Some(&camera))
                    .copied()
                    .collect()
            };
            let snapped = snap_unique(
                &self.track_set,
                &in_camera,
                &frame.features.keypoints_xy[range.clone()],
                3.0,
            );
            self.snap_errors += u64::from(snapped.is_err());
            seeds.extend(
                snapped
                    .ok()?
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, range.start + matched.keypoint_idx)),
            );
        }
        Some(seeds)
    }

    /// Estimates this frame's pose against the map, seeding the projection
    /// search from the flow `klt_survivors` of [`Self::flow`].
    ///
    /// The track set is left consistent either way: reconciled against the
    /// accepted matches on success, advanced onto the flow survivors on
    /// rejection, and cleared when neither can be applied.
    pub(crate) fn estimate(
        &mut self,
        input: FrameInput<'_>,
        klt_survivors: Option<Vec<FlowSurvivor>>,
        map: &Map,
        camera: &PinholeCamera,
        timings: &mut TrackingTimings,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let search_scale = self.search_scale(&input);

        // Without rig flow, optical flow follows the primary camera only; its
        // keypoints come first.
        let pre_seeded = klt_survivors
            .as_deref()
            .and_then(|survivors| self.snap_survivors(survivors, input.frame));

        let result = self.estimator.estimate_pose_timed(
            input.frame,
            &input.candidate_pose,
            &input.pose_before,
            map,
            camera,
            input.current_keyframe_idx,
            search_scale,
            pre_seeded,
            timings,
        );

        self.last_gap = None;
        let result = result.and_then(|estimate| self.accept(&input, estimate));
        let settled = self.settle(result, klt_survivors, input.frame);
        // Reconciliation can associate new map points.
        self.refresh_positions(map);
        settled
    }

    /// Reconciles the tracks with an accepted estimate, or advances them onto
    /// the flow survivors of a rejected one. On `WindowedU8` the refill
    /// tracks that survived flow outlive the reconcile, as on a flow-only
    /// frame, since no match can carry them.
    fn settle(
        &mut self,
        result: Result<Estimate, MapProjectionRejectReason>,
        klt_survivors: Option<Vec<FlowSurvivor>>,
        frame: &Frame,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        match result {
            Ok(estimate) => {
                let carried = match &klt_survivors {
                    Some(survivors) if self.windowed() => {
                        self.track_set.advanced_unassociated(survivors)
                    }
                    _ => Vec::new(),
                };
                if self.reconcile(&estimate.matches, frame) {
                    self.track_set.restore(carried);
                }
                self.thin_tracks();
                Ok(estimate)
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);
                Err(reason)
            }
        }
    }
}

/// One camera's pose by `refine_pose_analytic` from `start`, over the
/// observations within the prior gate.
fn refine_mono(
    observations: &[RigObservation<'_>],
    start: &Pose3d,
    config: &crate::tracking::pose_estimation::pnp::PnpConfig,
    max_iter: usize,
) -> Option<Pose3d> {
    let camera = observations.first()?.camera;
    let prior_sq = config.prior_reproj_threshold_px.powi(2);
    let mut points_world = Vec::with_capacity(observations.len());
    let mut points_image = Vec::with_capacity(observations.len());
    for o in observations {
        if camera
            .reprojection_error_sq_world(start, &o.point_world, o.pixel[0], o.pixel[1])
            .is_some_and(|e| e <= prior_sq)
        {
            points_world.push(vec3_to_f32(o.point_world));
            points_image.push(kornia_algebra::Vec2F32::new(
                o.pixel[0] as f32,
                o.pixel[1] as f32,
            ));
        }
    }
    if points_world.len() < config.min_correspondences {
        return None;
    }
    let params = kornia_3d::pnp::LMRefineParams {
        max_iterations: max_iter.max(1),
        robust: config.robust,
        robust_scale_sq: config.robust_scale_sq,
        ..kornia_3d::pnp::LMRefineParams::default()
    };
    let refined = refine_pose_analytic(
        &points_world,
        &points_image,
        &mat3_to_f32(camera.intrinsic_matrix()),
        &mat3_to_f32(start.rotation),
        &vec3_to_f32(start.translation),
        &params,
        config.refine_monotone,
    )?;
    Some(Pose3d::new(
        mat3_to_f64(refined.rotation),
        vec3_to_f64(refined.translation),
    ))
}

/// Where each of `tracks` should appear at `pose` (world to `camera`): its
/// map point's projection, carried back into raw pixels by the distortion
/// offset at the track's last pixel; without one, its last ray rotated by
/// `gyro_rotation` when given, else the last pixel itself.
fn predicted_pixels(
    tracks: &[Track],
    map: &Map,
    pose: &Pose3d,
    camera: &PinholeCamera,
    gyro_rotation: Option<kornia_algebra::Mat3F64>,
) -> Vec<[f32; 2]> {
    tracks
        .iter()
        .map(|track| {
            let raw = track.pixel();
            track
                .map_point_idx()
                .and_then(|idx| map.map_points().get(idx))
                .filter(|mp| !mp.culled)
                .and_then(|mp| project_raw(camera, &pose.transform_point(&mp.position), raw))
                .or_else(|| {
                    gyro_rotation.and_then(|rotation| rotated_ray_pixel(camera, &rotation, raw))
                })
                .unwrap_or(raw)
        })
        .collect()
}

/// `p_cam` projected through the ideal pinhole and carried back into raw
/// pixels by the distortion offset at `raw`, a nearby raw pixel.
pub(crate) fn project_raw(
    camera: &PinholeCamera,
    p_cam: &kornia_algebra::Vec3F64,
    raw: [f32; 2],
) -> Option<[f32; 2]> {
    let uv = camera.project_to_pixel(p_cam, 1e-6)?;
    let undistorted = camera.undistort(raw[0] as f64, raw[1] as f64)?;
    Some([
        (uv.x + raw[0] as f64 - undistorted.x) as f32,
        (uv.y + raw[1] as f64 - undistorted.y) as f32,
    ])
}

#[cfg(test)]
mod tests {
    use kornia_algebra::Vec3F64;

    use super::*;
    use crate::mapping::map::tests::test_frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey};

    /// A tracker whose tracks follow `n` map points, each with a distinct
    /// descriptor, plus one track without a map point.
    fn tracked_map(n: usize) -> (Tracker, Map) {
        let descriptors: Vec<[u8; 32]> = (0..n).map(|i| [i as u8; 32]).collect();
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, descriptors)))
            .unwrap();
        let mut tracker = Tracker::new(MapProjectionConfig::default());
        let mut matches = Vec::new();
        let mut keypoints = Vec::new();
        for feature in 0..n {
            let id = map
                .insert_landmark(LandmarkSeed {
                    position: Vec3F64::new(0.0, 0.0, 4.0),
                    color: [0; 3],
                    reference: ObservationKey {
                        keyframe_idx: 0,
                        feature_idx: feature,
                    },
                })
                .unwrap();
            matches.push(MapKeypointMatch {
                map_point_idx: id,
                keypoint_idx: feature,
            });
            keypoints.push([10.0 + feature as f32, 20.0]);
        }
        tracker
            .track_set
            .reconcile_from_matches(&matches, &keypoints)
            .unwrap();
        tracker.track_set.spawn([5.0, 5.0]);
        (tracker, map)
    }

    fn survivors_of(tracker: &Tracker) -> Vec<FlowSurvivor> {
        tracker
            .tracks()
            .iter()
            .map(|track| FlowSurvivor {
                track_id: track.id(),
                pixel: [track.pixel()[0] + 1.0, track.pixel()[1]],
                error: 0.0,
            })
            .collect()
    }

    #[test]
    fn flow_frame_carries_the_map_point_descriptors() {
        let (tracker, map) = tracked_map(5);
        let survivors = survivors_of(&tracker);
        let size = ImageSize {
            width: 640,
            height: 480,
        };

        let (frame, seeded) = tracker.flow_frame(7, &survivors, &map, size, 5).unwrap();

        assert_eq!(frame.idx, 7);
        assert_eq!(seeded.len(), 5);
        assert_eq!(frame.features.keypoints_xy.len(), 5);
        for &(mp_idx, kp_idx) in &seeded {
            assert_eq!(
                frame.features.descriptors[kp_idx],
                map.map_points()[mp_idx].descriptor
            );
            let track = tracker
                .tracks()
                .iter()
                .find(|t| t.map_point_idx() == Some(mp_idx))
                .unwrap();
            assert_eq!(
                frame.features.keypoints_xy[kp_idx],
                [track.pixel()[0] + 1.0, track.pixel()[1]]
            );
        }
        assert!(!frame.is_rig() && !frame.is_stereo());
    }

    fn gated(tracker: Tracker) -> Tracker {
        tracker.with_pose_gate(PoseGate::Enabled(
            crate::tracking::pose_gate::PoseGateLimits::default(),
        ))
    }

    /// A camera centred at `(x, 0, 0)`.
    fn centred_at(x: f64) -> Pose3d {
        Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-x, 0.0, 0.0),
        )
    }

    fn estimate_at(pose: Pose3d) -> Estimate {
        Estimate {
            pose,
            matches: (0..20).map(|i| (i, i)).collect(),
            inliers: 20,
            withheld: Vec::new(),
            most_voted_keyframe: None,
        }
    }

    fn input_at<'a>(frame: &'a Frame, candidate_pose: Pose3d) -> FrameInput<'a> {
        FrameInput {
            frame,
            candidate_pose,
            pose_before: Pose3d::IDENTITY,
            current_keyframe_idx: None,
            lost_for_sec: 0.0,
            trusted_prediction: true,
            gate: Some(GateReference {
                prediction: Pose3d::IDENTITY,
                dt_sec: 0.05,
                lost: None,
            }),
        }
    }

    #[test]
    fn a_gate_reject_carries_the_flow_survivors() {
        let (tracker, _map) = tracked_map(5);
        let mut tracker = gated(tracker);
        let frame = test_frame(1, vec![[0; 32]; 5]);
        let survivors = survivors_of(&tracker);
        let before: Vec<[f32; 2]> = tracker.tracks().iter().map(|t| t.pixel()).collect();

        let result = tracker.accept(
            &input_at(&frame, Pose3d::IDENTITY),
            estimate_at(centred_at(1095.0)),
        );
        assert_eq!(
            result.as_ref().err(),
            Some(&MapProjectionRejectReason::TranslationGap)
        );
        let reading = tracker.last_gap().unwrap();
        assert_eq!(reading.verdict, Some(GateVerdict::TranslationGap));
        assert!((reading.gap.trans_m - 1095.0).abs() < 1e-9);

        assert!(tracker.settle(result, Some(survivors), &frame).is_err());
        let after: Vec<[f32; 2]> = tracker.tracks().iter().map(|t| t.pixel()).collect();
        assert_eq!(after.len(), before.len());
        for (a, b) in after.iter().zip(&before) {
            assert_eq!(*a, [b[0] + 1.0, b[1]]);
        }
    }

    #[test]
    fn the_gate_measures_against_the_reference_not_the_search_seed() {
        // A KeyframeDue flow pose seeds the full search; the reference stays
        // the inertial prediction, so a pose at the seed is still rejected.
        let (tracker, _map) = tracked_map(5);
        let mut tracker = gated(tracker);
        let frame = test_frame(1, vec![[0; 32]; 5]);
        let seed = centred_at(3.0);
        let result = tracker.accept(&input_at(&frame, seed), estimate_at(seed));
        assert_eq!(
            result.err(),
            Some(MapProjectionRejectReason::TranslationGap)
        );
        let ok = tracker.accept(&input_at(&frame, seed), estimate_at(centred_at(0.1)));
        assert!(ok.is_ok());
        assert_eq!(
            tracker.last_gap().unwrap().verdict,
            Some(GateVerdict::Accepted)
        );
    }

    #[test]
    fn a_disabled_gate_measures_and_accepts() {
        let (mut tracker, _map) = tracked_map(5);
        let frame = test_frame(1, vec![[0; 32]; 5]);
        let far = estimate_at(centred_at(1095.0));
        assert!(
            tracker
                .accept(&input_at(&frame, Pose3d::IDENTITY), far)
                .is_ok()
        );
        assert_eq!(tracker.last_gap().unwrap().verdict, None);
        let mut input = input_at(&frame, Pose3d::IDENTITY);
        input.gate = None;
        tracker.last_gap = None;
        assert!(
            tracker
                .accept(&input, estimate_at(centred_at(1095.0)))
                .is_ok()
        );
        assert_eq!(
            tracker.last_gap(),
            None,
            "no reference without an inertial prediction"
        );
    }

    /// T2 (a): a refill track near an adopted landmark takes its map point
    /// and keeps its identity, so refill tracks can join the map.
    #[test]
    fn adoption_associates_the_nearest_refill_track() {
        let (mut tracker, _map) = tracked_map(5);
        let refill = *tracker.tracks().last().unwrap();
        assert_eq!(refill.map_point_idx(), None);
        let position = Vec3F64::new(1.0, 2.0, 5.0);
        let tracks_before = tracker.tracks().len();

        tracker.adopt([7.0, 6.0], 0, 99, position);

        assert_eq!(tracker.tracks().len(), tracks_before);
        let adopted = tracker
            .tracks()
            .iter()
            .find(|track| track.id() == refill.id())
            .unwrap();
        assert_eq!(adopted.map_point_idx(), Some(99));
        assert_eq!(adopted.position(), Some(position));
        assert_eq!(adopted.pixel(), refill.pixel());
    }

    /// T2 (b): on `WindowedU8` an ORB frame keeps the refill tracks that
    /// survived flow, as a flow-only frame does.
    #[test]
    fn an_orb_frame_keeps_the_refill_survivors_on_windowed_flow() {
        let (tracker, _map) = tracked_map(5);
        let mut tracker = tracker.with_lk_kernel(LkKernel::WindowedU8);
        let refill = *tracker.tracks().last().unwrap();
        let survivors = survivors_of(&tracker);
        let frame = test_frame(1, vec![[0; 32]; 5]);

        assert!(
            tracker
                .settle(Ok(estimate_at(Pose3d::IDENTITY)), Some(survivors), &frame)
                .is_ok()
        );

        let kept = tracker
            .tracks()
            .iter()
            .find(|track| track.id() == refill.id())
            .unwrap();
        assert_eq!(kept.pixel(), [refill.pixel()[0] + 1.0, refill.pixel()[1]]);
        assert_eq!(kept.age(), refill.age() + 1);
        assert_eq!(kept.map_point_idx(), None);
    }

    /// 80 exact observations and 16 coherent outliers 10 px off, all inside
    /// the 25 px prior gate, as a `FlowPose::AnalyticGn` frame sees them.
    fn analytic_flow_frame() -> (Frame, Vec<(usize, usize)>, Vec<Vec3F64>, PinholeCamera) {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut points = Vec::new();
        let mut pixels = Vec::new();
        for k in 0..96 {
            let (i, j) = (k % 12, k / 12);
            let point = Vec3F64::new(
                (i as f64 - 5.5) * 0.35,
                (j as f64 - 3.5) * 0.35,
                4.0 + ((i + j) % 3) as f64 * 0.5,
            );
            let uv = camera.project_to_pixel(&point, 1e-6).unwrap();
            let offset = if k % 6 == 5 { 10.0 } else { 0.0 };
            points.push(point);
            pixels.push([(uv.x + offset) as f32, uv.y as f32]);
        }
        let mut frame = test_frame(1, vec![[0; 32]; points.len()]);
        frame.keypoints_undist = pixels;
        let seeded = (0..points.len()).map(|k| (k, k)).collect();
        (frame, seeded, points, camera)
    }

    fn capped(tracker: Tracker) -> Tracker {
        tracker
            .with_lk_kernel(LkKernel::WindowedU8)
            .with_track_refill(TrackRefill::Enabled {
                per_camera: 3,
                min_spacing_px: 23.0,
                refill_below: 0.8,
                border_px: 9,
            })
    }

    /// T6: an ORB frame's reconcile adds one track per match; on
    /// `WindowedU8` the refill budget and spacing bound them.
    #[test]
    fn an_orb_frame_keeps_the_tracks_within_the_refill_budget() {
        let (tracker, _map) = tracked_map(5);
        let mut tracker = capped(tracker);
        let survivors = survivors_of(&tracker);
        let frame = test_frame(1, vec![[0; 32]; 5]);
        let estimate = Estimate {
            matches: (0..5).map(|i| (i, i)).collect(),
            ..estimate_at(Pose3d::IDENTITY)
        };
        // Keypoints 0-3 are within 4 px; 0, 4 and the refill track 30+ apart.
        let mut frame = frame;
        frame.features.keypoints_xy = vec![
            [40.0, 40.0],
            [42.0, 41.0],
            [41.0, 39.0],
            [43.0, 43.0],
            [80.0, 40.0],
        ];

        assert!(
            tracker
                .settle(Ok(estimate), Some(survivors), &frame)
                .is_ok()
        );

        let pixels: Vec<[f32; 2]> = tracker.tracks().iter().map(|t| t.pixel()).collect();
        assert!(pixels.len() <= 3, "{pixels:?}");
        for (i, a) in pixels.iter().enumerate() {
            for b in &pixels[i + 1..] {
                assert!((a[0] - b[0]).hypot(a[1] - b[1]) >= 23.0, "{pixels:?}");
            }
        }
    }

    /// T6: adoption adds a track per new landmark; thinning keeps the
    /// budget, oldest first.
    #[test]
    fn thinning_keeps_the_oldest_tracks_within_the_budget() {
        let mut tracker = capped(Tracker::new(MapProjectionConfig::default()));
        for k in 0..3 {
            tracker.track_set.spawn([30.0 + 40.0 * k as f32, 30.0]);
        }
        let survivors = survivors_of(&tracker)
            .into_iter()
            .map(|s| FlowSurvivor {
                pixel: [s.pixel[0] - 1.0, s.pixel[1]],
                ..s
            })
            .collect();
        tracker.track_set.advance(survivors).unwrap();
        let old: Vec<TrackId> = tracker.tracks().iter().map(|t| t.id()).collect();
        for k in 0..3 {
            tracker.adopt([30.0 + 40.0 * k as f32, 90.0], 0, 10 + k, Vec3F64::ZERO);
        }

        tracker.thin_tracks();

        let kept: Vec<TrackId> = tracker.tracks().iter().map(|t| t.id()).collect();
        assert_eq!(kept, old);
    }

    /// T4: the analytic flow pose keeps only its inliers as matches, and
    /// coherent outliers move it less than the 5 px Huber delta allowed
    /// (about 5 * 16 / 80 = 1 px at the inliers).
    #[test]
    fn the_analytic_flow_pose_drops_its_outliers() {
        let (frame, seeded, points, camera) = analytic_flow_frame();
        let mut tracker = Tracker::new(MapProjectionConfig::default());
        let mut input = input_at(&frame, Pose3d::IDENTITY);
        input.trusted_prediction = false;
        input.gate = None;

        let (estimate, _) = tracker
            .estimate_flow_analytic(input, seeded, &points, &camera, 3)
            .unwrap()
            .unwrap();

        let shift: f64 = (0..points.len())
            .filter(|kp| kp % 6 != 5)
            .map(|kp| {
                let uv = camera
                    .project_to_pixel(&estimate.pose.transform_point(&points[kp]), 1e-6)
                    .unwrap();
                uv.x - frame.keypoints_undist[kp][0] as f64
            })
            .sum::<f64>()
            / 80.0;
        assert!(shift.abs() < 0.7, "inlier shift {shift:.3} px");
        assert_eq!(estimate.matches.len(), 80);
        assert_eq!(estimate.inliers, 80);
        assert!(estimate.matches.iter().all(|&(_, kp)| kp % 6 != 5));
    }

    #[test]
    fn flow_frame_needs_min_tracks_with_a_map_point() {
        let (tracker, map) = tracked_map(5);
        let survivors = survivors_of(&tracker);
        let size = ImageSize {
            width: 640,
            height: 480,
        };

        // Six survivors, but only five carry a map point.
        assert!(tracker.flow_frame(0, &survivors, &map, size, 6).is_none());
    }
}
