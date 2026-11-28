//! Frame-to-map pose estimation and the optical-flow track continuity that
//! seeds it.
//!
//! [`Tracker`] owns the estimator and the track set. The system owns the map,
//! the runtime state and the mode transitions, and supplies the predicted pose
//! each frame; the tracker reports an estimate or a rejection and keeps the
//! track set consistent with whichever happened.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F32, Vec3F64};
use std::collections::HashMap;
use std::sync::Arc;

use kornia_image::{Image, ImageSize};

use crate::Frame;
use crate::features::Features;
use crate::frame::RigFeatures;
use crate::mapping::Map;
use crate::pose_conversion::{mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64};
use crate::rig::RigGeometry;
use crate::system::TrackRefill;
use crate::tracking::flow_tracker::{FlowSeeds, FlowStepInput, RigFlow, rotated_ray_pixel};
use crate::tracking::optical_flow::{
    FlowPrediction, FlowSurvivor, KltPyramidCache, KltTracker, LkKernel, MapKeypointMatch, Track,
    TrackId, TrackSet, carry_klt_survivors, snap_unique,
};
use crate::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use crate::tracking::pose_estimation::pnp::PnpConfig;
use crate::tracking::pose_estimation::rig_pnp::{RigObservation, solve_rig_pnp};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};
use crate::tracking::relocalization::{Relocalization, RelocalizationConfig, relocalize};
use crate::tracking::timings::TrackingTimings;

/// How far from an adopted landmark a refill track may be to take its map
/// point: the keypoint snap radius.
const ADOPTION_RADIUS_PX: f32 = 3.0;

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
    pub gyro_rotation: Option<Mat3F64>,
}

pub(crate) struct Tracker {
    estimator: MapProjectionEstimator,
    klt_tracker: KltTracker,
    klt_cache: KltPyramidCache,
    track_set: TrackSet,
    // Optical-flow and snapping failures dropped to "no seeds".
    klt_errors: u64,
    snap_errors: u64,
    followed_baseline: usize,
    /// The frame a flow-only frame is tracked on, refilled in place.
    flow_stand_in: Frame,
    /// The seeds of an analytic flow pose, refilled in place.
    flow_seeds: FlowSeeds,
    /// The rig whose every camera optical flow tracks (`klt_rig:
    /// AllCameras`); `None` tracks the primary camera only.
    rig: Option<Arc<RigGeometry>>,
    /// Pyramid caches of rig cameras 1 and up on the full-pyramid kernels.
    secondary_klt_caches: Vec<KltPyramidCache>,
    /// Refill corners of the latest `WindowedU8` step, spawned once the
    /// frame's tracks are settled.
    pending_spawns: Vec<(usize, [f32; 2])>,
    /// `(per_camera, min_spacing_px)` of `track_refill`, which also bounds
    /// the tracks ORB and adoption add on `WindowedU8`.
    track_cap: Option<(usize, f32)>,
    /// Buffers of a `WindowedU8` flow-only frame, reused so a warm one
    /// allocates nothing outside the pose solve: the step's survivors and
    /// the map points its pose kept, sorted.
    survivor_buffer: Vec<FlowSurvivor>,
    matched_points: Vec<usize>,
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
            followed_baseline: 0,
            flow_stand_in: Frame {
                idx: 0,
                features: Features {
                    keypoints_xy: Vec::new(),
                    orientations: Some(Vec::new()),
                    descriptors: Vec::new(),
                    octaves: Vec::new(),
                },
                pose_world_to_cam: Pose3d::IDENTITY,
                image_size: ImageSize {
                    width: 0,
                    height: 0,
                },
                keypoint_colors: Vec::new(),
                u_right: Vec::new(),
                depth: Vec::new(),
                keypoints_undist: Vec::new(),
                rig: RigFeatures::default(),
            },
            flow_seeds: FlowSeeds::default(),
            rig: None,
            secondary_klt_caches: Vec::new(),
            pending_spawns: Vec::new(),
            track_cap: None,
            survivor_buffer: Vec::new(),
            matched_points: Vec::new(),
        }
    }

    /// Runs optical flow on `kernel`.
    pub(crate) fn with_lk_kernel(mut self, kernel: LkKernel) -> Self {
        self.klt_tracker = self.klt_tracker.with_kernel(kernel);
        if self.windowed() {
            self.track_set = std::mem::take(&mut self.track_set).with_monotonic_ids();
        }
        self
    }

    /// Runs optical flow with `window`-pixel LK windows.
    pub(crate) fn with_lk_window(mut self, window: usize) -> Self {
        self.klt_tracker = self.klt_tracker.with_window(window);
        self
    }

    /// Caps the `WindowedU8` tracks per camera at `refill`'s budget and
    /// spacing.
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
        let cameras = rig.len();
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

    /// Whether the full-pyramid kernels track every rig camera, each from
    /// its own previous image.
    pub(crate) fn flows_rig(&self) -> bool {
        self.rig.is_some() && !self.windowed()
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
            &RelocalizationConfig::default(),
            accept,
        );
        if found.is_some() {
            self.reset_tracks();
        }
        found
    }

    /// Drops optical-flow continuity. Used when the map's world frame changes
    /// under the tracks, which invalidates them.
    pub(crate) fn reset_tracks(&mut self) {
        self.track_set.clear();
        self.followed_baseline = 0;
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
        let initial_pixels = guess.as_ref().map(|guess| {
            predicted_pixels(
                self.track_set.tracks(),
                guess.map,
                guess.pose,
                guess.camera,
                guess.gyro_rotation,
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
    /// on the full-pyramid kernels: `previous[c]` and `current[c]` are camera
    /// `c`'s. Survivors come out in camera order.
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
                    view.pinhole,
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
                self.pending_spawns.extend_from_slice(&output.spawned);
                let mut survivors = std::mem::take(&mut self.survivor_buffer);
                survivors.clear();
                survivors.extend_from_slice(&output.survivors);
                Some(survivors)
            }
            Err(_) => {
                self.klt_errors += 1;
                None
            }
        }
    }

    /// Takes back the survivors of a frame that did not consume them, for
    /// the next [`Self::flow_windowed`].
    pub(crate) fn recycle_survivors(&mut self, survivors: Option<Vec<FlowSurvivor>>) {
        if let Some(survivors) = survivors {
            self.survivor_buffer = survivors;
        }
    }

    /// Spawns the latest `WindowedU8` step's refill corners, once this
    /// frame's tracks are settled; they join the map when a keyframe's
    /// landmark is adopted next to them.
    pub(crate) fn spawn_refills(&mut self) {
        for &(camera, pixel) in &self.pending_spawns {
            self.track_set.spawn_on(pixel, camera);
        }
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
        position: Vec3F64,
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

    /// Ends a keyframe's adoption: the track budget, then the followed
    /// landmarks a flow frame's evidence is compared with.
    pub(crate) fn finish_adoption(&mut self) {
        self.thin_tracks();
        self.followed_baseline = self.followed_tracks();
    }

    /// Drops the youngest tracks of a camera past the refill budget or
    /// closer than its spacing to an older one (`WindowedU8` only).
    fn thin_tracks(&mut self) {
        if let (true, Some((per_camera, min_spacing_px))) = (self.windowed(), self.track_cap) {
            self.track_set.thin(per_camera, min_spacing_px);
        }
    }

    /// Rewrites every track's map-point position after a map change; only
    /// `WindowedU8` shards read them.
    pub(crate) fn refresh_positions(&mut self, map: &Map) {
        if self.windowed() {
            self.track_set.refresh_positions(map);
        }
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
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let search_scale = self.search_scale(&input);
        let pre_seeded = klt_survivors
            .as_deref()
            .and_then(|survivors| self.snap_survivors(survivors, input.frame));

        let result = self.estimator.estimate_pose(
            input.frame,
            &input.candidate_pose,
            &input.pose_before,
            map,
            camera,
            input.current_keyframe_idx,
            search_scale,
            pre_seeded,
        );
        let result = result.and_then(|estimate| self.accept(input.trusted_prediction, estimate));
        let settled = self.settle(result, klt_survivors, input.frame);
        // Reconciliation can associate new map points.
        self.refresh_positions(map);
        settled
    }

    /// The map-projection settings tracking searches with.
    pub(crate) fn map_projection_config(&self) -> &MapProjectionConfig {
        self.estimator.config()
    }

    fn search_scale(&self, input: &FrameInput<'_>) -> f32 {
        self.estimator
            .config()
            .search_scale_for(input.lost_for_sec, input.trusted_prediction)
    }

    /// With a trusted inertial prediction the threshold drops to ORB-SLAM3's
    /// inertial value; a solution below it is too weakly supported to
    /// override the prediction and would teleport the pose.
    fn accept(
        &self,
        trusted_prediction: bool,
        estimate: Estimate,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        if trusted_prediction && estimate.inliers < min_inliers {
            return Err(MapProjectionRejectReason::LowPnpInliers);
        }
        Ok(estimate)
    }

    /// The flow survivors snapped to `frame`'s ORB keypoints of their own
    /// camera, as `(map point, keypoint)` seeds. Without rig flow, optical
    /// flow follows the primary camera only; its keypoints come first.
    fn snap_survivors(
        &mut self,
        survivors: &[FlowSurvivor],
        frame: &Frame,
    ) -> Option<Vec<(usize, usize)>> {
        if self.rig.is_none() {
            let primary_keypoints = &frame.features.keypoints_xy[frame.camera_range(0)];
            let snapped = snap_unique(&self.track_set, survivors, primary_keypoints, 3.0);
            self.snap_errors += u64::from(snapped.is_err());
            return Some(
                snapped
                    .ok()?
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                    .collect(),
            );
        }
        let camera_of: HashMap<TrackId, usize> = self
            .track_set
            .tracks()
            .iter()
            .map(|track| (track.id(), track.camera()))
            .collect();
        let mut seeds = Vec::new();
        for camera in 0..frame.rig.camera_ranges.len().max(1) {
            let range = frame.camera_range(camera);
            let in_camera: Vec<FlowSurvivor> = survivors
                .iter()
                .filter(|s| camera_of.get(&s.track_id) == Some(&camera))
                .copied()
                .collect();
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

    /// Reconciles the tracks with an accepted ORB estimate, or advances them
    /// onto the flow survivors of a rejected one; then records how many
    /// tracks follow a landmark. On `WindowedU8` the refill tracks that
    /// survived flow outlive the reconcile, as on a flow-only frame, since no
    /// match can carry them.
    fn settle(
        &mut self,
        result: Result<Estimate, MapProjectionRejectReason>,
        klt_survivors: Option<Vec<FlowSurvivor>>,
        frame: &Frame,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        match &result {
            Ok(estimate) => {
                let carried = match &klt_survivors {
                    Some(survivors) if self.windowed() => {
                        self.track_set.advanced_unassociated(survivors)
                    }
                    _ => Vec::new(),
                };
                if reconcile(
                    &mut self.track_set,
                    self.rig.is_some(),
                    &estimate.matches,
                    frame,
                ) {
                    self.track_set.restore(carried);
                }
                self.thin_tracks();
            }
            Err(_) => carry_klt_survivors(&mut self.track_set, klt_survivors),
        }
        self.followed_baseline = self.followed_tracks();
        result
    }

    fn followed_tracks(&self) -> usize {
        self.track_set
            .tracks()
            .iter()
            .filter(|track| track.map_point_idx().is_some())
            .count()
    }

    /// Tracks that followed a landmark right after the last ORB frame settled:
    /// the reference a flow frame's inliers are compared with for a keyframe,
    /// since a flow frame can only find the landmarks its tracks follow.
    pub(crate) fn followed_baseline(&self) -> usize {
        self.followed_baseline
    }

    #[cfg(test)]
    pub(crate) fn set_followed_baseline(&mut self, followed: usize) {
        self.followed_baseline = followed;
    }

    /// Fills the flow stand-in frame from the survivors that carry a live map
    /// point: each keypoint is the survivor's pixel with its map point's
    /// descriptor, undistorted through `camera`. Returns the `(map point,
    /// keypoint)` seeds, or `None` when fewer than `min_tracks` survivors
    /// carry a map point.
    pub(crate) fn flow_frame(
        &mut self,
        idx: usize,
        survivors: &[FlowSurvivor],
        map: &Map,
        image_size: ImageSize,
        min_tracks: usize,
        camera: &PinholeCamera,
    ) -> Option<Vec<(usize, usize)>> {
        let map_points: HashMap<TrackId, (usize, usize)> = self
            .track_set
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id(), (track.map_point_idx()?, track.camera()))))
            .collect();
        let cameras = self.rig.as_ref().map_or(1, |rig| rig.len());
        let frame = &mut self.flow_stand_in;
        let features = &mut frame.features;
        features.keypoints_xy.clear();
        features.descriptors.clear();
        features.octaves.clear();
        let orientations = features.orientations.get_or_insert_with(Vec::new);
        orientations.clear();
        frame.rig.camera_ranges.clear();
        let mut seeded = Vec::with_capacity(survivors.len());
        // Grouped by camera into the frame's camera ranges under rig flow,
        // so the pose solve sees every camera's rays.
        for camera in 0..cameras {
            let begin = features.keypoints_xy.len();
            for survivor in survivors {
                let Some(&(mp_idx, _)) = map_points
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
                orientations.push(0.0);
                features.descriptors.push(mp.descriptor);
                features.octaves.push(0);
            }
            frame
                .rig
                .camera_ranges
                .push(begin..features.keypoints_xy.len());
        }
        if seeded.len() < min_tracks {
            return None;
        }
        // A primary-only frame is mono to the solver.
        match &self.rig {
            Some(rig) => frame.rig.geometry = Some(Arc::clone(rig)),
            None => {
                frame.rig.geometry = None;
                frame.rig.camera_ranges.clear();
            }
        }
        frame.idx = idx;
        frame.image_size = image_size;
        frame.keypoint_colors.clear();
        frame.keypoint_colors.resize(seeded.len(), [0; 3]);
        frame.keypoints_undist.clear();
        frame.ensure_undistorted(camera);
        Some(seeded)
    }

    /// The frame [`Self::flow_frame`] filled last. It holds no stereo depth
    /// and copied descriptors, so it never becomes a keyframe.
    pub(crate) fn flow_stand_in(&self) -> &Frame {
        &self.flow_stand_in
    }

    /// The seeds of an analytic flow pose: from a `WindowedU8` step's
    /// output and the tracks' stored positions without the map, or from the
    /// [`Self::flow_frame`] `seeded` and `map`. `None` below `min_tracks`.
    pub(crate) fn analytic_seeds(
        &mut self,
        source: AnalyticSeedSource<'_>,
        min_tracks: usize,
    ) -> Option<&FlowSeeds> {
        match source {
            AnalyticSeedSource::Windowed(flow) => {
                self.flow_seeds.assemble(&self.track_set, flow.output())
            }
            AnalyticSeedSource::Frame { seeded, map } => {
                self.flow_seeds
                    .assemble_from_frame(&self.flow_stand_in, seeded, map)
            }
        }
        (self.flow_seeds.len() >= min_tracks).then_some(&self.flow_seeds)
    }

    /// The seeds [`Self::analytic_seeds`] filled last.
    pub(crate) fn flow_seeds(&self) -> &FlowSeeds {
        &self.flow_seeds
    }

    /// The pose of the latest [`Self::analytic_seeds`] by Gauss-Newton from
    /// `candidate_pose`, no RANSAC (`FlowPose::AnalyticGn`): `solve_rig_pnp`
    /// over every camera under rig flow, `refine_pose_lm_analytic` on one
    /// camera. The matches are `(map point, keypoint)` inliers only, so the
    /// outliers' tracks end here. Returns the estimate with its inliers from
    /// cameras 1 and up, or `None` below `min_inliers`, where RANSAC PnP takes
    /// over.
    pub(crate) fn estimate_flow_analytic(
        &self,
        candidate_pose: &Pose3d,
        trusted_prediction: bool,
        camera: &PinholeCamera,
        max_iter: usize,
    ) -> Option<Result<(Estimate, usize), MapProjectionRejectReason>> {
        let seeds = &self.flow_seeds;
        let mut config = self.estimator.config().pnp.clone();
        config.lm_max_iterations = max_iter;
        // No RANSAC here: a Huber delta at the 2-dof chi-square 95 % level
        // (ORB-SLAM3's 2.45 px) and one outlier round bound what wrong
        // depths and moving objects can pull.
        config.robust_scale_sq = 5.991;
        config.outlier_rounds = 2;
        let rig = self.rig.as_deref();
        let mut observations = Vec::with_capacity(seeds.len());
        let mut pairs = Vec::with_capacity(seeds.len());
        for k in 0..seeds.len() {
            let pixel = seeds.undistorted[k];
            if !(pixel[0].is_finite() && pixel[1].is_finite()) {
                continue;
            }
            let view = rig
                .and_then(|rig| rig.view(seeds.cameras[k]))
                .filter(|_| seeds.cameras[k] > 0);
            observations.push(RigObservation {
                point_world: seeds.points[k],
                pixel: [pixel[0] as f64, pixel[1] as f64],
                camera: view.map_or(camera, |view| view.pinhole),
                camera_from_primary: view.map(|view| view.camera_from_primary),
            });
            pairs.push((seeds.map_points[k], seeds.keypoints[k]));
        }
        let pose = if rig.is_some() {
            solve_rig_pnp(&observations, candidate_pose, &config)?.0
        } else {
            refine_mono(&observations, candidate_pose, &config, max_iter)?
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
        };
        Some(
            self.accept(trusted_prediction, estimate)
                .map(|estimate| (estimate, secondary)),
        )
    }

    /// The pose of the [`Self::flow_frame`] in `input.frame` from its seeds
    /// alone: consistency filter and PnP, no projection search or refinement.
    /// The track set is left untouched; [`Self::accept_flow_only`] commits it.
    pub(crate) fn estimate_flow_only(
        &self,
        input: FrameInput<'_>,
        seeded: Vec<(usize, usize)>,
        map: &Map,
        camera: &PinholeCamera,
        timings: &mut TrackingTimings,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let estimate = self.estimator.estimate_from_seeds(
            input.frame,
            &input.candidate_pose,
            map,
            camera,
            input.current_keyframe_idx,
            self.search_scale(&input),
            seeded,
            timings,
        )?;
        self.accept(input.trusted_prediction, estimate)
    }

    /// Advances the tracks onto an accepted flow-only estimate: reconciled
    /// with the stand-in frame's matches, or on `WindowedU8` advanced onto
    /// `survivors` with the tracks whose map point the estimate kept and the
    /// tracks without one, refill's, so they live until a keyframe can give
    /// them one.
    pub(crate) fn accept_flow_only(&mut self, estimate: &Estimate, survivors: &[FlowSurvivor]) {
        if !self.windowed() {
            reconcile(
                &mut self.track_set,
                self.rig.is_some(),
                &estimate.matches,
                &self.flow_stand_in,
            );
            return;
        }
        let matched = &mut self.matched_points;
        matched.clear();
        matched.extend(estimate.matches.iter().map(|&(mp_idx, _)| mp_idx));
        matched.sort_unstable();
        let matched = &self.matched_points;
        let kept = self.track_set.advance_kept(survivors, |track| {
            track
                .map_point_idx()
                .is_none_or(|mp_idx| matched.binary_search(&mp_idx).is_ok())
        });
        if kept.is_err() {
            self.track_set.clear();
        }
    }

    /// The optical-flow tracks carried into the next frame.
    pub(crate) fn tracks(&self) -> &[Track] {
        self.track_set.tracks()
    }

    /// `(klt_errors, snap_errors)`: optical-flow and snapping failures so far,
    /// each dropped to "no seeds" instead of failing the frame.
    pub(crate) fn flow_errors(&self) -> (u64, u64) {
        (self.klt_errors, self.snap_errors)
    }
}

/// Where the analytic flow pose reads its seeds from.
pub(crate) enum AnalyticSeedSource<'a> {
    /// A `WindowedU8` step's output, with the tracks' stored positions.
    Windowed(&'a RigFlow),
    /// The stand-in frame's `(map point, keypoint)` seeds, with `map`.
    Frame {
        seeded: &'a [(usize, usize)],
        map: &'a Map,
    },
}

/// Re-associates `tracks` with `frame`'s accepted matches: the primary
/// camera's, or every camera's under rig flow. Returns whether the set
/// survived; it is cleared otherwise.
fn reconcile(
    tracks: &mut TrackSet,
    rig_flow: bool,
    matches: &[(usize, usize)],
    frame: &Frame,
) -> bool {
    let keypoints = if rig_flow {
        &frame.features.keypoints_xy[..]
    } else {
        &frame.features.keypoints_xy[frame.camera_range(0)]
    };
    let track_matches: Vec<MapKeypointMatch> = matches
        .iter()
        .filter(|&&(_, keypoint_idx)| keypoint_idx < keypoints.len())
        .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
            map_point_idx,
            keypoint_idx,
        })
        .collect();
    let reconciled = if rig_flow {
        tracks.reconcile_in_cameras(&track_matches, keypoints, |kp| frame.camera_of(kp))
    } else {
        tracks.reconcile_from_matches(&track_matches, keypoints)
    };
    if reconciled.is_err() {
        tracks.clear();
    }
    reconciled.is_ok()
}

/// One camera's pose by `refine_pose_lm_analytic` from `start`, over the
/// observations within the prior gate.
fn refine_mono(
    observations: &[RigObservation<'_>],
    start: &Pose3d,
    config: &PnpConfig,
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
            points_image.push(Vec2F32::new(o.pixel[0] as f32, o.pixel[1] as f32));
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
    let refined = kornia_3d::pnp::refine_pose_lm_analytic(
        &points_world,
        &points_image,
        &mat3_to_f32(camera.intrinsic_matrix()),
        &mat3_to_f32(start.rotation),
        &vec3_to_f32(start.translation),
        &params,
        false,
    )
    .ok()?;
    Some(Pose3d::new(
        mat3_to_f64(refined.rotation),
        vec3_to_f64(refined.translation),
    ))
}

/// Where each track is expected at `pose`: its live map point projected,
/// carried back into raw pixels by the distortion offset at the track's last
/// pixel; without one, its last ray rotated by `gyro_rotation` when given,
/// else its last pixel.
fn predicted_pixels(
    tracks: &[Track],
    map: &Map,
    pose: &Pose3d,
    camera: &PinholeCamera,
    gyro_rotation: Option<Mat3F64>,
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
    use super::*;
    use crate::mapping::map::tests::test_frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey};

    /// A tracker whose tracks follow `n` map points, plus one track without
    /// a map point.
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

    fn estimate_at(pose: Pose3d) -> Estimate {
        Estimate {
            pose,
            matches: (0..20).map(|i| (i, i)).collect(),
            inliers: 20,
        }
    }

    /// A refill track near an adopted landmark takes its map point and keeps
    /// its identity, so refill tracks can join the map.
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

    /// On `WindowedU8` an ORB frame keeps the refill tracks that survived
    /// flow, as a flow-only frame does.
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

    /// The full-pyramid kernels keep only the reconciled tracks, as before.
    #[test]
    fn an_orb_frame_drops_the_unassociated_tracks_on_the_default_kernel() {
        let (mut tracker, _map) = tracked_map(5);
        let refill = *tracker.tracks().last().unwrap();
        let survivors = survivors_of(&tracker);
        let frame = test_frame(1, vec![[0; 32]; 5]);

        assert!(
            tracker
                .settle(Ok(estimate_at(Pose3d::IDENTITY)), Some(survivors), &frame)
                .is_ok()
        );

        assert!(tracker.tracks().iter().all(|t| t.id() != refill.id()));
        assert!(tracker.tracks().iter().all(|t| t.map_point_idx().is_some()));
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

    /// An ORB frame's reconcile adds one track per match; on `WindowedU8`
    /// the refill budget and spacing bound them.
    #[test]
    fn an_orb_frame_keeps_the_tracks_within_the_refill_budget() {
        let (tracker, _map) = tracked_map(5);
        let mut tracker = capped(tracker);
        let survivors = survivors_of(&tracker);
        let mut frame = test_frame(1, vec![[0; 32]; 5]);
        let estimate = Estimate {
            matches: (0..5).map(|i| (i, i)).collect(),
            ..estimate_at(Pose3d::IDENTITY)
        };
        // Keypoints 0-3 are within 4 px; 0, 4 and the refill track 30+ apart.
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

    /// Adoption adds a track per new landmark; thinning keeps the budget,
    /// oldest first, and the followed baseline counts what is left.
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

        tracker.finish_adoption();

        let kept: Vec<TrackId> = tracker.tracks().iter().map(|t| t.id()).collect();
        assert_eq!(kept, old);
        assert_eq!(tracker.followed_baseline(), 0);
    }

    /// 80 exact observations and 16 coherent outliers 10 px off, all inside
    /// the 25 px prior gate, as an analytic flow frame sees them.
    fn analytic_seeds(tracker: &mut Tracker) -> (Vec<Vec3F64>, PinholeCamera) {
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
        let seeds = &mut tracker.flow_seeds;
        let mut points = Vec::new();
        for k in 0..96 {
            let (i, j) = (k % 12, k / 12);
            let point = Vec3F64::new(
                (i as f64 - 5.5) * 0.35,
                (j as f64 - 3.5) * 0.35,
                4.0 + ((i + j) % 3) as f64 * 0.5,
            );
            let uv = camera.project_to_pixel(&point, 1e-6).unwrap();
            let offset = if k % 6 == 5 { 10.0 } else { 0.0 };
            let pixel = [(uv.x + offset) as f32, uv.y as f32];
            seeds.pixels.push(pixel);
            seeds.undistorted.push(pixel);
            seeds.cameras.push(0);
            seeds.map_points.push(k);
            seeds.points.push(point);
            seeds.keypoints.push(k);
            points.push(point);
        }
        (points, camera)
    }

    /// The analytic flow pose keeps only its inliers as matches, and
    /// coherent outliers move it less than the Huber delta allows (about
    /// 2.45 * 16 / 80 = 0.5 px at the inliers).
    #[test]
    fn the_analytic_flow_pose_drops_its_outliers() {
        let mut tracker = Tracker::new(MapProjectionConfig::default());
        let (points, camera) = analytic_seeds(&mut tracker);

        let (estimate, secondary) = tracker
            .estimate_flow_analytic(&Pose3d::IDENTITY, false, &camera, 3)
            .unwrap()
            .unwrap();

        let shift: f64 = (0..points.len())
            .filter(|kp| kp % 6 != 5)
            .map(|kp| {
                let uv = camera
                    .project_to_pixel(&estimate.pose.transform_point(&points[kp]), 1e-6)
                    .unwrap();
                uv.x - tracker.flow_seeds.undistorted[kp][0] as f64
            })
            .sum::<f64>()
            / 80.0;
        assert!(shift.abs() < 0.7, "inlier shift {shift:.3} px");
        assert_eq!(estimate.matches.len(), 80);
        assert_eq!(estimate.inliers, 80);
        assert_eq!(secondary, 0);
        assert!(estimate.matches.iter().all(|&(_, kp)| kp % 6 != 5));
    }

    /// Below `min_inliers` the analytic pose gives way to RANSAC PnP.
    #[test]
    fn the_analytic_flow_pose_falls_back_below_min_inliers() {
        let mut tracker = Tracker::new(MapProjectionConfig::default());
        analytic_seeds(&mut tracker);
        let seeds = &mut tracker.flow_seeds;
        for list in [&mut seeds.pixels, &mut seeds.undistorted] {
            list.truncate(20);
        }
        seeds.cameras.truncate(20);
        seeds.map_points.truncate(20);
        seeds.points.truncate(20);
        seeds.keypoints.truncate(20);
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
        assert!(
            tracker
                .estimate_flow_analytic(&Pose3d::IDENTITY, false, &camera, 3)
                .is_none()
        );
    }
}
