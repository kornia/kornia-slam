//! Frame-to-map pose estimation and the optical-flow track continuity that
//! seeds it.
//!
//! [`Tracker`] owns the estimator and the track set. The system owns the map,
//! the runtime state and the mode transitions, and supplies the predicted pose
//! each frame; the tracker reports an estimate or a rejection and keeps the
//! track set consistent with whichever happened.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use std::collections::HashMap;

use kornia_image::{Image, ImageSize};

use crate::Frame;
use crate::features::Features;
use crate::frame::RigFeatures;
use crate::mapping::Map;
use crate::tracking::optical_flow::{
    FlowPrediction, FlowSurvivor, KltPyramidCache, KltTracker, LkKernel, MapKeypointMatch, Track,
    TrackId, TrackSet, carry_klt_survivors, snap_unique,
};
use crate::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};
use crate::tracking::relocalization::{Relocalization, RelocalizationConfig, relocalize};
use crate::tracking::timings::TrackingTimings;

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
        }
    }

    /// Runs optical flow on `kernel`.
    pub(crate) fn with_lk_kernel(mut self, kernel: LkKernel) -> Self {
        self.klt_tracker = self.klt_tracker.with_kernel(kernel);
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
        self.track_set = TrackSet::new();
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
            predicted_pixels(self.track_set.tracks(), guess.map, guess.pose, guess.camera)
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
        let result = result.and_then(|estimate| self.accept(&input, estimate));
        self.settle(result, klt_survivors, input.frame)
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
        input: &FrameInput<'_>,
        estimate: Estimate,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        if input.trusted_prediction && estimate.inliers < min_inliers {
            return Err(MapProjectionRejectReason::LowPnpInliers);
        }
        Ok(estimate)
    }

    /// The flow survivors snapped to `frame`'s primary-camera ORB keypoints,
    /// as `(map point, keypoint)` seeds. Optical flow follows the primary
    /// camera only; its keypoints come first.
    fn snap_survivors(
        &mut self,
        survivors: &[FlowSurvivor],
        frame: &Frame,
    ) -> Option<Vec<(usize, usize)>> {
        let primary_keypoints = &frame.features.keypoints_xy[frame.camera_range(0)];
        let snapped = snap_unique(&self.track_set, survivors, primary_keypoints, 3.0);
        self.snap_errors += u64::from(snapped.is_err());
        Some(
            snapped
                .ok()?
                .into_iter()
                .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                .collect(),
        )
    }

    /// Reconciles the tracks with an accepted ORB estimate, or advances them
    /// onto the flow survivors of a rejected one; then records how many
    /// tracks follow a landmark.
    fn settle(
        &mut self,
        result: Result<Estimate, MapProjectionRejectReason>,
        klt_survivors: Option<Vec<FlowSurvivor>>,
        frame: &Frame,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        match &result {
            Ok(estimate) => self.reconcile(&estimate.matches, frame),
            Err(_) => carry_klt_survivors(&mut self.track_set, klt_survivors),
        }
        self.followed_baseline = self.followed_tracks();
        result
    }

    /// Re-associates the tracks with `frame`'s accepted primary-camera
    /// matches; the set is cleared when they cannot be applied.
    fn reconcile(&mut self, matches: &[(usize, usize)], frame: &Frame) {
        let keypoints = &frame.features.keypoints_xy[frame.camera_range(0)];
        let track_matches: Vec<MapKeypointMatch> = matches
            .iter()
            .filter(|&&(_, keypoint_idx)| keypoint_idx < keypoints.len())
            .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                map_point_idx,
                keypoint_idx,
            })
            .collect();
        if self
            .track_set
            .reconcile_from_matches(&track_matches, keypoints)
            .is_err()
        {
            self.track_set = TrackSet::new();
        }
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
        let map_points: HashMap<TrackId, usize> = self
            .track_set
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id(), track.map_point_idx()?)))
            .collect();
        let frame = &mut self.flow_stand_in;
        let features = &mut frame.features;
        features.keypoints_xy.clear();
        features.descriptors.clear();
        features.octaves.clear();
        let orientations = features.orientations.get_or_insert_with(Vec::new);
        orientations.clear();
        let mut seeded = Vec::with_capacity(survivors.len());
        for survivor in survivors {
            let Some(&mp_idx) = map_points.get(&survivor.track_id) else {
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
        if seeded.len() < min_tracks {
            return None;
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
        self.accept(&input, estimate)
    }

    /// Advances the tracks onto an accepted flow-only estimate of the stand-in
    /// frame.
    pub(crate) fn accept_flow_only(&mut self, estimate: &Estimate) {
        let keypoints = &self.flow_stand_in.features.keypoints_xy;
        let track_matches: Vec<MapKeypointMatch> = estimate
            .matches
            .iter()
            .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                map_point_idx,
                keypoint_idx,
            })
            .collect();
        if self
            .track_set
            .reconcile_from_matches(&track_matches, keypoints)
            .is_err()
        {
            self.track_set = TrackSet::new();
        }
    }

    /// The optical-flow tracks carried into the next frame.
    #[cfg(test)]
    pub(crate) fn tracks(&self) -> &[Track] {
        self.track_set.tracks()
    }

    /// `(klt_errors, snap_errors)`: optical-flow and snapping failures so far,
    /// each dropped to "no seeds" instead of failing the frame.
    pub(crate) fn flow_errors(&self) -> (u64, u64) {
        (self.klt_errors, self.snap_errors)
    }
}

/// Where each track is expected at `pose`: its live map point projected, or
/// its last pixel for a track without one.
fn predicted_pixels(
    tracks: &[Track],
    map: &Map,
    pose: &Pose3d,
    camera: &PinholeCamera,
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
