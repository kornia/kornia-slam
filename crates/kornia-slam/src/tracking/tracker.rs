//! Frame-to-map pose estimation and the optical-flow track continuity that
//! seeds it.
//!
//! [`Tracker`] owns the estimator and the track set. The system owns the map,
//! the runtime state and the mode transitions, and supplies the predicted pose
//! each frame; the tracker reports an estimate or a rejection and keeps the
//! track set consistent with whichever happened.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::Image;

use crate::Frame;
use crate::mapping::Map;
use crate::tracking::optical_flow::{
    FlowPrediction, FlowSurvivor, KltPyramidCache, KltTracker, LkKernel, MapKeypointMatch, Track,
    TrackSet, carry_klt_survivors, snap_unique,
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
}

impl Tracker {
    pub(crate) fn new(config: MapProjectionConfig) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(config),
            klt_tracker: KltTracker::default(),
            klt_cache: KltPyramidCache::new(),
            track_set: TrackSet::new(),
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
        self.klt_tracker
            .track_cached(
                &mut self.klt_cache,
                self.track_set.tracks(),
                prediction,
                previous?,
                current,
                timings,
            )
            .ok()
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
        let search_scale = self
            .estimator
            .config()
            .search_scale_for(input.lost_for_sec, input.trusted_prediction);
        // Optical flow follows the primary camera only; its keypoints come first.
        let primary_keypoints = &input.frame.features.keypoints_xy[input.frame.camera_range(0)];
        let pre_seeded = klt_survivors
            .as_ref()
            .and_then(|survivors| {
                snap_unique(&self.track_set, survivors, primary_keypoints, 3.0).ok()
            })
            .map(|matches| {
                matches
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                    .collect()
            });

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

        // With a trusted inertial prediction the threshold drops to
        // ORB-SLAM3's inertial value; a solution below it is too weakly
        // supported to override the prediction and would teleport the pose.
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        let result = result.and_then(|estimate| {
            if input.trusted_prediction && estimate.inliers < min_inliers {
                Err(MapProjectionRejectReason::LowPnpInliers)
            } else {
                Ok(estimate)
            }
        });

        match result {
            Ok(estimate) => {
                let track_matches: Vec<MapKeypointMatch> = estimate
                    .matches
                    .iter()
                    .filter(|&&(_, keypoint_idx)| keypoint_idx < primary_keypoints.len())
                    .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                        map_point_idx,
                        keypoint_idx,
                    })
                    .collect();
                if self
                    .track_set
                    .reconcile_from_matches(&track_matches, primary_keypoints)
                    .is_err()
                {
                    self.track_set = TrackSet::new();
                }
                Ok(estimate)
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);
                Err(reason)
            }
        }
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
