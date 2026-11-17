//! Frame-to-map pose estimation and the optical-flow track continuity that
//! seeds it.
//!
//! [`Tracker`] owns the estimator and the track set. The system owns the map,
//! the runtime state and the mode transitions, and supplies the predicted pose
//! each frame; the tracker reports an estimate or a rejection and keeps the
//! track set consistent with whichever happened.

use std::collections::HashMap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::OrbFeatures;

use crate::Frame;
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
    // Optical-flow failures dropped to "no seeds" instead of failing the frame.
    klt_errors: u64,
    snap_errors: u64,
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
        }
    }

    /// Optical-flow tracking and keypoint-snap failures so far, in that order.
    pub(crate) fn flow_error_counts(&self) -> (u64, u64) {
        (self.klt_errors, self.snap_errors)
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

    /// The optical-flow tracks carried into the next frame.
    pub(crate) fn tracks(&self) -> &[Track] {
        self.track_set.tracks()
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

    /// Where each track should appear at `guess.pose`: its map point's
    /// projection, carried back into raw pixels by the distortion offset at
    /// the track's last pixel; the last pixel itself without a map point.
    fn predicted_pixels(&self, guess: &FlowGuess<'_>) -> Vec<[f32; 2]> {
        self.track_set
            .tracks()
            .iter()
            .map(|track| {
                let raw = track.pixel();
                track
                    .map_point_idx()
                    .and_then(|idx| guess.map.map_points().get(idx))
                    .filter(|mp| !mp.culled)
                    .and_then(|mp| {
                        let p_cam = guess.pose.transform_point(&mp.position);
                        guess.camera.project_to_pixel(&p_cam, 1e-6)
                    })
                    .and_then(|uv| {
                        let undistorted = guess.camera.undistort(raw[0] as f64, raw[1] as f64)?;
                        Some([
                            (uv.x + raw[0] as f64 - undistorted.x) as f32,
                            (uv.y + raw[1] as f64 - undistorted.y) as f32,
                        ])
                    })
                    .unwrap_or(raw)
            })
            .collect()
    }

    /// A frame made of the survivors that carry a map point: each keypoint
    /// is the survivor's pixel with its map point's descriptor, primary
    /// camera only. Returns it with the `(map point, keypoint)` seeds, or
    /// `None` when fewer than `min_tracks` survivors carry a map point.
    pub(crate) fn flow_frame(
        &self,
        idx: usize,
        survivors: &[FlowSurvivor],
        map: &Map,
        image_size: ImageSize,
        min_tracks: usize,
    ) -> Option<(Frame, Vec<(usize, usize)>)> {
        let map_points: HashMap<TrackId, usize> = self
            .track_set
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id(), track.map_point_idx()?)))
            .collect();
        let mut features = OrbFeatures {
            keypoints_xy: Vec::with_capacity(survivors.len()),
            orientations: Vec::with_capacity(survivors.len()),
            descriptors: Vec::with_capacity(survivors.len()),
            octaves: Vec::with_capacity(survivors.len()),
        };
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
            features.orientations.push(0.0);
            features.descriptors.push(mp.descriptor);
            features.octaves.push(0);
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
            rig: RigFeatures::default(),
        };
        Some((frame, seeded))
    }

    /// The pose of a [`Self::flow_frame`] from its seeds alone: consistency
    /// filter and PnP, no projection search or refinement. The track set is
    /// left untouched; [`Self::accept_flow_only`] commits it.
    pub(crate) fn estimate_flow_only(
        &self,
        input: FrameInput<'_>,
        seeded: Vec<(usize, usize)>,
        map: &Map,
        camera: &PinholeCamera,
        timings: &mut TrackingTimings,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let search_scale = self
            .estimator
            .config()
            .search_scale_for(input.lost_for_sec, input.trusted_prediction);
        let estimate = self.estimator.estimate_from_seeds(
            input.frame,
            &input.candidate_pose,
            map,
            camera,
            input.current_keyframe_idx,
            search_scale,
            seeded,
            timings,
        )?;
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        if input.trusted_prediction && estimate.inliers < min_inliers {
            return Err(MapProjectionRejectReason::LowPnpInliers);
        }
        Ok(estimate)
    }

    /// Advances the tracks onto an accepted flow-only estimate of `frame`.
    pub(crate) fn accept_flow_only(&mut self, estimate: &Estimate, frame: &Frame) {
        self.reconcile(&estimate.matches, &frame.features.keypoints_xy);
    }

    fn reconcile(&mut self, matches: &[(usize, usize)], primary_keypoints: &[[f32; 2]]) {
        let track_matches: Vec<MapKeypointMatch> = matches
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
        let search_scale = self
            .estimator
            .config()
            .search_scale_for(input.lost_for_sec, input.trusted_prediction);

        // Optical flow follows the primary camera only; its keypoints come first.
        let primary_keypoints = &input.frame.features.keypoints_xy[input.frame.camera_range(0)];
        let pre_seeded = klt_survivors
            .as_ref()
            .and_then(|survivors| {
                let snapped = snap_unique(&self.track_set, survivors, primary_keypoints, 3.0);
                self.snap_errors += u64::from(snapped.is_err());
                snapped.ok()
            })
            .map(|matches| {
                matches
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                    .collect()
            });

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

        // With a trusted inertial prediction the threshold drops to
        // ORB-SLAM3's inertial value; a solution below it is too weakly
        // supported to override the prediction and would teleport the pose.
        let min_inliers = self.estimator.config().min_inliers_with_prediction;
        let result = result.and_then(|estimate| {
            if input.trusted_prediction && estimate.inliers < min_inliers {
                debug_eprintln!(
                    "[proj] frame={} trusted_gate inliers={}",
                    input.frame.idx,
                    estimate.inliers
                );
                Err(MapProjectionRejectReason::LowPnpInliers)
            } else {
                Ok(estimate)
            }
        });

        match result {
            Ok(estimate) => {
                self.reconcile(&estimate.matches, primary_keypoints);
                Ok(estimate)
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);
                Err(reason)
            }
        }
    }
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
