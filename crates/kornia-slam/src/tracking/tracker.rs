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
    KltTracker, MapKeypointMatch, TrackSet, carry_klt_survivors, snap_unique,
};
use crate::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};

/// Everything the tracker needs about the current frame that it does not own.
pub(crate) struct FrameInput<'a> {
    pub frame: &'a Frame,
    pub previous_image: Option<&'a Image<u8, 1>>,
    pub current_image: &'a Image<u8, 1>,
    /// Pose predicted by the motion model, used to seed the projection search.
    pub candidate_pose: Pose3d,
    /// Pose at the end of the previous frame.
    pub pose_before: Pose3d,
    pub current_keyframe_idx: Option<usize>,
    /// How long tracking has been failing, which widens the search.
    pub lost_for_sec: f64,
}

pub(crate) struct Tracker {
    estimator: MapProjectionEstimator,
    klt_tracker: KltTracker,
    track_set: TrackSet,
}

impl Tracker {
    pub(crate) fn new(config: MapProjectionConfig) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(config),
            klt_tracker: KltTracker::default(),
            track_set: TrackSet::new(),
        }
    }

    /// Drops optical-flow continuity. Used when the map's world frame changes
    /// under the tracks, which invalidates them.
    pub(crate) fn reset_tracks(&mut self) {
        self.track_set = TrackSet::new();
    }

    /// Estimates this frame's pose against the map, seeding the projection
    /// search from optical flow where tracks survived.
    ///
    /// The track set is left consistent either way: reconciled against the
    /// accepted matches on success, advanced onto the flow survivors on
    /// rejection, and cleared when neither can be applied.
    pub(crate) fn estimate(
        &mut self,
        input: FrameInput<'_>,
        map: &Map,
        camera: &PinholeCamera,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let search_scale = self.estimator.config().search_scale_for(input.lost_for_sec);

        let klt_survivors = if self.track_set.is_empty() {
            None
        } else {
            input.previous_image.and_then(|previous_image| {
                self.klt_tracker
                    .track(self.track_set.tracks(), previous_image, input.current_image)
                    .ok()
            })
        };
        let pre_seeded = klt_survivors
            .as_ref()
            .and_then(|survivors| {
                snap_unique(
                    &self.track_set,
                    survivors,
                    &input.frame.features.keypoints_xy,
                    3.0,
                )
                .ok()
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

        match result {
            Ok(estimate) => {
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
                    .reconcile_from_matches(&track_matches, &input.frame.features.keypoints_xy)
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
