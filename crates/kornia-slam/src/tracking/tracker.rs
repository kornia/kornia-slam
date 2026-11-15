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
use crate::mapping::map::ORB_SCALE_FACTOR;
use crate::tracking::optical_flow::{
    FlowSurvivor, KltTracker, MapKeypointMatch, TrackSet, carry_klt_survivors, snap_unique,
};
use crate::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use crate::tracking::pose_estimation::pose_inertial::{
    FrameNavState, NavPrior, PointObservation, PoseInertialProblem, solve_pose_inertial,
};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};
use crate::tracking::relocalization::{Relocalization, RelocalizationConfig, relocalize};

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
    /// Whether `candidate_pose` is an inertial prediction trustworthy enough
    /// to veto a weakly supported visual estimate.
    pub trusted_prediction: bool,
}

/// The IMU side of an inertial tracking step.
pub(crate) struct InertialInput<'a> {
    pub problem: PoseInertialProblem<'a>,
    /// State propagated from the anchor through the preintegration.
    pub predicted: FrameNavState,
}

/// An estimate whose pose, velocity and bias were solved with the IMU edge.
pub(crate) struct InertialEstimate {
    pub estimate: Estimate,
    pub state: FrameNavState,
    /// The next frame's prior, when the marginal was usable.
    pub prior: Option<NavPrior>,
    /// The IMU edge's navigation χ² at the solution.
    pub imu_chi2: f64,
}

/// What each stage of an inertial tracking attempt saw, for diagnostics.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct InertialAttempt {
    /// Local-map matches around the prediction, and the solve's inliers on them.
    pub local: SolveCounts,
    /// The same for the visual estimator's matches, when the first solve fell short.
    pub fallback: Option<SolveCounts>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SolveCounts {
    pub observations: usize,
    pub inliers: usize,
}

/// Optical-flow continuity into the current frame: the tracks that survived,
/// and those snapped onto keypoints as `(map_point_idx, keypoint_idx)` seeds.
struct FlowSeeds {
    survivors: Option<Vec<FlowSurvivor>>,
    matches: Option<Vec<(usize, usize)>>,
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
        let search_scale = self
            .estimator
            .config()
            .search_scale_for(input.lost_for_sec, input.trusted_prediction);
        let FlowSeeds {
            survivors: klt_survivors,
            matches: pre_seeded,
        } = self.flow_seeds(&input);

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
                self.keep_tracks(input.frame, &estimate.matches);
                Ok(estimate)
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);
                Err(reason)
            }
        }
    }

    /// Matches the local map around the inertial prediction and solves pose,
    /// velocity and bias jointly with the IMU edge (ORB-SLAM3's
    /// `TrackLocalMap` once the IMU is initialized). If those matches cannot
    /// support the solve, the visual estimator's matches get one more try.
    pub(crate) fn estimate_inertial(
        &mut self,
        input: FrameInput<'_>,
        inertial: &InertialInput<'_>,
        map: &Map,
        camera: &PinholeCamera,
    ) -> (
        Result<InertialEstimate, MapProjectionRejectReason>,
        InertialAttempt,
    ) {
        let mut attempt = InertialAttempt::default();
        let config = self.estimator.config();
        let search_scale = config.search_scale_for(input.lost_for_sec, true);
        let gates = SolveGates {
            min_inliers: config.min_inliers_with_prediction,
            // Wide enough for any match the widened projection search can return.
            initial_px: 2.0 * f64::from(config.projection.search_radius * search_scale),
        };
        let FlowSeeds {
            survivors: klt_survivors,
            matches: seeds,
        } = self.flow_seeds(&input);

        let matches = self.estimator.match_local_map(
            input.frame,
            &inertial.predicted.pose_cw,
            map,
            camera,
            input.current_keyframe_idx,
            search_scale,
            seeds.as_deref().unwrap_or_default(),
        );
        let (counts, solved) = solve_on_matches(
            inertial,
            input.frame,
            map,
            camera,
            &matches,
            inertial.predicted,
            gates,
        );
        attempt.local = counts;
        let solved = solved.or_else(|| {
            let visual = self
                .estimator
                .estimate_pose(
                    input.frame,
                    &input.candidate_pose,
                    &input.pose_before,
                    map,
                    camera,
                    input.current_keyframe_idx,
                    search_scale,
                    None,
                )
                .ok()?;
            let initial = FrameNavState {
                pose_cw: visual.pose,
                ..inertial.predicted
            };
            let (counts, solved) = solve_on_matches(
                inertial,
                input.frame,
                map,
                camera,
                &visual.matches,
                initial,
                gates,
            );
            attempt.fallback = Some(counts);
            solved
        });

        let result = match solved {
            Some(solved) => {
                self.keep_tracks(input.frame, &solved.estimate.matches);
                Ok(solved)
            }
            None => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);
                Err(MapProjectionRejectReason::LowPnpInliers)
            }
        };
        (result, attempt)
    }

    /// Flows the track set into the current image and snaps the survivors
    /// onto this frame's keypoints as seed matches.
    fn flow_seeds(&self, input: &FrameInput<'_>) -> FlowSeeds {
        let klt_survivors = if self.track_set.is_empty() {
            None
        } else {
            input.previous_image.and_then(|previous_image| {
                self.klt_tracker
                    .track(self.track_set.tracks(), previous_image, input.current_image)
                    .ok()
            })
        };
        let seeds = klt_survivors
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
        FlowSeeds {
            survivors: klt_survivors,
            matches: seeds,
        }
    }

    /// Re-seeds the track set from an accepted estimate's matches.
    fn keep_tracks(&mut self, frame: &Frame, matches: &[(usize, usize)]) {
        let track_matches: Vec<MapKeypointMatch> = matches
            .iter()
            .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                map_point_idx,
                keypoint_idx,
            })
            .collect();
        if self
            .track_set
            .reconcile_from_matches(&track_matches, &frame.features.keypoints_xy)
            .is_err()
        {
            self.track_set = TrackSet::new();
        }
    }
}

/// Acceptance thresholds shared by every inertial solve of one frame.
#[derive(Clone, Copy)]
struct SolveGates {
    min_inliers: usize,
    /// Reprojection error, in pixels, beyond which a match starts as an outlier.
    initial_px: f64,
}

/// Runs the inertial solve on `matches`, accepting it with at least
/// `gates.min_inliers` inliers; the estimate keeps only the inlier matches.
fn solve_on_matches(
    inertial: &InertialInput<'_>,
    frame: &Frame,
    map: &Map,
    camera: &PinholeCamera,
    matches: &[(usize, usize)],
    initial: FrameNavState,
    gates: SolveGates,
) -> (SolveCounts, Option<InertialEstimate>) {
    let (kept, observations): (Vec<(usize, usize)>, Vec<PointObservation>) = matches
        .iter()
        .filter_map(|&(map_point_idx, keypoint_idx)| {
            let point = map
                .map_points()
                .get(map_point_idx)
                .filter(|point| !point.culled)?;
            let pixel = frame.undistorted_xy(keypoint_idx, camera)?;
            let octave = frame.features.octaves.get(keypoint_idx).copied()?;
            let observation = PointObservation {
                point_w: point.position,
                pixel: [pixel[0] as f64, pixel[1] as f64],
                information: octave_information(octave),
            };
            Some(((map_point_idx, keypoint_idx), observation))
        })
        .unzip();
    let mut counts = SolveCounts {
        observations: observations.len(),
        inliers: 0,
    };
    if observations.len() < gates.min_inliers {
        return (counts, None);
    }

    let solution = solve_pose_inertial(&inertial.problem, initial, &observations, gates.initial_px);
    let inliers = solution.inlier_count();
    counts.inliers = inliers;
    if inliers < gates.min_inliers {
        return (counts, None);
    }
    let matches = kept
        .into_iter()
        .zip(&solution.inliers)
        .filter_map(|(matched, &inlier)| inlier.then_some(matched))
        .collect();
    let estimate = InertialEstimate {
        estimate: Estimate {
            pose: solution.state.pose_cw,
            matches,
            inliers,
        },
        state: solution.state,
        prior: solution.prior,
        imu_chi2: solution.imu_chi2,
    };
    (counts, Some(estimate))
}

/// Pixel information `1/σ²` of a keypoint detected at `octave`, with σ the
/// pyramid scale (ORB-SLAM3's `mvInvLevelSigma2`).
fn octave_information(octave: u8) -> f64 {
    ORB_SCALE_FACTOR.powi(-2 * i32::from(octave))
}
