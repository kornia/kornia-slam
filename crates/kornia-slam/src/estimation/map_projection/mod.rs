//! Map-projection-based estimator: matching, PnP, and tracking flow.
//!
//! ```text
//!        Map Points (3D)
//!        *   *       *
//!         \  |      /
//!    project & match (ORB)
//!           \|/
//!    .---------------.
//!   /    * . *      /   current frame
//!  /       *       /    (2D keypoints)
//!  '---------------'
//!          |
//!      solve PnP
//!          |
//!     [pose_w2c]
//!          |
//!   refine with local map
//!          |
//! Estimated { pose, inliers, matches }
//! ```

mod keypoint_grid;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec2F32;
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;

use super::pnp::{self, PnpConfig};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::frame::Frame;
use crate::map::{Map, MapPoint};

use super::Estimate;
use keypoint_grid::KeypointGrid;

const ORB_SCALE_FACTOR: f64 = 1.2;
const ORB_N_LEVELS: usize = 8;
const VIEWING_COS_LIMIT: f64 = 0.5;

/// Tunable parameters for projection-guided matching.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionMatchConfig {
    /// Reject projected points with depth `<= min_depth`.
    pub min_depth: f64,
    /// Multiplier applied to the ORB-SLAM3-style search radius.
    pub search_radius: f32,
    /// Maximum Hamming distance to accept a descriptor match.
    pub max_hamming: u32,
}

impl Default for ProjectionMatchConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.0,
            search_radius: 1.0,
            max_hamming: 100,
        }
    }
}

/// Map-projection tracking thresholds.
#[derive(Debug, Clone)]
pub struct MapProjectionConfig {
    /// ORB descriptor matcher settings for tracking against reference observations.
    pub match_config: OrbMatchConfig,
    /// PnP pose-estimation thresholds.
    pub pnp: PnpConfig,
    /// Projection matching config for initial tracking.
    pub projection: ProjectionMatchConfig,
    /// Projection matching config for local-map refinement (wider search).
    pub local_projection: ProjectionMatchConfig,
}

impl Default for MapProjectionConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            pnp: PnpConfig::default(),
            projection: ProjectionMatchConfig::default(),
            local_projection: ProjectionMatchConfig {
                search_radius: 1.0,
                max_hamming: 100,
                ..ProjectionMatchConfig::default()
            },
        }
    }
}

/// Rejection reasons specific to the map-projection tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapProjectionRejectReason {
    /// Too few projection-guided matches were available to run the first PnP stage.
    LowProjectionMatches,
    /// PnP solving failed for the provided correspondences.
    PnpFailed,
    /// PnP solved, but too few final inliers remained after validation.
    LowPnpInliers,
    /// Too few descriptor matches were found against the current reference keyframe.
    LowReferenceMatches,
    /// Too few valid 3D-2D correspondences could be built from reference matches.
    LowReferenceCorrespondences,
}

/// Non-behavioral counters from one map-projection tracking attempt.
#[derive(Debug, Clone, Default)]
pub struct MapProjectionReport {
    /// Matches from the first projection-guided pass.
    pub projection_matches: usize,
    /// PnP inliers from the first projection-guided pass.
    pub projection_pnp_inliers: usize,
    /// Descriptor matches against the current reference keyframe.
    pub reference_matches: usize,
    /// Usable 3D-2D correspondences built from reference matches.
    pub reference_correspondences: usize,
    /// Matches from local-map projection refinement.
    pub local_projection_matches: usize,
    /// PnP inliers from local-map refinement.
    pub local_pnp_inliers: usize,
    /// Rejection reason if tracking failed.
    pub reject_reason: Option<MapProjectionRejectReason>,
}

/// Per-map-point projection/matching decisions for parity debugging.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionDebugFeatureCandidate {
    /// Keypoint index in the current debug frame.
    pub keypoint_idx: usize,
    /// Pyramid octave for this candidate keypoint.
    pub octave: usize,
    /// Whether the feature was already occupied before matching.
    pub occupied_before: bool,
    /// Hamming distance to the map-point descriptor.
    pub descriptor_dist: u32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectionDebugMatcherTrace {
    /// Search radius used for `GetFeaturesInArea` / grid query.
    pub search_radius: f32,
    /// Minimum accepted octave.
    pub min_level: isize,
    /// Maximum accepted octave.
    pub max_level: isize,
    /// Best candidate feature index.
    pub best_keypoint_idx: Option<usize>,
    /// Best descriptor distance.
    pub best_dist: Option<u32>,
    /// Best octave.
    pub best_level: Option<usize>,
    /// Second-best candidate feature index.
    pub second_best_keypoint_idx: Option<usize>,
    /// Second-best descriptor distance.
    pub second_best_dist: Option<u32>,
    /// Second-best octave.
    pub second_best_level: Option<usize>,
    /// Whether the ratio test rejected the best match.
    pub ratio_rejected: bool,
    /// Candidate features returned by the spatial query.
    pub feature_candidates: Vec<ProjectionDebugFeatureCandidate>,
}

/// Per-map-point projection/matching decisions for parity debugging.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionDebugPoint {
    /// Local map-point index in the slice passed to the matcher.
    pub map_point_idx: usize,
    /// Whether this map point projected into the current image under the
    /// configured visibility gate.
    pub visible: bool,
    /// Projected undistorted pixel if visible.
    pub projected_pixel: Option<[f32; 2]>,
    /// Predicted octave if the matcher computes one.
    pub predicted_octave: Option<usize>,
    /// Final matched keypoint index for this map point in the current frame.
    pub matched_keypoint_idx: Option<usize>,
    /// Matcher-side candidate list and best/second-best bookkeeping.
    pub matcher_trace: ProjectionDebugMatcherTrace,
}

/// Structured debug output for one projection-matching pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectionDebugFrame {
    /// Per-map-point projection and final match results.
    pub points: Vec<ProjectionDebugPoint>,
}

/// Debug-only override for the per-map-point projection state entering
/// `SearchByProjection`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionOverridePoint {
    /// Whether the map point should be considered visible.
    pub visible: bool,
    /// Projected undistorted pixel when visible.
    pub projected_pixel: Option<[f32; 2]>,
    /// Predicted octave when visible.
    pub predicted_octave: Option<usize>,
    /// Viewing cosine used to scale the search radius.
    pub view_cos: Option<f32>,
}

/// Map-projection-based pose estimator.
///
/// Estimates the camera pose by projecting map points into the current frame,
/// matching via ORB descriptors, and solving PnP.
pub struct MapProjectionEstimator {
    config: MapProjectionConfig,
}

impl MapProjectionEstimator {
    pub fn new(config: MapProjectionConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &MapProjectionConfig {
        &self.config
    }

    /// Run the same projection-matching path used by tracking, but return
    /// per-point visibility and match decisions for parity tooling.
    pub fn debug_match_map_to_frame(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        mp_filter: Option<&std::collections::HashSet<usize>>,
    ) -> ProjectionDebugFrame {
        self.match_map_to_frame_internal(map_points, frame, pose, camera, mp_filter, true)
            .3
            .expect("debug_match_map_to_frame must collect debug output")
    }

    /// Run projection matching with externally supplied projection/octave
    /// states. This is for parity tooling only and does not alter production
    /// tracking behavior.
    pub fn debug_match_map_to_frame_with_overrides(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        mp_filter: Option<&std::collections::HashSet<usize>>,
        overrides: &[ProjectionOverridePoint],
    ) -> ProjectionDebugFrame {
        self.match_map_to_frame_internal_with_overrides(
            map_points,
            frame,
            pose,
            camera,
            mp_filter,
            true,
            Some(overrides),
        )
        .3
        .expect("debug_match_map_to_frame_with_overrides must collect debug output")
    }

    /// Estimate the pose of `frame` against the map.
    pub fn estimate_pose(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        camera: &PinholeCamera,
        current_keyframe_idx: Option<usize>,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        self.estimate_pose_with_report(
            frame,
            candidate_pose,
            pose_before_tracking,
            map,
            camera,
            current_keyframe_idx,
        )
        .0
    }

    pub fn estimate_pose_with_report(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        camera: &PinholeCamera,
        current_keyframe_idx: Option<usize>,
    ) -> (
        Result<Estimate, MapProjectionRejectReason>,
        MapProjectionReport,
    ) {
        let pnp = &self.config.pnp;
        let mut report = MapProjectionReport::default();

        let (projection_matches, curr_keypoints_undist, grid) =
            self.match_map_to_frame(map.map_points(), frame, candidate_pose, camera, None);
        report.projection_matches = projection_matches.len();

        // Shared logic: try_track → refine_pose → Estimate, or propagate rejection.
        let try_track_and_refine = |correspondences: Vec<(usize, usize)>,
                                    pose_init: &Pose3d,
                                    report: &mut MapProjectionReport,
                                    is_projection_stage: bool|
         -> Result<Estimate, MapProjectionRejectReason> {
            let (mut pose, mut inliers) = self
                .solve_pnp(
                    map.map_points(),
                    &correspondences,
                    &curr_keypoints_undist,
                    &frame.features.scales,
                    camera,
                    pose_init,
                )
                .ok_or(MapProjectionRejectReason::PnpFailed)?;
            if is_projection_stage {
                report.projection_pnp_inliers = inliers;
            }
            if inliers < pnp.min_inliers_early {
                return Err(MapProjectionRejectReason::LowPnpInliers);
            }
            let mut matches = correspondences;
            if let Some(local) = self.refine_with_local_map(
                map,
                current_keyframe_idx,
                &matches,
                &curr_keypoints_undist,
                &frame.features.scales,
                &frame.features.descriptors,
                &grid,
                frame.image_size,
                camera,
                &pose,
            ) {
                report.local_projection_matches = local.matches.len();
                report.local_pnp_inliers = local.inliers;
                matches = local.matches;
                if local.inliers >= self.config.pnp.min_inliers {
                    pose = local.pose;
                    inliers = local.inliers;
                }
            }
            Ok(Estimate {
                pose,
                inliers,
                matches,
            })
        };

        // PnP from projection matches.
        let last_reject = if projection_matches.len() >= pnp.min_correspondences {
            match try_track_and_refine(projection_matches, candidate_pose, &mut report, true) {
                Ok(estimate) => return (Ok(estimate), report),
                Err(reason) => reason,
            }
        } else {
            MapProjectionRejectReason::LowProjectionMatches
        };

        // Fallback: match against reference keyframe descriptors.
        let current_kf = current_keyframe_idx.and_then(|ki| map.get_keyframe(ki));
        let Some(current_kf) = current_kf else {
            report.reject_reason = Some(last_reject);
            return (Err(last_reject), report);
        };

        let ref_matches = match_orb_descriptors(
            &current_kf.frame.features.orientations,
            &current_kf.frame.features.descriptors,
            &frame.features.orientations,
            &frame.features.descriptors,
            self.config.match_config,
        );
        report.reference_matches = ref_matches.len();

        const MIN_REF_MATCHES: usize = 15;
        if ref_matches.len() < MIN_REF_MATCHES {
            report.reject_reason = Some(MapProjectionRejectReason::LowReferenceMatches);
            return (Err(MapProjectionRejectReason::LowReferenceMatches), report);
        }

        let mut ref_correspondences = Vec::with_capacity(ref_matches.len());
        for (kf_desc_idx, curr_idx) in ref_matches {
            if let Some(Some(mp_idx)) = current_kf.map_point_by_desc_idx.get(kf_desc_idx)
                && *mp_idx < map.map_points().len()
            {
                ref_correspondences.push((*mp_idx, curr_idx));
            }
        }
        report.reference_correspondences = ref_correspondences.len();

        if ref_correspondences.len() < pnp.min_correspondences {
            report.reject_reason = Some(MapProjectionRejectReason::LowReferenceCorrespondences);
            return (
                Err(MapProjectionRejectReason::LowReferenceCorrespondences),
                report,
            );
        }

        match try_track_and_refine(
            ref_correspondences,
            pose_before_tracking,
            &mut report,
            false,
        ) {
            Ok(estimate) => (Ok(estimate), report),
            Err(reason) => {
                report.reject_reason = Some(reason);
                (Err(reason), report)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_with_local_map(
        &self,
        map: &Map,
        current_kf_idx: Option<usize>,
        tracked_matches: &[(usize, usize)],
        curr_keypoints_undist: &[[f32; 2]],
        curr_scales: &[f32],
        curr_descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        image_size: ImageSize,
        camera: &PinholeCamera,
        pose_init: &Pose3d,
    ) -> Option<Estimate> {
        let current_kf = current_kf_idx.and_then(|ki| map.get_keyframe(ki));
        let (local_map_points, local_to_global) =
            map.build_local_map_points(tracked_matches, current_kf);
        let min_corr = self.config.pnp.min_correspondences;
        if local_map_points.len() < min_corr {
            return None;
        }

        let local_matches = self.match_by_projection(
            &local_map_points,
            curr_keypoints_undist,
            curr_scales,
            curr_descriptors,
            grid,
            camera,
            pose_init,
            image_size,
            self.config.local_projection,
            None,
        );
        if local_matches.len() < min_corr {
            return None;
        }

        let global_matches: Vec<(usize, usize)> = local_matches
            .into_iter()
            .filter_map(|(local_mp_idx, curr_idx)| {
                local_to_global
                    .get(local_mp_idx)
                    .copied()
                    .map(|global_mp_idx| (global_mp_idx, curr_idx))
            })
            .collect();
        if global_matches.len() < min_corr {
            return None;
        }

        let (new_pose, inliers) = self.solve_pnp(
            map.map_points(),
            &global_matches,
            curr_keypoints_undist,
            curr_scales,
            camera,
            pose_init,
        )?;
        Some(Estimate {
            pose: new_pose,
            inliers,
            matches: global_matches,
        })
    }

    /// Gather 3D-2D correspondences from map points and keypoints, then solve PnP.
    fn solve_pnp(
        &self,
        map_points: &[MapPoint],
        correspondences: &[(usize, usize)],
        keypoints_undist: &[[f32; 2]],
        keypoint_scales: &[f32],
        camera: &PinholeCamera,
        pose_init: &Pose3d,
    ) -> Option<(Pose3d, usize)> {
        let mut points_world = Vec::with_capacity(correspondences.len());
        let mut points_image = Vec::with_capacity(correspondences.len());
        let mut keypoint_octaves = Vec::with_capacity(correspondences.len());
        for &(mp_idx, kp_idx) in correspondences {
            if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
                points_world.push(mp.position);
                points_image.push(Vec2F32::new(kp[0], kp[1]));
                keypoint_octaves.push(keypoint_octave(
                    keypoint_scales.get(kp_idx).copied().unwrap_or(1.0),
                ));
            }
        }
        pnp::solve_pnp_with_octaves(
            &points_world,
            &points_image,
            Some(&keypoint_octaves),
            camera,
            pose_init,
            &self.config.pnp,
        )
    }

    /// Undistorts keypoints, builds a spatial grid, and runs projection matching
    /// with narrow-to-wide fallback.
    fn match_map_to_frame(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        mp_filter: Option<&std::collections::HashSet<usize>>,
    ) -> (Vec<(usize, usize)>, Vec<[f32; 2]>, KeypointGrid) {
        let (matches, keypoints_undist, grid, _) = self.match_map_to_frame_internal(
            map_points, frame, pose, camera, mp_filter, false,
        );
        (matches, keypoints_undist, grid)
    }

    fn match_map_to_frame_internal(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        mp_filter: Option<&std::collections::HashSet<usize>>,
        collect_debug: bool,
    ) -> (
        Vec<(usize, usize)>,
        Vec<[f32; 2]>,
        KeypointGrid,
        Option<ProjectionDebugFrame>,
    ) {
        self.match_map_to_frame_internal_with_overrides(
            map_points,
            frame,
            pose,
            camera,
            mp_filter,
            collect_debug,
            None,
        )
    }

    fn match_map_to_frame_internal_with_overrides(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        mp_filter: Option<&std::collections::HashSet<usize>>,
        collect_debug: bool,
        projection_overrides: Option<&[ProjectionOverridePoint]>,
    ) -> (
        Vec<(usize, usize)>,
        Vec<[f32; 2]>,
        KeypointGrid,
        Option<ProjectionDebugFrame>,
    ) {
        const MIN_MATCHES_BEFORE_WIDE: usize = 20;

        let keypoints_undist: Vec<[f32; 2]> = frame
            .features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                [p.x as f32, p.y as f32]
            })
            .collect();

        let image_bounds = undistorted_image_bounds(camera, frame.image_size);
        let grid = KeypointGrid::new(
            &keypoints_undist,
            (
                image_bounds.0 as f32,
                image_bounds.1 as f32,
                image_bounds.2 as f32,
                image_bounds.3 as f32,
            ),
        );

        if let Some(overrides) = projection_overrides {
            assert_eq!(
                overrides.len(),
                map_points.len(),
                "projection override count must match map_points length"
            );
        }

        let config = self.config.projection;
        let (mut matches, mut debug) = self.match_by_projection_internal(
            map_points,
            &keypoints_undist,
            &frame.features.scales,
            &frame.features.descriptors,
            &grid,
            camera,
            pose,
            frame.image_size,
            config,
            mp_filter,
            collect_debug,
            projection_overrides,
        );

        if matches.len() < MIN_MATCHES_BEFORE_WIDE {
            (matches, debug) = self.match_by_projection_internal(
                map_points,
                &keypoints_undist,
                &frame.features.scales,
                &frame.features.descriptors,
                &grid,
                camera,
                pose,
                frame.image_size,
                ProjectionMatchConfig {
                    search_radius: config.search_radius * 2.0,
                    ..config
                },
                mp_filter,
                collect_debug,
                projection_overrides,
            );
        }

        (matches, keypoints_undist, grid, debug)
    }

    #[allow(clippy::too_many_arguments)]
    fn match_by_projection(
        &self,
        map_points: &[MapPoint],
        keypoints_xy: &[[f32; 2]],
        scales: &[f32],
        descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
        config: ProjectionMatchConfig,
        mp_filter: Option<&std::collections::HashSet<usize>>,
    ) -> Vec<(usize, usize)> {
        self.match_by_projection_internal(
            map_points,
            keypoints_xy,
            scales,
            descriptors,
            grid,
            camera,
            pose_world_to_cam,
            image_size,
            config,
            mp_filter,
            false,
            None,
        )
        .0
    }

    #[allow(clippy::too_many_arguments)]
    fn match_by_projection_internal(
        &self,
        map_points: &[MapPoint],
        keypoints_xy: &[[f32; 2]],
        scales: &[f32],
        descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
        config: ProjectionMatchConfig,
        mp_filter: Option<&std::collections::HashSet<usize>>,
        collect_debug: bool,
        projection_overrides: Option<&[ProjectionOverridePoint]>,
    ) -> (Vec<(usize, usize)>, Option<ProjectionDebugFrame>) {
        let mut matched_kp = vec![false; keypoints_xy.len()];
        let mut matches = Vec::new();
        let mut debug_points = collect_debug.then(Vec::new);
        let camera_center = pose_world_to_cam.inverse().translation;
        let image_bounds = undistorted_image_bounds(camera, image_size);

        for (mp_idx, mp) in map_points.iter().enumerate() {
            if mp.culled {
                continue;
            }
            if let Some(filter) = mp_filter
                && !filter.contains(&mp_idx)
            {
                continue;
            }

            let override_state = projection_overrides.and_then(|overrides| overrides.get(mp_idx));
            let projected = if let Some(override_state) = override_state {
                if override_state.visible {
                    match (
                        override_state.projected_pixel,
                        override_state.predicted_octave,
                        override_state.view_cos,
                    ) {
                        (Some([u, v]), Some(predicted_octave), Some(view_cos)) => Some((
                            kornia_algebra::Vec2F64::new(u as f64, v as f64),
                            predicted_octave,
                            view_cos as f64,
                        )),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                project_visible_map_point(
                    mp,
                    pose_world_to_cam,
                    camera,
                    config.min_depth,
                    &camera_center,
                    image_bounds,
                )
            };

            let Some((pixel, predicted_octave, view_cos)) = projected else {
                if let Some(points) = debug_points.as_mut() {
                    points.push(ProjectionDebugPoint {
                        map_point_idx: mp_idx,
                        visible: false,
                        projected_pixel: None,
                        predicted_octave: None,
                        matched_keypoint_idx: None,
                        matcher_trace: ProjectionDebugMatcherTrace::default(),
                    });
                }
                continue;
            };
            let u = pixel.x as f32;
            let v = pixel.y as f32;
            let mut debug_point = ProjectionDebugPoint {
                map_point_idx: mp_idx,
                visible: true,
                projected_pixel: Some([u, v]),
                predicted_octave: Some(predicted_octave),
                matched_keypoint_idx: None,
                matcher_trace: ProjectionDebugMatcherTrace::default(),
            };

            let search_radius = config.search_radius
                * radius_by_viewing_cos(view_cos) as f32
                * ORB_SCALE_FACTOR.powi(predicted_octave as i32) as f32;
            debug_point.matcher_trace.search_radius = search_radius;
            debug_point.matcher_trace.min_level = predicted_octave.saturating_sub(1) as isize;
            debug_point.matcher_trace.max_level = predicted_octave as isize;
            let candidates = grid.query_features_in_area(
                u,
                v,
                search_radius,
                predicted_octave.saturating_sub(1) as isize,
                predicted_octave as isize,
                keypoints_xy,
                scales,
                keypoint_octave,
            );
            let mut best_dist = u32::MAX;
            let mut best_octave = usize::MAX;
            let mut second_dist = u32::MAX;
            let mut second_octave = usize::MAX;
            let mut best_kp = usize::MAX;

            for kp_idx in candidates {
                let kp_octave = keypoint_octave(scales.get(kp_idx).copied().unwrap_or(1.0));
                let dist = hamming_distance(&mp.descriptor, &descriptors[kp_idx]);
                debug_point
                    .matcher_trace
                    .feature_candidates
                    .push(ProjectionDebugFeatureCandidate {
                        keypoint_idx: kp_idx,
                        octave: kp_octave,
                        occupied_before: false,
                        descriptor_dist: dist,
                    });
                if matched_kp[kp_idx] {
                    continue;
                }
                if dist < best_dist {
                    second_dist = best_dist;
                    second_octave = best_octave;
                    debug_point.matcher_trace.second_best_keypoint_idx =
                        debug_point.matcher_trace.best_keypoint_idx;
                    debug_point.matcher_trace.second_best_dist = debug_point.matcher_trace.best_dist;
                    debug_point.matcher_trace.second_best_level =
                        debug_point.matcher_trace.best_level;
                    best_dist = dist;
                    best_octave = kp_octave;
                    best_kp = kp_idx;
                    debug_point.matcher_trace.best_keypoint_idx = Some(kp_idx);
                    debug_point.matcher_trace.best_dist = Some(dist);
                    debug_point.matcher_trace.best_level = Some(kp_octave);
                } else if dist < second_dist {
                    second_dist = dist;
                    second_octave = kp_octave;
                    debug_point.matcher_trace.second_best_keypoint_idx = Some(kp_idx);
                    debug_point.matcher_trace.second_best_dist = Some(dist);
                    debug_point.matcher_trace.second_best_level = Some(kp_octave);
                }
            }

            let passes_ratio = best_octave != second_octave
                || second_dist == u32::MAX
                || (best_dist as f32) <= 0.8 * (second_dist as f32);
            if best_dist <= config.max_hamming && best_kp != usize::MAX && !passes_ratio {
                debug_point.matcher_trace.ratio_rejected = true;
            }
            if best_dist <= config.max_hamming && best_kp != usize::MAX && passes_ratio {
                matched_kp[best_kp] = true;
                matches.push((mp_idx, best_kp));
                debug_point.matched_keypoint_idx = Some(best_kp);
            }

            if let Some(points) = debug_points.as_mut() {
                points.push(debug_point);
            }
        }

        (
            matches,
            debug_points.map(|points| ProjectionDebugFrame { points }),
        )
    }
}

fn project_visible_map_point(
    map_point: &MapPoint,
    pose_world_to_cam: &Pose3d,
    camera: &PinholeCamera,
    min_depth: f64,
    camera_center_world: &kornia_algebra::Vec3F64,
    image_bounds: (f64, f64, f64, f64),
) -> Option<(kornia_algebra::Vec2F64, usize, f64)> {
    let p_cam = pose_world_to_cam.transform_point(&map_point.position);
    let pixel = camera.project_to_pixel(&p_cam, min_depth)?;
    let (min_x, max_x, min_y, max_y) = image_bounds;
    if pixel.x < min_x || pixel.x > max_x || pixel.y < min_y || pixel.y > max_y {
        return None;
    }

    let po = map_point.position - *camera_center_world;
    let dist = po.length();
    if dist <= 0.0 {
        return None;
    }
    if dist < map_point.min_distance_invariance() || dist > map_point.max_distance_invariance() {
        return None;
    }

    let view_cos = po.dot(map_point.viewing_normal()) / dist;
    if view_cos < VIEWING_COS_LIMIT {
        return None;
    }

    Some((
        pixel,
        map_point.predict_scale(dist, ORB_SCALE_FACTOR, ORB_N_LEVELS),
        view_cos,
    ))
}

fn undistorted_image_bounds(
    camera: &PinholeCamera,
    image_size: ImageSize,
) -> (f64, f64, f64, f64) {
    if camera.k1 == 0.0 && camera.k2 == 0.0 && camera.p1 == 0.0 && camera.p2 == 0.0 {
        return (
            0.0,
            image_size.width as f64,
            0.0,
            image_size.height as f64,
        );
    }

    let top_left = camera.undistort(0.0, 0.0);
    let top_right = camera.undistort(image_size.width as f64, 0.0);
    let bottom_left = camera.undistort(0.0, image_size.height as f64);
    let bottom_right = camera.undistort(image_size.width as f64, image_size.height as f64);

    (
        top_left.x.min(bottom_left.x),
        top_right.x.max(bottom_right.x),
        top_left.y.min(top_right.y),
        bottom_left.y.max(bottom_right.y),
    )
}

fn radius_by_viewing_cos(view_cos: f64) -> f64 {
    if view_cos > 0.998 {
        2.5
    } else {
        4.0
    }
}

fn keypoint_octave(scale: f32) -> usize {
    let octave = ((scale as f64).ln() / ORB_SCALE_FACTOR.ln()).round() as isize;
    octave.clamp(0, ORB_N_LEVELS.saturating_sub(1) as isize) as usize
}

/// Human-readable name for a map-projection rejection reason.
pub fn map_projection_reject_reason_name(reason: MapProjectionRejectReason) -> &'static str {
    match reason {
        MapProjectionRejectReason::LowProjectionMatches => "low_projection_matches",
        MapProjectionRejectReason::PnpFailed => "pnp_failed",
        MapProjectionRejectReason::LowPnpInliers => "low_pnp_inliers",
        MapProjectionRejectReason::LowReferenceMatches => "low_reference_matches",
        MapProjectionRejectReason::LowReferenceCorrespondences => "low_reference_correspondences",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod estimator_tests {
    use crate::system::KeyframePolicy;

    #[test]
    fn test_need_new_keyframe_forced_by_gap() {
        let policy = KeyframePolicy::default();
        assert!(policy.should_insert(100, Some(90), 50, 100));
    }

    #[test]
    fn test_need_new_keyframe_max_gap_overrides_strong_tracking() {
        let policy = KeyframePolicy::default();
        assert!(policy.should_insert(100, Some(90), 100, 100));
    }

    #[test]
    fn test_need_new_keyframe_too_soon() {
        let policy = KeyframePolicy::default();
        assert!(!policy.should_insert(2, Some(1), 50, 100));
    }
}

#[cfg(test)]
mod matching_tests {
    use super::*;
    use std::collections::HashSet;

    use kornia_algebra::{Mat3F64, Vec3F64};

    fn make_test_estimator() -> MapProjectionEstimator {
        MapProjectionEstimator::new(MapProjectionConfig::default())
    }

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_match_by_projection_simple() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let desc_a = [0u8; 32];
        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: desc_a,
            color: [0; 3],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0], [100.0, 100.0]];
        let descriptors = vec![desc_a, [0xFF; 32]];

        let grid = KeypointGrid::new(&keypoints_xy, (0.0, 640.0, 0.0, 480.0));
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &[1.0, 1.0],
            &descriptors,
            &grid,
            &camera,
            &pose,
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
            None,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_behind_camera_rejected() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, -5.0),
            descriptor: [0u8; 32],
            color: [0; 3],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0u8; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, (0.0, 640.0, 0.0, 480.0));

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &[1.0],
            &descriptors,
            &grid,
            &camera,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
            None,
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn test_high_hamming_rejected() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: [0u8; 32],
            color: [0; 3],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0xFF; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, (0.0, 640.0, 0.0, 480.0));

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &[1.0],
            &descriptors,
            &grid,
            &camera,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
            None,
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn test_debug_match_map_to_frame_returns_structured_output() {
        use kornia_imgproc::features::OrbFeatures;

        let estimator = make_test_estimator();
        let camera = test_camera();
        let descriptor = [0u8; 32];
        let frame = Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: vec![[320.0, 240.0], [100.0, 100.0]],
                scales: vec![1.0, 1.0],
                orientations: vec![0.0, 0.0],
                descriptors: vec![descriptor, [0xFF; 32]],
            },
            pose_world_to_cam: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; 2],
        };
        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor,
            color: [0; 3],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }];

        let debug = estimator.debug_match_map_to_frame(
            &map_points,
            &frame,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            &camera,
            None,
        );

        assert_eq!(debug.points.len(), 1);
        let point = &debug.points[0];
        assert_eq!(point.map_point_idx, 0);
        assert!(point.visible);
        assert_eq!(point.projected_pixel, Some([320.0, 240.0]));
        assert_eq!(point.predicted_octave, Some(0));
        assert_eq!(point.matched_keypoint_idx, Some(0));
    }

    #[test]
    fn test_debug_match_map_to_frame_matches_normal_path_with_wide_fallback() {
        use kornia_imgproc::features::OrbFeatures;

        let estimator = make_test_estimator();
        let camera = test_camera();
        let descriptor = [0u8; 32];
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let frame = Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: vec![[324.0, 240.0]],
                scales: vec![1.0],
                orientations: vec![0.0],
                descriptors: vec![descriptor],
            },
            pose_world_to_cam: pose.clone(),
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]],
        };
        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor,
            color: [0; 3],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }];

        let (matches, _, _) = estimator.match_map_to_frame(&map_points, &frame, &pose, &camera, None);
        let debug = estimator.debug_match_map_to_frame(&map_points, &frame, &pose, &camera, None);

        let debug_matches: Vec<(usize, usize)> = debug
            .points
            .iter()
            .filter_map(|point| {
                point
                    .matched_keypoint_idx
                    .map(|kp_idx| (point.map_point_idx, kp_idx))
            })
            .collect();

        assert_eq!(matches, vec![(0, 0)]);
        assert_eq!(debug_matches, matches);
    }

    #[test]
    fn test_debug_match_map_to_frame_respects_mp_filter() {
        use kornia_imgproc::features::OrbFeatures;

        let estimator = make_test_estimator();
        let camera = test_camera();
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let descriptors = vec![[0u8; 32], [1u8; 32]];
        let frame = Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: vec![[320.0, 240.0], [360.0, 240.0]],
                scales: vec![1.0, 1.0],
                orientations: vec![0.0, 0.0],
                descriptors: descriptors.clone(),
            },
            pose_world_to_cam: pose.clone(),
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; 2],
        };
        let map_points = vec![
            MapPoint {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                descriptor: descriptors[0],
                color: [0; 3],
                keyframe_idx: 0,
                n_visible: 0,
                n_found: 0,
                culled: false,
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                min_distance: 0.0,
                max_distance: f64::INFINITY,
                observation_count_override: None,
                observed_keyframes_override: None,
            },
            MapPoint {
                position: Vec3F64::new(1.0, 0.0, 5.0),
                descriptor: descriptors[1],
                color: [0; 3],
                keyframe_idx: 0,
                n_visible: 0,
                n_found: 0,
                culled: false,
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                min_distance: 0.0,
                max_distance: f64::INFINITY,
                observation_count_override: None,
                observed_keyframes_override: None,
            },
        ];
        let filter = HashSet::from([1usize]);

        let (matches, _, _) =
            estimator.match_map_to_frame(&map_points, &frame, &pose, &camera, Some(&filter));
        let debug =
            estimator.debug_match_map_to_frame(&map_points, &frame, &pose, &camera, Some(&filter));

        let debug_matches: Vec<(usize, usize)> = debug
            .points
            .iter()
            .filter_map(|point| {
                point
                    .matched_keypoint_idx
                    .map(|kp_idx| (point.map_point_idx, kp_idx))
            })
            .collect();

        assert_eq!(debug.points.len(), 1);
        assert_eq!(debug.points[0].map_point_idx, 1);
        assert_eq!(matches, vec![(1, 1)]);
        assert_eq!(debug_matches, matches);
    }
}
