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

use std::collections::HashMap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, RansacParams, ransac_fundamental};
use kornia_algebra::{Vec2F32, Vec2F64};
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;

use super::pnp::{self, PnpConfig};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::frame::Frame;
use crate::map::{Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR};

use super::Estimate;
use keypoint_grid::KeypointGrid;

/// Half-width of the pyramid-octave window for projection matching. ORB-SLAM3
/// uses `[predicted-1, predicted]` (effectively half-width ~1, biased down);
/// our ORB extractor's octave assignments are noisier, so we widen it.
const OCTAVE_WINDOW_HALF_WIDTH: usize = 2;

/// Tunable parameters for projection-guided matching.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionMatchConfig {
    /// Reject projected points with depth `<= min_depth`.
    pub min_depth: f64,
    /// Keypoint search radius around each projected pixel.
    pub search_radius: f32,
    /// Maximum Hamming distance to accept a descriptor match.
    pub max_hamming: u32,
}

impl Default for ProjectionMatchConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.0,
            search_radius: 15.0,
            max_hamming: 50,
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
    /// Growth in `search_scale` per second spent failing to track (see
    /// `search_scale_for`).
    pub search_widen_per_sec: f32,
    /// Upper bound on `search_scale` (see `search_scale_for`).
    pub max_search_scale: f32,
    /// Fundamental-matrix RANSAC inlier threshold in pixels.
    pub geometric_filter_threshold_px: f64,
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
                search_radius: 30.0,
                max_hamming: 60,
                ..ProjectionMatchConfig::default()
            },
            search_widen_per_sec: 1.0,
            max_search_scale: 4.0,
            geometric_filter_threshold_px: 1.0,
        }
    }
}

impl MapProjectionConfig {
    /// `search_scale` to pass to `MapProjectionEstimator::estimate_pose` given
    /// how long tracking has been failing.
    ///
    /// Widens the search/PnP-prior gates in proportion to how long we've
    /// already been failing to track: a pose predicted by compounding
    /// IMU/constant-velocity integration over several seconds of loss carries
    /// far more uncertainty than a single-frame prediction, and the narrow
    /// gates sized for the latter would otherwise starve PnP of
    /// correspondences for the entire recently-lost grace period, making a
    /// longer grace period actively counterproductive.
    pub fn search_scale_for(&self, currently_lost_for_sec: f64) -> f32 {
        (1.0 + currently_lost_for_sec as f32 * self.search_widen_per_sec).min(self.max_search_scale)
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

    /// Estimate the pose of `frame` against the map.
    ///
    /// `search_scale` (1.0 = normal) grows the projection search radius and PnP
    /// reprojection gate; see [`MapProjectionConfig::search_scale_for`].
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_pose(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        camera: &PinholeCamera,
        current_keyframe_idx: Option<usize>,
        search_scale: f32,
        pre_seeded: Option<Vec<(usize, usize)>>,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let pnp = &self.config.pnp;

        // Bound the initial projection search to the local map around the
        // reference keyframe (its points plus covisibility neighbors'), so
        // per-frame tracking cost is independent of total map size. With no
        // reference KF yet the builder falls back to all non-culled points.
        let current_kf = current_keyframe_idx.and_then(|ki| map.get_keyframe(ki));
        let local_indices = map.build_local_map_point_indices(&[], current_kf);

        let (projection_matches, curr_keypoints_undist, grid) = self.match_map_to_frame(
            map,
            &local_indices,
            frame,
            candidate_pose,
            camera,
            search_scale,
        );

        // Shared logic: try_track → refine_pose → Estimate, or propagate rejection.
        let try_track_and_refine = |correspondences: Vec<(usize, usize)>,
                                    pose_init: &Pose3d|
         -> Result<Estimate, MapProjectionRejectReason> {
            let correspondences = self.geometric_consistency_filter(
                current_kf,
                &correspondences,
                &curr_keypoints_undist,
                camera,
            );
            let (mut pose, mut inliers) = self
                .solve_pnp(
                    map.map_points(),
                    &correspondences,
                    &curr_keypoints_undist,
                    camera,
                    pose_init,
                    search_scale,
                )
                .ok_or(MapProjectionRejectReason::PnpFailed)?;
            if inliers < pnp.min_inliers_early {
                return Err(MapProjectionRejectReason::LowPnpInliers);
            }
            let mut matches = correspondences;
            if let Some(local) = self.refine_with_local_map(
                map,
                current_keyframe_idx,
                &matches,
                &curr_keypoints_undist,
                &frame.features.descriptors,
                &frame.features.octaves,
                &grid,
                frame.image_size,
                camera,
                &pose,
                search_scale,
            ) {
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

        if let Some(seeded) = Self::eligible_pre_seeded(pre_seeded, pnp.min_correspondences)
            && let Ok(estimate) = try_track_and_refine(seeded, candidate_pose)
        {
            return Ok(estimate);
        }

        // PnP from projection matches.
        let last_reject = if projection_matches.len() >= pnp.min_correspondences {
            match try_track_and_refine(projection_matches, candidate_pose) {
                Ok(estimate) => return Ok(estimate),
                Err(reason) => reason,
            }
        } else {
            MapProjectionRejectReason::LowProjectionMatches
        };

        // Fallback: match against reference keyframe descriptors.
        let Some(current_kf) = current_kf else {
            return Err(last_reject);
        };

        let ref_matches = match_orb_descriptors(
            &current_kf.frame.features.orientations,
            &current_kf.frame.features.descriptors,
            &frame.features.orientations,
            &frame.features.descriptors,
            self.config.match_config,
        );

        const MIN_REF_MATCHES: usize = 15;
        if ref_matches.len() < MIN_REF_MATCHES {
            return Err(MapProjectionRejectReason::LowReferenceMatches);
        }

        let mut ref_correspondences = Vec::with_capacity(ref_matches.len());
        for (kf_desc_idx, curr_idx) in ref_matches {
            if let Some(Some(mp_idx)) = current_kf.map_point_by_desc_idx.get(kf_desc_idx)
                && *mp_idx < map.map_points().len()
            {
                ref_correspondences.push((*mp_idx, curr_idx));
            }
        }

        if ref_correspondences.len() < pnp.min_correspondences {
            return Err(MapProjectionRejectReason::LowReferenceCorrespondences);
        }

        try_track_and_refine(ref_correspondences, pose_before_tracking)
    }

    fn eligible_pre_seeded(
        pre_seeded: Option<Vec<(usize, usize)>>,
        min_correspondences: usize,
    ) -> Option<Vec<(usize, usize)>> {
        pre_seeded.filter(|correspondences| correspondences.len() >= min_correspondences)
    }

    /// Reject matches that disagree with reference-to-current epipolar geometry.
    /// Matches without a reference-keyframe observation remain unfiltered.
    fn geometric_consistency_filter(
        &self,
        current_kf: Option<&Keyframe>,
        correspondences: &[(usize, usize)],
        curr_keypoints_undist: &[[f32; 2]],
        camera: &PinholeCamera,
    ) -> Vec<(usize, usize)> {
        const MIN_PAIRS_FOR_FILTER: usize = 8;

        let Some(current_kf) = current_kf else {
            return correspondences.to_vec();
        };
        let mp_to_desc: HashMap<usize, usize> = current_kf
            .map_point_by_desc_idx
            .iter()
            .enumerate()
            .filter_map(|(desc_idx, mp_idx)| mp_idx.map(|mp_idx| (mp_idx, desc_idx)))
            .collect();

        let mut checkable_indices = Vec::new();
        let mut reference_points = Vec::new();
        let mut current_points = Vec::new();
        for (index, &(mp_idx, kp_idx)) in correspondences.iter().enumerate() {
            let Some(&desc_idx) = mp_to_desc.get(&mp_idx) else {
                continue;
            };
            let Some(reference) = current_kf.frame.undistorted_xy(desc_idx, camera) else {
                continue;
            };
            let Some(&current) = curr_keypoints_undist.get(kp_idx) else {
                continue;
            };
            checkable_indices.push(index);
            reference_points.push(Vec2F64::new(reference[0] as f64, reference[1] as f64));
            current_points.push(Vec2F64::new(current[0] as f64, current[1] as f64));
        }

        if checkable_indices.len() < MIN_PAIRS_FOR_FILTER {
            return correspondences.to_vec();
        }
        let params = RansacParams {
            threshold: self.config.geometric_filter_threshold_px,
            ..RansacParams::default()
        };
        let Ok(result) = ransac_fundamental(&reference_points, &current_points, &params) else {
            return correspondences.to_vec();
        };

        let mut keep = vec![true; correspondences.len()];
        for (&index, &is_inlier) in checkable_indices.iter().zip(&result.inliers) {
            if !is_inlier {
                keep[index] = false;
            }
        }
        correspondences
            .iter()
            .zip(keep)
            .filter_map(|(&correspondence, keep)| keep.then_some(correspondence))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_with_local_map(
        &self,
        map: &Map,
        current_kf_idx: Option<usize>,
        tracked_matches: &[(usize, usize)],
        curr_keypoints_undist: &[[f32; 2]],
        curr_descriptors: &[[u8; 32]],
        curr_octaves: &[u8],
        grid: &KeypointGrid,
        image_size: ImageSize,
        camera: &PinholeCamera,
        pose_init: &Pose3d,
        search_scale: f32,
    ) -> Option<Estimate> {
        let current_kf = current_kf_idx.and_then(|ki| map.get_keyframe(ki));
        let local_indices = map.build_local_map_point_indices(tracked_matches, current_kf);
        let min_corr = self.config.pnp.min_correspondences;
        if local_indices.len() < min_corr {
            return None;
        }

        let local_config = ProjectionMatchConfig {
            search_radius: self.config.local_projection.search_radius * search_scale,
            ..self.config.local_projection
        };
        let global_matches = self.match_by_projection(
            map.map_points(),
            &local_indices,
            curr_keypoints_undist,
            curr_descriptors,
            curr_octaves,
            grid,
            camera,
            pose_init,
            image_size,
            local_config,
        );
        if global_matches.len() < min_corr {
            return None;
        }

        let (new_pose, inliers) = self.solve_pnp(
            map.map_points(),
            &global_matches,
            curr_keypoints_undist,
            camera,
            pose_init,
            search_scale,
        )?;
        Some(Estimate {
            pose: new_pose,
            inliers,
            matches: global_matches,
        })
    }

    /// Gather 3D-2D correspondences from map points and keypoints, then solve PnP.
    ///
    /// `search_scale` widens the coarse prior-reprojection gate (see
    /// [`Self::estimate_pose`]) so a drifted `pose_init` doesn't starve the
    /// LM solve of correspondences; the tight final-inlier threshold that
    /// actually accepts the solution is unaffected.
    fn solve_pnp(
        &self,
        map_points: &[MapPoint],
        correspondences: &[(usize, usize)],
        keypoints_undist: &[[f32; 2]],
        camera: &PinholeCamera,
        pose_init: &Pose3d,
        search_scale: f32,
    ) -> Option<(Pose3d, usize)> {
        let mut points_world = Vec::with_capacity(correspondences.len());
        let mut points_image = Vec::with_capacity(correspondences.len());
        for &(mp_idx, kp_idx) in correspondences {
            if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
                points_world.push(mp.position);
                points_image.push(Vec2F32::new(kp[0], kp[1]));
            }
        }
        let pnp_config = if search_scale > 1.0 {
            PnpConfig {
                prior_reproj_threshold_px: self.config.pnp.prior_reproj_threshold_px
                    * search_scale as f64,
                ..self.config.pnp.clone()
            }
        } else {
            self.config.pnp.clone()
        };
        pnp::solve_pnp(&points_world, &points_image, camera, pose_init, &pnp_config)
    }

    /// Undistorts keypoints, builds a spatial grid, and runs projection matching
    /// with narrow-to-wide fallback over the candidate map-point indices.
    fn match_map_to_frame(
        &self,
        map: &Map,
        candidates: &[usize],
        frame: &Frame,
        pose: &Pose3d,
        camera: &PinholeCamera,
        search_scale: f32,
    ) -> (Vec<(usize, usize)>, Vec<[f32; 2]>, KeypointGrid) {
        const KEYPOINT_GRID_CELL_SIZE: f32 = 64.0;
        const MIN_MATCHES_BEFORE_WIDE: usize = 20;

        // Use the frame's undistortion cache when filled (the pipeline fills
        // it once per frame); only frames built outside the pipeline pay the
        // per-keypoint undistortion here.
        let keypoints_undist: Vec<[f32; 2]> =
            if frame.keypoints_undist.len() == frame.features.keypoints_xy.len() {
                frame.keypoints_undist.clone()
            } else {
                frame
                    .features
                    .keypoints_xy
                    .iter()
                    .map(|kp| {
                        let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                        [p.x as f32, p.y as f32]
                    })
                    .collect()
            };

        let grid = KeypointGrid::new(
            &keypoints_undist,
            frame.image_size.width as f32,
            frame.image_size.height as f32,
            KEYPOINT_GRID_CELL_SIZE,
        );

        let config = ProjectionMatchConfig {
            search_radius: self.config.projection.search_radius * search_scale,
            ..self.config.projection
        };
        let mut matches = self.match_by_projection(
            map.map_points(),
            candidates,
            &keypoints_undist,
            &frame.features.descriptors,
            &frame.features.octaves,
            &grid,
            camera,
            pose,
            frame.image_size,
            config,
        );

        if matches.len() < MIN_MATCHES_BEFORE_WIDE {
            matches = self.match_by_projection(
                map.map_points(),
                candidates,
                &keypoints_undist,
                &frame.features.descriptors,
                &frame.features.octaves,
                &grid,
                camera,
                pose,
                frame.image_size,
                ProjectionMatchConfig {
                    search_radius: config.search_radius * 2.0,
                    ..config
                },
            );
        }

        (matches, keypoints_undist, grid)
    }

    #[allow(clippy::too_many_arguments)]
    fn match_by_projection(
        &self,
        map_points: &[MapPoint],
        candidates: &[usize],
        keypoints_xy: &[[f32; 2]],
        descriptors: &[[u8; 32]],
        octaves: &[u8],
        grid: &KeypointGrid,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
        config: ProjectionMatchConfig,
    ) -> Vec<(usize, usize)> {
        let mut matched_kp = vec![false; keypoints_xy.len()];
        let mut matches = Vec::new();
        let camera_center = pose_world_to_cam.inverse().translation;

        for &mp_idx in candidates {
            let Some(mp) = map_points.get(mp_idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }

            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            let Ok(pixel) = camera.project_to_image(&p_cam, config.min_depth, image_size) else {
                continue;
            };
            let u = pixel.x as f32;
            let v = pixel.y as f32;

            // Scale-invariance gates (ORB-SLAM3, relaxed):
            //  - distance gate: only match within [0.8*min, 1.2*max]
            //  - octave gate: candidates must be near the predicted level
            // Active once the point's scale geometry has been computed.
            let octave_window = if mp.max_distance > 0.0 {
                let dist = (mp.position - camera_center).length();
                if dist < mp.min_distance_invariance() || dist > mp.max_distance_invariance() {
                    continue;
                }
                let level = mp.predict_scale(dist, ORB_SCALE_FACTOR, ORB_N_LEVELS) as i32;
                let half = OCTAVE_WINDOW_HALF_WIDTH as i32;
                let lo = (level - half).max(0) as u8;
                let hi = (level + half).min(ORB_N_LEVELS as i32 - 1) as u8;
                Some((lo, hi))
            } else {
                None
            };

            let candidates = grid.query_radius(u, v, config.search_radius, keypoints_xy);
            let mut best_dist = u32::MAX;
            let mut best_kp = usize::MAX;

            for kp_idx in candidates {
                if matched_kp[kp_idx] {
                    continue;
                }
                if let Some((lo, hi)) = octave_window {
                    let kp_octave = octaves.get(kp_idx).copied().unwrap_or(0);
                    if kp_octave < lo || kp_octave > hi {
                        continue;
                    }
                }
                let dist = hamming_distance(&mp.descriptor, &descriptors[kp_idx]);
                if dist < best_dist {
                    best_dist = dist;
                    best_kp = kp_idx;
                }
            }

            if best_dist <= config.max_hamming && best_kp != usize::MAX {
                matched_kp[best_kp] = true;
                matches.push((mp_idx, best_kp));
            }
        }

        matches
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
    fn test_need_new_keyframe_too_soon() {
        let policy = KeyframePolicy::default();
        assert!(!policy.should_insert(2, Some(1), 50, 100));
    }
}

#[cfg(test)]
mod matching_tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_imgproc::features::OrbFeatures;

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

    fn test_frame(keypoints: Vec<[f32; 2]>) -> Frame {
        let count = keypoints.len();
        Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: keypoints.clone(),
                orientations: vec![0.0; count],
                descriptors: vec![[0; 32]; count],
                octaves: vec![0; count],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; count],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: keypoints,
        }
    }

    fn two_view_keypoints(camera: &PinholeCamera, count: usize) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        let angle = 5.0_f64.to_radians();
        let (sin, cos) = angle.sin_cos();
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(cos, 0.0, -sin),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sin, 0.0, cos),
        );
        let current_pose = Pose3d::new(rotation, Vec3F64::new(0.2, 0.05, 0.0));
        let mut random_state = 12_345_678_901_234_567_u64;
        let random = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as f64 / 4_294_967_296.0
        };
        let mut reference = Vec::with_capacity(count);
        let mut current = Vec::with_capacity(count);
        for _ in 0..count {
            let point = Vec3F64::new(
                (random(&mut random_state) - 0.5) * 3.0,
                (random(&mut random_state) - 0.5) * 2.0,
                random(&mut random_state) * 3.0 + 3.0,
            );
            let reference_pixel = camera.project_to_pixel(&point, 0.0).unwrap();
            let current_pixel = camera
                .project_to_pixel(&current_pose.transform_point(&point), 0.0)
                .unwrap();
            reference.push([reference_pixel.x as f32, reference_pixel.y as f32]);
            current.push([current_pixel.x as f32, current_pixel.y as f32]);
        }
        (reference, current)
    }

    #[test]
    fn geometric_filter_rejects_epipolar_outlier() {
        let estimator = make_test_estimator();
        let camera = test_camera();
        let (reference, mut current) = two_view_keypoints(&camera, 30);
        current[29] = [80.0, 430.0];
        current.push([10.0, 10.0]);

        let mut keyframe = Keyframe::from_frame(test_frame(reference));
        for index in 0..30 {
            keyframe.associate_map_point(index, index);
        }
        let mut correspondences: Vec<(usize, usize)> = (0..30).map(|i| (i, i)).collect();
        let uncheckable = (1000, 30);
        correspondences.push(uncheckable);

        let filtered = estimator.geometric_consistency_filter(
            Some(&keyframe),
            &correspondences,
            &current,
            &camera,
        );

        assert!(!filtered.contains(&(29, 29)));
        assert!(filtered.contains(&uncheckable));
        assert_eq!(filtered.len(), 30);
    }

    #[test]
    fn geometric_filter_is_noop_without_enough_reference_pairs() {
        let estimator = make_test_estimator();
        let camera = test_camera();
        let (reference, current) = two_view_keypoints(&camera, 7);
        let mut keyframe = Keyframe::from_frame(test_frame(reference));
        for index in 0..7 {
            keyframe.associate_map_point(index, index);
        }
        let correspondences: Vec<(usize, usize)> = (0..7).map(|i| (i, i)).collect();

        assert_eq!(
            estimator.geometric_consistency_filter(
                Some(&keyframe),
                &correspondences,
                &current,
                &camera,
            ),
            correspondences
        );
        assert_eq!(
            estimator.geometric_consistency_filter(None, &correspondences, &current, &camera),
            correspondences
        );
    }

    #[test]
    fn klt_source_is_eligible_at_minimum_size() {
        let seeded = vec![(0, 0), (1, 1), (2, 2), (3, 3)];

        assert_eq!(
            MapProjectionEstimator::eligible_pre_seeded(Some(seeded.clone()), 4),
            Some(seeded)
        );
        assert_eq!(
            MapProjectionEstimator::eligible_pre_seeded(Some(vec![(0, 0); 3]), 4),
            None
        );
        assert_eq!(MapProjectionEstimator::eligible_pre_seeded(None, 4), None);
    }

    #[test]
    fn test_match_by_projection_simple() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let desc_a = [0u8; 32];
        let map_points = vec![MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            desc_a,
            0,
            [0; 3],
            0,
        )];

        let keypoints_xy = vec![[320.0f32, 240.0], [100.0, 100.0]];
        let descriptors = vec![desc_a, [0xFF; 32]];
        let octaves = vec![0u8, 0u8];

        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);

        let matches = estimator.match_by_projection(
            &map_points,
            &[0],
            &keypoints_xy,
            &descriptors,
            &octaves,
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
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_behind_camera_rejected() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let map_points = vec![MapPoint::new(
            Vec3F64::new(0.0, 0.0, -5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        )];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0u8; 32]];
        let octaves = vec![0u8];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &[0],
            &keypoints_xy,
            &descriptors,
            &octaves,
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
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn test_high_hamming_rejected() {
        let estimator = make_test_estimator();
        let camera = test_camera();

        let map_points = vec![MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        )];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0xFF; 32]];
        let octaves = vec![0u8];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &[0],
            &keypoints_xy,
            &descriptors,
            &octaves,
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
        );

        assert!(matches.is_empty());
    }
}
