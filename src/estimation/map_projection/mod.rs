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
    pub fn estimate_pose(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        camera: &PinholeCamera,
        current_keyframe_idx: Option<usize>,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        let pnp = &self.config.pnp;

        let (projection_matches, curr_keypoints_undist, grid) =
            self.match_map_to_frame(map.map_points(), frame, candidate_pose, camera);

        // Shared logic: try_track → refine_pose → Estimate, or propagate rejection.
        let try_track_and_refine = |correspondences: Vec<(usize, usize)>,
                                    pose_init: &Pose3d|
         -> Result<Estimate, MapProjectionRejectReason> {
            let (mut pose, mut inliers) = self
                .solve_pnp(
                    map.map_points(),
                    &correspondences,
                    &curr_keypoints_undist,
                    camera,
                    pose_init,
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
                &grid,
                frame.image_size,
                camera,
                &pose,
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
        let current_kf = current_keyframe_idx.and_then(|ki| map.get_keyframe(ki));
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

    #[allow(clippy::too_many_arguments)]
    fn refine_with_local_map(
        &self,
        map: &Map,
        current_kf_idx: Option<usize>,
        tracked_matches: &[(usize, usize)],
        curr_keypoints_undist: &[[f32; 2]],
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
            curr_descriptors,
            grid,
            camera,
            pose_init,
            image_size,
            self.config.local_projection,
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
        camera: &PinholeCamera,
        pose_init: &Pose3d,
    ) -> Option<(Pose3d, usize)> {
        let mut points_world = Vec::with_capacity(correspondences.len());
        let mut points_image = Vec::with_capacity(correspondences.len());
        for &(mp_idx, kp_idx) in correspondences {
            if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
                points_world.push(mp.position);
                points_image.push(Vec2F32::new(kp[0], kp[1]));
            }
        }
        pnp::solve_pnp(
            &points_world,
            &points_image,
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
    ) -> (Vec<(usize, usize)>, Vec<[f32; 2]>, KeypointGrid) {
        const KEYPOINT_GRID_CELL_SIZE: f32 = 64.0;
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

        let grid = KeypointGrid::new(
            &keypoints_undist,
            frame.image_size.width as f32,
            frame.image_size.height as f32,
            KEYPOINT_GRID_CELL_SIZE,
        );

        let config = self.config.projection;
        let mut matches = self.match_by_projection(
            map_points,
            &keypoints_undist,
            &frame.features.descriptors,
            &grid,
            camera,
            pose,
            frame.image_size,
            config,
        );

        if matches.len() < MIN_MATCHES_BEFORE_WIDE {
            matches = self.match_by_projection(
                map_points,
                &keypoints_undist,
                &frame.features.descriptors,
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
        keypoints_xy: &[[f32; 2]],
        descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
        config: ProjectionMatchConfig,
    ) -> Vec<(usize, usize)> {
        let mut matched_kp = vec![false; keypoints_xy.len()];
        let mut matches = Vec::new();

        for (mp_idx, mp) in map_points.iter().enumerate() {
            if mp.culled {
                continue;
            }

            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            let Ok(pixel) = camera.project_to_image(&p_cam, config.min_depth, image_size) else {
                continue;
            };
            let u = pixel.x as f32;
            let v = pixel.y as f32;

            let candidates = grid.query_radius(u, v, config.search_radius, keypoints_xy);
            let mut best_dist = u32::MAX;
            let mut best_kp = usize::MAX;

            for kp_idx in candidates {
                if matched_kp[kp_idx] {
                    continue;
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
        }];

        let keypoints_xy = vec![[320.0f32, 240.0], [100.0, 100.0]];
        let descriptors = vec![desc_a, [0xFF; 32]];

        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
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
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0u8; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
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
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0xFF; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
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
        );

        assert!(matches.is_empty());
    }
}
