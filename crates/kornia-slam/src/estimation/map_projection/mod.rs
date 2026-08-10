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
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{RansacParams, ransac_fundamental};
use kornia_algebra::{Vec2F32, Vec2F64, Vec3F64};
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
    /// Inlier threshold (pixels) for the fundamental-matrix RANSAC geometric
    /// consistency check applied to reference-keyframe/current-frame
    /// correspondence pairs before PnP (see
    /// [`MapProjectionEstimator::geometric_consistency_filter`]). Matches
    /// lightweight_vio's own tuned `F_threshold` default.
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
        // KLT-seeded correspondences (see `optical_flow::klt_correspondences`,
        // called by `Pipeline::tracking_step`), tried before anything below.
        // Unlike `projection_matches`, generating these doesn't depend on
        // `candidate_pose` at all, so a degraded prediction can't starve this
        // path the way it can the projection search.
        pre_seeded: Option<Vec<(usize, usize)>>,
        // Per-stage correspondence counts for whichever source(s) get
        // attempted this call, appended here rather than returned as part of
        // `MapProjectionRejectReason` — this is purely for diagnosing *why*
        // a source failed (starved by the prior-reprojection gate vs. an
        // outright LM failure vs. never having enough input matches to try),
        // not part of the tracking contract itself.
        debug_log: &mut Vec<String>,
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
                                    pose_init: &Pose3d,
                                    source: &'static str,
                                    debug_log: &mut Vec<String>|
         -> Result<Estimate, MapProjectionRejectReason> {
            let input_n = correspondences.len();
            // Independent 2D geometric-consistency check, ahead of PnP and
            // independent of any 3D map-point position or pose estimate: a
            // correspondence whose reference-keyframe/current-frame pixel
            // pair doesn't fit *any* fundamental matrix that the bulk of the
            // other pairs agree on is very likely a bad match, regardless of
            // whether the map/pose happens to also think it's consistent.
            let correspondences = self.geometric_consistency_filter(
                current_kf,
                &correspondences,
                &curr_keypoints_undist,
                camera,
            );
            let after_geom_n = correspondences.len();

            // Two attempts at making this source-aware for `source == "klt"`
            // have been tried and reverted, both measured via V103
            // `--evaluate` against repeated baselines:
            //   1. Solve via `solve_pnp_ransac` (EPnP+RANSAC, no dependency
            //      on `pose_init`): ATE RMSE ~1.1m -> ~936m, map diverging
            //      into the hundreds of meters, repeated full resets.
            //      EPnP-via-RANSAC has no way to reject a pose that's
            //      self-consistent with a small, weakly-conditioned
            //      correspondence set (common here — many KLT attempts had
            //      only 10-30 points) but globally wrong.
            //   2. Keep `pose_init` as the seed, but widen the coarse prior
            //      gate (unbounded, then 120px) and add a Huber loss to the
            //      LM solve itself. Unbounded: ATE RMSE 1.8-4.1m / 1-3
            //      resets across repeats. 120px: 1.5-2.3m / 1 reset, and
            //      V101 regressed separately (3.4m ATE / 14% drift vs. a
            //      0.68m baseline). Huber only protects against a *minority*
            //      outlier fraction within an otherwise-good correspondence
            //      set; it doesn't stop the LM solve (a local method) from
            //      confidently converging to a pose that fits the surviving
            //      correspondences well locally without being the true
            //      global pose when `pose_init` is far off — that's a
            //      quieter failure than RANSAC's divergence but still a net
            //      accuracy regression, and tuning the gate width alone
            //      didn't fix it.
            // Both reverted; `source` is solved identically to the other
            // sources via `self.config.pnp` below. `PnpConfig` now also has
            // `outlier_rounds` (multi-round hard-exclusion refit, mirroring
            // lightweight_vio's `PnPOptimizer::optimize_pose`) and a
            // `converged` check on the LM result (mirrors their "only
            // commit if Ceres reports CONVERGENCE, else keep previous
            // pose") — see `pnp::solve_pnp_with_diagnostics`. Both default
            // to today's single-pass behavior (`outlier_rounds: 1`) for
            // every existing caller; re-verify via `--evaluate` on all six
            // sequences before turning `outlier_rounds` up or revisiting a
            // widened `klt_pnp` gate with this in place.
            let (pnp_result, diag) = self.solve_pnp(
                map.map_points(),
                &correspondences,
                &curr_keypoints_undist,
                camera,
                pose_init,
                search_scale,
            );
            let diag_msg = format!(
                "prior_survivors={}/{} (need {}) last_round_active={} converged={:?}",
                diag.prior_survivors,
                diag.input,
                pnp.min_correspondences,
                diag.last_round_active,
                diag.converged
            );
            let Some((mut pose, mut inliers)) = pnp_result else {
                debug_log.push(format!(
                    "[track_diag] {source}: in={input_n} after_geom={after_geom_n} {diag_msg}"
                ));
                return Err(MapProjectionRejectReason::PnpFailed);
            };
            if inliers < pnp.min_inliers_early {
                debug_log.push(format!(
                    "[track_diag] {source}: in={input_n} after_geom={after_geom_n} {diag_msg} \
                     early_inliers={inliers} (need {})",
                    pnp.min_inliers_early
                ));
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

        // KLT-seeded correspondences, tried first when available — see the
        // `pre_seeded` doc above for why this jumps the queue ahead of the
        // pose-dependent projection search.
        match pre_seeded {
            Some(seeded) if seeded.len() >= pnp.min_correspondences => {
                if let Ok(estimate) = try_track_and_refine(seeded, candidate_pose, "klt", debug_log)
                {
                    return Ok(estimate);
                }
            }
            Some(seeded) => {
                debug_log.push(format!(
                    "[track_diag] klt: seeded={} below min_correspondences={}",
                    seeded.len(),
                    pnp.min_correspondences
                ));
            }
            None => debug_log.push(
                "[track_diag] klt: not attempted (disabled, no prev frame, or empty track_state)"
                    .to_string(),
            ),
        }

        // PnP from projection matches.
        let last_reject = if projection_matches.len() >= pnp.min_correspondences {
            match try_track_and_refine(projection_matches, candidate_pose, "projection", debug_log)
            {
                Ok(estimate) => return Ok(estimate),
                Err(reason) => reason,
            }
        } else {
            debug_log.push(format!(
                "[track_diag] projection: matches={} below min_correspondences={}",
                projection_matches.len(),
                pnp.min_correspondences
            ));
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
            debug_log.push(format!(
                "[track_diag] reference: ref_matches={} below MIN_REF_MATCHES={MIN_REF_MATCHES}",
                ref_matches.len()
            ));
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
            debug_log.push(format!(
                "[track_diag] reference: correspondences={} below min_correspondences={}",
                ref_correspondences.len(),
                pnp.min_correspondences
            ));
            return Err(MapProjectionRejectReason::LowReferenceCorrespondences);
        }

        try_track_and_refine(
            ref_correspondences,
            pose_before_tracking,
            "reference",
            debug_log,
        )
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

        let (new_pose, inliers) = self
            .solve_pnp(
                map.map_points(),
                &global_matches,
                curr_keypoints_undist,
                camera,
                pose_init,
                search_scale,
            )
            .0?;
        Some(Estimate {
            pose: new_pose,
            inliers,
            matches: global_matches,
        })
    }

    /// Filters `correspondences` (map-point-index, current-frame-keypoint-index
    /// pairs) by fundamental-matrix RANSAC between each map point's pixel
    /// observation in the reference keyframe and its matched pixel in the
    /// current frame — mirrors lightweight_vio's per-frame
    /// `apply_fundamental_matrix_filter`, adapted to kornia-slam's
    /// reference-keyframe-centric tracking (lightweight_vio checks against
    /// the immediately previous frame; here the reference keyframe plays
    /// that role, since that's what per-frame correspondences are already
    /// matched against).
    ///
    /// Unlike the PnP inlier check this runs *before* it and needs neither a
    /// 3D map-point position nor a pose estimate — it only asks whether the
    /// bulk of the 2D-2D pixel pairs agree on *some* consistent epipolar
    /// geometry. A pair that doesn't is very likely a bad match regardless
    /// of what the map/pose currently believe, so this catches a class of
    /// outlier the existing Hamming-distance + reprojection-radius gates
    /// don't: one that's a plausible descriptor match and a plausible
    /// reprojection under a *wrong* pose, but geometrically inconsistent
    /// with where the rest of the frame's matches say the camera actually
    /// moved.
    ///
    /// A correspondence whose map point has no observation in
    /// `current_kf` (e.g. it's only observed by covisible neighbors) can't
    /// be checked this way and is kept unfiltered, same as
    /// lightweight_vio's own `tracked_id < 0` skip. Also a no-op (returns
    /// `correspondences` unchanged) if there are too few checkable pairs for
    /// an 8-point solve, or if RANSAC itself can't find a confident fit —
    /// in both cases there's no basis to reject anything.
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
            .filter_map(|(desc_idx, mp_opt)| mp_opt.map(|mp_idx| (mp_idx, desc_idx)))
            .collect();

        let mut checkable_indices = Vec::new();
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for (i, &(mp_idx, kp_idx)) in correspondences.iter().enumerate() {
            let Some(&desc_idx) = mp_to_desc.get(&mp_idx) else {
                continue;
            };
            let Some(ref_pt) = current_kf.frame.undistorted_xy(desc_idx, camera) else {
                continue;
            };
            let Some(&cur_pt) = curr_keypoints_undist.get(kp_idx) else {
                continue;
            };
            checkable_indices.push(i);
            x1.push(Vec2F64::new(ref_pt[0] as f64, ref_pt[1] as f64));
            x2.push(Vec2F64::new(cur_pt[0] as f64, cur_pt[1] as f64));
        }

        if checkable_indices.len() < MIN_PAIRS_FOR_FILTER {
            return correspondences.to_vec();
        }

        let params = RansacParams {
            threshold: self.config.geometric_filter_threshold_px,
            ..RansacParams::default()
        };
        let Ok(result) = ransac_fundamental(&x1, &x2, &params) else {
            return correspondences.to_vec();
        };

        let mut keep = vec![true; correspondences.len()];
        for (&i, &is_inlier) in checkable_indices.iter().zip(result.inliers.iter()) {
            if !is_inlier {
                keep[i] = false;
            }
        }
        correspondences
            .iter()
            .zip(keep.iter())
            .filter_map(|(&c, &k)| k.then_some(c))
            .collect()
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
    ) -> (Option<(Pose3d, usize)>, pnp::PnpDiagnostics) {
        let (points_world, points_image) =
            Self::gather_points(map_points, correspondences, keypoints_undist);
        let pnp_config = if search_scale > 1.0 {
            PnpConfig {
                prior_reproj_threshold_px: self.config.pnp.prior_reproj_threshold_px
                    * search_scale as f64,
                ..self.config.pnp.clone()
            }
        } else {
            self.config.pnp.clone()
        };
        pnp::solve_pnp_with_diagnostics(
            &points_world,
            &points_image,
            camera,
            pose_init,
            &pnp_config,
        )
    }

    /// Resolves correspondences (map-point-index, keypoint-index pairs) into
    /// parallel 3D-world / 2D-image point arrays, dropping any pair whose
    /// index is out of range.
    fn gather_points(
        map_points: &[MapPoint],
        correspondences: &[(usize, usize)],
        keypoints_undist: &[[f32; 2]],
    ) -> (Vec<Vec3F64>, Vec<Vec2F32>) {
        let mut points_world = Vec::with_capacity(correspondences.len());
        let mut points_image = Vec::with_capacity(correspondences.len());
        for &(mp_idx, kp_idx) in correspondences {
            if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
                points_world.push(mp.position);
                points_image.push(Vec2F32::new(kp[0], kp[1]));
            }
        }
        (points_world, points_image)
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
