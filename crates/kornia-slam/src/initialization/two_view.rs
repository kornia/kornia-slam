//! Two-view geometric initialization for monocular tracking.
//              (3D point)
//                  X
//                /   \
//               /     \
//              /       \
//             /         \
//            /           \
//      .-----------. .-----------.
//     /   x1 *    / /    * x2   /
//    /           / /           /
//    '-----------' '-----------'
//      frame 1        frame 2

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
pub use kornia_3d::pose::TwoViewError;
use kornia_3d::pose::{TriangulationConfig, TwoViewEstimator, TwoViewModel};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbFeatures, OrbMatchConfig, match_orb_descriptors};

use crate::tracking::pose_estimation::Estimate;

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewAcceptanceConfig {
    /// Minimum descriptor matches required before two-view estimation.
    pub min_matches: usize,
    /// Minimum inlier count required to accept initialization.
    pub min_inliers: usize,
    /// Minimum triangulated points required to accept initialization.
    pub min_triangulated: usize,
}

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewInitConfig {
    /// ORB descriptor matcher settings.
    pub match_config: OrbMatchConfig,
    /// Triangulation thresholds applied during two-view estimation.
    pub triangulation_config: TriangulationConfig,
    /// Acceptance thresholds applied on top of the estimator result.
    pub acceptance_config: TwoViewAcceptanceConfig,
}

impl Default for TwoViewInitConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            // Match ORB-SLAM3's `secondBestGood < 0.75 * bestGood` cheirality
            // ambiguity threshold (default in kornia-3d is 0.70).
            triangulation_config: TriangulationConfig {
                cheirality_ambiguity_max: 0.75,
                ..TriangulationConfig::default()
            },
            acceptance_config: TwoViewAcceptanceConfig {
                min_matches: 100,
                min_inliers: 30,
                min_triangulated: 50,
            },
        }
    }
}

/// Two-view initialization rejection reason.
#[derive(Debug)]
pub enum TwoViewRejectReason {
    /// Not enough descriptor matches to run two-view estimation.
    LowMatches,
    /// Two-view estimation failed; the wrapped error carries the specific
    /// cause (RANSAC failure, ambiguous cheirality from pure rotation /
    /// planar / low-parallax motion, solver error, …).
    EstimationFailed(TwoViewError),
    /// Too few triangulated points.
    LowTriangulated,
    /// Too few inliers in estimated model.
    LowInliers,
    /// Not enough parallax.
    LowParallax,
}

/// Bootstrap-specific data produced alongside the shared [`Estimate`].
#[derive(Debug, Clone)]
pub struct TwoViewEstimate {
    /// Shared pose estimate.
    pub estimate: Estimate,
    /// Triangulated 3D points in the reference camera frame.
    pub points3d: Vec<Vec3F64>,
    /// Indices into `estimate.matches` that were inliers in two-view estimation.
    pub inlier_indices: Vec<usize>,
    /// Median positive depth in the two-view triangulation (if available).
    pub median_depth: Option<f64>,
    /// Which model the estimator committed to: `'F'` (fundamental/essential)
    /// or `'H'` (homography). Useful for diagnostics on planar scenes.
    pub model_kind: char,
}

/// Attempt two-view initialization between a reference frame and the current frame.
pub fn try_initialize_two_view(
    ref_features: &OrbFeatures,
    ref_pose: &Pose3d,
    curr_features: &OrbFeatures,
    camera: &PinholeCamera,
    config: &TwoViewInitConfig,
) -> Result<TwoViewEstimate, TwoViewRejectReason> {
    let acceptance = &config.acceptance_config;

    let matches = match_orb_descriptors(
        &ref_features.orientations,
        &ref_features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        config.match_config,
    );
    if matches.len() < acceptance.min_matches {
        return Err(TwoViewRejectReason::LowMatches);
    }

    let (reference_pts, current_pts) = camera.undistort_matched_pairs(
        &ref_features.keypoints_xy,
        &curr_features.keypoints_xy,
        &matches,
    );

    let k = camera.intrinsic_matrix();
    let estimator = TwoViewEstimator::builder()
        .triangulation(config.triangulation_config.clone())
        .build();
    let result = estimator
        .estimate(&reference_pts, &current_pts, &k, &k)
        .map_err(TwoViewRejectReason::EstimationFailed)?;

    let model_kind = match result.model {
        TwoViewModel::Homography(_) => 'H',
        _ => 'F',
    };

    if result.points3d.len() < acceptance.min_triangulated {
        return Err(TwoViewRejectReason::LowTriangulated);
    }

    if result.inlier_indices.len() < acceptance.min_inliers {
        return Err(TwoViewRejectReason::LowInliers);
    }

    let median_parallax_deg = result.median_parallax_deg(&reference_pts, &current_pts, camera);
    if median_parallax_deg < config.triangulation_config.min_parallax_deg {
        return Err(TwoViewRejectReason::LowParallax);
    }

    let mut t_scaled = result.translation;
    let mut depths: Vec<f64> = result
        .points3d
        .iter()
        .map(|p| p.z)
        .filter(|&z| z > 0.0)
        .collect();
    let median_depth = median_in_place(&mut depths).filter(|&d| d > 1e-6);
    if let Some(md) = median_depth {
        t_scaled /= md;
    }

    let pose = Pose3d::new(
        result.rotation * ref_pose.rotation,
        result.rotation * ref_pose.translation + t_scaled,
    );
    let inliers = result.inlier_indices.len();

    Ok(TwoViewEstimate {
        estimate: Estimate {
            pose,
            matches,
            inliers,
        },
        points3d: result.points3d,
        inlier_indices: result.inlier_indices,
        median_depth,
        model_kind,
    })
}

/// Computes the median of a mutable slice in-place.
fn median_in_place(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    Some(values[mid])
}
