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

/// Acceptance thresholds for two-view initialization.
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

impl TwoViewInitConfig {
    fn validate(&self) -> Result<(), TwoViewRejectReason> {
        let acceptance = &self.acceptance_config;
        if acceptance.min_matches == 0 {
            return Err(TwoViewRejectReason::InvalidConfig(
                "min_matches must be greater than zero".into(),
            ));
        }
        if acceptance.min_inliers > acceptance.min_matches {
            return Err(TwoViewRejectReason::InvalidConfig(
                "min_inliers cannot exceed min_matches".into(),
            ));
        }
        if acceptance.min_triangulated > acceptance.min_matches {
            return Err(TwoViewRejectReason::InvalidConfig(
                "min_triangulated cannot exceed min_matches".into(),
            ));
        }
        if !self.match_config.nn_ratio.is_finite()
            || self.match_config.nn_ratio <= 0.0
            || self.match_config.nn_ratio > 1.0
        {
            return Err(TwoViewRejectReason::InvalidConfig(
                "descriptor nn_ratio must be finite and in (0, 1]".into(),
            ));
        }
        if self.match_config.check_orientation && self.match_config.histo_length == 0 {
            return Err(TwoViewRejectReason::InvalidConfig(
                "orientation histogram length must be greater than zero".into(),
            ));
        }

        let triangulation = &self.triangulation_config;
        for (name, value) in [
            ("min_parallax_deg", triangulation.min_parallax_deg),
            ("max_midpoint_gap", triangulation.max_midpoint_gap),
            (
                "max_reprojection_error",
                triangulation.max_reprojection_error,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(TwoViewRejectReason::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if triangulation.min_cheirality_count == 0 {
            return Err(TwoViewRejectReason::InvalidConfig(
                "min_cheirality_count must be greater than zero".into(),
            ));
        }
        if !triangulation.cheirality_ambiguity_max.is_finite()
            || triangulation.cheirality_ambiguity_max <= 0.0
            || triangulation.cheirality_ambiguity_max > 1.0
        {
            return Err(TwoViewRejectReason::InvalidConfig(
                "cheirality_ambiguity_max must be finite and in (0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Two-view initialization rejection reason.
#[derive(Debug, thiserror::Error)]
pub enum TwoViewRejectReason {
    /// Invalid matcher, geometry, or acceptance thresholds.
    #[error("invalid two-view initialization configuration: {0}")]
    InvalidConfig(String),
    /// Not enough descriptor matches to run two-view estimation.
    #[error("not enough descriptor matches: found {found}, need at least {required}")]
    LowMatches { found: usize, required: usize },
    /// Two-view estimation failed; the wrapped error carries the specific
    /// cause (RANSAC failure, ambiguous cheirality from pure rotation /
    /// planar / low-parallax motion, solver error, …).
    #[error("two-view estimation failed: {0}")]
    EstimationFailed(TwoViewError),
    /// Too few triangulated points.
    #[error("too few triangulated points: found {found}, need at least {required}")]
    LowTriangulated { found: usize, required: usize },
    /// Too few inliers in estimated model.
    #[error("too few model inliers: found {found}, need at least {required}")]
    LowInliers { found: usize, required: usize },
    /// Not enough parallax.
    #[error("insufficient parallax: {actual_deg:.3}deg, need at least {required_deg:.3}deg")]
    LowParallax { actual_deg: f64, required_deg: f64 },
}

/// Bootstrap-specific pose, correspondences, and triangulated geometry.
#[derive(Debug, Clone)]
pub struct TwoViewEstimate {
    /// Estimated world-to-camera pose for the current frame.
    pub pose: Pose3d,
    /// Matched pairs `(reference_keypoint_idx, current_keypoint_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// Number of inlier correspondences supporting the selected model.
    pub inliers: usize,
    /// Triangulated 3D points in the reference camera frame.
    pub points3d: Vec<Vec3F64>,
    /// Indices into `matches` that were inliers in two-view estimation.
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
    config.validate()?;
    let acceptance = &config.acceptance_config;

    let matches = match_orb_descriptors(
        &ref_features.orientations,
        &ref_features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        config.match_config,
    );
    if matches.len() < acceptance.min_matches {
        return Err(TwoViewRejectReason::LowMatches {
            found: matches.len(),
            required: acceptance.min_matches,
        });
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
        return Err(TwoViewRejectReason::LowTriangulated {
            found: result.points3d.len(),
            required: acceptance.min_triangulated,
        });
    }

    if result.inlier_indices.len() < acceptance.min_inliers {
        return Err(TwoViewRejectReason::LowInliers {
            found: result.inlier_indices.len(),
            required: acceptance.min_inliers,
        });
    }

    let median_parallax_deg = result.median_parallax_deg(&reference_pts, &current_pts, camera);
    if median_parallax_deg < config.triangulation_config.min_parallax_deg {
        return Err(TwoViewRejectReason::LowParallax {
            actual_deg: median_parallax_deg,
            required_deg: config.triangulation_config.min_parallax_deg,
        });
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
        pose,
        matches,
        inliers,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_features() -> OrbFeatures {
        OrbFeatures {
            keypoints_xy: Vec::new(),
            orientations: Vec::new(),
            descriptors: Vec::new(),
            octaves: Vec::new(),
        }
    }

    #[test]
    fn low_match_rejection_reports_counts() {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let config = TwoViewInitConfig::default();

        let result = try_initialize_two_view(
            &empty_features(),
            &Pose3d::IDENTITY,
            &empty_features(),
            &camera,
            &config,
        );

        assert!(matches!(
            result,
            Err(TwoViewRejectReason::LowMatches {
                found: 0,
                required: 100
            })
        ));
    }

    #[test]
    fn invalid_configuration_is_rejected_before_matching() {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut config = TwoViewInitConfig::default();
        config.match_config.histo_length = 0;

        let result = try_initialize_two_view(
            &empty_features(),
            &Pose3d::IDENTITY,
            &empty_features(),
            &camera,
            &config,
        );

        assert!(matches!(result, Err(TwoViewRejectReason::InvalidConfig(_))));
    }
}
