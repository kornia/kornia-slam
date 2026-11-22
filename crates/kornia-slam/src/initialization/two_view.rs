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
use kornia_3d::pose::{TriangulationConfig, TwoViewEstimator, TwoViewModel, TwoViewResult};
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_imgproc::features::{OrbFeatures, OrbMatchConfig, match_orb_descriptors};

use crate::frame::undistorted_camera;
use crate::tracking::pose_estimation::Estimate;

/// Acceptance thresholds of two-view initialization.
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
                max_midpoint_gap: 0.25,
                max_reprojection_error: 3.0,
                ..TriangulationConfig::default()
            },
            acceptance_config: TwoViewAcceptanceConfig::default(),
        }
    }
}

impl Default for TwoViewAcceptanceConfig {
    fn default() -> Self {
        Self {
            min_matches: 100,
            min_inliers: 30,
            min_triangulated: 50,
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

    let mut matches = match_orb_descriptors(
        &ref_features.orientations,
        &ref_features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        config.match_config,
    );
    if matches.len() < acceptance.min_matches {
        return Err(TwoViewRejectReason::LowMatches);
    }

    let (reference_pts, current_pts) = undistort_matches(
        camera,
        &ref_features.keypoints_xy,
        &curr_features.keypoints_xy,
        &mut matches,
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

    let median_parallax_deg = median_parallax_deg(&result, &reference_pts, &current_pts, camera);
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

/// Undistorts both keypoints of each match, dropping from `matches` the pairs
/// `camera` cannot undistort so that `matches` and the returned points stay
/// index-aligned.
fn undistort_matches(
    camera: &PinholeCamera,
    reference: &[[f32; 2]],
    current: &[[f32; 2]],
    matches: &mut Vec<(usize, usize)>,
) -> (Vec<Vec2F64>, Vec<Vec2F64>) {
    let undistort = |keypoints: &[[f32; 2]], i: usize| {
        let kp = keypoints.get(i)?;
        camera.undistort(kp[0] as f64, kp[1] as f64)
    };
    let mut reference_pts = Vec::with_capacity(matches.len());
    let mut current_pts = Vec::with_capacity(matches.len());
    matches.retain(
        |&(r, c)| match (undistort(reference, r), undistort(current, c)) {
            (Some(p), Some(q)) => {
                reference_pts.push(p);
                current_pts.push(q);
                true
            }
            _ => false,
        },
    );
    (reference_pts, current_pts)
}

/// Median inlier parallax, in degrees, of correspondences already undistorted
/// through `camera`.
fn median_parallax_deg(
    result: &TwoViewResult,
    reference: &[Vec2F64],
    current: &[Vec2F64],
    camera: &PinholeCamera,
) -> f64 {
    result.median_parallax_deg(reference, current, &undistorted_camera(camera))
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
    use kornia_3d::camera::CameraModel;
    use kornia_algebra::Mat3F64;

    fn euroc_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.283_408_11,
            k2: 0.073_959_07,
            p1: 0.000_193_59,
            p2: 1.761_871_14e-5,
        }
    }

    #[test]
    fn parallax_of_undistorted_points_is_the_angle_between_the_true_bearings() {
        let camera = euroc_camera();
        let reference = Vec3F64::new(1.0, 0.6, 2.0);
        let current = reference + Vec3F64::new(-0.3, 0.0, 0.0);
        let undistort = |p_cam: &Vec3F64| {
            let pixel = camera.project(p_cam).unwrap();
            camera.undistort(pixel.x, pixel.y).unwrap()
        };
        let result = TwoViewResult {
            model: TwoViewModel::Fundamental(Mat3F64::IDENTITY),
            rotation: Mat3F64::IDENTITY,
            translation: Vec3F64::new(1.0, 0.0, 0.0),
            points3d: vec![reference],
            inlier_indices: vec![0],
            inliers: vec![true],
        };

        let parallax = median_parallax_deg(
            &result,
            &[undistort(&reference)],
            &[undistort(&current)],
            &camera,
        );

        let expected = reference
            .normalize()
            .dot(current.normalize())
            .acos()
            .to_degrees();
        assert!(
            (parallax - expected).abs() < 1e-6,
            "{parallax} vs {expected}"
        );
    }

    #[test]
    fn undistorting_matches_drops_unusable_pairs_and_keeps_the_rest_aligned() {
        // k1 = -0.5 folds the radial map at r = 0.82, so no point images
        // beyond a distorted radius of 0.54: (600, 240) has no preimage.
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.5,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let reference = [[300.0, 200.0], [600.0, 240.0], [340.0, 260.0]];
        let current = [[310.0, 210.0], [330.0, 250.0], [350.0, 270.0]];
        let mut matches = vec![(0, 0), (1, 1), (2, 2)];

        let (reference_pts, current_pts) =
            undistort_matches(&camera, &reference, &current, &mut matches);

        assert_eq!(matches, vec![(0, 0), (2, 2)]);
        for (k, &(r, c)) in matches.iter().enumerate() {
            let [ru, rv] = reference[r];
            let [cu, cv] = current[c];
            assert_eq!(
                Some(reference_pts[k]),
                camera.undistort(ru as f64, rv as f64)
            );
            assert_eq!(Some(current_pts[k]), camera.undistort(cu as f64, cv as f64));
        }
    }
}
