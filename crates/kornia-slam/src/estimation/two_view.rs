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
use kornia_3d::pose::{
    TwoViewConfig, TwoViewDiagnostics, TwoViewModel, two_view_estimate_with_diagnostics,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_imgproc::features::{
    OrbFeatures, OrbMatchConfig, hamming_distance, match_orb_descriptors,
};

use super::Estimate;

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewAcceptanceConfig {
    /// Minimum descriptor matches required before two-view estimation.
    pub min_matches: usize,
    /// Minimum inlier count required to accept initialization.
    pub min_inliers: usize,
    /// Minimum triangulated points required to accept initialization.
    pub min_triangulated: usize,
    /// Minimum median parallax required to accept initialization.
    pub min_parallax_deg: f64,
}

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewInitConfig {
    /// ORB descriptor matcher settings.
    pub match_config: OrbMatchConfig,
    /// ORB-SLAM3 initialization matcher search window in pixels.
    pub initialization_window_px: Option<f32>,
    /// Two-view model estimation settings.
    pub estimation_config: TwoViewConfig,
    /// Acceptance thresholds applied on top of the estimator result.
    pub acceptance_config: TwoViewAcceptanceConfig,
}

impl Default for TwoViewInitConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.9,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            initialization_window_px: Some(100.0),
            estimation_config: {
                let mut cfg = TwoViewConfig::default();
                cfg.triangulation.use_orb_slam3_check_rt = true;
                cfg.triangulation.check_rt_reproj_th_sq = 4.0;
                cfg.triangulation.min_cheirality_count = 1;
                // ORB-SLAM3 chi-square thresholds (sigma=1 px): sqrt(3.841) ≈ 1.96 for F,
                // sqrt(5.991) ≈ 2.45 for H. Our RANSAC threshold is compared against the
                // squared residual internally, so pass the linear threshold here.
                cfg.ransac_f.threshold = 3.841_f64.sqrt();
                cfg.ransac_h.threshold = 5.991_f64.sqrt();
                cfg
            },
            acceptance_config: TwoViewAcceptanceConfig {
                min_matches: 100,
                min_inliers: 30,
                min_triangulated: 50,
                min_parallax_deg: 1.0,
            },
        }
    }
}

/// Two-view initialization rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoViewRejectReason {
    /// Not enough descriptor matches to run two-view estimation.
    LowMatches,
    /// Two-view estimation failed.
    EstimationFailed,
    /// Homography was selected instead of fundamental model.
    WrongModel,
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
}

/// Non-behavioral counters from a two-view initialization attempt.
#[derive(Debug, Clone)]
pub struct TwoViewInitReport {
    /// Descriptor matches before geometric estimation.
    pub matches: usize,
    /// Selected model name, or `none` if no model was estimated.
    pub model: &'static str,
    /// Geometric inliers from the selected model.
    pub inliers: usize,
    /// Accepted triangulated point count.
    pub triangulated: usize,
    /// Median positive triangulated depth.
    pub median_depth: Option<f64>,
    /// Rejection reason, if initialization failed.
    pub reject_reason: Option<TwoViewRejectReason>,

    /// Was two-view estimation attempted at all (matches >= min_matches).
    pub two_view_ran: bool,
    /// Did the pipeline accept the geometry (returns Ok).
    pub two_view_accepted: bool,
    /// Fundamental RANSAC inliers.
    pub inliers_f: usize,
    /// Homography RANSAC inliers.
    pub inliers_h: usize,
    /// ORB-SLAM3-style homography score (chi-square capped) in pixel units, sigma=1.
    pub orb_slam3_sh: f64,
    /// ORB-SLAM3-style fundamental score in pixel units, sigma=1.
    pub orb_slam3_sf: f64,
    /// ORB-SLAM3-style RH = SH / (SH + SF).
    pub orb_slam3_rh: f64,
    /// Per-candidate triangulation counts for the selected model.
    pub candidate_ngood: Vec<usize>,
    /// Best candidate index inside `candidate_ngood`, if any.
    pub best_candidate_idx: Option<usize>,
    /// Second-best candidate count.
    pub second_best_good: usize,
    /// Median parallax of the selected pose (degrees).
    pub median_parallax_deg: f64,
}

impl Default for TwoViewInitReport {
    fn default() -> Self {
        Self {
            matches: 0,
            model: "none",
            inliers: 0,
            triangulated: 0,
            median_depth: None,
            reject_reason: None,
            two_view_ran: false,
            two_view_accepted: false,
            inliers_f: 0,
            inliers_h: 0,
            orb_slam3_sh: 0.0,
            orb_slam3_sf: 0.0,
            orb_slam3_rh: 0.0,
            candidate_ngood: Vec::new(),
            best_candidate_idx: None,
            second_best_good: 0,
            median_parallax_deg: 0.0,
        }
    }
}

/// Attempt two-view initialization between a reference frame and the current frame.
pub fn try_initialize_two_view(
    ref_features: &OrbFeatures,
    ref_pose: &Pose3d,
    curr_features: &OrbFeatures,
    camera: &PinholeCamera,
    config: &TwoViewInitConfig,
) -> Result<TwoViewEstimate, TwoViewRejectReason> {
    try_initialize_two_view_with_report(ref_features, ref_pose, curr_features, camera, config).0
}

/// Attempt two-view initialization and return trace counters alongside the result.
pub fn try_initialize_two_view_with_report(
    ref_features: &OrbFeatures,
    ref_pose: &Pose3d,
    curr_features: &OrbFeatures,
    camera: &PinholeCamera,
    config: &TwoViewInitConfig,
) -> (
    Result<TwoViewEstimate, TwoViewRejectReason>,
    TwoViewInitReport,
) {
    let acceptance = &config.acceptance_config;
    let mut report = TwoViewInitReport::default();

    let matches = if let Some(window_px) = config.initialization_window_px {
        match_orb_descriptors_for_initialization(
            ref_features,
            curr_features,
            config.match_config,
            window_px,
        )
    } else {
        match_orb_descriptors(
            &ref_features.orientations,
            &ref_features.descriptors,
            &curr_features.orientations,
            &curr_features.descriptors,
            config.match_config,
        )
    };
    report.matches = matches.len();
    if matches.len() < acceptance.min_matches {
        report.reject_reason = Some(TwoViewRejectReason::LowMatches);
        return (Err(TwoViewRejectReason::LowMatches), report);
    }

    let (reference_pts, current_pts) = camera.undistort_matched_pairs(
        &ref_features.keypoints_xy,
        &curr_features.keypoints_xy,
        &matches,
    );

    let k = camera.intrinsic_matrix();
    report.two_view_ran = true;
    let estimation = two_view_estimate_with_diagnostics(
        &reference_pts,
        &current_pts,
        &k,
        &k,
        &config.estimation_config,
    )
    .map_err(|_| TwoViewRejectReason::EstimationFailed);
    let (result, diagnostics) = match estimation {
        Ok(pair) => pair,
        Err(reason) => {
            report.reject_reason = Some(reason);
            fill_diagnostics_empty(&mut report, &reference_pts, &current_pts);
            return (Err(reason), report);
        }
    };

    fill_diagnostics(&mut report, &diagnostics, &reference_pts, &current_pts);

    report.model = two_view_model_name(result.model);
    report.inliers = result.inlier_indices.len();
    report.triangulated = result.points3d.len();
    // Use ORB-SLAM3's sorted-cos-parallax-at-index-min(50,N-1) from CheckRT,
    // not a per-inlier bearing-angle median. Falls back to the bearing median
    // if CheckRT wasn't used (flag off in some configs).
    report.median_parallax_deg = if diagnostics.best_candidate_parallax_deg > 0.0 {
        diagnostics.best_candidate_parallax_deg
    } else {
        result.median_parallax_deg(&reference_pts, &current_pts, camera)
    };

    if !matches!(result.model, TwoViewModel::Fundamental(_)) {
        report.reject_reason = Some(TwoViewRejectReason::WrongModel);
        return (Err(TwoViewRejectReason::WrongModel), report);
    }

    if result.points3d.len() < acceptance.min_triangulated {
        report.reject_reason = Some(TwoViewRejectReason::LowTriangulated);
        return (Err(TwoViewRejectReason::LowTriangulated), report);
    }

    if result.inlier_indices.len() < acceptance.min_inliers {
        report.reject_reason = Some(TwoViewRejectReason::LowInliers);
        return (Err(TwoViewRejectReason::LowInliers), report);
    }

    let median_parallax_deg = report.median_parallax_deg;
    if median_parallax_deg < acceptance.min_parallax_deg {
        report.reject_reason = Some(TwoViewRejectReason::LowParallax);
        return (Err(TwoViewRejectReason::LowParallax), report);
    }

    let mut t_scaled = result.translation;
    let mut depths: Vec<f64> = result
        .points3d
        .iter()
        .map(|p| p.z)
        .filter(|&z| z > 0.0)
        .collect();
    let median_depth = median_in_place(&mut depths).filter(|&d| d > 1e-6);
    report.median_depth = median_depth;
    if let Some(md) = median_depth {
        t_scaled /= md;
    }

    let pose = Pose3d::new(
        result.rotation * ref_pose.rotation,
        result.rotation * ref_pose.translation + t_scaled,
    );
    let inliers = result.inlier_indices.len();

    let estimate = TwoViewEstimate {
        estimate: Estimate {
            pose,
            matches,
            inliers,
        },
        points3d: result.points3d,
        inlier_indices: result.inlier_indices,
        median_depth,
    };

    report.two_view_accepted = true;
    (Ok(estimate), report)
}

fn fill_diagnostics(
    report: &mut TwoViewInitReport,
    diag: &TwoViewDiagnostics,
    ref_pts: &[Vec2F64],
    cur_pts: &[Vec2F64],
) {
    report.inliers_f = diag.f_inlier_count;
    report.inliers_h = diag.h_inlier_count;
    report.candidate_ngood = diag.candidate_ngood.clone();
    report.best_candidate_idx = diag.best_candidate_idx;
    report.second_best_good = diag.second_best_good;
    let sh = orb_slam3_homography_score(&diag.h_matrix, ref_pts, cur_pts);
    let sf = orb_slam3_fundamental_score(&diag.f_matrix, ref_pts, cur_pts);
    report.orb_slam3_sh = sh;
    report.orb_slam3_sf = sf;
    report.orb_slam3_rh = if sh + sf > 0.0 { sh / (sh + sf) } else { 0.0 };
}

fn fill_diagnostics_empty(
    report: &mut TwoViewInitReport,
    _ref_pts: &[Vec2F64],
    _cur_pts: &[Vec2F64],
) {
    // Called when RANSAC itself failed; leave diagnostics at defaults.
    report.candidate_ngood.clear();
    report.best_candidate_idx = None;
}

/// ORB-SLAM3 symmetric-transfer score for a homography (CheckHomography).
/// Assumes sigma = 1 px and chi-square threshold 5.991.
fn orb_slam3_homography_score(h: &Mat3F64, x1: &[Vec2F64], x2: &[Vec2F64]) -> f64 {
    let th = 5.991;
    let inv = match invert_mat3(h) {
        Some(m) => m,
        None => return 0.0,
    };
    let mut score = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // x2in1 = H^-1 * x2
        let w2 = inv.col(0).x * p2.x + inv.col(1).x * p2.y + inv.col(2).x;
        let w = inv.col(0).y * p2.x + inv.col(1).y * p2.y + inv.col(2).y;
        let denom = inv.col(0).z * p2.x + inv.col(1).z * p2.y + inv.col(2).z;
        if denom.abs() < 1e-12 {
            continue;
        }
        let u = w2 / denom;
        let v = w / denom;
        let d2_1 = (p1.x - u) * (p1.x - u) + (p1.y - v) * (p1.y - v);
        if d2_1 <= th {
            score += th - d2_1;
        }

        // x1in2 = H * x1
        let u2 = h.col(0).x * p1.x + h.col(1).x * p1.y + h.col(2).x;
        let v2 = h.col(0).y * p1.x + h.col(1).y * p1.y + h.col(2).y;
        let w_fwd = h.col(0).z * p1.x + h.col(1).z * p1.y + h.col(2).z;
        if w_fwd.abs() < 1e-12 {
            continue;
        }
        let u2p = u2 / w_fwd;
        let v2p = v2 / w_fwd;
        let d2_2 = (p2.x - u2p) * (p2.x - u2p) + (p2.y - v2p) * (p2.y - v2p);
        if d2_2 <= th {
            score += th - d2_2;
        }
    }
    score
}

/// ORB-SLAM3 symmetric-transfer score for a fundamental matrix (CheckFundamental).
/// Sigma = 1 px, chi-square threshold per residual 3.841, bonus threshold 5.991.
fn orb_slam3_fundamental_score(f: &Mat3F64, x1: &[Vec2F64], x2: &[Vec2F64]) -> f64 {
    let th = 3.841;
    let th_score = 5.991;
    let mut score = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // l2 = F * x1; residual = (x2 . l2)^2 / (a2^2 + b2^2)
        let a2 = f.col(0).x * p1.x + f.col(1).x * p1.y + f.col(2).x;
        let b2 = f.col(0).y * p1.x + f.col(1).y * p1.y + f.col(2).y;
        let c2 = f.col(0).z * p1.x + f.col(1).z * p1.y + f.col(2).z;
        let num2 = a2 * p2.x + b2 * p2.y + c2;
        let denom2 = a2 * a2 + b2 * b2;
        if denom2 > 1e-12 {
            let d2_1 = num2 * num2 / denom2;
            if d2_1 <= th {
                score += th_score - d2_1;
            }
        }

        // l1 = F^T * x2
        let a1 = f.col(0).x * p2.x + f.col(0).y * p2.y + f.col(0).z;
        let b1 = f.col(1).x * p2.x + f.col(1).y * p2.y + f.col(1).z;
        let c1 = f.col(2).x * p2.x + f.col(2).y * p2.y + f.col(2).z;
        let num1 = a1 * p1.x + b1 * p1.y + c1;
        let denom1 = a1 * a1 + b1 * b1;
        if denom1 > 1e-12 {
            let d2_2 = num1 * num1 / denom1;
            if d2_2 <= th {
                score += th_score - d2_2;
            }
        }
    }
    score
}

fn invert_mat3(m: &Mat3F64) -> Option<Mat3F64> {
    let inv = m.inverse();
    // glam returns zero matrix when singular. Detect by checking a diagonal.
    let prod = *m * inv;
    let trace = prod.col(0).x + prod.col(1).y + prod.col(2).z;
    if (trace - 3.0).abs() > 1e-3 {
        return None;
    }
    Some(inv)
}

/// Human-readable name for a two-view rejection reason.
pub fn two_view_reject_reason_name(reason: TwoViewRejectReason) -> &'static str {
    match reason {
        TwoViewRejectReason::LowMatches => "low_matches",
        TwoViewRejectReason::EstimationFailed => "estimation_failed",
        TwoViewRejectReason::WrongModel => "wrong_model",
        TwoViewRejectReason::LowTriangulated => "low_triangulated",
        TwoViewRejectReason::LowInliers => "low_inliers",
        TwoViewRejectReason::LowParallax => "low_parallax",
    }
}

fn two_view_model_name(model: TwoViewModel) -> &'static str {
    match model {
        TwoViewModel::Fundamental(_) => "fundamental",
        TwoViewModel::Homography(_) => "homography",
    }
}

fn match_orb_descriptors_for_initialization(
    ref_features: &OrbFeatures,
    curr_features: &OrbFeatures,
    config: OrbMatchConfig,
    window_px: f32,
) -> Vec<(usize, usize)> {
    assert_eq!(
        ref_features.orientations.len(),
        ref_features.descriptors.len(),
        "ref orientation/descriptor length mismatch",
    );
    assert_eq!(
        ref_features.scales.len(),
        ref_features.descriptors.len(),
        "ref scale/descriptor length mismatch",
    );
    assert_eq!(
        curr_features.orientations.len(),
        curr_features.descriptors.len(),
        "current orientation/descriptor length mismatch",
    );
    assert_eq!(
        curr_features.scales.len(),
        curr_features.descriptors.len(),
        "current scale/descriptor length mismatch",
    );

    if ref_features.descriptors.is_empty() || curr_features.descriptors.is_empty() {
        return Vec::new();
    }

    let mut matches12: Vec<Option<usize>> = vec![None; ref_features.descriptors.len()];
    let mut matches21: Vec<Option<usize>> = vec![None; curr_features.descriptors.len()];
    let mut matched_distance: Vec<u32> = vec![u32::MAX; curr_features.descriptors.len()];
    let mut rot_hist: Vec<Vec<usize>> = vec![Vec::new(); config.histo_length];
    let factor = 1.0f32 / config.histo_length as f32;

    for (i1, ref_desc) in ref_features.descriptors.iter().enumerate() {
        if !is_base_octave(ref_features.scales[i1]) {
            continue;
        }
        let ref_pt = ref_features.keypoints_xy[i1];
        let mut best_dist = u32::MAX;
        let mut second_dist = u32::MAX;
        let mut best_idx = None;

        for (i2, curr_desc) in curr_features.descriptors.iter().enumerate() {
            if !is_base_octave(curr_features.scales[i2]) {
                continue;
            }
            let curr_pt = curr_features.keypoints_xy[i2];
            if (curr_pt[0] - ref_pt[0]).abs() > window_px
                || (curr_pt[1] - ref_pt[1]).abs() > window_px
            {
                continue;
            }

            let dist = hamming_distance(ref_desc, curr_desc);
            if matched_distance[i2] <= dist {
                continue;
            }

            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = Some(i2);
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        let Some(i2) = best_idx else {
            continue;
        };
        if best_dist > config.th_low || (best_dist as f32) >= config.nn_ratio * second_dist as f32 {
            continue;
        }

        if let Some(prev_i1) = matches21[i2] {
            matches12[prev_i1] = None;
        }
        matches12[i1] = Some(i2);
        matches21[i2] = Some(i1);
        matched_distance[i2] = best_dist;

        if config.check_orientation {
            let mut rot = ref_features.orientations[i1].to_degrees()
                - curr_features.orientations[i2].to_degrees();
            if rot < 0.0 {
                rot += 360.0;
            }
            let mut bin = (rot * factor).round() as usize;
            if bin == config.histo_length {
                bin = 0;
            }
            rot_hist[bin].push(i1);
        }
    }

    if config.check_orientation {
        let (ind1, ind2, ind3) = three_maxima(&rot_hist);
        for (bin_idx, bin) in rot_hist.iter().enumerate() {
            if Some(bin_idx) == ind1 || Some(bin_idx) == ind2 || Some(bin_idx) == ind3 {
                continue;
            }
            for &i1 in bin {
                matches12[i1] = None;
            }
        }
    }

    matches12
        .into_iter()
        .enumerate()
        .filter_map(|(i1, i2)| i2.map(|i2| (i1, i2)))
        .collect()
}

fn is_base_octave(scale: f32) -> bool {
    (scale - 1.0).abs() <= 1e-3
}

fn three_maxima(histo: &[Vec<usize>]) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut ind1 = None;
    let mut ind2 = None;
    let mut ind3 = None;
    let mut max1 = 0usize;
    let mut max2 = 0usize;
    let mut max3 = 0usize;

    for (i, bin) in histo.iter().enumerate() {
        let size = bin.len();
        if size > max1 {
            max3 = max2;
            ind3 = ind2;
            max2 = max1;
            ind2 = ind1;
            max1 = size;
            ind1 = Some(i);
        } else if size > max2 {
            max3 = max2;
            ind3 = ind2;
            max2 = size;
            ind2 = Some(i);
        } else if size > max3 {
            max3 = size;
            ind3 = Some(i);
        }
    }

    let threshold = (0.1 * max1 as f32) as usize;
    if max2 < threshold {
        ind2 = None;
        ind3 = None;
    } else if max3 < threshold {
        ind3 = None;
    }

    (ind1, ind2, ind3)
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

    fn descriptor(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn match_config() -> OrbMatchConfig {
        OrbMatchConfig {
            nn_ratio: 0.9,
            th_low: 50,
            check_orientation: true,
            histo_length: 30,
        }
    }

    #[test]
    fn initialization_matcher_rejects_matches_outside_search_window() {
        let ref_features = OrbFeatures {
            keypoints_xy: vec![[10.0, 10.0]],
            scales: vec![1.0],
            orientations: vec![0.0],
            descriptors: vec![descriptor(0)],
        };
        let curr_features = OrbFeatures {
            keypoints_xy: vec![[200.0, 200.0], [15.0, 15.0]],
            scales: vec![1.0, 1.0],
            orientations: vec![0.0, 0.0],
            descriptors: vec![descriptor(0), descriptor(1)],
        };

        let matches = match_orb_descriptors_for_initialization(
            &ref_features,
            &curr_features,
            match_config(),
            20.0,
        );

        assert_eq!(matches, vec![(0, 1)]);
    }

    #[test]
    fn initialization_matcher_only_uses_base_octave_features() {
        let ref_features = OrbFeatures {
            keypoints_xy: vec![[10.0, 10.0], [20.0, 20.0]],
            scales: vec![1.2, 1.0],
            orientations: vec![0.0, 0.0],
            descriptors: vec![descriptor(0), descriptor(1)],
        };
        let curr_features = OrbFeatures {
            keypoints_xy: vec![[10.0, 10.0], [20.0, 20.0]],
            scales: vec![1.0, 1.0],
            orientations: vec![0.0, 0.0],
            descriptors: vec![descriptor(0), descriptor(1)],
        };

        let matches = match_orb_descriptors_for_initialization(
            &ref_features,
            &curr_features,
            match_config(),
            20.0,
        );

        assert_eq!(matches, vec![(1, 1)]);
    }

    #[test]
    fn initialization_matcher_keeps_one_to_one_current_frame_matches() {
        let ref_features = OrbFeatures {
            keypoints_xy: vec![[10.0, 10.0], [11.0, 11.0]],
            scales: vec![1.0, 1.0],
            orientations: vec![0.0, 0.0],
            descriptors: vec![descriptor(1), descriptor(0)],
        };
        let curr_features = OrbFeatures {
            keypoints_xy: vec![[12.0, 12.0]],
            scales: vec![1.0],
            orientations: vec![0.0],
            descriptors: vec![descriptor(0)],
        };

        let matches = match_orb_descriptors_for_initialization(
            &ref_features,
            &curr_features,
            match_config(),
            20.0,
        );

        assert_eq!(matches, vec![(1, 0)]);
    }
}
