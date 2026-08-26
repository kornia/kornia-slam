use std::collections::HashSet;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{PnPMethod, RansacParams, solve_pnp_ransac};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::map::Map;

/// Acceptance thresholds for geometric loop verification.
#[derive(Debug, Clone)]
pub struct LoopVerificationConfig {
    pub orb_match: OrbMatchConfig,
    pub min_correspondences: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f32,
    pub max_reprojection_rmse_px: f32,
    pub coverage_rows: usize,
    pub coverage_cols: usize,
    pub min_occupied_cells: usize,
    pub pnp_ransac: RansacParams,
}

impl Default for LoopVerificationConfig {
    fn default() -> Self {
        Self {
            orb_match: OrbMatchConfig::default(),
            min_correspondences: 30,
            min_inliers: 30,
            min_inlier_ratio: 0.4,
            max_reprojection_rmse_px: 3.0,
            coverage_rows: 4,
            coverage_cols: 4,
            min_occupied_cells: 6,
            pnp_ransac: RansacParams {
                max_iterations: 500,
                reproj_threshold_px: 3.0,
                confidence: 0.999,
                random_seed: None,
                refine: true,
            },
        }
    }
}

/// Why an appearance candidate did not become a loop edge.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopVerificationReject {
    MissingQueryKeyframe,
    MissingCandidateKeyframe,
    TooFewCorrespondences { actual: usize, required: usize },
    PnpFailed(String),
    TooFewInliers { actual: usize, required: usize },
    LowInlierRatio { actual: f32, required: f32 },
    HighReprojectionError { actual: f32, maximum: f32 },
    PoorCoverage { occupied: usize, required: usize },
}

/// An independently measured metric loop constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedLoopEdge {
    pub query_kf_idx: usize,
    pub candidate_kf_idx: usize,
    /// Transform taking candidate-camera coordinates into query-camera coordinates.
    pub candidate_to_query: Pose3d,
    pub correspondences: usize,
    pub inliers: usize,
    pub inlier_ratio: f32,
    pub reprojection_rmse_px: f32,
    pub occupied_cells: usize,
}

pub(super) struct VerificationInput {
    pub(super) world_points: Vec<Vec3AF32>,
    pub(super) image_points: Vec<Vec2F32>,
}

pub(super) fn verification_input(
    map: &Map,
    camera: &PinholeCamera,
    query_kf_idx: usize,
    candidate_kf_idx: usize,
    config: &LoopVerificationConfig,
) -> Result<VerificationInput, LoopVerificationReject> {
    let query = map
        .get_keyframe(query_kf_idx)
        .ok_or(LoopVerificationReject::MissingQueryKeyframe)?;
    let candidate = map
        .get_keyframe(candidate_kf_idx)
        .ok_or(LoopVerificationReject::MissingCandidateKeyframe)?;
    let matches = match_orb_descriptors(
        &candidate.frame.features.orientations,
        &candidate.frame.features.descriptors,
        &query.frame.features.orientations,
        &query.frame.features.descriptors,
        config.orb_match,
    );

    let mut world_points = Vec::with_capacity(matches.len());
    let mut image_points = Vec::with_capacity(matches.len());
    for (candidate_desc_idx, query_desc_idx) in matches {
        let Some(map_point_idx) = candidate.map_point(candidate_desc_idx) else {
            continue;
        };
        let Some(map_point) = map.map_points().get(map_point_idx) else {
            continue;
        };
        if map_point.culled {
            continue;
        }
        let Some(query_xy) = query.frame.undistorted_xy(query_desc_idx, camera) else {
            continue;
        };
        world_points.push(vec3_f32(map_point.position));
        image_points.push(Vec2F32::new(query_xy[0], query_xy[1]));
    }

    if world_points.len() < config.min_correspondences {
        return Err(LoopVerificationReject::TooFewCorrespondences {
            actual: world_points.len(),
            required: config.min_correspondences,
        });
    }
    Ok(VerificationInput {
        world_points,
        image_points,
    })
}

/// Verify a BoW loop candidate using candidate landmarks and query pixels.
pub fn verify_loop_candidate(
    map: &Map,
    camera: &PinholeCamera,
    query_kf_idx: usize,
    candidate_kf_idx: usize,
    config: &LoopVerificationConfig,
) -> Result<VerifiedLoopEdge, LoopVerificationReject> {
    let input = verification_input(map, camera, query_kf_idx, candidate_kf_idx, config)?;
    let intrinsics = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let result = solve_pnp_ransac(
        &input.world_points,
        &input.image_points,
        &intrinsics,
        None,
        PnPMethod::AP3PDefault,
        &config.pnp_ransac,
    )
    .map_err(|error| LoopVerificationReject::PnpFailed(error.to_string()))?;

    let inliers = result.inliers.len();
    if inliers < config.min_inliers {
        return Err(LoopVerificationReject::TooFewInliers {
            actual: inliers,
            required: config.min_inliers,
        });
    }
    let inlier_ratio = inliers as f32 / input.world_points.len() as f32;
    if inlier_ratio < config.min_inlier_ratio {
        return Err(LoopVerificationReject::LowInlierRatio {
            actual: inlier_ratio,
            required: config.min_inlier_ratio,
        });
    }

    let query_pose = pnp_pose(&result.pose);
    let squared_errors: Vec<f64> = result
        .inliers
        .iter()
        .filter_map(|&index| {
            let point = vec3_f64(input.world_points[index]);
            let pixel = input.image_points[index];
            camera.reprojection_error_sq_world(&query_pose, &point, pixel.x as f64, pixel.y as f64)
        })
        .collect();
    let reprojection_rmse_px =
        (squared_errors.iter().sum::<f64>() / squared_errors.len() as f64).sqrt() as f32;
    if !reprojection_rmse_px.is_finite() || reprojection_rmse_px > config.max_reprojection_rmse_px {
        return Err(LoopVerificationReject::HighReprojectionError {
            actual: reprojection_rmse_px,
            maximum: config.max_reprojection_rmse_px,
        });
    }

    let query = map.get_keyframe(query_kf_idx).unwrap();
    let occupied_cells = occupied_image_cells(
        result
            .inliers
            .iter()
            .map(|&index| input.image_points[index]),
        query.frame.image_size.width,
        query.frame.image_size.height,
        config.coverage_rows,
        config.coverage_cols,
    );
    if occupied_cells < config.min_occupied_cells {
        return Err(LoopVerificationReject::PoorCoverage {
            occupied: occupied_cells,
            required: config.min_occupied_cells,
        });
    }

    let candidate_pose = map
        .get_keyframe(candidate_kf_idx)
        .unwrap()
        .frame
        .pose_world_to_cam;
    Ok(VerifiedLoopEdge {
        query_kf_idx,
        candidate_kf_idx,
        candidate_to_query: Pose3d::between(&candidate_pose, &query_pose),
        correspondences: input.world_points.len(),
        inliers,
        inlier_ratio,
        reprojection_rmse_px,
        occupied_cells,
    })
}

fn occupied_image_cells(
    points: impl IntoIterator<Item = Vec2F32>,
    width: usize,
    height: usize,
    rows: usize,
    cols: usize,
) -> usize {
    if width == 0 || height == 0 || rows == 0 || cols == 0 {
        return 0;
    }
    points
        .into_iter()
        .map(|point| {
            let col = ((point.x.max(0.0) / width as f32) * cols as f32) as usize;
            let row = ((point.y.max(0.0) / height as f32) * rows as f32) as usize;
            row.min(rows - 1) * cols + col.min(cols - 1)
        })
        .collect::<HashSet<_>>()
        .len()
}

fn pnp_pose(pose: &kornia_3d::pnp::PnPResult) -> Pose3d {
    let rotation = pose.rotation;
    Pose3d::new(
        Mat3F64::from_cols(
            vec3_f64(rotation.col(0).into()),
            vec3_f64(rotation.col(1).into()),
            vec3_f64(rotation.col(2).into()),
        ),
        vec3_f64(pose.translation),
    )
}

fn vec3_f32(value: Vec3F64) -> Vec3AF32 {
    Vec3AF32::new(value.x as f32, value.y as f32, value.z as f32)
}

fn vec3_f64(value: Vec3AF32) -> Vec3F64 {
    Vec3F64::new(value.x as f64, value.y as f64, value.z as f64)
}
