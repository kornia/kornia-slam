//! Geometric loop verification and read-only SE(3) pose-graph diagnostics.

use std::collections::{HashMap, HashSet};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pgo::{PgoEdge, PgoParams, pose_graph_optimize};
use kornia_3d::pnp::{PnPMethod, RansacParams, solve_pnp_ransac};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, Vec2F32, Vec3AF32, Vec3F64};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};
use thiserror::Error;

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

/// Configuration for the read-only PGO solve.
#[derive(Debug, Clone)]
pub struct ShadowPgoConfig {
    pub loop_edge_weight: f32,
    pub max_iterations: usize,
    pub cost_tolerance: f32,
    pub gradient_tolerance: f32,
    pub initial_lambda: f32,
}

impl Default for ShadowPgoConfig {
    fn default() -> Self {
        Self {
            loop_edge_weight: 0.5,
            max_iterations: 30,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            initial_lambda: 1e-3,
        }
    }
}

/// Output of a shadow PGO run. No values are written back to the map.
#[derive(Debug, Clone)]
pub struct ShadowPgoDiagnostic {
    pub keyframe_indices: Vec<usize>,
    pub original_poses: Vec<Pose3d>,
    pub optimized_poses: Vec<Pose3d>,
    pub verified_loop: VerifiedLoopEdge,
    pub iterations: usize,
    pub converged: bool,
    pub median_translation_correction: f64,
    pub max_translation_correction: f64,
}

/// Shadow-graph construction or solve failure.
#[derive(Debug, Error)]
pub enum ShadowPgoError {
    #[error("pose graph needs at least two keyframes")]
    TooFewKeyframes,
    #[error("loop edge references a keyframe outside the map snapshot")]
    MissingLoopKeyframe,
    #[error("pose graph optimization failed: {0}")]
    Solver(String),
}

struct VerificationInput {
    world_points: Vec<Vec3AF32>,
    image_points: Vec<Vec2F32>,
}

fn verification_input(
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

/// Run PGO against a read-only keyframe snapshot and return diagnostic poses.
pub fn run_shadow_pgo(
    map: &Map,
    verified_loops: &[VerifiedLoopEdge],
    config: &ShadowPgoConfig,
) -> Result<ShadowPgoDiagnostic, ShadowPgoError> {
    let keyframes = map.keyframes();
    if keyframes.len() < 2 {
        return Err(ShadowPgoError::TooFewKeyframes);
    }
    let keyframe_indices: Vec<_> = keyframes
        .iter()
        .map(|keyframe| keyframe.frame.idx)
        .collect();
    let original_poses: Vec<_> = keyframes
        .iter()
        .map(|keyframe| keyframe.frame.pose_world_to_cam)
        .collect();
    let node_by_keyframe: HashMap<_, _> = keyframe_indices
        .iter()
        .enumerate()
        .map(|(node, &keyframe)| (keyframe, node))
        .collect();
    let mut edges = Vec::with_capacity(original_poses.len() - 1 + verified_loops.len());
    for node in 0..original_poses.len() - 1 {
        let measurement = Pose3d::between(&original_poses[node], &original_poses[node + 1]);
        edges.push(PgoEdge {
            pose_a: node,
            pose_b: node + 1,
            t_ab_meas: pose_to_se3(&measurement),
            weight: 1.0,
        });
    }
    for verified in verified_loops {
        let Some(&candidate_node) = node_by_keyframe.get(&verified.candidate_kf_idx) else {
            return Err(ShadowPgoError::MissingLoopKeyframe);
        };
        let Some(&query_node) = node_by_keyframe.get(&verified.query_kf_idx) else {
            return Err(ShadowPgoError::MissingLoopKeyframe);
        };
        edges.push(PgoEdge {
            pose_a: candidate_node,
            pose_b: query_node,
            t_ab_meas: pose_to_se3(&verified.candidate_to_query),
            weight: config.loop_edge_weight,
        });
    }
    let result = pose_graph_optimize(
        &original_poses,
        &edges,
        &[0],
        &PgoParams {
            max_iterations: config.max_iterations,
            cost_tolerance: config.cost_tolerance,
            gradient_tolerance: config.gradient_tolerance,
            initial_lambda: config.initial_lambda,
        },
    )
    .map_err(|error| ShadowPgoError::Solver(error.to_string()))?;

    let mut corrections: Vec<f64> = original_poses
        .iter()
        .zip(&result.poses)
        .map(|(before, after)| {
            (before.inverse().translation - after.inverse().translation).length()
        })
        .collect();
    corrections.sort_by(f64::total_cmp);
    let median_translation_correction = corrections[corrections.len() / 2];
    let max_translation_correction = corrections.last().copied().unwrap_or(0.0);
    let verified_loop = verified_loops
        .last()
        .cloned()
        .ok_or(ShadowPgoError::MissingLoopKeyframe)?;

    Ok(ShadowPgoDiagnostic {
        keyframe_indices,
        original_poses,
        optimized_poses: result.poses,
        verified_loop,
        iterations: result.iterations,
        converged: result.converged,
        median_translation_correction,
        max_translation_correction,
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

fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    let rotation = Mat3AF32::from_cols(
        vec3_f32(pose.rotation.col(0).into()),
        vec3_f32(pose.rotation.col(1).into()),
        vec3_f32(pose.rotation.col(2).into()),
    );
    SE3F32::new(SO3F32::from_matrix(&rotation), vec3_f32(pose.translation))
}

fn vec3_f32(value: Vec3F64) -> Vec3AF32 {
    Vec3AF32::new(value.x as f32, value.y as f32, value.z as f32)
}

fn vec3_f64(value: Vec3AF32) -> Vec3F64 {
    Vec3F64::new(value.x as f64, value.y as f64, value.z as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frame;
    use crate::map::{Keyframe, MapPoint};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn frame(idx: usize, pose: Pose3d, points: &[[f32; 2]], descriptors: Vec<[u8; 32]>) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: points.to_vec(),
                orientations: vec![0.0; points.len()],
                descriptors,
                octaves: vec![0; points.len()],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; points.len()],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: points.to_vec(),
        }
    }

    fn descriptor(index: usize) -> [u8; 32] {
        let mut descriptor = [0_u8; 32];
        for (byte_index, value) in descriptor.iter_mut().enumerate() {
            *value = (index.wrapping_mul(37).wrapping_add(byte_index * 13)) as u8;
        }
        descriptor
    }

    fn synthetic_loop_map() -> (Map, Pose3d) {
        let camera = camera();
        let query_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.15, -0.05, 0.1));
        let mut map = Map::new();
        let mut world = Vec::new();
        let mut query_pixels = Vec::new();
        let mut candidate_pixels = Vec::new();
        let mut descriptors = Vec::new();
        for index in 0..36 {
            let x = (index % 6) as f64 * 0.35 - 0.875;
            let y = (index / 6) as f64 * 0.28 - 0.7;
            let point = Vec3F64::new(x, y, 4.0 + (index % 3) as f64 * 0.2);
            let query_point = query_pose.transform_point(&point);
            query_pixels.push([
                (camera.fx * query_point.x / query_point.z + camera.cx) as f32,
                (camera.fy * query_point.y / query_point.z + camera.cy) as f32,
            ]);
            candidate_pixels.push([
                (camera.fx * point.x / point.z + camera.cx) as f32,
                (camera.fy * point.y / point.z + camera.cy) as f32,
            ]);
            world.push(point);
            descriptors.push(descriptor(index));
        }
        let candidate = Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            &candidate_pixels,
            descriptors.clone(),
        ));
        let query = Keyframe::from_frame(frame(10, query_pose, &query_pixels, descriptors.clone()));
        map.upsert_keyframe(candidate);
        map.upsert_keyframe(query);
        for (index, point) in world.into_iter().enumerate() {
            let map_point =
                map.push_map_point(MapPoint::new(point, descriptors[index], 0, [0; 3], 0));
            map.get_keyframe_mut(0)
                .unwrap()
                .associate_map_point(index, map_point);
        }
        (map, query_pose)
    }

    #[test]
    fn correspondence_input_filters_unassociated_and_culled_points() {
        let (mut map, _) = synthetic_loop_map();
        map.get_keyframe_mut(0).unwrap().clear_map_point(0);
        let culled = map.get_keyframe(0).unwrap().map_point(1).unwrap();
        map.map_points_mut()[culled].mark_culled();
        let config = LoopVerificationConfig {
            min_correspondences: 1,
            ..LoopVerificationConfig::default()
        };
        let input = verification_input(&map, &camera(), 10, 0, &config).unwrap();
        assert_eq!(input.world_points.len(), 34);
        assert_eq!(input.image_points.len(), 34);
    }

    #[test]
    fn verify_loop_recovers_metric_relative_pose() {
        let (map, expected_query_pose) = synthetic_loop_map();
        let config = LoopVerificationConfig {
            min_correspondences: 20,
            min_inliers: 20,
            min_inlier_ratio: 0.8,
            min_occupied_cells: 4,
            pnp_ransac: RansacParams {
                random_seed: Some(7),
                ..LoopVerificationConfig::default().pnp_ransac
            },
            ..LoopVerificationConfig::default()
        };
        let verified = verify_loop_candidate(&map, &camera(), 10, 0, &config).unwrap();
        assert!(verified.inliers >= 20);
        assert!(verified.reprojection_rmse_px < 1e-2);
        assert!(
            (verified.candidate_to_query.translation - expected_query_pose.translation).length()
                < 1e-2
        );
    }

    #[test]
    fn shadow_pgo_reduces_terminal_loop_gap_without_moving_anchor() {
        let mut map = Map::new();
        for index in 0..5 {
            let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-(index as f64), 0.0, 0.0));
            map.upsert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())));
        }
        let verified = VerifiedLoopEdge {
            query_kf_idx: 4,
            candidate_kf_idx: 0,
            candidate_to_query: Pose3d::IDENTITY,
            correspondences: 40,
            inliers: 35,
            inlier_ratio: 0.875,
            reprojection_rmse_px: 1.0,
            occupied_cells: 8,
        };
        let diagnostic = run_shadow_pgo(&map, &[verified], &ShadowPgoConfig::default()).unwrap();
        assert_eq!(diagnostic.optimized_poses[0], diagnostic.original_poses[0]);
        let before = diagnostic.original_poses[4].inverse().translation.length();
        let after = diagnostic.optimized_poses[4].inverse().translation.length();
        assert!(after < before, "expected {after} < {before}");
        assert!(diagnostic.max_translation_correction > 0.0);
    }
}
