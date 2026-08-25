//! Geometric loop verification and pose-graph diagnostics.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pgo::{PgoEdge, PgoParams, pose_graph_optimize};
use kornia_3d::pnp::{PnPMethod, RansacParams, solve_pnp_ransac};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, Vec2F32, Vec3AF32, Vec3F64};
use kornia_imgproc::features::{OrbMatchConfig, hamming_distance, match_orb_descriptors};
use thiserror::Error;

use crate::gravity_pgo::gravity_pose_graph_optimize;
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

/// Bounded landmark search performed after an accepted loop correction.
#[derive(Debug, Clone)]
pub struct LoopFusionConfig {
    pub min_covisibility_weight: usize,
    pub max_neighbors_per_side: usize,
    pub search_radius_px: f32,
    pub max_hamming: u32,
    pub max_reprojection_error_px: f64,
}

impl Default for LoopFusionConfig {
    fn default() -> Self {
        Self {
            min_covisibility_weight: 15,
            max_neighbors_per_side: 5,
            search_radius_px: 7.0,
            max_hamming: 50,
            max_reprojection_error_px: 3.0,
        }
    }
}

/// Live-map changes made while fusing a verified loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopFusionStats {
    pub observations_added: usize,
    pub map_points_merged: usize,
}

/// Temporal and map-neighbourhood consistency required before a verified loop
/// becomes a pose-graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopEpisodeConfig {
    pub min_consistent_edges: usize,
    pub max_query_gap: usize,
    pub candidate_neighborhood_radius: usize,
}

impl Default for LoopEpisodeConfig {
    fn default() -> Self {
        Self {
            min_consistent_edges: 3,
            max_query_gap: 5,
            candidate_neighborhood_radius: 10,
        }
    }
}

/// Decision produced for one geometrically verified observation.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEpisodeDecision {
    Pending {
        hits: usize,
        required: usize,
    },
    Ready {
        representative: VerifiedLoopEdge,
        hits: usize,
    },
    Suppressed {
        representative_query_kf_idx: usize,
        representative_candidate_kf_idx: usize,
    },
}

#[derive(Debug, Clone)]
struct LoopEpisode {
    candidate_anchor_order: usize,
    last_query_order: usize,
    hits: usize,
    representative: VerifiedLoopEdge,
    ready: bool,
}

/// Collapses adjacent verified observations into one physical revisit episode.
#[derive(Debug, Clone)]
pub struct LoopEpisodeTracker {
    config: LoopEpisodeConfig,
    current: Option<LoopEpisode>,
}

impl LoopEpisodeTracker {
    pub fn new(mut config: LoopEpisodeConfig) -> Self {
        config.min_consistent_edges = config.min_consistent_edges.max(1);
        Self {
            config,
            current: None,
        }
    }

    pub fn observe(
        &mut self,
        query_order: usize,
        candidate_order: usize,
        edge: VerifiedLoopEdge,
    ) -> LoopEpisodeDecision {
        let compatible = self.current.as_ref().is_some_and(|episode| {
            query_order >= episode.last_query_order
                && query_order - episode.last_query_order <= self.config.max_query_gap
                && candidate_order.abs_diff(episode.candidate_anchor_order)
                    <= self.config.candidate_neighborhood_radius
        });
        if !compatible {
            self.current = Some(LoopEpisode {
                candidate_anchor_order: candidate_order,
                last_query_order: query_order,
                hits: 1,
                representative: edge,
                ready: self.config.min_consistent_edges == 1,
            });
            let episode = self.current.as_ref().unwrap();
            return if episode.ready {
                LoopEpisodeDecision::Ready {
                    representative: episode.representative.clone(),
                    hits: episode.hits,
                }
            } else {
                LoopEpisodeDecision::Pending {
                    hits: episode.hits,
                    required: self.config.min_consistent_edges,
                }
            };
        }

        let episode = self.current.as_mut().unwrap();
        episode.last_query_order = query_order;
        if episode.ready {
            return LoopEpisodeDecision::Suppressed {
                representative_query_kf_idx: episode.representative.query_kf_idx,
                representative_candidate_kf_idx: episode.representative.candidate_kf_idx,
            };
        }
        episode.hits += 1;
        if edge_quality_better(&edge, &episode.representative) {
            episode.representative = edge;
        }
        if episode.hits >= self.config.min_consistent_edges {
            episode.ready = true;
            LoopEpisodeDecision::Ready {
                representative: episode.representative.clone(),
                hits: episode.hits,
            }
        } else {
            LoopEpisodeDecision::Pending {
                hits: episode.hits,
                required: self.config.min_consistent_edges,
            }
        }
    }
}

fn edge_quality_better(candidate: &VerifiedLoopEdge, current: &VerifiedLoopEdge) -> bool {
    candidate
        .inliers
        .cmp(&current.inliers)
        .then_with(|| candidate.inlier_ratio.total_cmp(&current.inlier_ratio))
        .then_with(|| candidate.occupied_cells.cmp(&current.occupied_cells))
        .then_with(|| {
            current
                .reprojection_rmse_px
                .total_cmp(&candidate.reprojection_rmse_px)
        })
        .is_gt()
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

/// Live-map changes made after an explicitly enabled, usable PGO result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PgoApplicationStats {
    pub keyframes_corrected: usize,
    pub map_points_corrected: usize,
    pub observations_added: usize,
    pub map_points_merged: usize,
}

/// Manifold used for a pose-graph solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgoMode {
    Se3,
    Gravity4Dof,
}

/// Runtime inertial frame needed by gravity-preserving PGO diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct InertialPgoContext {
    pub gravity_world: Vec3F64,
    pub imu_t_bc: Option<Pose3d>,
}

/// Output of a PGO run and its optional live-map application statistics.
#[derive(Debug, Clone)]
pub struct ShadowPgoDiagnostic {
    pub mode: PgoMode,
    pub keyframe_indices: Vec<usize>,
    pub original_poses: Vec<Pose3d>,
    pub optimized_poses: Vec<Pose3d>,
    pub verified_loop: VerifiedLoopEdge,
    pub iterations: usize,
    pub converged: bool,
    pub usable: bool,
    pub node_count: usize,
    pub sequential_edge_count: usize,
    pub loop_edge_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub new_loop_residual_before: f64,
    pub new_loop_residual_after: f64,
    pub solve_time_ms: f64,
    pub median_translation_correction: f64,
    pub max_translation_correction: f64,
    pub max_gravity_alignment_error_rad: Option<f64>,
    pub imu_residual_rms_before: Option<f64>,
    pub imu_residual_rms_after: Option<f64>,
    pub application: Option<PgoApplicationStats>,
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

/// Adds and merges landmark observations across a geometrically verified loop.
pub fn fuse_verified_loop(
    map: &mut Map,
    camera: &PinholeCamera,
    edge: &VerifiedLoopEdge,
    config: &LoopFusionConfig,
) -> LoopFusionStats {
    let query_side = loop_side_keyframes(map, edge.query_kf_idx, config);
    let candidate_side = loop_side_keyframes(map, edge.candidate_kf_idx, config);
    let mut stats = LoopFusionStats::default();
    fuse_loop_direction(
        map,
        camera,
        &candidate_side,
        &query_side,
        config,
        &mut stats,
    );
    fuse_loop_direction(
        map,
        camera,
        &query_side,
        &candidate_side,
        config,
        &mut stats,
    );
    stats
}

fn loop_side_keyframes(map: &Map, anchor: usize, config: &LoopFusionConfig) -> Vec<usize> {
    if map.get_keyframe(anchor).is_none() {
        return Vec::new();
    }
    let mut side = vec![anchor];
    side.extend(
        map.covisible_keyframes(anchor, config.min_covisibility_weight)
            .into_iter()
            .take(config.max_neighbors_per_side)
            .map(|(keyframe_idx, _)| keyframe_idx),
    );
    side
}

fn fuse_loop_direction(
    map: &mut Map,
    camera: &PinholeCamera,
    source_keyframes: &[usize],
    target_keyframes: &[usize],
    config: &LoopFusionConfig,
    stats: &mut LoopFusionStats,
) {
    let mut seen_points = HashSet::new();
    let mut source_points = Vec::new();
    for &source_kf_idx in source_keyframes {
        let Some(source) = map.get_keyframe(source_kf_idx) else {
            continue;
        };
        for (source_desc_idx, &map_point_idx) in source.map_point_by_desc_idx.iter().enumerate() {
            let Some(map_point_idx) = map_point_idx else {
                continue;
            };
            if seen_points.insert(map_point_idx) {
                source_points.push((map_point_idx, source_kf_idx, source_desc_idx));
            }
        }
    }

    let search_radius_sq = config.search_radius_px * config.search_radius_px;
    let reprojection_limit_sq = config.max_reprojection_error_px * config.max_reprojection_error_px;
    for &target_kf_idx in target_keyframes {
        for &(source_point_idx, source_kf_idx, source_desc_idx) in &source_points {
            if source_kf_idx == target_kf_idx {
                continue;
            }
            let proposal = {
                let Some(source_point) = map.map_points().get(source_point_idx) else {
                    continue;
                };
                if source_point.culled
                    || source_point.observation_kf_indices.contains(&target_kf_idx)
                {
                    continue;
                }
                let Some(target) = map.get_keyframe(target_kf_idx) else {
                    continue;
                };
                let point_target = target
                    .frame
                    .pose_world_to_cam
                    .transform_point(&source_point.position);
                if point_target.z <= 0.0 {
                    continue;
                }
                let Ok(projected) =
                    camera.project_to_image(&point_target, 0.0, target.frame.image_size)
                else {
                    continue;
                };
                let projected_x = projected.x as f32;
                let projected_y = projected.y as f32;
                let mut best: Option<(usize, u32, f32)> = None;
                for target_desc_idx in 0..target.frame.features.descriptors.len() {
                    let Some(keypoint) = target.frame.undistorted_xy(target_desc_idx, camera)
                    else {
                        continue;
                    };
                    let dx = keypoint[0] - projected_x;
                    let dy = keypoint[1] - projected_y;
                    let distance_sq = dx * dx + dy * dy;
                    if distance_sq > search_radius_sq || distance_sq as f64 > reprojection_limit_sq
                    {
                        continue;
                    }
                    let hamming = hamming_distance(
                        &source_point.descriptor,
                        &target.frame.features.descriptors[target_desc_idx],
                    );
                    if hamming > config.max_hamming {
                        continue;
                    }
                    if best.is_none_or(|(_, best_hamming, best_distance)| {
                        (hamming, distance_sq).lt(&(best_hamming, best_distance))
                    }) {
                        best = Some((target_desc_idx, hamming, distance_sq));
                    }
                }
                best.map(|(target_desc_idx, _, _)| {
                    (target_desc_idx, target.map_point(target_desc_idx))
                })
            };
            let Some((target_desc_idx, target_point_idx)) = proposal else {
                continue;
            };

            let target_point_idx = target_point_idx.filter(|&index| {
                map.map_points()
                    .get(index)
                    .is_some_and(|point| !point.culled)
            });
            let Some(target_point_idx) = target_point_idx else {
                map.register_observation_at(source_point_idx, target_kf_idx, target_desc_idx);
                if let Some(target) = map.get_keyframe_mut(target_kf_idx) {
                    target.associate_map_point(target_desc_idx, source_point_idx);
                }
                stats.observations_added += 1;
                continue;
            };
            if target_point_idx == source_point_idx {
                continue;
            }

            let reciprocal_is_consistent = {
                let Some(target_point) = map.map_points().get(target_point_idx) else {
                    continue;
                };
                let Some(source) = map.get_keyframe(source_kf_idx) else {
                    continue;
                };
                let point_source = source
                    .frame
                    .pose_world_to_cam
                    .transform_point(&target_point.position);
                if point_source.z <= 0.0 {
                    false
                } else if let Ok(projected) =
                    camera.project_to_image(&point_source, 0.0, source.frame.image_size)
                {
                    source
                        .frame
                        .undistorted_xy(source_desc_idx, camera)
                        .is_some_and(|keypoint| {
                            let dx = projected.x - keypoint[0] as f64;
                            let dy = projected.y - keypoint[1] as f64;
                            dx * dx + dy * dy <= reprojection_limit_sq
                        })
                } else {
                    false
                }
            };
            if reciprocal_is_consistent
                && map
                    .merge_map_points(source_point_idx, target_point_idx)
                    .is_some()
            {
                stats.map_points_merged += 1;
            }
        }
    }
}

/// Run PGO against a read-only keyframe snapshot and return diagnostic poses.
pub fn run_shadow_pgo(
    map: &Map,
    verified_loops: &[VerifiedLoopEdge],
    config: &ShadowPgoConfig,
    inertial: Option<InertialPgoContext>,
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
    let node_count = original_poses.len();
    let sequential_edge_count = node_count - 1;
    let loop_edge_count = verified_loops.len();
    let initial_cost = pose_graph_cost(&original_poses, &edges);
    let new_loop_edge = edges.last().ok_or(ShadowPgoError::MissingLoopKeyframe)?;
    let new_loop_residual_before = weighted_edge_residual_norm(&original_poses, new_loop_edge)
        .ok_or(ShadowPgoError::MissingLoopKeyframe)?;
    let solve_started = Instant::now();
    let params = PgoParams {
        max_iterations: config.max_iterations,
        cost_tolerance: config.cost_tolerance,
        gradient_tolerance: config.gradient_tolerance,
        initial_lambda: config.initial_lambda,
    };
    let (mode, optimized_poses, iterations, converged) = if let Some(context) = inertial {
        let result = gravity_pose_graph_optimize(
            &original_poses,
            &edges,
            &[0],
            context.gravity_world,
            &params,
        )
        .map_err(|error| ShadowPgoError::Solver(error.to_string()))?;
        (
            PgoMode::Gravity4Dof,
            result.poses,
            result.iterations,
            result.converged,
        )
    } else {
        let result = pose_graph_optimize(&original_poses, &edges, &[0], &params)
            .map_err(|error| ShadowPgoError::Solver(error.to_string()))?;
        (
            PgoMode::Se3,
            result.poses,
            result.iterations,
            result.converged,
        )
    };
    let solve_time_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let final_cost = pose_graph_cost(&optimized_poses, &edges);
    let new_loop_residual_after = weighted_edge_residual_norm(&optimized_poses, new_loop_edge)
        .ok_or(ShadowPgoError::MissingLoopKeyframe)?;
    let max_gravity_alignment_error_rad = inertial.map(|context| {
        max_gravity_alignment_error(&original_poses, &optimized_poses, context.gravity_world)
    });
    let imu_residual_rms_before = inertial.and_then(|context| {
        map.inertial_residual_rms(
            &keyframe_indices,
            &original_poses,
            context.gravity_world,
            context.imu_t_bc,
        )
    });
    let imu_residual_rms_after = inertial.and_then(|context| {
        map.inertial_residual_rms(
            &keyframe_indices,
            &optimized_poses,
            context.gravity_world,
            context.imu_t_bc,
        )
    });
    let metrics_finite = [
        initial_cost,
        final_cost,
        new_loop_residual_before,
        new_loop_residual_after,
        solve_time_ms,
    ]
    .into_iter()
    .all(f64::is_finite);
    let inertial_metrics_finite = max_gravity_alignment_error_rad
        .into_iter()
        .chain(imu_residual_rms_before)
        .chain(imu_residual_rms_after)
        .all(f64::is_finite);
    let gravity_preserved = max_gravity_alignment_error_rad.is_none_or(|error| error <= 1e-4);
    let usable = converged
        && metrics_finite
        && inertial_metrics_finite
        && gravity_preserved
        && final_cost < initial_cost
        && new_loop_residual_after <= new_loop_residual_before;

    let mut corrections: Vec<f64> = original_poses
        .iter()
        .zip(&optimized_poses)
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
        mode,
        keyframe_indices,
        original_poses,
        optimized_poses,
        verified_loop,
        iterations,
        converged,
        usable,
        node_count,
        sequential_edge_count,
        loop_edge_count,
        initial_cost,
        final_cost,
        new_loop_residual_before,
        new_loop_residual_after,
        solve_time_ms,
        median_translation_correction,
        max_translation_correction,
        max_gravity_alignment_error_rad,
        imu_residual_rms_before,
        imu_residual_rms_after,
        application: None,
    })
}

fn max_gravity_alignment_error(
    original_poses: &[Pose3d],
    optimized_poses: &[Pose3d],
    gravity_world: Vec3F64,
) -> f64 {
    let gravity_axis = gravity_world.normalize();
    original_poses
        .iter()
        .zip(optimized_poses)
        .map(|(before, after)| {
            let before_camera = (before.rotation * gravity_axis).normalize();
            let after_camera = (after.rotation * gravity_axis).normalize();
            before_camera.dot(after_camera).clamp(-1.0, 1.0).acos()
        })
        .fold(0.0, f64::max)
}

fn pose_graph_cost(poses: &[Pose3d], edges: &[PgoEdge]) -> f64 {
    0.5 * edges
        .iter()
        .filter_map(|edge| weighted_edge_residual_squared(poses, edge))
        .sum::<f64>()
}

fn weighted_edge_residual_norm(poses: &[Pose3d], edge: &PgoEdge) -> Option<f64> {
    weighted_edge_residual_squared(poses, edge).map(f64::sqrt)
}

fn weighted_edge_residual_squared(poses: &[Pose3d], edge: &PgoEdge) -> Option<f64> {
    let pose_a = pose_to_se3(poses.get(edge.pose_a)?);
    let pose_b = pose_to_se3(poses.get(edge.pose_b)?);
    let error = edge.t_ab_meas.inverse() * (pose_b * pose_a.inverse());
    let (translation, rotation) = error.log();
    let weight = edge.weight as f64;
    Some(
        weight
            * weight
            * (translation.x as f64 * translation.x as f64
                + translation.y as f64 * translation.y as f64
                + translation.z as f64 * translation.z as f64
                + rotation.x as f64 * rotation.x as f64
                + rotation.y as f64 * rotation.y as f64
                + rotation.z as f64 * rotation.z as f64),
    )
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

    fn verified_edge(
        query_kf_idx: usize,
        candidate_kf_idx: usize,
        inliers: usize,
        inlier_ratio: f32,
        reprojection_rmse_px: f32,
        occupied_cells: usize,
    ) -> VerifiedLoopEdge {
        VerifiedLoopEdge {
            query_kf_idx,
            candidate_kf_idx,
            candidate_to_query: Pose3d::IDENTITY,
            correspondences: inliers + 5,
            inliers,
            inlier_ratio,
            reprojection_rmse_px,
            occupied_cells,
        }
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
    fn loop_episode_requires_consistency_and_selects_strongest_edge() {
        let mut tracker = LoopEpisodeTracker::new(LoopEpisodeConfig::default());
        let first = verified_edge(100, 20, 31, 0.70, 1.6, 6);
        let strongest = verified_edge(108, 24, 42, 0.84, 1.4, 8);
        let third = verified_edge(116, 16, 36, 0.90, 1.1, 9);

        assert!(matches!(
            tracker.observe(20, 5, first),
            LoopEpisodeDecision::Pending {
                hits: 1,
                required: 3
            }
        ));
        assert!(matches!(
            tracker.observe(21, 6, strongest.clone()),
            LoopEpisodeDecision::Pending {
                hits: 2,
                required: 3
            }
        ));
        assert_eq!(
            tracker.observe(23, 4, third),
            LoopEpisodeDecision::Ready {
                representative: strongest,
                hits: 3,
            }
        );
    }

    #[test]
    fn loop_episode_suppresses_redundant_edges_after_ready() {
        let mut tracker = LoopEpisodeTracker::new(LoopEpisodeConfig {
            min_consistent_edges: 2,
            ..LoopEpisodeConfig::default()
        });
        let representative = verified_edge(100, 20, 35, 0.8, 1.2, 8);
        tracker.observe(20, 5, representative.clone());
        assert!(matches!(
            tracker.observe(21, 6, verified_edge(108, 24, 32, 0.8, 1.3, 7)),
            LoopEpisodeDecision::Ready { .. }
        ));
        assert_eq!(
            tracker.observe(24, 7, verified_edge(120, 28, 45, 0.9, 1.0, 9)),
            LoopEpisodeDecision::Suppressed {
                representative_query_kf_idx: representative.query_kf_idx,
                representative_candidate_kf_idx: representative.candidate_kf_idx,
            }
        );
    }

    #[test]
    fn loop_episode_resets_after_query_gap_or_candidate_jump() {
        let config = LoopEpisodeConfig {
            min_consistent_edges: 3,
            max_query_gap: 3,
            candidate_neighborhood_radius: 4,
        };
        let mut tracker = LoopEpisodeTracker::new(config);
        tracker.observe(10, 5, verified_edge(50, 20, 35, 0.8, 1.2, 8));
        assert!(matches!(
            tracker.observe(14, 6, verified_edge(70, 24, 36, 0.8, 1.2, 8)),
            LoopEpisodeDecision::Pending { hits: 1, .. }
        ));
        assert!(matches!(
            tracker.observe(15, 20, verified_edge(75, 100, 37, 0.8, 1.2, 8)),
            LoopEpisodeDecision::Pending { hits: 1, .. }
        ));
    }

    fn direct_loop_edge() -> VerifiedLoopEdge {
        verified_edge(10, 0, 35, 0.8, 1.0, 8)
    }

    #[test]
    fn loop_fusion_attaches_a_point_to_an_unassociated_loop_keypoint() {
        let mut map = Map::new();
        let descriptor = [7; 32];
        map.upsert_keyframe(Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![descriptor],
        )));
        map.upsert_keyframe(Keyframe::from_frame(frame(
            10,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![descriptor],
        )));
        let point = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            descriptor,
            0,
            [0; 3],
            0,
        ));
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, point);

        let stats = fuse_verified_loop(
            &mut map,
            &camera(),
            &direct_loop_edge(),
            &LoopFusionConfig {
                max_neighbors_per_side: 0,
                ..LoopFusionConfig::default()
            },
        );

        assert_eq!(stats.observations_added, 1);
        assert_eq!(stats.map_points_merged, 0);
        assert_eq!(map.get_keyframe(10).unwrap().map_point(0), Some(point));
        assert!(map.map_points()[point].observation_kf_indices.contains(&10));
    }

    #[test]
    fn loop_fusion_merges_consistent_duplicate_points() {
        let mut map = Map::new();
        let descriptor = [9; 32];
        map.upsert_keyframe(Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![descriptor],
        )));
        map.upsert_keyframe(Keyframe::from_frame(frame(
            10,
            Pose3d::IDENTITY,
            &[[320.8, 240.0]],
            vec![descriptor],
        )));
        let candidate_point = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            descriptor,
            0,
            [0; 3],
            0,
        ));
        let query_point = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.01, 0.0, 5.0),
            descriptor,
            0,
            [0; 3],
            10,
        ));
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, candidate_point);
        map.get_keyframe_mut(10)
            .unwrap()
            .associate_map_point(0, query_point);

        let stats = fuse_verified_loop(
            &mut map,
            &camera(),
            &direct_loop_edge(),
            &LoopFusionConfig {
                max_neighbors_per_side: 0,
                ..LoopFusionConfig::default()
            },
        );

        assert_eq!(stats.observations_added, 0);
        assert_eq!(stats.map_points_merged, 1);
        assert!(!map.map_points()[candidate_point].culled);
        assert!(map.map_points()[query_point].culled);
        assert_eq!(
            map.get_keyframe(10).unwrap().map_point(0),
            Some(candidate_point)
        );
    }

    #[test]
    fn loop_fusion_rejects_descriptor_mismatch() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![[0; 32]],
        )));
        map.upsert_keyframe(Keyframe::from_frame(frame(
            10,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![[u8::MAX; 32]],
        )));
        let point = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
        ));
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, point);

        let stats = fuse_verified_loop(
            &mut map,
            &camera(),
            &direct_loop_edge(),
            &LoopFusionConfig {
                max_neighbors_per_side: 0,
                ..LoopFusionConfig::default()
            },
        );

        assert_eq!(stats, LoopFusionStats::default());
        assert_eq!(map.get_keyframe(10).unwrap().map_point(0), None);
    }

    #[test]
    fn loop_fusion_rejects_duplicate_with_inconsistent_reciprocal_projection() {
        let mut map = Map::new();
        let descriptor = [5; 32];
        map.upsert_keyframe(Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![descriptor],
        )));
        map.upsert_keyframe(Keyframe::from_frame(frame(
            10,
            Pose3d::IDENTITY,
            &[[320.0, 240.0]],
            vec![descriptor],
        )));
        let source_point = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            descriptor,
            0,
            [0; 3],
            0,
        ));
        let inconsistent_target = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            descriptor,
            0,
            [0; 3],
            10,
        ));
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, source_point);
        map.get_keyframe_mut(10)
            .unwrap()
            .associate_map_point(0, inconsistent_target);

        let stats = fuse_verified_loop(
            &mut map,
            &camera(),
            &direct_loop_edge(),
            &LoopFusionConfig {
                max_neighbors_per_side: 0,
                ..LoopFusionConfig::default()
            },
        );

        assert_eq!(stats, LoopFusionStats::default());
        assert!(!map.map_points()[source_point].culled);
        assert!(!map.map_points()[inconsistent_target].culled);
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
        let diagnostic =
            run_shadow_pgo(&map, &[verified], &ShadowPgoConfig::default(), None).unwrap();
        assert_eq!(diagnostic.optimized_poses[0], diagnostic.original_poses[0]);
        assert_eq!(diagnostic.node_count, 5);
        assert_eq!(diagnostic.sequential_edge_count, 4);
        assert_eq!(diagnostic.loop_edge_count, 1);
        assert!(diagnostic.initial_cost.is_finite());
        assert!(diagnostic.final_cost < diagnostic.initial_cost);
        assert!(
            diagnostic.new_loop_residual_after < diagnostic.new_loop_residual_before,
            "expected loop residual {} < {}",
            diagnostic.new_loop_residual_after,
            diagnostic.new_loop_residual_before,
        );
        assert!(diagnostic.solve_time_ms.is_finite());
        assert!(diagnostic.solve_time_ms >= 0.0);
        assert_eq!(diagnostic.usable, diagnostic.converged);
        let before = diagnostic.original_poses[4].inverse().translation.length();
        let after = diagnostic.optimized_poses[4].inverse().translation.length();
        assert!(after < before, "expected {after} < {before}");
        assert!(diagnostic.max_translation_correction > 0.0);
    }

    #[test]
    fn inertial_shadow_pgo_uses_four_dof_and_preserves_gravity() {
        let gravity = Vec3F64::new(0.3, -9.4, 1.1);
        let gravity_axis = gravity.normalize();
        let tilt = kornia_algebra::SO3F64::exp(Vec3F64::new(0.2, -0.1, 0.05)).matrix();
        let mut map = Map::new();
        for index in 0..5 {
            let yaw = index as f64 * 0.03;
            let yaw_world = kornia_algebra::SO3F64::exp(gravity_axis * -yaw).matrix();
            let rotation = tilt * yaw_world;
            let center = Vec3F64::new(index as f64, index as f64 * 0.05, 0.0);
            let pose = Pose3d::new(rotation, -(rotation * center));
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

        let diagnostic = run_shadow_pgo(
            &map,
            &[verified],
            &ShadowPgoConfig::default(),
            Some(InertialPgoContext {
                gravity_world: gravity,
                imu_t_bc: None,
            }),
        )
        .unwrap();

        assert_eq!(diagnostic.mode, PgoMode::Gravity4Dof);
        assert_eq!(diagnostic.optimized_poses[0], diagnostic.original_poses[0]);
        assert!(diagnostic.final_cost < diagnostic.initial_cost);
        assert!(diagnostic.max_gravity_alignment_error_rad.unwrap() <= 1e-4);
        assert_eq!(diagnostic.imu_residual_rms_before, None);
        assert_eq!(diagnostic.imu_residual_rms_after, None);
    }
}
