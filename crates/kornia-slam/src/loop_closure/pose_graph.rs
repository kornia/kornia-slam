use std::collections::HashMap;

use kornia_3d::pgo::{PgoEdge, PgoParams};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use thiserror::Error;

use crate::map::Map;
use crate::pose_conversion::pose_to_se3;
use crate::sparse_pgo::{
    GravityManifold, Se3Manifold, sparse_pose_graph_optimize, weighted_relative_residual,
};

use super::VerifiedLoopEdge;

/// Configuration for pose-graph optimization.
#[derive(Debug, Clone)]
pub struct PgoConfig {
    pub loop_edge_weight: f32,
    pub max_iterations: usize,
    pub cost_tolerance: f32,
    pub gradient_tolerance: f32,
    pub initial_lambda: f32,
}

impl Default for PgoConfig {
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

/// Runtime inertial frame needed by gravity-preserving PGO.
#[derive(Debug, Clone, Copy)]
pub struct InertialPgoContext {
    pub gravity_world: Vec3F64,
}

/// Pose snapshot and optimized poses required for safe live-map writeback.
#[derive(Debug, Clone)]
pub struct PgoResult {
    pub keyframe_indices: Vec<usize>,
    pub original_poses: Vec<Pose3d>,
    pub optimized_poses: Vec<Pose3d>,
    pub iterations: usize,
    pub usable: bool,
}

/// Pose-graph construction or solve failure.
#[derive(Debug, Error)]
pub enum PgoError {
    #[error("pose graph needs at least two keyframes")]
    TooFewKeyframes,
    #[error("loop edge references a keyframe outside the map snapshot")]
    MissingLoopKeyframe,
    #[error("pose graph optimization failed: {0}")]
    Solver(String),
}

/// Optimize a read-only keyframe snapshot for safe live-map writeback.
pub fn optimize_pose_graph(
    map: &Map,
    verified_loops: &[VerifiedLoopEdge],
    config: &PgoConfig,
    inertial: Option<InertialPgoContext>,
) -> Result<PgoResult, PgoError> {
    let keyframes = map.keyframes();
    if keyframes.len() < 2 {
        return Err(PgoError::TooFewKeyframes);
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
            return Err(PgoError::MissingLoopKeyframe);
        };
        let Some(&query_node) = node_by_keyframe.get(&verified.query_kf_idx) else {
            return Err(PgoError::MissingLoopKeyframe);
        };
        edges.push(PgoEdge {
            pose_a: candidate_node,
            pose_b: query_node,
            t_ab_meas: pose_to_se3(&verified.candidate_to_query),
            weight: config.loop_edge_weight,
        });
    }
    let initial_cost = pose_graph_cost(&original_poses, &edges);
    let new_loop_edge = edges.last().ok_or(PgoError::MissingLoopKeyframe)?;
    let new_loop_residual_before = weighted_edge_residual_norm(&original_poses, new_loop_edge)
        .ok_or(PgoError::MissingLoopKeyframe)?;
    let params = PgoParams {
        max_iterations: config.max_iterations,
        cost_tolerance: config.cost_tolerance,
        gradient_tolerance: config.gradient_tolerance,
        initial_lambda: config.initial_lambda,
    };
    let result = if let Some(context) = inertial {
        let manifold = GravityManifold::new(context.gravity_world)
            .map_err(|error| PgoError::Solver(error.to_string()))?;
        sparse_pose_graph_optimize(&original_poses, &edges, &[0], &params, &manifold)
            .map_err(|error| PgoError::Solver(error.to_string()))?
    } else {
        sparse_pose_graph_optimize(&original_poses, &edges, &[0], &params, &Se3Manifold)
            .map_err(|error| PgoError::Solver(error.to_string()))?
    };
    let optimized_poses = result.poses;
    let final_cost = pose_graph_cost(&optimized_poses, &edges);
    let new_loop_residual_after = weighted_edge_residual_norm(&optimized_poses, new_loop_edge)
        .ok_or(PgoError::MissingLoopKeyframe)?;
    let max_gravity_alignment_error_rad = inertial.map(|context| {
        max_gravity_alignment_error(&original_poses, &optimized_poses, context.gravity_world)
    });
    let metrics_finite = [
        initial_cost,
        final_cost,
        new_loop_residual_before,
        new_loop_residual_after,
    ]
    .into_iter()
    .all(f64::is_finite);
    let inertial_metrics_finite = max_gravity_alignment_error_rad.is_none_or(f64::is_finite);
    let gravity_preserved = max_gravity_alignment_error_rad.is_none_or(|error| error <= 1e-4);
    let usable = result.converged
        && metrics_finite
        && inertial_metrics_finite
        && gravity_preserved
        && final_cost < initial_cost
        && new_loop_residual_after <= new_loop_residual_before;

    Ok(PgoResult {
        keyframe_indices,
        original_poses,
        optimized_poses,
        iterations: result.iterations,
        usable,
    })
}

pub(super) fn max_gravity_alignment_error(
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

pub(super) fn pose_graph_cost(poses: &[Pose3d], edges: &[PgoEdge]) -> f64 {
    0.5 * edges
        .iter()
        .filter_map(|edge| weighted_edge_residual_squared(poses, edge))
        .sum::<f64>()
}

fn weighted_edge_residual_norm(poses: &[Pose3d], edge: &PgoEdge) -> Option<f64> {
    weighted_edge_residual_squared(poses, edge).map(f64::sqrt)
}

fn weighted_edge_residual_squared(poses: &[Pose3d], edge: &PgoEdge) -> Option<f64> {
    let residual = weighted_relative_residual(
        poses.get(edge.pose_a)?,
        poses.get(edge.pose_b)?,
        &edge.t_ab_meas,
        edge.weight,
    )
    .ok()?;
    Some(
        residual
            .into_iter()
            .map(|value| f64::from(value) * f64::from(value))
            .sum(),
    )
}
