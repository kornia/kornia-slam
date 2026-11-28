//! The poses-only pose graph a loop solve runs on, built from the map's keyframes at each solve.
//!
//! Poses are world-to-camera, as everywhere in this crate. Every keyframe is a node and the first
//! is fixed. Consecutive keyframes in map order are joined by an odometry edge whose noise is a
//! random walk in the path between them; verified loops add loop edges. The solve hands the
//! solver the odometry edges in map order, then the loop edges in the order they were added.

mod gate;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use kornia_3d::pgo::PgoParams;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{CauchyKernel, RobustKernel};
use kornia_algebra::{Mat3F64, Vec3F64};
use thiserror::Error;

use crate::loop_closure::sparse_pgo::{
    GravityManifold, InformationEdge, Se3Manifold, SparsePgoError, SparsePgoResult,
    sparse_pose_graph_optimize, weighted_relative_residual,
};
use crate::pose_conversion::pose_to_se3;

pub(crate) use gate::{
    GateDecision, LoopGate, LoopProposal, OdometryConsistencyGate, TemporalConsistencyGate,
};

/// A node's index, in map order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub(crate) usize);

/// The tangent space the solve moves poses in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GraphManifold {
    /// Full 6-DoF.
    Se3,
    /// x, y, z and yaw about this world gravity direction; roll and pitch stay.
    Gravity(Vec3F64),
}

/// A verified loop and its noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LoopMeasurement {
    /// `T_query * T_candidate^-1` for world-to-camera `T`, as in [`VerifiedLoopEdge`].
    ///
    /// [`VerifiedLoopEdge`]: crate::loop_closure::VerifiedLoopEdge
    pub(crate) candidate_to_query: Pose3d,
    /// One standard deviation of each translation component (m).
    pub(crate) sigma_m: f64,
    /// One standard deviation of each rotation component (rad).
    pub(crate) sigma_rad: f64,
}

/// Odometry noise and solver settings.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PoseGraphConfig {
    pub(crate) manifold: GraphManifold,
    /// Random-walk odometry noise, variance per metre travelled (m^2/m, rad^2/m): a chain of
    /// edges composes to the same sigma whatever the keyframe spacing.
    pub(crate) walk_m2_per_m: f64,
    pub(crate) walk_rad2_per_m: f64,
    /// Floors on an odometry edge's sigmas, so a zero-length edge is not infinitely certain.
    pub(crate) min_sigma_m: f64,
    pub(crate) min_sigma_rad: f64,
    /// Cauchy scale on loop edges, in whitened (sigma) units; `None` is least squares.
    pub(crate) loop_cauchy_scale: Option<f64>,
    pub(crate) max_iterations: usize,
    pub(crate) cost_tolerance: f32,
    pub(crate) gradient_tolerance: f32,
    pub(crate) initial_lambda: f32,
}

#[derive(Debug, Error)]
pub(crate) enum PoseGraphError {
    #[error("keyframe {0} already has a node")]
    DuplicateKeyframe(usize),
    #[error("keyframe {0} has no node")]
    UnknownKeyframe(usize),
    #[error("a loop edge needs two distinct nodes")]
    SelfLoop,
    #[error("pose-graph solve failed: {0}")]
    Solver(String),
}

/// A graph node: the keyframe's world-to-camera pose as the map held it, and as solved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub(crate) keyframe: usize,
    pub(crate) snapshot: Pose3d,
    pub(crate) pose: Pose3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeKind {
    Odometry,
    Loop,
}

/// A relative-pose constraint with diagonal information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Edge {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    /// `T_to * T_from^-1` for world-to-camera `T`.
    pub(crate) measurement: Pose3d,
    pub(crate) sigma_m: f64,
    pub(crate) sigma_rad: f64,
    pub(crate) kind: EdgeKind,
}

/// How a solve ended. Costs are `0.5 * sum rho(|S r|^2)`; the loop residual is the whitened norm
/// of the last loop edge added.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SolveReport {
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) cost_before: f64,
    pub(crate) cost_after: f64,
    pub(crate) loop_residual_before: f64,
    pub(crate) loop_residual_after: f64,
}

/// See the module doc.
#[derive(Debug, Clone)]
pub(crate) struct PoseGraph {
    config: PoseGraphConfig,
    nodes: Vec<Node>,
    by_keyframe: HashMap<usize, NodeId>,
    odometry: Vec<Edge>,
    loops: Vec<Edge>,
}

impl PoseGraph {
    pub(crate) fn new(config: PoseGraphConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            by_keyframe: HashMap::new(),
            odometry: Vec::new(),
            loops: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Odometry edges in map order, then loop edges in the order they were added.
    pub(crate) fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.odometry.iter().chain(&self.loops)
    }

    pub(crate) fn node_of_keyframe(&self, keyframe: usize) -> Option<NodeId> {
        self.by_keyframe.get(&keyframe).copied()
    }

    /// Keyframes in node order with their snapshot and solved poses: the
    /// `apply_pose_graph_correction` arguments.
    pub(crate) fn poses_before_after(&self) -> (Vec<usize>, Vec<Pose3d>, Vec<Pose3d>) {
        let keyframes = self.nodes.iter().map(|node| node.keyframe).collect();
        let before = self.nodes.iter().map(|node| node.snapshot).collect();
        let after = self.nodes.iter().map(|node| node.pose).collect();
        (keyframes, before, after)
    }

    /// Appends the next keyframe in map order at world-to-camera `pose`, joined to the one before
    /// it by an odometry edge whose sigmas grow as the square root of the path between them.
    ///
    /// # Errors
    /// The keyframe already has a node.
    pub(crate) fn add_keyframe(
        &mut self,
        keyframe: usize,
        pose: Pose3d,
    ) -> Result<NodeId, PoseGraphError> {
        if self.by_keyframe.contains_key(&keyframe) {
            return Err(PoseGraphError::DuplicateKeyframe(keyframe));
        }
        let to = NodeId(self.nodes.len());
        if let Some(previous) = self.nodes.last() {
            let step_m = (pose.camera_centre() - previous.snapshot.camera_centre()).length();
            let config = &self.config;
            self.odometry.push(Edge {
                from: NodeId(to.0 - 1),
                to,
                measurement: Pose3d::between(&previous.snapshot, &pose),
                sigma_m: (config.walk_m2_per_m * step_m)
                    .sqrt()
                    .max(config.min_sigma_m),
                sigma_rad: (config.walk_rad2_per_m * step_m)
                    .sqrt()
                    .max(config.min_sigma_rad),
                kind: EdgeKind::Odometry,
            });
        }
        self.nodes.push(Node {
            keyframe,
            snapshot: pose,
            pose,
        });
        self.by_keyframe.insert(keyframe, to);
        Ok(to)
    }

    /// Adds verified `(candidate, query, measurement)` loops between keyframes that own nodes,
    /// then solves once and moves every node to its solved pose.
    ///
    /// # Errors
    /// No loop is given, a keyframe has no node, both ends are one node, or the solve failed.
    pub(crate) fn add_loop_edges(
        &mut self,
        loops: impl IntoIterator<Item = (usize, usize, LoopMeasurement)>,
    ) -> Result<SolveReport, PoseGraphError> {
        for (candidate_kf, query_kf, measurement) in loops {
            let from = self
                .node_of_keyframe(candidate_kf)
                .ok_or(PoseGraphError::UnknownKeyframe(candidate_kf))?;
            let to = self
                .node_of_keyframe(query_kf)
                .ok_or(PoseGraphError::UnknownKeyframe(query_kf))?;
            if from == to {
                return Err(PoseGraphError::SelfLoop);
            }
            self.loops.push(Edge {
                from,
                to,
                measurement: measurement.candidate_to_query,
                sigma_m: measurement.sigma_m,
                sigma_rad: measurement.sigma_rad,
                kind: EdgeKind::Loop,
            });
        }
        let new_loop = *self
            .loops
            .last()
            .ok_or_else(|| PoseGraphError::Solver("no loop edge to add".into()))?;
        let (poses, report) = self.solve(&new_loop)?;
        for (node, pose) in self.nodes.iter_mut().zip(poses) {
            node.pose = pose;
        }
        Ok(report)
    }

    /// Solve over every node, the first fixed.
    fn solve(&self, new_loop: &Edge) -> Result<(Vec<Pose3d>, SolveReport), PoseGraphError> {
        // The f32 residual `t_b - R_b R_a^T t_a` cancels at the scale of the camera centres, so a
        // robust solve runs about node 0's centre: `T_b W (T_a W)^-1 = T_b T_a^-1`, and the gravity
        // retraction commutes with a translation. Least squares keeps `optimize_pose_graph`'s.
        let origin = self
            .config
            .loop_cauchy_scale
            .and(self.nodes.first())
            .map(|first| first.pose.camera_centre());
        let poses: Vec<Pose3d> = self
            .nodes
            .iter()
            .map(|node| origin.map_or(node.pose, |origin| about(&node.pose, origin)))
            .collect();
        // Odometry then loops: the f32 normal equations then sum in the order
        // `optimize_pose_graph` uses, which LM's accept path is sensitive to.
        let edges: Vec<InformationEdge> = self
            .edges()
            .map(|edge| self.information_edge(edge))
            .collect();
        let params = PgoParams {
            max_iterations: self.config.max_iterations,
            cost_tolerance: self.config.cost_tolerance,
            gradient_tolerance: self.config.gradient_tolerance,
            initial_lambda: self.config.initial_lambda,
        };
        let solver = |error: SparsePgoError| PoseGraphError::Solver(error.to_string());
        let result: SparsePgoResult = match self.config.manifold {
            GraphManifold::Se3 => {
                sparse_pose_graph_optimize(&poses, &edges, &[0], &params, &Se3Manifold)
                    .map_err(solver)?
            }
            GraphManifold::Gravity(gravity) => {
                let manifold = GravityManifold::new(gravity).map_err(solver)?;
                sparse_pose_graph_optimize(&poses, &edges, &[0], &params, &manifold)
                    .map_err(solver)?
            }
        };
        let new_loop = self.information_edge(new_loop);
        let report = SolveReport {
            iterations: result.iterations,
            converged: result.converged,
            cost_before: graph_cost(&poses, &edges),
            cost_after: graph_cost(&result.poses, &edges),
            loop_residual_before: whitened_squared(&poses, &new_loop).map_or(f64::NAN, f64::sqrt),
            loop_residual_after: whitened_squared(&result.poses, &new_loop)
                .map_or(f64::NAN, f64::sqrt),
        };
        let solved = match origin {
            Some(origin) => result
                .poses
                .iter()
                .map(|pose| about(pose, -origin))
                .collect(),
            None => result.poses,
        };
        Ok((solved, report))
    }

    fn information_edge(&self, edge: &Edge) -> InformationEdge {
        let cauchy_scale = match (edge.kind, self.config.loop_cauchy_scale) {
            (EdgeKind::Loop, Some(scale)) => Some(scale as f32),
            _ => None,
        };
        let mut sqrt_information = [[0.0f32; 6]; 6];
        for (row, values) in sqrt_information.iter_mut().enumerate().take(3) {
            values[row] = (1.0 / edge.sigma_m) as f32;
        }
        let rotation = match (self.config.manifold, cauchy_scale) {
            // Every node keeps `R·g`, so a robust edge's weight would see the roll and pitch of its
            // measurement that no state can meet: it compares only the twist about gravity, in
            // camera `from` (the residual's frame). Least squares keeps the full block, which is
            // `optimize_pose_graph`'s.
            (GraphManifold::Gravity(gravity), Some(_)) => {
                let up = self.nodes[edge.from.0].pose.rotation * gravity.normalize();
                Mat3F64::from_cols(up * up.x, up * up.y, up * up.z)
            }
            _ => Mat3F64::IDENTITY,
        };
        for row in 0..3 {
            for col in 0..3 {
                sqrt_information[3 + row][3 + col] =
                    (rotation.col(col)[row] / edge.sigma_rad) as f32;
            }
        }
        InformationEdge {
            pose_a: edge.from.0,
            pose_b: edge.to.0,
            t_ab_meas: pose_to_se3(&edge.measurement),
            sqrt_information,
            cauchy_scale,
        }
    }
}

/// World-to-camera `pose` in a world whose origin is at `origin`.
fn about(pose: &Pose3d, origin: Vec3F64) -> Pose3d {
    Pose3d::new(pose.rotation, pose.translation + pose.rotation * origin)
}

/// `0.5 * sum rho(|S r|^2)` in f64 over f32 whitened residuals, the solver's own cost.
fn graph_cost(poses: &[Pose3d], edges: &[InformationEdge]) -> f64 {
    0.5 * edges
        .iter()
        .filter_map(|edge| {
            let squared = whitened_squared(poses, edge)?;
            Some(match edge.cauchy_scale {
                None => squared,
                Some(scale) => CauchyKernel.rho(squared, f64::from(scale) * f64::from(scale)),
            })
        })
        .sum::<f64>()
}

fn whitened_squared(poses: &[Pose3d], edge: &InformationEdge) -> Option<f64> {
    let raw = weighted_relative_residual(
        poses.get(edge.pose_a)?,
        poses.get(edge.pose_b)?,
        &edge.t_ab_meas,
        1.0,
    )
    .ok()?;
    Some(
        (0..6)
            .map(|row| {
                let value: f32 = (0..6)
                    .map(|col| edge.sqrt_information[row][col] * raw[col])
                    .sum();
                f64::from(value) * f64::from(value)
            })
            .sum(),
    )
}
