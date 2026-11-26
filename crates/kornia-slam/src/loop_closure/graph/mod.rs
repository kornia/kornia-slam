//! One global, poses-only pose graph over the odometry keyframes.
//!
//! Poses are world-to-camera, as everywhere in this crate. The odometry world
//! drifts; the map world is the odometry world as the solve corrects it, and the
//! first node is fixed. A keyframe inside an existing node's footprint makes no
//! node, so the graph grows with the area covered rather than with time.

mod gate;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};

use kornia_3d::pgo::PgoParams;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{CauchyKernel, RobustKernel};
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use thiserror::Error;

use crate::loop_closure::sparse_pgo::{
    GravityManifold, InformationEdge, Se3Manifold, SparsePgoResult, sparse_pose_graph_optimize,
    weighted_relative_residual,
};
use crate::pose_conversion::pose_to_se3;
use crate::upstream::kornia_3d::pose_ext::camera_centre;

pub use gate::{
    GateDecision, LoopGate, LoopProposal, OdometryConsistencyGate, TemporalConsistencyGate,
};

/// A node's index, in creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

/// The tangent space the solve moves poses in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphManifold {
    /// Full 6-DoF; distances and turns are measured in 3D.
    Se3,
    /// x, y, z and yaw about this world gravity direction; roll and pitch stay,
    /// and distances and turns are measured in the plane normal to it.
    Gravity(Vec3F64),
}

/// A verified loop and its noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopMeasurement {
    /// `T_query * T_candidate^-1` for world-to-camera `T`, as in [`VerifiedLoopEdge`].
    ///
    /// [`VerifiedLoopEdge`]: crate::loop_closure::VerifiedLoopEdge
    pub candidate_to_query: Pose3d,
    /// One standard deviation of each translation component (m).
    pub sigma_m: f64,
    /// One standard deviation of each rotation component (rad).
    pub sigma_rad: f64,
}

/// Node, drift and solver settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphConfig {
    pub manifold: GraphManifold,
    /// Odometry drift, linear in path: translation sigma per metre travelled, rotation sigma per
    /// radian turned and per metre travelled.
    pub drift_m_per_m: f64,
    pub drift_rad_per_rad: f64,
    pub drift_rad_per_m: f64,
    /// Random-walk odometry noise, variance per metre travelled (m^2/m, rad^2/m), added in
    /// quadrature to the linear terms. Unlike them, it composes along a chain of edges to the same
    /// sigma whatever the node spacing.
    pub walk_m2_per_m: f64,
    pub walk_rad2_per_m: f64,
    /// Floors on an odometry edge's sigmas, so a zero-length edge is not infinitely certain.
    pub min_sigma_m: f64,
    pub min_sigma_rad: f64,
    /// Cauchy scale on loop edges, in whitened (sigma) units; `None` is least squares.
    pub loop_cauchy_scale: Option<f64>,
    pub max_iterations: usize,
    pub cost_tolerance: f32,
    pub gradient_tolerance: f32,
    pub initial_lambda: f32,
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        Self {
            manifold: GraphManifold::Se3,
            drift_m_per_m: 0.02,
            drift_rad_per_rad: 0.03,
            drift_rad_per_m: 0.005,
            walk_m2_per_m: 0.0,
            walk_rad2_per_m: 0.0,
            min_sigma_m: 0.01,
            min_sigma_rad: 0.002,
            loop_cauchy_scale: Some(3.0),
            max_iterations: 50,
            cost_tolerance: 1e-7,
            gradient_tolerance: 1e-7,
            initial_lambda: 1e-3,
        }
    }
}

/// `p_map = map_from_odometry * p_odometry`: where the map world places the odometry world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correction {
    pub map_from_odometry: Pose3d,
}

#[derive(Debug, Error)]
pub enum PoseGraphError {
    #[error("keyframe {0} already has a node")]
    DuplicateKeyframe(usize),
    #[error("keyframe {0} has no node")]
    UnknownKeyframe(usize),
    #[error("node {0} does not exist")]
    UnknownNode(usize),
    #[error("a loop edge needs two distinct nodes")]
    SelfLoop,
    #[error("pose-graph solve failed: {0}")]
    Solver(String),
}

/// A graph node. Poses are world-to-camera.
#[derive(Debug, Clone, PartialEq)]
pub struct Node<P> {
    /// The keyframe that made it; `None` once forgotten.
    pub keyframe: Option<usize>,
    pub t_s: f64,
    /// In the odometry world.
    pub odometry: Pose3d,
    /// In the map world.
    pub pose: Pose3d,
    pub payload: P,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeKind {
    Odometry,
    Loop,
}

/// A relative-pose constraint with diagonal information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    /// `T_to * T_from^-1` for world-to-camera `T`, as in [`InformationEdge::t_ab_meas`].
    pub measurement: Pose3d,
    pub sigma_m: f64,
    pub sigma_rad: f64,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    seq: u64,
    edge: Edge,
}

/// Where the newest pose hangs from: `estimate = odometry * vio_ref^-1 * offset * pose(node)`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Anchor {
    node: NodeId,
    /// `T_ref * T_node^-1`, the reference camera seen from the node in the map world.
    offset: Pose3d,
    odometry_ref: Pose3d,
    path_m: f64,
    turn_rad: f64,
    t_s: f64,
}

/// How [`PoseGraph::add_linked_keyframe`] joins a keyframe across a gap in tracked odometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyframeLink {
    /// An odometry edge from the anchor at no less than these sigmas: a pose carried across the
    /// gap, such as a coast on the IMU.
    Bridged { sigma_m: f64, sigma_rad: f64 },
    /// No odometry edge, since nothing measured the jump from the anchor: a loop edge from `from`
    /// instead, and the path restarts at the new node.
    Jump {
        from: NodeId,
        measurement: LoopMeasurement,
    },
}

/// How a solve ended. Costs are `0.5 * sum rho(|S r|^2)`; the loop residual is the whitened norm
/// of the last loop edge added.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    pub iterations: usize,
    pub converged: bool,
    pub cost_before: f64,
    pub cost_after: f64,
    pub loop_residual_before: f64,
    pub loop_residual_after: f64,
}

/// See the module doc. `P` is a payload each node carries.
#[derive(Debug, Clone)]
pub struct PoseGraph<P> {
    config: PoseGraphConfig,
    nodes: Vec<Node<P>>,
    by_keyframe: HashMap<usize, NodeId>,
    /// Keyed by (from, to, kind): a repeated loop between two nodes replaces, so edges grow with
    /// area too.
    edges: BTreeMap<(NodeId, NodeId, EdgeKind), Slot>,
    next_seq: u64,
    anchor: Option<Anchor>,
    last_odometry: Option<Pose3d>,
    path_m: f64,
    loops_accepted: u64,
}

impl<P> PoseGraph<P> {
    pub fn new(config: PoseGraphConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            by_keyframe: HashMap::new(),
            edges: BTreeMap::new(),
            next_seq: 0,
            anchor: None,
            last_odometry: None,
            path_m: 0.0,
            loops_accepted: 0,
        }
    }

    pub fn config(&self) -> &PoseGraphConfig {
        &self.config
    }

    /// The next solve moves poses in `manifold`; gravity can be refined between solves.
    pub fn set_manifold(&mut self, manifold: GraphManifold) {
        self.config.manifold = manifold;
    }

    pub fn nodes(&self) -> &[Node<P>] {
        &self.nodes
    }

    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values().map(|slot| &slot.edge)
    }

    pub fn loops_accepted(&self) -> u64 {
        self.loops_accepted
    }

    pub fn node_of_keyframe(&self, keyframe: usize) -> Option<NodeId> {
        self.by_keyframe.get(&keyframe).copied()
    }

    /// Drop everything: an odometry reset starts a new world, so the old nodes no longer apply.
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }

    /// The map-world pose of a keyframe whose odometry pose is `odometry`.
    pub fn estimate(&self, odometry: &Pose3d) -> Pose3d {
        match &self.anchor {
            Some(anchor) => odometry
                .compose(&anchor.odometry_ref.inverse())
                .compose(&anchor.offset)
                .compose(&self.nodes[anchor.node.0].pose),
            None => *odometry,
        }
    }

    /// Live keyframes in node order with their odometry and map-world poses: the
    /// `apply_pose_graph_correction` snapshot.
    pub fn poses_before_after(&self) -> (Vec<usize>, Vec<Pose3d>, Vec<Pose3d>) {
        let live = self
            .nodes
            .iter()
            .filter_map(|node| node.keyframe.map(|keyframe| (keyframe, node)));
        let mut keyframes = Vec::with_capacity(self.by_keyframe.len());
        let mut before = Vec::with_capacity(self.by_keyframe.len());
        let mut after = Vec::with_capacity(self.by_keyframe.len());
        for (keyframe, node) in live {
            keyframes.push(keyframe);
            before.push(node.odometry);
            after.push(node.pose);
        }
        (keyframes, before, after)
    }

    /// The keyframe left the source map; its node and edges stay as constraints.
    pub fn forget_keyframe(&mut self, keyframe: usize) {
        if let Some(node) = self.by_keyframe.remove(&keyframe) {
            self.nodes[node.0].keyframe = None;
        }
    }

    /// Reseed a node from a source that moved the keyframe, such as bundle adjustment or a
    /// written-back solve: its odometry and map-world poses both become `pose`, and its odometry
    /// edges are measured again.
    ///
    /// # Errors
    /// The keyframe has no node.
    pub fn refresh_odometry(
        &mut self,
        keyframe: usize,
        pose: Pose3d,
    ) -> Result<(), PoseGraphError> {
        let node = self
            .node_of_keyframe(keyframe)
            .ok_or(PoseGraphError::UnknownKeyframe(keyframe))?;
        self.nodes[node.0].odometry = pose;
        self.nodes[node.0].pose = pose;
        let nodes = &self.nodes;
        for edge in self
            .edges
            .values_mut()
            .map(|slot| &mut slot.edge)
            .filter(|edge| {
                edge.kind == EdgeKind::Odometry && (edge.from == node || edge.to == node)
            })
        {
            edge.measurement =
                Pose3d::between(&nodes[edge.from.0].odometry, &nodes[edge.to.0].odometry);
        }
        if let Some(anchor) = self.anchor.as_mut()
            && anchor.node == node
            && anchor.offset == Pose3d::IDENTITY
        {
            anchor.odometry_ref = pose;
        }
        Ok(())
    }

    /// The next keyframe's path is measured from `odometry`, for a source whose world moved under
    /// the graph, such as a written-back solve: the move is not distance travelled.
    pub fn rebase_path(&mut self, odometry: Pose3d) {
        self.last_odometry = Some(odometry);
    }

    /// Feed one odometry keyframe, `odometry` world-to-camera in the odometry world: a node
    /// joined to the anchor by a drift odometry edge.
    ///
    /// # Errors
    /// The keyframe already has a node.
    pub fn add_keyframe(
        &mut self,
        keyframe: usize,
        t_s: f64,
        odometry: Pose3d,
        payload: P,
    ) -> Result<NodeId, PoseGraphError> {
        if self.by_keyframe.contains_key(&keyframe) {
            return Err(PoseGraphError::DuplicateKeyframe(keyframe));
        }
        if self.anchor.is_none() {
            let node = self.push_node(keyframe, t_s, odometry, odometry, payload);
            self.anchor = Some(self.anchor_at_node(node, t_s));
            self.last_odometry = Some(odometry);
            return Ok(node);
        }
        self.advance_path(&odometry);
        let estimate = self.estimate(&odometry);
        Ok(self.push_node_with_odometry(keyframe, t_s, odometry, estimate, payload))
    }

    /// A node for `keyframe`, joined by `link` instead of the drift odometry edge, without a loop
    /// search or a solve. The first keyframe ignores `link`.
    ///
    /// # Errors
    /// The keyframe already has a node, or a jump's `from` node does not exist.
    pub fn add_linked_keyframe(
        &mut self,
        keyframe: usize,
        t_s: f64,
        odometry: Pose3d,
        payload: P,
        link: KeyframeLink,
    ) -> Result<NodeId, PoseGraphError> {
        if self.by_keyframe.contains_key(&keyframe) {
            return Err(PoseGraphError::DuplicateKeyframe(keyframe));
        }
        let Some(anchor) = self.anchor else {
            let node = self.push_node(keyframe, t_s, odometry, odometry, payload);
            self.anchor = Some(self.anchor_at_node(node, t_s));
            self.last_odometry = Some(odometry);
            return Ok(node);
        };
        let to = match link {
            KeyframeLink::Bridged { sigma_m, sigma_rad } => {
                self.advance_path(&odometry);
                let estimate = self.estimate(&odometry);
                let (measurement, drift_m, drift_rad) = self.odometry_since_anchor(&odometry);
                let to = self.push_node(keyframe, t_s, odometry, estimate, payload);
                self.insert_edge(Edge {
                    from: anchor.node,
                    to,
                    measurement,
                    sigma_m: sigma_m.max(drift_m),
                    sigma_rad: sigma_rad.max(drift_rad),
                    kind: EdgeKind::Odometry,
                });
                to
            }
            KeyframeLink::Jump { from, measurement } => {
                let from_pose = self
                    .nodes
                    .get(from.0)
                    .ok_or(PoseGraphError::UnknownNode(from.0))?
                    .pose;
                // The jump is not distance travelled.
                self.last_odometry = Some(odometry);
                let estimate = measurement.candidate_to_query.compose(&from_pose);
                let to = self.push_node(keyframe, t_s, odometry, estimate, payload);
                self.insert_edge(Edge {
                    from,
                    to,
                    measurement: measurement.candidate_to_query,
                    sigma_m: measurement.sigma_m,
                    sigma_rad: measurement.sigma_rad,
                    kind: EdgeKind::Loop,
                });
                to
            }
        };
        self.anchor = Some(self.anchor_at_node(to, t_s));
        Ok(to)
    }

    /// Add verified loops between keyframes that own nodes, then solve once.
    ///
    /// # Errors
    /// A keyframe has no node, both ends are one node, or the solve failed; then nothing of the
    /// loops stays.
    pub fn add_loop_edges(
        &mut self,
        loops: impl IntoIterator<Item = (usize, usize, LoopMeasurement)>,
    ) -> Result<SolveReport, PoseGraphError> {
        let mut edges = Vec::new();
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
            edges.push(Edge {
                from,
                to,
                measurement: measurement.candidate_to_query,
                sigma_m: measurement.sigma_m,
                sigma_rad: measurement.sigma_rad,
                kind: EdgeKind::Loop,
            });
        }
        self.commit_loops(edges, |_| {})
    }

    /// [`Self::add_loop_edges`] for one loop.
    ///
    /// # Errors
    /// As [`Self::add_loop_edges`].
    pub fn add_loop_edge(
        &mut self,
        candidate_kf: usize,
        query_kf: usize,
        measurement: LoopMeasurement,
    ) -> Result<SolveReport, PoseGraphError> {
        self.add_loop_edges([(candidate_kf, query_kf, measurement)])
    }

    fn push_node(
        &mut self,
        keyframe: usize,
        t_s: f64,
        odometry: Pose3d,
        pose: Pose3d,
        payload: P,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            keyframe: Some(keyframe),
            t_s,
            odometry,
            pose,
            payload,
        });
        self.by_keyframe.insert(keyframe, id);
        id
    }

    fn anchor_at_node(&self, node: NodeId, t_s: f64) -> Anchor {
        Anchor {
            node,
            offset: Pose3d::IDENTITY,
            odometry_ref: self.nodes[node.0].odometry,
            path_m: 0.0,
            turn_rad: 0.0,
            t_s,
        }
    }

    fn advance_path(&mut self, odometry: &Pose3d) {
        if let (Some(last), Some(anchor)) = (self.last_odometry, self.anchor.as_mut()) {
            let step = (camera_centre(odometry) - camera_centre(&last)).length();
            let turn = turn_between(&self.config.manifold, &last, odometry);
            self.path_m += step;
            anchor.path_m += step;
            anchor.turn_rad += turn;
        }
        self.last_odometry = Some(*odometry);
    }

    /// The odometry since the anchor as `T_now * T_anchor_node^-1`, with its drift sigmas.
    fn odometry_since_anchor(&self, odometry: &Pose3d) -> (Pose3d, f64, f64) {
        let Some(anchor) = self.anchor else {
            return (
                Pose3d::IDENTITY,
                self.config.min_sigma_m,
                self.config.min_sigma_rad,
            );
        };
        let relative = odometry
            .compose(&anchor.odometry_ref.inverse())
            .compose(&anchor.offset);
        let sigma_m = (self.config.drift_m_per_m * anchor.path_m)
            .hypot((self.config.walk_m2_per_m * anchor.path_m).sqrt())
            .max(self.config.min_sigma_m);
        let sigma_rad = (self.config.drift_rad_per_rad * anchor.turn_rad
            + self.config.drift_rad_per_m * anchor.path_m)
            .hypot((self.config.walk_rad2_per_m * anchor.path_m).sqrt())
            .max(self.config.min_sigma_rad);
        (relative, sigma_m, sigma_rad)
    }

    fn push_node_with_odometry(
        &mut self,
        keyframe: usize,
        t_s: f64,
        odometry: Pose3d,
        estimate: Pose3d,
        payload: P,
    ) -> NodeId {
        let (measurement, sigma_m, sigma_rad) = self.odometry_since_anchor(&odometry);
        let from = self.anchor.map_or(NodeId(0), |anchor| anchor.node);
        let to = self.push_node(keyframe, t_s, odometry, estimate, payload);
        self.insert_edge(Edge {
            from,
            to,
            measurement,
            sigma_m,
            sigma_rad,
            kind: EdgeKind::Odometry,
        });
        self.anchor = Some(self.anchor_at_node(to, t_s));
        to
    }

    /// Insert `loops` and solve; on a solver error nothing of them stays.
    fn commit_loops(
        &mut self,
        loops: Vec<Edge>,
        on_success: impl FnOnce(&mut Self),
    ) -> Result<SolveReport, PoseGraphError> {
        let Some(last) = loops.last().copied() else {
            return Err(PoseGraphError::Solver("no loop edge to add".into()));
        };
        let mut previous = Vec::with_capacity(loops.len());
        for edge in &loops {
            previous.push(self.insert_edge(*edge));
        }
        match self.solve(&last) {
            Ok((poses, report)) => {
                for (node, pose) in self.nodes.iter_mut().zip(poses) {
                    node.pose = pose;
                    // Nothing measures a forgotten node again, so its solved pose is its odometry
                    // from now on: a live neighbour's refresh re-measures their edge against it.
                    if node.keyframe.is_none() {
                        node.odometry = pose;
                    }
                }
                self.loops_accepted += loops.len() as u64;
                on_success(self);
                Ok(report)
            }
            Err(error) => {
                for (key, old) in previous.into_iter().rev() {
                    match old {
                        Some(old) => self.edges.insert(key, old),
                        None => self.edges.remove(&key),
                    };
                }
                Err(error)
            }
        }
    }

    fn insert_edge(&mut self, edge: Edge) -> ((NodeId, NodeId, EdgeKind), Option<Slot>) {
        let key = (edge.from, edge.to, edge.kind);
        let seq = self.next_seq;
        self.next_seq += 1;
        (key, self.edges.insert(key, Slot { seq, edge }))
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
            .map(|first| camera_centre(&first.pose));
        let poses: Vec<Pose3d> = self
            .nodes
            .iter()
            .map(|node| origin.map_or(node.pose, |origin| about(&node.pose, origin)))
            .collect();
        // Odometry then loops, each in insertion order: the f32 normal equations then sum in the
        // order `optimize_pose_graph` uses, which LM's accept path is sensitive to.
        let mut slots: Vec<&Slot> = self.edges.values().collect();
        slots.sort_by_key(|slot| (slot.edge.kind, slot.seq));
        let edges: Vec<InformationEdge> = slots
            .into_iter()
            .map(|slot| self.information_edge(&slot.edge))
            .collect::<Result<_, _>>()?;
        let params = PgoParams {
            max_iterations: self.config.max_iterations,
            cost_tolerance: self.config.cost_tolerance,
            gradient_tolerance: self.config.gradient_tolerance,
            initial_lambda: self.config.initial_lambda,
        };
        let solver =
            |error: crate::loop_closure::SparsePgoError| PoseGraphError::Solver(error.to_string());
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
        let new_loop = self.information_edge(new_loop)?;
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

    fn information_edge(&self, edge: &Edge) -> Result<InformationEdge, PoseGraphError> {
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
        let t_ab_meas = pose_to_se3(&edge.measurement);
        Ok(InformationEdge {
            pose_a: edge.from.0,
            pose_b: edge.to.0,
            t_ab_meas,
            sqrt_information,
            cauchy_scale,
        })
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

/// The world-frame turn from `a` to `b`: its twist about gravity on a gravity manifold, so it
/// holds for a tilted camera too, else its full angle (rad, non-negative).
fn turn_between(manifold: &GraphManifold, a: &Pose3d, b: &Pose3d) -> f64 {
    let world = b.rotation.transpose() * a.rotation;
    match manifold {
        GraphManifold::Se3 => SO3F64::from_matrix(&world).log().length(),
        GraphManifold::Gravity(gravity) => twist_about(&world, gravity.normalize()).abs(),
    }
}

/// The swing-twist twist angle of `rotation` about the unit `axis`, radians in (-pi, pi].
fn twist_about(rotation: &Mat3F64, axis: Vec3F64) -> f64 {
    let [x, y, z, w] = SO3F64::from_matrix(rotation).to_array();
    wrap_angle(2.0 * Vec3F64::new(x, y, z).dot(axis).atan2(w))
}

fn wrap_angle(angle: f64) -> f64 {
    let wrapped = angle.rem_euclid(std::f64::consts::TAU);
    if wrapped > std::f64::consts::PI {
        wrapped - std::f64::consts::TAU
    } else {
        wrapped
    }
}
