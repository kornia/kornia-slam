//! One global, poses-only pose graph over the odometry keyframes.
//!
//! Poses are world-to-camera, as everywhere in this crate. The odometry world
//! drifts; the map world is the odometry world as the solve corrects it, and the
//! first node is fixed. A keyframe inside an existing node's footprint makes no
//! node, so the graph grows with the area covered rather than with time.

mod gate;
mod verify;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};

use kornia_3d::pgo::PgoParams;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use thiserror::Error;

use crate::loop_closure::sparse_pgo::{
    GravityManifold, InformationEdge, RobustKernel, Se3Manifold, SparsePgoResult,
    sparse_pose_graph_optimize, weighted_relative_residual,
};
use crate::pose_conversion::pose_to_se3;

pub use gate::{
    GateDecision, LoopGate, LoopProposal, OdometryConsistencyGate, TemporalConsistencyGate,
};
pub use verify::{LoopCandidate, LoopMeasurement, LoopQuery, LoopVerifier, RejectAll};

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

/// When a keyframe is close enough to an existing node to make none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeSpacing {
    /// Within this of a node, both in distance and in yaw, a keyframe makes no node.
    pub min_distance_m: f64,
    pub min_yaw_rad: f64,
    /// Only nodes this close are compared for the two thresholds above.
    pub neighbour_radius_m: f64,
    /// A node is made anyway after this long without making one or re-anchoring onto one, so an
    /// odometry edge never spans more.
    pub max_interval_s: f64,
}

impl Default for NodeSpacing {
    fn default() -> Self {
        Self {
            min_distance_m: 0.3,
            min_yaw_rad: 15f64.to_radians(),
            neighbour_radius_m: 1.0,
            max_interval_s: 30.0,
        }
    }
}

/// Node, drift, candidate and solver settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphConfig {
    pub manifold: GraphManifold,
    /// `None` makes every keyframe a node.
    pub node_spacing: Option<NodeSpacing>,
    /// Odometry drift, linear in path: translation sigma per metre travelled, rotation sigma per
    /// radian turned and per metre travelled.
    pub drift_m_per_m: f64,
    pub drift_rad_per_rad: f64,
    pub drift_rad_per_m: f64,
    /// Floors on an odometry edge's sigmas, so a zero-length edge is not infinitely certain.
    pub min_sigma_m: f64,
    pub min_sigma_rad: f64,
    /// A loop candidate lies within this many drift sigmas of the current estimate.
    pub loop_gate_sigmas: f64,
    pub loop_max_yaw_rad: f64,
    /// A node visited less than this much path ago is the same pass, not a loop.
    pub loop_min_path_m: f64,
    /// At most this many candidates go to the verifier per keyframe, nearest first.
    pub loop_max_candidates: usize,
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
            node_spacing: Some(NodeSpacing::default()),
            drift_m_per_m: 0.02,
            drift_rad_per_rad: 0.03,
            drift_rad_per_m: 0.005,
            min_sigma_m: 0.01,
            min_sigma_rad: 0.002,
            loop_gate_sigmas: 3.0,
            loop_max_yaw_rad: 40f64.to_radians(),
            loop_min_path_m: 2.0,
            loop_max_candidates: 3,
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
    /// Path length at the last visit, created or re-anchored to.
    visited_at_path_m: f64,
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

/// What one keyframe did to the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeOutcome {
    /// Inside an existing node's footprint and no loop: no node.
    Skipped,
    NewNode(NodeId),
    /// A loop was accepted and the graph solved.
    LoopClosed,
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

/// See the module doc. `P` is whatever the verifier needs to match two places.
#[derive(Debug, Clone)]
pub struct PoseGraph<P> {
    config: PoseGraphConfig,
    nodes: Vec<Node<P>>,
    by_keyframe: HashMap<usize, NodeId>,
    /// Keyed by (from, to, kind): a repeated loop between two nodes replaces, so edges grow with
    /// area too.
    edges: BTreeMap<(NodeId, NodeId, EdgeKind), Edge>,
    anchor: Option<Anchor>,
    last_odometry: Option<Pose3d>,
    path_m: f64,
    loops_accepted: u64,
    loops_rejected: u64,
}

impl<P> PoseGraph<P> {
    pub fn new(config: PoseGraphConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            by_keyframe: HashMap::new(),
            edges: BTreeMap::new(),
            anchor: None,
            last_odometry: None,
            path_m: 0.0,
            loops_accepted: 0,
            loops_rejected: 0,
        }
    }

    pub fn config(&self) -> &PoseGraphConfig {
        &self.config
    }

    pub fn nodes(&self) -> &[Node<P>] {
        &self.nodes
    }

    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    pub fn loops_accepted(&self) -> u64 {
        self.loops_accepted
    }

    pub fn loops_rejected(&self) -> u64 {
        self.loops_rejected
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

    /// The correction now, yaw and translation only on a gravity manifold; identity before the
    /// first keyframe.
    pub fn correction(&self) -> Correction {
        let Some(anchor) = &self.anchor else {
            return Correction {
                map_from_odometry: Pose3d::IDENTITY,
            };
        };
        let full = self.nodes[anchor.node.0]
            .pose
            .inverse()
            .compose(&anchor.offset.inverse())
            .compose(&anchor.odometry_ref);
        let map_from_odometry = match self.config.manifold {
            GraphManifold::Se3 => full,
            GraphManifold::Gravity(gravity) => {
                let axis = gravity.normalize();
                let yaw = twist_about(&full.rotation, axis);
                Pose3d::new(SO3F64::exp(axis * yaw).matrix(), full.translation)
            }
        };
        Correction { map_from_odometry }
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
        for edge in self.edges.values_mut().filter(|edge| {
            edge.kind == EdgeKind::Odometry && (edge.from == node || edge.to == node)
        }) {
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

    /// Feed one odometry keyframe, `odometry` world-to-camera in the odometry world.
    ///
    /// # Errors
    /// The keyframe already has a node, or the loop solve failed; then the graph is left as it was
    /// before this keyframe's loop.
    pub fn add_keyframe(
        &mut self,
        keyframe: usize,
        t_s: f64,
        odometry: Pose3d,
        payload: P,
        verifier: &mut dyn LoopVerifier<P>,
    ) -> Result<KeyframeOutcome, PoseGraphError> {
        if self.by_keyframe.contains_key(&keyframe) {
            return Err(PoseGraphError::DuplicateKeyframe(keyframe));
        }
        let Some(anchor) = self.anchor else {
            let node = self.push_node(keyframe, t_s, odometry, odometry, payload);
            self.anchor = Some(self.anchor_at_node(node, t_s));
            self.last_odometry = Some(odometry);
            return Ok(KeyframeOutcome::NewNode(node));
        };
        self.advance_path(&odometry);
        let estimate = self.estimate(&odometry);
        let candidates = self.loop_candidates(&estimate);

        let reuse = self.config.node_spacing.is_some_and(|spacing| {
            t_s - anchor.t_s < spacing.max_interval_s
                && self.is_near_existing_node(&estimate, &spacing)
        });
        if reuse {
            // A revisit inside an existing node's footprint: a verified loop re-anchors onto it
            // without a new node, which is what keeps the node count tied to area.
            // The anchor itself is skipped: the loop edge needs two distinct nodes.
            let query = LoopQuery {
                keyframe,
                t_s,
                odometry,
                estimate,
                payload: &payload,
            };
            for &candidate in candidates.iter().filter(|&&node| node != anchor.node) {
                if let Some(measurement) = self.consult(verifier, &query, candidate) {
                    self.close_by_reanchor(t_s, &odometry, candidate, &measurement)?;
                    return Ok(KeyframeOutcome::LoopClosed);
                }
            }
            return Ok(KeyframeOutcome::Skipped);
        }

        let new_node = self.push_node_with_odometry(keyframe, t_s, odometry, estimate, payload);
        for &candidate in &candidates {
            let node = &self.nodes[new_node.0];
            let query = LoopQuery {
                keyframe,
                t_s,
                odometry,
                estimate,
                payload: &node.payload,
            };
            let measurement = verifier.verify(&query, &self.candidate(candidate));
            match measurement {
                Some(measurement) => {
                    let edge = Edge {
                        from: candidate,
                        to: new_node,
                        measurement: measurement.candidate_to_query,
                        sigma_m: measurement.sigma_m,
                        sigma_rad: measurement.sigma_rad,
                        kind: EdgeKind::Loop,
                    };
                    self.commit_loops(vec![edge], |_| {})?;
                    self.nodes[candidate.0].visited_at_path_m = self.path_m;
                    return Ok(KeyframeOutcome::LoopClosed);
                }
                None => self.loops_rejected += 1,
            }
        }
        Ok(KeyframeOutcome::NewNode(new_node))
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
        let visited: Vec<_> = edges.iter().map(|edge| edge.from).collect();
        let report = self.commit_loops(edges, |_| {})?;
        for node in visited {
            self.nodes[node.0].visited_at_path_m = self.path_m;
        }
        Ok(report)
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
            visited_at_path_m: self.path_m,
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
            let step = (centre(odometry) - centre(&last)).length();
            let turn = turn_between(&self.config.manifold, &last, odometry);
            self.path_m += step;
            anchor.path_m += step;
            anchor.turn_rad += turn;
        }
        self.last_odometry = Some(*odometry);
    }

    fn is_near_existing_node(&self, estimate: &Pose3d, spacing: &NodeSpacing) -> bool {
        let manifold = &self.config.manifold;
        self.nodes.iter().any(|node| {
            let distance = distance_between(manifold, &node.pose, estimate);
            distance <= spacing.neighbour_radius_m
                && distance <= spacing.min_distance_m
                && turn_between(manifold, &node.pose, estimate) <= spacing.min_yaw_rad
        })
    }

    /// Nodes from an earlier pass whose distance to `estimate` is inside the drift gate, nearest
    /// first.
    fn loop_candidates(&self, estimate: &Pose3d) -> Vec<NodeId> {
        if self.config.loop_max_candidates == 0 {
            return Vec::new();
        }
        let manifold = &self.config.manifold;
        let mut gated: Vec<(f64, NodeId)> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let path_since = self.path_m - node.visited_at_path_m;
                if path_since < self.config.loop_min_path_m {
                    return None;
                }
                let sigma = self.config.min_sigma_m + self.config.drift_m_per_m * path_since;
                let distance = distance_between(manifold, &node.pose, estimate);
                let inside = distance <= self.config.loop_gate_sigmas * sigma
                    && turn_between(manifold, &node.pose, estimate) < self.config.loop_max_yaw_rad;
                inside.then_some((distance, NodeId(index)))
            })
            .collect();
        gated.sort_by(|a, b| a.0.total_cmp(&b.0));
        gated
            .into_iter()
            .take(self.config.loop_max_candidates)
            .map(|(_, node)| node)
            .collect()
    }

    fn candidate(&self, node: NodeId) -> LoopCandidate<'_, P> {
        let candidate = &self.nodes[node.0];
        LoopCandidate {
            node,
            keyframe: candidate.keyframe,
            t_s: candidate.t_s,
            odometry: candidate.odometry,
            pose: candidate.pose,
            payload: &candidate.payload,
        }
    }

    fn consult(
        &mut self,
        verifier: &mut dyn LoopVerifier<P>,
        query: &LoopQuery<'_, P>,
        node: NodeId,
    ) -> Option<LoopMeasurement> {
        let measurement = verifier.verify(query, &self.candidate(node));
        if measurement.is_none() {
            self.loops_rejected += 1;
        }
        measurement
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
        let sigma_m = (self.config.drift_m_per_m * anchor.path_m).max(self.config.min_sigma_m);
        let sigma_rad = (self.config.drift_rad_per_rad * anchor.turn_rad
            + self.config.drift_rad_per_m * anchor.path_m)
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
        self.edges.insert(
            (from, to, EdgeKind::Odometry),
            Edge {
                from,
                to,
                measurement,
                sigma_m,
                sigma_rad,
                kind: EdgeKind::Odometry,
            },
        );
        self.anchor = Some(self.anchor_at_node(to, t_s));
        to
    }

    /// Chain the odometry since the anchor with the verified measurement into one loop edge from
    /// the anchor node to the candidate, then hang the head from the candidate.
    fn close_by_reanchor(
        &mut self,
        t_s: f64,
        odometry: &Pose3d,
        candidate: NodeId,
        measurement: &LoopMeasurement,
    ) -> Result<(), PoseGraphError> {
        let Some(anchor) = self.anchor else {
            return Ok(());
        };
        let (since_anchor, odo_sigma_m, odo_sigma_rad) = self.odometry_since_anchor(odometry);
        let edge = Edge {
            from: anchor.node,
            to: candidate,
            measurement: measurement
                .candidate_to_query
                .inverse()
                .compose(&since_anchor),
            sigma_m: odo_sigma_m.hypot(measurement.sigma_m),
            sigma_rad: odo_sigma_rad.hypot(measurement.sigma_rad),
            kind: EdgeKind::Loop,
        };
        let odometry_ref = *odometry;
        let offset = measurement.candidate_to_query;
        self.commit_loops(vec![edge], |graph| {
            graph.anchor = Some(Anchor {
                node: candidate,
                offset,
                odometry_ref,
                path_m: 0.0,
                turn_rad: 0.0,
                t_s,
            });
        })?;
        self.nodes[candidate.0].visited_at_path_m = self.path_m;
        Ok(())
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
            let key = (edge.from, edge.to, edge.kind);
            previous.push((key, self.edges.insert(key, *edge)));
        }
        match self.solve(&last) {
            Ok((poses, report)) => {
                for (node, pose) in self.nodes.iter_mut().zip(poses) {
                    node.pose = pose;
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

    /// Solve over every node, the first fixed.
    fn solve(&self, new_loop: &Edge) -> Result<(Vec<Pose3d>, SolveReport), PoseGraphError> {
        let poses: Vec<Pose3d> = self.nodes.iter().map(|node| node.pose).collect();
        let edges: Vec<InformationEdge> = self
            .edges
            .values()
            .map(|edge| self.information_edge(edge))
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
        Ok((result.poses, report))
    }

    fn information_edge(&self, edge: &Edge) -> Result<InformationEdge, PoseGraphError> {
        let mut sqrt_information = [[0.0f32; 6]; 6];
        for (row, values) in sqrt_information.iter_mut().enumerate() {
            let sigma = if row < 3 {
                edge.sigma_m
            } else {
                edge.sigma_rad
            };
            values[row] = (1.0 / sigma) as f32;
        }
        let kernel = match (edge.kind, self.config.loop_cauchy_scale) {
            (EdgeKind::Loop, Some(scale)) => RobustKernel::Cauchy {
                scale: scale as f32,
            },
            _ => RobustKernel::None,
        };
        let t_ab_meas = pose_to_se3(&edge.measurement);
        Ok(InformationEdge {
            pose_a: edge.from.0,
            pose_b: edge.to.0,
            t_ab_meas,
            sqrt_information,
            kernel,
        })
    }
}

/// `0.5 * sum rho(|S r|^2)` in f64 over f32 whitened residuals, the solver's own cost.
fn graph_cost(poses: &[Pose3d], edges: &[InformationEdge]) -> f64 {
    0.5 * edges
        .iter()
        .filter_map(|edge| {
            let squared = whitened_squared(poses, edge)?;
            Some(match edge.kernel {
                RobustKernel::None => squared,
                RobustKernel::Cauchy { scale } => {
                    let c2 = f64::from(scale) * f64::from(scale);
                    c2 * (squared / c2).ln_1p()
                }
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

fn centre(pose: &Pose3d) -> Vec3F64 {
    pose.inverse().translation
}

/// Camera-centre distance, in the plane normal to gravity on a gravity manifold.
fn distance_between(manifold: &GraphManifold, a: &Pose3d, b: &Pose3d) -> f64 {
    let offset = centre(b) - centre(a);
    match manifold {
        GraphManifold::Se3 => offset.length(),
        GraphManifold::Gravity(gravity) => {
            let axis = gravity.normalize();
            (offset - axis * offset.dot(axis)).length()
        }
    }
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
