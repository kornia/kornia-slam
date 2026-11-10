//! Loop closing: place recognition, geometric verification, consistency,
//! fusion, and pose-graph optimization.
//!
//! A verified loop passes the [`LoopGate`]s before it may enter the graph: the
//! odometry gate refuses a correction the drift model cannot explain, as a
//! false loop from repeated structure often implies. Accepted loops close
//! through a [`PoseGraph`]: by default a `w * I` snapshot of the map at each
//! solve, [`optimize_pose_graph`]'s computation; [`LoopGraphSettings`] can keep
//! it across solves, so keyframes the map culls or evicts stay as nodes and
//! their loops keep constraining it.

mod closer;
mod episode;
mod fusion;
mod graph;
mod odometry_gate;
pub mod place_recognition;
mod pose_graph;
mod sparse_pgo;
mod verification;

pub(crate) use closer::{LoopCloser, LoopClosingContext};
pub use closer::{
    LoopClosingConfig, LoopClosureEvent, LoopGraphInformation, LoopGraphSettings,
    OdometryRandomWalk,
};
pub use episode::{LoopEpisodeConfig, LoopEpisodeDecision, LoopEpisodeTracker};
pub use fusion::{LoopFusionConfig, LoopFusionStats, fuse_verified_loop};
pub use graph::{
    Correction, Edge, EdgeKind, GateDecision, GraphManifold, KeyframeOutcome, LoopCandidate,
    LoopGate, LoopMeasurement, LoopProposal, LoopQuery, LoopVerifier, Node, NodeId, NodeSpacing,
    OdometryConsistencyGate, PoseGraph, PoseGraphConfig, PoseGraphError, RejectAll, SolveReport,
    TemporalConsistencyGate,
};
pub use odometry_gate::{
    DriftModel, LoopCorrection, LoopGateVerdict, LoopOdometryGate, LoopTemporalConsistency,
    TemporalDecision,
};
pub use pose_graph::{InertialPgoContext, PgoConfig, PgoError, PgoResult, optimize_pose_graph};
pub use sparse_pgo::{
    GravityManifold, InformationEdge, PoseGraphEdge, PoseManifold, RobustKernel, Se3Manifold,
    SparsePgoError, SparsePgoResult, sparse_pose_graph_optimize,
};
pub use verification::{
    LoopVerificationConfig, LoopVerificationReject, VerifiedLoopEdge, verify_loop_candidate,
};

#[cfg(test)]
use closer::{optimize_through_graph, snapshot_graph_config, solve_in_graph, sync_graph};
#[cfg(test)]
use pose_graph::{max_gravity_alignment_error, pose_graph_cost};
#[cfg(test)]
use verification::verification_input;

#[cfg(test)]
mod tests;
