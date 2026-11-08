//! Loop closing: place recognition, geometric verification, consistency,
//! fusion, and pose-graph optimization.
//!
//! A verified loop passes the [`LoopGate`]s before it may enter the graph: the
//! odometry gate refuses a correction the drift model cannot explain, which is
//! what repeated structure produces. Accepted loops close through one
//! [`PoseGraph`] kept across solves, so keyframes the map culls or evicts stay
//! as nodes and their loops keep constraining it. [`optimize_pose_graph`] is
//! the earlier per-solve snapshot optimizer, kept as the `w * I` reference.
//!
//! A solve can run off the tracking thread ([`crate::LoopClosingExecution`]): it
//! reads a copy of the keyframe poses and the graph, and its result is written
//! back at a frame boundary, moving keyframes created during the solve with the
//! last one it saw, as ORB-SLAM3's `CorrectLoop` and VINS-Fusion's pose graph do.

mod closer;
mod episode;
mod fusion;
mod graph;
mod odometry_gate;
pub mod place_recognition;
mod pose_graph;
mod sparse_pgo;
mod verification;
mod worker;

pub(crate) use closer::{LoopCloser, LoopClosingContext, LoopClosingOutcome, LoopTiming};
pub use closer::{LoopClosingConfig, LoopClosureEvent, LoopGraphInformation};
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
pub(crate) use worker::LoopSolver;

#[cfg(test)]
use closer::{keyframe_poses, optimize_through_graph, snapshot_graph_config, solve_in_graph};
#[cfg(test)]
use pose_graph::{max_gravity_alignment_error, pose_graph_cost};
#[cfg(test)]
use verification::verification_input;

#[cfg(test)]
mod tests;
