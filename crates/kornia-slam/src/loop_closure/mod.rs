//! Loop closing: place recognition, geometric verification, consistency,
//! fusion, and pose-graph optimization.

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
use graph::{
    GateDecision, GraphManifold, LoopGate, LoopMeasurement, LoopProposal, OdometryConsistencyGate,
    PoseGraph, PoseGraphConfig, PoseGraphError, TemporalConsistencyGate,
};
pub use odometry_gate::{DriftModel, LoopOdometryGate, LoopTemporalConsistency};
use odometry_gate::{LoopCorrection, TemporalDecision};
pub use pose_graph::{InertialPgoContext, PgoConfig, PgoError, PgoResult, optimize_pose_graph};
pub use verification::{
    LoopVerificationConfig, LoopVerificationReject, VerifiedLoopEdge, verify_loop_candidate,
};

#[cfg(test)]
use closer::{keyframe_poses, solve_in_graph};
#[cfg(test)]
use pose_graph::{max_gravity_alignment_error, pose_graph_cost};
#[cfg(test)]
use verification::verification_input;

#[cfg(test)]
mod tests;
