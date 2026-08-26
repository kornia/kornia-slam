//! Geometric loop verification, consistency, fusion, and pose-graph optimization.

mod episode;
mod fusion;
mod pose_graph;
mod verification;

pub use episode::{LoopEpisodeConfig, LoopEpisodeDecision, LoopEpisodeTracker};
pub use fusion::{LoopFusionConfig, LoopFusionStats, fuse_verified_loop};
pub use pose_graph::{InertialPgoContext, PgoConfig, PgoError, PgoResult, optimize_pose_graph};
pub use verification::{
    LoopVerificationConfig, LoopVerificationReject, VerifiedLoopEdge, verify_loop_candidate,
};

#[cfg(test)]
use pose_graph::{max_gravity_alignment_error, pose_graph_cost};
#[cfg(test)]
use verification::verification_input;

#[cfg(test)]
mod tests;
