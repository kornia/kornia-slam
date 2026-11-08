//! What a loop verifier sees of a proximity candidate, and what it answers.

use kornia_3d::pose::Pose3d;

use super::NodeId;

/// The keyframe being added. Poses are world-to-camera.
#[derive(Debug, Clone, Copy)]
pub struct LoopQuery<'a, P> {
    pub keyframe: usize,
    pub t_s: f64,
    /// In the odometry world.
    pub odometry: Pose3d,
    /// In the map world, from the current correction.
    pub estimate: Pose3d,
    pub payload: &'a P,
}

/// A node offered as a loop candidate. Poses are world-to-camera.
#[derive(Debug, Clone, Copy)]
pub struct LoopCandidate<'a, P> {
    pub node: NodeId,
    pub keyframe: Option<usize>,
    pub t_s: f64,
    pub odometry: Pose3d,
    pub pose: Pose3d,
    pub payload: &'a P,
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

/// Decides whether a proximity candidate is the same place, and measures it.
pub trait LoopVerifier<P> {
    fn verify(
        &mut self,
        query: &LoopQuery<'_, P>,
        candidate: &LoopCandidate<'_, P>,
    ) -> Option<LoopMeasurement>;
}

/// Refuses every candidate: the graph then only deduplicates and chains odometry.
#[derive(Debug, Clone, Copy, Default)]
pub struct RejectAll;

impl<P> LoopVerifier<P> for RejectAll {
    fn verify(
        &mut self,
        _: &LoopQuery<'_, P>,
        _: &LoopCandidate<'_, P>,
    ) -> Option<LoopMeasurement> {
        None
    }
}
