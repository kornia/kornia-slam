//! Pose-graph solves detached from the map, and where they run.
//!
//! A [`LoopSolveJob`] copies what a solve reads, so it can run off the tracking
//! thread; its [`LoopSolveOutcome`] is written back at a frame boundary by the
//! loop closer. As in VINS-Fusion's `optimize4DoF` thread, at most one solve is
//! in flight and loops accepted meanwhile wait for the next job, which takes
//! them all; an in-flight solve is never aborted.

use std::time::{Duration, Instant};

use kornia_3d::pose::Pose3d;

use crate::loop_closure::VerifiedLoopEdge;
use crate::loop_closure::closer::{LoopGraphInformation, solve_in_graph};
use crate::loop_closure::{InertialPgoContext, PgoConfig, PgoError, PgoResult, PoseGraph};

/// What one solve reads, copied from the map and the closer when its loop was accepted.
pub(crate) struct LoopSolveJob {
    /// Map order, world-to-camera.
    pub keyframe_poses: Vec<(usize, Pose3d)>,
    /// Verified loops not yet in `graph`; the last is the newest.
    pub loops: Vec<VerifiedLoopEdge>,
    pub graph: PoseGraph<()>,
    pub optimizer: PgoConfig,
    pub information: LoopGraphInformation,
    pub inertial: Option<InertialPgoContext>,
}

/// A finished solve. `graph` holds the solved graph; the closer adopts it only once the result
/// is written back.
pub(crate) struct LoopSolveOutcome {
    pub graph: PoseGraph<()>,
    /// The result, and whether the map was a new world to the graph.
    pub solved: Result<(PgoResult, bool), PgoError>,
    pub loops: Vec<VerifiedLoopEdge>,
    pub solve_time: Duration,
}

impl LoopSolveJob {
    pub(crate) fn run(self) -> LoopSolveOutcome {
        let started = Instant::now();
        let mut graph = self.graph;
        let solved = solve_in_graph(
            &mut graph,
            &self.keyframe_poses,
            &self.loops,
            &self.optimizer,
            self.information,
            self.inertial,
        );
        LoopSolveOutcome {
            graph,
            solved,
            loops: self.loops,
            solve_time: started.elapsed(),
        }
    }
}
