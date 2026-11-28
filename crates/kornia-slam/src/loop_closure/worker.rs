//! Pose-graph solves detached from the map, and where they run.
//!
//! A [`LoopSolveJob`] copies what a solve reads, so it can run off the tracking
//! thread; its [`LoopSolveOutcome`] is written back at a frame boundary by the
//! loop closer. As in VINS-Fusion's `optimize4DoF` thread, at most one solve is
//! in flight and loops accepted meanwhile wait for the next job, which takes
//! them all; an in-flight solve is never aborted.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use kornia_3d::pose::Pose3d;

use crate::features::Descriptor;
use crate::loop_closure::closer::{LoopGraphSettings, solve_in_graph};
use crate::loop_closure::fusion_plan::{FusionOp, FusionSnapshot};
use crate::loop_closure::{InertialPgoContext, PgoConfig, PgoError, PgoResult, VerifiedLoopEdge};

/// What one solve reads, copied from the map and the closer when its loop was accepted.
pub(crate) struct LoopSolveJob<D: Descriptor> {
    pub world_epoch: u64,
    /// Map order, world-to-camera.
    pub keyframe_poses: Vec<(usize, Pose3d)>,
    /// Every verified loop; the last is the newest.
    pub loops: Vec<VerifiedLoopEdge>,
    pub optimizer: PgoConfig,
    pub settings: LoopGraphSettings,
    pub inertial: Option<InertialPgoContext>,
    /// The loop neighbourhoods to fuse once solved; `None` fuses at writeback instead.
    pub fusion: Option<FusionSnapshot<D>>,
}

/// A finished solve.
pub(crate) struct LoopSolveOutcome {
    pub world_epoch: u64,
    pub solved: Result<PgoResult, PgoError>,
    /// Odometry and loop edges of the solved graph.
    pub graph_edges: usize,
    pub solve_time: Duration,
    /// The fusion found on the solved snapshot, for the writeback to apply.
    pub fusion_ops: Vec<FusionOp>,
    pub fusion_plan_time: Duration,
}

impl<D: Descriptor> LoopSolveJob<D> {
    pub(crate) fn run(self) -> LoopSolveOutcome {
        let started = Instant::now();
        let solved = solve_in_graph(
            &self.keyframe_poses,
            &self.loops,
            &self.optimizer,
            self.settings,
            self.inertial,
        );
        let solve_time = started.elapsed();
        let fusion_ops = match (&solved, self.fusion) {
            (Ok(result), Some(snapshot)) if result.usable => snapshot.plan(result),
            _ => Vec::new(),
        };
        LoopSolveOutcome {
            world_epoch: self.world_epoch,
            solved,
            graph_edges: self.keyframe_poses.len().saturating_sub(1) + self.loops.len(),
            solve_time,
            fusion_ops,
            fusion_plan_time: started.elapsed().saturating_sub(solve_time),
        }
    }
}

/// The loop-closing worker stopped (it panicked), so no result will come.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerStopped;

/// Where a [`LoopSolveJob`] runs.
pub(crate) enum LoopSolver<D: Descriptor> {
    /// In the keyframe's own call, written back before it returns.
    Synchronous,
    /// Inline when submitted, but written back through the background path `frames` frame
    /// boundaries later (at least one): the asynchronous writeback, reproducibly.
    Replay {
        frames: usize,
        ready: Option<(usize, Box<LoopSolveOutcome>)>,
    },
    /// On the `slam-loop-closing` thread, written back at the first frame boundary after it ends.
    Asynchronous(LoopWorker<D>),
}

impl<D: Descriptor> LoopSolver<D> {
    pub(crate) fn replay(frames: usize) -> Self {
        Self::Replay {
            frames,
            ready: None,
        }
    }

    /// # Errors
    /// The worker thread could not be spawned.
    pub(crate) fn asynchronous() -> std::io::Result<Self> {
        LoopWorker::spawn(LoopSolveJob::run).map(Self::Asynchronous)
    }

    pub(crate) fn is_synchronous(&self) -> bool {
        matches!(self, Self::Synchronous)
    }

    pub(crate) fn in_flight(&self) -> bool {
        match self {
            Self::Synchronous => false,
            Self::Replay { ready, .. } => ready.is_some(),
            Self::Asynchronous(worker) => worker.in_flight,
        }
    }

    /// Starts `job`; a synchronous solver returns its outcome at once.
    pub(crate) fn submit(
        &mut self,
        job: LoopSolveJob<D>,
    ) -> Result<Option<LoopSolveOutcome>, WorkerStopped> {
        match self {
            Self::Synchronous => Ok(Some(job.run())),
            Self::Replay { frames, ready } => {
                *ready = Some(((*frames).max(1), Box::new(job.run())));
                Ok(None)
            }
            Self::Asynchronous(worker) => worker.submit(job).map(|()| None),
        }
    }

    /// Called once per frame boundary: the finished outcome, if one is due.
    pub(crate) fn poll(&mut self) -> Result<Option<LoopSolveOutcome>, WorkerStopped> {
        match self {
            Self::Synchronous => Ok(None),
            Self::Replay { ready, .. } => {
                let Some((remaining, _)) = ready.as_mut() else {
                    return Ok(None);
                };
                *remaining -= 1;
                if *remaining > 0 {
                    return Ok(None);
                }
                Ok(ready.take().map(|(_, outcome)| *outcome))
            }
            Self::Asynchronous(worker) => worker.poll(),
        }
    }

    /// The outcome of the solve in flight, waiting for it however many frames or how long it
    /// still needs; `None` when nothing is in flight.
    pub(crate) fn wait(&mut self) -> Result<Option<LoopSolveOutcome>, WorkerStopped> {
        match self {
            Self::Synchronous => Ok(None),
            Self::Replay { ready, .. } => Ok(ready.take().map(|(_, outcome)| *outcome)),
            Self::Asynchronous(worker) => worker.wait(),
        }
    }
}

/// The background thread. Dropping it waits for an in-flight solve to end, up to about a second
/// on a large graph.
pub(crate) struct LoopWorker<D: Descriptor> {
    jobs: Option<mpsc::Sender<LoopSolveJob<D>>>,
    results: mpsc::Receiver<LoopSolveOutcome>,
    join_handle: Option<JoinHandle<()>>,
    in_flight: bool,
}

impl<D: Descriptor> LoopWorker<D> {
    fn spawn(solve: fn(LoopSolveJob<D>) -> LoopSolveOutcome) -> std::io::Result<Self> {
        let (jobs, job_queue) = mpsc::channel::<LoopSolveJob<D>>();
        let (result_sender, results) = mpsc::channel();
        let join_handle = thread::Builder::new()
            .name("slam-loop-closing".into())
            .spawn(move || {
                for job in job_queue {
                    if result_sender.send(solve(job)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            jobs: Some(jobs),
            results,
            join_handle: Some(join_handle),
            in_flight: false,
        })
    }

    fn submit(&mut self, job: LoopSolveJob<D>) -> Result<(), WorkerStopped> {
        let jobs = self.jobs.as_ref().ok_or(WorkerStopped)?;
        jobs.send(job).map_err(|_| WorkerStopped)?;
        self.in_flight = true;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<LoopSolveOutcome>, WorkerStopped> {
        if !self.in_flight {
            return Ok(None);
        }
        match self.results.try_recv() {
            Ok(outcome) => {
                self.in_flight = false;
                Ok(Some(outcome))
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.in_flight = false;
                Err(WorkerStopped)
            }
        }
    }

    fn wait(&mut self) -> Result<Option<LoopSolveOutcome>, WorkerStopped> {
        if !self.in_flight {
            return Ok(None);
        }
        self.in_flight = false;
        self.results.recv().map(Some).map_err(|_| WorkerStopped)
    }
}

impl<D: Descriptor> Drop for LoopWorker<D> {
    fn drop(&mut self) {
        // Closing the queue ends the worker's loop once its current solve returns.
        self.jobs = None;
        if let Some(handle) = self.join_handle.take()
            && handle.join().is_err()
        {
            eprintln!("kornia-slam: loop-closing worker panicked; loop correction had stopped");
        }
    }
}

#[cfg(test)]
impl<D: Descriptor> LoopSolver<D> {
    /// An asynchronous solver whose worker panics on its first job.
    pub(crate) fn panicking_for_test() -> Self {
        Self::Asynchronous(
            LoopWorker::spawn(|_| panic!("loop-closing worker panic for test"))
                .expect("spawn test worker"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{LoopSolveJob, LoopSolver, WorkerStopped};
    use crate::features::RBrief;
    use crate::loop_closure::{LoopGraphSettings, PgoConfig, PgoError};

    fn empty_job() -> LoopSolveJob<RBrief> {
        LoopSolveJob {
            world_epoch: 7,
            keyframe_poses: Vec::new(),
            loops: Vec::new(),
            optimizer: PgoConfig::default(),
            settings: LoopGraphSettings::default(),
            inertial: None,
            fusion: None,
        }
    }

    fn poll_until_done(
        solver: &mut LoopSolver<RBrief>,
    ) -> Result<Option<super::LoopSolveOutcome>, WorkerStopped> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let polled = solver.poll();
            if !matches!(polled, Ok(None)) || Instant::now() > deadline {
                return polled;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn replayed_outcome_arrives_after_its_frames() {
        let mut solver = LoopSolver::replay(3);
        assert!(solver.submit(empty_job()).unwrap().is_none());
        assert!(solver.in_flight());
        assert!(solver.poll().unwrap().is_none());
        assert!(solver.poll().unwrap().is_none());
        let outcome = solver.poll().unwrap().expect("due at the third boundary");
        assert_eq!(outcome.world_epoch, 7);
        assert!(matches!(outcome.solved, Err(PgoError::TooFewKeyframes)));
        assert!(!solver.in_flight());
    }

    #[test]
    fn asynchronous_outcome_arrives_from_the_worker() {
        let mut solver = LoopSolver::asynchronous().unwrap();
        assert!(solver.submit(empty_job()).unwrap().is_none());
        assert!(solver.in_flight());
        let outcome = poll_until_done(&mut solver)
            .unwrap()
            .expect("solved within 30 s");
        assert_eq!(outcome.world_epoch, 7);
        assert!(!solver.in_flight());
    }

    #[test]
    fn waiting_returns_the_outcome_in_flight() {
        for mut solver in [LoopSolver::replay(3), LoopSolver::asynchronous().unwrap()] {
            assert!(solver.wait().unwrap().is_none());
            solver.submit(empty_job()).unwrap();
            let outcome = solver.wait().unwrap().expect("a solve was in flight");
            assert_eq!(outcome.world_epoch, 7);
            assert!(!solver.in_flight());
        }
    }

    #[test]
    fn a_panicking_worker_reports_stopped() {
        let mut solver = LoopSolver::<RBrief>::panicking_for_test();
        solver.submit(empty_job()).unwrap();
        assert_eq!(poll_until_done(&mut solver).err(), Some(WorkerStopped));
        assert!(!solver.in_flight());
        assert_eq!(solver.submit(empty_job()).err(), Some(WorkerStopped));
    }
}
