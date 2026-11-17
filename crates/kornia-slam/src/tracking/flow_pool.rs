//! Long-lived flow workers, one shard each, joined every frame (plan section
//! 3). Not rayon: `rayon::scope` allocates per frame, cannot pin, and shares
//! the global pool ORB runs on.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::tracking::flow_tracker::{FlowShard, ShardGuess};
use crate::tracking::timings::{MAX_FLOW_SHARDS, ShardTimings, TrackingTimings};
use crate::upstream::kornia_imgproc::lk_window::{LkFrame, LkWindowError};

/// The frames of one flow step, one entry per rig camera.
pub(crate) struct FlowFrames {
    pub previous: Vec<Arc<LkFrame>>,
    pub current: Vec<Arc<LkFrame>>,
}

/// A flow worker that could not start, or stopped.
#[derive(Debug, thiserror::Error)]
pub enum FlowPoolError {
    #[error("flow worker spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("flow worker {0} is gone")]
    WorkerGone(usize),
    #[error("core {0} could not be pinned")]
    Pin(usize),
    #[error("a flow frame is still shared after the join")]
    FrameShared,
    #[error("flow frame: {0}")]
    Frame(#[from] LkWindowError),
}

/// One worker's reused job; moved to the worker and back every step.
struct FlowJob {
    shard: FlowShard,
    previous: Vec<Arc<LkFrame>>,
    current: Vec<Arc<LkFrame>>,
    guess: ShardGuess,
    timings: ShardTimings,
}

/// One named worker `klt-flow-<n>` and its two channels.
struct FlowWorker {
    jobs: SyncSender<FlowJob>,
    done: Receiver<FlowJob>,
    /// The job while it is home; `None` while the worker holds it.
    job: Option<FlowJob>,
    handle: Option<JoinHandle<()>>,
}

/// Where the latest [`FlowPool::step`] spent its time.
#[derive(Debug, Clone, Copy, Default)]
struct StepTimings {
    dispatch: Duration,
    parallel: Duration,
    join: Duration,
    shards: [ShardTimings; MAX_FLOW_SHARDS],
}

/// Runs shard 0 on the calling thread and shard `n` on worker `n`.
pub(crate) struct FlowPool {
    threads: usize,
    workers: Vec<FlowWorker>,
    /// Shard 0's timings, run inline.
    inline_timings: ShardTimings,
    timings: StepTimings,
}

impl FlowPool {
    /// `threads - 1` named workers `klt-flow-<n>`; `threads == 1` spawns nothing.
    pub(crate) fn new(threads: usize, first_core: Option<usize>) -> Result<Self, FlowPoolError> {
        if first_core.is_some() && !cfg!(feature = "pin_threads") {
            eprintln!(
                "kornia-slam: frontend.flow_first_core is set, but this build has no pin_threads \
                 feature; flow workers are not pinned"
            );
        }
        if threads > 1 {
            todo!("lane T: flow workers");
        }
        Ok(Self {
            threads: 1,
            workers: Vec::new(),
            inline_timings: ShardTimings::default(),
            timings: StepTimings::default(),
        })
    }

    /// Runs `shards[0]` inline and `shards[n]` on worker `n`, then joins.
    pub(crate) fn step(
        &mut self,
        shards: &mut [FlowShard],
        frames: &FlowFrames,
        guess: &ShardGuess,
    ) -> Result<(), FlowPoolError> {
        self.timings = StepTimings::default();
        let Some((inline, _)) = shards.split_first_mut() else {
            return Ok(());
        };
        let started = Instant::now();
        inline.run(&frames.previous, &frames.current, guess, &mut self.inline_timings);
        self.timings.parallel = started.elapsed();
        self.timings.shards[0] = self.inline_timings;
        Ok(())
    }

    /// Adds the latest step's times to `timings`.
    pub(crate) fn add_timings(&self, timings: &mut TrackingTimings) {
        timings.flow_dispatch += self.timings.dispatch;
        timings.flow_parallel += self.timings.parallel;
        timings.flow_join += self.timings.join;
        for (total, shard) in timings.shards.iter_mut().zip(&self.timings.shards) {
            total.flow_template += shard.flow_template;
            total.flow_forward += shard.flow_forward;
            total.flow_backward += shard.flow_backward;
            total.flow_project += shard.flow_project;
            total.flow_refill += shard.flow_refill;
        }
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }
}

impl Drop for FlowPool {
    /// Closes each worker's job channel, so it returns, then joins it.
    fn drop(&mut self) {
        for worker in self.workers.drain(..) {
            let FlowWorker { jobs, handle, .. } = worker;
            drop(jobs);
            if let Some(handle) = handle
                && handle.join().is_err()
            {
                eprintln!("kornia-slam: a flow worker panicked");
            }
        }
    }
}
