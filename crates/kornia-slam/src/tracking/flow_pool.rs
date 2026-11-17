//! Long-lived flow workers, one shard each, joined every frame (plan section
//! 3). Not rayon: `rayon::scope` allocates per frame, cannot pin, and shares
//! the global pool ORB runs on.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;

use crate::tracking::flow_tracker::{FlowShard, ShardGuess};
use crate::tracking::timings::ShardTimings;
use crate::upstream::kornia_imgproc::lk_window::LkFrame;

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

/// Runs shard 0 on the calling thread and shard `n` on worker `n`.
pub(crate) struct FlowPool {
    threads: usize,
    workers: Vec<FlowWorker>,
    /// Shard 0's timings, run inline.
    inline_timings: ShardTimings,
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
        })
    }

    /// Runs `shards[0]` inline and `shards[n]` on worker `n`, then joins.
    pub(crate) fn step(
        &mut self,
        _shards: &mut [FlowShard],
        _frames: &FlowFrames,
        _guess: &ShardGuess,
    ) -> Result<(), FlowPoolError> {
        todo!("lane T: dispatch and join")
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
