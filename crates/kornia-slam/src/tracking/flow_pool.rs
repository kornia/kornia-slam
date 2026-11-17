//! Long-lived flow workers, one shard each, joined every frame (plan section
//! 3). Not rayon: `rayon::scope` allocates per frame, cannot pin, and shares
//! the global pool ORB runs on.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
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

/// How long the join spins on a worker before it parks (plan section 3).
const JOIN_SPIN: Duration = Duration::from_micros(100);

/// One worker's reused job; moved to the worker and back every step.
#[derive(Default)]
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
        let threads = threads.max(1);
        let mut workers = Vec::with_capacity(threads - 1);
        for n in 1..threads {
            let core = first_core.map(|first| first + n - 1);
            workers.push(FlowWorker::spawn(n, core)?);
        }
        Ok(Self {
            threads,
            workers,
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
        let Some((inline, rest)) = shards.split_first_mut() else {
            return Ok(());
        };
        let started = Instant::now();
        for (n, (worker, shard)) in self.workers.iter_mut().zip(rest.iter_mut()).enumerate() {
            worker.dispatch(n + 1, shard, frames, guess)?;
        }
        let dispatched = Instant::now();
        self.timings.dispatch = dispatched - started;

        inline.run(&frames.previous, &frames.current, guess, &mut self.inline_timings);
        for (n, worker) in self.workers.iter_mut().enumerate() {
            worker.wait(n + 1)?;
        }
        let joined = Instant::now();
        self.timings.parallel = joined - dispatched;

        self.timings.shards[0] = self.inline_timings;
        for (n, (worker, shard)) in self.workers.iter_mut().zip(rest.iter_mut()).enumerate() {
            let job = worker.job.as_mut().ok_or(FlowPoolError::WorkerGone(n + 1))?;
            std::mem::swap(&mut job.shard, shard);
            if let Some(slot) = self.timings.shards.get_mut(n + 1) {
                *slot = job.timings;
            }
        }
        self.timings.join = joined.elapsed();
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

impl FlowWorker {
    /// Starts worker `n`, pinned to `core` when given; waits until it runs.
    fn spawn(n: usize, core: Option<usize>) -> Result<Self, FlowPoolError> {
        let (jobs, inbox) = sync_channel::<FlowJob>(1);
        let (outbox, done) = sync_channel::<FlowJob>(1);
        let (ready_tx, ready) = sync_channel::<bool>(1);
        let handle = std::thread::Builder::new()
            .name(format!("klt-flow-{n}"))
            .spawn(move || {
                let pinned = core.is_none_or(pin_current_thread);
                if ready_tx.send(pinned).is_err() || !pinned {
                    return;
                }
                while let Ok(mut job) = inbox.recv() {
                    job.shard
                        .run(&job.previous, &job.current, &job.guess, &mut job.timings);
                    // The tracking core must hold the only frame references
                    // after the join, to rebuild them in place.
                    job.previous.clear();
                    job.current.clear();
                    if outbox.send(job).is_err() {
                        break;
                    }
                }
            })?;
        match ready.recv() {
            Ok(true) => {}
            Ok(false) => return Err(FlowPoolError::Pin(core.unwrap_or_default())),
            Err(_) => return Err(FlowPoolError::WorkerGone(n)),
        }
        Ok(Self {
            jobs,
            done,
            job: Some(FlowJob::default()),
            handle: Some(handle),
        })
    }

    /// Moves `shard` and this step's frames and guess to the worker.
    fn dispatch(
        &mut self,
        n: usize,
        shard: &mut FlowShard,
        frames: &FlowFrames,
        guess: &ShardGuess,
    ) -> Result<(), FlowPoolError> {
        let mut job = self.job.take().ok_or(FlowPoolError::WorkerGone(n))?;
        std::mem::swap(&mut job.shard, shard);
        job.previous.clear();
        job.previous.extend(frames.previous.iter().cloned());
        job.current.clear();
        job.current.extend(frames.current.iter().cloned());
        job.guess.clone_from(guess);
        self.jobs
            .send(job)
            .map_err(|_| FlowPoolError::WorkerGone(n))
    }

    /// Takes the job back: a short spin, since a shard is about as long as
    /// the inline one, then a blocking receive.
    fn wait(&mut self, n: usize) -> Result<(), FlowPoolError> {
        let spin_until = Instant::now() + JOIN_SPIN;
        let job = loop {
            match self.done.try_recv() {
                Ok(job) => break Some(job),
                Err(TryRecvError::Disconnected) => break None,
                Err(TryRecvError::Empty) if Instant::now() < spin_until => {
                    std::hint::spin_loop();
                }
                Err(TryRecvError::Empty) => break self.done.recv().ok(),
            }
        };
        self.job = Some(job.ok_or(FlowPoolError::WorkerGone(n))?);
        Ok(())
    }
}

/// Pins the calling thread to `core`; `false` when the OS refuses.
#[cfg(feature = "pin_threads")]
fn pin_current_thread(core: usize) -> bool {
    core_affinity::set_for_current(core_affinity::CoreId { id: core })
}

/// Without `pin_threads` the deployment confines the process instead.
#[cfg(not(feature = "pin_threads"))]
fn pin_current_thread(_core: usize) -> bool {
    true
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
