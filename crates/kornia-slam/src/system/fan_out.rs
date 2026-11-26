//! The rayon pools of `pipeline.fan_out`.
//!
//! rayon's idle workers yield 32 times before they sleep, so on 2 cores the
//! global pool's workers take time from the mapping thread between every
//! fork-join of ORB, pyramids and matching. `FlowPool` avoids rayon for the
//! same reason.

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

use super::config::FanOut;

/// Where the tracking thread's feature extraction forks.
pub(crate) struct TrackingFanOut(Option<ThreadPool>);

impl TrackingFanOut {
    /// # Errors
    /// The private pool's threads could not be spawned.
    pub(crate) fn new(fan_out: FanOut) -> Result<Self, ThreadPoolBuildError> {
        match fan_out {
            FanOut::Global => Ok(Self(None)),
            FanOut::Bounded(bounded) => ThreadPoolBuilder::new()
                .num_threads(bounded.tracking_threads)
                .thread_name(|n| format!("kornia-fanout-{n}"))
                .build()
                .map(|pool| Self(Some(pool))),
        }
    }

    /// Runs `f` on the private pool, or on the calling thread under `Global`.
    pub(crate) fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.0 {
            None => f(),
            Some(pool) => pool.install(f),
        }
    }
}

/// Makes the calling worker thread a 1-thread rayon pool of its own, so its
/// parallel iterators and faer's run inline; keep the pool alive for the
/// thread's life. `None` under `Global`. rayon leaks the registry, once per
/// worker thread.
pub(crate) fn bind_worker_thread(fan_out: FanOut) -> Option<ThreadPool> {
    match fan_out {
        FanOut::Global => None,
        FanOut::Bounded(_) => match ThreadPoolBuilder::new()
            .num_threads(1)
            .use_current_thread()
            .build()
        {
            Ok(pool) => Some(pool),
            Err(error) => {
                eprintln!("kornia-slam: worker stays on the global rayon pool: {error}");
                None
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{TrackingFanOut, bind_worker_thread};
    use crate::system::config::{BoundedFanOut, FanOut};

    fn bounded(tracking_threads: usize) -> FanOut {
        FanOut::Bounded(BoundedFanOut { tracking_threads })
    }

    #[test]
    fn bounded_extraction_runs_on_a_private_pool() {
        let fan_out = TrackingFanOut::new(bounded(1)).unwrap();
        let (name, threads) = fan_out.run(|| {
            (
                std::thread::current().name().map(str::to_owned),
                rayon::current_num_threads(),
            )
        });
        assert_eq!(name.as_deref(), Some("kornia-fanout-0"));
        assert_eq!(threads, 1);
    }

    #[test]
    fn global_extraction_runs_on_the_calling_thread() {
        let fan_out = TrackingFanOut::new(FanOut::Global).unwrap();
        let caller = std::thread::current().id();
        assert_eq!(fan_out.run(|| std::thread::current().id()), caller);
    }

    #[test]
    fn a_bound_worker_forks_only_into_itself() {
        let threads = std::thread::spawn(|| {
            let _pool = bind_worker_thread(bounded(1));
            let worker = std::thread::current().id();
            let inline = rayon::join(
                || std::thread::current().id(),
                || std::thread::current().id(),
            );
            (rayon::current_num_threads(), worker, inline)
        })
        .join()
        .unwrap();
        assert_eq!(threads.0, 1);
        assert_eq!(threads.2, (threads.1, threads.1));
        assert!(bind_worker_thread(FanOut::Global).is_none());
    }
}
