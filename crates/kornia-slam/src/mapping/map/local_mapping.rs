use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use super::{LocalBaMergeResult, LocalBaSnapshot, Map};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LocalMappingMode {
    Synchronous,
    #[default]
    Asynchronous,
}

impl FromStr for LocalMappingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "sync" | "synchronous" => Ok(Self::Synchronous),
            "async" | "asynchronous" => Ok(Self::Asynchronous),
            _ => Err(format!(
                "invalid local-mapping mode '{value}'; expected sync or async"
            )),
        }
    }
}

pub struct KeyframeJob {
    pub imu_initialized: bool,
    pub imu_t_bc: Option<Pose3d>,
    pub gravity_world: Vec3F64,
}

fn solve_snapshot(
    mut snapshot: LocalBaSnapshot,
    camera: &PinholeCamera,
    job: &KeyframeJob,
) -> LocalBaSnapshot {
    if job.imu_initialized {
        snapshot.run_inertial(camera, job.imu_t_bc, job.gravity_world);
    } else {
        snapshot.run_visual(camera);
    }
    snapshot
}

pub struct LocalMapping {
    backend: LocalMappingBackend,
}

enum LocalMappingBackend {
    Synchronous {
        map: Arc<Mutex<Map>>,
        camera: PinholeCamera,
        results: Mutex<Vec<LocalBaMergeResult>>,
    },
    Asynchronous {
        handle: LocalMappingHandle,
        publication_gate: Arc<Mutex<()>>,
    },
}

impl LocalMapping {
    pub fn new(mode: LocalMappingMode, map: Arc<Mutex<Map>>, camera: PinholeCamera) -> Self {
        let backend = match mode {
            LocalMappingMode::Synchronous => LocalMappingBackend::Synchronous {
                map,
                camera,
                results: Mutex::new(Vec::new()),
            },
            LocalMappingMode::Asynchronous => {
                let publication_gate = Arc::new(Mutex::new(()));
                let handle = LocalMappingHandle::spawn(map, Arc::clone(&publication_gate), camera);
                LocalMappingBackend::Asynchronous {
                    handle,
                    publication_gate,
                }
            }
        };
        Self { backend }
    }

    pub fn publication_gate(&self) -> Option<Arc<Mutex<()>>> {
        match &self.backend {
            LocalMappingBackend::Asynchronous {
                publication_gate, ..
            } => Some(Arc::clone(publication_gate)),
            LocalMappingBackend::Synchronous { .. } => None,
        }
    }

    pub fn submit(&self, job: KeyframeJob) -> bool {
        match &self.backend {
            LocalMappingBackend::Synchronous {
                map,
                camera,
                results,
            } => {
                let snapshot = map
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .local_ba_snapshot();
                let snapshot = solve_snapshot(snapshot, camera, &job);
                if let Some(result) = map
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .merge_local_ba_snapshot(snapshot)
                {
                    results
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(result);
                }
                true
            }
            LocalMappingBackend::Asynchronous { handle, .. } => handle.submit(job),
        }
    }

    pub fn drain_results(&self) -> Vec<LocalBaMergeResult> {
        match &self.backend {
            LocalMappingBackend::Synchronous { results, .. } => std::mem::take(
                &mut *results
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            LocalMappingBackend::Asynchronous { handle, .. } => handle.drain_results(),
        }
    }
}

#[derive(Default)]
struct PendingState {
    latest: Option<KeyframeJob>,
    shutdown: bool,
}

#[derive(Default)]
struct PendingJob {
    state: Mutex<PendingState>,
    ready: Condvar,
}

struct WorkerExit {
    pending: Arc<PendingJob>,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let mut state = self
            .pending
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.shutdown = true;
        state.latest = None;
        self.pending.ready.notify_all();
    }
}

struct LocalMappingHandle {
    pending: Arc<PendingJob>,
    results: mpsc::Receiver<LocalBaMergeResult>,
    join_handle: Option<JoinHandle<()>>,
}

impl LocalMappingHandle {
    fn spawn(
        map: Arc<Mutex<Map>>,
        publication_gate: Arc<Mutex<()>>,
        camera: PinholeCamera,
    ) -> Self {
        let pending = Arc::new(PendingJob::default());
        let worker_pending = Arc::clone(&pending);
        let (result_sender, results) = mpsc::channel();

        let join_handle = thread::Builder::new()
            .name("kornia-local-mapping".into())
            .spawn(move || {
                let _worker_exit = WorkerExit {
                    pending: Arc::clone(&worker_pending),
                };
                loop {
                    let job = {
                        let mut state = worker_pending
                            .state
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        while state.latest.is_none() && !state.shutdown {
                            state = worker_pending
                                .ready
                                .wait(state)
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                        }
                        if state.shutdown {
                            break;
                        }
                        state.latest.take().expect("pending job disappeared")
                    };

                    // The gate makes compound keyframe publication atomic to local
                    // mapping. The map lock is held only for the private clone.
                    let snapshot = {
                        let _publication = publication_gate
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        map.lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .local_ba_snapshot()
                    };

                    let snapshot = solve_snapshot(snapshot, &camera, &job);

                    // Merge is short and cannot interleave with keyframe publication
                    // or a world-frame transformation. Epoch mismatch rejects stale BA.
                    let should_stop = {
                        let _publication = publication_gate
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let merged = map
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .merge_local_ba_snapshot(snapshot);
                        merged.is_some_and(|result| result_sender.send(result).is_err())
                    };
                    if should_stop {
                        break;
                    }
                }
            })
            .expect("failed to spawn local-mapping worker");

        Self {
            pending,
            results,
            join_handle: Some(join_handle),
        }
    }

    /// Replaces any not-yet-started BA with the newest request.
    fn submit(&self, job: KeyframeJob) -> bool {
        let mut state = self
            .pending
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.shutdown {
            return false;
        }
        state.latest = Some(job);
        self.pending.ready.notify_one();
        true
    }

    fn drain_results(&self) -> Vec<LocalBaMergeResult> {
        self.results.try_iter().collect()
    }
}

impl Drop for LocalMappingHandle {
    fn drop(&mut self) {
        {
            let mut state = self
                .pending
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.shutdown = true;
            state.latest = None;
        }
        self.pending.ready.notify_one();

        if let Some(handle) = self.join_handle.take()
            && handle.join().is_err()
        {
            eprintln!("local-mapping worker panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kornia_3d::camera::PinholeCamera;
    use kornia_algebra::Vec3F64;

    use super::{KeyframeJob, LocalMapping, LocalMappingBackend, LocalMappingMode};
    use crate::map::Map;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn visual_job() -> KeyframeJob {
        KeyframeJob {
            imu_initialized: false,
            imu_t_bc: None,
            gravity_world: Vec3F64::ZERO,
        }
    }

    #[test]
    fn parses_local_mapping_modes_and_aliases() {
        assert_eq!(
            LocalMappingMode::from_str("sync").unwrap(),
            LocalMappingMode::Synchronous
        );
        assert_eq!(
            LocalMappingMode::from_str("synchronous").unwrap(),
            LocalMappingMode::Synchronous
        );
        assert_eq!(
            LocalMappingMode::from_str("async").unwrap(),
            LocalMappingMode::Asynchronous
        );
        assert_eq!(
            LocalMappingMode::from_str("asynchronous").unwrap(),
            LocalMappingMode::Asynchronous
        );
    }

    #[test]
    fn rejects_disabled_local_mapping_mode() {
        let error = LocalMappingMode::from_str("disabled").unwrap_err();
        assert!(error.contains("expected sync or async"));
    }

    #[test]
    fn synchronous_mode_finishes_the_job_before_submit_returns() {
        let mapping = LocalMapping::new(
            LocalMappingMode::Synchronous,
            Arc::new(Mutex::new(Map::new())),
            test_camera(),
        );

        assert!(mapping.publication_gate().is_none());
        assert!(mapping.submit(visual_job()));
        assert_eq!(mapping.drain_results().len(), 1);
    }

    #[test]
    fn asynchronous_mode_completes_the_job_on_the_worker() {
        let mapping = LocalMapping::new(
            LocalMappingMode::Asynchronous,
            Arc::new(Mutex::new(Map::new())),
            test_camera(),
        );

        assert!(mapping.publication_gate().is_some());
        assert!(mapping.submit(visual_job()));

        let LocalMappingBackend::Asynchronous { handle, .. } = &mapping.backend else {
            unreachable!("asynchronous mode selected the wrong backend");
        };
        handle
            .results
            .recv_timeout(Duration::from_secs(2))
            .expect("local-mapping worker did not publish a result");
    }
}
