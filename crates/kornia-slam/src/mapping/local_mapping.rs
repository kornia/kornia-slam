use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::map::{LocalBaMergeResult, LocalBaSnapshot, Map};

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
                // Cull under the same lock as the merge, so a completed result
                // is never observable before cleanup. A rejected snapshot culls
                // nothing.
                let merged = {
                    let mut map = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let merged = map.merge_local_ba_snapshot(snapshot);
                    if merged.is_some() {
                        crate::mapping::culling::cull_landmarks(&mut map);
                    }
                    merged
                };
                if let Some(result) = merged {
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
                        let merged = {
                            let mut map =
                                map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                            let merged = map.merge_local_ba_snapshot(snapshot);
                            if merged.is_some() {
                                crate::mapping::culling::cull_landmarks(&mut map);
                            }
                            merged
                        };
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

    /// A map holding one landmark that fails the culling policy, with a live
    /// association, so cleanup is observable.
    fn map_with_a_cullable_landmark() -> (Arc<Mutex<Map>>, usize) {
        use crate::frame::Frame;
        use crate::map::{Keyframe, LandmarkSeed, ObservationKey};
        use kornia_3d::pose::Pose3d;
        use kornia_image::ImageSize;
        use kornia_imgproc::features::OrbFeatures;

        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: vec![[10.0, 10.0]],
                orientations: vec![0.0],
                descriptors: vec![[0u8; 32]],
                octaves: vec![0],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }))
        .unwrap();
        let doomed = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        // Seen often, matched rarely: below the found-ratio threshold.
        map.set_tracking_stats_for_test(doomed, 10, 1);
        (Arc::new(Mutex::new(map)), doomed)
    }

    /// Culling runs under the same lock as the merge, so by the time a result
    /// is observable the retirement and its two-sided cleanup have happened.
    #[test]
    fn synchronous_completion_implies_cleanup_is_already_done() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let mapping = LocalMapping::new(
            LocalMappingMode::Synchronous,
            Arc::clone(&map),
            test_camera(),
        );

        assert!(mapping.submit(visual_job()));
        assert_eq!(mapping.drain_results().len(), 1);

        let map = map.lock().unwrap();
        assert!(map.map_points()[doomed].culled, "retired before completion");
        assert!(map.map_points()[doomed].observations().is_empty());
        assert_eq!(
            map.get_keyframe(0).unwrap().map_point(0),
            None,
            "the keyframe side was cleaned up too"
        );
    }

    #[test]
    fn asynchronous_completion_implies_cleanup_is_already_done() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let mapping = LocalMapping::new(
            LocalMappingMode::Asynchronous,
            Arc::clone(&map),
            test_camera(),
        );

        assert!(mapping.submit(visual_job()));
        let LocalMappingBackend::Asynchronous { handle, .. } = &mapping.backend else {
            unreachable!("asynchronous mode selected the wrong backend");
        };
        handle
            .results
            .recv_timeout(Duration::from_secs(2))
            .expect("local-mapping worker did not publish a result");

        let map = map.lock().unwrap();
        assert!(map.map_points()[doomed].culled);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
    }

    /// An epoch-rejected snapshot publishes nothing, so it must not cull
    /// either — the live map is left exactly as it was.
    #[test]
    fn a_rejected_snapshot_leaves_a_cullable_landmark_untouched() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let snapshot = map.lock().unwrap().local_ba_snapshot();
        // Advance the world frame, so the snapshot belongs to an older epoch.
        map.lock().unwrap().scale_world(2.0);

        assert!(
            map.lock()
                .unwrap()
                .merge_local_ba_snapshot(snapshot)
                .is_none(),
            "an older-epoch snapshot is refused"
        );

        let map = map.lock().unwrap();
        assert!(!map.map_points()[doomed].culled, "no cull after a refusal");
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(doomed));
    }
}
