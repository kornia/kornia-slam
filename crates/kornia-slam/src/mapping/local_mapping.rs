use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::mapping::keyframes::{KeyframeCullingStrategy, cull_redundant_keyframes};
use crate::mapping::map::{BaSnapshot, BaUpdate, BaUpdateError, LocalBaMergeResult, Map};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LocalMappingMode {
    /// On the tracking thread, before the frame result is returned.
    Synchronous,
    /// On a background worker.
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

fn capture_local_window(map: &Map) -> BaSnapshot {
    map.ba_snapshot(&super::bundle_adjustment::local_window(map))
}

fn solve_snapshot(snapshot: BaSnapshot, camera: &PinholeCamera, job: &KeyframeJob) -> BaUpdate {
    if job.imu_initialized {
        super::bundle_adjustment::run_local_inertial_ba(
            snapshot,
            camera,
            job.imu_t_bc,
            job.gravity_world,
            &[],
            &[],
        )
    } else {
        super::bundle_adjustment::run_local_ba(snapshot, camera)
    }
}

pub struct LocalMapping {
    backend: LocalMappingBackend,
}

/// An accepted local-mapping writeback and the keyframes retired as part of
/// that same map transaction. Kept crate-private so public result reporting
/// remains compatible while `SlamSystem` can retire its ID-indexed state.
pub(crate) struct LocalMappingCompletion {
    pub(crate) result: LocalBaMergeResult,
    pub(crate) removed_keyframes: Vec<usize>,
}

enum LocalMappingBackend {
    Synchronous {
        map: Arc<Mutex<Map>>,
        camera: PinholeCamera,
        keyframe_culling: KeyframeCullingStrategy,
        results: Mutex<Vec<LocalMappingCompletion>>,
    },
    Asynchronous {
        handle: LocalMappingHandle,
        publication_gate: Arc<Mutex<()>>,
    },
}

/// The coordinator's writeback decision, shared by both backends.
///
/// Culling runs under the caller's map lock, immediately after an accepted
/// merge and before the result is published, so a completed result is never
/// observable ahead of its cleanup. A refused snapshot — one whose world epoch
/// no longer matches — publishes nothing and must therefore cull nothing.
fn merge_and_cull(map: &mut Map, update: BaUpdate) -> Result<LocalBaMergeResult, BaUpdateError> {
    let merged = map.apply_ba_update(update)?;
    crate::mapping::culling::cull_landmarks(map);
    Ok(merged)
}

fn merge_and_cull_keyframes(
    map: &mut Map,
    update: BaUpdate,
    job: &KeyframeJob,
    strategy: &KeyframeCullingStrategy,
) -> Result<LocalMappingCompletion, BaUpdateError> {
    let result = merge_and_cull(map, update)?;
    // Before inertial initialization, keep the conservative historical
    // behaviour: the redundancy culler must not merge long IMU intervals.
    let removed_keyframes = match strategy {
        KeyframeCullingStrategy::Disabled => Vec::new(),
        KeyframeCullingStrategy::Redundancy(_)
            if job.imu_t_bc.is_some() && !job.imu_initialized =>
        {
            Vec::new()
        }
        KeyframeCullingStrategy::Redundancy(config) => {
            cull_redundant_keyframes(map, config, None, None).removed
        }
    };
    Ok(LocalMappingCompletion {
        result,
        removed_keyframes,
    })
}

impl LocalMapping {
    pub fn new(mode: LocalMappingMode, map: Arc<Mutex<Map>>, camera: PinholeCamera) -> Self {
        Self::new_with_keyframe_culling(mode, map, camera, KeyframeCullingStrategy::Disabled)
    }

    pub(crate) fn new_with_keyframe_culling(
        mode: LocalMappingMode,
        map: Arc<Mutex<Map>>,
        camera: PinholeCamera,
        keyframe_culling: KeyframeCullingStrategy,
    ) -> Self {
        let backend = match mode {
            LocalMappingMode::Synchronous => LocalMappingBackend::Synchronous {
                map,
                camera,
                keyframe_culling,
                results: Mutex::new(Vec::new()),
            },
            LocalMappingMode::Asynchronous => {
                let publication_gate = Arc::new(Mutex::new(()));
                let handle = LocalMappingHandle::spawn(
                    map,
                    Arc::clone(&publication_gate),
                    camera,
                    keyframe_culling,
                );
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
                keyframe_culling,
                results,
            } => {
                let snapshot = capture_local_window(
                    &map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                );
                let update = solve_snapshot(snapshot, camera, &job);
                // Cull under the same lock as the merge, so a completed result
                // is never observable before cleanup. A rejected snapshot culls
                // nothing.
                let merged = merge_and_cull_keyframes(
                    &mut map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                    update,
                    &job,
                    keyframe_culling,
                );
                if let Ok(result) = merged {
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
        self.drain_completions()
            .into_iter()
            .map(|completion| completion.result)
            .collect()
    }

    pub(crate) fn drain_completions(&self) -> Vec<LocalMappingCompletion> {
        match &self.backend {
            LocalMappingBackend::Synchronous { results, .. } => std::mem::take(
                &mut *results
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            LocalMappingBackend::Asynchronous { handle, .. } => handle.drain_completions(),
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
    results: mpsc::Receiver<LocalMappingCompletion>,
    join_handle: Option<JoinHandle<()>>,
}

impl LocalMappingHandle {
    fn spawn(
        map: Arc<Mutex<Map>>,
        publication_gate: Arc<Mutex<()>>,
        camera: PinholeCamera,
        keyframe_culling: KeyframeCullingStrategy,
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
                    // mapping. The map lock is held only to copy the local window.
                    let snapshot = {
                        let _publication = publication_gate
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        capture_local_window(
                            &map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                        )
                    };

                    let update = solve_snapshot(snapshot, &camera, &job);

                    // Merge is short and cannot interleave with keyframe publication
                    // or a world-frame transformation. Epoch mismatch rejects stale BA.
                    let should_stop = {
                        let _publication = publication_gate
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let merged = merge_and_cull_keyframes(
                            &mut map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                            update,
                            &job,
                            &keyframe_culling,
                        );
                        merged.is_ok_and(|result| result_sender.send(result).is_err())
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

    fn drain_completions(&self) -> Vec<LocalMappingCompletion> {
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
            // A fault, not solver chatter: always reported, since a host's
            // panic hook may not write the panic itself to stderr.
            eprintln!("kornia-slam: local-mapping worker panicked; mapping had stopped");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;

    use super::{
        BaUpdateError, KeyframeJob, LocalMapping, LocalMappingBackend, LocalMappingMode,
        merge_and_cull_keyframes,
    };
    use crate::mapping::keyframes::KeyframeCullingStrategy;
    use crate::mapping::keyframes::culling::keyframe_tests::{culling_fixture, enabled};
    use crate::mapping::map::{InertialAlignment, KeyframeVelocity, Map};
    use kornia_algebra::SO3F64;
    use kornia_sensors::imu::ImuBias;

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

    #[test]
    fn redundancy_culling_is_reported_in_the_completion() {
        let map = Arc::new(Mutex::new(culling_fixture()));
        let mapping = LocalMapping::new_with_keyframe_culling(
            LocalMappingMode::Synchronous,
            Arc::clone(&map),
            test_camera(),
            KeyframeCullingStrategy::Redundancy(enabled()),
        );

        assert!(mapping.submit(visual_job()));
        let completions = mapping.drain_completions();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].removed_keyframes, vec![7, 11, 19]);
        assert!(map.lock().unwrap().get_keyframe(7).is_none());
    }

    #[test]
    fn uninitialized_imu_suppresses_redundancy_culling() {
        let mut map = culling_fixture();
        let update = map.full_ba_snapshot().into_update();
        let job = KeyframeJob {
            imu_initialized: false,
            imu_t_bc: Some(Pose3d::IDENTITY),
            gravity_world: Vec3F64::ZERO,
        };

        let completion = merge_and_cull_keyframes(
            &mut map,
            update,
            &job,
            &KeyframeCullingStrategy::Redundancy(enabled()),
        )
        .expect("current snapshot is accepted");

        assert!(completion.removed_keyframes.is_empty());
        assert!(map.get_keyframe(7).is_some());
    }

    #[test]
    fn rejected_ba_merge_does_not_cull_or_publish_keyframes() {
        let mut map = culling_fixture();
        let update = map.full_ba_snapshot().into_update();
        map.apply_inertial_alignment(InertialAlignment {
            scale: 2.0,
            rotation: SO3F64::IDENTITY,
            keyframe_velocities: map
                .keyframes()
                .iter()
                .map(|kf| KeyframeVelocity {
                    keyframe_idx: kf.frame.idx,
                    velocity_world: Vec3F64::ZERO,
                })
                .collect(),
            bias: ImuBias::default(),
        })
        .expect("valid alignment advances the world epoch");

        assert!(matches!(
            merge_and_cull_keyframes(
                &mut map,
                update,
                &visual_job(),
                &KeyframeCullingStrategy::Redundancy(enabled()),
            ),
            Err(BaUpdateError::ObsoleteWorldFrame)
        ));
        assert!(map.get_keyframe(7).is_some());
    }

    #[test]
    fn asynchronous_culling_waits_for_the_publication_gate() {
        let map = Arc::new(Mutex::new(culling_fixture()));
        let mapping = LocalMapping::new_with_keyframe_culling(
            LocalMappingMode::Asynchronous,
            Arc::clone(&map),
            test_camera(),
            KeyframeCullingStrategy::Redundancy(enabled()),
        );
        let gate = mapping
            .publication_gate()
            .expect("asynchronous mapping has a publication gate");
        let guard = gate.lock().unwrap();
        assert!(mapping.submit(visual_job()));
        let LocalMappingBackend::Asynchronous { handle, .. } = &mapping.backend else {
            unreachable!("asynchronous mode selected the wrong backend");
        };
        assert!(matches!(
            handle.results.recv_timeout(Duration::from_millis(100)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(map.lock().unwrap().get_keyframe(7).is_some());
        drop(guard);

        let completion = handle
            .results
            .recv_timeout(Duration::from_secs(2))
            .expect("local-mapping worker did not publish a result");
        assert_eq!(completion.removed_keyframes, vec![7, 11, 19]);
        assert!(map.lock().unwrap().get_keyframe(7).is_none());
    }

    /// A map holding one landmark that fails the culling policy, with a live
    /// association, so cleanup is observable.
    fn map_with_a_cullable_landmark() -> (Arc<Mutex<Map>>, usize) {
        use crate::frame::Frame;
        use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey};
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
        // Retirement is logical: the landmark keeps its records.
        assert!(map.map_points()[doomed].is_observed_by(0));
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
    /// Through the coordinator's own decision, not `apply_ba_update`
    /// directly: a test that bypassed `merge_and_cull` would pass even if the
    /// worker culled after every refused merge.
    #[test]
    fn the_coordinator_does_not_cull_after_a_refused_merge() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let snapshot = map.lock().unwrap().full_ba_snapshot().into_update();
        // Advance the world frame through a valid public correction, so the
        // snapshot belongs to an older epoch. Raw scale/rotation are private
        // precisely because they leave geometry stale.
        map.lock()
            .unwrap()
            .apply_inertial_alignment(InertialAlignment {
                scale: 2.0,
                rotation: SO3F64::IDENTITY,
                keyframe_velocities: vec![KeyframeVelocity {
                    keyframe_idx: 0,
                    velocity_world: Vec3F64::ZERO,
                }],
                bias: ImuBias::default(),
            })
            .expect("a valid alignment advances the world epoch");

        let merged = super::merge_and_cull(&mut map.lock().unwrap(), snapshot);

        assert!(
            matches!(merged, Err(BaUpdateError::ObsoleteWorldFrame)),
            "an older-epoch snapshot is refused"
        );
        let map = map.lock().unwrap();
        assert!(!map.map_points()[doomed].culled, "no cull after a refusal");
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(doomed));
    }

    /// A malformed result is refused at the same seam, and a refusal must not
    /// cull: the coordinator publishes nothing, so it cleans up nothing.
    #[test]
    fn the_coordinator_does_not_cull_after_an_invalid_numerical_update() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let mut update = map.lock().unwrap().full_ba_snapshot().into_update();
        update.map_points[doomed].x = f64::NAN;

        let merged = super::merge_and_cull(&mut map.lock().unwrap(), update);

        assert_eq!(
            merged.unwrap_err(),
            BaUpdateError::NonFiniteLandmark {
                landmark_idx: doomed
            },
        );
        let map = map.lock().unwrap();
        assert!(
            !map.map_points()[doomed].culled,
            "an invalid update must not cull"
        );
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(doomed));
    }

    /// The same seam on the accepting side: an accepted merge culls, even when
    /// it published no numerical change.
    #[test]
    fn the_coordinator_culls_after_an_accepted_merge() {
        let (map, doomed) = map_with_a_cullable_landmark();
        let snapshot = map.lock().unwrap().full_ba_snapshot().into_update();

        let merged = super::merge_and_cull(&mut map.lock().unwrap(), snapshot);

        assert!(merged.is_ok(), "a current-epoch snapshot is accepted");
        assert_eq!(
            merged.expect("accepted").map_points_updated,
            0,
            "nothing moved, so the merge published no change"
        );
        let map = map.lock().unwrap();
        assert!(
            map.map_points()[doomed].culled,
            "culling still runs after an accepted no-op merge"
        );
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
    }
}
