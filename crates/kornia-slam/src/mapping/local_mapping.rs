use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::mapping::bundle_adjustment::{
    LocalBaParams, local_window_of, run_local_ba_with, run_local_inertial_ba_calibrated,
    run_local_inertial_ba_with,
};
use crate::mapping::keyframe_mapping::{
    DeferredKeyframe, DeferredKeyframeContext, DeferredKeyframeResult,
};
use crate::mapping::map::{
    BaOutcome, BaSnapshot, BaSolve, BaUpdate, BaUpdateError, LocalBaMergeResult, Map,
};
use crate::mapping::online_calibration::OnlineCalibrationState;

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
    pub local_ba: LocalBaParams,
    /// Wheel-odometry edges between keyframes; empty without a wheel feed.
    pub wheel_factors: Vec<kornia_factors::wheel_odometry::WheelOdomFactor>,
    /// PnP inliers the keyframe was tracked with; selects
    /// [`LargeWindow`](crate::mapping::bundle_adjustment::LargeWindow).
    pub tracked_inliers: usize,
}

impl KeyframeJob {
    /// Whether this job solves in the large window; VI-BA only, as
    /// ORB-SLAM3's `LocalInertialBA`.
    fn large_window(&self) -> bool {
        self.imu_initialized && self.local_ba.large_window_applies(self.tracked_inliers)
    }

    /// The local-BA settings this job solves with.
    fn ba_params(&self) -> LocalBaParams {
        if self.large_window() {
            self.local_ba.large()
        } else {
            self.local_ba
        }
    }
}

fn capture_local_window(map: &Map, job: &KeyframeJob) -> BaSnapshot {
    map.ba_snapshot(&local_window_of(map, job.ba_params().active_keyframes, []))
}

fn solve_snapshot(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    job: &KeyframeJob,
    calibration: &OnlineCalibrationState,
) -> BaUpdate {
    let params = job.ba_params();
    if job.imu_initialized && params.online_calibration.is_enabled() {
        run_local_inertial_ba_calibrated(
            snapshot,
            camera,
            job.imu_t_bc,
            job.gravity_world,
            &job.wheel_factors,
            &[],
            &params,
            calibration,
        )
    } else if job.imu_initialized {
        run_local_inertial_ba_with(
            snapshot,
            camera,
            job.imu_t_bc,
            job.gravity_world,
            &job.wheel_factors,
            &[],
            &params,
        )
    } else {
        run_local_ba_with(snapshot, camera, &params)
    }
}

/// Local-BA tallies since the worker started; see `RunCounters` for the meaning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LocalMappingCounts {
    pub ba_failed: u64,
    pub vi_ba_failed: u64,
    pub prior_fallbacks: u64,
    pub ba_updates_rejected: u64,
    pub ba_solves: u64,
    pub ba_solve_ns: u64,
    pub large_window_solves: u64,
    pub vi_ba_base_solves: u64,
    pub vi_ba_base_iterations: u64,
    pub vi_ba_iterations_max: u64,
    pub ba_stale_keyframes: u64,
    pub ba_propagated_keyframes: u64,
    pub ba_propagations_stopped_at_bridge: u64,
}

// Atomics because the asynchronous worker writes them while tracking reads.
#[derive(Default)]
struct LocalMappingCounters {
    ba_failed: AtomicU64,
    vi_ba_failed: AtomicU64,
    prior_fallbacks: AtomicU64,
    ba_updates_rejected: AtomicU64,
    ba_solves: AtomicU64,
    ba_solve_ns: AtomicU64,
    large_window_solves: AtomicU64,
    vi_ba_base_solves: AtomicU64,
    vi_ba_base_iterations: AtomicU64,
    vi_ba_iterations_max: AtomicU64,
    ba_stale_keyframes: AtomicU64,
    ba_propagated_keyframes: AtomicU64,
    ba_propagations_stopped_at_bridge: AtomicU64,
}

impl LocalMappingCounters {
    /// `iterations` is the solver's count when it ran (VI-BA only).
    fn record_solve(
        &self,
        job: &KeyframeJob,
        outcome: BaOutcome,
        iterations: Option<usize>,
        started: Instant,
    ) {
        let add = |counter: &AtomicU64, n: u64| counter.fetch_add(n, Ordering::Relaxed);
        if outcome.prior_fallback {
            add(&self.prior_fallbacks, 1);
        }
        if outcome.solve == BaSolve::NotRun {
            return;
        }
        add(&self.ba_solves, 1);
        if job.large_window() {
            add(&self.large_window_solves, 1);
        }
        if job.imu_initialized
            && let Some(iterations) = iterations
        {
            let iterations = u64::try_from(iterations).unwrap_or(u64::MAX);
            self.vi_ba_iterations_max
                .fetch_max(iterations, Ordering::Relaxed);
            if !job.large_window() {
                add(&self.vi_ba_base_solves, 1);
                add(&self.vi_ba_base_iterations, iterations);
            }
        }
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        add(&self.ba_solve_ns, elapsed_ns);
        if outcome.solve == BaSolve::Failed {
            add(
                if job.imu_initialized {
                    &self.vi_ba_failed
                } else {
                    &self.ba_failed
                },
                1,
            );
        }
    }

    fn record_merge(&self, merged: &Result<LocalBaMergeResult, BaUpdateError>) {
        let add = |counter: &AtomicU64, n: usize| {
            counter.fetch_add(u64::try_from(n).unwrap_or(u64::MAX), Ordering::Relaxed)
        };
        match merged {
            Ok(result) => {
                add(&self.ba_stale_keyframes, result.newer_keyframes);
                add(&self.ba_propagated_keyframes, result.propagated_keyframes);
                add(
                    &self.ba_propagations_stopped_at_bridge,
                    usize::from(result.stopped_at_bridge),
                );
            }
            Err(_) => {
                add(&self.ba_updates_rejected, 1);
            }
        }
    }

    fn counts(&self) -> LocalMappingCounts {
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        LocalMappingCounts {
            ba_failed: read(&self.ba_failed),
            vi_ba_failed: read(&self.vi_ba_failed),
            prior_fallbacks: read(&self.prior_fallbacks),
            ba_updates_rejected: read(&self.ba_updates_rejected),
            ba_solves: read(&self.ba_solves),
            ba_solve_ns: read(&self.ba_solve_ns),
            large_window_solves: read(&self.large_window_solves),
            vi_ba_base_solves: read(&self.vi_ba_base_solves),
            vi_ba_base_iterations: read(&self.vi_ba_base_iterations),
            vi_ba_iterations_max: read(&self.vi_ba_iterations_max),
            ba_stale_keyframes: read(&self.ba_stale_keyframes),
            ba_propagated_keyframes: read(&self.ba_propagated_keyframes),
            ba_propagations_stopped_at_bridge: read(&self.ba_propagations_stopped_at_bridge),
        }
    }
}

/// Solves `job`'s window and records what the solve did.
fn solve_counted(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    job: &KeyframeJob,
    counters: &LocalMappingCounters,
    calibration: &OnlineCalibrationState,
) -> BaUpdate {
    let started = Instant::now();
    let update = solve_snapshot(snapshot, camera, job, calibration);
    let iterations = update.solve_stats().map(|stats| stats.iterations);
    counters.record_solve(job, update.outcome, iterations, started);
    update
}

pub struct LocalMapping {
    backend: LocalMappingBackend,
    counters: Arc<LocalMappingCounters>,
}

enum LocalMappingBackend {
    Synchronous {
        map: Arc<Mutex<Map>>,
        camera: PinholeCamera,
        results: Mutex<Vec<LocalBaMergeResult>>,
        /// The online calibration the next solve starts from; boxed, as it
        /// dwarfs the asynchronous variant.
        calibration: Box<Mutex<OnlineCalibrationState>>,
        /// Set by [`LocalMapping::with_deferred`].
        deferred: Option<DeferredKeyframeContext>,
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

/// [`merge_and_cull`], keeping the calibration state `update` carries only if
/// the merge is accepted, as the boundary prior is.
fn merge_keeping_calibration(
    map: &mut Map,
    mut update: BaUpdate,
    calibration: &mut OnlineCalibrationState,
) -> Result<LocalBaMergeResult, BaUpdateError> {
    let next = update.online_calibration.take();
    let merged = merge_and_cull(map, update);
    if merged.is_ok()
        && let Some(next) = next
    {
        *calibration = next;
    }
    merged
}

impl LocalMapping {
    pub fn new(mode: LocalMappingMode, map: Arc<Mutex<Map>>, camera: PinholeCamera) -> Self {
        let counters = Arc::new(LocalMappingCounters::default());
        let backend = match mode {
            LocalMappingMode::Synchronous => LocalMappingBackend::Synchronous {
                map,
                camera,
                results: Mutex::new(Vec::new()),
                calibration: Box::new(Mutex::new(OnlineCalibrationState::default())),
                deferred: None,
            },
            LocalMappingMode::Asynchronous => {
                let publication_gate = Arc::new(Mutex::new(()));
                let handle = LocalMappingHandle::spawn(
                    map,
                    Arc::clone(&publication_gate),
                    camera,
                    Arc::clone(&counters),
                );
                LocalMappingBackend::Asynchronous {
                    handle,
                    publication_gate,
                }
            }
        };
        Self { backend, counters }
    }

    /// Enables the deferred keyframe path; the build calls it only with
    /// `keyframe_orb: Mapping`.
    #[allow(dead_code)] // skeleton: the build calls it once lane T builds the path
    pub(crate) fn with_deferred(mut self, context: DeferredKeyframeContext) -> Self {
        match &mut self.backend {
            LocalMappingBackend::Synchronous { deferred, .. } => *deferred = Some(context),
            LocalMappingBackend::Asynchronous { handle, .. } => {
                // The worker is already running: the context goes into the
                // shared state before any job is submitted.
                handle
                    .pending
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .deferred_context = Some(Arc::new(context));
            }
        }
        self
    }

    /// Queues `job` (FIFO, never replaced, unlike `submit`'s latest-wins
    /// job); `false` without [`Self::with_deferred`].
    #[allow(dead_code)] // skeleton: lane T calls it
    pub(crate) fn submit_deferred(&self, _job: DeferredKeyframe) -> bool {
        if !self.has_deferred_context() {
            return false;
        }
        todo!("lane T: queue a deferred keyframe")
    }

    /// The deferred keyframes finished since the last call, in queue order.
    #[allow(dead_code)] // skeleton: lane T calls it
    pub(crate) fn drain_deferred(&self) -> Vec<DeferredKeyframeResult> {
        if !self.has_deferred_context() {
            return Vec::new();
        }
        todo!("lane T: deferred keyframe results")
    }

    fn has_deferred_context(&self) -> bool {
        match &self.backend {
            LocalMappingBackend::Synchronous { deferred, .. } => deferred.is_some(),
            LocalMappingBackend::Asynchronous { handle, .. } => handle
                .pending
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .deferred_context
                .is_some(),
        }
    }

    /// Local-BA tallies so far. In asynchronous mode a job still on the
    /// worker is not counted yet.
    pub(crate) fn counts(&self) -> LocalMappingCounts {
        self.counters.counts()
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
                calibration,
                ..
            } => {
                let snapshot = capture_local_window(
                    &map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                    &job,
                );
                let mut calibration = calibration
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let update = solve_counted(snapshot, camera, &job, &self.counters, &calibration);
                // Cull under the same lock as the merge, so a completed result
                // is never observable before cleanup. A rejected snapshot culls
                // nothing.
                let merged = merge_keeping_calibration(
                    &mut map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                    update,
                    &mut calibration,
                );
                self.counters.record_merge(&merged);
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
    /// Keyframes whose ORB runs here, in order; never replaced.
    deferred: VecDeque<DeferredKeyframe>,
    deferred_context: Option<Arc<DeferredKeyframeContext>>,
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
        state.deferred.clear();
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
        counters: Arc<LocalMappingCounters>,
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
                // The online calibration the next solve starts from.
                let mut calibration = OnlineCalibrationState::default();
                loop {
                    let job = {
                        let mut state = worker_pending
                            .state
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        while state.latest.is_none() && state.deferred.is_empty() && !state.shutdown
                        {
                            state = worker_pending
                                .ready
                                .wait(state)
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                        }
                        if state.shutdown {
                            break;
                        }
                        // Queued only with a context (`submit_deferred`), and
                        // taken before `latest`.
                        if let Some(_deferred) = state.deferred.pop_front() {
                            todo!("lane T: deferred keyframe ORB, overlap, search and insert");
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
                            &job,
                        )
                    };

                    let update = solve_counted(snapshot, &camera, &job, &counters, &calibration);

                    // Merge is short and cannot interleave with keyframe publication
                    // or a world-frame transformation. Epoch mismatch rejects stale BA.
                    let should_stop = {
                        let _publication = publication_gate
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let merged = merge_keeping_calibration(
                            &mut map.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
                            update,
                            &mut calibration,
                        );
                        counters.record_merge(&merged);
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
            state.deferred.clear();
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
    use kornia_algebra::Vec3F64;

    use super::{
        BaUpdateError, KeyframeJob, LocalBaParams, LocalMapping, LocalMappingBackend,
        LocalMappingMode,
    };
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
            local_ba: LocalBaParams::default(),
            wheel_factors: Vec::new(),
            tracked_inliers: 0,
        }
    }

    /// Card V (E-1): a 0-inlier VI job never takes the large window, so it
    /// solves with the base settings, and the counters book it as a base solve.
    #[test]
    fn a_zero_inlier_vi_job_solves_with_the_base_budget_and_is_counted() {
        use super::{BaOutcome, BaSolve, LocalMappingCounters};
        use crate::mapping::bundle_adjustment::LargeWindow;
        use std::time::Instant;

        let local_ba = LocalBaParams {
            active_keyframes: 10,
            max_iterations: 7,
            initial_lambda: 1e-4,
            large_window: LargeWindow::Enabled {
                min_tracked_inliers: 0,
                active_keyframes: 10,
                max_iterations: 4,
                initial_lambda: 1e-2,
            },
            ..LocalBaParams::default()
        };
        let job = |tracked_inliers| KeyframeJob {
            imu_initialized: true,
            local_ba,
            tracked_inliers,
            ..visual_job()
        };
        let base = job(0).ba_params();
        assert_eq!(
            (
                base.active_keyframes,
                base.max_iterations,
                base.initial_lambda
            ),
            (10, 7, 1e-4)
        );
        assert_eq!(job(1).ba_params().max_iterations, 4, "the large window");

        let counters = LocalMappingCounters::default();
        let solved = BaOutcome {
            solve: BaSolve::Solved,
            prior_fallback: false,
        };
        counters.record_solve(&job(0), solved, Some(5), Instant::now());
        counters.record_solve(&job(0), solved, Some(7), Instant::now());
        counters.record_solve(&job(1), solved, Some(4), Instant::now());
        counters.record_solve(&visual_job(), solved, None, Instant::now());
        let counts = counters.counts();
        assert_eq!(counts.vi_ba_base_solves, 2);
        assert_eq!(counts.vi_ba_base_iterations, 12);
        assert_eq!(counts.vi_ba_iterations_max, 7);
        assert_eq!(counts.large_window_solves, 1);
        assert_eq!(counts.ba_solves, 4);
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
            rig: Default::default(),
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
