//! Loop closing for one keyframe: place recognition, verification, episode
//! consistency, pose-graph optimization and map writeback.
//!
//! [`LoopCloser`] owns the bag-of-words database, the acceptance history and
//! the pose graph, and decides what happens to the map. It reports what the runtime must apply
//! to its own tracking state through [`LoopClosingOutcome`] rather than
//! reaching into it.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::loop_closure::fusion_plan::{FusionOp, FusionSnapshot, apply_fusion};
use crate::loop_closure::place_recognition::{
    Candidate, KeyFrameDatabase, Vocabulary, compute_bow,
};
use crate::loop_closure::pose_graph::{max_gravity_alignment_error, unusable_reason};
use crate::loop_closure::worker::{LoopSolveJob, LoopSolveOutcome, LoopSolver, WorkerStopped};
use crate::loop_closure::{
    Correction, DriftModel, GateDecision, GraphManifold, InertialPgoContext, LoopCorrection,
    LoopEpisodeConfig, LoopEpisodeDecision, LoopEpisodeTracker, LoopFusionConfig, LoopGate,
    LoopMeasurement, LoopOdometryGate, LoopProposal, LoopTemporalConsistency,
    LoopVerificationConfig, OdometryConsistencyGate, PgoConfig, PgoError, PgoResult, PoseGraph,
    PoseGraphConfig, PoseGraphError, RejectAll, TemporalConsistencyGate, VerifiedLoopEdge,
    fuse_verified_loop, verify_loop_candidate,
};
use crate::mapping::Map;
use crate::mapping::map::KeyframePoseChange;
use crate::pose_conversion::apply_reference_pose_correction;

/// Verification, episode consistency, fusion and pose-graph settings of loop correction.
#[derive(Debug, Clone, Default)]
pub struct LoopClosingConfig {
    pub episode: LoopEpisodeConfig,
    pub fusion: LoopFusionConfig,
    pub verification: LoopVerificationConfig,
    pub odometry_gate: LoopOdometryGate,
    pub temporal_consistency: LoopTemporalConsistency,
    pub optimizer: PgoConfig,
    /// Set from [`LoopClosingMode::Enabled`] when the system is built.
    ///
    /// [`LoopClosingMode::Enabled`]: crate::system::LoopClosingMode::Enabled
    pub graph: LoopGraphSettings,
}

/// How accepted loops are solved. The default is [`optimize_pose_graph`]'s computation: `w * I`
/// on a fresh snapshot of the map at every solve.
///
/// [`optimize_pose_graph`]: crate::loop_closure::optimize_pose_graph
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct LoopGraphSettings {
    pub information: LoopGraphInformation,
    /// Keep the graph across solves, so a keyframe the map culls or evicts stays a node and the
    /// loops that touch it keep constraining the map. `false` builds the graph from the map's
    /// keyframes at each solve and refuses a loop whose keyframe left the map.
    pub persistent: bool,
}

/// The information odometry and loop edges carry in the pose graph.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoopGraphInformation {
    /// `w * I` from [`PgoConfig`]: the scalar weights of [`optimize_pose_graph`].
    ///
    /// [`optimize_pose_graph`]: crate::loop_closure::optimize_pose_graph
    #[default]
    ScalarWeights,
    /// Odometry edges a random walk in path; loop edges at the verified pose's own noise
    /// (`loop_sigma_m`, `loop_sigma_rad`) under a Cauchy kernel at `loop_cauchy_scale` sigmas.
    /// Omitted fields take [`LoopGraphInformation::drift`]'s values.
    Drift {
        #[cfg_attr(feature = "serde", serde(default))]
        odometry: OdometryRandomWalk,
        #[cfg_attr(feature = "serde", serde(default = "default_loop_sigma_m"))]
        loop_sigma_m: f64,
        #[cfg_attr(feature = "serde", serde(default = "default_loop_sigma_rad"))]
        loop_sigma_rad: f64,
        #[cfg_attr(feature = "serde", serde(default = "default_loop_cauchy_scale"))]
        loop_cauchy_scale: f64,
    },
}

#[cfg(feature = "serde")]
fn default_loop_sigma_m() -> f64 {
    DriftModel::default().translation_floor_m
}

#[cfg(feature = "serde")]
fn default_loop_sigma_rad() -> f64 {
    DriftModel::default().yaw_floor_rad
}

#[cfg(feature = "serde")]
fn default_loop_cauchy_scale() -> f64 {
    3.0
}

impl LoopGraphInformation {
    /// `Drift` with the default random walk, the [`DriftModel`] floors as the loop noise, and
    /// Cauchy at 3 sigma.
    pub fn drift() -> Self {
        let floors = DriftModel::default();
        Self::Drift {
            odometry: OdometryRandomWalk::default(),
            loop_sigma_m: floors.translation_floor_m,
            loop_sigma_rad: floors.yaw_floor_rad,
            loop_cauchy_scale: 3.0,
        }
    }
}

/// Odometry noise between keyframes as a random walk in path: `sigma = sqrt(rate * path)`, so a
/// chain of odometry edges has the same sigma whatever the keyframe spacing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OdometryRandomWalk {
    pub translation_m2_per_m: f64,
    /// Applied to every rotation axis; only yaw moves on a gravity manifold.
    pub rotation_rad2_per_m: f64,
    /// Conditioning floors for edges that barely move; small, because a per-edge floor
    /// compounds along the chain like density-dependent noise.
    pub floor_m: f64,
    pub floor_rad: f64,
}

impl Default for OdometryRandomWalk {
    /// The worst EuRoC Machine Hall rates with loop closing off: translation from MH_04's
    /// relative-displacement error over 5-40 m, yaw from MH_02's revisit drift.
    fn default() -> Self {
        Self {
            translation_m2_per_m: 0.0145,
            rotation_rad2_per_m: 2.7e-6,
            floor_m: 0.001,
            floor_rad: 1e-4,
        }
    }
}

/// Concise externally visible result of a loop-closure attempt.
#[derive(Debug, Clone)]
pub enum LoopClosureEvent {
    Accepted {
        edge: VerifiedLoopEdge,
        applied: bool,
    },
    PgoFailed {
        query_kf_idx: usize,
        candidate_kf_idx: usize,
        reason: String,
    },
}

/// Tracking state the loop closer reads but never owns.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoopClosingContext {
    pub pose_world_to_cam: Pose3d,
    pub velocity_world: Vec3F64,
    pub current_keyframe_idx: Option<usize>,
    pub imu_initialized: bool,
    pub gravity_world: Vec3F64,
}

/// What the runtime must apply after a loop-closure attempt.
#[derive(Debug, Default)]
pub(crate) struct LoopClosingOutcome {
    /// The best place-recognition match, for the runtime's debug log.
    pub debug_message: Option<String>,
    /// One `[loop-gate]` or `[loop-temporal]` line per decision.
    pub gate_messages: Vec<String>,
    pub events: Vec<LoopClosureEvent>,
    /// Corrected `(pose_world_to_cam, velocity_world)`, set only when the map
    /// was actually corrected.
    pub tracking_correction: Option<(Pose3d, Vec3F64)>,
    pub pgo_applied: bool,
    /// The accumulated map correction, set only when the map was corrected.
    pub correction: Option<Correction>,
    pub timing: LoopTiming,
    /// A loop was accepted and waits for a background solve.
    pub queued: bool,
    /// A background solve was written back (or discarded) this long after it was submitted.
    pub written_back: Option<Duration>,
}

/// Where one loop-closure attempt spent its time, for the runtime's `[loop-time]` log.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LoopTiming {
    /// BoW and database query.
    pub detect: Duration,
    /// Verification, gates and episode logic: everything not counted below.
    pub verify: Duration,
    /// Graph sync, loop edges and the solve.
    pub solve: Duration,
    /// Map writeback, `geometry` included.
    pub writeback: Duration,
    pub geometry: Duration,
    pub fuse: Duration,
    /// Copying the loop neighbourhoods at submit, on the calling thread.
    pub fuse_capture: Duration,
    /// The background fusion search on the solved snapshot, its operations and the stale ones.
    pub fuse_plan: Duration,
    pub fusion_ops: usize,
    pub fusion_stale: usize,
    pub graph_edges: usize,
    pub iterations: usize,
    pub landmarks_refreshed: usize,
}

/// Place recognition and loop closing for every keyframe.
///
/// Keyframes are indexed even while correction is held back until inertial
/// initialization, so the database is complete by the time a closure is
/// allowed.
pub(crate) struct LoopCloser {
    vocabulary: Vocabulary,
    kf_database: KeyFrameDatabase,
    acceptance: LoopAcceptance,
}

impl LoopCloser {
    /// `require_imu_initialized` holds correction back until inertial
    /// initialization.
    pub(crate) fn new(
        vocabulary: Vocabulary,
        config: LoopClosingConfig,
        require_imu_initialized: bool,
    ) -> Self {
        Self {
            vocabulary,
            kf_database: KeyFrameDatabase::new(),
            acceptance: LoopAcceptance::new(config, require_imu_initialized),
        }
    }

    /// Keyframes of `map` that look like the given descriptors, best first,
    /// for relocalization. The database outlives map resets, so entries of
    /// keyframes no longer in the map are skipped.
    pub(crate) fn relocalization_candidates(
        &self,
        descriptors: &[[u8; 32]],
        map: &Map,
        max_candidates: usize,
    ) -> Vec<usize> {
        let query = compute_bow(&self.vocabulary, descriptors);
        self.kf_database
            .detect_candidates(&query, &HashSet::new(), 0.0)
            .into_iter()
            .map(|candidate| candidate.kf_idx)
            .filter(|&kf_idx| map.get_keyframe(kf_idx).is_some())
            .take(max_candidates)
            .collect()
    }

    /// Indexes a freshly inserted keyframe for place recognition, queries the
    /// database for appearance-based loop candidates, and closes a verified
    /// loop when correction is allowed.
    ///
    /// Mirrors ORB-SLAM3's `LoopClosing::DetectLoop`: the acceptance threshold is
    /// the lowest BoW similarity to a covisible neighbour, and the covisibility
    /// set is excluded so only a revisited place can match. The query runs before
    /// this keyframe is added, so it never matches itself.
    pub(crate) fn on_keyframe(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        kf_idx: usize,
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        const MIN_COVIS_WEIGHT: usize = 15;
        let started = Instant::now();
        let Some(kf) = map.get_keyframe(kf_idx) else {
            return LoopClosingOutcome::default();
        };
        let bow = compute_bow(&self.vocabulary, &kf.frame.features.descriptors);
        if bow.0.is_empty() {
            return LoopClosingOutcome::default();
        }
        let neighbors = map.covisible_keyframes_min_weight(kf_idx, MIN_COVIS_WEIGHT);
        let candidates = self.kf_database.detect_loop_candidates(
            kf_idx,
            &bow,
            neighbors.iter().map(|&(nb_idx, _w)| nb_idx),
        );
        self.kf_database.add(kf_idx, bow);

        let debug_message = candidates.first().map(|best| {
            format!(
                "[loop] kf={kf_idx} matched kf={} score={:.3} shared_words={} ({} candidates)",
                best.kf_idx,
                best.score,
                best.shared_words,
                candidates.len()
            )
        });

        if self.acceptance.require_imu_initialized && !context.imu_initialized {
            return LoopClosingOutcome {
                debug_message,
                ..Default::default()
            };
        }
        let detect = started.elapsed();
        let mut outcome = self
            .acceptance
            .close(map, camera, kf_idx, &candidates, context);
        outcome.timing.detect = detect;
        LoopClosingOutcome {
            debug_message,
            ..outcome
        }
    }

    /// Every loop correction applied so far, composed: where the corrected map places the
    /// world tracking would have drifted into without loop closing.
    pub(crate) fn map_correction(&self) -> Correction {
        Correction {
            map_from_odometry: self.acceptance.map_from_odometry,
        }
    }

    pub(crate) fn graph_nodes(&self) -> usize {
        self.acceptance.graph.nodes().len()
    }

    /// How many corrections have been written to the map: it advances with every applied loop,
    /// so consumers of [`Self::map_correction`] re-anchor when it changes.
    pub(crate) fn correction_epoch(&self) -> u64 {
        self.acceptance.correction_epoch
    }

    /// Whether a background solve is running or waiting for its writeback.
    pub(crate) fn solve_in_flight(&self) -> bool {
        self.acceptance.solver.in_flight()
    }

    /// Runs solves on `solver` instead of inline.
    pub(crate) fn with_solver(mut self, solver: LoopSolver) -> Self {
        self.acceptance.solver = solver;
        self
    }

    /// Whether solves are written back at frame boundaries, through [`Self::on_frame`].
    pub(crate) fn runs_in_background(&self) -> bool {
        !self.acceptance.solver.is_synchronous()
    }

    /// Called at every frame boundary: writes back a finished background solve, then submits
    /// the loops accepted meanwhile.
    pub(crate) fn on_frame(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        self.acceptance.on_frame(map, camera, context)
    }
}

#[cfg(test)]
impl LoopCloser {
    pub(crate) fn correction_requires_imu(&self) -> bool {
        self.acceptance.require_imu_initialized
    }

    pub(crate) fn indexed_keyframes(&self) -> usize {
        self.kf_database.len()
    }

    pub(crate) fn verified_loop_count(&self) -> usize {
        self.acceptance.verified_loops.len()
    }

    /// What the closer does with a loop the episode tracker made ready.
    pub(crate) fn close_loop_for_test(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        representative: VerifiedLoopEdge,
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        let mut outcome = LoopClosingOutcome::default();
        self.acceptance
            .close_loop(map, camera, representative, context, &mut outcome);
        outcome
    }

    pub(crate) fn mark_verified_for_test(&mut self, a: usize, b: usize) {
        self.acceptance
            .verified_loop_pairs
            .insert(normalized_loop_pair(a, b));
    }
}

/// Verifies loop candidates and applies accepted closures to the map, keeping
/// the history that makes acceptance consistent across keyframes.
struct LoopAcceptance {
    config: LoopClosingConfig,
    require_imu_initialized: bool,
    episode_tracker: LoopEpisodeTracker,
    odometry_gate: OdometryConsistencyGate,
    temporal_gate: TemporalConsistencyGate,
    verified_loops: Vec<VerifiedLoopEdge>,
    verified_loop_pairs: HashSet<(usize, usize)>,
    /// With [`LoopGraphSettings::persistent`], every keyframe the map has held since the last
    /// reset, kept across solves: a culled or evicted keyframe stays a node, so the loops that
    /// touch it still constrain the map. Otherwise the last solve's snapshot.
    graph: PoseGraph<()>,
    /// Verified loops whose edge is in `graph`.
    loops_in_graph: HashSet<(usize, usize)>,
    map_from_odometry: Pose3d,
    correction_epoch: u64,
    solver: LoopSolver,
    /// Accepted loops waiting for a background solve; each reports `Accepted` at its writeback.
    queued: Vec<VerifiedLoopEdge>,
    /// The loops the solve in flight reports, and when it was submitted.
    in_flight: Vec<VerifiedLoopEdge>,
    submitted_at: Option<Instant>,
    /// Background results discarded in a row because the map changed world during the solve.
    discards: usize,
    /// The worker stopped: loops are still accepted, never corrected.
    stopped: bool,
}

/// A background result is discarded while the map keeps changing world under it, and the loops
/// are given up on after this many in a row.
const MAX_DISCARDS: usize = 3;

impl LoopAcceptance {
    fn new(config: LoopClosingConfig, require_imu_initialized: bool) -> Self {
        let episode_tracker = LoopEpisodeTracker::new(config.episode);
        let odometry_gate = OdometryConsistencyGate(config.odometry_gate);
        let temporal_gate =
            TemporalConsistencyGate::new(config.temporal_consistency, config.episode);
        let graph = PoseGraph::new(snapshot_graph_config(
            &config.optimizer,
            config.graph.information,
            None,
        ));
        Self {
            config,
            require_imu_initialized,
            episode_tracker,
            odometry_gate,
            temporal_gate,
            verified_loops: Vec::new(),
            verified_loop_pairs: HashSet::new(),
            graph,
            loops_in_graph: HashSet::new(),
            map_from_odometry: Pose3d::IDENTITY,
            correction_epoch: 0,
            solver: LoopSolver::Synchronous,
            queued: Vec::new(),
            in_flight: Vec::new(),
            submitted_at: None,
            discards: 0,
            stopped: false,
        }
    }

    /// Verifies `candidates` against `kf_idx`, and on an accepted closure
    /// optimizes the pose graph, corrects the map and fuses the loop.
    fn close(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        kf_idx: usize,
        candidates: &[Candidate],
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        let started = Instant::now();
        let mut outcome = LoopClosingOutcome::default();
        if let Some(representative) =
            self.accept(map, camera, kf_idx, candidates, context, &mut outcome)
        {
            if self.solver.is_synchronous() {
                self.close_loop(map, camera, representative, context, &mut outcome);
            } else {
                self.record(representative.clone());
                self.queued.push(representative);
                outcome.queued = true;
                self.submit_queued(map, camera, context, &mut outcome);
            }
        }
        let timing = &mut outcome.timing;
        timing.graph_edges = self.graph.edges().count();
        timing.verify = started
            .elapsed()
            .saturating_sub(timing.solve + timing.writeback + timing.fuse + timing.fuse_capture);
        outcome
    }

    /// Verification, the gates and episode consistency: the loop to close, once an episode is
    /// ready.
    fn accept(
        &mut self,
        map: &Map,
        camera: &PinholeCamera,
        kf_idx: usize,
        candidates: &[Candidate],
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) -> Option<VerifiedLoopEdge> {
        for candidate in candidates {
            let pair = normalized_loop_pair(kf_idx, candidate.kf_idx);
            if self.verified_loop_pairs.contains(&pair) {
                continue;
            }
            if let Ok(edge) = verify_loop_candidate(
                map,
                camera,
                kf_idx,
                candidate.kf_idx,
                &self.config.verification,
            ) {
                let query_order = map
                    .keyframes()
                    .iter()
                    .position(|keyframe| keyframe.frame.idx == kf_idx)
                    .expect("verified query keyframe must be in the map");
                let candidate_order = map
                    .keyframes()
                    .iter()
                    .position(|keyframe| keyframe.frame.idx == candidate.kf_idx)
                    .expect("verified candidate keyframe must be in the map");
                let gravity = context.imu_initialized.then_some(context.gravity_world);
                let correction = loop_correction(map, query_order, candidate_order, &edge, gravity);
                if let Some(correction) = correction {
                    let proposal = LoopProposal {
                        edge: &edge,
                        correction,
                        query_order,
                        candidate_order,
                    };
                    let log = &mut outcome.gate_messages;
                    if let GateDecision::Reject(_) = self.odometry_gate.decide(&proposal, log) {
                        continue;
                    }
                    if self.temporal_gate.is_enabled()
                        && !self.region_has_loop(map, candidate_order)
                        && self.temporal_gate.decide(&proposal, log) == GateDecision::Hold
                    {
                        break;
                    }
                }
                let decision =
                    self.episode_tracker
                        .observe(query_order, candidate_order, edge.clone());
                return match decision {
                    LoopEpisodeDecision::Pending { .. }
                    | LoopEpisodeDecision::Suppressed { .. } => None,
                    LoopEpisodeDecision::Ready { representative, .. } => Some(representative),
                };
            }
        }
        None
    }

    /// Remembers an accepted loop, so its pair is never verified again and its edge joins every
    /// later solve until it is in the graph.
    fn record(&mut self, edge: VerifiedLoopEdge) {
        self.verified_loop_pairs.insert(normalized_loop_pair(
            edge.query_kf_idx,
            edge.candidate_kf_idx,
        ));
        self.verified_loops.push(edge);
    }

    /// Solves the graph with `representative` and every pending loop inline, corrects the map
    /// when the solve is usable, and records the loop either way.
    fn close_loop(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        representative: VerifiedLoopEdge,
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) {
        let job = self.solve_job(
            map,
            self.pending_loops(Some(&representative)),
            context,
            None,
        );
        let solved = job.run();
        self.apply_synchronous(map, camera, solved, &representative, context, outcome);
        self.record(representative);
    }

    /// Every verified loop not yet in the graph, then `newest` when it is not recorded yet. A
    /// snapshot graph holds none, so it takes them all.
    fn pending_loops(&self, newest: Option<&VerifiedLoopEdge>) -> Vec<VerifiedLoopEdge> {
        let persistent = self.config.graph.persistent;
        self.verified_loops
            .iter()
            .filter(|edge| {
                !persistent
                    || !self.loops_in_graph.contains(&normalized_loop_pair(
                        edge.query_kf_idx,
                        edge.candidate_kf_idx,
                    ))
            })
            .chain(newest)
            .cloned()
            .collect()
    }

    /// The solve of `loops` (the last is the newest), detached from the map: the keyframe poses
    /// now and the graph as it stands. The graph is cloned, so `self.graph` stays the rollback
    /// until a result is applied.
    fn solve_job(
        &self,
        map: &Map,
        loops: Vec<VerifiedLoopEdge>,
        context: LoopClosingContext,
        fusion: Option<FusionSnapshot>,
    ) -> LoopSolveJob {
        LoopSolveJob {
            world_epoch: map.world_epoch(),
            keyframe_poses: keyframe_poses(map),
            loops,
            graph: self.graph.clone(),
            optimizer: self.config.optimizer.clone(),
            settings: self.config.graph,
            inertial: context.imu_initialized.then_some(InertialPgoContext {
                gravity_world: context.gravity_world,
            }),
            fusion,
        }
    }

    /// Today's inline path: the snapshot is the live map, so the solved poses are written as they
    /// are.
    fn apply_synchronous(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        solved: LoopSolveOutcome,
        representative: &VerifiedLoopEdge,
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) {
        outcome.timing.solve = solved.solve_time;
        let result = solved.solved.map(|(result, new_world)| {
            if new_world {
                self.loops_in_graph.clear();
                self.map_from_odometry = Pose3d::IDENTITY;
            }
            result
        });
        let failed = |reason: String| LoopClosureEvent::PgoFailed {
            query_kf_idx: representative.query_kf_idx,
            candidate_kf_idx: representative.candidate_kf_idx,
            reason,
        };
        match result {
            Ok(result) => {
                outcome.timing.iterations = result.iterations;
                if result.usable {
                    let tracking_correction = context
                        .current_keyframe_idx
                        .and_then(|reference_kf_idx| {
                            pose_graph_reference_correction(
                                reference_kf_idx,
                                &result.keyframe_indices,
                                &result.original_poses,
                                &result.optimized_poses,
                            )
                        })
                        .map(|(reference_before, reference_after, world)| {
                            (
                                apply_reference_pose_correction(
                                    context.pose_world_to_cam,
                                    reference_before,
                                    reference_after,
                                ),
                                world.rotation * context.velocity_world,
                                world,
                            )
                        });
                    if let Some((pose, velocity, world)) = tracking_correction {
                        let writeback_started = Instant::now();
                        let written = map.apply_pose_graph_correction(
                            &result.keyframe_indices,
                            &result.original_poses,
                            &result.optimized_poses,
                        );
                        outcome.timing.writeback = writeback_started.elapsed();
                        match written {
                            Ok(written) => {
                                outcome.timing.geometry = written.geometry_refresh;
                                outcome.timing.landmarks_refreshed = written.landmarks_refreshed;
                                let fuse_started = Instant::now();
                                fuse_verified_loop(
                                    map,
                                    camera,
                                    representative,
                                    &self.config.fusion,
                                );
                                outcome.timing.fuse = fuse_started.elapsed();
                                outcome.tracking_correction = Some((pose, velocity));
                                outcome.pgo_applied = true;
                                self.correction_epoch += 1;
                                self.map_from_odometry = world.compose(&self.map_from_odometry);
                                outcome.correction = Some(Correction {
                                    map_from_odometry: self.map_from_odometry,
                                });
                            }
                            Err(error) => outcome
                                .events
                                .push(failed(format!("live map correction rejected: {error}"))),
                        }
                    } else {
                        outcome.events.push(failed(
                            "current reference keyframe is outside the PGO snapshot".into(),
                        ));
                    }
                } else {
                    outcome
                        .events
                        .push(failed(result.unusable_reason.unwrap_or_default()));
                }
            }
            Err(error) => outcome.events.push(failed(error.to_string())),
        }
        if outcome.pgo_applied {
            self.loops_in_graph.extend(
                solved
                    .loops
                    .iter()
                    .map(|edge| normalized_loop_pair(edge.query_kf_idx, edge.candidate_kf_idx)),
            );
            self.graph = solved.graph;
        }
        outcome.events.push(LoopClosureEvent::Accepted {
            edge: representative.clone(),
            applied: outcome.pgo_applied,
        });
    }

    /// Submits every queued loop as one solve when none is in flight: loops accepted during a
    /// solve coalesce into the next, which is never aborted.
    fn submit_queued(
        &mut self,
        map: &Map,
        camera: &PinholeCamera,
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) {
        if self.stopped || self.queued.is_empty() || self.solver.in_flight() {
            return;
        }
        let capture_started = Instant::now();
        let fusion = FusionSnapshot::capture(map, camera, &self.queued, &self.config.fusion);
        outcome.timing.fuse_capture = capture_started.elapsed();
        let job = self.solve_job(map, self.pending_loops(None), context, Some(fusion));
        self.in_flight = std::mem::take(&mut self.queued);
        self.submitted_at = Some(Instant::now());
        if let Err(WorkerStopped) = self.solver.submit(job) {
            self.stop(outcome);
        }
    }

    fn on_frame(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        let mut outcome = LoopClosingOutcome::default();
        match self.solver.poll() {
            Ok(Some(solved)) => self.apply_background(map, camera, solved, context, &mut outcome),
            Ok(None) => {}
            Err(WorkerStopped) => self.stop(&mut outcome),
        }
        self.submit_queued(map, camera, context, &mut outcome);
        outcome.timing.graph_edges = self.graph.edges().count();
        outcome
    }

    /// The worker is gone: loop correction stays off, and tracking carries on uncorrected.
    fn stop(&mut self, outcome: &mut LoopClosingOutcome) {
        self.stopped = true;
        self.submitted_at = None;
        let mut loops = std::mem::take(&mut self.in_flight);
        loops.append(&mut self.queued);
        let (query_kf_idx, candidate_kf_idx) = loops
            .last()
            .map_or((0, 0), |edge| (edge.query_kf_idx, edge.candidate_kf_idx));
        outcome.events.push(LoopClosureEvent::PgoFailed {
            query_kf_idx,
            candidate_kf_idx,
            reason: "loop-closing worker stopped; loop correction is off".into(),
        });
        outcome
            .events
            .extend(loops.into_iter().map(|edge| LoopClosureEvent::Accepted {
                edge,
                applied: false,
            }));
    }

    /// Writes a finished background solve back at a frame boundary. A result from another world
    /// is discarded and its loops queued again; anything else reports every loop it carried.
    fn apply_background(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        solved: LoopSolveOutcome,
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) {
        outcome.timing.solve = solved.solve_time;
        outcome.timing.fuse_plan = solved.fusion_plan_time;
        outcome.timing.fusion_ops = solved.fusion_ops.len();
        outcome.written_back = self
            .submitted_at
            .take()
            .map(|submitted| submitted.elapsed());
        let report = std::mem::take(&mut self.in_flight);
        let Some(newest) = report.last().cloned() else {
            return;
        };
        let failed = |reason: String| LoopClosureEvent::PgoFailed {
            query_kf_idx: newest.query_kf_idx,
            candidate_kf_idx: newest.candidate_kf_idx,
            reason,
        };
        if solved.world_epoch != map.world_epoch() {
            self.discards += 1;
            if self.discards < MAX_DISCARDS {
                self.queued.splice(0..0, report);
                return;
            }
            self.discards = 0;
            outcome.events.push(failed(format!(
                "the map changed world during {MAX_DISCARDS} solves in a row"
            )));
        } else {
            self.discards = 0;
            let result = solved.solved.map(|(result, new_world)| {
                if new_world {
                    self.loops_in_graph.clear();
                    self.map_from_odometry = Pose3d::IDENTITY;
                }
                result
            });
            match result {
                Ok(result) => {
                    outcome.timing.iterations = result.iterations;
                    let written = match result.unusable_reason.clone() {
                        Some(reason) => Err(reason),
                        None => self.write_back(
                            map,
                            camera,
                            &result,
                            solved.world_epoch,
                            &solved.fusion_ops,
                            context,
                            outcome,
                        ),
                    };
                    if let Err(reason) = written {
                        outcome.events.push(failed(reason));
                    }
                }
                Err(error) => outcome.events.push(failed(error.to_string())),
            }
            if outcome.pgo_applied {
                self.loops_in_graph.extend(
                    solved
                        .loops
                        .iter()
                        .map(|edge| normalized_loop_pair(edge.query_kf_idx, edge.candidate_kf_idx)),
                );
                self.graph = solved.graph;
            }
        }
        let applied = outcome.pgo_applied;
        outcome.events.extend(
            report
                .into_iter()
                .map(|edge| LoopClosureEvent::Accepted { edge, applied }),
        );
    }

    /// Moves the live map by the solve, then the tracking state and the map correction by the
    /// correction the camera's view takes ([`viewed_correction`]), and applies the fusion planned
    /// on the solve.
    #[allow(clippy::too_many_arguments)]
    fn write_back(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        result: &PgoResult,
        world_epoch: u64,
        fusion: &[FusionOp],
        context: LoopClosingContext,
        outcome: &mut LoopClosingOutcome,
    ) -> Result<(), String> {
        let reference = context
            .current_keyframe_idx
            .filter(|&kf_idx| map.get_keyframe(kf_idx).is_some())
            .ok_or("current reference keyframe is not in the map")?;
        let seen = local_landmark_references(map, reference);
        let writeback_started = Instant::now();
        let written = map.apply_graph_correction(
            world_epoch,
            &result.keyframe_indices,
            &result.original_poses,
            &result.optimized_poses,
        );
        outcome.timing.writeback = writeback_started.elapsed();
        let written = written.map_err(|error| format!("live map correction rejected: {error}"))?;
        outcome.timing.geometry = written.geometry_refresh;
        outcome.timing.landmarks_refreshed = written.landmarks_refreshed;
        outcome.pgo_applied = true;
        self.correction_epoch += 1;
        if let Some(change) = viewed_correction(
            &written.keyframes,
            reference,
            &seen,
            context.pose_world_to_cam,
        ) {
            if change.kf_idx != reference {
                outcome.gate_messages.push(format!(
                    "[loop-view] reference kf={reference} follows kf={}: most of its local map does",
                    change.kf_idx
                ));
            }
            let world = change.after.inverse().compose(&change.before);
            outcome.tracking_correction = Some((
                apply_reference_pose_correction(
                    context.pose_world_to_cam,
                    change.before,
                    change.after,
                ),
                world.rotation * context.velocity_world,
            ));
            self.map_from_odometry = world.compose(&self.map_from_odometry);
        }
        outcome.correction = Some(Correction {
            map_from_odometry: self.map_from_odometry,
        });
        let fuse_started = Instant::now();
        let fused = apply_fusion(map, camera, fusion, &self.config.fusion);
        outcome.timing.fuse = fuse_started.elapsed();
        outcome.timing.fusion_stale = fused.stale;
        Ok(())
    }

    /// Whether an accepted loop already matched a keyframe within the episode
    /// neighbourhood of `candidate_order`.
    fn region_has_loop(&self, map: &Map, candidate_order: usize) -> bool {
        let radius = self.config.episode.candidate_neighborhood_radius;
        self.verified_loops.iter().any(|accepted| {
            map.keyframes()
                .iter()
                .position(|keyframe| keyframe.frame.idx == accepted.candidate_kf_idx)
                .is_some_and(|order| order.abs_diff(candidate_order) <= radius)
        })
    }
}

/// The keyframes whose views make up the camera's local map: the reference keyframe and its
/// best covisible neighbours, as tracking's local map selects them.
const VIEWED_NEIGHBOURS: usize = 10;
const VIEWED_MIN_COVISIBILITY: usize = 15;

/// Where a correction may move the camera centre and still count as the same correction (m).
const VIEWED_AGREEMENT_M: f64 = 0.01;

/// For every live landmark of the reference keyframe's local map, the keyframe it moves with.
fn local_landmark_references(map: &Map, reference: usize) -> Vec<usize> {
    let mut keyframes = vec![reference];
    keyframes.extend(
        map.covisible_keyframes_min_weight(reference, VIEWED_MIN_COVISIBILITY)
            .into_iter()
            .take(VIEWED_NEIGHBOURS)
            .map(|(kf, _)| kf),
    );
    let mut seen = HashSet::new();
    keyframes
        .iter()
        .filter_map(|&kf| map.get_keyframe(kf))
        .flat_map(|keyframe| keyframe.map_point_by_desc_idx.iter().flatten())
        .filter(|&&landmark| seen.insert(landmark))
        .filter_map(|&landmark| map.map_points().get(landmark))
        .filter(|point| !point.culled)
        .map(|point| point.keyframe_idx)
        .collect()
}

/// The keyframe whose correction the camera takes from a background writeback: the one most
/// landmarks of its local map move like, judged by where each would move the camera centre.
///
/// A landmark moves with its own reference keyframe. After loop fusion much of the local map is
/// referenced by keyframes of the earlier visit, which a later solve may barely move while the
/// current reference keyframe moves by the full correction. Tracking then puts the camera back on
/// those landmarks, so moving it, and the map correction, by the reference keyframe alone makes
/// the odometry `M^-1 * T` jump. The reference keyframe's correction is kept whenever at least as
/// many landmarks agree with it as with any other.
fn viewed_correction(
    changes: &[KeyframePoseChange],
    reference: usize,
    landmark_references: &[usize],
    camera: Pose3d,
) -> Option<KeyframePoseChange> {
    let change = |kf: usize| changes.iter().find(|change| change.kf_idx == kf).copied();
    let correction =
        |kf: usize| change(kf).map(|change| change.after.inverse().compose(&change.before));
    let reference_correction = correction(reference)?;
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for &kf in landmark_references {
        match counts.iter_mut().find(|(seen, _)| *seen == kf) {
            Some((_, count)) => *count += 1,
            None => counts.push((kf, 1)),
        }
    }
    let centre = camera.inverse().translation;
    let moved: Vec<(usize, Vec3F64, usize)> = counts
        .into_iter()
        .filter_map(|(kf, count)| Some((kf, correction(kf)?.transform_point(&centre), count)))
        .collect();
    let support = |to: Vec3F64| -> usize {
        moved
            .iter()
            .filter(|(_, other, _)| (*other - to).length() <= VIEWED_AGREEMENT_M)
            .map(|(_, _, count)| count)
            .sum()
    };
    let reference_support = support(reference_correction.transform_point(&centre));
    let best = moved
        .iter()
        .map(|&(kf, to, _)| (support(to), std::cmp::Reverse(kf)))
        .max();
    match best {
        Some((count, std::cmp::Reverse(kf))) if count > reference_support => change(kf),
        _ => change(reference),
    }
}

/// [`optimize_pose_graph`]'s snapshot, edges, solver and usability test through a [`PoseGraph`]
/// of every keyframe, at `information`.
///
/// [`optimize_pose_graph`]: crate::loop_closure::optimize_pose_graph
#[cfg(test)]
pub(super) fn optimize_through_graph(
    map: &Map,
    verified_loops: &[VerifiedLoopEdge],
    config: &PgoConfig,
    information: LoopGraphInformation,
    inertial: Option<InertialPgoContext>,
) -> Result<PgoResult, PgoError> {
    let mut graph = PoseGraph::new(snapshot_graph_config(config, information, inertial));
    let settings = LoopGraphSettings {
        information,
        persistent: false,
    };
    solve_in_graph(
        &mut graph,
        &keyframe_poses(map),
        verified_loops,
        config,
        settings,
        inertial,
    )
    .map(|(result, _)| result)
}

/// The map's keyframes in map order with their world-to-camera poses: what a solve reads.
pub(super) fn keyframe_poses(map: &Map) -> Vec<(usize, Pose3d)> {
    map.keyframes()
        .iter()
        .map(|keyframe| (keyframe.frame.idx, keyframe.frame.pose_world_to_cam))
        .collect()
}

/// Syncs `graph` to the map's `keyframes` (map order, world-to-camera), adds `loops` (the last
/// is the new one) and solves; `true` when the map is a new world and the graph started over. A
/// persistent graph drops earlier loops whose keyframes it never held; a snapshot refuses any
/// loop whose keyframe is not in the map, as [`optimize_pose_graph`] does.
///
/// [`optimize_pose_graph`]: crate::loop_closure::optimize_pose_graph
pub(super) fn solve_in_graph(
    graph: &mut PoseGraph<()>,
    keyframes: &[(usize, Pose3d)],
    loops: &[VerifiedLoopEdge],
    config: &PgoConfig,
    settings: LoopGraphSettings,
    inertial: Option<InertialPgoContext>,
) -> Result<(PgoResult, bool), PgoError> {
    let information = settings.information;
    let solver = |error: PoseGraphError| PgoError::Solver(error.to_string());
    if keyframes.len() < 2 {
        return Err(PgoError::TooFewKeyframes);
    }
    let Some((new_loop, earlier)) = loops.split_last() else {
        return Err(PgoError::MissingLoopKeyframe);
    };
    let new_world = sync_graph(graph, keyframes, settings.persistent).map_err(solver)?;
    graph.set_manifold(snapshot_graph_config(config, information, inertial).manifold);
    let (sigma_m, sigma_rad) = match information {
        LoopGraphInformation::ScalarWeights => {
            let sigma = 1.0 / f64::from(config.loop_edge_weight);
            (sigma, sigma)
        }
        LoopGraphInformation::Drift {
            loop_sigma_m,
            loop_sigma_rad,
            ..
        } => (loop_sigma_m, loop_sigma_rad),
    };
    let known = |kf: usize| graph.node_of_keyframe(kf).is_some();
    let held = |edge: &VerifiedLoopEdge| known(edge.query_kf_idx) && known(edge.candidate_kf_idx);
    if !settings.persistent && !earlier.iter().all(held) {
        return Err(PgoError::MissingLoopKeyframe);
    }
    let loops: Vec<_> = earlier
        .iter()
        .filter(|edge| held(edge))
        .chain(std::iter::once(new_loop))
        .map(|edge| {
            let measurement = LoopMeasurement {
                candidate_to_query: edge.candidate_to_query,
                sigma_m,
                sigma_rad,
            };
            (edge.candidate_kf_idx, edge.query_kf_idx, measurement)
        })
        .collect();
    let report = graph.add_loop_edges(loops).map_err(|error| match error {
        PoseGraphError::UnknownKeyframe(_) => PgoError::MissingLoopKeyframe,
        error => PgoError::Solver(error.to_string()),
    })?;
    let (keyframe_indices, original_poses, optimized_poses) = graph.poses_before_after();
    let gravity_error = inertial.map(|context| {
        max_gravity_alignment_error(&original_poses, &optimized_poses, context.gravity_world)
    });
    let unusable_reason = unusable_reason(
        report.converged,
        report.iterations,
        (report.cost_before, report.cost_after),
        (report.loop_residual_before, report.loop_residual_after),
        gravity_error,
    );
    let result = PgoResult {
        keyframe_indices,
        original_poses,
        optimized_poses,
        iterations: report.iterations,
        usable: unusable_reason.is_none(),
        unusable_reason,
    };
    Ok((result, new_world))
}

/// Makes `graph`'s live keyframes the map's, in map order, at the map's poses: keyframes the map
/// dropped are forgotten where the graph last placed them, new ones are appended. A map sharing
/// no keyframe with the graph is a new world, and the graph starts over (`true`); a graph that
/// does not persist starts over every time.
pub(super) fn sync_graph(
    graph: &mut PoseGraph<()>,
    keyframes: &[(usize, Pose3d)],
    persistent: bool,
) -> Result<bool, PoseGraphError> {
    let in_map: HashSet<usize> = keyframes.iter().map(|&(kf, _)| kf).collect();
    let (live, _, _) = graph.poses_before_after();
    let new_world = !live.is_empty() && !live.iter().any(|kf| in_map.contains(kf));
    if new_world || !persistent {
        graph.reset();
    } else {
        for &kf in live.iter().filter(|kf| !in_map.contains(kf)) {
            let node = graph
                .node_of_keyframe(kf)
                .ok_or(PoseGraphError::UnknownKeyframe(kf))?;
            // Frozen at its last map-world pose, so its edges keep measuring from there.
            let pose = graph.nodes()[node.0].pose;
            graph.refresh_odometry(kf, pose)?;
            graph.forget_keyframe(kf);
        }
        let last_held = keyframes
            .iter()
            .rev()
            .find(|&&(kf, _)| graph.node_of_keyframe(kf).is_some());
        if let Some(&(_, pose)) = last_held {
            graph.rebase_path(pose);
        }
    }
    for &(kf, pose) in keyframes {
        if graph.node_of_keyframe(kf).is_none() {
            graph.add_keyframe(kf, 0.0, pose, (), &mut RejectAll)?;
        }
    }
    for &(kf, pose) in keyframes {
        graph.refresh_odometry(kf, pose)?;
    }
    let (live, _, _) = graph.poses_before_after();
    if live.iter().ne(keyframes.iter().map(|(kf, _)| kf)) {
        return Err(PoseGraphError::Solver(
            "graph keyframes are out of map order".into(),
        ));
    }
    Ok(new_world)
}

/// Every keyframe a node. Scalar weights put `sigma = 1 / w = 1` on odometry edges; `Drift` makes
/// them its random walk, with no linear or per-radian term.
pub(super) fn snapshot_graph_config(
    config: &PgoConfig,
    information: LoopGraphInformation,
    inertial: Option<InertialPgoContext>,
) -> PoseGraphConfig {
    let defaults = PoseGraphConfig::default();
    let (walk_m2_per_m, walk_rad2_per_m, min_sigma_m, min_sigma_rad, loop_cauchy_scale) =
        match information {
            LoopGraphInformation::ScalarWeights => (0.0, 0.0, 1.0, 1.0, None),
            LoopGraphInformation::Drift {
                odometry,
                loop_cauchy_scale,
                ..
            } => (
                odometry.translation_m2_per_m,
                odometry.rotation_rad2_per_m,
                odometry.floor_m,
                odometry.floor_rad,
                Some(loop_cauchy_scale),
            ),
        };
    PoseGraphConfig {
        manifold: inertial.map_or(GraphManifold::Se3, |context| {
            GraphManifold::Gravity(context.gravity_world)
        }),
        node_spacing: None,
        drift_m_per_m: 0.0,
        drift_rad_per_rad: 0.0,
        drift_rad_per_m: 0.0,
        walk_m2_per_m,
        walk_rad2_per_m,
        min_sigma_m,
        min_sigma_rad,
        loop_max_candidates: 0,
        loop_cauchy_scale,
        max_iterations: config.max_iterations,
        cost_tolerance: config.cost_tolerance,
        gradient_tolerance: config.gradient_tolerance,
        initial_lambda: config.initial_lambda,
        ..defaults
    }
}

/// The correction `edge` would apply, over the keyframe path between its ends.
fn loop_correction(
    map: &Map,
    query_order: usize,
    candidate_order: usize,
    edge: &VerifiedLoopEdge,
    gravity: Option<Vec3F64>,
) -> Option<LoopCorrection> {
    let keyframes = map.keyframes();
    let pose = |order: usize| keyframes.get(order).map(|kf| kf.frame.pose_world_to_cam);
    let span = candidate_order.min(query_order)..=candidate_order.max(query_order);
    Some(LoopCorrection::measure(
        pose(candidate_order)?,
        pose(query_order)?,
        keyframes
            .get(span)?
            .iter()
            .map(|kf| kf.frame.pose_world_to_cam),
        edge,
        gravity,
    ))
}

fn normalized_loop_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

#[cfg(test)]
fn pose_graph_tracking_correction(
    current_pose: Pose3d,
    reference_kf_idx: usize,
    keyframe_indices: &[usize],
    poses_before: &[Pose3d],
    poses_after: &[Pose3d],
) -> Option<Pose3d> {
    let (reference_before, reference_after, _) = pose_graph_reference_correction(
        reference_kf_idx,
        keyframe_indices,
        poses_before,
        poses_after,
    )?;
    Some(apply_reference_pose_correction(
        current_pose,
        reference_before,
        reference_after,
    ))
}

fn pose_graph_reference_correction(
    reference_kf_idx: usize,
    keyframe_indices: &[usize],
    poses_before: &[Pose3d],
    poses_after: &[Pose3d],
) -> Option<(Pose3d, Pose3d, Pose3d)> {
    let node = keyframe_indices
        .iter()
        .position(|&keyframe_idx| keyframe_idx == reference_kf_idx)?;
    let reference_before = *poses_before.get(node)?;
    let reference_after = *poses_after.get(node)?;
    let world_correction = reference_after.inverse().compose(&reference_before);
    Some((reference_before, reference_after, world_correction))
}

#[cfg(test)]
mod tests {
    use super::LoopClosingConfig;
    use super::{
        LoopAcceptance, LoopClosingContext, pose_graph_reference_correction,
        pose_graph_tracking_correction,
    };
    use crate::loop_closure::place_recognition::Candidate;
    use crate::mapping::Map;
    use crate::pose_conversion::apply_reference_pose_correction;
    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};

    fn assert_pose_close(actual: Pose3d, expected: Pose3d) {
        assert!((actual.translation - expected.translation).length() < 1e-10);
        for (actual, expected) in actual
            .rotation
            .to_cols_array()
            .iter()
            .zip(expected.rotation.to_cols_array())
        {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn reference_pose_correction_preserves_relative_camera_pose() {
        let reference_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(-1.0, 0.5, 0.2),
        );
        let relative_pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.05, 0.2)).matrix(),
            Vec3F64::new(-0.5, 0.1, 0.3),
        );
        let current_before = relative_pose.compose(&reference_before);
        let reference_after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.1, -0.1)).matrix(),
            Vec3F64::new(-2.0, -0.3, 0.8),
        );

        let corrected =
            apply_reference_pose_correction(current_before, reference_before, reference_after);

        assert_pose_close(Pose3d::between(&reference_after, &corrected), relative_pose);
    }

    #[test]
    fn pose_graph_tracking_correction_uses_current_reference_keyframe() {
        let reference_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(-1.0, 0.5, 0.2),
        );
        let reference_after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.1, -0.1)).matrix(),
            Vec3F64::new(-2.0, -0.3, 0.8),
        );
        let relative_pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.05, 0.2)).matrix(),
            Vec3F64::new(-0.5, 0.1, 0.3),
        );
        let current_before = relative_pose.compose(&reference_before);

        let corrected = pose_graph_tracking_correction(
            current_before,
            20,
            &[10, 20],
            &[Pose3d::IDENTITY, reference_before],
            &[Pose3d::IDENTITY, reference_after],
        )
        .unwrap();

        assert_pose_close(Pose3d::between(&reference_after, &corrected), relative_pose);
        assert!(
            pose_graph_tracking_correction(
                current_before,
                99,
                &[10, 20],
                &[Pose3d::IDENTITY, reference_before],
                &[Pose3d::IDENTITY, reference_after],
            )
            .is_none()
        );
    }

    #[test]
    fn pose_graph_reference_correction_rotates_live_world_velocity() {
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.4, 0.0)).matrix();
        let reference_before = Pose3d::IDENTITY;
        let reference_after = Pose3d::new(yaw.transpose(), Vec3F64::ZERO);
        let (_, _, correction) = pose_graph_reference_correction(
            20,
            &[10, 20],
            &[Pose3d::IDENTITY, reference_before],
            &[Pose3d::IDENTITY, reference_after],
        )
        .unwrap();
        let velocity = Vec3F64::new(1.0, 0.2, -0.5);

        let corrected = correction.rotation * velocity;

        assert!((corrected - yaw * velocity).length() < 1e-10);
    }

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn context() -> LoopClosingContext {
        LoopClosingContext {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity_world: Vec3F64::new(1.0, 0.0, 0.0),
            current_keyframe_idx: Some(0),
            imu_initialized: false,
            gravity_world: Vec3F64::new(0.0, 0.0, -9.81),
        }
    }

    /// An attempt that verifies nothing must leave the runtime's tracking state
    /// alone: no correction to apply, no PGO, nothing to report.
    #[test]
    fn outcome_is_inert_when_no_candidate_verifies() {
        let mut closer = LoopAcceptance::new(LoopClosingConfig::default(), false);
        let mut map = Map::new();
        let candidates = [Candidate {
            kf_idx: 7,
            shared_words: 40,
            score: 0.9,
        }];

        let outcome = closer.close(&mut map, &test_camera(), 3, &candidates, context());

        assert!(outcome.tracking_correction.is_none());
        assert!(!outcome.pgo_applied);
        assert!(outcome.events.is_empty());
    }

    /// With no candidates at all the closer must not touch the map or report.
    #[test]
    fn outcome_is_inert_without_candidates() {
        let mut closer = LoopAcceptance::new(LoopClosingConfig::default(), false);
        let mut map = Map::new();

        let outcome = closer.close(&mut map, &test_camera(), 0, &[], context());

        assert!(outcome.tracking_correction.is_none());
        assert!(!outcome.pgo_applied);
        assert!(outcome.events.is_empty());
        assert_eq!(map.keyframes().len(), 0);
    }

    /// A correction the runtime applies must move the pose and rotate the world
    /// velocity by the same world correction.
    #[test]
    fn reference_correction_yields_pose_and_velocity_together() {
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.3, 0.0)).matrix();
        let reference_before = Pose3d::IDENTITY;
        let reference_after = Pose3d::new(yaw.transpose(), Vec3F64::ZERO);
        let (before, after, world) =
            pose_graph_reference_correction(4, &[4], &[reference_before], &[reference_after])
                .expect("reference keyframe is in the snapshot");

        let pose = apply_reference_pose_correction(Pose3d::IDENTITY, before, after);
        let velocity = world.rotation * Vec3F64::new(1.0, 0.0, 0.0);

        assert_pose_close(pose, reference_after);
        assert!((velocity.length() - 1.0).abs() < 1e-10);
        assert!(velocity.x < 1.0);
    }
}
