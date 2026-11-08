//! Loop closing for one keyframe: place recognition, verification, episode
//! consistency, pose-graph optimization and map writeback.
//!
//! [`LoopCloser`] owns the bag-of-words database and the acceptance history,
//! and decides what happens to the map. It reports what the runtime must apply
//! to its own tracking state through [`LoopClosingOutcome`] rather than
//! reaching into it.

use std::collections::HashSet;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::loop_closure::odometry_gate::TemporalConsistencyTracker;
use crate::loop_closure::place_recognition::{
    Candidate, KeyFrameDatabase, Vocabulary, compute_bow,
};
use crate::loop_closure::{
    InertialPgoContext, LoopCorrection, LoopEpisodeConfig, LoopEpisodeDecision, LoopEpisodeTracker,
    LoopFusionConfig, LoopOdometryGate, LoopTemporalConsistency, LoopVerificationConfig, PgoConfig,
    TemporalDecision, VerifiedLoopEdge, fuse_verified_loop, optimize_pose_graph,
    verify_loop_candidate,
};
use crate::mapping::Map;
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
        LoopClosingOutcome {
            debug_message,
            ..self
                .acceptance
                .close(map, camera, kf_idx, &candidates, context)
        }
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
    temporal_tracker: TemporalConsistencyTracker,
    verified_loops: Vec<VerifiedLoopEdge>,
    verified_loop_pairs: HashSet<(usize, usize)>,
}

impl LoopAcceptance {
    fn new(config: LoopClosingConfig, require_imu_initialized: bool) -> Self {
        let episode_tracker = LoopEpisodeTracker::new(config.episode);
        let temporal_tracker =
            TemporalConsistencyTracker::new(config.temporal_consistency, config.episode);
        Self {
            config,
            require_imu_initialized,
            episode_tracker,
            temporal_tracker,
            verified_loops: Vec::new(),
            verified_loop_pairs: HashSet::new(),
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
        let inertial_pgo = context.imu_initialized.then_some(InertialPgoContext {
            gravity_world: context.gravity_world,
        });
        let mut outcome = LoopClosingOutcome::default();
        let mut accepted = None;
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
                    if !self.admit(&edge, &correction, &mut outcome.gate_messages) {
                        continue;
                    }
                    if !self.temporally_consistent(
                        map,
                        query_order,
                        candidate_order,
                        &edge,
                        correction,
                        &mut outcome.gate_messages,
                    ) {
                        break;
                    }
                }
                let decision =
                    self.episode_tracker
                        .observe(query_order, candidate_order, edge.clone());
                match decision {
                    LoopEpisodeDecision::Pending { .. }
                    | LoopEpisodeDecision::Suppressed { .. } => {}
                    LoopEpisodeDecision::Ready { representative, .. } => {
                        let pair = normalized_loop_pair(
                            representative.query_kf_idx,
                            representative.candidate_kf_idx,
                        );
                        let mut loops = self.verified_loops.clone();
                        loops.push(representative.clone());
                        match optimize_pose_graph(map, &loops, &self.config.optimizer, inertial_pgo)
                        {
                            Ok(result) => {
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
                                            )
                                        });
                                    if let Some(tracking_correction) = tracking_correction {
                                        match map.apply_pose_graph_correction(
                                            &result.keyframe_indices,
                                            &result.original_poses,
                                            &result.optimized_poses,
                                        ) {
                                            Ok(_) => {
                                                fuse_verified_loop(
                                                    map,
                                                    camera,
                                                    &representative,
                                                    &self.config.fusion,
                                                );
                                                outcome.tracking_correction =
                                                    Some(tracking_correction);
                                                outcome.pgo_applied = true;
                                            }
                                            Err(error) => {
                                                outcome.events.push(LoopClosureEvent::PgoFailed {
                                                    query_kf_idx: representative.query_kf_idx,
                                                    candidate_kf_idx: representative
                                                        .candidate_kf_idx,
                                                    reason: format!(
                                                        "live map correction rejected: {error}"
                                                    ),
                                                })
                                            }
                                        }
                                    } else {
                                        outcome.events.push(LoopClosureEvent::PgoFailed {
                                            query_kf_idx: representative.query_kf_idx,
                                            candidate_kf_idx: representative.candidate_kf_idx,
                                            reason:
                                                "current reference keyframe is outside the PGO snapshot"
                                                    .into(),
                                        });
                                    }
                                }
                            }
                            Err(error) => outcome.events.push(LoopClosureEvent::PgoFailed {
                                query_kf_idx: representative.query_kf_idx,
                                candidate_kf_idx: representative.candidate_kf_idx,
                                reason: error.to_string(),
                            }),
                        }
                        outcome.events.push(LoopClosureEvent::Accepted {
                            edge: representative.clone(),
                            applied: outcome.pgo_applied,
                        });
                        accepted = Some((pair, representative));
                    }
                }
                break;
            }
        }
        if let Some((pair, edge)) = accepted {
            self.verified_loop_pairs.insert(pair);
            self.verified_loops.push(edge);
        }
        outcome
    }

    /// The odometry gate: `false` rejects the verified edge as if verification
    /// had failed, so the next candidate is tried.
    fn admit(
        &self,
        edge: &VerifiedLoopEdge,
        correction: &LoopCorrection,
        log: &mut Vec<String>,
    ) -> bool {
        let Some(verdict) = self.config.odometry_gate.judge(correction) else {
            return true;
        };
        log.push(format!(
            "[loop-gate] kf={}~{} path_m={:.2} path_rot_deg={:.1} corr_t={:.3} corr_yaw_deg={:.2} sigma_t={:.3} sigma_yaw_deg={:.2} k={} {}",
            edge.query_kf_idx,
            edge.candidate_kf_idx,
            correction.path_m,
            correction.path_rotation_rad.to_degrees(),
            correction.translation_m(),
            correction.yaw_rad.to_degrees(),
            verdict.sigma_t_m,
            verdict.sigma_yaw_rad.to_degrees(),
            verdict.k_sigma,
            if verdict.accepted { "accept" } else { "reject" },
        ));
        verdict.accepted
    }

    /// The temporal rule, applied only where no loop was accepted yet: `false`
    /// holds the detection back until enough consecutive ones agree.
    fn temporally_consistent(
        &mut self,
        map: &Map,
        query_order: usize,
        candidate_order: usize,
        edge: &VerifiedLoopEdge,
        correction: LoopCorrection,
        log: &mut Vec<String>,
    ) -> bool {
        if self.config.temporal_consistency == LoopTemporalConsistency::Disabled
            || self.region_has_loop(map, candidate_order)
        {
            return true;
        }
        let Some(decision) =
            self.temporal_tracker
                .observe(query_order, candidate_order, correction)
        else {
            return true;
        };
        let (passed, state) = match decision {
            TemporalDecision::Pass => (true, "pass".to_string()),
            TemporalDecision::Pending { hits, required } => {
                (false, format!("pending {hits}/{required}"))
            }
        };
        log.push(format!(
            "[loop-temporal] kf={}~{} corr_t={:.3} corr_yaw_deg={:.2} {state}",
            edge.query_kf_idx,
            edge.candidate_kf_idx,
            correction.translation_m(),
            correction.yaw_rad.to_degrees(),
        ));
        passed
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
