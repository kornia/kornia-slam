//! Loop-closing orchestration, geometric verification, fusion, and pose-graph optimization.

mod episode;
mod fusion;
mod pose_graph;
mod verification;

pub use episode::{LoopEpisodeConfig, LoopEpisodeDecision, LoopEpisodeTracker};
pub use fusion::{LoopFusionConfig, LoopFusionStats, fuse_verified_loop};
pub use pose_graph::{InertialPgoContext, PgoConfig, PgoError, PgoResult, optimize_pose_graph};
pub use verification::{
    LoopVerificationConfig, LoopVerificationReject, VerifiedLoopEdge, verify_loop_candidate,
};

use crate::map::Map;
use crate::place_recognition::{KeyFrameDatabase, Vocabulary, compute_bow};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct LoopClosingConfig {
    /// Mono+IMU maps become metric only after inertial initialization. Stereo
    /// maps are metric from bootstrap and leave this disabled.
    pub require_imu_initialized: bool,
    pub episode: LoopEpisodeConfig,
    pub fusion: LoopFusionConfig,
    pub verification: LoopVerificationConfig,
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

/// Runtime inputs required for map-side loop closing.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoopClosingContext {
    pub reference_keyframe_idx: Option<usize>,
    pub inertial: Option<InertialPgoContext>,
}

/// Reference geometry for correcting the live pose and world velocity.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReferencePoseCorrection {
    pub before: Pose3d,
    pub after: Pose3d,
    pub world: Pose3d,
}

/// A correction is returned only after the map accepted the PGO writeback.
#[derive(Debug, Default)]
pub(crate) struct LoopClosingOutcome {
    pub events: Vec<LoopClosureEvent>,
    pub reference_correction: Option<ReferencePoseCorrection>,
    pub debug_message: Option<String>,
}

/// Owns loop retrieval and acceptance history. The runtime owns synchronization,
/// live tracking state, and any BA scheduling following a correction.
pub(crate) struct LoopCloser {
    vocabulary: Option<Vocabulary>,
    kf_database: KeyFrameDatabase,
    pgo_config: Option<LoopClosingConfig>,
    loop_episode_tracker: Option<LoopEpisodeTracker>,
    verified_loops: Vec<VerifiedLoopEdge>,
    verified_loop_pairs: HashSet<(usize, usize)>,
}

impl LoopCloser {
    pub(crate) fn new(pgo_config: Option<LoopClosingConfig>) -> Self {
        let loop_episode_tracker = pgo_config
            .as_ref()
            .map(|config| LoopEpisodeTracker::new(config.episode));
        Self {
            vocabulary: None,
            kf_database: KeyFrameDatabase::new(),
            pgo_config,
            loop_episode_tracker,
            verified_loops: Vec::new(),
            verified_loop_pairs: HashSet::new(),
        }
    }

    pub(crate) fn set_vocabulary(&mut self, vocabulary: Vocabulary) {
        self.vocabulary = Some(vocabulary);
    }

    /// Queries before indexing the new keyframe, then verifies and applies a
    /// consistent loop. Indexing remains enabled when PGO is disabled or gated.
    /// The caller holds the runtime's publication gate throughout this operation.
    pub(crate) fn on_keyframe(
        &mut self,
        map: &mut Map,
        camera: &PinholeCamera,
        kf_idx: usize,
        context: LoopClosingContext,
    ) -> LoopClosingOutcome {
        let mut outcome = LoopClosingOutcome::default();
        let Some(vocabulary) = self.vocabulary.as_ref() else {
            return outcome;
        };
        const MIN_COVIS_WEIGHT: usize = 15;
        let (bow, neighbors) = {
            let Some(kf) = map.get_keyframe(kf_idx) else {
                return outcome;
            };
            let bow = compute_bow(vocabulary, &kf.frame.features.descriptors);
            if bow.0.is_empty() {
                return outcome;
            }
            let neighbors = map.covisible_keyframes(kf_idx, MIN_COVIS_WEIGHT);
            (bow, neighbors)
        };
        let candidates = self.kf_database.detect_loop_candidates(
            kf_idx,
            &bow,
            neighbors.iter().map(|&(nb_idx, _w)| nb_idx),
        );
        self.kf_database.add(kf_idx, bow);

        if let Some(best) = candidates.first().copied() {
            outcome.debug_message = Some(format!(
                "[loop] kf={kf_idx} matched kf={} score={:.3} shared_words={} ({} candidates)",
                best.kf_idx,
                best.score,
                best.shared_words,
                candidates.len()
            ));
        }

        let Some(pgo_config) = self.pgo_config.clone() else {
            return outcome;
        };
        if pgo_config.require_imu_initialized && context.inertial.is_none() {
            return outcome;
        }
        let inertial_pgo = context.inertial;
        let (events, accepted, reference_correction) = {
            let mut events = Vec::new();
            let mut accepted = None;
            let mut reference_correction = None;
            let mut pgo_applied = false;
            for candidate in &candidates {
                let pair = normalized_loop_pair(kf_idx, candidate.kf_idx);
                if self.verified_loop_pairs.contains(&pair) {
                    continue;
                }
                if let Ok(edge) = verify_loop_candidate(
                    map,
                    camera,
                    kf_idx,
                    candidate.kf_idx,
                    &pgo_config.verification,
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
                    let decision = self
                        .loop_episode_tracker
                        .as_mut()
                        .expect("PGO config must create an episode tracker")
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
                            match optimize_pose_graph(
                                map,
                                &loops,
                                &pgo_config.optimizer,
                                inertial_pgo,
                            ) {
                                Ok(result) => {
                                    if result.usable {
                                        let tracking_correction = context
                                            .reference_keyframe_idx
                                            .and_then(|reference_kf_idx| {
                                                pose_graph_reference_correction(
                                                    reference_kf_idx,
                                                    &result.keyframe_indices,
                                                    &result.original_poses,
                                                    &result.optimized_poses,
                                                )
                                            })
                                            .map(|(before, after, world)| {
                                                ReferencePoseCorrection {
                                                    before,
                                                    after,
                                                    world,
                                                }
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
                                                        &pgo_config.fusion,
                                                    );
                                                    reference_correction =
                                                        Some(tracking_correction);
                                                    pgo_applied = true;
                                                }
                                                Err(error) => {
                                                    events.push(LoopClosureEvent::PgoFailed {
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
                                            events.push(LoopClosureEvent::PgoFailed {
                                                    query_kf_idx: representative.query_kf_idx,
                                                    candidate_kf_idx: representative
                                                        .candidate_kf_idx,
                                                    reason: "current reference keyframe is outside the PGO snapshot"
                                                        .into(),
                                                });
                                        }
                                    }
                                }
                                Err(error) => events.push(LoopClosureEvent::PgoFailed {
                                    query_kf_idx: representative.query_kf_idx,
                                    candidate_kf_idx: representative.candidate_kf_idx,
                                    reason: error.to_string(),
                                }),
                            }
                            events.push(LoopClosureEvent::Accepted {
                                edge: representative.clone(),
                                applied: pgo_applied,
                            });
                            accepted = Some((pair, representative));
                        }
                    }
                    break;
                }
            }
            (events, accepted, reference_correction)
        };
        outcome.events = events;
        outcome.reference_correction = reference_correction;
        if let Some((pair, edge)) = accepted {
            self.verified_loop_pairs.insert(pair);
            self.verified_loops.push(edge);
        }
        outcome
    }
}

fn normalized_loop_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

pub(crate) fn pose_graph_reference_correction(
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
use pose_graph::{max_gravity_alignment_error, pose_graph_cost};
#[cfg(test)]
use verification::verification_input;

#[cfg(test)]
mod tests;
