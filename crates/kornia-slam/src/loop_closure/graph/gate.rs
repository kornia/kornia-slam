//! Gates a verified loop passes before it may enter the pose graph.

use crate::loop_closure::odometry_gate::TemporalConsistencyTracker;
use crate::loop_closure::{
    LoopCorrection, LoopEpisodeConfig, LoopOdometryGate, LoopTemporalConsistency, TemporalDecision,
    VerifiedLoopEdge,
};

/// A verified loop offered to a gate. Orders are positions in the map's keyframe list.
#[derive(Debug, Clone, Copy)]
pub struct LoopProposal<'a> {
    pub edge: &'a VerifiedLoopEdge,
    pub correction: LoopCorrection,
    pub query_order: usize,
    pub candidate_order: usize,
}

/// What a gate does with a proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    Accept,
    /// Refused as if verification had failed; the next candidate may be tried.
    Reject(String),
    /// Not yet: wait for more evidence before trying any other candidate.
    Hold,
}

/// One acceptance rule. A gate appends at most one log line per decision.
pub trait LoopGate {
    fn decide(&mut self, proposal: &LoopProposal<'_>, log: &mut Vec<String>) -> GateDecision;
}

/// Rejects a correction odometry drift cannot explain; see [`LoopOdometryGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdometryConsistencyGate(pub LoopOdometryGate);

impl LoopGate for OdometryConsistencyGate {
    fn decide(&mut self, proposal: &LoopProposal<'_>, log: &mut Vec<String>) -> GateDecision {
        let Some(verdict) = self.0.judge(&proposal.correction) else {
            return GateDecision::Accept;
        };
        let (edge, correction) = (proposal.edge, &proposal.correction);
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
        if verdict.accepted {
            GateDecision::Accept
        } else {
            GateDecision::Reject(format!(
                "correction {:.3} m / {:.2} deg beyond {} drift sigma",
                correction.translation_m(),
                correction.yaw_rad.to_degrees(),
                verdict.k_sigma
            ))
        }
    }
}

/// Holds a detection until enough consecutive ones imply the same correction; see
/// [`LoopTemporalConsistency`]. The caller skips it where a loop was accepted already.
#[derive(Debug, Clone)]
pub struct TemporalConsistencyGate {
    config: LoopTemporalConsistency,
    tracker: TemporalConsistencyTracker,
}

impl TemporalConsistencyGate {
    pub fn new(config: LoopTemporalConsistency, episode: LoopEpisodeConfig) -> Self {
        Self {
            config,
            tracker: TemporalConsistencyTracker::new(config, episode),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config != LoopTemporalConsistency::Disabled
    }
}

impl LoopGate for TemporalConsistencyGate {
    fn decide(&mut self, proposal: &LoopProposal<'_>, log: &mut Vec<String>) -> GateDecision {
        let correction = proposal.correction;
        let Some(decision) =
            self.tracker
                .observe(proposal.query_order, proposal.candidate_order, correction)
        else {
            return GateDecision::Accept;
        };
        let (decision, state) = match decision {
            TemporalDecision::Pass => (GateDecision::Accept, "pass".to_string()),
            TemporalDecision::Pending { hits, required } => {
                (GateDecision::Hold, format!("pending {hits}/{required}"))
            }
        };
        log.push(format!(
            "[loop-temporal] kf={}~{} corr_t={:.3} corr_yaw_deg={:.2} {state}",
            proposal.edge.query_kf_idx,
            proposal.edge.candidate_kf_idx,
            correction.translation_m(),
            correction.yaw_rad.to_degrees(),
        ));
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;

    fn edge() -> VerifiedLoopEdge {
        VerifiedLoopEdge {
            query_kf_idx: 40,
            candidate_kf_idx: 3,
            candidate_to_query: Pose3d::IDENTITY,
            correspondences: 40,
            inliers: 35,
            inlier_ratio: 0.875,
            reprojection_rmse_px: 1.0,
            occupied_cells: 8,
        }
    }

    fn proposal(edge: &VerifiedLoopEdge, translation: f64, query_order: usize) -> LoopProposal<'_> {
        LoopProposal {
            edge,
            correction: LoopCorrection {
                translation_world: Vec3F64::new(translation, 0.0, 0.0),
                yaw_rad: 0.0,
                path_m: 21.0,
                path_rotation_rad: 0.0,
            },
            query_order,
            candidate_order: 3,
        }
    }

    #[test]
    fn odometry_gate_rejects_beyond_drift_and_logs_each_decision() {
        let edge = edge();
        let mut gate = OdometryConsistencyGate(LoopOdometryGate::default());
        let mut log = Vec::new();
        assert!(matches!(
            gate.decide(&proposal(&edge, 6.0, 40), &mut log),
            GateDecision::Reject(_)
        ));
        assert_eq!(
            gate.decide(&proposal(&edge, 0.5, 40), &mut log),
            GateDecision::Accept
        );
        assert!(log[0].starts_with("[loop-gate] kf=40~3 ") && log[0].ends_with(" reject"));
        assert!(log[1].ends_with(" accept"));
    }

    #[test]
    fn disabled_odometry_gate_accepts_silently() {
        let edge = edge();
        let mut gate = OdometryConsistencyGate(LoopOdometryGate::Disabled);
        let mut log = Vec::new();
        assert_eq!(
            gate.decide(&proposal(&edge, 100.0, 40), &mut log),
            GateDecision::Accept
        );
        assert!(log.is_empty());
    }

    #[test]
    fn temporal_gate_holds_until_consecutive_detections_agree() {
        let edge = edge();
        let mut gate = TemporalConsistencyGate::new(
            LoopTemporalConsistency::enabled(2),
            LoopEpisodeConfig::default(),
        );
        let mut log = Vec::new();
        assert!(gate.is_enabled());
        assert_eq!(
            gate.decide(&proposal(&edge, 0.4, 40), &mut log),
            GateDecision::Hold
        );
        assert_eq!(
            gate.decide(&proposal(&edge, 0.45, 41), &mut log),
            GateDecision::Accept
        );
        assert!(log[0].ends_with(" pending 1/2"));
        assert!(log[1].ends_with(" pass"));
    }
}
