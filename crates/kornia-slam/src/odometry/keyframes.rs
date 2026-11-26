//! The keyframe rule and which variables leave the window when
//! (`sqrt_keypoint_vio.cpp:340-373, 552-685`; design card M0-M6).

use std::collections::{BTreeMap, BTreeSet};

use kornia_algebra::Vec3F64;

use super::config::OdometryTuning;
use super::state::FrameId;

/// A frame is a keyframe when under `keyframe_tracked_share` of its
/// camera-0 tracks have a landmark and more than `keyframe_min_frames`
/// frames passed since the last keyframe (`:370-373`).
pub(crate) fn is_keyframe(
    connected0: usize,
    unconnected0: usize,
    frames_after_keyframe: usize,
    tuning: &OdometryTuning,
) -> bool {
    let total = connected0 + unconnected0;
    total > 0
        && (connected0 as f64) < tuning.keyframe_tracked_share * total as f64
        && frames_after_keyframe > tuning.keyframe_min_frames
}

/// Landmarks each keyframe created, and how many of its landmarks the
/// current frame observes (any camera, as Basalt's `num_points_connected`).
#[derive(Debug, Clone, Default)]
pub(crate) struct KeyframeCounts {
    pub created: BTreeMap<FrameId, usize>,
    pub connected: BTreeMap<FrameId, usize>,
}

/// The variables one marginalization removes (M2-M6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MargPlan {
    /// The oldest state that stays; it becomes the prior's 15-DOF state.
    pub last_state_to_marg: FrameId,
    /// Older keyframe states: velocity and biases leave, the pose stays.
    pub states_vel_bias: Vec<FrameId>,
    /// Older states that are not keyframes: they leave whole.
    pub states_all: Vec<FrameId>,
    /// Keyframe poses dropped.
    pub keyframes_to_marg: Vec<FrameId>,
}

/// The marginalization of this frame, or `None` (M0, M1). `states` are the
/// frame states oldest first; `position` gives a keyframe's body position in
/// the world, from its pose or its state.
pub(crate) fn plan_marginalization(
    states: &[FrameId],
    keyframes: &BTreeSet<FrameId>,
    counts: &KeyframeCounts,
    position: impl Fn(FrameId) -> Option<Vec3F64>,
    optimization_started: bool,
    tuning: &OdometryTuning,
) -> Option<MargPlan> {
    if !optimization_started || states.is_empty() {
        return None;
    }
    if keyframes.len() <= tuning.max_keyframes && states.len() < tuning.max_states {
        return None;
    }
    let remove = (states.len() + 1).saturating_sub(tuning.max_states);
    let last_state_to_marg = states[remove.min(states.len() - 1)];
    let (states_vel_bias, states_all): (Vec<FrameId>, Vec<FrameId>) = states
        [..remove.min(states.len() - 1)]
        .iter()
        .partition(|id| keyframes.contains(id));
    let mut kept: BTreeSet<FrameId> = keyframes.clone();
    let mut keyframes_to_marg = Vec::new();
    while kept.len() > tuning.max_keyframes && !states_vel_bias.is_empty() {
        let candidates: Vec<FrameId> = kept
            .iter()
            .copied()
            .take(kept.len().saturating_sub(2))
            .collect();
        let Some(id) = by_tracked_share(&candidates, counts, tuning)
            .or_else(|| by_distance_score(&candidates, &kept, &position))
        else {
            break;
        };
        keyframes_to_marg.push(id);
        kept.remove(&id);
    }
    keyframes_to_marg.sort_unstable();
    Some(MargPlan {
        last_state_to_marg,
        states_vel_bias,
        states_all,
        keyframes_to_marg,
    })
}

/// M5: the oldest candidate whose landmarks the current frame tracks under
/// `keyframe_marg_feature_ratio` of those it created.
fn by_tracked_share(
    candidates: &[FrameId],
    counts: &KeyframeCounts,
    tuning: &OdometryTuning,
) -> Option<FrameId> {
    candidates.iter().copied().find(|id| {
        let connected = counts.connected.get(id).copied().unwrap_or(0);
        let created = counts.created.get(id).copied().unwrap_or(0);
        connected == 0 || (connected as f64) < tuning.keyframe_marg_feature_ratio * created as f64
    })
}

/// M6: the candidate of lowest `√d(i, newest) · Σ_j 1 / (d(i, j) + 1e-5)`,
/// DSO's score (`:638-671`).
fn by_distance_score(
    candidates: &[FrameId],
    keyframes: &BTreeSet<FrameId>,
    position: &impl Fn(FrameId) -> Option<Vec3F64>,
) -> Option<FrameId> {
    let newest = position(*keyframes.iter().next_back()?)?;
    let mut best: Option<(f64, FrameId)> = None;
    for &i in candidates {
        let pi = position(i)?;
        let denominator: f64 = candidates
            .iter()
            .filter_map(|&j| position(j))
            .map(|pj| 1.0 / ((pi - pj).length() + 1e-5))
            .sum();
        let score = (pi - newest).length().sqrt() * denominator;
        if best.is_none_or(|(s, _)| score < s) {
            best = Some((score, i));
        }
    }
    best.map(|(_, id)| id)
}
