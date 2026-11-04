//! Redundant-keyframe culling policy.
//!
//! Selection lives here because the thresholds are a mapping judgement about
//! which landmarks and keyframes have earned their place. Removal itself is
//! the map's canonical operation, so this selects ids and delegates — there is
//! no second cleanup loop, and the map never calls back into mapping.

use crate::mapping::map::{Keyframe, Map, StillNeighbour, StillSide};
use crate::mapping::stationary::StationaryFactor;
use crate::mapping::wheel_odom::WheelOdomFactor;
use std::collections::HashMap;

// Landmark culling is implemented in `mapping::culling`.

#[cfg(test)]
#[path = "culling_tests.rs"]
pub(crate) mod keyframe_tests;

/// Redundant-keyframe culling (ORB-SLAM's local-mapping `KeyFrameCulling`).
///
/// Off unless the caller runs [`cull_redundant_keyframes`]; the defaults are
/// ORB-SLAM3's where it has one. `#[non_exhaustive]`: start from
/// [`Default`] and set the fields.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct KeyframeCullingConfig {
    /// Fraction of a keyframe's live landmarks that must be redundant for the
    /// keyframe to be (ORB-SLAM: 0.9).
    pub redundant_ratio: f64,
    /// Other keyframes that must see a landmark for it to be redundant
    /// (ORB-SLAM: 3).
    pub min_other_observers: usize,
    /// Pyramid levels coarser than this keyframe's that another observation
    /// may sit at and still count: 1 is ORB-SLAM3's code, 0 its paper's
    /// "same or finer scale".
    pub scale_level_tolerance: u8,
    /// Newest keyframes never culled: the tracker's reference keyframe and
    /// whatever still names the newest ones (pending IMU or wheel intervals,
    /// queued local-mapping work). At least 1 is always kept.
    pub protect_newest: usize,
    /// No culling while the map holds this many keyframes or fewer
    /// (ORB-SLAM3's `Nd = 21` for inertial maps).
    pub min_keyframes: usize,
    /// Cull keyframes inside the IMU chain by merging their two edges. `false`
    /// restricts culling to keyframes no IMU edge touches.
    pub merge_imu_edges: bool,
    /// Longest IMU edge a merge may produce, seconds. One value whatever the
    /// IMU state: the culler is not told whether inertial initialization has
    /// run. ORB-SLAM3 allows 3 s once the IMU is initialized and 0.5 s before,
    /// so a caller that culls before initialization should set 0.5 (or not
    /// cull then); the 3.0 default is the initialized limit.
    pub max_merged_imu_span_sec: f64,
    /// Most keyframes one call removes.
    pub max_removals: usize,
    /// ORB-SLAM3's stereo rule: in a stereo keyframe, count only landmarks
    /// whose feature has a stereo depth in `(0, d]` (its `mThDepth`, the
    /// close-depth threshold), so far and mono-only points cannot make a
    /// keyframe that holds the only close metric points look redundant.
    /// `None` (the default) counts every live landmark, in stereo keyframes
    /// too; set it to the frontend's stereo close depth for ORB-SLAM3's
    /// behaviour. Monocular keyframes are unaffected.
    pub stereo_close_depth_m: Option<f64>,
}

impl Default for KeyframeCullingConfig {
    fn default() -> Self {
        Self {
            redundant_ratio: 0.9,
            min_other_observers: 3,
            scale_level_tolerance: 1,
            protect_newest: 3,
            min_keyframes: 21,
            merge_imu_edges: true,
            max_merged_imu_span_sec: 3.0,
            max_removals: usize::MAX,
            stereo_close_depth_m: None,
        }
    }
}

/// What [`cull_redundant_keyframes`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct KeyframeCullReport {
    /// Removed keyframes (`Frame::idx`), in removal order. Anything outside
    /// the map that holds keyframe ids — a place-recognition database, the
    /// tracker's last keyframe — must drop these.
    pub removed: Vec<usize>,
    pub imu_edges_merged: usize,
    pub wheel_edges_merged: usize,
    /// Stationary factors composed across a removed keyframe.
    pub stationary_edges_merged: usize,
    /// Keyframes removed at an end of a stationary chain, their stationary
    /// factor dropped and their IMU edge moved onto the still neighbour.
    pub stationary_boundaries_removed: usize,
    /// Redundant keyframes kept because their IMU, wheel or stationary edges
    /// could not be merged, or the merged IMU edge would have been too long.
    pub kept_for_edges: usize,
    /// Landmarks retired because a removed keyframe was their last observer.
    pub landmarks_retired: usize,
}

/// Removes redundant keyframes: those whose live landmarks are, by at least
/// `redundant_ratio`, each seen by `min_other_observers` other keyframes at
/// most `scale_level_tolerance` pyramid levels coarser than its own. The
/// default 1 is ORB-SLAM3's code rule and also counts a view one level
/// coarser; 0 gives its paper's "same or finer scale". In a stereo keyframe
/// only close stereo points count when `stereo_close_depth_m` is set, as in
/// ORB-SLAM3; by default every live landmark does.
///
/// The first keyframe (the gauge anchor) and the `protect_newest` newest are
/// never removed. Candidates are judged oldest first against the live map, so
/// each removal is seen by the next judgement. IMU edges through a removed
/// keyframe are merged by [`Map::remove_keyframe`]; `wheel_factors`, keyed by
/// `Frame::idx` as [`crate::mapping::bundle_adjustment::run_local_inertial_ba`] takes
/// them, have the pair through it composed into one. A keyframe either chain
/// cannot be merged across is kept, so no edge is ever cut.
///
/// `stationary_factors`, held the same way, are composed across a removed
/// keyframe with no span limit ([`StationaryFactor::compose`]):
/// `max_merged_imu_span_sec` caps IMU merges only. A keyframe linked by one
/// stationary factor in and one out, and by no IMU edge, is removable however
/// long the parked span. One at the boundary between a stationary and an IMU
/// edge is removable too: its stationary factor is dropped and the IMU edge
/// moves onto the still neighbour ([`Map::remove_keyframe_beside_still`]), so
/// the boundaries a stop leaves never pin keyframes. Boundaries are judged
/// after every other candidate, each at most once per call. A bare chain end
/// (a stationary factor and nothing on the other side) is kept. `None` culls
/// exactly as before.
pub fn cull_redundant_keyframes(
    map: &mut Map,
    config: &KeyframeCullingConfig,
    mut wheel_factors: Option<&mut Vec<WheelOdomFactor>>,
    mut stationary_factors: Option<&mut Vec<StationaryFactor>>,
) -> KeyframeCullReport {
    let mut report = KeyframeCullReport::default();
    let keyframes = map.keyframes();
    let protected_from = keyframes.len().saturating_sub(config.protect_newest.max(1));
    let candidates: Vec<usize> = keyframes
        .iter()
        .take(protected_from)
        .skip(1)
        .map(|kf| kf.frame.idx)
        .collect();

    // Boundaries go last, each once per call: removing one drops a stationary factor (its gyro
    // mean with it), so the chain behind it is composed first, and the keyframe that becomes the
    // next boundary waits for the next call instead of the IMU edge eating the whole stop now.
    let mut boundaries = Vec::new();
    for kf_idx in candidates {
        let step = cull_one(
            map,
            config,
            kf_idx,
            false,
            wheel_factors.as_deref_mut(),
            stationary_factors.as_deref_mut(),
            &mut report,
        );
        match step {
            Step::Done => {}
            Step::Boundary => boundaries.push(kf_idx),
            Step::Stop => return report,
        }
    }
    for kf_idx in boundaries {
        let step = cull_one(
            map,
            config,
            kf_idx,
            true,
            wheel_factors.as_deref_mut(),
            stationary_factors.as_deref_mut(),
            &mut report,
        );
        if step == Step::Stop {
            break;
        }
    }
    report
}

/// What [`cull_one`] did with a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Removed, kept, or not redundant.
    Done,
    /// A stationary boundary, left for the boundary pass.
    Boundary,
    /// A bound was reached; nothing more is removed.
    Stop,
}

/// Judges and, if allowed, removes one candidate, repairing the factor chains and the report.
/// A stationary boundary is removed only when `boundaries` is set.
#[allow(clippy::too_many_arguments)]
fn cull_one(
    map: &mut Map,
    config: &KeyframeCullingConfig,
    kf_idx: usize,
    boundaries: bool,
    wheel_factors: Option<&mut Vec<WheelOdomFactor>>,
    stationary_factors: Option<&mut Vec<StationaryFactor>>,
    report: &mut KeyframeCullReport,
) -> Step {
    if report.removed.len() >= config.max_removals || map.keyframes().len() <= config.min_keyframes
    {
        return Step::Stop;
    }
    if map.get_keyframe(kf_idx).is_none() || !is_redundant(map, kf_idx, config) {
        return Step::Done;
    }
    let wheel_merge = match wheel_factors.as_deref() {
        Some(factors) => match wheel_merge_through(factors, kf_idx) {
            Ok(merge) => merge,
            Err(()) => {
                report.kept_for_edges += 1;
                return Step::Done;
            }
        },
        None => None,
    };
    let stationary = match stationary_factors.as_deref() {
        Some(factors) => match stationary_plan(factors, kf_idx) {
            Ok(plan) => plan,
            Err(()) => {
                report.kept_for_edges += 1;
                return Step::Done;
            }
        },
        None => StationaryPlan::None,
    };
    let removal = match &stationary {
        // Removing a boundary moves its IMU edge onto the still neighbour,
        // which `merge_imu_edges = false` rules out like any IMU change.
        StationaryPlan::Boundary { .. } if !config.merge_imu_edges => None,
        StationaryPlan::Boundary { .. } if !boundaries => return Step::Boundary,
        // Stationary on both sides: no span cap, but no IMU edge either
        // (the map would refuse a one-sided IMU chain anyway).
        StationaryPlan::Compose(_) if touches_imu_edge(map, kf_idx) => None,
        StationaryPlan::Compose(_) => map.remove_keyframe(kf_idx).ok(),
        // The IMU edge keeps its span, so the cap does not apply.
        StationaryPlan::Boundary { still, .. } => {
            map.remove_keyframe_beside_still(kf_idx, *still).ok()
        }
        StationaryPlan::None if imu_allows_removal(map, kf_idx, config) => {
            map.remove_keyframe(kf_idx).ok()
        }
        StationaryPlan::None => None,
    };
    let Some(removal) = removal else {
        report.kept_for_edges += 1;
        return Step::Done;
    };
    report.removed.push(kf_idx);
    report.landmarks_retired += removal.landmarks_retired;
    if let (Some(factors), Some(merge)) = (wheel_factors, wheel_merge) {
        factors[merge.incoming] = merge.factor;
        factors.remove(merge.outgoing);
        report.wheel_edges_merged += 1;
    }
    match (stationary_factors, stationary) {
        (Some(factors), StationaryPlan::Compose(merge)) => {
            factors[merge.incoming] = merge.factor;
            factors.remove(merge.outgoing);
            report.stationary_edges_merged += 1;
        }
        // The IMU edge was moved, not merged.
        (Some(factors), StationaryPlan::Boundary { position, .. }) => {
            factors.remove(position);
            report.stationary_boundaries_removed += 1;
        }
        _ => report.imu_edges_merged += usize::from(removal.merged_imu_edge.is_some()),
    }
    Step::Done
}

/// ORB-SLAM's redundancy test for keyframe `kf_idx` against the live map.
fn is_redundant(map: &Map, kf_idx: usize, config: &KeyframeCullingConfig) -> bool {
    let slot_of: HashMap<usize, usize> = map
        .keyframes()
        .iter()
        .enumerate()
        .map(|(slot, kf)| (kf.frame.idx, slot))
        .collect();
    let Some(&slot) = slot_of.get(&kf_idx) else {
        return false;
    };
    let keyframe = &map.keyframes()[slot];
    let octave_in = |other: usize, feature: usize| {
        slot_of
            .get(&other)
            .map(|&s| feature_octave(&map.keyframes()[s], feature))
    };

    // ORB-SLAM3 skips `mvDepth[i] > mThDepth || mvDepth[i] < 0` in stereo.
    let close_depth = config
        .stereo_close_depth_m
        .filter(|_| keyframe.frame.is_stereo());
    let counts = |feature: usize| {
        close_depth.is_none_or(|max| {
            keyframe
                .frame
                .depth
                .get(feature)
                .is_some_and(|&d| d > 0.0 && f64::from(d) <= max)
        })
    };

    let mut live = 0usize;
    let mut redundant = 0usize;
    for (feature, slot) in keyframe.map_point_by_desc_idx.iter().enumerate() {
        let Some(mp) = slot
            .and_then(|id| map.map_points().get(id))
            .filter(|mp| !mp.culled)
        else {
            continue;
        };
        if !counts(feature) {
            continue;
        }
        live += 1;
        let coarsest =
            feature_octave(keyframe, feature).saturating_add(config.scale_level_tolerance);
        let seen_by = mp
            .observations()
            .iter()
            .filter(|o| o.key.keyframe_idx != kf_idx)
            .filter(|o| {
                octave_in(o.key.keyframe_idx, o.key.feature_idx)
                    .is_some_and(|octave| octave <= coarsest)
            })
            .take(config.min_other_observers)
            .count();
        if seen_by >= config.min_other_observers {
            redundant += 1;
        }
    }
    live > 0 && redundant as f64 >= config.redundant_ratio * live as f64
}

fn feature_octave(keyframe: &Keyframe, feature: usize) -> u8 {
    keyframe
        .frame
        .features
        .octaves
        .get(feature)
        .copied()
        .unwrap_or(0)
}

/// Whether the IMU edges through `kf_idx` may be merged under `config`. That
/// they *can* be is [`Map::remove_keyframe`]'s to decide.
fn imu_allows_removal(map: &Map, kf_idx: usize, config: &KeyframeCullingConfig) -> bool {
    let touching = map
        .imu_factors()
        .iter()
        .filter(|f| f.prev_kf_idx == kf_idx || f.curr_kf_idx == kf_idx);
    let (start, end) = touching.fold((f64::INFINITY, f64::NEG_INFINITY), |(s, e), f| {
        (s.min(f.t0), e.max(f.t1))
    });
    if start > end {
        return true;
    }
    config.merge_imu_edges && end - start <= config.max_merged_imu_span_sec
}

fn touches_imu_edge(map: &Map, kf_idx: usize) -> bool {
    map.imu_factors()
        .iter()
        .any(|f| f.prev_kf_idx == kf_idx || f.curr_kf_idx == kf_idx)
}

/// The composite replacing the factors into and out of a keyframe, and the
/// positions of the two it replaces.
struct EdgeMerge<F> {
    incoming: usize,
    outgoing: usize,
    factor: F,
}

type WheelMerge = EdgeMerge<WheelOdomFactor>;

/// What culling does with the stationary factors at a keyframe.
enum StationaryPlan {
    /// None touches it.
    None,
    /// One in and one out, composed into one.
    Compose(EdgeMerge<StationaryFactor>),
    /// One on one side only, at `position`: the keyframe ends the stationary
    /// chain, and [`Map::remove_keyframe_beside_still`] moves its IMU edge
    /// onto `still`.
    Boundary {
        position: usize,
        still: StillNeighbour,
    },
}

/// `Err` when the stationary factors at `kf_idx` are more than one on a side,
/// or do not compose.
fn stationary_plan(factors: &[StationaryFactor], kf_idx: usize) -> Result<StationaryPlan, ()> {
    let positions = |pick: fn(&StationaryFactor) -> usize| -> Vec<usize> {
        factors
            .iter()
            .enumerate()
            .filter(|(_, f)| pick(f) == kf_idx)
            .map(|(pos, _)| pos)
            .collect()
    };
    match (
        positions(|f| f.to_idx).as_slice(),
        positions(|f| f.from_idx).as_slice(),
    ) {
        ([], []) => Ok(StationaryPlan::None),
        (&[incoming], &[outgoing]) => factors[incoming]
            .compose(&factors[outgoing])
            .map(|factor| {
                StationaryPlan::Compose(EdgeMerge {
                    incoming,
                    outgoing,
                    factor,
                })
            })
            .ok_or(()),
        (&[position], []) => Ok(StationaryPlan::Boundary {
            position,
            still: StillNeighbour {
                keyframe_idx: factors[position].from_idx,
                side: StillSide::Before,
                span_s: factors[position].span_s,
            },
        }),
        ([], &[position]) => Ok(StationaryPlan::Boundary {
            position,
            still: StillNeighbour {
                keyframe_idx: factors[position].to_idx,
                side: StillSide::After,
                span_s: factors[position].span_s,
            },
        }),
        _ => Err(()),
    }
}

/// `Ok(None)` when no wheel factor touches `kf_idx`; `Err` when the ones that
/// do are not exactly one in and one out (a factor dropped for slip, or one
/// still pending in the caller), so removing it would cut the chain.
fn wheel_merge_through(
    factors: &[WheelOdomFactor],
    kf_idx: usize,
) -> Result<Option<WheelMerge>, ()> {
    let positions = |pick: fn(&WheelOdomFactor) -> usize| -> Vec<usize> {
        factors
            .iter()
            .enumerate()
            .filter(|(_, f)| pick(f) == kf_idx)
            .map(|(pos, _)| pos)
            .collect()
    };
    match (
        positions(|f| f.to_idx).as_slice(),
        positions(|f| f.from_idx).as_slice(),
    ) {
        ([], []) => Ok(None),
        (&[incoming], &[outgoing]) => factors[incoming]
            .compose(&factors[outgoing])
            .filter(|f| f.from_idx != f.to_idx)
            .map(|factor| {
                Some(EdgeMerge {
                    incoming,
                    outgoing,
                    factor,
                })
            })
            .ok_or(()),
        _ => Err(()),
    }
}
