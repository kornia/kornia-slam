//! Landmark and keyframe culling policy.
//!
//! Selection lives here because the thresholds are a mapping judgement about
//! which landmarks and keyframes have earned their place. Removal itself is
//! the map's canonical operation, so this selects ids and delegates — there is
//! no second cleanup loop, and the map never calls back into mapping.

use crate::mapping::map::{Keyframe, Map};
use crate::mapping::wheel_odom::WheelOdomFactor;
use std::collections::{HashMap, HashSet};

/// Minimum times a landmark must have been in view before its match rate is
/// judged at all.
const MIN_OBSERVATIONS: u32 = 5;
/// Match rate below which a sufficiently-observed landmark is dropped.
const MIN_FOUND_RATIO: f64 = 0.20;

/// Ids of the landmarks that fail the culling policy.
///
/// Two criteria, as before: a poor found ratio once a landmark has been seen
/// enough times to judge, and any landmark sitting behind a keyframe that
/// observes it.
fn select_for_culling(map: &Map) -> Vec<usize> {
    let mut selected: HashSet<usize> = HashSet::new();

    for (idx, mp) in map.map_points().iter().enumerate() {
        if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
            continue;
        }
        if mp.found_ratio() < MIN_FOUND_RATIO {
            selected.insert(idx);
        }
    }

    for kf in map.keyframes() {
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if let Some(mp) = map.map_points().get(*mp_idx)
                && !mp.culled
            {
                let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                if p_cam.z <= 1e-8 {
                    selected.insert(*mp_idx);
                }
            }
        }
    }

    let mut selected: Vec<usize> = selected.into_iter().collect();
    selected.sort_unstable();
    selected
}

/// Culls landmarks that fail the policy. Returns how many were retired.
///
/// Held under exclusive access so selection and removal see one state.
pub fn cull_landmarks(map: &mut Map) -> usize {
    let mut retired = 0usize;
    for idx in select_for_culling(map) {
        if map.remove_landmark(idx).unwrap_or(false) {
            retired += 1;
        }
    }
    retired
}

/// Redundant-keyframe culling (ORB-SLAM's local-mapping `KeyFrameCulling`).
///
/// Off unless the caller runs [`cull_redundant_keyframes`]; the defaults are
/// ORB-SLAM3's where it has one.
#[derive(Debug, Clone)]
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
    /// Redundant keyframes kept because their IMU or wheel edges could not be
    /// merged, or the merged IMU edge would have been too long.
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
/// `Frame::idx` as [`super::bundle_adjustment::run_local_inertial_ba`] takes
/// them, have the pair through it composed into one. A keyframe either chain
/// cannot be merged across is kept, so no edge is ever cut.
pub fn cull_redundant_keyframes(
    map: &mut Map,
    config: &KeyframeCullingConfig,
    mut wheel_factors: Option<&mut Vec<WheelOdomFactor>>,
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

    for kf_idx in candidates {
        if report.removed.len() >= config.max_removals
            || map.keyframes().len() <= config.min_keyframes
        {
            break;
        }
        if !is_redundant(map, kf_idx, config) {
            continue;
        }
        let wheel_merge = match wheel_factors.as_deref() {
            Some(factors) => match wheel_merge_through(factors, kf_idx) {
                Ok(merge) => merge,
                Err(()) => {
                    report.kept_for_edges += 1;
                    continue;
                }
            },
            None => None,
        };
        if !imu_allows_removal(map, kf_idx, config) {
            report.kept_for_edges += 1;
            continue;
        }
        let Ok(removal) = map.remove_keyframe(kf_idx) else {
            report.kept_for_edges += 1;
            continue;
        };
        report.removed.push(kf_idx);
        report.landmarks_retired += removal.landmarks_retired;
        report.imu_edges_merged += usize::from(removal.merged_imu_edge.is_some());
        if let (Some(factors), Some(merge)) = (wheel_factors.as_deref_mut(), wheel_merge) {
            factors[merge.incoming] = merge.factor;
            factors.remove(merge.outgoing);
            report.wheel_edges_merged += 1;
        }
    }
    report
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

/// The composite replacing the wheel factors into and out of a keyframe, and
/// the positions of the two it replaces.
struct WheelMerge {
    incoming: usize,
    outgoing: usize,
    factor: WheelOdomFactor,
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
                Some(WheelMerge {
                    incoming,
                    outgoing,
                    factor,
                })
            })
            .ok_or(()),
        _ => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    #[test]
    fn culls_low_found_ratio_and_clears_its_associations() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        let seed = |feature| LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: feature,
            },
        };
        let doomed = map.insert_landmark(seed(0)).unwrap();
        let kept = map.insert_landmark(seed(1)).unwrap();

        map.set_tracking_stats_for_test(doomed, 10, 1);
        map.set_tracking_stats_for_test(kept, 10, 5);

        assert_eq!(cull_landmarks(&mut map), 1);
        assert!(map.map_points()[doomed].culled);
        assert!(!map.map_points()[kept].culled);
        // Removal cleared the association; no second cleanup pass needed.
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(kept));
        // Retirement is logical: the landmark keeps its records.
        assert!(map.map_points()[doomed].is_observed_by(0));
    }

    #[test]
    fn spares_a_landmark_with_too_few_observations_to_judge() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        let idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        map.set_tracking_stats_for_test(idx, 4, 0);

        assert_eq!(cull_landmarks(&mut map), 0);
        assert!(!map.map_points()[idx].culled);
    }
}

#[cfg(test)]
#[path = "keyframe_culling_tests.rs"]
mod keyframe_tests;
