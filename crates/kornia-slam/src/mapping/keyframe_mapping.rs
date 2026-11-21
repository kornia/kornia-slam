//! Mapping work for an accepted keyframe: its core insertion, and the optional
//! work it earns afterwards.
//!
//! [`keyframe_insertion`] builds the core request — the keyframe, its tracked
//! links, its close stereo seeds and its IMU edge — without touching the map.
//! Admission, publishing that request and the tracker/IMU lifecycle stay with
//! the system. The rest is the sequence mapping owns: which neighbours to grow
//! against, growing each pair, then extending observations into those
//! neighbours. None of it is allowed to undo the accepted keyframe.
//!
//! ## Why a `Mutex` and not `&mut Map`
//!
//! The lock is taken and released once per pair, and once more for fusion.
//! That is deliberate, not incidental: one long exclusive borrow spanning the
//! whole sequence would hold the map against tracking and the local-mapping
//! worker for the entire growth pass. The coordinator therefore takes the
//! shared mutex and keeps the existing boundaries. It never touches the
//! publication gate — the caller already holds it.

use std::sync::{Arc, Mutex};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig};
use kornia_image::Image;
use kornia_imgproc::features::{OrbDetector, OrbMatchConfig};

use crate::mapping::growth;
use crate::mapping::map::{
    ImuFactor, Keyframe, LandmarkTarget, Map, MapInsertion, MapMutationError, ObservationKey,
    ObservationLink,
};
use crate::rig::RigGeometry;
use crate::tracking::optical_flow::Track;

/// The core request for an accepted keyframe, and how many of its landmarks
/// are close stereo seeds.
pub(crate) struct KeyframeInsertion {
    pub insertion: MapInsertion,
    pub close_points: usize,
}

/// A keyframe accepted on its flow pose whose ORB, overlap match, projection
/// search and insert run on the mapping thread (`keyframe_orb: Mapping`).
#[allow(dead_code)] // skeleton: lane T fills the deferred path
pub(crate) struct DeferredKeyframe {
    pub frame_idx: usize,
    pub timestamp_sec: f64,
    pub pose_world_to_cam: Pose3d,
    /// One level-0 plane per rig camera, from a reused pool.
    pub planes: Vec<Image<u8, 1>>,
    pub tracks: Vec<Track>,
    pub tracked_inliers: usize,
    pub matches: Vec<(usize, usize)>,
}

/// What the mapping thread made of a [`DeferredKeyframe`].
#[allow(dead_code)] // skeleton: lane T fills the deferred path
pub(crate) struct DeferredKeyframeResult {
    pub frame_idx: usize,
    pub keyframe_idx: Option<usize>,
    /// Map points created at this keyframe, for `adopt_landmarks`.
    pub new_map_points: Vec<usize>,
    /// The planes, returned to the pool.
    pub planes: Vec<Image<u8, 1>>,
}

/// What the mapping worker needs to finish a deferred keyframe.
#[allow(dead_code)] // skeleton: lane T fills the deferred path
pub(crate) struct DeferredKeyframeContext {
    /// The primary camera's detector (`OrbFrontend::detector`).
    pub detector: OrbDetector,
    /// The other rig cameras' detector (`OrbFrontend::secondary`).
    pub secondary: OrbDetector,
    /// The overlap match reads only the rig (`match_overlap`).
    pub rig: Arc<RigGeometry>,
}

/// Builds the core insertion for `keyframe`, published as one validated batch.
///
/// `tracked` holds `(landmark, feature)` matches from tracking. With
/// `stereo_close_depth` set, close stereo keypoints (`z <= threshold`, metres)
/// become metric landmarks directly; far ones are left to pair growth,
/// mirroring ORB-SLAM3's `CreateNewKeyFrame`.
pub(crate) fn keyframe_insertion(
    keyframe: Keyframe,
    tracked: &[(usize, usize)],
    stereo_close_depth: Option<f64>,
    camera: &PinholeCamera,
    imu_edge: Option<ImuFactor>,
) -> KeyframeInsertion {
    let keyframe_idx = keyframe.frame.idx;
    // Tracked links are recorded as claims first, so stereo seeding cannot
    // take a feature tracking already owns.
    let mut claimed: Vec<Option<usize>> = vec![None; keyframe.frame.features.descriptors.len()];
    let mut insertion = MapInsertion::default();
    // One feature per landmark and one landmark per feature; a repeat would
    // otherwise refuse the whole keyframe.
    // Out-of-range features are dropped *before* resolution: a claim that
    // cannot be published must not win its landmark and suppress a valid
    // later claim on the same one.
    let tracked_claims: Vec<(usize, usize)> = tracked
        .iter()
        .copied()
        .filter(|&(_, feature_idx)| feature_idx < claimed.len())
        .collect();
    for index in growth::accepted_pair_claims(&tracked_claims) {
        let (landmark_idx, feature_idx) = tracked_claims[index];
        claimed[feature_idx] = Some(landmark_idx);
        insertion.observations.push(ObservationLink {
            observation: ObservationKey {
                keyframe_idx,
                feature_idx,
            },
            landmark: LandmarkTarget::Existing(landmark_idx),
        });
    }

    if let Some(threshold) = stereo_close_depth
        && keyframe.frame.is_stereo()
    {
        insertion.landmarks = growth::stereo_seeds(&keyframe.frame, camera, threshold, &claimed);
    } else if let Some(threshold) = stereo_close_depth
        && keyframe.frame.is_rig()
    {
        let (seeds, links) = growth::rig_seeds(&keyframe.frame, threshold, &claimed);
        insertion.landmarks = seeds;
        insertion.observations.extend(links);
    }
    let close_points = insertion.landmarks.len();
    insertion.keyframes.push(keyframe);
    insertion.imu_factors.extend(imu_edge);

    KeyframeInsertion {
        insertion,
        close_points,
    }
}

/// Neighbours captured before the core keyframe is published.
///
/// Not a queued, independently valid snapshot: it names keyframes by id and is
/// consumed during the same publication interval that produced it.
pub(crate) struct KeyframeGrowthPlan {
    keyframe_idx: usize,
    /// Neighbours the keyframe triangulates new landmarks against.
    triangulation_neighbours: Vec<usize>,
    /// Neighbours its landmarks are fused into; equal to the above under
    /// [`GrowthNeighbours::RecentTen`].
    fusion_neighbours: Vec<usize>,
}

/// Which keyframes a new keyframe grows and fuses against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GrowthNeighbours {
    /// The 10 most recent keyframes, for triangulation and fusion alike.
    #[default]
    RecentTen,
    /// ORB-SLAM3's lists: `CreateNewMapPoints` (best covisible, then the
    /// previous-keyframe chain) and `SearchInNeighbors` (best covisible, their
    /// best 20, then temporal keyframes); `LocalMapping.cc:388-407, 714-763`.
    /// `covisible` is 10 (30 monocular).
    OrbSlam3 { covisible: usize },
}

/// The velocity a new keyframe enters its first VI-BA with (card R).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframeSeed {
    /// The tracker's velocity: the reference keyframe's plus one frame of IMU.
    #[default]
    ReferenceVelocity,
    /// The last keyframe's state propagated over the new keyframe edge
    /// (ORB-SLAM3 `Tracking.cc:1746-1762`).
    ImuPropagated,
}

/// A pair whose growth batch was refused, with the neighbour it was against.
pub(crate) struct PairGrowthFailure {
    pub neighbor_keyframe_idx: usize,
    pub error: MapMutationError,
}

/// What the growth pass actually committed.
///
/// `landmarks_added` counts landmarks published by pair growth;
/// `observations_added` counts links fusion newly created. A redundant fusion
/// proposal — one naming a link that already exists — is neither, because it
/// committed nothing. Message formatting stays with the caller.
#[derive(Default)]
pub(crate) struct KeyframeGrowthResult {
    pub neighbor_count: usize,
    pub landmarks_added: usize,
    pub observations_added: usize,
    pub pair_failures: Vec<PairGrowthFailure>,
    pub fusion_failures: Vec<MapMutationError>,
}

/// Maximum neighbours one keyframe grows against.
///
/// ORB-SLAM3's `CreateNewMapPoints` takes the 10 best covisible keyframes (30
/// in monocular); this takes the 10 most recent instead.
const MAX_COVIS_KFS: usize = 10;

/// Picks the neighbours for `keyframe_idx` by recency.
///
/// Must be called before the core keyframe is published: growing against a
/// list that already contained it would triangulate it against itself and drop
/// the oldest real neighbour. The id is filtered anyway, so the policy states
/// that intent rather than relying on call order alone.
pub(crate) fn prepare_keyframe_growth(
    map: &Map,
    keyframe_idx: usize,
    rule: GrowthNeighbours,
) -> KeyframeGrowthPlan {
    match rule {
        GrowthNeighbours::RecentTen => {
            let neighbours: Vec<usize> = map
                .keyframes()
                .iter()
                .rev()
                .map(|kf| kf.frame.idx)
                .filter(|&idx| idx != keyframe_idx)
                .take(MAX_COVIS_KFS)
                .collect();
            KeyframeGrowthPlan {
                keyframe_idx,
                fusion_neighbours: neighbours.clone(),
                triangulation_neighbours: neighbours,
            }
        }
        GrowthNeighbours::OrbSlam3 { covisible } => orb_slam3_growth(map, keyframe_idx, covisible),
    }
}

/// ORB-SLAM3 links keyframes sharing at least 15 landmarks
/// (`KeyFrame::UpdateConnections`).
const MIN_COVISIBILITY_WEIGHT: usize = 15;
/// Second-order fusion neighbours per target, and the size the temporal
/// keyframes fill the fusion list to (`LocalMapping.cc:741-763`).
const FUSION_TARGETS: usize = 20;

/// ORB-SLAM3's lists, triangulation from `CreateNewMapPoints`
/// (`LocalMapping.cc:388-407`) and fusion from `SearchInNeighbors`
/// (`:714-763`). The new keyframe is not in the map yet, so its covisibility
/// stands in as the previous keyframe's: that keyframe, then its best
/// covisible neighbours.
fn orb_slam3_growth(map: &Map, keyframe_idx: usize, covisible: usize) -> KeyframeGrowthPlan {
    // Newest first: ORB-SLAM3's `mPrevKF` chain.
    let recent: Vec<usize> = map
        .keyframes()
        .iter()
        .rev()
        .map(|kf| kf.frame.idx)
        .filter(|&idx| idx != keyframe_idx)
        .collect();
    let best_covisible = |kf_idx: usize, n: usize| -> Vec<usize> {
        map.covisible_keyframes_min_weight(kf_idx, MIN_COVISIBILITY_WEIGHT)
            .into_iter()
            .map(|(idx, _)| idx)
            .filter(|&idx| idx != keyframe_idx)
            .take(n)
            .collect()
    };
    let mut nearest = Vec::new();
    if let Some(&previous) = recent.first() {
        nearest.push(previous);
        extend_unique(&mut nearest, best_covisible(previous, covisible));
        nearest.truncate(covisible);
    }

    let mut triangulation_neighbours = nearest.clone();
    for &idx in recent.iter().take(covisible) {
        if triangulation_neighbours.len() > covisible {
            break;
        }
        extend_unique(&mut triangulation_neighbours, [idx]);
    }

    let mut fusion_neighbours = nearest.clone();
    for &target in &nearest {
        extend_unique(
            &mut fusion_neighbours,
            best_covisible(target, FUSION_TARGETS),
        );
    }
    for &idx in &recent {
        if fusion_neighbours.len() >= FUSION_TARGETS {
            break;
        }
        extend_unique(&mut fusion_neighbours, [idx]);
    }

    KeyframeGrowthPlan {
        keyframe_idx,
        triangulation_neighbours,
        fusion_neighbours,
    }
}

/// Appends each of `ids` not in `list` yet, in order.
fn extend_unique(list: &mut Vec<usize>, ids: impl IntoIterator<Item = usize>) {
    for id in ids {
        if !list.contains(&id) {
            list.push(id);
        }
    }
}

/// [`prepare_keyframe_growth`] for a re-seeded segment still inside its own
/// window: both lists keep only keyframes at or after `window_start_kf`.
pub(crate) fn prepare_seeded_window_growth(
    map: &Map,
    keyframe_idx: usize,
    rule: GrowthNeighbours,
    window_start_kf: usize,
) -> KeyframeGrowthPlan {
    let mut plan = prepare_keyframe_growth(map, keyframe_idx, rule);
    plan.triangulation_neighbours
        .retain(|&idx| idx >= window_start_kf);
    plan.fusion_neighbours.retain(|&idx| idx >= window_start_kf);
    plan
}

/// Grows the accepted keyframe against its captured neighbours, then fuses its
/// landmarks forward into them.
///
/// Each pair is prepared and published under its own lock, so a later pair sees
/// the feature claims an earlier one took. Every pair is optional: a proposal
/// that does not materialize is a skip, and a refused batch is reported without
/// disturbing the accepted keyframe or any earlier successful pair.
pub(crate) fn grow_keyframe(
    map: &Mutex<Map>,
    plan: KeyframeGrowthPlan,
    camera: &PinholeCamera,
    match_config: OrbMatchConfig,
    triangulation_config: &TriangulationConfig,
    min_baseline_ratio: f64,
) -> KeyframeGrowthResult {
    let mut result = KeyframeGrowthResult {
        neighbor_count: plan.triangulation_neighbours.len(),
        ..Default::default()
    };

    for &neighbor_keyframe_idx in &plan.triangulation_neighbours {
        let mut guard = map.lock().unwrap();
        let Some(request) = growth::pair_growth_request(
            &guard,
            neighbor_keyframe_idx,
            plan.keyframe_idx,
            match_config,
            triangulation_config,
            camera,
            min_baseline_ratio,
        ) else {
            continue;
        };
        let outcome = guard.apply_insertion(request);
        // Released before the next pair is prepared.
        drop(guard);
        match outcome {
            Ok(insertion) => result.landmarks_added += insertion.landmark_ids.len(),
            Err(error) => result.pair_failures.push(PairGrowthFailure {
                neighbor_keyframe_idx,
                error,
            }),
        }
    }

    // Forward SearchInNeighbors, one neighbour at a time under one lock. Each
    // neighbour's proposals are resolved against the map as the previous
    // neighbours left it: a link refreshes the landmark's representative
    // descriptor, and the next neighbour's search must match against that,
    // not against the descriptor it had before fusion began.
    let mut guard = map.lock().unwrap();
    for &neighbor_keyframe_idx in &plan.fusion_neighbours {
        let links = growth::neighbor_fusion_links(
            &guard,
            plan.keyframe_idx,
            &[neighbor_keyframe_idx],
            camera,
        );
        for link in links {
            let LandmarkTarget::Existing(landmark) = link.landmark else {
                continue;
            };
            match guard.link_observation(
                link.observation.keyframe_idx,
                link.observation.feature_idx,
                landmark,
            ) {
                Ok(true) => result.observations_added += 1,
                // Already linked: the proposal was redundant, not wrong.
                Ok(false) => {}
                // Proposals are resolved against a map held under this same
                // lock, so a refusal means an invariant we believed held did
                // not. Surface it rather than counting it as a no-op.
                Err(error) => result.fusion_failures.push(error),
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn stereo_keyframe(idx: usize, depths: Vec<f32>) -> Keyframe {
        let mut frame = empty_frame(idx);
        let n = depths.len();
        frame.features.keypoints_xy = vec![[320.0, 240.0]; n];
        frame.features.orientations = vec![0.0; n];
        frame.features.descriptors = vec![[0; 32]; n];
        frame.features.octaves = vec![0; n];
        frame.depth = depths;
        Keyframe::from_frame(frame)
    }

    /// Tracked links claim their features first, so a close stereo point on a
    /// tracked feature is not seeded twice; far points are left to growth.
    #[test]
    fn keyframe_insertion_seeds_only_close_unclaimed_stereo_points() {
        let keyframe = stereo_keyframe(9, vec![2.0, 2.0, 30.0, 2.0]);
        // Landmark 5 is claimed twice and feature 99 is out of range.
        let tracked = [(5, 0), (5, 3), (6, 99)];

        let KeyframeInsertion {
            insertion,
            close_points,
        } = keyframe_insertion(keyframe, &tracked, Some(10.0), &camera(), None);

        assert_eq!(insertion.observations.len(), 1);
        assert_eq!(insertion.observations[0].observation.feature_idx, 0);
        assert!(matches!(
            insertion.observations[0].landmark,
            LandmarkTarget::Existing(5)
        ));
        let seeded: Vec<usize> = insertion
            .landmarks
            .iter()
            .map(|seed| seed.reference.feature_idx)
            .collect();
        assert_eq!(seeded, vec![1, 3]);
        assert_eq!(close_points, 2);
        assert_eq!(insertion.keyframes.len(), 1);
        assert!(insertion.imu_factors.is_empty());
    }

    #[test]
    fn keyframe_insertion_skips_stereo_seeds_when_densification_is_off() {
        let keyframe = stereo_keyframe(9, vec![2.0, 2.0]);
        let KeyframeInsertion { close_points, .. } =
            keyframe_insertion(keyframe, &[], None, &camera(), None);
        assert_eq!(close_points, 0);
    }

    fn camera() -> PinholeCamera {
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

    /// A 4x5 grid of landmarks five metres out, the same geometry
    /// `growth::tests` uses for a well-conditioned pair.
    fn grid() -> Vec<Vec3F64> {
        (0..20)
            .map(|i| Vec3F64::new((i % 5) as f64 * 0.3 - 0.6, (i / 5) as f64 * 0.3 - 0.45, 5.0))
            .collect()
    }

    fn empty_frame(idx: usize) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
            rig: Default::default(),
        }
    }

    /// A frame observing `points` from a camera centre at `+x_offset`.
    fn observing_frame(idx: usize, x_offset: f64, points: &[Vec3F64]) -> Frame {
        let camera = camera();
        let n = points.len();
        let pose = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-x_offset, 0.0, 0.0),
        );
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: points
                    .iter()
                    .map(|p| {
                        let p = pose.transform_point(p);
                        [
                            (camera.fx * p.x / p.z + camera.cx) as f32,
                            (camera.fy * p.y / p.z + camera.cy) as f32,
                        ]
                    })
                    .collect(),
                orientations: vec![0.0; n],
                descriptors: (0..n).map(|i| [i as u8; 32]).collect(),
                octaves: vec![0; n],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
            rig: Default::default(),
        }
    }

    fn grow(map: &Mutex<Map>, plan: KeyframeGrowthPlan) -> KeyframeGrowthResult {
        grow_keyframe(
            map,
            plan,
            &camera(),
            OrbMatchConfig::default(),
            &TriangulationConfig::default(),
            0.0,
        )
    }

    /// Recency, capped, most recent first — and never the keyframe being grown,
    /// with ids that are neither contiguous nor increasing.
    #[test]
    fn neighbours_are_the_most_recent_ten_in_reverse_insertion_order() {
        let mut map = Map::new();
        let inserted = [70, 3, 41, 5, 900, 12, 64, 7, 55, 2, 88, 31];
        for idx in inserted {
            map.insert_keyframe(Keyframe::from_frame(empty_frame(idx)))
                .unwrap();
        }

        let plan = prepare_keyframe_growth(&map, 999, GrowthNeighbours::RecentTen);

        assert_eq!(
            plan.triangulation_neighbours,
            vec![31, 88, 2, 55, 7, 64, 12, 900, 5, 41],
            "the ten most recently inserted, newest first"
        );
        assert_eq!(plan.keyframe_idx, 999);
    }

    /// Called before publication, the plan cannot contain the new keyframe; the
    /// filter states that intent, so a later call site cannot reintroduce it.
    #[test]
    fn the_grown_keyframe_is_never_its_own_neighbour() {
        let mut map = Map::new();
        for idx in [4, 9] {
            map.insert_keyframe(Keyframe::from_frame(empty_frame(idx)))
                .unwrap();
        }

        let plan = prepare_keyframe_growth(&map, 9, GrowthNeighbours::RecentTen);

        assert_eq!(plan.triangulation_neighbours, vec![4]);
    }

    /// Without covisibility the ORB-SLAM3 lists are the previous keyframe,
    /// then the temporal chain: `covisible + 1` at most for triangulation,
    /// up to 20 for fusion; never the grown keyframe.
    #[test]
    fn orb_slam3_neighbours_fall_back_to_the_temporal_chain() {
        let mut map = Map::new();
        for idx in 1..=6 {
            map.insert_keyframe(Keyframe::from_frame(empty_frame(idx)))
                .unwrap();
        }

        let plan = prepare_keyframe_growth(&map, 6, GrowthNeighbours::OrbSlam3 { covisible: 2 });

        assert_eq!(plan.triangulation_neighbours, vec![5, 4]);
        assert_eq!(plan.fusion_neighbours, vec![5, 4, 3, 2, 1]);
    }

    /// Each pair is published before the next is prepared, so the second
    /// neighbour sees the features the first already claimed and cannot
    /// triangulate a second landmark onto them.
    #[test]
    fn a_published_pair_is_visible_to_the_next_pair() {
        let points = grid();
        let mut map = Map::new();
        for (idx, offset) in [(10, 0.0), (20, 1.0)] {
            map.insert_keyframe(Keyframe::from_frame(observing_frame(idx, offset, &points)))
                .unwrap();
        }
        let plan = prepare_keyframe_growth(&map, 30, GrowthNeighbours::RecentTen);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        assert_eq!(plan.triangulation_neighbours, vec![20, 10]);
        let map = Mutex::new(map);

        let result = grow(&map, plan);

        assert_eq!(result.neighbor_count, 2);
        assert!(result.pair_failures.is_empty());
        assert_eq!(
            result.landmarks_added,
            points.len(),
            "the second pair must not duplicate landmarks onto claimed features"
        );
        let map = map.lock().unwrap();
        let grown = map.get_keyframe(30).unwrap();
        for feature in 0..points.len() {
            assert!(
                grown.map_point(feature).is_some(),
                "feature {feature} should hold exactly one landmark"
            );
        }
    }

    /// A neighbour with nothing in common is a skip, not a failure, and the
    /// useful neighbour after it still runs.
    #[test]
    fn a_skipped_pair_does_not_stop_later_neighbours() {
        let points = grid();
        let mut map = Map::new();
        // Insertion order puts the useful neighbour first, so that reversing it
        // for recency visits the *skipped* one first: a skip that wrongly ended
        // the loop would then cost the useful pair, and this test would fail.
        map.insert_keyframe(Keyframe::from_frame(observing_frame(20, 1.0, &points)))
            .unwrap();
        // Shares no features: the pair has nothing to propose.
        map.insert_keyframe(Keyframe::from_frame(empty_frame(10)))
            .unwrap();
        let plan = prepare_keyframe_growth(&map, 30, GrowthNeighbours::RecentTen);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        assert_eq!(
            plan.triangulation_neighbours,
            vec![10, 20],
            "the skipped neighbour must be visited before the useful one"
        );
        let map = Mutex::new(map);

        let result = grow(&map, plan);

        assert_eq!(result.neighbor_count, 2);
        assert_eq!(
            result.landmarks_added,
            points.len(),
            "the neighbour after the skipped one still produced landmarks"
        );
        assert!(
            result.pair_failures.is_empty(),
            "a pair with nothing to propose is skipped, not refused"
        );
        // The accepted keyframe is untouched by the skip.
        assert!(map.lock().unwrap().get_keyframe(30).is_some());
    }

    /// Keyframe 30 is 1 m from keyframe 10 and holds one landmark 5 m deep, so
    /// its pair's baseline/depth is 0.2: a 0.25 ratio skips it, 0.15 grows it.
    #[test]
    fn the_baseline_ratio_reaches_each_pair() {
        let added = |ratio: f64| {
            let mut points = grid();
            let mut map = Map::new();
            map.insert_keyframe(Keyframe::from_frame(observing_frame(10, 0.0, &points)))
                .unwrap();
            points.push(Vec3F64::new(1.0, 0.0, 5.0));
            map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 1.0, &points)))
                .unwrap();
            map.insert_landmark(LandmarkSeed {
                position: points[20],
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 30,
                    feature_idx: 20,
                },
            })
            .unwrap();
            let plan = KeyframeGrowthPlan {
                keyframe_idx: 30,
                triangulation_neighbours: vec![10],
                fusion_neighbours: vec![],
            };
            grow_keyframe(
                &Mutex::new(map),
                plan,
                &camera(),
                OrbMatchConfig::default(),
                &TriangulationConfig::default(),
                ratio,
            )
            .landmarks_added
        };
        assert_eq!(added(0.25), 0);
        assert_eq!(added(0.15), 20);
    }

    /// Fusion runs after growth and over the same neighbour list, so it sees
    /// the landmarks growth just created and can extend them into a neighbour
    /// that does not observe them yet.
    #[test]
    fn fusion_extends_landmarks_created_by_the_preceding_growth() {
        let points = grid();
        let mut map = Map::new();
        // Observes the grid, but is not part of the pair that triangulates it.
        map.insert_keyframe(Keyframe::from_frame(observing_frame(10, 0.5, &points)))
            .unwrap();
        map.insert_keyframe(Keyframe::from_frame(observing_frame(20, 1.0, &points)))
            .unwrap();
        let plan = prepare_keyframe_growth(&map, 30, GrowthNeighbours::RecentTen);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        let map = Mutex::new(map);

        let result = grow(&map, plan);

        assert!(result.landmarks_added > 0, "growth produced landmarks");
        assert!(
            result.observations_added > 0,
            "fusion linked those landmarks into a neighbour"
        );
        assert!(result.fusion_failures.is_empty());

        let map = map.lock().unwrap();
        // Every committed link is recorded on both sides.
        for keyframe in map.keyframes() {
            for (feature, slot) in keyframe.map_point_by_desc_idx.iter().enumerate() {
                let Some(landmark) = *slot else { continue };
                assert!(
                    map.map_points()[landmark]
                        .observations()
                        .iter()
                        .any(|o| o.key.keyframe_idx == keyframe.frame.idx
                            && o.key.feature_idx == feature),
                    "kf {} feature {feature} has no matching record",
                    keyframe.frame.idx
                );
            }
        }
    }

    /// With nothing left to propose, the pass commits nothing and leaves the
    /// keyframe and its existing associations exactly as they were.
    #[test]
    fn no_useful_work_leaves_the_keyframe_and_its_links_intact() {
        let points = grid();
        let mut map = Map::new();
        for (idx, offset) in [(10, 0.0), (20, 1.0)] {
            map.insert_keyframe(Keyframe::from_frame(observing_frame(idx, offset, &points)))
                .unwrap();
        }
        let first_plan = prepare_keyframe_growth(&map, 30, GrowthNeighbours::RecentTen);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        let map = Mutex::new(map);
        let first = grow(&map, first_plan);
        assert!(first.landmarks_added > 0);

        let second_plan =
            prepare_keyframe_growth(&map.lock().unwrap(), 30, GrowthNeighbours::RecentTen);
        let before = map.lock().unwrap().state_fingerprint_for_test();

        let result = grow(&map, second_plan);

        assert_eq!(result.landmarks_added, 0);
        assert_eq!(result.observations_added, 0);
        assert!(result.pair_failures.is_empty());
        assert!(result.fusion_failures.is_empty());
        assert_eq!(
            map.lock().unwrap().state_fingerprint_for_test(),
            before,
            "a pass with nothing to do changed the map"
        );
    }

    #[test]
    fn a_seeded_window_grows_against_its_own_keyframes_only() {
        let mut map = Map::new();
        for idx in 0..13 {
            map.insert_keyframe(Keyframe::from_frame(empty_frame(idx)))
                .unwrap();
        }
        let open = prepare_keyframe_growth(&map, 13, GrowthNeighbours::RecentTen);
        assert_eq!(
            open.triangulation_neighbours,
            (3..13).rev().collect::<Vec<_>>()
        );

        let plan = prepare_seeded_window_growth(&map, 13, GrowthNeighbours::RecentTen, 8);
        let window: Vec<usize> = (8..13).rev().collect();
        assert_eq!(plan.keyframe_idx, 13);
        assert_eq!(plan.triangulation_neighbours, window);
        assert_eq!(plan.fusion_neighbours, window);

        let unbounded = prepare_seeded_window_growth(&map, 13, GrowthNeighbours::RecentTen, 0);
        assert_eq!(
            unbounded.triangulation_neighbours,
            open.triangulation_neighbours
        );
        assert_eq!(unbounded.fusion_neighbours, open.fusion_neighbours);
    }
}
