//! Coordination of the optional mapping work an accepted keyframe earns.
//!
//! Admission, core publication and the tracker/IMU lifecycle stay with the
//! system. What belongs here is the sequence mapping owns: which neighbours to
//! grow against, growing each pair, then extending observations into those
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

use std::sync::Mutex;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::TriangulationConfig;
use kornia_imgproc::features::OrbMatchConfig;

use crate::map::{LandmarkTarget, Map, MapMutationError};
use crate::mapping::growth;

/// Neighbours captured before the core keyframe is published.
///
/// Not a queued, independently valid snapshot: it names keyframes by id and is
/// consumed during the same publication interval that produced it.
pub(crate) struct KeyframeGrowthPlan {
    keyframe_idx: usize,
    neighbor_keyframe_indices: Vec<usize>,
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
/// Mirrors ORB-SLAM3's `CreateNewMapPoints`, which uses the 30 best covisible
/// keyframes; recency approximates covisibility until the graph is available.
const MAX_COVIS_KFS: usize = 10;

/// Picks the neighbours for `keyframe_idx` by recency.
///
/// Must be called before the core keyframe is published: growing against a
/// list that already contained it would triangulate it against itself and drop
/// the oldest real neighbour. The id is filtered anyway, so the policy states
/// that intent rather than relying on call order alone.
pub(crate) fn prepare_keyframe_growth(map: &Map, keyframe_idx: usize) -> KeyframeGrowthPlan {
    let neighbor_keyframe_indices = map
        .keyframes()
        .iter()
        .rev()
        .map(|kf| kf.frame.idx)
        .filter(|&idx| idx != keyframe_idx)
        .take(MAX_COVIS_KFS)
        .collect();
    KeyframeGrowthPlan {
        keyframe_idx,
        neighbor_keyframe_indices,
    }
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
) -> KeyframeGrowthResult {
    let mut result = KeyframeGrowthResult {
        neighbor_count: plan.neighbor_keyframe_indices.len(),
        ..Default::default()
    };

    for &neighbor_keyframe_idx in &plan.neighbor_keyframe_indices {
        let mut guard = map.lock().unwrap();
        let Some(request) = growth::pair_growth_request(
            &guard,
            neighbor_keyframe_idx,
            plan.keyframe_idx,
            match_config,
            triangulation_config,
            camera,
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

    // Forward SearchInNeighbors. Proposals are generated and published under one
    // lock: they are resolved against a specific map state, so nothing may
    // change between resolving them and applying them.
    let mut guard = map.lock().unwrap();
    let links = growth::neighbor_fusion_links(
        &guard,
        plan.keyframe_idx,
        &plan.neighbor_keyframe_indices,
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
            // Proposals are resolved against a map held under this same lock, so
            // a refusal means an invariant we believed held did not. Surface it
            // rather than counting it as a no-op.
            Err(error) => result.fusion_failures.push(error),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::map::Keyframe;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

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
        }
    }

    fn grow(map: &Mutex<Map>, plan: KeyframeGrowthPlan) -> KeyframeGrowthResult {
        grow_keyframe(
            map,
            plan,
            &camera(),
            OrbMatchConfig::default(),
            &TriangulationConfig::default(),
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

        let plan = prepare_keyframe_growth(&map, 999);

        assert_eq!(
            plan.neighbor_keyframe_indices,
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

        let plan = prepare_keyframe_growth(&map, 9);

        assert_eq!(plan.neighbor_keyframe_indices, vec![4]);
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
        let plan = prepare_keyframe_growth(&map, 30);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        assert_eq!(plan.neighbor_keyframe_indices, vec![20, 10]);
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
        let plan = prepare_keyframe_growth(&map, 30);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        assert_eq!(
            plan.neighbor_keyframe_indices,
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
        let plan = prepare_keyframe_growth(&map, 30);
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
        let first_plan = prepare_keyframe_growth(&map, 30);
        map.insert_keyframe(Keyframe::from_frame(observing_frame(30, 2.0, &points)))
            .unwrap();
        let map = Mutex::new(map);
        let first = grow(&map, first_plan);
        assert!(first.landmarks_added > 0);

        let second_plan = prepare_keyframe_growth(&map.lock().unwrap(), 30);
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
}
