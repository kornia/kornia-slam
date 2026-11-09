//! Loop fusion planned off the tracking thread and applied at the frame boundary.
//!
//! As in ORB-SLAM3's `SearchAndFuse`, the search only needs the loop neighbourhoods at their
//! corrected poses, so the worker runs it on a copy of them taken with the solve's snapshot,
//! moved by the solve. The copy keeps its own associations, so later proposals see earlier links
//! and merges as [`fuse_verified_loop`] does. The search returns a list of operations. The
//! writeback re-checks each one against the live map and skips the stale ones.
//!
//! [`fuse_verified_loop`]: crate::loop_closure::fuse_verified_loop

use std::collections::{HashMap, HashSet};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;

use crate::loop_closure::fusion::loop_side_keyframes;
use crate::loop_closure::{LoopFusionConfig, PgoResult, VerifiedLoopEdge};
use crate::mapping::map::{Map, ObservationKey};

/// One association change found by the fusion search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FusionOp {
    /// Link a free feature of `keyframe` to `landmark`.
    Link {
        keyframe: usize,
        feature: usize,
        landmark: usize,
    },
    /// Merge the landmark in `target_keyframe`'s `target_feature` into the projected `source`
    /// landmark, which `source_keyframe` sees through `source_feature`.
    Merge {
        source: usize,
        source_keyframe: usize,
        source_feature: usize,
        target: usize,
        target_keyframe: usize,
        target_feature: usize,
    },
}

/// What [`apply_fusion`] changed, and how many operations the map no longer allowed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FusionApplied {
    pub linked: usize,
    pub merged: usize,
    pub stale: usize,
}

struct SnapshotKeyframe {
    pose: Pose3d,
    image_size: ImageSize,
    keypoints: Vec<Option<[f32; 2]>>,
    descriptors: Vec<[u8; 32]>,
    slots: Vec<Option<usize>>,
}

struct SnapshotLandmark {
    position: Vec3F64,
    descriptor: [u8; 32],
    reference_keyframe: usize,
    observations: Vec<ObservationKey>,
    n_found: u32,
    culled: bool,
}

impl SnapshotLandmark {
    fn is_observed_by(&self, keyframe: usize) -> bool {
        self.observations
            .iter()
            .any(|key| key.keyframe_idx == keyframe)
    }
}

/// The loop neighbourhoods of the loops a solve reports, as the map held them at submit time.
pub(crate) struct FusionSnapshot {
    camera: PinholeCamera,
    config: LoopFusionConfig,
    /// The (query side, candidate side) of each loop, in report order.
    sides: Vec<(Vec<usize>, Vec<usize>)>,
    keyframes: HashMap<usize, SnapshotKeyframe>,
    landmarks: HashMap<usize, SnapshotLandmark>,
}

impl FusionSnapshot {
    /// Copies both sides of every loop whose ends are live, as [`fuse_verified_loop`] picks them.
    ///
    /// [`fuse_verified_loop`]: crate::loop_closure::fuse_verified_loop
    pub(crate) fn capture(
        map: &Map,
        camera: &PinholeCamera,
        loops: &[VerifiedLoopEdge],
        config: &LoopFusionConfig,
    ) -> Self {
        let mut snapshot = Self {
            camera: camera.clone(),
            config: config.clone(),
            sides: Vec::with_capacity(loops.len()),
            keyframes: HashMap::new(),
            landmarks: HashMap::new(),
        };
        for edge in loops {
            if map.get_keyframe(edge.query_kf_idx).is_none()
                || map.get_keyframe(edge.candidate_kf_idx).is_none()
            {
                continue;
            }
            let query = loop_side_keyframes(map, edge.query_kf_idx, config);
            let candidate = loop_side_keyframes(map, edge.candidate_kf_idx, config);
            for &kf in query.iter().chain(&candidate) {
                snapshot.copy_keyframe(map, kf);
            }
            snapshot.sides.push((query, candidate));
        }
        snapshot
    }

    fn copy_keyframe(&mut self, map: &Map, kf: usize) {
        if self.keyframes.contains_key(&kf) {
            return;
        }
        let Some(keyframe) = map.get_keyframe(kf) else {
            return;
        };
        let features = 0..keyframe.frame.features.descriptors.len();
        self.keyframes.insert(
            kf,
            SnapshotKeyframe {
                pose: keyframe.frame.pose_world_to_cam,
                image_size: keyframe.frame.image_size,
                keypoints: features
                    .map(|feature| keyframe.frame.undistorted_xy(feature, &self.camera))
                    .collect(),
                descriptors: keyframe.frame.features.descriptors.clone(),
                slots: keyframe.map_point_by_desc_idx.clone(),
            },
        );
        for &landmark in keyframe.map_point_by_desc_idx.iter().flatten() {
            let Some(point) = map.map_points().get(landmark) else {
                continue;
            };
            self.landmarks
                .entry(landmark)
                .or_insert_with(|| SnapshotLandmark {
                    position: point.position,
                    descriptor: point.descriptor,
                    reference_keyframe: point.keyframe_idx,
                    observations: point.observations().iter().map(|o| o.key).collect(),
                    n_found: point.n_found,
                    culled: point.culled,
                });
        }
    }

    /// Moves the copy by `solved` as [`Map::apply_graph_correction`] moves the map, then searches
    /// every loop in both directions, candidate side into query side first.
    pub(crate) fn plan(mut self, solved: &PgoResult) -> Vec<FusionOp> {
        let corrections: HashMap<usize, (Pose3d, Pose3d)> = solved
            .keyframe_indices
            .iter()
            .zip(solved.original_poses.iter().zip(&solved.optimized_poses))
            .map(|(&kf, (&before, &after))| (kf, (after.inverse().compose(&before), after)))
            .collect();
        let frontier = solved
            .original_poses
            .last()
            .zip(solved.optimized_poses.last())
            .map_or(Pose3d::IDENTITY, |(before, after)| {
                after.inverse().compose(before)
            });
        for (kf, keyframe) in &mut self.keyframes {
            keyframe.pose = match corrections.get(kf) {
                Some(&(_, after)) => after,
                None => keyframe.pose.compose(&frontier.inverse()),
            };
        }
        for landmark in self.landmarks.values_mut() {
            let correction = corrections
                .get(&landmark.reference_keyframe)
                .map_or(frontier, |&(correction, _)| correction);
            landmark.position = correction.transform_point(&landmark.position);
        }

        let mut search = Search {
            snapshot: &mut self,
            grids: HashMap::new(),
            ops: Vec::new(),
        };
        let sides = std::mem::take(&mut search.snapshot.sides);
        for (query, candidate) in &sides {
            search.direction(candidate, query);
            search.direction(query, candidate);
        }
        search.ops
    }
}

/// The copy under search, with each target keyframe's keypoints bucketed by the search radius.
struct Search<'a> {
    snapshot: &'a mut FusionSnapshot,
    grids: HashMap<usize, KeypointGrid>,
    ops: Vec<FusionOp>,
}

impl Search<'_> {
    /// `fuse_loop_direction` on the copy: every source landmark into every target keyframe.
    fn direction(&mut self, sources: &[usize], targets: &[usize]) {
        let mut seen = HashSet::new();
        let mut source_points = Vec::new();
        for &source_kf in sources {
            let Some(source) = self.snapshot.keyframes.get(&source_kf) else {
                continue;
            };
            for (feature, &landmark) in source.slots.iter().enumerate() {
                if let Some(landmark) = landmark
                    && seen.insert(landmark)
                {
                    source_points.push((landmark, source_kf, feature));
                }
            }
        }
        for &target_kf in targets {
            for &(source, source_kf, source_feature) in &source_points {
                if source_kf != target_kf {
                    self.fuse_into(source, source_kf, source_feature, target_kf);
                }
            }
        }
    }

    fn fuse_into(
        &mut self,
        source: usize,
        source_kf: usize,
        source_feature: usize,
        target_kf: usize,
    ) {
        let Some(point) = self.snapshot.landmarks.get(&source) else {
            return;
        };
        if point.culled || point.is_observed_by(target_kf) {
            return;
        }
        let Some((target_feature, held)) =
            best_feature(self.snapshot, &mut self.grids, point, target_kf)
        else {
            return;
        };
        let held = held.filter(|landmark| {
            self.snapshot
                .landmarks
                .get(landmark)
                .is_some_and(|point| !point.culled)
        });
        let Some(target) = held else {
            self.link(target_kf, target_feature, source);
            self.ops.push(FusionOp::Link {
                keyframe: target_kf,
                feature: target_feature,
                landmark: source,
            });
            return;
        };
        if target == source {
            return;
        }
        let consistent = self.snapshot.landmarks.get(&target).is_some_and(|point| {
            self.snapshot
                .keyframes
                .get(&source_kf)
                .is_some_and(|keyframe| {
                    reprojects_onto(
                        &self.snapshot.camera,
                        keyframe.pose,
                        keyframe.image_size,
                        point.position,
                        keyframe.keypoints.get(source_feature).copied().flatten(),
                        self.snapshot.config.max_reprojection_error_px,
                    )
                })
        });
        if consistent {
            self.merge(source, target);
            self.ops.push(FusionOp::Merge {
                source,
                source_keyframe: source_kf,
                source_feature,
                target,
                target_keyframe: target_kf,
                target_feature,
            });
        }
    }

    fn link(&mut self, keyframe: usize, feature: usize, landmark: usize) {
        if let Some(target) = self.snapshot.keyframes.get_mut(&keyframe)
            && let Some(slot) = target.slots.get_mut(feature)
        {
            *slot = Some(landmark);
        }
        if let Some(point) = self.snapshot.landmarks.get_mut(&landmark) {
            point.observations.push(ObservationKey {
                keyframe_idx: keyframe,
                feature_idx: feature,
            });
        }
    }

    /// `Map::merge_map_points` on the copy: the better-supported landmark survives and takes the
    /// other's features where it has none in that keyframe.
    fn merge(&mut self, first: usize, second: usize) {
        let landmarks = &self.snapshot.landmarks;
        let (Some(a), Some(b)) = (landmarks.get(&first), landmarks.get(&second)) else {
            return;
        };
        let support = |point: &SnapshotLandmark| (point.observations.len(), point.n_found);
        let second_is_stronger =
            support(b) > support(a) || (support(b) == support(a) && second < first);
        let (survivor, replaced) = if second_is_stronger {
            (second, first)
        } else {
            (first, second)
        };
        let Some(replaced_point) = self.snapshot.landmarks.get_mut(&replaced) else {
            return;
        };
        replaced_point.culled = true;
        let observations = std::mem::take(&mut replaced_point.observations);
        let replaced_found = replaced_point.n_found;
        for key in observations {
            let keyframe = self.snapshot.keyframes.get_mut(&key.keyframe_idx);
            let slot = keyframe.and_then(|keyframe| keyframe.slots.get_mut(key.feature_idx));
            if let Some(slot) = slot
                && *slot == Some(replaced)
            {
                *slot = None;
            }
            let survives_here = self
                .snapshot
                .landmarks
                .get(&survivor)
                .is_some_and(|point| point.is_observed_by(key.keyframe_idx));
            if !survives_here {
                self.link(key.keyframe_idx, key.feature_idx, survivor);
            }
        }
        if let Some(point) = self.snapshot.landmarks.get_mut(&survivor) {
            point.n_found = point.n_found.saturating_add(replaced_found);
        }
    }
}

/// The target feature nearest in descriptor, then in pixels, to where `point` projects, and
/// the landmark its slot holds: [`fuse_verified_loop`]'s choice, through the grid.
///
/// [`fuse_verified_loop`]: crate::loop_closure::fuse_verified_loop
fn best_feature(
    snapshot: &FusionSnapshot,
    grids: &mut HashMap<usize, KeypointGrid>,
    point: &SnapshotLandmark,
    target_kf: usize,
) -> Option<(usize, Option<usize>)> {
    let target = snapshot.keyframes.get(&target_kf)?;
    let in_camera = target.pose.transform_point(&point.position);
    if in_camera.z <= 0.0 {
        return None;
    }
    let projected = snapshot
        .camera
        .project_to_image(&in_camera, 0.0, target.image_size)
        .ok()?;
    let config = &snapshot.config;
    let radius = search_radius_px(config);
    let grid = grids
        .entry(target_kf)
        .or_insert_with(|| KeypointGrid::new(&target.keypoints, radius));
    let (x, y) = (projected.x as f32, projected.y as f32);
    let search_radius_sq = config.search_radius_px * config.search_radius_px;
    let reprojection_limit_sq = config.max_reprojection_error_px.powi(2);
    let mut best: Option<(u32, f32, usize)> = None;
    for feature in grid.near(x, y) {
        let Some(keypoint) = target.keypoints[feature] else {
            continue;
        };
        let dx = keypoint[0] - x;
        let dy = keypoint[1] - y;
        let distance_sq = dx * dx + dy * dy;
        if distance_sq > search_radius_sq || distance_sq as f64 > reprojection_limit_sq {
            continue;
        }
        let hamming = hamming_distance(&point.descriptor, &target.descriptors[feature]);
        if hamming > config.max_hamming {
            continue;
        }
        let candidate = (hamming, distance_sq, feature);
        if best.is_none_or(|best| {
            (candidate.0, candidate.1)
                .partial_cmp(&(best.0, best.1))
                .is_some_and(|order| order.is_lt() || (order.is_eq() && feature < best.2))
        }) {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, feature)| (feature, target.slots[feature]))
}

/// The pixel radius the search accepts: the tighter of the search and reprojection limits.
fn search_radius_px(config: &LoopFusionConfig) -> f32 {
    config
        .search_radius_px
        .min(config.max_reprojection_error_px as f32)
        .max(1.0)
}

/// Keypoint indices bucketed in square cells one search radius wide.
struct KeypointGrid {
    cell_px: f32,
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl KeypointGrid {
    fn new(keypoints: &[Option<[f32; 2]>], cell_px: f32) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (feature, keypoint) in keypoints.iter().enumerate() {
            if let Some([x, y]) = keypoint {
                cells
                    .entry(((x / cell_px).floor() as i32, (y / cell_px).floor() as i32))
                    .or_default()
                    .push(feature);
            }
        }
        Self { cell_px, cells }
    }

    /// Every keypoint within one cell of `(x, y)`, a superset of those within the radius.
    fn near(&self, x: f32, y: f32) -> impl Iterator<Item = usize> + '_ {
        let (cx, cy) = (
            (x / self.cell_px).floor() as i32,
            (y / self.cell_px).floor() as i32,
        );
        (cy - 1..=cy + 1)
            .flat_map(move |row| (cx - 1..=cx + 1).map(move |col| (col, row)))
            .filter_map(|cell| self.cells.get(&cell))
            .flatten()
            .copied()
    }
}

/// Whether `position` projects within `limit_px` of `keypoint` in a camera at `pose`.
fn reprojects_onto(
    camera: &PinholeCamera,
    pose: Pose3d,
    image_size: ImageSize,
    position: Vec3F64,
    keypoint: Option<[f32; 2]>,
    limit_px: f64,
) -> bool {
    let in_camera = pose.transform_point(&position);
    let Some(keypoint) = keypoint else {
        return false;
    };
    if in_camera.z <= 0.0 {
        return false;
    }
    camera
        .project_to_image(&in_camera, 0.0, image_size)
        .is_ok_and(|projected| {
            let dx = projected.x - keypoint[0] as f64;
            let dy = projected.y - keypoint[1] as f64;
            dx * dx + dy * dy <= limit_px * limit_px
        })
}

/// Applies planned operations to the live map in order. Each is checked against the map as it
/// now stands and skipped as stale when a landmark is gone, a slot changed hands, or the live
/// poses no longer put the landmark on the feature.
pub(crate) fn apply_fusion(
    map: &mut Map,
    camera: &PinholeCamera,
    ops: &[FusionOp],
    config: &LoopFusionConfig,
) -> FusionApplied {
    let mut applied = FusionApplied::default();
    for &op in ops {
        let done = match op {
            FusionOp::Link {
                keyframe,
                feature,
                landmark,
            } => {
                link_is_current(map, camera, keyframe, feature, landmark, config)
                    && map.link_observation(keyframe, feature, landmark) == Ok(true)
            }
            FusionOp::Merge {
                source,
                source_keyframe,
                source_feature,
                target,
                target_keyframe,
                target_feature,
            } => {
                merge_is_current(
                    map,
                    camera,
                    (source, source_keyframe, source_feature),
                    (target, target_keyframe, target_feature),
                    config,
                ) && map.merge_map_points_by_records(source, target).is_some()
            }
        };
        match (done, op) {
            (false, _) => applied.stale += 1,
            (true, FusionOp::Link { .. }) => applied.linked += 1,
            (true, FusionOp::Merge { .. }) => applied.merged += 1,
        }
    }
    applied
}

fn live_landmark(map: &Map, landmark: usize) -> Option<Vec3F64> {
    map.map_points()
        .get(landmark)
        .filter(|point| !point.culled)
        .map(|point| point.position)
}

/// Whether `landmark`, live, projects onto `keyframe`'s live `feature`.
fn lands_on(
    map: &Map,
    camera: &PinholeCamera,
    landmark: usize,
    keyframe: usize,
    feature: usize,
    config: &LoopFusionConfig,
) -> bool {
    let (Some(position), Some(keyframe)) =
        (live_landmark(map, landmark), map.get_keyframe(keyframe))
    else {
        return false;
    };
    reprojects_onto(
        camera,
        keyframe.frame.pose_world_to_cam,
        keyframe.frame.image_size,
        position,
        keyframe.frame.undistorted_xy(feature, camera),
        config.max_reprojection_error_px,
    )
}

fn link_is_current(
    map: &Map,
    camera: &PinholeCamera,
    keyframe: usize,
    feature: usize,
    landmark: usize,
    config: &LoopFusionConfig,
) -> bool {
    let observed = map
        .map_points()
        .get(landmark)
        .is_none_or(|point| point.is_observed_by(keyframe));
    let slot_free = map.get_keyframe(keyframe).is_some_and(|kf| {
        kf.map_point(feature)
            .is_none_or(|held| live_landmark(map, held).is_none())
    });
    !observed && slot_free && lands_on(map, camera, landmark, keyframe, feature, config)
}

fn merge_is_current(
    map: &Map,
    camera: &PinholeCamera,
    (source, source_keyframe, source_feature): (usize, usize, usize),
    (target, target_keyframe, target_feature): (usize, usize, usize),
    config: &LoopFusionConfig,
) -> bool {
    let holds = |keyframe: usize, feature: usize, landmark: usize| {
        map.get_keyframe(keyframe)
            .is_some_and(|kf| kf.map_point(feature) == Some(landmark))
    };
    let source_unseen = map
        .map_points()
        .get(source)
        .is_some_and(|point| !point.is_observed_by(target_keyframe));
    source != target
        && source_unseen
        && holds(source_keyframe, source_feature, source)
        && holds(target_keyframe, target_feature, target)
        && lands_on(map, camera, source, target_keyframe, target_feature, config)
        && lands_on(map, camera, target, source_keyframe, source_feature, config)
}
