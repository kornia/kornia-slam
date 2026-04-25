//! Map: keyframes, map points, local map selection, and culling.
//!
//! ```text
//!    +--------+
//!    | Frame  |
//!    +--------+
//!         |
//!         v
//!    +----------------------+
//!    | Keyframe             |
//!    | frame + desc -> mp   |
//!    +----------------------+
//!         |
//!         v
//!    +----------------------+
//!    | Map                  |
//!    | keyframes + points   |
//!    +----------------------+
//!
//!    ops:
//!      * upsert_keyframe
//!      * push_map_point
//!      * build_local_map_points
//!      * cull
//!      * run_local_ba
//! ```

use std::collections::{HashMap, HashSet};

use kornia_3d::ba::{self, BaObservation, BaParams};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_bow::metric::Hamming;
use kornia_bow::orb_slam3::pack_orb_descriptor;
use kornia_bow::{BoW, DirectIndex, Vocabulary};
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;

use crate::frame::Frame;

const ORB_SCALE_FACTOR: f64 = 1.2;
const ORB_N_LEVELS: usize = 8;

/// A frame promoted into the map, with descriptor-to-map-point associations.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub frame: Frame,
    /// For each descriptor index in `frame.features`, associated map-point index.
    pub map_point_by_desc_idx: Vec<Option<usize>>,
    /// Bag-of-Words histogram computed against the ORB-SLAM3 vocabulary. Populated
    /// lazily via [`Keyframe::compute_bow`].
    pub bow: Option<BoW>,
    /// Direct index: tree node → list of descriptor indices that traverse through
    /// it at a chosen mid-tree level. Used by `SearchForTriangulation`-style
    /// BoW-accelerated matching.
    pub direct_index: Option<DirectIndex>,
}

impl Keyframe {
    /// Creates a keyframe from a frame, with empty map-point associations.
    pub fn from_frame(frame: Frame) -> Self {
        let map_point_by_desc_idx = vec![None; frame.features.descriptors.len()];
        Self {
            frame,
            map_point_by_desc_idx,
            bow: None,
            direct_index: None,
        }
    }

    /// Computes the BoW vector and DirectIndex against the given vocabulary.
    ///
    /// `direct_index_level` selects the tree depth at which descriptors are
    /// grouped in the direct index (ORB-SLAM3 uses level 2 of a depth-6 tree).
    pub fn compute_bow(
        &mut self,
        vocab: &Vocabulary<10, Hamming<4>>,
        direct_index_level: usize,
    ) -> Result<(), kornia_bow::BowError> {
        if self.frame.features.descriptors.is_empty() {
            self.bow = None;
            self.direct_index = None;
            return Ok(());
        }
        let packed: Vec<_> = self
            .frame
            .features
            .descriptors
            .iter()
            .map(pack_orb_descriptor)
            .collect();
        let (bow, di) = vocab.transform_with_direct_index(&packed, direct_index_level)?;
        self.bow = Some(bow);
        self.direct_index = Some(di);
        Ok(())
    }

    /// Associates a descriptor slot with a persistent map point.
    pub fn associate_map_point(&mut self, desc_idx: usize, mp_idx: usize) {
        if let Some(slot) = self.map_point_by_desc_idx.get_mut(desc_idx) {
            *slot = Some(mp_idx);
        }
    }

    /// Clears the map-point association for a descriptor slot.
    pub fn clear_map_point(&mut self, desc_idx: usize) {
        if let Some(slot) = self.map_point_by_desc_idx.get_mut(desc_idx) {
            *slot = None;
        }
    }

    /// Returns the associated map-point index for a descriptor slot.
    pub fn map_point(&self, desc_idx: usize) -> Option<usize> {
        self.map_point_by_desc_idx.get(desc_idx).copied().flatten()
    }

    /// Counts how many descriptor slots currently reference a map point.
    pub fn num_associated_points(&self) -> usize {
        self.map_point_by_desc_idx
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }
}

/// A triangulated point ready for map insertion: (position, descriptor, color, prev_desc_idx, curr_desc_idx).
pub type TriangulatedPoint = (Vec3F64, [u8; 32], [u8; 3], usize, usize);

/// A persistent 3D landmark in the map.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// 3D position in world frame.
    pub position: Vec3F64,
    /// ORB descriptor used for projection-guided matching.
    pub descriptor: [u8; 32],
    /// Pixel color sampled at the keypoint that created this point.
    pub color: [u8; 3],
    /// Index of the keyframe that first observed this point.
    pub keyframe_idx: usize,
    /// Number of frames where this point was in the camera frustum.
    pub n_visible: u32,
    /// Number of frames where this point was successfully matched.
    pub n_found: u32,
    /// Whether this point has been culled (logically deleted).
    pub culled: bool,
    /// Average viewing direction from observing camera centers toward the point.
    pub normal: Vec3F64,
    /// Reference minimum distance for scale invariance.
    pub min_distance: f64,
    /// Reference maximum distance for scale invariance.
    pub max_distance: f64,
    /// Optional override for the observation count used by Fuse conflict
    /// resolution. ORB-SLAM3's `MapPoint::Observations()` reads `mObservations.size()`
    /// directly, which can disagree with the per-keyframe `mvpMapPoints` slot
    /// state (some `Erase`/`Replace` paths leave stale entries). Parity replays
    /// seed this from the dump's `observations` field so winner-selection
    /// matches C++ exactly. None falls back to the inverse-index count.
    pub observation_count_override: Option<u32>,
    /// Optional set of keyframe indices observed by this MP, seeded from
    /// `mObservations` for parity replay. ORB-SLAM3's
    /// `MapPoint::IsInKeyFrame(pKF)` reads this directly; Rust's default
    /// "is in kf" check scans the forward `mvpMapPoints` slot-list, which can
    /// disagree when stale entries exist. When set, Fuse's IsInKF gate
    /// honors this set in addition to the forward scan, so kornia skips
    /// candidates exactly where ORB-SLAM3 does.
    pub observed_keyframes_override: Option<std::collections::HashSet<usize>>,
}

impl MapPoint {
    /// Creates a fresh active map point.
    pub fn new(
        position: Vec3F64,
        descriptor: [u8; 32],
        color: [u8; 3],
        keyframe_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            color,
            keyframe_idx,
            n_visible: 1,
            n_found: 1,
            culled: false,
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            observation_count_override: None,
            observed_keyframes_override: None,
        }
    }

    /// Marks the point as logically deleted.
    pub fn mark_culled(&mut self) {
        self.culled = true;
    }

    /// Returns the tracking success ratio for this point.
    pub fn found_ratio(&self) -> f64 {
        if self.n_visible == 0 {
            return 0.0;
        }
        self.n_found as f64 / self.n_visible as f64
    }

    /// Sets viewing-geometry metadata used by ORB-SLAM3-style frustum gating.
    pub fn set_view_geometry(&mut self, normal: Vec3F64, min_distance: f64, max_distance: f64) {
        self.normal = if normal.length() > 0.0 {
            normal.normalize()
        } else {
            Vec3F64::new(0.0, 0.0, 1.0)
        };
        self.min_distance = min_distance.max(0.0);
        self.max_distance = max_distance.max(self.min_distance);
    }

    /// Returns the viewing normal used by frustum gating.
    pub fn viewing_normal(&self) -> Vec3F64 {
        self.normal
    }

    /// Lower distance-invariance bound with ORB-SLAM3 slack.
    pub fn min_distance_invariance(&self) -> f64 {
        0.8 * self.min_distance
    }

    /// Upper distance-invariance bound with ORB-SLAM3 slack.
    pub fn max_distance_invariance(&self) -> f64 {
        1.2 * self.max_distance
    }

    /// Predicts the expected pyramid level for a current viewing distance.
    pub fn predict_scale(&self, current_dist: f64, scale_factor: f64, n_levels: usize) -> usize {
        if !self.max_distance.is_finite() || current_dist <= 0.0 || scale_factor <= 1.0 || n_levels == 0
        {
            return 0;
        }
        let ratio = self.max_distance / current_dist;
        let level = (ratio.ln() / scale_factor.ln()).ceil() as isize;
        level.clamp(0, n_levels.saturating_sub(1) as isize) as usize
    }
}

/// Fusion policy for [`Map::fuse_projected_map_points_into_keyframe_limited_with_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseMode {
    /// Only assign projected map points to keypoints that don't already own one.
    /// Existing associations are never replaced. Safe default; used by the
    /// pipeline's `SearchInNeighbors` step to promote 2-obs points to 3+ obs
    /// without risking map corruption.
    AddOnly,
    /// When two map points compete for the same keypoint, keep the one observed
    /// by more keyframes and replace the other. Mirrors ORB-SLAM3
    /// `ORBmatcher::Fuse` replacement, but without its viewing-angle,
    /// scale-invariance, or chi² gates.
    ReplaceWeaker,
}

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
    observations_by_map_point: Vec<Vec<(usize, usize)>>,
    /// Frame index of the origin (first) keyframe. Pinned fixed in every local BA
    /// to lock the global frame — mirrors ORB-SLAM3's `Map::GetInitKFid()`.
    origin_kf_frame_idx: Option<usize>,
}

impl Map {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all keyframes.
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Returns the number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }

    /// Returns the number of active, non-culled map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map_points.iter().filter(|mp| !mp.culled).count()
    }

    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }

    /// Returns a mutable reference to the keyframe with frame index `idx`, if present.
    pub fn get_keyframe_mut(&mut self, idx: usize) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|kf| kf.frame.idx == idx)
    }

    /// Counts how many map points associated with `kf_idx` are observed by at least
    /// `min_obs` keyframes. This mirrors ORB-SLAM3's `KeyFrame::TrackedMapPoints(nMinObs)`.
    pub fn tracked_map_points(&self, kf_idx: usize, min_obs: usize) -> usize {
        let Some(pos) = self.keyframes.iter().position(|kf| kf.frame.idx == kf_idx) else {
            return 0;
        };
        self.keyframes[pos]
            .map_point_by_desc_idx
            .iter()
            .flatten()
            .filter(|&mp_idx| {
                self.map_points.get(*mp_idx).is_some_and(|mp| !mp.culled)
                    && self.map_point_observation_count(*mp_idx) >= min_obs
            })
            .count()
    }

    /// Inserts or replaces a keyframe by frame index.
    pub fn upsert_keyframe(&mut self, keyframe: Keyframe) {
        let mut affected_points = HashSet::new();
        if let Some(pos) = self
            .keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe.frame.idx)
        {
            for mp_idx in self.keyframes[pos].map_point_by_desc_idx.iter().flatten() {
                affected_points.insert(*mp_idx);
            }
            for mp_idx in keyframe.map_point_by_desc_idx.iter().flatten() {
                affected_points.insert(*mp_idx);
            }
            self.keyframes[pos] = keyframe;
        } else {
            if self.origin_kf_frame_idx.is_none() {
                self.origin_kf_frame_idx = Some(keyframe.frame.idx);
            }
            for mp_idx in keyframe.map_point_by_desc_idx.iter().flatten() {
                affected_points.insert(*mp_idx);
            }
            self.keyframes.push(keyframe);
        }
        self.rebuild_observation_index();
        self.refresh_map_points_metadata(affected_points);
    }

    /// Returns the origin keyframe's frame index, if any.
    pub fn origin_kf_frame_idx(&self) -> Option<usize> {
        self.origin_kf_frame_idx
    }

    /// Returns the global indices of non-culled map points observed by the
    /// most recent `n_kfs` keyframes. Used by the tracker to restrict
    /// projection-guided matching to a locally-relevant subset of the map,
    /// approximating ORB-SLAM3's `TrackLocalMap` scope.
    pub fn recent_kf_mp_indices(&self, n_kfs: usize) -> HashSet<usize> {
        let total = self.keyframes.len();
        let start = total.saturating_sub(n_kfs);
        let mut set = HashSet::new();
        for kf in &self.keyframes[start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if self.map_points.get(*mp_idx).is_some_and(|mp| !mp.culled) {
                    set.insert(*mp_idx);
                }
            }
        }
        set
    }

    pub fn associate_keyframe_map_point(&mut self, kf_idx: usize, desc_idx: usize, mp_idx: usize) -> bool {
        let Some(kf_pos) = self.keyframes.iter().position(|kf| kf.frame.idx == kf_idx) else {
            return false;
        };
        let Some(slot) = self.keyframes[kf_pos].map_point_by_desc_idx.get_mut(desc_idx) else {
            return false;
        };

        let mut affected_points = HashSet::new();
        if let Some(prev_mp_idx) = *slot {
            affected_points.insert(prev_mp_idx);
        }
        *slot = Some(mp_idx);
        affected_points.insert(mp_idx);
        self.rebuild_observation_index();
        self.refresh_map_points_metadata(affected_points);
        true
    }

    fn rebuild_observation_index(&mut self) {
        let mut observations_by_map_point = vec![Vec::new(); self.map_points.len()];
        for kf in &self.keyframes {
            for (desc_idx, mp_idx) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = *mp_idx else {
                    continue;
                };
                if self.map_points.get(mp_idx).is_some_and(|mp| !mp.culled)
                    && mp_idx < observations_by_map_point.len()
                {
                    observations_by_map_point[mp_idx].push((kf.frame.idx, desc_idx));
                }
            }
        }
        self.observations_by_map_point = observations_by_map_point;
    }

    fn map_point_observations(&self, mp_idx: usize) -> &[(usize, usize)] {
        self.observations_by_map_point
            .get(mp_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn map_point_observation_count(&self, mp_idx: usize) -> usize {
        if self.map_points.get(mp_idx).is_some_and(|mp| !mp.culled) {
            self.map_point_observations(mp_idx).len()
        } else {
            0
        }
    }

    fn map_point_observation_counts(&self) -> HashMap<usize, usize> {
        self.observations_by_map_point
            .iter()
            .enumerate()
            .filter_map(|(mp_idx, observations)| {
                let mp = self.map_points.get(mp_idx)?;
                if mp.culled {
                    return None;
                }
                let count = mp
                    .observation_count_override
                    .map(|c| c as usize)
                    .unwrap_or(observations.len());
                Some((mp_idx, count))
            })
            .collect()
    }

    fn active_observations_by_map_point(&self) -> HashMap<usize, Vec<usize>> {
        self.observations_by_map_point
            .iter()
            .enumerate()
            .filter_map(|(mp_idx, observations)| {
                self.map_points.get(mp_idx).is_some_and(|mp| !mp.culled).then(|| {
                    (
                        mp_idx,
                        observations.iter().map(|(kf_idx, _)| *kf_idx).collect::<Vec<_>>(),
                    )
                })
            })
            .collect()
    }

    fn keypoint_octave(scale: f32) -> usize {
        let raw = ((scale as f64).ln() / ORB_SCALE_FACTOR.ln()).round() as isize;
        raw.clamp(0, ORB_N_LEVELS.saturating_sub(1) as isize) as usize
    }

    fn refresh_map_points_metadata<I>(&mut self, mp_indices: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let unique: HashSet<_> = mp_indices.into_iter().collect();
        for mp_idx in unique {
            self.refresh_map_point_metadata(mp_idx);
        }
    }

    fn refresh_map_point_metadata(&mut self, mp_idx: usize) {
        let Some(mp) = self.map_points.get(mp_idx) else {
            return;
        };
        if mp.culled {
            return;
        }
        let observations = self.map_point_observations(mp_idx);
        if observations.is_empty() {
            return;
        }

        let mut descriptors = Vec::new();
        let mut normal_sum = Vec3F64::new(0.0, 0.0, 0.0);
        let mut normal_count = 0usize;
        let position = mp.position;
        let desired_ref_kf_idx = mp.keyframe_idx;
        let mut fallback_ref = None;
        let mut chosen_ref = None;

        for &(kf_idx, desc_idx) in observations {
            let Some(kf) = self.get_keyframe(kf_idx) else {
                continue;
            };
            let Some(descriptor) = kf.frame.features.descriptors.get(desc_idx).copied() else {
                continue;
            };
            descriptors.push((kf_idx, desc_idx, descriptor));

            let center = kf.frame.pose_world_to_cam.inverse().translation;
            let viewing_ray = position - center;
            let dist = viewing_ray.length();
            if dist > 0.0 {
                normal_sum += viewing_ray / dist;
                normal_count += 1;
                let scale = kf
                    .frame
                    .features
                    .scales
                    .get(desc_idx)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1.0) as f64;
                let candidate_ref = (kf_idx, dist, scale);
                if fallback_ref.is_none() {
                    fallback_ref = Some(candidate_ref);
                }
                if kf_idx == desired_ref_kf_idx && chosen_ref.is_none() {
                    chosen_ref = Some(candidate_ref);
                }
            }
        }

        if descriptors.is_empty() {
            return;
        }

        let descriptor = if descriptors.len() == 1 {
            descriptors[0].2
        } else {
            let mut best_descriptor = descriptors[0].2;
            let mut best_median = u32::MAX;
            for (i, &(_, _, descriptor_i)) in descriptors.iter().enumerate() {
                let mut distances = Vec::with_capacity(descriptors.len().saturating_sub(1));
                for (j, &(_, _, descriptor_j)) in descriptors.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    distances.push(hamming_distance(&descriptor_i, &descriptor_j));
                }
                distances.sort_unstable();
                let median = distances[distances.len() / 2];
                if median < best_median {
                    best_median = median;
                    best_descriptor = descriptor_i;
                }
            }
            best_descriptor
        };

        let Some((ref_kf_idx, ref_dist, ref_scale)) = chosen_ref.or(fallback_ref) else {
            return;
        };
        let max_distance = ref_dist * ref_scale;
        let min_distance =
            max_distance / ORB_SCALE_FACTOR.powi(ORB_N_LEVELS.saturating_sub(1) as i32);
        let normal = if normal_count > 0 {
            normal_sum / normal_count as f64
        } else {
            Vec3F64::new(0.0, 0.0, 1.0)
        };

        if let Some(mp_mut) = self.map_points.get_mut(mp_idx) {
            mp_mut.descriptor = descriptor;
            mp_mut.keyframe_idx = ref_kf_idx;
            mp_mut.set_view_geometry(normal, min_distance, max_distance);
        }
    }

    /// Returns keyframes that share active map points with `kf_idx`, sorted by
    /// descending shared-point count. This is the on-demand equivalent of
    /// ORB-SLAM3's covisibility graph connections.
    pub fn covisibility_neighbors(&self, kf_idx: usize, top_n: usize) -> Vec<(usize, usize)> {
        if top_n == 0 || self.get_keyframe(kf_idx).is_none() {
            return Vec::new();
        }

        let observers = self.active_observations_by_map_point();
        let mut shared_counts: HashMap<usize, usize> = HashMap::new();
        let Some(kf) = self.get_keyframe(kf_idx) else {
            return Vec::new();
        };

        let mut seen = HashSet::new();
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if !seen.insert(*mp_idx) {
                continue;
            }
            let Some(mp) = self.map_points.get(*mp_idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            let Some(mp_observers) = observers.get(mp_idx) else {
                continue;
            };
            for &other_kf_idx in mp_observers {
                if other_kf_idx != kf_idx {
                    *shared_counts.entry(other_kf_idx).or_insert(0) += 1;
                }
            }
        }

        let mut neighbors: Vec<_> = shared_counts.into_iter().collect();
        neighbors.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        neighbors.truncate(top_n);
        neighbors
    }

    fn local_keyframe_indices(
        &self,
        tracked_matches: &[(usize, usize)],
        current_keyframe: Option<&Keyframe>,
        max_keyframes: usize,
    ) -> Vec<usize> {
        const MAX_COVISIBLE_PER_SEED: usize = 10;
        const RECENT_FALLBACK_KEYFRAMES: usize = 10;

        let observers = self.active_observations_by_map_point();
        let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
        for &(mp_idx, _) in tracked_matches {
            let Some(mp_observers) = observers.get(&mp_idx) else {
                continue;
            };
            for &kf_idx in mp_observers {
                *keyframe_votes.entry(kf_idx).or_insert(0) += 1;
            }
        }

        let mut voted_kfs: Vec<(usize, usize)> = keyframe_votes.into_iter().collect();
        voted_kfs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut local_kfs = Vec::new();
        let mut seen = HashSet::new();
        let push_kf = |kf_idx: usize, local_kfs: &mut Vec<usize>, seen: &mut HashSet<usize>| {
            if local_kfs.len() < max_keyframes && seen.insert(kf_idx) {
                local_kfs.push(kf_idx);
            }
        };

        if let Some(kf) = current_keyframe {
            push_kf(kf.frame.idx, &mut local_kfs, &mut seen);
        }
        for (kf_idx, _) in voted_kfs {
            push_kf(kf_idx, &mut local_kfs, &mut seen);
        }

        // If tracking has too few matches to vote, keep the tracker alive with
        // a temporal fallback. Once covisibility exists, the neighbor expansion
        // below dominates this list.
        if local_kfs.len() < 2 {
            for kf in self.keyframes.iter().rev().take(RECENT_FALLBACK_KEYFRAMES) {
                push_kf(kf.frame.idx, &mut local_kfs, &mut seen);
            }
        }

        let seeds = local_kfs.clone();
        for seed_idx in seeds {
            if local_kfs.len() >= max_keyframes {
                break;
            }
            for (neighbor_idx, _) in self.covisibility_neighbors(seed_idx, MAX_COVISIBLE_PER_SEED) {
                push_kf(neighbor_idx, &mut local_kfs, &mut seen);
                if local_kfs.len() >= max_keyframes {
                    break;
                }
            }
        }

        local_kfs
    }

    /// Returns active map-point indices observed by a keyframe.
    pub fn keyframe_map_point_indices(&self, kf_idx: usize) -> Vec<usize> {
        let Some(kf) = self.get_keyframe(kf_idx) else {
            return Vec::new();
        };
        let mut seen = HashSet::new();
        let mut indices = Vec::new();
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if seen.insert(*mp_idx) && self.map_points.get(*mp_idx).is_some_and(|mp| !mp.culled) {
                indices.push(*mp_idx);
            }
        }
        indices
    }

    /// Replaces every observation of `from_mp_idx` with `to_mp_idx` and marks the
    /// replaced point culled. If a keyframe already observes the survivor, the
    /// duplicate observation is cleared to keep one map-point slot per keyframe.
    fn replace_map_point_slots(&mut self, from_mp_idx: usize, to_mp_idx: usize) -> bool {
        if from_mp_idx == to_mp_idx
            || from_mp_idx >= self.map_points.len()
            || to_mp_idx >= self.map_points.len()
            || self.map_points[to_mp_idx].culled
        {
            return false;
        }

        for kf in &mut self.keyframes {
            let has_survivor = kf
                .map_point_by_desc_idx
                .iter()
                .any(|slot| *slot == Some(to_mp_idx));
            let mut survivor_assigned_here = has_survivor;
            for slot in &mut kf.map_point_by_desc_idx {
                if *slot != Some(from_mp_idx) {
                    continue;
                }
                if survivor_assigned_here {
                    *slot = None;
                } else {
                    *slot = Some(to_mp_idx);
                    survivor_assigned_here = true;
                }
            }
        }

        if let Some(mp) = self.map_points.get_mut(from_mp_idx) {
            mp.mark_culled();
        }
        true
    }

    pub fn replace_map_point(&mut self, from_mp_idx: usize, to_mp_idx: usize) -> bool {
        if !self.replace_map_point_slots(from_mp_idx, to_mp_idx) {
            return false;
        }
        self.rebuild_observation_index();
        self.refresh_map_points_metadata([to_mp_idx]);
        true
    }

    /// Projects candidate map points into `kf_idx` and fuses descriptor matches.
    ///
    /// This is a compact monocular analogue of ORB-SLAM3 `ORBmatcher::Fuse`:
    /// points already observed by the target keyframe are skipped, unclaimed
    /// keypoints add a new observation, and conflicting map points are replaced
    /// by the one with more keyframe observations.
    pub fn fuse_projected_map_points_into_keyframe(
        &mut self,
        kf_idx: usize,
        candidate_mp_indices: &[usize],
        camera: &PinholeCamera,
    ) -> usize {
        self.fuse_projected_map_points_into_keyframe_limited_with_mode(
            kf_idx,
            candidate_mp_indices,
            camera,
            usize::MAX,
            FuseMode::ReplaceWeaker,
        )
    }

    /// Same as [`Map::fuse_projected_map_points_into_keyframe`], but stops after
    /// `max_fused` accepted observations/replacements.
    pub fn fuse_projected_map_points_into_keyframe_limited(
        &mut self,
        kf_idx: usize,
        candidate_mp_indices: &[usize],
        camera: &PinholeCamera,
        max_fused: usize,
    ) -> usize {
        self.fuse_projected_map_points_into_keyframe_limited_with_mode(
            kf_idx,
            candidate_mp_indices,
            camera,
            max_fused,
            FuseMode::ReplaceWeaker,
        )
    }

    /// Variant of [`Map::fuse_projected_map_points_into_keyframe_limited`] that
    /// selects whether conflicting map points may replace one another. With
    /// [`FuseMode::AddOnly`], only unclaimed keypoints receive a new observation;
    /// existing associations are never disturbed. This is the safe mode used by
    /// the pipeline's `SearchInNeighbors` step until full ORB-SLAM3 `Fuse` gates
    /// (viewing angle, scale invariance, chi² reprojection) are implemented.
    pub fn fuse_projected_map_points_into_keyframe_limited_with_mode(
        &mut self,
        kf_idx: usize,
        candidate_mp_indices: &[usize],
        camera: &PinholeCamera,
        max_fused: usize,
        mode: FuseMode,
    ) -> usize {
        // Matches ORB-SLAM3 `ORBmatcher::Fuse` gates:
        //   viewing angle cos >= 0.5, distance invariance, predicted octave
        //   filter `[pred-1, pred]`, chi-square reprojection gate
        //   `e^2 * invLevelSigma^2 <= 5.99`.
        const FUSE_TH: f32 = 3.0;
        const CHI2_MONO: f32 = 5.991;
        const ORB_SCALE_FACTOR_F32: f32 = 1.2;
        const ORB_N_LEVELS_LOCAL: usize = 8;
        const MAX_HAMMING: u32 = 50;
        const GRID_CELL_SIZE: f32 = 32.0;

        let Some(kf_pos) = self.keyframes.iter().position(|kf| kf.frame.idx == kf_idx) else {
            return 0;
        };
        if candidate_mp_indices.is_empty() || max_fused == 0 {
            return 0;
        }

        let pose_world_to_cam = self.keyframes[kf_pos].frame.pose_world_to_cam;
        let camera_center_world = pose_world_to_cam.inverse().translation;
        let image_size = self.keyframes[kf_pos].frame.image_size;
        let keypoints_undist: Vec<[f32; 2]> = self.keyframes[kf_pos]
            .frame
            .features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                [p.x as f32, p.y as f32]
            })
            .collect();
        let scales = self.keyframes[kf_pos].frame.features.scales.clone();
        let descriptors = self.keyframes[kf_pos].frame.features.descriptors.clone();
        let observation_counts = self.map_point_observation_counts();
        // Precompute per-kp octave; used for predicted-level filter and chi-square sigma.
        let kp_octaves: Vec<usize> = scales
            .iter()
            .map(|&s| Self::keypoint_octave(s))
            .collect();
        let max_level_scale =
            ORB_SCALE_FACTOR_F32.powi(ORB_N_LEVELS_LOCAL.saturating_sub(1) as i32);
        let max_query_radius = FUSE_TH * max_level_scale;
        let mut keypoint_grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (kp_idx, kp) in keypoints_undist.iter().enumerate() {
            let cell = (
                (kp[0] / GRID_CELL_SIZE).floor() as i32,
                (kp[1] / GRID_CELL_SIZE).floor() as i32,
            );
            keypoint_grid.entry(cell).or_default().push(kp_idx);
        }

        let mut fused = 0usize;
        let mut touched_map_points = HashSet::new();
        let mut seen_candidates = HashSet::new();
        for candidate_mp_idx in candidate_mp_indices.iter().copied() {
            if fused >= max_fused {
                break;
            }
            if !seen_candidates.insert(candidate_mp_idx) {
                continue;
            }
            let Some(candidate_mp) = self.map_points.get(candidate_mp_idx) else {
                continue;
            };
            // ORB-SLAM3 `MapPoint::IsInKeyFrame(pKF)` reads `mObservations`
            // directly. When `observed_keyframes_override` is set (parity
            // replay), use it as the authoritative source — that's what C++
            // does. Otherwise scan the forward `mvpMapPoints` slot-list.
            let is_in_kf = match candidate_mp.observed_keyframes_override.as_ref() {
                Some(set) => set.contains(&kf_idx),
                None => self.keyframes[kf_pos]
                    .map_point_by_desc_idx
                    .iter()
                    .any(|slot| *slot == Some(candidate_mp_idx)),
            };
            if candidate_mp.culled || is_in_kf {
                continue;
            }

            // Positive depth + image bounds.
            let p_cam = pose_world_to_cam.transform_point(&candidate_mp.position);
            let Ok(pixel) = camera.project_to_image(&p_cam, 0.0, image_size) else {
                continue;
            };
            let u = pixel.x as f32;
            let v = pixel.y as f32;

            // Distance invariance gate.
            let po = candidate_mp.position - camera_center_world;
            let dist3d = po.length();
            if dist3d <= 0.0
                || dist3d < candidate_mp.min_distance_invariance()
                || dist3d > candidate_mp.max_distance_invariance()
            {
                continue;
            }

            // Viewing angle gate (cos >= 0.5 → angle <= 60 deg).
            let normal = candidate_mp.viewing_normal();
            if normal.dot(po) < 0.5 * dist3d {
                continue;
            }

            // Predicted octave and per-level search radius.
            let predicted_level =
                candidate_mp.predict_scale(dist3d, ORB_SCALE_FACTOR, ORB_N_LEVELS);
            let pred_level_scale = ORB_SCALE_FACTOR_F32.powi(predicted_level as i32);
            let search_radius = FUSE_TH * pred_level_scale;
            let min_level = predicted_level.saturating_sub(1);
            let max_level = predicted_level;

            let mut best_dist = u32::MAX;
            let mut best_idx = None;
            let min_cell_x = ((u - max_query_radius) / GRID_CELL_SIZE).floor() as i32;
            let max_cell_x = ((u + max_query_radius) / GRID_CELL_SIZE).floor() as i32;
            let min_cell_y = ((v - max_query_radius) / GRID_CELL_SIZE).floor() as i32;
            let max_cell_y = ((v + max_query_radius) / GRID_CELL_SIZE).floor() as i32;
            for cell_y in min_cell_y..=max_cell_y {
                for cell_x in min_cell_x..=max_cell_x {
                    let Some(indices) = keypoint_grid.get(&(cell_x, cell_y)) else {
                        continue;
                    };
                    for &kp_idx in indices {
                        let kp_level = kp_octaves[kp_idx];
                        if kp_level < min_level || kp_level > max_level {
                            continue;
                        }
                        let kp = &keypoints_undist[kp_idx];
                        let dx = kp[0] - u;
                        let dy = kp[1] - v;
                        let e2 = dx * dx + dy * dy;
                        if e2 > search_radius * search_radius {
                            continue;
                        }
                        // Chi-square monocular gate: e^2 * invLevelSigma^2 <= 5.99,
                        // i.e. e^2 <= 5.99 * levelSigma^2 = 5.99 * scaleFactor^(2L).
                        let level_sigma2 = ORB_SCALE_FACTOR_F32.powi(2 * kp_level as i32);
                        if e2 > CHI2_MONO * level_sigma2 {
                            continue;
                        }
                        let Some(desc) = descriptors.get(kp_idx) else {
                            continue;
                        };
                        let dist = hamming_distance(&candidate_mp.descriptor, desc);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = Some(kp_idx);
                        }
                    }
                }
            }

            if best_dist > MAX_HAMMING {
                continue;
            }
            let Some(best_idx) = best_idx else {
                continue;
            };

            match self.keyframes[kf_pos].map_point(best_idx) {
                None => {
                    self.keyframes[kf_pos].associate_map_point(best_idx, candidate_mp_idx);
                    touched_map_points.insert(candidate_mp_idx);
                    fused += 1;
                }
                Some(existing_mp_idx) if existing_mp_idx == candidate_mp_idx => {}
                Some(existing_mp_idx) => match mode {
                    FuseMode::AddOnly => continue,
                    FuseMode::ReplaceWeaker => {
                        let existing_obs = observation_counts
                            .get(&existing_mp_idx)
                            .copied()
                            .unwrap_or(0);
                        let candidate_obs = observation_counts
                            .get(&candidate_mp_idx)
                            .copied()
                            .unwrap_or(0);
                        if existing_obs > candidate_obs {
                            if self.replace_map_point_slots(candidate_mp_idx, existing_mp_idx) {
                                touched_map_points.insert(existing_mp_idx);
                                fused += 1;
                            }
                        } else if self.replace_map_point_slots(existing_mp_idx, candidate_mp_idx) {
                            self.keyframes[kf_pos]
                                .associate_map_point(best_idx, candidate_mp_idx);
                            touched_map_points.insert(candidate_mp_idx);
                            fused += 1;
                        }
                    }
                },
            }
        }

        if !touched_map_points.is_empty() {
            self.rebuild_observation_index();
            self.refresh_map_points_metadata(touched_map_points);
        }

        fused
    }

    /// Inserts triangulated 3D points as map points and associates them to keyframes.
    ///
    /// For each entry, creates a `MapPoint` and associates it with `curr_kf`.
    /// If `prev_kf` is provided, associates it there too.
    /// Inserts triangulated 3D points as map points and associates them to keyframes.
    ///
    /// For each entry, creates a `MapPoint` and associates it with `curr_kf`.
    /// If `prev_kf` is provided, associates it there too.
    pub fn add_triangulated_points(
        &mut self,
        prev_kf: Option<&mut Keyframe>,
        curr_kf: &mut Keyframe,
        points: &[TriangulatedPoint],
        keyframe_idx: usize,
    ) -> usize {
        let first_mp_idx = self.map_points.len();
        let curr_center = curr_kf.frame.pose_world_to_cam.inverse().translation;
        let curr_max_scale = curr_kf
            .frame
            .features
            .scales
            .iter()
            .copied()
            .fold(1.0_f32, f32::max) as f64;
        let prev_center = prev_kf
            .as_ref()
            .map(|kf| kf.frame.pose_world_to_cam.inverse().translation);
        for (i, &(position, descriptor, color, _, curr_desc_idx)) in points.iter().enumerate() {
            let mut map_point = MapPoint::new(position, descriptor, color, keyframe_idx);
            let mut normal = position - curr_center;
            let mut n_normals = 0.0;
            if normal.length() > 0.0 {
                normal = normal.normalize();
                n_normals += 1.0;
            }
            if let Some(prev_center) = prev_center {
                let prev_normal = position - prev_center;
                if prev_normal.length() > 0.0 {
                    normal += prev_normal.normalize();
                    n_normals += 1.0;
                }
            }
            let ref_dist = (position - curr_center).length();
            let level_scale = curr_kf
                .frame
                .features
                .scales
                .get(curr_desc_idx)
                .copied()
                .unwrap_or(1.0) as f64;
            let max_distance = ref_dist * level_scale.max(1.0);
            let min_distance = if curr_max_scale > 0.0 {
                max_distance / curr_max_scale
            } else {
                max_distance
            };
            if n_normals > 0.0 {
                map_point.set_view_geometry(normal / n_normals, min_distance, max_distance);
            }
            self.push_map_point(map_point);
            curr_kf.associate_map_point(curr_desc_idx, first_mp_idx + i);
        }
        if let Some(prev) = prev_kf {
            for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().enumerate() {
                prev.associate_map_point(prev_desc_idx, first_mp_idx + i);
            }
        }
        points.len()
    }

    /// Appends a map point and returns its index.
    pub fn push_map_point(&mut self, map_point: MapPoint) -> usize {
        let idx = self.map_points.len();
        self.map_points.push(map_point);
        self.observations_by_map_point.push(Vec::new());
        idx
    }

    /// Returns a mutable reference to all map points.
    pub fn map_points_mut(&mut self) -> &mut Vec<MapPoint> {
        &mut self.map_points
    }

    /// Returns a mutable reference to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut Vec<Keyframe> {
        &mut self.keyframes
    }

    /// Returns indices of non-culled map points that project inside the image frustum.
    pub fn map_points_in_frustum(
        &self,
        camera: &PinholeCamera,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
    ) -> HashSet<usize> {
        let mut visible = HashSet::new();
        for (mp_idx, mp) in self.map_points.iter().enumerate() {
            if mp.culled {
                continue;
            }
            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            if camera.project_to_image(&p_cam, 0.0, image_size).is_ok() {
                visible.insert(mp_idx);
            }
        }
        visible
    }

    /// Update `n_visible` and `n_found` counters for map points.
    pub fn update_observation_counts(
        &mut self,
        visible: &HashSet<usize>,
        matched: &[(usize, usize)],
    ) {
        let matched_set: HashSet<usize> = matched.iter().map(|&(mp_idx, _)| mp_idx).collect();

        for &mp_idx in visible {
            if let Some(mp) = self.map_points.get_mut(mp_idx) {
                mp.n_visible = mp.n_visible.saturating_add(1);
                if matched_set.contains(&mp_idx) {
                    mp.n_found = mp.n_found.saturating_add(1);
                }
            }
        }
    }

    /// Builds a local map of visible points from nearby keyframes.
    pub fn build_local_map_points(
        &self,
        tracked_matches: &[(usize, usize)],
        current_keyframe: Option<&Keyframe>,
    ) -> (Vec<MapPoint>, Vec<usize>) {
        const MAX_LOCAL_KEYFRAMES: usize = 80;

        let local_kf_indices: HashSet<usize> = self
            .local_keyframe_indices(tracked_matches, current_keyframe, MAX_LOCAL_KEYFRAMES)
            .into_iter()
            .collect();

        let mut mp_indices: HashSet<usize> = HashSet::new();
        for &(mp_idx, _) in tracked_matches {
            if self.map_points.get(mp_idx).is_some_and(|mp| !mp.culled) {
                mp_indices.insert(mp_idx);
            }
        }
        for kf in &self.keyframes {
            if !local_kf_indices.contains(&kf.frame.idx) {
                continue;
            }
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if *mp_idx < self.map_points.len() {
                    mp_indices.insert(*mp_idx);
                }
            }
        }

        let mut global_indices: Vec<usize> = mp_indices.into_iter().collect();
        global_indices.sort_unstable();

        if global_indices.len() < 4 && self.map_points.len() >= 4 {
            global_indices = (0..self.map_points.len()).collect();
        }

        let mut local_map_points = Vec::new();
        let mut active_global_indices = Vec::new();
        for idx in global_indices {
            let Some(mp) = self.map_points.get(idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            active_global_indices.push(idx);
            local_map_points.push(mp.clone());
        }
        (local_map_points, active_global_indices)
    }

    /// Port of ORB-SLAM3 `LocalMapping::MapPointCulling` for the monocular case.
    ///
    /// Evaluates recently-added map points (≤3 KFs old) and culls those that fail
    /// either the found-ratio test (found/visible < 0.25) or the observation test
    /// (age ≥ 2 KFs and observed by ≤ 2 KFs). Points that survive until they are
    /// more than 3 KFs old are "promoted" and no longer revisited here.
    pub fn map_point_culling(&mut self) -> usize {
        const MONO_MIN_OBS: usize = 2;
        const FOUND_RATIO_MIN: f64 = 0.25;
        const RECENT_WINDOW_KFS: usize = 3;

        let n_kfs = self.keyframes.len();
        if n_kfs == 0 {
            return 0;
        }
        let current_kf_order = n_kfs - 1;

        let kf_order: HashMap<usize, usize> = self
            .keyframes
            .iter()
            .enumerate()
            .map(|(i, kf)| (kf.frame.idx, i))
            .collect();
        let obs_count = self.map_point_observation_counts();

        let mut n_culled = 0usize;
        for (mp_idx, mp) in self.map_points.iter_mut().enumerate() {
            if mp.culled {
                continue;
            }
            let Some(&first_order) = kf_order.get(&mp.keyframe_idx) else {
                continue;
            };
            let age = current_kf_order.saturating_sub(first_order);
            if age > RECENT_WINDOW_KFS {
                continue;
            }

            if mp.found_ratio() < FOUND_RATIO_MIN {
                mp.mark_culled();
                n_culled += 1;
                continue;
            }

            let obs = obs_count.get(&mp_idx).copied().unwrap_or(0);
            if age >= 2 && obs <= MONO_MIN_OBS {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        if n_culled > 0 {
            let culled_set: HashSet<usize> = self
                .map_points
                .iter()
                .enumerate()
                .filter(|(_, mp)| mp.culled)
                .map(|(i, _)| i)
                .collect();
            for kf in &mut self.keyframes {
                for desc_idx in 0..kf.map_point_by_desc_idx.len() {
                    if let Some(mp) = kf.map_point(desc_idx)
                        && culled_set.contains(&mp)
                    {
                        kf.clear_map_point(desc_idx);
                    }
                }
            }
            self.rebuild_observation_index();
        }

        n_culled
    }

    /// Port of ORB-SLAM3 `LocalMapping::KeyFrameCulling` for the monocular case.
    ///
    /// Only covisible neighbors of the current (most recent) keyframe are
    /// evaluated. A keyframe is redundant if >90% of its map points are also
    /// observed by more than 3 other keyframes at the same or finer scale (+1
    /// octave slack, matching the ORB-SLAM3 code path).
    pub fn keyframe_culling(&mut self) -> usize {
        const REDUNDANT_RATIO: f64 = 0.9;
        const N_TH_OBS: usize = 3;
        if self.keyframes.len() <= 1 {
            return 0;
        }

        let current_kf_idx = self
            .keyframes
            .last()
            .map(|kf| kf.frame.idx)
            .expect("checked non-empty keyframes");
        let origin = self.origin_kf_frame_idx;

        let mut observations_by_mp: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        for kf in &self.keyframes {
            for (desc_idx, mp_idx) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = *mp_idx else {
                    continue;
                };
                let Some(mp) = self.map_points.get(mp_idx) else {
                    continue;
                };
                if mp.culled {
                    continue;
                }
                let octave = kf
                    .frame
                    .features
                    .scales
                    .get(desc_idx)
                    .copied()
                    .map(Self::keypoint_octave)
                    .unwrap_or(0);
                observations_by_mp
                    .entry(mp_idx)
                    .or_default()
                    .entry(kf.frame.idx)
                    .and_modify(|existing| *existing = (*existing).min(octave))
                    .or_insert(octave);
            }
        }

        let mut to_remove_frame_indices: Vec<usize> = Vec::new();
        for (candidate_kf_idx, _) in self.covisibility_neighbors(current_kf_idx, self.keyframes.len()) {
            let Some(kf) = self.get_keyframe(candidate_kf_idx) else {
                continue;
            };
            if Some(kf.frame.idx) == origin {
                continue;
            }

            let mut total = 0usize;
            let mut redundant_observations = 0usize;
            for (desc_idx, mp_idx) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = *mp_idx else {
                    continue;
                };
                let Some(mp) = self.map_points.get(mp_idx) else {
                    continue;
                };
                if mp.culled {
                    continue;
                }
                total += 1;
                let Some(observers) = observations_by_mp.get(&mp_idx) else {
                    continue;
                };
                if observers.len() <= N_TH_OBS {
                    continue;
                }

                let scale_level = kf
                    .frame
                    .features
                    .scales
                    .get(desc_idx)
                    .copied()
                    .map(Self::keypoint_octave)
                    .unwrap_or(0);
                let mut supporting_obs = 0usize;
                for (&observer_kf_idx, &observer_scale_level) in observers {
                    if observer_kf_idx == kf.frame.idx {
                        continue;
                    }
                    if observer_scale_level <= scale_level + 1 {
                        supporting_obs += 1;
                        if supporting_obs > N_TH_OBS {
                            break;
                        }
                    }
                }

                if supporting_obs > N_TH_OBS {
                    redundant_observations += 1;
                }
            }
            if total > 0 && (redundant_observations as f64 / total as f64) > REDUNDANT_RATIO {
                to_remove_frame_indices.push(candidate_kf_idx);
            }
        }

        if to_remove_frame_indices.is_empty() {
            return 0;
        }

        let remove_set: HashSet<_> = to_remove_frame_indices.iter().copied().collect();
        let mut affected_points = HashSet::new();
        for kf in &self.keyframes {
            if remove_set.contains(&kf.frame.idx) {
                for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                    affected_points.insert(*mp_idx);
                }
            }
        }
        self.keyframes
            .retain(|kf| !remove_set.contains(&kf.frame.idx));
        self.rebuild_observation_index();

        let mut live_obs: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                live_obs.insert(*mp_idx);
            }
        }

        for (i, mp) in self.map_points.iter_mut().enumerate() {
            if !mp.culled && !live_obs.contains(&i) {
                mp.mark_culled();
            }
        }

        self.refresh_map_points_metadata(affected_points);

        to_remove_frame_indices.len()
    }

    /// Cull map points with poor observation ratios or that project behind cameras.
    pub fn cull(&mut self) {
        const MIN_OBSERVATIONS: u32 = 5;
        const MIN_FOUND_RATIO: f64 = 0.20;

        let mut n_culled = 0usize;

        for mp in self.map_points.iter_mut() {
            if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
                continue;
            }
            if mp.found_ratio() < MIN_FOUND_RATIO {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        let mut behind_camera: Vec<usize> = Vec::new();
        for kf in &self.keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 1e-8 {
                        behind_camera.push(*mp_idx);
                    }
                }
            }
        }

        for mp_idx in &behind_camera {
            if let Some(mp) = self.map_points.get_mut(*mp_idx)
                && !mp.culled
            {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        if n_culled > 0 {
            let culled_set: HashSet<usize> = self
                .map_points
                .iter()
                .enumerate()
                .filter(|(_, mp)| mp.culled)
                .map(|(i, _)| i)
                .collect();

            for kf in &mut self.keyframes {
                for desc_idx in 0..kf.map_point_by_desc_idx.len() {
                    if let Some(mp_idx) = kf.map_point(desc_idx)
                        && culled_set.contains(&mp_idx)
                    {
                        kf.clear_map_point(desc_idx);
                    }
                }
            }
            self.rebuild_observation_index();
        }
    }

    /// Run local bundle adjustment over recent keyframes and their observed map points.
    ///
    /// Collects the last N active keyframes, gathers observations (undistorting keypoints
    /// via camera), calls `kornia_3d::ba::bundle_adjust`, and writes back optimized poses
    /// and point positions.
    pub fn run_local_ba(&mut self, camera: &PinholeCamera) {
        // Optimize only the 3 most recent KF poses; every earlier KF is a fixed
        // constraint. Observations come from ALL KFs touching the optimized
        // map-point set, so old KFs anchor the geometry even though their poses
        // don't move. Keeping the active window tiny is what makes this stable.
        const MAX_ACTIVE_KFS: usize = 3;
        const MIN_OBSERVATIONS: usize = 8;

        let n_kfs = self.keyframes.len();
        if n_kfs < 2 {
            return;
        }

        let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

        let mut mp_set: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes[active_start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    mp_set.insert(*mp_idx);
                }
            }
        }
        if mp_set.is_empty() {
            return;
        }

        let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
        mp_global_indices.sort_unstable();
        let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let points: Vec<Vec3F64> = mp_global_indices
            .iter()
            .map(|&idx| self.map_points[idx].position)
            .collect();

        let poses: Vec<Pose3d> = self
            .keyframes
            .iter()
            .map(|kf| kf.frame.pose_world_to_cam)
            .collect();

        let mut observations = Vec::new();
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
            let is_fixed = kf_idx < active_start;
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                if let Some(mp_idx) = mp_opt {
                    let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                        continue;
                    };
                    if let Some(kp) = kf.frame.features.keypoints_xy.get(desc_idx) {
                        let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                        observations.push(BaObservation {
                            pose_idx: kf_idx,
                            point_idx,
                            pixel: [p.x as f32, p.y as f32],
                            fixed_pose: is_fixed,
                        });
                    }
                }
            }
        }

        if observations.len() < MIN_OBSERVATIONS {
            return;
        }

        let ba_result =
            match ba::bundle_adjust(&poses, &points, &observations, camera, &BaParams::default()) {
                Ok(r) => r,
                Err(_) => return,
            };

        for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
            if kf_idx >= active_start {
                self.keyframes[kf_idx].frame.pose_world_to_cam = *pose;
            }
        }

        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = ba_result.points[local_idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        let keypoints_xy = (0..n).map(|i| [i as f32, i as f32]).collect();
        test_frame_with_keypoints(idx, descriptors, keypoints_xy)
    }

    fn test_frame_with_keypoints(
        idx: usize,
        descriptors: Vec<[u8; 32]>,
        keypoints_xy: Vec<[f32; 2]>,
    ) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy,
                scales: vec![1.0; n],
                orientations: vec![0.0; n],
                descriptors,
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
        }
    }

    fn test_frame_with_scales(idx: usize, descriptors: Vec<[u8; 32]>, scales: Vec<f32>) -> Frame {
        let n = descriptors.len();
        assert_eq!(scales.len(), n);
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                scales,
                orientations: vec![0.0; n],
                descriptors,
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
        }
    }

    fn test_frame_with_pose(
        idx: usize,
        descriptors: Vec<[u8; 32]>,
        translation_world_to_cam: Vec3F64,
    ) -> Frame {
        let mut frame = test_frame(idx, descriptors);
        frame.pose_world_to_cam = Pose3d::new(Mat3F64::IDENTITY, translation_world_to_cam);
        frame
    }

    #[test]
    fn keyframe_from_frame_initializes_map_point_slots() {
        let keyframe = Keyframe::from_frame(test_frame(7, vec![[0u8; 32], [1u8; 32], [2u8; 32]]));

        assert_eq!(keyframe.frame.idx, 7);
        assert_eq!(keyframe.map_point_by_desc_idx.len(), 3);
        assert!(
            keyframe
                .map_point_by_desc_idx
                .iter()
                .all(|slot| slot.is_none())
        );
    }

    #[test]
    fn keyframe_association_helpers_work() {
        let mut keyframe = Keyframe::from_frame(test_frame(1, vec![[0u8; 32], [1u8; 32]]));

        keyframe.associate_map_point(1, 42);
        assert_eq!(keyframe.map_point(1), Some(42));
        assert_eq!(keyframe.num_associated_points(), 1);

        keyframe.clear_map_point(1);
        assert_eq!(keyframe.map_point(1), None);
        assert_eq!(keyframe.num_associated_points(), 0);
    }

    #[test]
    fn map_point_new_sets_active_defaults() {
        let mp = MapPoint::new(Vec3F64::new(1.0, 2.0, 3.0), [9u8; 32], [0; 3], 5);

        assert_eq!(mp.position, Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(mp.descriptor, [9u8; 32]);
        assert_eq!(mp.keyframe_idx, 5);
        assert_eq!(mp.n_visible, 1);
        assert_eq!(mp.n_found, 1);
        assert!(!mp.culled);
    }

    #[test]
    fn map_point_tracking_helpers_work() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], [0; 3], 0);
        mp.n_visible = 10;
        mp.n_found = 4;

        assert!((mp.found_ratio() - 0.4).abs() < 1e-9);
        mp.mark_culled();
        assert!(mp.culled);
    }

    #[test]
    fn upsert_keyframe_replaces_existing_idx() {
        let mut map = Map::new();

        map.upsert_keyframe(Keyframe::from_frame(test_frame(
            10,
            vec![[0u8; 32], [1u8; 32]],
        )));
        assert_eq!(map.keyframes().len(), 1);

        map.upsert_keyframe(Keyframe::from_frame(test_frame(10, vec![[2u8; 32]])));

        assert_eq!(map.keyframes().len(), 1);
        assert_eq!(
            map.get_keyframe(10)
                .expect("expected keyframe with idx 10")
                .frame
                .features
                .descriptors
                .len(),
            1
        );
    }

    #[test]
    fn push_map_point_returns_sequential_index() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 1.0),
            [0u8; 32],
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 1.0),
            [1u8; 32],
            [0; 3],
            0,
        ));

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.num_map_points(), 2);
    }

    #[test]
    fn covisibility_neighbors_are_sorted_by_shared_observations() {
        let mut map = Map::new();
        let mp0 = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            [0; 3],
            0,
        ));
        let mp1 = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            [0; 3],
            0,
        ));

        let mut kf0 = Keyframe::from_frame(test_frame(0, vec![[0u8; 32], [1u8; 32]]));
        kf0.associate_map_point(0, mp0);
        kf0.associate_map_point(1, mp1);
        let mut kf1 = Keyframe::from_frame(test_frame(1, vec![[0u8; 32], [1u8; 32]]));
        kf1.associate_map_point(0, mp0);
        kf1.associate_map_point(1, mp1);
        let mut kf2 = Keyframe::from_frame(test_frame(2, vec![[1u8; 32]]));
        kf2.associate_map_point(0, mp1);

        map.upsert_keyframe(kf0);
        map.upsert_keyframe(kf1);
        map.upsert_keyframe(kf2);

        assert_eq!(map.covisibility_neighbors(0, 10), vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn local_map_points_include_covisible_neighbor_points() {
        let mut map = Map::new();
        let mp0 = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            [0; 3],
            0,
        ));
        let mp1 = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            [0; 3],
            1,
        ));

        let mut current = Keyframe::from_frame(test_frame(0, vec![[0u8; 32]]));
        current.associate_map_point(0, mp0);
        let mut neighbor = Keyframe::from_frame(test_frame(1, vec![[0u8; 32], [1u8; 32]]));
        neighbor.associate_map_point(0, mp0);
        neighbor.associate_map_point(1, mp1);

        map.upsert_keyframe(current.clone());
        map.upsert_keyframe(neighbor);

        let (_, indices) = map.build_local_map_points(&[(mp0, 0)], Some(&current));

        assert_eq!(indices, vec![mp0, mp1]);
    }

    #[test]
    fn local_map_points_keep_global_indices_aligned_after_culling() {
        let mut map = Map::new();
        for i in 0..4 {
            map.push_map_point(MapPoint::new(
                Vec3F64::new(i as f64, 0.0, 5.0),
                [i as u8; 32],
                [0; 3],
                0,
            ));
        }
        map.map_points_mut()[0].mark_culled();
        map.map_points_mut()[2].mark_culled();

        let (points, indices) = map.build_local_map_points(&[], None);

        assert_eq!(points.len(), 2);
        assert_eq!(indices, vec![1, 3]);
        assert_eq!(points[0].descriptor, [1u8; 32]);
        assert_eq!(points[1].descriptor, [3u8; 32]);
    }

    #[test]
    fn observation_index_tracks_stored_keyframe_associations() {
        let mut map = Map::new();
        let mp = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [1u8; 32],
            [0; 3],
            0,
        ));

        let mut kf0 = Keyframe::from_frame(test_frame(0, vec![[1u8; 32]]));
        kf0.associate_map_point(0, mp);
        map.upsert_keyframe(kf0);
        map.upsert_keyframe(Keyframe::from_frame(test_frame(1, vec![[2u8; 32]])));

        assert_eq!(map.map_point_observation_count(mp), 1);
        assert_eq!(map.map_point_observations(mp), &[(0, 0)]);

        assert!(map.associate_keyframe_map_point(1, 0, mp));
        assert_eq!(map.map_point_observation_count(mp), 2);
        assert_eq!(map.map_point_observations(mp), &[(0, 0), (1, 0)]);
    }

    #[test]
    fn descriptor_refresh_picks_median_observer_descriptor() {
        let mut map = Map::new();
        let mp = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0x00; 32],
            [0; 3],
            0,
        ));

        let descriptor_a = [0x00; 32];
        let descriptor_b = [0x0F; 32];
        let descriptor_c = [0xFF; 32];

        let mut kf0 = Keyframe::from_frame(test_frame_with_pose(
            0,
            vec![descriptor_a],
            Vec3F64::ZERO,
        ));
        kf0.associate_map_point(0, mp);
        map.upsert_keyframe(kf0);

        let mut kf1 = Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![descriptor_b],
            Vec3F64::new(-1.0, 0.0, 0.0),
        ));
        kf1.associate_map_point(0, mp);
        map.upsert_keyframe(kf1);

        // Two observers tie on median distance, so the original descriptor stays.
        assert_eq!(map.map_points()[mp].descriptor, descriptor_a);

        map.upsert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            2,
            vec![descriptor_c],
            Vec3F64::new(1.0, 0.0, 0.0),
        )));
        assert!(map.associate_keyframe_map_point(2, 0, mp));

        assert_eq!(map.map_points()[mp].descriptor, descriptor_b);
        assert_eq!(map.map_point_observation_count(mp), 3);
        assert!(map.map_points()[mp].viewing_normal().z > 0.0);
        assert!(map.map_points()[mp].max_distance > map.map_points()[mp].min_distance);
    }

    #[test]
    fn fuse_projected_map_points_adds_new_observation() {
        let mut map = Map::new();
        let mp = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [7u8; 32],
            [0; 3],
            0,
        ));
        let kf = Keyframe::from_frame(test_frame_with_keypoints(
            0,
            vec![[7u8; 32]],
            vec![[320.0, 240.0]],
        ));
        map.upsert_keyframe(kf);

        let fused = map.fuse_projected_map_points_into_keyframe(0, &[mp], &test_camera());

        assert_eq!(fused, 1);
        assert_eq!(map.get_keyframe(0).and_then(|kf| kf.map_point(0)), Some(mp));
    }

    #[test]
    fn fuse_projected_map_points_replaces_weaker_duplicate() {
        let mut map = Map::new();
        let survivor = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [9u8; 32],
            [0; 3],
            0,
        ));
        let duplicate = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [9u8; 32],
            [0; 3],
            2,
        ));

        let mut kf0 = Keyframe::from_frame(test_frame(0, vec![[9u8; 32]]));
        kf0.associate_map_point(0, survivor);
        let mut kf1 = Keyframe::from_frame(test_frame(1, vec![[9u8; 32]]));
        kf1.associate_map_point(0, survivor);
        let mut target = Keyframe::from_frame(test_frame_with_keypoints(
            2,
            vec![[9u8; 32]],
            vec![[320.0, 240.0]],
        ));
        target.associate_map_point(0, duplicate);

        map.upsert_keyframe(kf0);
        map.upsert_keyframe(kf1);
        map.upsert_keyframe(target);

        let fused = map.fuse_projected_map_points_into_keyframe(2, &[survivor], &test_camera());

        assert_eq!(fused, 1);
        assert!(map.map_points()[duplicate].culled);
        assert_eq!(
            map.get_keyframe(2).and_then(|kf| kf.map_point(0)),
            Some(survivor)
        );
    }

    #[test]
    fn cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            [0; 3],
            0,
        ));
        map.map_points_mut()[first_idx].n_visible = 10;
        map.map_points_mut()[first_idx].n_found = 1;
        map.map_points_mut()[second_idx].n_visible = 10;
        map.map_points_mut()[second_idx].n_found = 5;

        map.cull();

        assert!(map.map_points()[first_idx].culled);
        assert!(!map.map_points()[second_idx].culled);
    }

    #[test]
    fn keyframe_culling_only_evaluates_local_covisibility_neighbors() {
        let mut map = Map::new();
        let mut shared_mp_indices = Vec::new();
        for i in 0..20 {
            shared_mp_indices.push(map.push_map_point(MapPoint::new(
                Vec3F64::new(i as f64 * 0.01, 0.0, 5.0),
                [i as u8; 32],
                [0; 3],
                0,
            )));
        }

        let mut remote_mp_indices = Vec::new();
        for i in 20..40 {
            remote_mp_indices.push(map.push_map_point(MapPoint::new(
                Vec3F64::new(i as f64 * 0.01, 0.0, 5.0),
                [i as u8; 32],
                [0; 3],
                10,
            )));
        }

        for kf_idx in 0..1 {
            let descriptors: Vec<[u8; 32]> = (0..20).map(|i| [i as u8; 32]).collect();
            let mut keyframe = Keyframe::from_frame(test_frame(kf_idx, descriptors));
            for (desc_idx, &mp_idx) in shared_mp_indices.iter().enumerate() {
                keyframe.associate_map_point(desc_idx, mp_idx);
            }
            map.upsert_keyframe(keyframe);
        }

        // Keyframe 10 is redundant within its own disconnected component, but it
        // is not covisible with the current keyframe (4), so ORB-SLAM3-style
        // local culling should not touch it.
        for kf_idx in 10..14 {
            let descriptors: Vec<[u8; 32]> = (20..40).map(|i| [i as u8; 32]).collect();
            let mut keyframe = Keyframe::from_frame(test_frame(kf_idx, descriptors));
            for (desc_idx, &mp_idx) in remote_mp_indices.iter().enumerate() {
                keyframe.associate_map_point(desc_idx, mp_idx);
            }
            map.upsert_keyframe(keyframe);
        }

        for kf_idx in 1..5 {
            let descriptors: Vec<[u8; 32]> = (0..20).map(|i| [i as u8; 32]).collect();
            let mut keyframe = Keyframe::from_frame(test_frame(kf_idx, descriptors));
            for (desc_idx, &mp_idx) in shared_mp_indices.iter().enumerate() {
                keyframe.associate_map_point(desc_idx, mp_idx);
            }
            map.upsert_keyframe(keyframe);
        }

        let removed = map.keyframe_culling();

        assert_eq!(removed, 3);
        assert!(map.get_keyframe(1).is_none());
        assert!(map.get_keyframe(2).is_none());
        assert!(map.get_keyframe(3).is_none());
        assert!(map.get_keyframe(0).is_some());
        assert!(map.get_keyframe(10).is_some());
        assert!(map.get_keyframe(11).is_some());
        assert!(map.get_keyframe(12).is_some());
        assert!(map.get_keyframe(13).is_some());
        assert_eq!(map.keyframes().len(), 6);
    }

    #[test]
    fn keyframe_culling_requires_same_or_finer_scale_support() {
        let mut map = Map::new();
        let mut mp_indices = Vec::new();
        for i in 0..20 {
            mp_indices.push(map.push_map_point(MapPoint::new(
                Vec3F64::new(i as f64 * 0.01, 0.0, 5.0),
                [i as u8; 32],
                [0; 3],
                0,
            )));
        }

        let base_descriptors: Vec<[u8; 32]> = (0..20).map(|i| [i as u8; 32]).collect();
        let mut origin = Keyframe::from_frame(test_frame_with_scales(
            0,
            base_descriptors.clone(),
            vec![1.0; 20],
        ));
        for (desc_idx, &mp_idx) in mp_indices.iter().enumerate() {
            origin.associate_map_point(desc_idx, mp_idx);
        }
        map.upsert_keyframe(origin);

        let mut candidate = Keyframe::from_frame(test_frame_with_scales(
            1,
            base_descriptors.clone(),
            vec![1.0; 20],
        ));
        for (desc_idx, &mp_idx) in mp_indices.iter().enumerate() {
            candidate.associate_map_point(desc_idx, mp_idx);
        }
        map.upsert_keyframe(candidate);

        for kf_idx in 2..5 {
            let mut supporting = Keyframe::from_frame(test_frame_with_scales(
                kf_idx,
                base_descriptors.clone(),
                vec![1.2_f32.powi(4); 20],
            ));
            for (desc_idx, &mp_idx) in mp_indices.iter().enumerate() {
                supporting.associate_map_point(desc_idx, mp_idx);
            }
            map.upsert_keyframe(supporting);
        }

        let removed = map.keyframe_culling();

        assert_eq!(removed, 2);
        assert!(map.get_keyframe(1).is_some());
        assert!(map.get_keyframe(2).is_none());
        assert!(map.get_keyframe(3).is_none());
    }
}
