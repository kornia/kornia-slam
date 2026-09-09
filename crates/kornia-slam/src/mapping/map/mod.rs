//! Map storage and the operations that read and change it.
//!
//! [`Map`] owns the keyframes, map points, IMU factors and the world epoch;
//! nothing else stores them. Entities live beside it in [`keyframe`] and
//! [`map_point`], and every operation is an inherent `impl Map` block under the
//! private `ops` module, grouped by responsibility:
//!
//! | Group | Examples |
//! |---|---|
//! | `mutation` | `insert_keyframe`, `insert_landmark`, `link_observation`, `apply_insertion`, `merge_map_points` |
//! | `correction` | `apply_pose_graph_correction`, `apply_inertial_alignment`, `scale_world`, `apply_ba_update` |
//! | `queries` | `covisible_keyframes` — raw weights; thresholds belong to consumers |
//! | `snapshot` | `ba_snapshot` — owned, immutable optimization inputs |
//!
//! ## Invariants
//!
//! A link exists on both sides or neither: a keyframe feature holding landmark
//! `L` implies `L` holds a matching [`LandmarkObservation`], and vice versa.
//! Within one keyframe, a feature holds at most one landmark and a landmark is
//! seen through at most one feature. Conflicts with either rule are refused
//! with a typed error; re-stating a link that already exists is a no-op, not an
//! error.
//!
//! Deletion is logical. `remove_landmark` clears every referencing feature,
//! empties the landmark's records and zeroes its derived geometry, but leaves
//! the slot in place — landmark ids are stable indices held inside keyframes,
//! so the vector is never compacted and a retired id is never reused.
//! Unlinking the last observation retires the landmark; unlinking the reference
//! observation promotes the smallest remaining `(keyframe, feature)` in its
//! place.
//!
//! Ids are valid for the current map lifetime only. `clear_active` is a whole-
//! map reset that invalidates every previous id and advances the world epoch;
//! ids are not unique across resets.
//!
//! [`Map::apply_insertion`] validates a whole batch — against the live map and
//! against the request's own claims — before its first write, so a rejected
//! request leaves entities, links, counters, factors and the epoch untouched.
//! That is validation-before-write, not rollback: it does not survive a panic,
//! and the map is never cloned to provide it. Single-operation methods share
//! the same primitives, so the two cannot drift apart.
//!
//! Callers are unaffected by the grouping: the methods stay inherent on `Map`,
//! so `map.keyframes()` and `map::MapPoint` resolve exactly as before. Culling
//! policy lives in [`crate::mapping::culling`], BA selection and solving in
//! [`crate::mapping::bundle_adjustment`], and scheduling in
//! [`crate::mapping::local_mapping`].

mod imu_factor;
mod keyframe;
mod map_point;
mod ops;

pub use imu_factor::ImuFactor;
pub use keyframe::Keyframe;
// Compatibility: the worker now lives in `mapping`, but callers still reach it
// through the map facade.
pub use crate::mapping::local_mapping::{KeyframeJob, LocalMapping, LocalMappingMode};
pub use map_point::{LandmarkObservation, MapPoint, ObservationKey};
pub use ops::{
    BaSnapshot, BaUpdate, InertialAlignment, InertialAlignmentError, KeyframeBaCorrection,
    KeyframeVelocity, LocalBaMergeResult, MapPointMergeResult, PoseGraphCorrectionError,
    PoseGraphCorrectionResult,
};
pub use ops::{
    InsertionResult, LandmarkSeed, LandmarkTarget, MapInsertion, MapMutationError, ObservationLink,
};

/// ORB pyramid scale factor between adjacent levels (matches the kornia-imgproc
/// ORB extractor default `downscale = 1.2`, which equals ORB-SLAM3's default).
pub const ORB_SCALE_FACTOR: f64 = 1.2;
/// Number of ORB pyramid levels (matches `OrbDetector` default `n_scales = 8`).
pub const ORB_N_LEVELS: usize = 8;

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
    imu_factors: Vec<ImuFactor>,
    // Incremented whenever the map's world coordinate frame changes. Local BA
    // snapshots from an older epoch must never be merged into the new frame.
    world_epoch: u64,
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

    /// Numeric writeback over stored keyframes, for correction and bundle
    /// adjustment.
    ///
    /// Private to this module tree — `map` and its descendants. Everything
    /// outside changes structure only through the canonical operations, which
    /// keep both sides of every link in step.
    fn keyframes_mut(&mut self) -> &mut [Keyframe] {
        &mut self.keyframes
    }

    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }

    /// Mutable access to one stored keyframe, for numeric writeback. Private
    /// for the same reason as [`Map::keyframes_mut`].
    fn get_keyframe_mut(&mut self, idx: usize) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|kf| kf.frame.idx == idx)
    }

    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Numeric writeback over stored landmarks.
    ///
    /// Private, like the keyframe accessors above. A slice rather than the
    /// `Vec` narrows the damage — no push or remove — but does not prevent it:
    /// `swap` and `sort` would still reassign landmark identities without
    /// repairing the associations that index them, so this stays inside the
    /// module tree where every caller is auditable.
    fn map_points_mut(&mut self) -> &mut [MapPoint] {
        &mut self.map_points
    }

    /// Returns the number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }

    /// Returns the number of non-culled (active) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map_points.iter().filter(|mp| !mp.culled).count()
    }

    /// Returns all keyframe-to-keyframe IMU factors in insertion order.
    pub fn imu_factors(&self) -> &[ImuFactor] {
        &self.imu_factors
    }
}

#[cfg(test)]
pub(crate) mod tests;
