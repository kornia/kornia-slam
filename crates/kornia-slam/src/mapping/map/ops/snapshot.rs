//! Owned inputs and numerical updates for optimization outside the map lock.
//!
//! Capture has no window or solver policy: mapping chooses a [`BaWindow`], the
//! map copies exactly those entities, and correction applies the numeric result.

use crate::mapping::map::{ImuFactor, Keyframe, Map, MapPoint};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};
use std::collections::HashSet;

/// The entities a BA capture copies, chosen by `mapping::bundle_adjustment`.
#[derive(Debug, Clone, Default)]
pub struct BaWindow {
    /// Keyframe slots in [`Map::keyframes`] order.
    pub keyframe_slots: Vec<usize>,
    /// Landmark ids.
    pub landmark_ids: Vec<usize>,
}

/// Consistent map inputs captured under the caller's map lock.
///
/// Holds only the entities of one [`BaWindow`], so capture cost follows the
/// window rather than the map. Keyframes keep map insertion order; landmarks
/// are sorted by id. IMU edges are those joining two captured keyframes.
#[derive(Debug, Clone)]
pub struct BaSnapshot {
    pub(super) keyframes: Vec<Keyframe>,
    pub(super) landmark_ids: Vec<usize>,
    pub(super) map_points: Vec<MapPoint>,
    pub(super) imu_factors: Vec<ImuFactor>,
    pub(super) world_epoch: u64,
}

/// Estimated quantities for a captured keyframe; identity stays in the snapshot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KeyframeBaState {
    pub pose_world_to_cam: Pose3d,
    pub velocity_world: Vec3F64,
    pub imu_bias: ImuBias,
}

/// Numerical output tied to its immutable capture and world epoch.
///
/// Only numeric estimates and refreshed preintegrations can be published;
/// observations, landmark lifetimes and other live structure are never copied
/// back. An unchanged update is still an accepted local-mapping completion.
#[derive(Debug)]
pub struct BaUpdate {
    pub(crate) keyframes: Vec<KeyframeBaState>,
    pub(crate) map_points: Vec<Vec3F64>,
    pub(crate) imu_preintegrations: Vec<PreintegratedImu>,
    pub(crate) snapshot: BaSnapshot,
}

impl BaSnapshot {
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Captured landmarks, parallel to [`BaSnapshot::landmark_ids`].
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Map ids of the captured landmarks, ascending.
    pub fn landmark_ids(&self) -> &[usize] {
        &self.landmark_ids
    }

    /// Snapshot slot of landmark `id`, if captured.
    pub fn landmark_slot(&self, id: usize) -> Option<usize> {
        self.landmark_ids.binary_search(&id).ok()
    }

    pub fn imu_factors(&self) -> &[ImuFactor] {
        &self.imu_factors
    }

    /// Starts an unchanged numeric result, including the original IMU edges.
    /// Solvers replace estimates in this result without changing their inputs.
    pub(crate) fn into_update(self) -> BaUpdate {
        BaUpdate {
            keyframes: self
                .keyframes
                .iter()
                .map(|kf| KeyframeBaState {
                    pose_world_to_cam: kf.frame.pose_world_to_cam,
                    velocity_world: kf.velocity_world,
                    imu_bias: kf.imu_bias,
                })
                .collect(),
            map_points: self.map_points.iter().map(|mp| mp.position).collect(),
            imu_preintegrations: self
                .imu_factors
                .iter()
                .map(|factor| factor.preintegrated.clone())
                .collect(),
            snapshot: self,
        }
    }
}

impl Map {
    /// Captures owned copies of the window's entities without selecting or
    /// solving a problem. Out-of-range slots and ids are skipped.
    pub fn ba_snapshot(&self, window: &BaWindow) -> BaSnapshot {
        let mut keyframe_slots = window.keyframe_slots.clone();
        keyframe_slots.sort_unstable();
        keyframe_slots.dedup();
        let keyframes: Vec<Keyframe> = keyframe_slots
            .iter()
            .filter_map(|&slot| self.keyframes.get(slot).cloned())
            .collect();

        let mut landmark_ids = window.landmark_ids.clone();
        landmark_ids.sort_unstable();
        landmark_ids.dedup();
        landmark_ids.retain(|&id| id < self.map_points.len());
        let map_points = landmark_ids
            .iter()
            .map(|&id| self.map_points[id].clone())
            .collect();

        let captured: HashSet<usize> = keyframes.iter().map(|kf| kf.frame.idx).collect();
        let imu_factors = self
            .imu_factors
            .iter()
            .filter(|f| captured.contains(&f.prev_kf_idx) && captured.contains(&f.curr_kf_idx))
            .cloned()
            .collect();

        BaSnapshot {
            keyframes,
            landmark_ids,
            map_points,
            imu_factors,
            world_epoch: self.world_epoch,
        }
    }

    /// Captures every entity; for tests that index updates by map slot.
    #[cfg(test)]
    pub(crate) fn full_ba_snapshot(&self) -> BaSnapshot {
        self.ba_snapshot(&BaWindow {
            keyframe_slots: (0..self.keyframes.len()).collect(),
            landmark_ids: (0..self.map_points.len()).collect(),
        })
    }
}
