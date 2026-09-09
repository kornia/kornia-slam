//! Owned inputs and numerical updates for optimization outside the map lock.
//!
//! Capture has no window or solver policy. Mapping selects its optimization
//! problem from these immutable inputs; correction applies the numeric result.

use crate::map::{ImuFactor, Keyframe, Map, MapPoint};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

/// Consistent map inputs captured under the caller's map lock.
///
/// This currently copies all entities, preserving insertion order and stable
/// landmark slots. Window selection belongs to `mapping::bundle_adjustment`.
#[derive(Debug, Clone)]
pub struct BaSnapshot {
    pub(super) keyframes: Vec<Keyframe>,
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
    // Initial BA also refreshes selected points whose positions did not move.
    pub(crate) refresh_points: Vec<usize>,
    pub(crate) snapshot: BaSnapshot,
}

impl BaSnapshot {
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
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
            refresh_points: Vec::new(),
            snapshot: self,
        }
    }
}

impl Map {
    /// Captures owned inputs for BA without selecting or solving a problem.
    pub fn ba_snapshot(&self) -> BaSnapshot {
        BaSnapshot {
            keyframes: self.keyframes.clone(),
            map_points: self.map_points.clone(),
            imu_factors: self.imu_factors.clone(),
            world_epoch: self.world_epoch,
        }
    }
}
