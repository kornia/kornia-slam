//! Applying optimizer results back to the map: BA snapshots, merges and pose-graph corrections.

use crate::map::{Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::SO3F64;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuBias;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
struct KeyframeBaState {
    idx: usize,
    pose_world_to_cam: Pose3d,
    velocity_world: Vec3F64,
    imu_bias: ImuBias,
}

/// A private copy of the map on which local bundle adjustment can run without
/// holding the live map lock.
#[derive(Debug, Clone)]
pub struct LocalBaSnapshot {
    pub(in crate::map) optimized: Map,
    world_epoch: u64,
    keyframes_before: Vec<KeyframeBaState>,
    map_points_before: Vec<Vec3F64>,
}
impl LocalBaSnapshot {
    /// Optimizes the visual local window in this private snapshot.
    pub fn run_visual(&mut self, camera: &PinholeCamera) {
        self.optimized.run_local_ba(camera);
    }

    /// Optimizes the visual-inertial local window in this private snapshot.
    pub fn run_inertial(
        &mut self,
        camera: &PinholeCamera,
        imu_t_bc: Option<Pose3d>,
        gravity_world: Vec3F64,
    ) {
        self.optimized
            .run_local_inertial_ba(camera, imu_t_bc, gravity_world);
    }
}
/// A keyframe state change accepted from an asynchronous local BA result.
#[derive(Debug, Clone, Copy)]
pub struct KeyframeBaCorrection {
    pub kf_idx: usize,
    pub pose_before: Pose3d,
    pub pose_after: Pose3d,
    pub velocity_world: Vec3F64,
    pub imu_bias: ImuBias,
}
/// Changes accepted while merging an asynchronous local BA snapshot.
#[derive(Debug, Default)]
pub struct LocalBaMergeResult {
    pub keyframe_corrections: Vec<KeyframeBaCorrection>,
    pub map_points_updated: usize,
}
/// Geometry changed by an accepted global pose-graph correction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoseGraphCorrectionResult {
    pub keyframes_corrected: usize,
    pub map_points_corrected: usize,
}
/// Refusal to apply a pose-graph result to the live map.
#[derive(Debug, thiserror::Error)]
pub enum PoseGraphCorrectionError {
    #[error("pose graph snapshot lengths do not match the live map")]
    LengthMismatch,
    #[error("pose graph snapshot no longer matches the live map")]
    StaleSnapshot,
    #[error("pose graph contains a non-finite optimized pose")]
    NonFinitePose,
    #[error("map point references a keyframe outside the pose graph")]
    MissingReferenceKeyframe,
}

impl Map {
    /// Creates a private local-BA snapshot. The returned value owns all solver
    /// inputs and can be optimized without holding a lock on this map.
    pub fn local_ba_snapshot(&self) -> LocalBaSnapshot {
        LocalBaSnapshot {
            optimized: self.clone(),
            world_epoch: self.world_epoch,
            keyframes_before: self
                .keyframes
                .iter()
                .map(|kf| KeyframeBaState {
                    idx: kf.frame.idx,
                    pose_world_to_cam: kf.frame.pose_world_to_cam,
                    velocity_world: kf.velocity_world,
                    imu_bias: kf.imu_bias,
                })
                .collect(),
            map_points_before: self.map_points.iter().map(|mp| mp.position).collect(),
        }
    }
    /// Merges solver-changed geometry from a local-BA snapshot.
    ///
    /// Entities inserted after the snapshot are left untouched. A snapshot is
    /// rejected wholesale if the live map has since been scaled, rotated, or
    /// cleared, because its coordinates then belong to another world frame.
    pub fn merge_local_ba_snapshot(
        &mut self,
        snapshot: LocalBaSnapshot,
    ) -> Option<LocalBaMergeResult> {
        if self.world_epoch != snapshot.world_epoch {
            return None;
        }

        let mut result = LocalBaMergeResult::default();
        for (before, optimized) in snapshot
            .keyframes_before
            .iter()
            .zip(snapshot.optimized.keyframes.iter())
        {
            if !keyframe_ba_state_changed(before, optimized) {
                continue;
            }
            let Some(live) = self.get_keyframe_mut(before.idx) else {
                continue;
            };

            let pose_before = live.frame.pose_world_to_cam;
            live.frame.pose_world_to_cam = optimized.frame.pose_world_to_cam;
            live.velocity_world = optimized.velocity_world;
            live.imu_bias = optimized.imu_bias;
            result.keyframe_corrections.push(KeyframeBaCorrection {
                kf_idx: before.idx,
                pose_before,
                pose_after: optimized.frame.pose_world_to_cam,
                velocity_world: optimized.velocity_world,
                imu_bias: optimized.imu_bias,
            });
        }

        let mut changed_points = Vec::new();
        for (idx, (&before, optimized)) in snapshot
            .map_points_before
            .iter()
            .zip(snapshot.optimized.map_points.iter())
            .enumerate()
        {
            if optimized.position == before {
                continue;
            }
            let Some(live) = self.map_points.get_mut(idx) else {
                continue;
            };
            if live.culled {
                continue;
            }
            live.position = optimized.position;
            changed_points.push(idx);
        }
        result.map_points_updated = changed_points.len();

        for idx in changed_points {
            self.update_map_point_geometry(idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }

        // VI-BA may repropagate an existing preintegration on its private copy.
        // Preserve newer live factors while copying those refreshed edges back.
        for optimized in &snapshot.optimized.imu_factors {
            if let Some(live) = self.imu_factors.iter_mut().find(|live| {
                live.prev_kf_idx == optimized.prev_kf_idx
                    && live.curr_kf_idx == optimized.curr_kf_idx
            }) {
                live.preintegrated = optimized.preintegrated.clone();
            }
        }
        self.cull();
        Some(result)
    }
    /// Atomically applies a pose-graph snapshot and transports landmarks through
    /// their reference keyframes so their reference-camera coordinates stay fixed.
    pub fn apply_pose_graph_correction(
        &mut self,
        keyframe_indices: &[usize],
        poses_before: &[Pose3d],
        poses_after: &[Pose3d],
    ) -> Result<PoseGraphCorrectionResult, PoseGraphCorrectionError> {
        let node_count = self.keyframes.len();
        if keyframe_indices.len() != node_count
            || poses_before.len() != node_count
            || poses_after.len() != node_count
        {
            return Err(PoseGraphCorrectionError::LengthMismatch);
        }
        if self
            .keyframes
            .iter()
            .zip(keyframe_indices)
            .zip(poses_before)
            .any(|((keyframe, &idx), &before)| {
                keyframe.frame.idx != idx || keyframe.frame.pose_world_to_cam != before
            })
        {
            return Err(PoseGraphCorrectionError::StaleSnapshot);
        }
        if poses_after.iter().any(|pose| {
            !pose.translation.x.is_finite()
                || !pose.translation.y.is_finite()
                || !pose.translation.z.is_finite()
                || pose
                    .rotation
                    .to_cols_array()
                    .into_iter()
                    .any(|value| !value.is_finite())
        }) {
            return Err(PoseGraphCorrectionError::NonFinitePose);
        }

        let node_by_keyframe: HashMap<_, _> = keyframe_indices
            .iter()
            .enumerate()
            .map(|(node, &idx)| (idx, node))
            .collect();
        if self
            .map_points
            .iter()
            .any(|point| !point.culled && !node_by_keyframe.contains_key(&point.keyframe_idx))
        {
            return Err(PoseGraphCorrectionError::MissingReferenceKeyframe);
        }

        let world_corrections: Vec<_> = poses_before
            .iter()
            .zip(poses_after)
            .map(|(&before, after)| after.inverse().compose(&before))
            .collect();
        let keyframes_corrected = poses_before
            .iter()
            .zip(poses_after)
            .filter(|(before, after)| before != after)
            .count();

        self.world_epoch = self.world_epoch.wrapping_add(1);
        for (node, keyframe) in self.keyframes.iter_mut().enumerate() {
            keyframe.frame.pose_world_to_cam = poses_after[node];
            keyframe.velocity_world = world_corrections[node].rotation * keyframe.velocity_world;
        }

        let mut changed_points = Vec::new();
        for (idx, point) in self.map_points.iter_mut().enumerate() {
            if point.culled {
                continue;
            }
            let node = node_by_keyframe[&point.keyframe_idx];
            let corrected = world_corrections[node].transform_point(&point.position);
            if corrected != point.position {
                point.position = corrected;
                changed_points.push(idx);
            }
        }
        for &idx in &changed_points {
            self.update_map_point_geometry(idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }

        Ok(PoseGraphCorrectionResult {
            keyframes_corrected,
            map_points_corrected: changed_points.len(),
        })
    }
    /// Applies a metric scale to camera centers and map points.
    pub fn scale_world(&mut self, scale: f64) {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        for kf in &mut self.keyframes {
            let mut cam_to_world = kf.frame.pose_world_to_cam.inverse();
            cam_to_world.translation *= scale;
            kf.frame.pose_world_to_cam = cam_to_world.inverse();
        }

        for mp in &mut self.map_points {
            mp.position *= scale;
        }
    }
    pub fn rotate_world(&mut self, r: &SO3F64) {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        for kf in self.keyframes_mut() {
            let cam_to_world = kf.frame.pose_world_to_cam.inverse();
            let new_translation = *r * cam_to_world.translation;
            let rot_so3 = SO3F64::from_matrix(&cam_to_world.rotation);
            let new_rot_so3 = *r * rot_so3;
            let new_rotation = new_rot_so3.matrix();
            kf.frame.pose_world_to_cam = Pose3d::from_rt(new_rotation, new_translation).inverse();
        }

        for mp in self.map_points_mut() {
            if !mp.culled {
                mp.position = *r * mp.position;
            }
        }
    }
}

fn keyframe_ba_state_changed(before: &KeyframeBaState, after: &Keyframe) -> bool {
    before.pose_world_to_cam != after.frame.pose_world_to_cam
        || before.velocity_world != after.velocity_world
        || before.imu_bias.gyro != after.imu_bias.gyro
        || before.imu_bias.accel != after.imu_bias.accel
}
