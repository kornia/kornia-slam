//! Whole-map corrections: pose-graph writeback, inertial alignment, and the
//! scale/rotation helpers they build on. Each keeps its own validation order
//! and world-epoch handling.

use crate::map::{Map, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::imu::ImuBias;
use std::collections::{HashMap, HashSet};

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeVelocity {
    pub keyframe_idx: usize,
    pub velocity_world: Vec3F64,
}

/// A validated inertial alignment to apply to the whole map: a metric scale, a
/// gravity-aligning world rotation, and the per-keyframe velocity/bias writes
/// that go with them.
#[derive(Debug, Clone)]
pub struct InertialAlignment {
    pub scale: f64,
    pub rotation: SO3F64,
    pub keyframe_velocities: Vec<KeyframeVelocity>,
    pub bias: ImuBias,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InertialAlignmentError {
    #[error("invalid initialization scale {0}")]
    InvalidScale(f64),
    #[error("initialization contains no keyframe velocities")]
    MissingVelocities,
    #[error("initialization contains duplicate velocity for keyframe {0}")]
    DuplicateKeyframe(usize),
    #[error("initialization references missing keyframe {0}")]
    MissingKeyframe(usize),
    #[error("initialization velocity for keyframe {0} is non-finite")]
    InvalidVelocity(usize),
    #[error("initialization bias is non-finite")]
    InvalidBias,
}

impl Map {
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

    /// Applies a validated inertial alignment: scales and rotates the world,
    /// then writes each keyframe's velocity and bias.
    ///
    /// Velocity assignments carry keyframe identities, so map insertion order
    /// cannot silently attach an optimizer result to the wrong keyframe.
    /// Everything is validated before the first mutation, so a rejected
    /// alignment leaves the map untouched. Returns the index of the last
    /// (max idx) keyframe in the assignment.
    pub fn apply_inertial_alignment(
        &mut self,
        alignment: InertialAlignment,
    ) -> Result<usize, InertialAlignmentError> {
        if !alignment.scale.is_finite() || alignment.scale <= 0.0 {
            return Err(InertialAlignmentError::InvalidScale(alignment.scale));
        }
        if alignment.keyframe_velocities.is_empty() {
            return Err(InertialAlignmentError::MissingVelocities);
        }
        if !alignment.bias.gyro.length().is_finite() || !alignment.bias.accel.length().is_finite() {
            return Err(InertialAlignmentError::InvalidBias);
        }

        let mut seen = HashSet::with_capacity(alignment.keyframe_velocities.len());
        for assignment in &alignment.keyframe_velocities {
            if !seen.insert(assignment.keyframe_idx) {
                return Err(InertialAlignmentError::DuplicateKeyframe(
                    assignment.keyframe_idx,
                ));
            }
            if self.get_keyframe(assignment.keyframe_idx).is_none() {
                return Err(InertialAlignmentError::MissingKeyframe(
                    assignment.keyframe_idx,
                ));
            }
            if !assignment.velocity_world.length().is_finite() {
                return Err(InertialAlignmentError::InvalidVelocity(
                    assignment.keyframe_idx,
                ));
            }
        }

        let last_keyframe_idx = alignment
            .keyframe_velocities
            .iter()
            .map(|assignment| assignment.keyframe_idx)
            .max()
            .expect("velocity list was checked above");

        self.scale_world(alignment.scale);
        self.rotate_world(&alignment.rotation);
        for assignment in alignment.keyframe_velocities {
            let keyframe = self
                .get_keyframe_mut(assignment.keyframe_idx)
                .expect("keyframe existence was checked before mutating the map");
            keyframe.velocity_world = alignment.rotation * assignment.velocity_world;
            keyframe.imu_bias = alignment.bias;
        }
        Ok(last_keyframe_idx)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{
        Keyframe, LandmarkSeed, ObservationKey,
        tests::{test_frame, test_frame_with_pose},
    };
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};

    #[test]
    fn inertial_alignment_assigns_velocities_by_keyframe_id() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(20, Vec::new())))
            .unwrap();
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, Vec::new())))
            .unwrap();
        let velocity_10 = Vec3F64::new(1.0, 2.0, 3.0);
        let velocity_20 = Vec3F64::new(4.0, 5.0, 6.0);

        let last = map
            .apply_inertial_alignment(InertialAlignment {
                scale: 1.0,
                rotation: SO3F64::IDENTITY,
                keyframe_velocities: vec![
                    KeyframeVelocity {
                        keyframe_idx: 10,
                        velocity_world: velocity_10,
                    },
                    KeyframeVelocity {
                        keyframe_idx: 20,
                        velocity_world: velocity_20,
                    },
                ],
                bias: ImuBias::default(),
            })
            .expect("valid alignment should apply");

        assert_eq!(last, 20);
        assert!((map.get_keyframe(10).unwrap().velocity_world - velocity_10).length() < 1e-12);
        assert!((map.get_keyframe(20).unwrap().velocity_world - velocity_20).length() < 1e-12);
    }

    #[test]
    fn inertial_alignment_rejects_an_unknown_keyframe_without_mutating() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, Vec::new())))
            .unwrap();

        let error = map
            .apply_inertial_alignment(InertialAlignment {
                scale: 2.0,
                rotation: SO3F64::IDENTITY,
                keyframe_velocities: vec![KeyframeVelocity {
                    keyframe_idx: 11,
                    velocity_world: Vec3F64::ZERO,
                }],
                bias: ImuBias::default(),
            })
            .unwrap_err();

        assert_eq!(error, InertialAlignmentError::MissingKeyframe(11));
        // Validation runs before the first mutation, so nothing was scaled.
        assert!((map.get_keyframe(10).unwrap().velocity_world).length() < 1e-12);
    }

    #[test]
    fn inertial_alignment_rotates_stored_velocities() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, Vec::new())))
            .unwrap();
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.4, 0.0));
        let velocity = Vec3F64::new(1.0, 0.2, -0.5);

        map.apply_inertial_alignment(InertialAlignment {
            scale: 1.0,
            rotation: yaw,
            keyframe_velocities: vec![KeyframeVelocity {
                keyframe_idx: 0,
                velocity_world: velocity,
            }],
            bias: ImuBias::default(),
        })
        .expect("valid alignment should apply");

        assert!((map.get_keyframe(0).unwrap().velocity_world - yaw * velocity).length() < 1e-12);
    }

    #[test]
    fn pose_graph_correction_preserves_point_in_reference_camera() {
        let before = [
            Pose3d::IDENTITY,
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-1.0, 0.0, 0.0),
            ),
        ];
        let after = [
            Pose3d::IDENTITY,
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-2.0, 0.0, 0.0),
            ),
        ];
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            10,
            vec![[0; 32]],
            before[0],
        )))
        .unwrap();
        map.insert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            20,
            vec![[1; 32]],
            before[1],
        )))
        .unwrap();
        let point_before = Vec3F64::new(1.0, 0.0, 5.0);
        let point_idx = map
            .insert_landmark(LandmarkSeed {
                position: point_before,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 20,
                    feature_idx: 0,
                },
            })
            .unwrap();
        let point_in_reference_before = before[1].transform_point(&point_before);

        let result = map
            .apply_pose_graph_correction(&[10, 20], &before, &after)
            .unwrap();
        let point_in_reference_after =
            after[1].transform_point(&map.map_points()[point_idx].position);

        assert!((point_in_reference_after - point_in_reference_before).length() < 1e-10);
        assert_eq!(result.keyframes_corrected, 1);
        assert_eq!(result.map_points_corrected, 1);
    }

    #[test]
    fn pose_graph_correction_rejects_stale_snapshot_without_mutation() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(7, vec![[0; 32]])))
            .unwrap();
        let point_idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 7,
                    feature_idx: 0,
                },
            })
            .unwrap();
        let live_pose = map.get_keyframe(7).unwrap().frame.pose_world_to_cam;
        let live_point = map.map_points()[point_idx].position;
        let stale = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(1.0, 0.0, 0.0),
        );

        assert!(
            map.apply_pose_graph_correction(&[7], &[stale], &[Pose3d::IDENTITY])
                .is_err()
        );
        assert_eq!(
            map.get_keyframe(7).unwrap().frame.pose_world_to_cam,
            live_pose
        );
        assert_eq!(map.map_points()[point_idx].position, live_point);
    }
}
