//! Geometry corrections: BA and pose-graph writeback, and inertial alignment.
//!
//! Each validates its complete numerical payload before its first write, so a
//! refused correction leaves the map — including the world epoch — untouched.
//! The raw scale and rotation helpers are private to the validated alignment
//! that owns them: applied alone they would advance the epoch and leave
//! derived geometry stale. Finalizing that geometry is `ops::geometry`.

use super::snapshot::KeyframeBaState;
use crate::map::{BaUpdate, Map};
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
    #[error("pose graph correction computes a non-finite pose, position or velocity")]
    NonFiniteCorrection,
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

/// Refusal to publish a bundle-adjustment result to the live map.
///
/// Every variant is decided before the first write, so a refused update leaves
/// the map — entities, links, counters, factors, geometry and epoch — exactly
/// as it was. Identities are reported from the immutable capture, which is the
/// only side that still knows what the solver was given.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BaUpdateError {
    #[error("bundle adjustment result belongs to an obsolete world frame")]
    ObsoleteWorldFrame,
    #[error("bundle adjustment result does not match the length of its captured inputs")]
    LengthMismatch,
    #[error("bundle adjustment produced a non-finite estimate for keyframe {keyframe_idx}")]
    NonFiniteKeyframe { keyframe_idx: usize },
    #[error("bundle adjustment produced a non-finite position for landmark {landmark_idx}")]
    NonFiniteLandmark { landmark_idx: usize },
    #[error(
        "bundle adjustment returned an invalid preintegration for the edge \
         {prev_kf_idx} -> {curr_kf_idx}"
    )]
    InvalidPreintegration {
        prev_kf_idx: usize,
        curr_kf_idx: usize,
    },
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InertialAlignmentError {
    #[error("invalid initialization scale {0}")]
    InvalidScale(f64),
    #[error("initialization rotation is not a valid unit quaternion")]
    InvalidRotation,
    #[error("initialization produced a non-finite world transform")]
    NonFiniteTransform,
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
        // A landmark whose reference pose is unchanged keeps its position, but
        // any other observer that moved still invalidates its viewing direction.
        let moved_keyframes: HashSet<usize> = keyframe_indices
            .iter()
            .zip(poses_before.iter().zip(poses_after))
            .filter(|(_, (before, after))| before != after)
            .map(|(&idx, _)| idx)
            .collect();

        // Finite inputs can still compose or transport into a non-finite
        // result, so the computed writes are prepared and checked before the
        // epoch advances and before the first field is assigned.
        if world_corrections.iter().any(|c| !finite_pose(c)) {
            return Err(PoseGraphCorrectionError::NonFiniteCorrection);
        }
        let corrected_velocities: Vec<Vec3F64> = self
            .keyframes
            .iter()
            .enumerate()
            .map(|(node, keyframe)| world_corrections[node].rotation * keyframe.velocity_world)
            .collect();
        let corrected_positions: Vec<Option<Vec3F64>> = self
            .map_points
            .iter()
            .map(|point| {
                (!point.culled).then(|| {
                    let node = node_by_keyframe[&point.keyframe_idx];
                    world_corrections[node].transform_point(&point.position)
                })
            })
            .collect();
        if corrected_velocities.iter().any(|v| !finite_vec3(*v))
            || corrected_positions
                .iter()
                .flatten()
                .any(|p| !finite_vec3(*p))
        {
            return Err(PoseGraphCorrectionError::NonFiniteCorrection);
        }

        self.world_epoch = self.world_epoch.wrapping_add(1);
        for (node, keyframe) in self.keyframes.iter_mut().enumerate() {
            keyframe.frame.pose_world_to_cam = poses_after[node];
            keyframe.velocity_world = corrected_velocities[node];
        }

        let mut changed_points = Vec::new();
        for (idx, point) in self.map_points.iter_mut().enumerate() {
            let Some(corrected) = corrected_positions[idx] else {
                continue;
            };
            if corrected != point.position {
                point.position = corrected;
                changed_points.push(idx);
            }
        }
        let map_points_corrected = changed_points.len();
        let observers_moved = self.landmarks_observed_by(&moved_keyframes);
        self.finalize_landmark_geometry(changed_points.into_iter().chain(observers_moved));

        Ok(PoseGraphCorrectionResult {
            keyframes_corrected,
            map_points_corrected,
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
        // Checked before anything is scaled: a bad rotation must not leave a
        // scaled-but-unrotated world behind.
        if !valid_rotation(&alignment.rotation) {
            return Err(InertialAlignmentError::InvalidRotation);
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

        // The world transform and the velocity writes are computed and checked
        // before the first mutation: a finite scale applied to a finite but
        // extreme coordinate can still overflow, and a half-applied alignment
        // is not recoverable.
        let assigned_velocities: Vec<Vec3F64> = alignment
            .keyframe_velocities
            .iter()
            .map(|assignment| alignment.rotation * assignment.velocity_world)
            .collect();
        let Some(transform) = self.prepare_world_transform(alignment.scale, &alignment.rotation)
        else {
            return Err(InertialAlignmentError::NonFiniteTransform);
        };
        if assigned_velocities.iter().any(|v| !finite_vec3(*v)) {
            return Err(InertialAlignmentError::NonFiniteTransform);
        }

        self.commit_world_transform(transform);
        for (assignment, velocity_world) in alignment
            .keyframe_velocities
            .iter()
            .zip(&assigned_velocities)
        {
            let keyframe = self
                .get_keyframe_mut(assignment.keyframe_idx)
                .expect("keyframe existence was checked before mutating the map");
            keyframe.velocity_world = *velocity_world;
            keyframe.imu_bias = alignment.bias;
        }
        // Every camera centre moved, so finalize once over the whole live map
        // after the complete correction — never between the scale and the
        // rotation, which would derive geometry from a half-aligned world.
        let live = self.live_landmarks();
        self.finalize_landmark_geometry(live);
        Ok(last_keyframe_idx)
    }

    /// The pose a keyframe stores after a metric scale.
    ///
    /// The scale acts on the camera centre, which lives in the inverted
    /// (camera-to-world) pose, so the stored world-to-camera pose is that
    /// intermediate inverted back. Formula unchanged from the raw scale
    /// helper this replaced.
    fn scaled_pose(pose: &Pose3d, scale: f64) -> Pose3d {
        let mut cam_to_world = pose.inverse();
        cam_to_world.translation *= scale;
        cam_to_world.inverse()
    }

    /// The pose a keyframe stores after a world rotation. Formula unchanged
    /// from the raw rotate helper this replaced.
    fn rotated_pose(pose: &Pose3d, r: &SO3F64) -> Pose3d {
        let cam_to_world = pose.inverse();
        let new_translation = *r * cam_to_world.translation;
        let rot_so3 = SO3F64::from_matrix(&cam_to_world.rotation);
        let new_rot_so3 = *r * rot_so3;
        let new_rotation = new_rot_so3.matrix();
        Pose3d::from_rt(new_rotation, new_translation).inverse()
    }

    /// Computes exactly the values an alignment will store, or `None` if any
    /// of them is not finite.
    ///
    /// The *stored* pose is what gets checked, not the camera centre it is
    /// derived from. Those differ by an inversion, and the inversion is itself
    /// an overflow site: a 45-degree rotation makes the two scaled translation
    /// components add rather than cancel, so a finite scaled centre can invert
    /// into an infinite stored translation.
    ///
    /// Applies the scale and then the rotation in that order, per keyframe and
    /// per landmark, reproducing the arithmetic of the two half-transforms it
    /// replaced — including their asymmetry over retired landmarks, which are
    /// scaled but not rotated.
    fn prepare_world_transform(&self, scale: f64, rotation: &SO3F64) -> Option<WorldTransform> {
        let mut poses = Vec::with_capacity(self.keyframes.len());
        for keyframe in &self.keyframes {
            let scaled = Self::scaled_pose(&keyframe.frame.pose_world_to_cam, scale);
            let stored = Self::rotated_pose(&scaled, rotation);
            if !finite_pose(&stored) {
                return None;
            }
            poses.push(stored);
        }

        let mut positions = Vec::with_capacity(self.map_points.len());
        for point in &self.map_points {
            let scaled = point.position * scale;
            let stored = if point.culled {
                scaled
            } else {
                *rotation * scaled
            };
            if !finite_vec3(stored) {
                return None;
            }
            positions.push(stored);
        }

        Some(WorldTransform { poses, positions })
    }

    /// Writes a prepared world transform. Every value was checked by
    /// [`Map::prepare_world_transform`], so this only assigns.
    ///
    /// The epoch advances twice, once for each half-transform this replaced:
    /// snapshots are invalidated by any change, so the count is not load
    /// bearing, but preserving it keeps the extraction observationally
    /// identical.
    fn commit_world_transform(&mut self, transform: WorldTransform) {
        self.world_epoch = self.world_epoch.wrapping_add(2);
        for (keyframe, pose) in self.keyframes_mut().iter_mut().zip(transform.poses) {
            keyframe.frame.pose_world_to_cam = pose;
        }
        for (point, position) in self.map_points_mut().iter_mut().zip(transform.positions) {
            point.position = position;
        }
    }
}

/// The exact values an inertial alignment will store, indexed by the map's own
/// keyframe and landmark order. Prepared and validated before the first write.
struct WorldTransform {
    poses: Vec<Pose3d>,
    positions: Vec<Vec3F64>,
}

/// A keyframe state change accepted from a BA result.
#[derive(Debug, Clone, Copy)]
pub struct KeyframeBaCorrection {
    pub kf_idx: usize,
    pub pose_before: Pose3d,
    pub pose_after: Pose3d,
    pub velocity_world: Vec3F64,
    pub imu_bias: ImuBias,
}

/// Changes accepted by BA writeback and published by the local-mapping worker.
#[derive(Debug, Default)]
pub struct LocalBaMergeResult {
    pub keyframe_corrections: Vec<KeyframeBaCorrection>,
    pub map_points_updated: usize,
}

impl Map {
    /// Applies solver-changed geometry from a BA update.
    ///
    /// Entities inserted after the snapshot are left untouched. A snapshot is
    /// rejected wholesale if the live map has since been scaled, rotated, or
    /// cleared, because its coordinates then belong to another world frame.
    ///
    /// The complete numerical payload is validated before the first write, so
    /// a malformed result cannot land partially: a valid change early in the
    /// arrays is not published just because a later value is unusable. An
    /// unchanged but valid update is still a successful completion.
    pub fn apply_ba_update(
        &mut self,
        update: BaUpdate,
    ) -> Result<LocalBaMergeResult, BaUpdateError> {
        let snapshot = &update.snapshot;
        if self.world_epoch != snapshot.world_epoch {
            return Err(BaUpdateError::ObsoleteWorldFrame);
        }
        // Lengths are checked rather than left to `zip`, which would silently
        // truncate a malformed result and publish the prefix.
        if update.keyframes.len() != snapshot.keyframes.len()
            || update.map_points.len() != snapshot.map_points.len()
            || update.imu_preintegrations.len() != snapshot.imu_factors.len()
        {
            return Err(BaUpdateError::LengthMismatch);
        }
        for (captured, optimized) in snapshot.keyframes.iter().zip(&update.keyframes) {
            if !finite_pose(&optimized.pose_world_to_cam)
                || !finite_vec3(optimized.velocity_world)
                || !finite_bias(&optimized.imu_bias)
            {
                return Err(BaUpdateError::NonFiniteKeyframe {
                    keyframe_idx: captured.frame.idx,
                });
            }
        }
        // Landmark slots are stable, so the captured index is the landmark id.
        for (landmark_idx, &position) in update.map_points.iter().enumerate() {
            if !finite_vec3(position) {
                return Err(BaUpdateError::NonFiniteLandmark { landmark_idx });
            }
        }
        for (edge, preintegrated) in snapshot.imu_factors.iter().zip(&update.imu_preintegrations) {
            if !valid_preintegration(preintegrated) {
                return Err(BaUpdateError::InvalidPreintegration {
                    prev_kf_idx: edge.prev_kf_idx,
                    curr_kf_idx: edge.curr_kf_idx,
                });
            }
        }

        let mut result = LocalBaMergeResult::default();
        // Which estimates to publish is decided against the capture baseline;
        // which landmarks became stale is decided against the *live* pose,
        // because that is the camera centre their geometry was derived from.
        let mut moved_keyframes = HashSet::new();
        for (before, optimized) in snapshot.keyframes.iter().zip(update.keyframes.iter()) {
            if !keyframe_ba_state_changed(before, optimized) {
                continue;
            }
            let Some(live) = self.get_keyframe_mut(before.frame.idx) else {
                continue;
            };

            let pose_before = live.frame.pose_world_to_cam;
            live.frame.pose_world_to_cam = optimized.pose_world_to_cam;
            live.velocity_world = optimized.velocity_world;
            live.imu_bias = optimized.imu_bias;
            if pose_before != optimized.pose_world_to_cam {
                moved_keyframes.insert(before.frame.idx);
            }
            result.keyframe_corrections.push(KeyframeBaCorrection {
                kf_idx: before.frame.idx,
                pose_before,
                pose_after: optimized.pose_world_to_cam,
                velocity_world: optimized.velocity_world,
                imu_bias: optimized.imu_bias,
            });
        }

        let mut changed_points = Vec::new();
        for (idx, (before, &optimized)) in snapshot
            .map_points
            .iter()
            .zip(update.map_points.iter())
            .enumerate()
        {
            if optimized == before.position {
                continue;
            }
            let Some(live) = self.map_points.get_mut(idx) else {
                continue;
            };
            if live.culled {
                continue;
            }
            live.position = optimized;
            changed_points.push(idx);
        }
        result.map_points_updated = changed_points.len();

        // Counters describe position changes; the refresh set is larger,
        // covering every live landmark whose observers moved.
        let observers_moved = self.landmarks_observed_by(&moved_keyframes);
        self.finalize_landmark_geometry(changed_points.into_iter().chain(observers_moved));

        // VI-BA may repropagate an existing preintegration on its private copy.
        // Preserve newer live factors while copying those refreshed edges back.
        for (edge, preintegrated) in snapshot.imu_factors.iter().zip(&update.imu_preintegrations) {
            if let Some(live) = self.imu_factors.iter_mut().find(|live| {
                live.prev_kf_idx == edge.prev_kf_idx && live.curr_kf_idx == edge.curr_kf_idx
            }) {
                live.preintegrated = preintegrated.clone();
            }
        }
        Ok(result)
    }
}
/// Tolerance on a rotation's unit-quaternion norm.
///
/// [`SO3F64`] wraps a raw quaternion and enforces nothing, so a correction has
/// to check the representation it was handed. The bound is loose enough to
/// accept rotations that reached `f64` through `f32` storage or an `f32`
/// matrix conversion — worst case a few multiples of `f32::EPSILON` (~1.2e-7) —
/// and tight enough that a genuinely unnormalized or rescaled quaternion is
/// refused. Invalid input is rejected, never silently normalized.
const ROTATION_NORM_TOLERANCE: f64 = 1e-6;

fn finite_vec3(v: Vec3F64) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn finite_mat3(m: &kornia_algebra::Mat3F64) -> bool {
    m.to_cols_array().into_iter().all(f64::is_finite)
}

fn finite_pose(pose: &Pose3d) -> bool {
    finite_vec3(pose.translation) && finite_mat3(&pose.rotation)
}

fn finite_bias(bias: &ImuBias) -> bool {
    finite_vec3(bias.gyro) && finite_vec3(bias.accel)
}

/// Accepts a rotation only if its stored quaternion is finite and of unit
/// norm within [`ROTATION_NORM_TOLERANCE`].
fn valid_rotation(rotation: &SO3F64) -> bool {
    let q = rotation.to_array();
    if !q.iter().all(|value| value.is_finite()) {
        return false;
    }
    let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    (norm - 1.0).abs() <= ROTATION_NORM_TOLERANCE
}

/// Checks every numeric field a repropagated measurement carries.
///
/// A zero-duration measurement is a legitimate placeholder and stays
/// acceptable; a negative or non-finite duration is not. Covariance
/// definiteness and calibration policy are deliberately out of scope here —
/// this is a finiteness gate, not a sensor-model review.
fn valid_preintegration(preintegrated: &kornia_sensors::imu::PreintegratedImu) -> bool {
    preintegrated.dt.is_finite()
        && preintegrated.dt >= 0.0
        && finite_mat3(&preintegrated.delta_rotation)
        && finite_vec3(preintegrated.delta_velocity)
        && finite_vec3(preintegrated.delta_position)
        && finite_bias(&preintegrated.bias)
        && preintegrated.calib.gyro_noise.is_finite()
        && preintegrated.calib.accel_noise.is_finite()
        && preintegrated.calib.gyro_bias_noise.is_finite()
        && preintegrated.calib.accel_bias_noise.is_finite()
        && preintegrated.covariance.iter().all(|v| v.is_finite())
        && preintegrated.bias_covariance.iter().all(|v| v.is_finite())
        && finite_mat3(&preintegrated.d_rotation_d_bias_gyro)
        && finite_mat3(&preintegrated.d_velocity_d_bias_gyro)
        && finite_mat3(&preintegrated.d_velocity_d_bias_accel)
        && finite_mat3(&preintegrated.d_position_d_bias_gyro)
        && finite_mat3(&preintegrated.d_position_d_bias_accel)
}

fn keyframe_ba_state_changed(before: &crate::map::Keyframe, after: &KeyframeBaState) -> bool {
    before.frame.pose_world_to_cam != after.pose_world_to_cam
        || before.velocity_world != after.velocity_world
        || before.imu_bias.gyro != after.imu_bias.gyro
        || before.imu_bias.accel != after.imu_bias.accel
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

    /// A landmark transported through an unmoved reference keyframe keeps its
    /// position exactly — but a *different* observing camera moved, so the
    /// direction averaged over all observers is stale until refreshed.
    #[test]
    fn pose_graph_refreshes_a_moved_non_reference_observer() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])))
            .unwrap();
        let moved_before = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-2.0, 0.0, 0.0),
        );
        map.insert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![[1; 32]],
            moved_before,
        )))
        .unwrap();
        let point_idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        assert!(map.link_observation(1, 0, point_idx).unwrap());
        let position_before = map.map_points()[point_idx].position;
        let bounds_before = map.map_points()[point_idx].max_distance;

        let moved_after = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-4.0, 0.0, 0.0),
        );
        let result = map
            .apply_pose_graph_correction(
                &[0, 1],
                &[Pose3d::IDENTITY, moved_before],
                &[Pose3d::IDENTITY, moved_after],
            )
            .expect("valid pose graph correction");

        assert_eq!(result.keyframes_corrected, 1);
        assert_eq!(
            result.map_points_corrected, 0,
            "the reference pose is unchanged, so the position does not move"
        );
        assert_eq!(map.map_points()[point_idx].position, position_before);
        assert_eq!(map.map_points()[point_idx].max_distance, bounds_before);
        let expected =
            (Vec3F64::new(0.0, 0.0, 1.0) + Vec3F64::new(-4.0, 0.0, 5.0) / 41.0_f64.sqrt()) / 2.0;
        let direction = map.map_points()[point_idx].mean_viewing_direction;
        assert!(
            (direction - expected).length() < 1e-10,
            "viewing direction {direction:?} should be {expected:?}"
        );
    }

    /// Alignment scales and rotates the whole world at once. Geometry is
    /// finalized after the complete correction, never between the scale and
    /// the rotation, and a retired landmark stays retired with nothing stale.
    #[test]
    fn inertial_alignment_refreshes_geometry_and_leaves_retired_landmarks_clear() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32], [1; 32]])))
            .unwrap();
        let live = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        let retired = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 1.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 1,
                },
            })
            .unwrap();
        assert!(map.remove_landmark(retired).unwrap());
        assert!((map.map_points()[live].max_distance - 5.0).abs() < 1e-12);

        // A quarter turn about +Y sends (0, 0, 1) to (1, 0, 0).
        let quarter_turn = SO3F64::exp(Vec3F64::new(0.0, std::f64::consts::FRAC_PI_2, 0.0));
        map.apply_inertial_alignment(InertialAlignment {
            scale: 2.0,
            rotation: quarter_turn,
            keyframe_velocities: vec![KeyframeVelocity {
                keyframe_idx: 0,
                velocity_world: Vec3F64::ZERO,
            }],
            bias: ImuBias::default(),
        })
        .expect("valid alignment should apply");

        let point = &map.map_points()[live];
        assert!(
            (point.max_distance - 10.0).abs() < 1e-9,
            "distance bounds scale with the world: {}",
            point.max_distance
        );
        assert!(
            (point.mean_viewing_direction - Vec3F64::new(1.0, 0.0, 0.0)).length() < 1e-9,
            "viewing direction should be rotated a quarter turn: {:?}",
            point.mean_viewing_direction
        );

        let dead = &map.map_points()[retired];
        assert!(dead.culled, "a retired landmark is never revived");
        assert!(dead.observations().is_empty());
        assert_eq!(dead.mean_viewing_direction, Vec3F64::ZERO);
        assert_eq!(dead.min_distance, 0.0);
        assert_eq!(dead.max_distance, 0.0);
    }
}

#[cfg(test)]
mod ba_tests {
    use super::*;
    use crate::map::{
        Keyframe, LandmarkSeed, MapInsertion, ObservationKey,
        tests::{test_frame, test_frame_with_pose},
    };
    use kornia_algebra::SO3F64;

    /// Moving a landmark's reference camera changes the distance bounds and
    /// viewing direction derived from that camera centre, even though the
    /// landmark itself does not move. The map owes that refresh to every
    /// caller; no solver supplies a list of points to fix up.
    #[test]
    fn ba_moving_a_reference_camera_refreshes_geometry_without_moving_the_point() {
        let (mut map, point) = snapshot_fixture();
        let before = map.map_points()[point].position;
        let mut update = map.ba_snapshot().into_update();
        // World-to-camera translation (1, 0, 0) puts the camera centre at
        // (-1, 0, 0), so the landmark at (0, 0, 5) sits (1, 0, 5) away.
        update.keyframes[0].pose_world_to_cam.translation.x = 1.0;
        let result = map.apply_ba_update(update).unwrap();
        assert_eq!(result.keyframe_corrections.len(), 1);
        assert_eq!(result.map_points_updated, 0);
        assert_eq!(map.map_points()[point].position, before);
        assert!((map.map_points()[point].max_distance - 26.0_f64.sqrt()).abs() < 1e-10);
        let direction = map.map_points()[point].mean_viewing_direction;
        let expected = Vec3F64::new(1.0, 0.0, 5.0) / 26.0_f64.sqrt();
        assert!(
            (direction - expected).length() < 1e-10,
            "viewing direction {direction:?} should be {expected:?}"
        );
    }

    /// The viewing direction averages over *all* observing camera centres, so
    /// moving a non-reference observer changes it while the reference distance
    /// bounds stay put.
    #[test]
    fn ba_moving_a_non_reference_observer_refreshes_only_the_viewing_direction() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![[1u8; 32]],
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-2.0, 0.0, 0.0),
            ),
        )))
        .unwrap();
        let point = map.insert_landmark(seeded(0, 0, 5.0)).unwrap();
        assert!(map.link_observation(1, 0, point).unwrap());
        let bounds_before = map.map_points()[point].max_distance;

        let mut update = map.ba_snapshot().into_update();
        // Camera centre of keyframe 1 moves from (2, 0, 0) to (4, 0, 0).
        update.keyframes[1].pose_world_to_cam.translation.x = -4.0;
        let result = map.apply_ba_update(update).unwrap();

        assert_eq!(result.keyframe_corrections.len(), 1);
        assert_eq!(result.map_points_updated, 0);
        assert_eq!(
            map.map_points()[point].max_distance,
            bounds_before,
            "the reference camera did not move, so the bounds must not change"
        );
        let expected =
            (Vec3F64::new(0.0, 0.0, 1.0) + Vec3F64::new(-4.0, 0.0, 5.0) / 41.0_f64.sqrt()) / 2.0;
        let direction = map.map_points()[point].mean_viewing_direction;
        assert!(
            (direction - expected).length() < 1e-10,
            "viewing direction {direction:?} should be {expected:?}"
        );
    }

    /// A landmark linked while BA was running is outside the captured arrays,
    /// so no estimate may be written back to it — but an existing camera that
    /// BA moved is still one of its observers, so its metadata is stale.
    #[test]
    fn a_landmark_linked_after_capture_is_refreshed_without_writeback() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        let mut update = map.ba_snapshot().into_update();

        // Inserted after capture: the snapshot holds no landmarks at all.
        let later = map.insert_landmark(seeded(0, 0, 5.0)).unwrap();
        let position = map.map_points()[later].position;

        update.keyframes[0].pose_world_to_cam.translation.x = 1.0;
        let result = map.apply_ba_update(update).unwrap();

        assert_eq!(result.map_points_updated, 0);
        assert_eq!(
            map.map_points()[later].position,
            position,
            "a landmark outside the capture keeps its own position"
        );
        assert_eq!(
            map.get_keyframe(0).unwrap().map_point(0),
            Some(later),
            "its link survived"
        );
        assert!(
            (map.map_points()[later].max_distance - 26.0_f64.sqrt()).abs() < 1e-10,
            "its metadata reflects the camera centre BA moved"
        );
    }
    #[test]
    fn local_ba_snapshot_merge_updates_only_snapshot_entities() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_landmark(seeded(0, 0, 5.0)).unwrap();

        let mut snapshot = map.ba_snapshot().into_update();
        snapshot.keyframes[0].pose_world_to_cam.translation.x = 2.0;
        snapshot.map_points[0].x = 3.0;

        map.insert_keyframe(Keyframe::from_frame(test_frame(1, vec![[1u8; 32]])))
            .unwrap();
        let later_point = map
            .insert_landmark(seeded_at(1, 0, Vec3F64::new(9.0, 0.0, 5.0)))
            .unwrap();

        let merged = map
            .apply_ba_update(snapshot)
            .expect("snapshot should still use the live world frame");

        assert_eq!(merged.keyframe_corrections.len(), 1);
        assert_eq!(merged.keyframe_corrections[0].kf_idx, 0);
        assert_eq!(
            map.get_keyframe(0)
                .unwrap()
                .frame
                .pose_world_to_cam
                .translation
                .x,
            2.0
        );
        assert_eq!(map.map_points()[0].position.x, 3.0);
        assert_eq!(
            map.get_keyframe(1).unwrap().frame.pose_world_to_cam,
            Pose3d::IDENTITY
        );
        assert_eq!(map.map_points()[later_point].position.x, 9.0);
    }

    /// A fresh bootstrap clears the active map before publishing, so any
    /// asynchronous update still in flight for the abandoned map must be
    /// refused: its coordinates belong to a world frame that no longer exists.
    #[test]
    fn clearing_the_active_map_rejects_an_in_flight_ba_snapshot() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_landmark(seeded_at(0, 0, Vec3F64::new(1.0, 0.0, 5.0)))
            .unwrap();

        let mut snapshot = map.ba_snapshot().into_update();
        snapshot.map_points[0].x = 7.0;

        map.clear_active();

        assert_eq!(
            map.apply_ba_update(snapshot).unwrap_err(),
            BaUpdateError::ObsoleteWorldFrame
        );
        assert!(
            map.keyframes().is_empty() && map.map_points().is_empty(),
            "the refused update must not repopulate the cleared map"
        );
    }

    #[test]
    fn local_ba_snapshot_merge_rejects_an_obsolete_world_frame() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_landmark(seeded_at(0, 0, Vec3F64::new(1.0, 0.0, 5.0)))
            .unwrap();

        let mut snapshot = map.ba_snapshot().into_update();
        snapshot.map_points[0].x = 7.0;
        map.apply_inertial_alignment(InertialAlignment {
            scale: 2.0,
            rotation: SO3F64::IDENTITY,
            keyframe_velocities: vec![KeyframeVelocity {
                keyframe_idx: 0,
                velocity_world: Vec3F64::ZERO,
            }],
            bias: ImuBias::default(),
        })
        .expect("a valid alignment advances the world epoch");

        assert_eq!(
            map.apply_ba_update(snapshot).unwrap_err(),
            BaUpdateError::ObsoleteWorldFrame
        );
        assert_eq!(map.map_points()[0].position.x, 2.0);
    }

    #[test]
    fn pose_graph_correction_invalidates_older_ba_snapshot() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])))
            .unwrap();
        let mut snapshot = map.ba_snapshot().into_update();
        snapshot.keyframes[0].pose_world_to_cam.translation.x = 9.0;
        let corrected = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-1.0, 0.0, 0.0),
        );

        map.apply_pose_graph_correction(&[0], &[Pose3d::IDENTITY], &[corrected])
            .unwrap();

        assert_eq!(
            map.apply_ba_update(snapshot).unwrap_err(),
            BaUpdateError::ObsoleteWorldFrame
        );
        assert_eq!(
            map.get_keyframe(0).unwrap().frame.pose_world_to_cam,
            corrected
        );
    }

    // ── canonical mutation vs. an in-flight BA snapshot ──────────────────

    fn seeded(kf: usize, feature: usize, z: f64) -> LandmarkSeed {
        seeded_at(kf, feature, Vec3F64::new(0.0, 0.0, z))
    }

    fn seeded_at(kf: usize, feature: usize, position: Vec3F64) -> LandmarkSeed {
        LandmarkSeed {
            position,
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: kf,
                feature_idx: feature,
            },
        }
    }

    fn snapshot_fixture() -> (Map, usize) {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        let point = map.insert_landmark(seeded(0, 0, 5.0)).unwrap();
        (map, point)
    }

    /// A snapshot taken before a landmark was retired must not resurrect it.
    #[test]
    fn merging_an_older_snapshot_does_not_resurrect_a_retired_landmark() {
        let (mut map, point) = snapshot_fixture();
        let mut snapshot = map.ba_snapshot().into_update();
        // The solver must actually have moved this landmark: writeback skips
        // unchanged positions, so an untouched snapshot would never reach the
        // retirement guard and the test would pass without exercising it.
        snapshot.map_points[point].x += 1.0;

        assert!(map.remove_landmark(point).unwrap());
        let merged = map
            .apply_ba_update(snapshot)
            .expect("the snapshot is still in the live world frame");

        assert!(map.map_points()[point].culled, "still retired");
        assert!(map.map_points()[point].observations().is_empty());
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(
            merged.map_points_updated, 0,
            "a retired landmark is not written back"
        );
    }

    /// Entities added while BA was running must survive the writeback.
    #[test]
    fn entities_added_after_a_snapshot_survive_its_merge() {
        let (mut map, _) = snapshot_fixture();
        let snapshot = map.ba_snapshot().into_update();

        let result = map
            .apply_insertion(MapInsertion {
                keyframes: vec![Keyframe::from_frame(test_frame(1, vec![[1u8; 32]; 2]))],
                landmarks: vec![seeded(1, 0, 7.0)],
                ..Default::default()
            })
            .expect("valid batch");
        let newer = result.landmark_ids[0];

        // Move an older entity so the merge has real work to publish; a no-op
        // writeback would satisfy the survival assertions vacuously.
        let mut snapshot = snapshot;
        snapshot.map_points[0].x += 0.5;
        let merged = map
            .apply_ba_update(snapshot)
            .expect("the snapshot is still in the live world frame");
        assert_eq!(
            merged.map_points_updated, 1,
            "the older landmark was actually written back"
        );

        assert!(map.get_keyframe(1).is_some(), "newer keyframe survived");
        assert!(!map.map_points()[newer].culled, "newer landmark survived");
        assert_eq!(
            map.get_keyframe(1).unwrap().map_point(0),
            Some(newer),
            "its link survived"
        );
    }

    /// A merge must not reapply the pre-merge observation lists over a merge
    /// that happened in the meantime.
    #[test]
    fn merging_after_a_landmark_merge_keeps_structure_consistent() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        map.insert_keyframe(Keyframe::from_frame(test_frame(1, vec![[1u8; 32]; 2])))
            .unwrap();
        let survivor = map.insert_landmark(seeded(0, 0, 5.0)).unwrap();
        let duplicate = map.insert_landmark(seeded(0, 1, 5.01)).unwrap();
        map.link_observation(1, 0, survivor).unwrap();

        let snapshot = map.ba_snapshot().into_update();
        map.merge_map_points(survivor, duplicate);
        // A landmark merge is not a world correction, so the capture is still
        // publishable — asserting that keeps the consistency checks below from
        // passing vacuously on a refusal.
        map.apply_ba_update(snapshot)
            .expect("a landmark merge does not invalidate the capture");

        // Whatever survived, the two sides still agree everywhere.
        for kf in map.keyframes() {
            for (feature, slot) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = *slot else { continue };
                let mp = &map.map_points()[mp_idx];
                assert!(!mp.culled, "kf {} links a retired landmark", kf.frame.idx);
                assert!(
                    mp.observations()
                        .iter()
                        .any(|o| o.key.keyframe_idx == kf.frame.idx
                            && o.key.feature_idx == feature),
                    "link from kf {} has no record",
                    kf.frame.idx
                );
            }
        }
    }
}

/// Atomic rejection: an invalid numerical correction must leave the live map
/// completely untouched, not partially written.
///
/// Each case puts a real, valid change early in the arrays and the invalid
/// value at the end, then compares a full state fingerprint across the
/// refusal — entity counts alone would miss a published prefix.
#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::map::{
        ImuFactor, Keyframe, LandmarkSeed, MapInsertion, ObservationKey, tests::test_frame,
    };
    use kornia_algebra::{Mat3F64, SO3F64};
    use kornia_sensors::imu::{ImuCalib, PreintegratedImu};

    fn seeded(kf: usize, feature: usize, position: Vec3F64) -> LandmarkSeed {
        LandmarkSeed {
            position,
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: kf,
                feature_idx: feature,
            },
        }
    }

    /// Two keyframes and two landmarks, so a case can change the first and
    /// corrupt the last.
    fn fixture() -> Map {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_keyframe(Keyframe::from_frame(test_frame(1, vec![[1u8; 32]])))
            .unwrap();
        map.insert_landmark(seeded(0, 0, Vec3F64::new(0.0, 0.0, 5.0)))
            .unwrap();
        map.insert_landmark(seeded(1, 0, Vec3F64::new(0.0, 0.0, 7.0)))
            .unwrap();
        map
    }

    fn placeholder_edge(prev: usize, curr: usize) -> ImuFactor {
        ImuFactor {
            prev_kf_idx: prev,
            curr_kf_idx: curr,
            preintegrated: PreintegratedImu::new(
                ImuBias::default(),
                ImuCalib {
                    gyro_noise: 1e-4,
                    accel_noise: 1e-3,
                    gyro_bias_noise: 1e-5,
                    accel_bias_noise: 1e-3,
                },
            ),
            raw_samples: Vec::new(),
            t0: 0.0,
            t1: 0.1,
        }
    }

    /// Every numeric field a solver can write, corrupted with both NaN and
    /// infinity, reported against the identity the capture carries.
    #[test]
    fn every_non_finite_estimate_is_refused_without_any_write() {
        type Corrupt = fn(&mut BaUpdate, f64);
        let cases: Vec<(&str, Corrupt, BaUpdateError)> = vec![
            (
                "pose translation",
                |update, bad| update.keyframes[1].pose_world_to_cam.translation.y = bad,
                BaUpdateError::NonFiniteKeyframe { keyframe_idx: 1 },
            ),
            (
                "pose rotation",
                |update, bad| {
                    let mut cols = update.keyframes[1]
                        .pose_world_to_cam
                        .rotation
                        .to_cols_array();
                    cols[4] = bad;
                    update.keyframes[1].pose_world_to_cam.rotation =
                        Mat3F64::from_cols_array(&cols);
                },
                BaUpdateError::NonFiniteKeyframe { keyframe_idx: 1 },
            ),
            (
                "velocity",
                |update, bad| update.keyframes[1].velocity_world.z = bad,
                BaUpdateError::NonFiniteKeyframe { keyframe_idx: 1 },
            ),
            (
                "gyro bias",
                |update, bad| update.keyframes[1].imu_bias.gyro.x = bad,
                BaUpdateError::NonFiniteKeyframe { keyframe_idx: 1 },
            ),
            (
                "accelerometer bias",
                |update, bad| update.keyframes[1].imu_bias.accel.y = bad,
                BaUpdateError::NonFiniteKeyframe { keyframe_idx: 1 },
            ),
            (
                "landmark coordinate",
                |update, bad| update.map_points[1].x = bad,
                BaUpdateError::NonFiniteLandmark { landmark_idx: 1 },
            ),
        ];

        for (what, corrupt, expected) in cases {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut map = fixture();
                let before = map.state_fingerprint_for_test();
                let mut update = map.ba_snapshot().into_update();
                // A genuine, publishable change early in the arrays.
                update.keyframes[0].pose_world_to_cam.translation.x = 1.0;
                update.map_points[0].z = 6.0;
                corrupt(&mut update, bad);

                assert_eq!(
                    map.apply_ba_update(update).unwrap_err(),
                    expected,
                    "{what} = {bad} should be refused"
                );
                assert_eq!(
                    map.state_fingerprint_for_test(),
                    before,
                    "{what} = {bad} wrote something before refusing"
                );
            }
        }
    }

    /// A malformed result must be refused outright rather than have its prefix
    /// published, which is what `zip` alone would do.
    #[test]
    fn mismatched_result_lengths_are_refused_without_any_write() {
        type Truncate = fn(&mut BaUpdate);
        let cases: Vec<(&str, Truncate)> = vec![
            ("keyframes", |update| {
                update.keyframes.pop();
            }),
            ("landmarks", |update| {
                update.map_points.pop();
            }),
            ("preintegrations", |update| {
                update.imu_preintegrations.pop();
            }),
            ("extra keyframe", |update| {
                let extra = update.keyframes[0];
                update.keyframes.push(extra);
            }),
        ];

        for (what, truncate) in cases {
            let mut map = fixture();
            map.apply_insertion(MapInsertion {
                imu_factors: vec![placeholder_edge(0, 1)],
                ..Default::default()
            })
            .unwrap();
            let before = map.state_fingerprint_for_test();
            let mut update = map.ba_snapshot().into_update();
            update.keyframes[0].pose_world_to_cam.translation.x = 1.0;
            truncate(&mut update);

            assert_eq!(
                map.apply_ba_update(update).unwrap_err(),
                BaUpdateError::LengthMismatch,
                "a {what} length mismatch should be refused"
            );
            assert_eq!(
                map.state_fingerprint_for_test(),
                before,
                "a {what} length mismatch wrote something before refusing"
            );
        }
    }

    /// Repropagated measurements are numerical output too: covariance and
    /// Jacobian blocks are checked, not just the deltas.
    #[test]
    fn a_non_finite_preintegration_is_refused_after_a_valid_pose_update() {
        type Corrupt = fn(&mut PreintegratedImu);
        let cases: Vec<(&str, Corrupt)> = vec![
            ("delta rotation", |p| {
                let mut cols = p.delta_rotation.to_cols_array();
                cols[0] = f64::NAN;
                p.delta_rotation = Mat3F64::from_cols_array(&cols);
            }),
            ("delta velocity", |p| p.delta_velocity.x = f64::INFINITY),
            ("delta position", |p| p.delta_position.z = f64::NAN),
            ("duration", |p| p.dt = f64::NAN),
            ("negative duration", |p| p.dt = -0.1),
            ("bias", |p| p.bias.gyro.y = f64::NAN),
            ("calibration", |p| p.calib.accel_noise = f64::NAN),
            ("covariance", |p| p.covariance[80] = f64::NAN),
            ("bias covariance", |p| p.bias_covariance[35] = f64::INFINITY),
            ("gyro-bias rotation jacobian", |p| {
                let mut cols = p.d_rotation_d_bias_gyro.to_cols_array();
                cols[8] = f64::NAN;
                p.d_rotation_d_bias_gyro = Mat3F64::from_cols_array(&cols);
            }),
            ("accel-bias position jacobian", |p| {
                let mut cols = p.d_position_d_bias_accel.to_cols_array();
                cols[3] = f64::INFINITY;
                p.d_position_d_bias_accel = Mat3F64::from_cols_array(&cols);
            }),
        ];

        for (what, corrupt) in cases {
            let mut map = fixture();
            map.apply_insertion(MapInsertion {
                imu_factors: vec![placeholder_edge(0, 1)],
                ..Default::default()
            })
            .unwrap();
            let before = map.state_fingerprint_for_test();
            let mut update = map.ba_snapshot().into_update();
            update.keyframes[0].pose_world_to_cam.translation.x = 1.0;
            corrupt(&mut update.imu_preintegrations[0]);

            assert_eq!(
                map.apply_ba_update(update).unwrap_err(),
                BaUpdateError::InvalidPreintegration {
                    prev_kf_idx: 0,
                    curr_kf_idx: 1,
                },
                "a non-finite {what} should be refused"
            );
            assert_eq!(
                map.state_fingerprint_for_test(),
                before,
                "a non-finite {what} wrote something before refusing"
            );
        }
    }

    /// A zero-duration measurement is a legitimate placeholder, and fixtures
    /// rely on it. The duration check rejects negatives, not zero.
    #[test]
    fn a_zero_duration_preintegration_is_accepted() {
        let mut map = fixture();
        map.apply_insertion(MapInsertion {
            imu_factors: vec![placeholder_edge(0, 1)],
            ..Default::default()
        })
        .unwrap();
        let update = map.ba_snapshot().into_update();
        assert_eq!(update.imu_preintegrations[0].dt, 0.0);

        let result = map
            .apply_ba_update(update)
            .expect("a zero-duration placeholder is valid");
        assert_eq!(result.map_points_updated, 0);
    }

    /// `SO3F64` wraps a raw quaternion and guarantees nothing, so alignment
    /// checks the representation before it scales anything.
    #[test]
    fn an_invalid_alignment_rotation_is_refused_before_scaling() {
        let unnormalized = SO3F64::from_array([0.0, 0.0, 0.0, 0.5]);
        let non_finite = SO3F64::from_array([f64::NAN, 0.0, 0.0, 1.0]);

        for rotation in [unnormalized, non_finite] {
            let mut map = fixture();
            let before = map.state_fingerprint_for_test();

            let error = map
                .apply_inertial_alignment(InertialAlignment {
                    scale: 2.0,
                    rotation,
                    keyframe_velocities: vec![KeyframeVelocity {
                        keyframe_idx: 0,
                        velocity_world: Vec3F64::new(1.0, 0.0, 0.0),
                    }],
                    bias: ImuBias::default(),
                })
                .unwrap_err();

            assert_eq!(error, InertialAlignmentError::InvalidRotation);
            assert_eq!(
                map.state_fingerprint_for_test(),
                before,
                "an invalid rotation left a scaled world behind"
            );
        }
    }

    /// A rotation that reached `f64` through `f32` storage carries a norm
    /// error of a few `f32` epsilons and must still be accepted.
    #[test]
    fn a_rotation_rounded_through_f32_is_still_accepted() {
        let exact = SO3F64::exp(Vec3F64::new(0.3, -0.4, 0.5));
        let rounded = exact.to_array().map(|v| v as f32 as f64);
        let rotation = SO3F64::from_array(rounded);
        let norm = rounded.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() > 0.0 && (norm - 1.0).abs() <= ROTATION_NORM_TOLERANCE,
            "fixture should be slightly off unit norm: {norm}"
        );

        let mut map = fixture();
        map.apply_inertial_alignment(InertialAlignment {
            scale: 1.5,
            rotation,
            keyframe_velocities: vec![KeyframeVelocity {
                keyframe_idx: 0,
                velocity_world: Vec3F64::new(1.0, 0.0, 0.0),
            }],
            bias: ImuBias::default(),
        })
        .expect("an f32-rounded rotation is a valid rotation");
    }

    /// Finite inputs can overflow while being scaled: the composed world
    /// transform is checked before the first write.
    #[test]
    fn an_overflowing_alignment_is_refused_before_any_write() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        // `insert_landmark` requires a finite length, which caps a coordinate
        // near 1e154; scaling that overflows.
        map.insert_landmark(seeded(0, 0, Vec3F64::new(1e154, 0.0, 0.0)))
            .unwrap();
        let before = map.state_fingerprint_for_test();

        let error = map
            .apply_inertial_alignment(InertialAlignment {
                scale: 1e300,
                rotation: SO3F64::IDENTITY,
                keyframe_velocities: vec![KeyframeVelocity {
                    keyframe_idx: 0,
                    velocity_world: Vec3F64::ZERO,
                }],
                bias: ImuBias::default(),
            })
            .unwrap_err();

        assert_eq!(error, InertialAlignmentError::NonFiniteTransform);
        assert_eq!(
            map.state_fingerprint_for_test(),
            before,
            "an overflowing scale was partly applied"
        );
    }

    /// The stored pose is the *inverse* of the scaled camera-to-world pose, and
    /// that inversion is its own overflow site: with a 45-degree rotation the
    /// two scaled translation components add instead of cancelling. Validating
    /// the camera centre alone would let an infinite stored translation
    /// through, taking the epoch with it.
    #[test]
    fn an_alignment_overflowing_only_after_inversion_is_refused() {
        let quarter = std::f64::consts::FRAC_PI_4;
        let (sin, cos) = quarter.sin_cos();
        let yaw = Mat3F64::from_cols_array(&[cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0]);
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.set_keyframe_pose_for_test(0, Pose3d::new(yaw, Vec3F64::new(1.5, 0.0, 0.0)));
        let before = map.state_fingerprint_for_test();

        // The scaled camera centre stays finite at ~1.38e308 either way; only
        // the stored pose overflows.
        let error = map
            .apply_inertial_alignment(InertialAlignment {
                scale: 1.3e308,
                rotation: SO3F64::IDENTITY,
                keyframe_velocities: vec![KeyframeVelocity {
                    keyframe_idx: 0,
                    velocity_world: Vec3F64::ZERO,
                }],
                bias: ImuBias::default(),
            })
            .unwrap_err();

        assert_eq!(error, InertialAlignmentError::NonFiniteTransform);
        assert_eq!(
            map.state_fingerprint_for_test(),
            before,
            "an alignment that overflows on inversion was partly applied"
        );
    }

    /// The same hazard on the pose-graph side: every transported position is
    /// computed and checked before the epoch advances.
    #[test]
    fn an_overflowing_pose_graph_correction_is_refused_before_any_write() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        map.insert_landmark(seeded(0, 0, Vec3F64::new(0.0, 0.0, 5.0)))
            .unwrap();
        // Both poses are finite, but the correction composes as
        // `t_before - t_after`, which leaves f64 range.
        let live = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1e308, 0.0, 0.0));
        map.set_keyframe_pose_for_test(0, live);
        let before = map.state_fingerprint_for_test();

        let after = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-1e308, 0.0, 0.0));
        let error = map
            .apply_pose_graph_correction(&[0], &[live], &[after])
            .unwrap_err();

        assert!(
            matches!(error, PoseGraphCorrectionError::NonFiniteCorrection),
            "expected a non-finite computed correction, got {error:?}"
        );
        assert_eq!(
            map.state_fingerprint_for_test(),
            before,
            "an overflowing pose graph correction was partly applied"
        );
    }
}
