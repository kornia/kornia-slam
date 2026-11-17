//! Upstream target: kornia_3d::pgo. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

use super::{
    SparsePgoError, normalized_gravity, pose_to_se3, se3_to_pose, validate_delta, validate_pose,
};

/// The local parameterisation a pose is optimised in. Poses are world-to-camera.
pub trait PoseManifold<const DOF: usize> {
    /// The pose moved by `delta` in this manifold's tangent space.
    fn retract(&self, pose: &Pose3d, delta: &[f32; DOF]) -> Result<Pose3d, SparsePgoError>;
}

/// Full 6-DoF: a right perturbation of the world-to-camera pose.
pub struct Se3Manifold;

impl PoseManifold<6> for Se3Manifold {
    fn retract(&self, pose: &Pose3d, delta: &[f32; 6]) -> Result<Pose3d, SparsePgoError> {
        validate_delta(delta)?;
        se3_to_pose(&pose_to_se3(pose)?.retract(delta))
    }
}

/// 4-DoF for gravity-aligned graphs: the camera centre moves in world coordinates and the pose
/// turns about the world `gravity_axis`, so roll and pitch never change.
pub struct GravityManifold {
    pub(crate) gravity_axis: Vec3F64,
}

impl GravityManifold {
    pub fn new(gravity_axis: Vec3F64) -> Result<Self, SparsePgoError> {
        Ok(Self {
            gravity_axis: normalized_gravity(gravity_axis)?,
        })
    }
}

impl PoseManifold<4> for GravityManifold {
    fn retract(&self, pose: &Pose3d, delta: &[f32; 4]) -> Result<Pose3d, SparsePgoError> {
        validate_pose(pose)?;
        validate_delta(delta)?;

        let center = pose.inverse().translation
            + Vec3F64::new(delta[0] as f64, delta[1] as f64, delta[2] as f64);
        // Positive world-frame yaw corresponds to a negative right perturbation
        // of the world-to-camera rotation.
        let yaw = SO3F64::exp(self.gravity_axis * -(delta[3] as f64)).matrix();
        let rotation = pose.rotation * yaw;
        let retracted = Pose3d::new(rotation, -(rotation * center));
        validate_pose(&retracted)?;
        Ok(retracted)
    }
}
