//! Upstream target: kornia_3d::pose. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SE3F32, SO3F32, Vec3F64};

use crate::upstream::kornia_algebra::cast::{mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64};

/// Camera centre in the world frame of a world-to-camera pose.
pub fn camera_centre(pose_world_to_cam: &Pose3d) -> Vec3F64 {
    pose_world_to_cam.inverse().translation
}

pub fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    // The quaternion of a scaled matrix is not unit, and `retract` keeps it so.
    let rotation = SO3F32::from_matrix(&mat3_to_f32(pose.rotation));
    SE3F32::new(
        SO3F32::from_quaternion(rotation.q.normalize()),
        vec3_to_f32(pose.translation),
    )
}

/// `Pose3d::inverse` is a transpose, so a scaled rotation fed back through it
/// compounds its scale (the Sophus invariant ORB-SLAM3 and Basalt rely on).
pub fn se3_to_pose(se3: &SE3F32) -> Pose3d {
    Pose3d::new(mat3_to_f64(se3.r.matrix()), vec3_to_f64(se3.t)).orthonormalized()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    #[test]
    fn camera_centre_inverts_the_world_to_camera_translation() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix();
        let centre = Vec3F64::new(4.0, -1.5, 12.0);
        let pose = Pose3d::new(rotation, -(rotation * centre));
        assert!((camera_centre(&pose) - centre).length() < 1e-12);
    }
}
