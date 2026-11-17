//! Upstream target: kornia_3d::pose. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

/// Camera centre in the world frame of a world-to-camera pose.
pub fn camera_centre(pose_world_to_cam: &Pose3d) -> Vec3F64 {
    pose_world_to_cam.inverse().translation
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
