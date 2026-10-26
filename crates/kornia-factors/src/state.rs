//! Keyframe state as the inertial factors read it.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuBias;

/// A keyframe's world-to-camera pose `T_cw`, world-frame velocity (m/s) and
/// IMU bias.
#[derive(Debug, Clone, Copy)]
pub struct NavState<'a> {
    pub pose_cw: &'a Pose3d,
    pub velocity: Vec3F64,
    pub bias: ImuBias,
}
