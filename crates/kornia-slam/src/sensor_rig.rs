//! Fixed sensor calibration, independent of the estimated SLAM state.

use kornia_3d::{camera::PinholeCamera, pose::Pose3d};
use kornia_sensors::imu::ImuCalib;

/// Camera calibration and optional IMU calibration used by a SLAM system.
///
/// The camera model describes the images supplied to the system. For rectified
/// images, both its intrinsics and the IMU extrinsics must refer to that camera.
/// Estimated bias, gravity, poses and buffered measurements are runtime state.
pub struct SensorRig {
    pub camera: PinholeCamera,
    /// `None` selects visual-only operation.
    pub imu: Option<ImuCalibration>,
}

/// IMU noise parameters and its rigid relationship to the camera.
pub struct ImuCalibration {
    pub noise: ImuCalib,
    /// Camera-to-body transform `T_BC`: `X_body = T_BC * X_camera`.
    /// The body frame is the frame of the supplied IMU measurements.
    pub camera_to_body: Pose3d,
}

impl ImuCalibration {
    /// Uses the system's historical IMU noise values with the given extrinsic.
    /// Supply `noise` explicitly when calibration for the actual sensor is available.
    pub fn new(camera_to_body: Pose3d) -> Self {
        Self {
            noise: default_imu_noise(),
            camera_to_body,
        }
    }
}

// Also retained for preintegration in the existing visual-only map paths.
pub(crate) fn default_imu_noise() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.6968e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9393e-5,
        accel_bias_noise: 3.0e-3,
    }
}
