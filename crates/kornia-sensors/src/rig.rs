//! Fixed calibration of a camera rig with optional stereo and IMU, independent
//! of any estimated state.

use crate::imu::ImuCalib;
use kornia_3d::{camera::PinholeCamera, pose::Pose3d};

/// Camera calibration plus optional stereo and IMU calibration.
///
/// The camera model describes the images the rig supplies. For rectified
/// images, its intrinsics, the stereo baseline and the IMU extrinsics must all
/// refer to that camera.
/// Estimated bias, gravity and poses are runtime state of the consumer.
#[derive(Debug, Clone)]
pub struct SensorRig {
    pub camera: PinholeCamera,
    /// Metric baseline (metres) of a rectified stereo pair whose left view is
    /// `camera`. `None` for a monocular rig.
    pub stereo_baseline_m: Option<f64>,
    /// `None` selects visual-only operation.
    pub imu: Option<ImuCalibration>,
}

impl SensorRig {
    /// A visual-only rig. Add IMU calibration with [`SensorRig::with_imu`].
    pub fn new(camera: PinholeCamera) -> Self {
        Self {
            camera,
            stereo_baseline_m: None,
            imu: None,
        }
    }

    /// Declares the images as a rectified stereo pair with the given baseline in metres.
    pub fn with_stereo_baseline(mut self, baseline_m: f64) -> Self {
        self.stereo_baseline_m = Some(baseline_m);
        self
    }

    /// Adds an IMU with the given camera-to-body extrinsic and the default
    /// noise parameters (see [`ImuCalibration::new`]).
    pub fn with_imu(mut self, camera_to_body: Pose3d) -> Self {
        self.imu = Some(ImuCalibration::new(camera_to_body));
        self
    }

    /// Stereo `bf = fx * baseline` (pixel·metres), the constant in
    /// `depth = bf / disparity`. `None` for a monocular rig.
    pub fn stereo_bf(&self) -> Option<f64> {
        self.stereo_baseline_m.map(|b| self.camera.fx * b)
    }

    /// Camera-to-body transform `T_BC`, or `None` for a visual-only rig.
    pub fn camera_to_body(&self) -> Option<Pose3d> {
        self.imu.as_ref().map(|imu| imu.camera_to_body)
    }

    /// Noise parameters to preintegrate with; the defaults for a rig without IMU
    /// calibration.
    pub fn imu_noise(&self) -> ImuCalib {
        self.imu
            .as_ref()
            .map_or_else(default_imu_noise, |imu| imu.noise)
    }
}

/// IMU noise parameters and its rigid relationship to the camera.
#[derive(Debug, Clone, Copy)]
pub struct ImuCalibration {
    pub noise: ImuCalib,
    /// Camera-to-body transform `T_BC`: `X_body = T_BC * X_camera`.
    /// The body frame is the frame of the supplied IMU measurements.
    pub camera_to_body: Pose3d,
}

impl ImuCalibration {
    /// Uses the EuRoC ADIS16448 noise densities with the given extrinsic.
    /// Supply `noise` explicitly when calibration for the actual sensor is
    /// available.
    pub fn new(camera_to_body: Pose3d) -> Self {
        Self {
            noise: default_imu_noise(),
            camera_to_body,
        }
    }
}

/// EuRoC MAV ADIS16448 noise densities and random walks (`imu0/sensor.yaml`).
fn default_imu_noise() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.6968e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9393e-5,
        accel_bias_noise: 3.0e-3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn a_rig_without_imu_is_visual_only() {
        let rig = SensorRig::new(test_camera());
        assert!(rig.imu.is_none());
        assert!(rig.camera_to_body().is_none());
    }

    #[test]
    fn stereo_bf_is_focal_times_baseline() {
        let rig = SensorRig::new(test_camera());
        assert!(rig.stereo_bf().is_none());

        let rig = rig.with_stereo_baseline(0.11);
        assert_eq!(rig.stereo_bf(), Some(rig.camera.fx * 0.11));
    }

    #[test]
    fn default_imu_noise_is_euroc() {
        for rig in [
            SensorRig::new(test_camera()),
            SensorRig::new(test_camera()).with_imu(Pose3d::IDENTITY),
        ] {
            let noise = rig.imu_noise();
            assert_eq!(noise.gyro_noise, 1.6968e-4);
            assert_eq!(noise.accel_noise, 2.0e-3);
            assert_eq!(noise.gyro_bias_noise, 1.9393e-5);
            assert_eq!(noise.accel_bias_noise, 3.0e-3);
        }
    }
}
