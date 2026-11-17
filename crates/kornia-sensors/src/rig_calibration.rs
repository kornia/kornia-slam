//! A rig calibration file: exactly what [`SensorRig`] stores, so loading is a
//! copy and not a recomputation (ORB-SLAM3's `Stereo.T_c1_c2` plus
//! `IMU.T_b_c1`, `Settings.cc:342, 422`).

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;

use crate::imu::ImuCalib;
use crate::rig::{RigCamera, SensorRig};

/// Camera geometry of a rig, its primary camera-to-body extrinsic and IMU noise.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCalibration {
    /// The primary first.
    pub cameras: Vec<RigCameraCalibration>,
    /// `T_BC` of the primary, row-major, last row `0 0 0 1`.
    pub camera_to_body: Option<[[f64; 4]; 4]>,
    pub imu_noise: Option<ImuCalib>,
}

/// One camera of a [`RigCalibration`]: undistorted pinhole intrinsics and
/// `T_PC`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCameraCalibration {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
    pub width: usize,
    pub height: usize,
    /// `T_PC`, row-major (ORB-SLAM3 `Stereo.T_c1_c2`); identity for `cameras[0]`.
    pub primary_from_camera: [[f64; 4]; 4],
}

/// A [`RigCalibration`] that does not describe a rig.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RigCalibrationError {
    #[error("a rig calibration needs at least one camera")]
    NoCameras,
    #[error("cameras[0] is the primary: its primary_from_camera must be the identity")]
    PrimaryNotIdentity,
    #[error("camera {camera}: rotation is not orthonormal (defect {defect})")]
    NotRigid { camera: usize, defect: f64 },
    #[error("camera {camera}: the last row of the transform is not 0 0 0 1")]
    NotHomogeneous { camera: usize },
}

impl RigCalibration {
    pub fn from_sensor_rig(_rig: &SensorRig) -> Self {
        todo!("lane B: rig calibration file")
    }

    /// Replaces `rig`'s camera geometry, plus `camera_to_body` and the noise
    /// when `rig.imu` is `Some`. A rig without IMU stays without one: the IMU
    /// selection belongs to the system config.
    pub fn apply_to(&self, _rig: SensorRig) -> Result<SensorRig, RigCalibrationError> {
        todo!("lane B: rig calibration file")
    }
}

/// Rig cameras from per-camera body extrinsics `T_BC` (Kalibr, LaMAria, EuRoC):
/// `T_PC = T_BC[primary]⁻¹ ∘ T_BC[c]`, the primary first.
pub fn rig_cameras_from_body_extrinsics(
    _cameras: &[(PinholeCamera, ImageSize, Pose3d)],
    _primary: usize,
) -> Vec<RigCamera> {
    todo!("lane B: rig cameras from body extrinsics")
}
