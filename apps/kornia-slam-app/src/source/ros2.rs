//! ROS 2 topic settings; the node itself needs the `ros2` feature and a sourced
//! ROS 2 installation (see INTEGRATIONS.md).

use kornia_slam::SensorSelection;
use serde::Deserialize;

use super::{OpenedSource, SourceError, finite, positive};

#[cfg(any(feature = "ros2", test))]
mod convert;
#[cfg(feature = "ros2")]
mod node;
#[cfg(any(feature = "ros2", test))]
mod sync;

#[cfg(feature = "ros2")]
use node::Ros2Source;

/// Rectified images, their `CameraInfo` and optionally an IMU, subscribed from
/// ROS 2 topics. Relative topic names resolve under the node's namespace, so
/// they can be remapped with `--ros-args`.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "ros2"), expect(dead_code))]
#[serde(deny_unknown_fields)]
pub struct Ros2Config {
    #[serde(default = "default_node_name")]
    pub node_name: String,
    /// `sensor_msgs/Image`: mono8, rgb8 or bgr8; the left view with stereo cameras.
    #[serde(default = "default_image")]
    pub image: String,
    /// `sensor_msgs/CameraInfo`; its projection matrix `P` is the camera.
    #[serde(default = "default_camera_info")]
    pub camera_info: String,
    #[serde(default = "default_right_image")]
    pub right_image: String,
    #[serde(default = "default_right_camera_info")]
    pub right_camera_info: String,
    /// `sensor_msgs/Imu`, read with the IMU sensor enabled.
    #[serde(default = "default_imu")]
    pub imu: String,
    /// Camera-to-IMU transform `T_BC`; required with the IMU.
    #[serde(default)]
    pub camera_to_imu: Option<Extrinsic>,
    /// Frames waiting for the tracker; the oldest is dropped beyond this.
    #[serde(default = "default_queue")]
    pub queue: usize,
    /// Seconds to wait for `CameraInfo` at startup.
    #[serde(default = "default_wait_s")]
    pub wait_s: f64,
    /// 0 runs until stopped.
    #[serde(default)]
    pub max_frames: usize,
}

/// A rigid transform `X_to = R * X_from + t`, in metres.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extrinsic {
    /// Row-major 3×3.
    pub rotation: Vec<f64>,
    pub translation: Vec<f64>,
}

impl Extrinsic {
    fn validate(&self) -> Result<(), String> {
        if self.rotation.len() != 9 || self.translation.len() != 3 {
            return Err("camera_to_imu needs 9 rotation and 3 translation values".into());
        }
        for value in self.rotation.iter().chain(&self.translation) {
            finite("camera_to_imu", *value)?;
        }
        Ok(())
    }
}

fn default_node_name() -> String {
    "kornia_slam".into()
}

fn default_image() -> String {
    "left/image_rect".into()
}

fn default_camera_info() -> String {
    "left/camera_info".into()
}

fn default_right_image() -> String {
    "right/image_rect".into()
}

fn default_right_camera_info() -> String {
    "right/camera_info".into()
}

fn default_imu() -> String {
    "imu".into()
}

fn default_queue() -> usize {
    4
}

fn default_wait_s() -> f64 {
    10.0
}

impl Ros2Config {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        if sensors.imu && self.camera_to_imu.is_none() {
            return Err("Ros2 with the IMU needs `camera_to_imu`".into());
        }
        if let Some(extrinsic) = &self.camera_to_imu {
            extrinsic.validate()?;
        }
        positive("queue", self.queue as f64)?;
        positive("wait_s", self.wait_s)
    }

    #[cfg(feature = "ros2")]
    pub(super) fn open(&self, sensors: SensorSelection) -> Result<OpenedSource, SourceError> {
        let source = Ros2Source::open(self, sensors)?;
        Ok(OpenedSource {
            summary: Some(source.summary()),
            ground_truth: None,
            source: Box::new(source),
        })
    }

    #[cfg(not(feature = "ros2"))]
    pub(super) fn open(&self, _sensors: SensorSelection) -> Result<OpenedSource, SourceError> {
        Err(SourceError::other(
            "this build has no ROS 2 support; rebuild with `--features ros2`",
        ))
    }
}
