//! A recorded session: the keyframes, landmarks, IMU samples and tracked
//! frames of an online run, for the offline global VI-BA
//! ([`super::global_vi_ba`]). Plain arrays keep serde off kornia-rs types.

use kornia_sensors::SensorRig;

use super::map::{Keyframe, Map, SegmentId};

/// Version of [`SessionDump`] this crate writes and reads.
pub const SESSION_DUMP_VERSION: u32 = 1;

/// Rotation as a unit quaternion `[x, y, z, w]` and translation, metres.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionPose {
    pub rotation_xyzw: [f64; 4],
    pub translation: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
    /// `T_PC`, as `RigCamera::primary_from_camera`.
    pub primary_from_camera: SessionPose,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionRig {
    pub cameras: Vec<SessionCamera>,
    /// Primary camera to body (IMU), `X_b = T_bc · X_c`.
    pub camera_to_body: SessionPose,
    /// `[gyro_noise, accel_noise, gyro_bias_noise, accel_bias_noise]`, as `ImuCalib`.
    pub imu_noise: [f64; 4],
    /// Already applied to [`SessionDump::imu`]; recorded for the report.
    pub imu_time_offset_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionObservation {
    pub landmark: u32,
    pub camera: u8,
    pub octave: u8,
    /// Undistorted pixel in `camera`.
    pub u: f32,
    pub v: f32,
    /// Stereo depth (m) when the frame had one.
    pub depth: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionKeyframe {
    pub kf_idx: u64,
    pub timestamp_sec: f64,
    /// World to primary camera, `T_cw`.
    pub pose_cw: SessionPose,
    pub velocity_world: [f64; 3],
    pub bias_gyro: [f64; 3],
    pub bias_accel: [f64; 3],
    pub segment: SegmentId,
    pub inserted_while_lost: bool,
    pub anchor: bool,
    /// Removed from the live map by a bound before the dump.
    pub evicted: bool,
    pub observations: Vec<SessionObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionLandmark {
    pub idx: u32,
    pub position_world: [f64; 3],
}

/// One tracked frame as the smoothed trajectory recorded it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionFrame {
    pub timestamp_sec: f64,
    pub reference_kf: u64,
    pub frame_from_kf: SessionPose,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImuSample {
    pub t: f64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionDump {
    pub version: u32,
    pub rig: SessionRig,
    pub keyframes: Vec<SessionKeyframe>,
    pub landmarks: Vec<SessionLandmark>,
    pub frames: Vec<SessionFrame>,
    pub imu: Vec<ImuSample>,
}

/// Keyframes (live, then `evicted`) and live landmarks of `map`, tagged by `segment_of`.
pub fn snapshot_map(
    _map: &Map,
    _rig: &SensorRig,
    _evicted: &[Keyframe],
    _segment_of: impl Fn(usize) -> SegmentId,
) -> (Vec<SessionKeyframe>, Vec<SessionLandmark>) {
    todo!("lane D: session snapshot")
}
