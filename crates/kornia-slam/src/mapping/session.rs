//! A recorded session: the keyframes, landmarks, IMU samples and tracked
//! frames of an online run, for the offline global VI-BA
//! ([`super::global_vi_ba`]). Plain arrays keep serde off kornia-rs types.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_image::ImageSize;
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

impl SessionPose {
    pub fn from_pose(pose: &Pose3d) -> Self {
        Self {
            rotation_xyzw: SO3F64::from_matrix(&pose.rotation).to_array(),
            translation: pose.translation.to_array(),
        }
    }

    pub fn to_pose(&self) -> Pose3d {
        Pose3d::new(
            SO3F64::from_array(self.rotation_xyzw).matrix(),
            Vec3F64::from_array(self.translation),
        )
    }
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
    /// Already applied to [`SessionDump::imu`]; the offline report adds its
    /// residual `td` to it.
    pub imu_time_offset_ns: i64,
    /// Baseline (m) behind the stereo depths: the rectified pair's, or the
    /// primary-to-camera-1 distance of a rig; `None` without stereo.
    pub stereo_baseline_m: Option<f64>,
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
    /// The online world's gravity at the dump (m/s²); the IMU edges hold it fixed.
    pub gravity_world: [f64; 3],
}

/// The rig of `rig` as a dump records it. `image_size` is the primary
/// camera's, which a rig without [`SensorRig::cameras`] does not carry.
pub fn session_rig(rig: &SensorRig, image_size: ImageSize, imu_time_offset_ns: i64) -> SessionRig {
    let cameras = if rig.cameras.is_empty() {
        vec![session_camera(&rig.camera, image_size, &Pose3d::IDENTITY)]
    } else {
        rig.cameras
            .iter()
            .map(|c| session_camera(&c.camera, c.size, &c.primary_from_camera))
            .collect()
    };
    let noise = rig.imu_noise();
    let rig_baseline = rig
        .is_multi_camera()
        .then(|| rig.cameras.get(1))
        .flatten()
        .map(|c| c.primary_from_camera.translation.length());
    SessionRig {
        cameras,
        camera_to_body: SessionPose::from_pose(&rig.camera_to_body().unwrap_or(Pose3d::IDENTITY)),
        imu_noise: [
            noise.gyro_noise,
            noise.accel_noise,
            noise.gyro_bias_noise,
            noise.accel_bias_noise,
        ],
        imu_time_offset_ns,
        stereo_baseline_m: rig.stereo_baseline_m.or(rig_baseline),
    }
}

fn session_camera(
    camera: &PinholeCamera,
    size: ImageSize,
    primary_from_camera: &Pose3d,
) -> SessionCamera {
    SessionCamera {
        fx: camera.fx,
        fy: camera.fy,
        cx: camera.cx,
        cy: camera.cy,
        width: size.width as u32,
        height: size.height as u32,
        primary_from_camera: SessionPose::from_pose(primary_from_camera),
    }
}

/// Keyframes (live, then `evicted`) and live landmarks of `map`, tagged by
/// `segment_of` and stamped by `timestamp_of` (a keyframe holds no stamp of
/// its own). Observations of a landmark the map no longer holds, or of a
/// keypoint the camera cannot undistort, are left out.
pub fn snapshot_map(
    map: &Map,
    rig: &SensorRig,
    evicted: &[Keyframe],
    segment_of: impl Fn(usize) -> SegmentId,
    timestamp_of: impl Fn(usize) -> f64,
) -> (Vec<SessionKeyframe>, Vec<SessionLandmark>) {
    let points = map.map_points();
    let live = |mp: usize| points.get(mp).is_some_and(|p| !p.culled);
    let keyframes = map
        .keyframes()
        .iter()
        .map(|kf| (kf, false))
        .chain(evicted.iter().map(|kf| (kf, true)))
        .map(|(kf, evicted)| SessionKeyframe {
            kf_idx: kf.frame.idx as u64,
            timestamp_sec: timestamp_of(kf.frame.idx),
            pose_cw: SessionPose::from_pose(&kf.frame.pose_world_to_cam),
            velocity_world: kf.velocity_world.to_array(),
            bias_gyro: kf.imu_bias.gyro.to_array(),
            bias_accel: kf.imu_bias.accel.to_array(),
            segment: segment_of(kf.frame.idx),
            inserted_while_lost: kf.inserted_while_lost,
            anchor: kf.anchor,
            evicted,
            observations: observations(kf, rig, &live),
        })
        .collect();
    let landmarks = points
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.culled)
        .map(|(idx, p)| SessionLandmark {
            idx: idx as u32,
            position_world: p.position.to_array(),
        })
        .collect();
    (keyframes, landmarks)
}

fn observations(
    kf: &Keyframe,
    rig: &SensorRig,
    live: &impl Fn(usize) -> bool,
) -> Vec<SessionObservation> {
    kf.map_point_by_desc_idx
        .iter()
        .enumerate()
        .filter_map(|(i, mp)| {
            let landmark = mp.filter(|&mp| live(mp))?;
            let [u, v] = kf.frame.undistorted_xy(i, &rig.camera)?;
            Some(SessionObservation {
                landmark: landmark as u32,
                camera: kf.frame.camera_of(i) as u8,
                octave: kf.frame.features.octaves.get(i).copied().unwrap_or(0),
                u,
                v,
                depth: kf.frame.stereo_depth(i),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pose_survives_the_quaternion_round_trip() {
        let pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix(),
            Vec3F64::new(1.0, -2.0, 3.0),
        );
        let back = SessionPose::from_pose(&pose).to_pose();
        let diff: [f64; 9] = (back.rotation - pose.rotation).into();
        assert!(diff.iter().all(|v| v.abs() < 1e-12), "{diff:?}");
        assert_eq!(back.translation, pose.translation);
    }
}
