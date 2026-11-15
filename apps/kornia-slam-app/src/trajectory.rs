//! Per-frame trajectory export in the TUM format read by benchmark tools
//! (e.g. LaMAria's `evaluate_wrt_mps` / `evaluate_wrt_control_points`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use kornia_3d::pose::Pose3d;
use kornia_algebra::SO3F64;

/// An estimated pose; `world_from_sensor` maps sensor-frame points into the world frame.
pub struct StampedPose {
    pub timestamp_sec: f64,
    pub world_from_sensor: Pose3d,
}

/// Pose of the body (IMU) frame in the world when `camera_to_body` (`T_BC`) is
/// given, otherwise of the camera.
pub fn world_from_sensor(pose_world_to_cam: &Pose3d, camera_to_body: Option<&Pose3d>) -> Pose3d {
    let world_from_cam = pose_world_to_cam.inverse();
    match camera_to_body {
        Some(body_from_cam) => world_from_cam.compose(&body_from_cam.inverse()),
        None => world_from_cam,
    }
}

/// Writes one `timestamp_ns tx ty tz qx qy qz qw` line per pose, creating parent directories.
pub fn write_tum(path: &Path, poses: &[StampedPose]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = BufWriter::new(File::create(path)?);
    for pose in poses {
        writeln!(out, "{}", tum_line(pose))?;
    }
    out.flush()
}

// Timestamps arrive as f64 seconds, so nanoseconds are exact only while
// `t·1e9` fits f64 precision: device clocks (LaMAria) do, Unix epochs (EuRoC)
// round to ~100 ns.
fn tum_line(pose: &StampedPose) -> String {
    let t = pose.world_from_sensor.translation;
    let [qx, qy, qz, qw] = SO3F64::from_matrix(&pose.world_from_sensor.rotation).to_array();
    let timestamp_ns = (pose.timestamp_sec * 1e9).round() as u64;
    format!(
        "{timestamp_ns} {:.9} {:.9} {:.9} {qx:.9} {qy:.9} {qz:.9} {qw:.9}",
        t.x, t.y, t.z
    )
}

#[cfg(test)]
mod tests {
    use kornia_algebra::{Mat3F64, Vec3F64};

    use super::*;

    #[test]
    fn camera_pose_is_the_inverse_of_world_to_camera() {
        let pose_world_to_cam = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));
        let pose = world_from_sensor(&pose_world_to_cam, None);
        assert_eq!(pose.translation, Vec3F64::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn body_pose_applies_the_camera_to_body_lever_arm() {
        // Camera at the world origin, sitting 1 m along body z: the body origin
        // is 1 m behind the camera.
        let body_from_cam = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 1.0));
        let pose = world_from_sensor(&Pose3d::IDENTITY, Some(&body_from_cam));
        assert_eq!(pose.translation, Vec3F64::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn tum_line_uses_nanoseconds_and_scalar_last_quaternion() {
        let pose = StampedPose {
            timestamp_sec: 1389.350666375,
            world_from_sensor: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, -1.0)),
        };
        assert_eq!(
            tum_line(&pose),
            "1389350666375 0.500000000 0.000000000 -1.000000000 \
             0.000000000 0.000000000 0.000000000 1.000000000"
        );
    }
}
