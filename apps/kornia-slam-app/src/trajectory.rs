//! Per-frame trajectory file in the LaMAria/TUM layout:
//! `timestamp_ns tx ty tz qx qy qz qw`, one line per tracked image.

use std::io::Write;
use std::path::Path;

use kornia_3d::pose::Pose3d;
use kornia_algebra::SO3F64;

pub struct TrajectoryWriter {
    out: std::io::BufWriter<std::fs::File>,
    /// `T_CB`; `None` writes the camera pose.
    body_to_camera: Option<Pose3d>,
}

impl TrajectoryWriter {
    /// Writes `world_from_body` when `camera_to_body` (`T_BC`) is given, else
    /// `world_from_camera`.
    pub fn create(path: &Path, camera_to_body: Option<Pose3d>) -> std::io::Result<Self> {
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            out: std::io::BufWriter::new(std::fs::File::create(path)?),
            body_to_camera: camera_to_body.map(|t_bc| t_bc.inverse()),
        })
    }

    pub fn write(&mut self, timestamp_sec: f64, pose_world_to_cam: &Pose3d) -> std::io::Result<()> {
        let world_from_cam = pose_world_to_cam.inverse();
        let pose = match &self.body_to_camera {
            Some(t_cb) => world_from_cam.compose(t_cb),
            None => world_from_cam,
        };
        writeln!(self.out, "{}", format_line(timestamp_sec, &pose))
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

fn format_line(timestamp_sec: f64, world_from_frame: &Pose3d) -> String {
    let t = world_from_frame.translation;
    let [qx, qy, qz, qw] = SO3F64::from_matrix(&world_from_frame.rotation).to_array();
    format!(
        "{} {:.9} {:.9} {:.9} {qx:.9} {qy:.9} {qz:.9} {qw:.9}",
        timestamp_ns(timestamp_sec),
        t.x,
        t.y,
        t.z
    )
}

/// Sources read integer ns into f64 seconds; for recordings shorter than
/// days the round trip back to ns is exact.
fn timestamp_ns(timestamp_sec: f64) -> i64 {
    (timestamp_sec * 1e9).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    #[test]
    fn integer_ns_timestamps_survive_the_round_trip() {
        for ns in [905_433_591_250_i64, 1_321_883_591_212, 86_400_000_000_123] {
            assert_eq!(timestamp_ns(ns as f64 * 1e-9), ns);
        }
    }

    #[test]
    fn the_line_holds_the_world_from_frame_pose() {
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));
        assert_eq!(
            format_line(1.5, &pose),
            "1500000000 1.000000000 -2.000000000 3.000000000 \
             0.000000000 0.000000000 0.000000000 1.000000000"
        );
    }
}
