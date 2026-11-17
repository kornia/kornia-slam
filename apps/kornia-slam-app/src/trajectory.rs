//! Per-frame trajectory file in the LaMAria/TUM layout:
//! `timestamp_ns tx ty tz qx qy qz qw`, one line per tracked image.

use std::collections::HashMap;
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

/// `--trajectory-smoothed`: each tracked frame re-anchored at shutdown to
/// its reference keyframe's final pose, ORB-SLAM3's `SaveTrajectoryEuRoC`.
/// A keyframe culled by then stands in through its parent, the reference
/// keyframe it replaced, as ORB-SLAM3 walks `GetParent`.
#[derive(Default)]
pub struct SmoothedTrajectory {
    frames: Vec<SmoothedFrame>,
    /// Keyframe -> (parent keyframe, `T_kf_from_parent`), at first reference.
    parents: HashMap<usize, (usize, Pose3d)>,
    last_reference: Option<(usize, Pose3d)>,
}

struct SmoothedFrame {
    timestamp_sec: f64,
    reference_kf: usize,
    /// `T_frame_from_kf`, camera frames, when the frame was tracked.
    frame_from_kf: Pose3d,
    /// The causal `T_cw`, written when no keyframe of the chain survives.
    pose_world_to_cam: Pose3d,
}

impl SmoothedTrajectory {
    /// Records a tracked frame against its reference keyframe `reference_kf`
    /// at that keyframe's current `T_cw`.
    pub fn record(
        &mut self,
        timestamp_sec: f64,
        pose_world_to_cam: &Pose3d,
        reference_kf: usize,
        reference_pose: &Pose3d,
    ) {
        if let Some((parent, parent_pose)) = self.last_reference
            && parent != reference_kf
        {
            self.parents
                .entry(reference_kf)
                .or_insert((parent, Pose3d::between(&parent_pose, reference_pose)));
        }
        self.last_reference = Some((reference_kf, *reference_pose));
        self.frames.push(SmoothedFrame {
            timestamp_sec,
            reference_kf,
            frame_from_kf: Pose3d::between(reference_pose, pose_world_to_cam),
            pose_world_to_cam: *pose_world_to_cam,
        });
    }

    /// Writes every recorded frame; `final_pose(kf)` is the keyframe's `T_cw`
    /// at shutdown, `None` once culled. Returns how many frames had no
    /// surviving keyframe and were written at their causal pose.
    pub fn write(
        &self,
        mut writer: TrajectoryWriter,
        final_pose: impl Fn(usize) -> Option<Pose3d>,
    ) -> std::io::Result<usize> {
        let mut unanchored = 0;
        for frame in &self.frames {
            let pose = match self.final_keyframe_pose(frame.reference_kf, &final_pose) {
                Some(kf_pose) => frame.frame_from_kf.compose(&kf_pose),
                None => {
                    unanchored += 1;
                    frame.pose_world_to_cam
                }
            };
            writer.write(frame.timestamp_sec, &pose)?;
        }
        writer.finish()?;
        Ok(unanchored)
    }

    fn final_keyframe_pose(
        &self,
        kf: usize,
        final_pose: &impl Fn(usize) -> Option<Pose3d>,
    ) -> Option<Pose3d> {
        let mut kf_from_ancestor = Pose3d::IDENTITY;
        let mut current = kf;
        // Parents always precede their children, so the chain ends.
        loop {
            if let Some(pose) = final_pose(current) {
                return Some(kf_from_ancestor.compose(&pose));
            }
            let (parent, child_from_parent) = self.parents.get(&current)?;
            kf_from_ancestor = kf_from_ancestor.compose(child_from_parent);
            current = *parent;
        }
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

    fn translation(x: f64) -> Pose3d {
        Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, 0.0, 0.0))
    }

    fn written(path: &Path) -> Vec<f64> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| line.split(' ').nth(1).unwrap().parse().unwrap())
            .collect()
    }

    /// A frame follows its keyframe's final pose; a culled keyframe is
    /// reached through its parent; with no survivor the causal pose is kept.
    #[test]
    fn frames_follow_their_keyframe_or_its_parent_to_the_final_pose() {
        let mut smoothed = SmoothedTrajectory::default();
        smoothed.record(0.0, &translation(1.5), 10, &translation(1.0));
        smoothed.record(0.1, &translation(2.5), 20, &translation(2.0));
        smoothed.record(0.2, &translation(3.5), 30, &translation(3.0));
        let dir = std::env::temp_dir().join(format!("smoothed-{}", std::process::id()));
        let path = dir.join("t.tum");
        let writer = TrajectoryWriter::create(&path, None).unwrap();
        // Keyframe 10 moved by +1 m, 20 was culled, 30's chain is gone too.
        let final_pose = |kf| (kf == 10).then(|| translation(2.0));
        let mut gone = smoothed.parents.clone();
        gone.remove(&30);
        let unanchored = SmoothedTrajectory {
            parents: gone,
            ..smoothed
        }
        .write(writer, final_pose)
        .unwrap();
        // `T_cw` translations: camera centres are the negation.
        assert_eq!(written(&path), vec![-2.5, -3.5, -3.5]);
        assert_eq!(unanchored, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

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
