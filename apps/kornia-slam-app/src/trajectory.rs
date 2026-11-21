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

    /// Writes one line unless the pose is non-finite; returns whether it did.
    pub fn write(
        &mut self,
        timestamp_sec: f64,
        pose_world_to_cam: &Pose3d,
    ) -> std::io::Result<bool> {
        if !is_finite_pose(pose_world_to_cam) {
            return Ok(false);
        }
        let world_from_cam = pose_world_to_cam.inverse();
        let pose = match &self.body_to_camera {
            Some(t_cb) => world_from_cam.compose(t_cb),
            None => world_from_cam,
        };
        writeln!(self.out, "{}", format_line(timestamp_sec, &pose))?;
        Ok(true)
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// Whether every entry of `pose` is finite; a diverged estimate is never written.
pub fn is_finite_pose(pose: &Pose3d) -> bool {
    let t = pose.translation;
    pose.rotation.is_finite() && t.x.is_finite() && t.y.is_finite() && t.z.is_finite()
}

/// `--trajectory-smoothed`: each frame of `--trajectory-out` (tracked, plus
/// the inertial coasts under `--trajectory-include-coasted`) re-anchored at shutdown to
/// its reference keyframe's final pose, ORB-SLAM3's `SaveTrajectoryEuRoC`.
/// A keyframe culled by then stands in through its parent, the reference
/// keyframe it replaced, as ORB-SLAM3 walks `GetParent`.
#[derive(Default)]
pub struct SmoothedTrajectory {
    frames: Vec<SmoothedFrame>,
    /// Keyframe -> (parent keyframe, `T_kf_from_parent`), at first reference.
    parents: HashMap<usize, (usize, Pose3d)>,
    last_reference: Option<(usize, Pose3d)>,
    /// Scored frames with no reference keyframe to record against.
    unrecorded: usize,
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
    /// Records a scored frame against its reference keyframe `reference_kf`
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

    /// Records a scored frame, or counts it as unrecorded when the system has
    /// no reference keyframe (`T_cw`) for it.
    pub fn record_scored(
        &mut self,
        timestamp_sec: f64,
        pose_world_to_cam: &Pose3d,
        reference: Option<(usize, Pose3d)>,
    ) {
        match reference {
            Some((kf, kf_pose)) => self.record(timestamp_sec, pose_world_to_cam, kf, &kf_pose),
            None => self.unrecorded += 1,
        }
    }

    /// Scored frames [`Self::record_scored`] could not record.
    pub fn unrecorded(&self) -> usize {
        self.unrecorded
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

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Every recorded frame as `(timestamp, keyframe, T_frame_from_kf)`
    /// against the nearest keyframe of its parent chain that `kept` holds;
    /// a frame with no such keyframe is left out.
    pub fn anchored_frames<'a>(
        &'a self,
        kept: impl Fn(usize) -> bool + 'a,
    ) -> impl Iterator<Item = (f64, usize, Pose3d)> + 'a {
        self.frames.iter().filter_map(move |frame| {
            let (kf, kf_from_ancestor) =
                self.walk_parents(frame.reference_kf, |kf| kept(kf).then_some(kf))?;
            Some((
                frame.timestamp_sec,
                kf,
                frame.frame_from_kf.compose(&kf_from_ancestor),
            ))
        })
    }

    fn final_keyframe_pose(
        &self,
        kf: usize,
        final_pose: &impl Fn(usize) -> Option<Pose3d>,
    ) -> Option<Pose3d> {
        self.walk_parents(kf, final_pose)
            .map(|(pose, kf_from_ancestor)| kf_from_ancestor.compose(&pose))
    }

    /// The first `found(ancestor)` up the parent chain from `kf`, with
    /// `T_kf_from_ancestor`; `None` when the chain ends or loops first.
    fn walk_parents<T>(
        &self,
        kf: usize,
        found: impl Fn(usize) -> Option<T>,
    ) -> Option<(T, Pose3d)> {
        let mut kf_from_ancestor = Pose3d::IDENTITY;
        let mut current = kf;
        // A return to an older reference keyframe (A, B, A) records A's
        // parent as B after B's as A, so a chain can loop: a walk longer
        // than the parent map has revisited a keyframe.
        for _ in 0..=self.parents.len() {
            if let Some(value) = found(current) {
                return Some((value, kf_from_ancestor));
            }
            let (parent, child_from_parent) = self.parents.get(&current)?;
            kf_from_ancestor = kf_from_ancestor.compose(child_from_parent);
            current = *parent;
        }
        None
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

/// Sources read integer ns into f64 seconds. The round trip back to ns is
/// exact while timestamps stay under about 1e15 ns (11 days) from the
/// clock's origin, not the recording's start; epoch-based ns are not exact.
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

    /// Each frame stands on the nearest kept keyframe up its chain, with the
    /// relative pose composed along the way.
    #[test]
    fn frames_are_anchored_to_the_nearest_kept_ancestor() {
        let mut smoothed = SmoothedTrajectory::default();
        smoothed.record(0.0, &translation(1.5), 10, &translation(1.0));
        smoothed.record(0.1, &translation(2.5), 20, &translation(2.0));
        smoothed.record(0.2, &translation(3.5), 30, &translation(3.0));
        let anchored: Vec<_> = smoothed
            .anchored_frames(|kf| kf == 10)
            .map(|(t, kf, pose)| (t, kf, pose.translation.x))
            .collect();
        assert_eq!(
            anchored,
            vec![(0.0, 10, 0.5), (0.1, 10, 1.5), (0.2, 10, 2.5)]
        );
        assert_eq!(smoothed.anchored_frames(|kf| kf == 99).count(), 0);
    }

    /// A coast is recorded against the keyframe it coasts from, and the
    /// session dump sees it like a tracked frame.
    #[test]
    fn a_recorded_coast_is_an_anchored_frame() {
        let mut smoothed = SmoothedTrajectory::default();
        smoothed.record(0.0, &translation(1.5), 10, &translation(1.0));
        smoothed.record(0.1, &translation(1.9), 10, &translation(1.0));
        let anchored: Vec<_> = smoothed
            .anchored_frames(|kf| kf == 10)
            .map(|(t, kf, _)| (t, kf))
            .collect();
        assert_eq!(anchored, vec![(0.0, 10), (0.1, 10)]);
    }

    /// R-C1: a diverged pose is skipped, and the caller is told.
    #[test]
    fn a_non_finite_pose_is_never_written() {
        let dir = std::env::temp_dir().join(format!("nonfinite-{}", std::process::id()));
        let path = dir.join("t.tum");
        let mut writer = TrajectoryWriter::create(&path, None).unwrap();
        assert!(writer.write(0.0, &translation(1.0)).unwrap());
        assert!(!writer.write(0.1, &translation(f64::NAN)).unwrap());
        assert!(!writer.write(0.2, &translation(f64::INFINITY)).unwrap());
        let rotation =
            Mat3F64::from_cols_array(&[f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let bad_rotation = Pose3d::new(rotation, Vec3F64::new(0.0, 0.0, 0.0));
        assert!(!is_finite_pose(&bad_rotation));
        assert!(!writer.write(0.3, &bad_rotation).unwrap());
        writer.finish().unwrap();
        assert_eq!(written(&path), vec![-1.0]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// R-F6: a reference sequence 10, 20, 10 gives 20 the parent 10 and 10
    /// the parent 20; a walk that finds neither ends instead of spinning.
    #[test]
    fn walk_parents_ends_on_a_cycle() {
        let mut smoothed = SmoothedTrajectory::default();
        smoothed.record(0.0, &translation(1.5), 10, &translation(1.0));
        smoothed.record(0.1, &translation(2.5), 20, &translation(2.0));
        smoothed.record(0.2, &translation(1.5), 10, &translation(1.0));
        assert_eq!(smoothed.parents.get(&20).map(|p| p.0), Some(10));
        assert_eq!(smoothed.parents.get(&10).map(|p| p.0), Some(20));
        assert!(smoothed.walk_parents(10, |_| None::<()>).is_none());
        assert_eq!(smoothed.anchored_frames(|_| false).count(), 0);
        assert_eq!(
            smoothed
                .walk_parents(20, |kf| (kf == 10).then_some(kf))
                .map(|f| f.0),
            Some(10)
        );
    }

    /// R-F1: a scored frame without a reference keyframe is counted, so the
    /// session's frames plus the unrecorded ones equal the poses written.
    #[test]
    fn unrecorded_scored_frames_are_counted() {
        let mut smoothed = SmoothedTrajectory::default();
        smoothed.record_scored(0.0, &translation(1.5), Some((10, translation(1.0))));
        smoothed.record_scored(0.1, &translation(1.6), None);
        smoothed.record_scored(0.2, &translation(1.7), Some((10, translation(1.0))));
        assert_eq!(smoothed.frame_count(), 2);
        assert_eq!(smoothed.unrecorded(), 1);
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
