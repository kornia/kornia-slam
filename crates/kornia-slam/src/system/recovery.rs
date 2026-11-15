//! Recovery after a give-up: what of the inertial state survives it, where the
//! next segment of the map starts, and how the IMU bridges the gap.

use std::collections::VecDeque;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::ImuCalib;

use crate::inertial::ImuWindowStats;
use crate::inertial::initialization::rotation_from_to;
use crate::pose_conversion::rigid;

/// What survives a give-up of a metric, inertially initialized map.
///
/// Gravity, the biases and the metric scale belong to the device and the
/// world frame, not to the landmarks that were lost, so the next segment can
/// start in the same world without a new inertial initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct RecoveryConfig {
    /// Keep the inertial state across a give-up when the map is metric and
    /// inertially initialized; otherwise the next map starts from scratch.
    pub keep_inertial: bool,
    /// Fastest average speed, m/s, a coast may imply and still be trusted; a
    /// walker does not hold 4 m/s (visloc-rs `runaway`).
    pub max_coast_speed_mps: f64,
    /// Seconds of trajectory the runaway speed is averaged over.
    pub runaway_window_sec: f64,
    /// Factor on the covariance of the IMU edge that bridges a give-up: the
    /// coast it spans had no visual support.
    pub bridge_covariance_scale: f64,
    /// Longest gap, seconds, a bridge edge spans.
    pub max_bridge_sec: f64,
    /// Fewest stereo points that start a segment while the inertial state is
    /// kept; a cold start needs 50. The LaMAria rig overlap gives a median
    /// 1-5 on lost frames.
    pub min_seed_points: usize,
    /// Largest angle, degrees, between the mean measured specific force and
    /// world up at a give-up before the seed is levelled: about 3 deg is
    /// walking noise, and 10 deg leaks 1.7 m/s^2 into the coast.
    pub max_tilt_deg: f64,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            keep_inertial: false,
            max_coast_speed_mps: 4.0,
            runaway_window_sec: 10.0,
            bridge_covariance_scale: 100.0,
            max_bridge_sec: 10.0,
            min_seed_points: 50,
            max_tilt_deg: 10.0,
        }
    }
}

/// The keyframe a bridge edge starts from: the last one before a give-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BridgeAnchor {
    pub kf_idx: usize,
    pub timestamp_sec: f64,
}

/// Whether moving the camera centre from `before` to `after` in `gap_sec`
/// stays under `max_speed_mps` on average. A non-finite pose never does.
pub(crate) fn coast_is_plausible(
    before: &Pose3d,
    after: &Pose3d,
    gap_sec: f64,
    max_speed_mps: f64,
) -> bool {
    let travelled = (after.inverse().translation - before.inverse().translation).length();
    if !travelled.is_finite() {
        return false;
    }
    gap_sec <= 0.0 || travelled / gap_sec <= max_speed_mps
}

/// `noise` with every density scaled so the preintegrated covariance, and the
/// bias walk, grow by `covariance_scale`. Kept in the edge's calibration, the
/// inflation survives re-preintegration.
pub(crate) fn inflated_noise(noise: ImuCalib, covariance_scale: f64) -> ImuCalib {
    let k = covariance_scale.max(1.0).sqrt();
    ImuCalib {
        gyro_noise: noise.gyro_noise * k,
        accel_noise: noise.accel_noise * k,
        gyro_bias_noise: noise.gyro_bias_noise * k,
        accel_bias_noise: noise.accel_bias_noise * k,
    }
}

/// Shortest stretch of trajectory a runaway speed is averaged over: over less,
/// tracking jitter alone can look fast.
const MIN_RUNAWAY_SPAN_SEC: f64 = 1.0;

/// Average speed of the camera over the last `window_sec` of trajectory.
///
/// The visloc-rs `runaway` check (a walking track averaging more than 4 m/s
/// over 10 s has diverged), here measured online: coasted poses beyond it come
/// from a diverged inertial state, and a keyframe placed there poisons the map
/// (kornia-slam#182 I4: keyframes at 1e57 m).
#[derive(Debug, Clone)]
pub(crate) struct RunawayDetector {
    window_sec: f64,
    max_speed_mps: f64,
    track: VecDeque<(f64, Vec3F64)>,
}

impl RunawayDetector {
    pub(crate) fn new(window_sec: f64, max_speed_mps: f64) -> Self {
        Self {
            window_sec,
            max_speed_mps,
            track: VecDeque::new(),
        }
    }

    /// Adds the camera centre at `timestamp_sec`, dropping what left the window.
    pub(crate) fn record(&mut self, timestamp_sec: f64, centre: Vec3F64) {
        self.track.push_back((timestamp_sec, centre));
        while self
            .track
            .front()
            .is_some_and(|&(t, _)| t < timestamp_sec - self.window_sec)
        {
            self.track.pop_front();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.track.clear();
    }

    /// Average speed over the window, m/s, over at least
    /// [`MIN_RUNAWAY_SPAN_SEC`]; infinite for a non-finite centre, `None` before
    /// two centres are recorded.
    pub(crate) fn speed_mps(&self) -> Option<f64> {
        if self.track.len() < 2 {
            return None;
        }
        let (&(t0, first), &(t1, last)) = (self.track.front()?, self.track.back()?);
        let travelled = (last - first).length();
        if !travelled.is_finite() {
            return Some(f64::INFINITY);
        }
        Some(travelled / (t1 - t0).max(MIN_RUNAWAY_SPAN_SEC))
    }

    pub(crate) fn runaway(&self) -> bool {
        self.speed_mps()
            .is_some_and(|speed| speed > self.max_speed_mps)
    }
}

/// Camera centre in the world frame, for speed checks.
pub(crate) fn camera_centre(pose_world_to_cam: &Pose3d) -> Vec3F64 {
    pose_world_to_cam.inverse().translation
}

/// The attitude of `attitude` at the camera centre of `position`: a held
/// re-seed keeps the gyro's attitude but not the coast's translation.
pub(crate) fn held_at(attitude: &Pose3d, position: &Pose3d) -> Pose3d {
    let rotation = attitude.rotation;
    rigid(Pose3d::new(rotation, -(rotation * camera_centre(position))))
}

/// Seconds of specific force the tilt check averages: one sample is a
/// footstep, 2 s of walking averages it out (VINS `initFirstIMUPose` averages
/// its window too).
const TILT_WINDOW_SEC: f64 = 2.0;

/// The mean bias-corrected specific force over the last
/// [`TILT_WINDOW_SEC`], rotated to the world frame by the attitude of each
/// frame it was measured in. At rest it points up, against gravity; an
/// attitude error shows as the angle between them.
#[derive(Debug, Clone, Default)]
pub(crate) struct TiltMonitor {
    /// Per frame: stamp, world specific force summed over its samples, and
    /// how many samples.
    frames: VecDeque<(f64, Vec3F64, usize)>,
}

impl TiltMonitor {
    /// Adds one frame's IMU window, measured in the body at `body_to_world`.
    pub(crate) fn record(
        &mut self,
        timestamp_sec: f64,
        stats: &ImuWindowStats,
        accel_bias: Vec3F64,
        body_to_world: &Mat3F64,
    ) {
        let force_world = *body_to_world * (stats.mean_accel - accel_bias);
        self.frames.push_back((
            timestamp_sec,
            force_world * stats.samples as f64,
            stats.samples,
        ));
        while self
            .frames
            .front()
            .is_some_and(|&(t, _, _)| t < timestamp_sec - TILT_WINDOW_SEC)
        {
            self.frames.pop_front();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.frames.clear();
    }

    /// The measured up direction over the window, or `None` when empty.
    pub(crate) fn measured_up(&self) -> Option<Vec3F64> {
        let (sum, n) = self
            .frames
            .iter()
            .fold((Vec3F64::ZERO, 0usize), |(sum, n), &(_, f, k)| {
                (sum + f, n + k)
            });
        (n > 0 && sum.length() > 0.0).then(|| sum.normalize())
    }
}

/// Angle, degrees, between `measured_up` and world up (against
/// `gravity_world`).
pub(crate) fn tilt_deg(measured_up: Vec3F64, gravity_world: Vec3F64) -> f64 {
    let up = -gravity_world.normalize();
    measured_up
        .normalize()
        .dot(up)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// `pose_world_to_cam` turned so that `measured_up` becomes world up, about
/// its own camera centre; the smallest such turn, so the heading stays. A
/// re-alignment of the whole world about the origin instead moves a far
/// keyframe by metres.
pub(crate) fn leveled(
    pose_world_to_cam: &Pose3d,
    measured_up: Vec3F64,
    gravity_world: Vec3F64,
) -> Pose3d {
    let correction = rotation_from_to(measured_up, -gravity_world).matrix();
    let cam_to_world = pose_world_to_cam.inverse();
    let rotation = (correction * cam_to_world.rotation).transpose();
    rigid(Pose3d::new(
        rotation,
        -(rotation * cam_to_world.translation),
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use kornia_sensors::imu::{ImuBias, ImuMeasurement, PreintegratedImu};

    /// World-to-camera pose of a camera centred at `(x, 0, 0)`.
    fn centred_at(x: f64) -> Pose3d {
        let mut pose = Pose3d::IDENTITY;
        pose.translation = Vec3F64::new(-x, 0.0, 0.0);
        pose
    }

    #[test]
    fn a_walking_coast_is_plausible_and_a_runaway_is_not() {
        let before = centred_at(0.0);
        assert!(coast_is_plausible(&before, &centred_at(7.0), 5.0, 4.0));
        assert!(coast_is_plausible(&before, &centred_at(20.0), 5.0, 4.0));
        assert!(!coast_is_plausible(&before, &centred_at(20.5), 5.0, 4.0));
        assert!(!coast_is_plausible(&before, &centred_at(-8.1e57), 5.0, 4.0));
        assert!(!coast_is_plausible(
            &before,
            &centred_at(f64::NAN),
            5.0,
            4.0
        ));
    }

    /// `scaled == scale * base`, to rounding relative to the largest entry.
    pub(crate) fn assert_scaled(base: &[f64], scaled: &[f64], scale: f64) {
        let largest = base.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (a, b) in base.iter().zip(scaled) {
            assert!((b - scale * a).abs() <= 1e-9 * scale * largest, "{a} {b}");
        }
    }

    fn walk(detector: &mut RunawayDetector, from_sec: f64, to_sec: f64, speed: f64) {
        let mut t = from_sec;
        while t <= to_sec + 1e-9 {
            detector.record(t, Vec3F64::new(speed * t, 0.0, 0.0));
            t += 0.05;
        }
    }

    #[test]
    fn a_walk_is_no_runaway_and_a_fast_coast_over_the_window_is() {
        let mut detector = RunawayDetector::new(10.0, 4.0);
        assert_eq!(detector.speed_mps(), None, "nothing recorded");
        walk(&mut detector, 0.0, 30.0, 1.4);
        let speed = detector.speed_mps().unwrap();
        assert!((speed - 1.4).abs() < 1e-9 && !detector.runaway(), "{speed}");

        // A coast that runs off at 9 m/s for 5 s averages 5.2 m/s over the
        // window: (5 * 9 + 5 * 1.4) / 10.
        for i in 1..=100 {
            let t = 30.0 + i as f64 * 0.05;
            detector.record(t, Vec3F64::new(42.0 + 9.0 * (t - 30.0), 0.0, 0.0));
        }
        assert!(detector.runaway(), "{:?}", detector.speed_mps());
    }

    #[test]
    fn a_short_span_is_averaged_over_a_second_and_a_divergence_always_runs_away() {
        let mut detector = RunawayDetector::new(10.0, 4.0);
        walk(&mut detector, 0.0, 0.5, 6.0);
        assert!(!detector.runaway(), "3 m of jitter in half a second");
        detector.record(0.55, Vec3F64::new(-8.1e57, 0.0, 0.0));
        assert!(detector.runaway());
        detector.record(0.6, Vec3F64::new(f64::NAN, 0.0, 0.0));
        assert!(detector.runaway());
        detector.clear();
        assert!(!detector.runaway() && detector.speed_mps().is_none());
    }

    #[test]
    fn the_window_forgets_what_is_older_than_it() {
        let mut detector = RunawayDetector::new(10.0, 4.0);
        detector.record(0.0, Vec3F64::new(-1e6, 0.0, 0.0));
        walk(&mut detector, 1.0, 20.0, 1.0);
        assert!(!detector.runaway(), "the jump at t=0 left the window");
    }

    /// World gravity as the inertial initialization leaves it: along +Y.
    fn gravity() -> Vec3F64 {
        Vec3F64::new(0.0, 9.81, 0.0)
    }

    /// A second of samples reading the specific force `accel` in the body.
    fn still_window(accel: Vec3F64) -> ImuWindowStats {
        let samples: Vec<ImuMeasurement> = (0..=200)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * 0.005,
                gyro: Vec3F64::ZERO,
                accel,
            })
            .collect();
        ImuWindowStats::from_samples(&samples).unwrap()
    }

    fn rotation(axis_angle: Vec3F64) -> Mat3F64 {
        kornia_algebra::SO3F64::exp(axis_angle).matrix()
    }

    #[test]
    fn a_level_still_window_reads_no_tilt_and_a_tilted_attitude_reads_its_angle() {
        // At rest the body measures the negative of gravity.
        let window = still_window(-gravity());
        let mut level = TiltMonitor::default();
        level.record(1.0, &window, Vec3F64::ZERO, &Mat3F64::IDENTITY);
        let tilt = tilt_deg(level.measured_up().unwrap(), gravity());
        assert!(tilt < 1e-6, "{tilt}");

        let mut tilted = TiltMonitor::default();
        let attitude = rotation(Vec3F64::new(0.0, 0.0, 15f64.to_radians()));
        tilted.record(1.0, &window, Vec3F64::ZERO, &attitude);
        let tilt = tilt_deg(tilted.measured_up().unwrap(), gravity());
        assert!((tilt - 15.0).abs() < 0.5, "{tilt}");
    }

    #[test]
    fn the_tilt_check_subtracts_the_accel_bias_and_forgets_old_frames() {
        let bias = Vec3F64::new(0.3, 0.0, 0.0);
        let mut monitor = TiltMonitor::default();
        monitor.record(
            1.0,
            &still_window(-gravity() + bias),
            bias,
            &Mat3F64::IDENTITY,
        );
        assert!(tilt_deg(monitor.measured_up().unwrap(), gravity()) < 1e-6);

        let attitude = rotation(Vec3F64::new(30f64.to_radians(), 0.0, 0.0));
        monitor.record(10.0, &still_window(-gravity()), Vec3F64::ZERO, &attitude);
        let tilt = tilt_deg(monitor.measured_up().unwrap(), gravity());
        assert!(
            (tilt - 30.0).abs() < 1e-6,
            "the frame at 1 s left the window: {tilt}"
        );
        monitor.clear();
        assert!(monitor.measured_up().is_none());
    }

    /// Heading of a camera-to-world rotation: the azimuth of its optical
    /// axis in the plane normal to gravity (+Y), degrees.
    fn heading_deg(cam_to_world: &Mat3F64) -> f64 {
        let forward = *cam_to_world * Vec3F64::new(0.0, 0.0, 1.0);
        forward.x.atan2(forward.z).to_degrees()
    }

    #[test]
    fn leveled_removes_the_tilt_about_the_camera_centre_and_keeps_the_heading() {
        let truth_cam_to_world = rotation(Vec3F64::new(0.0, 40f64.to_radians(), 0.0));
        // 15 deg of attitude error about a horizontal axis between x and z.
        let axis = Vec3F64::new(1.0, 0.0, 1.0).normalize();
        let error = rotation(axis * 15f64.to_radians());
        let estimate_cam_to_world = error * truth_cam_to_world;
        let centre = Vec3F64::new(120.0, -3.0, 85.0);
        let pose = Pose3d::new(
            estimate_cam_to_world.transpose(),
            -(estimate_cam_to_world.transpose() * centre),
        );
        // What the tilt check measures through the erroneous attitude.
        let measured_up = error * (-gravity()).normalize();
        assert!((tilt_deg(measured_up, gravity()) - 15.0).abs() < 1e-6);

        let level = leveled(&pose, measured_up, gravity());

        let level_cam_to_world = level.inverse().rotation;
        let up_after = level_cam_to_world * (truth_cam_to_world.transpose() * (-gravity()));
        let tilt = tilt_deg(up_after, gravity());
        assert!(tilt < 0.5, "{tilt}");
        assert!((camera_centre(&level) - centre).length() < 1e-9);
        let heading = heading_deg(&level_cam_to_world) - heading_deg(&truth_cam_to_world);
        assert!(heading.abs() < 1.0, "{heading}");
    }

    #[test]
    fn a_held_pose_takes_one_attitude_and_the_other_position() {
        let attitude = Pose3d::new(
            rotation(Vec3F64::new(0.1, 0.4, -0.2)),
            Vec3F64::new(5.0, 6.0, 7.0),
        );
        let position = centred_at(3.0);
        let held = held_at(&attitude, &position);
        assert_eq!(held.rotation, rigid(attitude).rotation);
        assert!((camera_centre(&held) - Vec3F64::new(3.0, 0.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn inflated_noise_scales_the_preintegrated_covariance() {
        let noise = ImuCalib {
            gyro_noise: 1e-3,
            accel_noise: 1e-2,
            gyro_bias_noise: 1e-5,
            accel_bias_noise: 1e-4,
        };
        let samples: Vec<ImuMeasurement> = (0..=100)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * 0.01,
                gyro: Vec3F64::new(0.1, -0.2, 0.05),
                accel: Vec3F64::new(0.3, 0.1, 9.81),
            })
            .collect();
        let integrate = |noise| {
            PreintegratedImu::from_measurements(ImuBias::default(), noise, &samples, 0.0, 1.0)
        };
        let base = integrate(noise);
        let inflated = integrate(inflated_noise(noise, 100.0));
        assert_scaled(&base.covariance, &inflated.covariance, 100.0);
        assert_scaled(&base.bias_covariance, &inflated.bias_covariance, 100.0);
        assert_eq!(inflated.delta_position, base.delta_position);
    }
}
