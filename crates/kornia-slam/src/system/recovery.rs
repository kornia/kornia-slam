//! Recovery after a give-up: what of the inertial state survives it, where the
//! next segment of the map starts, and how the IMU bridges the gap.

use std::collections::VecDeque;

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuCalib;

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
