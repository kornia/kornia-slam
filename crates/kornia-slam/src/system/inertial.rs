//! Ongoing inertial estimation state and measurement-window management.
//!
//! Calibration lives in [`SensorRig`](crate::sensor_rig::SensorRig); this is
//! the estimated part — bias, gravity, the buffered samples and the
//! initializer. The system coordinates writeback to the map and the tracker.

use crate::initialization::{ImuInitConfig, ImuInitializer};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

pub(super) struct InertialState {
    pub(super) bias: ImuBias,
    pub(super) gravity_world: Vec3F64,
    /// Timestamp of the frame the map was bootstrapped from.
    pub(super) bootstrap_timestamp_sec: Option<f64>,
    pub(super) last_keyframe_timestamp_sec: Option<f64>,
    pub(super) initializer: ImuInitializer,
    pending_samples: Vec<ImuMeasurement>,
}

impl InertialState {
    pub(super) fn new() -> Self {
        Self {
            bias: ImuBias::default(),
            gravity_world: Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE),
            bootstrap_timestamp_sec: None,
            last_keyframe_timestamp_sec: None,
            // Matches ORB-SLAM3's LocalMapping::InitializeIMU VIBA0 gate
            // (nMinKF=10; minTime=1.0s stereo/2.0s mono — `ready()` doubles
            // this for mono). The previous min_keyframes=30/min_time_sec=15.0
            // was effectively skipping VIBA0/VIBA1 and attempting a
            // VIBA2-strength window on the very first try.
            initializer: ImuInitializer::new(ImuInitConfig {
                min_keyframes: 10,
                min_time_sec: 1.0,
                min_motion: 0.05,
            }),
            pending_samples: Vec::new(),
        }
    }

    pub(super) fn buffer_samples(&mut self, samples: Vec<ImuMeasurement>) {
        self.pending_samples.extend(samples);
    }

    /// Integrates the inclusive window `[t0, t1]` without consuming the
    /// measurements: frame prediction and keyframe edges need overlapping
    /// windows. The returned sample copy is what `Map::add_imu_factor` keeps
    /// for later repropagation, so once this returns [`Self::prune_before`] is
    /// free to drop them from the buffer.
    pub(super) fn preintegrate_window(
        &self,
        noise: ImuCalib,
        t0: f64,
        t1: f64,
    ) -> (PreintegratedImu, Vec<ImuMeasurement>) {
        let samples: Vec<ImuMeasurement> = self
            .pending_samples
            .iter()
            .filter(|m| m.timestamp >= t0 && m.timestamp <= t1)
            .copied()
            .collect();
        let pre = PreintegratedImu::from_measurements(self.bias, noise, &samples, t0, t1);
        (pre, samples)
    }

    /// Drops buffered samples strictly older than `timestamp` (typically the
    /// last keyframe: the next edge and all per-frame windows start there).
    pub(super) fn prune_before(&mut self, timestamp: f64) {
        self.pending_samples.retain(|m| m.timestamp >= timestamp);
    }
}

#[cfg(test)]
mod tests {
    use super::InertialState;
    use kornia_algebra::Vec3F64;
    use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

    fn noise() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1e-4,
            accel_noise: 1e-3,
            gyro_bias_noise: 1e-5,
            accel_bias_noise: 1e-4,
        }
    }

    fn sample(timestamp: f64) -> ImuMeasurement {
        ImuMeasurement {
            timestamp,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        }
    }

    /// Windows overlap by design, and the samples handed to a map factor must
    /// outlive pruning of the buffer they came from.
    #[test]
    fn overlapping_windows_survive_preintegration_and_keep_the_pruning_boundary() {
        let mut state = InertialState::new();
        state.buffer_samples(vec![sample(0.0), sample(0.5), sample(1.0), sample(1.5)]);

        let (_, first) = state.preintegrate_window(noise(), 0.0, 1.0);
        let (_, overlapping) = state.preintegrate_window(noise(), 0.5, 1.5);
        assert_eq!(
            first.iter().map(|m| m.timestamp).collect::<Vec<_>>(),
            vec![0.0, 0.5, 1.0]
        );
        assert_eq!(
            overlapping.iter().map(|m| m.timestamp).collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.5]
        );

        state.prune_before(1.0);
        assert_eq!(first.len(), 3, "map-factor samples outlive pruning");
        let (_, next) = state.preintegrate_window(noise(), 1.0, 1.5);
        assert_eq!(
            next.len(),
            2,
            "the boundary sample is kept for the next window"
        );
    }

    /// Preintegration must use the live bias estimate, not the value it was
    /// constructed with.
    #[test]
    fn preintegration_uses_the_live_bias() {
        let mut state = InertialState::new();
        state.bias.gyro = Vec3F64::new(0.01, 0.02, 0.03);
        state.bias.accel = Vec3F64::new(0.1, 0.2, 0.3);
        state.buffer_samples(vec![sample(0.0), sample(0.5)]);

        let (pre, _) = state.preintegrate_window(noise(), 0.0, 0.5);

        assert_eq!(pre.bias.gyro, state.bias.gyro);
        assert_eq!(pre.bias.accel, state.bias.accel);
        assert_eq!(pre.calib.gyro_noise, noise().gyro_noise);
    }
}
