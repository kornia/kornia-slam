//! Ongoing inertial estimation state and measurement-window management.

use crate::initialization::{ImuInitConfig, ImuInitializer};
use crate::sensor_rig::{ImuCalibration, default_imu_noise};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuMeasurement, PreintegratedImu};

/// Runtime state shared across initialization, tracking and keyframe processing.
/// Calibration stays in SensorRig; the system coordinates map/tracker writeback.
pub(super) struct InertialState {
    pub(super) bias: ImuBias,
    pub(super) gravity_world: Vec3F64,
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
            initializer: ImuInitializer::new(ImuInitConfig::default()),
            pending_samples: Vec::new(),
        }
    }

    pub(super) fn buffer_samples(&mut self, samples: Vec<ImuMeasurement>) {
        self.pending_samples.extend(samples);
    }

    /// Integrates the inclusive window without consuming measurements: frame
    /// prediction and keyframe edges may need overlapping windows. The returned
    /// sample copy lets map factors repropagate after the buffer is pruned.
    pub(super) fn preintegrate_window(
        &self,
        calibration: Option<&ImuCalibration>,
        t0: f64,
        t1: f64,
    ) -> (PreintegratedImu, Vec<ImuMeasurement>) {
        let samples: Vec<_> = self
            .pending_samples
            .iter()
            .filter(|m| m.timestamp >= t0 && m.timestamp <= t1)
            .copied()
            .collect();
        // Preserve existing visual-only map preintegration behavior.
        let noise = calibration.map_or_else(default_imu_noise, |imu| imu.noise);
        let pre = PreintegratedImu::from_measurements(self.bias, noise, &samples, t0, t1);
        (pre, samples)
    }

    /// Keep the boundary sample for the next integration window.
    pub(super) fn prune_before(&mut self, timestamp: f64) {
        self.pending_samples.retain(|m| m.timestamp >= timestamp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;

    fn sample(timestamp: f64) -> ImuMeasurement {
        ImuMeasurement {
            timestamp,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        }
    }

    #[test]
    fn overlapping_windows_survive_preintegration_and_keep_pruning_boundary() {
        let mut state = InertialState::new();
        state.buffer_samples(vec![sample(0.0), sample(0.5), sample(1.0), sample(1.5)]);
        let (_, first) = state.preintegrate_window(None, 0.0, 1.0);
        let (_, overlapping) = state.preintegrate_window(None, 0.5, 1.5);
        assert_eq!(
            first.iter().map(|m| m.timestamp).collect::<Vec<_>>(),
            vec![0.0, 0.5, 1.0]
        );
        assert_eq!(
            overlapping.iter().map(|m| m.timestamp).collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.5]
        );
        state.prune_before(1.0);
        assert_eq!(
            state
                .pending_samples
                .iter()
                .map(|m| m.timestamp)
                .collect::<Vec<_>>(),
            vec![1.0, 1.5]
        );
        assert_eq!(first.len(), 3); // Map-factor samples outlive pruning.
        let (_, next) = state.preintegrate_window(None, 1.0, 1.5);
        assert_eq!(next.len(), 2);
    }

    #[test]
    fn preintegration_uses_live_bias_and_rig_noise() {
        let mut state = InertialState::new();
        state.bias.gyro = Vec3F64::new(0.01, 0.02, 0.03);
        state.bias.accel = Vec3F64::new(0.1, 0.2, 0.3);
        state.buffer_samples(vec![sample(0.0), sample(0.5)]);
        let mut calibration = ImuCalibration::new(Pose3d::IDENTITY);
        calibration.noise.gyro_noise = 0.123;
        let (pre, _) = state.preintegrate_window(Some(&calibration), 0.0, 0.5);
        assert_eq!(pre.bias.gyro, state.bias.gyro);
        assert_eq!(pre.bias.accel, state.bias.accel);
        assert_eq!(pre.calib.gyro_noise, 0.123);
        let (fallback, _) = state.preintegrate_window(None, 0.0, 0.5);
        assert_eq!(fallback.calib.gyro_noise, default_imu_noise().gyro_noise);
    }
}
