//! Ongoing inertial estimation state and measurement-window management.
//!
//! Calibration lives in [`SensorRig`](crate::sensor_rig::SensorRig); this is
//! the estimated part — bias, gravity, the buffered samples and the
//! initializer. The system coordinates writeback to the map and the tracker.

use crate::initialization::{ImuInitConfig, ImuInitResult, ImuInitializer};
use crate::mapping::Map;
use crate::tracking::SystemState;
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

    /// Applies an accepted initialization to the map and the tracking state,
    /// taking the new bias and gravity for itself, and reports what changed.
    ///
    /// The caller still owns what happens next — the mode transition and the
    /// mapping-worker handoff are its decisions, not this one's.
    pub(super) fn apply_initialization(
        &mut self,
        map: &mut Map,
        state: &mut SystemState,
        init: ImuInitResult,
        start_kf_idx: usize,
    ) -> AppliedInitialization {
        let applied = AppliedInitialization {
            scale: init.scale,
            gravity_world: init.gravity_world,
            gyro_bias: init.bias.gyro,
        };
        self.initializer.apply_initialization(
            map,
            state,
            &mut self.bias,
            &mut self.gravity_world,
            init,
            start_kf_idx,
        );
        applied
    }
}

/// Re-attempt interval for inertial initialization, in seconds of new data.
///
/// Without a throttle, once `ready()` is true a rejected attempt keeps the mode
/// at `ImuInit` and never resets the start index, so the same (growing) window
/// is re-solved from scratch on every subsequent keyframe forever — an
/// ever-more-expensive no-op once a call starts failing. Mirrors the VIBA1 5 s
/// cadence.
const RETRY_INTERVAL_SEC: f64 = 5.0;

/// Whether enough new data has arrived to justify another attempt.
pub(super) fn due_for_retry(last_attempt_sec: Option<f64>, timestamp_sec: f64) -> bool {
    last_attempt_sec.is_none_or(|last| timestamp_sec - last >= RETRY_INTERVAL_SEC)
}

/// Accelerometer-bias prior for the first attempt.
///
/// VIBA0 is ORB-SLAM3's first `InitializeIMU` call (LocalMapping.cc:183-186):
/// heavily regularized, with mono suppressing accel bias almost entirely
/// because a short and early window cannot yet observe it.
pub(super) fn viba0_accel_bias_prior(is_mono: bool) -> f64 {
    if is_mono { 1e10 } else { 1e5 }
}

/// What an accepted initialization changed, for the caller to report and to
/// hand on to the mapping worker.
pub(super) struct AppliedInitialization {
    pub(super) scale: f64,
    pub(super) gravity_world: Vec3F64,
    pub(super) gyro_bias: Vec3F64,
}

#[cfg(test)]
mod tests {
    use super::{InertialState, due_for_retry, viba0_accel_bias_prior};
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

    /// The first attempt is never throttled; later ones wait for 5 s of new
    /// data. Without this, a rejected attempt re-solves an ever-growing window
    /// on every keyframe forever.
    #[test]
    fn retries_are_throttled_to_five_seconds_of_new_data() {
        assert!(due_for_retry(None, 0.0), "the first attempt is always due");
        assert!(due_for_retry(None, 1234.5));

        assert!(!due_for_retry(Some(10.0), 10.0));
        assert!(!due_for_retry(Some(10.0), 14.999));
        assert!(due_for_retry(Some(10.0), 15.0), "the boundary is inclusive");
        assert!(due_for_retry(Some(10.0), 20.0));
    }

    /// Mono suppresses the accelerometer bias almost entirely at VIBA0; stereo
    /// can observe it and regularizes far less.
    #[test]
    fn mono_suppresses_the_accel_bias_prior_far_harder_than_stereo() {
        let mono = viba0_accel_bias_prior(true);
        let stereo = viba0_accel_bias_prior(false);

        assert_eq!(mono, 1e10);
        assert_eq!(stereo, 1e5);
        assert!(mono > stereo);
    }
}
