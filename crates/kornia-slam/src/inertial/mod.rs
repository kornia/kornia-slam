//! Inertial estimation: the IMU sample buffer and its preintegration windows,
//! the live bias and gravity estimates, and their initialization.
//!
//! Calibration lives in [`SensorRig`](kornia_sensors::SensorRig); this is
//! the estimated part. The system coordinates writeback to the tracker.

pub mod initialization;
mod schedule;

pub use initialization::{AlignedTrackingState, ImuInitConfig, ImuInitResult, ImuInitializer};
pub(crate) use schedule::{InertialInitSchedule, viba0_accel_bias_prior};

use crate::mapping::Map;
use crate::mapping::map::{ImuFactor, InertialAlignmentError};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

pub(crate) struct InertialState {
    pub(crate) bias: ImuBias,
    pub(crate) gravity_world: Vec3F64,
    /// Timestamp of the frame the map was bootstrapped from.
    pub(crate) bootstrap_timestamp_sec: Option<f64>,
    pub(crate) last_keyframe_timestamp_sec: Option<f64>,
    pub(crate) initializer: ImuInitializer,
    pub(crate) schedule: InertialInitSchedule,
    pending_samples: Vec<ImuMeasurement>,
}

impl InertialState {
    pub(crate) fn new() -> Self {
        Self {
            bias: ImuBias::default(),
            gravity_world: Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE),
            bootstrap_timestamp_sec: None,
            last_keyframe_timestamp_sec: None,
            // Matches ORB-SLAM3's LocalMapping::InitializeIMU VIBA0 gate
            // (nMinKF=10; minTime=1.0s stereo/2.0s mono — `ready()` doubles
            // this for mono).
            initializer: ImuInitializer::new(ImuInitConfig {
                min_keyframes: 10,
                min_time_sec: 1.0,
                min_motion: 0.05,
            }),
            schedule: InertialInitSchedule::default(),
            pending_samples: Vec::new(),
        }
    }

    pub(crate) fn buffer_samples(&mut self, samples: Vec<ImuMeasurement>) {
        self.pending_samples.extend(samples);
    }

    #[cfg(test)]
    pub(crate) fn pending_sample_count(&self) -> usize {
        self.pending_samples.len()
    }

    /// Integrates the inclusive window `[t0, t1]` without consuming the
    /// measurements: frame prediction and keyframe edges need overlapping
    /// windows. The returned sample copy is what the published `ImuFactor` keeps
    /// for later repropagation, so once this returns [`Self::prune_before`] is
    /// free to drop them from the buffer.
    pub(crate) fn preintegrate_window(
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

    /// The IMU edge between two keyframes, carrying its raw samples for later
    /// repropagation, or `None` when the window holds no time.
    pub(crate) fn keyframe_edge(
        &self,
        noise: ImuCalib,
        prev_kf_idx: usize,
        curr_kf_idx: usize,
        t0: f64,
        t1: f64,
    ) -> Option<ImuFactor> {
        let (preintegrated, raw_samples) = self.preintegrate_window(noise, t0, t1);
        (preintegrated.dt > 0.0).then_some(ImuFactor {
            prev_kf_idx,
            curr_kf_idx,
            preintegrated,
            raw_samples,
            t0,
            t1,
        })
    }

    /// Drops buffered samples strictly older than `timestamp` (typically the
    /// last keyframe: the next edge and all per-frame windows start there).
    pub(crate) fn prune_before(&mut self, timestamp: f64) {
        self.pending_samples.retain(|m| m.timestamp >= timestamp);
    }

    /// Applies an initialization to the map, taking the new bias and gravity
    /// for itself, and reports what changed and where tracking resumes.
    ///
    /// A refused alignment leaves the map and this state untouched. The caller
    /// still owns adopting the aligned tracking state, the mode transition and
    /// the mapping-worker handoff.
    pub(crate) fn apply_initialization(
        &mut self,
        map: &mut Map,
        init: ImuInitResult,
        start_kf_idx: usize,
    ) -> Result<AppliedInitialization, InertialAlignmentError> {
        let (scale, gravity_world, gyro_bias) = (init.scale, init.gravity_world, init.bias.gyro);
        let aligned = self.initializer.apply_initialization(
            map,
            &mut self.bias,
            &mut self.gravity_world,
            init,
            start_kf_idx,
        )?;
        Ok(AppliedInitialization {
            scale,
            gravity_world,
            gyro_bias,
            aligned,
        })
    }
}

/// What an accepted initialization changed, for the caller to report and to
/// hand on to the tracker and the mapping worker.
pub(crate) struct AppliedInitialization {
    pub(crate) scale: f64,
    pub(crate) gravity_world: Vec3F64,
    pub(crate) gyro_bias: Vec3F64,
    /// Where tracking resumes; `None` if the window has no keyframe.
    pub(crate) aligned: Option<AlignedTrackingState>,
}

#[cfg(test)]
mod tests {
    use super::{ImuInitResult, InertialAlignmentError, InertialState, Map};
    use kornia_algebra::Vec3F64;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement};

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

    #[test]
    fn a_keyframe_edge_needs_a_window_with_time_in_it() {
        let mut state = InertialState::new();
        state.buffer_samples(vec![sample(0.0), sample(0.5), sample(1.0)]);

        assert!(state.keyframe_edge(noise(), 1, 2, 1.0, 1.0).is_none());

        let edge = state.keyframe_edge(noise(), 1, 2, 0.0, 1.0).unwrap();
        assert_eq!((edge.prev_kf_idx, edge.curr_kf_idx), (1, 2));
        assert_eq!((edge.t0, edge.t1), (0.0, 1.0));
        assert_eq!(edge.raw_samples.len(), 3);
    }

    /// A refused alignment must not be reported as applied, nor change the
    /// bias and gravity estimates.
    #[test]
    fn a_refused_alignment_is_an_error_and_adopts_nothing() {
        let mut inertial = InertialState::new();
        let default_gravity = inertial.gravity_world;
        let init = ImuInitResult {
            scale: 0.0,
            gravity_world: Vec3F64::new(0.0, 0.0, -9.81),
            velocities_world: Vec::new(),
            bias: ImuBias {
                gyro: Vec3F64::new(0.01, 0.02, 0.03),
                accel: Vec3F64::new(0.1, 0.2, 0.3),
            },
        };

        let result = inertial.apply_initialization(&mut Map::new(), init, 0);

        assert!(matches!(
            result,
            Err(InertialAlignmentError::InvalidScale(scale)) if scale == 0.0
        ));
        assert_eq!(inertial.bias.gyro, Vec3F64::ZERO);
        assert_eq!(inertial.bias.accel, Vec3F64::ZERO);
        assert_eq!(inertial.gravity_world, default_gravity);
    }
}
