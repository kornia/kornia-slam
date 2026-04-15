//! IMU state and pre-integration primitives for visual-inertial odometry.

use nalgebra::{UnitQuaternion, Vector3};

/// A single IMU sample.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuMeasurement {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Linear acceleration in the IMU/body frame, in m/s^2.
    pub accel: Vector3<f64>,
    /// Angular velocity in the IMU/body frame, in rad/s.
    pub gyro: Vector3<f64>,
}

impl ImuMeasurement {
    /// Creates a new IMU sample.
    pub fn new(timestamp: f64, accel: Vector3<f64>, gyro: Vector3<f64>) -> Self {
        Self {
            timestamp,
            accel,
            gyro,
        }
    }
}

/// Full IMU state used by the VIO prediction step.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuState {
    /// Position in world coordinates.
    pub position: Vector3<f64>,
    /// Velocity in world coordinates.
    pub velocity: Vector3<f64>,
    /// Rotation from body frame into world frame.
    pub orientation: UnitQuaternion<f64>,
    /// Additive accelerometer bias in the body frame.
    pub bias_accel: Vector3<f64>,
    /// Additive gyroscope bias in the body frame.
    pub bias_gyro: Vector3<f64>,
}

impl ImuState {
    /// Creates a state initialized at the origin with zero velocity and bias.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for ImuState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            bias_accel: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
        }
    }
}

/// Errors produced while integrating IMU data.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ImuPreintegrationError {
    /// Timestamps must be strictly increasing.
    #[error("imu timestamps must be strictly increasing, got {previous} then {current}")]
    NonIncreasingTimestamp { previous: f64, current: f64 },
}

/// Pre-integrated IMU delta between two camera frames.
///
/// This stores first-order deltas in the starting body frame. It is intentionally
/// limited to the minimal state needed to bootstrap VIO prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPreintegrator {
    /// Pre-integrated position delta in the starting body frame.
    pub delta_position: Vector3<f64>,
    /// Pre-integrated velocity delta in the starting body frame.
    pub delta_velocity: Vector3<f64>,
    /// Pre-integrated orientation delta from the starting body frame.
    pub delta_orientation: UnitQuaternion<f64>,
    /// Total integrated duration in seconds.
    pub total_dt: f64,
    /// Number of integrated sample intervals.
    pub measurement_count: usize,
}

impl ImuPreintegrator {
    /// Creates an empty pre-integrator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all accumulated deltas.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Integrates one measurement interval using midpoint bias-corrected values.
    pub fn integrate_pair(
        &mut self,
        previous: &ImuMeasurement,
        current: &ImuMeasurement,
        bias_accel: &Vector3<f64>,
        bias_gyro: &Vector3<f64>,
    ) -> Result<(), ImuPreintegrationError> {
        let dt = current.timestamp - previous.timestamp;
        if dt <= 0.0 {
            return Err(ImuPreintegrationError::NonIncreasingTimestamp {
                previous: previous.timestamp,
                current: current.timestamp,
            });
        }

        let accel_avg = (previous.accel + current.accel) * 0.5 - bias_accel;
        let gyro_avg = (previous.gyro + current.gyro) * 0.5 - bias_gyro;

        let orientation_before = self.delta_orientation;
        let accel_in_start = orientation_before.transform_vector(&accel_avg);

        self.delta_position += self.delta_velocity * dt + accel_in_start * (0.5 * dt * dt);
        self.delta_velocity += accel_in_start * dt;
        self.delta_orientation *= UnitQuaternion::from_scaled_axis(gyro_avg * dt);
        self.total_dt += dt;
        self.measurement_count += 1;

        Ok(())
    }

    /// Integrates a timestamp-ordered slice of measurements using the state's biases.
    pub fn integrate_measurements(
        &mut self,
        measurements: &[ImuMeasurement],
        state: &ImuState,
    ) -> Result<(), ImuPreintegrationError> {
        for pair in measurements.windows(2) {
            self.integrate_pair(&pair[0], &pair[1], &state.bias_accel, &state.bias_gyro)?;
        }
        Ok(())
    }

    /// Applies the pre-integrated delta to an IMU state with a gravity term.
    pub fn predict(&self, state: &ImuState, gravity: Vector3<f64>) -> ImuState {
        let dt = self.total_dt;
        let delta_position_world = state.orientation.transform_vector(&self.delta_position);
        let delta_velocity_world = state.orientation.transform_vector(&self.delta_velocity);

        ImuState {
            position: state.position
                + state.velocity * dt
                + gravity * (0.5 * dt * dt)
                + delta_position_world,
            velocity: state.velocity + gravity * dt + delta_velocity_world,
            orientation: state.orientation * self.delta_orientation,
            bias_accel: state.bias_accel,
            bias_gyro: state.bias_gyro,
        }
    }
}

impl Default for ImuPreintegrator {
    fn default() -> Self {
        Self {
            delta_position: Vector3::zeros(),
            delta_velocity: Vector3::zeros(),
            delta_orientation: UnitQuaternion::identity(),
            total_dt: 0.0,
            measurement_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_close(actual: &Vector3<f64>, expected: &Vector3<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() <= tol,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn zero_motion_produces_zero_delta() {
        let state = ImuState::default();
        let samples = vec![
            ImuMeasurement::new(0.0, Vector3::zeros(), Vector3::zeros()),
            ImuMeasurement::new(0.01, Vector3::zeros(), Vector3::zeros()),
            ImuMeasurement::new(0.02, Vector3::zeros(), Vector3::zeros()),
        ];

        let mut preintegrator = ImuPreintegrator::new();
        preintegrator
            .integrate_measurements(&samples, &state)
            .unwrap();

        assert_vec3_close(&preintegrator.delta_position, &Vector3::zeros(), 1e-12);
        assert_vec3_close(&preintegrator.delta_velocity, &Vector3::zeros(), 1e-12);
        assert_eq!(preintegrator.delta_orientation, UnitQuaternion::identity());
        assert_eq!(preintegrator.total_dt, 0.02);
        assert_eq!(preintegrator.measurement_count, 2);
    }

    #[test]
    fn bias_compensation_cancels_constant_measurement() {
        let state = ImuState {
            bias_accel: Vector3::new(0.2, -0.1, 0.05),
            bias_gyro: Vector3::new(0.01, 0.02, -0.03),
            ..ImuState::default()
        };
        let samples = vec![
            ImuMeasurement::new(0.0, state.bias_accel, state.bias_gyro),
            ImuMeasurement::new(0.1, state.bias_accel, state.bias_gyro),
        ];

        let mut preintegrator = ImuPreintegrator::new();
        preintegrator
            .integrate_measurements(&samples, &state)
            .unwrap();

        assert_vec3_close(&preintegrator.delta_position, &Vector3::zeros(), 1e-12);
        assert_vec3_close(&preintegrator.delta_velocity, &Vector3::zeros(), 1e-12);
        assert_eq!(preintegrator.delta_orientation, UnitQuaternion::identity());
    }

    #[test]
    fn constant_acceleration_predicts_position_and_velocity() {
        let state = ImuState::default();
        let mut samples = Vec::new();
        for i in 0..=100 {
            samples.push(ImuMeasurement::new(
                i as f64 * 0.01,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::zeros(),
            ));
        }

        let mut preintegrator = ImuPreintegrator::new();
        preintegrator
            .integrate_measurements(&samples, &state)
            .unwrap();

        let predicted = preintegrator.predict(&state, Vector3::zeros());
        assert_vec3_close(&predicted.velocity, &Vector3::new(1.0, 0.0, 0.0), 1e-9);
        assert_vec3_close(&predicted.position, &Vector3::new(0.5, 0.0, 0.0), 1e-9);
        assert_eq!(predicted.orientation, UnitQuaternion::identity());
    }

    #[test]
    fn reset_clears_accumulated_delta() {
        let mut preintegrator = ImuPreintegrator {
            delta_position: Vector3::new(1.0, 2.0, 3.0),
            delta_velocity: Vector3::new(4.0, 5.0, 6.0),
            delta_orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.1, 0.0, 0.0)),
            total_dt: 0.5,
            measurement_count: 10,
        };

        preintegrator.reset();

        assert_eq!(preintegrator, ImuPreintegrator::default());
    }

    #[test]
    fn rejects_non_increasing_timestamps() {
        let mut preintegrator = ImuPreintegrator::new();
        let previous = ImuMeasurement::new(1.0, Vector3::zeros(), Vector3::zeros());
        let current = ImuMeasurement::new(1.0, Vector3::zeros(), Vector3::zeros());

        let err = preintegrator
            .integrate_pair(&previous, &current, &Vector3::zeros(), &Vector3::zeros())
            .unwrap_err();

        assert_eq!(
            err,
            ImuPreintegrationError::NonIncreasingTimestamp {
                previous: 1.0,
                current: 1.0,
            }
        );
    }
}
