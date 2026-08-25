//! Synthetic IMU measurement generation.
//!
//! The simulator emits **raw [`ImuMeasurement`] values** and feeds them through
//! the production [`PreintegratedImu`]. Preintegration is not reimplemented and
//! not mocked, so a bug in `integrate` — or in its covariance or bias-Jacobian
//! propagation — falls inside the blast radius of any test built on this,
//! rather than outside it.
//!
//! # Measurement model
//!
//! The sensor equations themselves live in
//! [`ImuMeasurement::simulate`][kornia_sensors::imu::ImuMeasurement::simulate],
//! next to the preintegration they invert, and are not restated here — one copy
//! of a sign convention is the most this codebase should have. This module
//! supplies the kinematics to feed it, the sampling schedule, and the noise.
//!
//! Noise is drawn at the **discrete** standard deviation `density / √dt`, which
//! is the discretization [`PreintegratedImu::integrate`] assumes when it forms
//! `density² / dt` as the per-sample variance.

use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

use super::SimError;
use super::rng::SimRng;
use super::trajectory::{DEFAULT_GRAVITY, Trajectory};
use crate::vi_ba_schur::ImuFactor;

use kornia_algebra::Vec3F64;

/// EuRoC's ADIS16448 noise densities, used as the simulator's default.
///
/// Starting from real datasheet values rather than invented ones means a
/// sensitivity sweep is anchored at a point that actually corresponds to
/// hardware.
pub const EUROC_IMU_CALIB: ImuCalib = ImuCalib {
    gyro_noise: 1.6968e-4,
    accel_noise: 2.0e-3,
    gyro_bias_noise: 1.9393e-5,
    accel_bias_noise: 3.0e-3,
};

/// How IMU measurements are generated.
#[derive(Debug, Clone)]
pub struct ImuSimConfig {
    /// Sample rate in Hz.
    pub rate_hz: f64,
    /// The **true** bias baked into the measurements. This is the quantity a
    /// recovery test injects and then asks the estimator to find.
    pub bias: ImuBias,
    /// Noise densities. Also what [`PreintegratedImu`] uses to build its
    /// covariance, so generation and estimation stay consistent.
    pub calib: ImuCalib,
    /// Whether to add sensor noise. `false` gives exact measurements, which is
    /// the right setting for a first recovery test — it separates "the
    /// optimizer is wrong" from "the noise was too large".
    pub add_noise: bool,
}

impl Default for ImuSimConfig {
    fn default() -> Self {
        Self {
            rate_hz: 200.0, // EuRoC's IMU rate
            bias: ImuBias::default(),
            calib: EUROC_IMU_CALIB,
            add_noise: false,
        }
    }
}

/// Generates IMU measurements over the trajectory's full time domain.
///
/// Noise is drawn at the **discrete** standard deviation `density / √dt`,
/// which is the discretization [`PreintegratedImu::integrate`] assumes when it
/// forms `density² / dt` as the per-sample variance. Getting this wrong would
/// not make the simulator obviously broken — it would just silently miscalibrate
/// every covariance-dependent assertion built on it.
pub fn generate_imu(
    trajectory: &Trajectory,
    gravity: Vec3F64,
    config: &ImuSimConfig,
    rng: &mut SimRng,
) -> Result<Vec<ImuMeasurement>, SimError> {
    if config.rate_hz <= 0.0 || !config.rate_hz.is_finite() {
        return Err(SimError::InvalidConfig(format!(
            "IMU rate must be positive and finite, got {}",
            config.rate_hz
        )));
    }

    let dt = 1.0 / config.rate_hz;
    let (gyro_std, accel_std) = if config.add_noise {
        (
            config.calib.gyro_noise / dt.sqrt(),
            config.calib.accel_noise / dt.sqrt(),
        )
    } else {
        (0.0, 0.0)
    };

    let n_samples = ((trajectory.t_end() - trajectory.t_start()) / dt).floor() as usize + 1;
    let mut out = Vec::with_capacity(n_samples);

    for k in 0..n_samples {
        let t = trajectory.t_start() + k as f64 * dt;
        // Guard the final sample against floating-point overshoot past t_end.
        let t = t.min(trajectory.t_end());
        let state = trajectory.state(t)?;

        // The measurement model itself lives in kornia-sensors, beside the
        // preintegration it inverts. This loop only supplies the kinematics and
        // the noise; it does not restate the sensor equations.
        let mut measurement = ImuMeasurement::simulate(
            t,
            &state.rotation.matrix(),
            state.acceleration,
            state.angular_velocity,
            gravity,
            &config.bias,
        );
        if config.add_noise {
            measurement.gyro += rng.normal_vec3(gyro_std);
            measurement.accel += rng.normal_vec3(accel_std);
        }

        out.push(measurement);
    }

    Ok(out)
}

/// Builds one [`ImuFactor`] per consecutive pair of keyframe times.
///
/// `bias_estimate` is the bias the factors are **linearized at** — the
/// estimator's current guess, not the true bias. Passing the estimator's guess
/// (typically zero) while the measurements carry the true bias is precisely
/// what makes bias recovery a real test rather than a tautology.
pub fn build_imu_factors(
    measurements: &[ImuMeasurement],
    keyframe_times: &[f64],
    bias_estimate: ImuBias,
    calib: ImuCalib,
) -> Result<Vec<ImuFactor>, SimError> {
    if keyframe_times.len() < 2 {
        return Err(SimError::InvalidConfig(format!(
            "need at least 2 keyframe times to form an IMU edge, got {}",
            keyframe_times.len()
        )));
    }

    let mut factors = Vec::with_capacity(keyframe_times.len() - 1);
    for i in 0..keyframe_times.len() - 1 {
        let (t0, t1) = (keyframe_times[i], keyframe_times[i + 1]);
        if t1 <= t0 {
            return Err(SimError::InvalidConfig(format!(
                "keyframe times must be strictly increasing: {t0} then {t1}"
            )));
        }
        factors.push(ImuFactor {
            from_idx: i,
            to_idx: i + 1,
            preintegrated: PreintegratedImu::from_measurements(
                bias_estimate,
                calib,
                measurements,
                t0,
                t1,
            ),
        });
    }
    Ok(factors)
}

/// Convenience wrapper generating measurements under
/// [`DEFAULT_GRAVITY`].
pub fn generate_imu_default_gravity(
    trajectory: &Trajectory,
    config: &ImuSimConfig,
    rng: &mut SimRng,
) -> Result<Vec<ImuMeasurement>, SimError> {
    generate_imu(trajectory, DEFAULT_GRAVITY, config, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::trajectory::ArcConfig;
    use kornia_algebra::{Mat3F64, SO3F64};

    fn traj() -> Trajectory {
        Trajectory::arc(&ArcConfig {
            radius: 4.0,
            climb: 0.5,
            n_control: 20,
            t_start: 0.0,
            knot_dt: 0.25,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn stationary_level_body_reads_only_gravity() {
        // A body held level and still measures a specific force of -g: the
        // classic "accelerometer reads +9.81 upward while at rest" check. With
        // OpenCV Y-down gravity of [0, 9.81, 0], that is [0, -9.81, 0].
        let n = 8;
        let positions = vec![Vec3F64::new(1.0, 2.0, 3.0); n];
        let rotations = vec![SO3F64::exp(Vec3F64::ZERO); n];
        let trajectory = Trajectory::new(
            crate::sim::spline::RSpline3::new(positions, 0.0, 0.5).unwrap(),
            crate::sim::spline::So3Spline::new(rotations, 0.0, 0.5).unwrap(),
        )
        .unwrap();

        let mut rng = SimRng::new(1);
        let imu =
            generate_imu_default_gravity(&trajectory, &ImuSimConfig::default(), &mut rng).unwrap();

        for m in &imu {
            assert!(m.gyro.length() < 1e-12, "stationary body should not rotate");
            let expected = Vec3F64::new(0.0, -9.81, 0.0);
            assert!(
                (m.accel - expected).length() < 1e-9,
                "accel {:?} != {expected:?}",
                m.accel
            );
        }
    }

    #[test]
    fn injected_bias_appears_in_the_measurements() {
        let trajectory = traj();
        let bias = ImuBias {
            gyro: Vec3F64::new(0.01, -0.02, 0.005),
            accel: Vec3F64::new(0.05, 0.03, -0.04),
        };
        let mut rng = SimRng::new(2);

        let clean =
            generate_imu_default_gravity(&trajectory, &ImuSimConfig::default(), &mut rng).unwrap();
        let biased = generate_imu_default_gravity(
            &trajectory,
            &ImuSimConfig {
                bias,
                ..Default::default()
            },
            &mut rng,
        )
        .unwrap();

        assert_eq!(clean.len(), biased.len());
        for (c, b) in clean.iter().zip(biased.iter()) {
            assert!((b.gyro - c.gyro - bias.gyro).length() < 1e-12);
            assert!((b.accel - c.accel - bias.accel).length() < 1e-12);
        }
    }

    /// Preintegration error against ground truth, at one sample rate.
    fn preintegration_error(rate_hz: f64) -> (f64, f64, f64) {
        let trajectory = traj();
        let mut rng = SimRng::new(3);
        let config = ImuSimConfig {
            rate_hz,
            ..Default::default()
        };
        let imu = generate_imu_default_gravity(&trajectory, &config, &mut rng).unwrap();

        let (t0, t1) = (1.0, 1.5);
        let pim =
            PreintegratedImu::from_measurements(ImuBias::default(), config.calib, &imu, t0, t1);

        let s0 = trajectory.state(t0).unwrap();
        let s1 = trajectory.state(t1).unwrap();
        let dt = t1 - t0;
        let r_bw_0 = Mat3F64(*s0.rotation.matrix().transpose());

        // ΔR should equal R_wb(t0)ᵀ · R_wb(t1).
        let expected_dr = s0.rotation.inverse() * s1.rotation;
        let actual_dr = SO3F64::from_matrix(&pim.delta_rotation);
        let rot = (expected_dr.inverse() * actual_dr).log().length();

        // Δv = R_bw_0 · (v1 − v0 − g·dt)
        let expected_dv = r_bw_0 * (s1.velocity - s0.velocity - DEFAULT_GRAVITY * dt);
        let vel = (pim.delta_velocity - expected_dv).length();

        // Δp = R_bw_0 · (p1 − p0 − v0·dt − ½g·dt²)
        let expected_dp = r_bw_0
            * (s1.position - s0.position - s0.velocity * dt - DEFAULT_GRAVITY * (0.5 * dt * dt));
        let pos = (pim.delta_position - expected_dp).length();

        (rot, vel, pos)
    }

    /// The end-to-end consistency check between generation and preintegration:
    /// integrating noiseless measurements must reproduce the trajectory's own
    /// change in rotation, velocity and position.
    ///
    /// This asserts **first-order convergence** rather than a fixed tolerance,
    /// and the distinction matters. The simulator's measurements are analytically
    /// exact, but [`PreintegratedImu::integrate`] is a first-order (rectangular)
    /// scheme: it advances `Δv += ΔR·a·dt` using `ΔR` from the *start* of each
    /// step. The residual disagreement is therefore genuine discretization error,
    /// on the order of 5 mm/s over a 0.5 s edge at 400 Hz — not a bug, and not
    /// something a tighter tolerance could ever reach.
    ///
    /// Halving the timestep must halve the error. That is the assertion that
    /// actually discriminates: a frame or sign-convention mismatch between the
    /// measurement model and `vi_ba_schur`'s residual produces a *constant*
    /// error that does not shrink with rate, and a fixed-tolerance test would
    /// have to be loosened until it accepted exactly that.
    #[test]
    fn preintegration_converges_to_ground_truth_motion() {
        let coarse = preintegration_error(200.0);
        let fine = preintegration_error(400.0);
        let finer = preintegration_error(800.0);

        // Rotation is recovered to machine precision at every rate, not merely
        // to first order. This arc turns about a fixed axis, so the incremental
        // `exp(ω·dt)` factors all commute and their product telescopes exactly —
        // there is no discretization error left to converge. Asserting the tight
        // bound is worth more than a convergence ratio here: a wrong gyro sign
        // or a body/world frame swap would land many orders of magnitude away,
        // and dividing one rounding error by another only measures noise.
        for (rate, e) in [(200.0, coarse.0), (400.0, fine.0), (800.0, finer.0)] {
            assert!(
                e < 1e-12,
                "rotation error {e} rad at {rate} Hz is not exact"
            );
        }

        // Velocity and position do carry genuine first-order error, so here the
        // convergence rate is the meaningful check.
        assert!(finer.1 < 1e-4, "velocity error {} m/s at 800 Hz", finer.1);
        assert!(finer.2 < 1e-4, "position error {} m at 800 Hz", finer.2);

        for (name, c, f, ff) in [
            ("velocity", coarse.1, fine.1, finer.1),
            ("position", coarse.2, fine.2, finer.2),
        ] {
            let r1 = c / f;
            let r2 = f / ff;
            assert!(
                (1.6..2.6).contains(&r1),
                "{name} error ratio 200→400 Hz is {r1}, expected ~2 (first order)"
            );
            assert!(
                (1.6..2.6).contains(&r2),
                "{name} error ratio 400→800 Hz is {r2}, expected ~2 (first order)"
            );
        }
    }

    #[test]
    fn noise_has_the_discretized_standard_deviation() {
        // Generate on a stationary, level trajectory so the signal is constant
        // and every deviation is noise.
        let n = 8;
        let trajectory = Trajectory::new(
            crate::sim::spline::RSpline3::new(vec![Vec3F64::ZERO; n], 0.0, 2.0).unwrap(),
            crate::sim::spline::So3Spline::new(vec![SO3F64::exp(Vec3F64::ZERO); n], 0.0, 2.0)
                .unwrap(),
        )
        .unwrap();

        let config = ImuSimConfig {
            rate_hz: 200.0,
            add_noise: true,
            ..Default::default()
        };
        let mut rng = SimRng::new(4);
        let imu = generate_imu_default_gravity(&trajectory, &config, &mut rng).unwrap();

        let dt = 1.0 / config.rate_hz;
        let expected_gyro_std = config.calib.gyro_noise / dt.sqrt();

        let n_samples = imu.len() as f64;
        let mean = imu.iter().map(|m| m.gyro.x).sum::<f64>() / n_samples;
        let var = imu.iter().map(|m| (m.gyro.x - mean).powi(2)).sum::<f64>() / n_samples;

        let ratio = var.sqrt() / expected_gyro_std;
        assert!(
            (0.9..1.1).contains(&ratio),
            "gyro noise std ratio {ratio} — discretization is off"
        );
    }

    #[test]
    fn imu_factors_span_consecutive_keyframes() {
        let trajectory = traj();
        let mut rng = SimRng::new(5);
        let imu =
            generate_imu_default_gravity(&trajectory, &ImuSimConfig::default(), &mut rng).unwrap();
        let times: Vec<f64> = (0..5).map(|i| 0.5 + i as f64 * 0.3).collect();

        let factors = build_imu_factors(&imu, &times, ImuBias::default(), EUROC_IMU_CALIB).unwrap();
        assert_eq!(factors.len(), 4);
        for (i, f) in factors.iter().enumerate() {
            assert_eq!(f.from_idx, i);
            assert_eq!(f.to_idx, i + 1);
            assert!(
                (f.preintegrated.dt - 0.3).abs() < 1e-9,
                "edge dt {} != 0.3",
                f.preintegrated.dt
            );
        }
    }

    #[test]
    fn non_increasing_keyframe_times_are_rejected() {
        let trajectory = traj();
        let mut rng = SimRng::new(6);
        let imu =
            generate_imu_default_gravity(&trajectory, &ImuSimConfig::default(), &mut rng).unwrap();
        let bad = vec![1.0, 0.9];
        assert!(build_imu_factors(&imu, &bad, ImuBias::default(), EUROC_IMU_CALIB).is_err());
    }
}
