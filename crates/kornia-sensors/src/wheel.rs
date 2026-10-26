//! Wheel-encoder odometry: measurements, kinematics and preintegration.
//!
//! The base frame follows REP-103: x forward, y left, z up, on the floor
//! midway between the drive wheels. Wheels observe only planar motion
//! `(x, y, θ)`; [`PlanarMotion::delta_pose`] lifts it to SE(3) with zero z,
//! roll and pitch.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use std::f64::consts::TAU;

/// Errors from [`WheelPreintegration::integrate`]. The preintegration is left
/// unchanged, so the caller may skip the measurement and carry on.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum WheelError {
    #[error("wheel measurement is flagged invalid")]
    Invalid,
    #[error("wheel measurement has a non-finite timestamp, reading or result")]
    NonFinite,
    #[error("wheel measurement at {timestamp} s precedes the last one at {last} s")]
    OutOfOrder { timestamp: f64, last: f64 },
    #[error("wheel measurement repeats the timestamp {0} s")]
    Duplicate(f64),
    #[error("wheel measurement mixes encoder ticks and angular velocities")]
    MixedReadings,
    #[error("wheel noise must be finite and non-negative")]
    InvalidNoise,
}

/// What the wheel encoders report, `[left, right]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WheelReading {
    /// Cumulative encoder counts; positive drives the base forward.
    Ticks([i64; 2]),
    /// Wheel angular velocity in rad/s; positive drives the base forward.
    AngularVelocity([f64; 2]),
}

/// One wheel-encoder sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelMeasurement {
    /// Timestamp in seconds.
    pub timestamp: f64,
    pub reading: WheelReading,
    /// `false` marks a sample the driver flagged as bad; it is skipped.
    pub valid: bool,
}

impl WheelMeasurement {
    /// A valid sample of cumulative encoder counts.
    pub fn ticks(timestamp: f64, ticks: [i64; 2]) -> Self {
        Self {
            timestamp,
            reading: WheelReading::Ticks(ticks),
            valid: true,
        }
    }

    /// A valid sample of wheel angular velocities in rad/s.
    pub fn angular_velocity(timestamp: f64, rates: [f64; 2]) -> Self {
        Self {
            timestamp,
            reading: WheelReading::AngularVelocity(rates),
            valid: true,
        }
    }
}

/// Planar velocity of the base, in base axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTwist {
    /// Forward velocity, m/s.
    pub vx: f64,
    /// Leftward velocity, m/s.
    pub vy: f64,
    /// Yaw rate, rad/s, counter-clockwise seen from above.
    pub wz: f64,
}

/// Maps wheel angular velocities to the base twist.
pub trait WheelKinematics {
    /// Encoder counts per wheel revolution.
    fn ticks_per_revolution(&self) -> f64;

    /// Base twist from wheel angular velocities `[left, right]` in rad/s.
    fn twist(&self, wheel_rates: [f64; 2]) -> BodyTwist;

    /// `∂(vx, vy, wz)/∂left` and `∂(vx, vy, wz)/∂right` at `wheel_rates`.
    fn twist_jacobian(&self, wheel_rates: [f64; 2]) -> [Vec3F64; 2];

    /// Mean wheel angular velocities in rad/s from a change of `tick_delta`
    /// encoder counts over `dt` seconds.
    fn wheel_rates_from_ticks(&self, tick_delta: [i64; 2], dt: f64) -> [f64; 2] {
        let rad_per_tick = TAU / self.ticks_per_revolution();
        tick_delta.map(|t| t as f64 * rad_per_tick / dt)
    }
}

/// Two independently driven wheels on a common axle.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DifferentialDrive {
    /// Wheel radius, metres.
    pub wheel_radius: f64,
    /// Distance between the wheel contact points, metres.
    pub track_width: f64,
    /// Encoder counts per wheel revolution (after any gearing).
    pub ticks_per_revolution: f64,
}

impl WheelKinematics for DifferentialDrive {
    fn ticks_per_revolution(&self) -> f64 {
        self.ticks_per_revolution
    }

    fn twist(&self, [left, right]: [f64; 2]) -> BodyTwist {
        BodyTwist {
            vx: 0.5 * self.wheel_radius * (left + right),
            vy: 0.0,
            wz: self.wheel_radius * (right - left) / self.track_width,
        }
    }

    fn twist_jacobian(&self, _: [f64; 2]) -> [Vec3F64; 2] {
        let half_r = 0.5 * self.wheel_radius;
        let r_b = self.wheel_radius / self.track_width;
        [
            Vec3F64::new(half_r, 0.0, -r_b),
            Vec3F64::new(half_r, 0.0, r_b),
        ]
    }
}

/// White-noise densities driving the preintegrated covariance.
///
/// The defaults are unmeasured placeholders; tune them on the robot.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct WheelNoise {
    /// Per-wheel angular velocity noise, rad/s/√Hz.
    pub wheel_rate: f64,
    /// Lateral (side-slip) velocity noise of the base, m/s/√Hz. A
    /// differential drive has no lateral motion, so without it the
    /// covariance is singular.
    pub lateral_velocity: f64,
}

impl Default for WheelNoise {
    fn default() -> Self {
        Self {
            wheel_rate: 0.05,
            lateral_velocity: 0.01,
        }
    }
}

impl WheelNoise {
    fn is_valid(&self) -> bool {
        [self.wheel_rate, self.lateral_velocity]
            .iter()
            .all(|s| s.is_finite() && *s >= 0.0)
    }
}

/// Planar motion of the base between two instants, with its covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarMotion {
    /// `(dx, dy, dθ)`: base j in base i, metres and radians.
    pub delta: [f64; 3],
    /// Covariance of `delta` (m², m·rad, rad²).
    pub covariance: Mat3F64,
}

impl PlanarMotion {
    /// No motion, no uncertainty.
    pub const IDENTITY: Self = Self {
        delta: [0.0; 3],
        covariance: Mat3F64::ZERO,
    };

    /// The delta as a transform taking base-j coordinates to base-i.
    pub fn delta_pose(&self) -> Pose3d {
        let [dx, dy, dtheta] = self.delta;
        Pose3d::new(yaw_rotation(dtheta), Vec3F64::new(dx, dy, 0.0))
    }

    /// The motion `self` then `next` (base k in base i from i→j and j→k).
    ///
    /// The covariance propagates to first order through the SE(2)
    /// composition, `Jᵢ Σᵢ Jᵢᵀ + Jⱼ Σⱼ Jⱼᵀ`; the two motions' noise is
    /// independent.
    pub fn compose(&self, next: &Self) -> Self {
        let [x1, y1, th1] = self.delta;
        let [x2, y2, th2] = next.delta;
        let j_next = yaw_rotation(th1);
        let t2 = j_next * Vec3F64::new(x2, y2, 0.0);
        let delta = [
            x1 + t2.x,
            y1 + t2.y,
            (th1 + th2).sin().atan2((th1 + th2).cos()),
        ];
        let j_first = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-t2.y, t2.x, 1.0),
        );
        let a = j_first * self.covariance * j_first.transpose();
        let b = j_next * next.covariance * j_next.transpose();
        // Symmetrized, bit-exactly given the grouping: consumers check
        // symmetry before inverting.
        let covariance = ((a + a.transpose()) + (b + b.transpose())) * 0.5;
        Self { delta, covariance }
    }

    /// The motion of holding the twist increment `u = (vx, vy, wz)·dt` for
    /// the interval (an arc), with `u`'s covariance `q` mapped through the
    /// SE(2) exponential.
    fn from_twist_increment(u: [f64; 3], q: Mat3F64) -> Self {
        let (delta, jacobian) = se2_exp(u);
        Self {
            delta,
            covariance: jacobian * q * jacobian.transpose(),
        }
    }
}

fn yaw_rotation(theta: f64) -> Mat3F64 {
    SO3F64::exp(Vec3F64::new(0.0, 0.0, theta)).matrix()
}

/// SE(2) exponential of `(a, b, w)` as `(x, y, θ)`, and its Jacobian.
///
/// kornia-algebra's SE(2) is f32-only and its Jacobians are over the tangent
/// space, not over `(x, y, θ)` as the wheel factor's covariance is.
fn se2_exp([a, b, w]: [f64; 3]) -> ([f64; 3], Mat3F64) {
    // x = A·a − B·b, y = B·a + A·b with A = sin w / w, B = (1 − cos w) / w;
    // A' and B' are their derivatives in w. Below 1e-3 the closed forms
    // cancel catastrophically, so use the Taylor series instead.
    let (s, c) = w.sin_cos();
    let w2 = w * w;
    let (aa, bb, da, db) = if w.abs() < 1e-3 {
        (
            1.0 - w2 / 6.0 + w2 * w2 / 120.0,
            w / 2.0 - w * w2 / 24.0,
            -w / 3.0 + w * w2 / 30.0,
            0.5 - w2 / 8.0 + w2 * w2 / 144.0,
        )
    } else {
        let aa = s / w;
        let bb = (1.0 - c) / w;
        (aa, bb, (c - aa) / w, (s - bb) / w)
    };
    let delta = [aa * a - bb * b, bb * a + aa * b, w];
    let jacobian = Mat3F64::from_cols(
        Vec3F64::new(aa, bb, 0.0),
        Vec3F64::new(-bb, aa, 0.0),
        Vec3F64::new(da * a - db * b, db * a + da * b, 1.0),
    );
    (delta, jacobian)
}

/// Wheel odometry accumulated between the first and the last integrated
/// measurement.
///
/// Each interval between consecutive valid measurements is integrated as an
/// arc of constant twist: tick readings give the mean wheel rates over the
/// interval exactly, angular-velocity readings are averaged (trapezoidal).
/// To preintegrate consecutive keyframe intervals, read [`Self::motion`] and
/// call [`Self::restart`]; the next interval starts at the last measurement,
/// so composing the pieces gives the whole.
#[derive(Debug, Clone)]
pub struct WheelPreintegration<K = DifferentialDrive> {
    kinematics: K,
    noise: WheelNoise,
    motion: PlanarMotion,
    dt: f64,
    last: Option<WheelMeasurement>,
}

impl<K: WheelKinematics> WheelPreintegration<K> {
    /// An empty preintegration; the first measurement sets its start.
    pub fn new(kinematics: K, noise: WheelNoise) -> Result<Self, WheelError> {
        if !noise.is_valid() {
            return Err(WheelError::InvalidNoise);
        }
        Ok(Self {
            kinematics,
            noise,
            motion: PlanarMotion::IDENTITY,
            dt: 0.0,
            last: None,
        })
    }

    /// Base motion since the start, base-now in base-start.
    pub fn motion(&self) -> &PlanarMotion {
        &self.motion
    }

    /// Integrated time, seconds.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// The last integrated measurement, where the next interval starts.
    pub fn last(&self) -> Option<&WheelMeasurement> {
        self.last.as_ref()
    }

    /// Clears the accumulated motion, keeping the last measurement as the
    /// new start.
    pub fn restart(&mut self) {
        self.motion = PlanarMotion::IDENTITY;
        self.dt = 0.0;
    }

    /// Integrates the interval from the last measurement to `measurement`.
    ///
    /// # Errors
    ///
    /// A measurement that is flagged invalid, non-finite, older than or as
    /// old as the last one, or of a different [`WheelReading`] kind is
    /// rejected and the state left unchanged. Skipping it loses no tick-based
    /// motion: the next interval spans it.
    pub fn integrate(&mut self, measurement: &WheelMeasurement) -> Result<(), WheelError> {
        if !measurement.valid {
            return Err(WheelError::Invalid);
        }
        if !is_finite(measurement) {
            return Err(WheelError::NonFinite);
        }
        let Some(last) = self.last else {
            self.last = Some(*measurement);
            return Ok(());
        };
        let dt = measurement.timestamp - last.timestamp;
        if dt < 0.0 {
            return Err(WheelError::OutOfOrder {
                timestamp: measurement.timestamp,
                last: last.timestamp,
            });
        }
        if dt == 0.0 {
            return Err(WheelError::Duplicate(measurement.timestamp));
        }
        let rates = match (last.reading, measurement.reading) {
            (WheelReading::Ticks(from), WheelReading::Ticks(to)) => {
                let delta = [to[0].wrapping_sub(from[0]), to[1].wrapping_sub(from[1])];
                self.kinematics.wheel_rates_from_ticks(delta, dt)
            }
            (WheelReading::AngularVelocity(from), WheelReading::AngularVelocity(to)) => {
                [0.5 * (from[0] + to[0]), 0.5 * (from[1] + to[1])]
            }
            _ => return Err(WheelError::MixedReadings),
        };
        let step = self.interval_motion(rates, dt);
        let motion = self.motion.compose(&step);
        if !(motion.delta.iter().all(|v| v.is_finite()) && motion.covariance.is_finite()) {
            return Err(WheelError::NonFinite);
        }
        self.motion = motion;
        self.dt += dt;
        self.last = Some(*measurement);
        Ok(())
    }

    /// Motion over `dt` seconds at constant wheel rates, with white rate and
    /// lateral-velocity noise: `Var(u) = (J diag(σ_w²) Jᵀ + diag(0, σ_y², 0))·dt`.
    fn interval_motion(&self, rates: [f64; 2], dt: f64) -> PlanarMotion {
        let twist = self.kinematics.twist(rates);
        let u = [twist.vx * dt, twist.vy * dt, twist.wz * dt];
        let [d_left, d_right] = self.kinematics.twist_jacobian(rates);
        let j = Mat3F64::from_cols(d_left, d_right, Vec3F64::ZERO);
        let lateral =
            Mat3F64::from_diagonal(Vec3F64::new(0.0, self.noise.lateral_velocity.powi(2), 0.0));
        let q = (j * j.transpose() * self.noise.wheel_rate.powi(2) + lateral) * dt;
        PlanarMotion::from_twist_increment(u, q)
    }
}

fn is_finite(m: &WheelMeasurement) -> bool {
    m.timestamp.is_finite()
        && match m.reading {
            WheelReading::Ticks(_) => true,
            WheelReading::AngularVelocity(rates) => rates.iter().all(|r| r.is_finite()),
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRIVE: DifferentialDrive = DifferentialDrive {
        wheel_radius: 0.1,
        track_width: 0.5,
        ticks_per_revolution: 1024.0,
    };

    /// Preintegrates constant wheel rates sampled at 100 Hz for `seconds`.
    fn constant_rates(rates: [f64; 2], seconds: f64) -> WheelPreintegration {
        let mut pre = WheelPreintegration::new(DRIVE, WheelNoise::default()).unwrap();
        let steps = (seconds * 100.0).round() as usize;
        for k in 0..=steps {
            let m = WheelMeasurement::angular_velocity(k as f64 * 0.01, rates);
            pre.integrate(&m).unwrap();
        }
        pre
    }

    fn assert_delta(got: [f64; 3], want: [f64; 3], tol: f64) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "{got:?} vs {want:?}");
        }
    }

    fn assert_matrix_near(a: Mat3F64, b: Mat3F64, rel: f64) {
        let (a, b) = (a.to_cols_array(), b.to_cols_array());
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= rel * scale, "{a:?} vs {b:?}");
        }
    }

    fn assert_positive_definite(c: Mat3F64) {
        assert_eq!(c, c.transpose(), "asymmetric");
        let minor2 = c.x_axis.x * c.y_axis.y - c.y_axis.x * c.x_axis.y;
        assert!(
            c.x_axis.x > 0.0 && minor2 > 0.0 && c.determinant() > 0.0,
            "{c:?}"
        );
    }

    fn trace(c: Mat3F64) -> f64 {
        c.x_axis.x + c.y_axis.y + c.z_axis.z
    }

    #[test]
    fn ticks_convert_to_wheel_rates() {
        // Half a revolution in half a second is 2π rad/s.
        let rates = DRIVE.wheel_rates_from_ticks([512, -256], 0.5);
        assert!((rates[0] - TAU).abs() < 1e-12);
        assert!((rates[1] + 0.5 * TAU).abs() < 1e-12);
        let twist = DRIVE.twist(rates);
        assert!((twist.vx - 0.1 * 0.25 * TAU).abs() < 1e-12);
        assert!((twist.wz + 0.1 * 1.5 * TAU / 0.5).abs() < 1e-12);
    }

    #[test]
    fn equal_wheel_speeds_translate() {
        let pre = constant_rates([2.0, 2.0], 3.0);
        assert_delta(pre.motion().delta, [0.1 * 2.0 * 3.0, 0.0, 0.0], 1e-12);
        assert!((pre.dt() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn opposite_wheel_speeds_rotate_in_place() {
        let pre = constant_rates([-1.0, 1.0], 2.0);
        // ω = r(ωr − ωl)/b = 0.4 rad/s for 2 s.
        assert_delta(pre.motion().delta, [0.0, 0.0, 0.8], 1e-12);
    }

    #[test]
    fn constant_rates_follow_the_closed_form_arc() {
        let (v, w, t) = (0.1 * 3.0, 0.1 * 2.0 / 0.5, 2.5);
        let pre = constant_rates([2.0, 4.0], t);
        let want = [v / w * (w * t).sin(), v / w * (1.0 - (w * t).cos()), w * t];
        assert_delta(pre.motion().delta, want, 1e-12);
    }

    #[test]
    fn ticks_follow_the_closed_form_arc() {
        let mut pre = WheelPreintegration::new(DRIVE, WheelNoise::default()).unwrap();
        // 1024 and 2048 ticks per second at 128 Hz: 2π and 4π rad/s.
        for k in 0..=256i64 {
            let m = WheelMeasurement::ticks(k as f64 / 128.0, [k * 8, k * 16]);
            pre.integrate(&m).unwrap();
        }
        let (v, w, t) = (0.1 * 3.0 * TAU / 2.0, 0.1 * TAU / 0.5, 2.0);
        let want = [v / w * (w * t).sin(), v / w * (1.0 - (w * t).cos()), w * t];
        assert_delta(pre.motion().delta, want, 1e-12);
    }

    #[test]
    fn split_preintegration_composes_to_the_whole() {
        let rates = |k: usize| [1.0 + 0.01 * k as f64, 2.0 - 0.02 * k as f64];
        let new = || WheelPreintegration::new(DRIVE, WheelNoise::default()).unwrap();
        let (mut whole, mut split) = (new(), new());
        let mut first = None;
        for k in 0..=200 {
            let m = WheelMeasurement::angular_velocity(k as f64 * 0.01, rates(k));
            whole.integrate(&m).unwrap();
            split.integrate(&m).unwrap();
            if k == 80 {
                first = Some(*split.motion());
                split.restart();
            }
        }
        let composed = first.unwrap().compose(split.motion());
        assert_delta(composed.delta, whole.motion().delta, 1e-12);
        assert_matrix_near(composed.covariance, whole.motion().covariance, 1e-12);
    }

    #[test]
    fn covariance_is_positive_definite_and_grows() {
        let mut pre = WheelPreintegration::new(DRIVE, WheelNoise::default()).unwrap();
        let mut prev_trace = 0.0;
        for k in 0..=300 {
            let t = k as f64 * 0.01;
            let m = WheelMeasurement::angular_velocity(t, [2.0 + t.sin(), 3.0 - t.cos()]);
            pre.integrate(&m).unwrap();
            if k > 0 {
                let c = pre.motion().covariance;
                assert_positive_definite(c);
                assert!(trace(c) > prev_trace);
                prev_trace = trace(c);
            }
        }
    }

    #[test]
    fn standing_still_accumulates_only_noise() {
        let noise = WheelNoise::default();
        let mut pre = WheelPreintegration::new(DRIVE, noise).unwrap();
        for k in 0..=500 {
            pre.integrate(&WheelMeasurement::ticks(k as f64 * 0.01, [7, -3]))
                .unwrap();
        }
        assert_eq!(pre.motion().delta, [0.0; 3]);
        let t = 5.0;
        let (r, b, s2) = (0.1, 0.5, noise.wheel_rate * noise.wheel_rate);
        let want = Mat3F64::from_diagonal(Vec3F64::new(
            0.5 * r * r * s2 * t,
            noise.lateral_velocity.powi(2) * t,
            2.0 * r * r / (b * b) * s2 * t,
        ));
        assert_matrix_near(pre.motion().covariance, want, 1e-9);
    }

    #[test]
    fn bad_measurements_are_skipped_without_losing_ticks() {
        let mut pre = WheelPreintegration::new(DRIVE, WheelNoise::default()).unwrap();
        pre.integrate(&WheelMeasurement::ticks(0.0, [0, 0]))
            .unwrap();
        pre.integrate(&WheelMeasurement::ticks(0.1, [100, 100]))
            .unwrap();
        let before = pre.clone();

        let mut flagged = WheelMeasurement::ticks(0.2, [200, 200]);
        flagged.valid = false;
        assert_eq!(pre.integrate(&flagged), Err(WheelError::Invalid));
        assert_eq!(
            pre.integrate(&WheelMeasurement::ticks(0.1, [150, 150])),
            Err(WheelError::Duplicate(0.1))
        );
        assert!(matches!(
            pre.integrate(&WheelMeasurement::ticks(0.05, [50, 50])),
            Err(WheelError::OutOfOrder { .. })
        ));
        assert_eq!(
            pre.integrate(&WheelMeasurement::angular_velocity(0.3, [1.0, 1.0])),
            Err(WheelError::MixedReadings)
        );
        assert_eq!(
            pre.integrate(&WheelMeasurement::ticks(f64::NAN, [0, 0])),
            Err(WheelError::NonFinite)
        );
        assert_eq!(pre.motion(), before.motion());
        assert_eq!(pre.dt(), before.dt());

        pre.integrate(&WheelMeasurement::ticks(0.3, [300, 300]))
            .unwrap();
        let metres_per_tick = TAU * 0.1 / 1024.0;
        assert_delta(
            pre.motion().delta,
            [300.0 * metres_per_tick, 0.0, 0.0],
            1e-12,
        );
    }

    #[test]
    fn negative_noise_is_refused() {
        let noise = WheelNoise {
            wheel_rate: -0.1,
            ..WheelNoise::default()
        };
        assert_eq!(
            WheelPreintegration::new(DRIVE, noise).err(),
            Some(WheelError::InvalidNoise)
        );
    }

    #[test]
    fn delta_pose_lifts_the_planar_motion() {
        let motion = PlanarMotion {
            delta: [1.0, -0.5, 0.3],
            ..PlanarMotion::IDENTITY
        };
        let pose = motion.delta_pose();
        let t = pose.translation;
        assert_eq!([t.x, t.y, t.z], [1.0, -0.5, 0.0]);
        assert!((pose.rotation.x_axis.y.atan2(pose.rotation.x_axis.x) - 0.3).abs() < 1e-12);
        assert!((pose.rotation.z_axis.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn composed_covariance_is_the_linearized_propagation() {
        let first = PlanarMotion {
            delta: [0.3, -0.1, 0.7],
            covariance: Mat3F64::from_cols_array(&[
                2e-4, 1e-5, 0.0, 1e-5, 1e-4, 2e-5, 0.0, 2e-5, 3e-4,
            ]),
        };
        let next = PlanarMotion {
            delta: [0.2, 0.15, -0.4],
            covariance: Mat3F64::from_cols_array(&[
                1e-4, 0.0, 1e-5, 0.0, 4e-4, 0.0, 1e-5, 0.0, 2e-4,
            ]),
        };
        // Central differences of the composed delta wrt each input delta.
        const EPS: f64 = 1e-6;
        let jacobian = |which: usize| {
            let column = |k: usize| {
                let shifted = |sign: f64| {
                    let (mut a, mut b) = (first, next);
                    let d = if which == 0 {
                        &mut a.delta
                    } else {
                        &mut b.delta
                    };
                    d[k] += sign * EPS;
                    Vec3F64::from_array(a.compose(&b).delta)
                };
                (shifted(1.0) - shifted(-1.0)) * (0.5 / EPS)
            };
            Mat3F64::from_cols(column(0), column(1), column(2))
        };
        let (j_first, j_next) = (jacobian(0), jacobian(1));
        let want = j_first * first.covariance * j_first.transpose()
            + j_next * next.covariance * j_next.transpose();
        let got = first.compose(&next).covariance.to_cols_array();
        for (g, w) in got.iter().zip(want.to_cols_array()) {
            assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn se2_exp_jacobian_matches_finite_differences() {
        for u in [[0.3, -0.1, 0.7], [0.5, 0.2, 2e-4], [0.0, 0.0, 0.0]] {
            let (_, jac) = se2_exp(u);
            const EPS: f64 = 1e-6;
            for k in 0..3 {
                let (mut up, mut down) = (u, u);
                up[k] += EPS;
                down[k] -= EPS;
                let (p, m) = (se2_exp(up).0, se2_exp(down).0);
                for r in 0..3 {
                    let fd = (p[r] - m[r]) / (2.0 * EPS);
                    assert!((fd - jac.col(k)[r]).abs() < 1e-8, "u={u:?} ({r},{k})");
                }
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn kinematics_and_noise_load_from_ron() {
        let drive: DifferentialDrive =
            ron::from_str("(wheel_radius: 0.1, track_width: 0.5, ticks_per_revolution: 1024.0)")
                .unwrap();
        assert_eq!(drive, DRIVE);
        let noise: WheelNoise = ron::from_str("(wheel_rate: 0.2)").unwrap();
        assert_eq!(noise.wheel_rate, 0.2);
        assert_eq!(
            noise.lateral_velocity,
            WheelNoise::default().lateral_velocity
        );
    }
}
