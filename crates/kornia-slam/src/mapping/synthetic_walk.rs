//! Analytic walks with exact IMU streams, shared by the solver tests.

use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

/// Start velocity (m/s) of the synthetic walk and its acceleration
/// `A + W sin(ω t)` (m/s², rad/s). A constant acceleration alone would leave
/// scale to the free accel bias: `b_a = (1 - s) A` fits a world scaled by `s`
/// exactly. The varying part makes scale observable; roll, pitch and gravity
/// need the body to turn too ([`Sampling::Rotating`]).
const VELOCITY: [f64; 3] = [0.5, 0.0, 0.0];
const ACCELERATION: [f64; 3] = [0.4, 0.0, 0.2];
const WOBBLE: [f64; 3] = [0.3, 0.2, -0.3];
const WOBBLE_RAD_S: f64 = 3.0;

pub(crate) fn body_position(t: f64) -> Vec3F64 {
    let w = WOBBLE_RAD_S;
    Vec3F64::from_array(VELOCITY) * t
        + Vec3F64::from_array(ACCELERATION) * (0.5 * t * t)
        + Vec3F64::from_array(WOBBLE) * (t / w - (w * t).sin() / (w * w))
}

pub(crate) fn body_velocity(t: f64) -> [f64; 3] {
    let w = WOBBLE_RAD_S;
    (Vec3F64::from_array(VELOCITY)
        + Vec3F64::from_array(ACCELERATION) * t
        + Vec3F64::from_array(WOBBLE) * ((1.0 - (w * t).cos()) / w))
        .to_array()
}

/// How the synthetic IMU samples the specific force.
#[derive(Clone, Copy)]
pub(crate) enum Sampling {
    /// Each sample the mean over the interval before it: exact for `Hold`.
    IntervalMean,
    /// Each sample the signal at its stamp, as a real IMU reads it.
    Instantaneous,
    /// Interval means, and the body turns about two axes: without rotation
    /// a world tilted by `R_δ` and the bias `b_a = R_δᵀg − g` fit as well as
    /// the truth, so neither roll, pitch nor gravity's sign is observed (D2).
    Rotating,
}

/// `R_wb(t)` of the rotating walk: `Exp(α e_x) · Exp(0.7 α e_y)`,
/// `α = 0.25 (1 − cos 2t)` rad; the identity otherwise.
pub(crate) fn body_rotation(t: f64, sampling: Sampling) -> Mat3F64 {
    match sampling {
        Sampling::Rotating => {
            let a = 0.25 * (1.0 - (2.0 * t).cos());
            SO3F64::exp(Vec3F64::new(a, 0.0, 0.0)).matrix()
                * SO3F64::exp(Vec3F64::new(0.0, 0.7 * a, 0.0)).matrix()
        }
        _ => Mat3F64::IDENTITY,
    }
}

fn body_acceleration(t: f64) -> Vec3F64 {
    Vec3F64::from_array(ACCELERATION) + Vec3F64::from_array(WOBBLE) * (WOBBLE_RAD_S * t).sin()
}

/// The IMU reading stamped `t` (gyro, accel), held over `(t - dt, t]`: the
/// one the `Hold` preintegration reproduces exactly for `sampling`.
pub(crate) fn exact_reading(
    t: f64,
    dt: f64,
    sampling: Sampling,
    gravity: Vec3F64,
) -> ([f64; 3], [f64; 3]) {
    if let Sampling::Rotating = sampling {
        // Held over `(t − dt, t]` and turned by the rotation at its
        // start, as the `Hold` integrator applies it.
        let (r_wb0, r_wb1) = (body_rotation(t - dt, sampling), body_rotation(t, sampling));
        let dv = Vec3F64::from_array(body_velocity(t))
            - Vec3F64::from_array(body_velocity(t - dt))
            - gravity * dt;
        return (
            (SO3F64::from_matrix(&(r_wb0.transpose() * r_wb1)).log() * (1.0 / dt)).to_array(),
            ((r_wb0.transpose() * dv) * (1.0 / dt)).to_array(),
        );
    }
    let mean = match sampling {
        Sampling::Instantaneous => body_acceleration(t),
        Sampling::Rotating => unreachable!(),
        Sampling::IntervalMean if t <= 0.0 => Vec3F64::from_array(ACCELERATION),
        Sampling::IntervalMean => {
            (Vec3F64::from_array(body_velocity(t)) - Vec3F64::from_array(body_velocity(t - dt)))
                * (1.0 / dt)
        }
    };
    ([0.0; 3], (mean - gravity).to_array())
}

/// Deterministic standard-normal draws (xorshift64* and Box-Muller), so the
/// synthetic tests are reproducible without a `rand` dependency.
pub(crate) struct Gauss(pub(crate) u64);

impl Gauss {
    pub(crate) fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }

    pub(crate) fn vec3(&mut self, sigma: f64) -> Vec3F64 {
        Vec3F64::new(self.normal(), self.normal(), self.normal()) * sigma
    }
}
