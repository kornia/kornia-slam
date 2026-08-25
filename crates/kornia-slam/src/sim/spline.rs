//! Uniform cubic B-splines on R³ and SO(3).
//!
//! The trajectory is represented as **two independent splines** — a cubic
//! B-spline for translation and a cumulative cubic B-spline for rotation —
//! rather than a single spline on SE(3).
//!
//! Three reasons, in order of importance:
//!
//! 1. It needs no `SE3F64`. `kornia_algebra` has `SO3F64` but its SE(3) type is
//!    f32-only, and kornia-slam's estimation is f64 throughout. The split
//!    representation reaches f64 ground truth today.
//! 2. Derivatives stay simple. Translational velocity and acceleration are
//!    plain polynomial derivatives; angular velocity from a cumulative SO(3)
//!    spline has a closed form (see [`So3Spline::angular_velocity`]). A coupled
//!    SE(3) spline's derivatives are messier for no gain here.
//! 3. Rotation and translation become independently controllable, which is
//!    what makes degenerate scenarios — pure rotation with no translation,
//!    translation with no rotation — expressible at all.
//!
//! **The spline is the ground truth.** It is not an approximation of some other
//! "true" curve: whatever curve the control points describe *is* the trajectory
//! the measurements are generated from, so there is no modelling error between
//! ground truth and what the estimator is asked to recover. That is the whole
//! point of generating measurements analytically.
//!
//! # Upstreaming
//!
//! **This module is intended to move to `kornia-algebra`, beside the Lie
//! groups.** A split R³ + SO(3) B-spline is a general trajectory primitive,
//! kornia-rs has nothing like it (checked across all its crates), and it is a
//! manifold construct rather than a domain factor — so it sits on the right side
//! of the kornia-manifold roadmap's rule that domain factors stay in domain
//! crates.
//!
//! Dependencies are already satisfied upstream: this needs only `SO3F64` (landed
//! in kornia-rs PR #931) and `Vec3F64`. It deliberately does **not** need
//! `SE3F64`, which still does not exist — see §4.1 of the plan.
//!
//! Two things to handle when moving it:
//!
//! 1. It returns [`SimError`] for `TimeOutOfRange`, `TooFewControlPoints` and
//!    `InvalidKnotSpacing`. Those need a local error type upstream; they are the
//!    only coupling to kornia-slam.
//! 2. `kornia-algebra` is being renamed to `kornia-manifold` in Phase 0 of the
//!    manifold roadmap. Coordinate rather than race it — a new module landing
//!    mid-rename is avoidable churn.

use kornia_algebra::{SO3F64, Vec3F64};

use super::SimError;

/// Number of control points influencing any one spline segment (cubic → 4).
const SPLINE_ORDER: usize = 4;

/// Uniform cubic B-spline basis, evaluated at `u ∈ [0, 1)` within a segment.
///
/// Returns the four weights applied to control points `i .. i+3`, using the
/// standard De Boor basis matrix
///
/// ```text
///           [  1   4   1   0 ]
/// M = 1/6 · [ -3   0   3   0 ]
///           [  3  -6   3   0 ]
///           [ -1   3  -3   1 ]
/// ```
///
/// as `b(u) = [1, u, u², u³] · M`.
fn basis(u: f64) -> [f64; SPLINE_ORDER] {
    let (u2, u3) = (u * u, u * u * u);
    [
        (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
        (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
        (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
        u3 / 6.0,
    ]
}

/// First derivative of [`basis`] with respect to `u`.
fn basis_d1(u: f64) -> [f64; SPLINE_ORDER] {
    let u2 = u * u;
    [
        (-3.0 + 6.0 * u - 3.0 * u2) / 6.0,
        (-12.0 * u + 9.0 * u2) / 6.0,
        (3.0 + 6.0 * u - 9.0 * u2) / 6.0,
        3.0 * u2 / 6.0,
    ]
}

/// Second derivative of [`basis`] with respect to `u`.
fn basis_d2(u: f64) -> [f64; SPLINE_ORDER] {
    [
        (6.0 - 6.0 * u) / 6.0,
        (-12.0 + 18.0 * u) / 6.0,
        (6.0 - 18.0 * u) / 6.0,
        6.0 * u / 6.0,
    ]
}

/// Cumulative cubic B-spline basis: `λ_j(u) = Σ_{s ≥ j} b_s(u)`.
///
/// Only `λ_1..λ_3` are returned; `λ_0 ≡ 1` because the basis is a partition of
/// unity, and it multiplies the segment's anchor rotation directly.
fn cumulative_basis(u: f64) -> [f64; 3] {
    let (u2, u3) = (u * u, u * u * u);
    [
        (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0,
        (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0,
        u3 / 6.0,
    ]
}

/// First derivative of [`cumulative_basis`] with respect to `u`.
fn cumulative_basis_d1(u: f64) -> [f64; 3] {
    let u2 = u * u;
    [
        (3.0 - 6.0 * u + 3.0 * u2) / 6.0,
        (3.0 + 6.0 * u - 6.0 * u2) / 6.0,
        3.0 * u2 / 6.0,
    ]
}

/// Locates `t` within a uniform knot vector.
///
/// Returns the index of the first of the four control points governing the
/// segment, plus the normalized position `u ∈ [0, 1]` inside it.
fn locate(t: f64, t_start: f64, knot_dt: f64, n_control: usize) -> Result<(usize, f64), SimError> {
    let n_segments = n_control - (SPLINE_ORDER - 1);
    let t_end = t_start + n_segments as f64 * knot_dt;

    if !(t_start..=t_end).contains(&t) {
        return Err(SimError::TimeOutOfRange {
            t,
            start: t_start,
            end: t_end,
        });
    }

    let scaled = (t - t_start) / knot_dt;
    // `t == t_end` lands exactly on the segment boundary, where floor() would
    // index one segment past the end. Clamp it back to the final segment's
    // closing edge (u = 1), which evaluates to the same point by continuity.
    let segment = (scaled.floor() as usize).min(n_segments - 1);
    Ok((segment, scaled - segment as f64))
}

/// A uniform cubic B-spline on R³, used for the translational trajectory.
#[derive(Debug, Clone)]
pub struct RSpline3 {
    control: Vec<Vec3F64>,
    t_start: f64,
    knot_dt: f64,
}

impl RSpline3 {
    /// Builds a spline from control points on a uniform knot vector starting at
    /// `t_start` with spacing `knot_dt`.
    ///
    /// A cubic spline needs at least `SPLINE_ORDER` control points to define
    /// even one segment.
    pub fn new(control: Vec<Vec3F64>, t_start: f64, knot_dt: f64) -> Result<Self, SimError> {
        if control.len() < SPLINE_ORDER {
            return Err(SimError::TooFewControlPoints {
                got: control.len(),
                need: SPLINE_ORDER,
            });
        }
        if knot_dt <= 0.0 || !knot_dt.is_finite() {
            return Err(SimError::InvalidKnotSpacing(knot_dt));
        }
        Ok(Self {
            control,
            t_start,
            knot_dt,
        })
    }

    /// First time at which the spline is defined.
    pub fn t_start(&self) -> f64 {
        self.t_start
    }

    /// Last time at which the spline is defined.
    ///
    /// A cubic B-spline with `n` control points spans `n - 3` segments; the
    /// leading and trailing control points shape the curve without extending
    /// its valid domain.
    pub fn t_end(&self) -> f64 {
        self.t_start + (self.control.len() - (SPLINE_ORDER - 1)) as f64 * self.knot_dt
    }

    fn weighted(&self, segment: usize, weights: [f64; SPLINE_ORDER]) -> Vec3F64 {
        let mut acc = Vec3F64::ZERO;
        for (k, w) in weights.iter().enumerate() {
            acc += self.control[segment + k] * *w;
        }
        acc
    }

    /// Position at time `t`.
    pub fn position(&self, t: f64) -> Result<Vec3F64, SimError> {
        let (segment, u) = locate(t, self.t_start, self.knot_dt, self.control.len())?;
        Ok(self.weighted(segment, basis(u)))
    }

    /// Velocity at time `t`. Chain rule: `du/dt = 1 / knot_dt`.
    pub fn velocity(&self, t: f64) -> Result<Vec3F64, SimError> {
        let (segment, u) = locate(t, self.t_start, self.knot_dt, self.control.len())?;
        Ok(self.weighted(segment, basis_d1(u)) * (1.0 / self.knot_dt))
    }

    /// Acceleration at time `t`.
    pub fn acceleration(&self, t: f64) -> Result<Vec3F64, SimError> {
        let (segment, u) = locate(t, self.t_start, self.knot_dt, self.control.len())?;
        Ok(self.weighted(segment, basis_d2(u)) * (1.0 / (self.knot_dt * self.knot_dt)))
    }
}

/// A cumulative uniform cubic B-spline on SO(3), used for the rotational
/// trajectory.
///
/// Interpolating rotations cannot use the plain weighted sum that works on R³ —
/// a weighted average of rotation matrices is not a rotation. The cumulative
/// form composes incremental rotations on the manifold instead:
///
/// ```text
/// R(u) = R_i · Exp(λ₁(u)·d₁) · Exp(λ₂(u)·d₂) · Exp(λ₃(u)·d₃)
/// d_j  = Log(R_{i+j-1}⁻¹ · R_{i+j})
/// ```
///
/// so every evaluation is a product of rotations and therefore a rotation by
/// construction (Kim, Kim & Shin, *A General Construction Scheme for Unit
/// Quaternion Curves with Simple High Order Derivatives*, SIGGRAPH '95).
#[derive(Debug, Clone)]
pub struct So3Spline {
    control: Vec<SO3F64>,
    t_start: f64,
    knot_dt: f64,
}

impl So3Spline {
    /// Builds a rotation spline from control rotations on a uniform knot
    /// vector.
    pub fn new(control: Vec<SO3F64>, t_start: f64, knot_dt: f64) -> Result<Self, SimError> {
        if control.len() < SPLINE_ORDER {
            return Err(SimError::TooFewControlPoints {
                got: control.len(),
                need: SPLINE_ORDER,
            });
        }
        if knot_dt <= 0.0 || !knot_dt.is_finite() {
            return Err(SimError::InvalidKnotSpacing(knot_dt));
        }
        Ok(Self {
            control,
            t_start,
            knot_dt,
        })
    }

    /// First time at which the spline is defined.
    pub fn t_start(&self) -> f64 {
        self.t_start
    }

    /// Last time at which the spline is defined.
    pub fn t_end(&self) -> f64 {
        self.t_start + (self.control.len() - (SPLINE_ORDER - 1)) as f64 * self.knot_dt
    }

    /// The three incremental tangent vectors `d_j` for a segment.
    fn deltas(&self, segment: usize) -> [Vec3F64; 3] {
        let mut d = [Vec3F64::ZERO; 3];
        for (j, slot) in d.iter_mut().enumerate() {
            let prev = &self.control[segment + j];
            let next = &self.control[segment + j + 1];
            *slot = (prev.inverse() * *next).log();
        }
        d
    }

    /// Rotation at time `t`, as body-to-world.
    pub fn rotation(&self, t: f64) -> Result<SO3F64, SimError> {
        let (segment, u) = locate(t, self.t_start, self.knot_dt, self.control.len())?;
        let d = self.deltas(segment);
        let lambda = cumulative_basis(u);

        let mut r = self.control[segment];
        for j in 0..3 {
            r *= SO3F64::exp(d[j] * lambda[j]);
        }
        Ok(r)
    }

    /// Body-frame angular velocity `ω` at time `t`, defined by
    /// `Ṙ = R · hat(ω)`.
    ///
    /// # Derivation
    ///
    /// Write `R = A₀·A₁·A₂·A₃` with `A₀ = R_i` constant and
    /// `A_j = Exp(λ_j(u)·d_j)`. Because `d_j` is fixed, the family `λ ↦
    /// Exp(λ·d_j)` is a one-parameter subgroup, so `Ȧ_j = A_j · hat(λ̇_j·d_j)`
    /// exactly — no left/right Jacobian correction is needed. Differentiating
    /// the product and left-multiplying by `R⁻¹`:
    ///
    /// ```text
    /// R⁻¹Ṙ = Σ_j (A_{j+1}···A₃)⁻¹ · hat(λ̇_j·d_j) · (A_{j+1}···A₃)
    ///      = Σ_j hat( (A_{j+1}···A₃)⁻¹ · (λ̇_j·d_j) )
    /// ```
    ///
    /// using the identity `Rᵀ·hat(v)·R = hat(Rᵀ·v)`. Hence
    /// `ω = Σ_j (A_{j+1}···A₃)⁻¹ · (λ̇_j·d_j)`, accumulated below from `j = 3`
    /// downwards so the trailing product is built incrementally.
    ///
    /// This is exact, not a finite-difference approximation — which is the
    /// property that makes the generated gyro measurements trustworthy as
    /// ground truth. `angular_velocity_matches_finite_difference` pins it.
    pub fn angular_velocity(&self, t: f64) -> Result<Vec3F64, SimError> {
        let (segment, u) = locate(t, self.t_start, self.knot_dt, self.control.len())?;
        let d = self.deltas(segment);
        let lambda = cumulative_basis(u);
        let lambda_dot = cumulative_basis_d1(u);
        let du_dt = 1.0 / self.knot_dt;

        let mut omega = Vec3F64::ZERO;
        // `trailing` holds (A_{j+1}···A₃) as j descends.
        let mut trailing = SO3F64::exp(Vec3F64::ZERO);
        for j in (0..3).rev() {
            omega += trailing.inverse() * (d[j] * (lambda_dot[j] * du_dt));
            trailing = SO3F64::exp(d[j] * lambda[j]) * trailing;
        }
        Ok(omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Control points tracing a helix — curved in all three axes so no
    /// derivative component is trivially zero.
    fn helix_positions(n: usize) -> Vec<Vec3F64> {
        (0..n)
            .map(|i| {
                let a = i as f64 * 0.4;
                Vec3F64::new(2.0 * a.cos(), 1.5 * a.sin(), 0.3 * a)
            })
            .collect()
    }

    /// Control rotations with a non-constant, non-planar angular rate.
    fn twisting_rotations(n: usize) -> Vec<SO3F64> {
        (0..n)
            .map(|i| {
                let a = i as f64 * 0.3;
                SO3F64::exp(Vec3F64::new(0.4 * a.sin(), 0.25 * a, 0.3 * a.cos()))
            })
            .collect()
    }

    #[test]
    fn basis_is_a_partition_of_unity() {
        // If the weights did not sum to 1 the spline would not be
        // translation-invariant, which every other property depends on.
        for k in 0..=20 {
            let u = k as f64 / 20.0;
            let sum: f64 = basis(u).iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "basis sums to {sum} at u={u}");
            // Derivative weights must sum to zero for the same reason.
            let d1: f64 = basis_d1(u).iter().sum();
            let d2: f64 = basis_d2(u).iter().sum();
            assert!(d1.abs() < 1e-14, "d1 sums to {d1} at u={u}");
            assert!(d2.abs() < 1e-14, "d2 sums to {d2} at u={u}");
        }
    }

    #[test]
    fn cumulative_basis_matches_summed_basis() {
        // The cumulative form must be the running tail-sum of the plain basis,
        // or the rotation spline is not the same curve family as the
        // translation spline.
        for k in 0..=20 {
            let u = k as f64 / 20.0;
            let b = basis(u);
            let lambda = cumulative_basis(u);
            for j in 0..3 {
                let expected: f64 = b[(j + 1)..].iter().sum();
                assert!(
                    (lambda[j] - expected).abs() < 1e-14,
                    "lambda[{j}] = {} != {expected} at u={u}",
                    lambda[j]
                );
            }
        }
    }

    #[test]
    fn cumulative_basis_d1_matches_finite_difference() {
        let h = 1e-6;
        for k in 1..20 {
            let u = k as f64 / 20.0;
            let plus = cumulative_basis(u + h);
            let minus = cumulative_basis(u - h);
            let analytic = cumulative_basis_d1(u);
            for j in 0..3 {
                let numeric = (plus[j] - minus[j]) / (2.0 * h);
                assert!(
                    (analytic[j] - numeric).abs() < 1e-8,
                    "lambda_dot[{j}] analytic {} vs numeric {numeric} at u={u}",
                    analytic[j]
                );
            }
        }
    }

    #[test]
    fn spline_is_continuous_across_segment_boundaries() {
        let spline = RSpline3::new(helix_positions(10), 0.0, 0.5).unwrap();
        // A cubic B-spline is C² everywhere, so evaluating either side of an
        // interior knot must agree in value, velocity and acceleration.
        for seg in 1..6 {
            let t = seg as f64 * 0.5;
            let eps = 1e-9;
            let before = spline.position(t - eps).unwrap();
            let after = spline.position(t + eps).unwrap();
            assert!((before - after).length() < 1e-8, "position jump at t={t}");

            let v_before = spline.velocity(t - eps).unwrap();
            let v_after = spline.velocity(t + eps).unwrap();
            assert!(
                (v_before - v_after).length() < 1e-6,
                "velocity jump at t={t}"
            );

            let a_before = spline.acceleration(t - eps).unwrap();
            let a_after = spline.acceleration(t + eps).unwrap();
            assert!((a_before - a_after).length() < 1e-4, "accel jump at t={t}");
        }
    }

    /// The gate for the whole simulator: if the analytic derivatives are wrong,
    /// every generated IMU measurement is wrong in a way that looks like an
    /// estimator bug.
    #[test]
    fn velocity_and_acceleration_match_finite_difference() {
        let spline = RSpline3::new(helix_positions(12), 0.0, 0.5).unwrap();
        let h = 1e-5;

        let mut samples = 0;
        let mut t = spline.t_start() + 0.1;
        while t < spline.t_end() - 0.1 {
            let p_plus = spline.position(t + h).unwrap();
            let p_minus = spline.position(t - h).unwrap();
            let p_mid = spline.position(t).unwrap();

            let v_numeric = (p_plus - p_minus) * (1.0 / (2.0 * h));
            let v_analytic = spline.velocity(t).unwrap();
            assert!(
                (v_numeric - v_analytic).length() < 1e-6,
                "velocity mismatch at t={t}: {v_analytic:?} vs {v_numeric:?}"
            );

            let a_numeric = (p_plus - p_mid * 2.0 + p_minus) * (1.0 / (h * h));
            let a_analytic = spline.acceleration(t).unwrap();
            // Second-order central differences lose ~half the mantissa, so the
            // tolerance here is looser than the velocity one by necessity.
            assert!(
                (a_numeric - a_analytic).length() < 1e-3,
                "acceleration mismatch at t={t}: {a_analytic:?} vs {a_numeric:?}"
            );

            samples += 1;
            t += 0.13;
        }
        assert!(samples > 10, "test swept too few samples");
    }

    /// The rotational counterpart, and the subtler of the two: the closed form
    /// in `angular_velocity` is the piece most likely to be silently wrong.
    #[test]
    fn angular_velocity_matches_finite_difference() {
        let spline = So3Spline::new(twisting_rotations(12), 0.0, 0.5).unwrap();
        let h = 1e-6;

        let mut samples = 0;
        let mut t = spline.t_start() + 0.1;
        while t < spline.t_end() - 0.1 {
            let r_minus = spline.rotation(t - h).unwrap();
            let r_plus = spline.rotation(t + h).unwrap();

            // Body-frame rate: Log(R(t-h)⁻¹ · R(t+h)) / 2h.
            let numeric = (r_minus.inverse() * r_plus).log() * (1.0 / (2.0 * h));
            let analytic = spline.angular_velocity(t).unwrap();

            assert!(
                (numeric - analytic).length() < 1e-6,
                "angular velocity mismatch at t={t}: {analytic:?} vs {numeric:?}"
            );

            samples += 1;
            t += 0.13;
        }
        assert!(samples > 10, "test swept too few samples");
    }

    #[test]
    fn rotation_stays_on_the_manifold() {
        let spline = So3Spline::new(twisting_rotations(10), 0.0, 0.5).unwrap();
        let mut t = spline.t_start();
        while t <= spline.t_end() {
            let m = spline.rotation(t).unwrap().matrix();
            let det = m.determinant();
            assert!((det - 1.0).abs() < 1e-12, "determinant {det} at t={t}");
            t += 0.1;
        }
    }

    #[test]
    fn out_of_range_time_is_rejected() {
        let spline = RSpline3::new(helix_positions(8), 0.0, 0.5).unwrap();
        assert!(spline.position(-0.1).is_err());
        assert!(spline.position(spline.t_end() + 0.1).is_err());
        // Both endpoints are inclusive and must evaluate.
        assert!(spline.position(spline.t_start()).is_ok());
        assert!(spline.position(spline.t_end()).is_ok());
    }

    #[test]
    fn too_few_control_points_is_rejected() {
        assert!(RSpline3::new(helix_positions(3), 0.0, 0.5).is_err());
        assert!(RSpline3::new(helix_positions(4), 0.0, 0.5).is_ok());
    }
}
