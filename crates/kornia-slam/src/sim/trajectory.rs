//! Ground-truth trajectories built from a translation spline and a rotation
//! spline.
//!
//! # Frame conventions
//!
//! These follow the rest of kornia-slam and must not be varied casually — a
//! frame-convention mismatch between the simulator and the estimator produces
//! exactly the class of bug the simulator exists to catch, except undetectably.
//!
//! - **World axes are OpenCV-style Y-down**, matching [`ViBaParams::gravity`]'s
//!   default of `[0, 9.81, 0]`.
//! - **Body axes**: `x` right, `y` down, `z` forward.
//! - `R_wb` maps body → world; its columns are the body axes in world
//!   coordinates.
//! - [`Pose3d`] values returned here are **world → camera** (`T_cw`), the
//!   convention `BaObservation` and `ViBaKeyframe` expect.
//! - The camera-to-body extrinsic `T_bc` satisfies `X_body = T_bc · X_cam`,
//!   matching [`ViBaParams::imu_t_bc`].
//!
//! [`ViBaParams::gravity`]: crate::vi_ba_schur::ViBaParams::gravity
//! [`ViBaParams::imu_t_bc`]: crate::vi_ba_schur::ViBaParams::imu_t_bc
//!
//! # Upstreaming
//!
//! **This module splits across two crates when it moves.** Only two methods
//! touch `kornia-3d` at all:
//!
//! - [`Trajectory`], [`TrajectoryState`]'s fields, [`Trajectory::sample_uniform`],
//!   [`Trajectory::arc`] and `look_along` are `kornia-algebra`-only, and travel
//!   with [`super::spline`] to `kornia-algebra`/`kornia-manifold`.
//! - [`TrajectoryState::body_pose`] and [`TrajectoryState::camera_pose`] need
//!   `Pose3d`, and belong in `kornia-3d` alongside [`super::landmarks`].
//!
//! The frame conventions documented above are the part most worth carrying over
//! verbatim: they are what keeps the simulator and the estimators agreeing, and
//! a mismatch introduced during a move would be invisible in exactly the way
//! this whole module exists to prevent.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

use super::SimError;
use super::spline::{RSpline3, So3Spline};

/// Gravity in the simulator's default world frame: OpenCV Y-down, so "down" is
/// `+Y`. Matches [`crate::vi_ba_schur::ViBaParams`]'s default.
pub const DEFAULT_GRAVITY: Vec3F64 = Vec3F64 {
    x: 0.0,
    y: 9.81,
    z: 0.0,
};

/// The full kinematic state of the body at one instant, all in ground truth.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryState {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Body position in the world frame.
    pub position: Vec3F64,
    /// Body velocity in the world frame (m/s).
    pub velocity: Vec3F64,
    /// Body acceleration in the world frame (m/s²), *excluding* gravity.
    pub acceleration: Vec3F64,
    /// Body-to-world rotation.
    pub rotation: SO3F64,
    /// Angular velocity in the **body** frame (rad/s) — what a gyroscope reads.
    pub angular_velocity: Vec3F64,
}

impl TrajectoryState {
    /// World → body pose (`T_bw`).
    pub fn body_pose(&self) -> Pose3d {
        let r_bw = Mat3F64(*self.rotation.matrix().transpose());
        Pose3d::new(r_bw, r_bw * -self.position)
    }

    /// World → camera pose (`T_cw`) for a camera mounted at `t_bc`.
    ///
    /// `None` means the camera frame coincides with the body frame.
    pub fn camera_pose(&self, t_bc: Option<&Pose3d>) -> Pose3d {
        // T_wb, then T_wc = T_wb · T_bc, then invert to T_cw.
        let t_wb = Pose3d::new(self.rotation.matrix(), self.position);
        match t_bc {
            Some(bc) => t_wb.compose(bc).inverse(),
            None => t_wb.inverse(),
        }
    }
}

/// A ground-truth trajectory: independent splines for translation and rotation
/// over a shared time domain.
#[derive(Debug, Clone)]
pub struct Trajectory {
    translation: RSpline3,
    rotation: So3Spline,
}

impl Trajectory {
    /// Combines a translation and a rotation spline.
    ///
    /// The two must share a time domain — otherwise there are times at which
    /// the trajectory has a position but no orientation, and sampling would
    /// fail unpredictably partway through a run rather than at construction.
    pub fn new(translation: RSpline3, rotation: So3Spline) -> Result<Self, SimError> {
        let (ts, rs) = (translation.t_start(), rotation.t_start());
        let (te, re) = (translation.t_end(), rotation.t_end());
        if (ts - rs).abs() > 1e-9 || (te - re).abs() > 1e-9 {
            return Err(SimError::MismatchedSplineDomains {
                translation: (ts, te),
                rotation: (rs, re),
            });
        }
        Ok(Self {
            translation,
            rotation,
        })
    }

    /// First time at which the trajectory is defined.
    pub fn t_start(&self) -> f64 {
        self.translation.t_start()
    }

    /// Last time at which the trajectory is defined.
    pub fn t_end(&self) -> f64 {
        self.translation.t_end()
    }

    /// Samples the complete kinematic state at time `t`.
    pub fn state(&self, t: f64) -> Result<TrajectoryState, SimError> {
        Ok(TrajectoryState {
            timestamp: t,
            position: self.translation.position(t)?,
            velocity: self.translation.velocity(t)?,
            acceleration: self.translation.acceleration(t)?,
            rotation: self.rotation.rotation(t)?,
            angular_velocity: self.rotation.angular_velocity(t)?,
        })
    }

    /// Samples `count` states at evenly spaced times spanning the domain.
    ///
    /// The endpoints are inset by one interval so that finite-difference checks
    /// and IMU intervals around each sample stay inside the valid domain.
    pub fn sample_uniform(&self, count: usize) -> Result<Vec<TrajectoryState>, SimError> {
        if count < 2 {
            return Err(SimError::InvalidConfig(format!(
                "need at least 2 samples, got {count}"
            )));
        }
        let margin = (self.t_end() - self.t_start()) * 0.02;
        let (lo, hi) = (self.t_start() + margin, self.t_end() - margin);
        let step = (hi - lo) / (count - 1) as f64;
        (0..count)
            .map(|i| self.state(lo + i as f64 * step))
            .collect()
    }
}

/// Builds `R_wb` for a body travelling along `forward` while keeping `world_down`
/// as close to its own down axis as the geometry allows.
///
/// Returns `None` when `forward` is degenerate or parallel to `world_down`, in
/// which case "level" is undefined.
fn look_along(forward: Vec3F64, world_down: Vec3F64) -> Option<SO3F64> {
    let z = forward.normalize();
    if !z.length().is_finite() || z.length() < 0.5 {
        return None;
    }
    // right = down × forward, then re-orthogonalize down as forward × right so
    // the three axes form an exact orthonormal basis even when the inputs are
    // not perpendicular.
    let x = world_down.cross(z);
    if x.length() < 1e-9 {
        return None;
    }
    let x = x.normalize();
    let y = z.cross(x).normalize();
    // Columns are the body axes expressed in world coordinates.
    Some(SO3F64::from_matrix(&Mat3F64::from_cols(x, y, z)))
}

/// Shape parameters for [`Trajectory::arc`].
#[derive(Debug, Clone)]
pub struct ArcConfig {
    /// Radius of the arc in metres.
    pub radius: f64,
    /// Total descent along the world down axis over the whole arc (metres).
    /// Zero keeps the path planar.
    pub climb: f64,
    /// Total angle swept, in radians.
    ///
    /// This is the co-visibility knob, and the most consequential setting here.
    /// A full revolution turns the camera through 360°, so landmarks visible at
    /// the start are long gone by the end and few points are seen by enough
    /// keyframes to be constrained. A modest sweep behaves like a local BA
    /// window: large baseline, strong parallax, high co-visibility.
    pub sweep: f64,
    /// Number of spline control points.
    pub n_control: usize,
    /// Time of the first knot.
    pub t_start: f64,
    /// Knot spacing in seconds.
    pub knot_dt: f64,
}

impl Default for ArcConfig {
    fn default() -> Self {
        Self {
            radius: 4.0,
            climb: 0.0,
            sweep: std::f64::consts::FRAC_PI_3,
            n_control: 12,
            t_start: 0.0,
            knot_dt: 0.25,
        }
    }
}

/// Presets used to construct trajectories.
///
/// These are deliberately minimal — enough to exercise the estimators, not the
/// full scenario library. Degenerate-regime scenarios (`pure_rotation`,
/// `stationary`, `constant_velocity`, …) come later, once each can be tied to a
/// specific observed failure.
impl Trajectory {
    /// A forward-facing camera sweeping along a circular arc.
    ///
    /// The body always looks along its direction of travel, which is the
    /// SLAM-typical configuration: continuously changing viewpoint with real
    /// parallax, and a non-trivial angular rate so the gyro path is exercised.
    ///
    /// The resulting curve is the *spline through* these control points, not a
    /// mathematically exact circle — and that is fine, because the spline is the
    /// ground truth. [`ArcConfig::radius`] and friends shape the path; they are
    /// not promises about it.
    pub fn arc(config: &ArcConfig) -> Result<Self, SimError> {
        if config.radius <= 0.0 {
            return Err(SimError::InvalidConfig(format!(
                "arc radius must be positive, got {}",
                config.radius
            )));
        }
        if config.n_control < 4 {
            return Err(SimError::TooFewControlPoints {
                got: config.n_control,
                need: 4,
            });
        }

        let angle_step = config.sweep / (config.n_control - 1) as f64;

        let mut positions = Vec::with_capacity(config.n_control);
        let mut rotations = Vec::with_capacity(config.n_control);
        for i in 0..config.n_control {
            let a = i as f64 * angle_step;
            let fraction = if config.sweep.abs() > 0.0 {
                a / config.sweep
            } else {
                0.0
            };
            // Arc in the world XZ plane; Y (down) carries the climb.
            positions.push(Vec3F64::new(
                config.radius * a.cos(),
                config.climb * fraction,
                config.radius * a.sin(),
            ));
            // Tangent to the arc: the derivative of the position expression
            // above with respect to `a`. The climb term must be included —
            // omitting it points the camera horizontally while the body
            // actually travels at an angle, which shows up as a constant
            // several-degree mismatch between the forward axis and the velocity.
            let tangent = Vec3F64::new(
                -config.radius * a.sin(),
                config.climb / config.sweep,
                config.radius * a.cos(),
            );
            let r = look_along(tangent, DEFAULT_GRAVITY)
                .ok_or_else(|| SimError::InvalidConfig("degenerate arc tangent".to_string()))?;
            rotations.push(r);
        }

        Self::new(
            RSpline3::new(positions, config.t_start, config.knot_dt)?,
            So3Spline::new(rotations, config.t_start, config.knot_dt)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_state_is_self_consistent() {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 3.0,
            climb: 0.5,
            n_control: 16,
            t_start: 0.0,
            knot_dt: 0.25,
            ..Default::default()
        })
        .unwrap();
        let states = traj.sample_uniform(20).unwrap();
        assert_eq!(states.len(), 20);

        for s in &states {
            // Forward-facing means the body z axis aligns with the velocity.
            let forward = s.rotation * Vec3F64::new(0.0, 0.0, 1.0);
            let speed = s.velocity.length();
            assert!(speed > 1e-3, "trajectory should be moving");
            let alignment = forward.dot(s.velocity * (1.0 / speed));
            assert!(
                alignment > 0.99,
                "camera not looking along travel: alignment {alignment}"
            );
        }
    }

    #[test]
    fn camera_pose_round_trips_through_the_body_frame() {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 2.0,
            climb: 0.0,
            n_control: 12,
            t_start: 0.0,
            knot_dt: 0.3,
            ..Default::default()
        })
        .unwrap();
        let s = traj.state(1.0).unwrap();

        // With no extrinsic, camera pose is exactly the body pose.
        let t_cw = s.camera_pose(None);
        let t_bw = s.body_pose();
        assert!((t_cw.translation - t_bw.translation).length() < 1e-12);

        // T_cw maps the body's own world position to the camera origin.
        let origin = t_cw.transform_point(&s.position);
        assert!(origin.length() < 1e-9, "body origin not at camera origin");
    }

    #[test]
    fn extrinsic_offsets_the_camera_by_the_expected_amount() {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 2.0,
            climb: 0.0,
            n_control: 12,
            t_start: 0.0,
            knot_dt: 0.3,
            ..Default::default()
        })
        .unwrap();
        let s = traj.state(1.0).unwrap();

        // Camera sits 10 cm along the body x axis: X_body = T_bc · X_cam.
        let t_bc = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.1, 0.0, 0.0));
        let t_cw = s.camera_pose(Some(&t_bc));

        // The camera centre in world coords is p_wb + R_wb · t_bc.
        let expected = s.position + s.rotation * Vec3F64::new(0.1, 0.0, 0.0);
        let actual = t_cw.inverse().translation;
        assert!(
            (actual - expected).length() < 1e-12,
            "camera centre {actual:?} != {expected:?}"
        );
    }

    #[test]
    fn mismatched_spline_domains_are_rejected() {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 2.0,
            climb: 0.0,
            n_control: 12,
            t_start: 0.0,
            knot_dt: 0.3,
            ..Default::default()
        })
        .unwrap();
        let positions: Vec<Vec3F64> = (0..12).map(|i| Vec3F64::new(i as f64, 0.0, 0.0)).collect();
        // Same control count, different knot spacing → different end time.
        let shifted = RSpline3::new(positions, 0.0, 0.5).unwrap();
        assert!(Trajectory::new(shifted, traj.rotation.clone()).is_err());
    }
}
