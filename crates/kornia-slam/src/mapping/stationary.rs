//! Stationary factor for visual-inertial BA.
//!
//! When a robot is parked, its keyframes need no IMU preintegration to be
//! linked: the camera has not moved, nothing moved, and the IMU biases only
//! wandered by their random walk. A [`StationaryFactor`] says exactly that
//! between two keyframes `i` and `j`, over a span of any length:
//!
//! ```text
//!     r_p  = Log(T_i · T_j⁻¹)                 zero relative pose  (6)
//!     r_vi = v_i,  r_vj = v_j                 zero velocity       (3 + 3)
//!     r_bg = b_g,j − b_g,i                    gyro bias walk      (3)
//!     r_ba = b_a,j − b_a,i                    accel bias walk     (3)
//!     r_ω  = ω̄ − (b_g,i + b_g,j) / 2         mean gyro rate      (3, optional)
//! ```
//!
//! with `T = T_cw` perturbed as `T ← T · Exp(ξ)`. The pose residual is the
//! wheel factor's with identity extrinsics and an identity delta, so its
//! Jacobian is `J_l⁻¹(r) · Ad(T_i)` for `i` and the negation for `j`
//! ([`super::wheel_odom`]).
//!
//! The mean-gyro row keeps the gyro bias observable while parked: at rest the
//! raw gyro reads bias plus white noise, so the mean of `T` seconds of samples
//! has variance `σ_g² / T`, and the difference between the bias's time
//! average and its endpoint average is a Brownian bridge with variance
//! `σ_bg² · span / 12`. There is deliberately no accelerometer row: at rest
//! accelerometer bias and tilt cannot be told apart.
//!
//! Unlike an IMU edge, a stationary factor has no span limit, and two of them
//! through a removed keyframe compose into one ([`StationaryFactor::compose`]),
//! so keyframe culling can merge a parked stretch indefinitely. The caller
//! holds the factors (keyed by `Frame::idx`, like
//! [`super::wheel_odom::WheelOdomFactor`]) and decides which intervals were
//! stationary.

use crate::mapping::robust::huber_weight_and_cost;
use crate::mapping::wheel_odom::{self, Mat6};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, ImuCalib};

/// The mean raw gyro rate measured while the rig was still.
///
/// `#[non_exhaustive]`: build it with [`Self::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StationaryGyro {
    /// Mean raw gyro rate, body frame, rad/s.
    pub mean: Vec3F64,
    /// Seconds of samples behind `mean`.
    pub covered_s: f64,
}

impl StationaryGyro {
    /// The mean rate `mean` (rad/s, body frame) over `covered_s` seconds.
    pub fn new(mean: Vec3F64, covered_s: f64) -> Self {
        Self { mean, covered_s }
    }
}

/// A stationary constraint between keyframes `from_idx` and `to_idx`.
///
/// `#[non_exhaustive]`: build it with [`Self::new`] (and
/// [`Self::with_gyro`]), then set the public fields.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StationaryFactor {
    /// Earlier keyframe.
    pub from_idx: usize,
    /// Later keyframe.
    pub to_idx: usize,
    /// Seconds between the two keyframes; scales the bias random walk.
    pub span_s: f64,
    /// Mean gyro rate over the still span, if measured.
    pub gyro: Option<StationaryGyro>,
    /// IMU noise densities: `gyro_noise` weighs the mean-gyro row,
    /// `gyro_bias_noise` and `accel_bias_noise` the bias random walk.
    pub calib: ImuCalib,
    /// Sigma of the zero relative camera translation, metres.
    pub translation_sigma_m: f64,
    /// Sigma of the zero relative camera rotation, radians.
    pub rotation_sigma_rad: f64,
    /// Sigma of the zero world velocity at each end, m/s.
    pub velocity_sigma_m_s: f64,
    /// Huber threshold on the 12-row pose and velocity chi²; the bias rows
    /// stay quadratic, as in the IMU edge. `None` keeps the factor quadratic.
    pub huber_chi2: Option<f64>,
}

impl StationaryFactor {
    /// 2 mm: the camera of a parked robot does not move.
    pub const DEFAULT_TRANSLATION_SIGMA_M: f64 = 0.002;
    /// 0.002 rad (0.11°).
    pub const DEFAULT_ROTATION_SIGMA_RAD: f64 = 0.002;
    /// 1 cm/s.
    pub const DEFAULT_VELOCITY_SIGMA_M_S: f64 = 0.01;
    /// The 12-DOF chi² at 95 %.
    pub const DEFAULT_HUBER_CHI2: f64 = 21.03;

    /// A factor with the default sigmas and Huber threshold and no gyro mean.
    pub fn new(from_idx: usize, to_idx: usize, span_s: f64, calib: ImuCalib) -> Self {
        Self {
            from_idx,
            to_idx,
            span_s,
            gyro: None,
            calib,
            translation_sigma_m: Self::DEFAULT_TRANSLATION_SIGMA_M,
            rotation_sigma_rad: Self::DEFAULT_ROTATION_SIGMA_RAD,
            velocity_sigma_m_s: Self::DEFAULT_VELOCITY_SIGMA_M_S,
            huber_chi2: Some(Self::DEFAULT_HUBER_CHI2),
        }
    }

    /// `self` with the mean gyro rate of the still span.
    pub fn with_gyro(self, gyro: StationaryGyro) -> Self {
        Self {
            gyro: Some(gyro),
            ..self
        }
    }

    /// Whether the factor can enter a solve: two distinct keyframes, a
    /// positive finite span, sigmas and bias densities, a positive Huber
    /// threshold if any, and a finite gyro mean over a positive time (with a
    /// positive gyro noise density). [`super::bundle_adjustment::run_local_inertial_ba`]
    /// skips a factor that is not; `visual_inertial_bundle_adjust` refuses it.
    pub fn is_valid(&self) -> bool {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let gyro_ok = self.gyro.is_none_or(|g| {
            positive(g.covered_s)
                && positive(self.calib.gyro_noise)
                && [g.mean.x, g.mean.y, g.mean.z].iter().all(|v| v.is_finite())
        });
        self.from_idx != self.to_idx
            && positive(self.span_s)
            && positive(self.translation_sigma_m)
            && positive(self.rotation_sigma_rad)
            && positive(self.velocity_sigma_m_s)
            && positive(self.calib.gyro_bias_noise)
            && positive(self.calib.accel_bias_noise)
            && self.huber_chi2.is_none_or(positive)
            && gyro_ok
    }

    /// The factor from `self.from_idx` to `next.to_idx`, replacing the two
    /// when the keyframe between them is removed; `None` unless `next`
    /// starts where `self` ends and the result links two keyframes.
    ///
    /// Spans add; the gyro means combine weighted by their time; the sigmas
    /// are the looser of the two (they are not inflated: the relative pose of
    /// two keyframes of one stop is zero however many were culled between
    /// them, and only the bias walk grows, through `span_s`); the Huber
    /// threshold is the tighter; the calibration is `self`'s.
    pub fn compose(&self, next: &Self) -> Option<Self> {
        if self.to_idx != next.from_idx || self.from_idx == next.to_idx {
            return None;
        }
        let gyro = match (self.gyro, next.gyro) {
            (Some(a), Some(b)) => {
                let covered_s = a.covered_s + b.covered_s;
                Some(StationaryGyro {
                    mean: (a.mean * a.covered_s + b.mean * b.covered_s) / covered_s,
                    covered_s,
                })
            }
            (a, b) => a.or(b),
        };
        let huber_chi2 = match (self.huber_chi2, next.huber_chi2) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Some(Self {
            from_idx: self.from_idx,
            to_idx: next.to_idx,
            span_s: self.span_s + next.span_s,
            gyro,
            calib: self.calib,
            translation_sigma_m: self.translation_sigma_m.max(next.translation_sigma_m),
            rotation_sigma_rad: self.rotation_sigma_rad.max(next.rotation_sigma_rad),
            velocity_sigma_m_s: self.velocity_sigma_m_s.max(next.velocity_sigma_m_s),
            huber_chi2,
        })
    }

    /// The (diagonal) information of each residual block. Meaningful only
    /// for a factor that [`Self::is_valid`].
    pub(crate) fn information(&self) -> StationaryInfo {
        let t = 1.0 / (self.translation_sigma_m * self.translation_sigma_m);
        let r = 1.0 / (self.rotation_sigma_rad * self.rotation_sigma_rad);
        let mut pose = [[0.0; 6]; 6];
        for k in 0..3 {
            pose[k][k] = t;
            pose[k + 3][k + 3] = r;
        }
        let walk = |density: f64| 1.0 / (density * density * self.span_s);
        let gyro_bias_walk_var =
            self.calib.gyro_bias_noise * self.calib.gyro_bias_noise * self.span_s;
        StationaryInfo {
            pose,
            velocity: 1.0 / (self.velocity_sigma_m_s * self.velocity_sigma_m_s),
            gyro_bias: walk(self.calib.gyro_bias_noise),
            accel_bias: walk(self.calib.accel_bias_noise),
            gyro_mean: self.gyro.map(|g| {
                let white = self.calib.gyro_noise * self.calib.gyro_noise / g.covered_s;
                (g.mean, 1.0 / (white + gyro_bias_walk_var / 12.0))
            }),
        }
    }
}

/// Per-block information of a [`StationaryFactor`]; every block is diagonal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StationaryInfo {
    pub(crate) pose: Mat6,
    /// Per axis, for each of `v_i` and `v_j`.
    pub(crate) velocity: f64,
    pub(crate) gyro_bias: f64,
    pub(crate) accel_bias: f64,
    /// The measured mean rate and its per-axis information.
    pub(crate) gyro_mean: Option<(Vec3F64, f64)>,
}

/// The residual blocks of one stationary factor at a state, and the pose
/// Jacobian wrt keyframe `i` (keyframe `j`'s is its negation). The linear
/// blocks have constant Jacobians: identity on `v_i` / `v_j`, `∓I` on the
/// bias walk, `−I/2` on each gyro bias for the mean-gyro row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StationaryResidual {
    pub(crate) pose: [f64; 6],
    pub(crate) pose_jacobian_i: Mat6,
    pub(crate) velocity_i: Vec3F64,
    pub(crate) velocity_j: Vec3F64,
    pub(crate) gyro_bias: Vec3F64,
    pub(crate) accel_bias: Vec3F64,
    pub(crate) gyro_mean: Option<Vec3F64>,
    /// Robust weight of the pose and velocity rows.
    pub(crate) weight: f64,
    /// `½ρ(chi²_nav) + ½ chi²_bias`, `ρ` the Huber objective of `weight`
    /// ([`super::robust`]).
    pub(crate) cost: f64,
}

/// One keyframe's state as the factor reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StationaryState<'a> {
    pub(crate) pose_cw: &'a Pose3d,
    pub(crate) velocity: Vec3F64,
    pub(crate) bias: ImuBias,
}

pub(crate) fn residual(
    factor: &StationaryFactor,
    info: &StationaryInfo,
    i: StationaryState<'_>,
    j: StationaryState<'_>,
) -> StationaryResidual {
    let (pose, pose_jacobian_i) = wheel_odom::relative_pose_residual(
        i.pose_cw,
        j.pose_cw,
        &Pose3d::IDENTITY,
        &Pose3d::IDENTITY,
        &Pose3d::IDENTITY,
    );
    let gyro_bias = j.bias.gyro - i.bias.gyro;
    let accel_bias = j.bias.accel - i.bias.accel;
    let gyro_mean = info
        .gyro_mean
        .map(|(mean, _)| mean - (i.bias.gyro + j.bias.gyro) * 0.5);

    let chi2_nav = wheel_odom::quadratic_form(&info.pose, &pose)
        + info.velocity * (i.velocity.dot(i.velocity) + j.velocity.dot(j.velocity));
    let (weight, nav_cost) = huber_weight_and_cost(chi2_nav, factor.huber_chi2);
    let chi2_bias = info.gyro_bias * gyro_bias.dot(gyro_bias)
        + info.accel_bias * accel_bias.dot(accel_bias)
        + info
            .gyro_mean
            .zip(gyro_mean)
            .map_or(0.0, |((_, g), r)| g * r.dot(r));
    StationaryResidual {
        pose,
        pose_jacobian_i,
        velocity_i: i.velocity,
        velocity_j: j.velocity,
        gyro_bias,
        accel_bias,
        gyro_mean,
        weight,
        cost: nav_cost + 0.5 * chi2_bias,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2e-3,
            gyro_bias_noise: 2e-5,
            accel_bias_noise: 3e-3,
        }
    }

    fn with_gyro(mut f: StationaryFactor, mean: [f64; 3], covered_s: f64) -> StationaryFactor {
        f.gyro = Some(StationaryGyro {
            mean: Vec3F64::new(mean[0], mean[1], mean[2]),
            covered_s,
        });
        f
    }

    #[test]
    fn compose_adds_spans_and_weights_the_gyro_mean_by_time() {
        let a = with_gyro(
            StationaryFactor::new(3, 7, 40.0, calib()),
            [0.003, 0.0, -0.001],
            30.0,
        );
        let b = with_gyro(
            StationaryFactor::new(7, 9, 20.0, calib()),
            [0.006, 0.0, 0.002],
            10.0,
        );
        let c = a.compose(&b).unwrap();
        assert_eq!((c.from_idx, c.to_idx), (3, 9));
        assert_eq!(c.span_s, 60.0);
        let g = c.gyro.unwrap();
        assert_eq!(g.covered_s, 40.0);
        assert!((g.mean.x - 0.00375).abs() < 1e-15, "{:?}", g.mean);
        assert!((g.mean.z + 0.00025).abs() < 1e-15, "{:?}", g.mean);
        assert_eq!(
            c.translation_sigma_m,
            StationaryFactor::DEFAULT_TRANSLATION_SIGMA_M
        );
        assert!(c.is_valid());

        // One side without a gyro mean keeps the other's.
        let bare = StationaryFactor::new(7, 9, 20.0, calib());
        assert_eq!(a.compose(&bare).unwrap().gyro, a.gyro);
        assert_eq!(
            StationaryFactor::new(1, 3, 1.0, calib())
                .compose(&a)
                .unwrap()
                .gyro,
            a.gyro
        );

        // Looser sigma, tighter Huber.
        let loose = StationaryFactor {
            translation_sigma_m: 0.01,
            huber_chi2: Some(5.0),
            ..bare
        };
        let c = a.compose(&loose).unwrap();
        assert_eq!(c.translation_sigma_m, 0.01);
        assert_eq!(c.huber_chi2, Some(5.0));
        let quadratic = StationaryFactor {
            huber_chi2: None,
            ..bare
        };
        assert_eq!(
            StationaryFactor {
                huber_chi2: None,
                ..a
            }
            .compose(&quadratic)
            .unwrap()
            .huber_chi2,
            None
        );
    }

    #[test]
    fn compose_refuses_mismatched_ids_and_self_loops() {
        let a = StationaryFactor::new(3, 7, 1.0, calib());
        assert!(
            a.compose(&StationaryFactor::new(8, 9, 1.0, calib()))
                .is_none()
        );
        assert!(
            a.compose(&StationaryFactor::new(7, 3, 1.0, calib()))
                .is_none()
        );
    }

    #[test]
    fn validity() {
        let good = with_gyro(StationaryFactor::new(0, 1, 5.0, calib()), [0.0; 3], 4.0);
        assert!(good.is_valid());
        assert!(!StationaryFactor { to_idx: 0, ..good }.is_valid());
        for span_s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(!StationaryFactor { span_s, ..good }.is_valid(), "{span_s}");
        }
        assert!(
            !StationaryFactor {
                translation_sigma_m: 0.0,
                ..good
            }
            .is_valid()
        );
        assert!(
            !StationaryFactor {
                rotation_sigma_rad: f64::NAN,
                ..good
            }
            .is_valid()
        );
        assert!(
            !StationaryFactor {
                velocity_sigma_m_s: -1.0,
                ..good
            }
            .is_valid()
        );
        assert!(
            !StationaryFactor {
                huber_chi2: Some(0.0),
                ..good
            }
            .is_valid()
        );
        assert!(
            StationaryFactor {
                huber_chi2: None,
                ..good
            }
            .is_valid()
        );
        let mut c = calib();
        c.gyro_bias_noise = 0.0;
        assert!(!StationaryFactor { calib: c, ..good }.is_valid());
        let mut c = calib();
        c.gyro_noise = 0.0;
        assert!(!StationaryFactor { calib: c, ..good }.is_valid());
        // Without a gyro mean the white-noise density is not read.
        assert!(
            StationaryFactor {
                calib: c,
                gyro: None,
                ..good
            }
            .is_valid()
        );
        assert!(!with_gyro(good, [f64::NAN, 0.0, 0.0], 4.0).is_valid());
        assert!(!with_gyro(good, [0.0; 3], 0.0).is_valid());
    }

    #[test]
    fn information_follows_the_noise_model() {
        let f = with_gyro(StationaryFactor::new(0, 1, 120.0, calib()), [0.0; 3], 100.0);
        let info = f.information();
        assert_eq!(info.pose[0][0], 1.0 / (0.002 * 0.002));
        assert_eq!(info.pose[5][5], 1.0 / (0.002 * 0.002));
        assert_eq!(info.pose[0][1], 0.0);
        assert_eq!(info.velocity, 1.0 / (0.01 * 0.01));
        assert!((info.gyro_bias - 1.0 / (2e-5 * 2e-5 * 120.0)).abs() < 1e-6 * info.gyro_bias);
        assert!((info.accel_bias - 1.0 / (3e-3 * 3e-3 * 120.0)).abs() < 1e-9 * info.accel_bias);
        let var = 1.7e-4 * 1.7e-4 / 100.0 + 2e-5 * 2e-5 * 120.0 / 12.0;
        let (_, g) = info.gyro_mean.unwrap();
        assert!((g - 1.0 / var).abs() < 1e-9 / var);
    }
}
