//! Stationary inertial initialization: gravity and gyro bias from a window in
//! which the body did not move.
//!
//! [`ImuInitializer::try_initialize`] needs motion. It estimates gravity from
//! the difference between what the IMU integrated and what the visual
//! trajectory did, so a parked platform gives it nothing to work with and its
//! readiness gate (`min_motion`) refuses outright. A STEREO map does not need
//! motion for the one thing motion buys a monocular map, metric scale, so a
//! parked stereo-inertial rig can initialize from the IMU alone:
//!
//! * at rest the accelerometer measures the specific force `f = R_bw (a - g)
//!   + b_a` with `a = 0`, so `g_w = -R_wb (mean(f) - b_a)`;
//! * at rest the gyroscope measures `w = 0 + b_g`, so `b_g = mean(w)`;
//! * velocity is zero.
//!
//! The accelerometer bias is only HALF observable at rest. Its component
//! across gravity trades one for one against a tilt of the gravity direction,
//! so it is held at the caller's prior, and the refinements and
//! visual-inertial BA that follow estimate it once the platform moves. Its
//! component ALONG gravity is fixed by the known magnitude of g: a still IMU
//! reading `|f| = 9.66` has 0.15 m/s^2 of vertical bias (or scale error, which
//! at rest is the same thing), and is corrected by exactly that along the
//! measured direction. Leaving it unmodelled would put it into the velocities
//! of the first visual-inertial BA, ~0.2 m/s per 1.4 s edge on a parked rig,
//! while the tight bias random walk moves it into the bias only slowly.
//! This is the textbook zero-velocity
//! initialization; see e.g. Qin et al., VINS-Mono (IEEE T-RO 2018), and the
//! static initializer of Geneva et al., OpenVINS (ICRA 2020).
//!
//! The result is an ordinary [`ImuInitResult`], applied with
//! [`ImuInitializer::apply_initialization`] exactly as a motion solve is: the
//! map is rotated so gravity lands on world +y, and the window's keyframes get
//! their (zero) velocity and the bias.
//!
//! Deciding that the window WAS stationary is the caller's job: only the
//! caller knows what other sensors (wheels, the visual track) can vouch for
//! it. [`ImuWindowStats`] carries the spread figures such a detector needs.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, ImuMeasurement};

use super::{ImuInitResult, ImuInitializer, window_is_mono};
use crate::mapping::map::{Keyframe, Map};

/// Mean and spread of a window of raw IMU samples, in the IMU (body) frame.
///
/// Built with [`ImuWindowStats::from_samples`]; `#[non_exhaustive]` so a
/// figure can be added later without breaking callers.
///
/// ```compile_fail
/// // `#[non_exhaustive]`: built by `from_samples` only, so a figure can be
/// // added without a semver break.
/// use kornia_algebra::Vec3F64;
/// use kornia_slam::inertial::ImuWindowStats;
/// let _ = ImuWindowStats {
///     samples: 2,
///     duration_sec: 1.0,
///     mean_accel: Vec3F64::ZERO,
///     mean_gyro: Vec3F64::ZERO,
///     accel_rms_dev: 0.0,
///     gyro_rms_dev: 0.0,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImuWindowStats {
    /// Samples in the window.
    pub samples: usize,
    /// Last sample stamp minus first, seconds.
    pub duration_sec: f64,
    /// Mean specific force, m/s^2.
    pub mean_accel: Vec3F64,
    /// Mean angular rate, rad/s.
    pub mean_gyro: Vec3F64,
    /// RMS deviation of the accelerometer vector from its window mean,
    /// `sqrt(mean |f_k - mean(f)|^2)`, m/s^2: the norm of the per-axis
    /// standard deviations. A vector spread rather than the spread of `|f|`,
    /// so that a body tilting in place, which rotates gravity without changing
    /// its magnitude, still reads as motion.
    pub accel_rms_dev: f64,
    /// The same spread for the gyroscope, rad/s.
    pub gyro_rms_dev: f64,
}

impl ImuWindowStats {
    /// Statistics over `samples`, or `None` for fewer than two samples or any
    /// non-finite reading. Order does not matter except for `duration_sec`,
    /// which is taken between the earliest and the latest stamp.
    pub fn from_samples<'a>(samples: impl IntoIterator<Item = &'a ImuMeasurement>) -> Option<Self> {
        let mut n = 0usize;
        let (mut sum_a, mut sum_w) = (Vec3F64::ZERO, Vec3F64::ZERO);
        let (mut sum_aa, mut sum_ww) = (0.0f64, 0.0f64);
        let (mut t_min, mut t_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for m in samples {
            if !(finite(m.accel) && finite(m.gyro) && m.timestamp.is_finite()) {
                return None;
            }
            n += 1;
            sum_a += m.accel;
            sum_w += m.gyro;
            sum_aa += m.accel.dot(m.accel);
            sum_ww += m.gyro.dot(m.gyro);
            t_min = t_min.min(m.timestamp);
            t_max = t_max.max(m.timestamp);
        }
        if n < 2 {
            return None;
        }
        let inv = 1.0 / n as f64;
        let (mean_a, mean_w) = (sum_a * inv, sum_w * inv);
        // E|x - mean|^2 = E|x|^2 - |mean|^2; clamped against round-off going
        // negative on a perfectly constant signal.
        let rms = |sum_sq: f64, mean: Vec3F64| (sum_sq * inv - mean.dot(mean)).max(0.0).sqrt();
        Some(Self {
            samples: n,
            duration_sec: t_max - t_min,
            mean_accel: mean_a,
            mean_gyro: mean_w,
            accel_rms_dev: rms(sum_aa, mean_a),
            gyro_rms_dev: rms(sum_ww, mean_w),
        })
    }
}

fn finite(v: Vec3F64) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Why a stationary initialization was refused.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StationaryInitError {
    /// The window's first keyframe carries no stereo data. Without stereo the
    /// map has no metric scale, and a stationary window cannot supply one.
    #[error("monocular window: a stationary initialization needs a metric (stereo) map")]
    MonocularWindow,
    /// No keyframe at or after the window start: nothing to attach the state to.
    #[error("no keyframe at or after frame {start_idx}")]
    NoKeyframe {
        /// The requested window start.
        start_idx: usize,
    },
    /// The bias-corrected mean specific force is not finite, or its magnitude
    /// is further than the tolerance from standard gravity: the platform was
    /// accelerating, or the accelerometer (or its bias prior) is wrong.
    #[error(
        "implausible gravity: |mean accel - accel bias| = {norm:.4} m/s^2, \
         more than {tolerance} from standard gravity"
    )]
    ImplausibleGravity {
        /// Magnitude of the bias-corrected mean specific force.
        norm: f64,
        /// The tolerance it was checked against.
        tolerance: f64,
    },
}

impl ImuInitializer {
    /// Initializes from a window the CALLER has established was stationary.
    ///
    /// `stats` are the window's raw IMU statistics; `start_idx` is the
    /// `Frame::idx` of the first keyframe of that window. Gravity is taken in
    /// the body frame of the NEWEST keyframe at or after `start_idx` and
    /// mapped into the world through that keyframe's pose and `imu_t_bc`
    /// (`X_body = T_BC * X_cam`), so the window's keyframes must not have
    /// rotated relative to one another; that is part of being stationary.
    ///
    /// Returns, for [`ImuInitializer::apply_initialization`] with the same
    /// `start_idx`: scale 1 (the map is already metric), gravity in the
    /// CURRENT world frame at standard magnitude, a zero velocity for every
    /// keyframe at or after `start_idx`, gyro bias `mean(w)` and accelerometer
    /// bias `prior_bias.accel` corrected ALONG the measured specific force by
    /// `|mean(f) - prior_bias.accel| - g`: the across-gravity part is
    /// unobservable at rest and stays at the prior, the along-gravity part is
    /// what the known magnitude of g pins (see the module docs). The corrected
    /// specific force then has magnitude g exactly, so gravity needs no
    /// rescale.
    ///
    /// `max_gravity_norm_error` bounds `| |mean(f) - prior_bias.accel| - g |`
    /// in m/s^2: the largest vertical bias (or scale error) believed.
    ///
    /// # Errors
    ///
    /// See [`StationaryInitError`]. Nothing is written either way; this only
    /// computes.
    pub fn initialize_stationary(
        &self,
        map: &Map,
        imu_t_bc: Pose3d,
        prior_bias: ImuBias,
        start_idx: usize,
        stats: &ImuWindowStats,
        max_gravity_norm_error: f64,
    ) -> Result<ImuInitResult, StationaryInitError> {
        let mut keyframes: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        keyframes.sort_by_key(|kf| kf.frame.idx);
        let Some(newest) = keyframes.last() else {
            return Err(StationaryInitError::NoKeyframe { start_idx });
        };
        if window_is_mono(&keyframes) {
            return Err(StationaryInitError::MonocularWindow);
        }

        let f_body = stats.mean_accel - prior_bias.accel;
        let norm = f_body.length();
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN norm must be refused too
        if !((norm - GRAVITY_MAGNITUDE).abs() <= max_gravity_norm_error) {
            return Err(StationaryInitError::ImplausibleGravity {
                norm,
                tolerance: max_gravity_norm_error,
            });
        }

        // R_wb = R_wc * R_cb, with R_wc the camera-in-world rotation and
        // R_cb = R_bc^T. The same composition `try_initialize` uses.
        let r_wc = newest.frame.pose_world_to_cam.inverse().rotation;
        let r_cb = imu_t_bc.inverse().rotation;
        let r_wb = r_wc * r_cb;
        // The along-gravity bias: whatever |f| has over g, along f.
        let accel_bias = prior_bias.accel + f_body * ((norm - GRAVITY_MAGNITUDE) / norm);
        let gravity_world = -(r_wb * (stats.mean_accel - accel_bias));

        Ok(ImuInitResult {
            scale: 1.0,
            gravity_world,
            velocities_world: vec![Vec3F64::ZERO; keyframes.len()],
            bias: ImuBias {
                gyro: stats.mean_gyro,
                accel: accel_bias,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::mapping::map::{ImuFactor, MapInsertion};
    use kornia_algebra::{Mat3F64, SO3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::imu::{ImuCalib, PreintegratedImu};

    const RATE_HZ: f64 = 200.0;

    /// The robot's measured gravity read in the camera frame, ~23 deg off the
    /// camera's -y: a tilted IMU frame with nothing axis-aligned about it.
    fn tilted_r_wb() -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.40, -0.25, 0.9)).matrix()
    }

    fn extrinsic() -> Pose3d {
        // Asymmetric under transpose, so a convention slip changes the answer.
        Pose3d::from_rt(
            SO3F64::exp(Vec3F64::new(0.1, 1.2, -0.3)).matrix(),
            Vec3F64::new(0.01, -0.02, 0.03),
        )
    }

    fn bias() -> ImuBias {
        ImuBias {
            gyro: Vec3F64::new(0.0031, -0.0012, 0.0024),
            accel: Vec3F64::new(0.04, -0.02, 0.03),
        }
    }

    fn gravity_true() -> Vec3F64 {
        Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)
    }

    /// Deterministic zero-mean jitter: a sum of incommensurate sines.
    fn jitter(k: usize, phase: f64) -> Vec3F64 {
        let t = k as f64;
        Vec3F64::new(
            (1.7 * t + phase).sin(),
            (2.9 * t + 2.0 * phase).sin(),
            (4.3 * t + 3.0 * phase).sin(),
        )
    }

    /// Stationary samples for a body at attitude `r_wb` in a world whose
    /// gravity is `g_w`, with `bias` and white-ish noise of the given size.
    fn still_samples(
        t0: f64,
        n: usize,
        r_wb: Mat3F64,
        g_w: Vec3F64,
        bias: ImuBias,
        noise: (f64, f64),
    ) -> Vec<ImuMeasurement> {
        (0..n)
            .map(|k| ImuMeasurement {
                timestamp: t0 + k as f64 / RATE_HZ,
                gyro: bias.gyro + jitter(k, 0.3) * noise.1,
                accel: r_wb.transpose() * (-g_w) + bias.accel + jitter(k, 1.1) * noise.0,
            })
            .collect()
    }

    fn stereo_frame(idx: usize, pose_world_to_cam: Pose3d) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: vec![[10.0, 10.0]],
                orientations: vec![0.0],
                descriptors: vec![[0u8; 32]],
                octaves: vec![0],
            },
            pose_world_to_cam,
            image_size: ImageSize {
                width: 640,
                height: 400,
            },
            keypoint_colors: vec![[0; 3]],
            u_right: vec![5.0],
            depth: vec![2.0], // => is_stereo()
            keypoints_undist: vec![],
        }
    }

    fn initializer() -> ImuInitializer {
        ImuInitializer::new(super::super::ImuInitConfig {
            min_keyframes: 10,
            min_time_sec: 1.0,
            min_motion: 0.05,
        })
    }

    /// A stationary STEREO map in an arbitrary, non-gravity-aligned visual
    /// world: the camera sits at `centre` with the attitude the tilted IMU and
    /// the extrinsic imply, and every keyframe pair carries a still IMU edge.
    fn still_map(n_kf: usize, kf_dt: f64, r_vis: Mat3F64, centre: Vec3F64) -> Map {
        // True body attitude in a gravity-aligned world, and the visual world
        // rotated from it by `r_vis`: the gravity the solve must find is
        // `r_vis * g_true`.
        let r_wb_true = tilted_r_wb();
        let r_bc = extrinsic().rotation;
        let r_wc_vis = r_vis * r_wb_true * r_bc;
        let cam_in_world = Pose3d::new(r_wc_vis, centre);
        let calib = ImuCalib {
            gyro_noise: 1.2e-4,
            accel_noise: 1.9e-3,
            gyro_bias_noise: 2e-6,
            accel_bias_noise: 4e-5,
        };
        let mut map = Map::new();
        for k in 0..n_kf {
            map.insert_keyframe(Keyframe::from_frame(stereo_frame(
                k,
                cam_in_world.inverse(),
            )))
            .unwrap();
            if k > 0 {
                let t0 = (k - 1) as f64 * kf_dt;
                let t1 = k as f64 * kf_dt;
                let n = (kf_dt * RATE_HZ).round() as usize + 1;
                let raw = still_samples(t0, n, r_wb_true, gravity_true(), bias(), (0.0, 0.0));
                let pre =
                    PreintegratedImu::from_measurements(ImuBias::default(), calib, &raw, t0, t1);
                map.apply_insertion(MapInsertion {
                    imu_factors: vec![ImuFactor {
                        prev_kf_idx: k - 1,
                        curr_kf_idx: k,
                        preintegrated: pre,
                        raw_samples: raw,
                        t0,
                        t1,
                        bias_walk_extra_s: 0.0,
                    }],
                    ..Default::default()
                })
                .unwrap();
            }
        }
        map
    }

    fn angle_deg(a: Vec3F64, b: Vec3F64) -> f64 {
        a.normalize()
            .dot(b.normalize())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    /// The window statistics: the means are the bias-plus-gravity the samples
    /// carry, and the spread is the vector RMS deviation, so noise of known
    /// per-axis amplitude reads back at its known size.
    ///
    /// Mutation: computing the accel spread from the gyro sums turned this red.
    #[test]
    fn window_stats_report_means_and_vector_spread() {
        let still = still_samples(0.0, 400, tilted_r_wb(), gravity_true(), bias(), (0.0, 0.0));
        let s = ImuWindowStats::from_samples(&still).unwrap();
        assert_eq!(s.samples, 400);
        assert!((s.duration_sec - 399.0 / RATE_HZ).abs() < 1e-12);
        assert!((s.mean_gyro - bias().gyro).length() < 1e-12);
        let want_a = tilted_r_wb().transpose() * (-gravity_true()) + bias().accel;
        assert!((s.mean_accel - want_a).length() < 1e-9);
        assert!(s.accel_rms_dev < 1e-6 && s.gyro_rms_dev < 1e-9);

        // A tilt that swings gravity's DIRECTION at constant magnitude: |f| is
        // constant, so a spread of the norm would read zero. The vector spread
        // must not.
        let swinging: Vec<ImuMeasurement> = (0..400)
            .map(|k| {
                let tilt = SO3F64::exp(Vec3F64::new(0.05 * (k as f64 * 0.05).sin(), 0.0, 0.0));
                ImuMeasurement {
                    timestamp: k as f64 / RATE_HZ,
                    gyro: Vec3F64::ZERO,
                    accel: tilt.matrix() * Vec3F64::new(0.0, -GRAVITY_MAGNITUDE, 0.0),
                }
            })
            .collect();
        let s = ImuWindowStats::from_samples(&swinging).unwrap();
        assert!(
            s.accel_rms_dev > 0.2,
            "a tilting body must read as motion: rms={}",
            s.accel_rms_dev
        );

        assert!(ImuWindowStats::from_samples(&still[..1]).is_none());
        let mut bad = still.clone();
        bad[7].gyro.y = f64::NAN;
        assert!(ImuWindowStats::from_samples(&bad).is_none());
    }

    /// Gravity and gyro bias from a still window, with the IMU tilted, the
    /// extrinsic non-trivial, the visual world not gravity-aligned and a
    /// nonzero gyro bias: gravity comes out as the visual world's true down,
    /// the gyro bias as the true one, the accel bias as the prior, velocities
    /// zero and scale 1.
    ///
    /// Mutations this catches (each turned it red): dropping the minus sign on
    /// the gravity; using `R_bc` instead of `R_cb`; skipping the accel-bias
    /// subtraction (a 0.05 m/s^2 bias is 0.3 deg of tilt, over the 0.05 deg
    /// tolerance).
    #[test]
    fn stationary_init_recovers_gravity_and_gyro_bias_in_a_tilted_frame() {
        let r_vis = SO3F64::exp(Vec3F64::new(-0.2, 0.7, 0.35)).matrix();
        let map = still_map(4, 0.6, r_vis, Vec3F64::new(0.3, -0.1, 1.2));
        let samples: Vec<ImuMeasurement> = map
            .imu_factors()
            .iter()
            .flat_map(|f| f.raw_samples.iter().copied())
            .collect();
        let stats = ImuWindowStats::from_samples(&samples).unwrap();
        // The accel bias is unobservable at rest, so the prior must be the TRUE
        // one for the gravity to come out exact.
        let prior = ImuBias {
            gyro: Vec3F64::ZERO,
            accel: bias().accel,
        };

        let init = initializer()
            .initialize_stationary(&map, extrinsic(), prior, 0, &stats, 0.3)
            .expect("a still stereo window initializes");

        let want_g = r_vis * gravity_true();
        assert!(
            angle_deg(init.gravity_world, want_g) < 0.05,
            "gravity {:?} vs {:?}: {} deg",
            init.gravity_world,
            want_g,
            angle_deg(init.gravity_world, want_g)
        );
        assert!((init.gravity_world.length() - GRAVITY_MAGNITUDE).abs() < 1e-9);
        assert!((init.bias.gyro - bias().gyro).length() < 1e-9);
        // |f - prior| is g exactly here, so the along-gravity correction is nil.
        assert!((init.bias.accel - prior.accel).length() < 1e-9);
        assert_eq!(init.scale, 1.0);
        assert_eq!(init.velocities_world, vec![Vec3F64::ZERO; 4]);

        // Windowed: only keyframes from `start_idx` get a velocity.
        let init = initializer()
            .initialize_stationary(&map, extrinsic(), prior, 2, &stats, 0.3)
            .unwrap();
        assert_eq!(init.velocities_world.len(), 2);
    }

    /// The gravity-magnitude sanity check, a monocular window and an empty one
    /// are all refused.
    ///
    /// Mutation: comparing `norm` against the tolerance without subtracting
    /// `GRAVITY_MAGNITUDE` turned this red.
    #[test]
    fn stationary_init_refuses_implausible_gravity_mono_and_empty_windows() {
        let map = still_map(3, 0.6, Mat3F64::IDENTITY, Vec3F64::ZERO);
        let samples: Vec<ImuMeasurement> = map
            .imu_factors()
            .iter()
            .flat_map(|f| f.raw_samples.iter().copied())
            .collect();
        let mut stats = ImuWindowStats::from_samples(&samples).unwrap();
        let init = initializer();
        assert!(
            init.initialize_stationary(&map, extrinsic(), ImuBias::default(), 0, &stats, 0.3)
                .is_ok()
        );

        // A platform accelerating at 0.5 m/s^2 along gravity.
        stats.mean_accel *= (GRAVITY_MAGNITUDE + 0.5) / stats.mean_accel.length();
        assert!(matches!(
            init.initialize_stationary(&map, extrinsic(), ImuBias::default(), 0, &stats, 0.3),
            Err(StationaryInitError::ImplausibleGravity { norm, .. }) if (norm - 10.31).abs() < 0.01
        ));
        stats.mean_accel = Vec3F64::new(f64::NAN, 0.0, 0.0);
        assert!(matches!(
            init.initialize_stationary(&map, extrinsic(), ImuBias::default(), 0, &stats, 0.3),
            Err(StationaryInitError::ImplausibleGravity { .. })
        ));

        assert_eq!(
            init.initialize_stationary(&map, extrinsic(), ImuBias::default(), 9, &stats, 0.3)
                .err(),
            Some(StationaryInitError::NoKeyframe { start_idx: 9 })
        );

        let mut mono = Map::new();
        let mut frame = stereo_frame(0, Pose3d::IDENTITY);
        frame.depth.clear();
        frame.u_right.clear();
        mono.insert_keyframe(Keyframe::from_frame(frame)).unwrap();
        let stats = ImuWindowStats::from_samples(&samples).unwrap();
        assert_eq!(
            init.initialize_stationary(&mono, extrinsic(), ImuBias::default(), 0, &stats, 0.3)
                .err(),
            Some(StationaryInitError::MonocularWindow)
        );
    }

    /// A stationary initialization lands the map where a MOTION solve over the
    /// same still window lands it: `try_initialize` given the identical map
    /// (the visual-inertial optimizer, with its own gravity seeding) and the
    /// stationary estimate are applied to two copies, and the resulting
    /// gravity-aligned worlds, biases and velocities agree.
    ///
    /// This is the cross-check on conventions: two independent derivations of
    /// "which way is down", one through the preintegrated deltas and the
    /// optimizer's residual, one through the raw mean specific force.
    ///
    /// Mutation: negating `gravity_world` in `initialize_stationary` (the
    /// "assumed -Z" class of bug) turned this red at 180 deg; using `r_bc`
    /// for `r_cb` turned it red too.
    #[test]
    fn stationary_init_matches_the_motion_solve_on_the_same_window() {
        let r_vis = SO3F64::exp(Vec3F64::new(0.3, -0.4, 0.2)).matrix();
        let map = still_map(12, 0.25, r_vis, Vec3F64::new(0.2, 0.1, -0.4));
        let samples: Vec<ImuMeasurement> = map
            .imu_factors()
            .iter()
            .flat_map(|f| f.raw_samples.iter().copied())
            .collect();
        let stats = ImuWindowStats::from_samples(&samples).unwrap();
        let prior = ImuBias::default();
        let init = initializer();

        let motion = init
            .try_initialize(&map, Some(extrinsic()), prior, 0, 1e2, 1e5, false)
            .expect("the optimizer converges on a still window too");
        let stationary = init
            .initialize_stationary(&map, extrinsic(), prior, 0, &stats, 0.3)
            .expect("a still window initializes");

        // Raw results: the accel bias the optimizer may move is held near zero
        // by its 1e5 prior, so the two gravity directions are comparable.
        assert!(
            angle_deg(motion.gravity_world, stationary.gravity_world) < 0.5,
            "motion {:?} vs stationary {:?}",
            motion.gravity_world,
            stationary.gravity_world
        );
        assert!((motion.bias.gyro - stationary.bias.gyro).length() < 1e-3);
        for v in &motion.velocities_world {
            assert!(v.length() < 0.02, "motion velocity {v:?}");
        }

        // Applied: same aligned world, same canonical gravity.
        let (mut map_m, mut map_s) = (map.clone(), map.clone());
        let (mut bias_m, mut bias_s) = (prior, prior);
        let (mut g_m, mut g_s) = (Vec3F64::ZERO, Vec3F64::ZERO);
        let resume_m = init
            .apply_initialization(&mut map_m, &mut bias_m, &mut g_m, motion, 0)
            .unwrap()
            .unwrap();
        let resume_s = init
            .apply_initialization(&mut map_s, &mut bias_s, &mut g_s, stationary, 0)
            .unwrap()
            .unwrap();
        assert_eq!(g_m, g_s);
        let rot_deg = |p: &Pose3d, q: &Pose3d| {
            let r = p.rotation * q.rotation.transpose();
            (((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        };
        assert!(
            rot_deg(&resume_m.pose_world_to_cam, &resume_s.pose_world_to_cam) < 0.5,
            "aligned camera attitudes differ by {} deg",
            rot_deg(&resume_m.pose_world_to_cam, &resume_s.pose_world_to_cam)
        );
        assert_eq!(resume_s.velocity_world, Vec3F64::ZERO);
        // Invariant of BOTH: in the aligned world, the IMU's own mean specific
        // force maps onto world -y (up).
        let r_cb = extrinsic().inverse().rotation;
        for resume in [resume_m, resume_s] {
            let r_wb = resume.pose_world_to_cam.inverse().rotation * r_cb;
            let up = r_wb * stats.mean_accel;
            assert!(
                angle_deg(up, Vec3F64::new(0.0, -1.0, 0.0)) < 0.5,
                "specific force in the aligned world: {up:?}"
            );
        }
    }
    /// The along-gravity accelerometer bias IS observable at rest: an IMU at
    /// XLeRobot's measured `|a| = 9.66` with a zero prior recovers the 0.15
    /// m/s^2 along the measured direction, keeps the across-gravity part at the
    /// prior, and its corrected specific force has magnitude g, so gravity is
    /// unchanged in direction and exact in size.
    ///
    /// Mutation: keeping `accel` at the prior (the original behaviour) turned
    /// this red.
    #[test]
    fn stationary_init_recovers_the_along_gravity_accel_bias() {
        let r_vis = SO3F64::exp(Vec3F64::new(0.1, -0.3, 0.2)).matrix();
        let map = still_map(4, 0.6, r_vis, Vec3F64::ZERO);
        // A vertical bias of -0.15 m/s^2 along the specific force: |f| = 9.66.
        let f_true = tilted_r_wb().transpose() * (-gravity_true());
        let along = f_true.normalize() * -0.15;
        let samples: Vec<ImuMeasurement> = still_samples(
            0.0,
            241,
            tilted_r_wb(),
            gravity_true(),
            ImuBias {
                gyro: bias().gyro,
                accel: along,
            },
            (0.0, 0.0),
        );
        let stats = ImuWindowStats::from_samples(&samples).unwrap();
        assert!((stats.mean_accel.length() - 9.66).abs() < 0.001);

        let init = initializer()
            .initialize_stationary(&map, extrinsic(), ImuBias::default(), 0, &stats, 0.3)
            .expect("9.66 is within tolerance");
        assert!(
            (init.bias.accel - along).length() < 1e-9,
            "bias {:?} vs {:?}",
            init.bias.accel,
            along
        );
        assert!((init.gravity_world.length() - GRAVITY_MAGNITUDE).abs() < 1e-9);
        let want_g = r_vis * gravity_true();
        assert!(angle_deg(init.gravity_world, want_g) < 0.05);
        assert!(((stats.mean_accel - init.bias.accel).length() - GRAVITY_MAGNITUDE).abs() < 1e-9);
    }
}
