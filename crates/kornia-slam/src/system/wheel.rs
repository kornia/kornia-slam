//! Wheel-odometry ingest for [`super::SlamSystem`]: the stamp-indexed reading ring, the
//! keyframe-interval integration that turns two stamps into a [`WheelOdomFactor`], the per-frame
//! motion prior, and the base-to-camera mount ring a tilting head needs.
//!
//! Ported from cu-kornia-vio's `wheel` module (the XLeRobot tracker), so a SLAM system and that
//! tracker integrate the same readings the same way.
#![allow(
    dead_code,
    reason = "the parked gate, the motion prior and the coverage report are the next SlamSystem steps"
)]

use std::collections::VecDeque;

use crate::mapping::wheel_odom::{WheelOdomFactor, WheelOdomMeasurement};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

/// A wheel-odometry configuration or mount that cannot be used.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WheelError {
    /// The configured or published camera mount rotation is not a rotation.
    #[error(
        "camera mount rotation is not a rigid rotation: max|R^T R - I| = \
         {orthonormality_error:e}, det = {determinant} (want 0 and +1)"
    )]
    MountNotRigid {
        orthonormality_error: f64,
        determinant: f64,
    },
    /// A noise figure or interval is not a usable number. A zero sigma is an infinite
    /// information matrix: the wheels would then outweigh every visual constraint.
    #[error("wheel odometry parameter `{field}` must be finite and > 0, got {value}")]
    ParamInvalid { field: &'static str, value: f64 },
}

/// `Some((max|R^T R - I|, det R))` when `r` is not a rotation, `None` when it is.
fn rotation_residual(r: &Mat3F64) -> Option<(f64, f64)> {
    let residual = r.transpose() * *r - Mat3F64::IDENTITY;
    let orthonormality_error = residual
        .to_cols_array()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    let determinant = r.determinant();
    let rigid = orthonormality_error.is_finite()
        && orthonormality_error <= 1e-6
        && (determinant - 1.0).abs() <= 1e-6;
    (!rigid).then_some((orthonormality_error, determinant))
}

/// Readings held. 8192 is ~80 s at 100 Hz: the ring must reach back to the last keyframe's stamp,
/// and a parked robot inserts none, so this is what bounds an idle stretch before that interval's
/// start is evicted and its edge refused.
pub const DEFAULT_WHEEL_BUFFER_CAPACITY: usize = 8192;

/// Mount samples held. Only a mount that MOVED is stored (see [`MOUNT_SAME_RAD`]), so a still
/// head costs no slots however long the robot parks, and the last keyframe's mount survives an
/// idle stretch the way its reading does. 256 changes is ~8 s of a head tilting at 30 Hz.
const MOUNT_CAPACITY: usize = 256;

/// A mount within this of the newest stored one is the same mount and is not stored. Far below
/// what a head-joint encoder resolves; it only keeps repeated identical samples out of the ring.
const MOUNT_SAME_RAD: f64 = 1e-4;
/// See [`MOUNT_SAME_RAD`].
const MOUNT_SAME_M: f64 = 1e-4;

/// Pending keyframe intervals kept waiting for readings. Past this the stream has stopped, not
/// lagged, and the oldest is refused as expired.
const MAX_PENDING_EDGES: usize = 8;

/// Consecutive frames outside the readings' span before the tracker says so, and how often it
/// repeats after that: ~2 s, then ~1 min, at the head's 15 Hz.
const UNCOVERED_WARN_AFTER: u64 = 30;
const UNCOVERED_WARN_EVERY: u64 = 900;

/// Integrated factors kept for local BA. The solve frees the newest three keyframes and drops a
/// factor with both ends fixed, so only the last few can act; 16 is slack for a late edge.
const MAX_FACTORS: usize = 16;

/// Default variance floors, so a parked robot's zero-length interval is a stiff but finite
/// constraint: 1 mm and 1 mrad, the values the factor shipped with. They are ten times the
/// encoders' quantisation (one tick of a 4096-step servo on a 0.060 m wheel is 0.092 mm, and on
/// one side of a 0.508 m track 0.18 mrad of heading, XLeRobot), deliberately: a floor is also the
/// whole uncertainty of a parked interval, and what the base-to-camera mount gets wrong (a head
/// that yaws on its mast) lands there too. Lowering them is a measured decision, per robot, via
/// [`WheelOdomConfig::min_translation_sigma_m`] and [`WheelOdomConfig::min_heading_sigma_rad`].
pub const DEFAULT_MIN_TRANSLATION_SIGMA_M: f64 = 1e-3;
/// The heading counterpart of [`DEFAULT_MIN_TRANSLATION_SIGMA_M`], radians.
pub const DEFAULT_MIN_HEADING_SIGMA_RAD: f64 = 1e-3;

/// The frame-stamp uncertainty the timing term assumes by default, seconds. See
/// [`WheelOdomConfig::clock_sigma_s`].
pub const DEFAULT_CLOCK_SIGMA_S: f64 = 0.005;

/// A reading-to-reading step faster than this is not motion but a discontinuity in the odometry
/// (an integrator restarted or re-zeroed while the stream kept flowing), m/s. Several times
/// anything a wheeled base drives.
const MAX_PLAUSIBLE_SPEED_M_S: f64 = 2.0;
/// ...or turned faster than this, rad/s.
const MAX_PLAUSIBLE_RATE_RAD_S: f64 = 2.0 * std::f64::consts::PI;
/// Slack on those two for stamp jitter between closely spaced readings: a step under this is
/// never a discontinuity, however close the stamps.
const STEP_SLACK_M: f64 = 0.02;
/// See [`STEP_SLACK_M`], radians.
const STEP_SLACK_RAD: f64 = 0.05;

/// One wheel-odometry reading: the cumulative base pose in the odometry frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelSample {
    /// Capture time, on the same clock as the frames' `tov`.
    pub stamp_ns: u64,
    /// Base position, metres.
    pub x: f64,
    /// Base position, metres.
    pub y: f64,
    /// Base heading, radians, wrapped or not.
    pub theta: f64,
}

impl WheelSample {
    fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.theta.is_finite()
    }

    /// Exactly the odometry origin: what a freshly started integrator reports first.
    fn at_origin(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.theta == 0.0
    }
}

/// Whether the step from reading `s0` to the next reading `s1` is a discontinuity in the
/// odometry rather than motion: the stream jumped, past [`STEP_SLACK_M`] / [`STEP_SLACK_RAD`],
/// back to exactly the origin (the body, or its odometry, restarted: 2026-10-10 a deploy restart
/// read as a 5.78 deg wheel turn of a parked robot), or it moved faster than a base can. Integrating across one would hand local BA and
/// the motion check the jump as if the base had driven it.
fn is_discontinuity(s0: &WheelSample, s1: &WheelSample) -> bool {
    let dt = s1.stamp_ns.saturating_sub(s0.stamp_ns) as f64 * 1e-9;
    let step_m = (s1.x - s0.x).hypot(s1.y - s0.y);
    let step_rad = wrap(s1.theta - s0.theta).abs();
    // A step within the slack is harmless whatever caused it, and lets a base driving through
    // the origin pass.
    let beyond_slack = step_m > STEP_SLACK_M || step_rad > STEP_SLACK_RAD;
    (s1.at_origin() && beyond_slack)
        || step_m > MAX_PLAUSIBLE_SPEED_M_S * dt + STEP_SLACK_M
        || step_rad > MAX_PLAUSIBLE_RATE_RAD_S * dt + STEP_SLACK_RAD
}

/// The wheel path's configuration. Build it with [`WheelOdomConfig::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct WheelOdomConfig {
    /// Base-to-camera transform (`X_cam = t_cb * X_base`), until a `CameraMount` arrives.
    pub t_cb: Pose3d,
    /// Translation noise, m per sqrt(m) of path: an interval of `d` metres gets variance
    /// `translation_noise^2 * d` on x and y.
    pub translation_noise: f64,
    /// Heading noise, rad per sqrt(rad) of turning: `rotation_noise^2 * |turned|` on theta.
    pub rotation_noise: f64,
    /// Heading noise per distance driven, rad per sqrt(m): `heading_noise_per_m^2 * d` on theta.
    /// A straight run drifts in heading too (uneven floor, tyre deformation), and through the
    /// compounding that heading error becomes lateral error growing as `d^3`. Zero by default.
    pub heading_noise_per_m: f64,
    /// Sigma of a SYSTEMATIC heading error per metre, rad/m: a left/right wheel radius mismatch
    /// turns every metre by the same unknown amount, so its variance grows as `d^2`, not `d`.
    /// Zero by default.
    pub heading_scale_sigma_per_m: f64,
    /// Out-of-plane translation growth, m per m of path: `z_sigma` of an interval is
    /// `hypot(z_sigma, z_slope * d)`, since a floor's slope and bumps add height with distance.
    /// Roll and pitch do not get one: the relative tilt of two base poses is bounded by the
    /// floor, it does not random-walk. Zero by default.
    pub z_slope: f64,
    /// Sigma of the interval ends' stamps against the readings', seconds. An end is interpolated
    /// at a frame stamp, so a stamp off by `dt` moves it by `v * dt`; zero while parked, it is
    /// what a fast interval's covariance must carry however short the path.
    pub clock_sigma_s: f64,
    /// Floor on the x and y sigma of every interval, metres. See
    /// [`DEFAULT_MIN_TRANSLATION_SIGMA_M`].
    pub min_translation_sigma_m: f64,
    /// Floor on the heading sigma of every interval, radians. See
    /// [`DEFAULT_MIN_HEADING_SIGMA_RAD`].
    pub min_heading_sigma_rad: f64,
    /// Out-of-plane translation sigma, metres: the planar stiffness on z.
    pub z_sigma: f64,
    /// Roll and pitch sigma, radians: the planar stiffness on tilt.
    pub tilt_sigma: f64,
    /// Huber threshold on the factor's 6-DOF chi-square; `None` keeps it quadratic.
    pub huber_chi2: Option<f64>,
    /// Longest gap between two readings an interval may span. A wheel stream that went silent
    /// for longer says nothing about the motion inside the hole; interpolating across it would.
    pub max_gap_s: f64,
    /// Readings held; see [`DEFAULT_WHEEL_BUFFER_CAPACITY`].
    pub buffer_capacity: usize,
    /// Longest gap between keyframes while parked; `None` (the default) keeps kornia-slam's own
    /// policy. kornia-slam forces a keyframe every few frames however still the scene, so a
    /// parked robot filled `max_keyframes` and dropped its map every ~75 s (4 drops in 5 min,
    /// 2026-10-11). Parked keyframes add nothing but that; one per interval keeps the IMU edges
    /// of the visual-inertial BA bounded. See `Tracker::parked_keyframe_suppressed`.
    pub parked_keyframe_interval_s: Option<f64>,
}

impl WheelOdomConfig {
    /// Validates the mount and the noise model. `cam_in_base` is the camera's pose in the base
    /// frame (the URDF transform); the factor wants its inverse, taken here once.
    ///
    /// The out-of-plane sigmas default to kornia-slam's (1 cm, 0.01 rad), the gap to 0.5 s, the
    /// Huber kernel to off, the clock sigma to [`DEFAULT_CLOCK_SIGMA_S`] and the distance terms
    /// (`heading_noise_per_m`, `heading_scale_sigma_per_m`, `z_slope`) to zero; set the fields
    /// afterwards and call [`WheelOdomConfig::validate`].
    pub fn new(
        cam_in_base: Pose3d,
        translation_noise: f64,
        rotation_noise: f64,
    ) -> Result<Self, WheelError> {
        let cfg = Self {
            t_cb: cam_in_base.inverse(),
            translation_noise,
            rotation_noise,
            heading_noise_per_m: 0.0,
            heading_scale_sigma_per_m: 0.0,
            z_slope: 0.0,
            clock_sigma_s: DEFAULT_CLOCK_SIGMA_S,
            min_translation_sigma_m: DEFAULT_MIN_TRANSLATION_SIGMA_M,
            min_heading_sigma_rad: DEFAULT_MIN_HEADING_SIGMA_RAD,
            z_sigma: WheelOdomMeasurement::DEFAULT_Z_SIGMA,
            tilt_sigma: WheelOdomMeasurement::DEFAULT_TILT_SIGMA,
            huber_chi2: None,
            max_gap_s: 0.5,
            buffer_capacity: DEFAULT_WHEEL_BUFFER_CAPACITY,
            parked_keyframe_interval_s: None,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Every check [`WheelOdomConfig::new`] runs, for a config edited after construction.
    pub fn validate(&self) -> Result<(), WheelError> {
        validate_mount(&self.t_cb)?;
        for (field, value) in [
            ("translation_noise", self.translation_noise),
            ("rotation_noise", self.rotation_noise),
            ("z_sigma", self.z_sigma),
            ("tilt_sigma", self.tilt_sigma),
            ("min_translation_sigma_m", self.min_translation_sigma_m),
            ("min_heading_sigma_rad", self.min_heading_sigma_rad),
            ("max_gap_s", self.max_gap_s),
            ("huber_chi2", self.huber_chi2.unwrap_or(1.0)),
            (
                "parked_keyframe_interval_s",
                self.parked_keyframe_interval_s.unwrap_or(1.0),
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(WheelError::ParamInvalid { field, value });
            }
        }
        // Zero turns these terms off, which is their default.
        for (field, value) in [
            ("heading_noise_per_m", self.heading_noise_per_m),
            ("heading_scale_sigma_per_m", self.heading_scale_sigma_per_m),
            ("z_slope", self.z_slope),
            ("clock_sigma_s", self.clock_sigma_s),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(WheelError::ParamInvalid { field, value });
            }
        }
        Ok(())
    }

    pub(crate) fn max_gap_ns(&self) -> u64 {
        (self.max_gap_s * 1e9) as u64
    }

    /// The factor measurement for one integrated interval.
    ///
    /// The planar covariance is the one [`WheelState::delta`] propagated, plus the timing term
    /// `(v0^2 + v1^2) sigma_t^2` on x and y and `(w0^2 + w1^2) sigma_t^2` on theta (`v`, `w` the
    /// speeds at the two ends), moved into the frame kornia-slam's residual lives in, and floored
    /// at the encoders' quantisation so a parked interval stays finite.
    ///
    /// The residual is `Log(dT^-1 T_ij)`, a perturbation on the RIGHT of `dT`: its translation is
    /// in the END base frame, while the propagation runs in the start one. A measured delta off
    /// by `(dt, dtheta)` in the start frame gives the residual `-(R(dtheta)^T dt, dtheta)`, so the
    /// covariance is rotated by the interval's own turn. An isotropic x/y block, all the old model
    /// had, is the same in both frames; the propagated one is not.
    pub(crate) fn measurement(&self, d: &WheelDelta) -> WheelOdomMeasurement {
        let sigma_t2 = self.clock_sigma_s.powi(2);
        let [v0, v1] = d.speed_m_s;
        let [w0, w1] = d.rate_rad_s;
        let mut start = d.covariance;
        start[0][0] += (v0 * v0 + v1 * v1) * sigma_t2;
        start[1][1] += (v0 * v0 + v1 * v1) * sigma_t2;
        start[2][2] += (w0 * w0 + w1 * w1) * sigma_t2;
        let (s, c) = d.delta[2].sin_cos();
        let j = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        // The upper triangle, mirrored: exactly symmetric, which kornia-slam checks, where the
        // full product would be symmetric only to rounding.
        let mut cov = [[0.0; 3]; 3];
        for (r, k) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] {
            let v = (0..3)
                .map(|a| (0..3).map(|b| j[r][a] * start[a][b] * j[k][b]).sum::<f64>())
                .sum();
            cov[r][k] = v;
            cov[k][r] = v;
        }
        // Raising a diagonal adds a positive semi-definite matrix, so a propagated covariance
        // (positive semi-definite by construction) comes out positive definite.
        cov[0][0] = cov[0][0].max(self.min_translation_sigma_m.powi(2));
        cov[1][1] = cov[1][1].max(self.min_translation_sigma_m.powi(2));
        cov[2][2] = cov[2][2].max(self.min_heading_sigma_rad.powi(2));
        WheelOdomMeasurement::new(d.delta, cov)
            .with_out_of_plane_sigmas(self.z_sigma.hypot(self.z_slope * d.path_m), self.tilt_sigma)
    }
}

/// Refuses a mount whose rotation is not a proper rotation, or whose lever arm is not finite.
pub(crate) fn validate_mount(t_cb: &Pose3d) -> Result<(), WheelError> {
    if let Some((orthonormality_error, determinant)) = rotation_residual(&t_cb.rotation) {
        return Err(WheelError::MountNotRigid {
            orthonormality_error,
            determinant,
        });
    }
    for (field, value) in [
        ("cam_in_base_translation.x", t_cb.translation.x),
        ("cam_in_base_translation.y", t_cb.translation.y),
        ("cam_in_base_translation.z", t_cb.translation.z),
    ] {
        if !value.is_finite() {
            return Err(WheelError::ParamInvalid { field, value });
        }
    }
    Ok(())
}

/// Planar base motion over one interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WheelDelta {
    /// `(dx, dy, dtheta)`: base at the end, in the base at the start. `dtheta` is the angle
    /// actually turned, unwrapped.
    pub delta: [f64; 3],
    /// Path length, metres: the sum of the reading-to-reading distances, not `|(dx, dy)|`, so a
    /// there-and-back interval is as uncertain as the distance driven.
    pub path_m: f64,
    /// Angle turned, radians, summed in absolute value for the same reason.
    pub path_rad: f64,
    /// Covariance of `delta` in the START base frame, row-major (m^2, m rad, rad^2): the noise
    /// model compounded reading by reading, so it carries the lateral/heading cross terms a
    /// heading error grows into. Zero for a parked interval; [`WheelOdomConfig::measurement`]
    /// adds the timing term and the floors.
    pub covariance: [[f64; 3]; 3],
    /// Base speed at the start and at the end, m/s, from the readings bracketing each; zero
    /// where an end is held rather than extrapolated.
    pub speed_m_s: [f64; 2],
    /// Yaw rate at the start and at the end, rad/s, likewise.
    pub rate_rad_s: [f64; 2],
}

impl WheelDelta {
    /// The delta as a transform taking base-end coordinates to base-start.
    fn pose(&self) -> Pose3d {
        let [dx, dy, dtheta] = self.delta;
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, dtheta)).matrix(),
            Vec3F64::new(dx, dy, 0.0),
        )
    }
}

/// Why an interval could not be integrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WheelWindowError {
    /// The interval ends at or before it starts.
    NonPositiveInterval,
    /// No reading at or before the start: the stream began later, or the start was evicted.
    NotCovered,
    /// No reading at or after the end YET. The one recoverable refusal: retried later.
    Pending,
    /// Two consecutive readings inside the interval are further apart than `max_gap`.
    Gap,
    /// The odometry jumped inside the interval (see [`is_discontinuity`]): restarted, re-zeroed,
    /// or faster than a base drives. Nothing measured the motion across the jump.
    Discontinuity,
}

/// Wheel-path counters. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WheelStats {
    /// Readings accepted into the ring.
    pub accepted: u64,
    /// Readings refused because their stamp did not advance.
    pub out_of_order: u64,
    /// Readings refused for a non-finite pose.
    pub non_finite: u64,
    /// Readings evicted by the capacity bound before anything read them.
    pub evicted: u64,
    /// Readings the bus queue evicted because the tracker did not drain it in time, cumulative.
    pub queue_evicted: u64,
    /// Mount samples accepted, and refused (out of order or not rigid).
    pub mounts_accepted: u64,
    /// See [`WheelStats::mounts_accepted`].
    pub mounts_refused: u64,
    /// Keyframe intervals integrated into a factor.
    pub edges_added: u64,
    /// Intervals refused: start not covered, a gap, an odometry discontinuity, or a bad interval.
    pub edges_refused: u64,
    /// Of [`WheelStats::edges_refused`], the intervals that spanned an odometry discontinuity
    /// (a restart or re-zero of the odometry).
    pub edges_discontinuous: u64,
    /// Intervals refused because the readings never caught up.
    pub edges_expired: u64,
    /// Frames whose tracking prior or coast came from the wheels.
    pub prior_frames: u64,
    /// Frames on which no reading covered the interval and the prior fell back.
    pub prior_unavailable: u64,
    /// Frames whose stamp lay outside `[oldest reading, newest reading + max_gap]`: the stream is
    /// silent or on another clock, and the wheels are not being used.
    pub frames_uncovered: u64,
    /// Keyframes kornia-slam's policy asked for and the parked gate withheld; see
    /// [`WheelOdomConfig::parked_keyframe_interval_s`]. Counted once per keyframe interval: past its
    /// frame budget the policy asks again on every frame, and those frames are one keyframe.
    pub parked_keyframes_skipped: u64,
}

/// A keyframe interval waiting for its readings.
#[derive(Debug, Clone, Copy)]
struct PendingEdge {
    from_idx: usize,
    from_ns: u64,
    to_idx: usize,
    to_ns: u64,
}

/// Wheel path under which an interval counts as parked, metres: the stationary initialization's
/// net-motion floor, applied to the path.
pub(crate) const PARKED_PATH_M: f64 = 0.005;

/// The heading counterpart of [`PARKED_PATH_M`], radians turned.
pub(crate) const PARKED_PATH_RAD: f64 = 0.01;

/// The tracker's wheel half: readings, mounts, pending intervals and the factors built from them.
pub(crate) struct WheelState {
    pub cfg: WheelOdomConfig,
    ring: VecDeque<WheelSample>,
    mounts: VecDeque<(u64, Pose3d)>,
    /// The newest mount stamp seen, stored or not: the order check for the next one.
    newest_mount_ns: Option<u64>,
    pending: Vec<PendingEdge>,
    factors: Vec<WheelOdomFactor>,
    /// Consecutive frames counted in [`WheelStats::frames_uncovered`].
    uncovered_streak: u64,
    pub stats: WheelStats,
}

impl WheelState {
    pub fn new(cfg: WheelOdomConfig) -> Self {
        Self {
            ring: VecDeque::with_capacity(cfg.buffer_capacity.min(1024)),
            mounts: VecDeque::new(),
            newest_mount_ns: None,
            pending: Vec::new(),
            factors: Vec::new(),
            uncovered_streak: 0,
            stats: WheelStats::default(),
            cfg,
        }
    }

    /// Appends readings, refusing any whose stamp does not advance or whose pose is not finite.
    pub fn push(&mut self, samples: impl IntoIterator<Item = WheelSample>) -> usize {
        let mut accepted = 0;
        for s in samples {
            if !s.is_finite() {
                self.stats.non_finite += 1;
                continue;
            }
            if self.ring.back().is_some_and(|b| s.stamp_ns <= b.stamp_ns) {
                self.stats.out_of_order += 1;
                continue;
            }
            self.ring.push_back(s);
            accepted += 1;
            while self.ring.len() > self.cfg.buffer_capacity {
                self.ring.pop_front();
                self.stats.evicted += 1;
            }
        }
        self.stats.accepted += accepted as u64;
        accepted
    }

    /// Records the base-to-camera transform valid from `stamp_ns`.
    pub fn push_mount(&mut self, stamp_ns: u64, t_cb: Pose3d) -> Result<(), WheelError> {
        if let Err(e) = validate_mount(&t_cb) {
            self.stats.mounts_refused += 1;
            return Err(e);
        }
        if self.newest_mount_ns.is_some_and(|t| stamp_ns <= t) {
            self.stats.mounts_refused += 1;
            return Ok(());
        }
        self.newest_mount_ns = Some(stamp_ns);
        self.stats.mounts_accepted += 1;
        if self
            .mounts
            .back()
            .is_some_and(|(_, last)| same_mount(last, &t_cb))
        {
            return Ok(());
        }
        self.mounts.push_back((stamp_ns, t_cb));
        while self.mounts.len() > MOUNT_CAPACITY {
            self.mounts.pop_front();
        }
        Ok(())
    }

    /// The base-to-camera transform at `stamp_ns`: the newest mount at or before it, the oldest
    /// one for an earlier stamp, and the configured one before any mount arrived.
    pub fn t_cb_at(&self, stamp_ns: u64) -> Pose3d {
        let newer = self.mounts.partition_point(|(t, _)| *t <= stamp_ns);
        match newer {
            0 => self.mounts.front().map_or(self.cfg.t_cb, |(_, p)| *p),
            n => self.mounts[n - 1].1,
        }
    }

    /// Base motion over `[t0_ns, t1_ns]`.
    ///
    /// `hold_ns` lets the end (and the start) run past the newest reading by up to that much,
    /// extrapolated: a prior wants the motion now, while a BA factor (`hold_ns = 0`) must cover
    /// the interval.
    pub fn delta(
        &self,
        t0_ns: u64,
        t1_ns: u64,
        hold_ns: u64,
    ) -> Result<WheelDelta, WheelWindowError> {
        self.delta_within(t0_ns, t1_ns, hold_ns, self.cfg.max_gap_ns())
    }

    /// [`Self::delta`] with no limit on the time between readings: a stretch with no reading is
    /// taken as the straight line between the readings around it. Only for a caller that has
    /// another sensor vouching for that stretch; a discontinuity is still refused.
    pub fn delta_across_gaps(
        &self,
        t0_ns: u64,
        t1_ns: u64,
    ) -> Result<WheelDelta, WheelWindowError> {
        self.delta_within(t0_ns, t1_ns, 0, u64::MAX)
    }

    fn delta_within(
        &self,
        t0_ns: u64,
        t1_ns: u64,
        hold_ns: u64,
        max_gap: u64,
    ) -> Result<WheelDelta, WheelWindowError> {
        if t1_ns <= t0_ns {
            return Err(WheelWindowError::NonPositiveInterval);
        }
        // Last reading at or before the start.
        let first_after = self.ring.partition_point(|s| s.stamp_ns <= t0_ns);
        if first_after == 0 {
            return Err(WheelWindowError::NotCovered);
        }
        let a = first_after - 1;
        // First reading at or after the end, or the held newest one.
        let b = self.ring.partition_point(|s| s.stamp_ns < t1_ns);
        let held = b == self.ring.len();
        if held {
            let newest = self.ring.back().expect("a is a reading");
            if t1_ns - newest.stamp_ns > hold_ns {
                return Err(WheelWindowError::Pending);
            }
        }
        let b = b.min(self.ring.len() - 1);
        for i in a..b {
            let (s0, s1) = (&self.ring[i], &self.ring[i + 1]);
            if s1.stamp_ns - s0.stamp_ns > max_gap {
                return Err(WheelWindowError::Gap);
            }
            if is_discontinuity(s0, s1) {
                return Err(WheelWindowError::Discontinuity);
            }
        }

        let start = if a + 1 == self.ring.len() {
            self.extrapolate(t0_ns)
        } else {
            self.interpolate(a, t0_ns)
        };
        let end = if held {
            self.extrapolate(t1_ns)
        } else {
            self.interpolate(b - 1, t1_ns)
        };
        // Walk start -> readings strictly inside -> end, summing the path, unwrapping theta and
        // compounding the covariance.
        let (mut prev, mut turned, mut path_m, mut path_rad) = (start, 0.0, 0.0, 0.0);
        let (sin, cos) = start.theta.sin_cos();
        let mut cov = Propagation::new(&self.cfg);
        let inside = self
            .ring
            .range(a + 1..)
            .copied()
            .take_while(|s| s.stamp_ns < t1_ns)
            .filter(|s| s.stamp_ns > t0_ns);
        for s in inside.chain(std::iter::once(end)) {
            let dth = wrap(s.theta - prev.theta);
            let (dxw, dyw) = (s.x - prev.x, s.y - prev.y);
            let step = [cos * dxw + sin * dyw, -sin * dxw + cos * dyw];
            cov.step(&self.cfg, step, turned + 0.5 * dth, dth);
            turned += dth;
            path_rad += dth.abs();
            path_m += dxw.hypot(dyw);
            prev = s;
        }
        let (dxw, dyw) = (end.x - start.x, end.y - start.y);
        // The speeds at the ends: the reading pair each end was interpolated in, or the pair an
        // extrapolated end was continued from.
        let last_pair = self.ring.len().checked_sub(2);
        let start_pair = if a + 1 == self.ring.len() {
            last_pair
        } else {
            Some(a)
        };
        let end_pair = if held { last_pair } else { Some(b - 1) };
        let (v0, w0) = start_pair.map_or((0.0, 0.0), |i| self.pair_rates(i));
        let (v1, w1) = end_pair.map_or((0.0, 0.0), |i| self.pair_rates(i));
        Ok(WheelDelta {
            delta: [cos * dxw + sin * dyw, -sin * dxw + cos * dyw, turned],
            path_m,
            path_rad,
            covariance: cov.planar(),
            speed_m_s: [v0, v1],
            rate_rad_s: [w0, w1],
        })
    }

    /// Speed and yaw rate between reading `i` and the next, both non-negative. Zero for a pair
    /// [`WheelState::extrapolate`] would not continue (further apart than `max_gap`, or across a
    /// discontinuity): an end held at a reading does not move with its stamp.
    fn pair_rates(&self, i: usize) -> (f64, f64) {
        let (Some(s0), Some(s1)) = (self.ring.get(i), self.ring.get(i + 1)) else {
            return (0.0, 0.0);
        };
        let dt_ns = s1.stamp_ns - s0.stamp_ns;
        if dt_ns > self.cfg.max_gap_ns() || is_discontinuity(s0, s1) {
            return (0.0, 0.0);
        }
        let dt = dt_ns as f64 * 1e-9;
        (
            (s1.x - s0.x).hypot(s1.y - s0.y) / dt,
            wrap(s1.theta - s0.theta).abs() / dt,
        )
    }

    /// The reading at `t_ns`, linearly interpolated between reading `i` and the next one.
    fn interpolate(&self, i: usize, t_ns: u64) -> WheelSample {
        let s0 = self.ring[i];
        let Some(s1) = self.ring.get(i + 1).copied() else {
            return s0;
        };
        if t_ns <= s0.stamp_ns {
            return s0;
        }
        if t_ns >= s1.stamp_ns {
            return s1;
        }
        let a = (t_ns - s0.stamp_ns) as f64 / (s1.stamp_ns - s0.stamp_ns) as f64;
        WheelSample {
            stamp_ns: t_ns,
            x: s0.x + a * (s1.x - s0.x),
            y: s0.y + a * (s1.y - s0.y),
            theta: s0.theta + a * wrap(s1.theta - s0.theta),
        }
    }

    /// The reading at `t_ns` at or past the newest one, continued at the last pair's velocity. A
    /// pair further apart than `max_gap`, or one across a discontinuity, has no velocity worth
    /// trusting (the stream resumed after a hole, or restarted), so the newest reading is held.
    fn extrapolate(&self, t_ns: u64) -> WheelSample {
        let n = self.ring.len();
        let s1 = self.ring[n - 1];
        if n < 2 || t_ns <= s1.stamp_ns {
            return s1;
        }
        let s0 = self.ring[n - 2];
        let dt = s1.stamp_ns - s0.stamp_ns;
        if dt > self.cfg.max_gap_ns() || is_discontinuity(&s0, &s1) {
            return s1;
        }
        let a = (t_ns - s1.stamp_ns) as f64 / dt as f64;
        WheelSample {
            stamp_ns: t_ns,
            x: s1.x + a * (s1.x - s0.x),
            y: s1.y + a * (s1.y - s0.y),
            theta: s1.theta + a * wrap(s1.theta - s0.theta),
        }
    }

    /// The CAMERA motion the wheels measured over `[t0_ns, t1_ns]`, as the world-to-camera
    /// increment `M` with `T_cw(t1) = M * T_cw(t0)`, or `None` when no reading covers it.
    ///
    /// With `X_cam = T_cb X_base` at each end and the delta `dT` taking base-end to base-start
    /// coordinates, camera-end to camera-start is `T_cb(t0) dT T_cb(t1)^-1`; `M` is its inverse.
    pub fn camera_motion(&mut self, t0_ns: u64, t1_ns: u64) -> Option<Pose3d> {
        match self.delta(t0_ns, t1_ns, self.cfg.max_gap_ns()) {
            Ok(d) => {
                self.stats.prior_frames += 1;
                let c0_c1 = self
                    .t_cb_at(t0_ns)
                    .compose(&d.pose())
                    .compose(&self.t_cb_at(t1_ns).inverse());
                Some(c0_c1.inverse())
            }
            Err(_) => {
                self.stats.prior_unavailable += 1;
                None
            }
        }
    }

    /// Stamp of the newest reading.
    pub fn newest_stamp_ns(&self) -> Option<u64> {
        self.ring.back().map(|s| s.stamp_ns)
    }

    /// Readings held.
    #[cfg(test)]
    pub fn buffered(&self) -> usize {
        self.ring.len()
    }

    /// Whether the base stood still over `[t0_ns, t1_ns]`: the PATH, not the net delta, under the
    /// stationary initialization's wheel floors, so a there-and-back is motion. `false` when the
    /// readings do not cover the interval: nothing then vouches for the base.
    pub fn parked_between(&self, t0_ns: u64, t1_ns: u64) -> bool {
        self.delta(t0_ns, t1_ns, self.cfg.max_gap_ns())
            .is_ok_and(|d| d.path_m <= PARKED_PATH_M && d.path_rad <= PARKED_PATH_RAD)
    }

    /// Checks a frame stamp against the readings' span. Returns `frame - newest reading` in ns
    /// and the frames in the streak when the tracker should warn: a clock offset between the body (wheels) and the head
    /// (frames) otherwise turns the wheels off with nothing but the stop-time counters to say so.
    /// Silent before the first reading.
    pub fn check_coverage(&mut self, stamp_ns: u64) -> Option<(i128, u64)> {
        let (Some(oldest), Some(newest)) = (self.ring.front(), self.ring.back()) else {
            return None;
        };
        let covered = oldest.stamp_ns <= stamp_ns
            && stamp_ns <= newest.stamp_ns.saturating_add(self.cfg.max_gap_ns());
        if covered {
            self.uncovered_streak = 0;
            return None;
        }
        self.stats.frames_uncovered += 1;
        self.uncovered_streak += 1;
        let n = self.uncovered_streak;
        (n >= UNCOVERED_WARN_AFTER
            && (n - UNCOVERED_WARN_AFTER).is_multiple_of(UNCOVERED_WARN_EVERY))
        .then(|| (i128::from(stamp_ns) - i128::from(newest.stamp_ns), n))
    }

    /// Queues the interval a keyframe insertion closed, then integrates every queued interval
    /// whose readings have arrived. Returns the factors integrated by this call, each with its
    /// path length in metres, for the motion consistency check ([`crate::consistency`]), which
    /// may [`drop_factor`](Self::drop_factor) one before any solve sees it.
    pub fn on_keyframe(
        &mut self,
        prev: Option<(usize, u64)>,
        to_idx: usize,
        to_ns: u64,
    ) -> Vec<(WheelOdomFactor, f64)> {
        if let Some((from_idx, from_ns)) = prev {
            self.pending.push(PendingEdge {
                from_idx,
                from_ns,
                to_idx,
                to_ns,
            });
        }
        let mut still = Vec::new();
        let mut added = Vec::new();
        for e in std::mem::take(&mut self.pending) {
            match self.delta(e.from_ns, e.to_ns, 0) {
                Ok(d) => {
                    self.stats.edges_added += 1;
                    let factor = WheelOdomFactor::new(
                        e.from_idx,
                        e.to_idx,
                        self.cfg.measurement(&d),
                        self.t_cb_at(e.from_ns),
                        self.t_cb_at(e.to_ns),
                        self.cfg.huber_chi2,
                    );
                    self.factors.push(factor);
                    added.push((factor, d.path_m));
                }
                Err(WheelWindowError::Pending) => still.push(e),
                Err(err) => {
                    self.stats.edges_refused += 1;
                    if err == WheelWindowError::Discontinuity {
                        self.stats.edges_discontinuous += 1;
                    }
                }
            }
        }
        let expired = still.len().saturating_sub(MAX_PENDING_EDGES);
        self.stats.edges_expired += expired as u64;
        still.drain(..expired);
        self.pending = still;
        let excess = self.factors.len().saturating_sub(MAX_FACTORS);
        self.factors.drain(..excess);
        added
    }

    /// Removes the factor between keyframes `from_idx` and `to_idx`, if one is held: the motion
    /// check found the wheels disagreeing with the gyro and vision over it (slip).
    pub fn drop_factor(&mut self, from_idx: usize, to_idx: usize) -> bool {
        let before = self.factors.len();
        self.factors
            .retain(|f| !(f.from_idx == from_idx && f.to_idx == to_idx));
        self.factors.len() < before
    }

    /// The integrated factors, keyed by `Frame::idx` as `run_local_inertial_ba` wants them.
    pub fn factors(&self) -> &[WheelOdomFactor] {
        &self.factors
    }

    /// The factors, for keyframe culling to compose the pair through a removed keyframe.
    pub fn factors_mut(&mut self) -> &mut Vec<WheelOdomFactor> {
        &mut self.factors
    }

    /// Keyframes a queued interval still names: culling must keep them, or the factor
    /// integrated later would name a keyframe the map no longer holds.
    pub fn pending_keyframes(&self) -> impl Iterator<Item = usize> + '_ {
        self.pending.iter().flat_map(|e| [e.from_idx, e.to_idx])
    }

    /// Drops readings and mounts no pending interval, keyframe or frame can still ask for.
    ///
    /// `keep_from` are the stamps a future lookup may start at (the last keyframe's, the last
    /// frame's); the reading at or before the oldest of those and the pending starts is kept.
    pub fn prune(&mut self, keep_from: impl IntoIterator<Item = Option<u64>>) {
        let oldest = keep_from
            .into_iter()
            .flatten()
            .chain(self.pending.iter().map(|e| e.from_ns))
            .min();
        let Some(oldest) = oldest else {
            return;
        };
        while self.ring.len() > 1 && self.ring[1].stamp_ns <= oldest {
            self.ring.pop_front();
        }
        while self.mounts.len() > 1 && self.mounts[1].0 <= oldest {
            self.mounts.pop_front();
        }
    }

    /// Forgets the intervals and factors of a map that was dropped: their keyframe indices name
    /// keyframes that no longer exist. The readings and mounts are kept, because the pose that
    /// seeds the next map is propagated on them.
    pub fn reset_map(&mut self) {
        self.pending.clear();
        self.factors.clear();
    }
}

/// The unicycle error propagation behind [`WheelDelta::covariance`], in the start base frame.
///
/// The state is `(x, y, theta, e)`: the pose relative to the start, and `e`, the systematic
/// heading error per metre, drawn once per interval with variance `ks^2`. Per reading-to-reading
/// step `u` (the step vector in the start frame, `ds` its signed length along the mid heading):
///
/// ```text
///     P <- F P F^T + Q,    F = [[1, 0, -u_y, 0],      Q = diag(kt^2 |u|, kt^2 |u|,
///                               [0, 1,  u_x, 0],               kr^2 |dtheta| + kh^2 |u|, 0)
///                               [0, 0,  1,  ds],
///                               [0, 0,  0,   1]]
/// ```
///
/// `(-u_y, u_x)` is how the step moves when the heading it was driven at is off, which is what
/// turns a heading error into lateral error (`var_y ~ kh^2 d^3 / 3`, `cov(y, theta) ~ kh^2 d^2 / 2`
/// on a straight run). `e` is a state rather than a per-step `(ks ds)^2`, because summed over N
/// steps that would be `ks^2 d^2 / N`: a systematic error that shrank with the reading rate.
struct Propagation {
    p: [[f64; 4]; 4],
}

impl Propagation {
    fn new(cfg: &WheelOdomConfig) -> Self {
        let mut p = [[0.0; 4]; 4];
        p[3][3] = cfg.heading_scale_sigma_per_m.powi(2);
        Self { p }
    }

    /// One step `u` (start frame, metres) driven at `mid_heading` (relative to the start)
    /// while turning `dtheta`.
    fn step(&mut self, cfg: &WheelOdomConfig, u: [f64; 2], mid_heading: f64, dtheta: f64) {
        let len = u[0].hypot(u[1]);
        let (sin, cos) = mid_heading.sin_cos();
        let ds = u[0] * cos + u[1] * sin;
        let f = [
            [1.0, 0.0, -u[1], 0.0],
            [0.0, 1.0, u[0], 0.0],
            [0.0, 0.0, 1.0, ds],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut fp = [[0.0; 4]; 4];
        for (r, row) in fp.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x = (0..4).map(|k| f[r][k] * self.p[k][c]).sum();
            }
        }
        for (r, row) in self.p.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x = (0..4).map(|k| fp[r][k] * f[c][k]).sum();
            }
        }
        let var_t = cfg.translation_noise.powi(2) * len;
        self.p[0][0] += var_t;
        self.p[1][1] += var_t;
        self.p[2][2] +=
            cfg.rotation_noise.powi(2) * dtheta.abs() + cfg.heading_noise_per_m.powi(2) * len;
    }

    /// The `(x, y, theta)` block: `e` marginalised out.
    fn planar(&self) -> [[f64; 3]; 3] {
        let p = &self.p;
        [
            [p[0][0], p[0][1], p[0][2]],
            [p[1][0], p[1][1], p[1][2]],
            [p[2][0], p[2][1], p[2][2]],
        ]
    }
}

/// Whether two mounts differ by less than [`MOUNT_SAME_RAD`] and [`MOUNT_SAME_M`].
fn same_mount(a: &Pose3d, b: &Pose3d) -> bool {
    let r = a.rotation.transpose() * b.rotation;
    let cos = ((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5;
    cos.clamp(-1.0, 1.0).acos() < MOUNT_SAME_RAD
        && (a.translation - b.translation).length() < MOUNT_SAME_M
}

/// `a` wrapped to `(-pi, pi]`.
fn wrap(a: f64) -> f64 {
    let w = a.rem_euclid(std::f64::consts::TAU);
    if w > std::f64::consts::PI {
        w - std::f64::consts::TAU
    } else {
        w
    }
}

/// `R` from nine ROW-major numbers, as a human writes a matrix (glam is column-major).
pub(crate) fn mat3_from_rows(rows: &[f64; 9]) -> Mat3F64 {
    Mat3F64::from_cols_array(rows).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn cfg() -> WheelOdomConfig {
        WheelOdomConfig::new(Pose3d::IDENTITY, 0.02, 0.05).expect("valid")
    }

    fn state(samples: &[(u64, f64, f64, f64)]) -> WheelState {
        let mut w = WheelState::new(cfg());
        w.push(samples.iter().map(|&(t, x, y, theta)| WheelSample {
            stamp_ns: t * MS,
            x,
            y,
            theta,
        }));
        w
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    /// Straight drive at 1 m/s: the interval between two stamps that fall BETWEEN readings is
    /// the interpolated distance, in the start's base frame.
    #[test]
    fn test_integration_between_stamps_interpolates_both_ends() {
        let w = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.1, 0.0, 0.0),
            (200, 0.2, 0.0, 0.0),
        ]);
        let d = w.delta(50 * MS, 150 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.1, 0.0, 0.0]), "{:?}", d.delta);
        assert!((d.path_m - 0.1).abs() < 1e-12);
    }

    /// The delta is in the START's base frame: driving along odom +y while facing +y is
    /// straight ahead, +x of the base.
    #[test]
    fn test_the_delta_is_expressed_in_the_start_base_frame() {
        let h = std::f64::consts::FRAC_PI_2;
        let w = state(&[(0, 1.0, 1.0, h), (400, 1.0, 1.5, h)]);
        let d = w.delta(0, 400 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.5, 0.0, 0.0]), "{:?}", d.delta);
    }

    /// A heading that wraps through +-pi is the small turn it is, not a 2 pi spin, and an
    /// in-place turn has zero path length but its full turned angle.
    #[test]
    fn test_heading_unwraps_across_pi() {
        let pi = std::f64::consts::PI;
        let w = state(&[(0, 0.0, 0.0, pi - 0.1), (100, 0.0, 0.0, -pi + 0.1)]);
        let d = w.delta(0, 100 * MS, 0).expect("covered");
        assert!((d.delta[2] - 0.2).abs() < 1e-12, "{:?}", d.delta);
        assert!((d.path_rad - 0.2).abs() < 1e-12);
        assert_eq!(d.path_m, 0.0);
    }

    /// A there-and-back interval nets zero motion but keeps the uncertainty of what was driven.
    #[test]
    fn test_path_length_sums_the_readings_not_the_net_motion() {
        let w = state(&[
            (0, 1.0, 0.0, 0.0),
            (200, 1.3, 0.0, 0.0),
            (400, 1.0, 0.0, 0.0),
        ]);
        let d = w.delta(0, 400 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.0, 0.0, 0.0]));
        assert!((d.path_m - 0.6).abs() < 1e-12);
        assert!((d.covariance[0][0] - 0.02f64.powi(2) * 0.6).abs() < 1e-15);
        // Parked: floored, not infinite information.
        let parked = w.cfg.measurement(&parked_delta());
        assert_eq!(
            parked.covariance[0][0],
            DEFAULT_MIN_TRANSLATION_SIGMA_M.powi(2)
        );
        assert!(parked.information().is_some());
    }

    fn parked_delta() -> WheelDelta {
        WheelDelta {
            delta: [0.0; 3],
            path_m: 0.0,
            path_rad: 0.0,
            covariance: [[0.0; 3]; 3],
            speed_m_s: [0.0; 2],
            rate_rad_s: [0.0; 2],
        }
    }

    /// Deterministic standard normals: SplitMix64 into Box-Muller, so the Monte Carlo tests
    /// below draw the same paths on every run and need no RNG dependency.
    struct Normal(u64);
    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // (0, 1]: never 0, which the log below cannot take.
            ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        }
        fn sample(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// A 1 m drive at 1 m/s read at 100 Hz, curving at `curvature` rad/m, from the origin.
    fn drive(curvature: f64) -> Vec<(u64, f64, f64, f64)> {
        (0..=100u64)
            .map(|k| {
                let s = k as f64 * 0.01;
                let th = curvature * s;
                let (x, y) = if curvature == 0.0 {
                    (s, 0.0)
                } else {
                    (th.sin() / curvature, (1.0 - th.cos()) / curvature)
                };
                (k * 10, x, y, th)
            })
            .collect()
    }

    fn noisy_cfg() -> WheelOdomConfig {
        let mut c = cfg();
        c.translation_noise = 0.01;
        c.rotation_noise = 0.05;
        c.heading_noise_per_m = 0.03;
        c.heading_scale_sigma_per_m = 0.02;
        c
    }

    fn state_with(cfg: WheelOdomConfig, samples: &[(u64, f64, f64, f64)]) -> WheelState {
        let mut w = WheelState::new(cfg);
        w.push(samples.iter().map(|&(t, x, y, theta)| WheelSample {
            stamp_ns: t * MS,
            x,
            y,
            theta,
        }));
        w
    }

    /// The empirical covariance of the end pose over `n` paths driven with the noise model
    /// itself: per step, isotropic translation noise, heading noise from turning and from
    /// distance, and one systematic heading error per metre for the whole path.
    fn monte_carlo(
        cfg: &WheelOdomConfig,
        path: &[(u64, f64, f64, f64)],
        n: usize,
    ) -> [[f64; 3]; 3] {
        let mut rng = Normal(0x5eed_f00d);
        let mut ends = Vec::with_capacity(n);
        for _ in 0..n {
            let e = cfg.heading_scale_sigma_per_m * rng.sample();
            let (mut x, mut y, mut th) = (path[0].1, path[0].2, path[0].3);
            for pair in path.windows(2) {
                let ((_, x0, y0, t0), (_, x1, y1, t1)) = (pair[0], pair[1]);
                let dth = t1 - t0;
                // The nominal step in the body frame at its mid heading, re-driven at the noisy
                // heading.
                let mid = t0 + 0.5 * dth;
                let (s, c) = mid.sin_cos();
                let (bx, by) = (
                    c * (x1 - x0) + s * (y1 - y0),
                    -s * (x1 - x0) + c * (y1 - y0),
                );
                let len = bx.hypot(by);
                let (sn, cn) = (th + 0.5 * dth).sin_cos();
                let sig_t = cfg.translation_noise * len.sqrt();
                x += cn * bx - sn * by + sig_t * rng.sample();
                y += sn * bx + cn * by + sig_t * rng.sample();
                let sig_r = (cfg.rotation_noise.powi(2) * dth.abs()
                    + cfg.heading_noise_per_m.powi(2) * len)
                    .sqrt();
                th += dth + e * bx + sig_r * rng.sample();
            }
            ends.push([x, y, th]);
        }
        let mean: Vec<f64> = (0..3)
            .map(|i| ends.iter().map(|p| p[i]).sum::<f64>() / n as f64)
            .collect();
        let mut cov = [[0.0; 3]; 3];
        for p in &ends {
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]) / (n - 1) as f64;
                }
            }
        }
        cov
    }

    /// Every entry within `tol` of the propagated one, relative to `sqrt(P_ii P_jj)`: a cross
    /// term is judged against the variances it couples, so a zero one cannot pass by being small.
    fn assert_cov_close(propagated: &[[f64; 3]; 3], empirical: &[[f64; 3]; 3], tol: f64) {
        for i in 0..3 {
            for j in 0..3 {
                let scale = (propagated[i][i] * propagated[j][j]).sqrt();
                let err = (propagated[i][j] - empirical[i][j]).abs() / scale;
                assert!(
                    err < tol,
                    "[{i}][{j}]: propagated {:e}, empirical {:e} ({:.1} %)\n{propagated:?}\n{empirical:?}",
                    propagated[i][j],
                    empirical[i][j],
                    err * 100.0
                );
            }
        }
    }

    /// The propagated covariance is what 8000 paths driven with the same noise actually spread
    /// to, on a straight metre (where the heading-to-lateral cross term is the whole point) and
    /// on a 90 deg arc (where the heading error rotates into both axes).
    ///
    /// 8000, not 2000: a sample variance's relative error is `sqrt(2 / n)`, 3.2 % at 2000, so a
    /// 5 % bound there is 1.6 sigma per entry and one of nine fails on noise alone (5.3 % on
    /// `var_y` did). At 8000 it is 3 sigma; at 40000 every entry agreed within 1.3 %.
    ///
    /// Mutation: dropping F (adding Q only, no compounding) turned this red: lateral variance
    /// at a quarter of the truth and every cross term zero.
    #[test]
    fn delta_covariance_matches_monte_carlo_on_a_straight_metre() {
        let cfg = noisy_cfg();
        for curvature in [0.0, std::f64::consts::FRAC_PI_2] {
            let path = drive(curvature);
            let w = state_with(cfg.clone(), &path);
            let d = w.delta(0, 1000 * MS, 0).expect("covered");
            let p = d.covariance;
            if curvature == 0.0 {
                // The case the test exists for: a heading error grows into strongly
                // correlated lateral error, which a diagonal model has no room for.
                assert!(p[1][2] / (p[1][1] * p[2][2]).sqrt() > 0.5, "{p:?}");
            }
            assert_cov_close(&p, &monte_carlo(&cfg, &path, 8000), 0.05);
        }
    }

    /// With no turning at all, heading still drifts with distance: linearly from the random
    /// walk, quadratically from the systematic term.
    ///
    /// Mutation: dropping `heading_noise_per_m` from Q turned this red (theta held at the floor).
    #[test]
    fn heading_variance_grows_with_distance_on_a_straight_run() {
        let mut cfg = noisy_cfg();
        cfg.heading_scale_sigma_per_m = 0.0;
        let w = state_with(cfg.clone(), &drive(0.0));
        let var = |t_ms: u64| w.delta(0, t_ms * MS, 0).expect("covered").covariance[2][2];
        let kh2 = cfg.heading_noise_per_m.powi(2);
        assert!((var(500) - kh2 * 0.5).abs() < 1e-12, "{}", var(500));
        assert!((var(1000) - kh2 * 1.0).abs() < 1e-12, "{}", var(1000));
        let m = cfg.measurement(&w.delta(0, 1000 * MS, 0).unwrap());
        assert!(m.covariance[2][2] > DEFAULT_MIN_HEADING_SIGMA_RAD.powi(2) * 10.0);
        // The systematic term adds ks^2 d^2 on top.
        cfg.heading_scale_sigma_per_m = 0.02;
        let w = state_with(cfg, &drive(0.0));
        let v = w.delta(0, 1000 * MS, 0).unwrap().covariance[2][2];
        assert!((v - (kh2 + 0.02f64.powi(2))).abs() < 1e-12, "{v}");
    }

    /// The timing term: nothing on a parked interval (the floors alone), and `2 v^2 sigma_t^2`
    /// (`2 w^2 sigma_t^2` on theta) on one driven at a constant `v` (`w`).
    ///
    /// Mutation: a constant `2 sigma_t^2` floor regardless of speed turned this red.
    #[test]
    fn timing_floor_is_zero_parked_and_v_sigma_t_moving() {
        let c = cfg();
        let sigma = DEFAULT_CLOCK_SIGMA_S;
        let parked = state(&[
            (0, 1.0, 2.0, 0.3),
            (100, 1.0, 2.0, 0.3),
            (200, 1.0, 2.0, 0.3),
        ]);
        let m = c.measurement(&parked.delta(50 * MS, 150 * MS, 0).unwrap());
        assert_eq!(m.covariance[0][0], DEFAULT_MIN_TRANSLATION_SIGMA_M.powi(2));
        assert_eq!(m.covariance[1][1], DEFAULT_MIN_TRANSLATION_SIGMA_M.powi(2));
        assert_eq!(m.covariance[2][2], DEFAULT_MIN_HEADING_SIGMA_RAD.powi(2));
        // 0.5 m/s straight; both ends between readings.
        let moving = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.05, 0.0, 0.0),
            (200, 0.1, 0.0, 0.0),
        ]);
        let d = moving.delta(50 * MS, 150 * MS, 0).unwrap();
        assert!((d.speed_m_s[0] - 0.5).abs() < 1e-9 && (d.speed_m_s[1] - 0.5).abs() < 1e-9);
        let m = c.measurement(&d);
        let want = 2.0 * 0.25 * sigma * sigma;
        assert!((m.covariance[1][1] - d.covariance[1][1] - want).abs() < 1e-15);
        // 0.5 rad/s on the spot.
        let turning = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.0, 0.0, 0.05),
            (200, 0.0, 0.0, 0.1),
        ]);
        let d = turning.delta(50 * MS, 150 * MS, 0).unwrap();
        let m = c.measurement(&d);
        assert!((m.covariance[2][2] - d.covariance[2][2] - want).abs() < 1e-15);
    }

    #[test]
    fn z_sigma_grows_with_path() {
        let mut c = cfg();
        c.z_slope = 0.01;
        let mut d = parked_delta();
        assert_eq!(c.measurement(&d).z_sigma, c.z_sigma);
        d.path_m = 2.0;
        let m = c.measurement(&d);
        assert!(
            (m.z_sigma - 0.01f64.hypot(0.02)).abs() < 1e-15,
            "{}",
            m.z_sigma
        );
        assert_eq!(m.tilt_sigma, c.tilt_sigma, "tilt does not random-walk");
        // Off by default: the constant z_sigma of before.
        d.path_m = 5.0;
        assert_eq!(cfg().measurement(&d).z_sigma, cfg().z_sigma);
    }

    /// On a turning interval the measurement has real cross terms; it is exactly symmetric and
    /// kornia-slam turns it into a symmetric positive definite 6x6 information.
    #[test]
    fn the_measurement_is_symmetric_and_its_information_positive_definite() {
        let w = state_with(noisy_cfg(), &drive(std::f64::consts::FRAC_PI_2));
        let m = w.cfg.measurement(&w.delta(0, 1000 * MS, 0).unwrap());
        let c = m.covariance;
        assert!(c[0][1] != 0.0 && c[0][2] != 0.0 && c[1][2] != 0.0, "{c:?}");
        for (i, row) in c.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                assert_eq!(*x, c[j][i]);
            }
        }
        let info = m.information().expect("positive definite");
        // Cholesky succeeds only on a symmetric positive definite matrix.
        let mut l = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                assert!((info[i][j] - info[j][i]).abs() <= 1e-9 * info[i][i].abs().max(1.0));
            }
            for j in 0..=i {
                let s: f64 = info[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                if i == j {
                    assert!(s > 0.0, "not positive definite at {i}");
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
    }

    /// The covariance is handed over in the END base frame, where kornia-slam's right-side
    /// residual lives: after a 90 deg turn the start frame's forward variance is the end
    /// frame's lateral one, and the heading cross terms follow.
    ///
    /// Mutation: passing the start-frame covariance through unrotated turned this red.
    #[test]
    fn the_covariance_is_rotated_into_the_residual_frame() {
        let mut c = noisy_cfg();
        c.clock_sigma_s = 0.0;
        let w = state_with(c.clone(), &drive(std::f64::consts::FRAC_PI_2));
        let d = w.delta(0, 1000 * MS, 0).unwrap();
        let (p, m) = (d.covariance, c.measurement(&d).covariance);
        // R(90 deg)^T maps start x to end -y and start y to end x.
        assert!((m[0][0] - p[1][1]).abs() < 1e-12 * p[1][1].max(1.0));
        assert!((m[1][1] - p[0][0]).abs() < 1e-12 * p[0][0].max(1.0));
        assert!((m[0][2] - p[1][2]).abs() < 1e-12);
        assert!((m[1][2] + p[0][2]).abs() < 1e-12);
        assert!(
            (m[0][0] - p[0][0]).abs() > 1e-6,
            "the test needs anisotropy: {p:?}"
        );
    }

    #[test]
    fn test_uncovered_intervals_are_refused_and_a_late_end_is_pending() {
        let w = state(&[
            (100, 0.0, 0.0, 0.0),
            (200, 0.1, 0.0, 0.0),
            (1200, 1.1, 0.0, 0.0),
        ]);
        assert_eq!(
            w.delta(50 * MS, 150 * MS, 0),
            Err(WheelWindowError::NotCovered)
        );
        assert_eq!(
            w.delta(150 * MS, 150 * MS, 0),
            Err(WheelWindowError::NonPositiveInterval)
        );
        assert_eq!(
            w.delta(150 * MS, 1300 * MS, 0),
            Err(WheelWindowError::Pending)
        );
        // A 1 s hole is longer than the 0.5 s max_gap.
        assert_eq!(w.delta(150 * MS, 1100 * MS, 0), Err(WheelWindowError::Gap));
        // The prior may hold the newest reading for up to max_gap.
        let held = w.delta(1200 * MS, 1300 * MS, 500 * MS).expect("held");
        assert!(close(held.delta, [0.0; 3]));
    }

    /// An odometry that restarts (back to exactly the origin) or jumps faster than a base drives
    /// inside an interval refuses it, with no time gap to give it away; the same motion spread
    /// over a plausible time, and a robot parked at the origin, integrate as before.
    #[test]
    fn test_an_odometry_discontinuity_refuses_the_interval() {
        // 2026-10-10: parked at 0.1 rad, the body restarts, the next reading is the origin.
        let restarted = state(&[
            (0, 0.3, 0.0, 0.1),
            (100, 0.3, 0.0, 0.1),
            (200, 0.0, 0.0, 0.0),
            (300, 0.0, 0.0, 0.0),
        ]);
        assert_eq!(
            restarted.delta(50 * MS, 250 * MS, 0),
            Err(WheelWindowError::Discontinuity)
        );
        // Both ends after the restart: a clean interval again.
        let after = restarted.delta(210 * MS, 290 * MS, 0).expect("clean");
        assert!(close(after.delta, [0.0; 3]));
        // A 1 rad heading jump in 10 ms (100 rad/s), not at the origin.
        let jumped = state(&[(0, 0.0, 0.0, 0.2), (10, 0.0, 0.0, 1.2), (20, 0.0, 0.0, 1.2)]);
        assert_eq!(
            jumped.delta(5 * MS, 15 * MS, 0),
            Err(WheelWindowError::Discontinuity)
        );
        // The prior does not extrapolate the jump's velocity either.
        let held = jumped.delta(20 * MS, 40 * MS, 500 * MS);
        assert!(close(held.expect("held").delta, [0.0; 3]));
        // A real 0.5 rad/s turn and a 0.5 m/s drive at 100 Hz, through the origin, integrate.
        let real: Vec<_> = (0..20u64)
            .map(|k| {
                (
                    k * 10,
                    0.005 * k as f64 - 0.05,
                    0.0,
                    0.005 * k as f64 - 0.05,
                )
            })
            .collect();
        assert!(state(&real).delta(0, 190 * MS, 0).is_ok());
        let parked = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.0, 0.0, 0.0),
            (200, 0.0, 0.0, 0.0),
        ]);
        assert!(parked.delta(50 * MS, 150 * MS, 0).is_ok());
        let mut w = restarted;
        w.on_keyframe(Some((1, 50 * MS)), 2, 250 * MS);
        assert!(w.factors().is_empty());
        assert_eq!((w.stats.edges_refused, w.stats.edges_discontinuous), (1, 1));
    }

    /// An interval whose readings arrive after the keyframe that closed it is integrated at a
    /// later keyframe, with the mount valid at each END, not at the time it was built.
    #[test]
    fn test_a_pending_interval_completes_when_the_readings_catch_up() {
        let mut w = state(&[(0, 0.0, 0.0, 0.0), (100, 0.1, 0.0, 0.0)]);
        let tilted = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.2, 0.0));
        w.push_mount(150 * MS, tilted).expect("rigid");
        w.on_keyframe(Some((3, 50 * MS)), 4, 200 * MS);
        assert!(w.factors().is_empty(), "the end is not covered yet");
        w.push([WheelSample {
            stamp_ns: 250 * MS,
            x: 0.25,
            y: 0.0,
            theta: 0.0,
        }]);
        w.on_keyframe(None, 5, 260 * MS);
        let f = w.factors();
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].from_idx, f[0].to_idx), (3, 4));
        assert!(close(f[0].measurement.delta, [0.15, 0.0, 0.0]));
        // The oldest mount stands in for a stamp before it: it is nearer the truth than the
        // configured one once a head is publishing.
        assert_eq!(f[0].t_cb_from, tilted);
        assert_eq!(f[0].t_cb_to, tilted);
        assert_eq!(
            WheelState::new(cfg()).t_cb_at(0),
            Pose3d::IDENTITY,
            "configured until then"
        );
        assert_eq!(w.stats.edges_added, 1);
    }

    #[test]
    fn test_a_stream_that_stopped_expires_its_intervals() {
        let mut w = state(&[(0, 0.0, 0.0, 0.0)]);
        for k in 0..(MAX_PENDING_EDGES + 3) {
            let t = (k as u64 + 1) * 100 * MS;
            w.on_keyframe(Some((k, t - 100 * MS)), k + 1, t);
        }
        assert_eq!(w.stats.edges_expired, 3);
        assert_eq!(w.pending.len(), MAX_PENDING_EDGES);
    }

    #[test]
    fn test_out_of_order_and_non_finite_readings_are_refused() {
        let mut w = state(&[(100, 0.0, 0.0, 0.0)]);
        let s = |t: u64, x: f64| WheelSample {
            stamp_ns: t * MS,
            x,
            y: 0.0,
            theta: 0.0,
        };
        assert_eq!(
            w.push([s(100, 1.0), s(50, 1.0), s(200, f64::NAN), s(300, 1.0)]),
            1
        );
        assert_eq!((w.stats.out_of_order, w.stats.non_finite), (2, 1));
    }

    /// Camera looking along base +x (OpenCV axes: x right = base -y, y down = base -z). Driving
    /// forward 0.5 m moves the camera 0.5 m along its own +z, so a world point ahead gets 0.5 m
    /// closer in camera coordinates.
    #[test]
    fn test_camera_motion_maps_base_forward_onto_camera_forward() {
        let r_bc = mat3_from_rows(&[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
        let cam_in_base = Pose3d::new(r_bc, Vec3F64::new(0.1, 0.0, 0.9));
        let mut w = WheelState::new(WheelOdomConfig::new(cam_in_base, 0.02, 0.05).unwrap());
        w.push([(0, 0.0), (400, 0.5)].map(|(t, x)| WheelSample {
            stamp_ns: t * MS,
            x,
            y: 0.0,
            theta: 0.0,
        }));
        let m = w.camera_motion(0, 400 * MS).expect("covered");
        let p_cam0 = Vec3F64::new(0.0, 0.0, 2.0);
        let p_cam1 = m.transform_point(&p_cam0);
        assert!(
            (p_cam1 - Vec3F64::new(0.0, 0.0, 1.5)).length() < 1e-12,
            "{p_cam1:?}"
        );
    }

    #[test]
    fn test_a_non_rigid_mount_is_refused() {
        let scaled = Pose3d::new(Mat3F64::IDENTITY * 2.0, Vec3F64::ZERO);
        assert!(matches!(
            WheelOdomConfig::new(scaled, 0.02, 0.05),
            Err(WheelError::MountNotRigid { .. })
        ));
        let mut w = WheelState::new(cfg());
        assert!(w.push_mount(1, scaled).is_err());
        assert_eq!(w.stats.mounts_refused, 1);
        assert!(matches!(
            WheelOdomConfig::new(Pose3d::IDENTITY, 0.0, 0.05),
            Err(WheelError::ParamInvalid {
                field: "translation_noise",
                ..
            })
        ));
    }

    #[test]
    fn test_prune_keeps_the_reading_before_the_oldest_stamp_still_needed() {
        let mut w = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.1, 0.0, 0.0),
            (200, 0.2, 0.0, 0.0),
        ]);
        w.prune([Some(150 * MS), None]);
        assert_eq!(w.ring.front().unwrap().stamp_ns, 100 * MS);
        assert!(w.delta(150 * MS, 200 * MS, 0).is_ok());
    }

    /// A prior past the newest reading continues at the last pair's velocity, from both ends:
    /// a lag shorter than the interval, and one longer than it (start past the newest too).
    #[test]
    fn test_a_lagging_prior_extrapolates_instead_of_holding() {
        let w = state(&[(0, 0.0, 0.0, 0.0), (100, 0.1, 0.0, 0.0)]);
        let short = w
            .delta(50 * MS, 150 * MS, 500 * MS)
            .expect("within the hold");
        assert!(close(short.delta, [0.1, 0.0, 0.0]), "{:?}", short.delta);
        let long = w
            .delta(150 * MS, 250 * MS, 500 * MS)
            .expect("within the hold");
        assert!(close(long.delta, [0.1, 0.0, 0.0]), "{:?}", long.delta);
        // A factor still refuses to guess.
        assert_eq!(
            w.delta(50 * MS, 150 * MS, 0),
            Err(WheelWindowError::Pending)
        );
    }

    /// A still head stores one mount however long it publishes, so the mount at an old
    /// keyframe stamp survives; a moving one is stored per change.
    #[test]
    fn test_a_still_mount_takes_one_slot() {
        let mut w = WheelState::new(cfg());
        let tilted = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.2, 0.0));
        w.push_mount(MS, tilted).unwrap();
        for k in 2..(MOUNT_CAPACITY as u64 + 10) {
            w.push_mount(k * MS, Pose3d::IDENTITY).unwrap();
        }
        assert_eq!(w.mounts.len(), 2);
        assert_eq!(w.t_cb_at(MS), tilted);
        assert_eq!(w.stats.mounts_accepted, MOUNT_CAPACITY as u64 + 9);
        // Order is still checked against the newest sample, stored or not.
        w.push_mount(5 * MS, tilted).unwrap();
        assert_eq!(w.stats.mounts_refused, 1);
    }
}
