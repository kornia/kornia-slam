//! Inertial (visual-inertial) initialization.
//!
//! This module owns the initialization *window* (which keyframes take part and
//! when it opened), the readiness gate on that window (keyframe count,
//! preintegrated IMU time, motion), the progressive VIBA0 / VIBA1 / VIBA2
//! schedule that mirrors ORB-SLAM3's `LocalMapping::InitializeIMU` re-triggering,
//! and the single joint LM solve each stage runs for scale, gravity direction,
//! IMU biases and per-keyframe velocities.
//!
//! It only *computes* results: applying them to the map and to system state
//! (scaling/rotating the map, latching the bias and gravity estimate) is
//! `SlamSystem`'s job.
//!
//! Layout: this file holds the configuration, the public request/result types,
//! the readiness gate and the schedule; `solve.rs` holds the joint solve.

mod factor;
mod solve;

#[cfg(test)]
mod tests;

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuBias;

pub use crate::map::KeyframeVelocity;
use crate::map::{Keyframe, Map};

/// A keyframe window is monocular unless its first keyframe carries stereo data.
fn window_is_mono(kfs: &[&Keyframe]) -> bool {
    !kfs.first().map(|kf| kf.frame.is_stereo()).unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitConfig {
    pub min_keyframes: usize,
    /// Stereo readiness threshold on preintegrated IMU time; a monocular
    /// window multiplies it by `mono_min_time_factor`.
    pub min_time_sec: f64,
    pub min_motion: f64,
    /// ORB-SLAM3 uses minTime=2.0s (mono) vs 1.0s (stereo) — `min_time_sec`
    /// holds the stereo (stricter/smaller) value; mono doubles it.
    pub mono_min_time_factor: f64,
    /// Throttles retries: without this, once the window is ready, a rejected
    /// attempt keeps the system in ImuInit mode and never resets start_idx, so
    /// the exact same (growing) window gets re-solved from scratch on every
    /// single subsequent keyframe forever — an ever-more-expensive no-op once
    /// a call starts failing. Re-attempt at most once every 5s of new data
    /// (mirrors the VIBA1 5s cadence), not every keyframe.
    pub retry_interval_sec: f64,
    /// VIBA1 fires once `mTinit` exceeds this (LocalMapping.cc:200-228).
    pub viba1_after_sec: f64,
    /// VIBA2 fires once `mTinit` exceeds this (LocalMapping.cc:200-228).
    pub viba2_after_sec: f64,
    /// No refinement pass fires once `mTinit` reaches this.
    pub refine_until_sec: f64,
    /// Reject a diverged refinement instead of replacing the last valid bias.
    pub max_accel_bias: f64,
}

impl Default for ImuInitConfig {
    fn default() -> Self {
        // Matches ORB-SLAM3's LocalMapping::InitializeIMU VIBA0 gate
        // (nMinKF=10; minTime=1.0s stereo/2.0s mono — `readiness()` doubles
        // this for mono). The previous min_keyframes=30/min_time_sec=15.0
        // was effectively skipping VIBA0/VIBA1 and attempting a
        // VIBA2-strength window on the very first try.
        Self {
            min_keyframes: 10,
            min_time_sec: 1.0,
            min_motion: 0.05,
            mono_min_time_factor: 2.0,
            retry_interval_sec: 5.0,
            viba1_after_sec: 5.0,
            viba2_after_sec: 15.0,
            refine_until_sec: 50.0,
            max_accel_bias: 1.0, // m/s^2
        }
    }
}

impl ImuInitConfig {
    fn validate(&self) -> Result<(), ImuInitRejectReason> {
        if self.min_keyframes < 2 {
            return Err(ImuInitRejectReason::InvalidConfig(
                "min_keyframes must be at least 2".into(),
            ));
        }
        for (name, value) in [
            ("min_time_sec", self.min_time_sec),
            ("min_motion", self.min_motion),
            ("retry_interval_sec", self.retry_interval_sec),
            ("viba1_after_sec", self.viba1_after_sec),
            ("viba2_after_sec", self.viba2_after_sec),
            ("refine_until_sec", self.refine_until_sec),
            ("max_accel_bias", self.max_accel_bias),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ImuInitRejectReason::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if !self.mono_min_time_factor.is_finite() || self.mono_min_time_factor < 1.0 {
            return Err(ImuInitRejectReason::InvalidConfig(
                "mono_min_time_factor must be finite and at least 1".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Requests, results, rejections
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale: f64,
    pub gravity_world: Vec3F64,
    pub keyframe_velocities: Vec<KeyframeVelocity>,
    pub bias: ImuBias,
}

/// Why an attempted solve was rejected.
///
/// Distinct from [`ImuInitNotReadyReason`], which says why a window was never
/// attempted at all: those gates clear themselves as keyframes arrive, and the
/// schedule only logs them. A reject here is the outcome of one solve that
/// actually ran — VIBA0 retries it after `retry_interval_sec`, VIBA1/VIBA2
/// latch and never retry.
///
/// `InvalidConfig` and `InsufficientKeyframes` duplicate readiness gates on
/// purpose: [`ImuInitializer::try_initialize`] is public, so it re-checks them
/// and a caller that bypasses the schedule still gets a typed rejection
/// instead of a meaningless solve. `MissingExtrinsics` is raised by the
/// schedule before any solve. The remaining variants can only be discovered by
/// solving.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImuInitRejectReason {
    #[error("invalid IMU initialization configuration: {0}")]
    InvalidConfig(String),
    #[error("camera-to-body IMU extrinsics are missing")]
    MissingExtrinsics,
    #[error("not enough keyframes: found {found}, need at least {required}")]
    InsufficientKeyframes { found: usize, required: usize },
    #[error("no usable IMU factors connect the initialization window")]
    NoUsableFactors,
    #[error("inertial optimizer failed: {0}")]
    OptimizerFailed(String),
    #[error("invalid estimated scale {0}")]
    InvalidScale(f64),
    #[error("estimated IMU bias is non-finite")]
    NonFiniteBias,
    #[error("estimated accelerometer bias norm {actual:.3} exceeds {maximum:.3} m/s^2")]
    ImplausibleAccelBias { actual: f64, maximum: f64 },
}

/// Progressive visual-inertial BA stage, mirroring ORB-SLAM3's VIBA0
/// (immediate) / VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s) schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InertialStage {
    Viba0,
    Viba1,
    Viba2,
}

impl InertialStage {
    /// Log label used by the `[imu_init]` diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Viba0 => "VIBA0",
            Self::Viba1 => "VIBA1",
            Self::Viba2 => "VIBA2",
        }
    }
}

/// Kept at VIBA2 because this system lacks the intervening pose-adjusting
/// inertial BA that lets ORB-SLAM3 safely remove the prior (kornia-slam#51).
const VIBA_PRIOR_A: f64 = 1e5;

/// Zero-prior weights on the gyroscope and accelerometer bias for one solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasPriors {
    pub gyro: f64,
    pub accel: f64,
}

impl BiasPriors {
    /// ORB-SLAM3 `LocalMapping.cc:183-228` priors. Keep the numbers here only.
    pub fn for_stage(stage: InertialStage, is_mono: bool) -> Self {
        match stage {
            // VIBA0: ORB-SLAM3's first InitializeIMU call
            // (LocalMapping.cc:183-186) — heavily-regularized, mono suppresses
            // accel bias almost entirely (priorA=1e10) since a short/early
            // window can't yet observe it; stereo uses priorA=1e5.
            InertialStage::Viba0 => Self {
                gyro: 1e2,
                accel: if is_mono { 1e10 } else { 1e5 },
            },
            InertialStage::Viba1 => Self {
                gyro: 1.0,
                accel: VIBA_PRIOR_A,
            },
            // The accel prior is deliberately NOT relaxed at VIBA2 (see
            // `VIBA_PRIOR_A`); only the gyro prior is dropped.
            InertialStage::Viba2 => Self {
                gyro: 0.0,
                accel: VIBA_PRIOR_A,
            },
        }
    }
}

/// How the solve seeds the world-to-gravity rotation `Rwg` and the per-keyframe
/// velocities. Selects the same branch ORB-SLAM3 does at LocalMapping.cc:1226
/// (`!isImuInitialized()`).
#[derive(Debug, Clone, Copy)]
pub enum RwgSeed {
    /// First (VIBA0) solve: derive Rwg and velocities from scratch from the
    /// visual trajectory (finite-difference velocity + gravity accumulation).
    FromVisualTrajectory,
    /// Refinement solve: the map is already gravity-aligned and metric, so
    /// seed Rwg from the current gravity estimate and velocities from each
    /// keyframe's current IMU-propagated estimate.
    FromCurrentGravity(Vec3F64),
}

/// One inertial-initialization solve request.
#[derive(Debug, Clone)]
pub struct InertialInitRequest {
    pub start_kf_idx: usize,
    pub imu_t_bc: Pose3d,
    pub bias: ImuBias,
    pub stage: InertialStage,
    pub seed: RwgSeed,
}

/// What the schedule did for one accepted keyframe.
#[derive(Debug)]
pub enum InertialInitOutcome {
    /// Nothing to do at this keyframe (throttled, or no stage is due).
    NotDue,
    /// The window is not yet initializable; carries the gate snapshot.
    NotReady(ImuInitNotReady),
    /// A solve ran for `stage`.
    Attempted {
        stage: InertialStage,
        result: Result<ImuInitResult, ImuInitRejectReason>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Readiness gate
// ─────────────────────────────────────────────────────────────────────────────

/// Which readiness gate the initialization window is still short of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuInitNotReadyReason {
    /// No window has been opened yet (no bootstrap keyframe).
    NoWindow,
    /// The configuration itself is unusable.
    InvalidConfig,
    /// Too few keyframes in the window.
    Keyframes,
    /// Not enough preintegrated IMU time in the window.
    ImuTime,
    /// Not enough translation between the first and last keyframe.
    Motion,
}

/// Snapshot of the readiness gate for a window that cannot be initialized yet.
///
/// `min_time_sec` is the *effective* threshold: already doubled for a
/// monocular window (see [`ImuInitializer::readiness`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ImuInitNotReady {
    pub start_kf_idx: usize,
    pub first_kf_idx: Option<usize>,
    pub last_kf_idx: Option<usize>,
    pub keyframes: usize,
    pub min_keyframes: usize,
    pub imu_time_sec: f64,
    pub min_time_sec: f64,
    pub motion: f64,
    pub min_motion: f64,
    pub reason: ImuInitNotReadyReason,
}

impl std::fmt::Display for ImuInitNotReady {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[imu_init_gate] start_idx={} first_idx={:?} last_idx={:?} kfs={}/{} imu_time={:.2}/{:.1}s",
            self.start_kf_idx,
            self.first_kf_idx,
            self.last_kf_idx,
            self.keyframes,
            self.min_keyframes,
            self.imu_time_sec,
            self.min_time_sec,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schedule
// ─────────────────────────────────────────────────────────────────────────────

pub struct ImuInitializer {
    pub config: ImuInitConfig,
    /// First keyframe of the current initialization window.
    start_kf_idx: Option<usize>,
    /// Timestamp the current inertial-init window started (first keyframe at
    /// or after `start_kf_idx`). Mirrors ORB-SLAM3's `mFirstTs` / `mTinit` —
    /// used to gate the VIBA1/VIBA2 progressive visual-inertial BA refinement
    /// passes (mTinit>5s / mTinit>15s respectively, after the initial VIBA0
    /// solve) at LocalMapping.cc:200-228.
    window_start_sec: Option<f64>,
    /// Timestamp of the last VIBA0 attempt (successful or not), so retries are
    /// throttled to a fixed cadence instead of firing on every single keyframe
    /// forever once the window is ready — with an ever-growing window
    /// (start_idx never resets) and a solve that scales with window size,
    /// unthrottled per-keyframe retries turn into an ever-more-expensive no-op
    /// once a call starts getting rejected.
    last_attempt_sec: Option<f64>,
    /// VIBA1/VIBA2 fire at most once each, mirroring
    /// Map::GetIniertialBA1()/GetIniertialBA2() latching in ORB-SLAM3.
    viba1_done: bool,
    viba2_done: bool,
}

impl ImuInitializer {
    pub fn new(config: ImuInitConfig) -> Self {
        Self {
            config,
            start_kf_idx: None,
            window_start_sec: None,
            last_attempt_sec: None,
            viba1_done: false,
            viba2_done: false,
        }
    }

    /// Opens a fresh initialization window at `start_kf_idx`, clearing the
    /// retry throttle and both refinement latches.
    pub fn begin_window(&mut self, start_kf_idx: usize, timestamp_sec: f64) {
        self.start_kf_idx = Some(start_kf_idx);
        self.window_start_sec = Some(timestamp_sec);
        self.last_attempt_sec = None;
        self.viba1_done = false;
        self.viba2_done = false;
    }

    /// First keyframe of the current window, if one is open.
    pub fn window_start_kf_idx(&self) -> Option<usize> {
        self.start_kf_idx
    }

    /// Call on every accepted keyframe while `imu_initialized == false`.
    ///
    /// Runs VIBA0 — ORB-SLAM3's first `InitializeIMU` call
    /// (LocalMapping.cc:183-186) — as soon as the window is ready, throttled
    /// to `config.retry_interval_sec`.
    pub fn on_keyframe_uninitialized(
        &mut self,
        map: &Map,
        timestamp_sec: f64,
        imu_t_bc: Option<Pose3d>,
        bias: ImuBias,
    ) -> InertialInitOutcome {
        if let Err(not_ready) = self.readiness(map, self.start_kf_idx) {
            return InertialInitOutcome::NotReady(not_ready);
        }
        let due_for_retry = self
            .last_attempt_sec
            .is_none_or(|last| timestamp_sec - last >= self.config.retry_interval_sec);
        if !due_for_retry {
            return InertialInitOutcome::NotDue;
        }
        let Some(start_kf_idx) = self.start_kf_idx else {
            return InertialInitOutcome::NotDue;
        };
        self.last_attempt_sec = Some(timestamp_sec);

        let result = self.attempt(
            map,
            start_kf_idx,
            imu_t_bc,
            bias,
            InertialStage::Viba0,
            RwgSeed::FromVisualTrajectory,
        );
        InertialInitOutcome::Attempted {
            stage: InertialStage::Viba0,
            result,
        }
    }

    /// Call on every accepted keyframe once initialized; fires VIBA1 then
    /// VIBA2 at most once each.
    ///
    /// VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s): progressive re-solves with
    /// relaxed priors over the same (now-growing) window that VIBA0 used,
    /// mirroring LocalMapping.cc:200-228. Each fires at most once and refines
    /// bg/ba/scale/gravity further — tracking is already running on VIBA0's
    /// result by the time these get a chance to fire, so a rejection here
    /// just means "try again never" for that stage, not a tracking failure.
    pub fn on_keyframe_initialized(
        &mut self,
        map: &Map,
        timestamp_sec: f64,
        imu_t_bc: Option<Pose3d>,
        bias: ImuBias,
        gravity_world: Vec3F64,
    ) -> InertialInitOutcome {
        let (Some(start_kf_idx), Some(window_start_sec)) =
            (self.start_kf_idx, self.window_start_sec)
        else {
            return InertialInitOutcome::NotDue;
        };
        let mtinit = timestamp_sec - window_start_sec;
        if mtinit >= self.config.refine_until_sec {
            return InertialInitOutcome::NotDue;
        }

        let stage = if !self.viba1_done && mtinit > self.config.viba1_after_sec {
            InertialStage::Viba1
        } else if self.viba1_done && !self.viba2_done && mtinit > self.config.viba2_after_sec {
            InertialStage::Viba2
        } else {
            return InertialInitOutcome::NotDue;
        };

        let result = self.attempt(
            map,
            start_kf_idx,
            imu_t_bc,
            bias,
            stage,
            RwgSeed::FromCurrentGravity(gravity_world),
        );

        // Latch regardless of accept/reject: each stage gets exactly one shot.
        if stage == InertialStage::Viba1 {
            self.viba1_done = true;
        } else {
            self.viba2_done = true;
        }

        InertialInitOutcome::Attempted { stage, result }
    }

    /// Builds the request and solves, turning missing extrinsics into the
    /// typed rejection the caller logs.
    fn attempt(
        &self,
        map: &Map,
        start_kf_idx: usize,
        imu_t_bc: Option<Pose3d>,
        bias: ImuBias,
        stage: InertialStage,
        seed: RwgSeed,
    ) -> Result<ImuInitResult, ImuInitRejectReason> {
        let imu_t_bc = imu_t_bc.ok_or(ImuInitRejectReason::MissingExtrinsics)?;
        self.try_initialize(
            map,
            &InertialInitRequest {
                start_kf_idx,
                imu_t_bc,
                bias,
                stage,
                seed,
            },
        )
    }
}

// ── readiness gate ────────────────────────────────────────────────────────

impl ImuInitializer {
    /// Reports whether the window that starts at `start_idx` can be
    /// initialized, and if not, exactly which gate it is short of.
    pub fn readiness(&self, map: &Map, start_idx: Option<usize>) -> Result<(), ImuInitNotReady> {
        let mut report = ImuInitNotReady {
            start_kf_idx: start_idx.unwrap_or(0),
            first_kf_idx: None,
            last_kf_idx: None,
            keyframes: 0,
            min_keyframes: self.config.min_keyframes,
            imu_time_sec: 0.0,
            min_time_sec: self.config.min_time_sec,
            motion: 0.0,
            min_motion: self.config.min_motion,
            reason: ImuInitNotReadyReason::NoWindow,
        };

        if self.config.validate().is_err() {
            report.reason = ImuInitNotReadyReason::InvalidConfig;
            return Err(report);
        }
        let Some(start_idx) = start_idx else {
            return Err(report);
        };

        let kfs: Vec<&Keyframe> = map.keyframes_from(start_idx).collect();
        report.first_kf_idx = kfs.first().map(|kf| kf.frame.idx);
        report.last_kf_idx = kfs.last().map(|kf| kf.frame.idx);
        report.keyframes = kfs.len();
        // ORB-SLAM3 uses minTime=2.0s (mono) vs 1.0s (stereo) — `config.min_time_sec`
        // holds the stereo (stricter/smaller) value; mono doubles it.
        report.min_time_sec = if window_is_mono(&kfs) {
            self.config.min_time_sec * self.config.mono_min_time_factor
        } else {
            self.config.min_time_sec
        };
        report.imu_time_sec = map.imu_time_from(start_idx);

        if kfs.len() < self.config.min_keyframes {
            report.reason = ImuInitNotReadyReason::Keyframes;
            return Err(report);
        }
        if report.imu_time_sec < report.min_time_sec {
            report.reason = ImuInitNotReadyReason::ImuTime;
            return Err(report);
        }

        let t0 = kfs
            .first()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        let t1 = kfs
            .last()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        report.motion = (t1 - t0).length();
        if report.motion < self.config.min_motion {
            report.reason = ImuInitNotReadyReason::Motion;
            return Err(report);
        }
        Ok(())
    }

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        self.readiness(map, start_idx).is_ok()
    }
}
