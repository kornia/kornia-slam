//! Sliding-window visual-inertial odometry: a Basalt-style fixed-lag
//! smoother over the rectified stereo pair (`experiments/euroc-2x-vio-design.md`).
//!
//! [`SlidingWindowOdometry`] owns its IMU queue, a two-camera flow frontend
//! and the window; it reads only its own state and the images it is given,
//! so in `Observe` it changes nothing the pipeline outputs.

mod config;
mod frontend;
mod init;
mod keyframes;
mod landmarks;
mod linearize;
mod marginalize;
mod solve;
mod state;
#[cfg(any(test, feature = "bench_internals"))]
mod synthetic;
#[cfg(test)]
mod tests;
mod window;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::{
    ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu, SampleInterpolation,
};

pub use config::{Odometry, OdometryImuNoise, OdometryMode, OdometryTuning, SlidingWindowConfig};

use crate::tracking::flow_pool::FlowPoolError;
use crate::tracking::optical_flow::TrackSetError;
use frontend::FlowFrontend;
use state::FrameId;
use window::{FrameInput, FrameReport, Observation, StereoView, Window};

/// World gravity of the window, m/s².
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
/// IMU samples kept behind the latest frame, s.
const IMU_KEEP_SEC: f64 = 1.0;

/// Why a frame got no estimate.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OdometryError {
    #[error("the window is not initialized")]
    NotInitialized,
    #[error("no finite accelerometer sample to start from")]
    NoGravity,
    #[error("no IMU interval between the frames")]
    ImuGap,
    #[error("a state or the prior is not finite")]
    NonFinite,
    #[error("a prior variable is not in the window")]
    PriorOutsideWindow,
    #[error("the marginal prior is not finite")]
    Marginalization,
    #[error(transparent)]
    Flow(#[from] FlowPoolError),
    #[error(transparent)]
    Tracks(#[from] TrackSetError),
}

/// The estimate after one frame: the newest state of the window.
#[derive(Debug, Clone, Copy)]
pub struct OdometryOutput {
    pub timestamp_sec: f64,
    /// World to left camera; the world is z-up, its origin the first body
    /// position.
    pub pose_world_to_cam: Pose3d,
    /// World frame, m/s.
    pub velocity_world: Vec3F64,
    pub bias: ImuBias,
    pub keyframe: bool,
    pub optimized: bool,
}

/// Run totals of the estimator; `*_ms` hold one value per frame.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OdometryCounters {
    pub frames: u64,
    pub poses: u64,
    pub keyframes: u64,
    pub solves: u64,
    pub solve_failures: u64,
    pub resets: u64,
    pub errors: u64,
    pub attempts: u64,
    pub rejected_steps: u64,
    pub landmarks_sum: u64,
    pub rows_left_sum: u64,
    pub rows_right_sum: u64,
    pub stereo_links: u64,
    pub fb_rejects: u64,
    pub row_gate_rejects: u64,
    /// Forward solves or survivor-filter rejects (the 60 px cap among them).
    pub movement_rejects: u64,
    pub landmarks_fixed: u64,
    pub marg_pinv: u64,
    pub imu_chi2_nav_sum: f64,
    pub bias_walk_chi2_sum: f64,
    pub chi2_count: u64,
    /// Each IMU edge's `r_navᵀ Ω r_nav / 9` as it enters the prior: the
    /// noise-arm statistic (`imu_chi2_nav` reads the newest edge, which the
    /// newest state absorbs).
    pub imu_chi2_marg_sum: f64,
    pub imu_chi2_marg_count: u64,
    pub visual_chi2_sum: f64,
    pub visual_chi2_count: u64,
    /// `log10` of the twin prior's yaw nullspace residual, the largest seen.
    pub nullspace_yaw_log10_max: f64,
    pub flow_ms: Vec<f32>,
    pub solve_ms: Vec<f32>,
    pub nullspace_ms: Vec<f32>,
    pub total_ms: Vec<f32>,
}

impl OdometryCounters {
    fn new() -> Self {
        Self {
            nullspace_yaw_log10_max: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    fn mean(sum: f64, count: u64) -> f64 {
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    pub fn iterations_mean(&self) -> f64 {
        Self::mean(self.attempts as f64, self.solves)
    }

    pub fn landmarks_mean(&self) -> f64 {
        Self::mean(self.landmarks_sum as f64, self.frames)
    }

    pub fn rows_left_mean(&self) -> f64 {
        Self::mean(self.rows_left_sum as f64, self.solves)
    }

    pub fn rows_right_mean(&self) -> f64 {
        Self::mean(self.rows_right_sum as f64, self.solves)
    }

    pub fn imu_chi2_nav(&self) -> f64 {
        Self::mean(self.imu_chi2_nav_sum, self.chi2_count)
    }

    pub fn imu_chi2_marg(&self) -> f64 {
        Self::mean(self.imu_chi2_marg_sum, self.imu_chi2_marg_count)
    }

    pub fn bias_walk_chi2(&self) -> f64 {
        Self::mean(self.bias_walk_chi2_sum, self.chi2_count)
    }

    pub fn visual_chi2(&self) -> f64 {
        Self::mean(self.visual_chi2_sum, self.visual_chi2_count)
    }
}

fn ms(d: Duration) -> f32 {
    (d.as_secs_f64() * 1e3) as f32
}

/// The estimator of `odometry: SlidingWindow`, run on the calling thread.
pub(crate) struct SlidingWindowOdometry {
    noise: ImuCalib,
    interpolation: SampleInterpolation,
    t_bc: Pose3d,
    imu: VecDeque<ImuMeasurement>,
    frontend: FlowFrontend,
    window: Window,
    /// Timestamp (s) of the newest frame in the window.
    last: Option<f64>,
    observations: Vec<Observation>,
    counters: OdometryCounters,
}

impl SlidingWindowOdometry {
    /// `None` when the rig has no IMU or no stereo baseline.
    pub(crate) fn new(
        config: &SlidingWindowConfig,
        tuning: &OdometryTuning,
        rig: &SensorRig,
        interpolation: SampleInterpolation,
    ) -> Result<Option<Self>, FlowPoolError> {
        let (Some(t_bc), Some(baseline)) = (rig.camera_to_body(), rig.stereo_baseline_m) else {
            return Ok(None);
        };
        let cameras = [
            StereoView {
                camera: rig.camera.clone(),
                camera_from_primary: Pose3d::IDENTITY,
            },
            StereoView {
                camera: rig.camera.clone(),
                camera_from_primary: Pose3d::new(
                    Mat3F64::IDENTITY,
                    Vec3F64::new(-baseline, 0.0, 0.0),
                ),
            },
        ];
        let noise = match config.imu_noise {
            OdometryImuNoise::Rig => rig.imu_noise(),
            OdometryImuNoise::Explicit(noise) => noise,
        };
        Ok(Some(Self {
            noise,
            interpolation,
            t_bc,
            imu: VecDeque::new(),
            frontend: FlowFrontend::new(tuning, &cameras)?,
            window: Window::new(tuning, cameras, t_bc, Vec3F64::from_array(GRAVITY)),
            last: None,
            observations: Vec::new(),
            counters: OdometryCounters::new(),
        }))
    }

    pub(crate) fn push_imu(&mut self, samples: &[ImuMeasurement]) {
        self.imu
            .extend(samples.iter().filter(|s| s.timestamp.is_finite()).copied());
    }

    pub(crate) fn counters(&self) -> &OdometryCounters {
        &self.counters
    }

    pub(crate) fn count_error(&mut self) {
        self.counters.errors += 1;
    }

    /// The first accelerometer reading at or after `t`, else the latest.
    fn accel_at(&mut self, t: f64) -> Option<Vec3F64> {
        let samples = self.imu.make_contiguous();
        samples
            .iter()
            .find(|s| s.timestamp >= t)
            .or_else(|| samples.last())
            .map(|s| s.accel)
    }

    pub(crate) fn process_pair(
        &mut self,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<OdometryOutput, OdometryError> {
        let started = Instant::now();
        self.counters.frames += 1;
        let id = (timestamp_sec * 1e9).round() as FrameId;
        let pim = match (self.last, self.window.newest()) {
            (Some(t0), Some((_, state))) => {
                let bias = state.value().bias;
                Some(PreintegratedImu::from_measurements_with(
                    bias,
                    self.noise,
                    self.imu.make_contiguous(),
                    t0,
                    timestamp_sec,
                    self.interpolation,
                ))
            }
            _ => None,
        };
        // R_cur_prev of the left camera: R_cb ΔRᵀ R_bc.
        let r_bc = self.t_bc.rotation;
        let rotation = pim
            .as_ref()
            .map(|p| r_bc.transpose() * p.delta_rotation.transpose() * r_bc);
        self.observations.clear();
        self.observations
            .extend_from_slice(self.frontend.process(left, right, rotation)?);
        let flow_done = Instant::now();
        self.counters.flow_ms.push(ms(flow_done - started));
        self.update_frontend_counts();

        let observations = std::mem::take(&mut self.observations);
        let result = match pim {
            Some(pim) if pim.dt > 0.0 => self.window.add_frame(FrameInput {
                id,
                imu: Some(pim),
                observations: &observations,
            }),
            Some(_) => Err(OdometryError::ImuGap),
            None => self.start(id, timestamp_sec, &observations),
        };
        let result = match result {
            Err(
                OdometryError::ImuGap | OdometryError::NonFinite | OdometryError::Marginalization,
            ) if self.last.is_some() => {
                self.counters.resets += 1;
                self.start(id, timestamp_sec, &observations)
            }
            other => other,
        };
        self.observations = observations;
        let report = result?;
        self.last = Some(timestamp_sec);
        self.prune_imu(timestamp_sec);
        let done = Instant::now();
        let nullspace = report.nullspace_time;
        self.counters.nullspace_ms.push(ms(nullspace));
        self.counters
            .solve_ms
            .push(ms((done - flow_done).saturating_sub(nullspace)));
        self.counters
            .total_ms
            .push(ms((done - started).saturating_sub(nullspace)));
        self.record(&report);
        self.output(timestamp_sec, &report)
    }

    /// Starts the window at this frame (`sqrt_keypoint_vio.cpp:185-220`).
    fn start(
        &mut self,
        id: FrameId,
        timestamp_sec: f64,
        observations: &[Observation],
    ) -> Result<FrameReport, OdometryError> {
        let accel = self
            .accel_at(timestamp_sec)
            .ok_or(OdometryError::NoGravity)?;
        self.window.initialize(id, accel)?;
        self.window.add_frame(FrameInput {
            id,
            imu: None,
            observations,
        })
    }

    fn prune_imu(&mut self, timestamp_sec: f64) {
        while self
            .imu
            .front()
            .is_some_and(|s| s.timestamp < timestamp_sec - IMU_KEEP_SEC)
        {
            self.imu.pop_front();
        }
    }

    fn update_frontend_counts(&mut self) {
        let counts = self.frontend.counts();
        self.counters.stereo_links = counts.stereo_links;
        self.counters.fb_rejects = counts.fb_rejects;
        self.counters.row_gate_rejects = counts.row_gate_rejects;
        self.counters.movement_rejects = counts.flow_failures;
    }

    fn record(&mut self, report: &FrameReport) {
        let c = &mut self.counters;
        c.keyframes += u64::from(report.keyframe);
        c.landmarks_sum += self.window.landmark_count() as u64;
        c.marg_pinv += u64::from(report.pseudo_inverse);
        if let Some(solve) = report.solve {
            c.solves += 1;
            c.solve_failures += u64::from(solve.failed);
            c.attempts += solve.attempts as u64;
            c.rejected_steps += solve.rejected as u64;
            c.landmarks_fixed += solve.landmarks_fixed as u64;
            c.rows_left_sum += solve.rows[0] as u64;
            c.rows_right_sum += solve.rows[1] as u64;
            let rows = solve.rows[0] + solve.rows[1];
            if rows > 0 {
                c.visual_chi2_sum += solve.vision_cost / rows as f64;
                c.visual_chi2_count += 1;
            }
        }
        if let (Some(nav), Some(bias)) = (report.imu_chi2_nav, report.bias_walk_chi2) {
            c.imu_chi2_nav_sum += nav;
            c.bias_walk_chi2_sum += bias;
            c.chi2_count += 1;
        }
        c.imu_chi2_marg_sum += report.imu_chi2_marg.iter().sum::<f64>();
        c.imu_chi2_marg_count += report.imu_chi2_marg.len() as u64;
        if let Some(n) = report.nullspace {
            c.nullspace_yaw_log10_max = c.nullspace_yaw_log10_max.max(n[5].max(1e-300).log10());
        }
    }

    fn output(
        &mut self,
        timestamp_sec: f64,
        report: &FrameReport,
    ) -> Result<OdometryOutput, OdometryError> {
        let (_, state) = self.window.newest().ok_or(OdometryError::NotInitialized)?;
        let value = state.value();
        self.counters.poses += 1;
        Ok(OdometryOutput {
            timestamp_sec,
            pose_world_to_cam: value.pose_cw,
            velocity_world: value.velocity,
            bias: value.bias,
            keyframe: report.keyframe,
            optimized: report.optimized,
        })
    }
}

/// A steady window step for `benches/bench_odometry_window.rs`.
#[cfg(feature = "bench_internals")]
#[doc(hidden)]
pub mod bench {
    use kornia_sensors::imu::{ImuBias, ImuCalib, PreintegratedImu};

    use super::config::OdometryTuning;
    use super::synthetic::{StereoWalk, camera_to_body, gravity, stereo_views};
    use super::window::{FrameInput, Observation, Window};

    /// The frames of a 12 s noisy walk, precomputed, fed one per step; the
    /// window restarts at the end.
    pub struct WindowBench {
        frames: Vec<(i64, Option<PreintegratedImu>, Vec<Observation>)>,
        first_accel: kornia_algebra::Vec3F64,
        window: Window,
        next: usize,
    }

    const WARM_UP_FRAMES: usize = 40;

    impl WindowBench {
        pub fn new() -> Self {
            let calib = ImuCalib {
                gyro_noise: 1.6968e-4,
                accel_noise: 2.0e-3,
                gyro_bias_noise: 1.9393e-5,
                accel_bias_noise: 3.0e-3,
            };
            let walk = StereoWalk::new(12.0, ImuBias::default(), Some((calib, 1)));
            let mut gauss = crate::mapping::synthetic_walk::Gauss(3);
            let frames = (0..walk.frames())
                .map(|k| {
                    let t = StereoWalk::time(k);
                    let imu = (k > 0).then(|| {
                        let t0 = StereoWalk::time(k - 1);
                        PreintegratedImu::from_measurements(
                            ImuBias::default(),
                            calib,
                            walk.samples(t0, t),
                            t0,
                            t,
                        )
                    });
                    (
                        StereoWalk::id(k),
                        imu,
                        walk.observations(k, Some((&mut gauss, 0.5))),
                    )
                })
                .collect();
            let mut bench = Self {
                frames,
                first_accel: walk.accel_at(0.0),
                window: Window::new(
                    &OdometryTuning::default(),
                    stereo_views(),
                    camera_to_body(),
                    gravity(),
                ),
                next: 0,
            };
            for _ in 0..WARM_UP_FRAMES {
                bench.step();
            }
            bench
        }

        /// One frame through the window.
        pub fn step(&mut self) {
            if self.next == self.frames.len() {
                self.next = 0;
            }
            let (id, imu, observations) = &self.frames[self.next];
            if self.next == 0 && self.window.initialize(*id, self.first_accel).is_err() {
                return;
            }
            let _ = self.window.add_frame(FrameInput {
                id: *id,
                imu: imu.clone(),
                observations,
            });
            self.next += 1;
        }
    }

    impl Default for WindowBench {
        fn default() -> Self {
            Self::new()
        }
    }
}
