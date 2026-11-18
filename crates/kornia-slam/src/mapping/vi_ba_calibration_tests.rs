//! Card G (`experiments/pr180-G-card.md`), tests GT0-GT7: online calibration
//! in the local VI-BA, on windows from the varying-motion generator.

use std::f64::consts::PI;

use super::tests::spin_vision;
use super::*;
use crate::mapping::calibration_states::prior_rows;
use crate::mapping::global_vi_ba::{
    ClockOffsetPrior, FocalSlicePrior, GyroIntrinsicsPrior, OfflineCalibrationConfig,
    RotationSlicePrior,
};
use crate::mapping::online_calibration::GATE_MIN_SPEED_MPS;
use kornia_algebra::Mat3F64;
use kornia_sensors::imu::{ImuCalib, SampleInterpolation};

/// 5 keyframes, 4 edges of 0.5 s, all but the first keyframe free.
const KEYFRAMES: usize = 5;
const EDGE_SEC: f64 = 0.5;
const ACTIVE: usize = 4;
const SAMPLE_SEC: f64 = 1e-3;
/// `α(t) = (A / 2πf)(1 − cos 2πf t)`, A = 1 rad/s, f = 1 Hz.
const RATE: f64 = 1.0;
const FREQ: f64 = 1.0;
/// Forward speed (m/s) and sway amplitude (m): the speed stays above the gate.
const V0: f64 = 0.5;
const SWAY: f64 = 0.05;

/// LaMAria's IMU noise (`R_11_5cp.json`, `imu0`); the bias walk as `sim.ron`.
fn lamaria_calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.75e-4,
        accel_noise: 7.8e-4,
        gyro_bias_noise: 2e-6,
        accel_bias_noise: 4e-5,
    }
}

#[derive(Clone, Copy)]
struct Motion {
    /// Rotation about body x as well as about body y, the gravity axis.
    two_axis: bool,
    v0: f64,
    sway: f64,
}

impl Motion {
    const VARYING: Self = Self {
        two_axis: true,
        v0: V0,
        sway: SWAY,
    };
    const YAW_ONLY: Self = Self {
        two_axis: false,
        v0: V0,
        sway: SWAY,
    };
    const PURE_ROTATION: Self = Self {
        two_axis: false,
        v0: 0.0,
        sway: 0.0,
    };

    fn angle(t: f64) -> f64 {
        RATE / (2.0 * PI * FREQ) * (1.0 - (2.0 * PI * FREQ * t).cos())
    }

    /// `R_wb(t) = Exp(α e_x) · Exp(α e_y)`, or `Exp(α e_y)` on one axis.
    fn rotation_wb(&self, t: f64) -> Mat3F64 {
        let a = Self::angle(t);
        let yaw = SO3F64::exp(Vec3F64::new(0.0, a, 0.0)).matrix();
        if self.two_axis {
            SO3F64::exp(Vec3F64::new(a, 0.0, 0.0)).matrix() * yaw
        } else {
            yaw
        }
    }

    fn position(&self, t: f64) -> Vec3F64 {
        Vec3F64::new(self.v0 * t, self.sway * (2.0 * PI * FREQ * t).sin(), 0.0)
    }

    fn velocity(&self, t: f64) -> Vec3F64 {
        Vec3F64::new(
            self.v0,
            self.sway * 2.0 * PI * FREQ * (2.0 * PI * FREQ * t).cos(),
            0.0,
        )
    }

    /// `T_bw(t)`.
    fn body_pose(&self, t: f64) -> Pose3d {
        let r_bw = self.rotation_wb(t).transpose();
        Pose3d::new(r_bw, (r_bw * self.position(t)) * -1.0)
    }

    /// The noise-free sample covering camera time `(τ − dt, τ]`, stamped
    /// `τ − td` on the IMU clock. It carries the rotation and the velocity
    /// change over its interval, so the Hold integrator (each sample held
    /// over the interval ending at its stamp, rotated by `ΔR` at the
    /// interval start) reproduces the motion without a half-sample lag,
    /// which would read as a 0.5 ms clock offset.
    fn sample(&self, tau: f64, td: f64, gravity: Vec3F64) -> ImuMeasurement {
        let start = tau - SAMPLE_SEC;
        let r_bw0 = self.rotation_wb(start).transpose();
        let step = SO3F64::from_matrix(&(r_bw0 * self.rotation_wb(tau))).log();
        let dv = self.velocity(tau) - self.velocity(start) - gravity * SAMPLE_SEC;
        ImuMeasurement {
            timestamp: tau - td,
            gyro: step * (1.0 / SAMPLE_SEC),
            accel: (r_bw0 * dv) * (1.0 / SAMPLE_SEC),
        }
    }

    /// The noise-free reading at camera time `τ`, stamped `τ − td`: body
    /// rate and specific force at that instant, by central differences.
    fn instantaneous_sample(&self, tau: f64, td: f64, gravity: Vec3F64) -> ImuMeasurement {
        let h = 1e-5;
        let r_bw = self.rotation_wb(tau).transpose();
        let turn = self.rotation_wb(tau - h).transpose() * self.rotation_wb(tau + h);
        let accel = (self.velocity(tau + h) - self.velocity(tau - h)) * (0.5 / h);
        ImuMeasurement {
            timestamp: tau - td,
            gyro: SO3F64::from_matrix(&turn).log() * (0.5 / h),
            accel: r_bw * (accel - gravity),
        }
    }
}

/// What a window injects; the solver gets the factory calibration.
#[derive(Clone, Copy)]
struct Truth {
    motion: Motion,
    /// Clock offset (s): a camera stamp `t` is IMU time `t − td`.
    td: f64,
    /// The primary camera's rotation offset, camera frame (rad).
    theta0: Vec3F64,
    imu_t_bc: Option<Pose3d>,
    /// A secondary camera: its factory extrinsic to camera 0 and its
    /// rotation offset (rad).
    secondary: Option<(Pose3d, Vec3F64)>,
    /// Focal scale minus one of every camera.
    focal: f64,
    interpolation: SampleInterpolation,
    /// Samples read at their stamps, as a real IMU reads them, rather than
    /// the interval means `Hold` integrates exactly.
    instantaneous: bool,
}

impl Default for Truth {
    fn default() -> Self {
        Self {
            motion: Motion::VARYING,
            td: 0.0,
            theta0: Vec3F64::ZERO,
            imu_t_bc: Some(extrinsic()),
            secondary: None,
            focal: 0.0,
            interpolation: SampleInterpolation::Hold,
            instantaneous: false,
        }
    }
}

fn extrinsic() -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.05)).matrix(),
        Vec3F64::new(0.02, -0.01, 0.03),
    )
}

fn secondary_extrinsic() -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.0, 0.6, 0.0)).matrix(),
        Vec3F64::new(-0.1, 0.0, 0.0),
    )
}

fn rotation_pose(theta: Vec3F64) -> Pose3d {
    Pose3d::new(SO3F64::exp(theta).matrix(), Vec3F64::ZERO)
}

/// Narrow lens, stiff enough that the fixed landmarks pin the poses.
fn narrow_camera(scale: f64) -> PinholeCamera {
    PinholeCamera {
        fx: 4000.0 * scale,
        fy: 4000.0 * scale,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

struct Window {
    keyframes: Vec<ViBaKeyframe>,
    points: Vec<Vec3F64>,
    observations: Vec<BaObservation>,
    imu_edges: Vec<ImuFactor>,
    map_edges: Vec<MapImuFactor>,
    camera: PinholeCamera,
    params: ViBaParams,
    /// True `T_bw` per keyframe.
    body: Vec<Pose3d>,
}

/// Fixed landmarks in front of camera 1 of every free keyframe, projected
/// through its true pose `camera_from_primary · T_cw`.
fn secondary_vision(
    keyframes: &[ViBaKeyframe],
    camera: &PinholeCamera,
    camera_from_primary: &Pose3d,
    points: &mut Vec<Vec3F64>,
    observations: &mut Vec<BaObservation>,
) {
    for (pose_idx, kf) in keyframes.iter().enumerate().skip(1) {
        let t_wc = camera_from_primary.compose(&kf.pose).inverse();
        for ix in -3..=3 {
            for iy in -3..=3 {
                let z = if (ix + iy) % 2 == 0 { 2.0 } else { 12.0 };
                let p_cam = Vec3F64::new(ix as f64 * 0.2 * z, iy as f64 * 0.2 * z, z);
                observations.push(BaObservation {
                    pose_idx,
                    point_idx: points.len(),
                    pixel: [
                        (camera.fx * p_cam.x / z + camera.cx) as f32,
                        (camera.fy * p_cam.y / z + camera.cy) as f32,
                    ],
                    fixed_point: true,
                    ..Default::default()
                });
                points.push(t_wc.transform_point(&p_cam));
            }
        }
    }
}

/// The varying-motion generator (card G): the true camera poses are
/// `T_cw = Exp(θ_0) · imu_t_bc⁻¹ · T_bw`, so `imu_t_bc · Exp(−θ_0) · T_cw`
/// is the true body; each map edge holds the samples over `[t0, t1]` on the
/// IMU clock plus the nearest one outside each end.
fn window(truth: &Truth) -> Window {
    let defaults = ViBaParams::default();
    let gravity = defaults.gravity;
    let camera = narrow_camera(1.0);
    let true_camera = narrow_camera(1.0 + truth.focal);
    let times: Vec<f64> = (0..KEYFRAMES).map(|k| k as f64 * EDGE_SEC).collect();
    let body: Vec<Pose3d> = times.iter().map(|&t| truth.motion.body_pose(t)).collect();
    let camera_from_body =
        rotation_pose(truth.theta0).compose(&truth.imu_t_bc.unwrap_or(Pose3d::IDENTITY).inverse());
    let keyframes: Vec<ViBaKeyframe> = times
        .iter()
        .zip(&body)
        .enumerate()
        .map(|(k, (&t, pose_bw))| ViBaKeyframe {
            pose: camera_from_body.compose(pose_bw),
            velocity: truth.motion.velocity(t),
            bias: ImuBias::default(),
            fixed: k == 0,
            inertial_fixed: false,
        })
        .collect();

    let (mut points, mut observations) = spin_vision(&keyframes, &true_camera);
    let mut observation_views = vec![0u8; observations.len()];
    let mut rig_views = Vec::new();
    if let Some((camera_from_primary, phi)) = truth.secondary {
        let true_pose = rotation_pose(phi).compose(&camera_from_primary);
        secondary_vision(
            &keyframes,
            &true_camera,
            &true_pose,
            &mut points,
            &mut observations,
        );
        observation_views.resize(observations.len(), 1);
        rig_views.push(RigObservationCamera {
            camera: camera.clone(),
            camera_from_primary,
        });
    }

    let last = ((KEYFRAMES - 1) as f64 * EDGE_SEC / SAMPLE_SEC).round() as i64;
    let samples: Vec<ImuMeasurement> = (-100..=last + 100)
        .map(|k| {
            let tau = k as f64 * SAMPLE_SEC;
            if truth.instantaneous {
                truth.motion.instantaneous_sample(tau, truth.td, gravity)
            } else {
                truth.motion.sample(tau, truth.td, gravity)
            }
        })
        .collect();
    let map_edges: Vec<MapImuFactor> = (0..KEYFRAMES - 1)
        .map(|e| {
            let (t0, t1) = (times[e], times[e + 1]);
            let lo = samples
                .partition_point(|s| s.timestamp < t0)
                .saturating_sub(1);
            let hi = (samples.partition_point(|s| s.timestamp <= t1) + 1).min(samples.len());
            let raw_samples = samples[lo..hi].to_vec();
            let preintegrated = PreintegratedImu::from_measurements_with(
                ImuBias::default(),
                lamaria_calib(),
                &raw_samples,
                t0,
                t1,
                truth.interpolation,
            );
            MapImuFactor {
                prev_kf_idx: e,
                curr_kf_idx: e + 1,
                preintegrated,
                raw_samples,
                t0,
                t1,
                bias_walk_extra_s: 0.0,
                interpolation: truth.interpolation,
            }
        })
        .collect();
    let imu_edges = map_edges
        .iter()
        .map(|m| ImuFactor::new(m.prev_kf_idx, m.curr_kf_idx, m.preintegrated.clone()))
        .collect();
    Window {
        keyframes,
        points,
        observations,
        imu_edges,
        map_edges,
        camera,
        params: ViBaParams {
            imu_t_bc: truth.imu_t_bc,
            rig_views,
            observation_views,
            max_iterations: 50,
            ..defaults
        },
        body,
    }
}

/// The card's priors for the states that are on.
fn config(rotation: bool, focal: bool, gyro: bool, clock: bool) -> OnlineCalibrationConfig {
    OnlineCalibrationConfig {
        camera_rotation: rotation.then_some(RotationSlicePrior {
            step_sigma_deg: 0.05,
            value_sigma_deg: 1.0,
        }),
        camera_focal: focal.then_some(FocalSlicePrior {
            step_sigma: 5e-4,
            value_sigma: 0.01,
        }),
        gyro_intrinsics: gyro.then_some(GyroIntrinsicsPrior { sigma: 0.005 }),
        clock_offset: clock.then_some(ClockOffsetPrior { sigma_ms: 2.0 }),
    }
}

fn input<'a>(
    window: &'a Window,
    config: OnlineCalibrationConfig,
    start: &'a OnlineCalibrationState,
    active_keyframes: usize,
) -> OnlineCalibrationInput<'a> {
    OnlineCalibrationInput {
        config,
        start,
        map_edges: window.map_edges.iter().collect(),
        active_keyframes,
    }
}

/// One solve through the public entry, from `start`.
fn calibrated(
    window: &Window,
    config: OnlineCalibrationConfig,
    start: &OnlineCalibrationState,
    active_keyframes: usize,
) -> (ViBaResult, OnlineCalibrationState) {
    let input = input(window, config, start, active_keyframes);
    visual_inertial_bundle_adjust_calibrated(
        &window.keyframes,
        &window.points,
        &window.observations,
        &window.imu_edges,
        &[],
        &[],
        &window.camera,
        &window.params,
        &input,
    )
    .unwrap()
}

/// One solve from the factory calibration through the shared body, with the
/// marginal sigmas; `force_open` opens the gate whatever the speed.
fn solve_window(
    window: &Window,
    config: OnlineCalibrationConfig,
    force_open: bool,
) -> (ViBaResult, CalibratedOutcome, Layout) {
    let start = OnlineCalibrationState::default();
    let input = input(window, config, &start, ACTIVE);
    let kf_local = free_keyframe_slots(
        &window.keyframes,
        &window.observations,
        &window.imu_edges,
        &[],
        &[],
    );
    let mut solve = CalibratedSolve::new(&window.keyframes, &kf_local, &window.params, &input);
    solve.marginal_sigma = true;
    solve.free |= force_open;
    let (result, outcome) = solve_vi_ba(
        &window.keyframes,
        &window.points,
        &window.observations,
        &window.imu_edges,
        &[],
        &[],
        &window.camera,
        &window.params,
        Some(&solve),
    )
    .unwrap();
    (result, outcome, solve.layout.clone())
}

/// The data-only sigma of a column: its marginal sigma with the information
/// of its value and step priors taken out (both are diagonal rows).
fn data_sigma(posterior: f64, step_sigma: f64, value_sigma: f64) -> f64 {
    let information = posterior.powi(-2) - step_sigma.powi(-2) - value_sigma.powi(-2);
    if information > 0.0 {
        information.sqrt().recip()
    } else {
        f64::INFINITY
    }
}

fn rotation_deg(values: &Calibration, layout: &Layout, camera: usize) -> [f64; 3] {
    let at = layout.rotation(camera, 0).unwrap();
    values.vec3(at).to_array().map(f64::to_degrees)
}

fn angle_deg(a: &Mat3F64, b: &Mat3F64) -> f64 {
    SO3F64::from_matrix(&(a.transpose() * *b))
        .log()
        .length()
        .to_degrees()
}

fn assert_same_preintegration(a: &PreintegratedImu, b: &PreintegratedImu, what: &str) {
    assert_eq!(a.delta_rotation, b.delta_rotation, "{what}: ΔR");
    assert_eq!(a.delta_velocity, b.delta_velocity, "{what}: Δv");
    assert_eq!(a.delta_position, b.delta_position, "{what}: Δp");
    assert_eq!(a.dt, b.dt, "{what}: dt");
    assert_eq!(a.covariance, b.covariance, "{what}: covariance");
    assert_eq!(a.bias_covariance, b.bias_covariance, "{what}: bias walk");
    assert_eq!(a.d_rotation_d_bias_gyro, b.d_rotation_d_bias_gyro, "{what}");
    assert_eq!(a.d_velocity_d_bias_gyro, b.d_velocity_d_bias_gyro, "{what}");
    assert_eq!(
        a.d_velocity_d_bias_accel, b.d_velocity_d_bias_accel,
        "{what}"
    );
    assert_eq!(a.d_position_d_bias_gyro, b.d_position_d_bias_gyro, "{what}");
    assert_eq!(
        a.d_position_d_bias_accel, b.d_position_d_bias_accel,
        "{what}"
    );
}

/// GT0: at the factory gyro and without the clock state (which integrates
/// `Midpoint` whatever the edge), the window-stream re-integration is
/// `MapImuFactor::repreintegrate`, for `Hold` and `Midpoint`, on both
/// window-end edges and an interior one.
#[test]
fn window_reintegration_matches_repreintegrate() {
    for interpolation in [SampleInterpolation::Hold, SampleInterpolation::Midpoint] {
        let window = window(&Truth {
            interpolation,
            ..Truth::default()
        });
        let start = OnlineCalibrationState::default();
        let input = input(&window, config(false, false, true, false), &start, ACTIVE);
        let kf_local = free_keyframe_slots(
            &window.keyframes,
            &window.observations,
            &window.imu_edges,
            &[],
            &[],
        );
        let solve = CalibratedSolve::new(&window.keyframes, &kf_local, &window.params, &input);
        let bias = ImuBias {
            gyro: Vec3F64::new(1e-3, -2e-3, 5e-4),
            accel: Vec3F64::new(0.02, -0.01, 0.03),
        };
        for e in [0, 1, KEYFRAMES - 2] {
            let edge = &window.map_edges[e];
            let reintegrated = solve.reintegrate(edge, bias, &solve.previous).unwrap();
            assert_same_preintegration(
                &reintegrated,
                &edge.repreintegrate(bias),
                &format!("{interpolation:?} edge {e}"),
            );
        }
    }
}

/// GT1: a clock offset injected in the IMU stamps is recovered in one solve,
/// with its sign; 2 ms reaches past the stream on a window-end edge. The
/// samples are read at their stamps and the edges `Hold`, as a run with the
/// default integration records them: the clock state integrates `Midpoint`
/// regardless (D3). Red on the old code (2026-11-18), whose `Hold` lags by
/// half a sample: −5 ms came back as −4.43 ms.
#[test]
fn online_td_recovers_injected_offset() {
    for (td, tolerance) in [(5e-3, 0.5e-3), (-5e-3, 0.5e-3), (2e-3, 0.2e-3)] {
        let window = window(&Truth {
            td,
            instantaneous: true,
            ..Truth::default()
        });
        let (result, state) = calibrated(
            &window,
            config(false, false, false, true),
            &OnlineCalibrationState::default(),
            ACTIVE,
        );
        assert!(!result.calibration_gated);
        let estimate = state.estimate().unwrap().clock_offset_sec.unwrap();
        assert!(
            (estimate - td).abs() <= tolerance,
            "td {:.3} ms estimated {:.4} ms",
            td * 1e3,
            estimate * 1e3
        );
    }
}

/// GT2: the rotation offsets, each a single solve from 0. The secondary
/// camera's is observed by vision on every axis, the primary's by the IMU
/// across two rotation axes; about the only rotation axis of a yaw-only
/// window the value and step priors hold it.
#[test]
fn online_rotation_observability() {
    let (step, value) = (0.05f64.to_radians(), 1.0f64.to_radians());
    let injected = 0.3f64.to_radians();
    let rotation = config(true, false, false, false);

    // Secondary camera, 0.3° about its x axis.
    let window_rig = window(&Truth {
        secondary: Some((secondary_extrinsic(), Vec3F64::new(injected, 0.0, 0.0))),
        ..Truth::default()
    });
    let (_, outcome, layout) = solve_window(&window_rig, rotation, false);
    let sigma = outcome.marginal_sigma.as_ref().unwrap();
    let x = layout.rotation(1, 0).unwrap();
    let sigma_data = data_sigma(sigma[x], step, value);
    assert!(
        sigma_data <= 0.01f64.to_radians(),
        "camera 1 x: data sigma {:.4}°",
        sigma_data.to_degrees()
    );
    let phi = rotation_deg(&outcome.values, &layout, 1);
    assert!((phi[0] - 0.3).abs() <= 0.03, "camera 1 offset {phi:?}°");
    assert!(
        phi[1].abs() <= 0.03 && phi[2].abs() <= 0.03,
        "camera 1 offset {phi:?}°"
    );

    // Primary camera, 0.3° about its x axis in `imu_t_bc`: offline's sign.
    let window_primary = window(&Truth {
        theta0: Vec3F64::new(injected, 0.0, 0.0),
        ..Truth::default()
    });
    let (_, outcome, layout) = solve_window(&window_primary, rotation, false);
    let sigma = outcome.marginal_sigma.as_ref().unwrap();
    let x = layout.rotation(0, 0).unwrap();
    let sigma_data = data_sigma(sigma[x], step, value);
    assert!(
        sigma_data <= 0.01f64.to_radians(),
        "camera 0 x: data sigma {:.4}°",
        sigma_data.to_degrees()
    );
    let theta = rotation_deg(&outcome.values, &layout, 0);
    assert!((theta[0] - 0.3).abs() <= 0.03, "camera 0 offset {theta:?}°");
    assert!(
        theta[1].abs() <= 0.03 && theta[2].abs() <= 0.03,
        "camera 0 offset {theta:?}°"
    );

    // Yaw only, 0.3° about the yaw axis: camera and body axes coincide, so
    // the yaw axis is the camera's y.
    let window_yaw = window(&Truth {
        motion: Motion::YAW_ONLY,
        theta0: Vec3F64::new(0.0, injected, 0.0),
        imu_t_bc: None,
        ..Truth::default()
    });
    let (_, outcome, layout) = solve_window(&window_yaw, rotation, false);
    let theta = rotation_deg(&outcome.values, &layout, 0);
    assert!(theta[1].abs() < 0.03, "yaw offset {theta:?}°");
    assert!(
        theta[0].abs() <= 0.03 && theta[2].abs() <= 0.03,
        "tilt {theta:?}°"
    );
}

/// GT3: pure rotation with the focal 0.5 % long, focal and gyro both on: the
/// two trade about the rotation axis, and their sum is what is observed.
#[test]
fn focal_and_gyro_scale_trade() {
    let window = window(&Truth {
        motion: Motion::PURE_ROTATION,
        imu_t_bc: None,
        focal: 0.005,
        ..Truth::default()
    });
    // No motion: the gate would hold the states.
    let (_, outcome, layout) = solve_window(&window, config(false, true, true, false), true);
    let focal = outcome.values.values[layout.focal(0, 0).unwrap()];
    let gyro = outcome.values.gyro(&layout).scale.y;
    let sum = focal + gyro;
    assert!(
        (0.004..=0.006).contains(&sum),
        "focal {:.4} % + gyro y {:.4} % = {:.4} %",
        focal * 1e2,
        gyro * 1e2,
        sum * 1e2
    );
}

/// GT4: the online value rows are `prior_rows`'s slice-0 rows bit for bit;
/// each step row is `(current − previous) / step_sigma` on the state's
/// column. `previous` is a constant of the solve, so it has no column.
#[test]
fn online_prior_rows_are_value_and_step_rows() {
    let online = config(true, true, true, true);
    let offline = OfflineCalibrationConfig {
        time_slices: 1,
        camera_rotation: online.camera_rotation,
        camera_focal: online.camera_focal,
        gyro_intrinsics: online.gyro_intrinsics,
        clock_offset: online.clock_offset,
    };
    let cameras = 2;
    let layout = Layout::online(&online, cameras);
    let offline_layout = Layout::new(&offline, cameras);
    assert_eq!(layout.len, offline_layout.len);
    let current = Calibration {
        values: (0..layout.len).map(|c| 1e-3 * (c as f64 + 1.0)).collect(),
    };
    let previous = Calibration {
        values: (0..layout.len).map(|c| -2e-4 * c as f64).collect(),
    };
    let rows = online_prior_rows(&layout, &online, &current, &previous);
    let offline_rows = prior_rows(&offline_layout, &offline, &current);
    // Per camera: three rotation components, then one focal.
    let drift = cameras * 4;
    assert_eq!(rows.len(), offline_rows.len() + drift);
    let steps = |c: usize| {
        if c < layout.focal(0, 0).unwrap() {
            0.05f64.to_radians()
        } else {
            5e-4
        }
    };
    for (k, offline_row) in offline_rows.iter().take(drift).enumerate() {
        let (value, step) = (&rows[2 * k], &rows[2 * k + 1]);
        assert_eq!(value.residual, offline_row.residual, "value row {k}");
        assert_eq!(value.terms, offline_row.terms, "value row {k}");
        let &[(column, w)] = step.terms.as_slice() else {
            panic!("step row {k} has {} terms", step.terms.len());
        };
        assert_eq!(column, offline_row.terms[0].0);
        assert_eq!(w, 1.0 / steps(column));
        assert_eq!(
            step.residual,
            w * (current.values[column] - previous.values[column]),
            "step row {k}"
        );
    }
    for (online_row, offline_row) in rows[2 * drift..].iter().zip(&offline_rows[drift..]) {
        assert_eq!(online_row.residual, offline_row.residual);
        assert_eq!(online_row.terms, offline_row.terms);
    }
}

/// GT5: the gate. Below 0.2 m/s on the oldest free keyframe, or one free
/// keyframe short of `active_keyframes`, a solve adds no column and returns
/// the carried estimate unchanged.
#[test]
fn online_calibration_gate() {
    assert!(!gate_open(Some(0.19), ACTIVE, ACTIVE));
    assert!(gate_open(Some(GATE_MIN_SPEED_MPS), ACTIVE, ACTIVE));
    assert!(!gate_open(Some(0.21), ACTIVE - 1, ACTIVE));
    assert!(!gate_open(None, ACTIVE, ACTIVE));

    let config = config(true, false, false, true);
    let mut window = window(&Truth {
        td: 1e-3,
        ..Truth::default()
    });
    let (_, carried) = calibrated(&window, config, &OnlineCalibrationState::default(), ACTIVE);
    assert!(carried.estimate().is_some());
    let columns = Layout::online(&config, 1).len;
    let direction = window.keyframes[1].velocity.normalize();

    window.keyframes[1].velocity = direction * 0.19;
    let (result, state) = calibrated(&window, config, &carried, ACTIVE);
    assert!(result.calibration_gated);
    assert_eq!(result.calibration_columns, 0);
    assert_eq!(state, carried);

    window.keyframes[1].velocity = direction * 0.21;
    let (result, _) = calibrated(&window, config, &carried, ACTIVE);
    assert!(!result.calibration_gated);
    assert_eq!(result.calibration_columns, columns);

    let (result, state) = calibrated(&window, config, &carried, ACTIVE + 1);
    assert_eq!(result.calibration_columns, 0);
    assert_eq!(state, carried);
}

/// GT6: the primary offset lives on the IMU side (D1), so the solved
/// keyframes stay in the camera frame tracking estimates, and the body
/// `imu_t_bc · Exp(−θ_0) · T_cw` is the true one.
#[test]
fn online_imu_side_rotation_keeps_camera_frame() {
    let injected = 0.3f64.to_radians();
    let truth = Truth {
        theta0: Vec3F64::new(injected, 0.0, 0.0),
        ..Truth::default()
    };
    let window = window(&truth);
    let (result, outcome, layout) = solve_window(&window, config(true, false, false, false), false);
    let theta = outcome.values.vec3(layout.rotation(0, 0).unwrap());
    let imu_t_bc = imu_t_bc_with_offset(truth.imu_t_bc.as_ref(), theta);
    for k in 1..KEYFRAMES {
        let (solved, start) = (&result.keyframes[k].pose, &window.keyframes[k].pose);
        let rotation = angle_deg(&solved.rotation, &start.rotation);
        assert!(rotation < 0.005, "keyframe {k} turned {rotation:.5}°");
        let centre = (solved.inverse().translation - start.inverse().translation).length();
        assert!(centre < 1e-3, "keyframe {k} moved {centre:.2e} m");
        let body = imu_t_bc.compose(solved);
        let error = angle_deg(&body.rotation, &window.body[k].rotation);
        assert!(error < 0.03, "keyframe {k} body off by {error:.4}°");
    }
}

/// GT7: only the gyro and clock states re-integrate (D2).
#[test]
fn online_reintegration_only_for_imu_states() {
    for (rotation, focal) in [(true, false), (false, true), (true, true)] {
        let layout = Layout::online(&config(rotation, focal, false, false), 2);
        assert!(!layout.imu_reintegrates());
    }
    assert!(Layout::online(&config(false, false, false, true), 2).imu_reintegrates());
    assert!(Layout::online(&config(false, false, true, false), 2).imu_reintegrates());

    let window = window(&Truth::default());
    let (_, outcome, _) = solve_window(&window, config(true, false, false, false), false);
    assert_eq!(outcome.reintegrated, 0);
    let (_, outcome, _) = solve_window(&window, config(true, false, false, true), false);
    assert_eq!(outcome.reintegrated, KEYFRAMES - 1);
}
