use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::SampleInterpolation;

use super::*;
use crate::mapping::session::{
    ImuSample, SESSION_DUMP_VERSION, SessionCamera, SessionDump, SessionFrame, SessionKeyframe,
    SessionLandmark, SessionObservation, SessionPose, SessionRig,
};
use crate::mapping::synthetic_walk::{
    Sampling, body_position, body_rotation, body_velocity, exact_reading,
};

const FX: f64 = 400.0;
const CX: f64 = 320.0;
const CY: f64 = 240.0;
const GRAVITY: [f64; 3] = [0.0, 9.81, 0.0];
/// A camera at the IMU accelerating past a wall of landmarks, with exact
/// pixels and an exact IMU stream (specific force `a - g`, no rotation).
/// `offset` moves the dumped landmarks and every keyframe but the first off
/// the truth.
fn synthetic_session(offset: f64) -> SessionDump {
    sampled_session(offset, Sampling::IntervalMean)
}

fn sampled_session(offset: f64, sampling: Sampling) -> SessionDump {
    let landmarks: Vec<Vec3F64> = (0..5)
        .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| (i, j, k))))
        .map(|(i, j, k)| Vec3F64::new(i as f64 - 1.0, j as f64 - 1.0, 4.0 + 2.0 * k as f64))
        .collect();
    let keyframes = (0..9)
        .map(|k| {
            let t = 0.25 * k as f64;
            let p = body_position(t);
            let r_bw = body_rotation(t, sampling).transpose();
            let observations = landmarks
                .iter()
                .enumerate()
                .map(|(l, x)| {
                    let c = r_bw * (*x - p);
                    SessionObservation {
                        landmark: l as u32,
                        camera: 0,
                        octave: 0,
                        u: (FX * c.x / c.z + CX) as f32,
                        v: (FX * c.y / c.z + CY) as f32,
                        depth: None,
                    }
                })
                .collect();
            let shift = if k == 0 {
                Vec3F64::ZERO
            } else {
                Vec3F64::new(offset, -offset, 0.5 * offset)
            };
            SessionKeyframe {
                kf_idx: 10 * k as u64,
                timestamp_sec: t,
                pose_cw: SessionPose::from_pose(&Pose3d::new(r_bw, r_bw * -(p + shift))),
                velocity_world: body_velocity(t),
                bias_gyro: [0.0; 3],
                bias_accel: [0.0; 3],
                segment: 0,
                inserted_while_lost: false,
                anchor: false,
                evicted: false,
                observations,
            }
        })
        .collect();
    // Each sample holds over the interval before it, so it carries that
    // interval's mean acceleration and the preintegrated `Δv` stays exact.
    const DT: f64 = 0.005;
    let imu = (0..=400)
        .map(|i| {
            let t = i as f64 * DT;
            let (gyro, accel) = exact_reading(t, DT, sampling, Vec3F64::from_array(GRAVITY));
            ImuSample { t, gyro, accel }
        })
        .collect();
    let identity = SessionPose::from_pose(&Pose3d::IDENTITY);
    SessionDump {
        version: SESSION_DUMP_VERSION,
        rig: SessionRig {
            cameras: vec![SessionCamera {
                fx: FX,
                fy: FX,
                cx: CX,
                cy: CY,
                width: 640,
                height: 480,
                primary_from_camera: identity,
            }],
            camera_to_body: identity,
            imu_noise: [1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3],
            imu_time_offset_ns: 0,
            stereo_baseline_m: None,
        },
        keyframes,
        landmarks: landmarks
            .iter()
            .enumerate()
            .map(|(l, x)| SessionLandmark {
                idx: l as u32,
                position_world: (*x + Vec3F64::new(offset, offset, -offset)).to_array(),
            })
            .collect(),
        frames: vec![SessionFrame {
            timestamp_sec: 0.3,
            reference_kf: 10,
            frame_from_kf: SessionPose::from_pose(&Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(-0.025, 0.0, 0.0),
            )),
        }],
        imu,
        gravity_world: GRAVITY,
    }
}

fn solve(session: &SessionDump, config: &OfflineBaConfig) -> OfflineBaResult {
    solve_with(session, SampleInterpolation::Hold, config)
}

/// Every solve in this module takes it, so the parallelism test reads a
/// global no other solve is changing.
static FAER_GLOBAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn faer_global() -> std::sync::MutexGuard<'static, ()> {
    FAER_GLOBAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn solve_with(
    session: &SessionDump,
    interpolation: SampleInterpolation,
    config: &OfflineBaConfig,
) -> OfflineBaResult {
    let _global = faer_global();
    offline_visual_inertial_ba(
        session,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
        interpolation,
        config,
    )
    .unwrap()
}

/// P2's mechanism: with no iteration the dumped keyframes come back as
/// `T_bc · T_cw`, and a frame as its keyframe composed with `frame_from_kf`.
#[test]
fn zero_iterations_write_the_start_back_out() {
    let session = synthetic_session(0.0);
    let config = OfflineBaConfig {
        max_iterations: 0,
        ..OfflineBaConfig::default()
    };
    let result = solve(&session, &config);
    for (pose, kf) in result.body_poses.iter().zip(&session.keyframes) {
        let start = kf.pose_cw.to_pose();
        assert!((pose.translation - start.translation).length() < 1e-12);
    }
    let frames = reanchor_frames(&session, &result.body_poses);
    assert_eq!(frames.len(), 1);
    assert_eq!(result.report.frames, 1);
    let expected = -(body_position(0.25) + Vec3F64::new(0.025, 0.0, 0.0));
    assert!((frames[0].1.translation - expected).length() < 1e-12);
    assert_eq!(result.report.iterations, 0);
}

/// Exact measurements from a perturbed start: the solve returns to the
/// truth, which holds the first keyframe where the gauge prior keeps it.
#[test]
fn a_perturbed_start_converges_to_the_truth() {
    let session = synthetic_session(0.03);
    let config = OfflineBaConfig {
        tighten_from_iteration: 100,
        ..OfflineBaConfig::default()
    };
    let result = solve(&session, &config);
    let report = &result.report;
    assert!(report.iterations > 0);
    assert!(
        report.final_cost < 1e-6 * report.initial_cost,
        "cost {} -> {}",
        report.initial_cost,
        report.final_cost
    );
    assert_eq!(report.components, 1);
    assert_eq!(report.dropped_outlier, 0);
    for (pose, kf) in result.body_poses.iter().zip(&session.keyframes) {
        let truth = -body_position(kf.timestamp_sec);
        let error = (pose.translation - truth).length();
        assert!(
            error < 1e-4,
            "keyframe at {} s off by {error} m",
            kf.timestamp_sec
        );
    }
}

/// A gyro-intrinsics state on an exact stream stays near zero and the solve
/// still converges: the border Schur complement and the numeric columns.
#[test]
fn a_calibration_state_on_exact_data_stays_near_zero() {
    let session = synthetic_session(0.03);
    let config = OfflineBaConfig {
        tighten_from_iteration: 100,
        calibration: OfflineCalibrationConfig {
            gyro_intrinsics: Some(GyroIntrinsicsPrior { sigma: 0.005 }),
            camera_focal: Some(FocalSlicePrior {
                step_sigma: 5e-4,
                value_sigma: 0.01,
            }),
            ..OfflineCalibrationConfig::default()
        },
        ..OfflineBaConfig::default()
    };
    let result = solve(&session, &config);
    assert!(result.report.final_cost < 1e-3 * result.report.initial_cost);
    let gyro = result.calibration.gyro_intrinsics.unwrap();
    assert!(
        gyro.scale.length() < 1e-4 && gyro.shear.length() < 1e-4,
        "{gyro:?}"
    );
    assert!(
        result.calibration.camera_focal[0]
            .iter()
            .all(|s| s.abs() < 1e-4),
        "{:?}",
        result.calibration.camera_focal
    );
}

#[test]
fn an_older_dump_is_refused() {
    let mut session = synthetic_session(0.0);
    session.version = SESSION_DUMP_VERSION + 1;
    let _global = faer_global();
    let error = offline_visual_inertial_ba(
        &session,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
        SampleInterpolation::Hold,
        &OfflineBaConfig::default(),
    );
    assert!(matches!(error, Err(OfflineBaError::VersionMismatch(_))));
}

/// E10: a run that integrated with `Midpoint` gets `Midpoint` edges offline.
/// On a stream read at the stamps, the trapezoid leaves `O(dt²)` of the
/// start's IMU cost; the right Riemann sum lags half a sample. Red on the
/// old code (2026-11-18), which always held: 1.47 against 2.1e-4.
#[test]
fn offline_edges_integrate_as_the_run_did() {
    let session = sampled_session(0.0, Sampling::Instantaneous);
    let config = OfflineBaConfig {
        max_iterations: 0,
        ..OfflineBaConfig::default()
    };
    let cost = |interpolation| {
        solve_with(&session, interpolation, &config)
            .report
            .initial_cost
    };
    let (hold, midpoint) = (
        cost(SampleInterpolation::Hold),
        cost(SampleInterpolation::Midpoint),
    );
    assert!(
        midpoint < 1e-2 * hold,
        "start cost {midpoint:.3e} with Midpoint, {hold:.3e} with Hold"
    );
}

/// D3: with the clock state on, the IMU cost at the true states, as a
/// function of `td`, is least at the true offset (zero) on a stream read at
/// the stamps. Red on the old code (2026-11-18): the right Riemann sum lags
/// by half a sample, so the cost was least at +2.45 ms.
#[test]
fn the_clock_state_does_not_absorb_the_integration_lag() {
    let session = sampled_session(0.0, Sampling::Instantaneous);
    let calibration = OfflineCalibrationConfig {
        clock_offset: Some(ClockOffsetPrior { sigma_ms: 2.0 }),
        ..OfflineCalibrationConfig::default()
    };
    let layout = Layout::new(&calibration, 1);
    let (problem, mut state) = Problem::new(
        &session,
        VisualInformation::Unit,
        StereoDepthSigma::default(),
        &layout,
    )
    .unwrap();
    let (edges, _) = build_edges(&problem, false);
    let imu = ImuModel::new(
        &problem,
        OfflineImuKernel::None,
        &layout,
        SampleInterpolation::Hold,
    );
    let clock = layout.clock().unwrap();
    let cost_at = |state: &mut State, td_ms: f64| {
        state.calibration.values[clock] = td_ms * 1e-3;
        let pims = imu.integrate_all(&edges, state, &layout);
        edges
            .iter()
            .zip(&pims)
            .map(|(edge, pim)| imu.cost(edge, pim, state))
            .sum::<f64>()
    };
    let (best_ms, _) = (-50..=50)
        .map(|k| k as f64 * 0.1)
        .map(|td_ms| (td_ms, cost_at(&mut state, td_ms)))
        .fold((f64::NAN, f64::INFINITY), |best, (td, c)| {
            if c < best.1 { (td, c) } else { best }
        });
    assert!(best_ms.abs() < 0.2, "IMU cost least at td {best_ms:.2} ms");
}

/// D14: the yaw gauge row is the derivative of `up · Log(R₀ᵀ R)` under the
/// solver's right perturbation, `J_r⁻ᵀ(φ) up`, also at a tilted keyframe
/// (roll and pitch are free). Red on the old code (2026-11-18), whose row was
/// `up`: its ω_x entry was 0 against a derivative of −1.47e3 (0.15 of
/// `1/σ_yaw`) at a 0.4 rad tilt.
#[test]
fn the_yaw_gauge_row_is_its_residual_derivative() {
    let session = synthetic_session(0.0);
    let layout = Layout::new(&OfflineCalibrationConfig::default(), 1);
    let (problem, start) = Problem::new(
        &session,
        VisualInformation::Unit,
        StereoDepthSigma::default(),
        &layout,
    )
    .unwrap();
    let observers = problem.observers();
    let anchors = problem.gauge_anchors(&[], &observers, &start);
    let anchor = &anchors[0];
    let step = |d: usize, h: f64| {
        let mut dx = vec![0.0; 15 * start.keyframes.len()];
        dx[d] = h;
        dx
    };
    let landmarks = vec![Vec3F64::ZERO; start.landmarks.len()];
    let mut tilt = step(3, 0.4);
    tilt[4..6].copy_from_slice(&[0.1, -0.3]);
    let tilted = start.retracted(&tilt, &landmarks, &[]);
    let (_, rows) = problem.gauge_rows(anchor, &tilted);
    let h = 1e-6;
    for (d, &analytic) in rows[3].1.iter().enumerate().skip(3) {
        let at = |h: f64| {
            let state = tilted.retracted(&step(d, h), &landmarks, &[]);
            problem.gauge_rows(anchor, &state).1[3].0
        };
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        assert!(
            (numeric - analytic).abs() <= 1e-4 * numeric.abs().max(1.0),
            "ω{}: numeric {numeric:.6e}, row {analytic:.6e}",
            d - 3
        );
    }
}

/// D8: a solve that converges before `tighten_from_iteration` still runs
/// the 3 px stage. Red on the old code (2026-11-18): it broke out first, and
/// a 5 px outlier stayed in.
#[test]
fn the_tight_stage_runs_even_after_an_early_convergence() {
    let mut session = synthetic_session(0.03);
    session.keyframes[4].observations[7].u += 5.0;
    let config = OfflineBaConfig {
        relative_cost_tolerance: 1e-4,
        tighten_from_iteration: 50,
        ..OfflineBaConfig::default()
    };
    let report = solve(&session, &config).report;
    assert!(report.iterations < 50, "{} iterations", report.iterations);
    assert_eq!(report.dropped_outlier, 0);
    assert_eq!(report.dropped_tight_outlier, 1);
}

/// D2: the solve observes gravity. Told the wrong sign, it cannot reach the
/// cost of the truth on a rotating walk. Red on the walk that never turns
/// (2026-11-18): `b_a = −2g` absorbed the flip, cost 1.14e-8 against
/// 1.13e-8 at the truth; turning, 6.5e2 against 1.5e-8.
#[test]
fn a_flipped_gravity_is_not_absorbed_by_the_accel_bias() {
    let config = OfflineBaConfig {
        tighten_from_iteration: 100,
        ..OfflineBaConfig::default()
    };
    let truth = solve(&sampled_session(0.0, Sampling::Rotating), &config).report;
    let mut flipped = sampled_session(0.0, Sampling::Rotating);
    flipped.gravity_world = GRAVITY.map(|g| -g);
    let flipped = solve(&flipped, &config).report;
    assert!(
        flipped.final_cost > 1e3 * truth.final_cost.max(1e-6),
        "final cost {:.3e} with gravity flipped, {:.3e} at the truth",
        flipped.final_cost,
        truth.final_cost
    );
}

/// D2: from a perturbed start the rotating walk converges to the truth.
#[test]
fn a_perturbed_rotating_start_converges_to_the_truth() {
    let session = sampled_session(0.03, Sampling::Rotating);
    let config = OfflineBaConfig {
        tighten_from_iteration: 100,
        ..OfflineBaConfig::default()
    };
    let result = solve(&session, &config);
    assert!(result.report.final_cost < 1e-6 * result.report.initial_cost);
    for (pose, kf) in result.body_poses.iter().zip(&session.keyframes) {
        let truth = body_position(kf.timestamp_sec);
        let error = (pose.inverse().translation - truth).length();
        assert!(
            error < 1e-4,
            "keyframe at {} s off by {error} m",
            kf.timestamp_sec
        );
    }
}

/// D12: the solved `td` is a residual on top of the offset the dump already
/// applied, so the report also prints their sum, the offset to apply next.
/// The old code (2026-11-18) printed the residual alone, and
/// `imu_time_offset_ns` was never read.
#[test]
fn the_report_adds_the_applied_clock_offset() {
    let mut session = synthetic_session(0.0);
    session.rig.imu_time_offset_ns = 2_000_000;
    let config = OfflineBaConfig {
        max_iterations: 0,
        calibration: OfflineCalibrationConfig {
            clock_offset: Some(ClockOffsetPrior { sigma_ms: 2.0 }),
            ..OfflineCalibrationConfig::default()
        },
        ..OfflineBaConfig::default()
    };
    let report = solve(&session, &config).report;
    let td = report.td_ms.unwrap();
    assert_eq!(report.td_total_ms, Some(2.0 + td));
    assert!(
        report
            .to_string()
            .contains(&format!("td_ms={td:.4} td_total_ms={:.4}", 2.0 + td))
    );
}

/// A-8: the solve runs sequential and leaves faer's global parallelism as
/// it found it, on success and on an error. Red on the old code
/// (2026-11-21): it stayed `Par::Seq`.
#[test]
fn offline_ba_restores_global_parallelism() {
    let _global = faer_global();
    let previous = faer::get_global_parallelism();
    let threaded = faer::Par::rayon(3);
    faer::set_global_parallelism(threaded);
    let session = synthetic_session(0.03);
    let solved = offline_visual_inertial_ba(
        &session,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
        SampleInterpolation::Hold,
        &OfflineBaConfig::default(),
    );
    let after_solve = faer::get_global_parallelism();
    let mut older = session;
    older.version = SESSION_DUMP_VERSION + 1;
    let refused = offline_visual_inertial_ba(
        &older,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
        SampleInterpolation::Hold,
        &OfflineBaConfig::default(),
    );
    let after_error = faer::get_global_parallelism();
    faer::set_global_parallelism(previous);
    assert!(solved.is_ok());
    assert!(refused.is_err());
    assert_eq!(after_solve, threaded);
    assert_eq!(after_error, threaded);
}
