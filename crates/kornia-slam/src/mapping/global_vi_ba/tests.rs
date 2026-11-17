use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};

use super::*;
use crate::mapping::session::{
    ImuSample, SESSION_DUMP_VERSION, SessionCamera, SessionDump, SessionFrame, SessionKeyframe,
    SessionLandmark, SessionObservation, SessionPose, SessionRig,
};

const FX: f64 = 400.0;
const CX: f64 = 320.0;
const CY: f64 = 240.0;
const GRAVITY: [f64; 3] = [0.0, 9.81, 0.0];
/// Start velocity (m/s) and constant acceleration (m/s²) of the synthetic
/// walk, which never rotates; the acceleration makes the scale observable.
const VELOCITY: [f64; 3] = [0.5, 0.0, 0.0];
const ACCELERATION: [f64; 3] = [0.4, 0.0, 0.2];

fn body_position(t: f64) -> Vec3F64 {
    Vec3F64::from_array(VELOCITY) * t + Vec3F64::from_array(ACCELERATION) * (0.5 * t * t)
}

fn body_velocity(t: f64) -> [f64; 3] {
    (Vec3F64::from_array(VELOCITY) + Vec3F64::from_array(ACCELERATION) * t).to_array()
}

/// A camera at the IMU accelerating past a wall of landmarks, with exact
/// pixels and an exact IMU stream (specific force `a - g`, no rotation). `offset` moves the dumped landmarks and every
/// keyframe but the first off the truth.
fn synthetic_session(offset: f64) -> SessionDump {
    let landmarks: Vec<Vec3F64> = (0..5)
        .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| (i, j, k))))
        .map(|(i, j, k)| Vec3F64::new(i as f64 - 1.0, j as f64 - 1.0, 4.0 + 2.0 * k as f64))
        .collect();
    let keyframes = (0..9)
        .map(|k| {
            let t = 0.25 * k as f64;
            let p = body_position(t);
            let observations = landmarks
                .iter()
                .enumerate()
                .map(|(l, x)| {
                    let c = *x - p;
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
                pose_cw: SessionPose::from_pose(&Pose3d::new(Mat3F64::IDENTITY, -(p + shift))),
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
    let imu = (0..=400)
        .map(|i| ImuSample {
            t: i as f64 * 0.005,
            gyro: [0.0; 3],
            accel: [
                ACCELERATION[0] - GRAVITY[0],
                ACCELERATION[1] - GRAVITY[1],
                ACCELERATION[2] - GRAVITY[2],
            ],
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
    offline_visual_inertial_ba(
        session,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
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
    let error = offline_visual_inertial_ba(
        &session,
        &VisualInformation::Unit,
        StereoDepthSigma::default(),
        &OfflineBaConfig::default(),
    );
    assert!(matches!(error, Err(OfflineBaError::VersionMismatch(_))));
}
