//! Per-frame motion prediction: where the camera is expected to be before the
//! projection search runs.
//!
//! Two models, in priority order. With an initialized IMU the preintegrated
//! window propagates both pose and metric world velocity. Otherwise, and when
//! the IMU window is empty, the last visual inter-frame motion is applied
//! again — a constant-velocity assumption.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::PreintegratedImu;

use crate::tracking::pose_estimation::pose_inertial::{Anchor, FrameNavState, NavPrior};

/// Preintegration prepared by the system, which owns bias and sample buffering.
pub(crate) struct InertialPrediction<'a> {
    pub preintegrated: &'a PreintegratedImu,
    /// Camera-to-body transform `T_BC`, or `None` for a camera-frame body.
    pub camera_to_body: Option<Pose3d>,
    pub gravity_world: Vec3F64,
}

/// Where a frame's IMU edge starts, with the preintegration from there to the frame.
pub(crate) struct InertialAnchor<'a> {
    pub origin: AnchorOrigin<'a>,
    pub preintegrated: PreintegratedImu,
}

pub(crate) enum AnchorOrigin<'a> {
    /// The last keyframe, held at its bundle-adjusted state.
    Keyframe(FrameNavState),
    /// The previous frame, under the prior its own solve left.
    PreviousFrame(&'a NavPrior),
}

impl InertialAnchor<'_> {
    pub(crate) fn as_anchor(&self) -> Anchor<'_> {
        match &self.origin {
            AnchorOrigin::Keyframe(state) => Anchor::Fixed(state),
            AnchorOrigin::PreviousFrame(prior) => Anchor::Prior(prior),
        }
    }

    /// The anchor state propagated through the preintegration, keeping its bias.
    pub(crate) fn predict(
        &self,
        gravity_world: Vec3F64,
        camera_to_body: Option<Pose3d>,
    ) -> FrameNavState {
        let anchor = match &self.origin {
            AnchorOrigin::Keyframe(state) => state,
            AnchorOrigin::PreviousFrame(prior) => &prior.mean,
        };
        let (pose_cw, velocity) = predict_pose_imu(
            anchor.pose_cw,
            anchor.velocity,
            gravity_world,
            &self.preintegrated,
            camera_to_body,
        );
        FrameNavState {
            pose_cw,
            velocity,
            bias: anchor.bias,
        }
    }
}

/// Predicted world-to-camera pose for this frame, plus the propagated world
/// velocity when the inertial model produced one.
///
/// A `None` velocity means the inertial model did not run and the caller must
/// leave its velocity untouched — an empty IMU window falls back to the visual
/// model rather than reporting a zero velocity.
pub(crate) fn predict_pose(
    pose_before: Pose3d,
    visual_velocity: Option<Pose3d>,
    velocity_world: Vec3F64,
    inertial: Option<InertialPrediction<'_>>,
) -> (Pose3d, Option<Vec3F64>) {
    if let Some(inertial) = inertial
        && inertial.preintegrated.dt > 0.0
    {
        let (pose, velocity) = predict_pose_imu(
            pose_before,
            velocity_world,
            inertial.gravity_world,
            inertial.preintegrated,
            inertial.camera_to_body,
        );
        return (pose, Some(velocity));
    }
    (predict_pose_visual(pose_before, visual_velocity), None)
}

/// Applies the last inter-frame motion again.
fn predict_pose_visual(pose_before: Pose3d, visual_velocity: Option<Pose3d>) -> Pose3d {
    visual_velocity
        .map(|v| v.compose(&pose_before))
        .unwrap_or(pose_before)
}

/// Propagates the camera pose and body velocity through one preintegrated IMU
/// window.
fn predict_pose_imu(
    pose_w2c: Pose3d,
    vel_world: Vec3F64,
    gravity_world: Vec3F64,
    preint: &PreintegratedImu,
    camera_to_body: Option<Pose3d>,
) -> (Pose3d, Vec3F64) {
    let body_to_world = body_to_world(&pose_w2c, camera_to_body);
    let (r_j, v_j, p_j) = preint.predict(
        &body_to_world.rotation,
        &vel_world,
        &body_to_world.translation,
        &gravity_world,
    );

    let pred_body_to_world = Pose3d::from_rt(r_j, p_j);
    let pred_cam_to_world = match &camera_to_body {
        Some(t_bc) => pred_body_to_world.compose(t_bc),
        None => pred_body_to_world,
    };
    (pred_cam_to_world.inverse(), v_j)
}

fn body_to_world(pose_w2c: &Pose3d, camera_to_body: Option<Pose3d>) -> Pose3d {
    let cam_to_world = pose_w2c.inverse();
    match &camera_to_body {
        Some(t_bc) => cam_to_world.compose(&t_bc.inverse()),
        None => cam_to_world,
    }
}

#[cfg(test)]
mod tests {
    use super::{AnchorOrigin, InertialAnchor, InertialPrediction, predict_pose};
    use crate::tracking::pose_estimation::pose_inertial::FrameNavState;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_factors::imu;
    use kornia_factors::state::NavState;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

    fn preintegration(dt: f64) -> PreintegratedImu {
        let mut preintegrated = PreintegratedImu::new(
            ImuBias::default(),
            ImuCalib {
                gyro_noise: 1e-4,
                accel_noise: 1e-3,
                gyro_bias_noise: 1e-5,
                accel_bias_noise: 1e-4,
            },
        );
        preintegrated.dt = dt;
        preintegrated
    }

    /// The IMU factor must read zero at the state the prediction produced, for
    /// a 1 kHz window, a non-trivial extrinsic and a pose far from the origin.
    #[test]
    fn the_imu_factor_is_zero_at_the_prediction() {
        let bias = ImuBias {
            gyro: Vec3F64::new(0.002, -0.001, 0.003),
            accel: Vec3F64::new(0.1, -0.2, 0.05),
        };
        let calib = ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        };
        let samples: Vec<ImuMeasurement> = (0..=50)
            .map(|k| {
                let t = 100.0 + f64::from(k) * 1e-3;
                ImuMeasurement {
                    timestamp: t,
                    gyro: Vec3F64::new(0.3, -0.5, 0.2 + 0.1 * (40.0 * t).sin()),
                    accel: Vec3F64::new(-9.5, 0.4 + 0.2 * (30.0 * t).cos(), 1.8),
                }
            })
            .collect();
        let preintegrated =
            PreintegratedImu::from_measurements(bias, calib, &samples, 100.0, 100.05);
        let t_bc = Pose3d::new(
            SO3F64::exp(Vec3F64::new(1.2, -0.4, 2.0)).matrix(),
            Vec3F64::new(0.05, -0.1, 0.02),
        );
        let state = FrameNavState {
            pose_cw: Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.4, 2.1, -0.7)).matrix(),
                Vec3F64::new(12.0, -30.0, 7.0),
            ),
            velocity: Vec3F64::new(0.8, -0.3, 0.4),
            bias,
        };
        let gravity = Vec3F64::new(0.3, 9.7, -1.2);
        let anchor = InertialAnchor {
            origin: AnchorOrigin::Keyframe(state),
            preintegrated,
        };

        let predicted = anchor.predict(gravity, Some(t_bc));

        fn nav(s: &FrameNavState) -> NavState<'_> {
            NavState {
                pose_cw: &s.pose_cw,
                velocity: s.velocity,
                bias: s.bias,
            }
        }
        let (residual, _, _) = imu::residual_and_jacobians(
            nav(&state),
            nav(&predicted),
            &anchor.preintegrated,
            &gravity,
            Some(&t_bc),
        );
        let omega = imu::information(&anchor.preintegrated, 1.0).omega;
        let chi2 = residual.dot(&(omega * residual));
        assert!(chi2 < 1e-6, "IMU χ² at the prediction: {chi2}");
    }

    fn assert_pose_close(actual: Pose3d, expected: Pose3d) {
        assert!((actual.translation - expected.translation).length() < 1e-10);
        for (actual, expected) in actual
            .rotation
            .to_cols_array()
            .iter()
            .zip(expected.rotation.to_cols_array())
        {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    /// Without an inertial window the last inter-frame motion is applied again.
    #[test]
    fn visual_model_applies_the_last_inter_frame_motion() {
        let pose_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.1, 0.0)).matrix(),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let velocity = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(0.2, 0.0, 0.0));

        let (pose, predicted_velocity) =
            predict_pose(pose_before, Some(velocity), Vec3F64::ZERO, None);

        assert_pose_close(pose, velocity.compose(&pose_before));
        assert!(predicted_velocity.is_none());
    }

    /// With no visual motion recorded yet, the pose is carried unchanged.
    #[test]
    fn visual_model_without_velocity_holds_the_pose() {
        let pose_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.2, 0.0, 0.0)).matrix(),
            Vec3F64::new(0.0, 1.0, 0.0),
        );

        let (pose, predicted_velocity) = predict_pose(pose_before, None, Vec3F64::ZERO, None);

        assert_pose_close(pose, pose_before);
        assert!(predicted_velocity.is_none());
    }

    /// An empty IMU window must fall back to the visual model and report no
    /// velocity, rather than propagating a zero-length preintegration.
    #[test]
    fn stalled_imu_window_falls_back_to_the_visual_model() {
        let pose_before = Pose3d::IDENTITY;
        let velocity = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(0.3, 0.0, 0.0));
        let preintegrated = preintegration(0.0);

        let (pose, predicted_velocity) = predict_pose(
            pose_before,
            Some(velocity),
            Vec3F64::new(5.0, 5.0, 5.0),
            Some(InertialPrediction {
                preintegrated: &preintegrated,
                camera_to_body: None,
                gravity_world: Vec3F64::new(0.0, 0.0, -9.81),
            }),
        );

        assert_pose_close(pose, velocity.compose(&pose_before));
        assert!(
            predicted_velocity.is_none(),
            "a stalled window must leave the caller's velocity untouched"
        );
    }

    /// A non-empty window predicts through the IMU and reports a velocity, in
    /// preference to the visual model.
    #[test]
    fn inertial_model_takes_priority_and_reports_velocity() {
        let preintegrated = preintegration(0.05);
        let visual = Pose3d::new(SO3F64::IDENTITY.matrix(), Vec3F64::new(9.0, 0.0, 0.0));

        let (pose, predicted_velocity) = predict_pose(
            Pose3d::IDENTITY,
            Some(visual),
            Vec3F64::new(1.0, 0.0, 0.0),
            Some(InertialPrediction {
                preintegrated: &preintegrated,
                camera_to_body: None,
                gravity_world: Vec3F64::new(0.0, 0.0, -9.81),
            }),
        );

        assert!(predicted_velocity.is_some());
        assert!(
            (pose.translation - visual.compose(&Pose3d::IDENTITY).translation).length() > 1e-6,
            "the visual model must not have been used"
        );
    }

    /// The camera-to-body extrinsic is applied on the way into the body frame
    /// and taken back out on the way to the camera pose. With no motion and no
    /// gravity it must therefore round-trip exactly, for any extrinsic.
    #[test]
    fn camera_to_body_extrinsic_round_trips_through_the_prediction() {
        let preintegrated = preintegration(0.02);
        let t_bc = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, 0.3)).matrix(),
            Vec3F64::new(0.1, -0.05, 0.02),
        );
        let pose_before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.0)).matrix(),
            Vec3F64::new(0.4, 0.3, -0.2),
        );

        let (pose, velocity) = predict_pose(
            pose_before,
            None,
            Vec3F64::ZERO,
            Some(InertialPrediction {
                preintegrated: &preintegrated,
                camera_to_body: Some(t_bc),
                gravity_world: Vec3F64::ZERO,
            }),
        );

        assert_pose_close(pose, pose_before);
        assert_eq!(velocity, Some(Vec3F64::ZERO));
    }
}
