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

/// Preintegration prepared by the system, which owns bias and sample buffering.
pub(crate) struct InertialPrediction<'a> {
    pub preintegrated: &'a PreintegratedImu,
    /// Camera-to-body transform `T_BC`, or `None` for a camera-frame body.
    pub camera_to_body: Option<Pose3d>,
    pub gravity_world: Vec3F64,
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
    use super::{InertialPrediction, predict_pose};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_sensors::imu::{ImuBias, ImuCalib, PreintegratedImu};

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
