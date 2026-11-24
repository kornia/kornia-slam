//! The first state, from the first accelerometer sample (Basalt
//! `sqrt_keypoint_vio.cpp:185-220`): attitude rotating the measured specific
//! force onto world +z, at rest at the world origin, biases zero.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::imu::ImuBias;

use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

/// `R_wb` taking the unit specific force `accel` (body frame) onto `+z`, by
/// the shortest rotation; `None` for a zero or non-finite reading.
pub(crate) fn attitude_from_accel(accel: Vec3F64) -> Option<kornia_algebra::Mat3F64> {
    let norm = accel.length();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    let a = accel * (1.0 / norm);
    let z = Vec3F64::new(0.0, 0.0, 1.0);
    let axis = a.cross(z);
    let sin = axis.length();
    let cos = a.dot(z);
    let rotation = if sin < 1e-12 {
        if cos > 0.0 {
            Vec3F64::ZERO
        } else {
            Vec3F64::new(std::f64::consts::PI, 0.0, 0.0)
        }
    } else {
        axis * (sin.atan2(cos) / sin)
    };
    Some(SO3F64::exp(rotation).matrix())
}

/// The first frame's state for camera-to-body `t_bc`: the body at the world
/// origin with attitude [`attitude_from_accel`], at rest, zero biases.
pub(crate) fn first_state(t_bc: &Pose3d, accel: Vec3F64) -> Option<FrameNavState> {
    let r_wb = attitude_from_accel(accel)?;
    let pose_bw = Pose3d::new(r_wb.transpose(), Vec3F64::ZERO);
    Some(FrameNavState {
        pose_cw: t_bc.inverse().compose(&pose_bw),
        velocity: Vec3F64::ZERO,
        bias: ImuBias::default(),
    })
}
