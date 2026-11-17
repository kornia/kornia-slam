//! Upstream target: kornia_3d::pnp::refine. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pnp::LMRefineParams;
use kornia_algebra::{Mat3AF32, Mat3F64, SMatF64, Vec2F32, Vec3AF32, Vec3F64};

/// Rows `d(u)/d(p_c)` and `d(v)/d(p_c)` of a pinhole projection at the camera-frame
/// point `point_cam` (`z > 0`), in pixels per metre.
pub fn pinhole_point_jacobian(_fx: f64, _fy: f64, _point_cam: Vec3F64) -> [Vec3F64; 2] {
    todo!("lane C: analytic pose refine")
}

/// Pixel Jacobian with respect to a right perturbation `T_cw · exp(δ)`, `δ = (υ, ω)` (the
/// order of `SE3F64::retract`), of the world-to-camera pose with rotation `rotation_cw`,
/// at world point `point_world` whose camera-frame position is `point_cam`.
pub fn pose_jacobian_2x6(
    _rotation_cw: &Mat3F64,
    _point_world: Vec3F64,
    _point_cam: Vec3F64,
    _fx: f64,
    _fy: f64,
) -> SMatF64<2, 6> {
    todo!("lane C: analytic pose refine")
}

/// Whether [`refine_pose_analytic`] implements `params`' robust kernel (Identity, or Huber
/// with a finite positive scale).
pub fn supports_kernel(_params: &LMRefineParams) -> bool {
    todo!("lane C: analytic pose refine")
}

/// A refined world-to-camera pose and whether a tolerance, not the iteration cap, ended it.
pub struct RefinedPose {
    pub rotation: Mat3AF32,
    pub translation: Vec3AF32,
    pub converged: bool,
}

/// Pose-only LM on undistorted pixels; `None` for an unsupported kernel, fewer than 3
/// points, a start point behind the camera, or a singular damped system.
pub fn refine_pose_analytic(
    _points_world: &[Vec3AF32],
    _points_image: &[Vec2F32],
    _k: &Mat3AF32,
    _rotation: &Mat3AF32,
    _translation: &Vec3AF32,
    _params: &LMRefineParams,
) -> Option<RefinedPose> {
    todo!("lane C: analytic pose refine")
}
