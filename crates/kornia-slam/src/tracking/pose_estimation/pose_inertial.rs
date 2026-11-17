//! Pose-inertial solve of one frame: reprojection rows plus an IMU edge from
//! an anchor, after ORB-SLAM3's `PoseInertialOptimizationLastKeyFrame` and
//! `...LastFrame` (`Optimizer.cc:4491-5289`); the idea of PR #195, credit Astik.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, Vec2F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

use crate::mapping::vi_ba_schur::RigObservationCamera;

/// The inertial state of one frame: world-to-camera pose, world velocity and
/// IMU bias.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameNavState {
    pub pose_cw: Pose3d,
    pub velocity: Vec3F64,
    pub bias: ImuBias,
}

/// A Gaussian prior on a frame's state, information layout `[ρ ω v b_g b_a]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NavPrior {
    pub mean: FrameNavState,
    pub information: Mat15F64,
}

/// What the IMU edge starts from: a keyframe held fixed, or the previous
/// frame free under its prior.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Anchor<'a> {
    Fixed(&'a FrameNavState),
    Prior(&'a NavPrior),
}

/// One reprojection row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoseInertialObservation {
    pub point_w: Vec3F64,
    /// Undistorted pixel.
    pub pixel: Vec2F64,
    /// Inverse pixel variance.
    pub information: f64,
    /// 0 for the primary camera, `k` for rig view `k - 1`.
    pub view: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct PoseInertialSolution {
    pub state: FrameNavState,
    /// The 15×15 marginal of the solved frame, the next frame's prior; `None`
    /// when it was not positive definite.
    pub prior: Option<NavPrior>,
    pub inliers: Vec<bool>,
    pub imu_chi2: f64,
}

/// Solves the frame's pose, velocity and bias from `observations` and the IMU
/// edge `preintegrated` from `anchor`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_pose_inertial(
    _anchor: Anchor<'_>,
    _preintegrated: &PreintegratedImu,
    _gravity_world: Vec3F64,
    _camera_to_body: Option<Pose3d>,
    _cameras: &[RigObservationCamera],
    _primary: &PinholeCamera,
    _initial: FrameNavState,
    _observations: &[PoseInertialObservation],
    _monocular: bool,
) -> PoseInertialSolution {
    todo!("lane A: pose-inertial solve")
}
