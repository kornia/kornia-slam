//! Window variables with their first-estimate linearization points, after
//! Basalt's `PoseVelBiasStateWithLin` and `PoseStateWithLin`
//! (`basalt/utils/imu_types.h:112-138, 209-216`).

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SVecF64, Vec3F64};
use kornia_sensors::imu::ImuBias;

use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

/// A frame's identity in the window: its timestamp in nanoseconds.
pub(crate) type FrameId = i64;

/// Tangent size of a frame state, `[ρ ω v b_g b_a]`.
pub(crate) const STATE_DOF: usize = 15;
/// Tangent size of a keyframe pose, `[ρ ω]`.
pub(crate) const POSE_DOF: usize = 6;

/// A 15-DOF frame state. Free, an increment moves `lin`; linearized (in the
/// marginal prior), `lin` is fixed and increments add to `delta`, the value
/// being `lin ⊞ delta`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateWithLin {
    lin: FrameNavState,
    delta: SVecF64<STATE_DOF>,
    linearized: bool,
}

impl StateWithLin {
    pub(crate) fn new(state: FrameNavState) -> Self {
        Self {
            lin: state,
            delta: SVecF64::zeros(),
            linearized: false,
        }
    }

    /// The current value.
    pub(crate) fn value(&self) -> FrameNavState {
        if self.linearized {
            self.lin.plus(&self.delta)
        } else {
            self.lin
        }
    }

    /// Where its Jacobians are evaluated: `lin` once linearized (FEJ), the
    /// value otherwise.
    pub(crate) fn jacobian_point(&self) -> FrameNavState {
        self.lin
    }

    pub(crate) fn is_linearized(&self) -> bool {
        self.linearized
    }

    pub(crate) fn delta(&self) -> &SVecF64<STATE_DOF> {
        &self.delta
    }

    pub(crate) fn apply_increment(&mut self, increment: &SVecF64<STATE_DOF>) {
        if self.linearized {
            self.delta += increment;
        } else {
            self.lin = self.lin.plus(increment);
        }
    }

    /// Fixes the linearization point at the current value; a state already
    /// linearized keeps its point.
    pub(crate) fn set_linearized(&mut self) {
        if !self.linearized {
            self.linearized = true;
            self.delta = SVecF64::zeros();
        }
    }

    /// The 6-DOF pose part, with the same linearization.
    pub(crate) fn to_pose(self) -> PoseWithLin {
        PoseWithLin {
            lin: self.lin.pose_cw,
            delta: self.delta.fixed_rows::<POSE_DOF>(0).into_owned(),
            linearized: self.linearized,
        }
    }
}

/// A keyframe pose that left the frame states, `T_cw` of camera 0.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoseWithLin {
    lin: Pose3d,
    delta: SVecF64<POSE_DOF>,
    linearized: bool,
}

impl PoseWithLin {
    #[cfg(test)]
    pub(crate) fn new(pose: Pose3d) -> Self {
        Self {
            lin: pose,
            delta: SVecF64::zeros(),
            linearized: false,
        }
    }

    pub(crate) fn value(&self) -> Pose3d {
        if self.linearized {
            pose_plus(&self.lin, &self.delta)
        } else {
            self.lin
        }
    }

    pub(crate) fn jacobian_point(&self) -> Pose3d {
        self.lin
    }

    pub(crate) fn is_linearized(&self) -> bool {
        self.linearized
    }

    pub(crate) fn delta(&self) -> &SVecF64<POSE_DOF> {
        &self.delta
    }

    pub(crate) fn apply_increment(&mut self, increment: &SVecF64<POSE_DOF>) {
        if self.linearized {
            self.delta += increment;
        } else {
            self.lin = pose_plus(&self.lin, increment);
        }
    }

    pub(crate) fn set_linearized(&mut self) {
        if !self.linearized {
            self.linearized = true;
            self.delta = SVecF64::zeros();
        }
    }
}

/// `T_cw · Exp(ρ, ω)`, as [`FrameNavState::plus`] moves the pose.
pub(crate) fn pose_plus(pose: &Pose3d, delta: &SVecF64<POSE_DOF>) -> Pose3d {
    let mut full = SVecF64::<STATE_DOF>::zeros();
    full.fixed_rows_mut::<POSE_DOF>(0).copy_from(delta);
    FrameNavState {
        pose_cw: *pose,
        velocity: Vec3F64::ZERO,
        bias: ImuBias::default(),
    }
    .plus(&full)
    .pose_cw
}
