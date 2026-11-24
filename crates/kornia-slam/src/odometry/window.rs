//! The window's variables, factors and prior, which the linearization and
//! the solve read (Basalt `sqrt_keypoint_vio.cpp`).

use std::collections::BTreeMap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, Vec3F64};
use kornia_sensors::imu::PreintegratedImu;

use super::config::OdometryTuning;
use super::landmarks::LandmarkStore;
use super::linearize::{Order, VarKind};
use super::marginalize::{MargPrior, PriorEntry};
use super::state::{FrameId, PoseWithLin, StateWithLin};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

/// One camera of the rectified pair: intrinsics and `T_c,c0`.
#[derive(Debug, Clone)]
pub(crate) struct StereoView {
    pub camera: PinholeCamera,
    pub camera_from_primary: Pose3d,
}

/// An IMU factor between consecutive states, with its information.
#[derive(Debug, Clone)]
pub(crate) struct ImuEdge {
    pub to: FrameId,
    pub pim: PreintegratedImu,
    pub omega: Mat15F64,
}

pub(crate) struct Window {
    pub(super) tuning: OdometryTuning,
    pub(super) cameras: [StereoView; 2],
    pub(super) t_bc: Pose3d,
    pub(super) gravity: Vec3F64,
    pub(super) states: BTreeMap<FrameId, StateWithLin>,
    pub(super) poses: BTreeMap<FrameId, PoseWithLin>,
    /// Keyed by the edge's first state.
    pub(super) imu: BTreeMap<FrameId, ImuEdge>,
    pub(super) landmarks: LandmarkStore,
    pub(super) prior: MargPrior,
}

impl Window {
    pub(crate) fn new(
        tuning: &OdometryTuning,
        cameras: [StereoView; 2],
        t_bc: Pose3d,
        gravity: Vec3F64,
    ) -> Self {
        Self {
            tuning: *tuning,
            cameras,
            t_bc,
            gravity,
            states: BTreeMap::new(),
            poses: BTreeMap::new(),
            imu: BTreeMap::new(),
            landmarks: LandmarkStore::default(),
            prior: MargPrior::zero(0),
        }
    }

    /// Clears the window and starts it at frame `id` in `state`, linearized
    /// under the first state's prior.
    pub(crate) fn initialize_at(&mut self, id: FrameId, state: FrameNavState) {
        self.states.clear();
        self.poses.clear();
        self.imu.clear();
        self.landmarks = LandmarkStore::default();
        let mut first = StateWithLin::new(state);
        first.set_linearized();
        self.states.insert(id, first);
        let g = self.gravity * (1.0 / self.gravity.length());
        self.prior = MargPrior::initial(
            id,
            g.to_array(),
            self.tuning.gauge_weight,
            self.tuning.initial_gyro_bias_sigma,
            self.tuning.initial_accel_bias_sigma,
        );
    }

    /// Camera 0's current world-to-camera pose of window frame `id`.
    pub(super) fn frame_pose(&self, id: FrameId) -> Option<Pose3d> {
        self.states
            .get(&id)
            .map(|s| s.value().pose_cw)
            .or_else(|| self.poses.get(&id).map(PoseWithLin::value))
    }

    /// Camera 0's pose at its Jacobian (first-estimate) point.
    pub(super) fn frame_jacobian_pose(&self, id: FrameId) -> Option<Pose3d> {
        self.states
            .get(&id)
            .map(|s| s.jacobian_point().pose_cw)
            .or_else(|| self.poses.get(&id).map(PoseWithLin::jacobian_point))
    }

    /// The variable order: keyframe poses by time, then states by time.
    pub(super) fn order(&self, order: &mut Order) {
        order.clear();
        for &id in self.poses.keys() {
            order.push(id, VarKind::Pose);
        }
        for &id in self.states.keys() {
            order.push(id, VarKind::State);
        }
    }

    /// The deltas of `prior`'s variables, in its order.
    pub(super) fn prior_delta(&self, prior: &MargPrior) -> Vec<f64> {
        let mut delta = Vec::with_capacity(prior.dim());
        for entry in &prior.order {
            self.push_delta(entry, &mut delta);
        }
        delta
    }

    fn push_delta(&self, entry: &PriorEntry, delta: &mut Vec<f64>) {
        let dim = entry.kind.dim();
        if let Some(state) = self.states.get(&entry.id) {
            delta.extend(state.delta().iter().take(dim));
        } else if let Some(pose) = self.poses.get(&entry.id) {
            delta.extend(pose.delta().iter().take(dim));
        } else {
            delta.extend(std::iter::repeat_n(0.0, dim));
        }
    }
}
