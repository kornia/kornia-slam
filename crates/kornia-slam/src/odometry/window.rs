//! The per-frame sequence of the estimator below the frontend: predict, add
//! observations, keyframe and landmarks, optimize, marginalize (Basalt
//! `sqrt_keypoint_vio.cpp:306-485`).

use std::collections::{BTreeMap, BTreeSet};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, Vec3F64};
use kornia_factors::inertial::preintegration::information;
use kornia_sensors::imu::PreintegratedImu;

use super::OdometryError;
use super::config::OdometryTuning;
use super::init::first_state;
use super::keyframes::{KeyframeCounts, MargPlan, is_keyframe, plan_marginalization};
use super::landmarks::{Landmark, LandmarkId, LandmarkStore, triangulate_from_host};
use super::linearize::{
    Cost, DenseSystem, LandmarkBlock, Order, Rows, VarKind, imu_edge, landmark_rows,
};
use super::marginalize::{Fate, MargPrior, PriorEntry, marginalize_system, nullspace_residual};
use super::solve::{SolveReport, SolveScratch, optimize};
use super::state::{FrameId, PoseWithLin, StateWithLin};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

/// One camera of the rectified pair: intrinsics and `T_c,c0`.
#[derive(Debug, Clone)]
pub(crate) struct StereoView {
    pub camera: PinholeCamera,
    pub camera_from_primary: Pose3d,
}

/// A rectified pixel of landmark (track) `landmark` in camera `camera`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Observation {
    pub landmark: LandmarkId,
    pub camera: u8,
    pub pixel: [f64; 2],
}

/// One frame for the window: its id, the IMU preintegrated from the
/// previous frame at the previous state's bias (`None` on the first frame),
/// and its observations sorted by (camera, landmark).
pub(crate) struct FrameInput<'a> {
    pub id: FrameId,
    pub imu: Option<PreintegratedImu>,
    pub observations: &'a [Observation],
}

/// What one frame did.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameReport {
    pub keyframe: bool,
    pub optimized: bool,
    pub landmarks_created: usize,
    pub solve: Option<SolveReport>,
    pub marginalized: bool,
    pub pseudo_inverse: bool,
    /// `nᵀHn / (‖n‖² ‖H‖_F)` of the nullspace twin, `x y z roll pitch yaw`.
    pub nullspace: Option<[f64; 6]>,
    /// `r_navᵀ Ω r_nav / 9` and `r_biasᵀ Ω r_bias / 6` of the newest IMU
    /// edge after the solve.
    pub imu_chi2_nav: Option<f64>,
    pub bias_walk_chi2: Option<f64>,
    pub nullspace_time: std::time::Duration,
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
    pub(super) keyframes: BTreeSet<FrameId>,
    /// Every observation of each window frame, landmark or not.
    pub(super) frame_observations: BTreeMap<FrameId, Vec<Observation>>,
    pub(super) landmarks: LandmarkStore,
    pub(super) prior: MargPrior,
    /// The prior without the gauge rows, through the same
    /// marginalizations: its nullspace is checked (`:82-84, 823-880`).
    pub(super) twin: MargPrior,
    pub(super) counts: KeyframeCounts,
    pub(super) lost: Vec<LandmarkId>,
    optimization_started: bool,
    frames_after_keyframe: usize,
    scratch: SolveScratch,
    #[cfg(test)]
    pub(super) ledger: Option<super::tests::Ledger>,
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
            keyframes: BTreeSet::new(),
            frame_observations: BTreeMap::new(),
            landmarks: LandmarkStore::default(),
            prior: MargPrior::zero(0),
            twin: MargPrior::zero(0),
            counts: KeyframeCounts::default(),
            lost: Vec::new(),
            optimization_started: false,
            frames_after_keyframe: 0,
            scratch: SolveScratch::default(),
            #[cfg(test)]
            ledger: None,
        }
    }

    /// Clears the window and starts it at frame `id` from its first
    /// accelerometer sample (body frame, m/s²).
    pub(crate) fn initialize(
        &mut self,
        id: FrameId,
        first_accel: Vec3F64,
    ) -> Result<(), OdometryError> {
        let state = first_state(&self.t_bc, first_accel).ok_or(OdometryError::NoGravity)?;
        self.initialize_at(id, state);
        Ok(())
    }

    /// As [`Self::initialize`] from a given first state.
    pub(crate) fn initialize_at(&mut self, id: FrameId, state: FrameNavState) {
        self.states.clear();
        self.poses.clear();
        self.imu.clear();
        self.keyframes.clear();
        self.frame_observations.clear();
        self.landmarks = LandmarkStore::default();
        self.counts = KeyframeCounts::default();
        self.lost.clear();
        self.optimization_started = false;
        self.frames_after_keyframe = 0;
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
        self.twin = MargPrior::zero(id);
        #[cfg(test)]
        if let Some(ledger) = self.ledger.as_mut() {
            ledger.start(id, &self.prior);
        }
    }

    pub(crate) fn newest(&self) -> Option<(FrameId, &StateWithLin)> {
        self.states.iter().next_back().map(|(id, s)| (*id, s))
    }

    pub(crate) fn add_frame(
        &mut self,
        input: FrameInput<'_>,
    ) -> Result<FrameReport, OdometryError> {
        let mut report = FrameReport::default();
        let (newest_id, newest) = self.newest().ok_or(OdometryError::NotInitialized)?;
        let newest = *newest;
        if input.id != newest_id {
            let pim = input.imu.ok_or(OdometryError::ImuGap)?;
            if pim.dt <= 0.0 || input.id <= newest_id {
                return Err(OdometryError::ImuGap);
            }
            self.predict(newest_id, &newest, input.id, pim)?;
        }
        let connected0 = self.add_observations(input.id, input.observations);
        let unconnected0 = input
            .observations
            .iter()
            .filter(|o| o.camera == 0 && !self.landmarks.contains(o.landmark))
            .count();
        self.frames_after_keyframe += 1;
        if self.keyframes.is_empty()
            || is_keyframe(
                connected0,
                unconnected0,
                self.frames_after_keyframe,
                &self.tuning,
            )
        {
            self.keyframes.insert(input.id);
            self.frames_after_keyframe = 0;
            report.keyframe = true;
            report.landmarks_created = self.create_landmarks(input.id);
        }
        self.lost = self.landmarks.lost_in(input.id);

        if self.optimization_started || self.states.len() > self.tuning.min_states_to_optimize {
            self.optimization_started = true;
            let mut scratch = std::mem::take(&mut self.scratch);
            let solve = optimize(self, &mut scratch);
            self.scratch = scratch;
            report.optimized = true;
            report.solve = Some(solve);
            self.edge_chi2(&mut report);
        }
        if !self.is_finite() {
            return Err(OdometryError::NonFinite);
        }
        if let Some(plan) = self.plan() {
            self.marginalize(&plan, &mut report)?;
            report.marginalized = true;
        }
        Ok(report)
    }

    /// The new state from the newest one and `pim`, integrated at its bias.
    fn predict(
        &mut self,
        from: FrameId,
        state: &StateWithLin,
        to: FrameId,
        pim: PreintegratedImu,
    ) -> Result<(), OdometryError> {
        let current = state.value();
        let (r_bw, t_wb) = kornia_factors::inertial::preintegration::body_frame(
            &current.pose_cw,
            Some(&self.t_bc),
        );
        let r_wb = r_bw.transpose();
        let (r_wb_next, v_next, p_next) =
            pim.predict(&r_wb, &current.velocity, &t_wb, &self.gravity);
        let pose_bw = Pose3d::new(r_wb_next.transpose(), -(r_wb_next.transpose() * p_next));
        let next = FrameNavState {
            pose_cw: self.t_bc.inverse().compose(&pose_bw),
            velocity: v_next,
            bias: current.bias,
        };
        let omega = information(&pim, 1.0).omega;
        self.states.insert(to, StateWithLin::new(next));
        self.imu.insert(from, ImuEdge { to, pim, omega });
        Ok(())
    }

    /// Stores the frame's observations and adds those of existing landmarks;
    /// returns how many camera-0 ones have a landmark.
    fn add_observations(&mut self, id: FrameId, observations: &[Observation]) -> usize {
        self.counts.connected.clear();
        let mut connected0 = 0;
        for o in observations {
            if let Some(lm) = self.landmarks.get_mut(o.landmark) {
                lm.observations.insert((id, o.camera), o.pixel);
                *self.counts.connected.entry(lm.host).or_insert(0) += 1;
                connected0 += usize::from(o.camera == 0);
            }
        }
        self.frame_observations.insert(id, observations.to_vec());
        connected0
    }

    /// Landmarks for the keyframe's camera-0 tracks that have none, each
    /// triangulated against its first window observation far enough from
    /// the host, with all its window observations (`:380-465`).
    fn create_landmarks(&mut self, id: FrameId) -> usize {
        let Some(host_cw) = self.frame_pose(id) else {
            return 0;
        };
        let Some(current) = self.frame_observations.get(&id) else {
            return 0;
        };
        let new: Vec<Observation> = current
            .iter()
            .filter(|o| o.camera == 0 && !self.landmarks.contains(o.landmark))
            .copied()
            .collect();
        let mut created = 0;
        for o in new {
            let mut observations = BTreeMap::new();
            for (&frame, list) in &self.frame_observations {
                for camera in 0..2u8 {
                    if let Ok(k) =
                        list.binary_search_by(|x| (x.camera, x.landmark).cmp(&(camera, o.landmark)))
                    {
                        observations.insert((frame, camera), list[k].pixel);
                    }
                }
            }
            let candidates = observations
                .iter()
                .filter_map(|(&(frame, camera), &pixel)| {
                    let pose = self.camera_pose(frame, camera)?;
                    Some((pose, pixel))
                });
            let Some(value) = triangulate_from_host(
                &host_cw,
                o.pixel,
                candidates,
                &self.cameras[0].camera,
                self.tuning.min_triangulation_m,
                self.tuning.max_inverse_distance,
            ) else {
                continue;
            };
            self.landmarks.insert(
                o.landmark,
                Landmark {
                    host: id,
                    value,
                    observations,
                },
            );
            created += 1;
        }
        self.counts.created.insert(id, created);
        created
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

    /// Camera `camera`'s current world-to-camera pose of frame `id`.
    pub(super) fn camera_pose(&self, id: FrameId, camera: u8) -> Option<Pose3d> {
        let view = self.cameras.get(usize::from(camera))?;
        Some(view.camera_from_primary.compose(&self.frame_pose(id)?))
    }

    /// The body origin in the world of window frame `id`.
    fn body_position(&self, id: FrameId) -> Option<Vec3F64> {
        let pose_bw = self.t_bc.compose(&self.frame_pose(id)?);
        Some(pose_bw.inverse().translation)
    }

    fn is_finite(&self) -> bool {
        let pose_finite = |p: &Pose3d| {
            p.rotation.to_cols_array().iter().all(|v| v.is_finite())
                && p.translation.to_array().iter().all(|v| v.is_finite())
        };
        self.states.values().all(|s| {
            let v = s.value();
            pose_finite(&v.pose_cw)
                && v.velocity.to_array().iter().all(|x| x.is_finite())
                && v.bias.gyro.to_array().iter().all(|x| x.is_finite())
                && v.bias.accel.to_array().iter().all(|x| x.is_finite())
        }) && self.poses.values().all(|p| pose_finite(&p.value()))
            && self.prior.b.iter().all(|v| v.is_finite())
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

    fn plan(&self) -> Option<MargPlan> {
        let states: Vec<FrameId> = self.states.keys().copied().collect();
        plan_marginalization(
            &states,
            &self.keyframes,
            &self.counts,
            |id| self.body_position(id),
            self.optimization_started,
            &self.tuning,
        )
    }

    /// The newest edge's normalized chi² at the current states.
    fn edge_chi2(&self, report: &mut FrameReport) {
        let Some((&from, edge)) = self.imu.iter().next_back() else {
            return;
        };
        let (Some(i), Some(j)) = (self.states.get(&from), self.states.get(&edge.to)) else {
            return;
        };
        let (vi, vj) = (i.value(), j.value());
        let (r, _, _) = kornia_factors::inertial::preintegration::residual_and_jacobians(
            vi.nav_state(),
            vj.nav_state(),
            &edge.pim,
            &self.gravity,
            Some(&self.t_bc),
        );
        let nav = r.fixed_rows::<9>(0);
        let bias = r.fixed_rows::<6>(9);
        report.imu_chi2_nav = Some(nav.dot(&(edge.omega.fixed_view::<9, 9>(0, 0) * nav)) / 9.0);
        report.bias_walk_chi2 = Some(bias.dot(&(edge.omega.fixed_view::<6, 6>(9, 9) * bias)) / 6.0);
    }

    /// The twin prior's nullspace, at the linearization points.
    pub(super) fn nullspace(&self) -> [f64; 6] {
        let mut centroid = Vec3F64::ZERO;
        let mut velocities = Vec::with_capacity(self.twin.order.len());
        for entry in &self.twin.order {
            let (pose, velocity) = match self.states.get(&entry.id) {
                Some(s) => (
                    s.jacobian_point().pose_cw,
                    Some(s.jacobian_point().velocity.to_array()),
                ),
                None => match self.poses.get(&entry.id) {
                    Some(p) => (p.jacobian_point(), None),
                    None => (Pose3d::IDENTITY, None),
                },
            };
            centroid += pose.inverse().translation;
            velocities.push(velocity.filter(|_| entry.kind == VarKind::State));
        }
        centroid *= 1.0 / self.twin.order.len().max(1) as f64;
        nullspace_residual(&self.twin, centroid.to_array(), &velocities)
    }
}

impl Window {
    /// Applies `plan` (M7-M10): the joint of the prior, the IMU edges inside
    /// the order and the rows of the landmarks that leave, Schur-complemented
    /// into the new prior; then the bookkeeping.
    fn marginalize(
        &mut self,
        plan: &MargPlan,
        report: &mut FrameReport,
    ) -> Result<(), OdometryError> {
        let lsm = plan.last_state_to_marg;
        let mut order = Order::default();
        for &id in self.poses.keys() {
            order.push(id, VarKind::Pose);
        }
        for &id in self.states.keys().take_while(|&&id| id <= lsm) {
            order.push(id, VarKind::State);
        }
        let mut factors = DenseSystem::default();
        factors.reset(order.dim());
        #[cfg(test)]
        let mut ledger = Vec::new();
        for (&from, edge) in &self.imu {
            let (Some(oi), Some(oj)) = (order.pose_offset(from), order.pose_offset(edge.to)) else {
                continue;
            };
            let Some((_, residual, j_i, j_j)) = imu_edge(self, from, edge) else {
                continue;
            };
            factors.add_pair(Some((oi, &j_i)), Some((oj, &j_j)), &edge.omega, &residual);
            #[cfg(test)]
            ledger.extend(self.ledger_imu(from, edge, &residual, &j_i, &j_j));
        }
        let leaving: BTreeSet<LandmarkId> = self
            .landmarks
            .iter()
            .filter(|(_, lm)| plan.keyframes_to_marg.contains(&lm.host))
            .map(|(id, _)| *id)
            .chain(self.lost.iter().copied())
            .collect();
        let mut block = LandmarkBlock::default();
        let mut scratch_cost = Cost::default();
        for id in &leaving {
            let Some(lm) = self.landmarks.get(*id) else {
                continue;
            };
            if landmark_rows(
                self,
                lm,
                &order,
                Rows::In(&order),
                &mut block,
                &mut scratch_cost,
            )
            .is_none()
            {
                continue;
            }
            if block.eliminate_reduced_into(&mut factors) {
                #[cfg(test)]
                ledger.extend(self.ledger_landmark(&order, &block));
            }
        }
        #[cfg(test)]
        if let Some(l) = self.ledger.as_mut() {
            l.extend(ledger);
        }
        let fate = |id: FrameId| {
            if plan.states_all.contains(&id) || plan.keyframes_to_marg.contains(&id) {
                Fate::Drop
            } else if plan.states_vel_bias.contains(&id) {
                Fate::KeepPose
            } else {
                Fate::Keep
            }
        };
        let mut joint = factors.clone();
        self.prior
            .accumulate(&self.prior_delta(&self.prior), &order, &mut joint)
            .map_err(|_| OdometryError::PriorOutsideWindow)?;
        let (mut prior, pseudo_inverse) =
            marginalize_system(&joint, &order, fate).map_err(|_| OdometryError::Marginalization)?;
        let started = std::time::Instant::now();
        self.twin
            .accumulate(&self.prior_delta(&self.twin), &order, &mut factors)
            .map_err(|_| OdometryError::PriorOutsideWindow)?;
        let (mut twin, _) = marginalize_system(&factors, &order, fate)
            .map_err(|_| OdometryError::Marginalization)?;
        report.nullspace_time += started.elapsed();
        report.pseudo_inverse = pseudo_inverse;

        if let Some(state) = self.states.get_mut(&lsm) {
            state.set_linearized();
        }
        for id in &plan.states_vel_bias {
            if let Some(state) = self.states.remove(id) {
                let mut pose = state.to_pose();
                pose.set_linearized();
                self.poses.insert(*id, pose);
            }
        }
        for id in &plan.states_all {
            self.states.remove(id);
        }
        for id in &plan.keyframes_to_marg {
            self.poses.remove(id);
            self.keyframes.remove(id);
            self.counts.created.remove(id);
        }
        self.imu.retain(|from, _| *from >= lsm);
        let delta = self.prior_delta(&prior);
        prior.recentre(&delta);
        twin.recentre(&delta);
        #[cfg(test)]
        if let Some(ledger) = self.ledger.as_mut() {
            ledger.check_point(&prior);
        }
        self.prior = prior;
        self.twin = twin;
        let started = std::time::Instant::now();
        report.nullspace = Some(self.nullspace());
        report.nullspace_time += started.elapsed();

        for id in &leaving {
            self.landmarks.remove(*id);
        }
        let removed: Vec<FrameId> = plan
            .states_all
            .iter()
            .chain(&plan.keyframes_to_marg)
            .copied()
            .collect();
        self.landmarks.remove_frames(&removed);
        for id in &removed {
            self.frame_observations.remove(id);
        }
        Ok(())
    }
}
