//! Joint pose–inertial tracking solve (ORB-SLAM3's
//! `PoseInertialOptimizationLastFrame` / `…LastKeyFrame`): the current frame's
//! pose, velocity and IMU bias against its matched map points and one
//! preintegrated IMU edge from an anchor state.
//!
//! The anchor is either a keyframe held at its bundle-adjusted state, or the
//! previous frame, left free under the Gaussian prior its own solve produced.
//! The solution's anchor is then marginalized out (Schur complement), and the
//! current frame's 15×15 information becomes the next frame's prior: a
//! one-frame sliding window, relinearized every frame.
//!
//! Each state is 15 DOF, `[ρ ω | v | b_g | b_a]`, with the pose `T_cw`
//! perturbed on the right; the stacked problem orders the current frame first.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, QuatF64, SE3F64, SMatF64, SO3F64, SVecF64, Vec3F64, Vec15F64};
use kornia_factors::imu;
use kornia_factors::reprojection::residual_and_pose_jacobian;
use kornia_factors::robust::huber_weight_and_cost;
use kornia_factors::state::NavState;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

/// 2-DOF χ² at 95 %: the reprojection inlier gate and Huber threshold.
const VISUAL_CHI2: f64 = 5.991;
/// Huber threshold on the prior's χ² (ORB-SLAM3's δ = 5).
const PRIOR_CHI2: f64 = 25.0;
/// Inlier/outlier reclassification rounds; only the first ones are robustified.
const ROUNDS: usize = 4;
const ROBUST_ROUNDS: usize = 2;
const ITERATIONS_PER_ROUND: usize = 10;
const MIN_DEPTH: f64 = 1e-3;
const INITIAL_DAMPING: f64 = 1e-4;
const MAX_DAMPING: f64 = 1e4;
const CONVERGED_STEP: f64 = 1e-9;

const STATE_DOF: usize = 15;
const DOF: usize = 2 * STATE_DOF;
type Hessian = SMatF64<DOF, DOF>;
type Gradient = SVecF64<DOF>;

/// One frame's navigation state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameNavState {
    /// World-to-camera pose.
    pub pose_cw: Pose3d,
    /// Body velocity in the world frame (m/s).
    pub velocity: Vec3F64,
    pub bias: ImuBias,
}

impl FrameNavState {
    fn nav(&self) -> NavState<'_> {
        NavState {
            pose_cw: &self.pose_cw,
            velocity: self.velocity,
            bias: self.bias,
        }
    }

    /// The same state with its rotation projected back onto SO(3).
    fn normalized(self) -> Self {
        Self {
            pose_cw: from_se3(&to_se3(&self.pose_cw)),
            ..self
        }
    }

    /// `x ⊞ δ` for the state whose tangent starts at `offset` in `delta`.
    fn retract(&self, delta: &Gradient, offset: usize) -> Self {
        let block = |k: usize| {
            Vec3F64::new(
                delta[offset + k],
                delta[offset + k + 1],
                delta[offset + k + 2],
            )
        };
        let pose = to_se3(&self.pose_cw).rplus(block(0), block(3));
        Self {
            pose_cw: from_se3(&pose),
            velocity: self.velocity + block(6),
            bias: ImuBias {
                gyro: self.bias.gyro + block(9),
                accel: self.bias.accel + block(12),
            },
        }
    }
}

/// Gaussian on a frame's state, in its tangent space at `mean`.
#[derive(Debug, Clone)]
pub(crate) struct NavPrior {
    pub mean: FrameNavState,
    pub information: Mat15F64,
}

/// Where the IMU edge to the current frame starts.
#[derive(Clone, Copy)]
pub(crate) enum Anchor<'a> {
    /// A keyframe held at its bundle-adjusted state.
    Fixed(&'a FrameNavState),
    /// The previous frame, free under its prior.
    Prior(&'a NavPrior),
}

/// A matched map point with its undistorted pixel and pixel information `1/σ²`.
pub(crate) struct PointObservation {
    pub point_w: Vec3F64,
    pub pixel: [f64; 2],
    pub information: f64,
}

/// Everything held fixed during one solve.
pub(crate) struct PoseInertialProblem<'a> {
    pub anchor: Anchor<'a>,
    /// IMU preintegrated from the anchor to the current frame.
    pub preintegrated: &'a PreintegratedImu,
    pub gravity_world: Vec3F64,
    /// Camera-to-body transform `T_BC`; `None` when the camera is the body.
    pub camera_to_body: Option<Pose3d>,
    pub camera: &'a PinholeCamera,
}

pub(crate) struct PoseInertialSolution {
    pub state: FrameNavState,
    /// The next frame's prior; `None` when the marginal is not positive definite.
    pub prior: Option<NavPrior>,
    /// Which observations ended as inliers.
    pub inliers: Vec<bool>,
    /// Mahalanobis χ² of the IMU edge's navigation rows at the solution: how
    /// far vision pulled the state from what the IMU measured.
    pub imu_chi2: f64,
}

impl PoseInertialSolution {
    pub fn inlier_count(&self) -> usize {
        self.inliers.iter().filter(|&&inlier| inlier).count()
    }
}

/// Solves the current frame's state from `initial`, reclassifying
/// observations as inliers between rounds as ORB-SLAM3 does: an observation
/// rejected in one round can return in the next. Observations reprojecting
/// more than `initial_gate_px` (at octave 0) from `initial` start as outliers.
pub(crate) fn solve_pose_inertial(
    problem: &PoseInertialProblem<'_>,
    initial: FrameNavState,
    observations: &[PointObservation],
    initial_gate_px: f64,
) -> PoseInertialSolution {
    let solver = Solver {
        problem,
        observations,
        imu_information: imu::information(problem.preintegrated, 1.0).omega,
    };
    let mut current = initial.normalized();
    let mut anchor = match problem.anchor {
        Anchor::Fixed(state) => *state,
        Anchor::Prior(prior) => prior.mean,
    }
    .normalized();
    let mut inliers = vec![false; observations.len()];
    solver.classify(&current, &mut inliers, initial_gate_px * initial_gate_px);
    for round in 0..ROUNDS {
        solver.optimize(&mut current, &mut anchor, &inliers, round < ROBUST_ROUNDS);
        solver.classify(&current, &mut inliers, VISUAL_CHI2);
    }
    PoseInertialSolution {
        state: current,
        prior: solver.marginal_prior(&current, &anchor, &inliers),
        imu_chi2: solver.imu_chi2(&current, &anchor),
        inliers,
    }
}

struct Linearization {
    hessian: Hessian,
    gradient: Gradient,
    cost: f64,
}

struct Solver<'a> {
    problem: &'a PoseInertialProblem<'a>,
    observations: &'a [PointObservation],
    imu_information: Mat15F64,
}

impl Solver<'_> {
    fn anchor_is_free(&self) -> bool {
        matches!(self.problem.anchor, Anchor::Prior(_))
    }

    /// Levenberg–Marquardt on the stacked state; with a fixed anchor its block
    /// is pinned to identity so its step is zero.
    fn optimize(
        &self,
        current: &mut FrameNavState,
        anchor: &mut FrameNavState,
        active: &[bool],
        robust: bool,
    ) {
        let mut lin = self.linearize(current, anchor, active, robust);
        let mut damping = INITIAL_DAMPING;
        for _ in 0..ITERATIONS_PER_ROUND {
            let Some(delta) = self.step(&lin, damping) else {
                damping *= 10.0;
                continue;
            };
            let trial_current = current.retract(&delta, 0);
            let trial_anchor = if self.anchor_is_free() {
                anchor.retract(&delta, STATE_DOF)
            } else {
                *anchor
            };
            let trial = self.linearize(&trial_current, &trial_anchor, active, robust);
            if trial.cost < lin.cost {
                *current = trial_current;
                *anchor = trial_anchor;
                lin = trial;
                damping *= 0.1;
                if delta.norm_squared() < CONVERGED_STEP {
                    break;
                }
            } else {
                damping *= 10.0;
                if damping > MAX_DAMPING {
                    break;
                }
            }
        }
    }

    fn step(&self, lin: &Linearization, damping: f64) -> Option<Gradient> {
        let mut hessian = lin.hessian;
        if !self.anchor_is_free() {
            hessian
                .fixed_view_mut::<STATE_DOF, STATE_DOF>(STATE_DOF, STATE_DOF)
                .fill_with_identity();
        }
        for k in 0..DOF {
            hessian[(k, k)] += damping * hessian[(k, k)];
        }
        Some(-hessian.cholesky()?.solve(&lin.gradient))
    }

    fn linearize(
        &self,
        current: &FrameNavState,
        anchor: &FrameNavState,
        active: &[bool],
        robust: bool,
    ) -> Linearization {
        let mut lin = Linearization {
            hessian: Hessian::zeros(),
            gradient: Gradient::zeros(),
            cost: 0.0,
        };
        self.add_visual(&mut lin, current, active, robust);
        self.add_imu(&mut lin, current, anchor);
        if let Anchor::Prior(prior) = self.problem.anchor {
            add_prior(&mut lin, anchor, prior);
        }
        lin
    }

    fn add_visual(
        &self,
        lin: &mut Linearization,
        current: &FrameNavState,
        active: &[bool],
        robust: bool,
    ) {
        let mut hessian = SMatF64::<6, 6>::zeros();
        let mut gradient = SVecF64::<6>::zeros();
        for (observation, _) in self.observations.iter().zip(active).filter(|(_, a)| **a) {
            let Some((residual, jacobian)) = residual_and_pose_jacobian(
                &current.pose_cw,
                &observation.point_w,
                observation.pixel,
                self.problem.camera,
                MIN_DEPTH,
            ) else {
                // Dropping it would make pushing a point behind the camera a
                // free way to shed its cost; the step that does so is refused.
                lin.cost = f64::INFINITY;
                continue;
            };
            let chi2 = residual.norm_squared() * observation.information;
            let (weight, cost) = huber_weight_and_cost(chi2, robust.then_some(VISUAL_CHI2));
            lin.cost += cost;
            let jt = jacobian.transpose() * (weight * observation.information);
            hessian += jt * jacobian;
            gradient += jt * residual;
        }
        add_block(&mut lin.hessian, 0, 0, &hessian);
        add_rows(&mut lin.gradient, 0, &gradient);
    }

    fn add_imu(&self, lin: &mut Linearization, current: &FrameNavState, anchor: &FrameNavState) {
        let (residual, j_anchor, j_current) = imu::residual_and_jacobians(
            anchor.nav(),
            current.nav(),
            self.problem.preintegrated,
            &self.problem.gravity_world,
            self.problem.camera_to_body.as_ref(),
        );
        let omega = &self.imu_information;
        lin.cost += 0.5 * residual.dot(&(omega * residual));
        let jc_t_omega = j_current.transpose() * omega;
        add_block(&mut lin.hessian, 0, 0, &(jc_t_omega * j_current));
        add_rows(&mut lin.gradient, 0, &(jc_t_omega * residual));
        if self.anchor_is_free() {
            let ja_t_omega = j_anchor.transpose() * omega;
            let cross = jc_t_omega * j_anchor;
            add_block(&mut lin.hessian, 0, STATE_DOF, &cross);
            add_block(&mut lin.hessian, STATE_DOF, 0, &cross.transpose());
            add_block(
                &mut lin.hessian,
                STATE_DOF,
                STATE_DOF,
                &(ja_t_omega * j_anchor),
            );
            add_rows(&mut lin.gradient, STATE_DOF, &(ja_t_omega * residual));
        }
    }

    fn imu_chi2(&self, current: &FrameNavState, anchor: &FrameNavState) -> f64 {
        let (residual, _, _) = imu::residual_and_jacobians(
            anchor.nav(),
            current.nav(),
            self.problem.preintegrated,
            &self.problem.gravity_world,
            self.problem.camera_to_body.as_ref(),
        );
        let nav = residual.fixed_rows::<9>(0);
        nav.dot(&(self.imu_information.fixed_view::<9, 9>(0, 0) * nav))
    }

    /// Marks each observation an inlier when it is in front of the camera and
    /// its reprojection χ² is within `chi2_gate`.
    fn classify(&self, current: &FrameNavState, inliers: &mut [bool], chi2_gate: f64) {
        for (observation, inlier) in self.observations.iter().zip(inliers.iter_mut()) {
            *inlier = residual_and_pose_jacobian(
                &current.pose_cw,
                &observation.point_w,
                observation.pixel,
                self.problem.camera,
                MIN_DEPTH,
            )
            .is_some_and(|(residual, _)| {
                residual.norm_squared() * observation.information <= chi2_gate
            });
        }
    }

    /// The current frame's information with the anchor marginalized out:
    /// `H_cc − H_ca · H_aa⁻¹ · H_ac`, or `H_cc` for a fixed anchor.
    fn marginal_prior(
        &self,
        current: &FrameNavState,
        anchor: &FrameNavState,
        inliers: &[bool],
    ) -> Option<NavPrior> {
        let lin = self.linearize(current, anchor, inliers, false);
        let h = &lin.hessian;
        let h_cc = h.fixed_view::<STATE_DOF, STATE_DOF>(0, 0).into_owned();
        let information = if self.anchor_is_free() {
            let h_aa = h
                .fixed_view::<STATE_DOF, STATE_DOF>(STATE_DOF, STATE_DOF)
                .into_owned();
            let h_ac = h
                .fixed_view::<STATE_DOF, STATE_DOF>(STATE_DOF, 0)
                .into_owned();
            h_cc - h_ac.transpose() * h_aa.cholesky()?.solve(&h_ac)
        } else {
            h_cc
        };
        let information = (information + information.transpose()) * 0.5;
        information.cholesky()?;
        Some(NavPrior {
            mean: *current,
            information,
        })
    }
}

/// Huber-robustified `‖x_a ⊟ x̄‖²_Λ` on the free anchor.
fn add_prior(lin: &mut Linearization, anchor: &FrameNavState, prior: &NavPrior) {
    let (residual, jacobian) = prior_residual(anchor, &prior.mean);
    let information = &prior.information;
    let chi2 = residual.dot(&(information * residual));
    let (weight, cost) = huber_weight_and_cost(chi2, Some(PRIOR_CHI2));
    lin.cost += cost;
    let jt_info = jacobian.transpose() * information * weight;
    add_block(
        &mut lin.hessian,
        STATE_DOF,
        STATE_DOF,
        &(jt_info * jacobian),
    );
    add_rows(&mut lin.gradient, STATE_DOF, &(jt_info * residual));
}

/// `x ⊟ x̄` with the pose part `Log(T̄⁻¹·T)`, and its Jacobian with respect to
/// `x`'s tangent (the right-Jacobian inverse on the pose, identity elsewhere).
fn prior_residual(state: &FrameNavState, mean: &FrameNavState) -> (Vec15F64, Mat15F64) {
    let (rho, omega) = to_se3(&mean.pose_cw).rminus(&to_se3(&state.pose_cw));
    let mut residual = Vec15F64::zeros();
    let blocks = [
        rho,
        omega,
        state.velocity - mean.velocity,
        state.bias.gyro - mean.bias.gyro,
        state.bias.accel - mean.bias.accel,
    ];
    for (k, block) in blocks.into_iter().enumerate() {
        residual
            .fixed_rows_mut::<3>(3 * k)
            .copy_from(&SVecF64::<3>::from(block));
    }
    let mut jacobian = Mat15F64::identity();
    jacobian
        .fixed_view_mut::<6, 6>(0, 0)
        .copy_from(&SE3F64::right_jacobian_inverse(rho, omega));
    (residual, jacobian)
}

/// `from_mat3` of a slightly skewed matrix is not unit length and `from_quat`
/// assumes it is, so without renormalizing the skew doubles on every retract.
fn to_se3(pose: &Pose3d) -> SE3F64 {
    let rotation = QuatF64::from_mat3(&pose.rotation).normalize();
    SE3F64::new(SO3F64::from_quaternion(rotation), pose.translation)
}

fn from_se3(pose: &SE3F64) -> Pose3d {
    Pose3d::new(pose.r.matrix(), pose.t)
}

fn add_block<const R: usize, const C: usize>(
    hessian: &mut Hessian,
    row: usize,
    col: usize,
    block: &SMatF64<R, C>,
) {
    let mut view = hessian.fixed_view_mut::<R, C>(row, col);
    view += block;
}

fn add_rows<const N: usize>(gradient: &mut Gradient, row: usize, rows: &SVecF64<N>) {
    let mut view = gradient.fixed_rows_mut::<N>(row);
    view += rows;
}

#[cfg(test)]
mod tests {
    use kornia_algebra::Mat3F64;
    use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

    use super::*;

    const IMU_RATE_HZ: f64 = 1000.0;
    const TEST_GATE_PX: f64 = 60.0;

    fn gravity() -> Vec3F64 {
        Vec3F64::new(0.0, 0.0, -9.81)
    }

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 450.0,
            fy: 450.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn noise() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        }
    }

    /// Constant world acceleration from rest-free initial velocity, no rotation,
    /// camera-frame body: the IMU model integrates it exactly.
    struct Motion {
        velocity: Vec3F64,
        acceleration: Vec3F64,
    }

    impl Motion {
        fn state(&self, t: f64) -> FrameNavState {
            let position = self.velocity * t + self.acceleration * (0.5 * t * t);
            FrameNavState {
                pose_cw: Pose3d::new(Mat3F64::IDENTITY, position * -1.0),
                velocity: self.velocity + self.acceleration * t,
                bias: ImuBias::default(),
            }
        }

        fn preintegrated(&self, t0: f64, t1: f64) -> PreintegratedImu {
            let n = ((t1 - t0) * IMU_RATE_HZ).round() as usize;
            let samples: Vec<ImuMeasurement> = (0..=n)
                .map(|k| ImuMeasurement {
                    timestamp: t0 + k as f64 / IMU_RATE_HZ,
                    gyro: Vec3F64::ZERO,
                    accel: self.acceleration - gravity(),
                })
                .collect();
            PreintegratedImu::from_measurements(ImuBias::default(), noise(), &samples, t0, t1)
        }
    }

    fn motion() -> Motion {
        Motion {
            velocity: Vec3F64::new(0.8, -0.1, 0.2),
            acceleration: Vec3F64::new(0.5, 0.3, -0.2),
        }
    }

    fn world_points() -> Vec<Vec3F64> {
        (0..40)
            .map(|k| {
                let (row, col) = ((k / 8) as f64, (k % 8) as f64);
                Vec3F64::new(
                    -1.75 + 0.5 * col,
                    -1.0 + 0.5 * row,
                    5.0 + 0.3 * (k % 5) as f64,
                )
            })
            .collect()
    }

    fn observe(state: &FrameNavState, points: &[Vec3F64]) -> Vec<PointObservation> {
        let camera = camera();
        points
            .iter()
            .map(|p| {
                let p_c = state.pose_cw.transform_point(p);
                PointObservation {
                    point_w: *p,
                    pixel: [
                        camera.fx * p_c.x / p_c.z + camera.cx,
                        camera.fy * p_c.y / p_c.z + camera.cy,
                    ],
                    information: 1.0,
                }
            })
            .collect()
    }

    fn perturbed(state: &FrameNavState) -> FrameNavState {
        let mut delta = Gradient::zeros();
        for (k, value) in [0.08, -0.05, 0.06, 0.02, -0.015, 0.01, 0.3, -0.2, 0.1]
            .into_iter()
            .enumerate()
        {
            delta[k] = value;
        }
        state.retract(&delta, 0)
    }

    fn position_error(a: &FrameNavState, b: &FrameNavState) -> f64 {
        (a.pose_cw.inverse().translation - b.pose_cw.inverse().translation).length()
    }

    fn solve(
        anchor: Anchor<'_>,
        preintegrated: &PreintegratedImu,
        initial: FrameNavState,
        observations: &[PointObservation],
    ) -> PoseInertialSolution {
        let camera = camera();
        let problem = PoseInertialProblem {
            anchor,
            preintegrated,
            gravity_world: gravity(),
            camera_to_body: None,
            camera: &camera,
        };
        solve_pose_inertial(&problem, initial, observations, TEST_GATE_PX)
    }

    #[test]
    fn recovers_pose_and_velocity_from_a_keyframe_anchor() {
        let motion = motion();
        let (keyframe, truth) = (motion.state(0.0), motion.state(0.25));
        let solution = solve(
            Anchor::Fixed(&keyframe),
            &motion.preintegrated(0.0, 0.25),
            perturbed(&truth),
            &observe(&truth, &world_points()),
        );
        assert!(position_error(&solution.state, &truth) < 1e-4);
        assert!((solution.state.velocity - truth.velocity).length() < 1e-3);
        assert_eq!(solution.inlier_count(), world_points().len());
        assert!(solution.prior.is_some());
    }

    #[test]
    fn the_imu_holds_the_pose_when_vision_alone_is_underdetermined() {
        let motion = motion();
        let (keyframe, truth) = (motion.state(0.0), motion.state(0.25));
        let points = world_points();
        let two_points = &points[..2];
        let solution = solve(
            Anchor::Fixed(&keyframe),
            &motion.preintegrated(0.0, 0.25),
            perturbed(&truth),
            &observe(&truth, two_points),
        );
        assert!(position_error(&solution.state, &truth) < 1e-3);
        assert!((solution.state.velocity - truth.velocity).length() < 1e-2);
    }

    #[test]
    fn gross_outliers_are_classified_out() {
        let motion = motion();
        let (keyframe, truth) = (motion.state(0.0), motion.state(0.25));
        let mut observations = observe(&truth, &world_points());
        for observation in observations.iter_mut().step_by(4) {
            observation.pixel[0] += 40.0;
        }
        let solution = solve(
            Anchor::Fixed(&keyframe),
            &motion.preintegrated(0.0, 0.25),
            perturbed(&truth),
            &observations,
        );
        for (k, inlier) in solution.inliers.iter().enumerate() {
            assert_eq!(*inlier, k % 4 != 0, "observation {k}");
        }
        assert!(position_error(&solution.state, &truth) < 1e-4);
    }

    #[test]
    fn a_gross_outlier_cannot_escape_behind_the_camera() {
        let motion = motion();
        let (keyframe, truth) = (motion.state(0.0), motion.state(0.25));
        let mut observations = observe(&truth, &world_points());
        // Just in front of the camera and hundreds of pixels off: rotating it
        // behind the camera would remove a huge cost if that were free.
        let camera_centre = truth.pose_cw.inverse().translation;
        observations.push(PointObservation {
            point_w: camera_centre + Vec3F64::new(0.3, 0.0, 0.02),
            pixel: [320.0, 240.0],
            information: 1.0,
        });
        let solution = solve(
            Anchor::Fixed(&keyframe),
            &motion.preintegrated(0.0, 0.25),
            perturbed(&truth),
            &observations,
        );
        assert!(!solution.inliers[observations.len() - 1]);
        assert!(position_error(&solution.state, &truth) < 1e-4);
        assert!(solution.imu_chi2 < 1.0);
    }

    #[test]
    fn chains_frames_through_the_marginal_prior() {
        let motion = motion();
        let points = world_points();
        let (keyframe, first, second) = (motion.state(0.0), motion.state(0.05), motion.state(0.1));
        let first_solution = solve(
            Anchor::Fixed(&keyframe),
            &motion.preintegrated(0.0, 0.05),
            perturbed(&first),
            &observe(&first, &points),
        );
        let prior = first_solution.prior.expect("positive-definite marginal");
        assert!((prior.information - prior.information.transpose()).norm() < 1e-6);

        let second_solution = solve(
            Anchor::Prior(&prior),
            &motion.preintegrated(0.05, 0.1),
            perturbed(&second),
            &observe(&second, &points),
        );
        assert!(position_error(&second_solution.state, &second) < 1e-4);
        assert!((second_solution.state.velocity - second.velocity).length() < 1e-3);
        assert!(second_solution.prior.is_some());
    }

    /// A slightly skewed rotation, as an f32 round trip or a map correction
    /// leaves, must come out of every retract on SO(3) rather than compounding.
    #[test]
    fn retract_projects_a_skewed_rotation_back_onto_so3() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -1.1, 0.7)).matrix();
        let mut columns = rotation.to_cols_array();
        columns[..3].iter_mut().for_each(|x| *x *= 1.003);
        let skewed = Mat3F64::from_cols_array(&columns);
        let mut state = FrameNavState {
            pose_cw: Pose3d::new(skewed, Vec3F64::new(4.0, -2.0, 9.0)),
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
        };
        let step = Gradient::from_fn(|k, _| if k < 6 { 1e-3 } else { 0.0 });
        for _ in 0..20 {
            state = state.retract(&step, 0);
        }
        let r = state.pose_cw.rotation;
        let error = (r.transpose() * r - Mat3F64::IDENTITY)
            .to_cols_array()
            .iter()
            .fold(0.0_f64, |max, x| x.abs().max(max));
        assert!(error < 1e-12, "RᵀR − I = {error:e}");
    }
}
