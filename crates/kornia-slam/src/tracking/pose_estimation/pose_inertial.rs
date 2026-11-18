//! Pose-inertial solve of one frame: reprojection rows plus an IMU edge from
//! an anchor, after ORB-SLAM3's `PoseInertialOptimizationLastKeyFrame` and
//! `...LastFrame` (`Optimizer.cc:4491-5289`); the idea of PR #195, credit Astik.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{HuberKernel, RobustKernel};
use kornia_algebra::{Mat15F64, SMatF64, SO3F64, SVecF64, Vec2F64, Vec3F64};
use kornia_factors::inertial::NavState;
use kornia_factors::inertial::preintegration::{information, residual_and_jacobians};
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

use crate::mapping::vi_ba_schur::RigObservationCamera;
use crate::upstream::kornia_3d::refine_analytic::pose_jacobian_2x6;

/// Four rounds of ten iterations; outliers are re-classified after each
/// (`Optimizer.cc:4698-4712`).
const ROUNDS: usize = 4;
const ITERATIONS_PER_ROUND: usize = 10;
/// Visual chi² gates per round: `LastKeyFrame` and `LastFrame` monocular
/// (`Optimizer.cc:4700-4701, 5097-5098`). Every row is a 2-dof pixel
/// residual, so a stereo frame takes them too: the 3-dof stereo gates
/// belong to a row with the right-image u, which this solve has not (E6).
const CHI2_KEYFRAME_MONO: [f64; ROUNDS] = [12.0, 7.5, 5.991, 5.991];
const CHI2_FRAME_MONO: [f64; ROUNDS] = [5.991; ROUNDS];
/// Points tracked closer than this get 1.5x the gate (`Optimizer.cc:4720-4737`).
const CLOSE_DEPTH_M: f64 = 10.0;
/// Huber knee of a reprojection row, `thHuberMono²`; the kernel is dropped
/// in the last round (`Optimizer.cc:4750`).
const HUBER_REPROJECTION_CHI2: f64 = 5.991;
/// Huber knee of the previous frame's prior, `δ = 5` (`Optimizer.cc:5091`).
const HUBER_PRIOR_CHI2: f64 = 25.0;
/// Below this many inliers, rows inside the wider rescue gate count again
/// (`Optimizer.cc:4796-4822`).
const RESCUE_BELOW_INLIERS: usize = 30;
const RESCUE_CHI2_MONO: f64 = 18.0;
/// Pixel rows behind this depth are outliers.
const MIN_DEPTH_M: f64 = 1e-3;

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
/// edge `preintegrated` from `anchor`. A `Fixed` anchor is held; a `Prior`
/// anchor is free under its prior and marginalized out of the result's.
/// A solve that leaves the finite numbers returns `initial` with no inliers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_pose_inertial(
    anchor: Anchor<'_>,
    preintegrated: &PreintegratedImu,
    gravity_world: Vec3F64,
    camera_to_body: Option<Pose3d>,
    cameras: &[RigObservationCamera],
    primary: &PinholeCamera,
    initial: FrameNavState,
    observations: &[PoseInertialObservation],
) -> PoseInertialSolution {
    let (previous, prior) = match anchor {
        Anchor::Fixed(state) => (*state, None),
        Anchor::Prior(prior) => (prior.mean, Some(prior)),
    };
    let problem = Problem {
        prior,
        preintegrated,
        information: information(preintegrated, 1.0).omega,
        gravity: gravity_world,
        t_bc: camera_to_body,
        cameras,
        primary,
        observations,
    };
    let gates = match anchor {
        Anchor::Fixed(_) => CHI2_KEYFRAME_MONO,
        Anchor::Prior(_) => CHI2_FRAME_MONO,
    };

    let mut states = States {
        previous,
        current: initial,
    };
    let mut inliers = vec![true; observations.len()];
    for (round, gate) in gates.iter().enumerate() {
        problem.optimize(&mut states, &inliers, round + 1 < ROUNDS);
        for (o, inlier) in inliers.iter_mut().enumerate() {
            *inlier = problem
                .chi2_and_depth(o, &states.current)
                .is_some_and(|(chi2, depth)| {
                    chi2 <= if depth < CLOSE_DEPTH_M {
                        1.5 * gate
                    } else {
                        *gate
                    }
                });
        }
    }
    if inliers.iter().filter(|&&inlier| inlier).count() < RESCUE_BELOW_INLIERS {
        for (o, inlier) in inliers.iter_mut().enumerate() {
            *inlier |= problem
                .chi2_and_depth(o, &states.current)
                .is_some_and(|(chi2, _)| chi2 < RESCUE_CHI2_MONO);
        }
    }

    if !states.is_finite() {
        return PoseInertialSolution {
            state: initial,
            prior: None,
            inliers: vec![false; observations.len()],
            imu_chi2: f64::NAN,
        };
    }
    let system = problem.build(&states, &inliers, false);
    PoseInertialSolution {
        state: states.current,
        prior: problem
            .marginal_information(&system.hessian)
            .map(|information| NavPrior {
                mean: states.current,
                information,
            }),
        inliers,
        imu_chi2: system.imu_chi2,
    }
}

impl FrameNavState {
    fn nav_state(&self) -> NavState<'_> {
        NavState {
            pose_cw: &self.pose_cw,
            velocity: self.velocity,
            bias: self.bias,
        }
    }

    /// `self ⊞ δ`, `δ` in `[ρ ω v b_g b_a]`: the pose perturbed on the right,
    /// `T_cw · Exp(ρ, ω)`, the rest additively.
    fn plus(&self, delta: &SVecF64<15>) -> Self {
        let rho = Vec3F64::new(delta[0], delta[1], delta[2]);
        let omega = Vec3F64::new(delta[3], delta[4], delta[5]);
        let rotation = self.pose_cw.rotation;
        Self {
            pose_cw: Pose3d::new(
                rotation * SO3F64::exp(omega).matrix(),
                self.pose_cw.translation + rotation * (SO3F64::left_jacobian(omega) * rho),
            ),
            velocity: self.velocity + Vec3F64::new(delta[6], delta[7], delta[8]),
            bias: ImuBias {
                gyro: self.bias.gyro + Vec3F64::new(delta[9], delta[10], delta[11]),
                accel: self.bias.accel + Vec3F64::new(delta[12], delta[13], delta[14]),
            },
        }
    }

    /// `self ⊟ mean` to first order, the inverse of [`Self::plus`].
    fn minus(&self, mean: &Self) -> SVecF64<15> {
        let rotation_mean_t = mean.pose_cw.rotation.transpose();
        let omega = SO3F64::from_matrix(&(rotation_mean_t * self.pose_cw.rotation)).log();
        let rho = rotation_mean_t * (self.pose_cw.translation - mean.pose_cw.translation);
        let velocity = self.velocity - mean.velocity;
        let gyro = self.bias.gyro - mean.bias.gyro;
        let accel = self.bias.accel - mean.bias.accel;
        let mut delta = SVecF64::<15>::zeros();
        for (k, v) in [rho, omega, velocity, gyro, accel].into_iter().enumerate() {
            delta[3 * k] = v.x;
            delta[3 * k + 1] = v.y;
            delta[3 * k + 2] = v.z;
        }
        delta
    }

    fn is_finite(&self) -> bool {
        let finite = |v: Vec3F64| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        self.pose_cw
            .rotation
            .to_cols_array()
            .iter()
            .all(|v| v.is_finite())
            && finite(self.pose_cw.translation)
            && finite(self.velocity)
            && finite(self.bias.gyro)
            && finite(self.bias.accel)
    }
}

/// The two states of the solve: the anchor (held, or free under its prior)
/// and the current frame.
#[derive(Clone, Copy)]
struct States {
    previous: FrameNavState,
    current: FrameNavState,
}

impl States {
    fn is_finite(&self) -> bool {
        self.previous.is_finite() && self.current.is_finite()
    }
}

/// Normal equations over `[previous | current]`, 30 states, and the cost.
struct System {
    hessian: SMatF64<30, 30>,
    /// `Jᵀ W r`.
    gradient: SVecF64<30>,
    cost: f64,
    imu_chi2: f64,
}

struct Problem<'a> {
    /// The previous frame's prior; `None` holds the anchor fixed.
    prior: Option<&'a NavPrior>,
    preintegrated: &'a PreintegratedImu,
    information: Mat15F64,
    gravity: Vec3F64,
    t_bc: Option<Pose3d>,
    cameras: &'a [RigObservationCamera],
    primary: &'a PinholeCamera,
    observations: &'a [PoseInertialObservation],
}

impl Problem<'_> {
    /// Levenberg-Marquardt over the free states with the rows in `inliers`.
    fn optimize(&self, states: &mut States, inliers: &[bool], robust: bool) {
        let mut lambda = 1e-3;
        let mut system = self.build(states, inliers, robust);
        for _ in 0..ITERATIONS_PER_ROUND {
            let Some(step) = self.step(&system, lambda) else {
                lambda *= 10.0;
                continue;
            };
            let trial = self.apply(states, &step);
            let trial_system = self.build(&trial, inliers, robust);
            if trial_system.cost.is_finite() && trial_system.cost < system.cost {
                let gain = system.cost - trial_system.cost;
                *states = trial;
                system = trial_system;
                lambda = (lambda / 10.0).max(1e-9);
                if gain <= 1e-9 * system.cost.max(1e-12) {
                    break;
                }
            } else {
                lambda *= 10.0;
                if lambda > 1e8 {
                    break;
                }
            }
        }
    }

    /// The damped step `-(H + λ diag H)⁻¹ g` over the free states.
    fn step(&self, system: &System, lambda: f64) -> Option<SVecF64<30>> {
        let damp = |d: f64| d + lambda * d.max(1e-9);
        let mut step = SVecF64::<30>::zeros();
        if self.prior.is_some() {
            let mut hessian = system.hessian;
            for d in 0..30 {
                hessian[(d, d)] = damp(hessian[(d, d)]);
            }
            step.copy_from(&hessian.cholesky()?.solve(&(-system.gradient)));
        } else {
            let mut hessian: Mat15F64 = system.hessian.fixed_view::<15, 15>(15, 15).into_owned();
            for d in 0..15 {
                hessian[(d, d)] = damp(hessian[(d, d)]);
            }
            let gradient: SVecF64<15> = system.gradient.fixed_rows::<15>(15).into_owned();
            step.fixed_rows_mut::<15>(15)
                .copy_from(&hessian.cholesky()?.solve(&(-gradient)));
        }
        step.iter().all(|v| v.is_finite()).then_some(step)
    }

    fn apply(&self, states: &States, step: &SVecF64<30>) -> States {
        let previous: SVecF64<15> = step.fixed_rows::<15>(0).into_owned();
        let current: SVecF64<15> = step.fixed_rows::<15>(15).into_owned();
        States {
            previous: if self.prior.is_some() {
                states.previous.plus(&previous)
            } else {
                states.previous
            },
            current: states.current.plus(&current),
        }
    }

    /// The normal equations at `states`: the IMU edge, the prior when the
    /// previous frame is free, and the reprojection rows in `inliers`, under
    /// the Huber kernel when `robust`.
    fn build(&self, states: &States, inliers: &[bool], robust: bool) -> System {
        let mut hessian = SMatF64::<30, 30>::zeros();
        let mut gradient = SVecF64::<30>::zeros();

        let (residual, j_previous, j_current) = residual_and_jacobians(
            states.previous.nav_state(),
            states.current.nav_state(),
            self.preintegrated,
            &self.gravity,
            self.t_bc.as_ref(),
        );
        let omega = self.information;
        let imu_chi2 = residual.dot(&(omega * residual));
        let mut cost = 0.5 * imu_chi2;
        for (row, j_row) in [(0, &j_previous), (15, &j_current)] {
            for (col, j_col) in [(0, &j_previous), (15, &j_current)] {
                let mut block = hessian.fixed_view_mut::<15, 15>(row, col);
                block += j_row.transpose() * omega * j_col;
            }
            let mut rows = gradient.fixed_rows_mut::<15>(row);
            rows += j_row.transpose() * omega * residual;
        }

        if let Some(prior) = self.prior {
            let r = states.previous.minus(&prior.mean);
            let chi2 = r.dot(&(prior.information * r));
            let weight = HuberKernel.weight(chi2, HUBER_PRIOR_CHI2);
            cost += 0.5 * HuberKernel.rho(chi2, HUBER_PRIOR_CHI2);
            let mut block = hessian.fixed_view_mut::<15, 15>(0, 0);
            block += prior.information * weight;
            let mut rows = gradient.fixed_rows_mut::<15>(0);
            rows += prior.information * r * weight;
        }

        for (o, observation) in self.observations.iter().enumerate() {
            if !inliers[o] {
                continue;
            }
            let Some(row) = self.reprojection(o, &states.current) else {
                continue;
            };
            let chi2 = observation.information * row.residual.norm_squared();
            let (weight, row_cost) = if robust {
                (
                    HuberKernel.weight(chi2, HUBER_REPROJECTION_CHI2),
                    HuberKernel.rho(chi2, HUBER_REPROJECTION_CHI2),
                )
            } else {
                (1.0, chi2)
            };
            cost += 0.5 * row_cost;
            let scale = weight * observation.information;
            let mut block = hessian.fixed_view_mut::<6, 6>(15, 15);
            block += row.jacobian.transpose() * row.jacobian * scale;
            let mut rows = gradient.fixed_rows_mut::<6>(15);
            rows += row.jacobian.transpose() * row.residual * scale;
        }

        System {
            hessian,
            gradient,
            cost,
            imu_chi2,
        }
    }

    /// Observation `o`'s pixel residual and its Jacobian with respect to the
    /// current pose; `None` behind the camera.
    fn reprojection(&self, o: usize, current: &FrameNavState) -> Option<ReprojectionRow> {
        let observation = &self.observations[o];
        let (camera, pose) = match usize::from(observation.view).checked_sub(1) {
            None => (self.primary, current.pose_cw),
            // A right perturbation of T_pw is the same one of T_c'w = T_c'p T_pw.
            Some(k) => {
                let view = self.cameras.get(k)?;
                (
                    &view.camera,
                    view.camera_from_primary.compose(&current.pose_cw),
                )
            }
        };
        let point_cam = pose.transform_point(&observation.point_w);
        if point_cam.z < MIN_DEPTH_M {
            return None;
        }
        let residual = SVecF64::<2>::new(
            camera.fx * point_cam.x / point_cam.z + camera.cx - observation.pixel.x,
            camera.fy * point_cam.y / point_cam.z + camera.cy - observation.pixel.y,
        );
        Some(ReprojectionRow {
            residual,
            jacobian: pose_jacobian_2x6(
                &pose.rotation,
                observation.point_w,
                point_cam,
                camera.fx,
                camera.fy,
            ),
            depth: point_cam.z,
        })
    }

    /// Observation `o`'s weighted chi² and depth at `current`; `None` behind
    /// the camera.
    fn chi2_and_depth(&self, o: usize, current: &FrameNavState) -> Option<(f64, f64)> {
        let row = self.reprojection(o, current)?;
        Some((
            self.observations[o].information * row.residual.norm_squared(),
            row.depth,
        ))
    }

    /// The current frame's information: its block of `hessian` with a free
    /// previous frame marginalized out (`Optimizer.cc:4831-4870, 5232-5285`).
    /// `None` when not positive definite.
    fn marginal_information(&self, hessian: &SMatF64<30, 30>) -> Option<Mat15F64> {
        let h_cc: Mat15F64 = hessian.fixed_view::<15, 15>(15, 15).into_owned();
        let marginal = if self.prior.is_some() {
            let h_pp: Mat15F64 = hessian.fixed_view::<15, 15>(0, 0).into_owned();
            let h_pc: Mat15F64 = hessian.fixed_view::<15, 15>(0, 15).into_owned();
            h_cc - h_pc.transpose() * h_pp.cholesky()?.solve(&h_pc)
        } else {
            h_cc
        };
        let symmetric = (marginal + marginal.transpose()) * 0.5;
        symmetric.cholesky().map(|_| symmetric)
    }
}

struct ReprojectionRow {
    residual: SVecF64<2>,
    jacobian: SMatF64<2, 6>,
    depth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn at_rest() -> FrameNavState {
        FrameNavState {
            pose_cw: Pose3d::IDENTITY,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
        }
    }

    /// 50 ms at rest without gravity, with loose noise.
    fn still_edge() -> PreintegratedImu {
        let calib = ImuCalib {
            gyro_noise: 1.0,
            accel_noise: 1.0,
            gyro_bias_noise: 1.0,
            accel_bias_noise: 1.0,
        };
        let mut preintegrated = PreintegratedImu::new(ImuBias::default(), calib);
        for _ in 0..10 {
            let sample = ImuMeasurement {
                timestamp: 0.0,
                gyro: Vec3F64::ZERO,
                accel: Vec3F64::ZERO,
            };
            preintegrated.integrate(&sample, 0.005);
        }
        preintegrated
    }

    /// 40 points 12 m ahead, past the close depth, seen exactly from the
    /// identity pose, and one row `offset_px` off.
    fn rows(offset_px: f64) -> Vec<PoseInertialObservation> {
        let camera = camera();
        let mut rows: Vec<_> = (0..40)
            .map(|k| {
                let point_w =
                    Vec3F64::new((k % 8) as f64 * 1.5 - 5.0, (k / 8) as f64 * 1.5 - 3.0, 12.0);
                PoseInertialObservation {
                    point_w,
                    pixel: Vec2F64::new(
                        camera.fx * point_w.x / point_w.z + camera.cx,
                        camera.fy * point_w.y / point_w.z + camera.cy,
                    ),
                    information: 1.0,
                    view: 0,
                }
            })
            .collect();
        let mut outlier = rows[17];
        outlier.point_w.x += 0.4;
        outlier.pixel.x = camera.fx * outlier.point_w.x / outlier.point_w.z + camera.cx + offset_px;
        rows.push(outlier);
        rows
    }

    /// E6: every row is a 2-dof pixel residual, so a stereo frame takes the
    /// monocular gates too. A row at χ² ≈ 6.6 lies inside the 3-dof 7.815
    /// and outside the 2-dof 5.991 (5 % of inliers past it, not 2 %).
    #[test]
    fn a_stereo_frame_gates_its_pixel_rows_with_two_dof_thresholds() {
        let anchor = at_rest();
        let observations = rows(6.9f64.sqrt());
        let solution = solve_pose_inertial(
            Anchor::Fixed(&anchor),
            &still_edge(),
            Vec3F64::ZERO,
            None,
            &[],
            &camera(),
            at_rest(),
            &observations,
        );
        let outlier = observations.len() - 1;
        assert_eq!(solution.inliers.iter().filter(|&&i| i).count(), 40);
        assert!(
            !solution.inliers[outlier],
            "χ² ≈ 6.6 is past the 2-dof 5 % gate"
        );
    }
}
