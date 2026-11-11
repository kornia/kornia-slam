//! Marginal prior that carries the inertial chain past the local VI-BA window.
//!
//! When the oldest active keyframe `k` freezes, its 15-DOF state is
//! eliminated from the joint of the prior on `x_k`, the IMU edge `k → k+1`
//! at full information and a pseudo-prior on the frozen pose, leaving a
//! 15×15 information prior on `x_{k+1}` (Schur complement, as in Leutenegger
//! et al., OKVIS, 2015). Visual factors never enter it, so the window's
//! vision is not counted twice. The prior keeps its first-estimate
//! linearization point: its Hessian is never rebuilt.
//!
//! It replaces the 1e-2-scaled boundary edge
//! ([`ViBaParams::boundary_imu_info_scale`]), which let the window's velocity
//! follow the finite difference of its visual positions and dropped all
//! inertial history older than the window.

use faer::Mat;
use faer::prelude::Solve;
use kornia_algebra::Vec3F64;
use thiserror::Error;

use crate::mapping::vi_ba_schur::{
    ImuFactor, ViBaKeyframe, ViBaParams, accum_jt_omega_j, idx15, imu_information_matrix,
    imu_residual_and_jacobians,
};
use crate::mapping::wheel_odom::se3_log;

const DOF: usize = 15;

/// A prior moved further than this from its linearization point, in metres
/// or radians of the pose tangent, is dropped: its FEJ Hessian no longer
/// describes the state.
const STALE_TRANSLATION_M: f64 = 0.5;
const STALE_ROTATION_RAD: f64 = 10.0 * std::f64::consts::PI / 180.0;

/// What the VI-BA window puts between the frozen past and its oldest active
/// keyframe.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryPrior {
    /// The boundary IMU edge, its information scaled by
    /// [`ViBaParams::boundary_imu_info_scale`] (ORB-SLAM3).
    #[default]
    ScaledEdge,
    /// A [`MarginalPrior`] on the oldest active keyframe; the boundary edge
    /// enters only through it.
    Marginal(FrozenPoseSigma),
}

/// Standard deviation of the pose pseudo-prior a keyframe gets as it
/// freezes: how well the frozen pose is known.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct FrozenPoseSigma {
    /// Metres, on each translation axis.
    pub frozen_pose_sigma_m: f64,
    /// Radians, on each rotation axis.
    pub frozen_pose_sigma_rad: f64,
}

impl Default for FrozenPoseSigma {
    fn default() -> Self {
        Self {
            frozen_pose_sigma_m: 0.002,
            frozen_pose_sigma_rad: 0.001,
        }
    }
}

impl FrozenPoseSigma {
    /// Both sigmas finite and positive.
    pub fn is_valid(&self) -> bool {
        [self.frozen_pose_sigma_m, self.frozen_pose_sigma_rad]
            .iter()
            .all(|s| s.is_finite() && *s > 0.0)
    }
}

/// Why [`marginalize_frozen`] produced no prior.
#[derive(Debug, Error, PartialEq)]
pub enum MarginalizationError {
    #[error("frozen pose sigma is not finite and positive")]
    InvalidSigma,
    #[error("frozen keyframe block is not positive definite")]
    NotPositiveDefinite,
    #[error("marginal prior is not finite")]
    NonFinite,
}

/// Quadratic prior `½ dᵀ H d + bᵀ d` on one keyframe's 15-DOF state, in
/// `d = x ⊖ lin_point`: pose `Log(T₀⁻¹ T)` of the world→camera pose (the
/// solver's right perturbation, `[ρ; ω]`), velocity (world, m/s) and biases
/// additive.
#[derive(Debug, Clone)]
pub struct MarginalPrior {
    /// `Frame::idx` of the keyframe it constrains.
    pub frame_idx: usize,
    /// Map world epoch the prior was built in.
    pub world_epoch: u64,
    pub lin_point: ViBaKeyframe,
    /// `H`, row-major 15×15.
    pub info: [f64; 225],
    /// `b`, the cost gradient at `lin_point`.
    pub rhs: [f64; 15],
}

impl MarginalPrior {
    /// `kf ⊖ lin_point`.
    pub fn tangent(&self, kf: &ViBaKeyframe) -> [f64; 15] {
        let pose = se3_log(&self.lin_point.pose.inverse().compose(&kf.pose));
        let v = kf.velocity - self.lin_point.velocity;
        let bg = kf.bias.gyro - self.lin_point.bias.gyro;
        let ba = kf.bias.accel - self.lin_point.bias.accel;
        let mut d = [0.0; DOF];
        d[..6].copy_from_slice(&pose);
        for (k, x) in [v, bg, ba].into_iter().enumerate() {
            d[6 + 3 * k..9 + 3 * k].copy_from_slice(&[x.x, x.y, x.z]);
        }
        d
    }

    /// `b + H d`.
    pub fn gradient(&self, d: &[f64; 15]) -> [f64; 15] {
        std::array::from_fn(|r| {
            self.rhs[r] + (0..DOF).map(|c| self.info[idx15(r, c)] * d[c]).sum::<f64>()
        })
    }

    /// `½ dᵀ H d + bᵀ d` at `kf`.
    pub fn cost(&self, kf: &ViBaKeyframe) -> f64 {
        let d = self.tangent(kf);
        let g = self.gradient(&d);
        // ½ dᵀHd + bᵀd = ½ dᵀ(b + Hd) + ½ bᵀd.
        (0..DOF).map(|r| 0.5 * d[r] * (g[r] + self.rhs[r])).sum()
    }

    /// Whether `kf` moved too far from the linearization point for the
    /// first-estimate Hessian to hold.
    pub fn is_stale(&self, kf: &ViBaKeyframe) -> bool {
        let d = self.tangent(kf);
        let norm = |s: &[f64]| s.iter().map(|x| x * x).sum::<f64>().sqrt();
        norm(&d[0..3]) > STALE_TRANSLATION_M || norm(&d[3..6]) > STALE_ROTATION_RAD
    }

    /// The prior's mean velocity (world, m/s) and its per-axis sigma
    /// `√(tr Σ_vv / 3)`; `None` while `H` is not positive definite (the
    /// first priors of a chain are rank-deficient).
    pub fn velocity_mean_and_sigma(&self) -> Option<(Vec3F64, f64)> {
        let h = Mat::<f64>::from_fn(DOF, DOF, |r, c| self.info[idx15(r, c)]);
        let chol = h.llt(faer::Side::Lower).ok()?;
        let rhs = Mat::<f64>::from_fn(DOF, DOF + 1, |r, c| match c {
            DOF => -self.rhs[r],
            _ => f64::from(u8::from(r == c)),
        });
        let x = chol.solve(&rhs);
        let mean = self.lin_point.velocity + Vec3F64::new(x[(6, DOF)], x[(7, DOF)], x[(8, DOF)]);
        let sigma = ((x[(6, 6)] + x[(7, 7)] + x[(8, 8)]) / 3.0).sqrt();
        [sigma, mean.x, mean.y, mean.z]
            .iter()
            .all(|x| x.is_finite())
            .then_some((mean, sigma))
    }
}

/// Adds `prior` on the free keyframe at system offset `base` to the reduced
/// system (`m_mat += H`, `m_vec −= b + H d`, unit Jacobian at the
/// linearization point) and returns its cost at `kf`. Never robustified.
pub(super) fn accumulate_prior(
    m_mat: &mut Mat<f64>,
    m_vec: &mut [f64],
    base: usize,
    prior: &MarginalPrior,
    kf: &ViBaKeyframe,
) -> f64 {
    let d = prior.tangent(kf);
    let g = prior.gradient(&d);
    for r in 0..DOF {
        m_vec[base + r] -= g[r];
        for c in 0..DOF {
            m_mat[(base + r, base + c)] += prior.info[idx15(r, c)];
        }
    }
    prior.cost(kf)
}

/// The prior on `next` left when `frozen` is eliminated from `prior` (on
/// `frozen`, `None` to start a chain), the IMU `edge` from `frozen` to `next`
/// at full information and a pseudo-prior of `pseudo` on the frozen pose.
/// Everything is linearized at the given states; the result's
/// `lin_point` is `next`. `frame_idx` and `world_epoch` are the caller's to
/// stamp.
pub(super) fn marginalize_frozen(
    prior: Option<&MarginalPrior>,
    frozen: &ViBaKeyframe,
    next: &ViBaKeyframe,
    edge: &ImuFactor,
    params: &ViBaParams,
    pseudo: FrozenPoseSigma,
) -> Result<MarginalPrior, MarginalizationError> {
    if !pseudo.is_valid() {
        return Err(MarginalizationError::InvalidSigma);
    }
    let (res, j_frozen, j_next) = imu_residual_and_jacobians(
        frozen,
        next,
        &edge.preintegrated,
        &params.gravity,
        params.imu_t_bc.as_ref(),
    );
    let omega = imu_information_matrix(&edge.preintegrated, params.imu_weight);

    // Joint over [frozen | next]; `m` is the negative gradient, as in the solver.
    let mut h = Mat::<f64>::zeros(2 * DOF, 2 * DOF);
    let mut m = vec![0.0; 2 * DOF];
    let blocks = [(0, &j_frozen), (DOF, &j_next)];
    for &(row, j_row) in &blocks {
        for &(col, j_col) in &blocks {
            accum_jt_omega_j(
                &mut h,
                &mut m,
                row,
                col,
                j_row,
                j_col,
                &omega,
                &res,
                row == col,
            );
        }
    }
    if let Some(prior) = prior {
        accumulate_prior(&mut h, &mut m, 0, prior, frozen);
    }
    for d in 0..3 {
        h[(d, d)] += pseudo.frozen_pose_sigma_m.powi(-2);
        h[(3 + d, 3 + d)] += pseudo.frozen_pose_sigma_rad.powi(-2);
    }

    // H' = H₂₂ − H₂₁ H₁₁⁻¹ H₁₂,  m' = m₂ − H₂₁ H₁₁⁻¹ m₁.
    let h11 = Mat::<f64>::from_fn(DOF, DOF, |r, c| h[(r, c)]);
    let chol = h11
        .llt(faer::Side::Lower)
        .map_err(|_| MarginalizationError::NotPositiveDefinite)?;
    let rhs = Mat::<f64>::from_fn(DOF, DOF + 1, |r, c| match c {
        DOF => m[r],
        _ => h[(r, DOF + c)],
    });
    let x = chol.solve(&rhs);
    let schur = |r: usize, c: usize| (0..DOF).map(|k| h[(DOF + r, k)] * x[(k, c)]).sum::<f64>();
    let mut info = [0.0; 225];
    for r in 0..DOF {
        for c in 0..DOF {
            info[idx15(r, c)] = h[(DOF + r, DOF + c)] - schur(r, c);
        }
    }
    for r in 0..DOF {
        for c in (r + 1)..DOF {
            let avg = 0.5 * (info[idx15(r, c)] + info[idx15(c, r)]);
            info[idx15(r, c)] = avg;
            info[idx15(c, r)] = avg;
        }
    }
    let rhs: [f64; 15] = std::array::from_fn(|r| -(m[DOF + r] - schur(r, DOF)));
    if !info.iter().chain(&rhs).all(|x| x.is_finite()) {
        return Err(MarginalizationError::NonFinite);
    }
    Ok(MarginalPrior {
        frame_idx: 0,
        world_epoch: 0,
        lin_point: ViBaKeyframe {
            fixed: false,
            ..next.clone()
        },
        info,
        rhs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::SO3F64;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

    const EDGE_S: f64 = 0.2;
    const IMU_HZ: f64 = 200.0;

    /// Deterministic uniform samples in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn vec3(&mut self, scale: f64) -> Vec3F64 {
            Vec3F64::new(self.next(), self.next(), self.next()) * scale
        }
    }

    fn se3_exp(xi: &[f64]) -> Pose3d {
        let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
        let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
        Pose3d::new(
            SO3F64::exp(omega).matrix(),
            SO3F64::left_jacobian(omega) * rho,
        )
    }

    /// The solver's step in f64: pose right-perturbed, the rest additive.
    fn retract(kf: &ViBaKeyframe, d: &[f64]) -> ViBaKeyframe {
        let mut out = kf.clone();
        out.pose = kf.pose.compose(&se3_exp(&d[0..6]));
        out.velocity += Vec3F64::new(d[6], d[7], d[8]);
        out.bias.gyro += Vec3F64::new(d[9], d[10], d[11]);
        out.bias.accel += Vec3F64::new(d[12], d[13], d[14]);
        out
    }

    /// Noise densities large enough that the 90-DOF batch stays well
    /// conditioned for a 1e-9 comparison.
    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1e-2,
            accel_noise: 2e-2,
            gyro_bias_noise: 1e-3,
            accel_bias_noise: 1e-2,
        }
    }

    /// Camera at world position `p` with a small yaw, as `T_cw`.
    fn pose_at(p: Vec3F64, yaw: f64) -> Pose3d {
        let r_wc = SO3F64::exp(Vec3F64::new(0.0, yaw, 0.0)).matrix();
        Pose3d::new(r_wc, p).inverse()
    }

    /// An edge of constant world acceleration `a` and yaw rate `w`,
    /// starting at yaw `yaw0`, under `gravity`.
    fn edge(k: usize, a: Vec3F64, yaw0: f64, w: f64, gravity: Vec3F64) -> ImuFactor {
        let mut pim = PreintegratedImu::new(ImuBias::default(), calib());
        let n = (EDGE_S * IMU_HZ).round() as usize;
        let dt = EDGE_S / n as f64;
        for s in 0..n {
            let r_bw = SO3F64::exp(Vec3F64::new(0.0, yaw0 + w * s as f64 * dt, 0.0)).inverse();
            let sample = ImuMeasurement {
                timestamp: 0.0,
                gyro: Vec3F64::new(0.0, w, 0.0),
                accel: r_bw * (a - gravity),
            };
            pim.integrate(&sample, dt);
        }
        ImuFactor {
            from_idx: k,
            to_idx: k + 1,
            preintegrated: pim,
        }
    }

    /// Six keyframes of constant acceleration with a slow yaw, states
    /// perturbed off truth, and the five IMU edges between them.
    fn chain() -> (Vec<ViBaKeyframe>, Vec<ImuFactor>, ViBaParams) {
        let params = ViBaParams::default();
        let (a, v0, w) = (
            Vec3F64::new(0.3, 0.0, -0.2),
            Vec3F64::new(0.5, 0.0, 0.2),
            0.1,
        );
        let mut rng = Lcg(7);
        let keyframes = (0..6)
            .map(|k| {
                let t = k as f64 * EDGE_S;
                let truth = ViBaKeyframe {
                    pose: pose_at(v0 * t + a * (0.5 * t * t), w * t),
                    velocity: v0 + a * t,
                    bias: ImuBias::default(),
                    fixed: false,
                };
                let d: [f64; 15] =
                    std::array::from_fn(|i| rng.next() * if i < 6 { 5e-3 } else { 2e-2 });
                retract(&truth, &d)
            })
            .collect();
        let edges = (0..5)
            .map(|k| edge(k, a, w * k as f64 * EDGE_S, w, params.gravity))
            .collect();
        (keyframes, edges, params)
    }

    /// Gauss-Newton step of the edges `edges` and pose pseudo-priors on
    /// every keyframe of `keyframes`, plus `prior` on the first; all at the
    /// given states.
    fn gn_step(
        keyframes: &[ViBaKeyframe],
        edges: &[ImuFactor],
        params: &ViBaParams,
        pseudo: FrozenPoseSigma,
        prior: Option<&MarginalPrior>,
    ) -> Vec<f64> {
        let n = keyframes.len() * DOF;
        let mut h = Mat::<f64>::zeros(n, n);
        let mut m = vec![0.0; n];
        for e in edges {
            let (res, ji, jj) = imu_residual_and_jacobians(
                &keyframes[e.from_idx],
                &keyframes[e.to_idx],
                &e.preintegrated,
                &params.gravity,
                None,
            );
            let omega = imu_information_matrix(&e.preintegrated, params.imu_weight);
            let blocks = [(e.from_idx * DOF, &ji), (e.to_idx * DOF, &jj)];
            for &(row, j_row) in &blocks {
                for &(col, j_col) in &blocks {
                    accum_jt_omega_j(
                        &mut h,
                        &mut m,
                        row,
                        col,
                        j_row,
                        j_col,
                        &omega,
                        &res,
                        row == col,
                    );
                }
            }
        }
        for k in 0..keyframes.len() {
            for d in 0..3 {
                h[(k * DOF + d, k * DOF + d)] += pseudo.frozen_pose_sigma_m.powi(-2);
                h[(k * DOF + 3 + d, k * DOF + 3 + d)] += pseudo.frozen_pose_sigma_rad.powi(-2);
            }
        }
        if let Some(prior) = prior {
            accumulate_prior(&mut h, &mut m, 0, prior, &keyframes[0]);
        }
        let chol = h.llt(faer::Side::Lower).unwrap();
        let x = chol.solve(&Mat::<f64>::from_fn(n, 1, |r, _| m[r]));
        (0..n).map(|r| x[(r, 0)]).collect()
    }

    /// Eliminating keyframes 0-2 one at a time and solving the window 3-5
    /// against the prior is the batch solve of all six, in the linear
    /// (one Gauss-Newton step) sense: the window's velocity and bias steps
    /// agree to 1e-9. The pose pseudo-priors stand in for vision; the
    /// frozen ones are the same in both, so nothing is counted twice.
    #[test]
    fn sliding_prior_matches_the_batch_solve() {
        let (keyframes, edges, params) = chain();
        let pseudo = FrozenPoseSigma::default();
        let batch = gn_step(&keyframes, &edges, &params, pseudo, None);

        let mut prior = None;
        for k in 0..3 {
            prior = Some(
                marginalize_frozen(
                    prior.as_ref(),
                    &keyframes[k],
                    &keyframes[k + 1],
                    &edges[k],
                    &params,
                    pseudo,
                )
                .unwrap(),
            );
        }
        let window_edges: Vec<ImuFactor> = edges[3..]
            .iter()
            .map(|e| ImuFactor {
                from_idx: e.from_idx - 3,
                to_idx: e.to_idx - 3,
                preintegrated: e.preintegrated.clone(),
            })
            .collect();
        let sliding = gn_step(
            &keyframes[3..],
            &window_edges,
            &params,
            pseudo,
            prior.as_ref(),
        );

        for k in 0..3 {
            for d in 0..DOF {
                let (b, s) = (batch[(3 + k) * DOF + d], sliding[k * DOF + d]);
                assert!(
                    (b - s).abs() < 1e-9,
                    "kf {} dof {d}: batch {b} sliding {s}",
                    3 + k
                );
            }
        }
        let last_v = batch[5 * DOF + 6];
        assert!(last_v.abs() > 1e-3, "step too small to test: {last_v}");
    }

    /// The prior's cost at `lin_point ⊕ d` is `½ dᵀHd + bᵀd`, and `⊖` undoes
    /// `⊕`, so the solver's retraction and the prior agree on the tangent.
    #[test]
    fn prior_cost_is_the_quadratic_in_the_retracted_tangent() {
        let mut rng = Lcg(3);
        let a: Vec<f64> = (0..225).map(|_| rng.next()).collect();
        let mut info = [0.0; 225];
        for r in 0..DOF {
            for c in 0..DOF {
                info[idx15(r, c)] = (0..DOF).map(|k| a[k * DOF + r] * a[k * DOF + c]).sum();
            }
            info[idx15(r, r)] += 1.0;
        }
        let lin_point = ViBaKeyframe {
            pose: pose_at(rng.vec3(2.0), 0.7),
            velocity: rng.vec3(1.0),
            bias: ImuBias {
                gyro: rng.vec3(1e-2),
                accel: rng.vec3(1e-1),
            },
            fixed: false,
        };
        let prior = MarginalPrior {
            frame_idx: 4,
            world_epoch: 1,
            lin_point: lin_point.clone(),
            info,
            rhs: std::array::from_fn(|_| rng.next()),
        };
        for scale in [1e-3, 0.1, 0.4] {
            let d: [f64; 15] = std::array::from_fn(|_| rng.next() * scale);
            let kf = retract(&lin_point, &d);
            let back = prior.tangent(&kf);
            for i in 0..DOF {
                assert!((back[i] - d[i]).abs() < 1e-12, "{scale}: tangent {i}");
            }
            let quadratic: f64 = (0..DOF)
                .map(|r| {
                    let hd: f64 = (0..DOF).map(|c| info[idx15(r, c)] * d[c]).sum();
                    0.5 * d[r] * hd + prior.rhs[r] * d[r]
                })
                .sum();
            let cost = prior.cost(&kf);
            assert!(
                (cost - quadratic).abs() < 1e-12,
                "{scale}: {cost} vs {quadratic}"
            );
        }
        assert!(!prior.is_stale(&lin_point));
    }
}
