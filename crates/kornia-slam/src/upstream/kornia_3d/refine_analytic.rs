//! Upstream target: kornia_3d::pnp::refine. Kept here until upstreamed; no kornia-slam types.
//!
//! Pose-only Levenberg-Marquardt with an analytic Jacobian.
//!
//! Same objective and control flow as kornia-3d's `refine_pose_lm` on
//! kornia-algebra's LM (Huber or L2 reprojection cost, the same damping,
//! tolerances and step rejection), but the 2x6 Jacobian of each residual is
//! closed-form and accumulated straight into a 6x6 normal matrix. The generic
//! solver's numerical f32 Jacobian costs about a microsecond per
//! correspondence per iteration and converges slowly; a tracker's per-frame
//! hot path needs the closed form.

// Fixed-size matrix loops read as the math they implement.
#![allow(clippy::needless_range_loop)]

use kornia_3d::pnp::LMRefineParams;
use kornia_3d::ransac::{RobustKernel, RobustKernelKind};
use kornia_algebra::{
    Mat3AF32, Mat3F64, Mat6F64, SMatF64, SO3F32, SO3F64, Vec2F32, Vec3AF32, Vec3F64, Vec6F64,
};

use crate::upstream::kornia_algebra::cast::{mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64};

const LAMBDA_FACTOR: f64 = 10.0;
const LAMBDA_MIN: f64 = 1e-10;
const LAMBDA_MAX: f64 = 1e10;
const STEP_TOLERANCE: f64 = 1e-6;

/// The robust kernel and squared scale `params` select, or `None` for a kernel
/// this solver does not implement.
fn kernel_from_params(params: &LMRefineParams) -> Option<(RobustKernelKind, f64)> {
    if params.robust == RobustKernelKind::Identity || !params.robust_scale_sq.is_finite() {
        return Some((RobustKernelKind::Identity, 0.0));
    }
    match params.robust {
        RobustKernelKind::Huber if params.robust_scale_sq > 0.0 => {
            Some((RobustKernelKind::Huber, f64::from(params.robust_scale_sq)))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct Pose {
    r: Mat3F64,
    t: Vec3F64,
}

impl Pose {
    fn transform(&self, p: Vec3F64) -> Vec3F64 {
        self.r * p + self.t
    }

    /// `self * exp(delta)` with `delta = (upsilon, omega)`: kornia's SE3
    /// `retract`, so the damped steps follow the generic solver's path.
    fn right_update(&self, delta: &Vec6F64) -> Self {
        let upsilon = Vec3F64::new(delta[0], delta[1], delta[2]);
        let omega = Vec3F64::new(delta[3], delta[4], delta[5]);
        Self {
            r: self.r * SO3F64::exp(omega).matrix(),
            t: self.t + self.r * (SO3F64::left_jacobian(omega) * upsilon),
        }
    }
}

struct Problem<'a> {
    points: &'a [Vec3F64],
    pixels: &'a [[f64; 2]],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    kernel: (RobustKernelKind, f64),
}

impl Problem<'_> {
    /// Robust cost at `pose`; `None` when a point is not in front of the
    /// camera, which the generic solver treats as an infeasible step.
    fn cost(&self, pose: &Pose) -> Option<f64> {
        let (kernel, c_sq) = self.kernel;
        let mut total = 0.0;
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let pc = pose.transform(*p);
            if pc.z <= 0.0 {
                return None;
            }
            let ru = self.fx * pc.x / pc.z + self.cx - uv[0];
            let rv = self.fy * pc.y / pc.z + self.cy - uv[1];
            total += kernel.rho(ru * ru + rv * rv, c_sq);
        }
        Some(total)
    }

    /// Weighted normal equations `(J^T W J, J^T W r)` at `pose`.
    fn normal_equations(&self, pose: &Pose) -> (Mat6F64, Vec6F64) {
        let (kernel, c_sq) = self.kernel;
        let mut h = Mat6F64::zeros();
        let mut g = Vec6F64::zeros();
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let pc = pose.transform(*p);
            let iz = 1.0 / pc.z;
            let r = [
                self.fx * pc.x * iz + self.cx - uv[0],
                self.fy * pc.y * iz + self.cy - uv[1],
            ];
            let w = kernel.weight(r[0] * r[0] + r[1] * r[1], c_sq);
            let jac = pose_jacobian_2x6(&pose.r, *p, pc, self.fx, self.fy);
            let (ju, jv): ([f64; 6], [f64; 6]) = (
                std::array::from_fn(|i| jac[(0, i)]),
                std::array::from_fn(|i| jac[(1, i)]),
            );
            for i in 0..6 {
                g[i] += w * (ju[i] * r[0] + jv[i] * r[1]);
                for j in i..6 {
                    h[(i, j)] += w * (ju[i] * ju[j] + jv[i] * jv[j]);
                }
            }
        }
        for i in 0..6 {
            for j in 0..i {
                h[(i, j)] = h[(j, i)];
            }
        }
        (h, g)
    }
}

/// Rows `d(u)/d(p_c)` and `d(v)/d(p_c)` of a pinhole projection at the camera-frame
/// point `point_cam` (`z > 0`), in pixels per metre.
pub fn pinhole_point_jacobian(fx: f64, fy: f64, point_cam: Vec3F64) -> [Vec3F64; 2] {
    let iz = 1.0 / point_cam.z;
    [
        Vec3F64::new(fx * iz, 0.0, -fx * point_cam.x * iz * iz),
        Vec3F64::new(0.0, fy * iz, -fy * point_cam.y * iz * iz),
    ]
}

/// Pixel Jacobian with respect to a right perturbation `T_cw · exp(δ)`, `δ = (υ, ω)` (the
/// order of `SE3F64::retract`), of the world-to-camera pose with rotation `rotation_cw`,
/// at world point `point_world` whose camera-frame position is `point_cam`.
pub fn pose_jacobian_2x6(
    rotation_cw: &Mat3F64,
    point_world: Vec3F64,
    point_cam: Vec3F64,
    fx: f64,
    fy: f64,
) -> SMatF64<2, 6> {
    // d(pixel)/d(p_c) [R | -R [p]x] for p_c = R exp(delta) p + t.
    let rows = pinhole_point_jacobian(fx, fy, point_cam).map(|row| {
        let a = rotation_cw.transpose() * row;
        let b = point_world.cross(a);
        [a.x, a.y, a.z, b.x, b.y, b.z]
    });
    SMatF64::<2, 6>::from_fn(|r, c| rows[r][c])
}

/// The damped Gauss-Newton step `-(H + lambda I)^-1 g`; `None` when the damped
/// system is not positive definite or the step is not finite.
fn damped_step(h: &Mat6F64, g: &Vec6F64, lambda: f64) -> Option<Vec6F64> {
    let step = (h + Mat6F64::identity() * lambda).cholesky()?.solve(&(-g));
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// Whether [`refine_pose_analytic`] implements `params`' robust kernel (Identity, or Huber
/// with a finite positive scale).
pub fn supports_kernel(params: &LMRefineParams) -> bool {
    kernel_from_params(params).is_some()
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
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    rotation: &Mat3AF32,
    translation: &Vec3AF32,
    params: &LMRefineParams,
) -> Option<RefinedPose> {
    let kernel = kernel_from_params(params)?;
    if points_world.len() != points_image.len() || points_world.len() < 3 {
        return None;
    }
    let points: Vec<Vec3F64> = points_world.iter().map(|p| vec3_to_f64(*p)).collect();
    let pixels: Vec<[f64; 2]> = points_image
        .iter()
        .map(|p| [p.x as f64, p.y as f64])
        .collect();
    let problem = Problem {
        points: &points,
        pixels: &pixels,
        fx: k.x_axis().x as f64,
        fy: k.y_axis().y as f64,
        cx: k.z_axis().x as f64,
        cy: k.z_axis().y as f64,
        kernel,
    };
    // The generic solver starts from the start rotation's unit quaternion,
    // which projects a not-quite-orthonormal prediction back onto SO(3).
    // The quaternion of a scaled matrix is not unit: normalize it, or the
    // scale survives into the result and compounds through the motion model.
    let rotation = SO3F32::from_quaternion(SO3F32::from_matrix(rotation).q.normalize()).matrix();
    let mut pose = Pose {
        r: mat3_to_f64(rotation),
        t: vec3_to_f64(*translation),
    };
    let mut cost = problem.cost(&pose)?;
    let mut lambda = params.initial_lambda as f64;
    let mut converged = false;
    for _ in 0..params.max_iterations {
        let (h, g) = problem.normal_equations(&pose);
        if g.norm() < params.gradient_tolerance as f64 {
            converged = true;
            break;
        }
        let delta = damped_step(&h, &g, lambda)?;
        if delta.norm() < STEP_TOLERANCE {
            converged = true;
            break;
        }
        let candidate = pose.right_update(&delta);
        let new_cost = problem.cost(&candidate).unwrap_or(f64::INFINITY);
        let relative_change = if cost > 0.0 {
            (cost - new_cost).abs() / cost
        } else {
            (cost - new_cost).abs()
        };
        if relative_change < params.cost_tolerance as f64 {
            // The generic solver keeps this last applied step.
            if new_cost.is_finite() {
                pose = candidate;
            }
            converged = true;
            break;
        }
        if new_cost < cost {
            pose = candidate;
            cost = new_cost;
            lambda = (lambda / LAMBDA_FACTOR).max(LAMBDA_MIN);
        } else {
            lambda *= LAMBDA_FACTOR;
            if lambda > LAMBDA_MAX {
                break;
            }
        }
    }
    Some(RefinedPose {
        rotation: mat3_to_f32(pose.r),
        translation: vec3_to_f32(pose.t),
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pnp::refine_pose_lm;

    /// Test-only oracles: the local bodies the kornia-rs calls replaced (P4 and P6 of
    /// `experiments/pr180-review-swaps-plan.md`).
    mod oracle {
        fn skew(w: [f64; 3]) -> [[f64; 3]; 3] {
            [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
        }

        fn identity_plus(a: f64, b: f64, w: [f64; 3]) -> [[f64; 3]; 3] {
            let k = skew(w);
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let kk: f64 = (0..3).map(|l| k[i][l] * k[l][j]).sum();
                    f64::from(u8::from(i == j)) + a * k[i][j] + b * kk
                })
            })
        }

        pub fn so3_exp(w: [f64; 3]) -> [[f64; 3]; 3] {
            let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            let (a, b) = if theta2 < 1e-12 {
                (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
            } else {
                let theta = theta2.sqrt();
                (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
            };
            identity_plus(a, b, w)
        }

        /// `V(w)` of the SE3 exponential, `t = V(w) upsilon`.
        pub fn se3_v(w: [f64; 3]) -> [[f64; 3]; 3] {
            let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            let (b, c) = if theta2 < 1e-12 {
                (0.5 - theta2 / 24.0, 1.0 / 6.0 - theta2 / 120.0)
            } else {
                let theta = theta2.sqrt();
                (
                    (1.0 - theta.cos()) / theta2,
                    (theta - theta.sin()) / (theta2 * theta),
                )
            };
            identity_plus(b, c, w)
        }

        /// Solves `a x = b` for a symmetric positive-definite 6x6 `a` (Cholesky).
        pub fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
            let mut l = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..=i {
                    let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                    if i == j {
                        let d = a[i][i] - s;
                        if d <= 0.0 || !d.is_finite() {
                            return None;
                        }
                        l[i][i] = d.sqrt();
                    } else {
                        l[i][j] = (a[i][j] - s) / l[j][j];
                    }
                }
            }
            let mut y = [0.0; 6];
            for i in 0..6 {
                y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
            }
            let mut x = [0.0; 6];
            for i in (0..6).rev() {
                x[i] = (y[i] - (i + 1..6).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
            }
            Some(x)
        }
    }

    /// Row `i`, column `j` of `m`.
    fn at(m: &Mat3F64, i: usize, j: usize) -> f64 {
        m.to_cols_array()[3 * j + i]
    }

    fn max_abs_difference(m: &Mat3F64, oracle: &[[f64; 3]; 3]) -> f64 {
        (0..9)
            .map(|n| (at(m, n / 3, n % 3) - oracle[n / 3][n % 3]).abs())
            .fold(0.0, f64::max)
    }

    fn rotation(w: [f64; 3]) -> Mat3F64 {
        SO3F64::exp(Vec3F64::from_array(w)).matrix()
    }

    fn scene(outlier: bool) -> (Vec<Vec3AF32>, Vec<Vec2F32>, Mat3AF32, Pose) {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(450.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 450.0, 0.0),
            Vec3AF32::new(376.0, 240.0, 1.0),
        );
        let truth = Pose {
            r: rotation([0.02, -0.03, 0.01]),
            t: Vec3F64::new(0.05, -0.02, 0.1),
        };
        let mut world = Vec::new();
        let mut image = Vec::new();
        for i in 0..60 {
            let p = Vec3F64::new(
                ((i * 37) % 17) as f64 * 0.1 - 0.8,
                ((i * 11) % 13) as f64 * 0.08 - 0.5,
                1.0 + ((i * 7) % 5) as f64 * 0.3,
            );
            let pc = truth.transform(p);
            let u = 450.0 * pc.x / pc.z + 376.0;
            let v = 450.0 * pc.y / pc.z + 240.0;
            // One gross outlier for the robust kernel to discount.
            let (du, dv) = if outlier && i == 5 {
                (40.0, -30.0)
            } else {
                (0.0, 0.0)
            };
            world.push(Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32));
            image.push(Vec2F32::new((u + du) as f32, (v + dv) as f32));
        }
        (world, image, k, truth)
    }

    fn params() -> LMRefineParams {
        LMRefineParams {
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 25.0,
            ..LMRefineParams::default()
        }
    }

    #[test]
    fn the_jacobian_matches_central_differences() {
        let pose = Pose {
            r: rotation([0.3, -0.2, 0.1]),
            t: Vec3F64::new(0.2, -0.1, 0.5),
        };
        let point = [Vec3F64::new(0.4, -0.3, 2.0)];
        let pixel = [[100.0, 80.0]];
        let problem = Problem {
            points: &point,
            pixels: &pixel,
            fx: 450.0,
            fy: 440.0,
            cx: 376.0,
            cy: 240.0,
            kernel: (RobustKernelKind::Identity, 0.0),
        };
        let residual = |pose: &Pose| {
            let pc = pose.transform(point[0]);
            [
                450.0 * pc.x / pc.z + 376.0 - pixel[0][0],
                440.0 * pc.y / pc.z + 240.0 - pixel[0][1],
            ]
        };
        let r0 = residual(&pose);
        let (h, g) = problem.normal_equations(&pose);
        // With one residual, g = J^T r and H = J^T J; recover J by differences.
        for k in 0..6 {
            let mut d = Vec6F64::zeros();
            d[k] = 1e-6;
            let plus = residual(&pose.right_update(&d));
            d[k] = -1e-6;
            let minus = residual(&pose.right_update(&d));
            let jk = [(plus[0] - minus[0]) / 2e-6, (plus[1] - minus[1]) / 2e-6];
            let gk = jk[0] * r0[0] + jk[1] * r0[1];
            assert!(
                (gk - g[k]).abs() < 1e-3 * (1.0 + gk.abs()),
                "g[{k}] {} vs {gk}",
                g[k]
            );
            let hkk = jk[0] * jk[0] + jk[1] * jk[1];
            assert!(
                (hkk - h[(k, k)]).abs() < 1e-3 * (1.0 + hkk),
                "h[{k}] {} vs {hkk}",
                h[(k, k)]
            );
        }
    }

    /// P5 of `experiments/pr180-C-card.md`: J read directly, not through `JᵀJ`.
    #[test]
    fn pose_jacobian_2x6_matches_central_differences() {
        let pose = Pose {
            r: rotation([0.3, -0.2, 0.1]),
            t: Vec3F64::new(0.2, -0.1, 0.5),
        };
        let point = Vec3F64::new(0.4, -0.3, 2.0);
        let (fx, fy) = (450.0, 440.0);
        let pixel = |pose: &Pose| {
            let pc = pose.transform(point);
            [fx * pc.x / pc.z, fy * pc.y / pc.z]
        };
        let jacobian = pose_jacobian_2x6(&pose.r, point, pose.transform(point), fx, fy);
        for k in 0..6 {
            let mut d = Vec6F64::zeros();
            d[k] = 1e-6;
            let plus = pixel(&pose.right_update(&d));
            d[k] = -1e-6;
            let minus = pixel(&pose.right_update(&d));
            for row in 0..2 {
                let numeric = (plus[row] - minus[row]) / 2e-6;
                let analytic = jacobian[(row, k)];
                assert!(
                    (numeric - analytic).abs() < 1e-6 * (1.0 + analytic.abs()),
                    "J[{row}][{k}] {analytic} vs {numeric}"
                );
            }
        }
    }

    /// `Q diag(eigenvalues) Qᵀ` for a fixed orthogonal `Q`.
    fn symmetric_with(eigenvalues: [f64; 6]) -> Mat6F64 {
        let seed = Mat6F64::from_fn(|r, c| ((r * 5 + c * 11) as f64 * 0.29).cos());
        let q = seed.qr().q();
        q * Mat6F64::from_diagonal(&Vec6F64::from(eigenvalues)) * q.transpose()
    }

    fn rows(m: &Mat6F64) -> [[f64; 6]; 6] {
        std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
    }

    #[test]
    fn the_cholesky_step_matches_the_local_solver() {
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        for kappa in [1.0f64, 1e2, 1e4] {
            let h = symmetric_with(std::array::from_fn(|i| kappa.powf(i as f64 / 5.0)));
            let new = damped_step(&h, &g, 0.0).unwrap();
            let old = Vec6F64::from(oracle::solve6(&rows(&h), &(-g).into()).unwrap());
            let relative = (new - old).norm() / old.norm();
            assert!(relative <= 1e-12, "kappa {kappa}: {relative:e}");
        }
    }

    #[test]
    fn a_system_that_is_not_positive_definite_is_refused_by_both_solvers() {
        let h = symmetric_with([1.0, 2.0, 3.0, 4.0, 5.0, -1.0]);
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        assert!(damped_step(&h, &g, 0.0).is_none());
        assert!(oracle::solve6(&rows(&h), &(-g).into()).is_none());
    }

    #[test]
    fn so3_exp_and_left_jacobian_match_the_local_bodies() {
        let axis = Vec3F64::new(0.3, -0.2, 0.9).normalize();
        for theta in [1e-7, 1e-5, 1e-3, 0.05, 0.5, 2.0, 3.0] {
            let w = axis * theta;
            let w_array = w.to_array();
            let exp = SO3F64::exp(w).matrix();
            let exp_error = max_abs_difference(&exp, &oracle::so3_exp(w_array));
            assert!(exp_error <= 1e-15, "exp at {theta}: {exp_error:e}");
            let jacobian = SO3F64::left_jacobian(w);
            let jacobian_error = max_abs_difference(&jacobian, &oracle::se3_v(w_array));
            let bound = 4e-16 / theta + 1e-15;
            assert!(
                jacobian_error <= bound,
                "left Jacobian at {theta}: {jacobian_error:e} > {bound:e}"
            );
        }
    }

    #[test]
    fn huber_matches_the_local_loss() {
        let (c_sq, delta) = (25.0f64, 5.0f64);
        for squared in [0.0f64, 1.0, 24.9, 25.0, 25.1, 400.0, 1e6] {
            let s = squared.sqrt();
            let old_rho = if s <= delta {
                squared
            } else {
                2.0 * delta * s - delta * delta
            };
            let old_weight = if squared <= delta * delta {
                1.0
            } else {
                delta / s
            };
            let rho = RobustKernelKind::Huber.rho(squared, c_sq);
            let weight = RobustKernelKind::Huber.weight(squared, c_sq);
            assert!(
                (rho - old_rho).abs() <= 1e-15 * old_rho.max(1.0),
                "{squared}"
            );
            assert!(
                (weight - old_weight).abs() <= 1e-15 * old_weight,
                "{squared}"
            );
        }
    }

    #[test]
    fn recovers_a_perturbed_pose() {
        let (world, image, k, truth) = scene(false);
        let start = Mat3AF32::IDENTITY;
        let refined =
            refine_pose_analytic(&world, &image, &k, &start, &Vec3AF32::ZERO, &params()).unwrap();
        let (r, t) = (refined.rotation, refined.translation);
        assert!(refined.converged);
        for i in 0..3 {
            for j in 0..3 {
                let col = [r.x_axis(), r.y_axis(), r.z_axis()][j].to_array();
                assert!((col[i] as f64 - at(&truth.r, i, j)).abs() < 1e-4);
            }
        }
        assert!((t.x as f64 - truth.t.x).abs() < 1e-4);
        assert!((t.y as f64 - truth.t.y).abs() < 1e-4);
        assert!((t.z as f64 - truth.t.z).abs() < 1e-4);
    }

    #[test]
    fn agrees_with_the_generic_solver() {
        let (world, image, k, _) = scene(true);
        let start = Mat3AF32::IDENTITY;
        let refined =
            refine_pose_analytic(&world, &image, &k, &start, &Vec3AF32::ZERO, &params()).unwrap();
        let (r, t) = (refined.rotation, refined.translation);
        let generic =
            refine_pose_lm(&world, &image, &k, &start, &Vec3AF32::ZERO, None, &params()).unwrap();
        assert!((t - generic.translation).length() < 1e-3);
        for (a, b) in [
            (r.x_axis(), generic.rotation.x_axis()),
            (r.y_axis(), generic.rotation.y_axis()),
            (r.z_axis(), generic.rotation.z_axis()),
        ] {
            assert!((a - b).length() < 1e-3);
        }
    }

    #[test]
    fn a_scaled_start_rotation_is_projected_onto_so3_like_the_generic_solver() {
        let (world, image, k, truth) = scene(false);
        // A prediction that drifted off SO(3): a rotation scaled by 0.9.
        let rotation = rotation([0.03, -0.01, 0.02]);
        let col = |j: usize| {
            Vec3AF32::new(
                0.9 * at(&rotation, 0, j) as f32,
                0.9 * at(&rotation, 1, j) as f32,
                0.9 * at(&rotation, 2, j) as f32,
            )
        };
        let scaled = Mat3AF32::from_cols(col(0), col(1), col(2));
        let refined =
            refine_pose_analytic(&world, &image, &k, &scaled, &Vec3AF32::ZERO, &params()).unwrap();
        let t = refined.translation;
        assert!((t.x as f64 - truth.t.x).abs() < 1e-4);
        assert!((t.y as f64 - truth.t.y).abs() < 1e-4);
        assert!((t.z as f64 - truth.t.z).abs() < 1e-4);
        let r = refined.rotation;
        for column in [r.x_axis(), r.y_axis(), r.z_axis()] {
            assert!((column.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn a_point_behind_the_start_pose_fails_like_the_generic_solver() {
        let (mut world, image, k, _) = scene(true);
        world[0] = Vec3AF32::new(0.0, 0.0, -1.0);
        let out = refine_pose_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &params(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn unsupported_kernels_are_declined() {
        let (world, image, k, _) = scene(true);
        let cauchy = LMRefineParams {
            robust: RobustKernelKind::Cauchy,
            robust_scale_sq: 25.0,
            ..LMRefineParams::default()
        };
        let out = refine_pose_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &cauchy,
        );
        assert!(out.is_none());
    }
}
