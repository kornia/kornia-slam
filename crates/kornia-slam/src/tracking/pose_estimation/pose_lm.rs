//! Pose-only Levenberg-Marquardt with an analytic Jacobian.
//!
//! Same objective and control flow as kornia-3d's `refine_pose_lm` on
//! kornia-algebra's LM (Huber or L2 reprojection cost, the same damping,
//! tolerances and step rejection), but the 2x6 Jacobian of each residual is
//! closed-form and accumulated straight into a 6x6 normal matrix. The generic
//! solver's numerical f32 Jacobian costs about a microsecond per
//! correspondence per iteration and converges slowly; this is the tracker's
//! per-frame hot path, so it is solved here. It belongs upstream in kornia-3d.

// Fixed-size matrix loops read as the math they implement.
#![allow(clippy::needless_range_loop)]

use kornia_3d::pnp::LMRefineParams;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, SO3F32, Vec2F32, Vec3AF32};

const LAMBDA_FACTOR: f64 = 10.0;
const LAMBDA_MIN: f64 = 1e-10;
const LAMBDA_MAX: f64 = 1e10;
const STEP_TOLERANCE: f64 = 1e-6;

/// Robust cost of one squared residual norm, and its IRLS weight.
#[derive(Clone, Copy)]
enum Loss {
    L2,
    Huber { delta: f64 },
}

impl Loss {
    /// The loss `params` selects, or `None` for one this solver does not
    /// implement.
    fn from_params(params: &LMRefineParams) -> Option<Self> {
        if params.robust == RobustKernelKind::Identity || !params.robust_scale_sq.is_finite() {
            return Some(Self::L2);
        }
        match params.robust {
            RobustKernelKind::Huber if params.robust_scale_sq > 0.0 => Some(Self::Huber {
                delta: (params.robust_scale_sq as f64).sqrt(),
            }),
            _ => None,
        }
    }

    fn rho(self, squared: f64) -> f64 {
        match self {
            Self::L2 => squared,
            Self::Huber { delta } => {
                let s = squared.sqrt();
                if s <= delta {
                    squared
                } else {
                    2.0 * delta * s - delta * delta
                }
            }
        }
    }

    fn weight(self, squared: f64) -> f64 {
        match self {
            Self::L2 => 1.0,
            Self::Huber { delta } => {
                if squared <= delta * delta {
                    1.0
                } else {
                    delta / squared.sqrt()
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Pose {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

impl Pose {
    fn transform(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + self.t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + self.t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + self.t[2],
        ]
    }

    /// `self * exp(delta)` with `delta = (upsilon, omega)`: kornia's SE3
    /// `retract`, so the damped steps follow the generic solver's path.
    fn right_update(&self, delta: &[f64; 6]) -> Self {
        let (e, v) = se3_exp_parts([delta[3], delta[4], delta[5]]);
        let vu: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| v[i][k] * delta[k]).sum());
        let mut r = [[0.0; 3]; 3];
        let mut t = self.t;
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (0..3).map(|k| self.r[i][k] * e[k][j]).sum();
            }
            t[i] += (0..3).map(|k| self.r[i][k] * vu[k]).sum::<f64>();
        }
        Self { r, t }
    }
}

/// `(Exp(w), V(w))` of the SE3 exponential, `t = V(w) upsilon`.
fn se3_exp_parts(w: [f64; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
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
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let mut v = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let kk: f64 = (0..3).map(|l| k[i][l] * k[l][j]).sum();
            v[i][j] = f64::from(u8::from(i == j)) + b * k[i][j] + c * kk;
        }
    }
    (so3_exp(w), v)
}

fn so3_exp(w: [f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let (a, b) = if theta2 < 1e-12 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        let theta = theta2.sqrt();
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let kk: f64 = (0..3).map(|l| k[i][l] * k[l][j]).sum();
            m[i][j] = f64::from(u8::from(i == j)) + a * k[i][j] + b * kk;
        }
    }
    m
}

struct Problem<'a> {
    points: &'a [[f64; 3]],
    pixels: &'a [[f64; 2]],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    loss: Loss,
}

impl Problem<'_> {
    /// Robust cost at `pose`; `None` when a point is not in front of the
    /// camera, which the generic solver treats as an infeasible step.
    fn cost(&self, pose: &Pose) -> Option<f64> {
        let mut total = 0.0;
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let pc = pose.transform(*p);
            if pc[2] <= 0.0 {
                return None;
            }
            let ru = self.fx * pc[0] / pc[2] + self.cx - uv[0];
            let rv = self.fy * pc[1] / pc[2] + self.cy - uv[1];
            total += self.loss.rho(ru * ru + rv * rv);
        }
        Some(total)
    }

    /// Weighted normal equations `(J^T W J, J^T W r)` at `pose`.
    fn normal_equations(&self, pose: &Pose) -> ([[f64; 6]; 6], [f64; 6]) {
        let mut h = [[0.0; 6]; 6];
        let mut g = [0.0; 6];
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let [x, y, z] = pose.transform(*p);
            let iz = 1.0 / z;
            let r = [
                self.fx * x * iz + self.cx - uv[0],
                self.fy * y * iz + self.cy - uv[1],
            ];
            let w = self.loss.weight(r[0] * r[0] + r[1] * r[1]);
            // d(pixel)/d(p_c) [R | -R [p]x] for p_c = R exp(delta) p + t.
            let jpi = [
                [self.fx * iz, 0.0, -self.fx * x * iz * iz],
                [0.0, self.fy * iz, -self.fy * y * iz * iz],
            ];
            let rows = jpi.map(|row| -> [f64; 3] {
                std::array::from_fn(|k| (0..3).map(|l| row[l] * pose.r[l][k]).sum())
            });
            let [px, py, pz] = *p;
            let jac = rows.map(|a| {
                [
                    a[0],
                    a[1],
                    a[2],
                    -(a[1] * pz - a[2] * py),
                    -(a[2] * px - a[0] * pz),
                    -(a[0] * py - a[1] * px),
                ]
            });
            let (ju, jv) = (jac[0], jac[1]);
            for i in 0..6 {
                g[i] += w * (ju[i] * r[0] + jv[i] * r[1]);
                for j in i..6 {
                    h[i][j] += w * (ju[i] * ju[j] + jv[i] * jv[j]);
                }
            }
        }
        for i in 0..6 {
            for j in 0..i {
                h[i][j] = h[j][i];
            }
        }
        (h, g)
    }
}

/// Solves `a x = b` for a symmetric positive-definite 6x6 `a` (Cholesky).
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
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

fn norm(v: &[f64; 6]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Whether [`refine_pose`] implements `params`' robust kernel.
pub(crate) fn supports(params: &LMRefineParams) -> bool {
    Loss::from_params(params).is_some()
}

/// A refined pose and whether a tolerance, not the iteration cap, ended it.
pub(crate) struct Refined {
    pub rotation: Mat3AF32,
    pub translation: Vec3AF32,
    pub converged: bool,
}

/// Refines a world-to-camera pose against `points_world` / `points_image`
/// (pixels, no distortion). `None` when the solver does not implement
/// `params`' robust kernel, when an initial point is not in front of the
/// camera, or when the damped system is singular.
pub(crate) fn refine_pose(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    rotation: &Mat3AF32,
    translation: &Vec3AF32,
    params: &LMRefineParams,
) -> Option<Refined> {
    let loss = Loss::from_params(params)?;
    if points_world.len() != points_image.len() || points_world.len() < 3 {
        return None;
    }
    let points: Vec<[f64; 3]> = points_world
        .iter()
        .map(|p| [p.x as f64, p.y as f64, p.z as f64])
        .collect();
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
        loss,
    };
    // The generic solver starts from the start rotation's quaternion, which
    // projects a not-quite-orthonormal prediction back onto SO(3).
    let rotation = SO3F32::from_matrix(rotation).matrix();
    let columns = [rotation.x_axis(), rotation.y_axis(), rotation.z_axis()].map(|c| c.to_array());
    let mut pose = Pose {
        r: std::array::from_fn(|i| std::array::from_fn(|j| columns[j][i] as f64)),
        t: [
            translation.x as f64,
            translation.y as f64,
            translation.z as f64,
        ],
    };
    let mut cost = problem.cost(&pose)?;
    let mut lambda = params.initial_lambda as f64;
    let mut converged = false;
    for _ in 0..params.max_iterations {
        let (mut h, g) = problem.normal_equations(&pose);
        if norm(&g) < params.gradient_tolerance as f64 {
            converged = true;
            break;
        }
        for (i, row) in h.iter_mut().enumerate() {
            row[i] += lambda;
        }
        let delta = solve6(&h, &g.map(|v| -v))?;
        if norm(&delta) < STEP_TOLERANCE {
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
    let r = pose.r;
    let rotation = Mat3AF32::from_cols(
        Vec3AF32::new(r[0][0] as f32, r[1][0] as f32, r[2][0] as f32),
        Vec3AF32::new(r[0][1] as f32, r[1][1] as f32, r[2][1] as f32),
        Vec3AF32::new(r[0][2] as f32, r[1][2] as f32, r[2][2] as f32),
    );
    let translation = Vec3AF32::new(pose.t[0] as f32, pose.t[1] as f32, pose.t[2] as f32);
    Some(Refined {
        rotation,
        translation,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pnp::refine_pose_lm;

    fn scene(outlier: bool) -> (Vec<Vec3AF32>, Vec<Vec2F32>, Mat3AF32, Pose) {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(450.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 450.0, 0.0),
            Vec3AF32::new(376.0, 240.0, 1.0),
        );
        let truth = Pose {
            r: so3_exp([0.02, -0.03, 0.01]),
            t: [0.05, -0.02, 0.1],
        };
        let mut world = Vec::new();
        let mut image = Vec::new();
        for i in 0..60 {
            let p = [
                ((i * 37) % 17) as f64 * 0.1 - 0.8,
                ((i * 11) % 13) as f64 * 0.08 - 0.5,
                1.0 + ((i * 7) % 5) as f64 * 0.3,
            ];
            let pc = truth.transform(p);
            let u = 450.0 * pc[0] / pc[2] + 376.0;
            let v = 450.0 * pc[1] / pc[2] + 240.0;
            // One gross outlier for the robust kernel to discount.
            let (du, dv) = if outlier && i == 5 {
                (40.0, -30.0)
            } else {
                (0.0, 0.0)
            };
            world.push(Vec3AF32::new(p[0] as f32, p[1] as f32, p[2] as f32));
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
            r: so3_exp([0.3, -0.2, 0.1]),
            t: [0.2, -0.1, 0.5],
        };
        let point = [[0.4, -0.3, 2.0]];
        let pixel = [[100.0, 80.0]];
        let problem = Problem {
            points: &point,
            pixels: &pixel,
            fx: 450.0,
            fy: 440.0,
            cx: 376.0,
            cy: 240.0,
            loss: Loss::L2,
        };
        let residual = |pose: &Pose| {
            let pc = pose.transform(point[0]);
            [
                450.0 * pc[0] / pc[2] + 376.0 - pixel[0][0],
                440.0 * pc[1] / pc[2] + 240.0 - pixel[0][1],
            ]
        };
        let r0 = residual(&pose);
        let (h, g) = problem.normal_equations(&pose);
        // With one residual, g = J^T r and H = J^T J; recover J by differences.
        for k in 0..6 {
            let mut d = [0.0; 6];
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
                (hkk - h[k][k]).abs() < 1e-3 * (1.0 + hkk),
                "h[{k}] {} vs {hkk}",
                h[k][k]
            );
        }
    }

    #[test]
    fn recovers_a_perturbed_pose() {
        let (world, image, k, truth) = scene(false);
        let start = Mat3AF32::IDENTITY;
        let refined = refine_pose(&world, &image, &k, &start, &Vec3AF32::ZERO, &params()).unwrap();
        let (r, t) = (refined.rotation, refined.translation);
        assert!(refined.converged);
        for i in 0..3 {
            for j in 0..3 {
                let col = [r.x_axis(), r.y_axis(), r.z_axis()][j].to_array();
                assert!((col[i] as f64 - truth.r[i][j]).abs() < 1e-4);
            }
        }
        assert!((t.x as f64 - truth.t[0]).abs() < 1e-4);
        assert!((t.y as f64 - truth.t[1]).abs() < 1e-4);
        assert!((t.z as f64 - truth.t[2]).abs() < 1e-4);
    }

    #[test]
    fn agrees_with_the_generic_solver() {
        let (world, image, k, _) = scene(true);
        let start = Mat3AF32::IDENTITY;
        let refined = refine_pose(&world, &image, &k, &start, &Vec3AF32::ZERO, &params()).unwrap();
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
        // A prediction that drifted off SO(3): the identity scaled by 1.1.
        let scaled = Mat3AF32::from_cols(
            Vec3AF32::new(1.1, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.1, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.1),
        );
        let refined = refine_pose(&world, &image, &k, &scaled, &Vec3AF32::ZERO, &params()).unwrap();
        let t = refined.translation;
        assert!((t.x as f64 - truth.t[0]).abs() < 1e-4);
        assert!((t.y as f64 - truth.t[1]).abs() < 1e-4);
        assert!((t.z as f64 - truth.t[2]).abs() < 1e-4);
        let r = refined.rotation;
        for column in [r.x_axis(), r.y_axis(), r.z_axis()] {
            assert!((column.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn a_point_behind_the_start_pose_fails_like_the_generic_solver() {
        let (mut world, image, k, _) = scene(true);
        world[0] = Vec3AF32::new(0.0, 0.0, -1.0);
        let out = refine_pose(
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
        let out = refine_pose(
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
