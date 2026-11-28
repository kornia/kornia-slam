//! Pose of a multi-camera rig from 3D-2D matches seen by any of its cameras.
//!
//! Each residual is ORB-SLAM3's `EdgeSE3ProjectXYZOnlyPoseToBody`: the world
//! point goes through the primary-camera pose and then the fixed
//! camera-from-primary extrinsic of the camera that observed it. The solve
//! mirrors [`super::pnp::solve_pnp`]: gate on the prior, Huber Gauss-Newton
//! with Levenberg damping, hard outlier rounds, then count final inliers.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{HuberKernel, RobustKernel};
use kornia_algebra::{Mat3F64, Mat6F64, SO3F64, Vec3F64, Vec6F64};

use super::pnp::PnpConfig;

/// One match: a world point and the undistorted pixel that camera `camera`
/// observed it at, with that camera's `T_CP`.
#[derive(Debug, Clone, Copy)]
pub struct RigObservation<'a> {
    pub point_world: Vec3F64,
    pub pixel: [f64; 2],
    pub camera: &'a PinholeCamera,
    /// `None` for the primary camera.
    pub camera_from_primary: Option<&'a Pose3d>,
}

impl RigObservation<'_> {
    fn point_in_camera(&self, pose_world_to_primary: &Pose3d) -> (Vec3F64, Vec3F64) {
        let p_primary = pose_world_to_primary.transform_point(&self.point_world);
        let p_camera = match self.camera_from_primary {
            Some(t) => t.transform_point(&p_primary),
            None => p_primary,
        };
        (p_primary, p_camera)
    }

    fn error_sq(&self, pose: &Pose3d) -> Option<f64> {
        let (_, p_camera) = self.point_in_camera(pose);
        self.camera
            .reprojection_error_sq_cam(&p_camera, self.pixel[0], self.pixel[1])
    }
}

/// Smallest depth (metres) a residual is linearized at.
const MIN_DEPTH_M: f64 = 1e-3;
/// Relative cost decrease below which an LM round stops.
const COST_TOLERANCE: f64 = 1e-9;
/// Smallest squared Cholesky pivot a damped system may have and still be solved: the
/// singularity threshold of the pivoted elimination this solve replaced.
const MIN_PIVOT_SQ: f64 = 1e-12;

/// Refines the primary-camera pose (world-to-camera) from `observations`;
/// returns it with the number of final inliers, or `None` when too few
/// matches survive the prior gate or the normal equations are singular.
pub fn solve_rig_pnp(
    observations: &[RigObservation<'_>],
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> Option<(Pose3d, usize)> {
    if observations.len() < config.min_correspondences {
        return None;
    }
    let prior_th2 = config.prior_reproj_threshold_px.powi(2);
    let mut active: Vec<usize> = (0..observations.len())
        .filter(|&i| {
            observations[i]
                .error_sq(pose_init)
                .is_some_and(|e| e <= prior_th2)
        })
        .collect();
    let final_th2 = config.final_reproj_threshold_px.powi(2);
    let c_sq = f64::from(config.robust_scale_sq);
    let rounds = config.outlier_rounds.max(1);
    let mut pose = *pose_init;
    for round in 0..rounds {
        if active.len() < config.min_correspondences {
            return None;
        }
        pose = refine(
            observations,
            &active,
            pose_init,
            c_sq,
            config.lm_max_iterations.max(1),
        )?;
        if round + 1 < rounds {
            active.retain(|&i| {
                observations[i]
                    .error_sq(&pose)
                    .is_some_and(|e| e <= final_th2)
            });
        }
    }
    let inliers = observations
        .iter()
        .filter(|o| o.error_sq(&pose).is_some_and(|e| e <= final_th2))
        .count();
    Some((pose, inliers))
}

/// Huber-weighted Levenberg-Marquardt over the pose, from `start`, with a
/// left perturbation `T <- Exp([rho; omega]) T`.
fn refine(
    observations: &[RigObservation<'_>],
    active: &[usize],
    start: &Pose3d,
    c_sq: f64,
    max_iterations: usize,
) -> Option<Pose3d> {
    let mut pose = *start;
    let mut cost = robust_cost(observations, active, &pose, c_sq);
    let mut lambda = 1e-3;
    for _ in 0..max_iterations {
        let (h, g) = normal_equations(observations, active, &pose, c_sq);
        let mut damped = h;
        for d in 0..6 {
            damped[(d, d)] += lambda * h[(d, d)].max(1e-9);
        }
        let step = solve_damped(damped, &g)?;
        let candidate = apply_step(&pose, &step);
        let candidate_cost = robust_cost(observations, active, &candidate, c_sq);
        if candidate_cost < cost {
            let decrease = (cost - candidate_cost) / cost.max(1e-12);
            pose = candidate;
            cost = candidate_cost;
            lambda = (lambda * 0.1).max(1e-9);
            if decrease < COST_TOLERANCE {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    Some(pose)
}

fn robust_cost(
    observations: &[RigObservation<'_>],
    active: &[usize],
    pose: &Pose3d,
    c_sq: f64,
) -> f64 {
    active
        .iter()
        .map(|&i| {
            let (_, p) = observations[i].point_in_camera(pose);
            let Some(r) = residual(&observations[i], &p) else {
                return c_sq * 1e3;
            };
            HuberKernel.rho(r[0] * r[0] + r[1] * r[1], c_sq)
        })
        .sum()
}

fn residual(observation: &RigObservation<'_>, p_camera: &Vec3F64) -> Option<[f64; 2]> {
    let pixel = observation.camera.project_to_pixel(p_camera, MIN_DEPTH_M)?;
    Some([
        pixel.x - observation.pixel[0],
        pixel.y - observation.pixel[1],
    ])
}

/// Gauss-Newton `H = Σ w Jᵀ J`, `g = -Σ w Jᵀ r` with
/// `J = J_proj · R_CP · [I | -[X_P]×]`; both triangles of `H` are filled.
fn normal_equations(
    observations: &[RigObservation<'_>],
    active: &[usize],
    pose: &Pose3d,
    c_sq: f64,
) -> (Mat6F64, Vec6F64) {
    let mut h = Mat6F64::zeros();
    let mut g = Vec6F64::zeros();
    for &i in active {
        let observation = &observations[i];
        let (p_primary, p_camera) = observation.point_in_camera(pose);
        let Some(r) = residual(observation, &p_camera) else {
            continue;
        };
        let w = HuberKernel.weight(r[0] * r[0] + r[1] * r[1], c_sq);
        let z = p_camera.z.max(MIN_DEPTH_M);
        let (fx, fy) = (observation.camera.fx, observation.camera.fy);
        let j_proj = [
            [fx / z, 0.0, -fx * p_camera.x / (z * z)],
            [0.0, fy / z, -fy * p_camera.y / (z * z)],
        ];
        let r_cp = observation
            .camera_from_primary
            .map_or(Mat3F64::IDENTITY, |t| t.rotation);
        let skew = SO3F64::hat(p_primary);
        // d X_P / d [rho; omega] = [I | -[X_P]x], then rotated into the camera.
        let mut jacobian = [[0.0; 6]; 2];
        for (row, j_row) in j_proj.iter().zip(jacobian.iter_mut()) {
            let a = r_cp.transpose() * Vec3F64::new(row[0], row[1], row[2]);
            let b = skew.transpose() * a * -1.0;
            *j_row = [a.x, a.y, a.z, b.x, b.y, b.z];
        }
        for (row, &res) in jacobian.iter().zip(&r) {
            for a in 0..6 {
                g[a] -= w * row[a] * res;
                for b in 0..6 {
                    h[(a, b)] += w * row[a] * row[b];
                }
            }
        }
    }
    (h, g)
}

fn apply_step(pose: &Pose3d, step: &Vec6F64) -> Pose3d {
    let rotation = SO3F64::exp(Vec3F64::new(step[3], step[4], step[5])).matrix();
    Pose3d::new(
        rotation * pose.rotation,
        rotation * pose.translation + Vec3F64::new(step[0], step[1], step[2]),
    )
}

/// `damped⁻¹ g` by Cholesky; `None` when a squared pivot is below [`MIN_PIVOT_SQ`] or the
/// system is not positive definite.
fn solve_damped(damped: Mat6F64, g: &Vec6F64) -> Option<Vec6F64> {
    let cholesky = damped.cholesky()?;
    let l = cholesky.l_dirty();
    if (0..6).any(|d| l[(d, d)] * l[(d, d)] < MIN_PIVOT_SQ) {
        return None;
    }
    Some(cholesky.solve(g))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The solver [`solve_damped`] replaced, kept as its oracle (P5 of
    /// `experiments/pr180-review-swaps-plan.md`): Gaussian elimination with partial pivoting,
    /// `None` when singular.
    fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
        for col in 0..6 {
            let pivot = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
            if a[pivot][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, pivot);
            b.swap(col, pivot);
            let pivot_row = a[col];
            for row in col + 1..6 {
                let factor = a[row][col] / pivot_row[col];
                for (x, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                    *x -= factor * p;
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = [0.0; 6];
        for row in (0..6).rev() {
            let tail: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
            x[row] = (b[row] - tail) / a[row][row];
        }
        Some(x)
    }

    /// `Q diag(eigenvalues) Qᵀ` for a fixed orthogonal `Q`.
    fn symmetric_with(eigenvalues: [f64; 6]) -> Mat6F64 {
        let seed = Mat6F64::from_fn(|r, c| ((r * 7 + c * 3) as f64 * 0.37).sin());
        let q = seed.qr().q();
        q * Mat6F64::from_diagonal(&Vec6F64::from(eigenvalues)) * q.transpose()
    }

    fn rows(m: &Mat6F64) -> [[f64; 6]; 6] {
        std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
    }

    #[test]
    fn cholesky_matches_pivoted_elimination() {
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        for (kappa, tolerance) in [(1.0f64, 1e-10), (1e2, 1e-10), (1e4, 1e-10), (1e8, 1e-6)] {
            let eigenvalues: [f64; 6] = std::array::from_fn(|i| kappa.powf(i as f64 / 5.0));
            let h = symmetric_with(eigenvalues);
            let new = solve_damped(h, &g).unwrap();
            let old = Vec6F64::from(solve6(rows(&h), g.into()).unwrap());
            let relative = (new - old).norm() / old.norm();
            assert!(relative <= tolerance, "kappa {kappa}: {relative:e}");
        }
    }

    #[test]
    fn a_rank_deficient_system_is_refused_by_both_solvers() {
        let h = symmetric_with([1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        assert!(solve_damped(h, &g).is_none());
        assert!(solve6(rows(&h), g.into()).is_none());
    }

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 240.0,
            fy: 240.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn yaw(deg: f64) -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix()
    }

    #[test]
    fn matches_the_single_camera_solver_with_noise_and_outliers() {
        let camera = camera();
        let truth = Pose3d::new(yaw(2.0), Vec3F64::new(0.05, -0.02, 0.1));
        let mut observations = Vec::new();
        let mut seed = 7u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for i in 0..200 {
            let p = Vec3F64::new(
                (i % 20) as f64 * 0.3 - 3.0,
                (i / 20) as f64 * 0.3 - 1.5,
                2.0 + (i % 7) as f64,
            );
            let world = truth.inverse().transform_point(&p);
            let pixel = camera.project_to_pixel(&p, 0.1).unwrap();
            let outlier = i % 4 == 0;
            let (du, dv) = if outlier {
                (noise() * 40.0, noise() * 40.0)
            } else {
                (noise(), noise())
            };
            observations.push(RigObservation {
                point_world: world,
                pixel: [pixel.x + du, pixel.y + dv],
                camera: &camera,
                camera_from_primary: None,
            });
        }
        let start = Pose3d::new(yaw(5.0), Vec3F64::new(0.1, -0.02, 0.05));
        let (pose, inliers) = solve_rig_pnp(&observations, &start, &PnpConfig::default()).unwrap();
        let points: Vec<Vec3F64> = observations.iter().map(|o| o.point_world).collect();
        let pixels: Vec<kornia_algebra::Vec2F32> = observations
            .iter()
            .map(|o| kornia_algebra::Vec2F32::new(o.pixel[0] as f32, o.pixel[1] as f32))
            .collect();
        let (mono, mono_inliers) =
            super::super::pnp::solve_pnp(&points, &pixels, &camera, &start, &PnpConfig::default())
                .unwrap();
        assert!(inliers >= 140, "inliers {inliers}");
        assert_eq!(inliers, mono_inliers);
        assert!((pose.translation - truth.translation).length() < 0.01);
        assert!((pose.translation - mono.translation).length() < 1e-3);
    }

    #[test]
    fn recovers_the_rig_pose_from_both_cameras() {
        let camera = camera();
        // Camera 1 looks 70 deg to the side of the primary, 11 cm away.
        let camera1_from_primary = Pose3d::new(yaw(70.0), Vec3F64::new(0.11, 0.0, 0.0)).inverse();
        let truth = Pose3d::new(yaw(5.0), Vec3F64::new(0.2, -0.1, 0.3));
        let mut points = Vec::new();
        for i in 0..40 {
            let (x, y) = ((i % 8) as f64 * 0.5 - 2.0, (i / 8) as f64 * 0.4 - 1.0);
            points.push((Vec3F64::new(x, y, 5.0 + (i % 3) as f64), None));
            points.push((
                Vec3F64::new(6.0 + (i % 3) as f64, y, x),
                Some(&camera1_from_primary),
            ));
        }
        let observations: Vec<RigObservation> = points
            .iter()
            .filter_map(|(p_primary, extrinsic)| {
                let point_world = truth.inverse().transform_point(p_primary);
                let p_camera = match extrinsic {
                    Some(t) => t.transform_point(p_primary),
                    None => *p_primary,
                };
                let pixel = camera.project_to_image(
                    &p_camera,
                    0.1,
                    kornia_image::ImageSize {
                        width: 640,
                        height: 480,
                    },
                );
                pixel.ok().map(|pixel| RigObservation {
                    point_world,
                    pixel: [pixel.x, pixel.y],
                    camera: &camera,
                    camera_from_primary: *extrinsic,
                })
            })
            .collect();
        assert!(observations.iter().any(|o| o.camera_from_primary.is_some()));
        let start = Pose3d::new(yaw(8.0), Vec3F64::new(0.25, -0.05, 0.2));
        let (pose, inliers) = solve_rig_pnp(&observations, &start, &PnpConfig::default()).unwrap();
        assert_eq!(inliers, observations.len());
        assert!((pose.translation - truth.translation).length() < 1e-6);
        let dr = SO3F64::from_matrix(&(pose.rotation * truth.rotation.transpose())).log();
        assert!(dr.length() < 1e-8);
    }
}
