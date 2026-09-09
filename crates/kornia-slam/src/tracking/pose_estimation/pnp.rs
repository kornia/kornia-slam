//! PnP pose estimation: solve 3D-2D correspondences with LM refinement.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{LMRefineParams, refine_pose_lm};
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};

/// PnP pose-estimation thresholds.
#[derive(Debug, Clone)]
pub struct PnpConfig {
    /// Reprojection threshold (px) for filtering correspondences against the prior pose.
    pub prior_reproj_threshold_px: f64,
    /// Reprojection threshold (px) for counting final inliers after LM refinement.
    pub final_reproj_threshold_px: f64,
    /// Minimum number of 3D-2D correspondences required for PnP solving.
    pub min_correspondences: usize,
    /// Minimum inliers for early acceptance before local map refinement.
    pub min_inliers_early: usize,
    /// Minimum inliers to accept a PnP solution.
    pub min_inliers: usize,
    /// M-estimator kernel applied per residual during LM refinement.
    pub robust: RobustKernelKind,
    /// Squared robust-loss scale passed to the LM solver.
    pub robust_scale_sq: f32,
    /// Number of hard-exclusion refit rounds.
    pub outlier_rounds: usize,
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            prior_reproj_threshold_px: 25.0,
            final_reproj_threshold_px: 3.0,
            min_correspondences: 4,
            min_inliers_early: 10,
            min_inliers: 30,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 25.0,
            outlier_rounds: 4,
        }
    }
}

/// Diagnostic counts from a PnP attempt.
#[derive(Debug, Clone, Copy, Default)]
pub struct PnpDiagnostics {
    /// Correspondences passed in before filtering.
    pub input: usize,
    /// Correspondences surviving the prior-pose reprojection filter.
    pub prior_survivors: usize,
    /// Correspondences entering the last refinement round that ran.
    pub last_round_active: usize,
    /// Whether the final LM round reported convergence.
    pub converged: Option<bool>,
}

/// Solve PnP from 3D world points and 2D image points with LM refinement.
///
/// Filters correspondences by reprojection error against `pose_init`, runs LM,
/// then counts final inliers. Returns the refined pose and inlier count.
pub fn solve_pnp(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> Option<(Pose3d, usize)> {
    solve_pnp_with_diagnostics(points_world_f64, points_image, camera, pose_init, config).0
}

/// Solves PnP and returns intermediate correspondence counts.
pub fn solve_pnp_with_diagnostics(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> (Option<(Pose3d, usize)>, PnpDiagnostics) {
    let mut diagnostics = PnpDiagnostics {
        input: points_world_f64.len(),
        ..PnpDiagnostics::default()
    };
    if points_world_f64.len() < config.min_correspondences {
        return (None, diagnostics);
    }

    // Filter by reprojection error against the prior pose.
    let prior_th2 = config.prior_reproj_threshold_px * config.prior_reproj_threshold_px;
    let mut prior_inlier_indices = Vec::new();
    for (i, pw) in points_world_f64.iter().enumerate() {
        if camera
            .reprojection_error_sq_world(
                pose_init,
                pw,
                points_image[i].x as f64,
                points_image[i].y as f64,
            )
            .is_some_and(|err_sq| err_sq <= prior_th2)
        {
            prior_inlier_indices.push(i);
        }
    }
    diagnostics.prior_survivors = prior_inlier_indices.len();
    if prior_inlier_indices.len() < config.min_correspondences {
        return (None, diagnostics);
    }

    // Normalize for the f32 LM solver: express points relative to the prior
    // camera center and scale to unit median depth. Reprojection is invariant
    // to this similarity transform, but the f32 normal equations are not —
    // once monocular scale drift grows the map (world coords and camera
    // translation in the tens), the unnormalized solve goes singular
    // ("LU solve failed") on perfectly well-posed problems. With
    // `center = -R^T t` the normalized prior translation is exactly zero.
    let center = pose_init.inverse().translation;
    let mut depths: Vec<f64> = prior_inlier_indices
        .iter()
        .map(|&i| pose_init.transform_point(&points_world_f64[i]).z)
        .collect();
    let mid = depths.len() / 2;
    depths.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let scale = 1.0 / depths[mid].max(1e-9);

    // Build f32 arrays for LM solver.
    let k = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let mut world_inliers = Vec::with_capacity(prior_inlier_indices.len());
    let mut image_inliers = Vec::with_capacity(prior_inlier_indices.len());
    for &i in &prior_inlier_indices {
        let pw = (points_world_f64[i] - center) * scale;
        world_inliers.push(Vec3AF32::new(pw.x as f32, pw.y as f32, pw.z as f32));
        image_inliers.push(points_image[i]);
    }

    let r_init_f32 = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose_init.rotation.col(0).x as f32,
            pose_init.rotation.col(0).y as f32,
            pose_init.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose_init.rotation.col(1).x as f32,
            pose_init.rotation.col(1).y as f32,
            pose_init.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose_init.rotation.col(2).x as f32,
            pose_init.rotation.col(2).y as f32,
            pose_init.rotation.col(2).z as f32,
        ),
    );
    // Normalized prior translation: t' = s * (t + R * center) = 0.
    let t_init_f32 = Vec3AF32::new(0.0, 0.0, 0.0);

    // Undo the similarity normalization: the LM pose maps
    // s*(w - center) -> cam, so the world-frame pose is
    // (R_lm, t_lm / s - R_lm * center).
    let unnormalize = |rotation: &Mat3AF32, translation: Vec3AF32| -> Pose3d {
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(
                rotation.col(0).x as f64,
                rotation.col(0).y as f64,
                rotation.col(0).z as f64,
            ),
            Vec3F64::new(
                rotation.col(1).x as f64,
                rotation.col(1).y as f64,
                rotation.col(1).z as f64,
            ),
            Vec3F64::new(
                rotation.col(2).x as f64,
                rotation.col(2).y as f64,
                rotation.col(2).z as f64,
            ),
        );
        let translation = Vec3F64::new(
            translation.x as f64,
            translation.y as f64,
            translation.z as f64,
        );
        Pose3d::new(rotation, translation / scale - rotation * center)
    };

    // Every refinement round starts from the same prior. Between rounds,
    // observations outside the final reprojection threshold are permanently
    // removed so a bad intermediate solution cannot compound into the next.
    let final_threshold_sq = config.final_reproj_threshold_px * config.final_reproj_threshold_px;
    let rounds = config.outlier_rounds.max(1);
    let mut active: Vec<usize> = (0..world_inliers.len()).collect();
    let mut last_lm = None;
    for round in 0..rounds {
        if active.len() < config.min_correspondences {
            return (None, diagnostics);
        }
        diagnostics.last_round_active = active.len();

        let active_world: Vec<Vec3AF32> = active.iter().map(|&i| world_inliers[i]).collect();
        let active_image: Vec<Vec2F32> = active.iter().map(|&i| image_inliers[i]).collect();
        let Ok(lm) = refine_pose_lm(
            &active_world,
            &active_image,
            &k,
            &r_init_f32,
            &t_init_f32,
            None,
            &LMRefineParams {
                robust: config.robust,
                robust_scale_sq: config.robust_scale_sq,
                ..LMRefineParams::default()
            },
        ) else {
            return (None, diagnostics);
        };

        if round + 1 < rounds {
            let pose = unnormalize(&lm.rotation, lm.translation);
            active.retain(|&i| {
                let original_index = prior_inlier_indices[i];
                camera
                    .reprojection_error_sq_world(
                        &pose,
                        &points_world_f64[original_index],
                        points_image[original_index].x as f64,
                        points_image[original_index].y as f64,
                    )
                    .is_some_and(|error_sq| error_sq <= final_threshold_sq)
            });
        }
        last_lm = Some(lm);
    }
    let Some(lm) = last_lm else {
        return (None, diagnostics);
    };

    // The upstream convergence flag is diagnostic only. Real-data solves can
    // reach the iteration cap while still producing a valid refined pose.
    diagnostics.converged = lm.converged;
    let pose_new = unnormalize(&lm.rotation, lm.translation);

    let final_inliers = count_reprojection_inliers(
        &pose_new,
        points_world_f64,
        points_image,
        camera,
        config.final_reproj_threshold_px,
    );

    (Some((pose_new, final_inliers)), diagnostics)
}

/// Count map-point reprojection inliers given a camera pose.
pub fn count_reprojection_inliers(
    pose_world_to_cam: &Pose3d,
    points_world: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    threshold_px: f64,
) -> usize {
    let th2 = threshold_px * threshold_px;
    let mut inliers = 0usize;
    for (pw, pi) in points_world.iter().zip(points_image.iter()) {
        if camera
            .reprojection_error_sq_world(pose_world_to_cam, pw, pi.x as f64, pi.y as f64)
            .is_some_and(|err_sq| err_sq <= th2)
        {
            inliers += 1;
        }
    }
    inliers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 300.0,
            fy: 300.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn synthetic_correspondences(
        camera: &PinholeCamera,
        pose: &Pose3d,
        count: usize,
    ) -> (Vec<Vec3F64>, Vec<Vec2F32>) {
        let points_world: Vec<Vec3F64> = (0..count)
            .map(|i| {
                let x = (i % 5) as f64 * 0.35 - 0.7;
                let y = (i / 5) as f64 * 0.28 - 0.7;
                let z = 4.0 + (i % 4) as f64 * 0.4;
                Vec3F64::new(x, y, z)
            })
            .collect();
        let points_image = points_world
            .iter()
            .map(|point| {
                let point_camera = pose.transform_point(point);
                let pixel = camera.project_to_pixel(&point_camera, 0.0).unwrap();
                Vec2F32::new(pixel.x as f32, pixel.y as f32)
            })
            .collect();
        (points_world, points_image)
    }

    #[test]
    fn robust_defaults_match_tracking_configuration() {
        let config = PnpConfig::default();

        assert_eq!(config.robust, RobustKernelKind::Huber);
        assert_eq!(config.robust_scale_sq, 25.0);
        assert_eq!(config.outlier_rounds, 4);
    }

    #[test]
    fn diagnostics_report_insufficient_input() {
        let camera = test_camera();
        let (points_world, points_image) = synthetic_correspondences(&camera, &Pose3d::IDENTITY, 3);

        let (result, diagnostics) = solve_pnp_with_diagnostics(
            &points_world,
            &points_image,
            &camera,
            &Pose3d::IDENTITY,
            &PnpConfig::default(),
        );

        assert!(result.is_none());
        assert_eq!(diagnostics.input, 3);
        assert_eq!(diagnostics.prior_survivors, 0);
        assert_eq!(diagnostics.last_round_active, 0);
    }

    #[test]
    fn diagnostics_report_prior_gate_starvation() {
        let camera = test_camera();
        let (points_world, mut points_image) =
            synthetic_correspondences(&camera, &Pose3d::IDENTITY, 8);
        for point in &mut points_image {
            point.x += 100.0;
        }
        let config = PnpConfig {
            prior_reproj_threshold_px: 1.0,
            ..PnpConfig::default()
        };

        let (result, diagnostics) = solve_pnp_with_diagnostics(
            &points_world,
            &points_image,
            &camera,
            &Pose3d::IDENTITY,
            &config,
        );

        assert!(result.is_none());
        assert_eq!(diagnostics.prior_survivors, 0);
        assert_eq!(diagnostics.last_round_active, 0);
    }

    #[test]
    fn solves_clean_synthetic_correspondences() {
        let camera = test_camera();
        let expected_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.1, -0.05, 0.15));
        let (points_world, points_image) = synthetic_correspondences(&camera, &expected_pose, 24);
        let config = PnpConfig {
            prior_reproj_threshold_px: 50.0,
            final_reproj_threshold_px: 1.0,
            ..PnpConfig::default()
        };

        let (pose, inliers) = solve_pnp(
            &points_world,
            &points_image,
            &camera,
            &Pose3d::IDENTITY,
            &config,
        )
        .expect("clean correspondences should solve");

        assert_eq!(inliers, points_world.len());
        assert!((pose.translation - expected_pose.translation).length() < 1e-3);
    }

    #[test]
    fn exclusion_rounds_remove_reprojection_outliers() {
        let camera = test_camera();
        let expected_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.1, -0.05, 0.15));
        let (points_world, mut points_image) =
            synthetic_correspondences(&camera, &expected_pose, 30);
        for point in points_image.iter_mut().step_by(6) {
            point.x += 30.0;
            point.y -= 20.0;
        }
        let config = PnpConfig {
            prior_reproj_threshold_px: 100.0,
            final_reproj_threshold_px: 2.0,
            ..PnpConfig::default()
        };

        let (result, diagnostics) = solve_pnp_with_diagnostics(
            &points_world,
            &points_image,
            &camera,
            &Pose3d::IDENTITY,
            &config,
        );
        let (_, inliers) = result.expect("robust solve should retain the inlier set");

        assert_eq!(diagnostics.prior_survivors, points_world.len());
        assert!(diagnostics.last_round_active < diagnostics.prior_survivors);
        assert!(inliers >= 25);
    }
}
