//! PnP pose estimation: solve 3D-2D correspondences with LM refinement.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{LMRefineParams, refine_pose_lm};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};

const ORB_SCALE_FACTOR: f64 = 1.2;
const ORB_MONO_CHI2: f64 = 5.991;
const ORB_POSE_OPT_PASSES: usize = 4;
const ORB_POSE_OPT_ITERATIONS: usize = 10;

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
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            prior_reproj_threshold_px: 25.0,
            final_reproj_threshold_px: 5.0,
            min_correspondences: 4,
            min_inliers_early: 10,
            min_inliers: 30,
        }
    }
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
    solve_pnp_with_octaves(
        points_world_f64,
        points_image,
        None,
        camera,
        pose_init,
        config,
    )
}

/// Solve PnP with ORB-SLAM3-style octave-weighted inlier classification.
pub fn solve_pnp_with_octaves(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    keypoint_octaves: Option<&[usize]>,
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> Option<(Pose3d, usize)> {
    if points_world_f64.len() < config.min_correspondences {
        return None;
    }
    if let Some(octaves) = keypoint_octaves
        && octaves.len() != points_world_f64.len()
    {
        return None;
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
    if prior_inlier_indices.len() < config.min_correspondences {
        return None;
    }

    let initial_inliers = count_reprojection_inliers_for_indices(
        pose_init,
        points_world_f64,
        points_image,
        keypoint_octaves,
        camera,
        config.final_reproj_threshold_px,
        &prior_inlier_indices,
    );

    let mut active_indices = prior_inlier_indices.clone();
    let mut current_pose = *pose_init;
    let mut best_pose = *pose_init;
    let mut best_inliers = initial_inliers;

    // ORB-SLAM3 PoseOptimization runs four optimize/classify passes. Outliers
    // are excluded from the next pass, but can be reclassified as inliers after
    // each pose update.
    for _ in 0..ORB_POSE_OPT_PASSES {
        if active_indices.len() < config.min_correspondences {
            break;
        }

        let pose_new = refine_pose_for_indices(
            points_world_f64,
            points_image,
            camera,
            &current_pose,
            &active_indices,
            ORB_POSE_OPT_ITERATIONS,
        )?;

        let pass_inlier_indices = reprojection_inlier_indices(
            &pose_new,
            points_world_f64,
            points_image,
            keypoint_octaves,
            camera,
            config.final_reproj_threshold_px,
            &prior_inlier_indices,
        );

        if pass_inlier_indices.len() >= best_inliers {
            best_inliers = pass_inlier_indices.len();
            best_pose = pose_new;
        }

        active_indices = pass_inlier_indices;
        current_pose = pose_new;
    }

    Some((best_pose, best_inliers))
}

fn refine_pose_for_indices(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    indices: &[usize],
    max_iterations: usize,
) -> Option<Pose3d> {
    if indices.len() < 3 {
        return None;
    }

    let k = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let mut world_inliers = Vec::with_capacity(indices.len());
    let mut image_inliers = Vec::with_capacity(indices.len());
    for &i in indices {
        let pw = &points_world_f64[i];
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
    let t_init_f32 = Vec3AF32::new(
        pose_init.translation.x as f32,
        pose_init.translation.y as f32,
        pose_init.translation.z as f32,
    );

    let lm = refine_pose_lm(
        &world_inliers,
        &image_inliers,
        &k,
        &r_init_f32,
        &t_init_f32,
        None,
        &LMRefineParams::default().with_max_iterations(max_iterations),
    )
    .ok()?;

    let r = &lm.rotation;
    let t = lm.translation;
    let r_new = Mat3F64::from_cols(
        Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
        Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
        Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
    );
    let t_new = Vec3F64::new(t.x as f64, t.y as f64, t.z as f64);
    Some(Pose3d::new(r_new, t_new))
}

/// Count map-point reprojection inliers given a camera pose.
pub fn count_reprojection_inliers(
    pose_world_to_cam: &Pose3d,
    points_world: &[Vec3F64],
    points_image: &[Vec2F32],
    keypoint_octaves: Option<&[usize]>,
    camera: &PinholeCamera,
    threshold_px: f64,
) -> usize {
    let all_indices: Vec<_> = (0..points_world.len()).collect();
    count_reprojection_inliers_for_indices(
        pose_world_to_cam,
        points_world,
        points_image,
        keypoint_octaves,
        camera,
        threshold_px,
        &all_indices,
    )
}

fn count_reprojection_inliers_for_indices(
    pose_world_to_cam: &Pose3d,
    points_world: &[Vec3F64],
    points_image: &[Vec2F32],
    keypoint_octaves: Option<&[usize]>,
    camera: &PinholeCamera,
    threshold_px: f64,
    indices: &[usize],
) -> usize {
    reprojection_inlier_indices(
        pose_world_to_cam,
        points_world,
        points_image,
        keypoint_octaves,
        camera,
        threshold_px,
        indices,
    )
    .len()
}

fn reprojection_inlier_indices(
    pose_world_to_cam: &Pose3d,
    points_world: &[Vec3F64],
    points_image: &[Vec2F32],
    keypoint_octaves: Option<&[usize]>,
    camera: &PinholeCamera,
    threshold_px: f64,
    indices: &[usize],
) -> Vec<usize> {
    let mut inliers = Vec::new();
    for &idx in indices {
        let Some(pw) = points_world.get(idx) else {
            continue;
        };
        let Some(pi) = points_image.get(idx) else {
            continue;
        };
        let th2 = reprojection_threshold_sq(keypoint_octaves, idx, threshold_px);
        if camera
            .reprojection_error_sq_world(pose_world_to_cam, pw, pi.x as f64, pi.y as f64)
            .is_some_and(|err_sq| err_sq <= th2)
        {
            inliers.push(idx);
        }
    }
    inliers
}

fn reprojection_threshold_sq(
    keypoint_octaves: Option<&[usize]>,
    point_idx: usize,
    fallback_threshold_px: f64,
) -> f64 {
    keypoint_octaves
        .and_then(|octaves| octaves.get(point_idx))
        .map(|&octave| ORB_MONO_CHI2 * ORB_SCALE_FACTOR.powi((2 * octave) as i32))
        .unwrap_or(fallback_threshold_px * fallback_threshold_px)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn octave_threshold_scales_like_orb_slam3() {
        let th0 = reprojection_threshold_sq(Some(&[0]), 0, 5.0);
        let th2 = reprojection_threshold_sq(Some(&[2]), 0, 5.0);

        assert!((th0 - ORB_MONO_CHI2).abs() < 1e-12);
        assert!((th2 - ORB_MONO_CHI2 * ORB_SCALE_FACTOR.powi(4)).abs() < 1e-12);
    }

    #[test]
    fn solve_pnp_with_octaves_rejects_consistent_pixel_outliers() {
        let camera = camera();
        let pose = Pose3d::IDENTITY;
        let mut points_world = Vec::new();
        let mut points_image = Vec::new();
        let mut octaves = Vec::new();

        for y in -2..=2 {
            for x in -2..=2 {
                let point = Vec3F64::new(x as f64 * 0.1, y as f64 * 0.1, 5.0);
                let pixel = camera.project_to_pixel(&point, 1e-8).unwrap();
                points_world.push(point);
                points_image.push(Vec2F32::new(pixel.x as f32, pixel.y as f32));
                octaves.push(0);
            }
        }

        // These pass the loose prior gate but fail the ORB mono chi-square gate.
        for i in 0..8 {
            let point = Vec3F64::new((i as f64 - 4.0) * 0.1, 0.35, 5.0);
            let pixel = camera.project_to_pixel(&point, 1e-8).unwrap();
            points_world.push(point);
            points_image.push(Vec2F32::new((pixel.x + 12.0) as f32, pixel.y as f32));
            octaves.push(0);
        }

        let config = PnpConfig {
            prior_reproj_threshold_px: 25.0,
            min_correspondences: 4,
            ..PnpConfig::default()
        };

        let (_pose, inliers) = solve_pnp_with_octaves(
            &points_world,
            &points_image,
            Some(&octaves),
            &camera,
            &pose,
            &config,
        )
        .expect("synthetic PnP should solve");

        assert_eq!(inliers, 25);
    }
}
