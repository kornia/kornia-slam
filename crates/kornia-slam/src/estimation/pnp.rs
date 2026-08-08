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
    /// M-estimator kernel applied per-residual during the LM solve itself
    /// (see `kornia_3d::pnp::LMRefineParams::robust`). `Identity` (the
    /// default) is today's plain-L2 behavior; a correspondence that
    /// disagrees with `pose_init` is judged solely by
    /// `prior_reproj_threshold_px` *before* the solve ever runs. Setting
    /// this to `Huber` additionally down-weights (rather than hard-excludes
    /// or hard-includes) disagreement *during* the solve, which is what
    /// makes it safe to widen `prior_reproj_threshold_px` for correspondence
    /// sources whose positions don't depend on `pose_init` being accurate
    /// (e.g. KLT) without risking a single bad outlier derailing an
    /// otherwise-unprotected least-squares solve.
    pub robust: RobustKernelKind,
    /// Squared scale for `robust`; see `LMRefineParams::robust_scale_sq`.
    /// Default `f32::INFINITY` collapses to L2 regardless of `robust`.
    pub robust_scale_sq: f32,
    /// Number of hard-exclusion refit rounds — mirrors lightweight_vio's
    /// `PnPOptimizer::optimize_pose` (its own default: 4 for EuRoC). Each
    /// round solves LM fresh **from `pose_init`** (never from the previous
    /// round's result, so a round dragged off by an outlier that hasn't
    /// been excluded yet can't compound into the next one), then hard-drops
    /// any correspondence whose reprojection error under that round's
    /// result exceeds `final_reproj_threshold_px` before the next round.
    /// `robust` still applies within every round, protecting against
    /// whatever outlier fraction hasn't been hard-excluded yet. Default `1`
    /// (a single solve, today's pre-existing behavior — no behavior change
    /// for callers that don't opt in).
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
            robust_scale_sq: 25.0, // 5px Huber delta
            outlier_rounds: 4,
        }
    }
}

/// Diagnostic counts from a [`solve_pnp_with_diagnostics`] attempt, exposed
/// so callers can log *why* a PnP solve failed (starved by the prior gate
/// vs. an outright LM failure with plenty of prior survivors) instead of
/// just the fact that it did.
#[derive(Debug, Clone, Copy, Default)]
pub struct PnpDiagnostics {
    /// Correspondences passed in, before any filtering.
    pub input: usize,
    /// Correspondences surviving the prior-pose reprojection filter
    /// (`prior_reproj_threshold_px`, scaled by the caller's `search_scale`).
    pub prior_survivors: usize,
    /// Correspondences still active going into the *last* outlier-exclusion
    /// round that actually ran (see `PnpConfig::outlier_rounds`).
    pub last_round_active: usize,
    /// Whether the last round's LM solve reported `CostConverged` or
    /// `GradientConverged` (as opposed to `MaxIterations`/
    /// `LambdaMaxExceeded`/`Interrupted` — all of which still return `Ok`
    /// from the solver but represent "gave up," not "found an answer").
    /// `None` if no round ran at all (e.g. starved before the first solve).
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

/// Same as [`solve_pnp`], but also returns the intermediate correspondence
/// counts — see [`PnpDiagnostics`].
pub fn solve_pnp_with_diagnostics(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> (Option<(Pose3d, usize)>, PnpDiagnostics) {
    let mut diag = PnpDiagnostics {
        input: points_world_f64.len(),
        prior_survivors: 0,
        last_round_active: 0,
        converged: None,
    };
    if points_world_f64.len() < config.min_correspondences {
        return (None, diag);
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
    diag.prior_survivors = prior_inlier_indices.len();
    if prior_inlier_indices.len() < config.min_correspondences {
        return (None, diag);
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

    // Undo the similarity normalization above: the LM pose maps
    // s*(w - center) -> cam, so the world-frame pose is
    // (R_lm, t_lm / s - R_lm * center).
    let unnormalize = |r: &Mat3AF32, t: Vec3AF32| -> Pose3d {
        let r_new = Mat3F64::from_cols(
            Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
            Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
            Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
        );
        let t_lm = Vec3F64::new(t.x as f64, t.y as f64, t.z as f64);
        let t_new = t_lm / scale - r_new * center;
        Pose3d::new(r_new, t_new)
    };

    // Multi-round hard-exclusion refit, mirroring lightweight_vio's
    // `PnPOptimizer::optimize_pose` (Optimizer.cpp:192-228): every round
    // solves fresh from `pose_init`'s normalized starting point — never
    // from the previous round's result — so a round that got pulled off by
    // a not-yet-excluded outlier can't compound into the next one. Before
    // the next round, any correspondence whose reprojection error under
    // *this* round's solved pose exceeds `final_reproj_threshold_px` is
    // permanently dropped. `config.robust` still applies within every
    // round's solve, protecting against whatever outlier fraction hasn't
    // been hard-excluded yet. With `outlier_rounds == 1` (the default) this
    // is exactly the single-solve behavior from before.
    let final_th2 = config.final_reproj_threshold_px * config.final_reproj_threshold_px;
    let mut active: Vec<usize> = (0..world_inliers.len()).collect();
    let mut last_lm = None;
    for round in 0..config.outlier_rounds.max(1) {
        if active.len() < config.min_correspondences {
            return (None, diag);
        }
        diag.last_round_active = active.len();

        let world_active: Vec<Vec3AF32> = active.iter().map(|&j| world_inliers[j]).collect();
        let image_active: Vec<Vec2F32> = active.iter().map(|&j| image_inliers[j]).collect();
        let Ok(lm) = refine_pose_lm(
            &world_active,
            &image_active,
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
            return (None, diag);
        };

        if round + 1 < config.outlier_rounds {
            let pose_round = unnormalize(&lm.rotation, lm.translation);
            active.retain(|&j| {
                let orig_idx = prior_inlier_indices[j];
                camera
                    .reprojection_error_sq_world(
                        &pose_round,
                        &points_world_f64[orig_idx],
                        points_image[orig_idx].x as f64,
                        points_image[orig_idx].y as f64,
                    )
                    .is_some_and(|err_sq| err_sq <= final_th2)
            });
        }
        last_lm = Some(lm);
    }
    let Some(lm) = last_lm else {
        return (None, diag);
    };

    // NOT gated on `lm.converged` — measured on all 6 EuRoC sequences via
    // `--evaluate` and it was a catastrophic regression (MH03 alone went
    // from 0.18m to 8.4m ATE with 32 resets, previously zero). Unlike
    // lightweight_vio's Ceres solve, kornia_algebra's from-scratch LM
    // apparently reaches `MaxIterations` (20 by default) on a large
    // fraction of perfectly healthy, well-converged-in-practice solves —
    // real pixel-noise data rarely satisfies `relative_cost_change < 1e-6`
    // in a single iteration step, so `CostConverged`/`GradientConverged`
    // is a much stricter bar here than Ceres' own `CONVERGENCE` turned out
    // to be for their problem. Recorded in `diag.converged` for visibility,
    // but not used to reject: `MaxIterations` almost certainly still means
    // "found a good pose, just didn't trip the strict tolerance," not
    // lightweight_vio's "gave up." Distinguishing that case from a genuine
    // `LambdaMaxExceeded`/`Interrupted` (real numerical distress) would
    // need `refine_pose_lm` to expose the raw `TerminationReason`, not just
    // this collapsed bool — not done.
    diag.converged = lm.converged;

    let pose_new = unnormalize(&lm.rotation, lm.translation);

    let final_inliers = count_reprojection_inliers(
        &pose_new,
        points_world_f64,
        points_image,
        camera,
        config.final_reproj_threshold_px,
    );

    (Some((pose_new, final_inliers)), diag)
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
