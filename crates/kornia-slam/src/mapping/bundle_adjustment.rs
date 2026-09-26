//! BA window selection, observation assembly, and numerical solving.
//!
//! Solvers consume immutable map snapshots and return numerical updates. The
//! map owns capture and writeback; the local-mapping worker owns scheduling.

use crate::mapping::map::{
    BaSnapshot, BaUpdate, BaUpdateError, Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR,
};
use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::{SchurBaError, bundle_adjust_schur};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::PreintegratedImu;
use std::collections::{HashMap, HashSet};

/// Relative standard deviation of a stereo depth measurement, as a fraction of
/// the measured depth (used to weight the BA depth residual). Depth-proportional
/// so far points—where disparity is least reliable—are downweighted.
pub const STEREO_DEPTH_REL_SIGMA: f32 = 0.05;
/// Floor on the stereo depth sigma (metres) to avoid over-trusting very near
/// points.
pub const STEREO_DEPTH_MIN_SIGMA: f32 = 0.02;

/// Keyframes a local BA optimizes; older keyframes enter only as fixed poses.
const MAX_ACTIVE_KFS: usize = 3;

/// Whether [`run_initial_ba`] refined the map or had too little to work with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialBaOutcome {
    Refined,
    /// Fewer than two keyframes, no live landmarks, or too few observations.
    Skipped,
}

/// Why [`run_initial_ba`] left the map unrefined.
#[derive(Debug, thiserror::Error)]
pub enum InitialBaError {
    #[error("bundle adjustment failed: {0}")]
    Solver(#[from] SchurBaError),
    #[error("writeback refused: {0}")]
    Writeback(#[from] BaUpdateError),
}

/// Run a 2-keyframe bundle adjustment over the bootstrap pair.
///
/// Operates on the two most recently inserted keyframes (the bootstrap
/// pair). Optimizes the newer KF's pose and all map points observed by
/// either KF; the older KF is held fixed as the gauge anchor. Mirrors
/// ORB-SLAM3's `GlobalBundleAdjustemnt(map, 20)` in
/// `CreateInitialMapMonocular`.
///
/// The map is left untouched unless the outcome is [`InitialBaOutcome::Refined`].
pub fn run_initial_ba(
    map: &mut Map,
    camera: &PinholeCamera,
) -> Result<InitialBaOutcome, InitialBaError> {
    let mut update = map.ba_snapshot().into_update();
    let snapshot = &update.snapshot;
    const MAX_ITERS: usize = 5;
    const HUBER_SCALE_SQ: f32 = 5.991;
    const MIN_OBSERVATIONS: usize = 8;

    let n = snapshot.keyframes().len();
    if n < 2 {
        return Ok(InitialBaOutcome::Skipped);
    }

    // Last two keyframes; older is gauge-fixed.
    let kf_indices = [n - 2, n - 1];

    let window = WindowPoints::collect(
        snapshot,
        kf_indices.iter().map(|&i| &snapshot.keyframes()[i]),
    );
    if window.is_empty() {
        return Ok(InitialBaOutcome::Skipped);
    }

    let poses: Vec<Pose3d> = kf_indices
        .iter()
        .map(|&i| snapshot.keyframes()[i].frame.pose_world_to_cam)
        .collect();

    let observations = window.observations(
        kf_indices
            .iter()
            .enumerate()
            .map(|(pose_idx, &kf_idx)| (pose_idx, &snapshot.keyframes()[kf_idx], pose_idx == 0)),
        camera,
    );

    if observations.len() < MIN_OBSERVATIONS {
        return Ok(InitialBaOutcome::Skipped);
    }

    let (sq_err_before, depth_before, kf1_t_before) =
        initial_ba_diagnostics(&poses, &window.positions, &observations, camera);

    let params = BaParams {
        max_iterations: MAX_ITERS,
        // Two-view monocular BA has a 1-DOF scale gauge (only KF0 is
        // fixed). Bump LM damping so the augmented normal equations stay
        // well-conditioned even though H is rank-deficient.
        initial_lambda: 1.0,
        robust: RobustKernelKind::Huber,
        robust_scale_sq: HUBER_SCALE_SQ,
        ..BaParams::default()
    };

    let ba_result = bundle_adjust_schur(&poses, &window.positions, &observations, camera, &params)?;

    let (sq_err_after, depth_after, kf1_t_after) =
        initial_ba_diagnostics(&ba_result.poses, &ba_result.points, &observations, camera);

    eprintln!(
        "[init_ba] before: reproj_rms={:.3}px median_depth={:.3} kf1_t_norm={:.3} obs={}",
        sq_err_before.sqrt(),
        depth_before,
        kf1_t_before,
        observations.len()
    );
    eprintln!(
        "[init_ba] after:  reproj_rms={:.3}px median_depth={:.3} kf1_t_norm={:.3} iters={} converged={}",
        sq_err_after.sqrt(),
        depth_after,
        kf1_t_after,
        ba_result.iterations,
        ba_result.converged
    );

    update.keyframes[kf_indices[1]].pose_world_to_cam = ba_result.poses[1];
    window.write_back(&ba_result.points, &mut update.map_points);
    map.apply_ba_update(update)?;

    // Diagnostics: sample one point's scale state for a sanity check.
    if let Some(&sample) = window.global.first()
        && let Some(mp) = map.map_points().get(sample)
    {
        let normal_len = mp.mean_viewing_direction.length();
        let predicted = mp.predict_scale(depth_after.max(1e-6), ORB_SCALE_FACTOR, ORB_N_LEVELS);
        eprintln!(
            "[init_ba] scale_state[mp{sample}]: min_dist={:.3} max_dist={:.3} normal_len={:.3} predict_scale@median={predicted}",
            mp.min_distance, mp.max_distance, normal_len,
        );
    }

    Ok(InitialBaOutcome::Refined)
}

/// Run local bundle adjustment over recent keyframes and their observed map points.
///
/// Collects the last N active keyframes, gathers observations (undistorting keypoints
/// via camera), calls the Schur solver, and returns optimized poses and points.
/// An early exit returns an unchanged update for the worker to publish normally.
pub fn run_local_ba(snapshot: BaSnapshot, camera: &PinholeCamera) -> BaUpdate {
    let mut update = snapshot.into_update();
    let snapshot = &update.snapshot;
    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

    let window = WindowPoints::collect(snapshot, &snapshot.keyframes()[active_start..]);
    if window.is_empty() {
        return update;
    }

    let poses: Vec<Pose3d> = snapshot
        .keyframes()
        .iter()
        .map(|kf| kf.frame.pose_world_to_cam)
        .collect();

    let observations = window.observations(
        snapshot
            .keyframes()
            .iter()
            .enumerate()
            .map(|(kf_idx, kf)| (kf_idx, kf, kf_idx < active_start)),
        camera,
    );

    if observations.len() < MIN_OBSERVATIONS {
        return update;
    }

    let ba_result = match bundle_adjust_schur(
        &poses,
        &window.positions,
        &observations,
        camera,
        &BaParams::default(),
    ) {
        Ok(r) => r,
        Err(_) => return update,
    };

    for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
        if kf_idx >= active_start {
            update.keyframes[kf_idx].pose_world_to_cam = *pose;
        }
    }

    window.write_back(&ba_result.points, &mut update.map_points);
    update
}

/// Solves the same local window with velocity, bias and active IMU edges.
/// Repropagated preintegrations remain in the returned update even if solving fails.
pub fn run_local_inertial_ba(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    imu_t_bc: Option<Pose3d>,
    gravity_world: Vec3F64,
) -> BaUpdate {
    let mut update = snapshot.into_update();
    let snapshot = &update.snapshot;
    use crate::mapping::vi_ba_schur::{
        ImuFactor as ViBaImuFactor, ViBaKeyframe, ViBaParams, visual_inertial_bundle_adjust,
    };

    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

    let window = WindowPoints::collect(snapshot, &snapshot.keyframes()[active_start..]);
    if window.is_empty() {
        return update;
    }

    // Build VI-BA keyframes (all KFs; fixed flag controls which are optimised).
    let vi_keyframes: Vec<ViBaKeyframe> = snapshot
        .keyframes()
        .iter()
        .enumerate()
        .map(|(kf_idx, kf)| ViBaKeyframe {
            pose: kf.frame.pose_world_to_cam,
            velocity: kf.velocity_world,
            bias: kf.imu_bias,
            fixed: kf_idx < active_start,
        })
        .collect();

    let observations = window.observations(
        snapshot
            .keyframes()
            .iter()
            .enumerate()
            .map(|(kf_idx, kf)| (kf_idx, kf, kf_idx < active_start)),
        camera,
    );

    if observations.len() < MIN_OBSERVATIONS {
        return update;
    }

    // Map global frame.idx → local keyframe slot (0..n_kfs).
    let frame_idx_to_slot: HashMap<usize, usize> = snapshot
        .keyframes()
        .iter()
        .enumerate()
        .map(|(slot, kf)| (kf.frame.idx, slot))
        .collect();

    // Repropagate any active edge whose current from-keyframe bias has
    // drifted past the point where `delta_*_with_bias`'s first-order
    // correction stays valid. Without this, a sliding window that keeps
    // re-optimizing the same edge across many calls while bias is still
    // moving compounds a purely numerical linearization error into what
    // looks like more residual, which pushes bias further — a feedback
    // loop independent of whatever real motion originally nudged bias.
    const REPROPAGATE_BIAS_THRESHOLD: f64 = 0.02;
    for (factor, preintegrated) in snapshot
        .imu_factors()
        .iter()
        .zip(&mut update.imu_preintegrations)
    {
        let Some(&from) = frame_idx_to_slot.get(&factor.prev_kf_idx) else {
            continue;
        };
        let Some(&to) = frame_idx_to_slot.get(&factor.curr_kf_idx) else {
            continue;
        };
        if from < active_start && to < active_start {
            continue;
        }
        let current_bias = snapshot.keyframes()[from].imu_bias;
        let d_accel = (current_bias.accel - preintegrated.bias.accel).length();
        let d_gyro = (current_bias.gyro - preintegrated.bias.gyro).length();
        if d_accel > REPROPAGATE_BIAS_THRESHOLD || d_gyro > REPROPAGATE_BIAS_THRESHOLD {
            *preintegrated = PreintegratedImu::from_measurements(
                current_bias,
                preintegrated.calib,
                &factor.raw_samples,
                factor.t0,
                factor.t1,
            );
        }
    }

    // Build IMU edges; include only edges where at least one endpoint is active.
    let imu_edges: Vec<ViBaImuFactor> = snapshot
        .imu_factors()
        .iter()
        .zip(&update.imu_preintegrations)
        .filter_map(|(f, preintegrated)| {
            let from = *frame_idx_to_slot.get(&f.prev_kf_idx)?;
            let to = *frame_idx_to_slot.get(&f.curr_kf_idx)?;
            if from < active_start && to < active_start {
                return None;
            }
            Some(ViBaImuFactor {
                from_idx: from,
                to_idx: to,
                preintegrated: preintegrated.clone(),
            })
        })
        .collect();

    // 15-DOF-per-keyframe state (pose+velocity+bias) with information
    // entries spanning many more orders of magnitude than the pure
    // visual 6-DOF problem (see the Marquardt-damping note in
    // visual_inertial_bundle_adjust) converges more slowly to the same
    // strict cost_tolerance: over half of non-converged calls were
    // hitting the default max_iterations=20 cap while still making
    // small, steady progress (final residuals *smaller* than many calls
    // that did converge), not diverging. Give it more room.
    let vi_result = match visual_inertial_bundle_adjust(
        &vi_keyframes,
        &window.positions,
        &observations,
        &imu_edges,
        camera,
        &ViBaParams {
            imu_t_bc,
            gravity: gravity_world,
            max_iterations: 50,
            ..ViBaParams::default()
        },
    ) {
        Ok(r) => r,
        Err(_) => return update,
    };

    // Write back optimised poses, velocities, and biases for active keyframes.
    for kf_idx in active_start..n_kfs {
        let vi_kf = &vi_result.keyframes[kf_idx];
        update.keyframes[kf_idx].pose_world_to_cam = vi_kf.pose;
        update.keyframes[kf_idx].velocity_world = vi_kf.velocity;
        update.keyframes[kf_idx].imu_bias = vi_kf.bias;
    }

    window.write_back(&vi_result.points, &mut update.map_points);
    update
}

/// Live landmarks a BA window observes, ordered by ascending map index; a
/// landmark's position in `global` is its solver index.
struct WindowPoints {
    global: Vec<usize>,
    local_of: HashMap<usize, usize>,
    positions: Vec<Vec3F64>,
}

impl WindowPoints {
    fn collect<'a>(
        snapshot: &BaSnapshot,
        keyframes: impl IntoIterator<Item = &'a Keyframe>,
    ) -> Self {
        let mut unique: HashSet<usize> = HashSet::new();
        for kf in keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = snapshot.map_points().get(*mp_idx)
                    && !mp.culled
                {
                    unique.insert(*mp_idx);
                }
            }
        }
        let mut global: Vec<usize> = unique.into_iter().collect();
        global.sort_unstable();
        let local_of = global
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let positions = global
            .iter()
            .map(|&idx| snapshot.map_points()[idx].position)
            .collect();
        Self {
            global,
            local_of,
            positions,
        }
    }

    fn is_empty(&self) -> bool {
        self.global.is_empty()
    }

    /// Reprojection (and stereo depth) observations of the window's landmarks
    /// from each `(pose_idx, keyframe, fixed)` in order.
    fn observations<'a>(
        &self,
        poses: impl IntoIterator<Item = (usize, &'a Keyframe, bool)>,
        camera: &PinholeCamera,
    ) -> Vec<BaObservation> {
        let mut observations = Vec::new();
        for (pose_idx, kf, fixed_pose) in poses {
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(&point_idx) = mp_opt.and_then(|mp_idx| self.local_of.get(&mp_idx)) else {
                    continue;
                };
                if let Some(pixel) = kf.frame.undistorted_xy(desc_idx, camera) {
                    let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                    observations.push(BaObservation {
                        pose_idx,
                        point_idx,
                        pixel,
                        fixed_pose,
                        fixed_point: false,
                        depth_meas,
                        depth_sigma,
                    });
                }
            }
        }
        observations
    }

    fn write_back(&self, optimized: &[Vec3F64], map_points: &mut [Vec3F64]) {
        for (&global_idx, point) in self.global.iter().zip(optimized) {
            if let Some(mp) = map_points.get_mut(global_idx) {
                *mp = *point;
            }
        }
    }
}

/// Depth measurement + sigma for a BA observation at `desc_idx` of `kf`.
///
/// Returns `(Some(z), sigma)` when the keyframe's keypoint has a valid stereo
/// depth (anchoring the BA's metric scale), else `(None, 1.0)` for a pure
/// reprojection observation.
fn stereo_depth_obs(kf: &Keyframe, desc_idx: usize) -> (Option<f32>, f32) {
    match kf.frame.stereo_depth(desc_idx) {
        Some(z) => (
            Some(z),
            (STEREO_DEPTH_REL_SIGMA * z).max(STEREO_DEPTH_MIN_SIGMA),
        ),
        None => (None, 1.0),
    }
}

/// Returns (mean_sq_reproj_error, median_depth_in_kf0_frame, kf1_translation_norm).
fn initial_ba_diagnostics(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
) -> (f64, f64, f64) {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for obs in observations {
        if let (Some(pose), Some(pt)) = (poses.get(obs.pose_idx), points.get(obs.point_idx))
            && let Some(err_sq) = camera.reprojection_error_sq_world(
                pose,
                pt,
                obs.pixel[0] as f64,
                obs.pixel[1] as f64,
            )
        {
            sum_sq += err_sq;
            count += 1;
        }
    }
    let mean_sq = if count > 0 {
        sum_sq / count as f64
    } else {
        0.0
    };

    let median_depth = match poses.first() {
        Some(kf0) => {
            let mut depths: Vec<f64> = points
                .iter()
                .map(|p| kf0.transform_point(p).z)
                .filter(|&z| z > 0.0)
                .collect();
            if depths.is_empty() {
                0.0
            } else {
                let mid = depths.len() / 2;
                depths.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                depths[mid]
            }
        }
        None => 0.0,
    };

    let kf1_t_norm = poses.get(1).map(|p| p.translation.length()).unwrap_or(0.0);

    (mean_sq, median_depth, kf1_t_norm)
}

#[cfg(test)]
#[path = "bundle_adjustment_tests.rs"]
mod tests;
