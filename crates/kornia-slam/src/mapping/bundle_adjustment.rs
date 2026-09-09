//! BA window selection, observation assembly, and numerical solving.
//!
//! Solvers consume immutable map snapshots and return numerical updates. The
//! map owns capture and writeback; the local-mapping worker owns scheduling.

use crate::map::{BaSnapshot, BaUpdate, Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR};
use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::bundle_adjust_schur;
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

/// Run a 2-keyframe bundle adjustment over the bootstrap pair.
///
/// Operates on the two most recently inserted keyframes (the bootstrap
/// pair). Optimizes the newer KF's pose and all map points observed by
/// either KF; the older KF is held fixed as the gauge anchor. Mirrors
/// ORB-SLAM3's `GlobalBundleAdjustemnt(map, 20)` in
/// `CreateInitialMapMonocular`.
///
/// Returns `true` if BA ran and wrote back optimized values; `false` if
/// there were too few observations or the optimizer errored (map left
/// untouched in that case).
pub fn run_initial_ba(map: &mut Map, camera: &PinholeCamera) -> bool {
    let mut update = map.ba_snapshot().into_update();
    let snapshot = &update.snapshot;
    const MAX_ITERS: usize = 5;
    const HUBER_SCALE_SQ: f32 = 5.991;
    const MIN_OBSERVATIONS: usize = 8;

    let n = snapshot.keyframes().len();
    if n < 2 {
        return false;
    }

    // Last two keyframes; older is gauge-fixed.
    let kf_indices = [n - 2, n - 1];

    let mut mp_set: HashSet<usize> = HashSet::new();
    for &kf_idx in &kf_indices {
        for mp_idx in snapshot.keyframes()[kf_idx]
            .map_point_by_desc_idx
            .iter()
            .flatten()
        {
            if let Some(mp) = snapshot.map_points().get(*mp_idx)
                && !mp.culled
            {
                mp_set.insert(*mp_idx);
            }
        }
    }
    if mp_set.is_empty() {
        return false;
    }

    let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
    mp_global_indices.sort_unstable();

    let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    let points: Vec<Vec3F64> = mp_global_indices
        .iter()
        .map(|&idx| snapshot.map_points()[idx].position)
        .collect();

    let poses: Vec<Pose3d> = kf_indices
        .iter()
        .map(|&i| snapshot.keyframes()[i].frame.pose_world_to_cam)
        .collect();

    let mut observations = Vec::new();
    for (pose_idx, &kf_idx) in kf_indices.iter().enumerate() {
        let is_fixed = pose_idx == 0;
        let kf = &snapshot.keyframes()[kf_idx];
        for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
            if let Some(mp_idx) = mp_opt {
                let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                    continue;
                };
                if let Some(p) = kf.frame.undistorted_xy(desc_idx, camera) {
                    let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                    observations.push(BaObservation {
                        pose_idx,
                        point_idx,
                        pixel: p,
                        fixed_pose: is_fixed,
                        fixed_point: false,
                        depth_meas,
                        depth_sigma,
                    });
                }
            }
        }
    }

    if observations.len() < MIN_OBSERVATIONS {
        return false;
    }

    let (sq_err_before, depth_before, kf1_t_before) =
        initial_ba_diagnostics(&poses, &points, &observations, camera);

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

    let ba_result = match bundle_adjust_schur(&poses, &points, &observations, camera, &params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[init_ba] bundle_adjust failed: {e}");
            return false;
        }
    };

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
    for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
        if let Some(mp) = update.map_points.get_mut(global_idx) {
            *mp = ba_result.points[local_idx];
        }
    }
    // Positions (and KF1's pose) moved: refresh scale geometry.
    update.refresh_points = mp_global_indices.clone();
    if map.apply_ba_update(update).is_none() {
        return false;
    }

    // Diagnostics: sample one point's scale state for a sanity check.
    if let Some(&sample) = mp_global_indices.first()
        && let Some(mp) = map.map_points().get(sample)
    {
        let normal_len = mp.mean_viewing_direction.length();
        let predicted = mp.predict_scale(depth_after.max(1e-6), ORB_SCALE_FACTOR, ORB_N_LEVELS);
        eprintln!(
            "[init_ba] scale_state[mp{sample}]: min_dist={:.3} max_dist={:.3} normal_len={:.3} predict_scale@median={predicted}",
            mp.min_distance, mp.max_distance, normal_len,
        );
    }

    true
}

/// Run local bundle adjustment over recent keyframes and their observed map points.
///
/// Collects the last N active keyframes, gathers observations (undistorting keypoints
/// via camera), calls the Schur solver, and returns optimized poses and points.
/// An early exit returns an unchanged update for the worker to publish normally.
pub fn run_local_ba(snapshot: BaSnapshot, camera: &PinholeCamera) -> BaUpdate {
    let mut update = snapshot.into_update();
    let snapshot = &update.snapshot;
    const MAX_ACTIVE_KFS: usize = 3;
    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

    let mut mp_set: HashSet<usize> = HashSet::new();
    for kf in &snapshot.keyframes()[active_start..] {
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if let Some(mp) = snapshot.map_points().get(*mp_idx)
                && !mp.culled
            {
                mp_set.insert(*mp_idx);
            }
        }
    }
    if mp_set.is_empty() {
        return update;
    }

    let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
    mp_global_indices.sort_unstable();

    let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    let points: Vec<Vec3F64> = mp_global_indices
        .iter()
        .map(|&idx| snapshot.map_points()[idx].position)
        .collect();

    let poses: Vec<Pose3d> = snapshot
        .keyframes()
        .iter()
        .map(|kf| kf.frame.pose_world_to_cam)
        .collect();

    let mut observations = Vec::new();
    for (kf_idx, kf) in snapshot.keyframes().iter().enumerate() {
        let is_fixed = kf_idx < active_start;
        for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
            if let Some(mp_idx) = mp_opt {
                let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                    continue;
                };
                if let Some(p) = kf.frame.undistorted_xy(desc_idx, camera) {
                    let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                    observations.push(BaObservation {
                        pose_idx: kf_idx,
                        point_idx,
                        pixel: p,
                        fixed_pose: is_fixed,
                        fixed_point: false,
                        depth_meas,
                        depth_sigma,
                    });
                }
            }
        }
    }

    if observations.len() < MIN_OBSERVATIONS {
        return update;
    }

    let ba_result =
        match bundle_adjust_schur(&poses, &points, &observations, camera, &BaParams::default()) {
            Ok(r) => r,
            Err(_) => return update,
        };

    for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
        if kf_idx >= active_start {
            update.keyframes[kf_idx].pose_world_to_cam = *pose;
        }
    }

    for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
        if let Some(mp) = update.map_points.get_mut(global_idx) {
            *mp = ba_result.points[local_idx];
        }
    }
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
    use crate::vi_ba_schur::{
        ImuFactor as ViBaImuFactor, ViBaKeyframe, ViBaParams, visual_inertial_bundle_adjust,
    };

    const MAX_ACTIVE_KFS: usize = 3;
    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

    let mut mp_set: HashSet<usize> = HashSet::new();
    for kf in &snapshot.keyframes()[active_start..] {
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if let Some(mp) = snapshot.map_points().get(*mp_idx)
                && !mp.culled
            {
                mp_set.insert(*mp_idx);
            }
        }
    }
    if mp_set.is_empty() {
        return update;
    }

    let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
    mp_global_indices.sort_unstable();

    let mp_global_to_local: HashMap<usize, usize> = mp_global_indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    let points: Vec<Vec3F64> = mp_global_indices
        .iter()
        .map(|&idx| snapshot.map_points()[idx].position)
        .collect();

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

    let mut observations = Vec::new();
    for (kf_idx, kf) in snapshot.keyframes().iter().enumerate() {
        let is_fixed = kf_idx < active_start;
        for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
            if let Some(mp_idx) = mp_opt {
                let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                    continue;
                };
                if let Some(p) = kf.frame.undistorted_xy(desc_idx, camera) {
                    let (depth_meas, depth_sigma) = stereo_depth_obs(kf, desc_idx);
                    observations.push(BaObservation {
                        pose_idx: kf_idx,
                        point_idx,
                        pixel: p,
                        fixed_pose: is_fixed,
                        fixed_point: false,
                        depth_meas,
                        depth_sigma,
                    });
                }
            }
        }
    }

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
        &points,
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

    for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
        if let Some(mp) = update.map_points.get_mut(global_idx) {
            *mp = vi_result.points[local_idx];
        }
    }
    update
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
