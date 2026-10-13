//! BA window selection, observation assembly, and numerical solving.
//!
//! Solvers consume immutable map snapshots and return numerical updates. The
//! map owns capture and writeback; the local-mapping worker owns scheduling.

use crate::mapping::map::{
    BaSnapshot, BaUpdate, BaUpdateError, BaWindow, Keyframe, LocalBaStats, Map, ORB_N_LEVELS,
    ORB_SCALE_FACTOR,
};
use crate::mapping::stationary::StationaryFactor;
use crate::mapping::vi_ba_schur::ViBaKeyframe;
use crate::mapping::wheel_odom::WheelOdomFactor;
use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::{SchurBaError, bundle_adjust_schur};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
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

/// Write-back guard of [`run_local_inertial_ba`]: an optimized landmark whose
/// distance from its nearest observer grew by more than this factor...
const WRITE_BACK_MAX_GROWTH: f64 = 10.0;
/// ...and ended beyond this many metres is not written back. Both, so that a
/// near landmark refined from 5 cm to 1 m, or a far one from 300 to 400 m,
/// is kept.
const WRITE_BACK_FAR_M: f64 = 50.0;

/// The local BA window: the three newest keyframes and the live
/// landmarks they observe, plus, as fixed context, every other keyframe that
/// observes those landmarks and the IMU neighbours of the active keyframes.
///
/// Mirrors ORB-SLAM3's `LocalBundleAdjustment` selection. Fixed keyframes
/// always precede the active ones in slot order, because the active keyframes
/// are the newest.
pub fn local_window(map: &Map) -> BaWindow {
    local_window_with_neighbors(map, [])
}

/// [`local_window`], with the caller's own keyframe links (`(from, to)` by
/// `Frame::idx`, such as the stationary factors it holds) treated like IMU
/// edges: an endpoint in the active set pulls the other in as fixed context.
///
/// Without it a keyframe linked to the active window only by a caller-held
/// factor, and observing none of its landmarks, would be left out, and the
/// factor dropped with it.
pub fn local_window_with_neighbors(
    map: &Map,
    neighbors: impl IntoIterator<Item = (usize, usize)>,
) -> BaWindow {
    let keyframes = map.keyframes();
    let active_start = keyframes.len().saturating_sub(MAX_ACTIVE_KFS);
    let active = &keyframes[active_start..];
    let landmark_ids = live_landmarks_of(map, active);

    let active_ids: HashSet<usize> = active.iter().map(|kf| kf.frame.idx).collect();
    let mut context: HashSet<usize> = landmark_ids
        .iter()
        .flat_map(|&id| map.map_points()[id].observer_keyframes())
        .collect();
    let imu_links = map
        .imu_factors()
        .iter()
        .map(|f| (f.prev_kf_idx, f.curr_kf_idx));
    for (from, to) in imu_links.chain(neighbors) {
        if active_ids.contains(&from) || active_ids.contains(&to) {
            context.insert(from);
            context.insert(to);
        }
    }

    let keyframe_slots = keyframes
        .iter()
        .enumerate()
        .filter(|(slot, kf)| *slot >= active_start || context.contains(&kf.frame.idx))
        .map(|(slot, _)| slot)
        .collect();
    BaWindow {
        keyframe_slots,
        landmark_ids,
    }
}

/// The bootstrap pair (the two newest keyframes) and the live landmarks they observe.
fn initial_window(map: &Map) -> BaWindow {
    let keyframes = map.keyframes();
    let start = keyframes.len().saturating_sub(2);
    BaWindow {
        keyframe_slots: (start..keyframes.len()).collect(),
        landmark_ids: live_landmarks_of(map, &keyframes[start..]),
    }
}

/// Live landmark ids observed by `keyframes`, ascending.
fn live_landmarks_of(map: &Map, keyframes: &[Keyframe]) -> Vec<usize> {
    let mut ids: Vec<usize> = keyframes
        .iter()
        .flat_map(|kf| kf.map_point_by_desc_idx.iter().flatten().copied())
        .filter(|&id| map.map_points().get(id).is_some_and(|mp| !mp.culled))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

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
    let mut update = map.ba_snapshot(&initial_window(map)).into_update();
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

    debug_eprintln!(
        "[init_ba] before: reproj_rms={:.3}px median_depth={:.3} kf1_t_norm={:.3} obs={}",
        sq_err_before.sqrt(),
        depth_before,
        kf1_t_before,
        observations.len()
    );
    debug_eprintln!(
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
        debug_eprintln!(
            "[init_ba] scale_state[mp{sample}]: min_dist={:.3} max_dist={:.3} normal_len={:.3} predict_scale@median={predicted}",
            mp.min_distance,
            mp.max_distance,
            normal_len,
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
///
/// `wheel_factors` index keyframes by `Frame::idx`, like the map's IMU
/// factors; those with an endpoint outside the window, or with both endpoints
/// fixed, are dropped, and so is any that fails [`WheelOdomFactor::is_valid`]
/// (one such factor would otherwise fail the whole solve).
///
/// `stationary_factors` are filtered the same way (by
/// [`StationaryFactor::is_valid`]). Capture the window with
/// [`local_window_with_neighbors`] so a stationary neighbour of the active
/// keyframes is in it.
///
/// A landmark the solve left behind any keyframe observing it, or flung far
/// away (its distance from the nearest observer grown more than tenfold and
/// beyond 50 m), keeps its old position: such a landmark had no depth the
/// window could observe, and written back it would poison the next window.
/// [`BaUpdate::solve_stats`] reports how many, with the solver's own counts.
pub fn run_local_inertial_ba(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    imu_t_bc: Option<Pose3d>,
    gravity_world: Vec3F64,
    wheel_factors: &[WheelOdomFactor],
    stationary_factors: &[StationaryFactor],
) -> BaUpdate {
    let mut update = snapshot.into_update();
    let snapshot = &update.snapshot;
    use crate::mapping::vi_ba_schur::{
        ImuFactor as ViBaImuFactor, ViBaParams, visual_inertial_bundle_adjust,
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
            // Keeps the walk over the edge's still time, which the samples
            // alone do not cover.
            *preintegrated = factor.repreintegrate(current_bias);
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

    let wheel_edges: Vec<WheelOdomFactor> = wheel_factors
        .iter()
        .filter_map(|f| {
            let from = *frame_idx_to_slot.get(&f.from_idx)?;
            let to = *frame_idx_to_slot.get(&f.to_idx)?;
            if from < active_start && to < active_start {
                return None;
            }
            // One bad factor would fail the whole solve, losing every
            // window update for as long as it stays in the window: drop it.
            if !f.is_valid() {
                debug_eprintln!(
                    "[vi_ba] skipping wheel factor {} -> {}: non-finite or not positive definite",
                    f.from_idx,
                    f.to_idx
                );
                return None;
            }
            Some(WheelOdomFactor {
                from_idx: from,
                to_idx: to,
                ..*f
            })
        })
        .collect();

    let stationary_edges: Vec<StationaryFactor> = stationary_factors
        .iter()
        .filter_map(|f| {
            let from = *frame_idx_to_slot.get(&f.from_idx)?;
            let to = *frame_idx_to_slot.get(&f.to_idx)?;
            if from < active_start && to < active_start {
                return None;
            }
            if !f.is_valid() {
                debug_eprintln!(
                    "[vi_ba] skipping stationary factor {} -> {}: invalid",
                    f.from_idx,
                    f.to_idx
                );
                return None;
            }
            Some(StationaryFactor {
                from_idx: from,
                to_idx: to,
                ..*f
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
        &wheel_edges,
        &stationary_edges,
        camera,
        &ViBaParams {
            imu_t_bc,
            gravity: gravity_world,
            max_iterations: 50,
            ..ViBaParams::default()
        },
    ) {
        Ok(r) => r,
        Err(error) => {
            debug_eprintln!("[vi_ba] local inertial BA failed, window unchanged: {error}");
            return update;
        }
    };

    // Write back optimised poses, velocities, and biases for active keyframes.
    for kf_idx in active_start..n_kfs {
        let vi_kf = &vi_result.keyframes[kf_idx];
        update.keyframes[kf_idx].pose_world_to_cam = vi_kf.pose;
        update.keyframes[kf_idx].velocity_world = vi_kf.velocity;
        update.keyframes[kf_idx].imu_bias = vi_kf.bias;
    }

    let keep = landmarks_to_write_back(
        &observations,
        (&vi_keyframes, &window.positions),
        (&vi_result.keyframes, &vi_result.points),
    );
    let points_rejected = keep.iter().filter(|&&k| !k).count();
    if points_rejected > 0 {
        debug_eprintln!(
            "[vi_ba] write-back guard kept {points_rejected} landmark(s) at their old position"
        );
    }
    window.write_back_where(&vi_result.points, &keep, &mut update.map_points);
    update.solve_stats = Some(LocalBaStats {
        initial_cost: vi_result.initial_cost,
        final_cost: vi_result.final_cost,
        iterations: vi_result.iterations,
        converged: vi_result.converged,
        depth_rows_gated: vi_result.depth_rows_gated,
        points_reseeded: vi_result.points_reseeded,
        cheirality_rejections: vi_result.cheirality_rejections,
        cheirality_points_held: vi_result.cheirality_points_held,
        points_rejected,
    });
    update
}

/// Per window landmark, whether its optimized position may be written back:
/// not behind any keyframe observing it, and not flung far (its distance from
/// the nearest observer grown more than [`WRITE_BACK_MAX_GROWTH`]-fold and
/// beyond [`WRITE_BACK_FAR_M`]). States are `(keyframes, landmarks)` before
/// and after the solve.
pub(super) fn landmarks_to_write_back(
    observations: &[BaObservation],
    before: (&[ViBaKeyframe], &[Vec3F64]),
    after: (&[ViBaKeyframe], &[Vec3F64]),
) -> Vec<bool> {
    let n = after.1.len();
    let mut behind = vec![false; n];
    let mut nearest = vec![(f64::INFINITY, f64::INFINITY); n];
    let camera_centre = |pose: &Pose3d| pose.inverse().translation;
    for obs in observations {
        let (Some(kf_before), Some(kf_after)) =
            (before.0.get(obs.pose_idx), after.0.get(obs.pose_idx))
        else {
            continue;
        };
        let (Some(p_before), Some(p_after)) =
            (before.1.get(obs.point_idx), after.1.get(obs.point_idx))
        else {
            continue;
        };
        let j = obs.point_idx;
        behind[j] |= kf_after.pose.transform_point(p_after).z <= 0.0;
        let d_before = (*p_before - camera_centre(&kf_before.pose)).length();
        let d_after = (*p_after - camera_centre(&kf_after.pose)).length();
        nearest[j] = (nearest[j].0.min(d_before), nearest[j].1.min(d_after));
    }
    (0..n)
        .map(|j| {
            let (d_before, d_after) = nearest[j];
            if d_before == f64::INFINITY {
                // Unobserved: the solver did not move it.
                return true;
            }
            let flung = d_after > WRITE_BACK_FAR_M && d_after > WRITE_BACK_MAX_GROWTH * d_before;
            !behind[j] && d_after.is_finite() && !flung
        })
        .collect()
}

/// Live landmarks a BA window observes, ordered by ascending map index; a
/// landmark's position in `global` is its solver index.
struct WindowPoints {
    global: Vec<usize>,
    /// Snapshot slot of each landmark in `global`.
    slots: Vec<usize>,
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
            for &mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(slot) = snapshot.landmark_slot(mp_idx)
                    && !snapshot.map_points()[slot].culled
                {
                    unique.insert(mp_idx);
                }
            }
        }
        let mut global: Vec<usize> = unique.into_iter().collect();
        global.sort_unstable();
        let slots: Vec<usize> = global
            .iter()
            .filter_map(|&idx| snapshot.landmark_slot(idx))
            .collect();
        let local_of = global
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let positions = slots
            .iter()
            .map(|&slot| snapshot.map_points()[slot].position)
            .collect();
        Self {
            global,
            slots,
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
        self.write_back_where(optimized, &vec![true; optimized.len()], map_points);
    }

    /// [`Self::write_back`] of the landmarks whose `keep` is true.
    fn write_back_where(&self, optimized: &[Vec3F64], keep: &[bool], map_points: &mut [Vec3F64]) {
        for ((&slot, point), _) in self
            .slots
            .iter()
            .zip(optimized)
            .zip(keep)
            .filter(|(_, keep)| **keep)
        {
            if let Some(mp) = map_points.get_mut(slot) {
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
