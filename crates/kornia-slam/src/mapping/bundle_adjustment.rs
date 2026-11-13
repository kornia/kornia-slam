//! BA window selection, observation assembly, and numerical solving.
//!
//! Solvers consume immutable map snapshots and return numerical updates. The
//! map owns capture and writeback; the local-mapping worker owns scheduling.

use crate::mapping::map::{
    BaSnapshot, BaUpdate, BaUpdateError, BaWindow, Keyframe, LocalBaStats, Map, ORB_N_LEVELS,
    ORB_SCALE_FACTOR, StateEnvelope,
};
use crate::mapping::marginalization::{
    BoundaryPrior, FrozenPoseSigma, MarginalPrior, marginalize_frozen,
};
use crate::mapping::stationary::StationaryFactor;
use crate::mapping::vi_ba_schur::{
    ImuFactor as ViBaImuFactor, ImuHuber, RigObservationCamera, ViBaKeyframe, ViBaParams,
};
use crate::mapping::visual_information::VisualInformation;
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

/// Local BA settings resolved against the rig. The defaults are the
/// solver's long-standing constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalBaParams {
    /// Keyframes a local BA optimizes; older keyframes enter only as fixed poses.
    pub active_keyframes: usize,
    /// See [`ViBaParams::boundary_imu_info_scale`](crate::mapping::vi_ba_schur::ViBaParams).
    pub boundary_imu_info_scale: f64,
    pub boundary_scales_bias_rows: bool,
    pub boundary: BoundaryPrior,
    pub imu_huber: ImuHuber,
    pub depth_sigma: DepthSigma,
    /// Reprojection weighting by keypoint octave. Only VI-BA honours it:
    /// kornia-3d's visual-only solver has no per-observation weight.
    pub visual_information: VisualInformation,
    /// Bounds a VI-BA result must stay inside to be written back.
    pub state_envelope: StateEnvelope,
    pub lost_keyframes: LostKeyframes,
}

/// How VI-BA treats keyframes inserted while tracking was lost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LostKeyframes {
    /// Full 15-DOF state, like any other keyframe.
    #[default]
    Free,
    /// Velocity and bias held at the propagated values: with no match to the
    /// map nothing observes them, and the solve would write back noise.
    FixedInertial,
}

impl Default for LocalBaParams {
    fn default() -> Self {
        Self {
            active_keyframes: 3,
            boundary_imu_info_scale: 1e-2,
            boundary_scales_bias_rows: true,
            boundary: BoundaryPrior::ScaledEdge,
            imu_huber: ImuHuber::AllEdges,
            depth_sigma: DepthSigma::Relative(STEREO_DEPTH_REL_SIGMA),
            visual_information: VisualInformation::Unit,
            state_envelope: StateEnvelope::Off,
            lost_keyframes: LostKeyframes::Free,
        }
    }
}

/// Standard deviation of a stereo depth measurement `z` (metres), floored at
/// [`STEREO_DEPTH_MIN_SIGMA`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepthSigma {
    /// `fraction · z`.
    Relative(f32),
    /// Disparity noise propagated to depth: `z² · px / (f · b)`.
    Disparity { px_over_focal_baseline: f32 },
}

impl DepthSigma {
    #[cfg(test)]
    fn sigma(self, z: f32) -> f32 {
        self.sigma_at(z, 1.0)
    }

    /// The disparity noise scales with the keypoint's pixel sigma
    /// (ORB-SLAM3's `EdgeStereo` weights `u_r` by the same `invSigma2`).
    fn sigma_at(self, z: f32, pixel_sigma: f32) -> f32 {
        let sigma = match self {
            Self::Relative(fraction) => fraction * z,
            Self::Disparity {
                px_over_focal_baseline,
            } => px_over_focal_baseline * pixel_sigma * z * z,
        };
        sigma.max(STEREO_DEPTH_MIN_SIGMA)
    }
}

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
    local_window_of(map, LocalBaParams::default().active_keyframes, neighbors)
}

/// [`local_window_with_neighbors`] with `active_keyframes` newest keyframes active.
pub fn local_window_of(
    map: &Map,
    active_keyframes: usize,
    neighbors: impl IntoIterator<Item = (usize, usize)>,
) -> BaWindow {
    let keyframes = map.keyframes();
    let active_start = keyframes.len().saturating_sub(active_keyframes);
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
    depth_sigma: DepthSigma,
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
        false,
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
        depth_sigma,
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
    run_local_ba_with(snapshot, camera, &LocalBaParams::default())
}

/// [`run_local_ba`] with `params`; the snapshot must come from a window of
/// the same `active_keyframes`.
pub fn run_local_ba_with(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    params: &LocalBaParams,
) -> BaUpdate {
    let mut update = snapshot.into_update();
    let snapshot = &update.snapshot;
    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(params.active_keyframes);

    let window = WindowPoints::collect(snapshot, &snapshot.keyframes()[active_start..], false);
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
        params.depth_sigma,
    );

    if observations.len() < MIN_OBSERVATIONS {
        return update;
    }

    // Tracked matches reach keyframes unfiltered, so a plain L2 solve lets
    // every mismatch pull the window; same kernel as VI-BA.
    let vi_defaults = ViBaParams::default();
    let ba_params = BaParams {
        robust: RobustKernelKind::Huber,
        robust_scale_sq: vi_defaults.huber_reproj_px_sq as f32,
        depth_robust_scale_sq: vi_defaults.huber_depth_sigma_sq as f32,
        ..BaParams::default()
    };
    let ba_result =
        match bundle_adjust_schur(&poses, &window.positions, &observations, camera, &ba_params) {
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
    run_local_inertial_ba_with(
        snapshot,
        camera,
        imu_t_bc,
        gravity_world,
        wheel_factors,
        stationary_factors,
        &LocalBaParams::default(),
    )
}

/// [`run_local_inertial_ba`] with `params`; the snapshot must come from a
/// window of the same `active_keyframes`.
pub fn run_local_inertial_ba_with(
    snapshot: BaSnapshot,
    camera: &PinholeCamera,
    imu_t_bc: Option<Pose3d>,
    gravity_world: Vec3F64,
    wheel_factors: &[WheelOdomFactor],
    stationary_factors: &[StationaryFactor],
    params: &LocalBaParams,
) -> BaUpdate {
    let mut update = snapshot.into_update();
    update.state_envelope = params.state_envelope;
    let snapshot = &update.snapshot;
    use crate::mapping::vi_ba_schur::visual_inertial_bundle_adjust;

    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = snapshot.keyframes().len();
    if n_kfs < 2 {
        return update;
    }

    let active_start = n_kfs.saturating_sub(params.active_keyframes);

    let window = WindowPoints::collect(snapshot, &snapshot.keyframes()[active_start..], true);
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
            inertial_fixed: params.lost_keyframes == LostKeyframes::FixedInertial
                && kf.inserted_while_lost,
        })
        .collect();

    let WindowObservations {
        observations,
        views: observation_views,
        sigma_px: observation_sigma_px,
    } = window.observations_with_views(
        snapshot
            .keyframes()
            .iter()
            .enumerate()
            .map(|(kf_idx, kf)| (kf_idx, kf, kf_idx < active_start)),
        camera,
        true,
        params.depth_sigma,
        params.visual_information,
    );
    let rig_views = rig_observation_cameras(snapshot.keyframes());

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
    let mut vi_params = ViBaParams {
        imu_t_bc,
        gravity: gravity_world,
        max_iterations: 50,
        boundary_imu_info_scale: params.boundary_imu_info_scale,
        boundary_scales_bias_rows: params.boundary_scales_bias_rows,
        boundary: params.boundary,
        imu_huber: params.imu_huber,
        rig_views: rig_views.clone(),
        observation_views: observation_views.clone(),
        observation_sigma_px,
        ..ViBaParams::default()
    };
    let boundary = imu_edges
        .iter()
        .find(|e| e.from_idx < active_start && e.to_idx >= active_start);
    if let (BoundaryPrior::Marginal(sigma), Some(edge)) = (params.boundary, boundary) {
        vi_params.prior = match carried_prior(snapshot, &vi_keyframes, edge, &vi_params, sigma) {
            Ok(prior) => Some(prior),
            Err(reason) => {
                debug_eprintln!("[vi_ba] PRIOR fallback: {reason}");
                None
            }
        };
    }
    let vi_result = match visual_inertial_bundle_adjust(
        &vi_keyframes,
        &window.positions,
        &observations,
        &imu_edges,
        &wheel_edges,
        &stationary_edges,
        camera,
        &vi_params,
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

    let camera_from_primary = |o: usize| {
        let k = (*observation_views.get(o)? as usize).checked_sub(1)?;
        rig_views.get(k).map(|view| view.camera_from_primary)
    };
    let keep = landmarks_to_write_back(
        &observations,
        camera_from_primary,
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
    if let (BoundaryPrior::Marginal(sigma), Some(edge)) = (params.boundary, boundary) {
        // A solve that fell back starts the chain from its own result.
        update.next_prior = vi_params.prior.take().or_else(|| {
            let (from, to) = (edge.from_idx, edge.to_idx);
            let frozen = &vi_result.keyframes;
            marginalize_frozen(None, &frozen[from], &frozen[to], edge, &vi_params, sigma)
                .ok()
                .map(|prior| stamped(prior, snapshot, to))
        });
    }
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

/// The boundary prior for the window whose boundary IMU edge is `edge`:
/// the map's prior if it is already on the edge's active keyframe, or moved
/// onto it by eliminating the edge's frozen keyframe. `Err` says why there
/// is none, and the solve falls back to the scaled edge.
fn carried_prior(
    snapshot: &BaSnapshot,
    keyframes: &[ViBaKeyframe],
    edge: &ViBaImuFactor,
    params: &ViBaParams,
    sigma: FrozenPoseSigma,
) -> Result<MarginalPrior, String> {
    let (from, to) = (edge.from_idx, edge.to_idx);
    let frame = |slot: usize| snapshot.keyframes()[slot].frame.idx;
    let stored = snapshot
        .marginal_prior()
        .filter(|prior| prior.world_epoch == snapshot.world_epoch())
        .ok_or("no prior")?;
    if stored.frame_idx == frame(to) {
        if stored.is_stale(&keyframes[to]) {
            return Err(format!("stale on kf {}", stored.frame_idx));
        }
        return Ok(stored.clone());
    }
    if stored.frame_idx != frame(from) {
        return Err(format!(
            "prior on kf {}, boundary {} -> {}",
            stored.frame_idx,
            frame(from),
            frame(to)
        ));
    }
    if stored.is_stale(&keyframes[from]) {
        return Err(format!("stale on kf {}", stored.frame_idx));
    }
    marginalize_frozen(
        Some(stored),
        &keyframes[from],
        &keyframes[to],
        edge,
        params,
        sigma,
    )
    .map(|prior| stamped(prior, snapshot, to))
    .map_err(|e| e.to_string())
}

/// `prior` as the prior on snapshot slot `slot`, in the snapshot's epoch.
fn stamped(prior: MarginalPrior, snapshot: &BaSnapshot, slot: usize) -> MarginalPrior {
    MarginalPrior {
        frame_idx: snapshot.keyframes()[slot].frame.idx,
        world_epoch: snapshot.world_epoch(),
        ..prior
    }
}

/// Per window landmark, whether its optimized position may be written back:
/// not behind any keyframe observing it, and not flung far (its distance from
/// the nearest observer grown more than [`WRITE_BACK_MAX_GROWTH`]-fold and
/// beyond [`WRITE_BACK_FAR_M`]). States are `(keyframes, landmarks)` before
/// and after the solve.
///
/// `camera_from_primary(o)` is observation `o`'s secondary rig camera
/// extrinsic, `None` for the primary camera.
pub(super) fn landmarks_to_write_back(
    observations: &[BaObservation],
    camera_from_primary: impl Fn(usize) -> Option<Pose3d>,
    before: (&[ViBaKeyframe], &[Vec3F64]),
    after: (&[ViBaKeyframe], &[Vec3F64]),
) -> Vec<bool> {
    let n = after.1.len();
    let mut behind = vec![false; n];
    let mut nearest = vec![(f64::INFINITY, f64::INFINITY); n];
    let camera_centre = |pose: &Pose3d| pose.inverse().translation;
    for (o, obs) in observations.iter().enumerate() {
        let (Some(kf_before), Some(kf_after)) =
            (before.0.get(obs.pose_idx), after.0.get(obs.pose_idx))
        else {
            continue;
        };
        let extrinsic = camera_from_primary(o);
        let in_camera = |pose: &Pose3d| match &extrinsic {
            Some(t) => t.compose(pose),
            None => *pose,
        };
        let (pose_before, pose_after) = (in_camera(&kf_before.pose), in_camera(&kf_after.pose));
        let (Some(p_before), Some(p_after)) =
            (before.1.get(obs.point_idx), after.1.get(obs.point_idx))
        else {
            continue;
        };
        let j = obs.point_idx;
        behind[j] |= pose_after.transform_point(p_after).z <= 0.0;
        let d_before = (*p_before - camera_centre(&pose_before)).length();
        let d_after = (*p_after - camera_centre(&pose_after)).length();
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

/// Secondary cameras of the window's rig, in rig order after the primary;
/// empty outside a rig.
fn rig_observation_cameras(keyframes: &[Keyframe]) -> Vec<RigObservationCamera> {
    keyframes
        .iter()
        .find_map(|kf| kf.frame.rig.geometry.as_ref().filter(|_| kf.frame.is_rig()))
        .map(|rig| {
            rig.views()
                .iter()
                .skip(1)
                .map(|view| RigObservationCamera {
                    camera: view.camera.clone(),
                    camera_from_primary: view.camera_from_primary,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Observations of a window with, per observation, its rig camera and its
/// pixel sigma (empty at unit information).
struct WindowObservations {
    observations: Vec<BaObservation>,
    views: Vec<u8>,
    sigma_px: Vec<f32>,
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
    /// With `all_cameras` unset only landmarks a primary-camera feature
    /// holds: a single-camera solve has no rows for the others, and a point
    /// without rows has a singular block.
    fn collect<'a>(
        snapshot: &BaSnapshot,
        keyframes: impl IntoIterator<Item = &'a Keyframe>,
        all_cameras: bool,
    ) -> Self {
        let mut unique: HashSet<usize> = HashSet::new();
        for kf in keyframes {
            let held = kf
                .map_point_by_desc_idx
                .iter()
                .enumerate()
                .filter(|&(desc_idx, _)| all_cameras || kf.frame.camera_of(desc_idx) == 0)
                .filter_map(|(_, mp)| mp.as_ref());
            for &mp_idx in held {
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
    /// from each `(pose_idx, keyframe, fixed)` in order, by the primary
    /// camera only: the single-camera solvers cannot model a rig's other
    /// cameras.
    fn observations<'a>(
        &self,
        poses: impl IntoIterator<Item = (usize, &'a Keyframe, bool)>,
        camera: &PinholeCamera,
        depth_sigma: DepthSigma,
    ) -> Vec<BaObservation> {
        self.observations_with_views(poses, camera, false, depth_sigma, VisualInformation::Unit)
            .observations
    }

    /// Like [`Self::observations`], with every rig camera's observations
    /// when `all_cameras` is set, each one's rig camera index and, unless
    /// `information` is unit, its pixel sigma.
    fn observations_with_views<'a>(
        &self,
        poses: impl IntoIterator<Item = (usize, &'a Keyframe, bool)>,
        camera: &PinholeCamera,
        all_cameras: bool,
        depth_sigma: DepthSigma,
        information: VisualInformation,
    ) -> WindowObservations {
        let mut observations = Vec::new();
        let mut views = Vec::new();
        let mut sigma_px = Vec::new();
        for (pose_idx, kf, fixed_pose) in poses {
            for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(&point_idx) = mp_opt.and_then(|mp_idx| self.local_of.get(&mp_idx)) else {
                    continue;
                };
                let view = kf.frame.camera_of(desc_idx);
                if view != 0 && !all_cameras {
                    continue;
                }
                if let Some(pixel) = kf.frame.undistorted_xy(desc_idx, camera) {
                    let octave = kf.frame.features.octaves.get(desc_idx).copied();
                    let pixel_sigma = information.sigma_px(octave.unwrap_or(0));
                    let (depth_meas, depth_sigma) =
                        stereo_depth_obs(kf, desc_idx, depth_sigma, pixel_sigma);
                    if !information.is_unit() {
                        sigma_px.push(pixel_sigma);
                    }
                    observations.push(BaObservation {
                        pose_idx,
                        point_idx,
                        pixel,
                        fixed_pose,
                        fixed_point: false,
                        depth_meas,
                        depth_sigma,
                    });
                    views.push(view as u8);
                }
            }
        }
        WindowObservations {
            observations,
            views,
            sigma_px,
        }
    }

    fn write_back(&self, optimized: &[Vec3F64], map_points: &mut [Vec3F64]) {
        for (&slot, point) in self.slots.iter().zip(optimized) {
            if let Some(mp) = map_points.get_mut(slot) {
                *mp = *point;
            }
        }
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
fn stereo_depth_obs(
    kf: &Keyframe,
    desc_idx: usize,
    depth_sigma: DepthSigma,
    pixel_sigma: f32,
) -> (Option<f32>, f32) {
    match kf.frame.stereo_depth(desc_idx) {
        Some(z) => (Some(z), depth_sigma.sigma_at(z, pixel_sigma)),
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
