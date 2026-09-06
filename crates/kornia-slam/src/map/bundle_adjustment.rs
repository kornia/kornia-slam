//! Bundle adjustment and publication of optimized map snapshots.

use super::{
    Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR, STEREO_DEPTH_MIN_SIGMA, STEREO_DEPTH_REL_SIGMA,
};
use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
struct KeyframeBaState {
    idx: usize,
    pose_world_to_cam: Pose3d,
    velocity_world: Vec3F64,
    imu_bias: ImuBias,
}

/// A private copy of the map on which local bundle adjustment can run without
/// holding the live map lock.
#[derive(Debug, Clone)]
pub struct LocalBaSnapshot {
    optimized: Map,
    world_epoch: u64,
    keyframes_before: Vec<KeyframeBaState>,
    map_points_before: Vec<Vec3F64>,
}

impl LocalBaSnapshot {
    /// Optimizes the visual local window in this private snapshot.
    pub fn run_visual(&mut self, camera: &PinholeCamera) {
        self.optimized.run_local_ba(camera);
    }

    /// Optimizes the visual-inertial local window in this private snapshot.
    pub fn run_inertial(
        &mut self,
        camera: &PinholeCamera,
        imu_t_bc: Option<Pose3d>,
        gravity_world: Vec3F64,
    ) {
        self.optimized
            .run_local_inertial_ba(camera, imu_t_bc, gravity_world);
    }
}

/// A keyframe state change accepted from an asynchronous local BA result.
#[derive(Debug, Clone, Copy)]
pub struct KeyframeBaCorrection {
    pub kf_idx: usize,
    pub pose_before: Pose3d,
    pub pose_after: Pose3d,
    pub velocity_world: Vec3F64,
    pub imu_bias: ImuBias,
}

/// Changes accepted while merging an asynchronous local BA snapshot.
#[derive(Debug, Default)]
pub struct LocalBaMergeResult {
    pub keyframe_corrections: Vec<KeyframeBaCorrection>,
    pub map_points_updated: usize,
}

impl Map {
    /// Creates a private local-BA snapshot. The returned value owns all solver
    /// inputs and can be optimized without holding a lock on this map.
    pub fn local_ba_snapshot(&self) -> LocalBaSnapshot {
        LocalBaSnapshot {
            optimized: self.clone(),
            world_epoch: self.world_epoch,
            keyframes_before: self
                .keyframes
                .iter()
                .map(|kf| KeyframeBaState {
                    idx: kf.frame.idx,
                    pose_world_to_cam: kf.frame.pose_world_to_cam,
                    velocity_world: kf.velocity_world,
                    imu_bias: kf.imu_bias,
                })
                .collect(),
            map_points_before: self.map_points.iter().map(|mp| mp.position).collect(),
        }
    }

    /// Merges solver-changed geometry from a local-BA snapshot.
    ///
    /// Entities inserted after the snapshot are left untouched. A snapshot is
    /// rejected wholesale if the live map has since been scaled, rotated, or
    /// cleared, because its coordinates then belong to another world frame.
    pub fn merge_local_ba_snapshot(
        &mut self,
        snapshot: LocalBaSnapshot,
    ) -> Option<LocalBaMergeResult> {
        if self.world_epoch != snapshot.world_epoch {
            return None;
        }

        let mut result = LocalBaMergeResult::default();
        for (before, optimized) in snapshot
            .keyframes_before
            .iter()
            .zip(snapshot.optimized.keyframes.iter())
        {
            if !keyframe_ba_state_changed(before, optimized) {
                continue;
            }
            let Some(live) = self.get_keyframe_mut(before.idx) else {
                continue;
            };

            let pose_before = live.frame.pose_world_to_cam;
            live.frame.pose_world_to_cam = optimized.frame.pose_world_to_cam;
            live.velocity_world = optimized.velocity_world;
            live.imu_bias = optimized.imu_bias;
            result.keyframe_corrections.push(KeyframeBaCorrection {
                kf_idx: before.idx,
                pose_before,
                pose_after: optimized.frame.pose_world_to_cam,
                velocity_world: optimized.velocity_world,
                imu_bias: optimized.imu_bias,
            });
        }

        let mut changed_points = Vec::new();
        for (idx, (&before, optimized)) in snapshot
            .map_points_before
            .iter()
            .zip(snapshot.optimized.map_points.iter())
            .enumerate()
        {
            if optimized.position == before {
                continue;
            }
            let Some(live) = self.map_points.get_mut(idx) else {
                continue;
            };
            if live.culled {
                continue;
            }
            live.position = optimized.position;
            changed_points.push(idx);
        }
        result.map_points_updated = changed_points.len();

        for idx in changed_points {
            self.update_map_point_geometry(idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }

        // VI-BA may repropagate an existing preintegration on its private copy.
        // Preserve newer live factors while copying those refreshed edges back.
        for optimized in &snapshot.optimized.imu_factors {
            if let Some(live) = self.imu_factors.iter_mut().find(|live| {
                live.prev_kf_idx == optimized.prev_kf_idx
                    && live.curr_kf_idx == optimized.curr_kf_idx
            }) {
                live.preintegrated = optimized.preintegrated.clone();
            }
        }
        self.cull();
        Some(result)
    }

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
    pub fn run_initial_ba(&mut self, camera: &PinholeCamera) -> bool {
        const MAX_ITERS: usize = 5;
        const HUBER_SCALE_SQ: f32 = 5.991;
        const MIN_OBSERVATIONS: usize = 8;

        let n = self.keyframes.len();
        if n < 2 {
            return false;
        }

        // Last two keyframes; older is gauge-fixed.
        let kf_indices = [n - 2, n - 1];

        let mut mp_set: HashSet<usize> = HashSet::new();
        for &kf_idx in &kf_indices {
            for mp_idx in self.keyframes[kf_idx]
                .map_point_by_desc_idx
                .iter()
                .flatten()
            {
                if let Some(mp) = self.map_points.get(*mp_idx)
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
            .map(|&idx| self.map_points[idx].position)
            .collect();

        let poses: Vec<Pose3d> = kf_indices
            .iter()
            .map(|&i| self.keyframes[i].frame.pose_world_to_cam)
            .collect();

        let mut observations = Vec::new();
        for (pose_idx, &kf_idx) in kf_indices.iter().enumerate() {
            let is_fixed = pose_idx == 0;
            let kf = &self.keyframes[kf_idx];
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

        self.keyframes[kf_indices[1]].frame.pose_world_to_cam = ba_result.poses[1];
        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = ba_result.points[local_idx];
            }
        }
        // Positions (and KF1's pose) moved: refresh scale geometry.
        for &global_idx in &mp_global_indices {
            self.update_map_point_geometry(global_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }

        // Diagnostics: sample one point's scale state for a sanity check.
        if let Some(&sample) = mp_global_indices.first()
            && let Some(mp) = self.map_points.get(sample)
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
    /// via camera), calls `kornia_3d::ba::bundle_adjust`, and writes back optimized poses
    /// and point positions.
    pub fn run_local_ba(&mut self, camera: &PinholeCamera) {
        const MAX_ACTIVE_KFS: usize = 3;
        const MIN_OBSERVATIONS: usize = 8;

        let n_kfs = self.keyframes.len();
        if n_kfs < 2 {
            return;
        }

        let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

        let mut mp_set: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes[active_start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    mp_set.insert(*mp_idx);
                }
            }
        }
        if mp_set.is_empty() {
            return;
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
            .map(|&idx| self.map_points[idx].position)
            .collect();

        let poses: Vec<Pose3d> = self
            .keyframes
            .iter()
            .map(|kf| kf.frame.pose_world_to_cam)
            .collect();

        let mut observations = Vec::new();
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
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
            return;
        }

        let ba_result =
            match bundle_adjust_schur(&poses, &points, &observations, camera, &BaParams::default())
            {
                Ok(r) => r,
                Err(_) => return,
            };

        for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
            if kf_idx >= active_start {
                self.keyframes[kf_idx].frame.pose_world_to_cam = *pose;
            }
        }

        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = ba_result.points[local_idx];
            }
        }
        for &global_idx in &mp_global_indices {
            self.update_map_point_geometry(global_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
    }

    pub fn run_local_inertial_ba(
        &mut self,
        camera: &PinholeCamera,
        imu_t_bc: Option<Pose3d>,
        gravity_world: Vec3F64,
    ) {
        use crate::vi_ba_schur::{
            ImuFactor as ViBaImuFactor, ViBaKeyframe, ViBaParams, visual_inertial_bundle_adjust,
        };

        const MAX_ACTIVE_KFS: usize = 3;
        const MIN_OBSERVATIONS: usize = 8;

        let n_kfs = self.keyframes.len();
        if n_kfs < 2 {
            return;
        }

        let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

        let mut mp_set: HashSet<usize> = HashSet::new();
        for kf in &self.keyframes[active_start..] {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    mp_set.insert(*mp_idx);
                }
            }
        }
        if mp_set.is_empty() {
            return;
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
            .map(|&idx| self.map_points[idx].position)
            .collect();

        // Build VI-BA keyframes (all KFs; fixed flag controls which are optimised).
        let vi_keyframes: Vec<ViBaKeyframe> = self
            .keyframes
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
        for (kf_idx, kf) in self.keyframes.iter().enumerate() {
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
            return;
        }

        // Map global frame.idx → local keyframe slot (0..n_kfs).
        let frame_idx_to_slot: HashMap<usize, usize> = self
            .keyframes
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
        for factor in self.imu_factors.iter_mut() {
            let Some(&from) = frame_idx_to_slot.get(&factor.prev_kf_idx) else {
                continue;
            };
            let Some(&to) = frame_idx_to_slot.get(&factor.curr_kf_idx) else {
                continue;
            };
            if from < active_start && to < active_start {
                continue;
            }
            let current_bias = self.keyframes[from].imu_bias;
            let d_accel = (current_bias.accel - factor.preintegrated.bias.accel).length();
            let d_gyro = (current_bias.gyro - factor.preintegrated.bias.gyro).length();
            if d_accel > REPROPAGATE_BIAS_THRESHOLD || d_gyro > REPROPAGATE_BIAS_THRESHOLD {
                factor.preintegrated = PreintegratedImu::from_measurements(
                    current_bias,
                    factor.preintegrated.calib,
                    &factor.raw_samples,
                    factor.t0,
                    factor.t1,
                );
            }
        }

        // Build IMU edges; include only edges where at least one endpoint is active.
        let imu_edges: Vec<ViBaImuFactor> = self
            .imu_factors
            .iter()
            .filter_map(|f| {
                let from = *frame_idx_to_slot.get(&f.prev_kf_idx)?;
                let to = *frame_idx_to_slot.get(&f.curr_kf_idx)?;
                if from < active_start && to < active_start {
                    return None;
                }
                Some(ViBaImuFactor {
                    from_idx: from,
                    to_idx: to,
                    preintegrated: f.preintegrated.clone(),
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
            Err(_) => return,
        };

        // Write back optimised poses, velocities, and biases for active keyframes.
        for kf_idx in active_start..n_kfs {
            let vi_kf = &vi_result.keyframes[kf_idx];
            self.keyframes[kf_idx].frame.pose_world_to_cam = vi_kf.pose;
            self.keyframes[kf_idx].velocity_world = vi_kf.velocity;
            self.keyframes[kf_idx].imu_bias = vi_kf.bias;
        }

        for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
            if let Some(mp) = self.map_points.get_mut(global_idx) {
                mp.position = vi_result.points[local_idx];
            }
        }
        for &global_idx in &mp_global_indices {
            self.update_map_point_geometry(global_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
    }
}
fn keyframe_ba_state_changed(before: &KeyframeBaState, after: &Keyframe) -> bool {
    before.pose_world_to_cam != after.frame.pose_world_to_cam
        || before.velocity_world != after.velocity_world
        || before.imu_bias.gyro != after.imu_bias.gyro
        || before.imu_bias.accel != after.imu_bias.accel
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
mod tests {
    use super::super::{MapPoint, tests::test_frame};
    use super::*;
    #[test]
    fn stereo_depth_obs_uses_proportional_sigma_with_floor() {
        let mut frame = test_frame(0, vec![[0u8; 32], [1u8; 32]]);
        frame.depth = vec![10.0, -1.0];
        frame.u_right = vec![5.0, -1.0];
        let kf = Keyframe::from_frame(frame);

        // Valid depth: sigma = 0.05 * 10 = 0.5.
        let (d, s) = stereo_depth_obs(&kf, 0);
        assert_eq!(d, Some(10.0));
        assert!((s - 0.5).abs() < 1e-6);

        // Sentinel depth: no measurement.
        let (d1, s1) = stereo_depth_obs(&kf, 1);
        assert_eq!(d1, None);
        assert!((s1 - 1.0).abs() < 1e-9);

        // Very near depth clamps to the sigma floor.
        let mut near = test_frame(1, vec![[0u8; 32]]);
        near.depth = vec![0.1]; // 0.05 * 0.1 = 0.005 < floor
        let kf_near = Keyframe::from_frame(near);
        let (_, s2) = stereo_depth_obs(&kf_near, 0);
        assert!((s2 - STEREO_DEPTH_MIN_SIGMA).abs() < 1e-9);
    }

    #[test]
    fn local_ba_snapshot_merge_updates_only_snapshot_entities() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));
        map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));

        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.keyframes[0]
            .frame
            .pose_world_to_cam
            .translation
            .x = 2.0;
        snapshot.optimized.map_points[0].position.x = 3.0;

        map.upsert_keyframe(Keyframe::from_frame(test_frame(1, vec![[1u8; 32]])));
        let later_point = map.push_map_point(MapPoint::new(
            Vec3F64::new(9.0, 0.0, 5.0),
            [1u8; 32],
            0,
            [0; 3],
            1,
        ));

        let merged = map
            .merge_local_ba_snapshot(snapshot)
            .expect("snapshot should still use the live world frame");

        assert_eq!(merged.keyframe_corrections.len(), 1);
        assert_eq!(merged.keyframe_corrections[0].kf_idx, 0);
        assert_eq!(
            map.get_keyframe(0)
                .unwrap()
                .frame
                .pose_world_to_cam
                .translation
                .x,
            2.0
        );
        assert_eq!(map.map_points()[0].position.x, 3.0);
        assert_eq!(
            map.get_keyframe(1).unwrap().frame.pose_world_to_cam,
            Pose3d::IDENTITY
        );
        assert_eq!(map.map_points()[later_point].position.x, 9.0);
    }

    #[test]
    fn local_ba_snapshot_merge_rejects_an_obsolete_world_frame() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));
        map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));

        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.map_points[0].position.x = 7.0;
        map.scale_world(2.0);

        assert!(map.merge_local_ba_snapshot(snapshot).is_none());
        assert_eq!(map.map_points()[0].position.x, 2.0);
    }

    #[test]
    fn pose_graph_correction_invalidates_older_local_ba_snapshot() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])));
        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.keyframes[0]
            .frame
            .pose_world_to_cam
            .translation
            .x = 9.0;
        let corrected = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-1.0, 0.0, 0.0),
        );

        map.apply_pose_graph_correction(&[0], &[Pose3d::IDENTITY], &[corrected])
            .unwrap();

        assert!(map.merge_local_ba_snapshot(snapshot).is_none());
        assert_eq!(
            map.get_keyframe(0).unwrap().frame.pose_world_to_cam,
            corrected
        );
    }
}
