//! Bundle adjustment entry points and their diagnostics.

use crate::map::{Map, ORB_N_LEVELS, ORB_SCALE_FACTOR, stereo_depth_obs};
use kornia_3d::ba::BaObservation;
use kornia_3d::ba::BaParams;
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::PreintegratedImu;
use std::collections::HashMap;
use std::collections::HashSet;

/// Quality metrics for a freshly-bootstrapped 2-keyframe map.
///
/// Used as the gate for accepting a bootstrap result. Mirrors
/// ORB-SLAM3's reset criteria in `CreateInitialMapMonocular`:
/// `medianDepth < 0 || TrackedMapPoints(1) < 50`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitialMapHealth {
    /// Number of map points with positive depth in both bootstrap KFs.
    pub valid_in_both: usize,
    /// Median depth of valid points in the older KF's frame.
    pub median_depth_older_kf: f64,
}

impl Map {
    /// Health metrics for the just-bootstrapped pair of keyframes.
    ///
    /// Inspects the last two keyframes in insertion order and reports how
    /// many associated map points still have positive depth in both KFs,
    /// plus the median depth in the older KF's frame. Used to decide whether
    /// a freshly-bootstrapped map is safe to commit.
    pub fn initial_map_health(&self) -> InitialMapHealth {
        let n = self.keyframes.len();
        if n < 2 {
            return InitialMapHealth::default();
        }
        let kf_older = &self.keyframes[n - 2];
        let kf_newer = &self.keyframes[n - 1];
        let pose_older = kf_older.frame.pose_world_to_cam;
        let pose_newer = kf_newer.frame.pose_world_to_cam;

        // Collect MPs observed by either KF, dedup.
        let mut seen: HashSet<usize> = HashSet::new();
        for mp_idx in kf_older
            .map_point_by_desc_idx
            .iter()
            .chain(kf_newer.map_point_by_desc_idx.iter())
            .flatten()
        {
            seen.insert(*mp_idx);
        }

        let mut depths_older: Vec<f64> = Vec::with_capacity(seen.len());
        let mut valid_in_both = 0usize;
        for idx in seen {
            let Some(mp) = self.map_points.get(idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            let z_older = pose_older.transform_point(&mp.position).z;
            let z_newer = pose_newer.transform_point(&mp.position).z;
            if z_older > 0.0 && z_newer > 0.0 {
                valid_in_both += 1;
                depths_older.push(z_older);
            }
        }

        let median_depth = if depths_older.is_empty() {
            0.0
        } else {
            let mid = depths_older.len() / 2;
            depths_older.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            depths_older[mid]
        };

        InitialMapHealth {
            valid_in_both,
            median_depth_older_kf: median_depth,
        }
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
