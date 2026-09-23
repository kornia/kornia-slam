//! Growing the map from an accepted keyframe, and fusing it into its
//! neighbours.
//!
//! Both work in two phases against the shared map: a read-only phase that
//! matches and triangulates, then a write phase that takes its own lock. The
//! phases are kept apart deliberately — triangulation is the expensive part and
//! must not hold the map against the local-mapping worker while it runs.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_imgproc::features::{OrbMatchConfig, hamming_distance, match_orb_descriptors};

use crate::mapping::map::ORB_SCALE_FACTOR;
use crate::mapping::{Keyframe, Map};

pub(crate) fn grow_map_points_from_keyframe_pair(
    map: &Arc<Mutex<Map>>,
    camera: &PinholeCamera,
    prev_kf_idx: usize,
    curr_kf: &mut Keyframe,
    match_config: OrbMatchConfig,
    triangulation_config: &TriangulationConfig,
) -> usize {
    const MIN_GROWTH_MATCHES: usize = 15;
    // 1-DOF chi-square gate at 95% for the point-to-epipolar-line distance
    // (ORB-SLAM3's CheckDistEpipolarLine), scaled per-octave below.
    const EPIPOLAR_CHI2: f64 = 3.84;

    // Read-only phase: match and triangulate against the neighbor
    // keyframe stored in the map. The shared borrow on the map ends with
    // this block so the write phase below can mutate the map.
    let points = {
        let map_guard = map.lock().unwrap();
        let Some(prev_kf) = map_guard.get_keyframe(prev_kf_idx) else {
            return 0;
        };

        // Only consider features that don't already have a map point in
        // either KF. Matching the full descriptor arrays and then
        // filtering discards almost everything once the KFs are mature
        // (the best matches always land on the already-tracked features).
        let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
            .filter(|&i| prev_kf.map_point(i).is_none())
            .collect();
        let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
            .filter(|&i| curr_kf.map_point(i).is_none())
            .collect();
        if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
            return 0;
        }

        // Both keyframe poses are known, so the fundamental matrix between
        // the pair is fully determined: F = K^-T [t]x R K^-1 with (R, t)
        // the prev->curr relative pose. Filtering matches against this F
        // replaces the F-matrix RANSAC of the two-view estimator (mirrors
        // ORB-SLAM3's SearchForTriangulation).
        let rel = Pose3d::between(
            &prev_kf.frame.pose_world_to_cam,
            &curr_kf.frame.pose_world_to_cam,
        );
        if rel.translation.length() <= 1e-8 {
            // No baseline: epipolar geometry degenerates and triangulation
            // would reject everything anyway.
            return 0;
        }
        let t = rel.translation;
        let t_skew = Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        );
        let k_inv = Mat3F64::from_cols(
            Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
            Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
        );
        let f_mat = k_inv.transpose() * (t_skew * rel.rotation) * k_inv;

        // Epipole of the prev camera in the curr image (projection of
        // prev's camera center). Near it every keypoint is close to every
        // epipolar line, so the chi-square gate below is uninformative
        // there: wrong matches survive and triangulate to depth-garbage
        // points that still reproject well in both views. Mirrors
        // ORB-SLAM3's epipole-proximity rejection in
        // SearchForTriangulation; RANSAC consensus used to absorb these.
        let prev_center_world = prev_kf.frame.pose_world_to_cam.inverse().translation;
        let epipole_cam = curr_kf
            .frame
            .pose_world_to_cam
            .transform_point(&prev_center_world);
        let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
            Vec2F64::new(
                camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
                camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
            )
        });

        // Brute-force descriptor matching over the unassociated subsets
        // (global second-best ratio test + orientation consistency live
        // inside the matcher and are essential for match quality).
        let prev_orients: Vec<f32> = prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.orientations[i])
            .collect();
        let prev_descs: Vec<[u8; 32]> = prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.descriptors[i])
            .collect();
        let curr_orients: Vec<f32> = curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.orientations[i])
            .collect();
        let curr_descs: Vec<[u8; 32]> = curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.descriptors[i])
            .collect();

        let sub_matches = match_orb_descriptors(
            &prev_orients,
            &prev_descs,
            &curr_orients,
            &curr_descs,
            match_config,
        );

        // Keep only matches consistent with the pose-derived epipolar
        // geometry: distance from the curr keypoint to its epipolar line
        // must pass the chi-square gate at the octave's detection sigma.
        let mut pair_indices: Vec<(usize, usize)> = Vec::new();
        let mut matched_prev: Vec<Vec2F64> = Vec::new();
        let mut matched_curr: Vec<Vec2F64> = Vec::new();
        for (prev_sub, curr_sub) in sub_matches {
            let (Some(&prev_idx), Some(&curr_idx)) =
                (prev_unassoc.get(prev_sub), curr_unassoc.get(curr_sub))
            else {
                continue;
            };
            let (Some(pu), Some(qu)) = (
                prev_kf.frame.undistorted_xy(prev_idx, camera),
                curr_kf.frame.undistorted_xy(curr_idx, camera),
            ) else {
                continue;
            };
            let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
            let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);

            // Reject curr keypoints near the epipole (radius grows with
            // octave; ORB-SLAM3 uses 100 * scaleFactor^octave px^2).
            if let Some(e) = epipole_px {
                let octave = curr_kf
                    .frame
                    .features
                    .octaves
                    .get(curr_idx)
                    .copied()
                    .unwrap_or(0);
                let dx = q.x - e.x;
                let dy = q.y - e.y;
                if dx * dx + dy * dy < 100.0 * ORB_SCALE_FACTOR.powi(octave as i32) {
                    continue;
                }
            }

            let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
            let line_norm_sq = l.x * l.x + l.y * l.y;
            if line_norm_sq <= 1e-12 {
                continue;
            }
            let d = l.x * q.x + l.y * q.y + l.z;
            let octave = curr_kf
                .frame
                .features
                .octaves
                .get(curr_idx)
                .copied()
                .unwrap_or(0);
            let sigma_sq = ORB_SCALE_FACTOR.powi(2 * octave as i32);
            if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
                continue;
            }

            pair_indices.push((prev_idx, curr_idx));
            matched_prev.push(p);
            matched_curr.push(q);
        }
        if pair_indices.len() < MIN_GROWTH_MATCHES {
            return 0;
        }

        let triangulated = match triangulate_matched_points(
            &matched_prev,
            &matched_curr,
            &prev_kf.frame.pose_world_to_cam,
            &curr_kf.frame.pose_world_to_cam,
            camera,
            triangulation_config,
        ) {
            Ok(pts) => pts,
            Err(_) => return 0,
        };

        let mut points = Vec::new();
        let mut claimed_prev: HashSet<usize> = HashSet::new();
        let mut claimed_curr: HashSet<usize> = HashSet::new();
        for tp in &triangulated {
            let Some(&(prev_idx, curr_idx)) = pair_indices.get(tp.pair_index) else {
                continue;
            };
            if curr_kf.map_point(curr_idx).is_some() {
                continue;
            }
            // The triangulator can emit two points on one feature in either
            // view. A feature holds one landmark, so the first claim wins:
            // publishing both left the loser with an observation record for a
            // feature the keyframe had reassigned to the winner. A refused
            // point reserves neither feature, so it cannot knock out a later
            // point that only shares its other one.
            if claimed_prev.contains(&prev_idx) || claimed_curr.contains(&curr_idx) {
                continue;
            }
            claimed_prev.insert(prev_idx);
            claimed_curr.insert(curr_idx);
            let color = curr_kf
                .frame
                .keypoint_colors
                .get(curr_idx)
                .copied()
                .unwrap_or([128; 3]);
            points.push((
                tp.position,
                curr_kf.frame.features.descriptors[curr_idx],
                color,
                prev_idx,
                curr_idx,
            ));
        }
        points
    };

    // Write phase: create the new map points; curr_kf is registered as
    // the first observer inside add_triangulated_points.
    let first_mp_idx = map.lock().unwrap().num_map_points();
    let added = map
        .lock()
        .unwrap()
        .add_triangulated_points(None, curr_kf, &points);

    // Register the neighbor as a second observer on each new map point.
    // This is the SearchInNeighbors-equivalent piece for the
    // triangulating pair: without it the new point would have a single
    // observation, biasing scale/normal geometry and making the cull
    // overly aggressive.
    for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().take(added).enumerate() {
        let mp_idx = first_mp_idx + i;
        map.lock()
            .unwrap()
            .register_observation_at(mp_idx, prev_kf_idx, prev_desc_idx);
        if let Some(prev_live) = map.lock().unwrap().get_keyframe_mut(prev_kf_idx) {
            prev_live.associate_map_point(prev_desc_idx, mp_idx);
        }
    }

    added
}

/// Forward Fuse pass: project each map point observed by the current KF
/// into every neighbor KF that doesn't already observe it. If the
/// projection lands near an unassociated keypoint with a matching
/// descriptor, register the observation. Mirrors a subset of ORB-SLAM3's
/// `SearchInNeighbors` (forward direction only; we don't yet do duplicate
/// merging or the second-hop covisible expansion).
pub(crate) fn fuse_into_neighbors(
    map: &Arc<Mutex<Map>>,
    camera: &PinholeCamera,
    curr_kf_idx: usize,
    neighbor_kf_indices: &[usize],
) -> usize {
    const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
    const FUSE_MAX_HAMMING: u32 = 50;

    // Collect map points observed by curr_kf. We snapshot the indices
    // here so we can hold no other borrow on the map during the loop.
    let curr_mp_indices: Vec<usize> = match map.lock().unwrap().get_keyframe(curr_kf_idx) {
        Some(kf) => kf
            .map_point_by_desc_idx
            .iter()
            .filter_map(|&mp| mp)
            .collect(),
        None => return 0,
    };
    if curr_mp_indices.is_empty() {
        return 0;
    }

    let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
    let mut n_fused = 0usize;

    for &nb_kf_idx in neighbor_kf_indices {
        if nb_kf_idx == curr_kf_idx {
            continue;
        }

        // Proposals: (kp_idx_in_nb_kf, mp_idx, hamming). Collected under a
        // shared borrow of the neighbor KF in the map (no clone), resolved
        // in the write phase below so a single keypoint can't be claimed
        // by two map points.
        let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
        {
            let map_guard = map.lock().unwrap();
            let Some(nb_kf) = map_guard.get_keyframe(nb_kf_idx) else {
                continue;
            };

            for &mp_idx in &curr_mp_indices {
                let mp = match map_guard.map_points().get(mp_idx) {
                    Some(mp) if !mp.culled => mp,
                    _ => continue,
                };
                // Skip if neighbor already observes this map point.
                if mp.is_observed_by(nb_kf_idx) {
                    continue;
                }

                // Project into the neighbor's frame.
                let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
                if p_cam.z <= 0.0 {
                    continue;
                }
                let Ok(pixel) = camera.project_to_image(&p_cam, 0.0, nb_kf.frame.image_size) else {
                    continue;
                };
                let u = pixel.x as f32;
                let v = pixel.y as f32;

                // Find the closest unassociated keypoint within the radius
                // that matches the map point's representative descriptor.
                let mut best_dist = u32::MAX;
                let mut best_kp = usize::MAX;
                for kp_idx in 0..nb_kf.frame.features.keypoints_xy.len() {
                    if nb_kf.map_point(kp_idx).is_some() {
                        continue;
                    }
                    let Some(kp) = nb_kf.frame.undistorted_xy(kp_idx, camera) else {
                        continue;
                    };
                    let dx = kp[0] - u;
                    let dy = kp[1] - v;
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let dist =
                        hamming_distance(&mp.descriptor, &nb_kf.frame.features.descriptors[kp_idx]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_kp = kp_idx;
                    }
                }

                if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                    proposals.push((best_kp, mp_idx, best_dist));
                }
            }
        }

        // Resolve proposals: if two map points want the same keypoint,
        // the one with the smaller Hamming distance wins. Track which
        // keypoints are already taken in this pass.
        proposals.sort_by_key(|&(_, _, dist)| dist);
        let mut taken_kp: HashSet<usize> = HashSet::new();
        for (kp_idx, mp_idx, _) in proposals {
            if taken_kp.contains(&kp_idx) {
                continue;
            }
            // Re-check that the live neighbor KF hasn't already had this
            // keypoint claimed (e.g. by a prior iteration in this fuse
            // call associating a different mp).
            let already = map
                .lock()
                .unwrap()
                .get_keyframe(nb_kf_idx)
                .and_then(|kf| kf.map_point(kp_idx))
                .is_some();
            if already {
                continue;
            }
            map.lock()
                .unwrap()
                .register_observation_at(mp_idx, nb_kf_idx, kp_idx);
            if let Some(nb_live) = map.lock().unwrap().get_keyframe_mut(nb_kf_idx) {
                nb_live.associate_map_point(kp_idx, mp_idx);
            }
            taken_kp.insert(kp_idx);
            n_fused += 1;
        }
    }

    n_fused
}
