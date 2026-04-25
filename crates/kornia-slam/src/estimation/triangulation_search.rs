//! BoW-accelerated stereo search for new-map-point triangulation.
//!
//! Port of ORB-SLAM3's `ORBmatcher::SearchForTriangulation` (ORBmatcher.cc:907).
//! Given two keyframes with BoW direct indices, walks matching tree nodes and
//! returns (idx1, idx2) pairs of *unmatched* features that satisfy a hamming
//! threshold, an epipolar-line distance check, and an orientation-histogram
//! consistency test.
//!
//! The matches returned here are the input to `triangulate_matched_points`
//! during continuous map growth (`LocalMapping::CreateNewMapPoints` equivalent).

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::{Mat3F64, Vec3F64};

use crate::map::Keyframe;

/// Tunables for [`search_for_triangulation`].
#[derive(Debug, Clone, Copy)]
pub struct TriangulationSearchConfig {
    /// Max accepted ORB descriptor hamming distance (ORB-SLAM3 `TH_LOW`).
    pub max_hamming_distance: u32,
    /// Enable orientation-histogram filtering.
    pub check_orientation: bool,
    /// Number of bins in the orientation histogram (ORB-SLAM3 `HISTO_LENGTH`).
    pub histogram_bins: usize,
    /// Skip matches whose epipole-to-kp2 distance² is below `margin_factor *
    /// scale_factor[octave]` (ORB-SLAM3: 100).
    pub epipole_margin_factor: f32,
    /// Chi² scaling for the epipolar-line distance check (ORB-SLAM3: 3.84).
    pub epipolar_chi2: f32,
}

impl Default for TriangulationSearchConfig {
    fn default() -> Self {
        Self {
            max_hamming_distance: 50,
            check_orientation: true,
            histogram_bins: 30,
            epipole_margin_factor: 100.0,
            epipolar_chi2: 3.84,
        }
    }
}

/// Computes hamming distance between two packed 32-byte ORB descriptors.
#[inline]
fn hamming_u8(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut d = 0u32;
    for i in 0..32 {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

/// Computes the fundamental matrix F21 such that `x2^T F21 x1 = 0`.
///
/// Matches the form used by ORB-SLAM3's `Pinhole::epipolarConstrain`, where
/// the epipolar line in image 2 is `l2 = F12^T x1` and residual is the
/// point-to-line distance of `x2` to `l2`. We compute `F12` under the
/// convention `F12 = K1^{-T} [t12]_x R12 K2^{-1}` with `T12 = T1w * Tw2`.
fn fundamental_12(
    t1w: &kornia_3d::pose::Pose3d,
    t2w: &kornia_3d::pose::Pose3d,
    k: &PinholeCamera,
) -> (Mat3F64, Vec3F64) {
    // T12 = T1w * T2w.inverse() → R12 = R1w * R2w^T, t12 = t1w - R12 * t2w
    let r12 = t1w.rotation * t2w.rotation.transpose();
    let t12 = t1w.translation - r12 * t2w.translation;
    let tx = Mat3F64::from_cols(
        Vec3F64::new(0.0, t12.z, -t12.y),
        Vec3F64::new(-t12.z, 0.0, t12.x),
        Vec3F64::new(t12.y, -t12.x, 0.0),
    );
    let k_mat = k.intrinsic_matrix();
    let k_inv = Mat3F64::from_cols(
        Vec3F64::new(1.0 / k.fx, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0 / k.fy, 0.0),
        Vec3F64::new(-k.cx / k.fx, -k.cy / k.fy, 1.0),
    );
    let k_inv_t = k_mat.transpose();
    // K^{-T} computed as (K^{-1})^T
    let k_inv_t_ = Mat3F64::from_cols(
        Vec3F64::new(1.0 / k.fx, 0.0, -k.cx / k.fx),
        Vec3F64::new(0.0, 1.0 / k.fy, -k.cy / k.fy),
        Vec3F64::new(0.0, 0.0, 1.0),
    );
    let _ = (k_inv, k_inv_t);
    let f12 = k_inv_t_ * tx * r12 * {
        // K^{-1}
        Mat3F64::from_cols(
            Vec3F64::new(1.0 / k.fx, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0 / k.fy, 0.0),
            Vec3F64::new(-k.cx / k.fx, -k.cy / k.fy, 1.0),
        )
    };
    // Projected epipole in image 2: e2 = K * (T2w * C1_world) where C1_world = -R1w^T t1w.
    let c1_world = -(t1w.rotation.transpose() * t1w.translation);
    let c1_in_cam2 = t2w.rotation * c1_world + t2w.translation;
    let _ = c1_in_cam2;
    (f12, t12)
}

/// Returns point-to-epipolar-line squared distance in image 2.
///
/// `l2 = [a, b, c] = x1^T F12` where `x1 = (u1, v1, 1)`. The squared pixel
/// distance of `x2 = (u2, v2)` to line `l2` is `(a u2 + b v2 + c)^2 / (a^2 + b^2)`.
#[inline]
fn epipolar_line_distance_sq(f12: &Mat3F64, u1: f32, v1: f32, u2: f32, v2: f32) -> Option<f32> {
    let u1 = u1 as f64;
    let v1 = v1 as f64;
    // row = x1^T F12 (treating F12 as column-major; use explicit indexing)
    let a = u1 * f12.x_axis.x + v1 * f12.x_axis.y + f12.x_axis.z;
    let b = u1 * f12.y_axis.x + v1 * f12.y_axis.y + f12.y_axis.z;
    let c = u1 * f12.z_axis.x + v1 * f12.z_axis.y + f12.z_axis.z;
    let num = a * u2 as f64 + b * v2 as f64 + c;
    let den = a * a + b * b;
    if den < 1e-12 {
        return None;
    }
    Some(((num * num) / den) as f32)
}

/// Projects keyframe-1's camera center into keyframe-2's image.
fn project_epipole_into_kf2(
    t1w: &kornia_3d::pose::Pose3d,
    t2w: &kornia_3d::pose::Pose3d,
    k: &PinholeCamera,
) -> Option<[f32; 2]> {
    let c1_world = -(t1w.rotation.transpose() * t1w.translation);
    let c1_in_cam2 = t2w.rotation * c1_world + t2w.translation;
    if c1_in_cam2.z <= 1e-6 {
        return None;
    }
    let u = k.fx * c1_in_cam2.x / c1_in_cam2.z + k.cx;
    let v = k.fy * c1_in_cam2.y / c1_in_cam2.z + k.cy;
    Some([u as f32, v as f32])
}

/// Returns candidate `(idx1, idx2)` pairs suitable for triangulation.
///
/// Walks the BoW direct indices of both keyframes (same-node intersection),
/// skips features already owning a map point, enforces a hamming threshold,
/// checks epipole margin + epipolar-line distance, and applies orientation
/// histogram filtering.
/// Diagnostic counters returned alongside the matched pairs.
#[derive(Debug, Default, Clone, Copy)]
pub struct TriangulationSearchStats {
    pub node_intersections: usize,
    pub candidate_feats_kf1: usize,
    pub candidate_feats_kf2: usize,
    pub rejected_already_matched: usize,
    pub rejected_hamming: usize,
    pub rejected_epipole_margin: usize,
    pub rejected_epipolar_line: usize,
    pub rejected_degenerate_line: usize,
    pub accepted_before_orientation: usize,
    pub dropped_by_orientation: usize,
}

pub fn search_for_triangulation(
    kf1: &Keyframe,
    kf2: &Keyframe,
    camera: &PinholeCamera,
    config: &TriangulationSearchConfig,
) -> Vec<(usize, usize)> {
    search_for_triangulation_with_stats(kf1, kf2, camera, config).0
}

pub fn search_for_triangulation_with_stats(
    kf1: &Keyframe,
    kf2: &Keyframe,
    camera: &PinholeCamera,
    config: &TriangulationSearchConfig,
) -> (Vec<(usize, usize)>, TriangulationSearchStats) {
    let mut stats = TriangulationSearchStats::default();
    let Some(di1) = kf1.direct_index.as_ref() else {
        return (Vec::new(), stats);
    };
    let Some(di2) = kf2.direct_index.as_ref() else {
        return (Vec::new(), stats);
    };

    let t1w = &kf1.frame.pose_world_to_cam;
    let t2w = &kf2.frame.pose_world_to_cam;

    let (f12, _t12) = fundamental_12(t1w, t2w, camera);

    let Some(epipole2) = project_epipole_into_kf2(t1w, t2w, camera) else {
        return (Vec::new(), stats);
    };

    let n1 = kf1.frame.features.descriptors.len();
    let n2 = kf2.frame.features.descriptors.len();
    let mut matched2 = vec![false; n2];
    let mut matches_12: Vec<i32> = vec![-1; n1];
    let mut match_rot_bin: Vec<i16> = vec![-1; n1];

    let factor = 1.0f32 / (config.histogram_bins as f32);

    // Walk both sorted direct indices keeping node IDs aligned.
    let a = &di1.0;
    let b = &di2.0;
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (node_a, feats_a) = &a[i];
        let (node_b, feats_b) = &b[j];
        use std::cmp::Ordering;
        match node_a.cmp(node_b) {
            Ordering::Less => {
                // advance i to the first node >= node_b
                match a[i..].binary_search_by_key(node_b, |(n, _)| *n) {
                    Ok(off) => i += off,
                    Err(off) => i += off,
                }
            }
            Ordering::Greater => match b[j..].binary_search_by_key(node_a, |(n, _)| *n) {
                Ok(off) => j += off,
                Err(off) => j += off,
            },
            Ordering::Equal => {
                stats.node_intersections += 1;
                stats.candidate_feats_kf1 += feats_a.len();
                stats.candidate_feats_kf2 += feats_b.len();
                for &fa in feats_a.iter() {
                    let idx1 = fa as usize;
                    if kf1.map_point(idx1).is_some() {
                        stats.rejected_already_matched += 1;
                        continue;
                    }
                    let d1 = &kf1.frame.features.descriptors[idx1];
                    let kp1 = kf1.frame.features.keypoints_xy[idx1];
                    let ang1 = kf1.frame.features.orientations[idx1];

                    let mut best_dist = config.max_hamming_distance + 1;
                    let mut best_idx2: i32 = -1;

                    for &fb in feats_b.iter() {
                        let idx2 = fb as usize;
                        if matched2[idx2] || kf2.map_point(idx2).is_some() {
                            stats.rejected_already_matched += 1;
                            continue;
                        }
                        let d2 = &kf2.frame.features.descriptors[idx2];
                        let dist = hamming_u8(d1, d2);
                        if dist > config.max_hamming_distance || dist >= best_dist {
                            stats.rejected_hamming += 1;
                            continue;
                        }

                        let kp2 = kf2.frame.features.keypoints_xy[idx2];
                        let scale2 = kf2.frame.features.scales[idx2];

                        // Epipole margin: drop matches whose kp2 is too close to the epipole.
                        let dex = epipole2[0] - kp2[0];
                        let dey = epipole2[1] - kp2[1];
                        if dex * dex + dey * dey < config.epipole_margin_factor * scale2 {
                            stats.rejected_epipole_margin += 1;
                            continue;
                        }

                        // Epipolar line distance in image 2 (units: px²).
                        let Some(dsq) =
                            epipolar_line_distance_sq(&f12, kp1[0], kp1[1], kp2[0], kp2[1])
                        else {
                            stats.rejected_degenerate_line += 1;
                            continue;
                        };
                        // ORB-SLAM3 threshold: dsq < 3.84 * sigmaLevel2(kp2.octave)
                        // sigmaLevel2 = scaleFactor^2.
                        if dsq >= config.epipolar_chi2 * scale2 * scale2 {
                            stats.rejected_epipolar_line += 1;
                            continue;
                        }

                        best_dist = dist;
                        best_idx2 = idx2 as i32;
                    }

                    if best_idx2 >= 0 {
                        let idx2 = best_idx2 as usize;
                        matches_12[idx1] = best_idx2;
                        matched2[idx2] = true;

                        if config.check_orientation {
                            let ang2 = kf2.frame.features.orientations[idx2];
                            // ORB-SLAM3 uses degrees and keypoint angle in [0, 360).
                            let mut rot = (ang1 - ang2).to_degrees();
                            if rot < 0.0 {
                                rot += 360.0;
                            }
                            let mut bin = (rot * factor).round() as i32;
                            if bin == config.histogram_bins as i32 {
                                bin = 0;
                            }
                            if bin >= 0 && (bin as usize) < config.histogram_bins {
                                match_rot_bin[idx1] = bin as i16;
                            }
                        }
                    }
                }
                i += 1;
                j += 1;
            }
        }
    }

    // Orientation-histogram filter: keep only the three most populated bins,
    // and drop any secondary bin whose mass is < 10% of the max.
    if config.check_orientation {
        let mut counts = vec![0i32; config.histogram_bins];
        for &b in &match_rot_bin {
            if b >= 0 {
                counts[b as usize] += 1;
            }
        }
        let (ind1, ind2, ind3) = three_maxima(&counts);

        for (idx1, &b) in match_rot_bin.iter().enumerate() {
            if b < 0 {
                continue;
            }
            let bi = b as i32;
            if bi != ind1 && bi != ind2 && bi != ind3 {
                matches_12[idx1] = -1;
            }
        }
    }

    let mut out = Vec::new();
    for (idx1, &m) in matches_12.iter().enumerate() {
        if m >= 0 {
            out.push((idx1, m as usize));
        }
    }
    (out, stats)
}

/// Returns the three histogram bins with the largest counts. Bins 2 and 3 are
/// suppressed (returned as -1) when their mass falls below 10% of bin 1's mass.
fn three_maxima(counts: &[i32]) -> (i32, i32, i32) {
    let mut max1 = 0i32;
    let mut max2 = 0i32;
    let mut max3 = 0i32;
    let mut ind1: i32 = -1;
    let mut ind2: i32 = -1;
    let mut ind3: i32 = -1;
    for (i, &s) in counts.iter().enumerate() {
        let i = i as i32;
        if s > max1 {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if s > max2 {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        } else if s > max3 {
            max3 = s;
            ind3 = i;
        }
    }
    if (max2 as f32) < 0.1 * (max1 as f32) {
        ind2 = -1;
        ind3 = -1;
    } else if (max3 as f32) < 0.1 * (max1 as f32) {
        ind3 = -1;
    }
    (ind1, ind2, ind3)
}
