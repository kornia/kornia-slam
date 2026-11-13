//! Stereo correspondence: per-keypoint disparity and metric depth from a
//! rectified, row-aligned stereo pair.
//!
//! Full-parity port of ORB-SLAM3's `Frame::ComputeStereoMatches`
//! (`extra/ORB_SLAM3/src/Frame.cc`). For every left keypoint it searches the
//! same image row of the right view for the best ORB descriptor match within
//! the valid disparity band, then refines the disparity to sub-pixel accuracy
//! with a sliding-window SAD correlation and a parabola fit. Matches whose
//! correlation error is a gross outlier (relative to the median) are dropped.
//!
//! ```text
//!   left kp (uL, vL)            right view, same row vL
//!        |                  .---------------------------.
//!        v                  |   *      * <- candidates  |
//!   uR in [uL-maxD, uL]     '---------------------------'
//!        |                          |
//!        +--- best ORB desc ---> uR0  --- SAD subpixel ---> bestuR
//!                                        |
//!                          disparity = uL - bestuR
//!                          depth     = bf / disparity
//! ```
//!
//! The caller supplies the left/right image pyramids (one [`Image`] per ORB
//! octave, matching the octaves recorded in the [`OrbFeatures`]); building a
//! scale-consistent pyramid for a specific dataset is the caller's job.

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_imgproc::features::{OrbFeatures, hamming_distance};

use crate::frame::Frame;

/// ORB descriptor distance upper bound (ORB-SLAM3 `ORBmatcher::TH_HIGH`).
pub const TH_HIGH: u32 = 100;
/// ORB descriptor distance lower bound (ORB-SLAM3 `ORBmatcher::TH_LOW`).
pub const TH_LOW: u32 = 50;

/// Configuration for [`compute_stereo_matches`].
#[derive(Debug, Clone)]
pub struct StereoMatchConfig {
    /// Stereo baseline times focal length `fx` (ORB-SLAM3's `mbf`), in
    /// pixel·metre units. `depth = bf / disparity`.
    pub bf: f32,
    /// Metric stereo baseline (ORB-SLAM3's `mb = bf / fx`). Sets the minimum
    /// usable depth `minZ = baseline`, hence the maximum disparity.
    pub baseline: f32,
    /// Per-octave scale multiplier `scale_factor^octave` (ORB-SLAM3's
    /// `mvScaleFactors`). Length must cover every octave in the features.
    pub scale_factors: Vec<f32>,
    /// Per-octave inverse scale `1 / scale_factor^octave` (ORB-SLAM3's
    /// `mvInvScaleFactors`). Same length as [`Self::scale_factors`].
    pub inv_scale_factors: Vec<f32>,
    /// Descriptor distance below which a coarse match is refined to sub-pixel.
    /// ORB-SLAM3 uses `(TH_HIGH + TH_LOW) / 2`.
    pub th_orb_dist: u32,
    /// Descriptor distance ceiling for the coarse row search
    /// (ORB-SLAM3's `TH_HIGH`).
    pub th_high: u32,
}

impl StereoMatchConfig {
    /// Builds a config from the metric baseline, focal length, and ORB pyramid
    /// geometry, deriving the per-octave scale tables and ORB-SLAM3's default
    /// descriptor thresholds.
    pub fn new(baseline: f32, fx: f32, scale_factor: f32, n_levels: usize) -> Self {
        let scale_factors: Vec<f32> = (0..n_levels).map(|o| scale_factor.powi(o as i32)).collect();
        let inv_scale_factors: Vec<f32> = scale_factors.iter().map(|s| 1.0 / s).collect();
        Self {
            bf: baseline * fx,
            baseline,
            scale_factors,
            inv_scale_factors,
            th_orb_dist: (TH_HIGH + TH_LOW) / 2,
            th_high: TH_HIGH,
        }
    }
}

/// Per-left-keypoint stereo result, parallel to the left [`OrbFeatures`].
///
/// `u_right[i]` / `depth[i]` are `-1.0` when keypoint `i` has no stereo match,
/// mirroring ORB-SLAM3's `mvuRight` / `mvDepth` sentinels.
#[derive(Debug, Clone, Default)]
pub struct StereoMatches {
    /// Matched x-coordinate in the right image (full resolution).
    pub u_right: Vec<f32>,
    /// Metric depth in the left camera frame.
    pub depth: Vec<f32>,
}

impl StereoMatches {
    /// Number of left keypoints with a valid stereo match.
    pub fn num_matched(&self) -> usize {
        self.depth.iter().filter(|&&d| d > 0.0).count()
    }
}

/// Half-width of the SAD correlation patch (ORB-SLAM3's `w`); patch is
/// `(2w+1) x (2w+1)`.
const SAD_HALF_WINDOW: i32 = 5;
/// Sliding-search half-range in pixels around the coarse match (ORB-SLAM3's
/// `L`).
const SAD_SEARCH_RANGE: i32 = 5;

/// Computes stereo correspondences for a rectified pair.
///
/// `left_pyramid` / `right_pyramid` hold one image per ORB octave (index =
/// octave); only level 0 is required if all keypoints are at octave 0. `left`
/// and `right` are the ORB features of the respective views. Returns a
/// [`StereoMatches`] of length `left.len()`.
pub fn compute_stereo_matches(
    left_pyramid: &[Image<u8, 1>],
    right_pyramid: &[Image<u8, 1>],
    left: &OrbFeatures,
    right: &OrbFeatures,
    cfg: &StereoMatchConfig,
) -> StereoMatches {
    let n_left = left.keypoints_xy.len();
    let mut u_right = vec![-1.0f32; n_left];
    let mut depth = vec![-1.0f32; n_left];

    let (Some(left_l0), Some(_)) = (left_pyramid.first(), right_pyramid.first()) else {
        return StereoMatches { u_right, depth };
    };
    let n_rows = left_l0.height();
    let n_right = right.keypoints_xy.len();
    if n_rows == 0 || n_right == 0 || n_left == 0 {
        return StereoMatches { u_right, depth };
    }

    // Row table: each right keypoint votes for the rows it could match, with a
    // band scaled by its octave (ORB-SLAM3 uses 2 * scale_factor[octave]).
    let mut row_indices: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
    for ir in 0..n_right {
        let kp_y = right.keypoints_xy[ir][1];
        let octave = right.octaves[ir] as usize;
        let r = 2.0 * cfg.scale_factors.get(octave).copied().unwrap_or(1.0);
        let min_r = ((kp_y - r).floor() as i32).max(0);
        let max_r = ((kp_y + r).ceil() as i32).min(n_rows as i32 - 1);
        for yi in min_r..=max_r {
            row_indices[yi as usize].push(ir);
        }
    }

    // Disparity search limits.
    let min_z = cfg.baseline;
    let min_d = 0.0f32;
    let max_d = if min_z > 0.0 { cfg.bf / min_z } else { 0.0 };

    // (sad_dist, left_idx) for accepted matches, used for the median reject.
    let mut dist_idx: Vec<(i64, usize)> = Vec::with_capacity(n_left);

    for il in 0..n_left {
        let u_l = left.keypoints_xy[il][0];
        let v_l = left.keypoints_xy[il][1];
        let level_l = left.octaves[il] as i32;

        let row = (v_l as i32).clamp(0, n_rows as i32 - 1) as usize;
        let candidates = &row_indices[row];
        if candidates.is_empty() {
            continue;
        }

        let min_u = u_l - max_d;
        let max_u = u_l - min_d;
        if max_u < 0.0 {
            continue;
        }

        // Coarse: best ORB descriptor match among same-row, near-octave
        // candidates within the disparity band.
        let mut best_dist = cfg.th_high;
        let mut best_ir: Option<usize> = None;
        let dl = &left.descriptors[il];
        for &ir in candidates {
            let octave_r = right.octaves[ir] as i32;
            if octave_r < level_l - 1 || octave_r > level_l + 1 {
                continue;
            }
            let u_r = right.keypoints_xy[ir][0];
            if u_r >= min_u && u_r <= max_u {
                let dist = hamming_distance(dl, &right.descriptors[ir]);
                if dist < best_dist {
                    best_dist = dist;
                    best_ir = Some(ir);
                }
            }
        }

        let Some(best_ir) = best_ir else { continue };
        if best_dist >= cfg.th_orb_dist {
            continue;
        }

        // Sub-pixel: sliding-window SAD at the keypoint's octave level.
        let octave = level_l.max(0) as usize;
        let Some(left_img) = left_pyramid.get(octave) else {
            continue;
        };
        let Some(right_img) = right_pyramid.get(octave) else {
            continue;
        };
        let inv_scale = cfg.inv_scale_factors.get(octave).copied().unwrap_or(1.0);

        let u_r0 = right.keypoints_xy[best_ir][0];
        let scaled_u_l = (u_l * inv_scale).round() as i32;
        let scaled_v_l = (v_l * inv_scale).round() as i32;
        let scaled_u_r0 = (u_r0 * inv_scale).round() as i32;

        let w = SAD_HALF_WINDOW;
        let l = SAD_SEARCH_RANGE;

        // Left patch must fit inside the octave image.
        let lw = left_img.width() as i32;
        let lh = left_img.height() as i32;
        if scaled_u_l - w < 0 || scaled_u_l + w >= lw || scaled_v_l - w < 0 || scaled_v_l + w >= lh
        {
            continue;
        }

        // Right search window column bounds (ORB-SLAM3's iniu/endu guard).
        let rw = right_img.width() as i32;
        let rh = right_img.height() as i32;
        let ini_u = scaled_u_r0 + l - w;
        let end_u = scaled_u_r0 + l + w + 1;
        if ini_u < 0 || end_u >= rw {
            continue;
        }
        if scaled_v_l - w < 0 || scaled_v_l + w >= rh {
            continue;
        }

        // Cache the left patch (row-major, (2w+1)^2 samples).
        let left_slice = left_img.as_slice();
        let patch_dim = (2 * w + 1) as usize;
        let mut left_patch = vec![0i32; patch_dim * patch_dim];
        for dy in -w..=w {
            let row_off = ((scaled_v_l + dy) as usize) * left_img.width();
            for dx in -w..=w {
                let idx = ((dy + w) as usize) * patch_dim + (dx + w) as usize;
                left_patch[idx] = left_slice[row_off + (scaled_u_l + dx) as usize] as i32;
            }
        }

        let right_slice = right_img.as_slice();
        let n_dists = (2 * l + 1) as usize;
        let mut dists = vec![0i64; n_dists];
        let mut best_sad = i64::MAX;
        let mut best_inc_r = 0i32;
        for inc_r in -l..=l {
            let mut sad = 0i64;
            for dy in -w..=w {
                let row_off = ((scaled_v_l + dy) as usize) * right_img.width();
                let lp_row = ((dy + w) as usize) * patch_dim;
                for dx in -w..=w {
                    let rc = scaled_u_r0 + inc_r + dx;
                    let rv = right_slice[row_off + rc as usize] as i32;
                    let lv = left_patch[lp_row + (dx + w) as usize];
                    sad += (lv - rv).unsigned_abs() as i64;
                }
            }
            dists[(l + inc_r) as usize] = sad;
            if sad < best_sad {
                best_sad = sad;
                best_inc_r = inc_r;
            }
        }

        // Reject optima at the search-window edge (no neighbours to fit).
        if best_inc_r == -l || best_inc_r == l {
            continue;
        }

        // Parabola fit on the three SADs around the optimum.
        let idx = (l + best_inc_r) as usize;
        let dist1 = dists[idx - 1] as f32;
        let dist2 = dists[idx] as f32;
        let dist3 = dists[idx + 1] as f32;
        let denom = 2.0 * (dist1 + dist3 - 2.0 * dist2);
        if denom == 0.0 {
            continue;
        }
        let delta_r = (dist1 - dist3) / denom;
        if !(-1.0..=1.0).contains(&delta_r) {
            continue;
        }

        // Re-scale the refined right coordinate back to full resolution.
        let scale = cfg.scale_factors.get(octave).copied().unwrap_or(1.0);
        let mut best_u_r = scale * (scaled_u_r0 as f32 + best_inc_r as f32 + delta_r);

        let mut disparity = u_l - best_u_r;
        if disparity >= min_d && disparity < max_d {
            if disparity <= 0.0 {
                disparity = 0.01;
                best_u_r = u_l - 0.01;
            }
            depth[il] = cfg.bf / disparity;
            u_right[il] = best_u_r;
            dist_idx.push((best_sad, il));
        }
    }

    // Median SAD outlier reject (ORB-SLAM3: drop matches above 1.5*1.4*median).
    if !dist_idx.is_empty() {
        dist_idx.sort_unstable();
        let median = dist_idx[dist_idx.len() / 2].0 as f32;
        let th_dist = 1.5 * 1.4 * median;
        for &(sad, il) in dist_idx.iter().rev() {
            if (sad as f32) < th_dist {
                break;
            }
            u_right[il] = -1.0;
            depth[il] = -1.0;
        }
    }

    StereoMatches { u_right, depth }
}

/// Back-projects every keypoint with a valid stereo depth into the camera
/// frame (ORB-SLAM3's `Frame::UnprojectStereo`), returning
/// `(keypoint_idx, point_in_camera_frame)`.
///
/// `camera` must be the rectified, zero-distortion camera the depths were
/// computed against; keypoint coordinates are taken as-is (already undistorted
/// for a rectified frame).
pub fn unproject_stereo(frame: &Frame, camera: &PinholeCamera) -> Vec<(usize, Vec3F64)> {
    let mut out = Vec::new();
    for (i, kp) in frame.features.keypoints_xy.iter().enumerate() {
        if let Some(z) = frame.stereo_depth(i) {
            let z = z as f64;
            let x = (kp[0] as f64 - camera.cx) / camera.fx * z;
            let y = (kp[1] as f64 - camera.cy) / camera.fy * z;
            out.push((i, Vec3F64::new(x, y, z)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbDetector;

    const ORB_SCALE_FACTOR: f32 = 1.2;
    const ORB_N_LEVELS: usize = 8;

    /// Deterministic high-frequency texture so FAST finds plenty of corners.
    fn noise(x: usize, y: usize) -> u8 {
        let h = (x as u32)
            .wrapping_mul(374_761_393)
            .wrapping_add((y as u32).wrapping_mul(668_265_263));
        let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        (h >> 16) as u8
    }

    /// Small independent per-view dither so the SAD correlation error is
    /// strictly positive (an exact-shift pair has zero SAD at the true
    /// disparity, which would make the median outlier reject discard
    /// everything — a degeneracy real imagery never exhibits).
    fn dither(seed: u32, x: usize, y: usize) -> i32 {
        let h = (x as u32)
            .wrapping_mul(2_246_822_519)
            .wrapping_add((y as u32).wrapping_mul(3_266_489_917))
            .wrapping_add(seed.wrapping_mul(668_265_263));
        let h = h ^ (h >> 15);
        (h % 11) as i32 - 5 // [-5, 5]
    }

    /// Builds a left image and a right image that is the left shifted left by
    /// `disparity` px (so a feature at column `c` in left lands at `c-disparity`
    /// in right, i.e. positive disparity / finite depth), each with a small,
    /// independent dither.
    fn synthetic_pair(
        width: usize,
        height: usize,
        disparity: usize,
    ) -> (Image<u8, 1>, Image<u8, 1>) {
        let mut left = vec![0u8; width * height];
        let mut right = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                left[y * width + x] = (noise(x, y) as i32 + dither(1, x, y)).clamp(0, 255) as u8;
                // right[x] = left-structure[x + disparity] + independent dither
                let sx = x + disparity;
                let base = if sx < width { noise(sx, y) as i32 } else { 0 };
                right[y * width + x] = (base + dither(2, x, y)).clamp(0, 255) as u8;
            }
        }
        let size = ImageSize { width, height };
        (
            Image::from_size_slice(size, &left).unwrap(),
            Image::from_size_slice(size, &right).unwrap(),
        )
    }

    #[test]
    fn recovers_depth_from_known_disparity() {
        let (width, height, disparity) = (320usize, 240usize, 8usize);
        let (left_img, right_img) = synthetic_pair(width, height, disparity);

        let detector = OrbDetector {
            n_keypoints: 800,
            ..Default::default()
        };
        let left = detector.detect_and_extract_u8(&left_img).unwrap();
        let right = detector.detect_and_extract_u8(&right_img).unwrap();
        assert!(left.keypoints_xy.len() > 50, "too few left keypoints");

        // Pyramids: only level 0 supplied; octave>0 keypoints are skipped.
        let left_pyr = [left_img];
        let right_pyr = [right_img];

        let baseline = 0.1f32;
        let fx = 200.0f32;
        let cfg = StereoMatchConfig::new(baseline, fx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        let expected_depth = cfg.bf / disparity as f32; // 20 / 8 = 2.5

        let result = compute_stereo_matches(&left_pyr, &right_pyr, &left, &right, &cfg);
        assert_eq!(result.depth.len(), left.keypoints_xy.len());

        let mut depths: Vec<f32> = result.depth.iter().copied().filter(|&d| d > 0.0).collect();
        assert!(
            depths.len() >= 20,
            "expected many stereo matches, got {}",
            depths.len()
        );

        depths.sort_by(|a, b| a.total_cmp(b));
        let median = depths[depths.len() / 2];
        assert!(
            (median - expected_depth).abs() < 0.1,
            "median depth {median} not near expected {expected_depth}"
        );

        // u_right should equal uL - disparity for matched points (within ~1px).
        for (il, &ur) in result.u_right.iter().enumerate() {
            if ur > 0.0 {
                let ul = left.keypoints_xy[il][0];
                assert!(
                    (ul - ur - disparity as f32).abs() < 1.5,
                    "disparity off: uL={ul} uR={ur}"
                );
            }
        }
    }

    #[test]
    fn unproject_stereo_back_projects_valid_depths() {
        use crate::frame::Frame;
        use kornia_3d::camera::PinholeCamera;
        use kornia_image::ImageSize;

        let camera = PinholeCamera {
            fx: 100.0,
            fy: 100.0,
            cx: 0.0,
            cy: 0.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let frame = Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: vec![[100.0, 0.0], [0.0, 200.0], [50.0, 50.0]],
                orientations: vec![0.0; 3],
                descriptors: vec![[0u8; 32]; 3],
                octaves: vec![0; 3],
            },
            pose_world_to_cam: kornia_3d::pose::Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; 3],
            u_right: vec![95.0, -1.0, 45.0],
            depth: vec![5.0, -1.0, 2.0],
            keypoints_undist: Vec::new(),
            rig: Default::default(),
        };

        let pts = unproject_stereo(&frame, &camera);
        // Keypoint 1 has sentinel depth and is skipped.
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].0, 0);
        let p0 = pts[0].1;
        assert!((p0.x - 5.0).abs() < 1e-9 && p0.y.abs() < 1e-9 && (p0.z - 5.0).abs() < 1e-9);
        let p2 = pts[1].1;
        assert_eq!(pts[1].0, 2);
        assert!(
            (p2.z - 2.0).abs() < 1e-9 && (p2.x - 1.0).abs() < 1e-9 && (p2.y - 1.0).abs() < 1e-9
        );
    }

    #[test]
    fn no_right_keypoints_yields_no_matches() {
        let (left_img, right_img) = synthetic_pair(160, 120, 6);
        let detector = OrbDetector {
            n_keypoints: 300,
            ..Default::default()
        };
        let left = detector.detect_and_extract_u8(&left_img).unwrap();
        let right = OrbFeatures {
            keypoints_xy: Vec::new(),
            orientations: Vec::new(),
            descriptors: Vec::new(),
            octaves: Vec::new(),
        };

        let left_pyr = [left_img];
        let right_pyr = [right_img];
        let cfg = StereoMatchConfig::new(0.1, 200.0, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let result = compute_stereo_matches(&left_pyr, &right_pyr, &left, &right, &cfg);
        assert_eq!(result.num_matched(), 0);
        assert_eq!(result.depth.len(), left.keypoints_xy.len());
        assert!(result.depth.iter().all(|&d| d < 0.0));
    }
}
