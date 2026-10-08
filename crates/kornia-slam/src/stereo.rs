//! Stereo utilities for kornia-slam.
//!
//! This crate used to ship a local port of ORB-SLAM3's `ComputeStereoMatches`.
//! That implementation now lives upstream as `kornia_3d::stereo::StereoMatcher`
//! (kornia/kornia-rs#1157). kornia-slam depends on the upstream matcher and
//! only keeps SLAM-specific helpers here.

// Re-export upstream types for downstream convenience.
pub use kornia_3d::stereo::{
    MAX_LEVELS, MAX_SAD_RADIUS, SadRefine, StereoDescriptors, StereoKeypoints, StereoMatchConfig,
    StereoMatchError, StereoMatcher, StereoMatches, SubPixelFit,
};

use crate::frame::Frame;
use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;

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
