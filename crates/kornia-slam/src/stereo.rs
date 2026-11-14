//! Stereo back-projection. Matching a rectified pair into per-keypoint depth
//! is [`kornia_3d::stereo::StereoMatcher`]'s job.

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;

use crate::frame::Frame;

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
    use kornia_imgproc::features::OrbFeatures;

    #[test]
    fn unproject_stereo_back_projects_valid_depths() {
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
}
