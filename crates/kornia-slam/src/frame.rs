//! Core frame-domain types shared across the crate.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;

/// `Frame` carries ORB features directly today.
///
/// If kornia-slam grows multiple frontend feature pipelines, this module is the
/// place to introduce a higher-level abstraction above `OrbFeatures`.
/// A camera observation: features extracted at a known pose.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame index.
    pub idx: usize,
    /// Features extracted for this frame.
    pub features: OrbFeatures,
    /// Camera pose in world-to-camera convention.
    pub pose_world_to_cam: Pose3d,
    /// Image dimensions.
    pub image_size: ImageSize,
    /// Pixel color sampled at each keypoint location (one [u8; 3] per keypoint).
    pub keypoint_colors: Vec<[u8; 3]>,
    /// Right-image x-coordinate of the stereo match per left keypoint
    /// (ORB-SLAM3's `mvuRight`); `-1.0` = no stereo match. Parallel to
    /// `features.keypoints_xy`. Empty when the frame is monocular.
    pub u_right: Vec<f32>,
    /// Metric depth per left keypoint (ORB-SLAM3's `mvDepth`); `-1.0` = no
    /// stereo match. Parallel to `features.keypoints_xy`. Empty when monocular.
    pub depth: Vec<f32>,
    /// Undistorted pixel coordinates per keypoint, parallel to
    /// `features.keypoints_xy`; NaN for a keypoint the camera cannot
    /// undistort. Filled once by [`Frame::ensure_undistorted`];
    /// empty until then. Undistortion is iterative and shows up in every
    /// stage that consumes keypoints (tracking, BA gathering, fuse), so it
    /// must not be recomputed per consumer.
    pub keypoints_undist: Vec<[f32; 2]>,
}

impl Frame {
    /// Whether this frame carries per-keypoint stereo depth.
    pub fn is_stereo(&self) -> bool {
        !self.depth.is_empty()
    }

    /// Computes and caches undistorted keypoint coordinates (no-op when
    /// already computed for the current keypoint set).
    pub fn ensure_undistorted(&mut self, camera: &PinholeCamera) {
        if self.keypoints_undist.len() == self.features.keypoints_xy.len() {
            return;
        }
        self.keypoints_undist = self
            .features
            .keypoints_xy
            .iter()
            .map(|kp| undistort_keypoint(camera, kp))
            .collect();
    }

    /// Undistorted coordinates of keypoint `i`, from the cache when present
    /// (frames built outside the pipeline fall back to undistorting on the
    /// fly); `None` when the camera cannot undistort it.
    pub fn undistorted_xy(&self, i: usize, camera: &PinholeCamera) -> Option<[f32; 2]> {
        if !self.keypoints_undist.is_empty() {
            return self.keypoints_undist.get(i).copied().filter(is_finite_xy);
        }
        self.features
            .keypoints_xy
            .get(i)
            .map(|kp| undistort_keypoint(camera, kp))
            .filter(is_finite_xy)
    }

    /// Metric depth for keypoint `i`, if it has a valid stereo match.
    pub fn stereo_depth(&self, i: usize) -> Option<f32> {
        match self.depth.get(i) {
            Some(&z) if z > 0.0 => Some(z),
            _ => None,
        }
    }
}

/// Undistorted pixel of a raw keypoint, NaN where `camera` has no valid
/// preimage for it.
pub(crate) fn undistort_keypoint(camera: &PinholeCamera, kp: &[f32; 2]) -> [f32; 2] {
    camera
        .undistort(kp[0] as f64, kp[1] as f64)
        .map_or([f32::NAN; 2], |p| [p.x as f32, p.y as f32])
}

/// The camera that images undistorted keypoints: `camera`'s intrinsics with
/// zero distortion.
pub(crate) fn undistorted_camera(camera: &PinholeCamera) -> PinholeCamera {
    PinholeCamera {
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
        ..camera.clone()
    }
}

/// Whether an undistortion-cache entry holds a usable pixel.
pub(crate) fn is_finite_xy(xy: &[f32; 2]) -> bool {
    xy[0].is_finite() && xy[1].is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_imgproc::features::OrbFeatures;

    fn empty_frame(u_right: Vec<f32>, depth: Vec<f32>) -> Frame {
        Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 1,
                height: 1,
            },
            keypoint_colors: Vec::new(),
            u_right,
            depth,
            keypoints_undist: Vec::new(),
        }
    }

    #[test]
    fn monocular_frame_reports_no_stereo() {
        let frame = empty_frame(Vec::new(), Vec::new());
        assert!(!frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), None);
    }

    #[test]
    fn a_keypoint_the_camera_cannot_undistort_keeps_its_slot_but_has_no_pixel() {
        // k1 = -0.5 folds the radial map, so (600, 240) has no preimage.
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.5,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let mut frame = empty_frame(Vec::new(), Vec::new());
        frame.features.keypoints_xy = vec![[330.0, 250.0], [600.0, 240.0], [310.0, 230.0]];

        let on_the_fly: Vec<_> = (0..3).map(|i| frame.undistorted_xy(i, &camera)).collect();
        frame.ensure_undistorted(&camera);
        let cached: Vec<_> = (0..3).map(|i| frame.undistorted_xy(i, &camera)).collect();

        assert_eq!(frame.keypoints_undist.len(), 3);
        assert_eq!(cached, on_the_fly);
        assert!(cached[0].is_some() && cached[2].is_some());
        assert_eq!(cached[1], None);
    }

    #[test]
    fn stereo_frame_reports_valid_depth() {
        let frame = empty_frame(vec![10.0, -1.0], vec![2.5, -1.0]);
        assert!(frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), Some(2.5));
        assert_eq!(frame.stereo_depth(1), None); // sentinel
        assert_eq!(frame.stereo_depth(2), None); // out of range
    }
}
