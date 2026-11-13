//! Core frame-domain types shared across the crate.

use std::ops::Range;
use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;

use crate::rig::{RigGeometry, RigView};

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
    /// `features.keypoints_xy`. Filled once by [`Frame::ensure_undistorted`];
    /// empty until then. Undistortion is iterative and shows up in every
    /// stage that consumes keypoints (tracking, BA gathering, fuse), so it
    /// must not be recomputed per consumer.
    pub keypoints_undist: Vec<[f32; 2]>,
    /// Per-camera layout of a multi-camera rig frame; empty otherwise.
    pub rig: RigFeatures,
}

/// How a multi-camera rig frame's features split over its cameras
/// (ORB-SLAM3's `Nleft`/`Nright`), and the overlap matches between the first two.
/// `image_size` and `pose_world_to_cam` of the frame are the primary camera's.
#[derive(Debug, Clone, Default)]
pub struct RigFeatures {
    /// The rig the features were extracted on; `None` outside a rig.
    pub geometry: Option<Arc<RigGeometry>>,
    /// Feature index range of each camera, in rig order, covering all features.
    pub camera_ranges: Vec<Range<usize>>,
    /// Per primary-camera feature: its match among camera 1's features
    /// (ORB-SLAM3's `mvLeftToRightMatch`).
    pub left_to_right: Vec<Option<usize>>,
    /// Per primary-camera feature: the point its overlap match triangulates
    /// to, in the primary camera frame (metres).
    pub stereo_points: Vec<Option<Vec3F64>>,
}

impl Frame {
    /// Whether the frame carries metric depth: rectified stereo or rig
    /// overlap triangulation (scale is then observable from one frame).
    pub fn is_metric(&self) -> bool {
        self.is_stereo() || self.is_rig()
    }

    /// Whether the features come from several rig cameras.
    pub fn is_rig(&self) -> bool {
        self.rig.camera_ranges.len() > 1
    }

    /// Rig camera of feature `i`; 0 outside a multi-camera rig.
    pub fn camera_of(&self, i: usize) -> usize {
        self.rig
            .camera_ranges
            .iter()
            .position(|range| range.contains(&i))
            .unwrap_or(0)
    }

    /// The rig view of feature `i`, for a rig frame's non-primary features;
    /// `None` for primary-camera features and outside a rig, which use the
    /// system camera and `pose_world_to_cam` unchanged.
    pub fn secondary_view(&self, i: usize) -> Option<&RigView> {
        let camera = self.camera_of(i);
        if camera == 0 {
            return None;
        }
        self.rig.geometry.as_ref()?.view(camera)
    }

    /// Camera model and world-to-camera pose that observe feature `i`:
    /// `primary` and `pose` unless `i` belongs to a secondary rig camera.
    pub fn observing<'a>(
        &'a self,
        i: usize,
        primary: &'a PinholeCamera,
        pose: &Pose3d,
    ) -> (&'a PinholeCamera, Pose3d) {
        match self.secondary_view(i) {
            Some(view) => (&view.camera, view.pose(pose)),
            None => (primary, *pose),
        }
    }

    /// Feature indices of rig camera `camera`; every feature for camera 0
    /// outside a rig.
    pub fn camera_range(&self, camera: usize) -> Range<usize> {
        match self.rig.camera_ranges.get(camera) {
            Some(range) => range.clone(),
            None if camera == 0 && !self.is_rig() => 0..self.features.keypoints_xy.len(),
            None => 0..0,
        }
    }

    /// Whether this frame carries per-keypoint stereo depth.
    pub fn is_stereo(&self) -> bool {
        !self.depth.is_empty()
    }

    /// Computes and caches undistorted keypoint coordinates (no-op when
    /// already computed for the current keypoint set). A rig camera's
    /// features use that camera's model.
    pub fn ensure_undistorted(&mut self, camera: &PinholeCamera) {
        if self.keypoints_undist.len() == self.features.keypoints_xy.len() {
            return;
        }
        let undistorted = self
            .features
            .keypoints_xy
            .iter()
            .enumerate()
            .map(|(i, kp)| {
                let camera = self.secondary_view(i).map_or(camera, |view| &view.camera);
                camera
                    .undistort(kp[0] as f64, kp[1] as f64)
                    .map_or(*kp, |p| [p.x as f32, p.y as f32])
            })
            .collect();
        self.keypoints_undist = undistorted;
    }

    /// Undistorted coordinates of keypoint `i`, from the cache when present
    /// (frames built outside the pipeline fall back to undistorting on the
    /// fly).
    pub fn undistorted_xy(&self, i: usize, camera: &PinholeCamera) -> Option<[f32; 2]> {
        if !self.keypoints_undist.is_empty() {
            return self.keypoints_undist.get(i).copied();
        }
        self.features.keypoints_xy.get(i).map(|kp| {
            camera
                .undistort(kp[0] as f64, kp[1] as f64)
                .map_or(*kp, |p| [p.x as f32, p.y as f32])
        })
    }

    /// Metric depth for keypoint `i`, if it has a valid stereo match.
    pub fn stereo_depth(&self, i: usize) -> Option<f32> {
        match self.depth.get(i) {
            Some(&z) if z > 0.0 => Some(z),
            _ => None,
        }
    }
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
            rig: RigFeatures::default(),
        }
    }

    #[test]
    fn monocular_frame_reports_no_stereo() {
        let frame = empty_frame(Vec::new(), Vec::new());
        assert!(!frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), None);
    }

    #[test]
    fn stereo_frame_reports_valid_depth() {
        let frame = empty_frame(vec![10.0, -1.0], vec![2.5, -1.0]);
        assert!(frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), Some(2.5));
        assert_eq!(frame.stereo_depth(1), None); // sentinel
        assert_eq!(frame.stereo_depth(2), None); // out of range
    }

    #[test]
    fn rig_features_belong_to_their_camera() {
        let mut frame = empty_frame(Vec::new(), Vec::new());
        assert_eq!(frame.camera_of(7), 0);
        assert!(!frame.is_rig());
        frame.rig.camera_ranges = vec![0..3, 3..5];
        assert!(frame.is_rig());
        assert_eq!(frame.camera_of(2), 0);
        assert_eq!(frame.camera_of(3), 1);
        assert_eq!(frame.camera_range(1), 3..5);
        assert!(!frame.is_stereo(), "a rig frame carries no rectified depth");
    }
}
