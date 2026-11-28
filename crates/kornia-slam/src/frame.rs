//! Core frame-domain types shared across the crate.

use std::ops::Range;
use std::sync::Arc;

use crate::features::RBrief;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

use crate::features::Features;
use crate::rig::{RigGeometry, RigView};

/// A camera observation: features extracted at a known pose. `D` is the
/// descriptor type; ORB's by default.
#[derive(Debug, Clone)]
pub struct Frame<D = RBrief> {
    /// Frame index.
    pub idx: usize,
    /// Features extracted for this frame.
    pub features: Features<D>,
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

impl<D> Frame<D> {
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
    pub fn secondary_view(&self, i: usize) -> Option<RigView<'_>> {
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
            Some(view) => (view.pinhole, view.pose(pose)),
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
                let camera = self.secondary_view(i).map_or(camera, |view| view.pinhole);
                undistort_keypoint(camera, kp)
            })
            .collect();
        self.keypoints_undist = undistorted;
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
            .map(|kp| {
                // The feature's own camera, as `ensure_undistorted` uses.
                undistort_keypoint(
                    self.secondary_view(i).map_or(camera, |view| view.pinhole),
                    kp,
                )
            })
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
    use crate::features::Features;

    fn empty_frame(u_right: Vec<f32>, depth: Vec<f32>) -> Frame {
        Frame {
            idx: 0,
            features: Features {
                keypoints_xy: Vec::new(),
                orientations: None,
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

    /// Without the cache, a secondary-camera feature is undistorted with its
    /// own camera's model, the same pixel the cache gives.
    #[test]
    fn the_uncached_fallback_uses_the_feature_view() {
        let pinhole = |fx: f64, k1: f64| PinholeCamera {
            fx,
            fy: fx,
            cx: 320.0,
            cy: 240.0,
            k1,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let primary = pinhole(400.0, 0.0);
        let secondary = pinhole(300.0, -0.2);
        let rig = kornia_sensors::SensorRig::from_cameras(vec![
            kornia_sensors::RigCamera::primary(primary.clone(), size),
            kornia_sensors::RigCamera::new(
                secondary.clone(),
                size,
                Pose3d::new(
                    kornia_algebra::Mat3F64::IDENTITY,
                    Vec3F64::new(0.1, 0.0, 0.0),
                ),
            ),
        ])
        .unwrap();
        let mut frame = empty_frame(Vec::new(), Vec::new());
        frame.features.keypoints_xy = vec![[500.0, 400.0], [500.0, 400.0]];
        frame.rig = RigFeatures {
            geometry: RigGeometry::from_rig(&rig).map(Arc::new),
            camera_ranges: vec![0..1, 1..2],
            ..RigFeatures::default()
        };

        let on_the_fly: Vec<_> = (0..2).map(|i| frame.undistorted_xy(i, &primary)).collect();
        assert_eq!(
            on_the_fly[1],
            Some(undistort_keypoint(&secondary, &[500.0, 400.0]))
        );
        assert_ne!(on_the_fly[0], on_the_fly[1]);
        frame.ensure_undistorted(&primary);
        let cached: Vec<_> = (0..2).map(|i| frame.undistorted_xy(i, &primary)).collect();
        assert_eq!(cached, on_the_fly);
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
