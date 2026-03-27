//! Core frame-domain types shared across the crate.

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
}
