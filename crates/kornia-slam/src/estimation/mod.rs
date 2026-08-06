//! Pose estimation algorithms and estimator modules.

pub mod imu_init;
pub mod inertial_init_factor;
pub mod map_projection;
pub mod optical_flow;
pub mod pnp;
pub mod two_view;

use kornia_3d::pose::Pose3d;

pub use imu_init::{ImuInitConfig, ImuInitResult, ImuInitializer};
pub use map_projection::MapProjectionEstimator;
pub use optical_flow::{Track, TrackState};

/// Successful pose estimate returned by any estimator.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// Estimated world-to-camera pose.
    pub pose: Pose3d,
    /// Matched pairs `(map_point_idx, keypoint_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// Number of inlier correspondences supporting the estimate.
    pub inliers: usize,
}
