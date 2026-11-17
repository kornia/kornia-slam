//! Pose estimation used by frame-to-map tracking.

pub mod map_projection;
pub mod pnp;
// Skeleton for the PR #180 lanes: lane A fills and calls the pose-inertial solve.
#[allow(dead_code)]
pub(crate) mod pose_inertial;
pub mod rig_pnp;

use kornia_3d::pose::Pose3d;

pub use map_projection::MapProjectionEstimator;

/// Successful pose estimate returned by any estimator.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// Estimated world-to-camera pose.
    pub pose: Pose3d,
    /// Matched pairs `(map_point_idx, keypoint_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// Number of inlier correspondences supporting the estimate.
    pub inliers: usize,
    /// The local map's most-voted keyframe; `None` without votes, or under
    /// [`LocalMapRules::Kornia`](crate::tracking::local_map::LocalMapRules).
    pub most_voted_keyframe: Option<usize>,
}
