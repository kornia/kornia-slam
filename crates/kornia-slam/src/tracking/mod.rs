//! Frame-to-map tracking: pose estimation, optical flow, the motion model and
//! loss-recovery policy, relocalization, and map projection.

pub mod local_map;
pub(crate) mod motion;
pub mod optical_flow;
mod policy;
pub mod pose_estimation;
pub mod relocalization;
pub(crate) mod tracker;

pub use crate::mapping::keyframes::{
    KeyframeEvidence, KeyframeInsertionStrategy, KeyframePolicy, median_compensated_parallax_px,
    median_parallax_px,
};
pub use local_map::{LocalMapSelectionConfig, select_local_map_points};
pub use policy::TrackingLossRecoveryPolicy;
pub use pose_estimation::MapProjectionEstimator;
