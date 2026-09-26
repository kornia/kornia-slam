//! Frame-to-map tracking: pose estimation, optical flow, the motion model and
//! the keyframe and loss-recovery policies.

pub mod local_map;
pub(crate) mod motion;
pub mod optical_flow;
mod policy;
pub mod pose_estimation;
pub(crate) mod tracker;

pub use local_map::{LocalMapSelectionConfig, select_local_map_points};
pub use policy::{KeyframePolicy, TrackingLossRecoveryPolicy};
pub use pose_estimation::MapProjectionEstimator;
