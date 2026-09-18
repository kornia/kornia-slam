//! Frame-to-map tracking: pose estimation, optical flow, state and policies.

pub mod local_map;
pub(crate) mod motion;
pub mod optical_flow;
mod policy;
pub mod pose_estimation;
mod state;
pub(crate) mod tracker;

pub use local_map::{LocalMapSelectionConfig, select_local_map_points};
pub use policy::{KeyframePolicy, TrackingLossRecoveryPolicy};
pub use pose_estimation::MapProjectionEstimator;
pub use state::{SystemMode, SystemState, TrackingResult, TrackingStatus};
