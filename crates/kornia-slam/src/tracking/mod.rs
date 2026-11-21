//! Frame-to-map tracking: pose estimation, optical flow, the motion model and
//! the keyframe and loss-recovery policies, relocalization, and the parallax
//! measure the keyframe policy can key off.

pub mod local_map;
// Skeleton for the 1 ms frontend: lane T fills and calls these.
#[allow(dead_code)]
pub(crate) mod flow_pool;
#[allow(dead_code)]
pub(crate) mod flow_tracker;
pub(crate) mod motion;
pub mod optical_flow;
pub(crate) mod parallax;
mod policy;
pub mod pose_estimation;
pub mod pose_gate;
pub mod relocalization;
pub mod timings;
pub(crate) mod tracker;

pub use flow_pool::FlowPoolError;
pub use local_map::{
    LocalMapRules, LocalMapSelection, LocalMapSelectionConfig, select_local_map_points,
};
pub use parallax::{median_compensated_parallax_px, median_parallax_px};
pub use policy::{
    KeyframeEvidence, KeyframePolicy, KeyframeRules, KeyframesWhileLost, TrackingLossRecoveryPolicy,
};
pub use pose_estimation::MapProjectionEstimator;
pub use timings::TrackingTimings;
