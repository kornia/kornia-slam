//! Frame-to-map tracking: pose estimation, optical flow, the motion model and
//! the keyframe and loss-recovery policies, relocalization, and the parallax
//! measure the keyframe policy can key off.

pub mod local_map;
pub(crate) mod motion;
pub mod optical_flow;
pub(crate) mod parallax;
mod policy;
pub mod pose_estimation;
pub mod relocalization;
pub(crate) mod tracker;

pub use local_map::{LocalMapSelectionConfig, select_local_map_points};
pub use parallax::{median_compensated_parallax_px, median_parallax_px};
pub use policy::{KeyframeEvidence, KeyframePolicy, TrackingLossRecoveryPolicy};
pub use pose_estimation::MapProjectionEstimator;
