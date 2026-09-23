//! Visual odometry and SLAM building blocks for kornia-rs.

pub mod frame;
pub mod initialization;
pub mod loop_closure;
pub mod mapping;
mod pose_conversion;
pub mod sensor_rig;
pub mod stereo;
pub mod system;
pub mod tracking;

pub use frame::Frame;
pub use kornia_imgproc::features::OrbFeatures;
pub use loop_closure::{LoopClosingConfig, LoopClosureEvent};
pub use sensor_rig::{ImuCalibration, SensorRig};
pub use system::{SlamConfig, SlamSystem, SystemMode, SystemState, TrackingResult, TrackingStatus};
pub use tracking::{KeyframePolicy, MapProjectionEstimator};
