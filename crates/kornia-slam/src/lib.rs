//! Visual odometry and SLAM building blocks for kornia-rs.

pub mod frame;
pub mod initialization;
pub mod loop_closure;
pub mod mapping;
pub mod place_recognition;
mod pose_conversion;
pub mod sensor_rig;
mod sparse_pgo;
pub mod stereo;
pub mod system;
pub mod tracking;
pub mod vi_ba_schur;

pub use frame::Frame;
pub use kornia_imgproc::features::OrbFeatures;
pub use loop_closure::{LoopClosingConfig, LoopClosureEvent};
pub use sensor_rig::{ImuCalibration, SensorRig};
pub use system::{SlamConfig, SlamSystem};
pub use tracking::MapProjectionEstimator;
pub use tracking::{KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus};
