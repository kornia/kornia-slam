//! Visual odometry and SLAM building blocks for kornia-rs.

pub mod estimation;
pub mod frame;
pub mod initialization;
pub mod loop_closure;
pub mod mapping;
pub mod pipeline;
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
pub use mapping::map;
pub use sensor_rig::{ImuCalibration, SensorRig};
pub use system::{LoopClosingConfig, LoopClosureEvent, SlamConfig, SlamSystem};
pub use tracking::MapProjectionEstimator;

#[allow(deprecated)]
pub use system::{PgoPipelineConfig, PipelineConfig, SlamPipeline};
pub use tracking::{KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus};
