//! Visual odometry and SLAM building blocks for kornia-rs.

pub mod estimation;
pub mod frame;
pub mod loop_closure;
pub mod map;
pub mod pipeline;
pub mod place_recognition;
mod pose_conversion;
mod sparse_pgo;
pub mod stereo;
pub mod system;
pub mod vi_ba_schur;

pub use estimation::MapProjectionEstimator;
pub use frame::Frame;
pub use kornia_imgproc::features::OrbFeatures;
pub use pipeline::{LoopClosureEvent, PgoPipelineConfig, PipelineConfig, SlamPipeline};
pub use system::{KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus};
