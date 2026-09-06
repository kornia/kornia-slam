//! Visual odometry and SLAM building blocks for kornia-rs.

pub mod frame;
pub mod initialization;
pub mod loop_closure;
pub mod map;
pub mod place_recognition;
mod pose_conversion;
mod sparse_pgo;
pub mod stereo;
pub mod system;
pub mod tracking;
pub mod vi_ba_schur;

pub use frame::Frame;
pub use kornia_imgproc::features::OrbFeatures;
pub use system::{LoopClosingConfig, LoopClosureEvent, SlamConfig, SlamSystem};
pub use tracking::pose_estimation::MapProjectionEstimator;
pub use tracking::{KeyframePolicy, SystemMode, SystemState, TrackingResult, TrackingStatus};

/// Deprecated name for [`SlamSystem`].
#[deprecated(note = "use SlamSystem")]
pub type SlamPipeline = SlamSystem;

/// Deprecated name for [`SlamConfig`].
#[deprecated(note = "use SlamConfig")]
pub type PipelineConfig = SlamConfig;

/// Deprecated name for [`LoopClosingConfig`].
#[deprecated(note = "use LoopClosingConfig")]
pub type PgoPipelineConfig = LoopClosingConfig;
