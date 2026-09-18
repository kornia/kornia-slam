//! Re-exports of the runtime, which lives in [`crate::system`].

#[allow(deprecated)]
pub use crate::system::{
    LoopClosingConfig, LoopClosureEvent, PgoPipelineConfig, PipelineConfig, SlamConfig,
    SlamPipeline, SlamSystem,
};
