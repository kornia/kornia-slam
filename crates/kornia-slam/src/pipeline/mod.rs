//! Declarative SLAM pipeline definitions: the sensors a pipeline consumes and
//! the stages, optional branches and settings it runs.
//!
//! With the `serde` feature, definitions load from versioned RON files.

mod config;
#[cfg(feature = "serde")]
mod file;
mod runtime;
mod stages;
mod validation;

pub use config::{
    CameraSelection, FrontendConfig, KeyframeConfig, LoopClosingMode, MappingConfig,
    MappingExecution, OrbFrontendConfig, OrbSlamPipeline, PIPELINE_CONFIG_VERSION, PipelineConfig,
    PipelineDefinition, SensorSelection,
};
#[cfg(feature = "serde")]
pub use file::LoadError;
pub use stages::Stage;
pub use validation::ConfigError;

#[cfg(test)]
mod tests;
