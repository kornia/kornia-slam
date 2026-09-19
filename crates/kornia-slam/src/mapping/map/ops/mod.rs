//! Operations on the map, grouped by responsibility. Each submodule adds
//! inherent methods to [`Map`](super::Map); the storage lives in `super`.

pub mod bundle_adjustment;
pub mod correction;
pub mod culling;
pub mod mutation;
pub mod selection;

pub use bundle_adjustment::InitialMapHealth;
pub use correction::{
    KeyframeBaCorrection, LocalBaMergeResult, LocalBaSnapshot, PoseGraphCorrectionError,
    PoseGraphCorrectionResult,
};
pub use mutation::{MapMutationError, MapPointMergeResult};
