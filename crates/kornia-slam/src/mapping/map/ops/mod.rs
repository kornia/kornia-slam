//! Map operations grouped by responsibility; storage remains owned by Map.

mod correction;
mod mutation;
mod queries;
mod snapshot;

pub use correction::{
    InertialAlignment, InertialAlignmentError, KeyframeBaCorrection, KeyframeVelocity,
    LocalBaMergeResult, PoseGraphCorrectionError, PoseGraphCorrectionResult,
};
pub use mutation::{
    InsertionResult, LandmarkSeed, LandmarkTarget, MapInsertion, MapMutationError,
    MapPointMergeResult, ObservationLink,
};
pub use snapshot::{BaSnapshot, BaUpdate};
