//! Map operations grouped by responsibility; storage remains owned by Map.

mod bundle_adjustment;
mod correction;
mod culling;
mod growth;
mod mutation;
mod selection;

pub use bundle_adjustment::{
    KeyframeBaCorrection, LocalBaMergeResult, LocalBaSnapshot, STEREO_DEPTH_MIN_SIGMA,
    STEREO_DEPTH_REL_SIGMA,
};
pub use correction::{
    InertialAlignment, InertialAlignmentError, KeyframeVelocity, PoseGraphCorrectionError,
    PoseGraphCorrectionResult,
};
pub use mutation::{MapPointMergeResult, TriangulatedPoint};
pub use selection::InitialMapHealth;
