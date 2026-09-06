//! Visual and inertial initialization for the SLAM system.

mod inertial;
mod two_view;

pub use inertial::{
    BiasPriors, ImuInitConfig, ImuInitNotReady, ImuInitNotReadyReason, ImuInitRejectReason,
    ImuInitResult, ImuInitializer, InertialInitOutcome, InertialInitRequest, InertialStage,
    KeyframeVelocity, RwgSeed,
};
pub use two_view::{
    TwoViewAcceptanceConfig, TwoViewEstimate, TwoViewInitConfig, TwoViewRejectReason,
    try_initialize_two_view,
};
