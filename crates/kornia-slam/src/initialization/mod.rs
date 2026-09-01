//! Visual and inertial initialization for the SLAM system.

mod inertial;
mod two_view;

pub use inertial::{ImuInitConfig, ImuInitReject, ImuInitResult, ImuInitializer, KeyframeVelocity};
pub use two_view::{
    TwoViewAcceptanceConfig, TwoViewEstimate, TwoViewInitConfig, TwoViewRejectReason,
    try_initialize_two_view,
};
