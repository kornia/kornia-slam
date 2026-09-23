//! Visual and inertial initialization for the SLAM system.

pub(crate) mod bootstrap;
pub mod inertial;
pub mod two_view;

pub use bootstrap::{InitialMapHealth, initial_map_health};
pub use inertial::{ImuInitConfig, ImuInitResult, ImuInitializer};
