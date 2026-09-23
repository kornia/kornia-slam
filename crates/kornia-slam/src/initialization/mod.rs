//! Visual and inertial initialization for the SLAM system.

pub(crate) mod bootstrap;
pub mod imu;
pub mod inertial_factor;
pub mod two_view;

pub use bootstrap::{InitialMapHealth, initial_map_health};
pub use imu::{ImuInitConfig, ImuInitResult, ImuInitializer};
