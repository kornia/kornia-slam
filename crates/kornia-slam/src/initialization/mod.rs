//! Visual and inertial initialization for the SLAM system.

pub(crate) mod bootstrap;
pub mod inertial;
pub mod two_view;

pub use bootstrap::{InitialMapHealth, initial_map_health};
pub use inertial::{ImuInitConfig, ImuInitResult, ImuInitializer};

/// The inertial initializer's former path; it now lives in [`inertial`].
pub use inertial as imu;
/// The inertial factors' former path; they now live in [`inertial::factor`].
pub use inertial::factor as inertial_factor;
