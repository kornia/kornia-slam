//! Inertial factors: the preintegrated IMU factor, the stationary factor and
//! the keyframe state they both read.

pub mod preintegration;
mod state;
pub mod stationary;

pub use state::NavState;
