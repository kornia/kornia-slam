//! Sliding-window visual-inertial odometry: a Basalt-style fixed-lag
//! smoother over the rectified stereo pair (`experiments/euroc-2x-vio-design.md`).

mod config;
// Test-only until the Observe hook runs the window from the system.
#[cfg(test)]
mod init;
#[cfg(test)]
mod keyframes;
#[cfg(test)]
mod landmarks;
#[cfg(test)]
mod linearize;
#[cfg(test)]
mod marginalize;
#[cfg(test)]
mod solve;
#[cfg(test)]
mod state;
#[cfg(test)]
mod synthetic;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod window;

/// Why a frame got no estimate.
#[cfg(test)]
#[derive(Debug, thiserror::Error)]
pub(crate) enum OdometryError {
    #[error("the window is not initialized")]
    NotInitialized,
    #[error("no finite accelerometer sample to start from")]
    NoGravity,
    #[error("no IMU interval between the frames")]
    ImuGap,
    #[error("a state or the prior is not finite")]
    NonFinite,
    #[error("a prior variable is not in the window")]
    PriorOutsideWindow,
    #[error("the marginal prior is not finite")]
    Marginalization,
}

pub use config::{Odometry, OdometryImuNoise, OdometryMode, OdometryTuning, SlidingWindowConfig};
