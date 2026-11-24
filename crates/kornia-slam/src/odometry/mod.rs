//! Sliding-window visual-inertial odometry: a Basalt-style fixed-lag
//! smoother over the rectified stereo pair (`experiments/euroc-2x-vio-design.md`).

mod config;
// Test-only until the Observe hook runs the window from the system.
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
mod tests;
#[cfg(test)]
mod window;

pub use config::{Odometry, OdometryImuNoise, OdometryMode, OdometryTuning, SlidingWindowConfig};
