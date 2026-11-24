//! Sliding-window visual-inertial odometry: a Basalt-style fixed-lag
//! smoother over the rectified stereo pair (`experiments/euroc-2x-vio-design.md`).

mod config;

pub use config::{Odometry, OdometryImuNoise, OdometryMode, OdometryTuning, SlidingWindowConfig};
