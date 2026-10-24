//! Visual odometry and SLAM building blocks for kornia-rs.
//!
//! [`SlamSystem`] is the runtime: [`SlamSystem::build`] assembles it from a
//! [`SystemConfig`] and the source's calibrated [`SensorRig`], and
//! [`SlamSystem::process_imu`] buffers IMU samples as they arrive and
//! [`SlamSystem::process_image`] tracks one image (or
//! [`SlamSystem::process_stereo`] a stereo pair).
//! Images are in the coordinates of the rig's camera, rectified for stereo,
//! except for a rig with [`SensorRig::fisheye`], whose raw fisheye images
//! the system maps into that camera keypoint by keypoint.
//!
//! ```no_run
//! # use kornia_3d::camera::PinholeCamera;
//! # use kornia_image::Image;
//! use kornia_slam::{SystemConfig, SensorRig, SlamSystem};
//!
//! # fn run(camera: PinholeCamera, images: Vec<Image<u8, 1>>) -> Result<(), Box<dyn std::error::Error>> {
//! let mut system = SlamSystem::build(SystemConfig::default(), SensorRig::new(camera))?;
//! for (idx, image) in images.iter().enumerate() {
//!     let result = system.process_image(idx as f64 / 20.0, image)?;
//!     println!("{idx}: {:?}", result.status);
//! }
//! # Ok(())
//! # }
//! ```

/// `eprintln!` only when `KORNIA_SLAM_DEBUG` is set. The solvers' diagnostic lines run per solve
/// (~100 unbuffered writes a keyframe); on an embedded host whose stderr is a file on eMMC they
/// blocked the caller's solve thread on journal commits for seconds (2026-10-11, XLeRobot).
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

/// Whether [`debug_eprintln!`] prints: `KORNIA_SLAM_DEBUG` set to anything, read once.
pub(crate) fn debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KORNIA_SLAM_DEBUG").is_some())
}

pub mod frame;
pub(crate) mod frontend;
pub mod inertial;
pub mod initialization;
pub mod loop_closure;
pub mod mapping;
mod pose_conversion;
pub mod stereo;
pub mod system;
pub mod tracking;

pub use frame::Frame;
pub use frontend::FrontendObservation;
pub use kornia_imgproc::features::OrbFeatures;
pub use kornia_sensors::{ImuCalibration, SensorRig};
pub use loop_closure::{LoopClosingConfig, LoopClosureEvent};
#[cfg(feature = "serde")]
pub use system::LoadError;
pub use system::{
    BuildError, CameraSelection, ConfigError, LoopClosingMode, ProcessError, SensorSelection,
    SlamSystem, SystemConfig, TrackingResult, TrackingStatus,
};
pub use tracking::{KeyframePolicy, MapProjectionEstimator};
