//! Visual odometry and SLAM building blocks for kornia-rs.
//!
//! [`SlamSystem`] is the runtime: [`SlamSystem::build`] assembles it from a
//! [`PipelineConfig`] and the source's calibrated [`SensorRig`], and
//! [`SlamSystem::process`] tracks one [`SensorFrame`] of images and IMU samples.
//! Images are in the coordinates of the rig's camera, rectified for stereo,
//! except for a rig with [`SensorRig::fisheye`], whose raw fisheye images
//! the system maps into that camera keypoint by keypoint.
//!
//! ```no_run
//! # use kornia_3d::camera::PinholeCamera;
//! # use kornia_image::Image;
//! use kornia_slam::{PipelineConfig, SensorFrame, SensorRig, SlamSystem};
//!
//! # fn run(camera: PinholeCamera, images: Vec<Image<u8, 1>>) -> Result<(), Box<dyn std::error::Error>> {
//! let mut system = SlamSystem::build(PipelineConfig::default(), SensorRig::new(camera))?;
//! for (idx, image) in images.iter().enumerate() {
//!     let result = system.process(SensorFrame {
//!         idx,
//!         timestamp_sec: idx as f64 / 20.0,
//!         image,
//!         right_image: None,
//!         imu_samples: &[],
//!     })?;
//!     println!("{idx}: {:?}", result.status);
//! }
//! # Ok(())
//! # }
//! ```

pub mod frame;
pub mod initialization;
pub mod loop_closure;
pub mod mapping;
pub mod pipeline;
mod pose_conversion;
pub mod stereo;
pub mod system;
pub mod tracking;

pub use frame::Frame;
pub use kornia_imgproc::features::OrbFeatures;
pub use kornia_sensors::{ImuCalibration, SensorRig};
pub use loop_closure::{LoopClosingConfig, LoopClosureEvent};
pub use pipeline::{BuildError, PipelineConfig};
pub use system::{
    FrontendObservation, ProcessError, SensorFrame, SlamSystem, TrackingResult, TrackingStatus,
};
pub use tracking::{KeyframePolicy, MapProjectionEstimator};
