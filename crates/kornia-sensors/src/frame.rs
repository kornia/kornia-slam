//! One synchronized sample of a camera rig.

use kornia_image::Image;

use crate::imu::ImuMeasurement;

/// One synchronized camera sample with the IMU samples since the previous one.
///
/// Images are grayscale and borrowed, so a consumer can process the frame
/// while its producer keeps the images. For a stereo rig they are the
/// rectified pair; for a rig with a fisheye model the left image is the raw
/// fisheye image (see [`SensorRig`](crate::SensorRig)).
#[derive(Clone, Copy)]
pub struct SensorFrame<'a> {
    /// Source frame index.
    pub idx: usize,
    /// Capture time in seconds, on the same clock as the IMU samples.
    pub timestamp_sec: f64,
    pub image: &'a Image<u8, 1>,
    pub right_image: Option<&'a Image<u8, 1>>,
    /// IMU samples in the body frame; may be empty.
    pub imu_samples: &'a [ImuMeasurement],
}
