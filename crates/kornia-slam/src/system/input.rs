//! Sensor input to [`SlamSystem::process`](super::SlamSystem::process).

use kornia_image::{Image, ImageError, ImageSize};
use kornia_sensors::imu::ImuMeasurement;

/// One synchronized camera sample with the IMU samples since the previous one.
///
/// Images are grayscale. With a stereo rig they are the rectified pair; with a
/// fisheye rig the left image is the raw fisheye image. Inputs for sensors the
/// pipeline does not select are ignored: the right image of a mono pipeline and
/// the IMU samples of a visual-only one.
#[derive(Clone, Copy)]
pub struct SensorFrame<'a> {
    /// Source frame index; keyframes and diagnostics refer to it.
    pub idx: usize,
    /// Capture time in seconds, on the same clock as the IMU samples.
    pub timestamp_sec: f64,
    pub image: &'a Image<u8, 1>,
    pub right_image: Option<&'a Image<u8, 1>>,
    /// IMU samples in the body frame; may be empty.
    pub imu_samples: &'a [ImuMeasurement],
}

/// A frame the system cannot process. The system state is unchanged.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("frame {idx}: timestamp {timestamp_sec} is not finite")]
    NonFiniteTimestamp { idx: usize, timestamp_sec: f64 },
    #[error("frame {idx}: the image is empty")]
    EmptyImage { idx: usize },
    #[error("frame {idx}: the stereo pipeline needs a right image")]
    MissingRightImage { idx: usize },
    #[error(
        "frame {idx}: right image is {}x{}, left is {}x{}",
        right.width, right.height, left.width, left.height
    )]
    StereoSizeMismatch {
        idx: usize,
        left: ImageSize,
        right: ImageSize,
    },
    #[error("frame {idx}: feature extraction failed: {source}")]
    Frontend {
        idx: usize,
        #[source]
        source: ImageError,
    },
}

impl SensorFrame<'_> {
    /// Checks the input before any system state is touched. `stereo` is whether
    /// the pipeline selected stereo cameras.
    pub(crate) fn validate(&self, stereo: bool) -> Result<(), ProcessError> {
        let idx = self.idx;
        if !self.timestamp_sec.is_finite() {
            return Err(ProcessError::NonFiniteTimestamp {
                idx,
                timestamp_sec: self.timestamp_sec,
            });
        }
        if is_empty(self.image.size()) {
            return Err(ProcessError::EmptyImage { idx });
        }
        if !stereo {
            return Ok(());
        }
        let right = self
            .right_image
            .ok_or(ProcessError::MissingRightImage { idx })?;
        if right.size() != self.image.size() {
            return Err(ProcessError::StereoSizeMismatch {
                idx,
                left: self.image.size(),
                right: right.size(),
            });
        }
        Ok(())
    }
}

fn is_empty(size: ImageSize) -> bool {
    size.width == 0 || size.height == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize) -> Image<u8, 1> {
        Image::from_size_val(ImageSize { width, height }, 0).unwrap()
    }

    fn frame<'a>(image: &'a Image<u8, 1>, right: Option<&'a Image<u8, 1>>) -> SensorFrame<'a> {
        SensorFrame {
            idx: 7,
            timestamp_sec: 1.0,
            image,
            right_image: right,
            imu_samples: &[],
        }
    }

    #[test]
    fn mono_input_ignores_the_right_image() {
        let (left, right) = (image(8, 6), image(4, 4));
        assert!(frame(&left, None).validate(false).is_ok());
        assert!(frame(&left, Some(&right)).validate(false).is_ok());
    }

    #[test]
    fn stereo_input_needs_a_matching_right_image() {
        let (left, right, small) = (image(8, 6), image(8, 6), image(4, 4));
        assert!(frame(&left, Some(&right)).validate(true).is_ok());
        assert!(matches!(
            frame(&left, None).validate(true),
            Err(ProcessError::MissingRightImage { idx: 7 })
        ));
        assert!(matches!(
            frame(&left, Some(&small)).validate(true),
            Err(ProcessError::StereoSizeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_timestamps() {
        let left = image(8, 6);
        for timestamp_sec in [f64::NAN, f64::INFINITY] {
            let input = SensorFrame {
                timestamp_sec,
                ..frame(&left, None)
            };
            assert!(matches!(
                input.validate(false),
                Err(ProcessError::NonFiniteTimestamp { .. })
            ));
        }
    }
}
