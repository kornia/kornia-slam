//! Validation of the input to [`SlamSystem::process`](super::SlamSystem::process).

use kornia_image::{ImageError, ImageSize};
use kornia_sensors::SensorFrame;

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

/// Checks the input before any system state is touched. `stereo` is whether
/// the pipeline selected stereo cameras; inputs for sensors it did not select
/// are ignored.
pub(crate) fn validate(input: &SensorFrame<'_>, stereo: bool) -> Result<(), ProcessError> {
    let idx = input.idx;
    if !input.timestamp_sec.is_finite() {
        return Err(ProcessError::NonFiniteTimestamp {
            idx,
            timestamp_sec: input.timestamp_sec,
        });
    }
    if is_empty(input.image.size()) {
        return Err(ProcessError::EmptyImage { idx });
    }
    if !stereo {
        return Ok(());
    }
    let right = input
        .right_image
        .ok_or(ProcessError::MissingRightImage { idx })?;
    if right.size() != input.image.size() {
        return Err(ProcessError::StereoSizeMismatch {
            idx,
            left: input.image.size(),
            right: right.size(),
        });
    }
    Ok(())
}

fn is_empty(size: ImageSize) -> bool {
    size.width == 0 || size.height == 0
}

#[cfg(test)]
mod tests {
    use kornia_image::Image;

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
        assert!(validate(&frame(&left, None), false).is_ok());
        assert!(validate(&frame(&left, Some(&right)), false).is_ok());
    }

    #[test]
    fn stereo_input_needs_a_matching_right_image() {
        let (left, right, small) = (image(8, 6), image(8, 6), image(4, 4));
        assert!(validate(&frame(&left, Some(&right)), true).is_ok());
        assert!(matches!(
            validate(&frame(&left, None), true),
            Err(ProcessError::MissingRightImage { idx: 7 })
        ));
        assert!(matches!(
            validate(&frame(&left, Some(&small)), true),
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
                validate(&input, false),
                Err(ProcessError::NonFiniteTimestamp { .. })
            ));
        }
    }
}
