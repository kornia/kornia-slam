//! Validation of the input to [`SlamSystem::process_image`](super::SlamSystem::process_image)
//! [`SlamSystem::process_stereo`](super::SlamSystem::process_stereo) and
//! [`SlamSystem::process_rig`](super::SlamSystem::process_rig).

use kornia_image::{Image, ImageError, ImageSize};
use kornia_sensors::RigCamera;

/// A frame the system cannot process. The system state is unchanged.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("timestamp {timestamp_sec} is not finite")]
    NonFiniteTimestamp { timestamp_sec: f64 },
    #[error("the image is empty")]
    EmptyImage,
    #[error("the stereo pipeline needs a right image, use process_stereo")]
    StereoRequired,
    #[error("the pipeline tracks a multi-camera rig, which needs an image per camera")]
    RigRequired,
    #[error("the pipeline has no multi-camera rig, use process_image or process_stereo")]
    NotARig,
    #[error("the rig has {expected} cameras, but {got} images were passed")]
    RigImageCount { expected: usize, got: usize },
    #[error(
        "rig camera {camera} image is {}x{}, its calibration is {}x{}",
        image.width, image.height, calibrated.width, calibrated.height
    )]
    RigSizeMismatch {
        camera: usize,
        image: ImageSize,
        calibrated: ImageSize,
    },
    #[error("the pipeline is monocular, use process_image")]
    MonoPipeline,
    #[error(
        "right image is {}x{}, left is {}x{}",
        right.width, right.height, left.width, left.height
    )]
    StereoSizeMismatch { left: ImageSize, right: ImageSize },
    #[error("feature extraction failed: {0}")]
    Frontend(#[source] ImageError),
}

/// Checks one rig capture before any system state is touched: one image per
/// rig camera, each of its calibrated size.
pub(crate) fn validate_rig_images(
    timestamp_sec: f64,
    images: &[&Image<u8, 1>],
    cameras: &[RigCamera],
) -> Result<(), ProcessError> {
    if images.len() != cameras.len() {
        return Err(ProcessError::RigImageCount {
            expected: cameras.len(),
            got: images.len(),
        });
    }
    for (camera, (image, calibration)) in images.iter().zip(cameras).enumerate() {
        validate_image(timestamp_sec, image)?;
        if image.size() != calibration.size {
            return Err(ProcessError::RigSizeMismatch {
                camera,
                image: image.size(),
                calibrated: calibration.size,
            });
        }
    }
    Ok(())
}

/// Checks one image before any system state is touched.
pub(crate) fn validate_image(timestamp_sec: f64, image: &Image<u8, 1>) -> Result<(), ProcessError> {
    if !timestamp_sec.is_finite() {
        return Err(ProcessError::NonFiniteTimestamp { timestamp_sec });
    }
    if is_empty(image.size()) {
        return Err(ProcessError::EmptyImage);
    }
    Ok(())
}

/// Checks a stereo pair before any system state is touched.
pub(crate) fn validate_stereo_pair(
    timestamp_sec: f64,
    left: &Image<u8, 1>,
    right: &Image<u8, 1>,
) -> Result<(), ProcessError> {
    validate_image(timestamp_sec, left)?;
    if right.size() != left.size() {
        return Err(ProcessError::StereoSizeMismatch {
            left: left.size(),
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
    use super::*;

    fn image(width: usize, height: usize) -> Image<u8, 1> {
        Image::from_size_val(ImageSize { width, height }, 0).unwrap()
    }

    #[test]
    fn stereo_input_needs_a_matching_right_image() {
        let (left, right, small) = (image(8, 6), image(8, 6), image(4, 4));
        assert!(validate_stereo_pair(1.0, &left, &right).is_ok());
        assert!(matches!(
            validate_stereo_pair(1.0, &left, &small),
            Err(ProcessError::StereoSizeMismatch { .. })
        ));
    }

    #[test]
    fn a_rig_capture_needs_one_calibrated_image_per_camera() {
        let camera = |width, height| RigCamera {
            camera: kornia_3d::camera::PinholeCamera {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
            size: ImageSize { width, height },
            primary_from_camera: kornia_3d::pose::Pose3d::IDENTITY,
        };
        let cameras = [camera(8, 6), camera(7, 5)];
        let (first, second) = (image(8, 6), image(7, 5));
        assert!(validate_rig_images(1.0, &[&first, &second], &cameras).is_ok());
        assert!(matches!(
            validate_rig_images(1.0, &[&first], &cameras),
            Err(ProcessError::RigImageCount {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            validate_rig_images(1.0, &[&first, &first], &cameras),
            Err(ProcessError::RigSizeMismatch { camera: 1, .. })
        ));
    }

    #[test]
    fn rejects_empty_images() {
        let (empty, right) = (image(0, 6), image(0, 6));
        assert!(matches!(
            validate_image(1.0, &empty),
            Err(ProcessError::EmptyImage)
        ));
        assert!(matches!(
            validate_stereo_pair(1.0, &empty, &right),
            Err(ProcessError::EmptyImage)
        ));
    }

    #[test]
    fn rejects_non_finite_timestamps() {
        let left = image(8, 6);
        for timestamp_sec in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                validate_image(timestamp_sec, &left),
                Err(ProcessError::NonFiniteTimestamp { .. })
            ));
        }
    }
}
