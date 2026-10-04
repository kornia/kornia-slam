//! Frame sources for SLAM examples.
//!
//! Both offline datasets (EuRoC) and live cameras (OAK-D) feed the same
//! `process_frame` loop. This module exposes a single trait, [`FrameSource`],
//! so the main binary can stay source-agnostic.

pub mod euroc;
pub mod hilti;
pub mod mcap;
#[cfg(feature = "oakd")]
pub mod oakd;
#[cfg(feature = "uvc")]
pub mod uvc;

use kornia_image::Image;
use kornia_imgproc::features::OrbFeatures;
use kornia_sensors::SensorRig;

use crate::datasets::StereoRectifier;
use crate::datasets::euroc::ImuSample;

pub use euroc::EurocSource;
pub use hilti::HiltiSource;
pub use mcap::McapSource;
#[cfg(feature = "oakd")]
pub use oakd::OakdSource;
#[cfg(feature = "uvc")]
pub use uvc::UvcSource;

/// One frame yielded by a source.
pub struct FrameItem {
    /// Absolute frame index (source-defined; EuRoC counts samples, OAK-D counts received frames).
    pub idx: usize,
    /// Capture timestamp in seconds (host clock for live sources).
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Grayscale image (rectified left view when the source is stereo).
    pub image: Image<u8, 1>,
    /// Rectified right view, when the source provides a stereo pair.
    pub right_image: Option<Image<u8, 1>>,
    /// IMU samples between the previous yielded camera frame and this one.
    #[allow(dead_code)]
    pub imu_samples: Vec<ImuSample>,
}

/// Pull-based interface for monocular SLAM frame producers.
///
/// `next_frame` returns `Ok(None)` when the stream is exhausted. Offline
/// datasets exhaust after their last sample; live sources may exhaust when
/// a CLI-imposed cap is reached.
pub trait FrameSource {
    /// Calibration of the sensors behind this source. Must be valid before the
    /// first `next_frame` call. For a stereo source the camera is the rectified
    /// one shared by both views, and the IMU extrinsic (when present) refers to it.
    fn rig(&self) -> SensorRig;

    /// Total frames the source will yield, if known.
    ///
    /// Live sources without a cap return `None`. The TUI uses this to render
    /// a progress bar; absent it, the bar shows elapsed-only.
    fn n_frames_hint(&self) -> Option<usize>;

    /// Pull the next frame. `Ok(None)` ⇒ end of stream.
    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError>;

    /// Map keypoints from raw-image pixels into the coordinate frame implied by
    /// [`Self::rig`]'s camera, filtering features the camera model cannot represent.
    ///
    /// Default is a no-op: sources that yield images already in `camera()`'s
    /// frame (EuRoC, rectified stereo) leave features untouched. A fisheye
    /// source extracts ORB on the raw fisheye image, then overrides this to
    /// undistort each keypoint to its virtual-pinhole pixel and drop features
    /// beyond the usable incidence angle. Called after extraction, before the
    /// features are handed to the SLAM pipeline. All per-feature arrays
    /// (keypoints, orientations, descriptors, octaves) stay aligned.
    fn undistort_features(&self, _features: &mut OrbFeatures) {}
}

/// Errors returned from a [`FrameSource`].
#[derive(thiserror::Error, Debug)]
pub enum SourceError {
    /// I/O error reading a frame from disk.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Other error (dataset parse, image decode, device error).
    #[error("{0}")]
    Other(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl SourceError {
    /// Wrap any boxable error as a `SourceError::Other`.
    pub fn other<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    {
        Self::Other(err.into())
    }
}

/// Rectify a raw stereo pair into freshly allocated `(left, right)` images.
fn rectify_pair(
    rectifier: &StereoRectifier,
    left: &Image<u8, 1>,
    right: &Image<u8, 1>,
) -> Result<(Image<u8, 1>, Image<u8, 1>), SourceError> {
    let mut left_rect = Image::from_size_val(left.size(), 0).map_err(SourceError::other)?;
    let mut right_rect = Image::from_size_val(right.size(), 0).map_err(SourceError::other)?;
    rectifier
        .rectify_left(left, &mut left_rect)
        .map_err(SourceError::other)?;
    rectifier
        .rectify_right(right, &mut right_rect)
        .map_err(SourceError::other)?;
    Ok((left_rect, right_rect))
}
