//! Frame sources for SLAM examples.
//!
//! Both offline datasets (EuRoC) and live cameras (OAK-D) feed the same
//! `SlamSystem::process` loop. This module exposes a single trait,
//! [`FrameSource`], so the main binary can stay source-agnostic.

pub mod euroc;
pub mod hilti;
pub mod mcap;
#[cfg(feature = "oakd")]
pub mod oakd;
#[cfg(feature = "uvc")]
pub mod uvc;

use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;

use crate::config::SourceConfig;
use crate::datasets::StereoRectifier;
use crate::datasets::euroc::GroundTruthPose;
use kornia_slam::{CameraSelection, SensorSelection};

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
    pub timestamp_sec: f64,
    /// Grayscale image (rectified left view when the source is stereo).
    pub image: Image<u8, 1>,
    /// Rectified right view, when the source provides a stereo pair.
    pub right_image: Option<Image<u8, 1>>,
    /// IMU samples between the previous yielded camera frame and this one.
    pub imu_samples: Vec<ImuMeasurement>,
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
    /// A source of raw fisheye images declares their model with
    /// [`SensorRig::with_fisheye`].
    fn rig(&self) -> SensorRig;

    /// Total frames the source will yield, if known.
    ///
    /// Live sources without a cap return `None`. The TUI uses this to render
    /// a progress bar; absent it, the bar shows elapsed-only.
    fn n_frames_hint(&self) -> Option<usize>;

    /// Pull the next frame. `Ok(None)` ⇒ end of stream.
    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError>;
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

/// A source opened from its run configuration.
pub struct OpenedSource {
    pub source: Box<dyn FrameSource>,
    /// Ground truth for evaluation; `None` when the dataset has none.
    pub ground_truth: Option<Vec<GroundTruthPose>>,
    /// One line describing what will be read, for the startup log.
    pub summary: Option<String>,
}

/// Opens the configured source in the mode the selected sensors need.
pub fn open(
    config: &SourceConfig,
    sensors: SensorSelection,
) -> Result<OpenedSource, Box<dyn std::error::Error>> {
    let stereo = sensors.cameras == CameraSelection::Stereo;
    let window = |total: usize, start: usize, n: Option<usize>| {
        format!(
            "Dataset: {total} frames (processing {start}..{})",
            start + n.unwrap_or(0)
        )
    };
    let opened = match config {
        SourceConfig::Euroc(e) => {
            let src = if stereo {
                EurocSource::open_stereo(&e.data, e.start_frame, e.max_frames)?
            } else {
                EurocSource::open(&e.data, e.start_frame, e.max_frames)?
            };
            OpenedSource {
                summary: Some(window(
                    src.dataset_len(),
                    e.start_frame,
                    src.n_frames_hint(),
                )),
                ground_truth: non_empty(src.ground_truth_poses_cloned()),
                source: Box::new(src),
            }
        }
        SourceConfig::Hilti(h) => {
            let src =
                HiltiSource::open(&h.data, &h.calib, h.start_frame, h.max_frames, h.rotate_180)?;
            OpenedSource {
                summary: Some(window(
                    src.dataset_len(),
                    h.start_frame,
                    src.n_frames_hint(),
                )),
                ground_truth: non_empty(src.ground_truth_poses_cloned()),
                source: Box::new(src),
            }
        }
        SourceConfig::Mcap(m) => {
            let src = match (&m.calib, stereo) {
                (Some(calib), true) => McapSource::open_stereo(
                    &m.path,
                    &m.channel,
                    &m.right_channel,
                    calib,
                    m.start_frame,
                    m.max_frames,
                )?,
                _ => McapSource::open(&m.path, &m.channel, m.start_frame, m.max_frames)?,
            };
            OpenedSource {
                summary: src
                    .n_frames_hint()
                    .map(|n| format!("MCAP: {n} frames from /{}", m.channel)),
                ground_truth: None,
                source: Box::new(src),
            }
        }
        #[cfg(feature = "oakd")]
        SourceConfig::Oakd(o) => {
            let src = match (&o.calib, stereo) {
                (Some(calib), true) => OakdSource::open_stereo(o.fps, calib, o.max_frames)?,
                _ => OakdSource::open(o.width, o.height, o.fps, o.max_frames)?,
            };
            OpenedSource {
                summary: None,
                ground_truth: None,
                source: Box::new(src),
            }
        }
        #[cfg(feature = "uvc")]
        SourceConfig::Uvc(u) => {
            let camera = kornia_3d::camera::PinholeCamera {
                fx: u.fx,
                fy: u.fy,
                cx: u.cx,
                cy: u.cy,
                k1: u.k1,
                k2: u.k2,
                p1: u.p1,
                p2: u.p2,
            };
            OpenedSource {
                summary: None,
                ground_truth: None,
                source: Box::new(UvcSource::open(
                    u.index,
                    u.width,
                    u.height,
                    camera,
                    u.max_frames,
                )?),
            }
        }
        #[cfg(not(feature = "oakd"))]
        SourceConfig::Oakd(_) => {
            return Err("this build has no OAK-D support; rebuild with `--features oakd`".into());
        }
        #[cfg(not(feature = "uvc"))]
        SourceConfig::Uvc(_) => {
            return Err("this build has no UVC support; rebuild with `--features uvc`".into());
        }
    };
    Ok(opened)
}

/// Datasets report missing ground truth as an empty list.
fn non_empty(poses: Vec<GroundTruthPose>) -> Option<Vec<GroundTruthPose>> {
    (!poses.is_empty()).then_some(poses)
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
