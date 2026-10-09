//! Frame sources for SLAM examples.
//!
//! Both offline datasets (EuRoC) and live cameras (OAK-D) feed the same
//! `SlamSystem::process` loop. This module exposes a single trait,
//! [`FrameSource`], so the main binary can stay source-agnostic. Each source
//! module also owns the run-file settings it is opened from.

pub mod euroc;
pub mod hilti;
pub mod mcap;
pub mod oakd;
pub mod uvc;

use std::path::{Path, PathBuf};

use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;
use serde::Deserialize;

use crate::datasets::StereoRectifier;
use crate::datasets::euroc::GroundTruthPose;
use kornia_slam::{CameraSelection, SensorSelection};

use euroc::EurocConfig;
use hilti::HiltiConfig;
use mcap::McapConfig;
use oakd::OakdConfig;
use uvc::UvcConfig;

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

/// Where a run's frames come from.
#[derive(Debug, Clone, Deserialize)]
pub enum SourceConfig {
    Euroc(EurocConfig),
    Hilti(HiltiConfig),
    Mcap(McapConfig),
    Oakd(OakdConfig),
    Uvc(UvcConfig),
}

impl SourceConfig {
    /// Checks, before anything is opened, that the source can provide the
    /// selected sensors and has the settings it needs. `SlamSystem::build`
    /// checks the opened source's calibration again.
    pub fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        match self {
            Self::Euroc(_) => Ok(()),
            Self::Hilti(hilti) => hilti.validate(sensors),
            Self::Mcap(mcap) => mcap.validate(sensors),
            Self::Oakd(oakd) => oakd.validate(sensors),
            Self::Uvc(uvc) => uvc.validate(sensors),
        }
    }

    /// Resolves relative paths against `base_dir`.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        match self {
            Self::Euroc(euroc) => euroc.resolve_paths(base_dir),
            Self::Hilti(hilti) => hilti.resolve_paths(base_dir),
            Self::Mcap(mcap) => mcap.resolve_paths(base_dir),
            Self::Oakd(oakd) => oakd.resolve_paths(base_dir),
            Self::Uvc(_) => {}
        }
    }

    /// Opens the source in the mode the selected sensors need.
    pub fn open(&self, sensors: SensorSelection) -> Result<OpenedSource, SourceError> {
        let stereo = sensors.cameras == CameraSelection::Stereo;
        match self {
            Self::Euroc(euroc) => euroc.open(stereo),
            Self::Hilti(hilti) => hilti.open(),
            Self::Mcap(mcap) => mcap.open(stereo),
            Self::Oakd(oakd) => oakd.open(stereo),
            Self::Uvc(uvc) => uvc.open(),
        }
    }
}

fn resolve(path: &mut PathBuf, base_dir: &Path) {
    if path.is_relative() {
        *path = base_dir.join(&*path);
    }
}

fn mono_only(source: &str, sensors: SensorSelection) -> Result<(), String> {
    if sensors.cameras == CameraSelection::Stereo {
        return Err(format!("{source} provides monocular images only"));
    }
    Ok(())
}

fn no_imu(source: &str, sensors: SensorSelection) -> Result<(), String> {
    if sensors.imu {
        return Err(format!("{source} provides no IMU data"));
    }
    Ok(())
}

/// Stereo from a raw camera pair needs a calibration to rectify it with.
fn stereo_needs_calib(
    source: &str,
    sensors: SensorSelection,
    calib: Option<&Path>,
) -> Result<(), String> {
    if sensors.cameras == CameraSelection::Stereo && calib.is_none() {
        return Err(format!("{source} with stereo cameras needs `calib`"));
    }
    Ok(())
}

fn positive(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} is {value}, but must be finite and > 0"))
    }
}

fn finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

/// Startup line for a dataset read through a frame window.
fn dataset_summary(total: usize, start: usize, n_frames: Option<usize>) -> String {
    format!(
        "Dataset: {total} frames (processing {start}..{})",
        start + n_frames.unwrap_or(0)
    )
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
