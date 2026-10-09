//! TUM-VI (`dataset-*_512_16`) as a [`FrameSource`]: the raw equidistant
//! fisheye `cam0`, optionally with the IMU, evaluated against motion capture.
//!
//! The sequences share the Hilti extraction's layout under `mav0/` and ship
//! their Kalibr chain as `dso/camchain.yaml`, so they are read by
//! [`HiltiSource`] with the upright images as recorded.

use std::path::{Path, PathBuf};

use kornia_slam::SensorSelection;
use serde::Deserialize;

use super::hilti::{FrameWindow, HiltiSource};
use super::{OpenedSource, SourceError, mono_only, resolve};

/// A TUM-VI sequence, e.g. `dataset-room1_512_16/`. Monocular raw fisheye.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TumViConfig {
    /// Sequence root, containing `mav0/` and `dso/camchain.yaml`.
    pub data: PathBuf,
    #[serde(default)]
    pub start_frame: usize,
    #[serde(default)]
    pub max_frames: usize,
}

impl TumViConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        mono_only("TUM-VI", sensors)
    }

    pub(super) fn resolve_paths(&mut self, base_dir: &Path) {
        resolve(&mut self.data, base_dir);
    }

    pub(super) fn open(&self, imu: bool) -> Result<OpenedSource, SourceError> {
        let source = HiltiSource::open(
            self.data.join("mav0"),
            self.data.join("dso").join("camchain.yaml"),
            FrameWindow {
                start_frame: self.start_frame,
                max_frames: self.max_frames,
            },
            false,
            imu,
        )?;
        Ok(source.opened(self.start_frame))
    }
}
