//! Live OAK-D camera settings; the device itself needs the `oakd` feature.

use std::path::{Path, PathBuf};

use kornia_slam::SensorSelection;
use serde::Deserialize;

use super::{OpenedSource, SourceError, no_imu, positive, resolve, stereo_needs_calib};

#[cfg(feature = "oakd")]
mod device;

#[cfg(feature = "oakd")]
use device::OakdSource;

/// A live OAK-D camera: CamB mono, or CamB+CamC stereo.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "oakd"), expect(dead_code))]
#[serde(deny_unknown_fields)]
pub struct OakdConfig {
    /// 0 runs until stopped.
    #[serde(default)]
    pub max_frames: usize,
    /// Mono resolution; stereo uses the calibration's.
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    /// Stereo calibration YAML; required with stereo cameras.
    #[serde(default)]
    pub calib: Option<PathBuf>,
}

fn default_width() -> u32 {
    640
}

fn default_height() -> u32 {
    400
}

fn default_fps() -> f32 {
    30.0
}

impl OakdConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        stereo_needs_calib("Oakd", sensors, self.calib.as_deref())?;
        no_imu("Oakd", sensors)?;
        positive("fps", self.fps.into())?;
        positive("width", self.width.into())?;
        positive("height", self.height.into())
    }

    pub(super) fn resolve_paths(&mut self, base_dir: &Path) {
        if let Some(calib) = &mut self.calib {
            resolve(calib, base_dir);
        }
    }

    #[cfg(feature = "oakd")]
    pub(super) fn open(&self, stereo: bool) -> Result<OpenedSource, SourceError> {
        let source = match (&self.calib, stereo) {
            (Some(calib), true) => OakdSource::open_stereo(self.fps, calib, self.max_frames)?,
            _ => OakdSource::open(self.width, self.height, self.fps, self.max_frames)?,
        };
        Ok(OpenedSource {
            summary: None,
            ground_truth: None,
            source: Box::new(source),
        })
    }

    #[cfg(not(feature = "oakd"))]
    pub(super) fn open(&self, _stereo: bool) -> Result<OpenedSource, SourceError> {
        Err(SourceError::other(
            "this build has no OAK-D support; rebuild with `--features oakd`",
        ))
    }
}
