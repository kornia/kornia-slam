//! Live UVC camera settings; the device itself needs the `uvc` feature.

use kornia_slam::SensorSelection;
use serde::Deserialize;

use super::{OpenedSource, SourceError, finite, mono_only, no_imu, positive};

#[cfg(feature = "uvc")]
mod device;

#[cfg(feature = "uvc")]
use device::UvcSource;

/// A live UVC camera (webcam, USB camera, CSI-to-UVC adapter). The intrinsics
/// must match the resolution the device actually streams at.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "uvc"), expect(dead_code))]
#[serde(deny_unknown_fields)]
pub struct UvcConfig {
    #[serde(default)]
    pub index: u32,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default)]
    pub max_frames: usize,
    /// Focal lengths and principal point, in pixels.
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Radial and tangential distortion.
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
}

fn default_width() -> u32 {
    640
}

fn default_height() -> u32 {
    480
}

impl UvcConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        mono_only("Uvc", sensors)?;
        no_imu("Uvc", sensors)?;
        positive("width", self.width.into())?;
        positive("height", self.height.into())?;
        positive("fx", self.fx)?;
        positive("fy", self.fy)?;
        for (name, value) in [
            ("cx", self.cx),
            ("cy", self.cy),
            ("k1", self.k1),
            ("k2", self.k2),
            ("p1", self.p1),
            ("p2", self.p2),
        ] {
            finite(name, value)?;
        }
        Ok(())
    }

    #[cfg(feature = "uvc")]
    pub(super) fn open(&self) -> Result<OpenedSource, SourceError> {
        let camera = kornia_3d::camera::PinholeCamera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: self.p2,
        };
        let source = UvcSource::open(self.index, self.width, self.height, camera, self.max_frames)?;
        Ok(OpenedSource {
            summary: None,
            ground_truth: None,
            source: Box::new(source),
        })
    }

    #[cfg(not(feature = "uvc"))]
    pub(super) fn open(&self) -> Result<OpenedSource, SourceError> {
        Err(SourceError::other(
            "this build has no UVC support; rebuild with `--features uvc`",
        ))
    }
}
