//! Run files: the source to read and the SLAM system to run on it.
//!
//! ```ron
//! (
//!     source: Euroc((data: "../data/MH_01_easy", max_frames: 500)),
//!     system: (version: 1, sensors: (cameras: Stereo, imu: true)),
//! )
//! ```
//!
//! Relative paths, in either section, are resolved against the run file's
//! directory.

use std::path::{Path, PathBuf};

use kornia_slam::system::ConfigError;
use kornia_slam::{CameraSelection, PipelineConfig, SensorSelection};
use serde::Deserialize;

/// One run: where the data comes from and how it is processed.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub source: SourceConfig,
    /// Omitted, the default monocular pipeline runs.
    #[serde(default)]
    pub system: PipelineConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub enum SourceConfig {
    Euroc(EurocConfig),
    Hilti(HiltiConfig),
    Mcap(McapConfig),
    Oakd(OakdConfig),
    Uvc(UvcConfig),
}

/// An EuRoC MAV sequence; stereo and IMU calibration come from its `sensor.yaml` files.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EurocConfig {
    /// Sequence root, e.g. `MH_01_easy/`.
    pub data: PathBuf,
    #[serde(default)]
    pub start_frame: usize,
    /// 0 processes the whole sequence.
    #[serde(default)]
    pub max_frames: usize,
}

/// A Hilti-Trimble SLAM Challenge 2026 sequence extracted to the EuRoC-style
/// layout by the challenge's `ros2bag_to_euroc.py`. Monocular raw fisheye.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HiltiConfig {
    /// Extracted sequence root, containing `cam0/` and `imu0/`.
    pub data: PathBuf,
    /// Kalibr camera-IMU chain YAML.
    pub calib: PathBuf,
    #[serde(default)]
    pub start_frame: usize,
    #[serde(default)]
    pub max_frames: usize,
    /// The sensors are mounted inverted; disable when the extraction already
    /// rotated the images.
    #[serde(default = "enabled")]
    pub rotate_180: bool,
}

/// A bubbaloop MCAP recording.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McapConfig {
    pub path: PathBuf,
    /// Channel suffix to read; the left channel with stereo cameras.
    #[serde(default = "mono_left")]
    pub channel: String,
    #[serde(default = "mono_right")]
    pub right_channel: String,
    /// Stereo calibration YAML; required with stereo cameras.
    #[serde(default)]
    pub calib: Option<PathBuf>,
    #[serde(default)]
    pub start_frame: usize,
    #[serde(default)]
    pub max_frames: usize,
}

/// A live OAK-D camera: CamB mono, or CamB+CamC stereo.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "oakd"), expect(dead_code))]
#[serde(deny_unknown_fields)]
pub struct OakdConfig {
    /// 0 runs until stopped.
    #[serde(default)]
    pub max_frames: usize,
    /// Mono resolution; stereo uses the calibration's.
    #[serde(default = "oakd_width")]
    pub width: u32,
    #[serde(default = "oakd_height")]
    pub height: u32,
    #[serde(default = "thirty")]
    pub fps: f32,
    /// Stereo calibration YAML; required with stereo cameras.
    #[serde(default)]
    pub calib: Option<PathBuf>,
}

/// A live UVC camera (webcam, USB camera, CSI-to-UVC adapter). The intrinsics
/// must match the resolution the device actually streams at.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "uvc"), expect(dead_code))]
#[serde(deny_unknown_fields)]
pub struct UvcConfig {
    #[serde(default)]
    pub index: u32,
    #[serde(default = "uvc_width")]
    pub width: u32,
    #[serde(default = "uvc_height")]
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

fn enabled() -> bool {
    true
}
fn mono_left() -> String {
    "mono_left".into()
}
fn mono_right() -> String {
    "mono_right".into()
}
fn oakd_width() -> u32 {
    640
}
fn oakd_height() -> u32 {
    400
}
fn uvc_width() -> u32 {
    640
}
fn uvc_height() -> u32 {
    480
}
fn thirty() -> f32 {
    30.0
}

/// A run file that cannot be read or does not describe a valid run.
#[derive(Debug, thiserror::Error)]
pub enum RunConfigError {
    #[error("{}: {source}", path.display())]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{}: {source}", path.display())]
    Parse {
        path: PathBuf,
        source: Box<ron::error::SpannedError>,
    },
    #[error("{}: system: {source}", path.display())]
    System { path: PathBuf, source: ConfigError },
    #[error("{}: source: {message}", path.display())]
    Source { path: PathBuf, message: String },
}

/// Reads only the system version, so a system written for another schema
/// version is reported as such rather than as unknown fields.
#[derive(Deserialize)]
struct VersionProbe {
    #[serde(default)]
    system: SystemVersion,
}

/// A missing system or version defers to the full parse.
#[derive(Deserialize)]
struct SystemVersion {
    #[serde(default = "current_version")]
    version: u32,
}

impl Default for SystemVersion {
    fn default() -> Self {
        Self {
            version: current_version(),
        }
    }
}

fn current_version() -> u32 {
    kornia_slam::system::PIPELINE_CONFIG_VERSION
}

impl RunConfig {
    /// Loads and validates a run file and resolves its relative paths.
    pub fn from_ron_file(path: impl AsRef<Path>) -> Result<Self, RunConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| RunConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let mut run = Self::from_ron_str(&text).map_err(|error| error.at(path))?;
        run.resolve_paths(path.parent().unwrap_or(Path::new("")));
        Ok(run)
    }

    /// Parses and validates run-file text; paths are left as written.
    pub fn from_ron_str(text: &str) -> Result<Self, Invalid> {
        // Optional settings such as `calib` are written without `Some(..)`.
        let ron = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME);
        let parse_error = |error| Invalid::Parse(Box::new(error));
        let probe: VersionProbe = ron.from_str(text).map_err(parse_error)?;
        PipelineConfig::check_version(probe.system.version).map_err(Invalid::System)?;
        let run: Self = ron.from_str(text).map_err(parse_error)?;
        run.system.validate().map_err(Invalid::System)?;
        run.source
            .validate(run.system.sensors)
            .map_err(Invalid::Source)?;
        Ok(run)
    }

    fn resolve_paths(&mut self, base_dir: &Path) {
        let resolve = |path: &mut PathBuf| {
            if path.is_relative() {
                *path = base_dir.join(&*path);
            }
        };
        match &mut self.source {
            SourceConfig::Euroc(euroc) => resolve(&mut euroc.data),
            SourceConfig::Hilti(hilti) => {
                resolve(&mut hilti.data);
                resolve(&mut hilti.calib);
            }
            SourceConfig::Mcap(mcap) => {
                resolve(&mut mcap.path);
                mcap.calib.iter_mut().for_each(resolve);
            }
            SourceConfig::Oakd(oakd) => oakd.calib.iter_mut().for_each(resolve),
            SourceConfig::Uvc(_) => {}
        }
        self.system.resolve_paths(base_dir);
    }
}

/// [`RunConfigError`] before the file it came from is known.
#[derive(Debug)]
pub enum Invalid {
    Parse(Box<ron::error::SpannedError>),
    System(ConfigError),
    Source(String),
}

impl Invalid {
    fn at(self, path: &Path) -> RunConfigError {
        let path = path.to_path_buf();
        match self {
            Self::Parse(source) => RunConfigError::Parse { path, source },
            Self::System(source) => RunConfigError::System { path, source },
            Self::Source(message) => RunConfigError::Source { path, message },
        }
    }
}

impl SourceConfig {
    /// Checks, before anything is opened, that the source can provide the
    /// selected sensors and has the settings it needs. `SlamSystem::build`
    /// checks the opened source's calibration again.
    fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        self.check_sensors(sensors)?;
        match self {
            Self::Euroc(_) | Self::Hilti(_) | Self::Mcap(_) => Ok(()),
            Self::Oakd(oakd) => {
                positive("fps", oakd.fps.into())?;
                positive("width", oakd.width.into())?;
                positive("height", oakd.height.into())
            }
            Self::Uvc(uvc) => {
                positive("width", uvc.width.into())?;
                positive("height", uvc.height.into())?;
                positive("fx", uvc.fx)?;
                positive("fy", uvc.fy)?;
                finite("cx", uvc.cx)?;
                finite("cy", uvc.cy)?;
                for (name, value) in [
                    ("k1", uvc.k1),
                    ("k2", uvc.k2),
                    ("p1", uvc.p1),
                    ("p2", uvc.p2),
                ] {
                    finite(name, value)?;
                }
                Ok(())
            }
        }
    }
}

impl SourceConfig {
    fn name(&self) -> &'static str {
        match self {
            Self::Euroc(_) => "Euroc",
            Self::Hilti(_) => "Hilti",
            Self::Mcap(_) => "Mcap",
            Self::Oakd(_) => "Oakd",
            Self::Uvc(_) => "Uvc",
        }
    }

    /// Stereo needs a rectifiable pair: EuRoC's own calibration, or a
    /// calibration file for MCAP and OAK-D. Only EuRoC supplies IMU data.
    fn check_sensors(&self, sensors: SensorSelection) -> Result<(), String> {
        let name = self.name();
        if sensors.cameras == CameraSelection::Stereo {
            match self {
                Self::Euroc(_) => {}
                Self::Mcap(McapConfig { calib: None, .. })
                | Self::Oakd(OakdConfig { calib: None, .. }) => {
                    return Err(format!("{name} with stereo cameras needs `calib`"));
                }
                Self::Mcap(_) | Self::Oakd(_) => {}
                Self::Hilti(_) | Self::Uvc(_) => {
                    return Err(format!("{name} provides monocular images only"));
                }
            }
        }
        if sensors.imu && !matches!(self, Self::Euroc(_)) {
            return Err(format!("{name} provides no IMU data"));
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests;
