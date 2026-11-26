//! Run files: the source to read and the SLAM system to run on it.
//!
//! ```ron
//! (
//!     source: Euroc((data: "../data/euroc/MH_01_easy", max_frames: 500)),
//!     system: (version: 1, sensors: (cameras: Stereo, imu: true)),
//! )
//! ```
//!
//! Relative paths, in either section, are resolved against the run file's
//! directory.

use std::path::{Path, PathBuf};

use kornia_sensors::SensorRig;
use kornia_sensors::rig_calibration::{RigCalibration, RigCalibrationError};
use kornia_slam::SystemConfig;
use kornia_slam::mapping::global_vi_ba::OfflineBaConfig;
use kornia_slam::system::ConfigError;
use serde::Deserialize;

use crate::source::{Rectification, SourceConfig};

/// One run: where the data comes from and how it is processed.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub source: SourceConfig,
    /// Omitted, the default monocular pipeline runs.
    #[serde(default)]
    pub system: SystemConfig,
    /// A rig calibration file (RON) whose camera geometry, and IMU extrinsic
    /// and noise, replace the source's; relative to the run file. Omitted, the
    /// source's own calibration is used.
    #[serde(default)]
    pub rig_calibration: Option<PathBuf>,
    /// Settings of the `offline-ba` subcommand; the online run never reads them.
    #[serde(default)]
    pub offline_ba: OfflineBaConfig,
    /// How a raw stereo pair is rectified.
    #[serde(default)]
    pub rectification: Rectification,
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

/// A rig calibration file that cannot replace the source's calibration.
#[derive(Debug, thiserror::Error)]
pub enum RigCalibrationFileError {
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
    #[error("{}: {in_file} cameras, the source's rig has {in_source}", path.display())]
    CameraCount {
        path: PathBuf,
        in_file: usize,
        in_source: usize,
    },
    #[error("{}: {source}", path.display())]
    Invalid {
        path: PathBuf,
        source: RigCalibrationError,
    },
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
    kornia_slam::system::SYSTEM_CONFIG_VERSION
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
        SystemConfig::check_version(probe.system.version).map_err(Invalid::System)?;
        let run: Self = ron.from_str(text).map_err(parse_error)?;
        run.system.validate().map_err(Invalid::System)?;
        run.source
            .validate(run.system.sensors)
            .map_err(Invalid::Source)?;
        Ok(run)
    }

    fn resolve_paths(&mut self, base_dir: &Path) {
        self.source.resolve_paths(base_dir);
        self.system.resolve_paths(base_dir);
        if let Some(path) = &mut self.rig_calibration
            && path.is_relative()
        {
            *path = base_dir.join(&*path);
        }
    }

    /// `rig` with the run's rig calibration file applied; `rig` unchanged
    /// without one. The file must describe as many cameras as `rig` has.
    pub fn apply_rig_calibration(
        &self,
        rig: SensorRig,
    ) -> Result<SensorRig, RigCalibrationFileError> {
        let Some(path) = &self.rig_calibration else {
            return Ok(rig);
        };
        let text =
            std::fs::read_to_string(path).map_err(|source| RigCalibrationFileError::Read {
                path: path.clone(),
                source,
            })?;
        let calibration: RigCalibration = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
            .from_str(&text)
            .map_err(|error| RigCalibrationFileError::Parse {
                path: path.clone(),
                source: Box::new(error),
            })?;
        let in_source = rig.cameras.len().max(1);
        if calibration.cameras.len() != in_source {
            return Err(RigCalibrationFileError::CameraCount {
                path: path.clone(),
                in_file: calibration.cameras.len(),
                in_source,
            });
        }
        calibration
            .apply_to(rig)
            .map_err(|source| RigCalibrationFileError::Invalid {
                path: path.clone(),
                source,
            })
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

#[cfg(test)]
mod tests;
