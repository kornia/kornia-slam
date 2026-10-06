use std::path::{Path, PathBuf};

use super::config::{PIPELINE_CONFIG_VERSION, PipelineConfig, PipelineDefinition};
use super::validation::ConfigError;

/// Failure to read, parse or serialize a RON pipeline configuration.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("failed to read pipeline config {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse pipeline config: {0}")]
    Parse(#[from] ron::error::SpannedError),
    #[error("failed to serialize pipeline config: {0}")]
    Serialize(#[from] ron::Error),
    #[error(transparent)]
    Invalid(#[from] ConfigError),
}

/// Reads only the version, so a file written for another schema version is
/// reported as such rather than as unknown fields.
#[derive(serde::Deserialize)]
struct VersionProbe {
    version: u32,
}

impl PipelineConfig {
    /// Loads and validates a RON file. Relative resource paths in it are
    /// resolved against the file's directory.
    pub fn from_ron_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| LoadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_ron_str(&text, path.parent().unwrap_or(Path::new("")))
    }

    /// Parses and validates RON text. Omitted fields take their defaults;
    /// relative resource paths are resolved against `base_dir`.
    pub fn from_ron_str(text: &str, base_dir: &Path) -> Result<Self, LoadError> {
        let VersionProbe { version } = ron::from_str(text)?;
        if version != PIPELINE_CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: version,
                supported: PIPELINE_CONFIG_VERSION,
            }
            .into());
        }
        let mut config: Self = ron::from_str(text)?;
        config.validate()?;
        config.resolve_paths(base_dir);
        Ok(config)
    }

    /// Serializes to pretty-printed RON that [`PipelineConfig::from_ron_str`] reads back.
    pub fn to_ron_string(&self) -> Result<String, LoadError> {
        Ok(ron::ser::to_string_pretty(
            self,
            ron::ser::PrettyConfig::default(),
        )?)
    }

    fn resolve_paths(&mut self, base_dir: &Path) {
        let PipelineDefinition::OrbSlam(orb) = &mut self.pipeline;
        if let Some(vocabulary) = orb.loop_closing.vocabulary_mut()
            && vocabulary.is_relative()
        {
            *vocabulary = base_dir.join(&*vocabulary);
        }
    }
}
