use std::path::PathBuf;

/// Error type used by dataset readers.
#[derive(thiserror::Error, Debug)]
pub enum DatasetError {
    /// Generic I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse failure with contextual message.
    #[error("parse error: {0}")]
    Parse(String),

    /// Referenced file does not exist.
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),
}

/// One dataset image sample.
#[derive(Debug, Clone)]
pub struct DatasetSample {
    /// Timestamp in seconds.
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Path to the image file.
    pub image_path: PathBuf,
}