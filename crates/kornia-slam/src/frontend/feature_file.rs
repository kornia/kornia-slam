//! Features exported offline, one file per sequence, read frame by frame.
//!
//! Little-endian layout, written by `scripts/export_clidd_features.py`:
//!
//! ```text
//! magic         8 bytes   "KSLMFEAT"
//! version       u32       1
//! dim           u32       floats per descriptor
//! n_frames      u32
//! n_frames times, by strictly increasing timestamp:
//!   timestamp   i64       image timestamp, ns
//!   n           u32       keypoints
//!   keypoints   f32[n][2] (x, y), pixels of the image the system is given,
//!                         integer coordinates at pixel centres
//!   scores      f32[n]    detector scores, unused here
//!   descriptors f32[n][dim]
//! ```

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::features::{CLIDD_DIM, Clidd, Features};

const MAGIC: &[u8; 8] = b"KSLMFEAT";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 8 + 4 + 4 + 4;
const FRAME_HEADER_BYTES: u64 = 8 + 4;
/// Largest gap between a frame's timestamp and the stored one it is given.
/// Timestamps reach the system as `f64` seconds, which round nanoseconds.
const MAX_TIMESTAMP_GAP_NS: u64 = 1_000_000;

/// A feature file that cannot be opened or read.
#[derive(Debug, thiserror::Error)]
#[error("feature file {}: {source}", path.display())]
pub struct FeatureFileError {
    pub path: PathBuf,
    #[source]
    pub source: io::Error,
}

/// An open feature file with its frame index; frames are read on demand.
pub(crate) struct FeatureFile {
    path: PathBuf,
    reader: BufReader<File>,
    frames: Vec<FrameEntry>,
}

#[derive(Debug, Clone, Copy)]
struct FrameEntry {
    timestamp_ns: i64,
    /// Byte offset of the frame's keypoints.
    offset: u64,
    keypoints: usize,
}

impl FeatureFile {
    /// Opens a file of [`Clidd`] descriptors and indexes its frames.
    pub(crate) fn open(path: &Path) -> Result<Self, FeatureFileError> {
        let error = |source| FeatureFileError {
            path: path.to_path_buf(),
            source,
        };
        let file = File::open(path).map_err(error)?;
        let len = file.metadata().map_err(error)?.len();
        let mut reader = BufReader::new(file);
        let frames = index(&mut reader, len).map_err(error)?;
        Ok(Self {
            path: path.to_path_buf(),
            reader,
            frames,
        })
    }

    /// The features of the stored frame nearest `timestamp_sec`, or `None`
    /// when none lies within 1 ms.
    pub(crate) fn features_near(
        &mut self,
        timestamp_sec: f64,
    ) -> Result<Option<Features<Clidd>>, FeatureFileError> {
        let Some(entry) = self.nearest(timestamp_sec) else {
            return Ok(None);
        };
        self.read_frame(entry)
            .map(Some)
            .map_err(|source| FeatureFileError {
                path: self.path.clone(),
                source,
            })
    }

    fn nearest(&self, timestamp_sec: f64) -> Option<FrameEntry> {
        let t = (timestamp_sec * 1e9).round() as i64;
        let after = self.frames.partition_point(|f| f.timestamp_ns < t);
        [after.checked_sub(1), Some(after)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.frames.get(i))
            .min_by_key(|f| f.timestamp_ns.abs_diff(t))
            .filter(|f| f.timestamp_ns.abs_diff(t) <= MAX_TIMESTAMP_GAP_NS)
            .copied()
    }

    fn read_frame(&mut self, entry: FrameEntry) -> io::Result<Features<Clidd>> {
        let n = entry.keypoints;
        self.reader.seek(SeekFrom::Start(entry.offset))?;
        let mut keypoints_xy = vec![[0.0f32; 2]; n];
        self.reader
            .read_exact(bytemuck::cast_slice_mut(&mut keypoints_xy))?;
        self.reader.seek_relative(4 * n as i64)?;
        let mut descriptors = vec![Clidd([0.0; CLIDD_DIM]); n];
        self.reader
            .read_exact(bytemuck::cast_slice_mut(&mut descriptors))?;
        from_little_endian(bytemuck::cast_slice_mut(&mut keypoints_xy));
        from_little_endian(bytemuck::cast_slice_mut(&mut descriptors));

        let mut values = keypoints_xy
            .iter()
            .flatten()
            .chain(descriptors.iter().flat_map(|d| d.0.iter()));
        if !values.all(|v| v.is_finite()) {
            return Err(invalid(format!(
                "non-finite value in the frame at {} ns",
                entry.timestamp_ns
            )));
        }
        Ok(Features {
            keypoints_xy,
            orientations: None,
            descriptors,
            octaves: vec![0; n],
        })
    }
}

/// Checks the header and records where each frame starts, skipping payloads.
fn index(reader: &mut BufReader<File>, file_len: u64) -> io::Result<Vec<FrameEntry>> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a kornia-slam feature file".into()));
    }
    let version = read_u32(reader)?;
    if version != VERSION {
        return Err(invalid(format!(
            "version {version}; this build reads {VERSION}"
        )));
    }
    let dim = read_u32(reader)? as u64;
    if dim != CLIDD_DIM as u64 {
        return Err(invalid(format!(
            "descriptors have {dim} floats; CLIDD has {CLIDD_DIM}"
        )));
    }
    let n_frames = read_u32(reader)?;

    let mut frames: Vec<FrameEntry> = Vec::new();
    let mut offset = HEADER_BYTES;
    for _ in 0..n_frames {
        let timestamp_ns = read_i64(reader)?;
        let keypoints = read_u32(reader)?;
        offset += FRAME_HEADER_BYTES;
        if frames
            .last()
            .is_some_and(|last| last.timestamp_ns >= timestamp_ns)
        {
            return Err(invalid(format!(
                "timestamps are not increasing at {timestamp_ns} ns"
            )));
        }
        frames.push(FrameEntry {
            timestamp_ns,
            offset,
            keypoints: keypoints as usize,
        });
        let payload = u64::from(keypoints) * (3 + dim) * 4;
        reader.seek_relative(payload as i64)?;
        offset += payload;
    }
    if offset != file_len {
        return Err(invalid(format!(
            "file has {file_len} bytes, its frames describe {offset}"
        )));
    }
    Ok(frames)
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn from_little_endian(values: &mut [f32]) {
    for value in values {
        *value = f32::from_le_bytes(value.to_ne_bytes());
    }
}

fn invalid(reason: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// One stored frame: timestamp (ns), keypoints, and descriptors.
    pub(crate) type StoredFrame = (i64, Vec<[f32; 2]>, Vec<Clidd>);

    /// A feature file in the temporary directory, removed on drop.
    pub(crate) struct TempFeatureFile(pub(crate) PathBuf);

    impl Drop for TempFeatureFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    pub(crate) fn encode(dim: u32, frames: &[StoredFrame]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        for word in [VERSION, dim, frames.len() as u32] {
            bytes.extend(word.to_le_bytes());
        }
        for (timestamp_ns, keypoints, descriptors) in frames {
            bytes.extend(timestamp_ns.to_le_bytes());
            bytes.extend((keypoints.len() as u32).to_le_bytes());
            let scores = vec![1.0f32; keypoints.len()];
            let floats = keypoints
                .iter()
                .flatten()
                .chain(&scores)
                .chain(descriptors.iter().flat_map(|d| d.0.iter()));
            for value in floats {
                bytes.extend(value.to_le_bytes());
            }
        }
        bytes
    }

    pub(crate) fn write(name: &str, bytes: &[u8]) -> TempFeatureFile {
        let path = std::env::temp_dir().join(format!(
            "kornia-slam-features-{}-{name}.kfeat",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        TempFeatureFile(path)
    }

    fn descriptor(value: f32) -> Clidd {
        Clidd([value; CLIDD_DIM])
    }

    fn two_frames() -> Vec<StoredFrame> {
        vec![
            (1_000_000_000, vec![[1.0, 2.0]], vec![descriptor(0.5)]),
            (
                1_050_000_000,
                vec![[3.0, 4.0], [5.0, 6.0]],
                vec![descriptor(0.25), descriptor(-0.25)],
            ),
        ]
    }

    #[test]
    fn frames_are_found_by_nearest_timestamp() {
        let file = write("nearest", &encode(CLIDD_DIM as u32, &two_frames()));
        let mut features = FeatureFile::open(&file.0).unwrap();

        let second = features.features_near(1.050_000_2).unwrap().unwrap();
        assert_eq!(second.keypoints_xy, [[3.0, 4.0], [5.0, 6.0]]);
        assert_eq!(second.descriptors, [descriptor(0.25), descriptor(-0.25)]);
        assert_eq!(second.octaves, [0, 0]);
        assert_eq!(second.orientations, None);

        let first = features.features_near(1.0).unwrap().unwrap();
        assert_eq!(first.keypoints_xy, [[1.0, 2.0]]);
        assert!(features.features_near(1.025).unwrap().is_none());
    }

    #[test]
    fn malformed_files_are_refused() {
        let frames = two_frames();
        let wrong_dim = write("dim", &encode(32, &frames[..0]));
        assert!(FeatureFile::open(&wrong_dim.0).is_err());

        let mut bytes = encode(CLIDD_DIM as u32, &frames);
        bytes.pop();
        let truncated = write("truncated", &bytes);
        assert!(FeatureFile::open(&truncated.0).is_err());

        let reversed: Vec<StoredFrame> = frames.into_iter().rev().collect();
        let unordered = write("unordered", &encode(CLIDD_DIM as u32, &reversed));
        assert!(FeatureFile::open(&unordered.0).is_err());
    }

    #[test]
    fn non_finite_values_are_refused_when_read() {
        let frames = vec![(0, vec![[f32::NAN, 0.0]], vec![descriptor(0.0)])];
        let file = write("nan", &encode(CLIDD_DIM as u32, &frames));
        let mut features = FeatureFile::open(&file.0).unwrap();
        assert!(features.features_near(0.0).is_err());
    }
}
