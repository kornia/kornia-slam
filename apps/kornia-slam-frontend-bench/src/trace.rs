//! Reader for the directory `kornia-slam --trace-out` writes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{0}: malformed")]
    Malformed(String),
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
}

pub struct TraceCamera {
    pub camera: PinholeCamera,
    pub size: ImageSize,
    pub stereo_baseline_m: f64,
}

/// One local-map point on one frame.
#[derive(Clone)]
pub struct LocalPoint {
    pub map_point_idx: usize,
    pub position: Vec3F64,
    pub min_distance: f64,
    pub max_distance: f64,
    pub descriptor: [u8; 32],
}

/// `(pixel, map_point_idx)` of one track at frame start.
pub type TraceTrack = ([f32; 2], Option<usize>);

pub struct TraceFrame {
    pub frame_idx: usize,
    pub trusted_prediction: bool,
    pub keyframe: bool,
    pub candidate_pose: Pose3d,
    pub pose: Pose3d,
}

pub struct Trace {
    dir: PathBuf,
    pub cameras: Vec<TraceCamera>,
    pub frames: Vec<TraceFrame>,
    /// `(pixel, map_point_idx)` per traced frame.
    pub tracks: BTreeMap<usize, Vec<TraceTrack>>,
    pub local_map: BTreeMap<usize, Vec<LocalPoint>>,
}

const FRAME_COLUMNS: usize = 31;

fn read(path: &Path) -> Result<String, TraceError> {
    std::fs::read_to_string(path).map_err(|source| TraceError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().skip(1).map(|line| line.split(',').collect())
}

fn num<T: std::str::FromStr>(fields: &[&str], i: usize, what: &str) -> Result<T, TraceError> {
    fields
        .get(i)
        .and_then(|f| f.trim().parse().ok())
        .ok_or_else(|| TraceError::Malformed(format!("{what} column {i}")))
}

fn pose_at(fields: &[&str], start: usize, what: &str) -> Result<Pose3d, TraceError> {
    let mut v = [0.0f64; 12];
    for (k, slot) in v.iter_mut().enumerate() {
        *slot = num(fields, start + k, what)?;
    }
    // Row-major in the file; glam matrices are column-major.
    let rotation =
        Mat3F64::from_cols_array(&[v[0], v[3], v[6], v[1], v[4], v[7], v[2], v[5], v[8]]);
    Ok(Pose3d::new(rotation, Vec3F64::new(v[9], v[10], v[11])))
}

impl Trace {
    pub fn load(dir: &Path) -> Result<Self, TraceError> {
        let mut cameras = Vec::new();
        for f in rows(&read(&dir.join("rig.csv"))?) {
            cameras.push(TraceCamera {
                camera: PinholeCamera {
                    fx: num(&f, 3, "rig")?,
                    fy: num(&f, 4, "rig")?,
                    cx: num(&f, 5, "rig")?,
                    cy: num(&f, 6, "rig")?,
                    k1: num(&f, 7, "rig")?,
                    k2: num(&f, 8, "rig")?,
                    p1: num(&f, 9, "rig")?,
                    p2: num(&f, 10, "rig")?,
                },
                size: ImageSize {
                    width: num(&f, 1, "rig")?,
                    height: num(&f, 2, "rig")?,
                },
                stereo_baseline_m: num(&f, 11, "rig")?,
            });
        }
        let mut frames = Vec::new();
        for f in rows(&read(&dir.join("frames.csv"))?) {
            // A trace still being written ends in a partial row.
            if f.len() < FRAME_COLUMNS {
                continue;
            }
            frames.push(TraceFrame {
                frame_idx: num(&f, 0, "frames")?,
                trusted_prediction: num::<u8>(&f, 2, "frames")? == 1,
                keyframe: num::<u8>(&f, 4, "frames")? == 1,
                candidate_pose: pose_at(&f, 7, "frames")?,
                pose: pose_at(&f, 19, "frames")?,
            });
        }
        let mut tracks: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for f in rows(&read(&dir.join("tracks.csv"))?) {
            if f.len() < 5 {
                continue;
            }
            let mp: i64 = num(&f, 4, "tracks")?;
            tracks.entry(num(&f, 0, "tracks")?).or_default().push((
                [num(&f, 2, "tracks")?, num(&f, 3, "tracks")?],
                usize::try_from(mp).ok(),
            ));
        }
        let path = dir.join("local_map.bin");
        let bytes = std::fs::read(&path).map_err(|source| TraceError::Io { path, source })?;
        let mut local_map: BTreeMap<usize, Vec<LocalPoint>> = BTreeMap::new();
        for record in bytes.chunks_exact(60) {
            let u32_at = |o: usize| {
                u32::from_le_bytes([record[o], record[o + 1], record[o + 2], record[o + 3]])
            };
            let f32_at = |o: usize| f32::from_bits(u32_at(o)) as f64;
            let mut descriptor = [0u8; 32];
            descriptor.copy_from_slice(&record[28..60]);
            local_map
                .entry(u32_at(0) as usize)
                .or_default()
                .push(LocalPoint {
                    map_point_idx: u32_at(4) as usize,
                    position: Vec3F64::new(f32_at(8), f32_at(12), f32_at(16)),
                    min_distance: f32_at(20),
                    max_distance: f32_at(24),
                    descriptor,
                });
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            cameras,
            frames,
            tracks,
            local_map,
        })
    }

    /// Images per frame: two for a rectified stereo pair, one per rig camera.
    pub fn n_images(&self) -> usize {
        match self.cameras.as_slice() {
            [only] if only.stereo_baseline_m > 0.0 => 2,
            cameras => cameras.len(),
        }
    }

    pub fn image(&self, frame_idx: usize, camera: usize) -> Result<Image<u8, 1>, TraceError> {
        let size = self
            .cameras
            .get(camera)
            .or(self.cameras.first())
            .map(|c| c.size)
            .ok_or_else(|| TraceError::Malformed("rig.csv has no camera".into()))?;
        let path = self.dir.join(format!("images/{frame_idx}_{camera}.u8"));
        let data = std::fs::read(&path).map_err(|source| TraceError::Io { path, source })?;
        Ok(Image::new(size, data)?)
    }
}
