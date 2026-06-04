//! KITTI odometry sequence as a [`FrameSource`].

use std::path::Path;

use kornia_3d::camera::PinholeCamera;
use kornia_io::png::read_image_png_mono8;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::KittiDataset;

/// Reads grayscale PNG frames from a KITTI sequence (`image_0/`, `calib.txt`, `times.txt`).
pub struct KittiSource {
    dataset: KittiDataset,
    cursor: usize,
    start: usize,
    end: usize,
}

impl KittiSource {
    /// Opens the dataset and configures the iteration window.
    ///
    /// `max_frames == 0` means "until the dataset is exhausted". `start_frame`
    /// is the index into the samples of the first frame to yield; later frames
    /// retain their absolute index in `FrameItem::idx`.
    pub fn open(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        let dataset = KittiDataset::open(root).map_err(SourceError::other)?;
        let n = dataset.samples().len();
        let start = start_frame.min(n);
        let end = if max_frames > 0 {
            (start + max_frames).min(n)
        } else {
            n
        };

        Ok(Self {
            dataset,
            cursor: start,
            start,
            end,
        })
    }

    /// Total sample count in the dataset (ignoring start/max).
    pub fn dataset_len(&self) -> usize {
        self.dataset.samples().len()
    }
}

impl FrameSource for KittiSource {
    fn camera(&self) -> PinholeCamera {
        self.dataset.camera()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.end - self.start)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.cursor >= self.end {
            return Ok(None);
        }
        let idx = self.cursor;
        let sample = &self.dataset.cam0_samples[idx];
        let image = read_image_png_mono8(&sample.image_path)
            .map_err(SourceError::other)?
            .into_inner();

        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec: sample.timestamp_sec,
            image,
            right_image: None,
        }))
    }
}
