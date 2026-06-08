//! EuRoC MAV dataset as a [`FrameSource`].

use std::path::Path;

use kornia_3d::camera::PinholeCamera;
use kornia_io::png::read_image_png_mono8;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::EurocDataset;
use crate::datasets::StereoRectifier;
use crate::datasets::euroc::{GroundTruthPose, ImuSample};
/// Reads left-camera (and optionally rectified left+right) PNG frames from an
/// EuRoC dataset in order.
pub struct EurocSource {
    dataset: EurocDataset,
    cursor: usize,
    start: usize,
    end: usize,
    /// When `Some`, the source rectifies the left+right pair and yields stereo.
    rectifier: Option<StereoRectifier>,
    with_imu: bool,
    imu_cursor: usize,
}

impl EurocSource {
    /// Opens the dataset and configures the iteration window.
    ///
    /// `max_frames == 0` means "until the dataset is exhausted". `start_frame`
    /// is the index into the left-camera samples of the first sample to yield; later
    /// samples retain their absolute index in `FrameItem::idx`.
    pub fn open(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        Self::open_inner(root, start_frame, max_frames, false, false)
    }

    /// Like [`Self::open`], but rectifies the left+right pair and yields stereo
    /// pairs. Errors if the dataset has no usable right camera.
    pub fn open_stereo(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        Self::open_inner(root, start_frame, max_frames, true, false)
    }

    pub fn open_imu(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        Self::open_inner(root, start_frame, max_frames, false, true)
    }

    pub fn open_imu_stereo(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        Self::open_inner(root, start_frame, max_frames, true, true)
    }

    fn open_inner(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
        stereo: bool,
        imu: bool,
    ) -> Result<Self, SourceError> {
        let dataset = EurocDataset::open(root).map_err(SourceError::other)?;
        let n = dataset.samples().len();
        let start = start_frame.min(n);
        let end = if max_frames > 0 {
            (start + max_frames).min(n)
        } else {
            n
        };

        let rectifier = if stereo {
            if !dataset.is_stereo() {
                return Err(SourceError::other(
                    "stereo requested but dataset has no usable right camera",
                ));
            }
            let right = dataset
                .right_calibration
                .expect("is_stereo() guarantees right-camera calibration");
            Some(StereoRectifier::new(&dataset.left_calibration, &right))
        } else {
            None
        };

        if imu && !dataset.is_imu() {
            return Err(SourceError::other(
                "IMU requested but dataset has no IMU samples",
            ));
        }

        let imu_cursor = if imu {
            if start == 0 {
                0
            } else {
                let boundary_ts = dataset
                    .left_samples
                    .get(start - 1)
                    .map(|sample| sample.timestamp_sec)
                    .unwrap_or(f64::INFINITY);
                dataset
                    .imu_samples
                    .partition_point(|sample| sample.timestamp_sec <= boundary_ts)
            }
        } else {
            0
        };

        Ok(Self {
            dataset,
            cursor: start,
            start,
            end,
            rectifier,
            with_imu: imu,
            imu_cursor,
        })
    }

    pub fn ground_truth_poses_cloned(&self) -> Vec<GroundTruthPose> {
        self.dataset.ground_truth().to_vec()
    }

    /// Total sample count in the dataset (ignoring start/max).
    pub fn dataset_len(&self) -> usize {
        self.dataset.samples().len()
    }
}

impl FrameSource for EurocSource {
    fn camera(&self) -> PinholeCamera {
        match &self.rectifier {
            Some(rect) => rect.rectified_camera(),
            None => self.dataset.camera(),
        }
    }

    fn stereo_bf(&self) -> Option<f64> {
        self.rectifier.as_ref().map(|r| r.bf())
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.end - self.start)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.cursor >= self.end {
            return Ok(None);
        }
        let idx = self.cursor;
        let sample = &self.dataset.left_samples[idx];
        let timestamp_sec = sample.timestamp_sec;
        let left_raw = read_image_png_mono8(&sample.image_path)
            .map_err(SourceError::other)?
            .into_inner();

        let (image, right_image) = match &self.rectifier {
            Some(rect) => {
                let right_path = &self.dataset.right_samples[idx].image_path;
                let right_raw = read_image_png_mono8(right_path)
                    .map_err(SourceError::other)?
                    .into_inner();
                (
                    rect.rectify_left(&left_raw),
                    Some(rect.rectify_right(&right_raw)),
                )
            }
            None => (left_raw, None),
        };
        let imu_samples = self.imu_samples_until(timestamp_sec);

        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            imu_samples,
        }))
    }
}

impl EurocSource {
    fn imu_samples_until(&mut self, timestamp_sec: f64) -> Vec<ImuSample> {
        if !self.with_imu {
            return Vec::new();
        }

        let start = self.imu_cursor;
        let rel_end = self.dataset.imu_samples[start..]
            .partition_point(|sample| sample.timestamp_sec <= timestamp_sec);
        let end = start + rel_end;
        self.imu_cursor = end;
        self.dataset.imu_samples[start..end].to_vec()
    }
}
