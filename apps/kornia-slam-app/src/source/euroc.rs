//! EuRoC MAV dataset as a [`FrameSource`].

use std::path::{Path, PathBuf};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_io::png::read_image_png_mono8;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;
use serde::Deserialize;

use super::{
    FrameItem, FrameSource, OpenedSource, SourceError, dataset_summary, non_empty, rectify_pair,
    resolve,
};
use crate::datasets::EurocDataset;
use crate::datasets::euroc::{GroundTruthPose, ImuSample};
use crate::datasets::{StereoRectifier, rectifier_from_euroc};

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

impl EurocConfig {
    pub(super) fn resolve_paths(&mut self, base_dir: &Path) {
        resolve(&mut self.data, base_dir);
    }

    pub(super) fn open(&self, stereo: bool) -> Result<OpenedSource, SourceError> {
        let source = EurocSource::open(&self.data, self.start_frame, self.max_frames, stereo)?;
        Ok(OpenedSource {
            summary: Some(dataset_summary(
                source.dataset_len(),
                self.start_frame,
                source.n_frames_hint(),
            )),
            ground_truth: non_empty(source.ground_truth_poses_cloned()),
            source: Box::new(source),
        })
    }
}

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
    /// samples retain their absolute index in `FrameItem::idx`. With `stereo`,
    /// the left+right pair is rectified and yielded together; the dataset must
    /// then have a usable right camera.
    pub fn open(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
        stereo: bool,
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
            Some(
                rectifier_from_euroc(&dataset.left_calibration, &right)
                    .map_err(SourceError::other)?,
            )
        } else {
            None
        };

        // IMU samples are yielded whenever the dataset ships them (`imu0`);
        // skip those preceding the first yielded camera frame.
        let with_imu = dataset.has_imu();
        let imu_cursor = if with_imu && start > 0 {
            let boundary_ts = dataset
                .left_samples
                .get(start - 1)
                .map(|sample| sample.timestamp_sec)
                .unwrap_or(f64::INFINITY);
            dataset
                .imu_samples
                .partition_point(|sample| sample.timestamp_sec <= boundary_ts)
        } else {
            0
        };

        Ok(Self {
            dataset,
            cursor: start,
            start,
            end,
            rectifier,
            with_imu,
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
    fn rig(&self) -> SensorRig {
        let mut rig = SensorRig::new(self.camera());
        if let Some(rect) = &self.rectifier {
            rig = rig.with_stereo_baseline(rect.baseline());
        }
        if let Some(t_bc) = self.camera_to_body() {
            rig = rig.with_imu(t_bc);
        }
        rig
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.end - self.start)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        let Some(idx) = self.advance() else {
            return Ok(None);
        };
        let sample = &self.dataset.left_samples[idx];
        let timestamp_sec = sample.timestamp_sec;
        let left_raw = read_image_png_mono8(&sample.image_path)
            .map_err(SourceError::other)?
            .into_inner();

        let (image, right_image) = match &self.rectifier {
            Some(rect) => {
                let right_path = &self
                    .dataset
                    .right_sample_for(idx)
                    .ok_or_else(|| SourceError::other("left frame has no right partner"))?
                    .image_path;
                let right_raw = read_image_png_mono8(right_path)
                    .map_err(SourceError::other)?
                    .into_inner();
                let (left, right) = rectify_pair(rect, &left_raw, &right_raw)?;
                (left, Some(right))
            }
            None => (left_raw, None),
        };
        let imu_samples = self.imu_samples_until(timestamp_sec);

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
    fn camera(&self) -> PinholeCamera {
        match &self.rectifier {
            Some(rect) => rect.rectified_camera(),
            None => self.dataset.camera(),
        }
    }

    /// `T_BC` for the camera returned by [`Self::camera`], when IMU data is present.
    fn camera_to_body(&self) -> Option<Pose3d> {
        if !self.with_imu {
            return None;
        }
        let (rotation, translation) = self.dataset.left_calibration.body_from_camera();
        match &self.rectifier {
            // The rectified virtual camera is the raw cam0 rotated by the
            // rectifying rotation (p_rect = R_rect · p_cam0), so
            // T_B,rect = T_BS · R_rectᵀ; the translation is unchanged.
            Some(rect) => {
                let r_rect_t = Mat3F64(*rect.left_rectifying_rotation().transpose());
                Some(Pose3d::from_rt(rotation * r_rect_t, translation))
            }
            None => Some(Pose3d::from_rt(rotation, translation)),
        }
    }

    /// Next left frame to yield: any frame in mono mode, and in stereo mode
    /// only frames with a synchronised right image. IMU samples of a skipped
    /// frame are delivered with the next yielded one.
    fn advance(&mut self) -> Option<usize> {
        while self.cursor < self.end {
            let idx = self.cursor;
            self.cursor += 1;
            if self.rectifier.is_none() || self.dataset.right_sample_for(idx).is_some() {
                return Some(idx);
            }
        }
        None
    }

    fn imu_samples_until(&mut self, timestamp_sec: f64) -> Vec<ImuMeasurement> {
        if !self.with_imu {
            return Vec::new();
        }

        let start = self.imu_cursor;
        let rel_end = self.dataset.imu_samples[start..]
            .partition_point(|sample| sample.timestamp_sec <= timestamp_sec);
        let end = start + rel_end;
        self.imu_cursor = end;
        self.dataset.imu_samples[start..end]
            .iter()
            .map(imu_measurement)
            .collect()
    }
}

fn imu_measurement(sample: &ImuSample) -> ImuMeasurement {
    ImuMeasurement {
        timestamp: sample.timestamp_sec,
        gyro: Vec3F64::new(sample.gyro[0], sample.gyro[1], sample.gyro[2]),
        accel: Vec3F64::new(sample.accel[0], sample.accel[1], sample.accel[2]),
    }
}
