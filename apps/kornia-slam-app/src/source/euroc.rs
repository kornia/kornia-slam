//! EuRoC MAV dataset as a [`FrameSource`].

use std::path::{Path, PathBuf};

use kornia_3d::camera::{Camera, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_io::png::read_image_png_mono8;
use kornia_sensors::imu::ImuMeasurement;
use kornia_sensors::{ImuCalibration, SensorRig};
use kornia_slam::CameraSelection;
use serde::Deserialize;

use super::{
    FrameItem, FrameSource, OpenedSource, PairRectifier, Rectification, SourceError,
    dataset_summary, non_empty, resolve, rig_from_body_extrinsics,
};
use crate::datasets::EurocDataset;
use crate::datasets::euroc::{GroundTruthPose, ImuSample};
use crate::datasets::rectifier_from_euroc;

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

    pub(super) fn open(
        &self,
        cameras: CameraSelection,
        rectification: Rectification,
    ) -> Result<OpenedSource, SourceError> {
        let source = EurocSource::open(
            &self.data,
            self.start_frame,
            self.max_frames,
            cameras,
            rectification,
        )?;
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

/// Reads left-camera (and optionally rectified left+right, or raw left+right
/// as a rig) PNG frames from an EuRoC dataset in order.
pub struct EurocSource {
    dataset: EurocDataset,
    cursor: usize,
    end: usize,
    /// Frames the window yields.
    n_frames: usize,
    /// When `Some`, the source rectifies the left+right pair and yields stereo.
    rectifier: Option<PairRectifier>,
    /// When `Some`, the source yields raw cam0 + cam1 as this two-camera rig.
    rig: Option<SensorRig>,
    /// Left image size; rectified images keep it.
    size: ImageSize,
    with_imu: bool,
    imu_cursor: usize,
}

impl EurocSource {
    /// Opens the dataset and configures the iteration window.
    ///
    /// `max_frames == 0` means "until the dataset is exhausted". `start_frame`
    /// is the index into the left-camera samples of the first sample to yield; later
    /// samples retain their absolute index in `FrameItem::idx`. With `Stereo`,
    /// the left+right pair is rectified as `rectification` says and yielded
    /// together; with `Rig`, both raw images are yielded as a two-camera rig.
    /// Either needs a usable right camera.
    pub fn open(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
        cameras: CameraSelection,
        rectification: Rectification,
    ) -> Result<Self, SourceError> {
        let dataset = EurocDataset::open(root).map_err(SourceError::other)?;
        // Rig, like Stereo, skips left frames without a right partner, so both
        // images belong to one instant.
        let paired = cameras != CameraSelection::Mono;
        let n = dataset.samples().len();
        let start = start_frame.min(n);
        let yields = |idx: usize| yields_frame(&dataset, paired, idx);
        let end = window_end(start, n, max_frames, yields);
        let n_frames = (start..end).filter(|&idx| yields(idx)).count();

        let rectifier = if cameras == CameraSelection::Stereo {
            if !dataset.is_stereo() {
                return Err(SourceError::other(
                    "stereo requested but dataset has no usable right camera",
                ));
            }
            let right = dataset
                .right_calibration
                .expect("is_stereo() guarantees right-camera calibration");
            let rectifier = rectifier_from_euroc(&dataset.left_calibration, &right)
                .map_err(SourceError::other)?;
            Some(PairRectifier::new(rectifier, rectification)?)
        } else {
            None
        };

        let size = image_size(&dataset, start)?;
        let rig = match cameras {
            CameraSelection::Rig => Some(raw_rig(&dataset, size)?),
            CameraSelection::Mono | CameraSelection::Stereo => None,
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
            end,
            n_frames,
            rectifier,
            rig,
            size,
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
        if let Some(rig) = &self.rig {
            return rig.clone();
        }
        let mut rig = match &self.rectifier {
            Some(rect) => SensorRig::rectified_stereo(self.camera(), self.size, rect.baseline()),
            None => SensorRig::mono(self.camera(), self.size),
        };
        if let Some(t_bc) = self.camera_to_body() {
            rig = rig.with_imu(t_bc);
        }
        rig
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.n_frames)
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

        let read_right = || -> Result<_, SourceError> {
            let right_path = &self
                .dataset
                .right_sample_for(idx)
                .ok_or_else(|| SourceError::other("left frame has no right partner"))?
                .image_path;
            Ok(read_image_png_mono8(right_path)
                .map_err(SourceError::other)?
                .into_inner())
        };
        let (image, right_image, rig_images) = match &self.rectifier {
            Some(rect) => {
                let (left, right) = rect.rectify(&left_raw, &read_right()?)?;
                (left, Some(right), Vec::new())
            }
            None if self.rig.is_some() => (left_raw, None, vec![read_right()?]),
            None => (left_raw, None, Vec::new()),
        };
        let imu_samples = self.imu_samples_until(timestamp_sec);

        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            rig_images,
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
                let r_rect_t = rect.left_rectifying_rotation().transpose();
                Some(Pose3d::from_rt(rotation * r_rect_t, translation))
            }
            None => Some(Pose3d::from_rt(rotation, translation)),
        }
    }

    /// Next left frame to yield. IMU samples of a skipped frame are delivered
    /// with the next yielded one.
    fn advance(&mut self) -> Option<usize> {
        while self.cursor < self.end {
            let idx = self.cursor;
            self.cursor += 1;
            if yields_frame(
                &self.dataset,
                self.rectifier.is_some() || self.rig.is_some(),
                idx,
            ) {
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

/// The `sensor.yaml` resolution of the left camera, or, when the file gives
/// none, the size of the left image at `start`, the first of the window.
fn image_size(dataset: &EurocDataset, start: usize) -> Result<ImageSize, SourceError> {
    let calibration = &dataset.left_calibration;
    if calibration.width > 0 && calibration.height > 0 {
        return Ok(ImageSize {
            width: calibration.width,
            height: calibration.height,
        });
    }
    let first = dataset
        .left_samples
        .get(start)
        .or_else(|| dataset.left_samples.first())
        .ok_or_else(|| SourceError::other("the dataset has no left images"))?;
    Ok(read_image_png_mono8(&first.image_path)
        .map_err(SourceError::other)?
        .size())
}

/// Whether the left sample `idx` is yielded: any sample in mono mode, and in
/// paired (stereo or rig) mode only one with a synchronised right image.
fn yields_frame(dataset: &EurocDataset, paired: bool, idx: usize) -> bool {
    !paired || dataset.right_sample_for(idx).is_some()
}

/// Raw cam0 (primary, `left_size`) and cam1 as a rig, each with its own radtan
/// intrinsics; `T_PC` comes from the two `T_BS` (body = imu0), and the IMU
/// refers to cam0 when the dataset ships one.
fn raw_rig(dataset: &EurocDataset, left_size: ImageSize) -> Result<SensorRig, SourceError> {
    let right = match &dataset.right_calibration {
        Some(right) if dataset.is_stereo() => right,
        _ => {
            return Err(SourceError::other(
                "rig requested but dataset has no usable right camera",
            ));
        }
    };
    if right.width == 0 || right.height == 0 {
        return Err(SourceError::other(
            "rig requested but the cam1 sensor.yaml has no resolution",
        ));
    }
    let right_size = ImageSize {
        width: right.width,
        height: right.height,
    };
    let cameras = [(&dataset.left_calibration, left_size), (right, right_size)].map(|(c, size)| {
        let (rotation, translation) = c.body_from_camera();
        (
            Camera::from(c.to_pinhole_camera()),
            size,
            Pose3d::from_rt(rotation, translation),
        )
    });
    let imu = dataset.has_imu().then(|| ImuCalibration::new(cameras[0].2));
    rig_from_body_extrinsics(&cameras, 0, imu)
}

/// End of the window from `start` that holds `max_frames` yielded samples
/// (0 = all of them), out of `n`.
fn window_end(start: usize, n: usize, max_frames: usize, yields: impl Fn(usize) -> bool) -> usize {
    match max_frames.checked_sub(1) {
        Some(last) => (start..n)
            .filter(|&idx| yields(idx))
            .nth(last)
            .map_or(n, |idx| idx + 1),
        None => n,
    }
}

fn imu_measurement(sample: &ImuSample) -> ImuMeasurement {
    ImuMeasurement {
        timestamp: sample.timestamp_sec,
        gyro: Vec3F64::new(sample.gyro[0], sample.gyro[1], sample.gyro[2]),
        accel: Vec3F64::new(sample.accel[0], sample.accel[1], sample.accel[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::window_end;
    use crate::datasets::euroc::EurocCameraCalibration;
    use kornia_3d::camera::Camera;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;

    /// MH_01 `cam0` / `cam1` `sensor.yaml`.
    fn mh01_cameras() -> (EurocCameraCalibration, EurocCameraCalibration) {
        let cam0 = EurocCameraCalibration {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
            width: 752,
            height: 480,
            t_bs: [
                0.0148655429818,
                -0.999880929698,
                0.00414029679422,
                -0.0216401454975,
                0.999557249008,
                0.0149672133247,
                0.025715529948,
                -0.064676986768,
                -0.0257744366974,
                0.00375618835797,
                0.999660727178,
                0.00981073058949,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        };
        let cam1 = EurocCameraCalibration {
            fx: 457.587,
            fy: 456.134,
            cx: 379.999,
            cy: 255.238,
            k1: -0.28368365,
            k2: 0.07451284,
            p1: -0.00010473,
            p2: -3.55590700e-05,
            width: 752,
            height: 480,
            t_bs: [
                0.0125552670891,
                -0.999755099723,
                0.0182237714554,
                -0.0198435579556,
                0.999598781151,
                0.0130119051815,
                0.0251588363115,
                0.0453689425024,
                -0.0253898008918,
                0.0179005838253,
                0.999517347078,
                0.00786212447038,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        };
        (cam0, cam1)
    }

    #[test]
    fn mh01_rig_puts_cam1_eleven_cm_to_the_right_with_its_own_distortion() {
        let (cam0, cam1) = mh01_cameras();
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        let cameras = [cam0, cam1].map(|c| {
            let (rotation, translation) = c.body_from_camera();
            (
                Camera::from(c.to_pinhole_camera()),
                size,
                Pose3d::from_rt(rotation, translation),
            )
        });
        let rig = super::rig_from_body_extrinsics(&cameras, 0, None).unwrap();

        let cameras = rig.cameras();
        assert_eq!(cameras.len(), 2);
        assert_eq!(cameras[0].primary_from_camera, Pose3d::IDENTITY);
        let t = cameras[1].primary_from_camera.translation;
        let norm = (t.x * t.x + t.y * t.y + t.z * t.z).sqrt();
        assert!((norm - 0.1101).abs() <= 0.0005, "|t| = {norm}");
        assert!(t.x > 0.109, "t = {t:?}");
        assert_eq!(cameras[0].pinhole().k1, -0.28340811);
        assert_eq!(cameras[1].pinhole().k1, -0.28368365);
        assert_eq!(rig.rectified_stereo_baseline_m(), None);
        for camera in cameras {
            assert_eq!((camera.size.width, camera.size.height), (752, 480));
            assert!(!camera.is_fisheye());
        }
    }

    #[test]
    fn a_bounded_window_counts_only_yielded_samples() {
        let paired = |idx: usize| idx != 0 && idx != 3;

        assert_eq!(window_end(0, 6, 1, paired), 2);
        assert_eq!(window_end(0, 6, 3, paired), 5);
        assert_eq!(window_end(0, 6, 10, paired), 6);
        assert_eq!(window_end(2, 6, 0, paired), 6);
    }
}
