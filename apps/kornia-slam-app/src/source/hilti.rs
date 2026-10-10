//! Hilti-Trimble SLAM Challenge 2026 dataset as a [`FrameSource`].
//!
//! The challenge cameras are Kannala-Brandt (equidistant) fisheye. The source
//! yields the raw fisheye images and declares their model in its rig rather
//! than resampling them into a pinhole view, which would crop the wide field of
//! view.
//!
//! The sensors are mounted inverted, so the extracted PNGs are upside-down. The
//! source rotates each frame 180° (a flat-array reverse, not a remap) so the
//! image matches the upright calibration; set `rotate_180: false` if the
//! extraction already rotated them.
//!
//! Monocular only for now: it reads `cam0`, with the IMU when the images are
//! not rotated (the Kalibr extrinsic describes the camera as mounted). Stereo
//! (`cam0`+`cam1`) is a follow-up. The same reader serves TUM-VI, which shares
//! the layout and the Kalibr calibration.

use std::path::{Path, PathBuf};

use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::Image;
use kornia_io::png::{read_image_png_mono8, read_image_png_mono16};

use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::SensorSelection;
use serde::Deserialize;

use super::{
    FrameItem, FrameSource, OpenedSource, SourceError, dataset_summary, mono_only, non_empty,
    resolve,
};
use crate::datasets::euroc::GroundTruthPose;
use crate::datasets::hilti::HiltiDataset;

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

fn enabled() -> bool {
    true
}

impl HiltiConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        mono_only("Hilti", sensors)?;
        if sensors.imu && self.rotate_180 {
            return Err(
                "Hilti with the IMU needs `rotate_180: false`: the camera-IMU \
                        extrinsic of rotated images is not supported yet"
                    .into(),
            );
        }
        Ok(())
    }

    pub(super) fn resolve_paths(&mut self, base_dir: &Path) {
        resolve(&mut self.data, base_dir);
        resolve(&mut self.calib, base_dir);
    }

    pub(super) fn open(&self, imu: bool) -> Result<OpenedSource, SourceError> {
        let source = HiltiSource::open(
            &self.data,
            &self.calib,
            FrameWindow {
                start_frame: self.start_frame,
                max_frames: self.max_frames,
            },
            self.rotate_180,
            imu,
        )?;
        Ok(source.opened(self.start_frame))
    }
}

/// Reads upright raw fisheye `cam0` frames from an extracted Hilti sequence in
/// order.
pub struct HiltiSource {
    dataset: HiltiDataset,
    /// Virtual pinhole that keypoints are mapped into.
    camera: PinholeCamera,
    /// Kannala-Brandt model of the upright raw image.
    fisheye: FisheyeCamera,
    rotate_180: bool,
    /// `T_BC` of the virtual pinhole, when the IMU is read.
    camera_to_body: Option<Pose3d>,
    imu_cursor: usize,
    cursor: usize,
    start: usize,
    end: usize,
}

/// Which `cam0` samples to read: from `start_frame`, at most `max_frames`
/// (0 = to the end).
#[derive(Debug, Clone, Copy)]
pub struct FrameWindow {
    pub start_frame: usize,
    pub max_frames: usize,
}

impl HiltiSource {
    /// Opens an extracted sequence and its Kalibr calibration.
    ///
    /// `rotate_180` applies the inverted-mount correction (leave it on for
    /// Hilti unless the extraction already rotated the images). `imu` yields
    /// the `imu0` samples with each frame and declares the IMU in the rig.
    pub fn open(
        data_root: impl AsRef<Path>,
        calibration: impl AsRef<Path>,
        window: FrameWindow,
        rotate_180: bool,
        imu: bool,
    ) -> Result<Self, SourceError> {
        let dataset = HiltiDataset::open(data_root, calibration).map_err(SourceError::other)?;
        let n = dataset.samples().len();
        let start = window.start_frame.min(n);
        let end = if window.max_frames > 0 {
            (start + window.max_frames).min(n)
        } else {
            n
        };
        if imu && dataset.imu_samples.is_empty() {
            return Err(SourceError::other(
                "the IMU is selected but imu0/data.csv is missing",
            ));
        }

        let calib = &dataset.cam0_calibration;
        let camera = calib.to_undistorted_pinhole();
        let fisheye = calib.to_fisheye_camera();
        let camera_to_body = imu.then(|| camera_to_body(&dataset.t_cam0_imu));
        // IMU samples before the first yielded frame belong to no window.
        let imu_cursor = match start.checked_sub(1) {
            Some(previous) => {
                let boundary = dataset.cam0_samples[previous].timestamp_sec;
                dataset
                    .imu_samples
                    .partition_point(|sample| sample.timestamp <= boundary)
            }
            None => 0,
        };

        Ok(Self {
            dataset,
            camera,
            fisheye,
            rotate_180,
            camera_to_body,
            imu_cursor,
            cursor: start,
            start,
            end,
        })
    }

    /// The source with what the app reports and evaluates against.
    pub fn opened(self, start_frame: usize) -> OpenedSource {
        OpenedSource {
            summary: Some(dataset_summary(
                self.dataset_len(),
                start_frame,
                self.n_frames_hint(),
            )),
            ground_truth: non_empty(self.ground_truth_poses_cloned()),
            source: Box::new(self),
        }
    }

    fn imu_samples_until(&mut self, timestamp_sec: f64) -> Vec<ImuMeasurement> {
        if self.camera_to_body.is_none() {
            return Vec::new();
        }
        let start = self.imu_cursor;
        let end = start
            + self.dataset.imu_samples[start..]
                .partition_point(|sample| sample.timestamp <= timestamp_sec);
        self.imu_cursor = end;
        self.dataset.imu_samples[start..end].to_vec()
    }

    pub fn ground_truth_poses_cloned(&self) -> Vec<GroundTruthPose> {
        self.dataset.ground_truth().to_vec()
    }

    /// Total `cam0` sample count (ignoring start/max).
    pub fn dataset_len(&self) -> usize {
        self.dataset.samples().len()
    }
}

/// `T_BC` from Kalibr's row-major `T_cam_imu` (IMU → camera). The virtual
/// pinhole keeps the fisheye camera's axes, so the extrinsic carries over.
fn camera_to_body(t_cam_imu: &[[f64; 4]; 4]) -> Pose3d {
    let column = |c: usize| Vec3F64::new(t_cam_imu[0][c], t_cam_imu[1][c], t_cam_imu[2][c]);
    let imu_to_camera = Pose3d::new(
        Mat3F64::from_cols(column(0), column(1), column(2)),
        column(3),
    );
    imu_to_camera.inverse()
}

/// Reads an 8-bit grayscale PNG, or a 16-bit one (TUM-VI's `_16` exports)
/// reduced to its high byte.
fn read_gray8(path: &Path) -> Result<Image<u8, 1>, SourceError> {
    if let Ok(image) = read_image_png_mono8(path) {
        return Ok(image.into_inner());
    }
    let wide = read_image_png_mono16(path)
        .map_err(SourceError::other)?
        .into_inner();
    let narrow: Vec<u8> = wide.as_slice().iter().map(|&v| (v >> 8) as u8).collect();
    Image::from_size_slice(wide.size(), &narrow).map_err(SourceError::other)
}

/// Rotates a single-channel image 180°. For one channel this is just a
/// reverse of the row-major pixel buffer: `out[i] = in[N-1-i]`.
fn rotate_180_mono(img: &Image<u8, 1>) -> Image<u8, 1> {
    let mut buf = img.as_slice().to_vec();
    buf.reverse();
    Image::from_size_slice(img.size(), &buf).expect("rotated buffer matches original size")
}

impl FrameSource for HiltiSource {
    fn rig(&self) -> SensorRig {
        let rig = SensorRig::new(self.camera.clone()).with_fisheye(self.fisheye.clone());
        match self.camera_to_body {
            Some(t_bc) => rig.with_imu(t_bc),
            None => rig,
        }
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
        let timestamp_sec = sample.timestamp_sec;
        let raw = read_gray8(&sample.image_path)?;
        let image = if self.rotate_180 {
            rotate_180_mono(&raw)
        } else {
            raw
        };

        let imu_samples = self.imu_samples_until(timestamp_sec);
        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image: None,
            imu_samples,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn camera_to_body_inverts_the_kalibr_extrinsic() {
        // Camera axes rotated 90° about the IMU z axis, offset along x.
        let t_cam_imu = [
            [0.0, 1.0, 0.0, 0.1],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let t_bc = camera_to_body(&t_cam_imu);
        let p_imu = Vec3F64::new(0.3, -0.2, 0.5);
        let p_cam = Vec3F64::new(-0.2 + 0.1, -0.3, 0.5);
        assert!((t_bc.transform_point(&p_cam) - p_imu).length() < 1e-12);
    }

    #[test]
    fn rotate_180_reverses_pixels() {
        let img = Image::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &[1u8, 2, 3, 4],
        )
        .unwrap();
        let rot = rotate_180_mono(&img);
        assert_eq!(rot.as_slice(), &[4u8, 3, 2, 1]);
    }
}
