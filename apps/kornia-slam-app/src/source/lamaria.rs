//! LaMAria (pinhole ASL export, or raw Aria fisheye images) as a [`FrameSource`].
//!
//! The PNGs are RGB copies of the grey SLAM cameras and are converted back to
//! grey; the raw extraction's JPEGs are decoded to grey. Every frame is yielded; there is no ground truth here, as LaMAria is
//! scored against control points by its own evaluator.

use std::path::{Path, PathBuf};

use kornia_3d::camera::Camera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::png::read_image_png_rgb8;
use kornia_sensors::imu::ImuMeasurement;
use kornia_sensors::{ImuCalibration, SensorRig};
use kornia_slam::{CameraSelection, SensorSelection};
use serde::Deserialize;

use super::{
    FrameItem, FrameSource, OpenedSource, SourceError, dataset_summary, decode_jpeg_to_luma,
    resolve, rig_from_body_extrinsics,
};
use crate::datasets::euroc::ImuSample;
use crate::datasets::lamaria::{LamariaDataset, RawImages};

/// Which images of a LaMAria sequence the cameras read.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum LamariaImages {
    /// The pinhole ASL export under `data`.
    #[default]
    Pinhole,
    /// Raw 640x480 Aria fisheye images: `calib` is `aria_calibrations/<seq>.json`
    /// and `root` the ASL extraction of the `.vrs` (`root/aria/{cam0,cam1}`).
    /// The IMU still comes from `data`. Needs the `Rig` cameras.
    RawFisheye { calib: PathBuf, root: PathBuf },
}

/// A LaMAria sequence: `data` holds `aria/{cam0,cam1,imu0}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LamariaConfig {
    pub data: PathBuf,
    /// `pinhole_calibrations/<seq>.json`.
    pub calib: PathBuf,
    /// Aria factory calibration record from the `.vrs` header; rectifies the IMU.
    #[serde(default)]
    pub factory_calib: Option<PathBuf>,
    /// Added to every IMU timestamp (ns).
    #[serde(default)]
    pub imu_time_offset_ns: i64,
    /// Camera of a monocular run: 0 (left SLAM camera) or 1 (right).
    #[serde(default)]
    pub camera: usize,
    #[serde(default)]
    pub start_frame: usize,
    /// 0 processes the whole sequence.
    #[serde(default)]
    pub max_frames: usize,
    /// Short-experiment window, in seconds since the sequence's first frame:
    /// frames before `start_sec` are skipped (their IMU too, except the
    /// interval up to the first kept frame) and frames after `end_sec` are
    /// not read. Combines with `start_frame`/`max_frames` by taking the
    /// narrower range. `None` keeps the whole range.
    #[serde(default)]
    pub start_sec: Option<f64>,
    #[serde(default)]
    pub end_sec: Option<f64>,
    #[serde(default)]
    pub images: LamariaImages,
}

impl LamariaConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        if sensors.cameras == CameraSelection::Stereo {
            return Err("LaMAria has no rectified pair; select the `Rig` cameras".into());
        }
        if self.images != LamariaImages::Pinhole && sensors.cameras != CameraSelection::Rig {
            return Err("LaMAria raw fisheye images need the `Rig` cameras".into());
        }
        if self.camera > 1 {
            return Err(format!(
                "LaMAria camera is {}, but must be 0 or 1",
                self.camera
            ));
        }
        Ok(())
    }

    pub(super) fn resolve_paths(&mut self, base_dir: &Path) {
        resolve(&mut self.data, base_dir);
        resolve(&mut self.calib, base_dir);
        if let Some(path) = self.factory_calib.as_mut() {
            resolve(path, base_dir);
        }
        if let LamariaImages::RawFisheye { calib, root } = &mut self.images {
            resolve(calib, base_dir);
            resolve(root, base_dir);
        }
    }

    pub(super) fn open(&self, sensors: SensorSelection) -> Result<OpenedSource, SourceError> {
        let imu = sensors.imu;
        let raw = match &self.images {
            LamariaImages::Pinhole => None,
            LamariaImages::RawFisheye { calib, root } => Some(RawImages { calib, root }),
        };
        let dataset = LamariaDataset::open(
            &self.data,
            &self.calib,
            raw,
            self.factory_calib.as_deref(),
            self.imu_time_offset_ns,
        )
        .map_err(SourceError::other)?;
        let total = dataset.len();
        let times: Vec<f64> = dataset.cameras[0]
            .iter()
            .map(|sample| sample.timestamp_sec)
            .collect();
        let (start, end) = self.frame_range(&times);
        let summary = format!(
            "{} (cam{}, IMU {}, offset {} ns)",
            dataset_summary(total, start, Some(end - start)),
            self.camera,
            if dataset.imu_rectified {
                "rectified"
            } else {
                "raw"
            },
            self.imu_time_offset_ns,
        );
        let imu_cursor = match imu {
            true => imu_start(&times, &dataset.imu_samples, start),
            false => 0,
        };
        let rig = sensor_rig(&dataset, self.camera, imu)?;
        let source = LamariaSource {
            dataset,
            sensor_rig: rig,
            camera: self.camera,
            rig: sensors.cameras == CameraSelection::Rig,
            with_imu: imu,
            cursor: start,
            end,
            imu_cursor,
        };
        Ok(OpenedSource {
            source: Box::new(source),
            ground_truth: None,
            summary: Some(summary),
        })
    }

    /// Frames `[start, end)` to read, given every frame's timestamp.
    fn frame_range(&self, times: &[f64]) -> (usize, usize) {
        let total = times.len();
        let mut start = self.start_frame.min(total);
        let mut end = match self.max_frames {
            0 => total,
            n => (start + n).min(total),
        };
        let first = times.first().copied().unwrap_or(0.0);
        if let Some(sec) = self.start_sec {
            start = start.max(times.partition_point(|t| t - first < sec));
        }
        if let Some(sec) = self.end_sec {
            end = end.min(times.partition_point(|t| t - first <= sec));
        }
        (start, end.max(start))
    }
}

pub struct LamariaSource {
    dataset: LamariaDataset,
    sensor_rig: SensorRig,
    camera: usize,
    /// Yield the other camera's image too.
    rig: bool,
    with_imu: bool,
    cursor: usize,
    end: usize,
    imu_cursor: usize,
}

impl FrameSource for LamariaSource {
    /// The selected camera is the primary one, and both cameras are offered
    /// as a multi-camera rig.
    fn rig(&self) -> SensorRig {
        self.sensor_rig.clone()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.end.saturating_sub(self.cursor))
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.cursor >= self.end {
            return Ok(None);
        }
        let idx = self.cursor;
        self.cursor += 1;
        let sample = &self.dataset.cameras[self.camera][idx];
        let timestamp_sec = sample.timestamp_sec;
        let image = read_gray(&sample.image_path)?;
        let rig_images = if self.rig {
            vec![read_gray(
                &self.dataset.cameras[1 - self.camera][idx].image_path,
            )?]
        } else {
            Vec::new()
        };
        let imu_samples = self.imu_samples_until(timestamp_sec);
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image: None,
            rig_images,
            imu_samples,
        }))
    }
}

impl LamariaSource {
    fn imu_samples_until(&mut self, timestamp_sec: f64) -> Vec<ImuMeasurement> {
        if !self.with_imu {
            return Vec::new();
        }
        let samples = &self.dataset.imu_samples;
        let start = self.imu_cursor;
        let end = start + samples[start..].partition_point(|s| s.timestamp_sec <= timestamp_sec);
        self.imu_cursor = end;
        samples[start..end]
            .iter()
            .map(|s| ImuMeasurement {
                timestamp: s.timestamp_sec,
                gyro: Vec3F64::from_array(s.gyro),
                accel: Vec3F64::from_array(s.accel),
            })
            .collect()
    }
}

/// Both cameras with `primary` first, and the IMU when `imu` is set.
fn sensor_rig(
    dataset: &LamariaDataset,
    primary: usize,
    imu: bool,
) -> Result<SensorRig, SourceError> {
    let cameras = &dataset.calibration.cameras;
    let extrinsics: Vec<(Camera, ImageSize, Pose3d)> = cameras
        .iter()
        .map(|c| {
            let size = ImageSize {
                width: c.width,
                height: c.height,
            };
            (c.model.clone(), size, c.camera_to_body)
        })
        .collect();
    let imu = imu.then(|| ImuCalibration {
        noise: dataset.calibration.imu_noise,
        camera_to_body: cameras[primary].camera_to_body,
    });
    rig_from_body_extrinsics(&extrinsics, primary, imu)
}

/// First IMU sample after frame `start - 1`, so the first kept frame gets
/// the interval since the frame before it, as in a full run.
fn imu_start(frame_times: &[f64], imu: &[ImuSample], start: usize) -> usize {
    match start.checked_sub(1) {
        Some(prev) => {
            let boundary = frame_times[prev];
            imu.partition_point(|sample| sample.timestamp_sec <= boundary)
        }
        None => 0,
    }
}

fn read_gray(path: &Path) -> Result<Image<u8, 1>, SourceError> {
    let jpeg = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg"));
    if jpeg {
        let bytes = std::fs::read(path).map_err(SourceError::other)?;
        return decode_jpeg_to_luma(&bytes).map_err(SourceError::other);
    }
    let rgb = read_image_png_rgb8(path).map_err(SourceError::other)?;
    let rgb: Image<u8, 3> = rgb.into_inner();
    let mut gray = Image::from_size_val(rgb.size(), 0).map_err(SourceError::other)?;
    gray_from_rgb_u8(&rgb, &mut gray).map_err(SourceError::other)?;
    Ok(gray)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    fn bits(pose: &Pose3d) -> Vec<u64> {
        let r = pose.rotation.to_cols_array();
        let t = pose.translation.to_array();
        r.iter().chain(&t).map(|v| v.to_bits()).collect()
    }

    fn config(images: &str) -> LamariaConfig {
        ron::from_str(&format!(r#"(data: "seq", calib: "pinhole.json"{images})"#)).unwrap()
    }

    fn sensors(cameras: CameraSelection) -> SensorSelection {
        SensorSelection { cameras, imu: true }
    }

    #[test]
    fn raw_fisheye_images_parse_and_default_to_pinhole() {
        assert_eq!(config("").images, LamariaImages::Pinhole);
        assert_eq!(
            config(r#", images: RawFisheye(calib: "aria.json", root: "raw")"#).images,
            LamariaImages::RawFisheye {
                calib: PathBuf::from("aria.json"),
                root: PathBuf::from("raw"),
            }
        );
    }

    #[test]
    fn time_window_selects_frames_and_none_changes_nothing() {
        // 10 frames at 0.5 s from t0 = 100 s.
        let times: Vec<f64> = (0..10).map(|i| 100.0 + 0.5 * i as f64).collect();
        assert_eq!(config("").frame_range(&times), (0, 10));
        assert_eq!(
            config(", start_frame: 2, max_frames: 3").frame_range(&times),
            (2, 5)
        );
        assert_eq!(
            config(", start_sec: Some(1.0), end_sec: Some(3.0)").frame_range(&times),
            (2, 7)
        );
        assert_eq!(
            config(", start_sec: Some(1.2)").frame_range(&times),
            (3, 10)
        );
        // The narrower of the frame and time limits wins.
        assert_eq!(
            config(", start_frame: 4, max_frames: 2, start_sec: Some(1.0)").frame_range(&times),
            (4, 6)
        );
        assert_eq!(
            config(", start_sec: Some(9.0), end_sec: Some(1.0)").frame_range(&times),
            (10, 10)
        );
    }

    #[test]
    fn time_window_keeps_imu_from_the_frame_before_the_window() {
        let frames: Vec<f64> = (0..10).map(|i| 0.5 * i as f64).collect();
        // IMU at 0.1 s steps from 0.0 to 4.9 s.
        let imu: Vec<ImuSample> = (0..50)
            .map(|i| ImuSample {
                timestamp_sec: 0.1 * i as f64,
                gyro: [0.0; 3],
                accel: [0.0; 3],
            })
            .collect();
        let (start, _) = config(", start_sec: Some(1.0)").frame_range(&frames);
        assert_eq!(start, 2);
        // Frame 1 is at 0.5 s: the first kept sample is the one after it.
        let first = imu_start(&frames, &imu, start);
        assert!(imu[first - 1].timestamp_sec <= 0.5);
        assert!(imu[first].timestamp_sec > 0.5 && imu[first].timestamp_sec < 1.0);
        assert_eq!(imu_start(&frames, &imu, 0), 0);
    }

    #[test]
    fn raw_fisheye_images_need_the_rig_cameras() {
        let raw = config(r#", images: RawFisheye(calib: "aria.json", root: "raw")"#);
        assert!(raw.validate(sensors(CameraSelection::Mono)).is_err());
        assert!(raw.validate(sensors(CameraSelection::Rig)).is_ok());
        assert!(config("").validate(sensors(CameraSelection::Mono)).is_ok());
    }

    #[test]
    fn raw_fisheye_paths_resolve_against_the_run_file() {
        let mut raw = config(r#", images: RawFisheye(calib: "aria.json", root: "/abs/raw")"#);
        raw.resolve_paths(Path::new("/runs"));
        assert_eq!(
            raw.images,
            LamariaImages::RawFisheye {
                calib: PathBuf::from("/runs/aria.json"),
                root: PathBuf::from("/abs/raw"),
            }
        );
    }

    #[test]
    fn raw_fisheye_jpeg_reads_as_grey() {
        let size = ImageSize {
            width: 16,
            height: 8,
        };
        let pixels: Vec<u8> = (0..size.width * size.height)
            .map(|i| (i * 2) as u8)
            .collect();
        let image = Image::<u8, 1>::new(size, pixels).unwrap();
        let mut bytes = Vec::new();
        kornia_io::jpeg::encode_image_jpeg_gray8(&image, 100, &mut bytes).unwrap();
        let path =
            std::env::temp_dir().join(format!("kornia-slam-lamaria-{}.jpg", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let gray = read_gray(&path);
        std::fs::remove_file(&path).unwrap();
        let gray = gray.unwrap();
        assert_eq!(gray.size(), size);
        let max_error = gray
            .as_slice()
            .iter()
            .zip(image.as_slice())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(max_error <= 8, "{max_error}");
    }

    /// The shared helper keeps the expression this source composed inline.
    #[test]
    fn rig_cameras_keep_the_inline_composition() {
        let camera = |fx| {
            Camera::from(kornia_3d::camera::PinholeCamera {
                fx,
                fy: fx,
                cx: 320.0,
                cy: 240.0,
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
            })
        };
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let camera_to_body = [
            Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.31, -1.2, 0.05)).matrix(),
                Vec3F64::new(0.0412, -0.0071, 0.0113),
            ),
            Pose3d::new(
                SO3F64::exp(Vec3F64::new(-0.27, 1.1, -0.02)).matrix(),
                Vec3F64::new(-0.0598, -0.0069, 0.0131),
            ),
        ];
        let extrinsics: Vec<(Camera, ImageSize, Pose3d)> = (0..2)
            .map(|c| (camera(240.0 + c as f64), size, camera_to_body[c]))
            .collect();
        for primary in [0, 1] {
            let rig = rig_from_body_extrinsics(&extrinsics, primary, None).unwrap();
            let inline = camera_to_body[primary]
                .inverse()
                .compose(&camera_to_body[1 - primary]);
            assert_eq!(rig.cameras()[0].primary_from_camera, Pose3d::IDENTITY);
            assert_eq!(rig.primary().pinhole().fx, 240.0 + primary as f64);
            assert_eq!(bits(&rig.cameras()[1].primary_from_camera), bits(&inline));
        }
    }
}
