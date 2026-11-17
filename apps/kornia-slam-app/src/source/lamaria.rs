//! LaMAria (pinhole ASL export) as a [`FrameSource`].
//!
//! The PNGs are RGB copies of the grey SLAM cameras and are converted back to
//! grey. Every frame is yielded; there is no ground truth here, as LaMAria is
//! scored against control points by its own evaluator.

use std::path::{Path, PathBuf};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::png::read_image_png_rgb8;
use kornia_sensors::ImuCalibration;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;
use kornia_sensors::rig_calibration::rig_cameras_from_body_extrinsics;
use kornia_slam::{CameraSelection, SensorSelection};
use serde::Deserialize;

use super::{FrameItem, FrameSource, OpenedSource, SourceError, dataset_summary, resolve};
use crate::datasets::lamaria::LamariaDataset;

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
}

impl LamariaConfig {
    pub(super) fn validate(&self, sensors: SensorSelection) -> Result<(), String> {
        if sensors.cameras == CameraSelection::Stereo {
            return Err("LaMAria has no rectified pair; select the `Rig` cameras".into());
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
    }

    pub(super) fn open(&self, sensors: SensorSelection) -> Result<OpenedSource, SourceError> {
        let imu = sensors.imu;
        let dataset = LamariaDataset::open(
            &self.data,
            &self.calib,
            self.factory_calib.as_deref(),
            self.imu_time_offset_ns,
        )
        .map_err(SourceError::other)?;
        let total = dataset.len();
        let start = self.start_frame.min(total);
        let end = match self.max_frames {
            0 => total,
            n => (start + n).min(total),
        };
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
        let imu_cursor = match start.checked_sub(1) {
            Some(prev) if imu => {
                let boundary = dataset.cameras[0][prev].timestamp_sec;
                dataset
                    .imu_samples
                    .partition_point(|sample| sample.timestamp_sec <= boundary)
            }
            _ => 0,
        };
        let source = LamariaSource {
            dataset,
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
}

pub struct LamariaSource {
    dataset: LamariaDataset,
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
        let cameras = &self.dataset.calibration.cameras;
        let primary = &cameras[self.camera];
        let extrinsics: Vec<(PinholeCamera, ImageSize, Pose3d)> = cameras
            .iter()
            .map(|c| {
                let size = ImageSize {
                    width: c.width,
                    height: c.height,
                };
                (c.camera.clone(), size, c.camera_to_body)
            })
            .collect();
        let mut rig = SensorRig::new(primary.camera.clone())
            .with_rig_cameras(rig_cameras_from_body_extrinsics(&extrinsics, self.camera));
        if self.with_imu {
            rig.imu = Some(ImuCalibration {
                noise: self.dataset.calibration.imu_noise,
                camera_to_body: primary.camera_to_body,
            });
        }
        rig
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

fn read_gray(path: &Path) -> Result<Image<u8, 1>, SourceError> {
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

    /// B1-P2 of `experiments/pr180-B-card.md`: the shared helper keeps the
    /// expression this source composed inline.
    #[test]
    fn rig_cameras_keep_the_inline_composition() {
        let camera = |fx| PinholeCamera {
            fx,
            fy: fx,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
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
        let extrinsics: Vec<(PinholeCamera, ImageSize, Pose3d)> = (0..2)
            .map(|c| (camera(240.0 + c as f64), size, camera_to_body[c]))
            .collect();
        for primary in [0, 1] {
            let rig = rig_cameras_from_body_extrinsics(&extrinsics, primary);
            let inline = camera_to_body[primary]
                .inverse()
                .compose(&camera_to_body[1 - primary]);
            assert_eq!(rig[0].primary_from_camera, Pose3d::IDENTITY);
            assert_eq!(rig[0].camera.fx, 240.0 + primary as f64);
            assert_eq!(bits(&rig[1].primary_from_camera), bits(&inline));
        }
    }
}
