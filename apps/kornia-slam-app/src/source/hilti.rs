//! Hilti-Trimble SLAM Challenge 2026 dataset as a [`FrameSource`].
//!
//! The challenge cameras are Kannala-Brandt (equidistant) fisheye. Rather than
//! resampling the whole image into a pinhole view (which crops the wide field of
//! view and stretches the edges), the source yields raw fisheye images and
//! declares the fisheye model in its rig; following ORB-SLAM3, the SLAM
//! frontend extracts ORB on the raw image and maps only the keypoints into the
//! rig's virtual pinhole. The existing pinhole geometry then works unchanged.
//!
//! The sensors are mounted inverted, so the extracted PNGs are upside-down. The
//! source rotates each frame 180° (a flat-array reverse, not a remap) so the
//! image matches the upright calibration; pass `rotate_180 = false` if the
//! extraction already rotated them.
//!
//! Monocular only for now: it reads `cam0`. Stereo (`cam0`+`cam1`) is a
//! follow-up.

use std::path::Path;

use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
use kornia_image::Image;
use kornia_io::png::read_image_png_mono8;

use kornia_sensors::SensorRig;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::euroc::GroundTruthPose;
use crate::datasets::hilti::HiltiDataset;

/// Reads upright raw fisheye `cam0` frames from an extracted Hilti sequence in
/// order.
pub struct HiltiSource {
    dataset: HiltiDataset,
    /// Virtual pinhole that keypoints are mapped into.
    camera: PinholeCamera,
    /// Kannala-Brandt model of the upright raw image.
    fisheye: FisheyeCamera,
    rotate_180: bool,
    cursor: usize,
    start: usize,
    end: usize,
}

impl HiltiSource {
    /// Opens an extracted Hilti sequence and its Kalibr calibration.
    ///
    /// `max_frames == 0` means "until the dataset is exhausted". `start_frame`
    /// is the index of the first `cam0` sample to yield. `rotate_180` applies
    /// the inverted-mount correction (leave it on unless the extraction already
    /// rotated the images).
    pub fn open(
        data_root: impl AsRef<Path>,
        calibration: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
        rotate_180: bool,
    ) -> Result<Self, SourceError> {
        let dataset = HiltiDataset::open(data_root, calibration).map_err(SourceError::other)?;
        let n = dataset.samples().len();
        let start = start_frame.min(n);
        let end = if max_frames > 0 {
            (start + max_frames).min(n)
        } else {
            n
        };

        let calib = &dataset.cam0_calibration;
        let camera = calib.to_undistorted_pinhole();
        let fisheye = calib.to_fisheye_camera();

        Ok(Self {
            dataset,
            camera,
            fisheye,
            rotate_180,
            cursor: start,
            start,
            end,
        })
    }

    pub fn ground_truth_poses_cloned(&self) -> Vec<GroundTruthPose> {
        self.dataset.ground_truth().to_vec()
    }

    /// Total `cam0` sample count (ignoring start/max).
    pub fn dataset_len(&self) -> usize {
        self.dataset.samples().len()
    }
}

/// Rotates a single-channel image 180° in place. For one channel this is just a
/// reverse of the row-major pixel buffer: `out[i] = in[N-1-i]`.
fn rotate_180_mono(img: &Image<u8, 1>) -> Image<u8, 1> {
    let mut buf = img.as_slice().to_vec();
    buf.reverse();
    Image::from_size_slice(img.size(), &buf).expect("rotated buffer matches original size")
}

impl FrameSource for HiltiSource {
    fn rig(&self) -> SensorRig {
        SensorRig::new(self.camera.clone()).with_fisheye(self.fisheye.clone())
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
        let raw = read_image_png_mono8(&sample.image_path)
            .map_err(SourceError::other)?
            .into_inner();
        let image = if self.rotate_180 {
            rotate_180_mono(&raw)
        } else {
            raw
        };

        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image: None,
            imu_samples: Vec::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::hilti::KannalaBrandtCalibration;
    use kornia_image::ImageSize;

    #[test]
    fn rig_declares_the_fisheye_model() {
        let calib = KannalaBrandtCalibration {
            fx: 461.64,
            fy: 459.72,
            cx: 732.95,
            cy: 720.54,
            k1: 0.0344,
            k2: -0.0216,
            k3: 0.0031,
            k4: -0.0005,
            width: 1472,
            height: 1440,
        };
        let source = HiltiSource {
            dataset: HiltiDataset {
                root: std::path::PathBuf::new(),
                cam0_samples: Vec::new(),
                cam1_samples: Vec::new(),
                imu_samples: Vec::new(),
                cam0_calibration: calib,
                cam1_calibration: calib,
                t_cam0_imu: [[0.0; 4]; 4],
                t_cam1_imu: [[0.0; 4]; 4],
                ground_truth: Vec::new(),
            },
            camera: calib.to_undistorted_pinhole(),
            fisheye: calib.to_fisheye_camera(),
            rotate_180: true,
            cursor: 0,
            start: 0,
            end: 0,
        };
        let rig = source.rig();
        assert_eq!(rig.camera.fx, calib.fx);
        assert_eq!(rig.fisheye.map(|fisheye| fisheye.k1), Some(calib.k1));
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
