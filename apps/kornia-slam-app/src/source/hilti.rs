//! Hilti-Trimble SLAM Challenge 2026 dataset as a [`FrameSource`].
//!
//! The challenge cameras are Kannala-Brandt (equidistant) fisheye. Rather than
//! resampling the whole image into a pinhole view (which crops the wide field of
//! view and stretches the edges), this source follows ORB-SLAM3: ORB is
//! extracted on the raw fisheye image, and only the *keypoints* are undistorted
//! — each is unprojected through the KB model to a bearing and reprojected
//! through a virtual pinhole (see [`HiltiSource::undistort_features`]). The
//! existing pinhole geometry then works unchanged.
//!
//! The sensors are mounted inverted, so the extracted PNGs are upside-down. The
//! source rotates each frame 180° (a flat-array reverse, not a remap) so the
//! image matches the upright calibration; pass `rotate_180 = false` if the
//! extraction already rotated them.
//!
//! Features whose incidence angle exceeds [`MAX_INCIDENCE_DEG`] are dropped: a
//! pinhole cannot represent rays at/beyond 90°, and precision degrades long
//! before that.
//!
//! Monocular only for now: it reads `cam0`. Stereo (`cam0`+`cam1`) is a
//! follow-up.

use std::path::Path;

use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
use kornia_algebra::Vec2F64;
use kornia_image::Image;
use kornia_imgproc::features::OrbFeatures;
use kornia_io::png::read_image_png_mono8;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::euroc::GroundTruthPose;
use crate::datasets::hilti::HiltiDataset;

/// Maximum incidence angle (degrees) kept when undistorting keypoints. Beyond
/// this the bearing's `z` is tiny and the pinhole reprojection is ill-posed.
const MAX_INCIDENCE_DEG: f64 = 88.0;

/// Reads upright fisheye `cam0` frames from an extracted Hilti sequence in
/// order, undistorting keypoints (not pixels) to a virtual pinhole.
pub struct HiltiSource {
    dataset: HiltiDataset,
    /// Virtual pinhole that keypoints are undistorted into; reported by `camera()`.
    camera: PinholeCamera,
    /// Source Kannala-Brandt fisheye (upright image) for unprojecting keypoints.
    fisheye: FisheyeCamera,
    rotate_180: bool,
    /// `cosθ` floor for the incidence-angle cap.
    min_bearing_z: f64,
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
            min_bearing_z: MAX_INCIDENCE_DEG.to_radians().cos(),
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
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
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

    fn undistort_features(&self, features: &mut OrbFeatures) {
        let n = features.keypoints_xy.len();
        let mut keypoints_xy = Vec::with_capacity(n);
        let mut orientations = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);
        let mut octaves = Vec::with_capacity(n);

        for i in 0..n {
            let [u, v] = features.keypoints_xy[i];
            let b = self.fisheye.unproject(&Vec2F64::new(u as f64, v as f64));
            // b is a unit bearing; b.z = cosθ. Drop rays too close to / past 90°.
            if b.z <= self.min_bearing_z {
                continue;
            }
            let xn = b.x / b.z;
            let yn = b.y / b.z;
            let pu = self.camera.fx * xn + self.camera.cx;
            let pv = self.camera.fy * yn + self.camera.cy;

            keypoints_xy.push([pu as f32, pv as f32]);
            orientations.push(features.orientations[i]);
            descriptors.push(features.descriptors[i]);
            octaves.push(features.octaves[i]);
        }

        features.keypoints_xy = keypoints_xy;
        features.orientations = orientations;
        features.descriptors = descriptors;
        features.octaves = octaves;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::hilti::KannalaBrandtCalibration;
    use kornia_image::ImageSize;

    fn test_calib() -> KannalaBrandtCalibration {
        KannalaBrandtCalibration {
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
        }
    }

    /// Build a `HiltiSource` without touching the filesystem.
    fn source(calib: &KannalaBrandtCalibration) -> HiltiSource {
        HiltiSource {
            // dataset is unused by the methods under test; build a minimal stub
            // via the public open path is overkill, so we only exercise the
            // pure-math pieces through a hand-built source.
            dataset: HiltiDataset {
                root: std::path::PathBuf::new(),
                cam0_samples: Vec::new(),
                cam1_samples: Vec::new(),
                imu_samples: Vec::new(),
                cam0_calibration: *calib,
                cam1_calibration: *calib,
                t_cam0_imu: [[0.0; 4]; 4],
                t_cam1_imu: [[0.0; 4]; 4],
                ground_truth: Vec::new(),
            },
            camera: calib.to_undistorted_pinhole(),
            fisheye: calib.to_fisheye_camera(),
            rotate_180: true,
            min_bearing_z: MAX_INCIDENCE_DEG.to_radians().cos(),
            cursor: 0,
            start: 0,
            end: 0,
        }
    }

    fn features_at(pts: &[[f32; 2]]) -> OrbFeatures {
        OrbFeatures {
            keypoints_xy: pts.to_vec(),
            orientations: vec![0.0; pts.len()],
            descriptors: vec![[0u8; 32]; pts.len()],
            octaves: vec![0u8; pts.len()],
        }
    }

    #[test]
    fn principal_point_is_a_fixed_point() {
        let calib = test_calib();
        let src = source(&calib);
        let mut f = features_at(&[[calib.cx as f32, calib.cy as f32]]);
        src.undistort_features(&mut f);
        assert_eq!(f.keypoints_xy.len(), 1);
        let [u, v] = f.keypoints_xy[0];
        assert!((u as f64 - calib.cx).abs() < 1e-2);
        assert!((v as f64 - calib.cy).abs() < 1e-2);
    }

    #[test]
    fn keeps_arrays_aligned_when_filtering() {
        let calib = test_calib();
        let src = source(&calib);
        // One central point (kept) and one extreme-corner point (likely dropped).
        let mut f = features_at(&[[calib.cx as f32, calib.cy as f32], [1.0, 1.0]]);
        f.orientations = vec![0.5, 1.5];
        f.octaves = vec![1, 2];
        src.undistort_features(&mut f);
        let m = f.keypoints_xy.len();
        assert_eq!(f.orientations.len(), m);
        assert_eq!(f.descriptors.len(), m);
        assert_eq!(f.octaves.len(), m);
        assert!(m >= 1);
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
