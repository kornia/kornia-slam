//! Feature preparation for one input frame: ORB extraction, stereo depth and,
//! for raw fisheye images, keypoint mapping into the rig's virtual pinhole.

use std::time::{Duration, Instant};

use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec2F64;
use kornia_image::{Image, ImageError, ImageSize, InterpolationMode};
use kornia_imgproc::features::{OrbDetector, OrbFeatures};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_sensors::{SensorFrame, SensorRig};

use crate::Frame;
use crate::stereo::{StereoMatchConfig, compute_stereo_matches};

/// Fisheye keypoints beyond this incidence angle are dropped: a pinhole cannot
/// represent rays at or past 90°, and precision degrades well before that.
const MAX_INCIDENCE_DEG: f64 = 88.0;

/// The latest frame's frontend output, for overlays and timing.
#[derive(Debug, Clone, Default)]
pub struct FrontendObservation {
    /// Extracted keypoints in input-image pixels; raw fisheye coordinates for
    /// a fisheye rig, before mapping into the geometry camera.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Left keypoints with a stereo match; `None` without stereo.
    pub stereo_matched: Option<usize>,
    /// Feature extraction and stereo matching time.
    pub duration: Duration,
}

/// ORB extraction plus the rig-specific preparation tracking expects.
pub(crate) struct OrbFrontend {
    detector: OrbDetector,
    stereo: Option<StereoMatchConfig>,
    fisheye: Option<FisheyeMapping>,
    observation: FrontendObservation,
}

impl OrbFrontend {
    /// A rig with a stereo baseline enables matching against the right image;
    /// `rig` must already be restricted to the selected sensors.
    pub(crate) fn new(detector: OrbDetector, rig: &SensorRig) -> Self {
        let stereo = rig.stereo_baseline_m.map(|baseline| {
            StereoMatchConfig::new(
                baseline as f32,
                rig.camera.fx as f32,
                detector.downscale,
                detector.n_scales,
            )
        });
        let fisheye = rig.fisheye.clone().map(|fisheye| FisheyeMapping {
            fisheye,
            camera: rig.camera.clone(),
            min_bearing_z: MAX_INCIDENCE_DEG.to_radians().cos(),
        });
        Self {
            detector,
            stereo,
            fisheye,
            observation: FrontendObservation::default(),
        }
    }

    pub(crate) fn observation(&self) -> &FrontendObservation {
        &self.observation
    }

    /// Extracts and prepares the features of a validated input. The
    /// observation is updated only when preparation succeeds.
    pub(crate) fn prepare(&mut self, input: &SensorFrame<'_>) -> Result<Frame, ImageError> {
        let start = Instant::now();
        let image = input.image;
        let mut features = self.detector.detect_and_extract_u8(image)?;

        let (u_right, depth, stereo_matched) = match (&self.stereo, input.right_image) {
            (Some(config), Some(right)) => {
                let right_features = self.detector.detect_and_extract_u8(right)?;
                let left_pyramid = self.build_pyramid(image)?;
                let right_pyramid = self.build_pyramid(right)?;
                let matches = compute_stereo_matches(
                    &left_pyramid,
                    &right_pyramid,
                    &features,
                    &right_features,
                    config,
                );
                let matched = matches.num_matched();
                (matches.u_right, matches.depth, Some(matched))
            }
            _ => (Vec::new(), Vec::new(), None),
        };

        self.observation.keypoints_xy.clear();
        self.observation
            .keypoints_xy
            .extend_from_slice(&features.keypoints_xy);
        // Every per-feature array, stereo included, must stay aligned, so the
        // mapping runs before colors are sampled. A fisheye rig is never stereo.
        if let Some(fisheye) = &self.fisheye {
            fisheye.map_keypoints(&mut features);
        }
        let keypoint_colors = sample_colors(image, &features.keypoints_xy);

        self.observation.stereo_matched = stereo_matched;
        self.observation.duration = start.elapsed();
        Ok(Frame {
            idx: input.idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: image.size(),
            keypoint_colors,
            u_right,
            depth,
            keypoints_undist: Vec::new(),
        })
    }

    /// Builds an ORB-consistent pyramid: level `o` is the full image
    /// downscaled by `downscale^o`, so a full-resolution keypoint at octave `o`
    /// maps into level `o` by multiplying its coordinates by `downscale^-o`.
    ///
    /// Unlike the detector's own pyramid, each level resizes the original
    /// image rather than the previous level; reconcile before sharing one.
    fn build_pyramid(&self, image: &Image<u8, 1>) -> Result<Vec<Image<u8, 1>>, ImageError> {
        let levels = self.detector.n_scales;
        let mut pyramid = Vec::with_capacity(levels);
        pyramid.push(image.clone());
        let (w0, h0) = (image.width() as f32, image.height() as f32);
        for level in 1..levels {
            let inv = 1.0 / self.detector.downscale.powi(level as i32);
            let size = ImageSize {
                width: ((w0 * inv).round() as usize).max(1),
                height: ((h0 * inv).round() as usize).max(1),
            };
            let mut dst = Image::from_size_val(size, 0u8)?;
            resize_fast_mono(image, &mut dst, InterpolationMode::Bilinear)?;
            pyramid.push(dst);
        }
        Ok(pyramid)
    }
}

/// Maps raw fisheye keypoints into a virtual pinhole by unprojecting each to a
/// bearing through the Kannala-Brandt model, as ORB-SLAM3 does for fisheye
/// cameras, instead of resampling the whole image.
struct FisheyeMapping {
    fisheye: FisheyeCamera,
    camera: PinholeCamera,
    /// `cos` of the maximum incidence angle.
    min_bearing_z: f64,
}

impl FisheyeMapping {
    /// Remaps keypoints in place and drops those beyond the incidence cap,
    /// keeping orientations, descriptors and octaves aligned.
    fn map_keypoints(&self, features: &mut OrbFeatures) {
        let n = features.keypoints_xy.len();
        let mut keypoints_xy = Vec::with_capacity(n);
        let mut orientations = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);
        let mut octaves = Vec::with_capacity(n);

        for i in 0..n {
            let [u, v] = features.keypoints_xy[i];
            let bearing = self.fisheye.unproject(&Vec2F64::new(u as f64, v as f64));
            if bearing.z <= self.min_bearing_z {
                continue;
            }
            let xn = bearing.x / bearing.z;
            let yn = bearing.y / bearing.z;
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

/// Gray value at each keypoint as an RGB triple, clamped to the image.
fn sample_colors(image: &Image<u8, 1>, keypoints_xy: &[[f32; 2]]) -> Vec<[u8; 3]> {
    let size = image.size();
    let pixels = image.as_slice();
    keypoints_xy
        .iter()
        .map(|kp| {
            let x = (kp[0] as usize).min(size.width.saturating_sub(1));
            let y = (kp[1] as usize).min(size.height.saturating_sub(1));
            let g = pixels[y * size.width + x];
            [g, g, g]
        })
        .collect()
}

#[cfg(test)]
mod tests;
