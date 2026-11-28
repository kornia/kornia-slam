//! Feature preparation for one input frame: ORB extraction, stereo depth and,
//! for raw fisheye images, keypoint mapping into the rig's virtual pinhole.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kornia_3d::camera::{Camera, CameraModel, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec2F64;
use kornia_image::{Image, ImageError, ImageSize, InterpolationMode};
use kornia_imgproc::features::{OrbDetector, OrbFeatures};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_sensors::SensorRig;

use crate::features::Features;
use crate::frame::{Frame, RigFeatures};
use crate::overlap::OverlapMatcher;
use crate::rig::RigGeometry;
use crate::stereo::{StereoMatchConfig, StereoMatches, compute_stereo_matches};

/// Fisheye keypoints beyond this incidence angle are dropped: a pinhole cannot
/// represent rays at or past 90°, and precision degrades well before that.
const MAX_INCIDENCE_DEG: f64 = 88.0;

/// The latest frame's frontend output, for overlays and timing.
#[derive(Debug, Clone, Default)]
pub struct FrontendObservation {
    /// Extracted keypoints in input-image pixels; raw fisheye coordinates for
    /// a fisheye rig, before mapping into the geometry camera.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Left keypoints with a stereo match (for a rig, primary keypoints
    /// with a triangulated overlap match); `None` without stereo or a rig.
    pub stereo_matched: Option<usize>,
    /// Feature extraction and stereo matching time.
    pub duration: Duration,
}

/// ORB extraction plus the rig-specific preparation tracking expects.
pub(crate) struct OrbFrontend {
    detector: OrbDetector,
    stereo: Option<StereoMatchConfig>,
    fisheye: Option<FisheyeMapping>,
    /// The rig a multi-camera frontend extracts on, with the matcher of the
    /// overlap of its first two cameras.
    rig: Option<OverlapMatcher>,
    observation: FrontendObservation,
    /// Filled by the preparation in progress and swapped into `observation`
    /// only when it succeeds, so a failed frame leaves the last one intact.
    staging: FrontendObservation,
    #[cfg(test)]
    fail_next_extraction: bool,
}

impl OrbFrontend {
    /// A stereo baseline enables matching against the right image;
    /// `multi_camera` extracts on every camera of `rig`, which must already
    /// be restricted to the selected sensors.
    pub(crate) fn new(
        detector: OrbDetector,
        rig: &SensorRig,
        stereo_baseline_m: Option<f64>,
        multi_camera: bool,
    ) -> Self {
        let primary = rig.primary();
        let camera = primary.pinhole();
        let stereo = stereo_baseline_m.map(|baseline| {
            StereoMatchConfig::new(
                baseline as f32,
                camera.fx as f32,
                detector.downscale,
                detector.n_scales,
            )
        });
        let fisheye = primary.is_fisheye().then(|| FisheyeMapping {
            model: primary.model.clone(),
            camera,
            min_bearing_z: MAX_INCIDENCE_DEG.to_radians().cos(),
        });
        let scale_factor = f64::from(detector.downscale);
        Self {
            detector,
            stereo,
            fisheye,
            rig: multi_camera
                .then(|| rig_overlap(rig, scale_factor))
                .flatten(),
            observation: FrontendObservation::default(),
            staging: FrontendObservation::default(),
            #[cfg(test)]
            fail_next_extraction: false,
        }
    }

    pub(crate) fn observation(&self) -> &FrontendObservation {
        &self.observation
    }

    /// Makes the next ORB extraction fail, to test the error paths.
    #[cfg(test)]
    pub(crate) fn fail_next_extraction(&mut self) {
        self.fail_next_extraction = true;
    }

    fn extract(&mut self, image: &Image<u8, 1>) -> Result<OrbFeatures, ImageError> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_extraction) {
            return Err(ImageError::ImageDataNotInitialized);
        }
        self.detector.detect_and_extract_u8(image)
    }

    /// Publishes the staged observation of a frame that was prepared.
    fn publish(&mut self, frame: Frame) -> Frame {
        std::mem::swap(&mut self.observation, &mut self.staging);
        frame
    }

    /// Extracts and prepares the features of one validated image, numbering
    /// the frame `idx`. The observation is updated only when preparation
    /// succeeds.
    pub(crate) fn prepare_image(
        &mut self,
        idx: usize,
        image: &Image<u8, 1>,
    ) -> Result<Frame, ImageError> {
        let frame = self.image_frame(idx, image)?;
        Ok(self.publish(frame))
    }

    fn image_frame(&mut self, idx: usize, image: &Image<u8, 1>) -> Result<Frame, ImageError> {
        let start = Instant::now();
        let features = self.extract(image)?;
        Ok(self.finish_frame(idx, image, features, None, start))
    }

    /// Like [`Self::prepare_image`] for a validated, rectified stereo pair,
    /// giving each left feature its right-image match and depth. Without a
    /// stereo rig the right image is ignored.
    pub(crate) fn prepare_stereo_pair(
        &mut self,
        idx: usize,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<Frame, ImageError> {
        let start = Instant::now();
        let features = self.extract(left)?;
        let right_features = match self.stereo {
            Some(_) => Some(self.extract(right)?),
            None => None,
        };
        let (Some(config), Some(right_features)) = (&self.stereo, right_features) else {
            let frame = self.finish_frame(idx, left, features, None, start);
            return Ok(self.publish(frame));
        };
        let left_pyramid = self.build_pyramid(left)?;
        let right_pyramid = self.build_pyramid(right)?;
        let matches = compute_stereo_matches(
            &left_pyramid,
            &right_pyramid,
            &features,
            &right_features,
            config,
        );
        let frame = self.finish_frame(idx, left, features, Some(matches), start);
        Ok(self.publish(frame))
    }

    /// Extracts features on every image of a validated rig capture, one image
    /// per rig camera in rig order, and matches the first two cameras inside
    /// their overlap. Without a rig only the first image is used.
    pub(crate) fn prepare_rig(
        &mut self,
        idx: usize,
        images: &[&Image<u8, 1>],
    ) -> Result<Frame, ImageError> {
        let start = Instant::now();
        let Some(primary) = images.first() else {
            return Err(ImageError::ImageDataNotInitialized);
        };
        if self.rig.is_none() {
            return self.prepare_image(idx, primary);
        }
        let mut features = OrbFeatures {
            keypoints_xy: Vec::new(),
            orientations: Vec::new(),
            descriptors: Vec::new(),
            octaves: Vec::new(),
        };
        let mut camera_ranges = Vec::with_capacity(images.len());
        let mut keypoint_colors = Vec::new();
        for image in images {
            let camera_features = self.extract(image)?;
            let begin = features.keypoints_xy.len();
            keypoint_colors.extend(sample_colors(image, &camera_features.keypoints_xy));
            features.keypoints_xy.extend(camera_features.keypoints_xy);
            features.orientations.extend(camera_features.orientations);
            features.descriptors.extend(camera_features.descriptors);
            features.octaves.extend(camera_features.octaves);
            camera_ranges.push(begin..features.keypoints_xy.len());
        }
        let Some(rig) = self.rig.as_mut() else {
            return Err(ImageError::ImageDataNotInitialized);
        };
        self.staging.keypoints_xy.clear();
        if let Some(range) = camera_ranges.first() {
            self.staging
                .keypoints_xy
                .extend_from_slice(&features.keypoints_xy[range.clone()]);
        }
        let mut frame = Frame {
            idx,
            features: Features::from(features),
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: primary.size(),
            keypoint_colors,
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
            rig: RigFeatures {
                geometry: Some(Arc::clone(rig.rig())),
                camera_ranges,
                left_to_right: Vec::new(),
                stereo_points: Vec::new(),
            },
        };
        // Raw fisheye or radtan pixels are not pinhole pixels: the overlap's
        // epipolar geometry and triangulation need the undistorted cache. A
        // zero-distortion pinhole rig keeps reading raw keypoints.
        if let (true, Some(view0)) = (rig.rig().needs_undistortion(), rig.rig().view(0)) {
            frame.ensure_undistorted(view0.pinhole);
        }
        let pixels = if frame.keypoints_undist.is_empty() {
            &frame.features.keypoints_xy
        } else {
            &frame.keypoints_undist
        };
        let matches = rig.run(&frame.features, pixels, &frame.rig.camera_ranges);
        self.staging.stereo_matched = Some(matches.num_matched());
        frame.rig.left_to_right = matches.left_to_right;
        frame.rig.stereo_points = matches.points;
        self.staging.duration = start.elapsed();
        Ok(self.publish(frame))
    }

    fn finish_frame(
        &mut self,
        idx: usize,
        image: &Image<u8, 1>,
        features: OrbFeatures,
        stereo: Option<StereoMatches>,
        start: Instant,
    ) -> Frame {
        let mut features = Features::from(features);
        let stereo_matched = stereo.as_ref().map(StereoMatches::num_matched);
        let StereoMatches { u_right, depth } = stereo.unwrap_or_default();

        self.staging.keypoints_xy.clear();
        self.staging
            .keypoints_xy
            .extend_from_slice(&features.keypoints_xy);
        // Every per-feature array, stereo included, must stay aligned, so the
        // mapping runs before colors are sampled. A fisheye rig is never stereo.
        if let Some(fisheye) = &self.fisheye {
            fisheye.map_keypoints(&mut features);
        }
        let keypoint_colors = sample_colors(image, &features.keypoints_xy);

        self.staging.stereo_matched = stereo_matched;
        self.staging.duration = start.elapsed();
        Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: image.size(),
            keypoint_colors,
            u_right,
            depth,
            keypoints_undist: Vec::new(),
            rig: RigFeatures::default(),
        }
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

/// The overlap matcher of a rig of two or more cameras; `None` for one camera.
fn rig_overlap(rig: &SensorRig, scale_factor: f64) -> Option<OverlapMatcher> {
    OverlapMatcher::new(
        Arc::new(RigGeometry::from_rig(rig)?),
        rig.b_from_a(0, 1)?,
        scale_factor,
    )
}

/// Maps raw fisheye keypoints into a virtual pinhole by unprojecting each to a
/// bearing through the fisheye model, as ORB-SLAM3 does for fisheye cameras,
/// instead of resampling the whole image.
struct FisheyeMapping {
    model: Camera,
    camera: PinholeCamera,
    /// `cos` of the maximum incidence angle.
    min_bearing_z: f64,
}

impl FisheyeMapping {
    /// Remaps keypoints in place and drops those beyond the incidence cap,
    /// keeping orientations, descriptors and octaves aligned.
    fn map_keypoints(&self, features: &mut Features) {
        let mut kept = Vec::with_capacity(features.len());
        let mut keypoints_xy = Vec::with_capacity(features.len());
        for (i, &[u, v]) in features.keypoints_xy.iter().enumerate() {
            let Some(bearing) = self
                .model
                .unproject(&Vec2F64::new(u as f64, v as f64))
                .filter(|bearing| bearing.z > self.min_bearing_z)
            else {
                continue;
            };
            let xn = bearing.x / bearing.z;
            let yn = bearing.y / bearing.z;
            let pu = self.camera.fx * xn + self.camera.cx;
            let pv = self.camera.fy * yn + self.camera.cy;
            keypoints_xy.push([pu as f32, pv as f32]);
            kept.push(i);
        }

        features.keypoints_xy = keypoints_xy;
        features.orientations = features
            .orientations
            .as_ref()
            .map(|orientations| kept.iter().map(|&i| orientations[i]).collect());
        features.descriptors = kept.iter().map(|&i| features.descriptors[i]).collect();
        features.octaves = kept.iter().map(|&i| features.octaves[i]).collect();
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
pub(crate) mod tests;
