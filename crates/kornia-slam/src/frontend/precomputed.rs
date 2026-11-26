//! A frontend that looks up features exported offline by image timestamp
//! instead of detecting them, so a learned extractor without a Rust port can
//! drive the system.

use std::path::Path;
use std::time::Instant;

use kornia_image::Image;
use kornia_sensors::SensorRig;

use super::feature_file::{FeatureFile, FeatureFileError};
use super::{FeatureFrontend, FisheyeMapping, FrontendError, FrontendObservation, assemble_frame};
use crate::Frame;
use crate::features::{Clidd, ScalePyramid};

/// Monocular [`Clidd`] features from a feature file, keyed by timestamp.
/// Keypoints are in the pixels of the image the system is given: raw for a
/// fisheye rig, which maps them like detected ones.
pub(crate) struct PrecomputedFrontend {
    file: FeatureFile,
    fisheye: Option<FisheyeMapping>,
    observation: FrontendObservation,
}

impl PrecomputedFrontend {
    pub(crate) fn open(path: &Path, rig: &SensorRig) -> Result<Self, FeatureFileError> {
        Ok(Self {
            file: FeatureFile::open(path)?,
            fisheye: FisheyeMapping::for_rig(rig),
            observation: FrontendObservation::default(),
        })
    }
}

impl FeatureFrontend<Clidd> for PrecomputedFrontend {
    /// CLIDD detects at one scale.
    fn scale_pyramid(&self) -> ScalePyramid {
        ScalePyramid::SINGLE_SCALE
    }

    fn prepare_image(
        &mut self,
        idx: usize,
        timestamp_sec: f64,
        image: &Image<u8, 1>,
    ) -> Result<Frame<Clidd>, FrontendError> {
        let start = Instant::now();
        let features = self
            .file
            .features_near(timestamp_sec)?
            .ok_or(FrontendError::MissingFeatures { timestamp_sec })?;
        let (width, height) = (image.width() as f32, image.height() as f32);
        let inside = |&[x, y]: &[f32; 2]| (0.0..width).contains(&x) && (0.0..height).contains(&y);
        if !features.keypoints_xy.iter().all(inside) {
            return Err(FrontendError::KeypointsOutsideImage { timestamp_sec });
        }
        Ok(assemble_frame(
            idx,
            image,
            features,
            None,
            self.fisheye.as_ref(),
            &mut self.observation,
            start,
        ))
    }

    /// Precomputed features carry no stereo matches: the right image is ignored.
    fn prepare_stereo_pair(
        &mut self,
        idx: usize,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        _right: &Image<u8, 1>,
    ) -> Result<Frame<Clidd>, FrontendError> {
        self.prepare_image(idx, timestamp_sec, left)
    }

    fn observation(&self) -> &FrontendObservation {
        &self.observation
    }
}

#[cfg(test)]
mod tests {
    use kornia_3d::camera::PinholeCamera;
    use kornia_image::ImageSize;

    use super::*;
    use crate::features::CLIDD_DIM;
    use crate::frontend::feature_file::tests::{encode, write};

    fn rig() -> SensorRig {
        SensorRig::new(PinholeCamera {
            fx: 100.0,
            fy: 100.0,
            cx: 16.0,
            cy: 8.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        })
    }

    fn image() -> Image<u8, 1> {
        Image::from_size_val(
            ImageSize {
                width: 32,
                height: 16,
            },
            7,
        )
        .unwrap()
    }

    #[test]
    fn frames_come_from_the_file_by_timestamp() {
        let frames = vec![
            (
                2_000_000_000,
                vec![[3.0, 4.0]],
                vec![Clidd([0.125; CLIDD_DIM])],
            ),
            (
                2_050_000_000,
                vec![[40.0, 4.0]],
                vec![Clidd([0.125; CLIDD_DIM])],
            ),
        ];
        let file = write("frontend", &encode(CLIDD_DIM as u32, &frames));
        let mut frontend = PrecomputedFrontend::open(&file.0, &rig()).unwrap();

        let frame = frontend.prepare_image(5, 2.0, &image()).unwrap();
        assert_eq!(frame.idx, 5);
        assert_eq!(frame.features.keypoints_xy, [[3.0, 4.0]]);
        assert_eq!(frame.keypoint_colors, [[7, 7, 7]]);
        assert_eq!(frontend.observation().keypoints_xy, [[3.0, 4.0]]);

        assert!(matches!(
            frontend.prepare_image(6, 2.05, &image()),
            Err(FrontendError::KeypointsOutsideImage { .. })
        ));
        assert!(matches!(
            frontend.prepare_image(7, 3.0, &image()),
            Err(FrontendError::MissingFeatures { .. })
        ));
    }
}
