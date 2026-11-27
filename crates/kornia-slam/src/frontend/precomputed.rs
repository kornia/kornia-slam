//! A frontend that looks up features exported offline by image timestamp
//! instead of detecting them, so a learned extractor without a Rust port can
//! drive the system.

use std::path::Path;
use std::slice;
use std::time::Instant;

use bytemuck::TransparentWrapper;
use kornia_3d::stereo::{StereoDescriptors, StereoKeypoints, StereoMatchConfig, StereoMatcher};
use kornia_image::Image;
use kornia_sensors::SensorRig;

use super::feature_file::FeatureFile;
use super::{FeatureFrontend, FisheyeMapping, FrontendError, FrontendObservation, assemble_frame};
use crate::Frame;
use crate::features::{CLIDD_DIM, Clidd, Descriptor, Features, ScalePyramid};
use crate::stereo::StereoMatches;

/// [`Clidd`] features from feature files, keyed by timestamp. Keypoints are
/// in the pixels of the images the system is given: raw for a fisheye rig,
/// which maps them like detected ones, and rectified for a stereo rig.
pub(crate) struct PrecomputedFrontend {
    file: FeatureFile,
    stereo: Option<PrecomputedStereo>,
    fisheye: Option<FisheyeMapping>,
    observation: FrontendObservation,
}

impl PrecomputedFrontend {
    /// Opens the features of the images the system is given and, for a rig
    /// with a stereo baseline, those of the right view; `rig` must already be
    /// restricted to the selected sensors.
    pub(crate) fn open(
        path: &Path,
        right_path: Option<&Path>,
        rig: &SensorRig,
    ) -> Result<Self, FrontendError> {
        let stereo = match (rig.stereo_baseline_m, right_path) {
            (Some(baseline_m), Some(right)) => {
                Some(PrecomputedStereo::open(right, rig.camera.fx, baseline_m)?)
            }
            _ => None,
        };
        Ok(Self {
            file: FeatureFile::open(path)?,
            stereo,
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
        let features = stored_features(&mut self.file, timestamp_sec, image)?;
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

    /// Without the right view's features the right image is ignored.
    fn prepare_stereo_pair(
        &mut self,
        idx: usize,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<Frame<Clidd>, FrontendError> {
        let Some(stereo) = &mut self.stereo else {
            return self.prepare_image(idx, timestamp_sec, left);
        };
        let start = Instant::now();
        let features = stored_features(&mut self.file, timestamp_sec, left)?;
        let right_features = stored_features(&mut stereo.right, timestamp_sec, right)?;
        let matches = stereo.match_views(left, right, &features, &right_features)?;
        Ok(assemble_frame(
            idx,
            left,
            features,
            Some(matches),
            self.fisheye.as_ref(),
            &mut self.observation,
            start,
        ))
    }

    fn observation(&self) -> &FrontendObservation {
        &self.observation
    }
}

/// The right view's features and the matcher that pairs them with the left's.
struct PrecomputedStereo {
    right: FeatureFile,
    matcher: StereoMatcher,
}

impl PrecomputedStereo {
    fn open(right: &Path, fx: f64, baseline_m: f64) -> Result<Self, FrontendError> {
        let mut config = StereoMatchConfig::new(fx as f32, baseline_m as f32);
        // The guided-search cut: for unit vectors the dot product is
        // 1 - d / 2 at squared distance d.
        config.min_similarity = 1.0 - 0.5 * Clidd::TH_LOW;
        Ok(Self {
            right: FeatureFile::open(right)?,
            matcher: StereoMatcher::new(config)?,
        })
    }

    /// ORB-SLAM3's row search with SAD refinement, run on CLIDD descriptors
    /// at full resolution.
    fn match_views(
        &self,
        left_image: &Image<u8, 1>,
        right_image: &Image<u8, 1>,
        left: &Features<Clidd>,
        right: &Features<Clidd>,
    ) -> Result<StereoMatches, FrontendError> {
        let mut matches = kornia_3d::stereo::StereoMatches::default();
        self.matcher.match_into(
            slice::from_ref(left_image),
            slice::from_ref(right_image),
            &stereo_keypoints(left),
            &stereo_keypoints(right),
            &mut matches,
        )?;
        Ok(StereoMatches {
            u_right: matches.u_right,
            depth: matches.depth,
        })
    }
}

fn stereo_keypoints(features: &Features<Clidd>) -> StereoKeypoints<'_> {
    StereoKeypoints {
        xy: &features.keypoints_xy,
        octaves: None,
        descriptors: StereoDescriptors::Float {
            data: Clidd::peel_slice(&features.descriptors).as_flattened(),
            dim: CLIDD_DIM,
        },
    }
}

/// The stored features of `image`, captured at `timestamp_sec`.
fn stored_features(
    file: &mut FeatureFile,
    timestamp_sec: f64,
    image: &Image<u8, 1>,
) -> Result<Features<Clidd>, FrontendError> {
    let features = file
        .features_near(timestamp_sec)?
        .ok_or(FrontendError::MissingFeatures { timestamp_sec })?;
    let (width, height) = (image.width() as f32, image.height() as f32);
    let inside = |&[x, y]: &[f32; 2]| (0.0..width).contains(&x) && (0.0..height).contains(&y);
    if !features.keypoints_xy.iter().all(inside) {
        return Err(FrontendError::KeypointsOutsideImage { timestamp_sec });
    }
    Ok(features)
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
        let mut frontend = PrecomputedFrontend::open(&file.0, None, &rig()).unwrap();

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

    fn unit(axis: usize) -> Clidd {
        let mut descriptor = [0.0; CLIDD_DIM];
        descriptor[axis] = 1.0;
        Clidd(descriptor)
    }

    #[test]
    fn stereo_pairs_take_depth_from_the_right_views_features() {
        let disparity = 8;
        let (left, right) = crate::frontend::tests::synthetic_pair(96, 48, disparity);
        let stamp = 1_000_000_000;
        let left_file = write(
            "stereo-left",
            &encode(
                CLIDD_DIM as u32,
                &[(
                    stamp,
                    vec![[48.0, 24.0], [60.0, 30.0]],
                    vec![unit(0), unit(1)],
                )],
            ),
        );
        let right_file = write(
            "stereo-right",
            &encode(
                CLIDD_DIM as u32,
                &[(
                    stamp,
                    vec![[44.0, 24.0], [40.0, 24.0]],
                    vec![unit(2), unit(0)],
                )],
            ),
        );
        let stereo_rig = rig().with_stereo_baseline(0.1);
        let mut frontend =
            PrecomputedFrontend::open(&left_file.0, Some(&right_file.0), &stereo_rig).unwrap();

        let frame = frontend.prepare_stereo_pair(0, 1.0, &left, &right).unwrap();
        // Sub-pixel refinement moves the match within the block texture's
        // pixel, not to the next one.
        let u_right = frame.u_right[0];
        assert!(
            (u_right - (48 - disparity) as f32).abs() < 0.5,
            "u_right {u_right}"
        );
        let bf = 100.0 * 0.1;
        assert!((frame.stereo_depth(0).unwrap() - bf / (48.0 - u_right)).abs() < 1e-4);
        assert_eq!(frame.stereo_depth(1), None, "no right feature describes it");
        assert_eq!(frontend.observation().stereo_matched, Some(1));

        let mono = PrecomputedFrontend::open(&left_file.0, Some(&right_file.0), &rig())
            .unwrap()
            .prepare_stereo_pair(0, 1.0, &left, &right)
            .unwrap();
        assert_eq!(
            mono.stereo_depth(0),
            None,
            "a rig without a baseline is monocular"
        );
    }
}
