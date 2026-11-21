//! Pixel noise of an ORB keypoint, the weight of its reprojection residual.

use kornia_sensors::{Fisheye624Camera, SensorRig};

use crate::mapping::map::ORB_SCALE_FACTOR;
use crate::rig::RigView;

/// How a reprojection residual is weighted by the octave its keypoint came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VisualInformation {
    /// 1 px at every octave.
    #[default]
    Unit,
    /// `scale_factor^octave` px, ORB-SLAM3's `invSigma2 = 1 / scale^(2·octave)`: a keypoint found
    /// at a coarse level is localized only to that level's pixel size.
    OctaveScaled,
}

impl VisualInformation {
    /// Standard deviation (px) of a keypoint detected at `octave`.
    pub fn sigma_px(self, octave: u8) -> f32 {
        match self {
            Self::Unit => 1.0,
            Self::OctaveScaled => ORB_SCALE_FACTOR.powi(i32::from(octave)) as f32,
        }
    }

    /// Whether every residual keeps unit weight, so callers can skip the scaling.
    pub fn is_unit(self) -> bool {
        self == Self::Unit
    }
}

/// How a raw fisheye view's observations are weighted for the stretch of the
/// raw-to-virtual-pinhole mapping (card F4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FisheyeWeight {
    /// Virtual pixels weighted as raw ones.
    #[default]
    Uniform,
    /// A raw pixel of noise spans `pinhole_magnification` virtual pixels, so
    /// the pixel sigma and the projection search radius scale by it.
    Magnification,
}

impl FisheyeWeight {
    /// `Uniform` unless some camera of `rig` supplies raw fisheye images.
    pub fn for_rig(self, rig: &SensorRig) -> Self {
        if rig.cameras.iter().any(|camera| camera.raw_model.is_some()) {
            self
        } else {
            Self::Uniform
        }
    }

    /// Pixel-sigma factor of an observation at the undistorted `pixel` of
    /// `view`: the geometric mean `sqrt(radial · tangential)` of the
    /// magnification (D3), which keeps the information determinant. 1 for
    /// `Uniform` and for views without a raw model.
    pub fn sigma_scale(self, view: Option<&RigView>, pixel: [f32; 2]) -> f32 {
        if self == Self::Uniform {
            return 1.0;
        }
        view.and_then(|view| view.pinhole_magnification(view.incidence_rad(pixel)))
            .map_or(1.0, |(radial, tangential)| {
                (radial * tangential).sqrt() as f32
            })
    }

    /// Projection-search radius factor at incidence `theta` (rad): the radial
    /// magnification, the larger axis of the stretch. 1 for `Uniform` and
    /// without a raw model.
    pub fn radius_scale(self, raw: Option<&Fisheye624Camera>, theta: f64) -> f32 {
        match (self, raw) {
            (Self::Magnification, Some(raw)) => raw.pinhole_magnification(theta).0 as f32,
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FisheyeWeight, VisualInformation};
    use crate::rig::RigGeometry;
    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;
    use kornia_sensors::{Fisheye624Camera, RigCamera, SensorRig};

    /// LaMAria R_11 `cam0`, its virtual pinhole and a two-camera rig of it.
    fn raw_rig(raw: Option<Fisheye624Camera>) -> SensorRig {
        let model = r11_cam0();
        let camera = PinholeCamera {
            fx: model.f,
            fy: model.f,
            cx: model.cx,
            cy: model.cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let view = |primary_from_camera| RigCamera {
            camera: camera.clone(),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            primary_from_camera,
            raw_model: raw,
        };
        SensorRig::new(camera.clone()).with_rig_cameras(vec![
            view(Pose3d::IDENTITY),
            view(Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(0.1, 0.0, 0.0),
            )),
        ])
    }

    fn r11_cam0() -> Fisheye624Camera {
        Fisheye624Camera {
            f: 241.6048925567284,
            cx: 322.895555771704,
            cy: 240.4445599560633,
            k: [
                -0.02484243376501498,
                0.09628938752134535,
                -0.063325466945624,
                0.006200726341569274,
                0.003488761399963396,
                -0.0007297520056856479,
            ],
            p: [0.00089532032215904, 0.000361066580790014],
            s: [
                -0.00136879244387001,
                -0.0001054083415853531,
                -0.0004635611298488213,
                -0.0001049856461262643,
            ],
        }
    }

    #[test]
    fn raw_fisheye_scales_at_70_degrees() {
        let rig = RigGeometry::from_rig(&raw_rig(Some(r11_cam0()))).unwrap();
        let view = rig.view(0).unwrap();
        let theta = 70f64.to_radians();
        let pixel = [
            (view.camera.cx + view.camera.fx * theta.tan()) as f32,
            view.camera.cy as f32,
        ];
        let sigma = FisheyeWeight::Magnification.sigma_scale(Some(view), pixel);
        assert!((sigma - 4.479).abs() < 1e-3, "{sigma}");
        let radius = FisheyeWeight::Magnification.radius_scale(view.raw_model.as_ref(), theta);
        assert!((radius - 9.059).abs() < 1e-3, "{radius}");
        assert_eq!(FisheyeWeight::Uniform.sigma_scale(Some(view), pixel), 1.0);
        assert_eq!(
            FisheyeWeight::Uniform.radius_scale(view.raw_model.as_ref(), theta),
            1.0
        );
        assert_eq!(FisheyeWeight::Magnification.sigma_scale(None, pixel), 1.0);
        assert_eq!(FisheyeWeight::Magnification.radius_scale(None, theta), 1.0);
    }

    #[test]
    fn raw_fisheye_magnification_on_a_pinhole_rig_is_uniform() {
        assert_eq!(
            FisheyeWeight::Magnification.for_rig(&raw_rig(None)),
            FisheyeWeight::Uniform
        );
        assert_eq!(
            FisheyeWeight::Magnification.for_rig(&raw_rig(Some(r11_cam0()))),
            FisheyeWeight::Magnification
        );
        assert_eq!(
            FisheyeWeight::Uniform.for_rig(&raw_rig(Some(r11_cam0()))),
            FisheyeWeight::Uniform
        );
    }

    #[test]
    fn octave_sigma_follows_the_pyramid_scale() {
        assert_eq!(VisualInformation::Unit.sigma_px(7), 1.0);
        assert_eq!(VisualInformation::OctaveScaled.sigma_px(0), 1.0);
        let sigma = VisualInformation::OctaveScaled.sigma_px(3);
        assert!((sigma - 1.728).abs() < 1e-6);
    }
}
