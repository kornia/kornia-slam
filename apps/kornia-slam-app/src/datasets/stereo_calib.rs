//! YAML stereo calibration for non-rectified sources (e.g. OAK-D mono pair).
//!
//! A device's factory calibration (per-camera intrinsics + Brown-Conrady
//! distortion, plus the left→right extrinsic) is dumped to YAML once and loaded
//! here to build a [`StereoRectifier`]. Distortion is the 8-coefficient OpenCV
//! order `[k1, k2, p1, p2, k3, k4, k5, k6]`; the rotation is row-major 3×3 and
//! the translation is in metres.
//!
//! TODO: this is a generic stereo-calibration container + loader, not specific
//! to this example. Move it into `kornia-imgproc::calibration` upstream
//! alongside the generalized rectifier (see the matching TODO in `rectify.rs`),
//! so any kornia consumer can load a stereo calib and build a rectifier.

use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
use serde::Deserialize;

use kornia_3d::stereo::{CameraCalib, StereoError, StereoRectifier};

#[derive(Deserialize)]
struct CamYaml {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    /// `[k1, k2, p1, p2, k3, k4, k5, k6]` (OpenCV order).
    distortion: [f64; 8],
}

/// Deserialized stereo calibration (see module docs for the schema).
#[derive(Deserialize)]
pub struct StereoCalib {
    /// Image width (pixels) both views are calibrated for.
    pub width: usize,
    /// Image height (pixels) both views are calibrated for.
    pub height: usize,
    left: CamYaml,
    right: CamYaml,
    /// Left→right rotation, row-major 3×3.
    r_left_to_right: [f64; 9],
    /// Left→right translation in metres.
    t_left_to_right_m: [f64; 3],
}

impl StereoCalib {
    /// Loads a calibration YAML from `path`.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&text)?)
    }

    /// Builds a [`StereoRectifier`] from this calibration.
    pub fn rectifier(&self) -> Result<StereoRectifier, StereoError> {
        let calib = |c: &CamYaml| CameraCalib {
            width: self.width,
            height: self.height,
            fx: c.fx,
            fy: c.fy,
            cx: c.cx,
            cy: c.cy,
            distortion: PolynomialDistortion {
                k1: c.distortion[0],
                k2: c.distortion[1],
                p1: c.distortion[2],
                p2: c.distortion[3],
                k3: c.distortion[4],
                k4: c.distortion[5],
                k5: c.distortion[6],
                k6: c.distortion[7],
            },
        };
        // Row-major [r00,r01,r02, r10,r11,r12, r20,r21,r22] → columns.
        let m = &self.r_left_to_right;
        let r = Mat3F64::from_cols(
            Vec3F64::new(m[0], m[3], m[6]),
            Vec3F64::new(m[1], m[4], m[7]),
            Vec3F64::new(m[2], m[5], m[8]),
        );
        let t = Vec3F64::new(
            self.t_left_to_right_m[0],
            self.t_left_to_right_m[1],
            self.t_left_to_right_m[2],
        );
        StereoRectifier::from_calib(&calib(&self.left), &calib(&self.right), r, t)
    }
}
