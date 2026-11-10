//! Pixel noise of an ORB keypoint, the weight of its reprojection residual.

use crate::mapping::map::ORB_SCALE_FACTOR;

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

#[cfg(test)]
mod tests {
    use super::VisualInformation;

    #[test]
    fn octave_sigma_follows_the_pyramid_scale() {
        assert_eq!(VisualInformation::Unit.sigma_px(7), 1.0);
        assert_eq!(VisualInformation::OctaveScaled.sigma_px(0), 1.0);
        let sigma = VisualInformation::OctaveScaled.sigma_px(3);
        assert!((sigma - 1.728).abs() < 1e-6);
    }
}
