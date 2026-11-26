//! CLIDD's learned float descriptor (Yao et al., "CLIDD: Cross-Layer
//! Independent Deformable Description", arXiv:2601.09230, 2026), as
//! `kornia.feature.CLIDD` produces it.

use bytemuck::{Pod, TransparentWrapper, Zeroable};
use kornia_imgproc::features::{l2_sq, match_descriptors_f32};

use super::{Descriptor, DescriptorSet};

/// Floats per descriptor of the 64-D variants (`N64`, `L64`, ...).
pub const CLIDD_DIM: usize = 64;

/// An L2-normalised CLIDD descriptor, compared by squared L2 distance:
/// `2 - 2 cos`, in `[0, 4]`.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, TransparentWrapper)]
#[repr(transparent)]
pub struct Clidd(pub [f32; CLIDD_DIM]);

// Thresholds follow ORB's through the random-hyperplane relation between a
// normalised Hamming distance and an angle, theta = pi * bits / 256
// (Charikar, 2002): ORB-SLAM3's TH_LOW = 50 is about 35 degrees, cos 0.82, the
// same cut as XFeat's `min_cossim` default.
impl Descriptor for Clidd {
    type Distance = f32;
    type MatchConfig = CliddMatchConfig;

    const TH_LOW: f32 = 0.37;

    fn distance(&self, other: &Self) -> f32 {
        let distance = l2_sq(&self.0, &other.0);
        if distance.is_nan() {
            f32::INFINITY
        } else {
            distance
        }
    }

    fn match_sets(
        query: DescriptorSet<'_, Self>,
        candidates: DescriptorSet<'_, Self>,
        config: &CliddMatchConfig,
    ) -> Vec<(usize, usize)> {
        // For unit vectors 1 - cos is half the squared distance, so the
        // cosine-space ratio of kornia-imgproc is the squared-distance ratio.
        match_descriptors_f32(
            Clidd::peel_slice(query.descriptors),
            Clidd::peel_slice(candidates.descriptors),
            Some(1.0 - 0.5 * config.max_distance),
            config.mutual,
            Some(config.max_ratio),
        )
    }
}

/// Unguided CLIDD matching: nearest neighbours within `max_distance` that
/// pass Lowe's ratio test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CliddMatchConfig {
    /// Largest squared L2 distance accepted.
    pub max_distance: f32,
    /// Largest best / second-best ratio of squared distances; 1 disables the test.
    pub max_ratio: f32,
    /// Keep only mutual nearest neighbours.
    pub mutual: bool,
}

impl Default for CliddMatchConfig {
    /// The mutual ratio test (0.9 on distances) that found 2.7x ORB's correct
    /// inliers on EuRoC pairs, capped at [`Clidd::TH_LOW`] as ORB's matcher is.
    fn default() -> Self {
        Self {
            max_distance: Clidd::TH_LOW,
            max_ratio: 0.81,
            mutual: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit axis `axis`, turned by `angle` radians towards the last axis.
    fn near(axis: usize, angle: f32) -> Clidd {
        let mut d = [0.0; CLIDD_DIM];
        d[axis] = angle.cos();
        d[CLIDD_DIM - 1] = angle.sin();
        Clidd(d)
    }

    fn set(descriptors: &[Clidd]) -> DescriptorSet<'_, Clidd> {
        DescriptorSet {
            orientations: None,
            descriptors,
        }
    }

    #[test]
    fn distance_is_twice_one_minus_cosine() {
        let a = near(1, 0.0);
        let b = near(1, std::f32::consts::FRAC_PI_3);
        assert!((a.distance(&b) - 1.0).abs() < 1e-6);
        assert_eq!(a.distance(&a), 0.0);
    }

    #[test]
    fn distance_is_never_nan() {
        let nan = Clidd([f32::NAN; CLIDD_DIM]);
        assert_eq!(nan.distance(&near(1, 0.0)), f32::INFINITY);
    }

    #[test]
    fn matching_pairs_close_distinct_neighbours() {
        let query = [near(1, 0.1), near(2, 0.1)];
        let candidates = [near(2, 0.0), near(1, 0.0), near(3, 0.0)];
        let config = CliddMatchConfig::default();
        assert_eq!(
            Clidd::match_sets(set(&query), set(&candidates), &config),
            vec![(0, 1), (1, 0)]
        );
    }

    #[test]
    fn matching_rejects_far_and_ambiguous_neighbours() {
        let config = CliddMatchConfig::default();
        let far = [near(1, 1.2)];
        let candidates = [near(1, 0.0), near(2, 0.0)];
        assert!(Clidd::match_sets(set(&far), set(&candidates), &config).is_empty());

        let query = [near(1, 0.0)];
        let twins = [near(1, 0.1), near(1, 0.1)];
        assert!(Clidd::match_sets(set(&query), set(&twins), &config).is_empty());
    }
}
