//! Detector-independent feature sets: the keypoints and descriptors a frame
//! carries, what a descriptor type must provide, and the scale pyramid that
//! gives per-keypoint octaves their meaning.

use std::cmp::Ordering;
use std::fmt::Debug;

use bytemuck::TransparentWrapper;
use kornia_imgproc::features::{
    OrbFeatures, OrbMatchConfig, hamming_distance, match_orb_descriptors,
};

/// A descriptor type kornia-slam can match and aggregate into landmarks.
pub trait Descriptor: Copy + Debug + PartialEq + Send + Sync + 'static {
    /// Dissimilarity between two descriptors; smaller is more alike.
    type Distance: Copy + PartialOrd + Debug + Send + Sync;
    /// Settings of [`Descriptor::match_sets`].
    type MatchConfig: Clone + Debug + Send + Sync;

    /// Distance within which a guided search accepts a match without a ratio
    /// test (ORB-SLAM3's `ORBmatcher::TH_LOW`).
    const TH_LOW: Self::Distance;

    /// Must never be NaN or otherwise unordered: matchers keep the first
    /// best candidate and compare thresholds with `<` and `>`.
    fn distance(&self, other: &Self) -> Self::Distance;

    /// Unguided matching of `query` against `candidates`: at most one
    /// candidate per query, as `(query, candidate)` index pairs in query order.
    fn match_sets(
        query: DescriptorSet<'_, Self>,
        candidates: DescriptorSet<'_, Self>,
        config: &Self::MatchConfig,
    ) -> Vec<(usize, usize)>;
}

/// ORB's 256-bit rotated-BRIEF descriptor, compared by Hamming distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TransparentWrapper)]
#[repr(transparent)]
pub struct RBrief(pub [u8; 32]);

impl Descriptor for RBrief {
    type Distance = u32;
    type MatchConfig = OrbMatchConfig;

    const TH_LOW: u32 = 50;

    fn distance(&self, other: &Self) -> u32 {
        hamming_distance(&self.0, &other.0)
    }

    fn match_sets(
        query: DescriptorSet<'_, Self>,
        candidates: DescriptorSet<'_, Self>,
        config: &OrbMatchConfig,
    ) -> Vec<(usize, usize)> {
        let query_descriptors = RBrief::peel_slice(query.descriptors);
        let candidate_descriptors = RBrief::peel_slice(candidates.descriptors);
        match (query.orientations, candidates.orientations) {
            (Some(query_orientations), Some(candidate_orientations)) => match_orb_descriptors(
                query_orientations,
                query_descriptors,
                candidate_orientations,
                candidate_descriptors,
                *config,
            ),
            _ => match_orb_descriptors(
                &vec![0.0; query_descriptors.len()],
                query_descriptors,
                &vec![0.0; candidate_descriptors.len()],
                candidate_descriptors,
                OrbMatchConfig {
                    check_orientation: false,
                    ..*config
                },
            ),
        }
    }
}

/// Descriptors with their keypoint orientations, if the detector has them:
/// the input of [`Descriptor::match_sets`].
#[derive(Debug, Clone, Copy)]
pub struct DescriptorSet<'a, D> {
    /// Parallel to `descriptors` when present.
    pub orientations: Option<&'a [f32]>,
    pub descriptors: &'a [D],
}

/// Per-keypoint parallel vectors for one image. Index `i` in every vector
/// describes the same keypoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Features<D = RBrief> {
    /// Keypoints as `[col, row]` in full-resolution pixels.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Keypoint orientations in radians, for detectors that estimate one.
    pub orientations: Option<Vec<f32>>,
    pub descriptors: Vec<D>,
    /// Pyramid level each keypoint was detected at (0 = full resolution);
    /// see [`ScalePyramid`].
    pub octaves: Vec<u8>,
}

impl<D> Default for Features<D> {
    fn default() -> Self {
        Self {
            keypoints_xy: Vec::new(),
            orientations: None,
            descriptors: Vec::new(),
            octaves: Vec::new(),
        }
    }
}

impl<D> Features<D> {
    pub fn descriptor_set(&self) -> DescriptorSet<'_, D> {
        DescriptorSet {
            orientations: self.orientations.as_deref(),
            descriptors: &self.descriptors,
        }
    }

    pub fn len(&self) -> usize {
        self.keypoints_xy.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints_xy.is_empty()
    }
}

impl From<OrbFeatures> for Features<RBrief> {
    fn from(orb: OrbFeatures) -> Self {
        Self {
            keypoints_xy: orb.keypoints_xy,
            orientations: Some(orb.orientations),
            descriptors: orb.descriptors.into_iter().map(RBrief).collect(),
            octaves: orb.octaves,
        }
    }
}

/// The detection pyramid octaves index into: level `o` is the image
/// downscaled by `factor^o`, for `o < n_levels`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalePyramid {
    pub factor: f64,
    pub n_levels: usize,
}

impl ScalePyramid {
    /// ORB-SLAM3's default pyramid, which kornia-imgproc's `OrbDetector` also
    /// uses (`downscale = 1.2`, `n_scales = 8`).
    pub const ORB: Self = Self {
        factor: 1.2,
        n_levels: 8,
    };
}

/// Index of the item whose descriptor has the smallest median distance to all
/// the others (ORB-SLAM3's `ComputeDistinctiveDescriptors`); the first one on
/// ties, and `None` for an empty set. O(n^2).
pub fn most_distinctive<T, D: Descriptor>(
    items: &[T],
    descriptor: impl Fn(&T) -> &D,
) -> Option<usize> {
    let n = items.len();
    let mid = n / 2;
    let mut distances = Vec::with_capacity(n);
    let mut best: Option<(usize, D::Distance)> = None;
    for (i, candidate) in items.iter().map(&descriptor).enumerate() {
        distances.clear();
        distances.extend(
            items
                .iter()
                .map(|other| candidate.distance(descriptor(other))),
        );
        distances.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median = distances[mid];
        if best.is_none_or(|(_, best_median)| median < best_median) {
            best = Some((i, median));
        }
    }
    best.map(|(i, _)| i)
}

/// A one-dimensional float descriptor for exercising the generic paths with
/// a non-Hamming distance and no orientations.
#[cfg(test)]
pub(crate) mod scalar {
    use super::{Descriptor, DescriptorSet};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) struct Scalar(pub f32);

    impl Descriptor for Scalar {
        type Distance = f32;
        type MatchConfig = ();

        const TH_LOW: f32 = 0.5;

        fn distance(&self, other: &Self) -> f32 {
            (self.0 - other.0).abs()
        }

        fn match_sets(
            query: DescriptorSet<'_, Self>,
            candidates: DescriptorSet<'_, Self>,
            _config: &(),
        ) -> Vec<(usize, usize)> {
            query
                .descriptors
                .iter()
                .enumerate()
                .filter_map(|(i, q)| {
                    let (j, d) = candidates
                        .descriptors
                        .iter()
                        .map(|c| q.distance(c))
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(&b.1))?;
                    (d <= Self::TH_LOW).then_some((i, j))
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orb(bits_set: usize) -> RBrief {
        let mut d = [0u8; 32];
        for bit in 0..bits_set {
            d[bit / 8] |= 1 << (bit % 8);
        }
        RBrief(d)
    }

    #[test]
    fn most_distinctive_picks_the_smallest_median_distance() {
        // Upper medians of each row's Hamming distances: 29, 1, 2, 2.
        let set = [orb(30), orb(1), orb(0), orb(2)];
        assert_eq!(most_distinctive(&set, |d| d), Some(1));
        // Equal medians resolve to the first descriptor.
        assert_eq!(most_distinctive(&[orb(4), orb(4)], |d| d), Some(0));
        assert_eq!(most_distinctive::<RBrief, RBrief>(&[], |d| d), None);
    }

    #[test]
    fn most_distinctive_works_for_float_distances() {
        use scalar::Scalar;
        // Upper medians: 9.6, 0.4, 0.5, 0.5.
        let set = [Scalar(10.0), Scalar(0.4), Scalar(0.0), Scalar(0.5)];
        assert_eq!(most_distinctive(&set, |d| d), Some(1));
    }

    #[test]
    fn orb_matching_accepts_sets_without_orientations() {
        let candidates = [orb(0), orb(40)];
        let query = DescriptorSet {
            orientations: None,
            descriptors: &[orb(0)],
        };
        let candidates = DescriptorSet {
            orientations: Some(&[0.0, 1.0]),
            descriptors: &candidates,
        };
        let config = OrbMatchConfig::default();
        assert_eq!(RBrief::match_sets(query, candidates, &config), vec![(0, 0)]);
    }

    #[test]
    fn orb_features_convert_field_by_field() {
        let orb_features = OrbFeatures {
            keypoints_xy: vec![[1.0, 2.0]],
            orientations: vec![0.5],
            descriptors: vec![orb(3).0],
            octaves: vec![2],
        };
        let features = Features::from(orb_features.clone());
        assert_eq!(features.keypoints_xy, orb_features.keypoints_xy);
        assert_eq!(features.orientations, Some(orb_features.orientations));
        assert_eq!(features.descriptors, [orb(3)]);
        assert_eq!(features.octaves, orb_features.octaves);
    }
}
