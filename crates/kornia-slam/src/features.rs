//! Detector-independent feature sets: the keypoints and descriptors a frame
//! carries, what a descriptor type must provide, and the scale pyramid that
//! gives per-keypoint octaves their meaning.

use std::cmp::Ordering;
use std::fmt::Debug;

use kornia_imgproc::features::{OrbFeatures, hamming_distance};

/// A descriptor type kornia-slam can match and aggregate into landmarks.
pub trait Descriptor: Copy + Debug + PartialEq + Send + Sync + 'static {
    /// Dissimilarity between two descriptors; smaller is more alike.
    type Distance: Copy + PartialOrd + Debug + Send + Sync;

    fn distance(&self, other: &Self) -> Self::Distance;
}

/// ORB's 256-bit rBRIEF descriptor, compared by Hamming distance.
impl Descriptor for [u8; 32] {
    type Distance = u32;

    fn distance(&self, other: &Self) -> u32 {
        hamming_distance(self, other)
    }
}

/// Per-keypoint parallel vectors for one image. Index `i` in every vector
/// describes the same keypoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Features<D = [u8; 32]> {
    /// Keypoints as `[col, row]` in full-resolution pixels.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Keypoint orientations in radians; empty for detectors without them.
    pub orientations: Vec<f32>,
    pub descriptors: Vec<D>,
    /// Pyramid level each keypoint was detected at (0 = full resolution);
    /// see [`ScalePyramid`].
    pub octaves: Vec<u8>,
}

impl<D> Features<D> {
    pub fn len(&self) -> usize {
        self.keypoints_xy.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints_xy.is_empty()
    }
}

impl From<OrbFeatures> for Features<[u8; 32]> {
    fn from(orb: OrbFeatures) -> Self {
        Self {
            keypoints_xy: orb.keypoints_xy,
            orientations: orb.orientations,
            descriptors: orb.descriptors,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn orb(bits_set: usize) -> [u8; 32] {
        let mut d = [0u8; 32];
        for bit in 0..bits_set {
            d[bit / 8] |= 1 << (bit % 8);
        }
        d
    }

    #[test]
    fn most_distinctive_picks_the_smallest_median_distance() {
        // Upper medians of each row's Hamming distances: 29, 1, 2, 2.
        let set = [orb(30), orb(1), orb(0), orb(2)];
        assert_eq!(most_distinctive(&set, |d| d), Some(1));
        // Equal medians resolve to the first descriptor.
        assert_eq!(most_distinctive(&[orb(4), orb(4)], |d| d), Some(0));
        assert_eq!(most_distinctive::<[u8; 32], [u8; 32]>(&[], |d| d), None);
    }

    #[test]
    fn orb_features_convert_field_by_field() {
        let orb = OrbFeatures {
            keypoints_xy: vec![[1.0, 2.0]],
            orientations: vec![0.5],
            descriptors: vec![orb(3)],
            octaves: vec![2],
        };
        let features = Features::from(orb.clone());
        assert_eq!(features.keypoints_xy, orb.keypoints_xy);
        assert_eq!(features.orientations, orb.orientations);
        assert_eq!(features.descriptors, orb.descriptors);
        assert_eq!(features.octaves, orb.octaves);
    }
}
