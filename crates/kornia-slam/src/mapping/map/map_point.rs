//! Map point: a triangulated landmark and its observations.

use kornia_algebra::Vec3F64;
use kornia_imgproc::features::hamming_distance;

/// Identifies one feature slot in one keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObservationKey {
    pub keyframe_idx: usize,
    pub feature_idx: usize,
}

/// One landmark observation: where it was seen, and the descriptor that
/// feature contributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandmarkObservation {
    pub key: ObservationKey,
    pub descriptor: [u8; 32],
}

/// ORB pyramid scale factor between adjacent levels (matches the kornia-imgproc
/// ORB extractor default `downscale = 1.2`, which equals ORB-SLAM3's default).
pub const ORB_SCALE_FACTOR: f64 = 1.2;
/// Number of ORB pyramid levels (matches `OrbDetector` default `n_scales = 8`).
pub const ORB_N_LEVELS: usize = 8;
/// A persistent 3D landmark in the map.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// 3D position in world frame.
    pub position: Vec3F64,
    /// Representative ORB descriptor used for projection-guided matching.
    /// Recomputed as the descriptor with minimum median Hamming distance to
    /// all observations whenever a new observation is added (mirrors
    /// ORB-SLAM3's `MapPoint::ComputeDistinctiveDescriptors`).
    pub descriptor: [u8; 32],
    /// Where this landmark has been seen, in insertion order. One record per
    /// link, and a keyframe appears at most once — enforced by
    /// [`MapPoint::add_observation`] rather than left to each caller.
    observations: Vec<LandmarkObservation>,
    /// Pyramid octave of this point's keypoint in the reference keyframe
    /// (`keyframe_idx`). Drives the scale-invariance distance bounds.
    pub reference_octave: u8,
    /// Mean unit viewing direction (world -> point), averaged over observing
    /// keyframes (ORB-SLAM3's `mNormalVector`). Zero until computed.
    pub mean_viewing_direction: Vec3F64,
    /// Minimum / maximum camera-to-point distance at which the point is
    /// expected to be matchable (ORB-SLAM3's `mfMinDistance` / `mfMaxDistance`,
    /// the raw values before the 0.8 / 1.2 query margins). Zero until computed.
    pub min_distance: f64,
    pub max_distance: f64,
    /// Pixel color sampled at the keypoint that created this point.
    pub color: [u8; 3],
    /// Frame index of the reference keyframe for this point's scale geometry.
    pub keyframe_idx: usize,
    /// Number of frames where this point was in the camera frustum.
    pub n_visible: u32,
    /// Number of frames where this point was successfully matched.
    pub n_found: u32,
    /// Whether this point has been culled (logically deleted).
    pub culled: bool,
}
impl MapPoint {
    /// Creates a fresh active map point with one observed descriptor from
    /// keyframe `keyframe_idx`, detected at pyramid octave `octave`.
    pub fn new(
        position: Vec3F64,
        descriptor: [u8; 32],
        octave: u8,
        color: [u8; 3],
        keyframe_idx: usize,
        feature_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            observations: vec![LandmarkObservation {
                key: ObservationKey {
                    keyframe_idx,
                    feature_idx,
                },
                descriptor,
            }],
            reference_octave: octave,
            mean_viewing_direction: Vec3F64::ZERO,
            min_distance: 0.0,
            max_distance: 0.0,
            color,
            keyframe_idx,
            n_visible: 1,
            n_found: 1,
            culled: false,
        }
    }

    /// Marks the point as logically deleted.
    pub fn mark_culled(&mut self) {
        self.culled = true;
    }

    /// Returns the tracking success ratio for this point.
    pub fn found_ratio(&self) -> f64 {
        if self.n_visible == 0 {
            return 0.0;
        }
        self.n_found as f64 / self.n_visible as f64
    }

    /// Records a new observed descriptor from keyframe `kf_idx` and refreshes
    /// the representative.
    ///
    /// The representative is the observed descriptor with minimum median
    /// Hamming distance to all others (ORB-SLAM3's
    /// `ComputeDistinctiveDescriptors`). With <=2 observations the choice is
    /// trivial; with >=3 we run the O(n^2) pairwise distance scan.
    /// Links this landmark to one feature slot, and reports whether the link
    /// was new.
    ///
    /// A keyframe that already observes this landmark is refused. Duplicate
    /// links inflate covisibility weights and local-map keyframe votes, both of
    /// which count entries, so the invariant is enforced here rather than left
    /// to each caller to remember.
    pub fn add_observation(&mut self, key: ObservationKey, descriptor: [u8; 32]) -> bool {
        if self.is_observed_by(key.keyframe_idx) {
            return false;
        }
        self.observations
            .push(LandmarkObservation { key, descriptor });
        self.recompute_representative_descriptor();
        true
    }

    /// Every link, in insertion order.
    pub fn observations(&self) -> &[LandmarkObservation] {
        &self.observations
    }

    /// Whether `keyframe_idx` observes this landmark.
    pub fn is_observed_by(&self, keyframe_idx: usize) -> bool {
        self.observations
            .iter()
            .any(|observation| observation.key.keyframe_idx == keyframe_idx)
    }

    /// The observing keyframes, in insertion order.
    pub fn observer_keyframes(&self) -> impl Iterator<Item = usize> + '_ {
        self.observations
            .iter()
            .map(|observation| observation.key.keyframe_idx)
    }

    /// Removes the link from `keyframe_idx`, if any, and reports whether one
    /// was removed. The representative descriptor is refreshed, so it can move
    /// when the winning observation goes.
    pub fn remove_observation(&mut self, keyframe_idx: usize) -> bool {
        let before = self.observations.len();
        self.observations
            .retain(|observation| observation.key.keyframe_idx != keyframe_idx);
        let removed = self.observations.len() != before;
        if removed {
            self.recompute_representative_descriptor();
        }
        removed
    }

    fn recompute_representative_descriptor(&mut self) {
        let n = self.observations.len();
        match n {
            0 => {}
            1 => self.descriptor = self.observations[0].descriptor,
            2 => self.descriptor = self.observations[0].descriptor,
            _ => {
                let mut dist_buf = vec![0u32; n];
                let mut best_idx = 0usize;
                let mut best_median = u32::MAX;
                for i in 0..n {
                    for (slot, other) in dist_buf.iter_mut().zip(self.observations.iter()) {
                        *slot =
                            hamming_distance(&self.observations[i].descriptor, &other.descriptor);
                    }
                    let mid = n / 2;
                    dist_buf.select_nth_unstable(mid);
                    let median = dist_buf[mid];
                    if median < best_median {
                        best_median = median;
                        best_idx = i;
                    }
                }
                self.descriptor = self.observations[best_idx].descriptor;
            }
        }
    }

    /// Predicts the ORB pyramid level the point is expected to be observable
    /// at given the current camera-to-point distance (ORB-SLAM3's
    /// `MapPoint::PredictScale`). Returns 0 when the scale geometry is unset.
    pub fn predict_scale(&self, distance: f64, scale_factor: f64, n_levels: usize) -> usize {
        if distance <= 0.0 || self.max_distance <= 0.0 || n_levels == 0 {
            return 0;
        }
        let ratio = self.max_distance / distance;
        let level = (ratio.ln() / scale_factor.ln()).ceil();
        if level.is_nan() || level < 0.0 {
            0
        } else {
            (level as usize).min(n_levels - 1)
        }
    }

    /// Lower bound of the matchable distance range, with ORB-SLAM3's 0.8 margin.
    pub fn min_distance_invariance(&self) -> f64 {
        0.8 * self.min_distance
    }

    /// Upper bound of the matchable distance range, with ORB-SLAM3's 1.2 margin.
    pub fn max_distance_invariance(&self) -> f64 {
        1.2 * self.max_distance
    }
}

/// A triangulated point ready for map insertion: (position, descriptor, color, prev_desc_idx, curr_desc_idx).
pub type TriangulatedPoint = (Vec3F64, [u8; 32], [u8; 3], usize, usize);

#[cfg(test)]
mod tests {
    use super::{MapPoint, ObservationKey};
    use kornia_algebra::Vec3F64;

    fn landmark() -> MapPoint {
        MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0, 0)
    }

    fn key(keyframe_idx: usize, feature_idx: usize) -> ObservationKey {
        ObservationKey {
            keyframe_idx,
            feature_idx,
        }
    }

    /// A keyframe already observing the landmark is refused, whatever feature
    /// slot it offers. Duplicate links inflate covisibility weights and
    /// local-map votes, both of which count entries.
    #[test]
    fn a_keyframe_cannot_observe_the_same_landmark_twice() {
        let mut mp = landmark();
        assert_eq!(mp.observations().len(), 1, "the reference observation");

        assert!(!mp.add_observation(key(0, 7), [1u8; 32]), "same keyframe");
        assert_eq!(mp.observations().len(), 1);

        assert!(
            mp.add_observation(key(1, 7), [1u8; 32]),
            "different keyframe"
        );
        assert_eq!(mp.observations().len(), 2);
    }

    /// Insertion order is preserved, which the representative-descriptor
    /// tie-break depends on.
    #[test]
    fn observations_keep_insertion_order() {
        let mut mp = landmark();
        mp.add_observation(key(5, 1), [5u8; 32]);
        mp.add_observation(key(3, 2), [3u8; 32]);

        assert_eq!(
            mp.observer_keyframes().collect::<Vec<_>>(),
            vec![0, 5, 3],
            "not sorted — insertion order"
        );
    }

    /// Each record carries the feature that produced its descriptor, so a
    /// descriptor can be traced back to the keypoint it came from.
    #[test]
    fn each_observation_names_the_feature_that_produced_it() {
        let mut mp = landmark();
        mp.add_observation(key(4, 11), [7u8; 32]);

        let observation = mp
            .observations()
            .iter()
            .find(|o| o.key.keyframe_idx == 4)
            .expect("the link was added");
        assert_eq!(observation.key.feature_idx, 11);
        assert_eq!(observation.descriptor, [7u8; 32]);
    }

    /// Removing the winning observation must move the representative, not leave
    /// a descriptor no observation supports.
    #[test]
    fn the_representative_moves_when_its_observation_is_removed() {
        let mut mp = landmark();
        mp.add_observation(key(1, 0), [0xFFu8; 32]);
        mp.add_observation(key(2, 0), [0xFFu8; 32]);
        assert_eq!(mp.descriptor, [0xFFu8; 32], "the majority descriptor wins");

        assert!(mp.remove_observation(1));
        assert!(mp.remove_observation(2));

        assert_eq!(mp.descriptor, [0u8; 32], "only the reference remains");
        assert!(
            !mp.remove_observation(99),
            "removing an absent link is a no-op"
        );
    }

    #[test]
    fn is_observed_by_reports_membership() {
        let mut mp = landmark();
        mp.add_observation(key(9, 3), [1u8; 32]);

        assert!(mp.is_observed_by(0));
        assert!(mp.is_observed_by(9));
        assert!(!mp.is_observed_by(4));
    }
}
