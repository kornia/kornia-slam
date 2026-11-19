//! Persistent 3D landmarks: descriptor aggregation and scale-invariance geometry.
//!
//! A `MapPoint` owns what it can derive from itself. Anything needing the
//! observing keyframes — the viewing direction and distance bounds, which read
//! camera centres — lives in the map's `ops::geometry` group and only writes
//! the fields back here. Corrections decide which landmarks a moved camera
//! made stale; nothing recomputes these fields from a `MapPoint` alone.

use kornia_algebra::Vec3F64;

use crate::features::{Descriptor, ScalePyramid, most_distinctive};

/// Identifies one feature slot in one keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObservationKey {
    pub keyframe_idx: usize,
    pub feature_idx: usize,
}

/// One landmark observation: where it was seen, and the descriptor that
/// feature contributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandmarkObservation<D = [u8; 32]> {
    pub key: ObservationKey,
    pub descriptor: D,
}

#[derive(Debug, Clone)]
pub struct MapPoint<D = [u8; 32]> {
    /// World-frame position.
    pub position: Vec3F64,
    /// Representative descriptor for projection-guided matching, picked by
    /// `recompute_representative_descriptor`.
    pub descriptor: D,
    /// Where this landmark has been seen, in insertion order. One record per
    /// link; a keyframe appears at most once.
    observations: Vec<LandmarkObservation<D>>,
    /// Octave in the reference keyframe; drives the distance bounds below.
    pub reference_octave: u8,
    /// ORB-SLAM3's `mNormalVector`. Zero until computed.
    pub mean_viewing_direction: Vec3F64,
    /// Matchable distance range, raw — before the 0.8 / 1.2 query margins
    /// applied by `min_distance_invariance` / `max_distance_invariance`.
    /// Zero until computed.
    pub min_distance: f64,
    pub max_distance: f64,
    pub color: [u8; 3],
    /// Reference keyframe for this point's scale geometry.
    pub keyframe_idx: usize,
    /// Frames where the point fell in the frustum, and where it then matched.
    pub n_visible: u32,
    pub n_found: u32,
    /// Logically deleted; still occupies its slot.
    pub culled: bool,
}

impl<D: Descriptor> MapPoint<D> {
    pub fn new(
        position: Vec3F64,
        descriptor: D,
        octave: u8,
        color: [u8; 3],
        keyframe_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            observations: Vec::new(),
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

    pub fn mark_culled(&mut self) {
        self.culled = true;
    }

    pub fn found_ratio(&self) -> f64 {
        if self.n_visible == 0 {
            return 0.0;
        }
        self.n_found as f64 / self.n_visible as f64
    }

    /// Records a link. Returns false if this keyframe already observes the
    /// landmark, leaving the records untouched.
    ///
    /// Does not finalize: the representative descriptor is stale until
    /// [`MapPoint::finalize_descriptor`] runs. A batch or a merge changing
    /// several links pays for one O(n^2) selection rather than one per link.
    pub(crate) fn add_observation(&mut self, key: ObservationKey, descriptor: D) -> bool {
        if self.is_observed_by(key.keyframe_idx) {
            return false;
        }
        self.observations
            .push(LandmarkObservation { key, descriptor });
        true
    }

    /// Drops the link from `keyframe_idx`, returning the removed record.
    /// Like [`MapPoint::add_observation`], leaves finalization to the caller.
    pub(crate) fn remove_observation(
        &mut self,
        keyframe_idx: usize,
    ) -> Option<LandmarkObservation<D>> {
        let at = self
            .observations
            .iter()
            .position(|o| o.key.keyframe_idx == keyframe_idx)?;
        Some(self.observations.remove(at))
    }

    /// Re-selects the representative descriptor from the current records.
    pub(crate) fn finalize_descriptor(&mut self) {
        self.recompute_representative_descriptor();
    }

    pub fn observations(&self) -> &[LandmarkObservation<D>] {
        &self.observations
    }

    pub fn is_observed_by(&self, keyframe_idx: usize) -> bool {
        self.observations
            .iter()
            .any(|o| o.key.keyframe_idx == keyframe_idx)
    }

    /// Observing keyframe ids, in insertion order.
    pub fn observer_keyframes(&self) -> impl Iterator<Item = usize> + '_ {
        self.observations.iter().map(|o| o.key.keyframe_idx)
    }

    /// Picks the observation descriptor with the minimum median distance to
    /// the others; trivial below 3 observations.
    fn recompute_representative_descriptor(&mut self) {
        let best = match self.observations.len() {
            0 => return,
            1 | 2 => 0,
            _ => most_distinctive(&self.observations, |o| &o.descriptor).unwrap_or(0),
        };
        self.descriptor = self.observations[best].descriptor;
    }

    /// Pyramid level the point should be observable at from `distance`
    /// (ORB-SLAM3's `PredictScale`). Returns 0 when the geometry is unset.
    pub fn predict_scale(&self, distance: f64, pyramid: ScalePyramid) -> usize {
        let ScalePyramid {
            factor: scale_factor,
            n_levels,
        } = pyramid;
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

    pub fn min_distance_invariance(&self) -> f64 {
        0.8 * self.min_distance
    }

    pub fn max_distance_invariance(&self) -> f64 {
        1.2 * self.max_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_point_new_sets_active_defaults() {
        let mp = MapPoint::new(Vec3F64::new(1.0, 2.0, 3.0), [9u8; 32], 0, [0; 3], 5);

        assert_eq!(mp.position, Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(mp.descriptor, [9u8; 32]);
        assert_eq!(mp.keyframe_idx, 5);
        assert_eq!(mp.n_visible, 1);
        assert_eq!(mp.n_found, 1);
        assert!(!mp.culled);
    }

    #[test]
    fn map_point_tracking_helpers_work() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0);
        mp.n_visible = 10;
        mp.n_found = 4;

        assert!((mp.found_ratio() - 0.4).abs() < 1e-9);
        mp.mark_culled();
        assert!(mp.culled);
    }

    /// T1a: `predict_scale` matches ORB-SLAM3's
    /// `nScale = clamp(ceil(log(maxDist/dist) / log(scaleFactor)), 0, nLevels-1)`.
    #[test]
    fn predict_scale_matches_orbslam3_closed_form() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0);
        mp.max_distance = 10.0;
        let pyramid = ScalePyramid::ORB;
        let (sf, n) = (pyramid.factor, pyramid.n_levels);

        for &dist in &[40.0_f64, 12.0, 10.0, 8.0, 3.7, 0.9, 0.01] {
            let want = {
                let ratio = mp.max_distance / dist;
                let lvl = (ratio.ln() / sf.ln()).ceil();
                if lvl.is_nan() || lvl < 0.0 {
                    0
                } else {
                    (lvl as usize).min(n - 1)
                }
            };
            assert_eq!(mp.predict_scale(dist, pyramid), want, "dist={dist}");
        }

        // Degenerate inputs return level 0.
        assert_eq!(mp.predict_scale(0.0, pyramid), 0);
        let mut unset = MapPoint::new(Vec3F64::ZERO, [0u8; 32], 0, [0; 3], 0);
        unset.max_distance = 0.0;
        assert_eq!(unset.predict_scale(5.0, pyramid), 0);
    }

    fn key(kf: usize, feature: usize) -> ObservationKey {
        ObservationKey {
            keyframe_idx: kf,
            feature_idx: feature,
        }
    }

    fn point() -> MapPoint {
        MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0)
    }

    #[test]
    fn a_repeated_keyframe_is_refused_and_leaves_the_records_untouched() {
        let mut mp = point();
        assert!(mp.add_observation(key(0, 0), [1u8; 32]));
        assert!(!mp.add_observation(key(0, 7), [9u8; 32]));
        mp.finalize_descriptor();
        assert_eq!(mp.observations().len(), 1);
        assert_eq!(mp.observations()[0].key, key(0, 0));
        assert!(mp.is_observed_by(0));
    }

    /// Three descriptors where the middle one is closest to both others, so it
    /// is the representative; removing it must hand the role to another.
    #[test]
    fn removing_an_observation_moves_the_representative() {
        let mut mp = point();
        // Four descriptors spaced 4 bits apart along a line: 0, 4, 8, 12.
        // Medians of the pairwise distances are 8, 4, 4, 8 — so the second wins
        // on the tie-break, and removing it hands the role to the third.
        let d0 = [0u8; 32];
        let mut d4 = [0u8; 32];
        d4[0] = 0b0000_1111;
        let mut d8 = [0u8; 32];
        d8[0] = 0b1111_1111;
        let mut d12 = [0u8; 32];
        d12[0] = 0b1111_1111;
        d12[1] = 0b0000_1111;

        mp.add_observation(key(0, 0), d0);
        mp.add_observation(key(1, 0), d4);
        mp.add_observation(key(2, 0), d8);
        mp.add_observation(key(3, 0), d12);
        // Records and finalization are separate now: the representative is
        // stale until asked for, so a batch pays for one selection, not four.
        mp.finalize_descriptor();
        assert_eq!(mp.descriptor, d4);

        let removed = mp.remove_observation(1).expect("link exists");
        mp.finalize_descriptor();
        assert_eq!(removed.descriptor, d4);
        assert_eq!(mp.observations().len(), 3);
        assert!(!mp.is_observed_by(1));
        assert_eq!(mp.descriptor, d8, "representative moved after removal");
    }

    #[test]
    fn removing_an_absent_keyframe_is_none() {
        let mut mp = point();
        mp.add_observation(key(0, 0), [1u8; 32]);
        assert!(mp.remove_observation(42).is_none());
        assert_eq!(mp.observations().len(), 1);
    }

    #[test]
    fn observer_order_follows_insertion() {
        let mut mp = point();
        for kf in [5usize, 2, 9] {
            mp.add_observation(key(kf, 0), [kf as u8; 32]);
        }
        assert_eq!(mp.observer_keyframes().collect::<Vec<_>>(), vec![5, 2, 9]);
    }
}
