//! Persistent 3D landmarks: descriptor aggregation and scale-invariance geometry.
//!
//! A `MapPoint` owns what it can derive from itself. Anything needing the
//! observing keyframes — the viewing direction and distance bounds, which read
//! camera centres — lives on [`Map::update_map_point_geometry`] and only writes
//! the fields back here.
//!
//! [`Map::update_map_point_geometry`]: super::Map::update_map_point_geometry

use kornia_algebra::Vec3F64;
use kornia_imgproc::features::hamming_distance;

#[derive(Debug, Clone)]
pub struct MapPoint {
    /// World-frame position.
    pub position: Vec3F64,
    /// Representative descriptor for projection-guided matching, picked by
    /// `recompute_representative_descriptor`.
    pub descriptor: [u8; 32],
    /// Observed descriptors, parallel to `observation_kf_indices`.
    pub observed_descriptors: Vec<[u8; 32]>,
    /// `Keyframe::frame.idx` of each observer, parallel to `observed_descriptors`.
    pub observation_kf_indices: Vec<usize>,
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

impl MapPoint {
    pub fn new(
        position: Vec3F64,
        descriptor: [u8; 32],
        octave: u8,
        color: [u8; 3],
        keyframe_idx: usize,
    ) -> Self {
        Self {
            position,
            descriptor,
            observed_descriptors: vec![descriptor],
            observation_kf_indices: vec![keyframe_idx],
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

    pub fn add_observation_descriptor(&mut self, kf_idx: usize, descriptor: [u8; 32]) {
        self.observed_descriptors.push(descriptor);
        self.observation_kf_indices.push(kf_idx);
        self.recompute_representative_descriptor();
    }

    /// Picks the descriptor with minimum median Hamming distance to all others
    /// (ORB-SLAM3's `ComputeDistinctiveDescriptors`). Trivial below 3
    /// observations; an O(n^2) pairwise scan above.
    fn recompute_representative_descriptor(&mut self) {
        let n = self.observed_descriptors.len();
        match n {
            0 => {}
            1 => self.descriptor = self.observed_descriptors[0],
            2 => self.descriptor = self.observed_descriptors[0],
            _ => {
                let mut dist_buf = vec![0u32; n];
                let mut best_idx = 0usize;
                let mut best_median = u32::MAX;
                for i in 0..n {
                    for (slot, other) in dist_buf.iter_mut().zip(self.observed_descriptors.iter()) {
                        *slot = hamming_distance(&self.observed_descriptors[i], other);
                    }
                    let mid = n / 2;
                    dist_buf.select_nth_unstable(mid);
                    let median = dist_buf[mid];
                    if median < best_median {
                        best_median = median;
                        best_idx = i;
                    }
                }
                self.descriptor = self.observed_descriptors[best_idx];
            }
        }
    }

    /// Pyramid level the point should be observable at from `distance`
    /// (ORB-SLAM3's `PredictScale`). Returns 0 when the geometry is unset.
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

    pub fn min_distance_invariance(&self) -> f64 {
        0.8 * self.min_distance
    }

    pub fn max_distance_invariance(&self) -> f64 {
        1.2 * self.max_distance
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ORB_N_LEVELS, ORB_SCALE_FACTOR};
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
        let sf = ORB_SCALE_FACTOR;
        let n = ORB_N_LEVELS;

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
            assert_eq!(mp.predict_scale(dist, sf, n), want, "dist={dist}");
        }

        // Degenerate inputs return level 0.
        assert_eq!(mp.predict_scale(0.0, sf, n), 0);
        let mut unset = MapPoint::new(Vec3F64::ZERO, [0u8; 32], 0, [0; 3], 0);
        unset.max_distance = 0.0;
        assert_eq!(unset.predict_scale(5.0, sf, n), 0);
    }
}
