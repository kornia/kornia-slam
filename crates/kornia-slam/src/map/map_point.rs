//! Map point: a triangulated landmark and its observations.

use kornia_algebra::Vec3F64;
use kornia_imgproc::features::hamming_distance;

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
    /// All descriptors of this point across observing keyframes. Parallel to
    /// `observation_kf_indices`. Used to recompute `descriptor`.
    pub observed_descriptors: Vec<[u8; 32]>,
    /// Frame indices (`Keyframe::frame.idx`) of the keyframes observing this
    /// point, parallel to `observed_descriptors`. Used to recompute the
    /// viewing direction.
    pub observation_kf_indices: Vec<usize>,
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
    pub fn add_observation_descriptor(&mut self, kf_idx: usize, descriptor: [u8; 32]) {
        self.observed_descriptors.push(descriptor);
        self.observation_kf_indices.push(kf_idx);
        self.recompute_representative_descriptor();
    }

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
