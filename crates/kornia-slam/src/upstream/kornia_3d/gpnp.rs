//! Upstream target: kornia_3d::pnp::gpnp. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{Estimator, Sampler};
use kornia_algebra::Vec3F64;

/// A 2D-3D match as a ray of a multi-camera rig; everything in the rig (primary) frame.
#[derive(Debug, Clone, Copy)]
pub struct RayMatch {
    /// Camera centre, metres.
    pub origin: Vec3F64,
    /// Unit.
    pub direction: Vec3F64,
    pub point_world: Vec3F64,
    /// Pixels per unit of tangent-plane error (the camera's fx): residuals come out in px².
    pub pixel_scale: f64,
}

/// World-to-rig poses (up to 4) from three matches that share one origin (bit-equal):
/// AP3P in a frame at that origin, lifted to the rig. Empty when the origins differ.
pub fn ap3p_lifted(_matches: &[RayMatch; 3]) -> Vec<Pose3d> {
    todo!("lane B: lifted AP3P")
}

/// `(pixel_scale · tan∠(p, d))²` with `p = R X + t − origin`; `f64::INFINITY` when `p·d ≤ 0`.
pub fn ray_residual_sq(_pose_world_to_rig: &Pose3d, _m: &RayMatch) -> f64 {
    todo!("lane B: lifted AP3P")
}

/// [`ap3p_lifted`] as a RANSAC estimator.
pub struct LiftedAp3pEstimator;

impl Estimator for LiftedAp3pEstimator {
    type Model = Pose3d;
    type Sample = RayMatch;
    const SAMPLE_SIZE: usize = 3;

    fn fit(&self, _samples: &[RayMatch], _out: &mut Vec<Pose3d>) {
        todo!("lane B: lifted AP3P")
    }

    fn residual(&self, _model: &Pose3d, _sample: &RayMatch) -> f64 {
        todo!("lane B: lifted AP3P")
    }
}

/// Draws only from `[0, prefix)`, with its own splitmix64 state (no `rand` here).
pub struct PrefixSampler {
    state: u64,
    prefix: usize,
}

impl Sampler for PrefixSampler {
    fn sample(&mut self, _n: usize, _out: &mut [usize]) {
        todo!("lane B: prefix sampler")
    }
}

/// RANSAC (`kornia_3d::ransac::run`, `ThresholdConsensus`): hypotheses from
/// `matches[..hypothesis_count]`, scored on all matches. Returns the pose and the inlier mask.
pub fn ransac_lifted_ap3p(
    _matches: &[RayMatch],
    _hypothesis_count: usize,
    _threshold_px: f64,
    _max_iterations: u32,
    _seed: u64,
) -> Option<(Pose3d, Vec<bool>)> {
    todo!("lane B: lifted AP3P")
}
