//! Upstream target: kornia_3d::pnp::gpnp. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pnp::ap3p::solve_ap3p_multi;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{Estimator, RansacConfig, Sampler, ThresholdConsensus, run};
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};

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

/// Shortest sum of three unit bearings, or RMS radius of three world points,
/// that still defines a frame to solve in.
const MIN_SPREAD: f64 = 1e-9;

/// World-to-rig poses (up to 4) from three matches that share one origin (bit-equal):
/// AP3P in a frame at that origin, lifted to the rig. Empty when the origins differ.
pub fn ap3p_lifted(matches: &[RayMatch; 3]) -> Vec<Pose3d> {
    let origin = matches[0].origin;
    if matches[1..].iter().any(|m| m.origin != origin) {
        return Vec::new();
    }
    let mean_bearing = matches
        .iter()
        .fold(Vec3F64::ZERO, |acc, m| acc + m.direction);
    if !spread_enough(mean_bearing.length()) {
        return Vec::new();
    }
    // A virtual pinhole at the origin looking along the mean bearing keeps the
    // three bearings in front of it.
    let to_virtual = rotation_taking_to_z(mean_bearing);
    let mut image = [Vec2F32::ZERO; 3];
    for (pixel, m) in image.iter_mut().zip(matches) {
        let b = to_virtual * m.direction;
        if b.z <= 0.0 {
            return Vec::new();
        }
        *pixel = Vec2F32::new((b.x / b.z) as f32, (b.y / b.z) as f32);
    }
    // kornia's AP3P is f32: centre and scale the points so its rounding is relative.
    let centre = matches
        .iter()
        .fold(Vec3F64::ZERO, |acc, m| acc + m.point_world)
        / 3.0;
    let scale = (matches
        .iter()
        .map(|m| (m.point_world - centre).dot(m.point_world - centre))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    if !spread_enough(scale) {
        return Vec::new();
    }
    let world = (*matches).map(|m| {
        let p = (m.point_world - centre) / scale;
        Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32)
    });
    let Ok(solutions) = solve_ap3p_multi(&world, &image, &Mat3AF32::IDENTITY) else {
        return Vec::new();
    };
    // X_virtual = R (X - c) + s t', and X_rig = R_sᵀ X_virtual + origin.
    let from_virtual = to_virtual.transpose();
    solutions
        .iter()
        .map(|solution| {
            let r = Mat3F64::from(solution.rotation.as_dmat3());
            let t = solution.translation;
            let t = Vec3F64::new(f64::from(t.x), f64::from(t.y), f64::from(t.z));
            Pose3d::new(
                from_virtual * r,
                from_virtual * (t * scale - r * centre) + origin,
            )
        })
        .collect()
}

fn spread_enough(spread: f64) -> bool {
    spread.is_finite() && spread > MIN_SPREAD
}

/// A rotation `R` with `R · axis / |axis| = +z`.
fn rotation_taking_to_z(axis: Vec3F64) -> Mat3F64 {
    let z = axis.normalize();
    let helper = if z.x.abs() < 0.9 {
        Vec3F64::new(1.0, 0.0, 0.0)
    } else {
        Vec3F64::new(0.0, 1.0, 0.0)
    };
    let x = (helper - z * helper.dot(z)).normalize();
    let y = z.cross(x);
    Mat3F64::from_cols(x, y, z).transpose()
}

/// `(pixel_scale · tan∠(p, d))²` with `p = R X + t − origin`; `f64::INFINITY` when `p·d ≤ 0`.
pub fn ray_residual_sq(pose_world_to_rig: &Pose3d, m: &RayMatch) -> f64 {
    let p = pose_world_to_rig.transform_point(&m.point_world) - m.origin;
    let along = p.dot(m.direction);
    if along <= 0.0 {
        return f64::INFINITY;
    }
    let tangent = p.cross(m.direction).length() / along;
    (m.pixel_scale * tangent).powi(2)
}

/// [`ap3p_lifted`] as a RANSAC estimator.
pub struct LiftedAp3pEstimator;

impl Estimator for LiftedAp3pEstimator {
    type Model = Pose3d;
    type Sample = RayMatch;
    const SAMPLE_SIZE: usize = 3;

    fn fit(&self, samples: &[RayMatch], out: &mut Vec<Pose3d>) {
        if let Ok(triple) = <&[RayMatch; 3]>::try_from(samples) {
            out.extend(ap3p_lifted(triple));
        }
    }

    fn residual(&self, model: &Pose3d, sample: &RayMatch) -> f64 {
        ray_residual_sq(model, sample)
    }
}

/// Draws only from `[0, prefix)`, with its own splitmix64 state (no `rand` here).
pub struct PrefixSampler {
    state: u64,
    prefix: usize,
}

impl PrefixSampler {
    /// A sampler over `[0, prefix)` seeded with `seed`.
    pub fn new(seed: u64, prefix: usize) -> Self {
        Self {
            state: seed,
            prefix,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, bound)` (Lemire's multiply-shift; the bias is below 2⁻⁵⁰).
    fn below(&mut self, bound: usize) -> usize {
        ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize
    }
}

impl Sampler for PrefixSampler {
    /// Floyd's combination algorithm over `[0, min(prefix, n))`.
    fn sample(&mut self, n: usize, out: &mut [usize]) {
        let population = self.prefix.min(n);
        let k = out.len();
        debug_assert!(
            k <= population,
            "sample size {k} exceeds prefix {population}"
        );
        for (filled, j) in (population.saturating_sub(k)..population).enumerate() {
            let t = self.below(j + 1);
            out[filled] = if out[..filled].contains(&t) { j } else { t };
        }
    }
}

/// RANSAC (`kornia_3d::ransac::run`, `ThresholdConsensus`): hypotheses from
/// `matches[..hypothesis_count]`, scored on all matches. Returns the pose and the inlier mask.
pub fn ransac_lifted_ap3p(
    matches: &[RayMatch],
    hypothesis_count: usize,
    threshold_px: f64,
    max_iterations: u32,
    seed: u64,
) -> Option<(Pose3d, Vec<bool>)> {
    let prefix = hypothesis_count.min(matches.len());
    if prefix < LiftedAp3pEstimator::SAMPLE_SIZE {
        return None;
    }
    let threshold_sq = threshold_px * threshold_px;
    let config = RansacConfig {
        max_iters: max_iterations,
        inlier_threshold: threshold_sq,
        ..RansacConfig::default()
    };
    let result = run(
        &LiftedAp3pEstimator,
        &ThresholdConsensus {
            threshold: threshold_sq,
        },
        &mut PrefixSampler::new(seed, prefix),
        matches,
        &config,
    );
    let pose = result.model?;
    Some((pose, result.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    const FX: f64 = 240.0;

    fn yaw(deg: f64) -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix()
    }

    /// `T_PC` of the 70 deg, 11 cm rig camera of `rig_pnp`'s tests.
    fn side_camera() -> Pose3d {
        Pose3d::new(yaw(70.0), Vec3F64::new(0.11, 0.0, 0.0))
    }

    fn truth() -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.05, -0.3, 0.1)).matrix(),
            Vec3F64::new(0.4, -0.2, 1.3),
        )
    }

    /// The match of a point at `p_camera` in the camera with `T_PC`.
    fn ray_match(primary_from_camera: &Pose3d, p_camera: Vec3F64) -> RayMatch {
        let p_rig = primary_from_camera.transform_point(&p_camera);
        RayMatch {
            origin: primary_from_camera.translation,
            direction: primary_from_camera.rotation * p_camera.normalize(),
            point_world: truth().inverse().transform_point(&p_rig),
            pixel_scale: FX,
        }
    }

    fn rotation_error(a: &Pose3d, b: &Pose3d) -> f64 {
        SO3F64::from_matrix(&(a.rotation * b.rotation.transpose()))
            .log()
            .length()
    }

    #[test]
    fn three_side_camera_rays_give_the_rig_pose() {
        let camera = side_camera();
        let matches = [
            ray_match(&camera, Vec3F64::new(-0.8, 0.3, 4.0)),
            ray_match(&camera, Vec3F64::new(0.6, -0.5, 5.5)),
            ray_match(&camera, Vec3F64::new(0.2, 0.7, 3.2)),
        ];
        let poses = ap3p_lifted(&matches);
        assert!(!poses.is_empty() && poses.len() <= 4);
        let truth = truth();
        let found = poses.iter().any(|pose| {
            rotation_error(pose, &truth) < 2e-5
                && (pose.translation - truth.translation).length() < 1e-4
        });
        assert!(found, "{poses:?}");
        for m in &matches {
            let best = poses
                .iter()
                .map(|pose| ray_residual_sq(pose, m))
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-4, "residual {best} px²");
        }
    }

    #[test]
    fn rays_from_two_origins_are_not_lifted() {
        let matches = [
            ray_match(&side_camera(), Vec3F64::new(-0.8, 0.3, 4.0)),
            ray_match(&side_camera(), Vec3F64::new(0.6, -0.5, 5.5)),
            ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.2, 0.7, 3.2)),
        ];
        assert!(ap3p_lifted(&matches).is_empty());
    }

    #[test]
    fn a_point_behind_the_ray_has_no_residual() {
        let m = ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.1, 0.2, 3.0));
        assert!(ray_residual_sq(&truth(), &m) < 1e-18);
        let behind = RayMatch {
            direction: -m.direction,
            ..m
        };
        assert_eq!(ray_residual_sq(&truth(), &behind), f64::INFINITY);
    }

    #[test]
    fn the_sampler_draws_distinct_indices_from_its_prefix() {
        let mut sampler = PrefixSampler::new(3, 7);
        let mut out = [0usize; 3];
        for _ in 0..1000 {
            sampler.sample(100, &mut out);
            assert!(out.iter().all(|&i| i < 7), "{out:?}");
            assert!(out[0] != out[1] && out[0] != out[2] && out[1] != out[2]);
        }
    }

    #[test]
    fn ransac_finds_the_pose_from_one_camera_and_scores_both() {
        let side = side_camera();
        let mut matches = Vec::new();
        for i in 0..40 {
            let (x, y) = ((i % 8) as f64 * 0.4 - 1.4, (i / 8) as f64 * 0.3 - 0.6);
            let mut m = ray_match(&side, Vec3F64::new(x, y, 4.0 + (i % 3) as f64));
            if i % 3 == 0 {
                m.point_world = m.point_world + Vec3F64::new(0.7, -0.4, 0.5);
            }
            matches.push(m);
        }
        for i in 0..30 {
            let (x, y) = ((i % 6) as f64 * 0.5 - 1.2, (i / 6) as f64 * 0.4 - 0.8);
            matches.push(ray_match(
                &Pose3d::IDENTITY,
                Vec3F64::new(x, y, 6.0 + (i % 2) as f64),
            ));
        }
        let (pose, inliers) = ransac_lifted_ap3p(&matches, 40, 4.0, 300, 0).unwrap();
        let truth = truth();
        assert!(rotation_error(&pose, &truth) < 1e-4);
        assert!((pose.translation - truth.translation).length() < 1e-3);
        assert_eq!(inliers.iter().filter(|&&b| b).count(), 26 + 30);
        assert!(inliers[40..].iter().all(|&b| b));
    }

    #[test]
    fn too_few_hypothesis_matches_give_nothing() {
        let m = ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.1, 0.2, 3.0));
        assert!(ransac_lifted_ap3p(&[m; 10], 2, 4.0, 300, 0).is_none());
    }
}
