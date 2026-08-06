//! Landmark sampling and synthetic visual observations.
//!
//! Produces [`BaObservation`] values directly, so the output drives
//! [`kornia_3d::ba_schur::bundle_adjust_schur`] and
//! [`crate::vi_ba_schur::visual_inertial_bundle_adjust`] with no adapter layer
//! and nothing mocked.

use kornia_3d::ba::BaObservation;
use kornia_3d::camera::{PinholeCamera, project_point};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use super::SimError;
use super::rng::SimRng;
use super::trajectory::Trajectory;

/// How landmarks are scattered around the trajectory.
#[derive(Debug, Clone)]
pub struct LandmarkConfig {
    /// Number of landmarks to sample.
    pub count: usize,
    /// Inner radius of the sampling shell (metres).
    pub min_range: f64,
    /// Outer radius of the sampling shell (metres).
    pub max_range: f64,
    /// Half-angle (radians) of the cone around the camera's forward axis into
    /// which landmarks are sampled.
    pub cone_half_angle: f64,
}

impl Default for LandmarkConfig {
    fn default() -> Self {
        Self {
            count: 400,
            // A shell, not a box. A uniform box around the trajectory gives an
            // unrealistic depth distribution — most volume ends up at the
            // corners, far from the path — and admits degenerate configurations
            // such as landmarks arbitrarily close to the camera centre.
            min_range: 2.0,
            max_range: 12.0,
            // Slightly wider than a typical 90° horizontal FOV, so some
            // landmarks fall outside the image and exercise the culling path
            // rather than every sample being trivially visible.
            cone_half_angle: 0.9,
        }
    }
}

/// Image bounds, depth gating and pixel noise for observation generation.
#[derive(Debug, Clone)]
pub struct ObservationConfig {
    /// Image width in pixels.
    pub image_width: f64,
    /// Image height in pixels.
    pub image_height: f64,
    /// Observations closer than this are dropped (metres).
    pub min_depth: f64,
    /// Observations farther than this are dropped (metres).
    pub max_depth: f64,
    /// Standard deviation of the Gaussian noise added to each pixel
    /// coordinate. Zero produces exact measurements.
    pub pixel_noise_std: f64,
    /// Optional cap on observations per keyframe — the landmark-density knob.
    /// `None` keeps every visible landmark.
    pub max_per_keyframe: Option<usize>,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            image_width: 752.0,
            image_height: 480.0,
            min_depth: 0.5,
            max_depth: 50.0,
            // 1 px is a realistic ORB keypoint localization error and matches
            // the sigma the estimators' chi-square thresholds assume.
            pixel_noise_std: 1.0,
            max_per_keyframe: None,
        }
    }
}

/// Landmarks plus the observations of them, kept index-consistent.
#[derive(Debug, Clone)]
pub struct VisualData {
    /// Ground-truth landmark positions in the world frame, after pruning.
    pub points: Vec<Vec3F64>,
    /// Observations indexing into [`Self::points`] and the caller's pose slice.
    pub observations: Vec<BaObservation>,
    /// For each entry of [`Self::points`], its index in the pre-pruning
    /// landmark list. Useful when correlating a failure back to the sample.
    pub source_index: Vec<usize>,
}

impl VisualData {
    /// Number of observations of each pose, indexed by pose.
    pub fn observations_per_pose(&self, n_poses: usize) -> Vec<usize> {
        let mut counts = vec![0usize; n_poses];
        for obs in &self.observations {
            if obs.pose_idx < n_poses {
                counts[obs.pose_idx] += 1;
            }
        }
        counts
    }

    /// Marks the given poses as fixed on every observation that references
    /// them.
    ///
    /// `bundle_adjust_schur` takes its gauge anchoring per-observation, so
    /// fixing a pose means setting the flag on all of its observations.
    pub fn fix_poses(&mut self, poses: &[usize]) {
        for obs in &mut self.observations {
            if poses.contains(&obs.pose_idx) {
                obs.fixed_pose = true;
            }
        }
    }
}

/// Samples landmarks in a shell around the trajectory.
///
/// Each landmark is placed relative to a randomly chosen point along the path,
/// inside a cone around that pose's forward axis, at a uniformly sampled range.
/// Anchoring to the path rather than to a single global volume keeps the
/// landmark density roughly constant along a long trajectory.
pub fn sample_landmarks(
    trajectory: &Trajectory,
    config: &LandmarkConfig,
    rng: &mut SimRng,
) -> Result<Vec<Vec3F64>, SimError> {
    if config.min_range <= 0.0 || config.max_range <= config.min_range {
        return Err(SimError::InvalidConfig(format!(
            "landmark shell must satisfy 0 < min_range < max_range, got {} .. {}",
            config.min_range, config.max_range
        )));
    }

    // Inset from the domain edges so anchors stay comfortably inside.
    let span = trajectory.t_end() - trajectory.t_start();
    let (lo, hi) = (
        trajectory.t_start() + 0.02 * span,
        trajectory.t_end() - 0.02 * span,
    );

    let mut points = Vec::with_capacity(config.count);
    for _ in 0..config.count {
        let state = trajectory.state(rng.uniform_range(lo, hi))?;

        // Direction inside a cone about the body forward (+z) axis. Sampling
        // cos(theta) uniformly gives an equal-area cap rather than one that
        // bunches toward the axis.
        let cos_min = config.cone_half_angle.cos();
        let cos_theta = rng.uniform_range(cos_min, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let azimuth = rng.uniform_range(0.0, std::f64::consts::TAU);
        let local = Vec3F64::new(
            sin_theta * azimuth.cos(),
            sin_theta * azimuth.sin(),
            cos_theta,
        );

        let range = rng.uniform_range(config.min_range, config.max_range);
        points.push(state.position + state.rotation * local * range);
    }
    Ok(points)
}

/// Projects landmarks into every pose, culling and adding pixel noise.
///
/// Landmarks seen by fewer than `min_observations` poses are dropped and the
/// remaining ones re-indexed. This is not tidying: a point with a single
/// observation is rank-deficient in the reduced camera system, and leaving such
/// points in makes the Schur complement singular for reasons that have nothing
/// to do with the behaviour under test.
pub fn generate_observations(
    poses: &[Pose3d],
    points: &[Vec3F64],
    camera: &PinholeCamera,
    config: &ObservationConfig,
    min_observations: usize,
    rng: &mut SimRng,
) -> Result<VisualData, SimError> {
    if min_observations < 2 {
        return Err(SimError::InvalidConfig(format!(
            "min_observations must be at least 2 to constrain a point, got {min_observations}"
        )));
    }

    // Pass 1: every visible (pose, point) pair.
    let mut raw: Vec<Vec<BaObservation>> = vec![Vec::new(); poses.len()];
    for (pose_idx, pose) in poses.iter().enumerate() {
        for (point_idx, point) in points.iter().enumerate() {
            let Some((u, v, depth)) = project_point(camera, pose, point) else {
                continue; // behind the camera
            };
            if depth < config.min_depth || depth > config.max_depth {
                continue;
            }
            let (nu, nv) = if config.pixel_noise_std > 0.0 {
                (
                    u + rng.normal_scaled(config.pixel_noise_std),
                    v + rng.normal_scaled(config.pixel_noise_std),
                )
            } else {
                (u, v)
            };
            // Bounds-check after noise: a measurement displaced outside the
            // image would not have been detected in the first place.
            if nu < 0.0 || nu >= config.image_width || nv < 0.0 || nv >= config.image_height {
                continue;
            }
            raw[pose_idx].push(BaObservation {
                pose_idx,
                point_idx,
                pixel: [nu as f32, nv as f32],
                fixed_pose: false,
                fixed_point: false,
                depth_meas: None,
                depth_sigma: 0.0,
            });
        }
    }

    // Pass 2: apply the per-keyframe density cap by random subsampling.
    if let Some(cap) = config.max_per_keyframe {
        for per_pose in raw.iter_mut() {
            if per_pose.len() > cap {
                // Partial Fisher-Yates: draw `cap` distinct entries to the
                // front, then truncate. Deterministic given the seed.
                for i in 0..cap {
                    let j = i + (rng.next_u64() as usize) % (per_pose.len() - i);
                    per_pose.swap(i, j);
                }
                per_pose.truncate(cap);
            }
        }
    }

    // Pass 3: drop under-observed points and re-index.
    let mut counts = vec![0usize; points.len()];
    for per_pose in &raw {
        for obs in per_pose {
            counts[obs.point_idx] += 1;
        }
    }

    let mut remap = vec![usize::MAX; points.len()];
    let mut kept_points = Vec::new();
    let mut source_index = Vec::new();
    for (idx, &count) in counts.iter().enumerate() {
        if count >= min_observations {
            remap[idx] = kept_points.len();
            kept_points.push(points[idx]);
            source_index.push(idx);
        }
    }

    let mut observations = Vec::new();
    for per_pose in raw {
        for mut obs in per_pose {
            let new_idx = remap[obs.point_idx];
            if new_idx != usize::MAX {
                obs.point_idx = new_idx;
                observations.push(obs);
            }
        }
    }

    if kept_points.is_empty() {
        return Err(SimError::NoVisibleLandmarks);
    }

    Ok(VisualData {
        points: kept_points,
        observations,
        source_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::trajectory::ArcConfig;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 376.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn setup() -> (Trajectory, Vec<Pose3d>, Vec<Vec3F64>) {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 4.0,
            climb: 0.5,
            n_control: 20,
            t_start: 0.0,
            knot_dt: 0.25,
            ..Default::default()
        })
        .unwrap();
        let mut rng = SimRng::new(1);
        let points = sample_landmarks(&traj, &LandmarkConfig::default(), &mut rng).unwrap();
        let poses = traj
            .sample_uniform(10)
            .unwrap()
            .iter()
            .map(|s| s.camera_pose(None))
            .collect();
        (traj, poses, points)
    }

    #[test]
    fn landmarks_land_inside_the_configured_shell() {
        let traj = Trajectory::arc(&ArcConfig {
            radius: 4.0,
            climb: 0.0,
            n_control: 20,
            t_start: 0.0,
            knot_dt: 0.25,
            ..Default::default()
        })
        .unwrap();
        let cfg = LandmarkConfig {
            count: 500,
            min_range: 3.0,
            max_range: 9.0,
            cone_half_angle: 0.7,
        };
        let mut rng = SimRng::new(5);
        let points = sample_landmarks(&traj, &cfg, &mut rng).unwrap();
        assert_eq!(points.len(), 500);

        // Every landmark must be within max_range of *some* point on the path,
        // which is what "shell around the trajectory" means.
        let path: Vec<Vec3F64> = traj
            .sample_uniform(200)
            .unwrap()
            .iter()
            .map(|s| s.position)
            .collect();
        for p in &points {
            let nearest = path
                .iter()
                .map(|q| (*p - *q).length())
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest <= cfg.max_range + 1e-6,
                "landmark {nearest} m from path, beyond max_range {}",
                cfg.max_range
            );
        }
    }

    #[test]
    fn noiseless_observations_reproject_exactly() {
        let (_, poses, points) = setup();
        let camera = test_camera();
        let cfg = ObservationConfig {
            pixel_noise_std: 0.0,
            ..Default::default()
        };
        let mut rng = SimRng::new(2);
        let data = generate_observations(&poses, &points, &camera, &cfg, 2, &mut rng).unwrap();

        assert!(!data.observations.is_empty());
        for obs in &data.observations {
            let (u, v, _) =
                project_point(&camera, &poses[obs.pose_idx], &data.points[obs.point_idx]).unwrap();
            // f32 pixel storage is the only loss here.
            assert!((obs.pixel[0] as f64 - u).abs() < 1e-3);
            assert!((obs.pixel[1] as f64 - v).abs() < 1e-3);
        }
    }

    #[test]
    fn observations_respect_image_bounds_and_depth_gate() {
        let (_, poses, points) = setup();
        let camera = test_camera();
        let cfg = ObservationConfig {
            min_depth: 3.0,
            max_depth: 8.0,
            pixel_noise_std: 0.0,
            ..Default::default()
        };
        let mut rng = SimRng::new(3);
        let data = generate_observations(&poses, &points, &camera, &cfg, 2, &mut rng).unwrap();

        for obs in &data.observations {
            let (u, v, depth) =
                project_point(&camera, &poses[obs.pose_idx], &data.points[obs.point_idx]).unwrap();
            assert!(
                (3.0..=8.0).contains(&depth),
                "depth {depth} outside the configured gate"
            );
            assert!((0.0..cfg.image_width).contains(&u));
            assert!((0.0..cfg.image_height).contains(&v));
        }
    }

    #[test]
    fn under_observed_points_are_pruned_and_reindexed() {
        let (_, poses, points) = setup();
        let camera = test_camera();
        let mut rng = SimRng::new(4);
        let data = generate_observations(
            &poses,
            &points,
            &camera,
            &ObservationConfig::default(),
            3,
            &mut rng,
        )
        .unwrap();

        let mut counts = vec![0usize; data.points.len()];
        for obs in &data.observations {
            assert!(obs.point_idx < data.points.len(), "stale point index");
            counts[obs.point_idx] += 1;
        }
        for (idx, &c) in counts.iter().enumerate() {
            assert!(c >= 3, "point {idx} kept with only {c} observations");
        }
        assert_eq!(data.source_index.len(), data.points.len());
    }

    #[test]
    fn density_cap_is_respected() {
        let (_, poses, points) = setup();
        let camera = test_camera();
        let cfg = ObservationConfig {
            max_per_keyframe: Some(20),
            pixel_noise_std: 0.0,
            ..Default::default()
        };
        let mut rng = SimRng::new(6);
        let data = generate_observations(&poses, &points, &camera, &cfg, 2, &mut rng).unwrap();
        for (pose_idx, count) in data.observations_per_pose(poses.len()).iter().enumerate() {
            assert!(*count <= 20, "pose {pose_idx} kept {count} observations");
        }
    }

    #[test]
    fn generation_is_deterministic_for_a_given_seed() {
        let (_, poses, points) = setup();
        let camera = test_camera();
        let cfg = ObservationConfig::default();
        let a =
            generate_observations(&poses, &points, &camera, &cfg, 2, &mut SimRng::new(9)).unwrap();
        let b =
            generate_observations(&poses, &points, &camera, &cfg, 2, &mut SimRng::new(9)).unwrap();

        assert_eq!(a.observations.len(), b.observations.len());
        for (x, y) in a.observations.iter().zip(b.observations.iter()) {
            assert_eq!(x.pose_idx, y.pose_idx);
            assert_eq!(x.point_idx, y.point_idx);
            assert_eq!(x.pixel, y.pixel);
        }
    }
}
