//! Assembles a trajectory, landmarks and sensor measurements into the exact
//! input types the estimators take.
//!
//! A [`Scene`] is the bridge between "here is a trajectory I chose" and the six
//! arguments of [`visual_inertial_bundle_adjust`]. Everything it hands back is
//! ground truth, which is what makes a failure attributable: the estimate is
//! compared against a quantity that was *defined*, not measured.
//!
//! [`visual_inertial_bundle_adjust`]: crate::vi_ba_schur::visual_inertial_bundle_adjust

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, ImuMeasurement};

use super::SimError;
use super::imu::{ImuSimConfig, build_imu_factors, generate_imu};
use super::landmarks::{
    LandmarkConfig, ObservationConfig, VisualData, generate_observations, sample_landmarks,
};
use super::rng::SimRng;
use super::trajectory::{DEFAULT_GRAVITY, Trajectory, TrajectoryState};
use crate::vi_ba_schur::{ImuFactor, ViBaKeyframe};

/// Everything needed to build a [`Scene`] from a [`Trajectory`].
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// Number of keyframes sampled along the trajectory.
    pub n_keyframes: usize,
    /// Camera intrinsics used for projection.
    pub camera: PinholeCamera,
    /// Landmark scattering.
    pub landmarks: LandmarkConfig,
    /// Projection, culling and pixel noise.
    pub observations: ObservationConfig,
    /// Minimum number of keyframes that must see a landmark for it to be kept.
    pub min_observations: usize,
    /// IMU generation. `None` builds a visual-only scene.
    pub imu: Option<ImuSimConfig>,
    /// Gravity in the world frame.
    pub gravity: Vec3F64,
    /// Camera-to-body extrinsic (`X_body = T_bc · X_cam`). `None` means the
    /// camera and body frames coincide.
    pub t_bc: Option<Pose3d>,
    /// The seed. Recorded on the [`Scene`] so a failing run is reproducible
    /// from its output alone.
    pub seed: u64,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            n_keyframes: 12,
            // EuRoC-like intrinsics, so the pixel scale relates to the datasets
            // the estimators are actually tuned against.
            camera: PinholeCamera {
                fx: 458.654,
                fy: 457.296,
                cx: 367.215,
                cy: 248.375,
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
            landmarks: LandmarkConfig::default(),
            observations: ObservationConfig::default(),
            min_observations: 3,
            imu: None,
            gravity: DEFAULT_GRAVITY,
            t_bc: None,
            seed: 0,
        }
    }
}

/// A fully generated scene: ground truth plus the measurements derived from it.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Ground-truth kinematic state at each keyframe.
    pub states: Vec<TrajectoryState>,
    /// Ground-truth world→camera poses at each keyframe.
    pub camera_poses: Vec<Pose3d>,
    /// Landmarks and their observations.
    pub visual: VisualData,
    /// Raw IMU stream, empty when the scene is visual-only.
    pub imu_measurements: Vec<ImuMeasurement>,
    /// The IMU settings the stream was generated with. `None` for a
    /// visual-only scene. Carried on the scene so [`Scene::imu_factors`] uses
    /// the same noise model that generated the data rather than assuming a
    /// default.
    pub imu_config: Option<ImuSimConfig>,
    /// Camera intrinsics used.
    pub camera: PinholeCamera,
    /// Gravity used.
    pub gravity: Vec3F64,
    /// Camera-to-body extrinsic used.
    pub t_bc: Option<Pose3d>,
    /// The seed this scene was generated from.
    pub seed: u64,
}

impl Scene {
    /// Generates a scene from a trajectory.
    pub fn build(trajectory: &Trajectory, config: &SceneConfig) -> Result<Self, SimError> {
        let mut rng = SimRng::new(config.seed);

        let states = trajectory.sample_uniform(config.n_keyframes)?;
        let camera_poses: Vec<Pose3d> = states
            .iter()
            .map(|s| s.camera_pose(config.t_bc.as_ref()))
            .collect();

        let points = sample_landmarks(trajectory, &config.landmarks, &mut rng)?;
        let visual = generate_observations(
            &camera_poses,
            &points,
            &config.camera,
            &config.observations,
            config.min_observations,
            &mut rng,
        )?;

        let imu_measurements = match &config.imu {
            Some(imu_config) => generate_imu(trajectory, config.gravity, imu_config, &mut rng)?,
            None => Vec::new(),
        };

        Ok(Self {
            states,
            camera_poses,
            visual,
            imu_measurements,
            imu_config: config.imu.clone(),
            camera: config.camera.clone(),
            gravity: config.gravity,
            t_bc: config.t_bc,
            seed: config.seed,
        })
    }

    /// Keyframe timestamps.
    pub fn keyframe_times(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.timestamp).collect()
    }

    /// Ground-truth [`ViBaKeyframe`] values: exact pose, exact world velocity
    /// and the true bias.
    ///
    /// This is the target a recovery test measures against, and also the
    /// starting point a perturbation is applied to.
    pub fn ground_truth_keyframes(&self) -> Vec<ViBaKeyframe> {
        self.states
            .iter()
            .zip(self.camera_poses.iter())
            .map(|(state, pose)| ViBaKeyframe {
                pose: *pose,
                velocity: state.velocity,
                bias: self.true_bias(),
                fixed: false,
            })
            .collect()
    }

    /// The true bias baked into [`Self::imu_measurements`]. Zero for a
    /// visual-only scene.
    pub fn true_bias(&self) -> ImuBias {
        self.imu_config.as_ref().map(|c| c.bias).unwrap_or_default()
    }

    /// Builds IMU factors between consecutive keyframes, linearized at
    /// `bias_estimate`.
    pub fn imu_factors(&self, bias_estimate: ImuBias) -> Result<Vec<ImuFactor>, SimError> {
        let Some(config) = self.imu_config.as_ref() else {
            return Err(SimError::InvalidConfig(
                "scene has no IMU measurements".to_string(),
            ));
        };
        build_imu_factors(
            &self.imu_measurements,
            &self.keyframe_times(),
            bias_estimate,
            config.calib,
        )
    }
}

/// Applies a known, seeded perturbation to a set of ground-truth keyframes.
///
/// This is the input side of a recovery test: displace the estimate by a known
/// amount, then check the optimizer walks it back. The perturbation is
/// deliberately applied in the tangent space so a rotation stays a rotation.
///
/// `fixed` keyframes are left untouched — they are the gauge anchor, and
/// perturbing them would move the frame the answer is expressed in rather than
/// creating an error to correct.
#[derive(Debug, Clone, Default)]
pub struct Perturbation {
    /// Standard deviation of the translation offset (metres).
    pub translation_std: f64,
    /// Standard deviation of the rotation offset (radians).
    pub rotation_std: f64,
    /// Standard deviation of the velocity offset (m/s).
    pub velocity_std: f64,
    /// Standard deviation of the landmark position offset (metres).
    pub point_std: f64,
    /// Bias the estimate is *started from*, overriding ground truth. `None`
    /// keeps the true bias.
    pub initial_bias: Option<ImuBias>,
}

impl Perturbation {
    /// Perturbs poses in place. Rotation is perturbed on the manifold via the
    /// exponential map, so the result is always a valid rotation.
    pub fn apply_to_poses(&self, poses: &mut [Pose3d], fixed: &[bool], rng: &mut SimRng) {
        use kornia_algebra::SO3F64;
        for (pose, is_fixed) in poses.iter_mut().zip(fixed.iter()) {
            if *is_fixed {
                continue;
            }
            if self.rotation_std > 0.0 {
                // Right-multiplied tangent perturbation: the result is a
                // product of rotations, never a matrix with noise added to its
                // entries, so it stays exactly on SO(3).
                let delta = SO3F64::exp(rng.normal_vec3(self.rotation_std));
                let r = SO3F64::from_matrix(&pose.rotation) * delta;
                pose.rotation = r.matrix();
            }
            if self.translation_std > 0.0 {
                pose.translation += rng.normal_vec3(self.translation_std);
            }
        }
    }

    /// Perturbs landmark positions in place.
    pub fn apply_to_points(&self, points: &mut [Vec3F64], rng: &mut SimRng) {
        if self.point_std <= 0.0 {
            return;
        }
        for p in points.iter_mut() {
            *p += rng.normal_vec3(self.point_std);
        }
    }

    /// Perturbs full VI-BA keyframes: pose, velocity and bias.
    pub fn apply_to_keyframes(&self, keyframes: &mut [ViBaKeyframe], rng: &mut SimRng) {
        let fixed: Vec<bool> = keyframes.iter().map(|k| k.fixed).collect();
        let mut poses: Vec<Pose3d> = keyframes.iter().map(|k| k.pose).collect();
        self.apply_to_poses(&mut poses, &fixed, rng);

        for (kf, pose) in keyframes.iter_mut().zip(poses) {
            kf.pose = pose;
            if kf.fixed {
                continue;
            }
            if self.velocity_std > 0.0 {
                kf.velocity += rng.normal_vec3(self.velocity_std);
            }
            if let Some(bias) = self.initial_bias {
                kf.bias = bias;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::trajectory::ArcConfig;

    fn traj() -> Trajectory {
        Trajectory::arc(&ArcConfig {
            radius: 4.0,
            climb: 0.5,
            n_control: 20,
            t_start: 0.0,
            knot_dt: 0.25,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn visual_only_scene_has_no_imu() {
        let scene = Scene::build(&traj(), &SceneConfig::default()).unwrap();
        assert!(scene.imu_measurements.is_empty());
        assert_eq!(scene.camera_poses.len(), 12);
        assert!(!scene.visual.observations.is_empty());
        assert!(scene.imu_factors(ImuBias::default()).is_err());
    }

    #[test]
    fn inertial_scene_produces_one_factor_per_gap() {
        let config = SceneConfig {
            n_keyframes: 8,
            imu: Some(ImuSimConfig::default()),
            ..Default::default()
        };
        let scene = Scene::build(&traj(), &config).unwrap();
        assert!(!scene.imu_measurements.is_empty());
        let factors = scene.imu_factors(ImuBias::default()).unwrap();
        assert_eq!(factors.len(), 7);
    }

    #[test]
    fn ground_truth_keyframes_match_the_scene_poses() {
        let scene = Scene::build(&traj(), &SceneConfig::default()).unwrap();
        let kfs = scene.ground_truth_keyframes();
        for (kf, pose) in kfs.iter().zip(scene.camera_poses.iter()) {
            assert!((kf.pose.translation - pose.translation).length() < 1e-15);
        }
    }

    #[test]
    fn perturbation_keeps_rotations_valid_and_spares_fixed_frames() {
        let scene = Scene::build(&traj(), &SceneConfig::default()).unwrap();
        let mut kfs = scene.ground_truth_keyframes();
        kfs[0].fixed = true;
        let original_first = kfs[0].pose;

        let perturbation = Perturbation {
            translation_std: 0.1,
            rotation_std: 0.02,
            velocity_std: 0.1,
            point_std: 0.05,
            initial_bias: None,
        };
        let mut rng = SimRng::new(99);
        perturbation.apply_to_keyframes(&mut kfs, &mut rng);

        assert!((kfs[0].pose.translation - original_first.translation).length() < 1e-15);
        for kf in &kfs {
            let det = kf.pose.rotation.determinant();
            assert!((det - 1.0).abs() < 1e-9, "perturbed rotation det {det}");
        }
        assert!((kfs[3].pose.translation - scene.camera_poses[3].translation).length() > 1e-6);
    }

    #[test]
    fn scene_generation_is_reproducible() {
        let config = SceneConfig {
            seed: 1234,
            imu: Some(ImuSimConfig {
                add_noise: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let a = Scene::build(&traj(), &config).unwrap();
        let b = Scene::build(&traj(), &config).unwrap();

        assert_eq!(a.visual.observations.len(), b.visual.observations.len());
        assert_eq!(a.imu_measurements.len(), b.imu_measurements.len());
        for (x, y) in a.imu_measurements.iter().zip(b.imu_measurements.iter()) {
            assert_eq!(x.gyro.x, y.gyro.x);
            assert_eq!(x.accel.z, y.accel.z);
        }
    }
}
