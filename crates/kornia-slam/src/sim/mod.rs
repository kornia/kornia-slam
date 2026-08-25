//! A measurement-level simulator for the estimators.
//!
//! Ground-truth spline trajectory → analytically exact IMU and landmark
//! observations → optional noise → straight into the optimizers. No images, no
//! renderer, no scene geometry.
//!
//! # Why measurement-level rather than photorealistic
//!
//! The instinct behind "simulator" is usually a renderer. For validating an
//! *estimator* that is the wrong tool: generating pixels means generating a
//! frontend problem, and it makes the ground truth only as good as the
//! renderer's geometric consistency. Generating measurements instead —
//! pixel coordinates of known landmarks, IMU readings from a differentiable
//! trajectory — has exact ground truth by construction and isolates the
//! component where the expensive bugs in this project have actually been.
//!
//! Simulating the frontend is a legitimate separate problem. It is not this
//! one.
//!
//! # What this is for, and what it is not
//!
//! **Nothing here is mocked.** The simulator produces inputs; the code under
//! test is the production optimizer ([`crate::vi_ba_schur`],
//! [`kornia_3d::ba_schur`]) and the production preintegrator
//! ([`kornia_sensors::imu::PreintegratedImu`]). The usual synthetic-test
//! failure mode — testing a mock of the thing rather than the thing — does not
//! apply.
//!
//! **This is not the acceptance gate.** Real sequences stay the gate. This is
//! the instrument to reach for *after* a real sequence says something is wrong
//! and the question is *which component*. On real data there is no way to ask
//! "what if the bias were exactly 0.05 m/s² and nothing else were wrong?"; here
//! that is the only kind of question there is.
//!
//! # Layout
//!
//! - [`rng`] — seeded, dependency-free deterministic random source.
//! - [`spline`] — the split R³ + SO(3) B-spline and its analytic derivatives.
//! - [`trajectory`] — ground-truth kinematic state and frame conventions.
//! - [`landmarks`] — landmark sampling and [`kornia_3d::ba::BaObservation`]
//!   generation.
//! - [`imu`] — raw [`kornia_sensors::imu::ImuMeasurement`] generation and
//!   preintegration into factors.
//! - [`scene`] — assembles the above into estimator inputs, plus the
//!   perturbation used by recovery tests.
//!
//! # Upstreaming
//!
//! Parts of this module are general enough to belong in kornia-rs rather than
//! here, and are expected to move once this has settled. Nothing below is
//! required for kornia-slam to work — the case is reuse, plus deleting
//! duplication that already exists upstream. Each module carries its own note
//! with the details; this is the index.
//!
//! | Piece | Destination | Why |
//! |---|---|---|
//! | [`landmarks`] + the pose helpers from [`trajectory`] | `kornia-3d`, beside `ba.rs` | Needs only symbols already there. Deletes a triplicated hand-rolled `Lcg` in `ba_schur.rs`, widens a 4-point/2-pose recovery test, gives `pgo.rs` its first tests |
//! | [`spline`] + the algebra-only core of [`trajectory`] | `kornia-algebra` (→ `kornia-manifold`), beside the Lie groups | Nothing comparable exists in kornia-rs; a manifold construct, not a domain factor. Needs `SO3F64` only — no `SE3F64` |
//!
//! Staying here: [`scene`] (needs `ViBaKeyframe`), [`imu`] (needs `ImuFactor`,
//! which is solver window-slot bookkeeping rather than sensor data), and
//! [`rng`] (a thin wrapper over `rand::StdRng`, nothing to move).
//!
//! Note the friction, since it shapes the sequencing: kornia-slam consumes
//! kornia-rs via a git branch, so each upstream move is a land-wait-bump round
//! trip. Worth batching into two PRs — one per destination crate — rather than
//! going piecemeal, and worth coordinating the `kornia-algebra` one with the
//! Phase 0 rename to `kornia-manifold`.
//!
//! # Example
//!
//! ```
//! use kornia_slam::sim::{ArcConfig, Scene, SceneConfig, Trajectory};
//!
//! let trajectory = Trajectory::arc(&ArcConfig::default())?;
//! let scene = Scene::build(&trajectory, &SceneConfig::default())?;
//!
//! // Ground truth, exactly — not an approximation of it.
//! assert_eq!(scene.camera_poses.len(), 12);
//! assert!(!scene.visual.observations.is_empty());
//! # Ok::<(), kornia_slam::sim::SimError>(())
//! ```

use thiserror::Error;

pub mod imu;
pub mod landmarks;
pub mod rng;
pub mod scene;
pub mod spline;
pub mod trajectory;

pub use imu::{EUROC_IMU_CALIB, ImuSimConfig, build_imu_factors, generate_imu};
pub use landmarks::{
    LandmarkConfig, ObservationConfig, VisualData, generate_observations, sample_landmarks,
};
pub use rng::SimRng;
pub use scene::{Perturbation, Scene, SceneConfig};
pub use spline::{RSpline3, So3Spline};
pub use trajectory::{ArcConfig, DEFAULT_GRAVITY, Trajectory, TrajectoryState};

/// Errors from simulator construction and sampling.
#[derive(Debug, Error)]
pub enum SimError {
    /// A time was requested outside the spline's valid domain.
    #[error("time {t} outside spline domain [{start}, {end}]")]
    TimeOutOfRange {
        /// The requested time.
        t: f64,
        /// First valid time.
        start: f64,
        /// Last valid time.
        end: f64,
    },

    /// A spline was constructed with fewer control points than its order.
    #[error("spline needs at least {need} control points, got {got}")]
    TooFewControlPoints {
        /// Number supplied.
        got: usize,
        /// Number required.
        need: usize,
    },

    /// Knot spacing was zero, negative or non-finite.
    #[error("knot spacing must be positive and finite, got {0}")]
    InvalidKnotSpacing(f64),

    /// The translation and rotation splines cover different time ranges.
    #[error(
        "translation spline spans [{}, {}] but rotation spline spans [{}, {}]",
        translation.0, translation.1, rotation.0, rotation.1
    )]
    MismatchedSplineDomains {
        /// Translation spline domain.
        translation: (f64, f64),
        /// Rotation spline domain.
        rotation: (f64, f64),
    },

    /// No landmark survived visibility culling.
    #[error("no landmark was observed by enough keyframes")]
    NoVisibleLandmarks,

    /// A configuration value was inconsistent or out of range.
    #[error("invalid simulator configuration: {0}")]
    InvalidConfig(String),
}
