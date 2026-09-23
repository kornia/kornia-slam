//! Inertial initialization: estimating metric scale, gravity, velocities and
//! IMU biases from a window of visually tracked keyframes.
//!
//! [`ImuInitializer`] owns the readiness gate and the result it applies; the
//! optimization itself is in `solve`, and its factors in [`factor`].

pub mod factor;
mod solve;

use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, QuatF64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

use crate::mapping::map::{InertialAlignment, Keyframe, KeyframeVelocity, Map};
use crate::tracking::SystemState;
use factor::{InertialInitFactor, KfConst, WeightedZeroPrior};
use kornia_algebra::optim::{LevenbergMarquardt, Problem, Variable, VariableType};
// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rotation that takes unit vector `from` to unit vector `to`.
fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
    let from = from.normalize();
    let to = to.normalize();
    let dot = from.dot(to).clamp(-1.0, 1.0);
    let cross = from.cross(to);

    // Anti-parallel: pick an arbitrary perpendicular axis.
    if dot < -1.0 + 1e-9 {
        let perp = if from.x.abs() < 0.9 {
            Vec3F64::new(1.0, 0.0, 0.0)
        } else {
            Vec3F64::new(0.0, 1.0, 0.0)
        };
        let axis = from.cross(perp).normalize();
        return SO3F64::from_quaternion(QuatF64::from_array([axis.x, axis.y, axis.z, 0.0]));
    }

    let w = ((1.0 + dot) / 2.0).sqrt();
    let s = 1.0 / (2.0 * w);
    SO3F64::from_quaternion(QuatF64::from_array([
        cross.x * s,
        cross.y * s,
        cross.z * s,
        w,
    ]))
}

/// A keyframe window is monocular unless its first keyframe carries stereo data.
fn window_is_mono(kfs: &[&Keyframe]) -> bool {
    !kfs.first().map(|kf| kf.frame.is_stereo()).unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitConfig {
    pub min_keyframes: usize,
    pub min_time_sec: f64,
    pub min_motion: f64,
}

#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale: f64,
    pub gravity_world: Vec3F64,
    pub velocities_world: Vec<Vec3F64>,
    pub bias: ImuBias,
}

// ─────────────────────────────────────────────────────────────────────────────
// ImuInitializer
// ─────────────────────────────────────────────────────────────────────────────

pub struct ImuInitializer {
    pub config: ImuInitConfig,
}

impl ImuInitializer {
    pub fn new(config: ImuInitConfig) -> Self {
        Self { config }
    }

    // ── readiness gate ────────────────────────────────────────────────────────

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        let Some(start_idx) = start_idx else {
            return false;
        };

        let kfs: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        if kfs.len() < self.config.min_keyframes {
            return false;
        }

        // ORB-SLAM3 uses minTime=2.0s (mono) vs 1.0s (stereo) — `config.min_time_sec`
        // holds the stereo (stricter/smaller) value; mono doubles it.
        let min_time_sec = if window_is_mono(&kfs) {
            self.config.min_time_sec * 2.0
        } else {
            self.config.min_time_sec
        };

        let imu_time: f64 = map
            .imu_factors()
            .iter()
            .filter(|f| f.curr_kf_idx >= start_idx)
            .map(|f| f.preintegrated.dt)
            .sum();
        if imu_time < min_time_sec {
            return false;
        }

        let t0 = kfs
            .first()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        let t1 = kfs
            .last()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        (t1 - t0).length() >= self.config.min_motion
    }
}

impl ImuInitializer {
    // ── apply to map & state ──────────────────────────────────────────────────

    pub fn apply_initialization(
        &self,
        map: &mut Map,
        state: &mut SystemState,
        imu_bias: &mut ImuBias,
        gravity_world: &mut Vec3F64,
        init: ImuInitResult,
        start_idx: usize,
    ) {
        eprintln!(
            "[imu_init] applying: scale={:.4}  g=({:.3},{:.3},{:.3})",
            init.scale, init.gravity_world.x, init.gravity_world.y, init.gravity_world.z
        );

        // 1–3. Scale to metric, rotate gravity onto +Y, and write each
        // keyframe's velocity and bias, as one validated map correction.
        // Velocities pair with the window's keyframes in map order, as the
        // solver produced them, but are published by keyframe identity.
        let g_norm = init.gravity_world / init.gravity_world.length();
        let rwg = rotation_from_to(g_norm, Vec3F64::new(0.0, 1.0, 0.0));
        let keyframe_velocities: Vec<KeyframeVelocity> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .zip(init.velocities_world)
            .map(|(kf, velocity_world)| KeyframeVelocity {
                keyframe_idx: kf.frame.idx,
                velocity_world,
            })
            .collect();
        if let Err(error) = map.apply_inertial_alignment(InertialAlignment {
            scale: init.scale,
            rotation: rwg,
            keyframe_velocities,
            bias: init.bias,
        }) {
            eprintln!("[imu_init] alignment refused, map left unchanged: {error}");
            return;
        }

        // 4. Update the tracker state from the last initialized keyframe.
        if let Some(last_kf) = map.keyframes().iter().rfind(|kf| kf.frame.idx >= start_idx) {
            state.velocity_world = last_kf.velocity_world;
            state.pose_world_to_cam = last_kf.frame.pose_world_to_cam;
        }

        state.velocity = None;
        state.imu_initialized = true;
        *gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        *imu_bias = init.bias;
    }
}

#[cfg(test)]
mod tests;
