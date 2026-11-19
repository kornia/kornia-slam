//! Inertial initialization: estimating metric scale, gravity, velocities and
//! IMU biases from a window of visually tracked keyframes.
//!
//! [`ImuInitializer`] owns the readiness gate and the result it applies; the
//! optimization itself is in `solve`, and its factors in [`factor`].

pub mod factor;
mod gate;
mod solve;
mod stationary;

pub use stationary::{ImuWindowStats, StationaryInitError};

use kornia_3d::pose::Pose3d;
use kornia_algebra::{QuatF64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

use crate::mapping::map::{
    InertialAlignment, InertialAlignmentError, Keyframe, KeyframeVelocity, Map,
};
// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rotation that takes unit vector `from` to unit vector `to`.
pub(crate) fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
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

/// Whether `kf` belongs to the initialization window from `start_idx`.
/// Anchors never do: no IMU edge reaches them.
fn in_window(kf: &Keyframe, start_idx: usize) -> bool {
    kf.frame.idx >= start_idx && !kf.anchor
}

/// Which keyframes a VIBA solve covers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VibaWindow {
    /// Every keyframe of the initialization window.
    #[default]
    All,
    /// The window without the keyframes inserted while tracking was lost.
    VisualOnly,
}

impl VibaWindow {
    fn admits(self, kf: &Keyframe) -> bool {
        match self {
            Self::All => true,
            Self::VisualOnly => !kf.inserted_while_lost,
        }
    }
}

/// A keyframe window is monocular unless its first keyframe carries metric
/// depth (stereo, or a rig's overlap points).
fn window_is_mono(kfs: &[&Keyframe]) -> bool {
    !kfs.first().map(|kf| kf.frame.is_metric()).unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitConfig {
    pub min_keyframes: usize,
    pub min_time_sec: f64,
    pub min_motion: f64,
    pub viba_window: VibaWindow,
    /// Drops edges whose velocity cost exceeds this multiple of the median,
    /// then solves again; `0.0` is off.
    pub viba_velocity_gate: f64,
}

#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale: f64,
    pub gravity_world: Vec3F64,
    pub keyframe_velocities: Vec<KeyframeVelocity>,
    pub bias: ImuBias,
    /// `(prev, curr)` keyframe indices of the edges the velocity gate left
    /// out, worst first; empty with the gate off.
    pub gated_edges: Vec<(usize, usize)>,
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

    fn in_solve_window(&self, kf: &Keyframe, start_idx: usize) -> bool {
        in_window(kf, start_idx) && self.config.viba_window.admits(kf)
    }

    // ── readiness gate ────────────────────────────────────────────────────────

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        let Some(start_idx) = start_idx else {
            return false;
        };

        let kfs: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| self.in_solve_window(kf, start_idx))
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

/// Where tracking resumes after an initialization is applied: the newest
/// window keyframe's world-to-camera pose and metric world-frame velocity.
#[derive(Debug, Clone, Copy)]
pub struct AlignedTrackingState {
    pub pose_world_to_cam: Pose3d,
    pub velocity_world: Vec3F64,
}

impl ImuInitializer {
    // ── apply to map ──────────────────────────────────────────────────────────

    /// Aligns the map to the solved scale and gravity and writes the window's
    /// velocities and bias, then takes the new bias and canonical gravity.
    ///
    /// Returns where tracking resumes, or `None` if the window has no
    /// keyframe; a refused alignment leaves the map, bias and gravity as they
    /// were.
    pub fn apply_initialization(
        &self,
        map: &mut Map,
        imu_bias: &mut ImuBias,
        gravity_world: &mut Vec3F64,
        init: ImuInitResult,
        start_idx: usize,
    ) -> Result<Option<AlignedTrackingState>, InertialAlignmentError> {
        debug_eprintln!(
            "[imu_init] applying: scale={:.4}  g=({:.3},{:.3},{:.3})",
            init.scale,
            init.gravity_world.x,
            init.gravity_world.y,
            init.gravity_world.z
        );

        // 1–3. Scale to metric, rotate gravity onto +Y, and write each solved
        // keyframe's velocity and bias, by keyframe identity, as one validated
        // map correction.
        let g_norm = init.gravity_world / init.gravity_world.length();
        let rwg = rotation_from_to(g_norm, Vec3F64::new(0.0, 1.0, 0.0));
        if let Err(error) = map.apply_inertial_alignment(InertialAlignment {
            scale: init.scale,
            rotation: rwg,
            keyframe_velocities: init.keyframe_velocities,
            bias: init.bias,
        }) {
            debug_eprintln!("[imu_init] alignment refused, map left unchanged: {error}");
            return Err(error);
        }

        *gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        *imu_bias = init.bias;

        // 4. Tracking resumes from the last initialized keyframe.
        Ok(map
            .keyframes()
            .iter()
            .rfind(|kf| in_window(kf, start_idx))
            .map(|last_kf| AlignedTrackingState {
                pose_world_to_cam: last_kf.frame.pose_world_to_cam,
                velocity_world: last_kf.velocity_world,
            }))
    }
}

#[cfg(test)]
mod tests;
