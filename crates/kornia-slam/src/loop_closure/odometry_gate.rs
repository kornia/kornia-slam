//! Odometry consistency of verified loops.
//!
//! Geometric verification proves that two keyframes see the same structure,
//! not that they are the same place: repeated structure verifies too. A true
//! loop's correction is explained by the drift accumulated since the matched
//! keyframe, so a correction far beyond that drift is rejected before it can
//! enter the pose graph, where it would stay.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

use super::{LoopEpisodeConfig, VerifiedLoopEdge};

/// Odometry drift as one standard deviation that grows linearly with the
/// keyframe path travelled: `σ = per_m · path_m + floor`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftModel {
    /// Translation drift per metre travelled (m/m).
    pub translation_per_m: f64,
    /// Translation σ at zero path (m), the noise of the verified pose itself.
    pub translation_floor_m: f64,
    /// Yaw drift per metre travelled (rad/m).
    pub yaw_per_m_rad: f64,
    /// Yaw σ at zero path (rad), the noise of the verified pose itself.
    pub yaw_floor_rad: f64,
}

impl Default for DriftModel {
    /// The per-metre terms are the RMS revisit drift of stereo-inertial
    /// odometry with loop closing off over a 61 m sim run (1.86 % and
    /// 0.026°/m), rounded up; the floors bound a stereo PnP pose at 2-3 m
    /// depth.
    fn default() -> Self {
        Self {
            translation_per_m: 0.019,
            translation_floor_m: 0.10,
            yaw_per_m_rad: 0.026_f64.to_radians(),
            yaw_floor_rad: 1.0_f64.to_radians(),
        }
    }
}

impl DriftModel {
    pub fn translation_sigma_m(&self, path_m: f64) -> f64 {
        self.translation_per_m * path_m + self.translation_floor_m
    }

    pub fn yaw_sigma_rad(&self, path_m: f64) -> f64 {
        self.yaw_per_m_rad * path_m + self.yaw_floor_rad
    }
}

/// Rejects a verified loop whose correction exceeds `k_sigma` drift-model σ
/// in translation or yaw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopOdometryGate {
    Disabled,
    Enabled { drift: DriftModel, k_sigma: f64 },
}

impl Default for LoopOdometryGate {
    fn default() -> Self {
        Self::Enabled {
            drift: DriftModel::default(),
            k_sigma: 3.0,
        }
    }
}

/// What the gate concluded about one correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopGateVerdict {
    pub sigma_t_m: f64,
    pub sigma_yaw_rad: f64,
    pub k_sigma: f64,
    pub accepted: bool,
}

impl LoopOdometryGate {
    /// `None` when the gate is disabled.
    pub fn judge(&self, correction: &LoopCorrection) -> Option<LoopGateVerdict> {
        let Self::Enabled { drift, k_sigma } = *self else {
            return None;
        };
        let sigma_t_m = drift.translation_sigma_m(correction.path_m);
        let sigma_yaw_rad = drift.yaw_sigma_rad(correction.path_m);
        let accepted = correction.translation_m() <= k_sigma * sigma_t_m
            && correction.yaw_rad.abs() <= k_sigma * sigma_yaw_rad;
        Some(LoopGateVerdict {
            sigma_t_m,
            sigma_yaw_rad,
            k_sigma,
            accepted,
        })
    }
}

/// The correction a verified loop would apply to the query keyframe, and the
/// keyframe path whose drift would have to explain it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopCorrection {
    /// Verified minus estimated query camera centre, in the world frame (m),
    /// with the candidate keyframe held where it is.
    pub translation_world: Vec3F64,
    /// Signed rotation of the correction about gravity; its full angle when
    /// gravity is unknown (rad).
    pub yaw_rad: f64,
    /// Summed camera-centre steps between consecutive keyframes (m).
    pub path_m: f64,
    /// Summed rotation angles between consecutive keyframes (rad).
    pub path_rotation_rad: f64,
}

impl LoopCorrection {
    /// Poses are world-to-camera; `path` holds the keyframe poses from one
    /// end to the other in map order.
    pub fn measure(
        candidate: Pose3d,
        query: Pose3d,
        path: impl IntoIterator<Item = Pose3d>,
        edge: &VerifiedLoopEdge,
        gravity_world: Option<Vec3F64>,
    ) -> Self {
        let (path_m, path_rotation_rad) = path_extent(path);
        let (translation_world, yaw_rad) =
            query_correction(&candidate, &query, &edge.candidate_to_query, gravity_world);
        Self {
            translation_world,
            yaw_rad,
            path_m,
            path_rotation_rad,
        }
    }

    pub fn translation_m(&self) -> f64 {
        self.translation_world.length()
    }
}

/// Correction of the estimated query pose by the verified relative pose,
/// holding the candidate fixed: centre offset (m, world) and rotation about
/// gravity (rad).
pub(super) fn query_correction(
    candidate_world_to_cam: &Pose3d,
    query_world_to_cam: &Pose3d,
    candidate_to_query: &Pose3d,
    gravity_world: Option<Vec3F64>,
) -> (Vec3F64, f64) {
    let verified_query = candidate_to_query.compose(candidate_world_to_cam);
    let translation =
        verified_query.inverse().translation - query_world_to_cam.inverse().translation;
    // World-frame rotation taking the estimated query orientation to the verified one.
    let world_rotation = verified_query.rotation.transpose() * query_world_to_cam.rotation;
    let rotation_vector = SO3F64::from_matrix(&world_rotation).log();
    let yaw = match gravity_world.filter(|g| g.length() > 0.0) {
        Some(gravity) => rotation_vector.dot(gravity.normalize()),
        None => rotation_vector.length(),
    };
    (translation, yaw)
}

/// Camera-centre path length (m) and accumulated rotation (rad) along
/// world-to-camera poses.
pub(super) fn path_extent(poses: impl IntoIterator<Item = Pose3d>) -> (f64, f64) {
    let mut poses = poses.into_iter();
    let Some(mut previous) = poses.next() else {
        return (0.0, 0.0);
    };
    let (mut length, mut rotation) = (0.0, 0.0);
    for pose in poses {
        length += (pose.inverse().translation - previous.inverse().translation).length();
        let relative = Pose3d::between(&previous, &pose);
        rotation += SO3F64::from_matrix(&relative.rotation).log().length();
        previous = pose;
    }
    (length, rotation)
}

/// Before the first loop in a place, requires consecutive detections that
/// imply the same correction, so one aliased match cannot open a region.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum LoopTemporalConsistency {
    #[default]
    Disabled,
    Enabled {
        min_detections: usize,
        /// Largest difference between consecutive corrections (m).
        max_translation_m: f64,
        /// Largest difference between consecutive yaw corrections (rad).
        max_yaw_rad: f64,
    },
}

impl LoopTemporalConsistency {
    /// `min_detections` with tolerances three verification-noise floors wide.
    pub fn enabled(min_detections: usize) -> Self {
        Self::Enabled {
            min_detections,
            max_translation_m: 0.3,
            max_yaw_rad: 3.0_f64.to_radians(),
        }
    }
}

/// Outcome of offering one gated detection to the temporal rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalDecision {
    Pass,
    Pending { hits: usize, required: usize },
}

#[derive(Debug, Clone, Copy)]
struct DetectionChain {
    candidate_anchor_order: usize,
    last_query_order: usize,
    last_correction: LoopCorrection,
    hits: usize,
}

/// Counts consecutive detections whose corrections agree, using the episode
/// tracker's query gap and candidate neighbourhood.
#[derive(Debug, Clone)]
pub(crate) struct TemporalConsistencyTracker {
    config: LoopTemporalConsistency,
    episode: LoopEpisodeConfig,
    chain: Option<DetectionChain>,
}

impl TemporalConsistencyTracker {
    pub(crate) fn new(config: LoopTemporalConsistency, episode: LoopEpisodeConfig) -> Self {
        Self {
            config,
            episode,
            chain: None,
        }
    }

    /// `None` when the rule is disabled.
    pub(crate) fn observe(
        &mut self,
        query_order: usize,
        candidate_order: usize,
        correction: LoopCorrection,
    ) -> Option<TemporalDecision> {
        let LoopTemporalConsistency::Enabled {
            min_detections,
            max_translation_m,
            max_yaw_rad,
        } = self.config
        else {
            return None;
        };
        let consistent = self.chain.as_ref().is_some_and(|chain| {
            query_order >= chain.last_query_order
                && query_order - chain.last_query_order <= self.episode.max_query_gap
                && candidate_order.abs_diff(chain.candidate_anchor_order)
                    <= self.episode.candidate_neighborhood_radius
                && (correction.translation_world - chain.last_correction.translation_world).length()
                    <= max_translation_m
                && (correction.yaw_rad - chain.last_correction.yaw_rad).abs() <= max_yaw_rad
        });
        let hits = if consistent {
            self.chain.map_or(1, |chain| chain.hits + 1)
        } else {
            1
        };
        let candidate_anchor_order = match self.chain {
            Some(chain) if consistent => chain.candidate_anchor_order,
            _ => candidate_order,
        };
        self.chain = Some(DetectionChain {
            candidate_anchor_order,
            last_query_order: query_order,
            last_correction: correction,
            hits,
        });
        let required = min_detections.max(1);
        Some(if hits >= required {
            TemporalDecision::Pass
        } else {
            TemporalDecision::Pending { hits, required }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;

    fn yaw_about_z(angle: f64) -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.0, 0.0, angle)).matrix()
    }

    /// World-to-camera pose of a camera at `centre` with world-to-camera rotation `rotation`.
    fn pose_at(centre: Vec3F64, rotation: Mat3F64) -> Pose3d {
        Pose3d::new(rotation, -(rotation * centre))
    }

    fn correction(translation: f64, yaw_deg: f64, path_m: f64) -> LoopCorrection {
        LoopCorrection {
            translation_world: Vec3F64::new(translation, 0.0, 0.0),
            yaw_rad: yaw_deg.to_radians(),
            path_m,
            path_rotation_rad: 0.0,
        }
    }

    #[test]
    fn drift_sigma_grows_linearly_with_path() {
        let model = DriftModel {
            translation_per_m: 0.02,
            translation_floor_m: 0.1,
            yaw_per_m_rad: 0.001,
            yaw_floor_rad: 0.01,
        };
        assert!((model.translation_sigma_m(0.0) - 0.1).abs() < 1e-12);
        assert!((model.translation_sigma_m(20.0) - 0.5).abs() < 1e-12);
        assert!((model.yaw_sigma_rad(20.0) - 0.03).abs() < 1e-12);
    }

    #[test]
    fn exact_verified_pose_means_no_correction() {
        let candidate = pose_at(Vec3F64::new(1.0, 2.0, 0.0), yaw_about_z(0.3));
        let query = pose_at(Vec3F64::new(1.5, 2.2, 0.1), yaw_about_z(0.5));
        let (translation, yaw) = query_correction(
            &candidate,
            &query,
            &Pose3d::between(&candidate, &query),
            Some(Vec3F64::new(0.0, 0.0, -9.81)),
        );
        assert!(translation.length() < 1e-12);
        assert!(yaw.abs() < 1e-12);
    }

    /// The verified pose puts the query 6 m from its estimate, with a 2° yaw
    /// about gravity: the correction reports both, in the world frame.
    #[test]
    fn correction_measures_centre_offset_and_yaw_about_gravity() {
        let candidate = pose_at(Vec3F64::new(0.0, 0.0, 0.0), yaw_about_z(0.0));
        let query_estimate = pose_at(Vec3F64::new(6.0, 0.0, 0.0), yaw_about_z(0.0));
        let query_verified = pose_at(
            Vec3F64::new(0.0, 0.0, 0.0),
            yaw_about_z(2.0_f64.to_radians()),
        );
        let (translation, yaw) = query_correction(
            &candidate,
            &query_estimate,
            &Pose3d::between(&candidate, &query_verified),
            Some(Vec3F64::new(0.0, 0.0, -9.81)),
        );
        assert!((translation - Vec3F64::new(-6.0, 0.0, 0.0)).length() < 1e-9);
        assert!((yaw.abs() - 2.0_f64.to_radians()).abs() < 1e-9);
        let (_, full_angle) = query_correction(
            &candidate,
            &query_estimate,
            &Pose3d::between(&candidate, &query_verified),
            None,
        );
        assert!((full_angle - 2.0_f64.to_radians()).abs() < 1e-9);
    }

    #[test]
    fn path_extent_sums_centre_steps_and_rotations() {
        let poses = [
            pose_at(Vec3F64::new(0.0, 0.0, 0.0), yaw_about_z(0.0)),
            pose_at(Vec3F64::new(3.0, 0.0, 0.0), yaw_about_z(0.1)),
            pose_at(Vec3F64::new(3.0, 4.0, 0.0), yaw_about_z(0.3)),
        ];
        let (length, rotation) = path_extent(poses);
        assert!((length - 7.0).abs() < 1e-12);
        assert!((rotation - 0.3).abs() < 1e-12);
        assert_eq!(path_extent([]), (0.0, 0.0));
    }

    /// The sim's false loop: 6 m correction over about 21 m of path. The 3σ
    /// translation bound there is under 1.5 m, while 0.5 m of drift passes.
    #[test]
    fn default_gate_rejects_false_loop_and_keeps_drift_sized_correction() {
        let gate = LoopOdometryGate::default();
        let false_loop = gate.judge(&correction(6.0, -2.0, 21.0)).unwrap();
        assert!(!false_loop.accepted);
        assert!(3.0 * false_loop.sigma_t_m < 1.5);
        assert!(gate.judge(&correction(0.5, 1.0, 21.0)).unwrap().accepted);
    }

    #[test]
    fn gate_rejects_on_yaw_alone() {
        let gate = LoopOdometryGate::default();
        let verdict = gate.judge(&correction(0.05, 10.0, 5.0)).unwrap();
        assert!(!verdict.accepted);
        assert!(gate.judge(&correction(0.05, -2.0, 5.0)).unwrap().accepted);
    }

    #[test]
    fn disabled_gate_judges_nothing() {
        assert!(
            LoopOdometryGate::Disabled
                .judge(&correction(100.0, 90.0, 1.0))
                .is_none()
        );
    }

    #[test]
    fn temporal_rule_needs_consecutive_agreeing_corrections() {
        let mut tracker = TemporalConsistencyTracker::new(
            LoopTemporalConsistency::enabled(3),
            LoopEpisodeConfig::default(),
        );
        assert_eq!(
            tracker.observe(100, 10, correction(0.4, 0.0, 20.0)),
            Some(TemporalDecision::Pending {
                hits: 1,
                required: 3
            })
        );
        // Disagrees by 1 m: the chain restarts.
        assert_eq!(
            tracker.observe(101, 11, correction(1.4, 0.0, 20.0)),
            Some(TemporalDecision::Pending {
                hits: 1,
                required: 3
            })
        );
        assert_eq!(
            tracker.observe(102, 12, correction(1.5, 0.5, 20.0)),
            Some(TemporalDecision::Pending {
                hits: 2,
                required: 3
            })
        );
        assert_eq!(
            tracker.observe(103, 12, correction(1.45, 0.2, 20.0)),
            Some(TemporalDecision::Pass)
        );
        // Too far along the query side: a new chain.
        assert_eq!(
            tracker.observe(120, 12, correction(1.45, 0.2, 20.0)),
            Some(TemporalDecision::Pending {
                hits: 1,
                required: 3
            })
        );
    }

    #[test]
    fn disabled_temporal_rule_decides_nothing() {
        let mut tracker = TemporalConsistencyTracker::new(
            LoopTemporalConsistency::Disabled,
            LoopEpisodeConfig::default(),
        );
        assert!(tracker.observe(1, 0, correction(0.0, 0.0, 1.0)).is_none());
    }
}
