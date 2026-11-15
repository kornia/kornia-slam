//! The pose-acceptance gate: a tracked pose is checked against the inertial
//! prediction before it becomes the state.
//!
//! OpenVINS gates every update on its innovation (`UpdaterMSCKF.cpp:215-229`);
//! here the spread is measured, because the IMU-only covariance (0.002 deg
//! per frame) is far below the PnP pose's own spread.

use kornia_3d::pose::Pose3d;
use kornia_algebra::SO3F64;

use crate::tracking::motion::{camera_centre, coast_is_plausible};

/// Whether a tracked pose that disagrees with the inertial prediction is
/// rejected; `Disabled` accepts on inliers alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoseGate {
    #[default]
    Disabled,
    Enabled(PoseGateLimits),
}

/// How far a tracked pose may sit from the prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct PoseGateLimits {
    /// Largest angle, degrees, between the tracked and predicted rotation.
    pub max_rotation_gap_deg: f64,
    /// Largest distance between the tracked and predicted camera centre, as a
    /// speed: the limit is this times the frame interval.
    pub max_translation_gap_mps: f64,
    /// While lost, the fastest average speed, m/s, from the last tracked pose
    /// that a recovered pose may imply.
    pub max_recover_speed_mps: f64,
}

impl Default for PoseGateLimits {
    fn default() -> Self {
        Self {
            max_rotation_gap_deg: 5.0,
            max_translation_gap_mps: 6.0,
            max_recover_speed_mps: 10.0,
        }
    }
}

/// What a tracked pose is checked against; poses are world-to-camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GateReference {
    /// The inertial prediction of this frame.
    pub prediction: Pose3d,
    /// Seconds since the previous frame.
    pub dt_sec: f64,
    /// While lost: the last tracked pose and the seconds since it. The coasted
    /// position drifts, so it is not the position reference then.
    pub lost: Option<(Pose3d, f64)>,
}

/// How far a tracked pose is from its reference and from the previous pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PoseGap {
    /// Angle to the predicted rotation, degrees.
    pub rot_deg: f64,
    /// Distance to the predicted camera centre, m.
    pub trans_m: f64,
    /// Camera-centre step from the previous pose, m.
    pub step_m: f64,
    /// Rotation step from the previous pose, degrees.
    pub step_rot_deg: f64,
    /// While lost: average speed from the last tracked pose, m/s.
    pub recover_mps: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateVerdict {
    Accepted,
    RotationGap,
    TranslationGap,
}

fn angle_deg(from: &Pose3d, to: &Pose3d) -> f64 {
    let relative = Pose3d::between(from, to);
    SO3F64::from_matrix(&relative.rotation)
        .log()
        .length()
        .to_degrees()
}

impl PoseGap {
    pub(crate) fn measure(reference: &GateReference, pose_before: &Pose3d, pose: &Pose3d) -> Self {
        let centre = camera_centre(pose);
        Self {
            rot_deg: angle_deg(&reference.prediction, pose),
            trans_m: (centre - camera_centre(&reference.prediction)).length(),
            step_m: (centre - camera_centre(pose_before)).length(),
            step_rot_deg: angle_deg(pose_before, pose),
            recover_mps: reference.lost.map(|(before_loss, gap_sec)| {
                (centre - camera_centre(&before_loss)).length() / gap_sec
            }),
        }
    }
}

impl PoseGate {
    pub(crate) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }

    /// Rotation is tested first; `Accepted` when disabled.
    pub(crate) fn judge(
        &self,
        reference: &GateReference,
        gap: &PoseGap,
        pose: &Pose3d,
    ) -> GateVerdict {
        let Self::Enabled(limits) = self else {
            return GateVerdict::Accepted;
        };
        if gap.rot_deg.is_nan() || gap.rot_deg > limits.max_rotation_gap_deg {
            return GateVerdict::RotationGap;
        }
        let position_ok = match reference.lost {
            Some((before_loss, gap_sec)) => {
                coast_is_plausible(&before_loss, pose, gap_sec, limits.max_recover_speed_mps)
            }
            None => gap.trans_m <= limits.max_translation_gap_mps * reference.dt_sec,
        };
        if position_ok {
            GateVerdict::Accepted
        } else {
            GateVerdict::TranslationGap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;

    fn centred_at(x: f64) -> Pose3d {
        let mut pose = Pose3d::IDENTITY;
        pose.translation = Vec3F64::new(-x, 0.0, 0.0);
        pose
    }

    fn turned(deg: f64) -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix(),
            Vec3F64::ZERO,
        )
    }

    fn tracking(prediction: Pose3d, dt_sec: f64) -> GateReference {
        GateReference {
            prediction,
            dt_sec,
            lost: None,
        }
    }

    fn judge(gate: PoseGate, reference: &GateReference, pose: &Pose3d) -> GateVerdict {
        let gap = PoseGap::measure(reference, &reference.prediction, pose);
        gate.judge(reference, &gap, pose)
    }

    const ON: PoseGate = PoseGate::Enabled(PoseGateLimits {
        max_rotation_gap_deg: 5.0,
        max_translation_gap_mps: 6.0,
        max_recover_speed_mps: 10.0,
    });

    #[test]
    fn the_gap_of_equal_poses_is_zero_and_a_turn_reads_its_angle() {
        let reference = tracking(turned(3.0), 0.05);
        let gap = PoseGap::measure(&reference, &turned(3.0), &turned(3.0));
        assert!(gap.rot_deg < 1e-9 && gap.trans_m < 1e-12 && gap.step_m < 1e-12);
        assert_eq!(gap.recover_mps, None);

        let gap = PoseGap::measure(
            &tracking(Pose3d::IDENTITY, 0.05),
            &turned(0.0),
            &turned(19.7),
        );
        assert!((gap.rot_deg - 19.7).abs() < 1e-9, "{}", gap.rot_deg);
        assert!((gap.step_rot_deg - 19.7).abs() < 1e-9);
    }

    #[test]
    fn rotation_and_position_limits_while_tracking() {
        let reference = tracking(Pose3d::IDENTITY, 0.05);
        assert_eq!(judge(ON, &reference, &turned(4.9)), GateVerdict::Accepted);
        assert_eq!(
            judge(ON, &reference, &turned(5.1)),
            GateVerdict::RotationGap
        );
        assert_eq!(
            judge(ON, &reference, &centred_at(0.29)),
            GateVerdict::Accepted
        );
        assert_eq!(
            judge(ON, &reference, &centred_at(0.31)),
            GateVerdict::TranslationGap
        );
        let slower = tracking(Pose3d::IDENTITY, 0.1);
        assert_eq!(judge(ON, &slower, &centred_at(0.59)), GateVerdict::Accepted);
        assert_eq!(
            judge(ON, &slower, &centred_at(0.61)),
            GateVerdict::TranslationGap
        );
    }

    #[test]
    fn while_lost_the_position_is_bounded_from_the_last_tracked_pose() {
        // The coasted prediction is far off; only the walker bound applies.
        let reference = GateReference {
            prediction: centred_at(40.0),
            dt_sec: 0.05,
            lost: Some((centred_at(0.0), 1.75)),
        };
        assert_eq!(
            judge(ON, &reference, &centred_at(17.0)),
            GateVerdict::Accepted
        );
        assert_eq!(
            judge(ON, &reference, &centred_at(18.0)),
            GateVerdict::TranslationGap
        );
        let gap = PoseGap::measure(&reference, &centred_at(40.0), &centred_at(17.5));
        assert!((gap.recover_mps.unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_disabled_gate_accepts_anything() {
        let reference = tracking(Pose3d::IDENTITY, 0.05);
        assert_eq!(
            judge(PoseGate::Disabled, &reference, &centred_at(1095.0)),
            GateVerdict::Accepted
        );
        assert_eq!(PoseGate::default(), PoseGate::Disabled);
        assert!(!PoseGate::default().is_enabled() && ON.is_enabled());
    }

    #[test]
    fn a_non_finite_pose_is_rejected() {
        let reference = tracking(Pose3d::IDENTITY, 0.05);
        assert_ne!(
            judge(ON, &reference, &centred_at(f64::NAN)),
            GateVerdict::Accepted
        );
    }
}
