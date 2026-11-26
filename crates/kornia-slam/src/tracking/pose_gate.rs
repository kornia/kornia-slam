//! The pose-acceptance gate: a tracked pose is checked against the inertial
//! prediction before it becomes the state.
//!
//! OpenVINS gates every update on its innovation (`UpdaterMSCKF.cpp:215-229`);
//! here the spread is measured, because the IMU-only covariance (0.002 deg
//! per frame) is far below the PnP pose's own spread.

use std::collections::VecDeque;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SMatF64, SVecF64};

use crate::pose_conversion::angle_deg;
use crate::upstream::kornia_3d::pose_ext::camera_centre;

/// Whether a tracked pose that disagrees with the inertial prediction is
/// rejected; `Disabled` accepts on inliers alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoseGate {
    #[default]
    Disabled,
    /// The pose against the IMU-propagated covariance (card AC-2b); judged
    /// by the system after the tracker, so `judge` accepts.
    Innovation(InnovationGate),
}

/// The innovation test of a tracked pose: `δ = Log(T_pred⁻¹ T)` against
/// `S = P_pred + P_vis` at 99.9 %, after OpenVINS (`UpdaterMSCKF.cpp:215-229`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct InnovationGate {
    pub mode: InnovationMode,
}

/// `Observe` counts and logs the poses past the bound; `Reject` coasts them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InnovationMode {
    #[default]
    Observe,
    Reject,
}

/// χ²₆ at the owner's 99.9 %, and the χ²₆ median the consistency scale
/// divides by.
pub(crate) const CHI2_6_999: f64 = 22.458;
const CHI2_6_MEDIAN: f64 = 5.348;
/// The median NIS of the last this many accepted frames (median noise about
/// 5 %), and the fewest it is read from.
const SCALE_WINDOW: usize = 200;
const SCALE_MIN_SAMPLES: usize = 50;

impl InnovationGate {
    /// Whether `nis` is past the bound scaled by `scale`; NaN is.
    pub(crate) fn is_over(nis: f64, scale: f64) -> bool {
        let scaled = nis / scale;
        scaled.is_nan() || scaled > CHI2_6_999
    }
}

/// `δᵀ S⁻¹ δ`; `None` when `S` is not positive definite.
pub(crate) fn nis<const N: usize>(delta: &SVecF64<N>, s: &SMatF64<N, N>) -> Option<f64> {
    let value = delta.dot(&s.cholesky()?.solve(delta));
    value.is_finite().then_some(value)
}

/// How far the NIS of accepted frames runs above χ²₆: `c = max(1, median /
/// 5.348)` over the last [`SCALE_WINDOW`] of them. The pixel noise behind
/// `P_vis` is a model, so the bound follows the measured consistency; the
/// floor keeps it from tightening below the model.
#[derive(Debug, Clone, Default)]
pub(crate) struct ConsistencyScale {
    window: VecDeque<f64>,
    sorted: Vec<f64>,
    scale: f64,
}

impl ConsistencyScale {
    pub(crate) fn push(&mut self, nis: f64) {
        if !nis.is_finite() {
            return;
        }
        if self.window.len() == SCALE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(nis);
        self.sorted.clear();
        self.sorted.extend(self.window.iter().copied());
        self.sorted.sort_unstable_by(f64::total_cmp);
        let n = self.sorted.len();
        let median = if n % 2 == 1 {
            self.sorted[n / 2]
        } else {
            0.5 * (self.sorted[n / 2 - 1] + self.sorted[n / 2])
        };
        self.scale = (median / CHI2_6_MEDIAN).max(1.0);
    }

    /// `c`, once [`SCALE_MIN_SAMPLES`] frames feed the median.
    pub(crate) fn scale(&self) -> Option<f64> {
        (self.window.len() >= SCALE_MIN_SAMPLES).then_some(self.scale)
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
    pub(crate) fn innovation(&self) -> Option<&InnovationGate> {
        match self {
            Self::Innovation(gate) => Some(gate),
            Self::Disabled => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{SO3F64, Vec3F64};

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

    /// The innovation gate is judged by the system; its default is
    /// `Observe`, and the gate's default stays `Disabled`.
    #[test]
    fn the_innovation_variant_observes_by_default() {
        let gate = PoseGate::Innovation(InnovationGate::default());
        assert!(gate.innovation().is_some());
        assert_eq!(PoseGate::default(), PoseGate::Disabled);
        assert!(PoseGate::default().innovation().is_none());
        assert_eq!(InnovationGate::default().mode, InnovationMode::Observe);
        assert!(!InnovationGate::is_over(CHI2_6_999, 1.0));
        assert!(InnovationGate::is_over(CHI2_6_999 * 1.01, 1.0));
        assert!(!InnovationGate::is_over(CHI2_6_999 * 1.9, 2.0));
    }
}
