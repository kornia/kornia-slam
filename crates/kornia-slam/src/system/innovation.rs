//! Card AC-2b: a tracked pose against the IMU prediction as an innovation
//! test. `δ = Log(T_pred⁻¹ T)` in `(υ, ω)` under `T_pred · Exp(δ)`, the
//! layout of the propagated covariance, against `S = P_pred + P_vis`
//! (OpenVINS `UpdaterMSCKF.cpp:215-229`); a refused pose coasts, as
//! ORB-SLAM3 carries a failed frame on the IMU (`Tracking.cc:1981-1998`).

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SMatF64, SO3F64, SVecF64};

use crate::Frame;
use crate::mapping::vi_ba_schur::RigObservationCamera;
use crate::tracking::pose_estimation::Estimate;
use crate::tracking::pose_estimation::map_projection::MapProjectionRejectReason;
use crate::tracking::pose_estimation::pose_inertial::{PoseInertialObservation, reprojection_row};
use crate::tracking::pose_gate::{GateReference, InnovationGate, InnovationMode, nis};
use crate::upstream::kornia_3d::pose_ext::camera_centre;

use super::SlamSystem;
use super::estimator::{centre_jacobian, rig_observation_cameras};

/// The NIS of one tracked pose, whole and in its two parts: the rotation
/// `ω` and the camera centre, each against its own marginal of `S`, so both
/// read the same wherever the world origin is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Innovation {
    pub nis6: f64,
    pub nis_rot: f64,
    pub nis_trans: f64,
}

/// `Log(prediction⁻¹ · pose)` in `(υ, ω)`, the inverse of the right
/// perturbation `prediction · Exp(υ, ω)` with `t = t_p + R_p Jl(ω) υ`.
fn pose_delta(prediction: &Pose3d, pose: &Pose3d) -> SVecF64<6> {
    let rotation_pt = prediction.rotation.transpose();
    let omega = SO3F64::from_matrix(&(rotation_pt * pose.rotation)).log();
    let upsilon = SO3F64::left_jacobian(omega).inverse()
        * (rotation_pt * (pose.translation - prediction.translation));
    SVecF64::<6>::new(upsilon.x, upsilon.y, upsilon.z, omega.x, omega.y, omega.z)
}

/// The innovation of `pose` against `prediction`, with the prediction's
/// pose covariance `p_pred` and the visual information `vis_information`
/// of `pose`, both in `δ`'s layout. `None` when a matrix is not positive
/// definite.
pub(crate) fn innovation(
    prediction: &Pose3d,
    pose: &Pose3d,
    p_pred: &SMatF64<6, 6>,
    vis_information: &SMatF64<6, 6>,
) -> Option<Innovation> {
    let delta = pose_delta(prediction, pose);
    let s = p_pred + vis_information.cholesky()?.inverse();
    let s = (s + s.transpose()) * 0.5;
    let omega: SVecF64<3> = delta.fixed_rows::<3>(3).into_owned();
    let s_rot: SMatF64<3, 3> = s.fixed_view::<3, 3>(3, 3).into_owned();
    let a = centre_jacobian(camera_centre(prediction));
    Some(Innovation {
        nis6: nis(&delta, &s)?,
        nis_rot: nis(&omega, &s_rot)?,
        nis_trans: nis(&(a * delta), &(a * s * a.transpose()))?,
    })
}

/// `Σ w Jᵀ J` over the rows within the final PnP gate at `pose_cw`
/// (`w χ² ≤ gate²`, `w = 1/σ²`), and how many there are.
fn visual_information(
    observations: &[PoseInertialObservation],
    primary: &PinholeCamera,
    cameras: &[RigObservationCamera],
    pose_cw: &Pose3d,
    gate_px: f64,
) -> (SMatF64<6, 6>, usize) {
    let gate_sq = gate_px * gate_px;
    let mut information = SMatF64::<6, 6>::zeros();
    let mut inliers = 0;
    for observation in observations {
        let Some(row) = reprojection_row(observation, primary, cameras, pose_cw) else {
            continue;
        };
        if observation.information * row.residual.norm_squared() > gate_sq {
            continue;
        }
        information += row.jacobian.transpose() * row.jacobian * observation.information;
        inliers += 1;
    }
    (information, inliers)
}

/// Whether a judged pose is past the bound, and why it is not judged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Within,
    Over,
    /// Fewer than 50 accepted frames feed the consistency scale.
    WarmingUp,
    /// No propagated covariance, or a matrix not positive definite.
    Unavailable,
}

impl Verdict {
    /// Only `Reject` refuses, and only a pose past the bound.
    fn refused_under(self, mode: InnovationMode) -> bool {
        self == Self::Over && mode == InnovationMode::Reject
    }
}

impl SlamSystem {
    /// Under `pose_gate: Innovation`, judges a tracked `estimate` against the
    /// trusted prediction of `gate` (`None` without one: not judged). Fails
    /// open when the statistic is missing or the scale is warming up. Only
    /// `Reject` refuses, with the larger part as the reason.
    pub(super) fn judge_pose_innovation(
        &mut self,
        frame: &Frame,
        estimate: &Estimate,
        gate: Option<&GateReference>,
        timestamp_sec: f64,
    ) -> Result<(), MapProjectionRejectReason> {
        let (Some(&InnovationGate { mode }), Some(reference)) =
            (self.tracker.pose_gate().innovation(), gate)
        else {
            return Ok(());
        };
        let (measured, inliers) =
            self.measure_innovation(frame, estimate, &reference.prediction, timestamp_sec);
        let scale = self.estimator.innovation_scale.scale();
        let verdict = match (measured, scale) {
            (None, _) => Verdict::Unavailable,
            (Some(_), None) => Verdict::WarmingUp,
            (Some(m), Some(c)) if InnovationGate::is_over(m.nis6, c) => Verdict::Over,
            (Some(_), Some(_)) => Verdict::Within,
        };
        let refused = verdict.refused_under(mode);
        self.count_innovation(verdict, refused, scale);
        if let Some(m) = measured.filter(|_| !refused) {
            self.estimator.innovation_scale.push(m.nis6);
        }
        if self.debug {
            let m = measured.unwrap_or(Innovation {
                nis6: f64::NAN,
                nis_rot: f64::NAN,
                nis_trans: f64::NAN,
            });
            self.dbg(format!(
                "[nis] frame={} nis6={:.3} nis_rot={:.3} nis_trans={:.3} c={:.3} inl={inliers} \
                 lost={} verdict={}",
                frame.idx,
                m.nis6,
                m.nis_rot,
                m.nis_trans,
                scale.unwrap_or(f64::NAN),
                u8::from(reference.lost.is_some()),
                match verdict {
                    Verdict::Within => "ok",
                    Verdict::Over if refused => "reject",
                    Verdict::Over => "over",
                    Verdict::WarmingUp => "warm",
                    Verdict::Unavailable => "na",
                },
            ));
        }
        match measured {
            Some(m) if refused && m.nis_rot >= m.nis_trans => {
                Err(MapProjectionRejectReason::RotationGap)
            }
            Some(_) if refused => Err(MapProjectionRejectReason::TranslationGap),
            _ => Ok(()),
        }
    }

    /// The innovation of `estimate` and its visual inlier count; `None`
    /// before IMU initialization or without a propagated covariance.
    fn measure_innovation(
        &self,
        frame: &Frame,
        estimate: &Estimate,
        prediction: &Pose3d,
        timestamp_sec: f64,
    ) -> (Option<Innovation>, usize) {
        let observations = self.pose_inertial_observations(frame, &estimate.matches);
        let gate_px = self
            .tracker
            .map_projection_config()
            .pnp
            .final_reproj_threshold_px;
        let (information, inliers) = visual_information(
            &observations,
            &self.rig.camera,
            &rig_observation_cameras(frame),
            &estimate.pose,
            gate_px,
        );
        if !self.state.imu_initialized {
            return (None, inliers);
        }
        let measured = self
            .predicted_covariance(timestamp_sec, prediction)
            .and_then(|covariance| {
                let p_pred: SMatF64<6, 6> = covariance.fixed_view::<6, 6>(0, 0).into_owned();
                innovation(prediction, &estimate.pose, &p_pred, &information)
            });
        (measured, inliers)
    }

    fn count_innovation(&mut self, verdict: Verdict, refused: bool, scale: Option<f64>) {
        let c = &mut self.counters;
        match verdict {
            Verdict::Within => c.pose_innovation_judged += 1,
            Verdict::Over => {
                c.pose_innovation_judged += 1;
                c.pose_innovation_over += 1;
            }
            Verdict::WarmingUp | Verdict::Unavailable => c.pose_innovation_unavailable += 1,
        }
        c.pose_innovation_rejects += u64::from(refused);
        if let Some(scale) = scale {
            c.pose_innovation_scale_milli_max = c
                .pose_innovation_scale_milli_max
                .max((scale * 1e3).round() as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::pose_gate::{CHI2_6_999, ConsistencyScale};
    use kornia_algebra::{Mat15F64, Vec3F64};
    use kornia_factors::inertial::NavState;
    use kornia_factors::inertial::preintegration::propagate_state_covariance;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// A camera at `centre` looking down world +z.
    fn at(centre: Vec3F64) -> Pose3d {
        Pose3d::new(kornia_algebra::Mat3F64::IDENTITY, -centre)
    }

    /// `n` landmarks 3-6 m ahead of `pose`, seen exactly from it.
    fn rows(pose: &Pose3d, n: usize) -> Vec<PoseInertialObservation> {
        let camera = camera();
        let centre = camera_centre(pose);
        (0..n)
            .map(|k| {
                let point_w = centre
                    + Vec3F64::new(
                        (k % 5) as f64 * 0.8 - 1.6,
                        (k / 5 % 4) as f64 * 0.7 - 1.0,
                        3.0 + (k % 3) as f64 * 1.5,
                    );
                let p = pose.transform_point(&point_w);
                PoseInertialObservation {
                    point_w,
                    pixel: kornia_algebra::Vec2F64::new(
                        camera.fx * p.x / p.z + camera.cx,
                        camera.fy * p.y / p.z + camera.cy,
                    ),
                    information: 1.0,
                    view: 0,
                }
            })
            .collect()
    }

    fn information_at(pose: &Pose3d, n: usize) -> SMatF64<6, 6> {
        visual_information(&rows(pose, n), &camera(), &[], pose, 1e9).0
    }

    fn exp(prediction: &Pose3d, delta: &SVecF64<6>) -> Pose3d {
        let upsilon = Vec3F64::new(delta[0], delta[1], delta[2]);
        let omega = Vec3F64::new(delta[3], delta[4], delta[5]);
        Pose3d::new(
            prediction.rotation * SO3F64::exp(omega).matrix(),
            prediction.translation + prediction.rotation * (SO3F64::left_jacobian(omega) * upsilon),
        )
    }

    /// A tracking-time pose covariance: 2 cm centre, 0.2 deg rotation.
    fn tracking_covariance() -> SMatF64<6, 6> {
        let mut p = SMatF64::<6, 6>::zeros();
        for d in 0..3 {
            p[(d, d)] = 0.02f64.powi(2);
            p[(d + 3, d + 3)] = 0.2f64.to_radians().powi(2);
        }
        p
    }

    /// A deterministic standard normal (xorshift and Box-Muller).
    struct Normal(u64);
    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn sample(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    #[test]
    fn the_nis_of_samples_drawn_from_s_has_mean_six() {
        let prediction = at(Vec3F64::new(1.0, -2.0, 0.5));
        let p_pred = tracking_covariance();
        let information = information_at(&prediction, 20);
        let s = p_pred + information.cholesky().unwrap().inverse();
        let l = s.cholesky().unwrap().l();
        let mut normal = Normal(0x9e37_79b9_7f4a_7c15);
        let n = 4000;
        let mut sum = 0.0;
        for _ in 0..n {
            let z = SVecF64::<6>::from_fn(|_, _| normal.sample());
            let pose = exp(&prediction, &(l * z));
            sum += innovation(&prediction, &pose, &p_pred, &information)
                .unwrap()
                .nis6;
        }
        let mean = sum / n as f64;
        assert!((mean - 6.0).abs() < 0.3, "mean NIS {mean}");
    }

    /// The same frame with the world moved 600 m (3_18 runs 380-610 m out):
    /// `δ`, `P_pred` and `P_vis` move by the same adjoint, so the NIS stays.
    #[test]
    fn the_nis_is_unchanged_when_the_world_is_shifted_600_m() {
        let shift = Vec3F64::new(400.0, -400.0, 200.0);
        assert!((shift.length() - 600.0).abs() < 1.0);
        let base = at(Vec3F64::new(0.3, 0.1, -0.2));
        let delta = SVecF64::<6>::new(0.03, -0.02, 0.05, 0.004, -0.003, 0.002);
        let pose = exp(&base, &delta);
        let p_pred = tracking_covariance();
        let near = innovation(&base, &pose, &p_pred, &information_at(&base, 20)).unwrap();

        // Exp(δ') = T(s) Exp(δ) T(-s): Ad = [[I, [s]×], [0, I]] on (υ, ω).
        let shifted = |p: &Pose3d| Pose3d::new(p.rotation, p.translation - p.rotation * shift);
        let mut ad = SMatF64::<6, 6>::identity();
        ad.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SMatF64::<3, 3>::new(
                0.0, -shift.z, shift.y, shift.z, 0.0, -shift.x, -shift.y, shift.x, 0.0,
            ));
        let far_prediction = shifted(&base);
        let far = innovation(
            &far_prediction,
            &shifted(&pose),
            &(ad * p_pred * ad.transpose()),
            &information_at(&far_prediction, 20),
        )
        .unwrap();
        for (a, b) in [
            (near.nis6, far.nis6),
            (near.nis_rot, far.nis_rot),
            (near.nis_trans, far.nis_trans),
        ] {
            assert!((a - b).abs() < 1e-4 * a.max(1.0), "{a} against {b}");
        }
    }

    /// 3 s at rest under gravity from a VI-BA-like marginal: 0.2 m/s
    /// velocity, 1 deg tilt, 0.05 m/s² accel bias.
    fn coast_covariance(seconds: f64) -> SMatF64<6, 6> {
        let mut marginal = Mat15F64::zeros();
        for d in 0..3 {
            marginal[(d, d)] = 0.02f64.powi(2);
            marginal[(d + 3, d + 3)] = 1f64.to_radians().powi(2);
            marginal[(d + 6, d + 6)] = 0.2f64.powi(2);
            marginal[(d + 9, d + 9)] = 1e-3f64.powi(2);
            marginal[(d + 12, d + 12)] = 0.05f64.powi(2);
        }
        let calib = ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3e-3,
        };
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let mut preintegrated = PreintegratedImu::new(ImuBias::default(), calib);
        let dt = 0.005;
        for _ in 0..(seconds / dt).round() as usize {
            preintegrated.integrate(
                &ImuMeasurement {
                    timestamp: 0.0,
                    gyro: Vec3F64::ZERO,
                    accel: -gravity,
                },
                dt,
            );
        }
        let pose = Pose3d::IDENTITY;
        let state = NavState {
            pose_cw: &pose,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
        };
        let propagated = propagate_state_covariance(
            state,
            &marginal,
            NavState {
                pose_cw: &pose,
                velocity: Vec3F64::ZERO,
                bias: ImuBias::default(),
            },
            &preintegrated,
            &gravity,
            None,
        )
        .unwrap();
        propagated.fixed_view::<6, 6>(0, 0).into_owned()
    }

    #[test]
    fn reject_refuses_a_1_m_step_at_20_inliers_and_takes_it_after_a_3_s_coast() {
        let prediction = Pose3d::IDENTITY;
        let pose = exp(
            &prediction,
            &SVecF64::<6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        );
        let information = information_at(&pose, 20);
        let tracking =
            innovation(&prediction, &pose, &tracking_covariance(), &information).unwrap();
        assert!(InnovationGate::is_over(tracking.nis6, 1.0), "{tracking:?}");

        let coasted = innovation(&prediction, &pose, &coast_covariance(3.0), &information).unwrap();
        assert!(
            !InnovationGate::is_over(coasted.nis6, 1.0),
            "{coasted:?} against {CHI2_6_999}"
        );
        // The covariance grows through the coast.
        let short = coast_covariance(0.05);
        let long = coast_covariance(3.0);
        assert!(long[(0, 0)] > 100.0 * short[(0, 0)]);
    }

    #[test]
    fn the_scale_waits_for_50_frames_then_follows_the_median_above_one() {
        let mut scale = ConsistencyScale::default();
        for _ in 0..49 {
            scale.push(5.348 * 3.0);
        }
        assert_eq!(scale.scale(), None);
        scale.push(5.348 * 3.0);
        assert!((scale.scale().unwrap() - 3.0).abs() < 1e-9);
        // The window keeps 200 frames: 200 small ones replace the large.
        for _ in 0..200 {
            scale.push(1.0);
        }
        assert_eq!(scale.scale(), Some(1.0));
        scale.push(f64::NAN);
        assert_eq!(scale.scale(), Some(1.0));
    }

    #[test]
    fn observe_never_refuses_and_reject_refuses_only_past_the_bound() {
        for verdict in [
            Verdict::Within,
            Verdict::Over,
            Verdict::WarmingUp,
            Verdict::Unavailable,
        ] {
            assert!(!verdict.refused_under(InnovationMode::Observe));
            assert_eq!(
                verdict.refused_under(InnovationMode::Reject),
                verdict == Verdict::Over
            );
        }
    }

    #[test]
    fn a_non_finite_or_singular_input_is_unavailable() {
        let prediction = Pose3d::IDENTITY;
        assert_eq!(
            innovation(
                &prediction,
                &prediction,
                &tracking_covariance(),
                &SMatF64::<6, 6>::zeros()
            ),
            None
        );
        assert!(InnovationGate::is_over(f64::NAN, 1.0));
    }
}
