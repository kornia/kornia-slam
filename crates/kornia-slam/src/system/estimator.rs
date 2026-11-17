//! The per-frame estimator choices of card A, ported from ORB-SLAM3 and
//! OpenVINS: keyframe evidence for ORB-SLAM3's insertion rules and the IMU
//! covariance trust. Each one is behind its knob, and the defaults read none
//! of this.

use std::collections::HashSet;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, Vec3F64};
use kornia_factors::inertial::NavState;
use kornia_factors::inertial::preintegration::propagate_state_covariance;
use kornia_sensors::imu::ImuBias;

use crate::Frame;
use crate::inertial::trust::{ImuTrust, ImuTrustMode};
use crate::mapping::Keyframe;
use crate::tracking::{KeyframeEvidence, KeyframeRules};

use super::SlamSystem;

/// Lane A's state carried across frames.
#[derive(Debug, Default)]
pub(super) struct EstimatorState {
    /// Frame index of the last relocalization.
    pub(super) last_relocalization_frame: Option<usize>,
    /// The newest keyframe's id and marginal covariance from VI-BA.
    pub(super) newest_marginal: Option<(usize, Mat15F64)>,
    /// The covariance trust of the frame at the timestamp it was taken for.
    pub(super) trust: Option<(f64, ImuTrust)>,
    pub(super) trusted_before: bool,
}

/// A keyframe's inertial state, copied out of the map.
struct KeyframeNav {
    pose_cw: Pose3d,
    velocity: Vec3F64,
    bias: ImuBias,
}

impl KeyframeNav {
    fn of(kf: &Keyframe) -> Self {
        Self {
            pose_cw: kf.frame.pose_world_to_cam,
            velocity: kf.velocity_world,
            bias: kf.imu_bias,
        }
    }

    fn nav_state(&self) -> NavState<'_> {
        NavState {
            pose_cw: &self.pose_cw,
            velocity: self.velocity,
            bias: self.bias,
        }
    }
}

impl SlamSystem {
    /// `evidence` with what ORB-SLAM3's keyframe rules read; unchanged under
    /// the default rules, which ignore it.
    pub(super) fn with_orb_slam3_evidence(
        &self,
        evidence: KeyframeEvidence,
        frame: &Frame,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> KeyframeEvidence {
        if self.keyframe_policy.rules == KeyframeRules::Kornia {
            return evidence;
        }
        let ref_points_min_obs = {
            let map = self.map.lock().unwrap();
            // `Tracking.cc:3097-3100`.
            let min_observers = if map.keyframes().len() <= 2 { 2 } else { 3 };
            self.state
                .current_keyframe_idx
                .and_then(|ki| map.get_keyframe(ki))
                .map_or(0, |kf| {
                    kf.map_point_by_desc_idx
                        .iter()
                        .flatten()
                        .filter(|&&mp_idx| {
                            map.map_points().get(mp_idx).is_some_and(|mp| {
                                !mp.culled && mp.observer_keyframes().count() >= min_observers
                            })
                        })
                        .count()
                })
        };
        let (tracked_close, untracked_close) = self.close_point_counts(frame, matches);
        KeyframeEvidence {
            ref_points_min_obs,
            tracked_close,
            untracked_close,
            recently_lost: self.state.lost_since_sec.is_some(),
            // A visual-only run has no phase before initialization.
            imu_initialized: self.state.imu_initialized || self.rig.imu.is_none(),
            mono_inertial: self.rig.imu.is_some() && !frame.is_metric(),
            secs_since_last_kf: self
                .inertial
                .last_keyframe_timestamp_sec
                .map(|t| timestamp_sec - t),
            frames_since_reloc: self
                .estimator
                .last_relocalization_frame
                .map(|reloc| frame.idx.saturating_sub(reloc)),
            ..evidence
        }
    }

    /// Primary-camera keypoints with a stereo depth closer than the close
    /// depth (ORB-SLAM3's `mThDepth`), tracked and not (`Tracking.cc:3106-3127`).
    fn close_point_counts(&self, frame: &Frame, matches: &[(usize, usize)]) -> (usize, usize) {
        let Some(close_depth) = self.stereo_close_depth else {
            return (0, 0);
        };
        let tracked: HashSet<usize> = matches.iter().map(|&(_, kp_idx)| kp_idx).collect();
        frame
            .camera_range(0)
            .filter(|&i| {
                frame
                    .stereo_depth(i)
                    .is_some_and(|z| f64::from(z) < close_depth)
            })
            .fold((0, 0), |(tracked_close, untracked_close), i| {
                if tracked.contains(&i) {
                    (tracked_close + 1, untracked_close)
                } else {
                    (tracked_close, untracked_close + 1)
                }
            })
    }

    /// Under [`ImuTrustMode::Covariance`], decides this frame's trust from its
    /// predicted pose and caches it; the search scale goes to the tracker.
    pub(super) fn update_imu_trust(
        &mut self,
        frame_idx: usize,
        timestamp_sec: f64,
        predicted_pose: &Pose3d,
    ) {
        let ImuTrustMode::Covariance { search_sigmas } = self.tracking_config.imu_trust else {
            return;
        };
        let trust = self.covariance_trust(timestamp_sec, predicted_pose, search_sigmas);
        if self.estimator.trusted_before && !trust.trusted {
            self.counters.imu_trust_lost += 1;
        }
        self.estimator.trusted_before = trust.trusted;
        self.estimator.trust = Some((timestamp_sec, trust));
        self.tracker.set_search_scale_override(trust.search_scale);
        if self.debug {
            let cov6 = trust.covariance.map_or_else(String::new, |covariance| {
                (0..36)
                    .map(|k| format!("{:.3e}", covariance[(k / 6, k % 6)]))
                    .collect::<Vec<_>>()
                    .join(",")
            });
            self.dbg(format!(
                "[trust] frame={frame_idx} t={timestamp_sec:.6} trusted={} sigma_px={:.2} \
                 scale={:.2} cov6=[{cov6}]",
                u8::from(trust.trusted),
                trust.pixel_sigma.unwrap_or(f64::NAN),
                trust.search_scale.unwrap_or(f32::NAN),
            ));
        }
    }

    /// The covariance rule at `timestamp_sec`: the cached verdict of this
    /// frame, or one taken at the current pose.
    pub(super) fn cached_covariance_trust(
        &self,
        timestamp_sec: f64,
        search_sigmas: f64,
    ) -> ImuTrust {
        match self.estimator.trust {
            Some((t, trust)) if t == timestamp_sec => trust,
            _ => self.covariance_trust(timestamp_sec, &self.state.pose_world_to_cam, search_sigmas),
        }
    }

    /// Trusted while `search_sigmas` predicted pixel sigmas fit inside the
    /// widest projection search; the search scale follows the spread. Without
    /// a marginal yet, the time rule decides.
    fn covariance_trust(
        &self,
        timestamp_sec: f64,
        predicted_pose: &Pose3d,
        search_sigmas: f64,
    ) -> ImuTrust {
        let by_time = ImuTrust::by_time(self.trusted_by_time(timestamp_sec));
        if !self.state.imu_initialized {
            return by_time;
        }
        let Some(covariance) = self.predicted_covariance(timestamp_sec, predicted_pose) else {
            return by_time;
        };
        let Some(pixel_sigma) = self.pixel_sigma(&covariance) else {
            return by_time;
        };
        let config = self.tracker.map_projection_config();
        let radius = f64::from(config.projection.search_radius);
        let lost_scale = f64::from(config.lost_search_scale);
        let spread = search_sigmas * pixel_sigma;
        let trusted = spread <= lost_scale * radius;
        ImuTrust {
            trusted,
            search_scale: trusted.then(|| (spread / radius).max(1.0).min(lost_scale) as f32),
            pixel_sigma: Some(pixel_sigma),
            covariance: Some(covariance),
        }
    }

    /// The newest marginal propagated along the keyframe IMU chain to the last
    /// keyframe, then through this frame's window to `predicted_pose`.
    fn predicted_covariance(
        &self,
        timestamp_sec: f64,
        predicted_pose: &Pose3d,
    ) -> Option<Mat15F64> {
        let (marginal_kf, marginal) = self.estimator.newest_marginal?;
        let last_kf = self.state.last_keyframe_idx?;
        let last_kf_sec = self.inertial.last_keyframe_timestamp_sec?;
        let gravity = self.inertial.gravity_world;
        let t_bc = self.rig.camera_to_body();
        let (mut covariance, last) = {
            let map = self.map.lock().unwrap();
            let mut covariance = marginal;
            let mut current = KeyframeNav::of(map.get_keyframe(marginal_kf)?);
            let mut current_idx = marginal_kf;
            for _ in 0..map.keyframes().len() {
                if current_idx == last_kf {
                    break;
                }
                let edge = map
                    .imu_factors()
                    .iter()
                    .find(|edge| edge.prev_kf_idx == current_idx)?;
                let next = KeyframeNav::of(map.get_keyframe(edge.curr_kf_idx)?);
                covariance = propagate_state_covariance(
                    current.nav_state(),
                    &covariance,
                    next.nav_state(),
                    &edge.preintegrated,
                    &gravity,
                    t_bc.as_ref(),
                )?;
                (current, current_idx) = (next, edge.curr_kf_idx);
            }
            (current_idx == last_kf).then_some((covariance, current))?
        };
        let (preintegrated, _) =
            self.inertial
                .preintegrate_window(self.rig.imu_noise(), last_kf_sec, timestamp_sec);
        if preintegrated.dt > 0.0 {
            let frame = NavState {
                pose_cw: predicted_pose,
                velocity: self.state.velocity_world,
                bias: self.inertial.bias,
            };
            covariance = propagate_state_covariance(
                last.nav_state(),
                &covariance,
                frame,
                &preintegrated,
                &gravity,
                t_bc.as_ref(),
            )?;
        }
        Some(covariance)
    }

    /// One-sigma reprojection spread, px: `f · √(σ_θ² + (σ_p / z)²)` from the
    /// largest rotation and translation variances and the reference
    /// keyframe's median landmark depth `z`.
    fn pixel_sigma(&self, covariance: &Mat15F64) -> Option<f64> {
        let largest = |base: usize| {
            covariance
                .fixed_view::<3, 3>(base, base)
                .into_owned()
                .symmetric_eigenvalues()
                .max()
                .max(0.0)
        };
        let depth = self.reference_median_depth()?;
        let spread = self.rig.camera.fx * (largest(3) + largest(0) / (depth * depth)).sqrt();
        spread.is_finite().then_some(spread)
    }

    /// Median camera-frame depth of the reference keyframe's landmarks.
    fn reference_median_depth(&self) -> Option<f64> {
        let map = self.map.lock().unwrap();
        let kf = map.get_keyframe(self.state.current_keyframe_idx?)?;
        let mut depths: Vec<f64> = kf
            .map_point_by_desc_idx
            .iter()
            .flatten()
            .filter_map(|&mp_idx| map.map_points().get(mp_idx))
            .filter(|mp| !mp.culled)
            .map(|mp| kf.frame.pose_world_to_cam.transform_point(&mp.position).z)
            .filter(|z| z.is_finite() && *z > 0.0)
            .collect();
        if depths.is_empty() {
            return None;
        }
        let mid = depths.len() / 2;
        let (_, median, _) = depths.select_nth_unstable_by(mid, f64::total_cmp);
        Some(*median)
    }
}
