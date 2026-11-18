//! The per-frame estimator choices of card A, ported from ORB-SLAM3 and
//! OpenVINS: keyframe evidence for ORB-SLAM3's insertion rules, the IMU
//! covariance trust, the pose-inertial solve and the G29 snap-back counter
//! and fix. Each one is behind its knob, and the defaults read none of this;
//! the G29 counter only counts.

use std::collections::HashSet;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat15F64, Vec2F64, Vec3F64};
use kornia_factors::inertial::NavState;
use kornia_factors::inertial::preintegration::propagate_state_covariance;
use kornia_sensors::imu::ImuBias;

use crate::Frame;
use crate::inertial::trust::{ImuTrust, ImuTrustMode};
use crate::mapping::Keyframe;
use crate::mapping::map::ORB_SCALE_FACTOR;
use crate::mapping::vi_ba_schur::RigObservationCamera;
use crate::tracking::pose_estimation::Estimate;
use crate::tracking::pose_estimation::pose_inertial::{
    Anchor, FrameNavState, NavPrior, PoseInertialObservation, solve_pose_inertial,
};
use crate::tracking::{KeyframeEvidence, KeyframeRules};
use crate::upstream::kornia_3d::pose_ext::camera_centre;

use super::{PoseInertial, SlamSystem, SnapbackFix};

/// A BA write-back moving the newest keyframe further than this counts as a
/// shift (G29).
const BA_SHIFT_M: f64 = 0.3;
/// The next tracked step undoes a shift when it moves back this many times
/// the shift along it.
const SNAPBACK_RATIO: std::ops::RangeInclusive<f64> = 0.5..=2.0;

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
    /// The last frame the pose-inertial solve tracked: its index, the map's
    /// world epoch then, and its prior for the next frame.
    pub(super) frame_prior: Option<(usize, u64, NavPrior)>,
    /// A local-BA write-back changed the map since the last solve
    /// (ORB-SLAM3's `mbMapUpdated`).
    pub(super) map_changed: bool,
    /// The newest keyframe's centre shift of the last write-back over
    /// [`BA_SHIFT_M`], until the next tracked frame measures its step.
    pub(super) pending_shift: Option<(usize, Vec3F64)>,
    /// `snapback_fix`: the next tracked frame is solved from the corrected
    /// keyframe.
    pub(super) anchor_next_frame: bool,
}

/// What the pose-inertial solve of a frame starts from.
enum SolveAnchor {
    /// The last keyframe, held fixed, from its timestamp.
    Keyframe(FrameNavState, f64),
    /// The previous frame under its prior; boxed, as it is ten times the
    /// keyframe variant.
    Frame(Box<NavPrior>),
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

    /// Under `tracking.pose_inertial`, refines a tracked `estimate` with the
    /// pose-inertial solve and takes the frame's velocity and bias from it.
    /// The visual estimate stays when the solve keeps too few inliers.
    pub(super) fn refine_pose_inertial(
        &mut self,
        frame: &Frame,
        estimate: &mut Estimate,
        prev_timestamp: f64,
        timestamp_sec: f64,
    ) {
        let mode = self.pose_inertial_mode();
        self.estimator.anchor_next_frame = false;
        if mode == PoseInertial::Off || !self.state.imu_initialized {
            return;
        }
        let Some(anchor) = self.solve_anchor(mode, frame.idx) else {
            self.estimator.frame_prior = None;
            return;
        };
        let (start_sec, keyframe_anchored) = match &anchor {
            SolveAnchor::Keyframe(_, kf_sec) => (*kf_sec, true),
            SolveAnchor::Frame(_) => (prev_timestamp, false),
        };
        let (preintegrated, _) =
            self.inertial
                .preintegrate_window(self.rig.imu_noise(), start_sec, timestamp_sec);
        if preintegrated.dt <= 0.0 {
            self.estimator.frame_prior = None;
            return;
        }
        let observations = self.pose_inertial_observations(frame, &estimate.matches);
        let initial = FrameNavState {
            pose_cw: estimate.pose,
            velocity: self.state.velocity_world,
            bias: self.inertial.bias,
        };
        let solution = solve_pose_inertial(
            match &anchor {
                SolveAnchor::Keyframe(state, _) => Anchor::Fixed(state),
                SolveAnchor::Frame(prior) => Anchor::Prior(prior),
            },
            &preintegrated,
            self.inertial.gravity_world,
            self.rig.camera_to_body(),
            &rig_observation_cameras(frame),
            &self.rig.camera,
            initial,
            &observations,
            !frame.is_metric(),
        );
        let inliers = solution.inliers.iter().filter(|&&inlier| inlier).count();
        let min_inliers = self.tracker.map_projection_config().pnp.min_inliers;
        self.dbg(format!(
            "[pose_inertial] frame={} anchor={} inliers={inliers} imu_chi2={:.3} prior_pd={}",
            frame.idx,
            if keyframe_anchored { "kf" } else { "frame" },
            solution.imu_chi2,
            u8::from(solution.prior.is_some()),
        ));
        if inliers < min_inliers {
            self.estimator.frame_prior = None;
            return;
        }

        self.counters.pose_inertial_solves += 1;
        if keyframe_anchored {
            self.counters.pose_inertial_keyframe_anchored += 1;
        }
        if solution.prior.is_none() {
            self.counters.pose_inertial_prior_dropped += 1;
        }
        estimate.pose = solution.state.pose_cw;
        estimate.inliers = inliers;
        self.state.velocity_world = solution.state.velocity;
        self.inertial.bias = solution.state.bias;
        let epoch = self.map.lock().unwrap().world_epoch();
        self.estimator.frame_prior = solution.prior.map(|prior| (frame.idx, epoch, prior));
        self.estimator.map_changed = false;
    }

    /// Whether the previous frame's solve carries velocity and bias into this
    /// frame, so the reference keyframe's are not copied (G13).
    pub(super) fn frame_state_carried(&self, frame_idx: usize) -> bool {
        self.tracking_config.pose_inertial != PoseInertial::Off
            && self.valid_frame_prior(frame_idx).is_some()
    }

    /// Whether this frame's solve and prediction start from the last
    /// keyframe: always with `LastKeyframe`; with `OrbSlam3` after the map
    /// changed or without the previous frame's prior (`Tracking.cc:2970-2992`).
    pub(super) fn keyframe_anchored(&self, frame_idx: usize) -> bool {
        match self.pose_inertial_mode() {
            PoseInertial::Off => false,
            PoseInertial::LastKeyframe => true,
            PoseInertial::OrbSlam3 => self.valid_frame_prior(frame_idx).is_none(),
        }
    }

    /// The previous frame's prior, when it is that frame's and the map has
    /// not changed since.
    fn valid_frame_prior(&self, frame_idx: usize) -> Option<NavPrior> {
        let (prior_frame, epoch, prior) = self.estimator.frame_prior?;
        (!self.estimator.map_changed
            && prior_frame + 1 == frame_idx
            && epoch == self.map.lock().unwrap().world_epoch())
        .then_some(prior)
    }

    fn solve_anchor(&self, mode: PoseInertial, frame_idx: usize) -> Option<SolveAnchor> {
        if mode == PoseInertial::OrbSlam3
            && let Some(prior) = self.valid_frame_prior(frame_idx)
        {
            return Some(SolveAnchor::Frame(Box::new(prior)));
        }
        let kf_sec = self.inertial.last_keyframe_timestamp_sec?;
        let map = self.map.lock().unwrap();
        let kf = map.get_keyframe(self.state.last_keyframe_idx?)?;
        Some(SolveAnchor::Keyframe(
            FrameNavState {
                pose_cw: kf.frame.pose_world_to_cam,
                velocity: kf.velocity_world,
                bias: kf.imu_bias,
            },
            kf_sec,
        ))
    }

    /// Where this frame's motion prediction starts: the last keyframe after
    /// the map changed (`Tracking.cc:1746-1761`), otherwise the last frame.
    /// From the keyframe, the window re-integrates from its velocity and
    /// bias, which the frame takes (E1, E7).
    pub(super) fn prediction_origin(
        &mut self,
        frame_idx: usize,
        pose_before: Pose3d,
        prev_timestamp: f64,
    ) -> (Pose3d, f64) {
        let from_keyframe = match self.tracking_config.pose_inertial {
            PoseInertial::OrbSlam3 => self.keyframe_anchored(frame_idx),
            PoseInertial::LastKeyframe => false,
            PoseInertial::Off => self.pose_inertial_mode() != PoseInertial::Off,
        };
        if !self.state.imu_initialized || !from_keyframe {
            return (pose_before, prev_timestamp);
        }
        let origin = self
            .inertial
            .last_keyframe_timestamp_sec
            .and_then(|kf_sec| {
                let map = self.map.lock().unwrap();
                let kf = map.get_keyframe(self.state.last_keyframe_idx?)?;
                Some((KeyframeNav::of(kf), kf_sec))
            });
        let Some((kf, kf_sec)) = origin else {
            return (pose_before, prev_timestamp);
        };
        self.state.velocity_world = kf.velocity;
        self.inertial.bias = kf.bias;
        (kf.pose_cw, kf_sec)
    }

    /// The tracked matches as reprojection rows: undistorted pixels weighted
    /// by their octave, `1/σ²` with `σ = 1.2^octave` (ORB-SLAM3's
    /// `invSigma2`).
    fn pose_inertial_observations(
        &self,
        frame: &Frame,
        matches: &[(usize, usize)],
    ) -> Vec<PoseInertialObservation> {
        let map = self.map.lock().unwrap();
        matches
            .iter()
            .filter_map(|&(mp_idx, kp_idx)| {
                let mp = map.map_points().get(mp_idx).filter(|mp| !mp.culled)?;
                let [x, y] = frame.undistorted_xy(kp_idx, &self.rig.camera)?;
                let octave = frame.features.octaves.get(kp_idx).copied().unwrap_or(0);
                Some(PoseInertialObservation {
                    point_w: mp.position,
                    pixel: Vec2F64::new(f64::from(x), f64::from(y)),
                    information: ORB_SCALE_FACTOR.powi(-2 * i32::from(octave)),
                    view: u8::try_from(frame.camera_of(kp_idx)).ok()?,
                })
            })
            .collect()
    }

    /// The solve this frame runs: the configured one, or with `snapback_fix`
    /// a `LastKeyframe` solve on the first tracked frame after a write-back
    /// (`Tracking.cc:2982-2991` without the rest of `OrbSlam3`).
    fn pose_inertial_mode(&self) -> PoseInertial {
        match self.tracking_config.pose_inertial {
            PoseInertial::Off
                if self.tracking_config.snapback_fix == SnapbackFix::AnchorNextFrame
                    && self.estimator.anchor_next_frame =>
            {
                PoseInertial::LastKeyframe
            }
            mode => mode,
        }
    }

    /// G29 at a BA write-back: the shift of the newest keyframe's centre,
    /// kept for the next tracked frame when over [`BA_SHIFT_M`].
    pub(super) fn note_ba_shift(
        &mut self,
        kf_idx: usize,
        pose_before: &Pose3d,
        pose_after: &Pose3d,
    ) {
        if self.tracking_config.snapback_fix == SnapbackFix::AnchorNextFrame {
            self.estimator.anchor_next_frame = true;
        }
        let shift = camera_centre(pose_after) - camera_centre(pose_before);
        if shift.length() > BA_SHIFT_M {
            self.counters.ba_keyframe_shifts += 1;
            self.estimator.pending_shift = Some((kf_idx, shift));
        }
    }

    /// G29 on a tracked frame: how much of the pending shift its step from
    /// the `predicted` pose to the `accepted` one undid.
    pub(super) fn note_tracked_step(&mut self, predicted: &Pose3d, accepted: &Pose3d) {
        let Some((kf_idx, shift)) = self.estimator.pending_shift.take() else {
            return;
        };
        let step = camera_centre(accepted) - camera_centre(predicted);
        let ratio = -step.dot(shift) / shift.dot(shift);
        if SNAPBACK_RATIO.contains(&ratio) {
            self.counters.ba_snapbacks += 1;
        }
        self.dbg(format!(
            "[g29] kf={kf_idx} shift={:.3} next_step={:.3} ratio={ratio:.2}",
            shift.length(),
            step.length(),
        ));
    }
}

/// The secondary cameras of `frame`'s rig, in rig order after the primary;
/// empty outside a rig.
fn rig_observation_cameras(frame: &Frame) -> Vec<RigObservationCamera> {
    frame
        .rig
        .geometry
        .as_ref()
        .filter(|_| frame.is_rig())
        .map(|rig| {
            rig.views()
                .iter()
                .skip(1)
                .map(|view| RigObservationCamera {
                    camera: view.camera.clone(),
                    camera_from_primary: view.camera_from_primary,
                })
                .collect()
        })
        .unwrap_or_default()
}
