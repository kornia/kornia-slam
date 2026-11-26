//! Policies controlling keyframe insertion and short tracking-loss recovery.

/// Keyframe insertion heuristics.
///
/// The defaults are the frame-count policy: a keyframe at least every
/// `max_frames_between` frames. Setting `max_frames_between` to `usize::MAX`
/// and `min_parallax_px` to a threshold gives ORB-SLAM3 / VINS-Mono style
/// insertion instead, where a keyframe is earned by motion (parallax) or by
/// losing the reference keyframe's landmarks, and a parked camera adds none.
/// With inertial input that mode also needs `max_interval_sec`, as ORB-SLAM3
/// forces a keyframe every 0.5 s in its inertial modes: IMU samples are
/// buffered until the next keyframe, so without a time bound a parked rig
/// buffers (and rescans) every sample of the stop and then bridges it with
/// one IMU edge. [`SlamSystem::build`](crate::system::SlamSystem::build)
/// refuses an inertial system with neither bound.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct KeyframePolicy {
    /// Minimum frame gap before allowing keyframe insertion.
    pub min_frames_between: usize,
    /// Force a keyframe if this frame gap is reached; `usize::MAX` disables
    /// the bound.
    pub max_frames_between: usize,
    /// Relative inlier ratio threshold (vs reference keyframe tracked map points).
    pub ref_ratio: f64,
    /// Insert once the median rotation-compensated parallax since the last
    /// keyframe reaches this many pixels ([`KeyframeEvidence::median_parallax_px`]).
    /// `None` (the default) ignores parallax. VINS-Mono uses 10 px at a
    /// 460 px focal length.
    pub min_parallax_px: Option<f64>,
    /// Force a keyframe once this many seconds have passed since the last
    /// one (outside `min_frames_between`). `None` (the default) sets no time
    /// bound. `SlamSystem` applies it from the frame timestamps;
    /// [`Self::should_insert_with`] does not see time.
    pub max_interval_sec: Option<f64>,
    /// The rule set the decision follows.
    pub rules: KeyframeRules,
    /// What a still device does to the bound asks (card AB).
    pub stillness: Stillness,
}

/// How keyframes are handled while the device is still: no voter says
/// moving and the image, the wheels or a quiet IMU vouches for rest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Stillness {
    /// No detector.
    #[default]
    Off,
    /// The detector runs and counts; no decision changes.
    Observe,
    /// Bound asks (frame, time and lost-interval keyframes) are withheld
    /// while still, up to a heartbeat; quality asks still insert.
    Gate,
}

impl Stillness {
    /// Whether the detector runs.
    pub fn observes(self) -> bool {
        self != Self::Off
    }

    /// Whether still frames change keyframe decisions.
    pub fn gates(self) -> bool {
        self == Self::Gate
    }
}

/// Which keyframe insertion rules apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframeRules {
    /// The fields of [`KeyframePolicy`].
    #[default]
    Kornia,
    /// ORB-SLAM3's `Tracking::NeedNewKeyFrame`, inertial branches
    /// (`Tracking.cc:3064-3214`); the other [`KeyframePolicy`] fields are
    /// ignored. `max_frames` is ORB-SLAM3's `mMaxFrames`, the camera rate.
    OrbSlam3 { max_frames: usize },
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            min_frames_between: 3,
            max_frames_between: 8,
            ref_ratio: 0.6,
            min_parallax_px: None,
            max_interval_sec: None,
            rules: KeyframeRules::Kornia,
            stillness: Stillness::Off,
        }
    }
}

/// What the tracker measured on the current frame, for
/// [`KeyframePolicy::should_insert_with`]. `#[non_exhaustive]`: build it with
/// [`Self::new`] or from [`Default`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct KeyframeEvidence {
    /// `Frame::idx` of the current frame.
    pub curr_idx: usize,
    /// `Frame::idx` of the last inserted keyframe, `None` before the first.
    pub last_keyframe_idx: Option<usize>,
    /// Map points the current frame tracked as inliers.
    pub tracked_inliers: usize,
    /// Map points the reference keyframe holds.
    pub n_ref_map_points: usize,
    /// Median rotation-compensated parallax of the tracked features against
    /// the last keyframe, pixels (see [`super::median_parallax_px`]); `None`
    /// when it was not measured.
    pub median_parallax_px: Option<f64>,
    /// Reference-keyframe map points seen at least 3 times (2 while the map
    /// holds at most 2 keyframes): ORB-SLAM3's `nRefMatches`.
    pub ref_points_min_obs: usize,
    /// Close stereo points tracked, and close stereo points not tracked.
    pub tracked_close: usize,
    pub untracked_close: usize,
    /// The first tracked frame after a loss (ORB-SLAM3's `RECENTLY_LOST`).
    pub recently_lost: bool,
    pub imu_initialized: bool,
    pub mono_inertial: bool,
    pub secs_since_last_kf: Option<f64>,
    pub frames_since_reloc: Option<usize>,
}

impl KeyframeEvidence {
    /// Evidence without a parallax measurement; set `median_parallax_px`
    /// when it was measured.
    pub fn new(
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> Self {
        Self {
            curr_idx,
            last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
            ..Self::default()
        }
    }

    /// `self` with the measured median parallax (`None` when it was not
    /// measured), pixels.
    pub fn with_median_parallax_px(self, median_parallax_px: Option<f64>) -> Self {
        Self {
            median_parallax_px,
            ..self
        }
    }
}

/// Why the policy asks for a keyframe: a `bound` ask comes from elapsed
/// frames or time alone, a `quality` ask from what the tracker measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyframeAsks {
    pub bound: bool,
    pub quality: bool,
}

impl KeyframeAsks {
    pub fn any(self) -> bool {
        self.bound || self.quality
    }
}

/// Fewest tracked inliers for which a low reference ratio still earns a
/// keyframe (ORB-SLAM3's `mnMatchesInliers > 15`): below it the keyframe
/// would hold too little to be worth adding.
const MIN_INLIERS_FOR_RATIO_INSERT: usize = 15;

impl KeyframePolicy {
    /// Decide whether a new keyframe should be inserted.
    ///
    /// Equivalent to [`Self::should_insert_with`] without a parallax measurement.
    pub fn should_insert(
        &self,
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> bool {
        self.should_insert_with(&KeyframeEvidence::new(
            curr_idx,
            last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ))
    }

    /// Decide whether a new keyframe should be inserted.
    ///
    /// In order: always for the first keyframe; never inside
    /// `min_frames_between`; always at `max_frames_between`; always when the
    /// reference keyframe holds no map points (tracking is weak); when the
    /// tracked ratio against the reference falls below `ref_ratio`; and, if
    /// `min_parallax_px` is set, when the measured parallax reaches it.
    pub fn should_insert_with(&self, evidence: &KeyframeEvidence) -> bool {
        self.asks(evidence).any()
    }

    /// [`Self::should_insert_with`] split into its bound and quality asks.
    /// The first keyframe is a quality ask.
    pub fn asks(&self, evidence: &KeyframeEvidence) -> KeyframeAsks {
        if let KeyframeRules::OrbSlam3 { max_frames } = self.rules {
            return orb_slam3_asks(max_frames, evidence);
        }
        let Some(last_kf_idx) = evidence.last_keyframe_idx else {
            return KeyframeAsks {
                bound: false,
                quality: true,
            };
        };
        let frames_since_last_kf = evidence.curr_idx.saturating_sub(last_kf_idx);
        if frames_since_last_kf < self.min_frames_between {
            return KeyframeAsks::default();
        }
        KeyframeAsks {
            bound: frames_since_last_kf >= self.max_frames_between,
            quality: self.quality_due(evidence),
        }
    }

    /// Whether what the tracker measured earns a keyframe on its own, past
    /// `min_frames_between`: an empty reference, the reference ratio, or
    /// parallax.
    fn quality_due(&self, evidence: &KeyframeEvidence) -> bool {
        if evidence.n_ref_map_points == 0 {
            return true;
        }
        let weak_threshold = (evidence.n_ref_map_points as f64 * self.ref_ratio) as usize;
        if evidence.tracked_inliers >= MIN_INLIERS_FOR_RATIO_INSERT
            && evidence.tracked_inliers < weak_threshold
        {
            return true;
        }

        matches!(
            (self.min_parallax_px, evidence.median_parallax_px),
            (Some(threshold), Some(parallax)) if parallax >= threshold
        )
    }
}

impl KeyframePolicy {
    /// Whether `max_interval_sec` forces a keyframe `secs_since_last_kf`
    /// seconds and `frames_since_last_kf` frames after the last one.
    pub(crate) fn interval_elapsed(
        &self,
        frames_since_last_kf: usize,
        secs_since_last_kf: f64,
    ) -> bool {
        self.rules == KeyframeRules::Kornia
            && frames_since_last_kf >= self.min_frames_between
            && self
                .max_interval_sec
                .is_some_and(|max| secs_since_last_kf >= max)
    }
}

/// ORB-SLAM3's `Tracking::NeedNewKeyFrame`, inertial branches
/// (`Tracking.cc:3064-3214`), with mapping taken as idle: `mMinFrames` is 0,
/// so `c1b` always holds and `c1a` adds nothing. The rules that read the
/// map's keyframe count (`nKFs <= 2`, `< 2`) are left out: they act only on
/// the first two keyframes.
///
/// Bound asks: the 0.25 s rule before IMU initialization and `c3` (0.5 s);
/// quality asks: `c2` and `c4`.
fn orb_slam3_asks(max_frames: usize, evidence: &KeyframeEvidence) -> KeyframeAsks {
    if evidence.last_keyframe_idx.is_none() {
        return KeyframeAsks {
            bound: false,
            quality: true,
        };
    }
    let secs_since_last_kf = evidence.secs_since_last_kf;
    if !evidence.imu_initialized {
        return KeyframeAsks {
            bound: secs_since_last_kf.is_some_and(|secs| secs >= 0.25),
            quality: false,
        };
    }
    if evidence.frames_since_reloc.is_some_and(|n| n < max_frames) {
        return KeyframeAsks::default();
    }
    let inliers = evidence.tracked_inliers;
    let need_close = evidence.tracked_close < 100 && evidence.untracked_close > 70;
    let ref_ratio = if evidence.mono_inertial && inliers <= 350 {
        0.90
    } else {
        0.75
    };
    let c2 = ((inliers as f64) < evidence.ref_points_min_obs as f64 * ref_ratio || need_close)
        && inliers > MIN_INLIERS_FOR_RATIO_INSERT;
    let c3 = secs_since_last_kf.is_some_and(|secs| secs >= 0.5);
    let c4 = evidence.mono_inertial
        && ((inliers < 75 && inliers > MIN_INLIERS_FOR_RATIO_INSERT) || evidence.recently_lost);
    KeyframeAsks {
        bound: c3,
        quality: c2 || c4,
    }
}

/// Recently-lost grace period policy (mirrors ORB-SLAM3's RECENTLY_LOST vs
/// LOST distinction), bridging interruptions (fast rotation, motion blur,
/// dropped frames) without throwing the map away.
///
/// With a settled IMU, tracking coasts on the inertial prediction for up to
/// `timeout_imu_sec` (ORB-SLAM3's `time_recently_lost`), and keyframes
/// keep extending the map at the predicted pose every
/// `keyframe_interval_while_lost_sec`: after a fast turn the camera faces
/// space the map does not cover, and only new landmarks there let tracking
/// resume. Without a settled inertial state tracking gets only the short
/// visual grace period, and a map that's too young is reset at once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingLossRecoveryPolicy {
    /// A map with at most this many keyframes is reset as soon as tracking
    /// fails, with no grace period.
    pub min_keyframes_for_grace: usize,
    /// Grace period once the IMU has been initialized for at least
    /// `min_imu_confidence_sec`.
    pub timeout_imu_sec: f64,
    /// Grace period otherwise (no IMU, or too recently initialized).
    pub timeout_visual_sec: f64,
    /// How long the IMU must have been initialized before `timeout_imu_sec`
    /// applies instead of `timeout_visual_sec`.
    pub min_imu_confidence_sec: f64,
    /// Seconds between keyframes inserted at the IMU-predicted pose
    /// while coasting.
    pub keyframe_interval_while_lost_sec: f64,
}

/// Whether keyframes are inserted at the IMU-predicted pose while coasting,
/// and as what; `Disabled` is for attribution runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframesWhileLost {
    #[default]
    Enabled,
    Disabled,
    /// Inserted as anchors: held fixed in every BA, with no IMU edge, so a
    /// pose nothing observed never becomes an optimised state.
    Anchor,
}

impl Default for TrackingLossRecoveryPolicy {
    fn default() -> Self {
        Self {
            min_keyframes_for_grace: 10,
            timeout_imu_sec: 5.0,
            timeout_visual_sec: 0.5,
            min_imu_confidence_sec: 2.0,
            keyframe_interval_while_lost_sec: 0.2,
        }
    }
}

impl TrackingLossRecoveryPolicy {
    /// Whether a lost frame should become a keyframe, given the seconds since
    /// the last one.
    pub fn keyframe_due(&self, since_last_keyframe_sec: f64) -> bool {
        since_last_keyframe_sec >= self.keyframe_interval_while_lost_sec
    }

    /// Grace period, in seconds, to allow before giving up and resetting.
    pub fn grace_period_sec(&self, imu_confident: bool) -> f64 {
        if imu_confident {
            self.timeout_imu_sec
        } else {
            self.timeout_visual_sec
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(frames_since: usize, inliers: usize, n_ref: usize) -> KeyframeEvidence {
        KeyframeEvidence {
            curr_idx: 100 + frames_since,
            last_keyframe_idx: Some(100),
            tracked_inliers: inliers,
            n_ref_map_points: n_ref,
            ..KeyframeEvidence::default()
        }
    }

    fn with_parallax(e: KeyframeEvidence, px: f64) -> KeyframeEvidence {
        KeyframeEvidence {
            median_parallax_px: Some(px),
            ..e
        }
    }

    /// Opt-in motion policy: no frame-count bound, 10 px of parallax.
    fn parallax_policy() -> KeyframePolicy {
        KeyframePolicy {
            max_frames_between: usize::MAX,
            min_parallax_px: Some(10.0),
            ..KeyframePolicy::default()
        }
    }

    /// `Tracking.cc:3064-3214`: 0.25 s before IMU initialization; then the
    /// reference ratio (0.9 monocular below 350 inliers), 0.5 s, and the
    /// monocular weak-tracking and recently-lost rules; none right after a
    /// relocalization.
    #[test]
    fn orb_slam3_rules_follow_need_new_keyframe() {
        let policy = KeyframePolicy {
            rules: KeyframeRules::OrbSlam3 { max_frames: 20 },
            ..KeyframePolicy::default()
        };
        let base = KeyframeEvidence {
            imu_initialized: true,
            secs_since_last_kf: Some(0.1),
            ref_points_min_obs: 200,
            ..evidence(1, 180, 200)
        };
        let insert = |e: KeyframeEvidence| policy.should_insert_with(&e);

        assert!(!insert(base), "180 of 200 is above the 0.75 ratio");
        assert!(insert(KeyframeEvidence {
            tracked_inliers: 140,
            ..base
        }));
        assert!(insert(KeyframeEvidence {
            mono_inertial: true,
            tracked_inliers: 170,
            ..base
        }));
        assert!(!insert(KeyframeEvidence {
            mono_inertial: true,
            tracked_inliers: 190,
            ..base
        }));
        assert!(insert(KeyframeEvidence {
            secs_since_last_kf: Some(0.5),
            ..base
        }));
        assert!(insert(KeyframeEvidence {
            mono_inertial: true,
            recently_lost: true,
            tracked_inliers: 190,
            ..base
        }));
        assert!(insert(KeyframeEvidence {
            tracked_close: 50,
            untracked_close: 80,
            ..base
        }));
        assert!(!insert(KeyframeEvidence {
            tracked_inliers: 140,
            frames_since_reloc: Some(5),
            ..base
        }));
        assert!(!insert(KeyframeEvidence {
            imu_initialized: false,
            tracked_inliers: 10,
            ..base
        }));
        assert!(insert(KeyframeEvidence {
            imu_initialized: false,
            secs_since_last_kf: Some(0.25),
            ..base
        }));
        assert!(!policy.interval_elapsed(1000, 1000.0));
    }

    #[test]
    fn the_default_keeps_the_frame_count_and_ratio_decision_and_ignores_parallax() {
        let policy = KeyframePolicy::default();
        // (inliers, n_ref, first frame gap that inserts). The ratio rule
        // inserts from min_frames_between (3) when 15 <= inliers <
        // 0.6 * n_ref, an empty reference always does, anything else waits
        // for max_frames_between (8).
        let cases = [
            (80, 100, 8),
            (60, 100, 8), // exactly at the ratio: not below it
            (59, 100, 3),
            (50, 100, 3),
            (15, 30, 3), // fewest inliers the ratio rule keeps
            (14, 30, 8),
            (10, 100, 8),
            (50, 0, 3),
        ];
        for frames in 0..12 {
            for (inliers, n_ref, inserts_from) in cases {
                let e = evidence(frames, inliers, n_ref);
                let expected = frames >= inserts_from;
                let case = format!("frames={frames} inliers={inliers} n_ref={n_ref}");
                assert_eq!(policy.should_insert_with(&e), expected, "{case}");
                assert_eq!(policy.asks(&e).any(), expected, "{case}");
                assert_eq!(
                    policy.should_insert(e.curr_idx, e.last_keyframe_idx, inliers, n_ref),
                    expected,
                    "{case}"
                );
                assert_eq!(
                    policy.should_insert_with(&with_parallax(e, 1e6)),
                    expected,
                    "{case}: parallax is ignored without min_parallax_px"
                );
            }
        }
    }

    /// The frame bound and ORB-SLAM3's 0.25 s / `c3` rules are bound asks;
    /// the ratio, an empty reference, `c2` and the first keyframe are quality.
    #[test]
    fn asks_split_into_bound_and_quality() {
        let policy = KeyframePolicy::default();
        let ask = |bound, quality| KeyframeAsks { bound, quality };
        assert_eq!(policy.asks(&evidence(8, 80, 100)), ask(true, false));
        assert_eq!(policy.asks(&evidence(8, 50, 100)), ask(true, true));
        assert_eq!(policy.asks(&evidence(4, 50, 100)), ask(false, true));
        assert_eq!(policy.asks(&evidence(4, 50, 0)), ask(false, true));
        assert_eq!(policy.asks(&evidence(2, 50, 0)), ask(false, false));
        let first = KeyframeEvidence {
            last_keyframe_idx: None,
            ..evidence(0, 0, 0)
        };
        assert_eq!(policy.asks(&first), ask(false, true));

        let orb = KeyframePolicy {
            rules: KeyframeRules::OrbSlam3 { max_frames: 20 },
            ..KeyframePolicy::default()
        };
        let base = KeyframeEvidence {
            imu_initialized: true,
            secs_since_last_kf: Some(0.6),
            ref_points_min_obs: 200,
            ..evidence(1, 180, 200)
        };
        assert_eq!(orb.asks(&base), ask(true, false));
        assert_eq!(
            orb.asks(&KeyframeEvidence {
                tracked_inliers: 140,
                ..base
            }),
            ask(true, true)
        );
        assert_eq!(
            orb.asks(&KeyframeEvidence {
                imu_initialized: false,
                secs_since_last_kf: Some(0.3),
                ..base
            }),
            ask(true, false)
        );
        assert_eq!(orb.asks(&first), ask(false, true));
    }

    #[test]
    fn parallax_earns_a_keyframe_inside_the_bounds() {
        let policy = parallax_policy();
        let steady = evidence(5, 80, 100);
        assert!(!policy.should_insert_with(&steady), "unmeasured");
        assert!(!policy.should_insert_with(&with_parallax(steady, 9.9)));
        assert!(policy.should_insert_with(&with_parallax(steady, 10.0)));
        assert!(
            !policy.should_insert_with(&with_parallax(evidence(2, 80, 100), 50.0)),
            "min_frames_between still applies"
        );
    }

    #[test]
    fn a_parked_camera_adds_no_keyframes_without_the_frame_bound() {
        let policy = parallax_policy();
        for frames in [8, 100, 10_000] {
            assert!(!policy.should_insert_with(&with_parallax(evidence(frames, 80, 100), 0.3)));
        }
        let bounded = KeyframePolicy {
            max_frames_between: 30,
            ..parallax_policy()
        };
        assert!(!bounded.should_insert_with(&with_parallax(evidence(29, 80, 100), 0.3)));
        assert!(bounded.should_insert_with(&with_parallax(evidence(30, 80, 100), 0.3)));
    }

    #[test]
    fn the_time_bound_forces_a_keyframe_outside_min_frames_between() {
        let policy = KeyframePolicy {
            max_interval_sec: Some(0.5),
            ..parallax_policy()
        };
        assert!(!policy.interval_elapsed(1000, 0.49));
        assert!(policy.interval_elapsed(1000, 0.5));
        assert!(
            !policy.interval_elapsed(2, 10.0),
            "min_frames_between still applies"
        );
        assert!(
            !parallax_policy().interval_elapsed(1000, 1e9),
            "no bound by default"
        );
    }

    #[test]
    fn losing_the_reference_still_inserts_without_parallax() {
        let policy = parallax_policy();
        // 50 of 100 is under the 0.6 ratio with enough inliers to keep.
        assert!(policy.should_insert_with(&with_parallax(evidence(5, 50, 100), 0.0)));
        // Too few inliers for the ratio rule; the keyframe would be poor.
        assert!(!policy.should_insert_with(&with_parallax(evidence(5, 10, 100), 0.0)));
        // A reference with no map points is weak tracking: always insert.
        assert!(policy.should_insert_with(&with_parallax(evidence(5, 0, 0), 0.0)));
        assert!(policy.should_insert_with(&KeyframeEvidence {
            last_keyframe_idx: None,
            ..evidence(0, 0, 0)
        }));
    }

    #[test]
    fn keyframes_while_lost_follow_the_interval() {
        let policy = TrackingLossRecoveryPolicy::default();
        let interval = policy.keyframe_interval_while_lost_sec;
        assert!(!policy.keyframe_due(interval * 0.5));
        assert!(policy.keyframe_due(interval));
    }

    #[test]
    fn a_settled_imu_coasts_longer_than_vision_alone() {
        let policy = TrackingLossRecoveryPolicy::default();
        assert!(policy.grace_period_sec(true) > policy.grace_period_sec(false));
    }
}
