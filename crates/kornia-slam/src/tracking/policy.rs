//! Policies controlling keyframe insertion and short tracking-loss recovery.

/// Keyframe insertion heuristics.
///
/// The defaults are the frame-count policy: a keyframe at least every
/// `max_frames_between` frames. Setting `max_frames_between` to `usize::MAX`
/// and `min_parallax_px` to a threshold gives ORB-SLAM3 / VINS-Mono style
/// insertion instead, where a keyframe is earned by motion (parallax) or by
/// losing the reference keyframe's landmarks, and a parked camera adds none.
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
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            min_frames_between: 3,
            max_frames_between: 8,
            ref_ratio: 0.6,
            min_parallax_px: None,
        }
    }
}

/// What the tracker measured on the current frame, for
/// [`KeyframePolicy::should_insert_with`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
        self.should_insert_with(&KeyframeEvidence {
            curr_idx,
            last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
            median_parallax_px: None,
        })
    }

    /// Decide whether a new keyframe should be inserted.
    ///
    /// In order: always for the first keyframe; never inside
    /// `min_frames_between`; always at `max_frames_between`; always when the
    /// reference keyframe holds no map points (tracking is weak); when the
    /// tracked ratio against the reference falls below `ref_ratio`; and, if
    /// `min_parallax_px` is set, when the measured parallax reaches it.
    pub fn should_insert_with(&self, evidence: &KeyframeEvidence) -> bool {
        let Some(last_kf_idx) = evidence.last_keyframe_idx else {
            return true;
        };

        let frames_since_last_kf = evidence.curr_idx.saturating_sub(last_kf_idx);
        if frames_since_last_kf < self.min_frames_between {
            return false;
        }
        if frames_since_last_kf >= self.max_frames_between {
            return true;
        }

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

/// Recently-lost grace period policy (mirrors ORB-SLAM3's RECENTLY_LOST vs
/// LOST distinction), bridging brief interruptions (motion blur, a few
/// dropped/occluded frames) without throwing the map away.
///
/// Deliberately short, unlike ORB-SLAM3's ~5s: our PnP has no RANSAC/robust
/// loss, so the projection search and the PnP prior-reprojection gate both
/// key off the same predicted pose. Once genuinely lost (not a brief blip),
/// that pose keeps compounding IMU/constant-velocity drift every extra frame
/// we wait, which does not improve recovery odds (verified against EuRoC
/// V101 frames ~600-770, a sustained-loss segment: granting several seconds
/// of patience there only delayed the same eventual reset, it never let
/// tracking resume early). A map that's too young, or an inertial state that
/// hasn't settled yet, gets no grace at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingLossRecoveryPolicy {
    /// Minimum keyframe count before any grace period is granted.
    pub min_keyframes_for_grace: usize,
    /// Grace period once the IMU has been initialized for at least
    /// `min_imu_confidence_sec`.
    pub timeout_imu_sec: f64,
    /// Grace period otherwise (no IMU, or too recently initialized).
    pub timeout_visual_sec: f64,
    /// How long the IMU must have been initialized before `timeout_imu_sec`
    /// applies instead of `timeout_visual_sec`.
    pub min_imu_confidence_sec: f64,
}

impl Default for TrackingLossRecoveryPolicy {
    fn default() -> Self {
        Self {
            min_keyframes_for_grace: 10,
            timeout_imu_sec: 1.0,
            timeout_visual_sec: 0.5,
            min_imu_confidence_sec: 2.0,
        }
    }
}

impl TrackingLossRecoveryPolicy {
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
            median_parallax_px: None,
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

    #[test]
    fn the_default_ignores_parallax_and_matches_should_insert() {
        let policy = KeyframePolicy::default();
        for frames in 0..12 {
            for (inliers, n_ref) in [(80, 100), (50, 100), (10, 100), (50, 0)] {
                let e = evidence(frames, inliers, n_ref);
                let plain = policy.should_insert(e.curr_idx, e.last_keyframe_idx, inliers, n_ref);
                assert_eq!(policy.should_insert_with(&e), plain);
                assert_eq!(policy.should_insert_with(&with_parallax(e, 1e6)), plain);
            }
        }
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
}
