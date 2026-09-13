//! Policies controlling keyframe insertion and short tracking-loss recovery.

/// Keyframe insertion heuristics.
#[derive(Debug, Clone)]
pub struct KeyframePolicy {
    /// Minimum frame gap before allowing keyframe insertion.
    pub min_frames_between: usize,
    /// Force a keyframe if this frame gap is reached.
    pub max_frames_between: usize,
    /// Relative inlier ratio threshold (vs reference keyframe tracked map points).
    pub ref_ratio: f64,
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            min_frames_between: 3,
            max_frames_between: 8,
            ref_ratio: 0.6,
        }
    }
}

impl KeyframePolicy {
    /// Decide whether a new keyframe should be inserted.
    pub fn should_insert(
        &self,
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> bool {
        let Some(last_kf_idx) = last_keyframe_idx else {
            return true;
        };

        let frames_since_last_kf = curr_idx.saturating_sub(last_kf_idx);
        if frames_since_last_kf < self.min_frames_between {
            return false;
        }
        if frames_since_last_kf >= self.max_frames_between {
            return true;
        }

        if n_ref_map_points == 0 {
            return true;
        }

        let weak_threshold = (n_ref_map_points as f64 * self.ref_ratio) as usize;
        tracked_inliers >= 15 && tracked_inliers < weak_threshold
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
#[derive(Debug, Clone)]
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
