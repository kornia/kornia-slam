//! Policies that remain owned by frame tracking.

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
