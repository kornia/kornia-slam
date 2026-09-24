//! Runtime state carried across frames, the system mode, and per-frame results.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;
use crate::initialization::AlignedTrackingState;
use kornia_algebra::Vec3F64;

/// Status of processing one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingStatus {
    /// Frame tracked successfully.
    Tracked,
    /// Frame processed but rejected (includes bootstrap frames before the map is ready).
    Skipped,
    /// Keyframe accepted and pose chained.
    KeyframeAccepted,
}

/// Result of processing one frame.
#[derive(Debug, Clone)]
pub struct TrackingResult {
    /// Current accumulated world-to-camera pose.
    pub pose_world_to_cam: Pose3d,
    /// Status for this frame.
    pub status: TrackingStatus,
}

/// Mutable pipeline state carried across frames.
#[derive(Debug, Clone)]
pub(crate) struct SystemState {
    pub pose_world_to_cam: Pose3d,
    pub velocity: Option<Pose3d>,
    /// Metric body velocity in the world frame (m/s); valid once `imu_initialized`.
    pub velocity_world: Vec3F64,
    /// Timestamp of the previous processed frame, bounding the per-frame
    /// preintegration window.
    pub last_frame_timestamp_sec: f64,
    /// Whether visual-inertial initialization succeeded (gates IMU pose prediction).
    pub imu_initialized: bool,
    /// Timestamp (sec) at which inertial initialization completed. IMU-only
    /// pose prediction isn't trustworthy yet for a short window after this,
    /// so a tracking loss shortly after init is treated as fully lost rather
    /// than granted the longer inertial grace period.
    pub imu_init_timestamp_sec: Option<f64>,
    /// Reference keyframe tracking runs against (ORB-SLAM3's `mpReferenceKF`).
    pub current_keyframe_idx: Option<usize>,
    /// Most recently inserted keyframe (ORB-SLAM3's `mpLastKeyFrame`); spaces
    /// keyframe insertion and anchors the next IMU edge. Equal to
    /// `current_keyframe_idx` until the reference keyframe can change on its own.
    pub last_keyframe_idx: Option<usize>,
    /// Timestamp (sec) of the first frame in the current run of tracking
    /// failures, or `None` while tracking is healthy. Drives the
    /// recently-lost grace period (mirrors ORB-SLAM3's `mTimeStampLost`).
    pub lost_since_sec: Option<f64>,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
}

/// Which stage the system is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SystemMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// IMU initialization for scale and gravity
    ImuInit,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            velocity_world: Vec3F64::ZERO,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            lost_since_sec: None,
            bootstrap_frame: None,
            imu_initialized: false,
            imu_init_timestamp_sec: None,
            last_frame_timestamp_sec: 0.0,
            mode: SystemMode::Bootstrap,
        }
    }

    pub fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.lost_since_sec = None;
        self.bootstrap_frame = None;
        // The new map starts at an unknown monocular scale, so the metric
        // IMU state no longer applies until inertial init runs again.
        self.imu_initialized = false;
        self.imu_init_timestamp_sec = None;
        self.velocity_world = Vec3F64::ZERO;
    }

    /// Resumes tracking from an applied inertial initialization.
    pub(crate) fn adopt_inertial_initialization(&mut self, aligned: Option<AlignedTrackingState>) {
        if let Some(aligned) = aligned {
            self.velocity_world = aligned.velocity_world;
            self.pose_world_to_cam = aligned.pose_world_to_cam;
        }
        self.velocity = None;
        self.imu_initialized = true;
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::SystemState;
    use crate::initialization::AlignedTrackingState;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;

    #[test]
    fn adopting_an_initialization_resumes_from_the_aligned_keyframe() {
        let mut state = SystemState::new();
        state.velocity = Some(Pose3d::IDENTITY);
        let mut pose = Pose3d::IDENTITY;
        pose.translation = Vec3F64::new(1.0, 2.0, 3.0);

        state.adopt_inertial_initialization(Some(AlignedTrackingState {
            pose_world_to_cam: pose,
            velocity_world: Vec3F64::new(0.5, 0.0, 0.0),
        }));

        assert!(state.imu_initialized);
        assert!(
            state.velocity.is_none(),
            "the visual motion model is dropped"
        );
        assert_eq!(state.pose_world_to_cam, pose);
        assert_eq!(state.velocity_world, Vec3F64::new(0.5, 0.0, 0.0));
    }
}
