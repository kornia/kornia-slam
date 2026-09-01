//! Tracking runtime state, mode transitions, and per-frame results.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;
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

/// Mutable system state carried across frames.
#[derive(Debug, Clone)]
pub struct SystemState {
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
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    /// Timestamp (sec) of the first frame in the current run of tracking
    /// failures, or `None` while tracking is healthy. Drives the
    /// recently-lost grace period (mirrors ORB-SLAM3's `mTimeStampLost`).
    pub lost_since_sec: Option<f64>,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
}

/// Pipeline mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
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
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}
