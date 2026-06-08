//! System runtime state, mode transitions, and tracking results.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;
use kornia_algebra::Vec3F64;

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
pub struct SystemState {
    pub pose_world_to_cam: Pose3d,
    pub velocity: Option<Pose3d>,
    pub velocity_world: Vec3F64,           // ADD: metric m/s in world frame
    pub last_frame_timestamp_sec: f64,     // ADD: for per-frame preint window
    pub imu_initialized: bool,             // ADD: gate for VI prediction path
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    pub consecutive_failures: usize,
    /// Maximum consecutive tracking failures before resetting to bootstrap.
    pub max_consecutive_failures: usize,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
}

/// Pipeline mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// IMU initialization for scale and gravity
    InertialInit,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            velocity_world:Vec3F64 { x: (0.0), y: (0.0), z: (0.0) },
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            consecutive_failures: 0,
            max_consecutive_failures: 15,
            bootstrap_frame: None,
            imu_initialized: false,
            last_frame_timestamp_sec: 0.0,
            mode: SystemMode::Bootstrap,
        }
    }

    pub fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.consecutive_failures = 0;
        self.bootstrap_frame = None;
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}
