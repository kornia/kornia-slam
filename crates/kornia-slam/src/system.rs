//! System runtime state, mode transitions, and tracking results.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

use crate::frame::Frame;

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
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    pub consecutive_failures: usize,
    /// Maximum consecutive tracking failures before resetting to bootstrap.
    pub max_consecutive_failures: usize,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
    /// IMU state: world-frame velocity (m/s). `None` when IMU is not active.
    pub imu_velocity: Option<Vec3F64>,
    /// IMU state: current bias estimates.
    pub imu_bias: ImuBias,
    /// Gravity vector in world frame (e.g. [0, 0, -9.81]).
    pub gravity: Vec3F64,
    /// Buffered IMU measurements since the last processed frame.
    pub imu_buffer: Vec<ImuMeasurement>,
    /// IMU calibration parameters. `None` when running visual-only.
    pub imu_calib: Option<ImuCalib>,
}

/// Pipeline mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            consecutive_failures: 0,
            max_consecutive_failures: 15,
            bootstrap_frame: None,
            mode: SystemMode::Bootstrap,
            imu_velocity: None,
            imu_bias: ImuBias::default(),
            gravity: Vec3F64::new(0.0, 0.0, -9.81),
            imu_buffer: Vec::new(),
            imu_calib: None,
        }
    }

    /// Enable IMU fusion by providing calibration parameters.
    pub fn enable_imu(&mut self, calib: ImuCalib) {
        self.imu_calib = Some(calib);
    }

    /// Buffer an IMU measurement. Call this for each IMU sample between camera frames.
    pub fn push_imu(&mut self, measurement: ImuMeasurement) {
        self.imu_buffer.push(measurement);
    }

    /// Preintegrate buffered IMU measurements up to the given timestamp,
    /// returning the preintegrated result and draining consumed samples.
    ///
    /// Samples with timestamp > `up_to` are kept in the buffer for the next frame.
    pub fn preintegrate_imu(&mut self, up_to: f64) -> Option<PreintegratedImu> {
        let calib = self.imu_calib?;

        // Partition: consume samples with timestamp <= up_to
        let split_idx = self.imu_buffer.partition_point(|m| m.timestamp <= up_to);
        if split_idx < 2 {
            return None;
        }

        let consumed: Vec<_> = self.imu_buffer.drain(..split_idx).collect();
        let mut pre = PreintegratedImu::new(self.imu_bias, calib);
        pre.integrate_batch(&consumed);
        Some(pre)
    }

    pub fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.consecutive_failures = 0;
        self.bootstrap_frame = None;
        self.imu_velocity = None;
        self.imu_buffer.clear();
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}
