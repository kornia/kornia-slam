//! One flow shard: the tracks of one worker, from start pixel to survivor,
//! with every buffer reused from frame to frame (plan section 3).

use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Mat3F64;

use crate::system::{AnchorGuard, ForwardBackward, TrackRefill};
use crate::tracking::optical_flow::{FlowSurvivor, Track, TrackId};
use crate::tracking::timings::ShardTimings;
use crate::upstream::kornia_imgproc::good_features::GoodFeaturesScratch;
use crate::upstream::kornia_imgproc::lk_window::{LkFrame, LkTemplate};

/// One camera's model for a flow step, refreshed in place each frame.
#[derive(Debug, Clone)]
pub(crate) struct FlowCamera {
    pub camera: PinholeCamera,
    /// `T_CP`; `None` for the primary camera.
    pub camera_from_primary: Option<Pose3d>,
    /// `R_cur_prev` of this camera from the gyro (`GyroGuess::RotatedRay`).
    pub gyro_rotation: Option<Mat3F64>,
}

/// What every shard needs to start its tracks; filled by the tracking core.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShardGuess {
    pub frame_idx: u64,
    /// World to primary camera at the IMU prediction, when trusted.
    pub pose_world_to_primary: Option<Pose3d>,
    pub cameras: Vec<FlowCamera>,
    pub max_level: usize,
    pub max_iter: usize,
    pub epsilon: f32,
    pub window: usize,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    /// Camera refilled this frame, if any.
    pub refill_camera: Option<usize>,
    /// Start-of-frame pixels of every track of `refill_camera`, all shards.
    pub refill_occupied: Vec<[f32; 2]>,
    pub anchor_guard: AnchorGuard,
    /// Retry pass: only `retry` tracks, from `max_level`.
    pub retry: bool,
}

/// One shard's tracks and every buffer it reuses; moved to a worker and back.
#[derive(Default)]
pub(crate) struct FlowShard {
    pub index: usize,
    /// Input, in (camera, track id) order.
    pub tracks: Vec<Track>,
    /// Output, in the order of `tracks`.
    pub survivors: Vec<FlowSurvivor>,
    /// Tracks that failed at level 0, for `klt_retry`.
    pub retry: Vec<TrackId>,
    pub spawned: Vec<(usize, [f32; 2])>,
    pub fb_rejects: u32,
    pub anchor_drops: u32,
    pub template_rebuilds: u32,
    templates: Vec<(TrackId, u64, LkTemplate)>,
    anchors: Vec<(TrackId, LkTemplate)>,
    starts: Vec<[f32; 2]>,
    features: GoodFeaturesScratch,
    occupied: Vec<[f32; 2]>,
}

impl FlowShard {
    /// Tracks `self.tracks` from `previous[c]` into `current[c]`.
    pub(crate) fn run(
        &mut self,
        _previous: &[Arc<LkFrame>],
        _current: &[Arc<LkFrame>],
        _guess: &ShardGuess,
        _timings: &mut ShardTimings,
    ) {
        todo!("lane T: flow shard")
    }
}
