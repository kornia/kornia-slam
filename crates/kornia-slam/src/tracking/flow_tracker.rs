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

/// Which flow shard a new track runs on, fixed at its spawn (plan section 3).
#[derive(Debug, Clone, Default)]
pub(crate) struct ShardLayout {
    threads: usize,
    cameras: usize,
    /// Tracks spawned so far per camera, for the `cameras + 1` split.
    spawned: Vec<u64>,
}

impl ShardLayout {
    /// Shards for `threads` flow threads on `cameras` cameras; validation
    /// allows 1, `cameras` or `cameras + 1` threads.
    pub(crate) fn new(threads: usize, cameras: usize) -> Self {
        Self {
            threads,
            cameras,
            spawned: vec![0; cameras],
        }
    }

    fn shards(&self) -> usize {
        self.threads.max(1)
    }

    /// The shard of the next track spawned in `camera`. With one thread per
    /// camera a camera is its shard; with one thread more, worker `c + 1`
    /// takes two thirds of camera `c` and the tracking core the rest.
    pub(crate) fn assign(&mut self, camera: usize) -> usize {
        if self.shards() == 1 {
            return 0;
        }
        let shard = if self.threads == self.cameras + 1 {
            match self.spawned.get_mut(camera) {
                Some(k) => {
                    let shard = if *k % 3 == 0 { 0 } else { camera + 1 };
                    *k += 1;
                    shard
                }
                None => 0,
            }
        } else {
            camera
        };
        shard.min(self.shards() - 1)
    }

    /// The shard that refills `camera`: the camera's own worker.
    pub(crate) fn refill_shard(&self, camera: usize) -> usize {
        let shard = match self.shards() {
            1 => 0,
            threads if threads == self.cameras + 1 => camera + 1,
            _ => camera,
        };
        shard.min(self.shards() - 1)
    }
}

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
