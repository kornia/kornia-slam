//! Per-stage wall time of one tracked frame, for profiling the frontend.

use std::time::{Duration, Instant};

use crate::system::MAX_FLOW_THREADS;

/// Flow shards a frame can time, one per flow thread.
pub const MAX_FLOW_SHARDS: usize = MAX_FLOW_THREADS;

/// Wall time of one `WindowedU8` flow shard's stages in the latest frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShardTimings {
    /// Template extraction, gradients included.
    pub flow_template: Duration,
    pub flow_forward: Duration,
    /// The forward-backward round trip.
    pub flow_backward: Duration,
    /// Survivors undistorted.
    pub flow_project: Duration,
    pub flow_refill: Duration,
}

/// Wall time of each tracking stage of the latest frame. A stage that runs
/// more than once in a frame is summed; a stage that did not run is zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackingTimings {
    /// u8-to-f32 casts, LK image pyramids and gradients.
    pub klt_build: Duration,
    /// Pyramidal LK on the prebuilt pyramids.
    pub klt_track: Duration,
    /// Fundamental-matrix filter of a flow frame's seeds against the
    /// reference keyframe.
    pub consistency_filter: Duration,
    /// PnP solve of a flow frame.
    pub pnp: Duration,
    /// `WindowedU8` flow on the tracking thread: the guess and the level-0
    /// copies.
    pub flow_prepare: Duration,
    /// Handing the shards to the flow workers.
    pub flow_dispatch: Duration,
    /// From dispatch to the last shard's return, the inline shard included.
    pub flow_parallel: Duration,
    pub flow_join: Duration,
    /// The analytic pose of a frame tracked from flow alone.
    pub flow_pose: Duration,
    /// Per flow shard, inside `flow_parallel`.
    pub shards: [ShardTimings; MAX_FLOW_SHARDS],
}

/// Runs `f`, adding its wall time to `slot`.
pub(crate) fn timed<T>(slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    *slot += start.elapsed();
    out
}
