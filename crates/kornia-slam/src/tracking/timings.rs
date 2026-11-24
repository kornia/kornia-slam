//! Per-stage wall time of one tracked frame, for profiling the frontend.

use std::time::{Duration, Instant};

use crate::system::MAX_FLOW_THREADS;

/// Flow shards a frame can time, one per flow thread.
pub const MAX_FLOW_SHARDS: usize = MAX_FLOW_THREADS;

/// Wall time of one flow shard's stages in the latest frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShardTimings {
    /// Template extraction, gradients included.
    pub flow_template: Duration,
    pub flow_forward: Duration,
    /// The forward-backward round trip.
    pub flow_backward: Duration,
    /// Survivors unprojected into rays.
    pub flow_project: Duration,
    pub flow_refill: Duration,
}

impl ShardTimings {
    /// Sum of the shard's stages.
    pub fn total(&self) -> Duration {
        self.flow_template
            + self.flow_forward
            + self.flow_backward
            + self.flow_project
            + self.flow_refill
    }
}

/// Wall time of each tracking stage of the latest frame. A stage that runs
/// more than once in a frame (the estimator's fallback paths) is summed; a
/// stage that did not run is zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackingTimings {
    /// u8-to-f32 casts, LK image pyramids and gradients.
    pub klt_build: Duration,
    /// Pyramidal LK on the prebuilt pyramids.
    pub klt_track: Duration,
    /// Undistortion reuse, grids and projection matching of the local map.
    pub projection_search: Duration,
    /// Fundamental-matrix filter against the reference keyframe.
    pub consistency_filter: Duration,
    /// Every PnP solve on the frame's estimation paths.
    pub pnp: Duration,
    /// Local-map projection search and its PnP solve.
    pub local_refine: Duration,
    /// Keyframe decision and insertion and, with synchronous mapping, the
    /// mapping work it triggers.
    pub keyframe_insert: Duration,
    /// Growth and forward fusion inside `keyframe_insert`; only the queueing
    /// with `keyframe_growth: Mapping`.
    pub keyframe_growth: Duration,
    /// Place recognition and inline loop closing inside `keyframe_insert`.
    pub loop_recognition: Duration,
    /// Time the tracking thread waited for the map lock, overlapping the
    /// other stages.
    pub map_wait: Duration,
    /// Flow path on the tracking core: the IMU guess and the level-0 copies.
    pub flow_prepare: Duration,
    /// Handing the shards to the flow workers.
    pub flow_dispatch: Duration,
    /// From dispatch to the last shard's return, the inline shard included.
    pub flow_parallel: Duration,
    pub flow_join: Duration,
    /// The pose of a frame tracked from flow alone.
    pub flow_pose: Duration,
    /// Track advance, counts and the keyframe decision of a flow frame.
    pub flow_bookkeeping: Duration,
    /// Time blocked on the map lock on the flow path; the other flow fields
    /// exclude it.
    pub flow_map_wait: Duration,
    /// Per flow shard, inside `flow_parallel`.
    pub shards: [ShardTimings; MAX_FLOW_SHARDS],
}

impl TrackingTimings {
    /// Sum of all tracking-thread stages; the shards run inside `flow_parallel`.
    pub fn total(&self) -> Duration {
        self.klt_build
            + self.klt_track
            + self.projection_search
            + self.consistency_filter
            + self.pnp
            + self.local_refine
            + self.keyframe_insert
            + self.flow_prepare
            + self.flow_dispatch
            + self.flow_parallel
            + self.flow_join
            + self.flow_pose
            + self.flow_bookkeeping
            + self.flow_map_wait
    }

    /// The flow frame's critical path: the tracking-core stages, dispatch,
    /// the slowest shard and the join.
    pub fn critical_path(&self) -> Duration {
        let slowest_shard = self
            .shards
            .iter()
            .map(ShardTimings::total)
            .max()
            .unwrap_or_default();
        self.total().saturating_sub(self.flow_parallel) + slowest_shard
    }
}

/// Runs `f`, adding its wall time to `slot`.
pub(crate) fn timed<T>(slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    *slot += start.elapsed();
    out
}
