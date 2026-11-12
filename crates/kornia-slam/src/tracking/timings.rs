//! Per-stage wall time of one tracked frame, for profiling the frontend.

use std::time::{Duration, Instant};

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
}

impl TrackingTimings {
    /// Sum of all stages.
    pub fn total(&self) -> Duration {
        self.klt_build
            + self.klt_track
            + self.projection_search
            + self.consistency_filter
            + self.pnp
            + self.local_refine
            + self.keyframe_insert
    }
}

/// Runs `f`, adding its wall time to `slot`.
pub(crate) fn timed<T>(slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    *slot += start.elapsed();
    out
}
