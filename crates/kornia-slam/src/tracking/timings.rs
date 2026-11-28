//! Per-stage wall time of one tracked frame, for profiling the frontend.

use std::time::{Duration, Instant};

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
}

/// Runs `f`, adding its wall time to `slot`.
pub(crate) fn timed<T>(slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    *slot += start.elapsed();
    out
}
