//! Tallies of optical-flow-first tracking (`klt_first`) over a run.

/// What became of the frames `klt_first` deferred, and the flow they ran on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlowCounters {
    /// Frames tracked by optical flow alone, without ORB.
    pub flow_frames: u64,
    /// Deferred frames whose flow-only pose was accepted but that were due as
    /// a keyframe, so ORB ran.
    pub flow_keyframe_due: u64,
    /// Deferred frames whose flow-only pose was refused.
    pub flow_refused: u64,
    /// Deferred frames without flow or with too few survivors on a map point.
    pub flow_unavailable: u64,
    /// `flow_keyframe_due` frames whose ORB estimate was then rejected.
    pub flow_due_orb_rejected: u64,
    /// Seeds of the flow-tracked frames.
    pub flow_seeds: u64,
    /// PnP inliers of the flow-tracked frames.
    pub flow_inliers: u64,
    /// Optical-flow solves that failed, dropped to "no seeds".
    pub klt_errors: u64,
    /// Survivor snaps to ORB keypoints that failed, dropped to "no seeds".
    pub snap_errors: u64,
}
