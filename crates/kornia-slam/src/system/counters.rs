//! Run counters: how many frames took each path and what local mapping did,
//! so failures and gaps are visible without debug logs.

/// Why the latest frame was [`TrackingStatus::Skipped`](super::TrackingStatus::Skipped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SkipKind {
    /// No map to track against: bootstrap is waiting, failed, or restarted
    /// after a give-up. The reported pose is the last one before the loss.
    Bootstrap,
    /// Tracking failed and the pose was carried on the motion prediction:
    /// the IMU when `inertial`, otherwise the constant-velocity model.
    Coasting { inertial: bool },
}

/// Tallies since [`SlamSystem::build`](super::SlamSystem::build). Every frame
/// is exactly one of tracked, bootstrap-skipped or coasted; local-mapping
/// figures are exact in synchronous mode, while in asynchronous mode a job
/// still on the worker is not counted yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunCounters {
    /// Frames processed (feature-extraction failures excluded).
    pub frames: u64,
    /// Frames with a tracked pose, keyframes included.
    pub tracked: u64,
    /// Keyframes accepted, bootstrap keyframes included.
    pub keyframes: u64,
    pub skipped_bootstrap: u64,
    pub coast_inertial: u64,
    pub coast_visual: u64,
    /// Losses that ran past the grace period and restarted bootstrap.
    pub give_ups: u64,
    /// Optical-flow tracking errors; the frame went on without flow seeds.
    pub klt_errors: u64,
    /// Flow-to-keypoint snap errors; the frame went on without flow seeds.
    pub snap_errors: u64,
    /// Visual-only local BA solves that returned an error.
    pub ba_failed: u64,
    /// Visual-inertial local BA solves that returned an error.
    pub vi_ba_failed: u64,
    /// VI solves that used the scaled boundary edge for want of a carried prior.
    pub prior_fallbacks: u64,
    /// Local BA updates the map refused to merge.
    pub ba_updates_rejected: u64,
    /// Local BA jobs where the solver ran (failures included).
    pub ba_solves: u64,
    /// Wall time of those solves, in nanoseconds.
    pub ba_solve_ns: u64,
    /// Frames whose reported pose rotation is off a rotation by more than 1e-6.
    pub nonrigid_poses: u64,
    /// Local BA keyframe corrections whose corrected rotation is off a rotation by more than 1e-6.
    pub nonrigid_keyframes: u64,
    /// Give-ups that kept the inertial state and re-seeded in the same world.
    pub reseeds: u64,
    /// Of those, re-seeds whose translation was held at the last tracked
    /// position: an implausible coast, a runaway, or a stall past the bridge.
    pub reseeds_held: u64,
    /// Re-seeds levelled because the measured specific force was tilted.
    pub tilt_relevels: u64,
    /// Bias components zeroed at a re-seed for leaving the state envelope.
    pub bias_resets: u64,
    /// Frames skipped while re-seeding with the inertial state kept.
    pub reseed_stall_frames: u64,
    /// Frames whose tracked pose the pose gate rejected for its rotation.
    pub gate_rejects_rotation: u64,
    /// Frames whose tracked pose the pose gate rejected for its position.
    pub gate_rejects_translation: u64,
}

impl RunCounters {
    /// Counts one processed frame with `status` and, if skipped, its `skip`.
    pub(crate) fn record_frame(&mut self, status: super::TrackingStatus, skip: Option<SkipKind>) {
        use super::TrackingStatus;
        self.frames += 1;
        match (status, skip) {
            (TrackingStatus::KeyframeAccepted, _) => {
                self.tracked += 1;
                self.keyframes += 1;
            }
            (TrackingStatus::Tracked, _) => self.tracked += 1,
            (TrackingStatus::Skipped, Some(SkipKind::Coasting { inertial: true })) => {
                self.coast_inertial += 1;
            }
            (TrackingStatus::Skipped, Some(SkipKind::Coasting { inertial: false })) => {
                self.coast_visual += 1;
            }
            (TrackingStatus::Skipped, Some(SkipKind::Bootstrap) | None) => {
                self.skipped_bootstrap += 1;
            }
        }
    }
}
