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
    /// `trusted` when the IMU was confident and the coast was not a runaway.
    Coasting { inertial: bool, trusted: bool },
    /// A re-seed is waiting for its map, and the pose was carried on the kept
    /// inertial state rather than held.
    Reseeding,
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
    /// Local-BA write-backs that moved the newest keyframe's centre over 0.3 m.
    pub ba_keyframe_shifts: u64,
    /// Of those, next frames whose tracked step undid 0.5-2x of the shift.
    pub ba_snapbacks: u64,
    /// VI-BA solves run with the large window.
    pub large_window_solves: u64,
    /// VI-BA solves run with the base settings, and their iterations summed
    /// (card V: a 0-inlier job never takes the large window).
    pub vi_ba_base_solves: u64,
    pub vi_ba_base_iterations: u64,
    /// Most iterations of any VI-BA solve.
    pub vi_ba_iterations_max: u64,
    /// Keyframes inserted while a local BA solved, summed over its merges:
    /// the merge found them newer than its snapshot (card R).
    pub ba_stale_keyframes: u64,
    /// Of those, keyframes the merge carried the correction to.
    pub ba_propagated_keyframes: u64,
    /// Merges whose propagation stopped at a re-seeded segment's first
    /// keyframe that a bridge edge reaches (card V).
    pub ba_propagations_stopped_at_bridge: u64,
    /// Frames tracked by the pose-inertial solve.
    pub pose_inertial_solves: u64,
    /// Of those, solves anchored at a keyframe rather than the previous frame.
    pub pose_inertial_keyframe_anchored: u64,
    /// Frame priors dropped because their marginal was not positive definite.
    pub pose_inertial_prior_dropped: u64,
    /// Frames rejected because the pose-inertial solve kept fewer than
    /// `min_inliers` (ORB-SLAM3 fails them too).
    pub pose_inertial_rejects: u64,
    /// Transitions of the IMU prediction from trusted to untrusted.
    pub imu_trust_lost: u64,
    /// Midpoint preintegrations that held the last sample to the frame stamp.
    pub imu_tail_holds: u64,
    /// Calls of relocalization that reached the candidate query.
    pub relocalization_attempts: u64,
    /// Relocalizations adopted.
    pub relocalizations: u64,
    /// Of those, adopted only because relocalization runs while lost: the
    /// IMU was confident.
    pub relocalizations_while_lost: u64,
    /// Relocalized poses refused for jumping too far from the coasted one.
    pub relocalization_refused_jump: u64,
    /// Relocalizations adopted into an older segment once lost.
    pub relocalizations_older_segment: u64,
    /// Frames where a relocalization was due but no loop closer exists.
    pub relocalization_without_vocabulary: u64,
    /// Wall time inside relocalization, in nanoseconds.
    pub relocalization_ns: u64,
    /// Frames tracked with a re-seeded segment's own window active.
    pub seeded_window_frames: u64,
    /// Local-map points that window filtered out of the visibility count.
    pub seeded_window_dropped_points: u64,
    /// Frames tracked from optical flow alone.
    pub flow_frames: u64,
    /// Coasted frames the runaway check flagged, so not trusted (S0-C1b).
    pub coast_runaway_frames: u64,
    /// Lost keyframes inserted while the runaway check flagged the coast;
    /// only without `reseed.keep_inertial` (S0-C1b).
    pub lost_keyframes_at_runaway: u64,
    /// Flow survivors with a map point, summed over flow frames.
    pub flow_seeds: u64,
    /// Pose inliers on flow frames.
    pub flow_inliers: u64,
    /// Of those, from rig cameras 1 and up.
    pub flow_secondary_inliers: u64,
    /// Tracks entering the forward solve.
    pub flow_tracks_tried: u64,
    /// Tracks dropped by the forward-backward round trip.
    pub flow_fb_rejects: u64,
    /// Tracks re-run at the coarse level.
    pub flow_retries: u64,
    /// Tracks spawned by refill.
    pub flow_refill_spawns: u64,
    /// Tracks dropped by the anchor guard.
    pub flow_anchor_drops: u64,
    /// `AnalyticGn` flow frames that fell back to RANSAC PnP.
    pub flow_pose_fallbacks: u64,
    /// Templates cut again because the cached one came from another frame.
    pub flow_template_rebuilds: u64,
    /// Track sets cleared by a duplicate map point in reconciliation.
    pub flow_reconcile_clears: u64,
    /// Track positions rewritten after a local-mapping result or a map correction.
    pub flow_position_refreshes: u64,
    /// Frames between a deferred keyframe falling due and its insert, summed.
    pub deferred_keyframe_lag_frames: u64,
    /// Longest the deferred keyframe queue has been.
    pub deferred_keyframe_queue_max: u64,
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
            (TrackingStatus::Skipped, Some(SkipKind::Coasting { inertial: true, .. })) => {
                self.coast_inertial += 1;
            }
            (
                TrackingStatus::Skipped,
                Some(SkipKind::Coasting {
                    inertial: false, ..
                }),
            ) => {
                self.coast_visual += 1;
            }
            // A carried re-seed frame is still a bootstrap skip in the summary.
            (TrackingStatus::Skipped, Some(SkipKind::Bootstrap | SkipKind::Reseeding) | None) => {
                self.skipped_bootstrap += 1;
            }
        }
    }
}
