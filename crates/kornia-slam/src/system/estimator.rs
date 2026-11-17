//! The per-frame estimator choices of card A, ported from ORB-SLAM3 and
//! OpenVINS: keyframe evidence for ORB-SLAM3's insertion rules. Each one is
//! behind its knob, and the defaults read none of this.

use std::collections::HashSet;

use crate::Frame;
use crate::tracking::{KeyframeEvidence, KeyframeRules};

use super::SlamSystem;

/// Lane A's state carried across frames.
#[derive(Debug, Default)]
pub(super) struct EstimatorState {
    /// Frame index of the last relocalization.
    pub(super) last_relocalization_frame: Option<usize>,
}

impl SlamSystem {
    /// `evidence` with what ORB-SLAM3's keyframe rules read; unchanged under
    /// the default rules, which ignore it.
    pub(super) fn with_orb_slam3_evidence(
        &self,
        evidence: KeyframeEvidence,
        frame: &Frame,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> KeyframeEvidence {
        if self.keyframe_policy.rules == KeyframeRules::Kornia {
            return evidence;
        }
        let ref_points_min_obs = {
            let map = self.map.lock().unwrap();
            // `Tracking.cc:3097-3100`.
            let min_observers = if map.keyframes().len() <= 2 { 2 } else { 3 };
            self.state
                .current_keyframe_idx
                .and_then(|ki| map.get_keyframe(ki))
                .map_or(0, |kf| {
                    kf.map_point_by_desc_idx
                        .iter()
                        .flatten()
                        .filter(|&&mp_idx| {
                            map.map_points().get(mp_idx).is_some_and(|mp| {
                                !mp.culled && mp.observer_keyframes().count() >= min_observers
                            })
                        })
                        .count()
                })
        };
        let (tracked_close, untracked_close) = self.close_point_counts(frame, matches);
        KeyframeEvidence {
            ref_points_min_obs,
            tracked_close,
            untracked_close,
            recently_lost: self.state.lost_since_sec.is_some(),
            // A visual-only run has no phase before initialization.
            imu_initialized: self.state.imu_initialized || self.rig.imu.is_none(),
            mono_inertial: self.rig.imu.is_some() && !frame.is_metric(),
            secs_since_last_kf: self
                .inertial
                .last_keyframe_timestamp_sec
                .map(|t| timestamp_sec - t),
            frames_since_reloc: self
                .estimator
                .last_relocalization_frame
                .map(|reloc| frame.idx.saturating_sub(reloc)),
            ..evidence
        }
    }

    /// Primary-camera keypoints with a stereo depth closer than the close
    /// depth (ORB-SLAM3's `mThDepth`), tracked and not (`Tracking.cc:3106-3127`).
    fn close_point_counts(&self, frame: &Frame, matches: &[(usize, usize)]) -> (usize, usize) {
        let Some(close_depth) = self.stereo_close_depth else {
            return (0, 0);
        };
        let tracked: HashSet<usize> = matches.iter().map(|&(_, kp_idx)| kp_idx).collect();
        frame
            .camera_range(0)
            .filter(|&i| {
                frame
                    .stereo_depth(i)
                    .is_some_and(|z| f64::from(z) < close_depth)
            })
            .fold((0, 0), |(tracked_close, untracked_close), i| {
                if tracked.contains(&i) {
                    (tracked_close + 1, untracked_close)
                } else {
                    (tracked_close, untracked_close + 1)
                }
            })
    }
}
