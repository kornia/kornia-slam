//! The still-device detector behind `keyframes.stillness` (card AB).
//!
//! A frame is still when no voter says moving and one position-aware voter
//! vouches for rest: the image (pixel-noise displacement against the last
//! keyframe), the wheels, or a quiet IMU when no image vote exists. Opening
//! is hard and closing easy: a false close costs one keyframe, a false
//! "still" costs accuracy. While still, `Gate` withholds the bound asks up
//! to a heartbeat (ORB-SLAM3 `LocalMapping.cc:1022-1024`: 3 s once the
//! inertial schedule settled, 0.5 s before), so each heartbeat keyframe keeps
//! an ordinary IMU edge.

use kornia_sensors::imu::ImuCalib;

use super::SlamSystem;
use super::counters::StillnessCounters;
use crate::frame::Frame;
use crate::inertial::still::{GyroView, ImuVote, StillRun};
use crate::tracking::KeyframeAsks;
use crate::tracking::parallax::{
    MIN_STILL_PAIRS, NoiseDisplacement, median_normalized_displacement,
};

pub(crate) const HEARTBEAT_SETTLED_SEC: f64 = 3.0;
pub(crate) const HEARTBEAT_UNSETTLED_SEC: f64 = 0.5;

/// A voter that vouched for rest or closed a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Voter {
    Image,
    Imu,
    Wheels,
}

impl Voter {
    fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Imu => "imu",
            Self::Wheels => "wheels",
        }
    }
}

/// The votes measured on one frame besides the IMU run's.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct FrameVotes {
    pub(crate) image: Option<NoiseDisplacement>,
    /// `Some(true)` parked, `Some(false)` moved, `None` uncovered.
    pub(crate) wheels: Option<bool>,
}

/// Whether a frame is still, and which voter vouched for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Verdict {
    pub(crate) voucher: Option<Voter>,
}

impl Verdict {
    pub(crate) fn still(self) -> bool {
        self.voucher.is_some()
    }
}

/// What a still frame's keyframe ask came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AskOutcome {
    Withheld,
    Heartbeat,
    Quality,
    LostWithheld,
}

impl AskOutcome {
    fn name(self) -> &'static str {
        match self {
            Self::Withheld => "withheld",
            Self::Heartbeat => "heartbeat",
            Self::Quality => "quality",
            Self::LostWithheld => "lost_withheld",
        }
    }
}

/// The run since the last keyframe and the detector's tallies.
#[derive(Debug, Default)]
pub(crate) struct StillnessState {
    run: Option<StillRun>,
    run_start_sec: f64,
    /// A non-IMU voter said moving: the run stays closed until the next keyframe.
    closed_by: Option<Voter>,
    run_counted: bool,
    /// The verdict of the frame judged last, so every caller on a frame gets the same one.
    judged: Option<(usize, Verdict)>,
    ask_counted: Option<usize>,
    pub(crate) counters: StillnessCounters,
}

impl StillnessState {
    /// Restarts the run when the last keyframe changed; no run without one.
    /// Returns whether a run started.
    fn sync(&mut self, last_keyframe: Option<(usize, f64)>) -> bool {
        let current = self.run.as_ref().map(|r| (r.kf_idx, self.run_start_sec));
        if current == last_keyframe {
            return false;
        }
        self.run = last_keyframe.map(|(idx, t)| StillRun::new(idx, t));
        self.run_start_sec = last_keyframe.map_or(0.0, |(_, t)| t);
        self.closed_by = None;
        self.run_counted = false;
        self.run.is_some()
    }

    /// Where the next IMU chunk starts, once one is due at `newest_sec`.
    fn imu_chunk_from(&self, newest_sec: f64) -> Option<f64> {
        let run = self.run.as_ref().filter(|r| r.is_open())?;
        (newest_sec >= run.end_sec() + crate::inertial::still::RUN_CHUNK_SEC).then(|| run.end_sec())
    }

    /// Judges a chunk; the closing line when it closed the run.
    fn advance_imu(
        &mut self,
        samples: &[kornia_sensors::imu::ImuMeasurement],
        calib: &ImuCalib,
        view: Option<GyroView>,
    ) -> Option<String> {
        let run = self.run.as_mut()?;
        let was_open = run.is_open();
        run.advance(samples, calib, view);
        let ImuVote::Moving(closer) = run.vote() else {
            return None;
        };
        if !was_open || self.closed_by.is_some() {
            return None;
        }
        self.counters.runs_closed_imu += 1;
        let (stat, threshold) = run.closing().unwrap_or((f64::NAN, f64::NAN));
        Some(format!(
            "[still] run kf={} closed by={} at={:.3} stat={stat:.4} threshold={threshold:.4}",
            run.kf_idx,
            closer.name(),
            run.end_sec()
        ))
    }

    /// The verdict on `frame_idx` at `t`, judged once per frame.
    fn judge(&mut self, frame_idx: usize, t: f64, votes: FrameVotes) -> (Verdict, bool) {
        if let Some((idx, verdict)) = self.judged
            && idx == frame_idx
        {
            return (verdict, false);
        }
        let verdict = match self.vote(t, votes) {
            Ok(voucher) => Verdict { voucher },
            Err(voter) => {
                self.close(voter);
                Verdict::default()
            }
        };
        self.count(verdict);
        self.judged = Some((frame_idx, verdict));
        (verdict, true)
    }

    /// The vouching voter (`None`: nobody vouched), or the voter that says moving.
    fn vote(&self, t: f64, votes: FrameVotes) -> Result<Option<Voter>, Voter> {
        let Some(run) = self.run.as_ref() else {
            return Ok(None);
        };
        if self.closed_by.is_some() || !run.is_open() {
            return Ok(None);
        }
        let image = votes.image.map(|d| d.is_still());
        if image == Some(false) {
            return Err(Voter::Image);
        }
        if votes.wheels == Some(false) {
            return Err(Voter::Wheels);
        }
        Ok(if image == Some(true) {
            Some(Voter::Image)
        } else if votes.wheels == Some(true) {
            Some(Voter::Wheels)
        } else if run.quiet_at(t) {
            Some(Voter::Imu)
        } else {
            None
        })
    }

    fn close(&mut self, voter: Voter) {
        self.closed_by = Some(voter);
        match voter {
            Voter::Image => self.counters.runs_closed_image += 1,
            Voter::Wheels => self.counters.runs_closed_wheels += 1,
            Voter::Imu => self.counters.runs_closed_imu += 1,
        }
    }

    fn count(&mut self, verdict: Verdict) {
        let Some(voucher) = verdict.voucher else {
            return;
        };
        let c = &mut self.counters;
        c.still_frames += 1;
        match voucher {
            Voter::Image => c.vouched_image += 1,
            Voter::Imu => c.vouched_imu += 1,
            Voter::Wheels => c.vouched_wheels += 1,
        }
        if !self.run_counted {
            self.run_counted = true;
            c.still_runs += 1;
        }
    }

    /// Counts a still frame's ask outcome, once per frame.
    fn note_ask(&mut self, frame_idx: usize, outcome: AskOutcome) -> bool {
        if self.ask_counted == Some(frame_idx) {
            return false;
        }
        self.ask_counted = Some(frame_idx);
        let c = &mut self.counters;
        match outcome {
            AskOutcome::Withheld => c.keyframe_asks_withheld += 1,
            AskOutcome::Heartbeat => c.heartbeat_keyframes += 1,
            AskOutcome::Quality => c.still_quality_keyframes += 1,
            AskOutcome::LostWithheld => c.lost_keyframes_withheld += 1,
        }
        true
    }
}

/// The gyro view check's bound: the image bound at the fewest pairs, in
/// radians at focal length `fx` (pixel sigma 1 at octave 0).
fn gyro_view_max_rad(fx: f64) -> f64 {
    (2.0 * NoiseDisplacement::bound(MIN_STILL_PAIRS)).sqrt() / fx
}

fn to_ns(t: f64) -> u64 {
    (t.max(0.0) * 1e9).round() as u64
}

impl SlamSystem {
    /// Keeps the run on the last keyframe and judges the IMU samples that
    /// arrived; once per frame, before the keyframe decision.
    pub(super) fn advance_stillness(&mut self) {
        if !self.keyframe_policy.stillness.observes() {
            return;
        }
        let last = self
            .state
            .last_keyframe_idx
            .zip(self.inertial.last_keyframe_timestamp_sec);
        if self.stillness.sync(last)
            && let Some((idx, t)) = last
        {
            self.dbg(format!("[still] run kf={idx} t={t:.3} started"));
        }
        let Some(imu) = self.rig.imu.as_ref().map(|imu| imu.noise) else {
            return;
        };
        let Some(from) = self
            .inertial
            .newest_sample_sec()
            .and_then(|newest| self.stillness.imu_chunk_from(newest))
        else {
            return;
        };
        let samples = self.inertial.samples_from(from);
        let view = self.state.imu_initialized.then(|| GyroView {
            bias: self.inertial.bias.gyro,
            max_rad: gyro_view_max_rad(self.rig.camera.fx),
        });
        if let Some(line) = self.stillness.advance_imu(&samples, &imu, view) {
            self.dbg(line);
        }
    }

    /// The image vote of `frame` against the run's keyframe.
    fn image_still_vote(
        &self,
        frame: &Frame,
        matches: &[(usize, usize)],
    ) -> Option<NoiseDisplacement> {
        let last = self.state.last_keyframe_idx?;
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        median_normalized_displacement(
            &map,
            last,
            frame,
            matches,
            &self.rig.camera,
            self.local_ba.visual_information,
        )
    }

    fn wheel_still_vote(&self, timestamp_sec: f64) -> Option<bool> {
        let start = self.inertial.last_keyframe_timestamp_sec?;
        self.wheel
            .as_ref()?
            .parked_vote(to_ns(start), to_ns(timestamp_sec))
    }

    fn still_verdict(
        &mut self,
        frame_idx: usize,
        timestamp_sec: f64,
        image: Option<NoiseDisplacement>,
    ) -> Verdict {
        let votes = FrameVotes {
            image,
            wheels: self.wheel_still_vote(timestamp_sec),
        };
        let (verdict, first) = self.stillness.judge(frame_idx, timestamp_sec, votes);
        if first && self.debug {
            let (q, pairs) = image.map_or((f64::NAN, 0), |d| (d.median, d.pairs));
            let bound = if pairs > 0 {
                NoiseDisplacement::bound(pairs)
            } else {
                f64::NAN
            };
            let imu = self
                .stillness
                .run
                .as_ref()
                .map_or("none".to_string(), |r| format!("{:?}", r.vote()));
            self.dbg(format!(
                "[still] frame={frame_idx} t={timestamp_sec:.3} still={} by={} q={q:.3} \
                 pairs={pairs} bound={bound:.3} imu={imu} wheels={:?}",
                u8::from(verdict.still()),
                verdict.voucher.map_or("-", Voter::name),
                votes.wheels
            ));
        }
        verdict
    }

    /// Whether a bound ask is let through: 3 s after the last keyframe once
    /// the inertial schedule settled, 0.5 s before; never without an IMU,
    /// whose edge is the only thing the heartbeat bounds.
    fn heartbeat_due(&self, timestamp_sec: f64) -> bool {
        if self.rig.imu.is_none() {
            return false;
        }
        let settled = self.state.imu_initialized
            && !self.inertial.schedule.refinements_pending(timestamp_sec);
        let interval = if settled {
            HEARTBEAT_SETTLED_SEC
        } else {
            HEARTBEAT_UNSETTLED_SEC
        };
        self.inertial
            .last_keyframe_timestamp_sec
            .is_some_and(|t| timestamp_sec - t >= interval)
    }

    /// The keyframe decision on a tracked frame once the detector runs:
    /// under `Gate` a still frame's bound-only ask waits for the heartbeat.
    pub(super) fn still_gated_decision(
        &mut self,
        asks: KeyframeAsks,
        frame: &Frame,
        matches: &[(usize, usize)],
        timestamp_sec: f64,
    ) -> bool {
        let image = self.image_still_vote(frame, matches);
        let verdict = self.still_verdict(frame.idx, timestamp_sec, image);
        if !verdict.still() || !asks.any() {
            return asks.any();
        }
        let outcome = if asks.quality {
            AskOutcome::Quality
        } else if !self.keyframe_policy.stillness.gates() {
            return true;
        } else if self.heartbeat_due(timestamp_sec) {
            AskOutcome::Heartbeat
        } else {
            AskOutcome::Withheld
        };
        self.note_still_ask(frame.idx, outcome);
        outcome != AskOutcome::Withheld
    }

    /// Whether a due lost-interval keyframe is withheld: a still frame under
    /// `Gate`, before the heartbeat. Judges the frame whenever the detector runs.
    pub(super) fn withhold_lost_keyframe(
        &mut self,
        frame_idx: usize,
        timestamp_sec: f64,
        due: bool,
    ) -> bool {
        if !self.keyframe_policy.stillness.observes() {
            return false;
        }
        let verdict = self.still_verdict(frame_idx, timestamp_sec, None);
        let withhold = due
            && verdict.still()
            && self.keyframe_policy.stillness.gates()
            && !self.heartbeat_due(timestamp_sec);
        if withhold {
            self.note_still_ask(frame_idx, AskOutcome::LostWithheld);
        }
        withhold
    }

    fn note_still_ask(&mut self, frame_idx: usize, outcome: AskOutcome) {
        if self.stillness.note_ask(frame_idx, outcome) {
            self.dbg(format!("[still] frame={frame_idx} ask={}", outcome.name()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn still_image() -> Option<NoiseDisplacement> {
        Some(NoiseDisplacement {
            median: 0.5,
            pairs: 100,
        })
    }

    fn moved_image() -> Option<NoiseDisplacement> {
        Some(NoiseDisplacement {
            median: 9.0,
            pairs: 100,
        })
    }

    fn state_on_keyframe() -> StillnessState {
        let mut state = StillnessState::default();
        assert!(state.sync(Some((4, 1.0))));
        assert!(!state.sync(Some((4, 1.0))));
        state
    }

    #[test]
    fn the_image_vouches_and_a_moving_vote_closes_the_run_until_the_next_keyframe() {
        let mut state = state_on_keyframe();
        let image = |image| FrameVotes {
            image,
            wheels: None,
        };
        let (verdict, first) = state.judge(5, 1.05, image(still_image()));
        assert!(first);
        assert_eq!(verdict.voucher, Some(Voter::Image));
        assert_eq!(state.judge(5, 1.05, image(moved_image())), (verdict, false));
        assert!(!state.judge(6, 1.1, image(moved_image())).0.still());
        assert!(!state.judge(7, 1.15, image(still_image())).0.still());
        assert_eq!(state.counters.runs_closed_image, 1);
        assert_eq!(state.counters.still_frames, 1);
        assert_eq!(state.counters.still_runs, 1);
        state.sync(Some((8, 1.2)));
        assert!(state.judge(9, 1.25, image(still_image())).0.still());
        assert_eq!(state.counters.still_runs, 2);
    }

    #[test]
    fn wheels_vouch_or_veto_and_no_vote_is_not_still() {
        let mut state = state_on_keyframe();
        let parked = FrameVotes {
            image: None,
            wheels: Some(true),
        };
        assert_eq!(state.judge(5, 1.05, parked).0.voucher, Some(Voter::Wheels));
        assert!(!state.judge(6, 1.1, FrameVotes::default()).0.still());
        let moved = FrameVotes {
            image: still_image(),
            wheels: Some(false),
        };
        assert!(!state.judge(7, 1.15, moved).0.still());
        assert_eq!(state.counters.runs_closed_wheels, 1);
        assert!(!StillnessState::default().judge(1, 0.0, parked).0.still());
    }

    #[test]
    fn an_ask_is_counted_once_per_frame() {
        let mut state = state_on_keyframe();
        assert!(state.note_ask(5, AskOutcome::Withheld));
        assert!(!state.note_ask(5, AskOutcome::Heartbeat));
        assert!(state.note_ask(6, AskOutcome::Heartbeat));
        assert_eq!(state.counters.keyframe_asks_withheld, 1);
        assert_eq!(state.counters.heartbeat_keyframes, 1);
    }

    #[test]
    fn the_gyro_view_bound_is_the_image_bound_in_radians() {
        assert!((gyro_view_max_rad(400.0) * 400.0 - 2.35).abs() < 0.01);
    }
}
