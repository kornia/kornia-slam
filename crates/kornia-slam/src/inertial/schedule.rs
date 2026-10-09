//! When inertial initialization runs: VIBA0 and its retries, then the VIBA1
//! and VIBA2 refinements.

/// When inertial initialization runs: the keyframe window it solves over, the
/// throttle on VIBA0 retries, and the one-shot VIBA1/VIBA2 refinements.
#[derive(Debug, Default)]
pub(crate) struct InertialInitSchedule {
    start_kf_idx: Option<usize>,
    /// ORB-SLAM3's `mFirstTs`; the refinements are gated on `mTinit`, the time
    /// since it (LocalMapping.cc:200-228).
    window_start_sec: Option<f64>,
    last_attempt_sec: Option<f64>,
    // Each refinement fires at most once, mirroring ORB-SLAM3's
    // `GetIniertialBA1()`/`GetIniertialBA2()` latches.
    viba1_done: bool,
    viba2_done: bool,
}

/// A progressive re-solve over the initialization window, with relaxed priors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refinement {
    Viba1,
    Viba2,
}

impl Refinement {
    /// Kept at VIBA2 because this pipeline lacks the intervening pose-adjusting
    /// inertial BA that lets ORB-SLAM3 safely remove the prior (kornia-slam#51).
    const PRIOR_A: f64 = 1e5;

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Viba1 => "VIBA1",
            Self::Viba2 => "VIBA2",
        }
    }

    pub(crate) fn prior_g(self) -> f64 {
        match self {
            Self::Viba1 => 1.0,
            Self::Viba2 => 0.0,
        }
    }

    pub(crate) fn prior_a(self) -> f64 {
        Self::PRIOR_A
    }
}

impl InertialInitSchedule {
    /// Opens a new window at `kf_idx`, re-arming both refinements.
    pub(crate) fn start(&mut self, kf_idx: usize, timestamp_sec: f64) {
        self.start_kf_idx = Some(kf_idx);
        self.window_start_sec = Some(timestamp_sec);
        self.viba1_done = false;
        self.viba2_done = false;
    }

    pub(crate) fn start_kf_idx(&self) -> Option<usize> {
        self.start_kf_idx
    }

    pub(crate) fn retry_due(&self, timestamp_sec: f64) -> bool {
        due_for_retry(self.last_attempt_sec, timestamp_sec)
    }

    pub(crate) fn record_attempt(&mut self, timestamp_sec: f64) {
        self.last_attempt_sec = Some(timestamp_sec);
    }

    /// The refinement due at `timestamp_sec`: VIBA1 after 5 s of window, VIBA2
    /// after 15 s, and nothing once the window is 50 s old.
    pub(crate) fn due_refinement(&self, timestamp_sec: f64) -> Option<Refinement> {
        let (Some(_), Some(window_start_sec)) = (self.start_kf_idx, self.window_start_sec) else {
            return None;
        };
        let mtinit = timestamp_sec - window_start_sec;
        if mtinit >= 50.0 {
            return None;
        }
        if !self.viba1_done && mtinit > 5.0 {
            Some(Refinement::Viba1)
        } else if self.viba1_done && !self.viba2_done && mtinit > 15.0 {
            Some(Refinement::Viba2)
        } else {
            None
        }
    }

    /// Marks a refinement as spent, whether or not its solve was accepted.
    pub(crate) fn complete(&mut self, refinement: Refinement) {
        match refinement {
            Refinement::Viba1 => self.viba1_done = true,
            Refinement::Viba2 => self.viba2_done = true,
        }
    }
}

/// Re-attempt interval for inertial initialization, in seconds of new data.
///
/// Without a throttle, once `ready()` is true a rejected attempt keeps the mode
/// at `ImuInit` and never resets the start index, so the same (growing) window
/// is re-solved from scratch on every subsequent keyframe forever — an
/// ever-more-expensive no-op once a call starts failing. Mirrors the VIBA1 5 s
/// cadence.
const RETRY_INTERVAL_SEC: f64 = 5.0;

/// Whether enough new data has arrived to justify another attempt.
fn due_for_retry(last_attempt_sec: Option<f64>, timestamp_sec: f64) -> bool {
    last_attempt_sec.is_none_or(|last| timestamp_sec - last >= RETRY_INTERVAL_SEC)
}

/// Accelerometer-bias prior for the first attempt.
///
/// VIBA0 is ORB-SLAM3's first `InitializeIMU` call (LocalMapping.cc:183-186):
/// heavily regularized, with mono suppressing accel bias almost entirely
/// because a short and early window cannot yet observe it.
pub(crate) fn viba0_accel_bias_prior(is_mono: bool) -> f64 {
    if is_mono { 1e10 } else { 1e5 }
}

#[cfg(test)]
mod tests {
    use super::{InertialInitSchedule, Refinement, due_for_retry, viba0_accel_bias_prior};

    /// The first attempt is never throttled; later ones wait for 5 s of new
    /// data. Without this, a rejected attempt re-solves an ever-growing window
    /// on every keyframe forever.
    #[test]
    fn retries_are_throttled_to_five_seconds_of_new_data() {
        assert!(due_for_retry(None, 0.0), "the first attempt is always due");
        assert!(due_for_retry(None, 1234.5));

        assert!(!due_for_retry(Some(10.0), 10.0));
        assert!(!due_for_retry(Some(10.0), 14.999));
        assert!(due_for_retry(Some(10.0), 15.0), "the boundary is inclusive");
        assert!(due_for_retry(Some(10.0), 20.0));
    }

    /// Mono suppresses the accelerometer bias almost entirely at VIBA0; stereo
    /// can observe it and regularizes far less.
    #[test]
    fn mono_suppresses_the_accel_bias_prior_far_harder_than_stereo() {
        let mono = viba0_accel_bias_prior(true);
        let stereo = viba0_accel_bias_prior(false);

        assert_eq!(mono, 1e10);
        assert_eq!(stereo, 1e5);
        assert!(mono > stereo);
    }

    /// VIBA1 waits for 5 s of window and VIBA2 for 15 s after VIBA1; each
    /// fires once, and neither fires once the window is 50 s old.
    #[test]
    fn refinements_fire_once_each_in_order() {
        let mut schedule = InertialInitSchedule::default();
        assert_eq!(schedule.due_refinement(100.0), None, "no window yet");

        schedule.start(7, 100.0);
        assert_eq!(schedule.due_refinement(105.0), None);
        assert_eq!(schedule.due_refinement(105.1), Some(Refinement::Viba1));
        assert_eq!(
            schedule.due_refinement(120.0),
            Some(Refinement::Viba1),
            "VIBA2 never runs before VIBA1"
        );

        schedule.complete(Refinement::Viba1);
        assert_eq!(schedule.due_refinement(115.0), None);
        assert_eq!(schedule.due_refinement(115.1), Some(Refinement::Viba2));

        schedule.complete(Refinement::Viba2);
        assert_eq!(schedule.due_refinement(130.0), None);
    }

    #[test]
    fn refinements_stop_once_the_window_is_fifty_seconds_old() {
        let mut schedule = InertialInitSchedule::default();
        schedule.start(0, 0.0);
        assert_eq!(schedule.due_refinement(49.9), Some(Refinement::Viba1));
        assert_eq!(schedule.due_refinement(50.0), None);
    }

    /// A new window re-arms both refinements but keeps the retry throttle.
    #[test]
    fn starting_a_window_rearms_refinements_but_not_the_retry_throttle() {
        let mut schedule = InertialInitSchedule::default();
        schedule.start(0, 0.0);
        schedule.record_attempt(10.0);
        schedule.complete(Refinement::Viba1);
        schedule.complete(Refinement::Viba2);

        schedule.start(40, 11.0);
        assert_eq!(schedule.start_kf_idx(), Some(40));
        assert_eq!(schedule.due_refinement(16.5), Some(Refinement::Viba1));
        assert!(!schedule.retry_due(12.0));
    }
}
