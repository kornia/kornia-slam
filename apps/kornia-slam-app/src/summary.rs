//! The `[summary]` lines printed at the end of a run: the system's run
//! counters and per-frame timing percentiles.

use kornia_slam::RunCounters;

/// Per-frame wall times in milliseconds, in frame order.
#[derive(Default)]
pub struct FrameTimes {
    frame_ms: Vec<f64>,
    frontend_ms: Vec<f64>,
}

impl FrameTimes {
    pub fn record(&mut self, frame_ms: f64, frontend_ms: f64) {
        self.frame_ms.push(frame_ms);
        self.frontend_ms.push(frontend_ms);
    }
}

pub fn format(counters: &RunCounters, times: &FrameTimes) -> String {
    let c = counters;
    format!(
        "[summary] frames={} tracked={} keyframes={} skipped_bootstrap={} coast_inertial={} \
         coast_visual={} give_ups={} klt_errors={} snap_errors={} ba_failed={} vi_ba_failed={} \
         prior_fallbacks={} ba_updates_rejected={} ba_solves={} ba_solve_ms={:.1} \
         nonrigid_poses={} nonrigid_keyframes={}\n\
         [summary] frame_ms {}\n\
         [summary] frontend_ms {}",
        c.frames,
        c.tracked,
        c.keyframes,
        c.skipped_bootstrap,
        c.coast_inertial,
        c.coast_visual,
        c.give_ups,
        c.klt_errors,
        c.snap_errors,
        c.ba_failed,
        c.vi_ba_failed,
        c.prior_fallbacks,
        c.ba_updates_rejected,
        c.ba_solves,
        c.ba_solve_ns as f64 / 1e6,
        c.nonrigid_poses,
        c.nonrigid_keyframes,
        format_percentiles(&times.frame_ms),
        format_percentiles(&times.frontend_ms),
    )
}

fn format_percentiles(values: &[f64]) -> String {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let [p50, p90, p99] = [50.0, 90.0, 99.0].map(|p| nearest_rank(&sorted, p));
    format!("p50={p50:.1} p90={p90:.1} p99={p99:.1}")
}

/// Nearest-rank percentile of ascending `sorted`; 0 when empty.
fn nearest_rank(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (percent / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let sorted: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(nearest_rank(&sorted, 50.0), 50.0);
        assert_eq!(nearest_rank(&sorted, 90.0), 90.0);
        assert_eq!(nearest_rank(&sorted, 99.0), 99.0);
        assert_eq!(nearest_rank(&[7.0], 99.0), 7.0);
        assert_eq!(nearest_rank(&[], 50.0), 0.0);
    }

    #[test]
    fn first_line_ends_with_the_rigidity_counters() {
        // `RunCounters` is non-exhaustive, so fields are set one by one.
        let mut counters = RunCounters::default();
        counters.ba_solve_ns = 2_500_000;
        counters.nonrigid_poses = 3;
        counters.nonrigid_keyframes = 1;
        let text = format(&counters, &FrameTimes::default());
        let first = text.lines().next().unwrap();
        assert!(
            first.ends_with("ba_solve_ms=2.5 nonrigid_poses=3 nonrigid_keyframes=1"),
            "{first}"
        );
    }
}
