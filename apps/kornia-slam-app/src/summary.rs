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
         prior_fallbacks={} ba_updates_rejected={} ba_solves={} ba_solve_ms={:.1}\n\
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
}
