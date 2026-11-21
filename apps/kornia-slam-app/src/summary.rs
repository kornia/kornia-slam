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
         nonrigid_poses={} nonrigid_keyframes={} reseeds={} reseeds_held={} tilt_relevels={} \
         bias_resets={} reseed_stall_frames={} gate_rejects_rotation={} \
         gate_rejects_translation={} ba_stale_keyframes={} ba_propagated_keyframes={}\n\
         [summary] frame_ms {}\n\
         [summary] frontend_ms {}\n\
         {}",
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
        c.reseeds,
        c.reseeds_held,
        c.tilt_relevels,
        c.bias_resets,
        c.reseed_stall_frames,
        c.gate_rejects_rotation,
        c.gate_rejects_translation,
        c.ba_stale_keyframes,
        c.ba_propagated_keyframes,
        format_percentiles(&times.frame_ms),
        format_percentiles(&times.frontend_ms),
        format_estimator_recovery(c),
    )
}

/// The estimator (IMU trust, pose-inertial solve, G29) and recovery
/// (relocalization, own window) counters, in a fixed order.
fn format_estimator_recovery(c: &RunCounters) -> String {
    format!(
        "[summary] estimator_recovery ba_keyframe_shifts={} ba_snapbacks={} \
         large_window_solves={} pose_inertial_solves={} pose_inertial_keyframe_anchored={} \
         pose_inertial_prior_dropped={} imu_trust_lost={} imu_tail_holds={} \
         relocalization_attempts={} relocalizations={} relocalizations_while_lost={} \
         relocalization_refused_jump={} relocalizations_older_segment={} \
         relocalization_without_vocabulary={} relocalization_ms={:.1} seeded_window_frames={} \
         seeded_window_dropped_points={}",
        c.ba_keyframe_shifts,
        c.ba_snapbacks,
        c.large_window_solves,
        c.pose_inertial_solves,
        c.pose_inertial_keyframe_anchored,
        c.pose_inertial_prior_dropped,
        c.imu_trust_lost,
        c.imu_tail_holds,
        c.relocalization_attempts,
        c.relocalizations,
        c.relocalizations_while_lost,
        c.relocalization_refused_jump,
        c.relocalizations_older_segment,
        c.relocalization_without_vocabulary,
        c.relocalization_ns as f64 / 1e6,
        c.seeded_window_frames,
        c.seeded_window_dropped_points,
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
    fn first_line_ends_with_the_rigidity_the_reseed_the_gate_then_the_merge_counters() {
        // `RunCounters` is non-exhaustive, so fields are set one by one.
        let mut counters = RunCounters::default();
        counters.ba_solve_ns = 2_500_000;
        counters.nonrigid_poses = 3;
        counters.nonrigid_keyframes = 1;
        counters.reseeds = 12;
        counters.reseeds_held = 2;
        counters.tilt_relevels = 1;
        counters.bias_resets = 0;
        counters.reseed_stall_frames = 140;
        counters.gate_rejects_rotation = 4;
        counters.gate_rejects_translation = 9;
        counters.ba_stale_keyframes = 21;
        counters.ba_propagated_keyframes = 5;
        let text = format(&counters, &FrameTimes::default());
        let first = text.lines().next().unwrap();
        assert!(
            first.ends_with(
                "ba_solve_ms=2.5 nonrigid_poses=3 nonrigid_keyframes=1 reseeds=12 \
                 reseeds_held=2 tilt_relevels=1 bias_resets=0 reseed_stall_frames=140 \
                 gate_rejects_rotation=4 gate_rejects_translation=9 ba_stale_keyframes=21 \
                 ba_propagated_keyframes=5"
            ),
            "{first}"
        );
    }

    #[test]
    fn the_last_line_holds_the_estimator_and_recovery_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.ba_keyframe_shifts = 1;
        counters.ba_snapbacks = 2;
        counters.large_window_solves = 3;
        counters.pose_inertial_solves = 4;
        counters.pose_inertial_keyframe_anchored = 5;
        counters.pose_inertial_prior_dropped = 6;
        counters.imu_trust_lost = 7;
        counters.imu_tail_holds = 8;
        counters.relocalization_attempts = 9;
        counters.relocalizations = 10;
        counters.relocalizations_while_lost = 11;
        counters.relocalization_refused_jump = 12;
        counters.relocalizations_older_segment = 13;
        counters.relocalization_without_vocabulary = 14;
        counters.relocalization_ns = 15_000_000;
        counters.seeded_window_frames = 16;
        counters.seeded_window_dropped_points = 17;
        let text = format(&counters, &FrameTimes::default());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(
            lines[3],
            "[summary] estimator_recovery ba_keyframe_shifts=1 ba_snapbacks=2 \
             large_window_solves=3 pose_inertial_solves=4 pose_inertial_keyframe_anchored=5 \
             pose_inertial_prior_dropped=6 imu_trust_lost=7 imu_tail_holds=8 \
             relocalization_attempts=9 relocalizations=10 relocalizations_while_lost=11 \
             relocalization_refused_jump=12 relocalizations_older_segment=13 \
             relocalization_without_vocabulary=14 relocalization_ms=15.0 \
             seeded_window_frames=16 seeded_window_dropped_points=17"
        );
    }
}
