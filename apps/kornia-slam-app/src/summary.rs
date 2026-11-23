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
         gate_rejects_translation={} ba_stale_keyframes={} ba_propagated_keyframes={} \
         ba_propagations_stopped_at_bridge={}\n\
         [summary] frame_ms {}\n\
         [summary] frontend_ms {}\n\
         {}\n\
         {}\n\
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
        c.ba_propagations_stopped_at_bridge,
        format_percentiles(&times.frame_ms),
        format_percentiles(&times.frontend_ms),
        format_estimator_recovery(c),
        format_flow_coast(c),
        format_outlier_erase(c),
    )
}

/// The local VI-BA outlier erase counters (G56), in a fixed order; zero when
/// the erase is off.
fn format_outlier_erase(c: &RunCounters) -> String {
    format!(
        "[summary] outlier_erase erase_tested={} erase_unlinked={} erase_tested_newest={} \
         erase_unlinked_newest={} erase_landmarks_retired={} erase_stale_links={}",
        c.erase_tested,
        c.erase_unlinked,
        c.erase_tested_newest,
        c.erase_unlinked_newest,
        c.erase_landmarks_retired,
        c.erase_stale_links,
    )
}

/// The flow-frame decisions (card Z, S0-C11) and the runaway coast counters
/// (S0-C1b), in a fixed order.
fn format_flow_coast(c: &RunCounters) -> String {
    format!(
        "[summary] flow_coast flow_frames={} flow_keyframe_due={} flow_refused={} \
         flow_unavailable={} flow_due_orb_rejected={} coast_runaway_frames={} \
         lost_keyframes_at_runaway={}",
        c.flow_frames,
        c.flow_keyframe_due,
        c.flow_refused,
        c.flow_unavailable,
        c.flow_due_orb_rejected,
        c.coast_runaway_frames,
        c.lost_keyframes_at_runaway,
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
         seeded_window_dropped_points={} vi_ba_base_solves={} vi_ba_base_iterations={} \
         vi_ba_iterations_max={} vi_ba_rejected_steps={} vi_ba_solves_3plus_rejects={}",
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
        c.vi_ba_base_solves,
        c.vi_ba_base_iterations,
        c.vi_ba_iterations_max,
        c.vi_ba_rejected_steps,
        c.vi_ba_solves_3plus_rejects,
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
        counters.ba_propagations_stopped_at_bridge = 6;
        let text = format(&counters, &FrameTimes::default());
        let first = text.lines().next().unwrap();
        assert!(
            first.ends_with(
                "ba_solve_ms=2.5 nonrigid_poses=3 nonrigid_keyframes=1 reseeds=12 \
                 reseeds_held=2 tilt_relevels=1 bias_resets=0 reseed_stall_frames=140 \
                 gate_rejects_rotation=4 gate_rejects_translation=9 ba_stale_keyframes=21 \
                 ba_propagated_keyframes=5 ba_propagations_stopped_at_bridge=6"
            ),
            "{first}"
        );
    }

    #[test]
    fn the_fourth_line_holds_the_estimator_and_recovery_counters_in_order() {
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
        counters.vi_ba_base_solves = 18;
        counters.vi_ba_base_iterations = 19;
        counters.vi_ba_iterations_max = 20;
        counters.vi_ba_rejected_steps = 21;
        counters.vi_ba_solves_3plus_rejects = 22;
        let text = format(&counters, &FrameTimes::default());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 6);
        assert_eq!(
            lines[3],
            "[summary] estimator_recovery ba_keyframe_shifts=1 ba_snapbacks=2 \
             large_window_solves=3 pose_inertial_solves=4 pose_inertial_keyframe_anchored=5 \
             pose_inertial_prior_dropped=6 imu_trust_lost=7 imu_tail_holds=8 \
             relocalization_attempts=9 relocalizations=10 relocalizations_while_lost=11 \
             relocalization_refused_jump=12 relocalizations_older_segment=13 \
             relocalization_without_vocabulary=14 relocalization_ms=15.0 \
             seeded_window_frames=16 seeded_window_dropped_points=17 vi_ba_base_solves=18 \
             vi_ba_base_iterations=19 vi_ba_iterations_max=20 vi_ba_rejected_steps=21 \
             vi_ba_solves_3plus_rejects=22"
        );
    }

    #[test]
    fn the_fifth_line_holds_the_flow_and_coast_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.flow_frames = 1;
        counters.flow_keyframe_due = 2;
        counters.flow_refused = 3;
        counters.flow_unavailable = 4;
        counters.flow_due_orb_rejected = 5;
        counters.coast_runaway_frames = 6;
        counters.lost_keyframes_at_runaway = 7;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().nth(4).unwrap(),
            "[summary] flow_coast flow_frames=1 flow_keyframe_due=2 flow_refused=3 \
             flow_unavailable=4 flow_due_orb_rejected=5 coast_runaway_frames=6 \
             lost_keyframes_at_runaway=7"
        );
    }

    #[test]
    fn the_last_line_holds_the_outlier_erase_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.erase_tested = 1;
        counters.erase_unlinked = 2;
        counters.erase_tested_newest = 3;
        counters.erase_unlinked_newest = 4;
        counters.erase_landmarks_retired = 5;
        counters.erase_stale_links = 6;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().last().unwrap(),
            "[summary] outlier_erase erase_tested=1 erase_unlinked=2 erase_tested_newest=3 \
             erase_unlinked_newest=4 erase_landmarks_retired=5 erase_stale_links=6"
        );
    }
}
