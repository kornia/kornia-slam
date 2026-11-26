//! The `[summary]` lines printed at the end of a run: the system's run
//! counters and per-frame timing percentiles.

use std::time::Duration;

use kornia_slam::RunCounters;

/// Per-frame wall times in milliseconds, in frame order.
#[derive(Default)]
pub struct FrameTimes {
    frame_ms: Vec<f64>,
    frontend_ms: Vec<f64>,
    wall: Vec<FrameWall>,
    sleep: Duration,
}

/// One frame's wall time as the app sees it: reading the frame, the
/// `process_*` calls, and how far their end lies behind the frame's paced
/// arrival. `frame_ms` leaves out ORB and the source work (F1).
pub struct FrameWall {
    pub timestamp_sec: f64,
    pub source: Duration,
    pub process: Duration,
    pub latency: Duration,
    pub keyframe: bool,
}

impl FrameTimes {
    pub fn record(&mut self, frame_ms: f64, frontend_ms: f64) {
        self.frame_ms.push(frame_ms);
        self.frontend_ms.push(frontend_ms);
    }

    pub fn record_wall(&mut self, wall: FrameWall) {
        self.wall.push(wall);
    }

    /// Time the loop slept for pacing.
    pub fn record_sleep(&mut self, sleep: Duration) {
        self.sleep += sleep;
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
         {}\n\
         {}\n\
         {}\n\
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
        format_flow_tracks(c),
        format_mapping_load(c),
        format_loss(c),
        format_loop_closure(c),
        format_stillness(c),
    )
}

fn format_loop_closure(c: &RunCounters) -> String {
    format!("[summary] loop_closure pgo_capped={}", c.pgo_capped)
}

/// How the tracking thread met local mapping's load (real-time fix ladder F4, F5a).
fn format_mapping_load(c: &RunCounters) -> String {
    format!(
        "[summary] mapping_load growth_queue_max={} growth_lag_frames_max={}",
        c.growth_queue_max, c.growth_lag_frames_max
    )
}

/// Per-frame tracking-thread stages the mapping-thread rungs move (F5a, F5b), in ms.
#[derive(Default)]
pub struct StageTimes {
    keyframe_insert: Vec<f64>,
    keyframe_growth: Vec<f64>,
    loop_recognition: Vec<f64>,
    map_wait: Vec<f64>,
}

impl StageTimes {
    pub fn record(&mut self, timings: &kornia_slam::TrackingTimings, keyframe: bool) {
        if keyframe {
            self.keyframe_insert.push(ms(timings.keyframe_insert));
            self.keyframe_growth.push(ms(timings.keyframe_growth));
            self.loop_recognition.push(ms(timings.loop_recognition));
        }
        self.map_wait.push(ms(timings.map_wait));
    }
}

/// Keyframe-frame stage percentiles, then the map-lock wait over every frame.
pub fn format_stages(times: &StageTimes) -> String {
    format!(
        "[summary] keyframe_insert_ms_keyframe {}\n\
         [summary] keyframe_growth_ms_keyframe {}\n\
         [summary] loop_recognition_ms_keyframe {}\n\
         [summary] map_wait_ms {}",
        format_percentiles(&times.keyframe_insert),
        format_percentiles(&times.keyframe_growth),
        format_percentiles(&times.loop_recognition),
        format_percentiles(&times.map_wait),
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

/// What flow-only frames rest on: seeds, inliers (and how many came from rig
/// cameras 1 and up), tried tracks and refill spawns; G50 reads these.
fn format_flow_tracks(c: &RunCounters) -> String {
    format!(
        "[summary] flow_tracks flow_seeds={} flow_inliers={} flow_secondary_inliers={} \
         flow_tracks_tried={} flow_refill_spawns={}",
        c.flow_seeds,
        c.flow_inliers,
        c.flow_secondary_inliers,
        c.flow_tracks_tried,
        c.flow_refill_spawns,
    )
}

/// The tracking-loss counters of card AC: later-segment alignments, seeds,
/// young deaths and coasts at a give-up, in a fixed order.
fn format_loss(c: &RunCounters) -> String {
    let milli = |value: u64| value as f64 / 1e3;
    format!(
        "[summary] loss later_alignments={} later_alignment_deg_max={:.3} \
         later_alignment_jump_m_max={:.3} seeds_gyro_held={} seed_rotation_deg_max={:.3} \
         young_segment_deaths={} startup_segment_deaths={} keeps_refused_before_viba2={} \
         coast_m_at_give_up_max={:.3} pose_innovation_judged={} pose_innovation_over={} \
         pose_innovation_rejects={} pose_innovation_unavailable={} pose_innovation_scale_max={:.3}",
        c.later_alignments,
        milli(c.later_alignment_mdeg_max),
        milli(c.later_alignment_jump_mm_max),
        c.seeds_gyro_held,
        milli(c.seed_rotation_mdeg_max),
        c.young_segment_deaths,
        c.startup_segment_deaths,
        c.keeps_refused_before_viba2,
        milli(c.coast_mm_at_give_up_max),
        c.pose_innovation_judged,
        c.pose_innovation_over,
        c.pose_innovation_rejects,
        c.pose_innovation_unavailable,
        milli(c.pose_innovation_scale_milli_max),
    )
}

/// The still-device detector's counters (card AB), in a fixed order.
fn format_stillness(c: &RunCounters) -> String {
    let s = &c.stillness;
    format!(
        "[summary] stillness still_frames={} still_runs={} keyframe_asks_withheld={} \
         heartbeat_keyframes={} still_quality_keyframes={} lost_keyframes_withheld={} \
         runs_closed_imu={} runs_closed_image={} runs_closed_wheels={} vouched_image={} \
         vouched_imu={} vouched_wheels={} runs_carried={}",
        s.still_frames,
        s.still_runs,
        s.keyframe_asks_withheld,
        s.heartbeat_keyframes,
        s.still_quality_keyframes,
        s.lost_keyframes_withheld,
        s.runs_closed_imu,
        s.runs_closed_image,
        s.runs_closed_wheels,
        s.vouched_image,
        s.vouched_imu,
        s.vouched_wheels,
        s.runs_carried,
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
         vi_ba_iterations_max={} vi_ba_rejected_steps={} vi_ba_solves_3plus_rejects={} \
         pose_inertial_rejects={}",
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
        c.pose_inertial_rejects,
    )
}

/// The app-side frame times (F1). `loop_wall` is the frame loop's wall time:
/// with nothing untimed it equals `sum_total_s + sum_sleep_s`.
pub fn format_wall(times: &FrameTimes, loop_wall: Duration) -> String {
    let line = |f: fn(&FrameWall) -> Option<f64>| {
        format_percentiles(&times.wall.iter().filter_map(f).collect::<Vec<_>>())
    };
    let period_ms = frame_period_ms(&times.wall);
    let late = times
        .wall
        .iter()
        .filter(|w| ms(w.latency) > period_ms)
        .count();
    let sum_total: Duration = times.wall.iter().map(|w| w.source + w.process).sum();
    format!(
        "[summary] source_ms {}\n\
         [summary] process_ms {}\n\
         [summary] total_ms {}\n\
         [summary] total_ms_keyframe {}\n\
         [summary] latency_ms {}\n\
         [summary] late_frames={late} frame_period_ms={period_ms:.1} final_latency_ms={:.1} \
         sum_total_s={:.3} sum_sleep_s={:.3} loop_wall_s={:.3}",
        line(|w| Some(ms(w.source))),
        line(|w| Some(ms(w.process))),
        line(|w| Some(ms(w.source + w.process))),
        line(|w| w.keyframe.then(|| ms(w.source + w.process))),
        line(|w| Some(ms(w.latency))),
        times.wall.last().map_or(0.0, |w| ms(w.latency)),
        sum_total.as_secs_f64(),
        times.sleep.as_secs_f64(),
        loop_wall.as_secs_f64(),
    )
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Median spacing of the frame timestamps in ms; 0 with fewer than 2 frames.
fn frame_period_ms(wall: &[FrameWall]) -> f64 {
    let mut steps: Vec<f64> = wall
        .windows(2)
        .map(|pair| (pair[1].timestamp_sec - pair[0].timestamp_sec) * 1000.0)
        .filter(|step| *step > 0.0)
        .collect();
    steps.sort_by(f64::total_cmp);
    nearest_rank(&steps, 50.0)
}

/// The sliding-window odometry's lines (`odometry: SlidingWindow`).
pub fn format_vio(c: &kornia_slam::OdometryCounters) -> String {
    let per_frame =
        |v: &[f32]| format_percentiles(&v.iter().map(|x| f64::from(*x)).collect::<Vec<_>>());
    let nullspace = if c.nullspace_yaw_log10_max.is_finite() {
        c.nullspace_yaw_log10_max
    } else {
        0.0
    };
    format!(
        "[summary] vio frames={} poses={} keyframes={} solves={} solve_failures={} resets={} \
         errors={} iterations_mean={:.3} rejected_steps={} landmarks_mean={:.1} \
         rows_left_mean={:.1} rows_right_mean={:.1} stereo_links={} fb_rejects={} \
         row_gate_rejects={} movement_rejects={} landmarks_fixed={} marg_pinv={} \
         imu_chi2_nav={:.4} imu_chi2_marg={:.4} bias_walk_chi2={:.4} visual_chi2={:.4} nullspace_yaw_log10_max={:.2}\n\
         [summary] vio_ms {}\n\
         [summary] vio_flow_ms {}\n\
         [summary] vio_solve_ms {}\n\
         [summary] vio_nullspace_ms {}",
        c.frames,
        c.poses,
        c.keyframes,
        c.solves,
        c.solve_failures,
        c.resets,
        c.errors,
        c.iterations_mean(),
        c.rejected_steps,
        c.landmarks_mean(),
        c.rows_left_mean(),
        c.rows_right_mean(),
        c.stereo_links,
        c.fb_rejects,
        c.row_gate_rejects,
        c.movement_rejects,
        c.landmarks_fixed,
        c.marg_pinv,
        c.imu_chi2_nav(),
        c.imu_chi2_marg(),
        c.bias_walk_chi2(),
        c.visual_chi2(),
        nullspace,
        per_frame(&c.total_ms),
        per_frame(&c.flow_ms),
        per_frame(&c.solve_ms),
        per_frame(&c.nullspace_ms),
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

    /// `E/score.py` reads `(\w+)=(-?[\d.]+)`: every vio field is plain
    /// fixed-point, and an empty run prints zeros.
    #[test]
    fn the_vio_lines_are_fixed_point_fields() {
        let text = format_vio(&kornia_slam::OdometryCounters::default());
        assert!(text.starts_with("[summary] vio frames=0 poses=0"), "{text}");
        assert!(text.contains("imu_chi2_marg=0.0000"), "{text}");
        assert!(text.contains("nullspace_yaw_log10_max=0.00"), "{text}");
        assert!(
            text.contains("[summary] vio_ms p50=0.0 p90=0.0 p99=0.0"),
            "{text}"
        );
        assert!(!text.contains("NaN") && !text.contains("inf"), "{text}");
    }

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
        counters.pose_inertial_rejects = 23;
        let text = format(&counters, &FrameTimes::default());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 11);
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
             vi_ba_solves_3plus_rejects=22 pose_inertial_rejects=23"
        );
    }

    #[test]
    fn the_line_before_stillness_holds_the_loop_closure_counters() {
        let mut counters = RunCounters::default();
        counters.pgo_capped = 3;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().rev().nth(1),
            Some("[summary] loop_closure pgo_capped=3")
        );
    }

    #[test]
    fn the_last_line_holds_the_stillness_counters_in_order() {
        let mut counters = RunCounters::default();
        let s = &mut counters.stillness;
        for (k, field) in [
            &mut s.still_frames,
            &mut s.still_runs,
            &mut s.keyframe_asks_withheld,
            &mut s.heartbeat_keyframes,
            &mut s.still_quality_keyframes,
            &mut s.lost_keyframes_withheld,
            &mut s.runs_closed_imu,
            &mut s.runs_closed_image,
            &mut s.runs_closed_wheels,
            &mut s.vouched_image,
            &mut s.vouched_imu,
            &mut s.vouched_wheels,
            &mut s.runs_carried,
        ]
        .into_iter()
        .enumerate()
        {
            *field = k as u64 + 1;
        }
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().last().unwrap(),
            "[summary] stillness still_frames=1 still_runs=2 keyframe_asks_withheld=3 \
             heartbeat_keyframes=4 still_quality_keyframes=5 lost_keyframes_withheld=6 \
             runs_closed_imu=7 runs_closed_image=8 runs_closed_wheels=9 vouched_image=10 \
             vouched_imu=11 vouched_wheels=12 runs_carried=13"
        );
    }

    #[test]
    fn the_flow_coast_and_flow_track_lines_hold_their_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.flow_frames = 1;
        counters.flow_keyframe_due = 2;
        counters.flow_refused = 3;
        counters.flow_unavailable = 4;
        counters.flow_due_orb_rejected = 5;
        counters.coast_runaway_frames = 6;
        counters.lost_keyframes_at_runaway = 7;
        counters.flow_seeds = 8;
        counters.flow_inliers = 9;
        counters.flow_secondary_inliers = 10;
        counters.flow_tracks_tried = 11;
        counters.flow_refill_spawns = 12;
        let text = format(&counters, &FrameTimes::default());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[4],
            "[summary] flow_coast flow_frames=1 flow_keyframe_due=2 flow_refused=3 \
             flow_unavailable=4 flow_due_orb_rejected=5 coast_runaway_frames=6 \
             lost_keyframes_at_runaway=7"
        );
        assert_eq!(
            lines[6],
            "[summary] flow_tracks flow_seeds=8 flow_inliers=9 flow_secondary_inliers=10 \
             flow_tracks_tried=11 flow_refill_spawns=12"
        );
    }

    #[test]
    fn the_sixth_line_holds_the_outlier_erase_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.erase_tested = 1;
        counters.erase_unlinked = 2;
        counters.erase_tested_newest = 3;
        counters.erase_unlinked_newest = 4;
        counters.erase_landmarks_retired = 5;
        counters.erase_stale_links = 6;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().nth(5).unwrap(),
            "[summary] outlier_erase erase_tested=1 erase_unlinked=2 erase_tested_newest=3 \
             erase_unlinked_newest=4 erase_landmarks_retired=5 erase_stale_links=6"
        );
    }

    #[test]
    fn the_eighth_line_holds_the_mapping_load_counters() {
        let mut counters = RunCounters::default();
        counters.growth_queue_max = 4;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().nth(7).unwrap(),
            "[summary] mapping_load growth_queue_max=4 growth_lag_frames_max=0"
        );
    }

    #[test]
    fn the_stage_lines_hold_keyframe_stages_and_every_frames_map_wait() {
        let ms = Duration::from_millis;
        let mut times = StageTimes::default();
        let timings = |insert, growth, wait| kornia_slam::TrackingTimings {
            keyframe_insert: ms(insert),
            keyframe_growth: ms(growth),
            loop_recognition: ms(1),
            map_wait: ms(wait),
            ..Default::default()
        };
        times.record(&timings(40, 20, 2), true);
        times.record(&timings(0, 0, 4), false);
        assert_eq!(
            format_stages(&times),
            "[summary] keyframe_insert_ms_keyframe p50=40.0 p90=40.0 p99=40.0\n\
             [summary] keyframe_growth_ms_keyframe p50=20.0 p90=20.0 p99=20.0\n\
             [summary] loop_recognition_ms_keyframe p50=1.0 p90=1.0 p99=1.0\n\
             [summary] map_wait_ms p50=2.0 p90=4.0 p99=4.0"
        );
    }

    #[test]
    fn the_wall_lines_hold_source_process_total_and_latency() {
        let ms = Duration::from_millis;
        let mut times = FrameTimes::default();
        for (i, (source, process, latency, keyframe)) in
            [(30, 20, 0, false), (30, 90, 70, true), (20, 40, 60, false)]
                .into_iter()
                .enumerate()
        {
            times.record_wall(FrameWall {
                timestamp_sec: 10.0 + 0.05 * i as f64,
                source: ms(source),
                process: ms(process),
                latency: ms(latency),
                keyframe,
            });
        }
        times.record_sleep(ms(500));
        times.record_sleep(ms(250));
        assert_eq!(
            format_wall(&times, ms(1000)),
            "[summary] source_ms p50=30.0 p90=30.0 p99=30.0\n\
             [summary] process_ms p50=40.0 p90=90.0 p99=90.0\n\
             [summary] total_ms p50=60.0 p90=120.0 p99=120.0\n\
             [summary] total_ms_keyframe p50=120.0 p90=120.0 p99=120.0\n\
             [summary] latency_ms p50=60.0 p90=70.0 p99=70.0\n\
             [summary] late_frames=2 frame_period_ms=50.0 final_latency_ms=60.0 \
             sum_total_s=0.230 sum_sleep_s=0.750 loop_wall_s=1.000"
        );
        assert_eq!(
            format_wall(&FrameTimes::default(), Duration::ZERO)
                .lines()
                .last(),
            Some(
                "[summary] late_frames=0 frame_period_ms=0.0 final_latency_ms=0.0 \
                 sum_total_s=0.000 sum_sleep_s=0.000 loop_wall_s=0.000"
            )
        );
    }

    #[test]
    fn the_ninth_line_holds_the_loss_counters_in_order() {
        let mut counters = RunCounters::default();
        counters.later_alignments = 2;
        counters.later_alignment_mdeg_max = 61_100;
        counters.later_alignment_jump_mm_max = 423_500;
        counters.seeds_gyro_held = 3;
        counters.seed_rotation_mdeg_max = 150_250;
        counters.young_segment_deaths = 4;
        counters.startup_segment_deaths = 5;
        counters.keeps_refused_before_viba2 = 1;
        counters.coast_mm_at_give_up_max = 7_125;
        counters.pose_innovation_judged = 900;
        counters.pose_innovation_over = 8;
        counters.pose_innovation_rejects = 6;
        counters.pose_innovation_unavailable = 50;
        counters.pose_innovation_scale_milli_max = 1_450;
        let text = format(&counters, &FrameTimes::default());
        assert_eq!(
            text.lines().nth(8).unwrap(),
            "[summary] loss later_alignments=2 later_alignment_deg_max=61.100 \
             later_alignment_jump_m_max=423.500 seeds_gyro_held=3 seed_rotation_deg_max=150.250 \
             young_segment_deaths=4 startup_segment_deaths=5 keeps_refused_before_viba2=1 \
             coast_m_at_give_up_max=7.125 pose_innovation_judged=900 pose_innovation_over=8 \
             pose_innovation_rejects=6 pose_innovation_unavailable=50 pose_innovation_scale_max=1.450"
        );
    }
}
