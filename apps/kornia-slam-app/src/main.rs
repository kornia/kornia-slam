//! `kornia-slam` command-line application: runs the SLAM runtime over the
//! source a run file describes.
//!
//! ```text
//! cargo run --release -p kornia-slam-app -- --config configs/euroc.ron
//! ```
//!
//! A run file names the source (EuRoC, Hilti, MCAP, or a live OAK-D or UVC
//! camera) and the system to run on it; see `configs/` and [`config`].

mod config;
mod datasets;
mod evaluation;
mod source;
mod tui;
mod utils;
use config::RunConfig;
use evaluation::associate_gt;
use kornia_algebra::Vec3F64;
use kornia_slam::system::PipelineDefinition;
use kornia_slam::{LoopClosureEvent, SensorFrame, SlamSystem};
use source::{FrameItem, OpenedSource};
use std::time::{Duration, Instant};
use utils::trajectory_point_from_pose;

#[cfg(feature = "viz")]
use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
};

#[derive(argh::FromArgs)]
#[argh(description = "Visual and visual-inertial ORB-SLAM over the source a run file describes")]
struct Args {
    /// run file (RON): the source to read and the system to run on it; see configs/
    #[argh(option)]
    config: String,

    /// after the run, align the trajectory to ground truth and report
    /// ATE/RPE/drift (writes kornia_slam_raw.csv and kornia_slam_aligned.csv);
    /// needs a source with ground truth (EuRoC, Hilti)
    #[argh(switch)]
    evaluate: bool,

    /// directory for the evaluation CSVs (created if missing; default: current dir)
    #[argh(option, default = "String::from(\".\")")]
    eval_out: String,

    /// spawn a Rerun viewer and stream to it (requires `--features viz`)
    #[argh(switch)]
    #[cfg(feature = "viz")]
    rerun_stream: bool,

    /// disable the terminal UI (status lines stream to stderr instead)
    #[argh(switch)]
    no_tui: bool,

    /// print per-frame diagnostics: bootstrap skip/reject reasons,
    /// map-projection reject reasons, keyframe growth and fuse counters
    #[argh(switch)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

    let run = RunConfig::from_ron_file(&args.config)?;
    if !tui_active {
        eprint!("{}", run.system);
    }

    let OpenedSource {
        mut source,
        ground_truth,
        summary,
    } = run.source.open(run.system.sensors)?;
    if args.evaluate && ground_truth.is_none() {
        return Err("--evaluate needs ground truth, and this source has none".into());
    }
    if !tui_active && let Some(summary) = summary {
        eprintln!("{summary}");
    }
    let target_dt = Duration::from_secs_f64(1.0 / 30.0);
    let mut last_frame_walltime = Instant::now();

    let n_frames_hint = source.n_frames_hint();
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    let vocabulary = orb
        .loop_closing
        .vocabulary()
        .map(std::path::Path::to_path_buf);
    let mut system = SlamSystem::build(run.system, source.rig()).map_err(|e| e.to_string())?;
    system.set_debug(args.debug);
    let camera = system.rig().camera.clone();
    if let Some(path) = vocabulary {
        eprintln!(
            "[place-recognition] loaded vocabulary from {}",
            path.display()
        );
    }
    if !tui_active && let Some(baseline) = system.rig().stereo_baseline_m {
        eprintln!(
            "Stereo: rectified fx={:.2} baseline={baseline:.4}m bf={:.2}",
            camera.fx,
            baseline * camera.fx,
        );
    }

    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("kornia-slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    let mut tui_state = if tui_active {
        let (term, guard) = tui::setup_terminal(std::path::Path::new("tui_stderr.log"))?;
        let mut app = tui::TuiApp::new(n_frames_hint.unwrap_or(0));
        app.debug_enabled = args.debug;
        Some((term, app, guard))
    } else {
        None
    };
    let (mut est_positions, mut gt_positions): (Vec<Vec3F64>, Vec<Vec3F64>) =
        (Vec::new(), Vec::new());
    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    let mut processed: usize = 0;

    while let Some(item) = source.next_frame()? {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_walltime);

        if elapsed < target_dt {
            std::thread::sleep(target_dt - elapsed);
        }

        last_frame_walltime = Instant::now();

        let FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            imu_samples,
        } = item;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            rec.set_time_sequence("frame", idx as i64);
            rec.set_duration_secs("timestamp", timestamp_sec);
        }

        let result = system.process(SensorFrame {
            idx,
            timestamp_sec,
            image: &image,
            right_image: right_image.as_ref(),
            imu_samples: &imu_samples,
        })?;
        let frame_ms = system.tracking_duration().as_secs_f64() * 1000.0;
        let frontend_ms = system.frontend_observation().duration.as_secs_f64() * 1000.0;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &image, &system.frontend_observation().keypoints_xy);
        }
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_active_map_points();
        let debug_msgs = system.drain_debug_messages();
        processed += 1;

        for event in system.drain_loop_closure_events() {
            match event {
                LoopClosureEvent::Accepted { edge, applied } => eprintln!(
                    "[loop-closure] accepted kf={} ~ kf={} inliers={} rmse={:.2}px applied={applied}",
                    edge.query_kf_idx,
                    edge.candidate_kf_idx,
                    edge.inliers,
                    edge.reprojection_rmse_px,
                ),
                LoopClosureEvent::PgoFailed {
                    query_kf_idx,
                    candidate_kf_idx,
                    reason,
                } => eprintln!("[pgo] failed kf={query_kf_idx} ~ kf={candidate_kf_idx}: {reason}"),
            }
        }

        if !tui_active {
            for line in &debug_msgs {
                eprintln!("{line}");
            }
            let status_line = format!(
                "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms fe={frontend_ms:.1}ms",
                result.status, keyframe_idx, map_point_count,
            );
            eprintln!("{status_line}");
        }

        let traj_pt = trajectory_point_from_pose(&result.pose_world_to_cam);
        trajectory.push(traj_pt);

        // Evaluation pairs each estimate with the nearest ground-truth pose.
        if args.evaluate
            && let Some(gt) = ground_truth
                .as_deref()
                .and_then(|gt| associate_gt(timestamp_sec, gt))
        {
            est_positions.push(Vec3F64::new(
                traj_pt[0] as f64,
                traj_pt[1] as f64,
                traj_pt[2] as f64,
            ));
            gt_positions.push(Vec3F64::new(gt.tx, gt.ty, gt.tz));
        }
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image.size());
            system.with_map_points(|map_points| log_map_points_to_rerun(rec, map_points));
        }

        if let Some((term, app, _guard)) = tui_state.as_mut() {
            for line in debug_msgs {
                app.push_debug_line(line);
            }
            app.frame_idx = idx;
            app.n_frames = n_frames_hint.unwrap_or(processed);
            app.frame_ms = frame_ms;
            app.status = match result.status {
                kornia_slam::TrackingStatus::Tracked => tui::TuiStatus::Tracked,
                kornia_slam::TrackingStatus::KeyframeAccepted => tui::TuiStatus::KeyframeAccepted,
                kornia_slam::TrackingStatus::Skipped => tui::TuiStatus::Skipped,
            };
            app.kf_idx = keyframe_idx;
            app.n_active_mp = map_point_count;
            let n_so_far = processed as f64;
            app.mean_ms = app.mean_ms + (frame_ms - app.mean_ms) / n_so_far;
            app.update_pose(&result.pose_world_to_cam);
            app.draw(term)?;
            match tui::poll_action()? {
                tui::TuiAction::Quit => break,
                tui::TuiAction::ToggleDebug => {
                    app.debug_enabled = !app.debug_enabled;
                    system.set_debug(app.debug_enabled);
                }
                tui::TuiAction::None => {}
            }
        }
    }

    // Restore terminal before printing the final summary.
    if let Some((mut term, _, _guard)) = tui_state.take() {
        tui::restore_terminal(&mut term)?;
    }

    let (total_pts, active_pts, obs_total, obs_max) = system.with_map_points(|map_points| {
        let mut active_pts: usize = 0;
        let mut obs_total: usize = 0;
        let mut obs_max: usize = 0;
        for mp in map_points.iter().filter(|mp| !mp.culled) {
            let n = mp.observations().len();
            active_pts += 1;
            obs_total += n;
            if n > obs_max {
                obs_max = n;
            }
        }
        (map_points.len(), active_pts, obs_total, obs_max)
    });
    let obs_mean = if active_pts > 0 {
        obs_total as f64 / active_pts as f64
    } else {
        0.0
    };
    eprintln!(
        "Done. Final map: total={total_pts}  active={active_pts}  obs_per_active_mp={obs_mean:.2}  max_obs={obs_max}"
    );
    if args.evaluate {
        evaluation::report(
            &est_positions,
            &gt_positions,
            std::path::Path::new(&args.eval_out),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use argh::FromArgs;
    use kornia_slam::SystemConfig;

    use super::*;

    fn parse(args: &[&str]) -> Result<Args, argh::EarlyExit> {
        Args::from_args(&["kornia-slam"], args)
    }

    #[test]
    fn a_run_file_is_required() {
        assert!(parse(&[]).is_err());
        let args = parse(&["--config", "configs/euroc.ron"]).unwrap();
        assert_eq!(args.config, "configs/euroc.ron");
        assert!(!args.evaluate);
        assert_eq!(args.eval_out, ".");
    }

    #[test]
    fn evaluation_is_a_global_option() {
        let args = parse(&["--config", "r.ron", "--evaluate", "--eval-out", "out"]).unwrap();
        assert!(args.evaluate);
        assert_eq!(args.eval_out, "out");
    }

    #[test]
    fn source_subcommands_and_algorithm_flags_are_gone() {
        for removed in [
            &["--config", "r.ron", "euroc", "--data", "d"][..],
            &["--config", "r.ron", "mcap", "--path", "r.mcap"],
            &["--config", "r.ron", "--data", "d"],
            &["--config", "r.ron", "--n-keypoints", "3000"],
            &["--config", "r.ron", "--stereo"],
        ] {
            assert!(parse(removed).is_err(), "{removed:?}");
        }
    }

    fn configs_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
    }

    #[test]
    fn shipped_run_files_load() {
        let mut loaded = 0;
        for entry in std::fs::read_dir(configs_dir()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "ron") {
                RunConfig::from_ron_file(&path)
                    .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
                loaded += 1;
            }
        }
        assert!(loaded > 0, "no run files in {}", configs_dir().display());
    }

    /// `euroc.ron` documents every system default, so it must stay equal to them.
    #[test]
    fn euroc_run_file_lists_the_system_defaults() {
        let run = RunConfig::from_ron_file(configs_dir().join("euroc.ron")).unwrap();
        assert_eq!(
            run.system.to_ron_string().unwrap(),
            SystemConfig::default().to_ron_string().unwrap(),
            "configs/euroc.ron no longer lists the system defaults"
        );
    }
}
