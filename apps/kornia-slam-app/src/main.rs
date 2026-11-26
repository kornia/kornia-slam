//! `kornia-slam` command-line application: runs the SLAM runtime over the
//! source a run file describes.
//!
//! ```text
//! cargo run --release -p kornia-slam-app -- --config configs/euroc.ron
//! ```
//!
//! A run file names the source (EuRoC, Hilti, MCAP, or a live OAK-D or UVC
//! camera) and the system to run on it; see `configs/` and [`config`].

mod blackout;
mod config;
mod datasets;
mod evaluation;
mod map_dump;
mod session;
mod source;
mod summary;
mod trace_out;
mod trajectory;
mod tui;
mod utils;
use config::RunConfig;
use evaluation::associate_gt;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::rig_calibration::RigCalibration;
use kornia_slam::system::PipelineDefinition;
use kornia_slam::{LoopClosureEvent, SkipKind, SlamSystem};
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

    /// write `timestamp_ns tx ty tz qx qy qz qw` per tracked frame to this
    /// file: the body (IMU) pose in the world when the rig has an IMU, else
    /// the camera pose (the LaMAria estimate format)
    #[argh(option)]
    trajectory_out: Option<String>,

    /// like --trajectory-out, for the frames skipped while coasting on the
    /// IMU prediction instead; kept apart so the scored file is unchanged
    #[argh(option)]
    trajectory_coasted: Option<String>,

    /// also write the trusted, non-runaway frames coasted on the IMU
    /// prediction, and the re-seed frames carried on it, to
    /// --trajectory-out, so it has one pose per frame through visual gaps
    /// (card L)
    #[argh(switch)]
    trajectory_include_coasted: bool,

    /// like --trajectory-out, written at shutdown with each frame re-anchored
    /// to its reference keyframe's final pose (ORB-SLAM3's SaveTrajectoryEuRoC)
    #[argh(option)]
    trajectory_smoothed: Option<String>,

    /// write the sliding-window odometry's pose of every frame it estimated
    /// to this file, in the --trajectory-out layout (body pose); needs
    /// `pipeline.odometry: SlidingWindow`
    #[argh(option)]
    trajectory_vio: Option<String>,

    /// like --trajectory-vio, each line followed by the state and its
    /// counts: `vx vy vz bgx bgy bgz bax bay baz landmarks obs_left
    /// obs_right rows_left rows_right keyframe`
    #[argh(option)]
    vio_states: Option<String>,

    /// black images from `<start_s>` after the first frame for `<duration_s>`,
    /// given as `<start_s>:<duration_s>`; frames and IMU are still processed
    #[argh(option)]
    blackout: Option<String>,

    /// write the live landmarks at shutdown to this CSV (position, observation
    /// count, descriptor)
    #[argh(option)]
    dump_map: Option<String>,

    /// spawn a Rerun viewer and stream to it (requires `--features viz`)
    #[argh(switch)]
    #[cfg(feature = "viz")]
    rerun_stream: bool,

    /// process frames as fast as they come instead of at most 30 per second
    #[argh(switch)]
    no_pace: bool,

    /// process each frame no earlier than its source timestamp after the
    /// first frame's, so a run is paced at the sensor rate instead of at most
    /// 30 per second; excludes --no-pace
    #[argh(switch)]
    pace_timestamps: bool,

    /// disable the terminal UI (status lines stream to stderr instead)
    #[argh(switch)]
    no_tui: bool,

    /// print per-frame diagnostics: bootstrap skip/reject reasons,
    /// map-projection reject reasons, keyframe growth and fuse counters
    #[argh(switch)]
    debug: bool,

    /// write per-frame stage timings (timings.csv) and, for frames in
    /// [trace-from, trace-to), the tracking inputs a frontend benchmark
    /// replays (frames.csv, tracks.csv, local_map.bin) into this directory
    #[argh(option)]
    trace_out: Option<String>,

    /// write the opened source's rig as a rig calibration file (RON) to this
    /// path, then exit
    #[argh(option)]
    write_rig_calibration: Option<String>,

    /// record the session (keyframes, landmarks, IMU samples and the frames
    /// of --trajectory-out) and write it at shutdown to this file (CBOR), for
    /// `offline-ba`
    #[argh(option)]
    session_out: Option<String>,

    /// first source frame index to trace (default 0)
    #[argh(option, default = "0")]
    trace_from: usize,

    /// source frame index to stop tracing at (default: never)
    #[argh(option, default = "usize::MAX")]
    trace_to: usize,

    #[argh(subcommand)]
    command: Option<Command>,
}

#[derive(argh::FromArgs)]
#[argh(subcommand)]
enum Command {
    OfflineBa(OfflineBaArgs),
    SessionSlice(SessionSliceArgs),
}

/// Offline global visual-inertial BA over a recorded session, with the run
/// file's `offline_ba` settings; no online run happens.
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "offline-ba")]
struct OfflineBaArgs {
    /// session file written by --session-out
    #[argh(option)]
    session: String,

    /// write the solved trajectory here, in the --trajectory-out layout
    #[argh(option)]
    trajectory_out: String,

    /// write the solve's counters to this JSON file
    #[argh(option)]
    report: Option<String>,
}

/// Cuts a time span out of a recorded session (break-case fixtures); the run
/// file is not read.
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "session-slice")]
struct SessionSliceArgs {
    /// session file written by --session-out
    #[argh(option)]
    session: String,

    /// first keyframe stamp to keep, in the session's clock (s)
    #[argh(option)]
    from_sec: f64,

    /// last keyframe stamp to keep, in the session's clock (s)
    #[argh(option)]
    to_sec: f64,

    /// write the slice here (CBOR)
    #[argh(option)]
    out: String,
}

/// ` orb=[..] ov= klt=build/track ps= cf= pnp= lr= kf=` in ms, for `--debug`.
fn stage_times(
    frontend: &kornia_slam::FrontendObservation,
    timings: &kornia_slam::TrackingTimings,
) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let orb: Vec<String> = frontend
        .orb_per_camera
        .iter()
        .map(|d| format!("{:.1}", ms(*d)))
        .collect();
    format!(
        " orb=[{}] ov={:.1} klt={:.1}/{:.1} ps={:.1} cf={:.1} pnp={:.1} lr={:.1} kf={:.1}",
        orb.join(","),
        ms(frontend.overlap),
        ms(timings.klt_build),
        ms(timings.klt_track),
        ms(timings.projection_search),
        ms(timings.consistency_filter),
        ms(timings.pnp),
        ms(timings.local_refine),
        ms(timings.keyframe_insert),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

    if args.pace_timestamps && args.no_pace {
        return Err("--pace-timestamps and --no-pace exclude each other".into());
    }
    if let Some(Command::SessionSlice(slice)) = &args.command {
        return run_session_slice(slice);
    }
    let run = RunConfig::from_ron_file(&args.config)?;
    if let Some(Command::OfflineBa(offline)) = &args.command {
        return run_offline_ba(&run, offline);
    }
    if !tui_active {
        eprint!("{}", run.system);
    }

    let imu_time_offset_ns = match &run.source {
        source::SourceConfig::Lamaria(lamaria) => lamaria.imu_time_offset_ns,
        _ => 0,
    };
    let OpenedSource {
        mut source,
        ground_truth,
        summary,
    } = run.source.open(run.system.sensors, run.rectification)?;
    if args.evaluate && ground_truth.is_none() {
        return Err("--evaluate needs ground truth, and this source has none".into());
    }
    if let Some(path) = &args.write_rig_calibration {
        let calibration = RigCalibration::from_sensor_rig(&source.rig());
        let text = ron::ser::to_string_pretty(&calibration, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, text)?;
        return Ok(());
    }
    let rig = run.apply_rig_calibration(source.rig())?;
    if !tui_active && let Some(summary) = summary {
        eprintln!("{summary}");
    }
    let target_dt = Duration::from_secs_f64(1.0 / 30.0);
    let mut last_frame_walltime = Instant::now();
    let mut timestamp_pacer = TimestampPacer::default();

    let n_frames_hint = source.n_frames_hint();
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    check_trajectory_vio(&args, orb)?;
    let vocabulary = orb
        .loop_closing
        .vocabulary()
        .map(std::path::Path::to_path_buf);
    let mut system = SlamSystem::build(run.system, rig).map_err(|e| e.to_string())?;
    system.set_debug(args.debug);
    let mut session_recorder = args.session_out.as_ref().map(|_| {
        system.set_session_recording(true);
        session::SessionRecorder::new(imu_time_offset_ns)
    });
    let mut trace_writer = args
        .trace_out
        .as_deref()
        .map(|dir| trace_out::TraceWriter::create(std::path::Path::new(dir)))
        .transpose()?;
    let camera = system.rig().camera.clone();
    let mut trajectory_out = args
        .trajectory_out
        .as_deref()
        .map(|path| {
            trajectory::TrajectoryWriter::create(
                std::path::Path::new(path),
                system.rig().camera_to_body(),
            )
        })
        .transpose()?;
    let mut poses_written: u64 = 0;
    let mut poses_non_finite: u64 = 0;
    let mut trajectory_coasted = args
        .trajectory_coasted
        .as_deref()
        .map(|path| {
            trajectory::TrajectoryWriter::create(
                std::path::Path::new(path),
                system.rig().camera_to_body(),
            )
        })
        .transpose()?;
    let mut trajectory_vio = args
        .trajectory_vio
        .as_deref()
        .map(|path| {
            trajectory::TrajectoryWriter::create(
                std::path::Path::new(path),
                system.rig().camera_to_body(),
            )
        })
        .transpose()?;
    let mut vio_states = args
        .vio_states
        .as_deref()
        .map(|path| {
            trajectory::TrajectoryWriter::create(
                std::path::Path::new(path),
                system.rig().camera_to_body(),
            )
        })
        .transpose()?;
    // The session dump carries the smoothed frames too.
    let mut smoothed = (args.trajectory_smoothed.is_some() || args.session_out.is_some())
        .then(trajectory::SmoothedTrajectory::default);
    let mut blackout = args
        .blackout
        .as_deref()
        .map(blackout::Blackout::parse)
        .transpose()?;
    if let Some(path) = vocabulary {
        eprintln!(
            "[place-recognition] loaded vocabulary from {}",
            path.display()
        );
    }
    if !tui_active && let Some(baseline) = system.rig().stereo_baseline_m {
        // Printed only when non-default, so the golden headers stay valid.
        let tables = match run.rectification {
            source::Rectification::RemapF32 => "",
            source::Rectification::FixedPoint => " rectification=FixedPoint",
        };
        eprintln!(
            "Stereo: rectified fx={:.2} baseline={baseline:.4}m bf={:.2}{tables}",
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
    let mut frame_times = summary::FrameTimes::default();
    let mut stages = summary::StageTimes::default();
    let loop_start = Instant::now();

    loop {
        let source_start = Instant::now();
        let Some(mut item) = source.next_frame()? else {
            break;
        };
        let source_time = source_start.elapsed();
        if let Some(blackout) = blackout.as_mut()
            && blackout.apply(&mut item)
            && !tui_active
        {
            eprintln!("[blackout] frame={}", item.idx);
        }
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_walltime);

        // The pacer runs unpaced too, so latency is measured in every mode.
        let pace_delay = timestamp_pacer.delay(item.timestamp_sec, now);
        if args.pace_timestamps {
            std::thread::sleep(pace_delay);
        } else if !args.no_pace && elapsed < target_dt {
            std::thread::sleep(target_dt - elapsed);
        }

        last_frame_walltime = Instant::now();
        frame_times.record_sleep(last_frame_walltime.duration_since(now));

        let FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            rig_images,
            imu_samples,
        } = item;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            rec.set_time_sequence("frame", idx as i64);
            rec.set_duration_secs("timestamp", timestamp_sec);
        }

        if let Some(recorder) = session_recorder.as_mut() {
            recorder.record(image.size(), &imu_samples);
        }
        let process_start = Instant::now();
        system.process_imu(&imu_samples);
        system.set_trace(trace_writer.is_some() && (args.trace_from..args.trace_to).contains(&idx));
        let result = match right_image.as_ref() {
            Some(right) => system.process_stereo(timestamp_sec, &image, right)?,
            None if !rig_images.is_empty() => {
                let images: Vec<&kornia_image::Image<u8, 1>> =
                    std::iter::once(&image).chain(&rig_images).collect();
                system.process_rig(timestamp_sec, &images)?
            }
            None => system.process_image(timestamp_sec, &image)?,
        };
        let process_end = Instant::now();
        let wall = summary::FrameWall {
            timestamp_sec,
            source: source_time,
            process: process_end.duration_since(process_start),
            latency: timestamp_pacer.lateness(timestamp_sec, process_end),
            keyframe: result.status == kornia_slam::TrackingStatus::KeyframeAccepted,
        };
        let frame_ms = system.tracking_duration().as_secs_f64() * 1000.0;
        let frontend_ms = system.frontend_observation().duration.as_secs_f64() * 1000.0;
        frame_times.record(frame_ms, frontend_ms);
        stages.record(system.tracking_timings(), wall.keyframe);
        if let Some(writer) = trace_writer.as_mut() {
            writer.write_timings(
                idx,
                result.status,
                system.frontend_observation(),
                system.tracking_timings(),
                system.tracking_duration(),
                &wall,
            )?;
            if let Some(trace) = system.frame_trace() {
                writer.write_trace(idx, trace)?;
                let images: Vec<&kornia_image::Image<u8, 1>> = std::iter::once(&image)
                    .chain(right_image.as_ref())
                    .chain(&rig_images)
                    .collect();
                writer.write_images(system.rig(), idx, &images)?;
            }
        }
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
                LoopClosureEvent::Accepted { edge, applied } => {
                    eprintln!(
                        "[loop-closure] accepted kf={} ~ kf={} inliers={} rmse={:.2}px applied={applied}",
                        edge.query_kf_idx,
                        edge.candidate_kf_idx,
                        edge.inliers,
                        edge.reprojection_rmse_px,
                    );
                    if applied {
                        let correction = system.map_correction().map_from_odometry;
                        let t = correction.translation;
                        let rotation = SO3F64::from_matrix(&correction.rotation);
                        let [qx, qy, qz, qw] = rotation.to_array();
                        eprintln!(
                            "[map-correction] map_from_odometry t=({:.3},{:.3},{:.3}) rot_deg={:.2} \
                             epoch={} q=({qx:.9},{qy:.9},{qz:.9},{qw:.9}) t_full=({:.9},{:.9},{:.9})",
                            t.x,
                            t.y,
                            t.z,
                            rotation.log().length().to_degrees(),
                            system.correction_epoch(),
                            t.x,
                            t.y,
                            t.z,
                        );
                    }
                }
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
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let mut status_line = format!(
                "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms fe={frontend_ms:.1}ms \
                 src={:.1} tot={:.1} lat={:.1}",
                result.status,
                keyframe_idx,
                map_point_count,
                ms(wall.source),
                ms(wall.source + wall.process),
                ms(wall.latency),
            );
            if args.debug {
                status_line.push_str(&stage_times(
                    system.frontend_observation(),
                    system.tracking_timings(),
                ));
            }
            eprintln!("{status_line}");
        }

        let scored = writes_scored_pose(
            result.status,
            system.last_skip(),
            args.trajectory_include_coasted,
        );
        let finite = trajectory::is_finite_pose(&result.pose_world_to_cam);
        if scored && !finite {
            poses_non_finite += 1;
        }
        if let Some(writer) = trajectory_out.as_mut()
            && scored
            && writer.write(timestamp_sec, &result.pose_world_to_cam)?
        {
            poses_written += 1;
        }
        if let Some(writer) = trajectory_coasted.as_mut()
            && is_inertial_coast(system.last_skip())
        {
            writer.write(timestamp_sec, &result.pose_world_to_cam)?;
        }
        if let (Some(writer), Some(vio)) = (trajectory_vio.as_mut(), system.odometry_output())
            && vio.timestamp_sec == timestamp_sec
        {
            writer.write(timestamp_sec, &vio.pose_world_to_cam)?;
        }
        if let (Some(writer), Some(vio)) = (vio_states.as_mut(), system.odometry_output())
            && vio.timestamp_sec == timestamp_sec
        {
            writer.write_with(
                timestamp_sec,
                &vio.pose_world_to_cam,
                &vio_state_columns(vio),
            )?;
        }
        // Gated as --trajectory-out, so the session dump and offline-ba see
        // the same frames that are scored.
        if let Some(smoothed) = smoothed.as_mut()
            && scored
            && finite
        {
            let reference = system
                .current_keyframe_idx()
                .and_then(|kf| system.keyframe_pose(kf).map(|pose| (kf, pose)));
            smoothed.record_scored(timestamp_sec, &result.pose_world_to_cam, reference);
        }
        frame_times.record_wall(wall);
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

    let loop_wall = loop_start.elapsed();
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
    eprintln!("{}", summary::format(&system.run_counters(), &frame_times));
    eprintln!("{}", summary::format_wall(&frame_times, loop_wall));
    eprintln!("{}", summary::format_stages(&stages));
    if trajectory_out.is_some() {
        eprintln!("[summary] poses_written={poses_written}");
        eprintln!("[summary] poses_non_finite={poses_non_finite}");
    }
    if let Some(writer) = trajectory_out {
        writer.finish()?;
    }
    if let Some(writer) = trajectory_coasted {
        writer.finish()?;
    }
    if let Some(writer) = trajectory_vio {
        writer.finish()?;
    }
    if let Some(writer) = vio_states {
        writer.finish()?;
    }
    if let Some(counters) = system.odometry_counters() {
        eprintln!("{}", summary::format_vio(counters));
    }
    if let (Some(smoothed), Some(path)) = (smoothed.as_ref(), args.trajectory_smoothed.as_deref()) {
        let writer = trajectory::TrajectoryWriter::create(
            std::path::Path::new(path),
            system.rig().camera_to_body(),
        )?;
        let unanchored = smoothed.write(writer, |kf| system.keyframe_pose(kf))?;
        eprintln!("Smoothed trajectory: {unanchored} frame(s) without a surviving keyframe");
    }
    if let (Some(recorder), Some(smoothed), Some(path)) = (
        session_recorder,
        smoothed.as_ref(),
        args.session_out.as_deref(),
    ) {
        let unanchored = recorder.write(std::path::Path::new(path), &system, smoothed)?;
        eprintln!("Session: {path}, {unanchored} frame(s) without a dumped keyframe");
        // R-F1: session frames plus unrecorded ones equal the scored poses.
        eprintln!(
            "[summary] session_frames={} session_unrecorded={}",
            smoothed.frame_count() - unanchored,
            smoothed.unrecorded() + unanchored,
        );
    }
    if let Some(path) = args.dump_map.as_deref() {
        system.with_map_points(|map_points| {
            map_dump::write(std::path::Path::new(path), map_points)
        })?;
    }
    if let Some(writer) = trace_writer.as_mut() {
        writer.flush()?;
    }
    if args.evaluate {
        evaluation::report(
            &est_positions,
            &gt_positions,
            std::path::Path::new(&args.eval_out),
        )?;
    }
    Ok(())
}

/// `offline-ba`: one global VI-BA over a recorded session, with the run
/// file's `offline_ba` settings and its local-BA visual error model.
fn run_offline_ba(run: &RunConfig, args: &OfflineBaArgs) -> Result<(), Box<dyn std::error::Error>> {
    use kornia_slam::mapping::global_vi_ba::{offline_visual_inertial_ba, reanchor_frames};

    let dump = session::read(std::path::Path::new(&args.session))?;
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    let local_ba = &orb.mapping.local_ba;
    let result = offline_visual_inertial_ba(
        &dump,
        &local_ba.visual_information,
        local_ba.stereo_depth_sigma,
        orb.inertial.sample_interpolation,
        &run.offline_ba,
    )?;
    eprint!("{}", result.report);
    // The solved states are `T_bw`: written as given, they are `world_from_body`.
    let mut writer =
        trajectory::TrajectoryWriter::create(std::path::Path::new(&args.trajectory_out), None)?;
    let mut rows = 0usize;
    for (timestamp_sec, pose_bw) in reanchor_frames(&dump, &result.body_poses) {
        rows += usize::from(writer.write(timestamp_sec, &pose_bw)?);
    }
    writer.finish()?;
    // R-F1: a frame without a solved keyframe or with a non-finite pose is
    // missing from the output; the count makes the hole visible.
    eprintln!(
        "[offline_ba] rows={rows} session_frames={}",
        dump.frames.len()
    );
    if rows < dump.frames.len() {
        eprintln!(
            "[offline_ba] warning: {} session frame(s) missing from {}",
            dump.frames.len() - rows,
            args.trajectory_out
        );
    }
    if let Some(path) = args.report.as_deref() {
        let out = std::io::BufWriter::new(std::fs::File::create(path)?);
        serde_json::to_writer_pretty(out, &result.report)?;
    }
    Ok(())
}

/// `session-slice`: writes the part of a session in `[from_sec, to_sec]`.
fn run_session_slice(args: &SessionSliceArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.from_sec.is_nan() || args.to_sec.is_nan() || args.from_sec > args.to_sec {
        return Err("--from-sec must be a time at or before --to-sec".into());
    }
    let dump = session::read(std::path::Path::new(&args.session))?;
    let sliced = session::slice(&dump, args.from_sec, args.to_sec);
    if sliced.keyframes.is_empty() {
        return Err(format!(
            "no keyframe of {} lies in {}..={} s",
            args.session, args.from_sec, args.to_sec
        )
        .into());
    }
    session::write_dump(std::path::Path::new(&args.out), &sliced)?;
    eprintln!(
        "[session_slice] keyframes={} landmarks={} frames={} imu={} -> {}",
        sliced.keyframes.len(),
        sliced.landmarks.len(),
        sliced.frames.len(),
        sliced.imu.len(),
        args.out
    );
    Ok(())
}

/// `--pace-timestamps`: holds each frame until as much wall time has passed
/// since the first frame as source time.
#[derive(Default)]
struct TimestampPacer {
    /// The first frame's timestamp (s) and wall time.
    start: Option<(f64, Instant)>,
}

impl TimestampPacer {
    /// How long to wait at `now` before processing the frame at `timestamp_sec`.
    fn delay(&mut self, timestamp_sec: f64, now: Instant) -> Duration {
        let (first_sec, first_wall) = *self.start.get_or_insert((timestamp_sec, now));
        let due = Duration::try_from_secs_f64(timestamp_sec - first_sec).unwrap_or(Duration::ZERO);
        due.saturating_sub(now.duration_since(first_wall))
    }

    /// How far `now` lies behind the arrival time of the frame at
    /// `timestamp_sec`; zero before the first frame or when early.
    fn lateness(&self, timestamp_sec: f64, now: Instant) -> Duration {
        let Some((first_sec, first_wall)) = self.start else {
            return Duration::ZERO;
        };
        let due = Duration::try_from_secs_f64(timestamp_sec - first_sec).unwrap_or(Duration::ZERO);
        now.duration_since(first_wall).saturating_sub(due)
    }
}

/// Bootstrap skips hold the pose from before the loss and a visual coast has
/// no metric motion, so only an inertial coast is a usable estimate.
fn is_inertial_coast(skip: Option<SkipKind>) -> bool {
    matches!(skip, Some(SkipKind::Coasting { inertial: true, .. }))
}

/// Whether a frame's pose goes to --trajectory-out: every tracked frame, plus
/// the trusted inertial coasts and the re-seed frames carried on the IMU when
/// they are included. An untrusted or runaway coast is never scored.
fn writes_scored_pose(
    status: kornia_slam::TrackingStatus,
    skip: Option<SkipKind>,
    include_coasted: bool,
) -> bool {
    status != kornia_slam::TrackingStatus::Skipped
        || (include_coasted
            && matches!(
                skip,
                Some(SkipKind::Coasting {
                    inertial: true,
                    trusted: true
                }) | Some(SkipKind::Reseeding)
            ))
}

/// The `--vio-states` columns after the pose.
fn vio_state_columns(vio: &kornia_slam::OdometryOutput) -> String {
    let (v, g, a) = (vio.velocity_world, vio.bias.gyro, vio.bias.accel);
    format!(
        "{:.6} {:.6} {:.6} {:.7} {:.7} {:.7} {:.6} {:.6} {:.6} {} {} {} {} {} {}",
        v.x,
        v.y,
        v.z,
        g.x,
        g.y,
        g.z,
        a.x,
        a.y,
        a.z,
        vio.landmarks,
        vio.observations[0],
        vio.observations[1],
        vio.rows[0],
        vio.rows[1],
        u8::from(vio.keyframe)
    )
}

/// `--trajectory-vio` writes the sliding-window odometry's poses, so it needs
/// that estimator.
fn check_trajectory_vio(
    args: &Args,
    orb: &kornia_slam::system::OrbSlamPipeline,
) -> Result<(), String> {
    if (args.trajectory_vio.is_some() || args.vio_states.is_some())
        && orb.odometry.sliding_window().is_none()
    {
        return Err(
            "--trajectory-vio and --vio-states need pipeline.odometry: SlidingWindow".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use argh::FromArgs;

    use super::*;

    fn parse(args: &[&str]) -> Result<Args, argh::EarlyExit> {
        Args::from_args(&["kornia-slam"], args)
    }

    #[test]
    fn parses_the_run_file_and_evaluation_options() {
        assert!(parse(&[]).is_err(), "a run file is required");

        let args = parse(&["--config", "configs/euroc.ron"]).unwrap();
        assert_eq!(args.config, "configs/euroc.ron");
        assert!(!args.evaluate);
        assert_eq!(args.eval_out, ".");
        assert!(!args.no_pace);
        assert!(parse(&["--config", "r.ron", "--no-pace"]).unwrap().no_pace);

        let args = parse(&["--config", "r.ron", "--evaluate", "--eval-out", "out"]).unwrap();
        assert!(args.evaluate);
        assert_eq!(args.eval_out, "out");
    }

    /// R-F5: a frame waits for its timestamp's offset from the first frame,
    /// minus the wall time already spent; a late frame or a timestamp that
    /// steps back does not wait.
    #[test]
    fn pace_delay_follows_the_timestamps() {
        let wall = Instant::now();
        let mut pacer = TimestampPacer::default();
        assert_eq!(pacer.delay(1.0, wall), Duration::ZERO);
        assert_eq!(
            pacer.delay(1.5, wall + Duration::from_millis(30)),
            Duration::from_millis(470)
        );
        assert_eq!(
            pacer.delay(1.75, wall + Duration::from_millis(800)),
            Duration::ZERO
        );
        assert_eq!(pacer.delay(0.5, wall), Duration::ZERO);

        assert!(!parse(&["--config", "r.ron"]).unwrap().pace_timestamps);
        assert!(
            parse(&["--config", "r.ron", "--pace-timestamps"])
                .unwrap()
                .pace_timestamps
        );
    }

    /// F1: a frame finished after its arrival time is late by the
    /// difference; one finished early, or before any frame, is not.
    #[test]
    fn lateness_is_the_finish_behind_the_arrival() {
        let wall = Instant::now();
        let mut pacer = TimestampPacer::default();
        assert_eq!(pacer.lateness(1.0, wall), Duration::ZERO);
        pacer.delay(1.0, wall);
        assert_eq!(
            pacer.lateness(1.0, wall + Duration::from_millis(40)),
            Duration::from_millis(40)
        );
        assert_eq!(
            pacer.lateness(1.5, wall + Duration::from_millis(620)),
            Duration::from_millis(120)
        );
        assert_eq!(
            pacer.lateness(1.5, wall + Duration::from_millis(300)),
            Duration::ZERO
        );
    }

    #[test]
    fn offline_ba_is_an_optional_subcommand() {
        assert!(parse(&["--config", "r.ron"]).unwrap().command.is_none());
        let args = parse(&[
            "--config",
            "r.ron",
            "offline-ba",
            "--session",
            "s.cbor",
            "--trajectory-out",
            "t.txt",
        ])
        .unwrap();
        let Some(Command::OfflineBa(offline)) = args.command else {
            panic!("offline-ba did not parse");
        };
        assert_eq!(offline.session, "s.cbor");
        assert_eq!(offline.trajectory_out, "t.txt");
        assert!(offline.report.is_none());
    }

    #[test]
    fn the_vio_trajectory_needs_the_sliding_window() {
        let args = parse(&["--config", "r.ron", "--trajectory-vio", "v.tum"]).unwrap();
        assert_eq!(args.trajectory_vio.as_deref(), Some("v.tum"));
        let mut orb = kornia_slam::system::OrbSlamPipeline::default();
        assert!(check_trajectory_vio(&args, &orb).is_err());
        orb.odometry = kornia_slam::Odometry::SlidingWindow(Default::default());
        assert!(check_trajectory_vio(&args, &orb).is_ok());
        let states = parse(&["--config", "r.ron", "--vio-states", "s.txt"]).unwrap();
        assert!(
            check_trajectory_vio(&states, &kornia_slam::system::OrbSlamPipeline::default())
                .is_err()
        );
        let without = parse(&["--config", "r.ron"]).unwrap();
        let default = kornia_slam::system::OrbSlamPipeline::default();
        assert!(check_trajectory_vio(&without, &default).is_ok());
    }

    #[test]
    fn only_inertial_coasts_go_to_the_coasted_trajectory() {
        let args = parse(&["--config", "r.ron", "--trajectory-coasted", "c.tum"]).unwrap();
        assert_eq!(args.trajectory_coasted.as_deref(), Some("c.tum"));
        assert!(
            parse(&["--config", "r.ron"])
                .unwrap()
                .trajectory_coasted
                .is_none()
        );

        for trusted in [false, true] {
            assert!(is_inertial_coast(Some(SkipKind::Coasting {
                inertial: true,
                trusted
            })));
            assert!(!is_inertial_coast(Some(SkipKind::Coasting {
                inertial: false,
                trusted
            })));
        }
        assert!(!is_inertial_coast(Some(SkipKind::Bootstrap)));
        assert!(!is_inertial_coast(None));
    }

    #[test]
    fn coasted_poses_reach_the_scored_trajectory_only_when_included() {
        use kornia_slam::TrackingStatus::{KeyframeAccepted, Skipped, Tracked};
        let inertial = Some(SkipKind::Coasting {
            inertial: true,
            trusted: true,
        });
        let untrusted = Some(SkipKind::Coasting {
            inertial: true,
            trusted: false,
        });
        let visual = Some(SkipKind::Coasting {
            inertial: false,
            trusted: true,
        });
        assert!(!writes_scored_pose(Skipped, inertial, false));
        assert!(writes_scored_pose(Skipped, inertial, true));
        assert!(!writes_scored_pose(Skipped, visual, true));
        assert!(
            !writes_scored_pose(Skipped, untrusted, true),
            "an untrusted or runaway coast is not scored"
        );
        assert!(!writes_scored_pose(
            Skipped,
            Some(SkipKind::Bootstrap),
            true
        ));
        assert!(!writes_scored_pose(Skipped, None, true));
        let reseeding = Some(SkipKind::Reseeding);
        assert!(writes_scored_pose(Skipped, reseeding, true));
        assert!(!writes_scored_pose(Skipped, reseeding, false));
        assert!(
            !is_inertial_coast(reseeding),
            "--trajectory-coasted keeps the coasts only"
        );
        for status in [Tracked, KeyframeAccepted] {
            for skip in [
                None,
                inertial,
                untrusted,
                visual,
                Some(SkipKind::Bootstrap),
                reseeding,
            ] {
                assert!(writes_scored_pose(status, skip, false));
                assert!(writes_scored_pose(status, skip, true));
            }
        }
        assert!(
            parse(&["--config", "r.ron", "--trajectory-include-coasted"])
                .unwrap()
                .trajectory_include_coasted
        );
        assert!(
            !parse(&["--config", "r.ron"])
                .unwrap()
                .trajectory_include_coasted
        );
    }
}
