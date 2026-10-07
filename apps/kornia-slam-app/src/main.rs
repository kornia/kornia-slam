//! `kornia-slam` command-line application: runs the SLAM runtime over a selectable frame source.
//!
//! Run on an offline EuRoC dataset:
//! ```text
//! cargo run --release -p kornia-slam-app -- euroc --data /path/to/V1_01_easy
//! ```
//!
//! Run on a bubbaloop MCAP recording (defaults to the mono_left channel):
//! ```text
//! cargo run --release -p kornia-slam-app -- mcap --path /path/to/recording.mcap
//! ```
//!
//! Run live on an OAK-D camera (requires `--features oakd`):
//! ```text
//! cargo run --release -p kornia-slam-app --features oakd -- oakd
//! ```
//!
//! Run live on a UVC camera (built-in webcam, USB cam, etc.; requires
//! `--features uvc`):
//! ```text
//! cargo run --release -p kornia-slam-app --features uvc -- uvc \
//!     --fx 600 --fy 600 --cx 320 --cy 240
//! ```

mod datasets;
mod evaluation;
mod source;
mod tui;
mod utils;
use crate::datasets::euroc::GroundTruthPose;
use evaluation::associate_gt;
use kornia_algebra::Vec3F64;
use kornia_slam::pipeline::{CameraSelection, PipelineConfig, PipelineDefinition};
use kornia_slam::{LoopClosureEvent, SensorFrame, SlamSystem};
#[cfg(feature = "oakd")]
use source::OakdSource;
#[cfg(feature = "uvc")]
use source::UvcSource;
use source::{EurocSource, FrameItem, FrameSource, HiltiSource, McapSource};
use std::time::{Duration, Instant};
use utils::trajectory_point_from_pose;

#[cfg(feature = "viz")]
use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
};
/// CLI arguments.
#[derive(argh::FromArgs)]
#[argh(description = "Visual and visual-inertial ORB-SLAM over a selectable frame source")]
struct Args {
    #[argh(subcommand)]
    source: SourceCmd,

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

    /// pipeline configuration (RON): sensors, ORB settings and loop closing.
    /// Defaults to monocular ORB without loop closing; see configs/
    #[argh(option)]
    config: Option<String>,
}

#[derive(argh::FromArgs)]
#[argh(subcommand)]
enum SourceCmd {
    Euroc(EurocCmd),
    Hilti(HiltiCmd),
    Mcap(McapCmd),
    #[cfg(feature = "oakd")]
    Oakd(OakdCmd),
    #[cfg(feature = "uvc")]
    Uvc(UvcCmd),
}

/// Run on an EuRoC MAV dataset.
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "euroc")]
struct EurocCmd {
    /// path to EuRoC dataset root (e.g. V1_01_easy/)
    #[argh(option)]
    data: String,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,

    /// after the run, align the trajectory to ground truth and report
    /// ATE/RPE/drift (writes kornia_slam_raw.csv and kornia_slam_aligned.csv)
    #[argh(switch)]
    evaluate: bool,

    /// directory for the evaluation CSVs (created if missing; default: current dir)
    #[argh(option, default = "String::from(\".\")")]
    eval_out: String,
}

/// Run on a Hilti-Trimble SLAM Challenge 2026 sequence extracted to the
/// EuRoC-style layout by the challenge `ros2bag_to_euroc.py` tool.
///
/// ORB runs on the raw fisheye `cam0` image; keypoints (not pixels) are
/// undistorted into a virtual pinhole, preserving the full field of view. Images
/// are rotated 180° by default (inverted sensor mount); pass `--no-rotate` if the
/// extraction already rotated them. These 2 MP frames need about 3000 keypoints
/// to bootstrap (`configs/hilti.ron`).
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "hilti")]
struct HiltiCmd {
    /// path to the extracted sequence root (the dir containing cam0/, imu0/)
    #[argh(option)]
    data: String,

    /// path to the Kalibr camera-IMU chain YAML
    #[argh(option)]
    calib: String,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,

    /// do not rotate frames 180° (use when the extraction already rotated them)
    #[argh(switch)]
    no_rotate: bool,

    /// after the run, align the trajectory to ground truth and report
    /// ATE/RPE/drift (writes kornia_slam_raw.csv and kornia_slam_aligned.csv)
    #[argh(switch)]
    evaluate: bool,

    /// directory for the evaluation CSVs (created if missing; default: current dir)
    #[argh(option, default = "String::from(\".\")")]
    eval_out: String,
}

/// Run on a bubbaloop MCAP recording.
///
/// Defaults to the `mono_left` channel — 640×400 grayscale JPEGs, ready for
/// the SLAM pipeline without color conversion. Pass `--channel mono_right`
/// or `--channel compressed` to switch sources within the same file.
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "mcap")]
struct McapCmd {
    /// path to an MCAP file recorded by the bubbaloop mcap-recorder
    #[argh(option)]
    path: String,

    /// channel suffix to read (e.g. mono_left, mono_right, compressed).
    /// In stereo mode this is the left channel.
    #[argh(option, default = "String::from(\"mono_left\")")]
    channel: String,

    /// right stereo channel suffix (stereo mode)
    #[argh(option, default = "String::from(\"mono_right\")")]
    right_channel: String,

    /// path to a stereo calibration YAML (required for stereo cameras)
    #[argh(option)]
    calib: Option<String>,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,
}

/// Run live on an OAK-D camera (CamB mono, or CamB+CamC stereo).
#[cfg(feature = "oakd")]
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "oakd")]
struct OakdCmd {
    /// maximum number of frames to process (0 = run forever, Ctrl-C to stop)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// frame width in pixels (mono only; stereo uses the calibration's width)
    #[argh(option, default = "640")]
    width: u32,

    /// frame height in pixels (mono only; stereo uses the calibration's height)
    #[argh(option, default = "400")]
    height: u32,

    /// camera FPS
    #[argh(option, default = "30.0")]
    fps: f32,

    /// path to a stereo calibration YAML (required for stereo cameras)
    #[argh(option)]
    calib: Option<String>,
}

/// Run live on a UVC camera (laptop webcam, USB cam, CSI-to-UVC adapter…),
/// using V4L2 / AVFoundation / MSMF via nokhwa.
///
/// Intrinsics flags must match the resolution the device actually streams at
/// — nokhwa may pick the closest supported mode if the exact one is missing.
#[cfg(feature = "uvc")]
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "uvc")]
struct UvcCmd {
    /// camera device index (0 = first camera)
    #[argh(option, default = "0")]
    index: u32,

    /// frame width in pixels
    #[argh(option, default = "640")]
    width: u32,

    /// frame height in pixels
    #[argh(option, default = "480")]
    height: u32,

    /// maximum number of frames to process (0 = run forever, Ctrl-C to stop)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// focal length x (pixels)
    #[argh(option)]
    fx: f64,

    /// focal length y (pixels)
    #[argh(option)]
    fy: f64,

    /// principal point x (pixels)
    #[argh(option)]
    cx: f64,

    /// principal point y (pixels)
    #[argh(option)]
    cy: f64,

    /// radial distortion k1
    #[argh(option, default = "0.0")]
    k1: f64,

    /// radial distortion k2
    #[argh(option, default = "0.0")]
    k2: f64,

    /// tangential distortion p1
    #[argh(option, default = "0.0")]
    p1: f64,

    /// tangential distortion p2
    #[argh(option, default = "0.0")]
    p2: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // TUI is the default; --rerun-stream or --no-tui falls back to plain stderr.
    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

    // ── Pipeline ───────────────────────────────────────────────────────────
    let pipeline = match args.config.as_deref() {
        Some(path) => PipelineConfig::from_ron_file(path).map_err(|e| format!("{path}: {e}"))?,
        None => PipelineConfig::default(),
    };
    let stereo = pipeline.sensors.cameras == CameraSelection::Stereo;
    if !tui_active {
        eprint!("{pipeline}");
    }

    // ── Source ─────────────────────────────────────────────────────────────
    let mut evaluate = false;
    let mut eval_out = String::from(".");
    let target_dt = Duration::from_secs_f64(1.0 / 30.0);
    let mut last_frame_walltime = Instant::now();
    let (mut source, euroc_gt): (Box<dyn FrameSource>, Option<Vec<GroundTruthPose>>) = match args
        .source
    {
        SourceCmd::Euroc(e) => {
            evaluate = e.evaluate;
            eval_out = e.eval_out.clone();
            let src = if stereo {
                EurocSource::open_stereo(&e.data, e.start_frame, e.max_frames)?
            } else {
                EurocSource::open(&e.data, e.start_frame, e.max_frames)?
            };
            if !tui_active {
                let total = src.dataset_len();
                let n = src.n_frames_hint().unwrap_or(0);
                eprintln!(
                    "Dataset: {total} frames (processing {}..{})",
                    e.start_frame,
                    e.start_frame + n,
                );
            }
            // Clone the GT poses before the source is moved into the box.
            let gt = src.ground_truth_poses_cloned();
            (Box::new(src), Some(gt))
        }
        SourceCmd::Hilti(h) => {
            evaluate = h.evaluate;
            eval_out = h.eval_out.clone();
            let src =
                HiltiSource::open(&h.data, &h.calib, h.start_frame, h.max_frames, !h.no_rotate)?;
            if !tui_active {
                let total = src.dataset_len();
                let n = src.n_frames_hint().unwrap_or(0);
                eprintln!(
                    "Dataset: {total} frames (processing {}..{})",
                    h.start_frame,
                    h.start_frame + n,
                );
            }
            // Clone the GT poses before the source is moved into the box.
            let gt = src.ground_truth_poses_cloned();
            (Box::new(src), Some(gt))
        }
        SourceCmd::Mcap(m) => {
            let path = std::path::Path::new(&m.path);
            let src = if stereo {
                let calib = m
                    .calib
                    .as_deref()
                    .ok_or("mcap stereo cameras require --calib <stereo calibration YAML>")?;
                McapSource::open_stereo(
                    path,
                    &m.channel,
                    &m.right_channel,
                    std::path::Path::new(calib),
                    m.start_frame,
                    m.max_frames,
                )?
            } else {
                McapSource::open(path, &m.channel, m.start_frame, m.max_frames)?
            };
            if !tui_active && let Some(n) = src.n_frames_hint() {
                eprintln!("MCAP: {n} frames from /{}", m.channel);
            }
            (Box::new(src), None)
        }
        #[cfg(feature = "oakd")]
        SourceCmd::Oakd(o) => {
            let src = if stereo {
                let calib = o
                    .calib
                    .as_deref()
                    .ok_or("oakd stereo cameras require --calib <stereo calibration YAML>")?;
                OakdSource::open_stereo(o.fps, std::path::Path::new(calib), o.max_frames)?
            } else {
                OakdSource::open(o.width, o.height, o.fps, o.max_frames)?
            };
            (Box::new(src), None)
        }
        #[cfg(feature = "uvc")]
        SourceCmd::Uvc(w) => {
            let camera = kornia_3d::camera::PinholeCamera {
                fx: w.fx,
                fy: w.fy,
                cx: w.cx,
                cy: w.cy,
                k1: w.k1,
                k2: w.k2,
                p1: w.p1,
                p2: w.p2,
            };
            (
                Box::new(UvcSource::open(
                    w.index,
                    w.width,
                    w.height,
                    camera,
                    w.max_frames,
                )?),
                None,
            )
        }
    };

    let n_frames_hint = source.n_frames_hint();
    let mut system =
        SlamSystem::build(pipeline.clone(), source.rig()).map_err(|e| e.to_string())?;
    system.set_debug(args.debug);
    let camera = system.rig().camera.clone();
    let PipelineDefinition::OrbSlam(orb) = &pipeline.pipeline;
    if let Some(path) = orb.loop_closing.vocabulary() {
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

    // ── Rerun ──────────────────────────────────────────────────────────────
    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("kornia-slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    // ── TUI ────────────────────────────────────────────────────────────────
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
    // ── Main loop ──────────────────────────────────────────────────────────
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

        let t0 = Instant::now();
        let result = system.process(SensorFrame {
            idx,
            timestamp_sec,
            image: &image,
            right_image: right_image.as_ref(),
            imu_samples: &imu_samples,
        })?;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let frontend_ms = system.frontend_observation().duration.as_secs_f64() * 1000.0;
        // Tracking time excludes feature extraction, as it did before the
        // system owned the frontend.
        let frame_ms = total_ms - frontend_ms;
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

        // Status line.
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

        // Trajectory.
        let traj_pt = trajectory_point_from_pose(&result.pose_world_to_cam);
        trajectory.push(traj_pt);

        // Evaluation collection (EuRoC, --evaluate only).
        if evaluate {
            let est_pos = Vec3F64::new(traj_pt[0] as f64, traj_pt[1] as f64, traj_pt[2] as f64);
            est_positions.push(est_pos);

            // Associate nearest ground-truth pose by timestamp.
            let gt_pos = euroc_gt
                .as_deref()
                .and_then(|gt| associate_gt(timestamp_sec, gt))
                .map(|gt| Vec3F64::new(gt.tx, gt.ty, gt.tz))
                .unwrap_or_else(|| *est_positions.last().unwrap());
            gt_positions.push(gt_pos);
        }
        // Rerun logging.
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image.size());
            system.with_map_points(|map_points| log_map_points_to_rerun(rec, map_points));
        }

        // TUI render.
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
    // ── Trajectory evaluation (EuRoC, --evaluate only) ─────────────────────
    if evaluate {
        evaluation::report(
            &est_positions,
            &gt_positions,
            std::path::Path::new(&eval_out),
        )?;
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
    fn config_selects_the_pipeline() {
        let args = parse(&["--config", "configs/stereo-imu.ron", "euroc", "--data", "d"]).unwrap();
        assert_eq!(args.config.as_deref(), Some("configs/stereo-imu.ron"));
        assert!(parse(&["euroc", "--data", "d"]).unwrap().config.is_none());
    }

    #[test]
    fn algorithm_and_sensor_flags_are_rejected() {
        for removed in [
            &["--n-keypoints", "3000", "euroc", "--data", "d"][..],
            &["--local-mapping", "sync", "euroc", "--data", "d"],
            &["--vocab", "ORBvoc.txt", "euroc", "--data", "d"],
            &["--apply-pgo", "euroc", "--data", "d"],
            &["euroc", "--data", "d", "--stereo"],
            &["euroc", "--data", "d", "--imu"],
            &["mcap", "--path", "r.mcap", "--stereo"],
        ] {
            assert!(parse(removed).is_err(), "{removed:?}");
        }
    }

    #[test]
    fn shipped_configs_load() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let mut loaded = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "ron") {
                PipelineConfig::from_ron_file(&path)
                    .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
                loaded += 1;
            }
        }
        assert!(loaded > 0, "no configs in {}", dir.display());
    }

    fn shipped(name: &str) -> PipelineConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs")
            .join(name);
        PipelineConfig::from_ron_file(&path).unwrap()
    }

    /// `mono.ron` documents every default, so it must stay equal to them.
    #[test]
    fn mono_config_lists_the_defaults() {
        let defaults = PipelineConfig::default().to_ron_string().unwrap();
        let mono = shipped("mono.ron").to_ron_string().unwrap();
        assert_eq!(
            mono, defaults,
            "configs/mono.ron no longer lists the defaults"
        );
    }
}
