//! Monocular ORB-SLAM example with a selectable frame source.
//!
//! Run on an offline EuRoC dataset:
//! ```text
//! cargo run --release -p orb_slam -- euroc --data /path/to/V1_01_easy
//! ```
//!
//! Run on a bubbaloop MCAP recording (defaults to the mono_left channel):
//! ```text
//! cargo run --release -p orb_slam -- mcap --path /path/to/recording.mcap
//! ```
//!
//! Run live on an OAK-D camera (requires `--features oakd`):
//! ```text
//! cargo run --release -p orb_slam --features oakd -- oakd
//! ```
//!
//! Run live on a UVC camera (built-in webcam, USB cam, etc.; requires
//! `--features uvc`):
//! ```text
//! cargo run --release -p orb_slam --features uvc -- uvc \
//!     --fx 600 --fy 600 --cx 320 --cy 240
//! ```

mod config;
#[path = "../../common/datasets/mod.rs"]
mod datasets;
mod evaluation;
mod pipeline;
#[path = "../../common/source/mod.rs"]
mod source;
mod tui;
mod utils;
use crate::datasets::euroc::GroundTruthPose;
use config::{PgoPipelineConfig, PipelineConfig};
use evaluation::associate_gt;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::Frame;
use kornia_slam::map::LocalMappingMode;
use kornia_slam::stereo::{StereoMatchConfig, compute_stereo_matches};
use pipeline::{LoopClosureEvent, Pipeline};
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
#[argh(description = "Monocular ORB-SLAM (EuRoC dataset or live OAK-D)")]
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

    /// local mapping mode: sync or async
    #[argh(option, default = "LocalMappingMode::Asynchronous")]
    local_mapping: LocalMappingMode,

    /// ORB keypoints to extract per frame (default 1000; the 2 MP Hilti fisheye
    /// frames need ~3000 to bootstrap)
    #[argh(option, default = "1000")]
    n_keypoints: usize,

    /// path to a bag-of-words vocabulary (`.bin` from `convert_orbvoc`, or a
    /// DBoW2 `ORBvoc.txt`) to enable appearance-based loop detection
    #[argh(option)]
    vocab: Option<String>,

    /// apply usable pose-graph corrections to the live metric map; initialized
    /// IMU input uses gravity-preserving four-degree-of-freedom optimization
    #[argh(switch)]
    apply_pgo: bool,
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

    /// rectify the left+right cameras and compute per-keypoint stereo depth
    #[argh(switch)]
    stereo: bool,

    /// enable IMU preintegration from mav0/imu0 (mono: metric scale + gravity;
    /// stereo: gravity + velocities, scale fixed). Errors if the dataset has no IMU
    #[argh(switch)]
    imu: bool,

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
/// extraction already rotated them. These 2 MP frames need `--n-keypoints ~3000`
/// to bootstrap.
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

    /// rectify a stereo pair and compute per-keypoint depth; requires --calib
    #[argh(switch)]
    stereo: bool,

    /// right stereo channel suffix (stereo mode)
    #[argh(option, default = "String::from(\"mono_right\")")]
    right_channel: String,

    /// path to a stereo calibration YAML (required for --stereo)
    #[argh(option)]
    calib: Option<String>,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,
}

/// Run live on an OAK-D camera (CamB mono, or CamB+CamC stereo with --stereo).
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

    /// open CamB+CamC and rectify online; requires --calib
    #[argh(switch)]
    stereo: bool,

    /// path to a stereo calibration YAML (required for --stereo)
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

/// ORB pyramid scale factor (matches `OrbDetector` default `downscale`).
const ORB_SCALE: f32 = 1.2;
/// ORB pyramid level count (matches `OrbDetector` default `n_scales`).
const ORB_LEVELS: usize = 8;

// TODO: dedupe with kornia-imgproc — `OrbDetector::build_pyramid` (and helpers
// `pyramid_size_at_level` / `pyramid_reduce_u8`) implement this same 1.2-scale,
// 8-level pyramid, but they're private. Once those are exposed publicly (e.g. in
// `kornia_imgproc::pyramid`), call them here instead. Note the semantic diff:
// this builder resizes the original full-res image each level, whereas kornia
// resizes the previous level (ORB-SLAM3 behavior) — reconcile before switching.
/// Builds an ORB-consistent u8 image pyramid: level `o` is the full image
/// downscaled by `ORB_SCALE^o`, so a full-resolution keypoint at octave `o`
/// maps into level `o` by multiplying its coordinates by `ORB_SCALE^-o`.
fn build_u8_pyramid(img: &Image<u8, 1>) -> Vec<Image<u8, 1>> {
    let mut pyramid = Vec::with_capacity(ORB_LEVELS);
    pyramid.push(img.clone());
    let (w0, h0) = (img.width() as f32, img.height() as f32);
    for level in 1..ORB_LEVELS {
        let inv = 1.0 / ORB_SCALE.powi(level as i32);
        let w = ((w0 * inv).round() as usize).max(1);
        let h = ((h0 * inv).round() as usize).max(1);
        let mut dst = Image::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            0u8,
        )
        .expect("pyramid level allocation");
        resize_fast_mono(img, &mut dst, InterpolationMode::Bilinear).expect("pyramid resize");
        pyramid.push(dst);
    }
    pyramid
}

fn validate_pgo_mode(
    apply_pgo: bool,
    has_vocabulary: bool,
    has_stereo: bool,
    has_imu: bool,
) -> Result<(), &'static str> {
    if apply_pgo && !has_vocabulary {
        return Err("--apply-pgo requires --vocab");
    }
    if apply_pgo && !has_stereo && !has_imu {
        return Err("--apply-pgo requires stereo or IMU input");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // TUI is the default; --rerun-stream or --no-tui falls back to plain stderr.
    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

    // ── Source ─────────────────────────────────────────────────────────────
    let mut evaluate = false;
    let mut eval_out = String::from(".");
    let mut imu_enabled = false;
    let target_dt = Duration::from_secs_f64(1.0 / 30.0);
    let mut last_frame_walltime = Instant::now();
    let (mut source, euroc_gt): (Box<dyn FrameSource>, Option<Vec<GroundTruthPose>>) = match args
        .source
    {
        SourceCmd::Euroc(e) => {
            evaluate = e.evaluate;
            eval_out = e.eval_out.clone();
            imu_enabled = e.imu;
            let src = if e.stereo {
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
            let src = if m.stereo {
                let calib = m
                    .calib
                    .as_deref()
                    .ok_or("mcap --stereo requires --calib <stereo calibration YAML>")?;
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
            let src = if o.stereo {
                let calib = o
                    .calib
                    .as_deref()
                    .ok_or("oakd --stereo requires --calib <stereo calibration YAML>")?;
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

    let camera = source.camera();
    let n_frames_hint = source.n_frames_hint();

    // Stereo config (when the source yields rectified pairs).
    let stereo_bf = source.stereo_bf();
    // Near/far split: close points (z < baseline * TH_DEPTH) get direct stereo
    // back-projection at each keyframe (ORB-SLAM3's ThDepth ~ 35 for EuRoC).
    const TH_DEPTH: f64 = 35.0;
    let stereo_close_depth_m = stereo_bf.map(|bf| (bf / camera.fx) * TH_DEPTH);
    let stereo_config = stereo_bf.map(|bf| {
        let baseline = bf / camera.fx;
        if !tui_active {
            eprintln!(
                "Stereo: rectified fx={:.2} baseline={:.4}m bf={:.2} close_depth={:.2}m",
                camera.fx,
                baseline,
                bf,
                baseline * TH_DEPTH,
            );
        }
        StereoMatchConfig::new(baseline as f32, camera.fx as f32, ORB_SCALE, ORB_LEVELS)
    });

    if let Err(error) = validate_pgo_mode(
        args.apply_pgo,
        args.vocab.is_some(),
        stereo_config.is_some(),
        imu_enabled,
    ) {
        return Err(error.into());
    }

    // ── ORB detector ───────────────────────────────────────────────────────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: args.n_keypoints,
        ..Default::default()
    };

    // ── SLAM system ────────────────────────────────────────────────────────
    let pipeline_config = PipelineConfig {
        debug: args.debug,
        local_mapping: args.local_mapping,
        stereo_close_depth_m,
        pgo: args.apply_pgo.then(|| PgoPipelineConfig {
            require_imu_initialized: imu_enabled,
            ..PgoPipelineConfig::default()
        }),
        ..PipelineConfig::default()
    };
    let mut system = Pipeline::new(camera.clone(), pipeline_config);
    if let Some(vocab_path) = args.vocab.as_deref() {
        use kornia_slam::place_recognition::{Vocabulary, load_orb_slam3_vocabulary};
        let vocab = if vocab_path.ends_with(".txt") {
            load_orb_slam3_vocabulary(vocab_path)
                .map_err(|e| format!("failed to load text vocabulary {vocab_path}: {e}"))?
        } else {
            Vocabulary::load(vocab_path)
                .map_err(|e| format!("failed to load vocabulary {vocab_path}: {e}"))?
        };
        eprintln!("[place-recognition] loaded vocabulary from {vocab_path}");
        system.set_vocabulary(vocab);
    }
    if imu_enabled {
        match source.imu_extrinsics() {
            Some(t_bc) => system.set_imu_extrinsics(t_bc),
            None => {
                return Err(
                    "--imu requested but the source has no camera-IMU extrinsic or IMU samples"
                        .into(),
                );
            }
        }
    }

    // ── Rerun ──────────────────────────────────────────────────────────────
    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("orb_slam").spawn()?;
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
    let mut previous_image: Option<Image<u8, 1>> = None;

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
            image: gray_u8,
            right_image,
            imu_samples,
        } = item;
        let image_size = gray_u8.size();
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            rec.set_time_sequence("frame", idx as i64);
            rec.set_duration_secs("timestamp", timestamp_sec);
        }
        let imu_measurements: Vec<ImuMeasurement> = if imu_enabled {
            imu_samples
                .into_iter()
                .map(|s| ImuMeasurement {
                    timestamp: s.timestamp_sec,
                    gyro: Vec3F64::new(s.gyro[0], s.gyro[1], s.gyro[2]),
                    accel: Vec3F64::new(s.accel[0], s.accel[1], s.accel[2]),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Extract ORB features (on the raw image — for a fisheye source this is
        // the distorted frame; keypoints are undistorted below).
        let mut features = detector.detect_and_extract_u8(&gray_u8)?;

        // Stereo: match the rectified right view to fill per-keypoint depth.
        let (u_right, depth) = match (&stereo_config, &right_image) {
            (Some(cfg), Some(right_img)) => {
                let right_features = detector.detect_and_extract_u8(right_img)?;
                let left_pyr = build_u8_pyramid(&gray_u8);
                let right_pyr = build_u8_pyramid(right_img);
                let matches =
                    compute_stereo_matches(&left_pyr, &right_pyr, &features, &right_features, cfg);
                if args.debug && !tui_active {
                    let n = matches.num_matched();
                    let mut ds: Vec<f32> =
                        matches.depth.iter().copied().filter(|&d| d > 0.0).collect();
                    let med = if ds.is_empty() {
                        0.0
                    } else {
                        ds.sort_by(|a, b| a.total_cmp(b));
                        ds[ds.len() / 2]
                    };
                    eprintln!("[stereo] frame={idx} matched={n} median_depth={med:.3}m");
                }
                (matches.u_right, matches.depth)
            }
            _ => (Vec::new(), Vec::new()),
        };
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }

        // Undistort keypoints into the camera's coordinate frame. No-op for
        // pinhole/rectified sources; for the fisheye source this remaps each
        // keypoint to its virtual-pinhole pixel and drops over-wide rays, so it
        // must run before colors/Frame so all per-feature arrays stay aligned.
        source.undistort_features(&mut features);

        // Sample pixel colors at each keypoint location.
        let image_bytes = gray_u8.as_slice();
        let keypoint_colors: Vec<[u8; 3]> = features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let x = (kp[0] as usize).min(image_size.width.saturating_sub(1));
                let y = (kp[1] as usize).min(image_size.height.saturating_sub(1));
                let g = image_bytes[y * image_size.width + x];
                [g, g, g]
            })
            .collect();

        // Run SLAM.
        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors,
            u_right,
            depth,
            keypoints_undist: Vec::new(),
        };
        let t0 = std::time::Instant::now();
        let result = system.process_frame(
            frame,
            previous_image.as_ref(),
            &gray_u8,
            timestamp_sec,
            imu_measurements,
        );
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
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
                "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms",
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
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
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
        previous_image = Some(gray_u8);
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
            let n = mp.observation_kf_indices.len();
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
mod pgo_mode_tests {
    use super::validate_pgo_mode;

    #[test]
    fn apply_pgo_requires_metric_input_and_accepts_mono_imu() {
        assert_eq!(
            validate_pgo_mode(true, false, true, false).unwrap_err(),
            "--apply-pgo requires --vocab"
        );
        assert_eq!(
            validate_pgo_mode(true, true, false, false).unwrap_err(),
            "--apply-pgo requires stereo or IMU input"
        );
        assert!(validate_pgo_mode(true, true, false, true).is_ok());
        assert!(validate_pgo_mode(true, true, true, true).is_ok());
        assert!(validate_pgo_mode(true, true, true, false).is_ok());
        assert!(validate_pgo_mode(false, false, false, false).is_ok());
    }
}
