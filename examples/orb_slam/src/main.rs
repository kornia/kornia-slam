//! ORB-SLAM example with minimal Rerun visualization.
//!
//! Runs an ORB-based SLAM pipeline on EuRoC MAV images, extracting ORB features
//! externally and feeding them to `process_frame`.
//!
//! ```text
//! cargo run --manifest-path examples/orb_slam/Cargo.toml -- --data /path/to/euroc/V1_01_easy
//! ```

mod config;
#[path = "../../common/datasets/mod.rs"]
mod datasets;
mod pipeline;
mod trace;
mod utils;

use config::PipelineConfig;
use kornia_3d::pose::Pose3d;
use kornia_imgproc::features::OrbDetector;
use kornia_io::png::read_image_png_mono8;
use kornia_slam::Frame;

use pipeline::Pipeline;
use trace::TraceWriter;

use datasets::EurocDataset;

use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
    trajectory_point_from_pose,
};

/// CLI arguments.
#[derive(argh::FromArgs)]
#[argh(description = "Monocular visual odometry on EuRoC dataset")]
struct Args {
    /// path to EuRoC dataset root (e.g. V1_01_easy/)
    #[argh(option)]
    data: String,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// spawn a Rerun viewer and stream to it
    #[argh(switch)]
    rerun_stream: bool,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,

    /// output prefix for CSV trace files
    #[argh(option)]
    trace_prefix: Option<String>,

    /// optional path to an ORB-SLAM3 ORBvoc.txt vocabulary file
    #[argh(option)]
    vocabulary: Option<String>,

    /// disable SearchInNeighbors-style map-point fusion (add-only mode, on by default)
    #[argh(switch)]
    no_neighbor_fusion: bool,

    /// frames after a keyframe insert where local mapping is treated as busy
    #[argh(option, default = "0")]
    mapping_cooldown: usize,

    /// optional path to write a TUM-format trajectory (timestamp tx ty tz qx qy qz qw)
    #[argh(option)]
    trajectory: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // ── Dataset ────────────────────────────────────────────────────────────
    let dataset = EurocDataset::open(&args.data)?;
    let samples = dataset.samples();

    let n_frames = if args.max_frames > 0 {
        args.max_frames.min(samples.len() - args.start_frame)
    } else {
        samples.len() - args.start_frame
    };
    eprintln!(
        "Dataset: {} frames (processing {}..{})",
        samples.len(),
        args.start_frame,
        args.start_frame + n_frames
    );

    // ── Camera (from EuRoC cam0 sensor.yaml) ───────────────────────────────
    let camera = dataset.camera();

    let mut pipeline_config = PipelineConfig::default();
    pipeline_config.enable_neighbor_fusion = !args.no_neighbor_fusion;
    pipeline_config.mapping_cooldown_frames = args.mapping_cooldown;

    // ── ORB detectors (used externally before feeding Pipeline) ────────────
    let tracking_detector = OrbDetector {
        n_keypoints: pipeline_config.tracking_orb_keypoints,
        ..Default::default()
    };
    let initial_detector = OrbDetector {
        n_keypoints: pipeline_config.initial_orb_keypoints(),
        ..Default::default()
    };

    // ── SLAM config & system ───────────────────────────────────────────────
    let mut system = Pipeline::new(camera.clone(), pipeline_config);
    if let Some(voc_path) = args.vocabulary.as_ref() {
        eprintln!("Loading ORB-SLAM3 vocabulary from {}", voc_path);
        let t0 = std::time::Instant::now();
        let vocab = kornia_bow::orb_slam3::load_orb_slam3_vocabulary(voc_path)?;
        eprintln!(
            "Vocabulary loaded: {} blocks in {:.2}s",
            vocab.blocks.len(),
            t0.elapsed().as_secs_f32()
        );
        system = system.with_vocabulary(std::sync::Arc::new(vocab), 2);
    }
    let mut trace_writer = args
        .trace_prefix
        .as_ref()
        .map(TraceWriter::new)
        .transpose()?;

    let mut trajectory_writer = args
        .trajectory
        .as_ref()
        .map(
            |p| -> Result<std::io::BufWriter<std::fs::File>, std::io::Error> {
                Ok(std::io::BufWriter::new(std::fs::File::create(p)?))
            },
        )
        .transpose()?;

    // ── Rerun ──────────────────────────────────────────────────────────────
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("orb_slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    // ── Main loop ──────────────────────────────────────────────────────────
    let mut trajectory: Vec<[f32; 3]> = Vec::with_capacity(n_frames);

    for (i, sample) in samples
        .iter()
        .skip(args.start_frame)
        .take(n_frames)
        .enumerate()
    {
        let idx = args.start_frame + i;

        // Load grayscale image and convert u8 → f32.
        let gray_u8 = read_image_png_mono8(&sample.image_path)?;
        let image_size = gray_u8.size();
        let gray_f32 = {
            let mut dst = kornia_image::Image::from_size_val(
                image_size,
                0.0f32,
                kornia_tensor::CpuAllocator,
            )?;
            gray_u8
                .as_slice()
                .iter()
                .zip(dst.as_slice_mut())
                .for_each(|(&s, d)| *d = s as f32 / 255.0);
            dst
        };

        // Extract ORB features.
        let detector = if system.use_initial_orb_extractor(idx) {
            &initial_detector
        } else {
            &tracking_detector
        };
        let features = detector.detect_and_extract(&gray_f32)?;
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }

        // Sample pixel colors at each keypoint location.
        let keypoint_colors: Vec<[u8; 3]> = features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let x = (kp[0] as usize).min(image_size.width.saturating_sub(1));
                let y = (kp[1] as usize).min(image_size.height.saturating_sub(1));
                let g = gray_u8.as_slice()[y * image_size.width + x];
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
        };
        let feature_count = frame.features.descriptors.len();
        let timestamp_ns = (sample.timestamp_sec * 1e9).round() as i64;
        let t0 = std::time::Instant::now();
        let (result, trace) = system.process_frame_with_trace(frame);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_map_points();
        let active_map_point_count = system.num_active_map_points();

        // Status line. pts=<active>/<total>.
        let status_line = format!(
            "[{idx:>5}] {:?}  kf={:<4} pts={}/{} {frame_ms:>6.1}ms",
            result.status, keyframe_idx, active_map_point_count, map_point_count,
        );
        eprintln!("{status_line}");

        // Collect trajectory.
        trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));

        // Rerun logging.
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
            log_map_points_to_rerun(rec, system.map_points());
        }

        if let Some(writer) = trajectory_writer.as_mut() {
            use std::io::Write;
            // camera-to-world pose (standard TUM convention)
            let twc = result.pose_world_to_cam.inverse();
            let t = twc.translation;
            let q = rotmat_to_quat_xyzw(&twc.rotation);
            writeln!(
                writer,
                "{:.9} {} {} {} {} {} {} {}",
                sample.timestamp_sec, t.x, t.y, t.z, q[0], q[1], q[2], q[3]
            )?;
        }

        if let Some(writer) = trace_writer.as_mut() {
            writer.write_frame(
                idx,
                timestamp_ns,
                feature_count,
                map_point_count,
                active_map_point_count,
                system.num_keyframes(),
                result.status,
                frame_ms,
                &trace,
            )?;
        }
    }

    let final_line = format!(
        "Done. Final map: {} active / {} total points, {} keyframes",
        system.num_active_map_points(),
        system.map_points().len(),
        system.num_keyframes(),
    );
    eprintln!("{final_line}");
    Ok(())
}

fn rotmat_to_quat_xyzw(r: &kornia_algebra::Mat3F64) -> [f64; 4] {
    // glam DMat3 stores columns; reshape to row-major for the trace formula.
    let c = r.to_cols_array();
    let m = [[c[0], c[3], c[6]], [c[1], c[4], c[7]], [c[2], c[5], c[8]]];
    let trace: f64 = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s: f64 = (trace + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s: f64 = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s: f64 = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s: f64 = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    }
}
