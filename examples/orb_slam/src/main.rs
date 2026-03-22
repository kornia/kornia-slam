//! ORB-SLAM example with minimal Rerun visualization.
//!
//! Runs an ORB-based SLAM pipeline on EuRoC MAV images, extracting ORB features
//! externally and feeding them to `process_frame`.
//!
//! ```text
//! cargo run --manifest-path examples/orb_slam/Cargo.toml -- --data /path/to/euroc/V1_01_easy
//! ```

#[path = "../../common/datasets/mod.rs"]
mod datasets;
mod config;
mod pipeline;
mod utils;

use config::PipelineConfig;
use kornia_3d::pose::Pose3d;
use kornia_io::png::read_image_png_mono8;
use kornia_slam::Frame;

use pipeline::Pipeline;

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

    // ── ORB detector (used externally before feeding Pipeline) ─────────────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: 1000,
        ..Default::default()
    };

    // ── SLAM config & system ───────────────────────────────────────────────
    let mut system = Pipeline::new(camera.clone(), PipelineConfig::default());

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
            )
            ?;
            gray_u8
                .as_slice()
                .iter()
                .zip(dst.as_slice_mut())
                .for_each(|(&s, d)| *d = s as f32 / 255.0);
            dst
        };

        // Extract ORB features.
        let features = detector.detect_and_extract(&gray_f32)?;
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }

        // Run SLAM.
        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
        };
        let result = system.process_frame(frame);
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_map_points();

        // Status line.
        let status_line = format!(
            "[{idx:>5}] {:?}  kf={:<4} pts={:<5}",
            result.status, keyframe_idx, map_point_count,
        );
        eprintln!("{status_line}");

        // Collect trajectory.
        trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));

        // Rerun logging.
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
            let map_points = system.map_points();
            log_map_points_to_rerun(rec, &map_points);
        }
    }

    let final_line = format!("Done. Final map: {} points", system.map_points().len());
    eprintln!("{final_line}");
    Ok(())
}
