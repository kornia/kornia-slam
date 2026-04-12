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
mod utils;

use config::PipelineConfig;
use kornia_3d::pose::Pose3d;
use kornia_io::png::read_image_png_mono8;
use kornia_slam::Frame;

use pipeline::Pipeline;

use datasets::EurocDataset;

use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_imu_covariance_to_rerun, log_imu_to_rerun,
    log_map_points_to_rerun, log_trajectory_to_rerun, trajectory_point_from_pose,
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

    /// enable IMU fusion (requires imu0/ in dataset)
    #[argh(switch)]
    imu: bool,
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

    // ── IMU ────────────────────────────────────────────────────────────
    // Always load IMU samples for visualization; only enable fusion with --imu.
    let imu_samples = dataset.imu_samples();
    if args.imu {
        if let Some(imu_calib) = dataset.imu_calib() {
            system.enable_imu(imu_calib);
            eprintln!("IMU enabled: {} samples", imu_samples.len());
        } else {
            eprintln!("Warning: --imu requested but no imu0/sensor.yaml found");
        }
    } else if !imu_samples.is_empty() {
        eprintln!("IMU data available ({} samples), pass --imu to enable fusion", imu_samples.len());
    }
    let mut imu_cursor = 0;

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

        // Advance IMU cursor up to this camera frame's timestamp.
        let imu_batch_start = imu_cursor;
        while imu_cursor < imu_samples.len()
            && imu_samples[imu_cursor].timestamp_sec <= sample.timestamp_sec
        {
            imu_cursor += 1;
        }

        // Feed IMU samples to pipeline (only when fusion is enabled)
        if args.imu {
            for s in &imu_samples[imu_batch_start..imu_cursor] {
                system.push_imu(s.to_imu_measurement());
            }
        }

        // Log IMU data to Rerun
        if let Some(ref rec) = rec {
            log_imu_to_rerun(rec, &imu_samples[imu_batch_start..imu_cursor]);
        }

        // Run SLAM.
        let frame = Frame {
            idx,
            timestamp: sample.timestamp_sec,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors,
        };
        let t0 = std::time::Instant::now();
        let result = system.process_frame(frame);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_map_points();

        // Status line.
        let status_line = format!(
            "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms",
            result.status, keyframe_idx, map_point_count,
        );
        eprintln!("{status_line}");

        // Collect trajectory.
        trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));

        // Rerun logging.
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
            log_map_points_to_rerun(rec, system.map_points());
            if args.imu {
                if let Some(pre) = system.latest_preintegrated_imu() {
                    log_imu_covariance_to_rerun(rec, pre);
                }
            }
        }
    }

    let final_line = format!("Done. Final map: {} points", system.map_points().len());
    eprintln!("{final_line}");
    Ok(())
}
