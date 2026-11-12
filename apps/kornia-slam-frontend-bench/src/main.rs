//! Replays a `kornia-slam --trace-out` recording through the visual frontend,
//! one thread, and writes per-frame process CPU time and accuracy columns.
//!
//! The trace supplies only what the backend would: the local map, the
//! predicted pose and the keyframe schedule. The bench keeps its own track
//! state from the first traced frame's tracks, like the OpenCV arm
//! (`cvfront.py`) it is compared against.

mod pipeline;
mod trace;

use std::io::Write;
use std::path::PathBuf;

use pipeline::{Keypoints, Mode, Pipeline};
use trace::Trace;

#[derive(argh::FromArgs)]
#[argh(description = "Frontend benchmark over a kornia-slam trace directory")]
struct Args {
    /// trace directory written by `kornia-slam --trace-out`
    #[argh(positional)]
    trace: PathBuf,
    /// output directory for times.txt, stages.csv and acc.csv
    #[argh(positional)]
    out: PathBuf,
    /// frontend design: today (default) or klt-first
    #[argh(option, default = "String::from(\"today\")")]
    mode: String,
    /// on trusted frames, start LK at the predicted pose over two levels
    #[argh(switch)]
    predict: bool,
    /// ORB keypoints of the primary image, as the system's frontend
    #[argh(option, default = "1000")]
    n_keypoints: usize,
    /// ORB keypoints of every other image (default: --n-keypoints)
    #[argh(option)]
    secondary_keypoints: Option<usize>,
    /// LK kernel: f32 (default) or u8 (fixed point)
    #[argh(option, default = "String::from(\"f32\")")]
    lk_kernel: String,
    /// first traced frame index to replay
    #[argh(option, default = "0")]
    first: usize,
    /// number of frames to replay
    #[argh(option, default = "usize::MAX")]
    count: usize,
}

/// Process CPU time in nanoseconds, the quantity a fixed core budget must fit.
fn process_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec and the clock id is a
    // constant the kernel always supports.
    unsafe {
        libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    let mode = Mode::parse(&args.mode).ok_or("--mode must be `today` or `klt-first`")?;
    let trace = Trace::load(&args.trace)?;
    std::fs::create_dir_all(&args.out)?;
    let frames: Vec<_> = trace
        .frames
        .iter()
        .filter(|f| f.frame_idx >= args.first)
        .take(args.count)
        .collect();
    let first = frames.first().ok_or("no traced frame in range")?;
    let keypoints = Keypoints {
        primary: args.n_keypoints,
        secondary: args.secondary_keypoints.unwrap_or(args.n_keypoints),
        lk_kernel: match args.lk_kernel.as_str() {
            "u8" => kornia_slam::tracking::optical_flow::LkKernel::FixedPointU8,
            "f32" => kornia_slam::tracking::optical_flow::LkKernel::F32,
            _ => return Err("--lk-kernel must be `f32` or `u8`".into()),
        },
    };
    let mut pipeline = Pipeline::new(
        &trace,
        mode,
        args.predict,
        keypoints,
        trace.tracks.get(&first.frame_idx),
    );

    let mut times = Vec::with_capacity(frames.len());
    let mut stages = std::io::BufWriter::new(std::fs::File::create(args.out.join("stages.csv"))?);
    writeln!(stages, "frame_idx,{}", pipeline::Stages::HEADER)?;
    let mut acc = std::io::BufWriter::new(std::fs::File::create(args.out.join("acc.csv"))?);
    writeln!(
        acc,
        "frame_idx,n_kp,n_stereo,n_seeded,n_proj,n_corr,n_inliers,rmse_px_at_trace_pose,orb_ran,trusted"
    )?;
    for frame in frames {
        let images = (0..trace.n_images())
            .map(|c| trace.image(frame.frame_idx, c))
            .collect::<Result<Vec<_>, _>>()?;
        let empty = Vec::new();
        let local = trace.local_map.get(&frame.frame_idx).unwrap_or(&empty);
        let start = process_cpu_ns();
        let out = pipeline.step(frame, &images, local)?;
        times.push(process_cpu_ns() - start);
        writeln!(stages, "{},{}", frame.frame_idx, out.stages.row())?;
        let rmse = pipeline.rmse_at(&frame.pose, local, &out);
        writeln!(
            acc,
            "{},{},{},{},{},{},{},{rmse:.4},{},{}",
            frame.frame_idx,
            out.n_kp,
            out.n_stereo,
            out.n_seeded,
            out.n_proj,
            out.correspondences.len(),
            out.inliers,
            u8::from(out.orb_ran),
            u8::from(frame.trusted_prediction),
        )?;
    }
    let text: String = times.iter().map(|t| format!("{t}\n")).collect();
    std::fs::write(args.out.join("times.txt"), text)?;
    let mut ms: Vec<f64> = times.iter().map(|&t| t as f64 / 1e6).collect();
    ms.sort_by(f64::total_cmp);
    let pick = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    println!(
        "kornia {} frames={} median={:.2} ms p95={:.2} ms",
        args.mode,
        ms.len(),
        pick(0.5),
        pick(0.95)
    );
    Ok(())
}
