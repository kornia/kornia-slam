//! EuRoC trajectory evaluation: ground-truth association, Sim3 alignment,
//! ATE/RPE/drift metrics, and CSV reporting.

use std::io::Write;
use std::path::Path;

use kornia_algebra::Vec3F64;
use kornia_slam::estimation::sim3::align_sim3;

use crate::datasets::euroc::GroundTruthPose;

// ── Association ──────────────────────────────────────────────────────────

/// Nearest ground-truth pose to timestamp `t` (by absolute time difference).
pub fn associate_gt(t: f64, gt: &[GroundTruthPose]) -> Option<&GroundTruthPose> {
    gt.iter().min_by(|a, b| {
        (a.timestamp_sec - t)
            .abs()
            .partial_cmp(&(b.timestamp_sec - t).abs())
            .unwrap()
    })
}

// ── Metrics ──────────────────────────────────────────────────────────────

pub fn compute_ate(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let sum: f64 = est
        .iter()
        .zip(gt)
        .map(|(e, g)| {
            let d = *e - *g;
            d.dot(d)
        })
        .sum();
    (sum / est.len() as f64).sqrt()
}

pub fn compute_rpe(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let sum: f64 = est
        .windows(2)
        .zip(gt.windows(2))
        .map(|(e, g)| {
            let d = (e[1] - e[0]) - (g[1] - g[0]);
            d.dot(d)
        })
        .sum();
    (sum / (est.len() - 1) as f64).sqrt()
}

pub fn compute_drift(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let diff = *est.last().unwrap() - *gt.last().unwrap();
    let final_error = diff.dot(diff).sqrt();
    let length: f64 = gt
        .windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            d.dot(d).sqrt()
        })
        .sum();
    final_error / length
}

// ── Reporting ────────────────────────────────────────────────────────────

/// Aligns the estimated trajectory to ground truth (Sim3), writes the raw and
/// aligned trajectories as CSVs under `out_dir` (created if missing), and
/// prints ATE/RPE/drift. A no-op for fewer than two samples or a length
/// mismatch.
pub fn report(est: &[Vec3F64], gt: &[Vec3F64], out_dir: &Path) -> std::io::Result<()> {
    if est.len() < 2 || gt.len() != est.len() {
        return Ok(());
    }
    std::fs::create_dir_all(out_dir)?;

    // Raw (unaligned, monocular-scale) trajectory.
    let mut raw = std::fs::File::create(out_dir.join("kornia_slam_raw.csv"))?;
    writeln!(raw, "idx,x,y,z")?;
    for (i, p) in est.iter().enumerate() {
        writeln!(raw, "{i},{:.9},{:.9},{:.9}", p.x, p.y, p.z)?;
    }

    let a = align_sim3(est, gt);
    let aligned: Vec<Vec3F64> = est
        .iter()
        .map(|p| a.translation + (a.rotation * *p) * a.scale)
        .collect();

    let mut af = std::fs::File::create(out_dir.join("kornia_slam_aligned.csv"))?;
    writeln!(af, "idx,x,y,z,gt_x,gt_y,gt_z")?;
    for (i, (p, g)) in aligned.iter().zip(gt).enumerate() {
        writeln!(
            af,
            "{i},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            p.x, p.y, p.z, g.x, g.y, g.z
        )?;
    }

    eprintln!();
    eprintln!("================ Trajectory Evaluation ================");
    eprintln!("Frames evaluated : {}", aligned.len());
    eprintln!("Scale            : {:.6}", a.scale);
    eprintln!("ATE RMSE         : {:.4} m", compute_ate(&aligned, gt));
    eprintln!("RPE RMSE         : {:.4} m", compute_rpe(&aligned, gt));
    eprintln!(
        "Final Drift      : {:.4} %",
        compute_drift(&aligned, gt) * 100.0
    );
    eprintln!("=======================================================");
    Ok(())
}
