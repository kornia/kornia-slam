//! Trajectory evaluation: ground-truth association, Sim3 alignment,
//! ATE/RPE/drift metrics, per-map ATE, and CSV reporting.

use std::io::Write;
use std::path::Path;

use kornia_algebra::linalg::svd::svd3_f64;
use kornia_algebra::{Mat3F64, Vec3F64};

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

// ── Sim3 alignment ───────────────────────────────────────────────────────

pub struct Sim3Alignment {
    pub scale: f64,
    pub rotation: Mat3F64,
    pub translation: Vec3F64,
}

impl Sim3Alignment {
    pub fn apply(&self, p: Vec3F64) -> Vec3F64 {
        self.translation + (self.rotation * p) * self.scale
    }
}

/// Umeyama Sim3 fit mapping `est` onto `gt` (least-squares scale, rotation,
/// translation).
pub fn align_sim3(est: &[Vec3F64], gt: &[Vec3F64]) -> Sim3Alignment {
    let n = est.len() as f64;

    // Centroids.
    let mut mu_est = Vec3F64::ZERO;
    let mut mu_gt = Vec3F64::ZERO;
    for i in 0..est.len() {
        mu_est += est[i];
        mu_gt += gt[i];
    }
    mu_est /= n;
    mu_gt /= n;

    // Cross-covariance and estimated variance.
    let mut sigma = Mat3F64::ZERO;
    let mut var_est = 0.0_f64;
    for i in 0..est.len() {
        let pe = est[i] - mu_est;
        let pg = gt[i] - mu_gt;
        sigma += Mat3F64::from_cols(pg * pe.x, pg * pe.y, pg * pe.z);
        var_est += pe.dot(pe);
    }
    *sigma /= n;
    var_est /= n;

    let svd = svd3_f64(&sigma);
    let u = *svd.u();
    let s = *svd.s();
    let v = *svd.v();

    // Reflection correction: ensure det(U * V^T) > 0.
    let mut diag_s = Mat3F64::IDENTITY;
    if (u * v.transpose()).determinant() < 0.0 {
        diag_s.z_axis.z = -1.0;
    }

    let r = u * diag_s * v.transpose();

    // Scale: trace(S * diag_s) / var_est (S is diagonal).
    let trace =
        s.x_axis.x * diag_s.x_axis.x + s.y_axis.y * diag_s.y_axis.y + s.z_axis.z * diag_s.z_axis.z;
    let scale = trace / var_est;

    let translation = mu_gt - (r * mu_est) * scale;

    Sim3Alignment {
        scale,
        rotation: r,
        translation,
    }
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

// ── Per-map ATE ──────────────────────────────────────────────────────────

/// Shortest map that gets its own alignment: a Sim3 fit to a handful of
/// frames is near-exact and would flatter the score.
pub const MIN_SEGMENT_FRAMES: usize = 20;

/// Labels frames with the map they were tracked in, from the first frame
/// that map tracked. Bootstrap frames before it carry the previous map's
/// pose, so they get no label.
#[derive(Debug, Default)]
pub struct SegmentLabeler {
    live_map: Option<u64>,
}

impl SegmentLabeler {
    pub fn label(&mut self, map_id: u64, tracked: bool) -> Option<u64> {
        if tracked {
            self.live_map = Some(map_id);
        }
        self.live_map.filter(|&id| id == map_id)
    }
}

/// ATE with one Sim3 per map, pooled over every frame of the maps long
/// enough to align. A reset starts a new monocular gauge, which a single
/// alignment over the whole run can't fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentAte {
    pub segments: usize,
    /// Fraction of evaluated frames inside an aligned map.
    pub coverage: f64,
    pub rmse: f64,
}

pub fn compute_segment_ate(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    labels: &[Option<u64>],
) -> Option<SegmentAte> {
    let mut start = 0;
    let (mut segments, mut covered, mut sq_sum) = (0, 0, 0.0);
    for run in labels.chunk_by(|a, b| a == b) {
        let range = start..start + run.len();
        start = range.end;
        if run[0].is_none() || run.len() < MIN_SEGMENT_FRAMES {
            continue;
        }
        let (est, gt) = (&est[range.clone()], &gt[range]);
        let alignment = align_sim3(est, gt);
        sq_sum += est
            .iter()
            .zip(gt)
            .map(|(e, g)| {
                let d = alignment.apply(*e) - *g;
                d.dot(d)
            })
            .sum::<f64>();
        segments += 1;
        covered += run.len();
    }
    (covered > 0).then(|| SegmentAte {
        segments,
        coverage: covered as f64 / labels.len() as f64,
        rmse: (sq_sum / covered as f64).sqrt(),
    })
}

// ── Reporting ────────────────────────────────────────────────────────────

/// Aligns the estimated trajectory to ground truth (Sim3), writes the raw and
/// aligned trajectories as CSVs under `out_dir` (created if missing), and
/// prints ATE/RPE/drift plus the per-map ATE over `labels` (one per sample,
/// from [`SegmentLabeler`]). A no-op for fewer than two samples or a length
/// mismatch.
pub fn report(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    labels: &[Option<u64>],
    out_dir: &Path,
) -> std::io::Result<()> {
    if est.len() < 2 || gt.len() != est.len() || labels.len() != est.len() {
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
    let aligned: Vec<Vec3F64> = est.iter().map(|p| a.apply(*p)).collect();

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
    match compute_segment_ate(est, gt, labels) {
        Some(s) => {
            eprintln!(
                "Segments         : {} covering {:.1} % of frames",
                s.segments,
                s.coverage * 100.0
            );
            eprintln!("Segment ATE RMSE : {:.4} m", s.rmse);
        }
        None => eprintln!("Segment ATE RMSE : n/a (no map tracked {MIN_SEGMENT_FRAMES}+ frames)"),
    }
    eprintln!("=======================================================");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn helix(n: usize) -> Vec<Vec3F64> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                Vec3F64::new(t.cos(), t.sin(), 0.05 * t)
            })
            .collect()
    }

    fn similarity(points: &[Vec3F64], scale: f64, offset: Vec3F64) -> Vec<Vec3F64> {
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        points
            .iter()
            .map(|p| rotation * *p * scale + offset)
            .collect()
    }

    #[test]
    fn each_map_gets_its_own_alignment() {
        let gt = helix(80);
        let mut est = similarity(&gt[..40], 0.3, Vec3F64::new(1.0, 2.0, 3.0));
        est.extend(similarity(&gt[40..], 2.0, Vec3F64::new(-5.0, 0.0, 1.0)));
        let labels: Vec<_> = (0..80).map(|i| Some(u64::from(i >= 40))).collect();

        let a = align_sim3(&est, &gt);
        let global: Vec<_> = est.iter().map(|p| a.apply(*p)).collect();
        assert!(compute_ate(&global, &gt) > 0.1);

        let s = compute_segment_ate(&est, &gt, &labels).unwrap();
        assert_eq!(s.segments, 2);
        assert_eq!(s.coverage, 1.0);
        assert!(s.rmse < 1e-9, "rmse {}", s.rmse);
    }

    #[test]
    fn unlabelled_and_short_maps_are_left_out() {
        let gt = helix(60);
        let est = similarity(&gt, 0.5, Vec3F64::ZERO);
        let mut labels = vec![None; 10];
        labels.extend(vec![Some(0); MIN_SEGMENT_FRAMES - 1]);
        labels.extend(vec![Some(1); 60 - labels.len()]);

        let s = compute_segment_ate(&est, &gt, &labels).unwrap();
        assert_eq!(s.segments, 1);
        assert_eq!(
            s.coverage,
            (60 - 10 - (MIN_SEGMENT_FRAMES - 1)) as f64 / 60.0
        );
        assert!(compute_segment_ate(&est, &gt, &[None; 60]).is_none());
    }

    #[test]
    fn bootstrap_frames_after_a_reset_are_unlabelled() {
        let mut labeler = SegmentLabeler::default();
        let frames = [(0, false), (0, true), (0, false), (1, false), (1, true)];
        let labels: Vec<_> = frames
            .iter()
            .map(|&(map_id, tracked)| labeler.label(map_id, tracked))
            .collect();
        assert_eq!(labels, [None, Some(0), Some(0), None, Some(1)]);
    }
}
