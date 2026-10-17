//! Trajectory evaluation: ground-truth association, Sim3 alignment,
//! ATE/RPE/drift metrics, and CSV reporting.

use std::io::Write;
use std::path::Path;

use kornia_algebra::Vec3F64;
use kornia_slam::sim3::align_sim3;

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

/// Registers `est` onto `gt` for scoring, returning the aligned trajectory and
/// a label describing the registration.
///
/// `anchored` marks an already-applied AprilTag metric correction: the anchor
/// put the map in the *tag* frame, which is generally not the ground-truth
/// frame, so a registration is still required. It is deliberately **rigid**
/// (scale fixed): a wrong anchor scale cannot be absorbed by a scale-1
/// transform and so still surfaces as ATE. The label reports the full-Sim3
/// scale as a diagnostic (≈1.000 when the anchor is metric-correct). Without
/// the anchor the trajectory is still in the map's arbitrary monocular frame,
/// so it takes the full best-fit Sim3.
///
/// `None` when a fit is refused (which the callers' length guards make
/// unreachable).
fn aligned_trajectory(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    anchored: bool,
) -> Option<(Vec<Vec3F64>, String)> {
    if anchored {
        let (rotation, translation, _scale) =
            kornia_algebra::linalg::rigid::umeyama_f64(est, gt, false).ok()?;
        let aligned = est.iter().map(|p| translation + rotation * *p).collect();
        let metric_scale = align_sim3(est, gt)
            .map(|a| format!("{:.6}", a.scale))
            .unwrap_or_else(|_| "n/a".to_string());
        Some((
            aligned,
            format!("anchored (metric); rigidly registered to GT (fit scale {metric_scale})"),
        ))
    } else {
        let a = align_sim3(est, gt).ok()?;
        let aligned = est
            .iter()
            .map(|p| a.translation + (a.rotation * *p) * a.scale)
            .collect();
        Some((
            aligned,
            format!("best-fit Sim3 to ground truth (scale {:.6})", a.scale),
        ))
    }
}

/// Writes the trajectory as CSVs under `out_dir` (created if missing) and
/// prints ATE/RPE/drift. A no-op for fewer than two samples or a length
/// mismatch.
///
/// `anchored` is true only when the AprilTag metric correction was actually
/// applied to the positions. See [`aligned_trajectory`] for how each case is
/// registered to ground truth.
pub fn report(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    out_dir: &Path,
    anchored: bool,
) -> std::io::Result<()> {
    if est.len() < 2 || gt.len() != est.len() {
        return Ok(());
    }
    let Some((aligned, alignment_label)) = aligned_trajectory(est, gt, anchored) else {
        return Ok(());
    };
    std::fs::create_dir_all(out_dir)?;

    // Raw trajectory as recorded (post-anchor when the anchor applied).
    let mut raw = std::fs::File::create(out_dir.join("kornia_slam_raw.csv"))?;
    writeln!(raw, "idx,x,y,z")?;
    for (i, p) in est.iter().enumerate() {
        writeln!(raw, "{i},{:.9},{:.9},{:.9}", p.x, p.y, p.z)?;
    }

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
    eprintln!("Alignment        : {alignment_label}");
    eprintln!("ATE RMSE         : {:.4} m", compute_ate(&aligned, gt));
    eprintln!("RPE RMSE         : {:.4} m", compute_rpe(&aligned, gt));
    eprintln!(
        "Final Drift      : {:.4} %",
        compute_drift(&aligned, gt) * 100.0
    );
    eprintln!("=======================================================");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    /// A metric trajectory seen in the tag frame, and the same trajectory rigidly
    /// placed in a ground-truth frame: the two differ by an unknown rigid offset
    /// only, which is the situation an anchored run is scored in.
    fn offset_pair() -> (Vec<Vec3F64>, Vec<Vec3F64>) {
        let est: Vec<Vec3F64> = (0..6)
            .map(|i| Vec3F64::new(i as f64 * 0.3, (i as f64 * 0.2).sin(), 1.0 + i as f64 * 0.1))
            .collect();
        let rotation = SO3F64::exp(Vec3F64::new(0.0, 0.5, 0.0)).matrix();
        let translation = Vec3F64::new(1.0, -2.0, 0.5);
        let gt = est.iter().map(|p| translation + rotation * *p).collect();
        (est, gt)
    }

    #[test]
    fn anchored_trajectory_is_registered_rigidly_to_ground_truth() {
        let (est, gt) = offset_pair();
        let (aligned, label) = aligned_trajectory(&est, &gt, true).expect("fit succeeds");
        for (p, g) in aligned.iter().zip(&gt) {
            assert!((*p - *g).length() < 1e-9, "rigid registration left a gap");
        }
        assert!(label.contains("anchored"), "{label}");
        // The diagnostic scale is ~1 because the offset is rigid, not a rescale.
        assert!(label.contains("1.000"), "{label}");
    }

    #[test]
    fn anchored_registration_cannot_hide_a_wrong_scale() {
        // Ground truth is the tag-frame trajectory rescaled by 1.5; a rigid
        // registration cannot absorb that, so the residual stays large.
        let (est, _) = offset_pair();
        let gt: Vec<Vec3F64> = est.iter().map(|p| *p * 1.5).collect();
        let (aligned, label) = aligned_trajectory(&est, &gt, true).expect("fit succeeds");
        let worst = aligned
            .iter()
            .zip(&gt)
            .map(|(p, g)| (*p - *g).length())
            .fold(0.0_f64, f64::max);
        assert!(worst > 0.1, "scale error must surface, got {worst}");
        assert!(label.contains("1.500"), "diagnostic scale: {label}");
    }

    #[test]
    fn unanchored_trajectory_takes_the_best_fit_sim3() {
        let (est, gt) = offset_pair();
        let (aligned, label) = aligned_trajectory(&est, &gt, false).expect("fit succeeds");
        for (p, g) in aligned.iter().zip(&gt) {
            assert!((*p - *g).length() < 1e-9);
        }
        assert!(label.contains("best-fit Sim3"), "{label}");
    }

    #[test]
    fn mismatched_lengths_are_not_scored() {
        let (est, mut gt) = offset_pair();
        gt.pop();
        assert!(aligned_trajectory(&est, &gt, true).is_none());
        assert!(aligned_trajectory(&est, &gt, false).is_none());
    }
}
