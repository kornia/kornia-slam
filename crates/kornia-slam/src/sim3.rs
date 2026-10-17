//! Similarity (Sim3) alignment of two 3D point sets.
//!
//! `align_sim3` delegates the plain fit to `kornia-algebra`'s f64 Umeyama.
//! `align_sim3_robust` wraps a weighted solve in a Huber-weighted IRLS loop
//! with a hard outlier rejection pass, for correspondences contaminated by
//! gross errors (e.g. AprilTag corners from a misdecoded tag).
//!
//! Frames and units are the caller's: both point sets must be expressed in the
//! same physical units, and the returned transform maps `src` coordinates onto
//! `dst` coordinates.
//!
//! `umeyama_weighted` and `align_sim3_gravity_preserving` have no equivalent in
//! kornia-rs today (upstream Umeyama is unweighted and has no robust or
//! gravity-preserving variant); both are upstream candidates.

use kornia_algebra::{Mat3F64, Vec3F64, linalg::svd::svd3_f64};
// ── Sim3 alignment ───────────────────────────────────────────────────────
/// A similarity transform `X' = s·R·X + t` mapping `src` coordinates onto
/// `dst` coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3Alignment {
    pub scale: f64,
    pub rotation: Mat3F64,
    pub translation: Vec3F64,
}

impl Sim3Alignment {
    /// Applies the similarity transform to a point.
    #[inline]
    pub fn apply(&self, p: Vec3F64) -> Vec3F64 {
        (self.rotation * p) * self.scale + self.translation
    }
}

/// Weighted Umeyama Sim3 fit mapping `src` onto `dst`.
///
/// `w` must be the same length as `src`/`dst`. Weights need not be
/// normalized; they are normalized internally by `Σw`.
fn umeyama_weighted(src: &[Vec3F64], dst: &[Vec3F64], w: &[f64]) -> Sim3Alignment {
    let sum_w: f64 = w.iter().sum();

    // Weighted centroids.
    let mut mu_src = Vec3F64::ZERO;
    let mut mu_dst = Vec3F64::ZERO;
    for i in 0..src.len() {
        mu_src += src[i] * w[i];
        mu_dst += dst[i] * w[i];
    }
    mu_src /= sum_w;
    mu_dst /= sum_w;

    // Weighted cross-covariance and weighted estimated variance.
    let mut sigma = Mat3F64::ZERO;
    let mut var_est = 0.0_f64;
    for i in 0..src.len() {
        let pe = src[i] - mu_src;
        let pg = dst[i] - mu_dst;
        sigma += Mat3F64::from_cols(pg * pe.x, pg * pe.y, pg * pe.z) * w[i];
        var_est += w[i] * pe.dot(pe);
    }
    sigma *= 1.0 / sum_w;
    var_est /= sum_w;

    // Coplanar correspondences (e.g. AprilTag corners) make the cross-covariance
    // exactly rank-deficient, whose singular decomposition has tied/zero
    // eigenvalues that the quaternion-based `svd3_f64` cannot handle robustly.
    // Add a tiny isotropic Tikhonov term (~1e-12 relative) to keep the SVD
    // well-conditioned; it is far below any 1e-9 accuracy tolerance.
    sigma += Mat3F64::IDENTITY * ((sigma.x_axis.x + sigma.y_axis.y + sigma.z_axis.z) / 3.0 * 1e-12);

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

    let translation = mu_dst - (r * mu_src) * scale;

    Sim3Alignment {
        scale,
        rotation: r,
        translation,
    }
}

/// A Sim3 alignment that cannot be performed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Sim3Error {
    /// Both point sets must carry one correspondence per point.
    #[error("Sim3 alignment needs equally sized point sets (src {src}, dst {dst})")]
    MismatchedInputLengths {
        /// Number of source points.
        src: usize,
        /// Number of destination points.
        dst: usize,
    },
    /// Both point sets are empty; at least one correspondence is required.
    #[error("Sim3 alignment needs at least one correspondence")]
    EmptyInput,
}

/// Validates the correspondence counts before either solver runs: non-empty
/// and equal. An empty pair would divide by a zero point count (0/0 centroids)
/// in both solvers, so it is refused rather than producing NaN.
fn check_lengths(src: usize, dst: usize) -> Result<(), Sim3Error> {
    if src == 0 {
        return Err(Sim3Error::EmptyInput);
    }
    if src == dst {
        Ok(())
    } else {
        Err(Sim3Error::MismatchedInputLengths { src, dst })
    }
}

/// Umeyama Sim3 fit mapping `est` onto `gt` (least-squares scale, rotation,
/// translation).
///
/// Delegates to `kornia-algebra`'s f64 Umeyama, which computes the SVD
/// numerically robustly. The weighted core below is only the fallback for the
/// SVD error that upstream API reports, so this stays infallible.
pub fn align_sim3(est: &[Vec3F64], gt: &[Vec3F64]) -> Result<Sim3Alignment, Sim3Error> {
    check_lengths(est.len(), gt.len())?;
    match kornia_algebra::linalg::rigid::umeyama_f64(est, gt, true) {
        Ok((rotation, translation, scale)) => Ok(Sim3Alignment {
            scale,
            rotation,
            translation,
        }),
        // Lengths were checked above; upstream echoing the mismatch is handled
        // explicitly rather than swallowed. Only the SVD failures are
        // recoverable and fall through to the weighted solver.
        Err(kornia_algebra::linalg::rigid::UmeyamaError::MismatchedInputLengths) => {
            Err(Sim3Error::MismatchedInputLengths {
                src: est.len(),
                dst: gt.len(),
            })
        }
        Err(_) => {
            let w = vec![1.0_f64; est.len()];
            Ok(umeyama_weighted(est, gt, &w))
        }
    }
}

/// Configuration for the Huber-weighted IRLS robust Sim3 fit.
#[derive(Debug, Clone, Copy)]
pub struct HuberIrlsConfig {
    /// Huber threshold: residuals below this get weight 1.
    pub delta: f64,
    /// Maximum number of IRLS iterations.
    pub max_iters: usize,
    /// Convergence tolerance on max relative change of scale/R/t.
    pub tol: f64,
}

impl Default for HuberIrlsConfig {
    fn default() -> Self {
        Self {
            delta: 1.0,
            max_iters: 20,
            tol: 1e-6,
        }
    }
}

/// Robust Sim3 fit mapping `est` onto `gt` using Huber-weighted IRLS on top
/// of the closed-form weighted Umeyama solve.
///
/// After the IRLS loop, points whose residual exceeds a robust threshold
/// (5.2x the median absolute deviation, i.e. ~3.5 sigma) are classified as
/// outliers and dropped; the weighted Umeyama solve is then recomputed on
/// the remaining inliers. This hard rejection prevents gross outliers from
/// biasing the final result even after the Huber downweighting.
pub fn align_sim3_robust(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    cfg: HuberIrlsConfig,
) -> Result<Sim3Alignment, Sim3Error> {
    check_lengths(est.len(), gt.len())?;
    let n = est.len();
    let mut w = vec![1.0_f64; n];
    let mut fit = umeyama_weighted(est, gt, &w);

    for _ in 0..cfg.max_iters {
        // Residuals under the current fit.
        let residuals: Vec<f64> = est
            .iter()
            .zip(gt)
            .map(|(a, b)| (fit.apply(*a) - *b).length())
            .collect();
        for i in 0..n {
            let r = residuals[i];
            w[i] = if r > 0.0 {
                (cfg.delta / r).min(1.0)
            } else {
                1.0
            };
        }

        let new_fit = umeyama_weighted(est, gt, &w);

        // Convergence check: max relative change across scale, rotation, translation.
        let scale_rel = ((new_fit.scale - fit.scale) / fit.scale.max(f64::EPSILON)).abs();

        let mut rot_rel = 0.0_f64;
        let r_diff = new_fit.rotation - fit.rotation;
        for c in 0..3 {
            let col_diff = r_diff.col(c);
            let col_old = fit.rotation.col(c);
            let d = col_diff.length();
            let o = col_old.length().max(f64::EPSILON);
            rot_rel = rot_rel.max(d / o);
        }

        let t_diff = (new_fit.translation - fit.translation).length();
        let t_old = fit.translation.length().max(f64::EPSILON);
        let t_rel = t_diff / t_old;

        let max_rel = scale_rel.max(rot_rel).max(t_rel);

        fit = new_fit;

        if max_rel < cfg.tol {
            break;
        }
    }

    // Residuals under the converged fit.
    let residuals: Vec<f64> = est
        .iter()
        .zip(gt)
        .map(|(a, b)| (fit.apply(*a) - *b).length())
        .collect();

    // Robust residual scale: 1.4826 x MAD (pseudo-sigma under Gaussian noise).
    let med = median_of(&residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    let mad = median_of(&abs_dev);
    let sigma = 1.4826 * mad;

    // Drop gross outliers (residual > 3.5 sigma) and re-fit on the inliers.
    if sigma > f64::EPSILON {
        let thr = 3.5 * sigma;
        let inliers: Vec<usize> = residuals
            .iter()
            .enumerate()
            .filter(|(_, r)| **r <= thr)
            .map(|(i, _)| i)
            .collect();

        if !inliers.is_empty() && inliers.len() < n {
            let est_in: Vec<Vec3F64> = inliers.iter().map(|&i| est[i]).collect();
            let gt_in: Vec<Vec3F64> = inliers.iter().map(|&i| gt[i]).collect();
            let w_in = vec![1.0_f64; inliers.len()];
            fit = umeyama_weighted(&est_in, &gt_in, &w_in);
        }
    }

    Ok(fit)
}

/// Fit `src` onto `dst` constrained to translation plus a yaw rotation about
/// `gravity_axis`, with scale fixed at 1.
///
/// This is ORB-SLAM3's four-degree-of-freedom inertial-correction convention:
/// an IMU-initialized world frame is gravity-aligned and metric, so a metric
/// anchor must not tilt it off gravity or re-scale it. The rotation is the
/// best in-plane rotation (2D Procrustes on the plane perpendicular to
/// gravity), which leaves the gravity direction an exact fixed point.
pub fn align_sim3_gravity_preserving(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    gravity_axis: Vec3F64,
) -> Result<Sim3Alignment, Sim3Error> {
    check_lengths(src.len(), dst.len())?;
    let g_len = gravity_axis.length();
    let g = if g_len > f64::EPSILON {
        gravity_axis / g_len
    } else {
        Vec3F64::new(0.0, 0.0, 1.0)
    };

    // An orthonormal basis (u, v, g), right-handed: v = g x u.
    let seed = if g.x.abs() <= g.y.abs() && g.x.abs() <= g.z.abs() {
        Vec3F64::new(1.0, 0.0, 0.0)
    } else if g.y.abs() <= g.z.abs() {
        Vec3F64::new(0.0, 1.0, 0.0)
    } else {
        Vec3F64::new(0.0, 0.0, 1.0)
    };
    let u_raw = seed.cross(g);
    let u = u_raw / u_raw.length().max(f64::EPSILON);
    let v = g.cross(u);

    let n = src.len() as f64;
    let mut mu_src = Vec3F64::ZERO;
    let mut mu_dst = Vec3F64::ZERO;
    for i in 0..src.len() {
        mu_src += src[i];
        mu_dst += dst[i];
    }
    mu_src /= n;
    mu_dst /= n;

    // Best in-plane rotation: 2D Procrustes of the perpendicular components.
    let mut dot = 0.0_f64;
    let mut cross_xy = 0.0_f64;
    for i in 0..src.len() {
        let a = src[i] - mu_src;
        let b = dst[i] - mu_dst;
        let (ax, ay) = (a.dot(u), a.dot(v));
        let (bx, by) = (b.dot(u), b.dot(v));
        dot += ax * bx + ay * by;
        cross_xy += ax * by - ay * bx;
    }
    let alpha = cross_xy.atan2(dot);
    let rotation = kornia_algebra::SO3F64::exp(g * alpha).matrix();
    let translation = mu_dst - rotation * mu_src;

    Ok(Sim3Alignment {
        scale: 1.0,
        rotation,
        translation,
    })
}

/// Median of a slice. Sorts a copy, so the input order never matters; an empty
/// slice returns `0`. `f64::total_cmp` gives NaN a deterministic position.
pub(crate) fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let m = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[m]
    } else {
        0.5 * (sorted[m - 1] + sorted[m])
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleRng(u64);

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            // Avoid a zero state, which would stall xorshift.
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        fn next_f64(&mut self) -> f64 {
            // Uniform in [0, 1).
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + self.next_f64() * (hi - lo)
        }
    }

    // helpers
    /// Builds a rotation matrix from an axis-angle representation via the
    /// Rodrigues formula. `axis` need not be pre-normalized.
    fn rotation_from_axis_angle(axis: Vec3F64, angle: f64) -> Mat3F64 {
        let len = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
        let (nx, ny, nz) = (axis.x / len, axis.y / len, axis.z / len);
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;

        // Row-major entries of R.
        let r00 = t * nx * nx + c;
        let r01 = t * nx * ny - s * nz;
        let r02 = t * nx * nz + s * ny;

        let r10 = t * nx * ny + s * nz;
        let r11 = t * ny * ny + c;
        let r12 = t * ny * nz - s * nx;

        let r20 = t * nx * nz - s * ny;
        let r21 = t * ny * nz + s * nx;
        let r22 = t * nz * nz + c;

        let col0 = Vec3F64::new(r00, r10, r20);
        let col1 = Vec3F64::new(r01, r11, r21);
        let col2 = Vec3F64::new(r02, r12, r22);

        Mat3F64::from_cols(col0, col1, col2)
    }

    fn random_unit_axis(rng: &mut SimpleRng) -> Vec3F64 {
        loop {
            let v = Vec3F64::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            let len2 = v.x * v.x + v.y * v.y + v.z * v.z;
            if len2 > 1e-6 {
                return v;
            }
        }
    }

    fn random_points(rng: &mut SimpleRng, n: usize) -> Vec<Vec3F64> {
        (0..n)
            .map(|_| {
                Vec3F64::new(
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                    rng.range(-10.0, 10.0),
                )
            })
            .collect()
    }

    fn apply_transform(scale: f64, r: &Mat3F64, t: Vec3F64, p: Vec3F64) -> Vec3F64 {
        (*r * p) * scale + t
    }

    fn vec3_norm(v: Vec3F64) -> f64 {
        (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
    }

    fn mat3_frobenius_diff(a: &Mat3F64, b: &Mat3F64) -> f64 {
        let d = *a - *b;
        // Sum of squares over all 9 entries via the 3 columns.
        let cols = [d.x_axis, d.y_axis, d.z_axis];
        cols.iter()
            .map(|c| c.x * c.x + c.y * c.y + c.z * c.z)
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn exact_recovery_plain_and_robust() {
        let mut rng = SimpleRng::new(42);

        let true_scale = rng.range(0.2, 5.0);
        let axis = random_unit_axis(&mut rng);
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let true_r = rotation_from_axis_angle(axis, angle);
        let true_t = Vec3F64::new(
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        );

        let n = 50;
        let src = random_points(&mut rng, n);
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|&p| apply_transform(true_scale, &true_r, true_t, p))
            .collect();

        // Plain solver.
        let fit = align_sim3(&src, &dst).unwrap();
        assert!(
            (fit.scale - true_scale).abs() < 1e-9,
            "plain scale error too large: {}",
            (fit.scale - true_scale).abs()
        );
        assert!(
            mat3_frobenius_diff(&fit.rotation, &true_r) < 1e-9,
            "plain rotation error too large"
        );
        assert!(
            vec3_norm(fit.translation - true_t) < 1e-9,
            "plain translation error too large"
        );
        for i in 0..n {
            let err = vec3_norm(fit.apply(src[i]) - dst[i]);
            assert!(err < 1e-9, "plain point {} residual too large: {}", i, err);
        }

        // Robust solver should agree exactly (no outliers, all weights -> 1).
        let robust_fit = align_sim3_robust(&src, &dst, HuberIrlsConfig::default()).unwrap();
        assert!(
            (robust_fit.scale - true_scale).abs() < 1e-9,
            "robust scale error too large: {}",
            (robust_fit.scale - true_scale).abs()
        );
        assert!(
            mat3_frobenius_diff(&robust_fit.rotation, &true_r) < 1e-9,
            "robust rotation error too large"
        );
        assert!(
            vec3_norm(robust_fit.translation - true_t) < 1e-9,
            "robust translation error too large"
        );
        for i in 0..n {
            let err = vec3_norm(robust_fit.apply(src[i]) - dst[i]);
            assert!(err < 1e-9, "robust point {} residual too large: {}", i, err);
        }
    }

    #[test]
    fn robust_outperforms_plain_with_outliers() {
        let mut rng = SimpleRng::new(7);

        let true_scale = rng.range(0.5, 2.0);
        let axis = random_unit_axis(&mut rng);
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let true_r = rotation_from_axis_angle(axis, angle);
        let true_t = Vec3F64::new(
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
        );

        let n_good = 47;
        let n_bad = 3;
        let n = n_good + n_bad;

        let src = random_points(&mut rng, n);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|&p| apply_transform(true_scale, &true_r, true_t, p))
            .collect();

        // Grossly corrupt the last 3 points' destinations.
        for d in dst.iter_mut().skip(n_good) {
            *d += Vec3F64::new(
                rng.range(40.0, 60.0),
                rng.range(40.0, 60.0),
                rng.range(40.0, 60.0),
            );
        }

        let plain_fit = align_sim3(&src, &dst).unwrap();
        let robust_fit = align_sim3_robust(&src, &dst, HuberIrlsConfig::default()).unwrap();

        // Plain solve should be visibly dragged off by the outliers.
        let plain_scale_err = (plain_fit.scale - true_scale).abs();
        assert!(
            plain_scale_err > 1e-3,
            "expected plain solve to be visibly off, got scale error {}",
            plain_scale_err
        );

        const SCALE_TOL: f64 = 1e-3;
        const ROTATION_TOL: f64 = 5e-3;
        const TRANSLATION_TOL: f64 = 1e-2;

        assert!(
            (robust_fit.scale - true_scale).abs() < SCALE_TOL,
            "robust scale error too large: {}",
            (robust_fit.scale - true_scale).abs()
        );
        assert!(
            mat3_frobenius_diff(&robust_fit.rotation, &true_r) < ROTATION_TOL,
            "robust rotation error too large: {}",
            mat3_frobenius_diff(&robust_fit.rotation, &true_r)
        );
        assert!(
            vec3_norm(robust_fit.translation - true_t) < TRANSLATION_TOL,
            "robust translation error too large: {}",
            vec3_norm(robust_fit.translation - true_t)
        );

        // Good points should have small residual under the robust fit...
        for i in 0..n_good {
            let err = vec3_norm(robust_fit.apply(src[i]) - dst[i]);
            assert!(
                err < 1e-2,
                "robust good-point {} residual too large: {}",
                i,
                err
            );
        }

        // ...while the corrupted points should still show large residuals
        // against the robust fit. Since the IRLS weight is
        // w = min(1, delta / r), a large residual here is direct evidence
        // that those points were downweighted to ~0 rather than pulling
        // the solution toward them.
        for i in n_good..n {
            let err = vec3_norm(robust_fit.apply(src[i]) - dst[i]);
            assert!(
                err > 1.0,
                "expected corrupted point {} to remain a large residual (implying near-zero weight), got {}",
                i,
                err
            );
        }
    }

    #[test]
    fn uniform_weights_match_plain_align() {
        let mut rng = SimpleRng::new(123);

        let true_scale = rng.range(0.2, 5.0);
        let axis = random_unit_axis(&mut rng);
        let angle = rng.range(0.0, std::f64::consts::TAU);
        let true_r = rotation_from_axis_angle(axis, angle);
        let true_t = Vec3F64::new(
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        );

        let n = 30;
        let src = random_points(&mut rng, n);
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|&p| apply_transform(true_scale, &true_r, true_t, p))
            .collect();

        let plain_fit = align_sim3(&src, &dst).unwrap();

        // `align_sim3` delegates to the upstream solver while the weighted core
        // is ours, so the two agree to solver precision rather than bit-exactly.
        let ones = vec![1.0_f64; n];
        let weighted_ones_fit = umeyama_weighted(&src, &dst, &ones);

        assert!((weighted_ones_fit.scale - plain_fit.scale).abs() < 1e-9);
        assert!(mat3_frobenius_diff(&weighted_ones_fit.rotation, &plain_fit.rotation) < 1e-9);
        assert!(vec3_norm(weighted_ones_fit.translation - plain_fit.translation) < 1e-9);

        // Weights must only matter up to a global scale factor: since the
        // weighted centroid, covariance, and var_est are all normalized by
        // Σw, any uniform nonzero constant reproduces the all-ones solve
        // exactly.
        let constant = vec![3.7_f64; n];
        let weighted_const_fit = umeyama_weighted(&src, &dst, &constant);

        assert!((weighted_const_fit.scale - weighted_ones_fit.scale).abs() < 1e-12);
        assert!(
            mat3_frobenius_diff(&weighted_const_fit.rotation, &weighted_ones_fit.rotation) < 1e-12
        );
        assert!(vec3_norm(weighted_const_fit.translation - weighted_ones_fit.translation) < 1e-12);
    }

    #[test]
    fn gravity_preserving_recovers_pure_yaw_and_translation() {
        let mut rng = SimpleRng::new(9);
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let g_hat = gravity / gravity.length();
        let angle = 1.2_f64;
        let yaw = rotation_from_axis_angle(g_hat, angle);
        let t = Vec3F64::new(1.0, -2.0, 0.5);

        let n = 24;
        let src = random_points(&mut rng, n);
        let dst: Vec<Vec3F64> = src.iter().map(|&p| (yaw * p) + t).collect();

        let fit = align_sim3_gravity_preserving(&src, &dst, gravity).unwrap();
        assert!((fit.scale - 1.0).abs() < 1e-12);
        assert!(mat3_frobenius_diff(&fit.rotation, &yaw) < 1e-9, "yaw error");
        assert!(vec3_norm(fit.translation - t) < 1e-9, "translation error");
        for i in 0..n {
            assert!(vec3_norm(fit.apply(src[i]) - dst[i]) < 1e-9);
        }
    }

    #[test]
    fn gravity_preserving_leaves_gravity_a_fixed_point() {
        let mut rng = SimpleRng::new(31);
        let gravity = Vec3F64::new(0.2, -0.3, -9.8);
        let g_hat = gravity / gravity.length();

        // Destinations come from an arbitrary (tilting) rotation; the fit must
        // still return a yaw-only rotation that keeps gravity fixed and scale 1.
        let tilt = rotation_from_axis_angle(Vec3F64::new(1.0, 0.5, 0.0), 0.7);
        let n = 20;
        let src = random_points(&mut rng, n);
        let dst: Vec<Vec3F64> = src.iter().map(|&p| tilt * p).collect();

        let fit = align_sim3_gravity_preserving(&src, &dst, gravity).unwrap();
        assert!((fit.scale - 1.0).abs() < 1e-12);
        let g_after = fit.rotation * g_hat;
        assert!(
            vec3_norm(g_after - g_hat) < 1e-9,
            "gravity direction moved: {}",
            vec3_norm(g_after - g_hat)
        );
    }

    #[test]
    fn median_of_sorts_its_copy_and_ignores_input_order() {
        // Unsorted on purpose: the middle element is not the median.
        assert_eq!(median_of(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median_of(&[1.0, 3.0, 5.0]), 3.0);
        assert_eq!(median_of(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_of(&[]), 0.0);
        assert_eq!(median_of(&[7.5]), 7.5);
    }

    #[test]
    fn robust_fit_rejects_outliers_with_unsorted_residual_order() {
        // Outliers come first in the arrays, so an unsorted "median" would miss
        // them and the MAD gate would keep the fit biased toward them.
        let mut rng = SimpleRng::new(5);
        let true_scale = 1.3;
        let true_r = rotation_from_axis_angle(Vec3F64::new(0.0, 1.0, 0.0), 0.8);
        let true_t = Vec3F64::new(-1.0, 0.5, 2.0);

        let n_bad = 6;
        let n_good = 24;
        let src = random_points(&mut rng, n_bad + n_good);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|&p| apply_transform(true_scale, &true_r, true_t, p))
            .collect();
        for d in dst.iter_mut().take(n_bad) {
            *d += Vec3F64::new(30.0, -25.0, 40.0);
        }

        let fit = align_sim3_robust(&src, &dst, HuberIrlsConfig::default()).unwrap();
        assert!((fit.scale - true_scale).abs() < 1e-2, "scale {}", fit.scale);
        assert!(mat3_frobenius_diff(&fit.rotation, &true_r) < 5e-2);
    }

    #[test]
    fn align_sim3_fits_coplanar_corners_exactly() {
        // A RANSAC hypothesis fits two observations = 8 coplanar tag corners.
        // The delegated upstream solver must recover the transform exactly
        // despite the rank-deficient covariance (the case the old svd3_f64
        // path needed a Tikhonov term for).
        let s = 0.05;
        let canonical = [
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(-s, s, 0.0),
            Vec3F64::new(-s, -s, 0.0),
        ];
        let src: Vec<Vec3F64> = canonical.iter().chain(canonical.iter()).copied().collect();
        let scale = 2.5;
        let r = rotation_from_axis_angle(Vec3F64::new(0.3, 1.0, 0.2), 0.9);
        let t = Vec3F64::new(1.0, -2.0, 0.5);
        let dst: Vec<Vec3F64> = src.iter().map(|&p| (r * p) * scale + t).collect();

        let fit = align_sim3(&src, &dst).unwrap();
        assert!((fit.scale - scale).abs() < 1e-9, "scale {}", fit.scale);
        assert!(mat3_frobenius_diff(&fit.rotation, &r) < 1e-9);
        assert!(vec3_norm(fit.translation - t) < 1e-9);
    }

    #[test]
    fn empty_point_sets_are_rejected_by_every_fit() {
        let empty: [Vec3F64; 0] = [];
        assert_eq!(align_sim3(&empty, &empty), Err(Sim3Error::EmptyInput));
        assert_eq!(
            align_sim3_robust(&empty, &empty, HuberIrlsConfig::default()),
            Err(Sim3Error::EmptyInput)
        );
        assert_eq!(
            align_sim3_gravity_preserving(&empty, &empty, Vec3F64::new(0.0, 0.0, -1.0)),
            Err(Sim3Error::EmptyInput)
        );
    }

    #[test]
    fn mismatched_point_counts_are_rejected_by_every_fit() {
        let a = vec![Vec3F64::ZERO; 4];
        let b = vec![Vec3F64::ZERO; 3];
        let expected = Sim3Error::MismatchedInputLengths { src: 4, dst: 3 };

        assert_eq!(align_sim3(&a, &b), Err(expected.clone()));
        assert_eq!(
            align_sim3_robust(&a, &b, HuberIrlsConfig::default()),
            Err(expected.clone())
        );
        assert_eq!(
            align_sim3_gravity_preserving(&a, &b, Vec3F64::new(0.0, 0.0, -1.0)),
            Err(expected)
        );
    }
}
