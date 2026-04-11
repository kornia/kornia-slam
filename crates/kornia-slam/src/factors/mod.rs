pub mod bias_random_walk;
pub mod inertial;
pub mod reprojection;

// === Shared whitening utilities ===
//
// IMU factors use matrix-valued information (inverse covariance) for whitening,
// unlike the scalar weight used by the reprojection factor.  The helpers below
// compute the square-root information matrix via Cholesky decomposition so that
// whitening reduces to a matrix–vector multiply.

/// Compute the square-root information matrix from a covariance matrix.
///
/// Given an n×n symmetric positive-definite covariance Σ (column-major, f64):
///   1. Cholesky: Σ = L·L^T
///   2. Invert:   L^{-1}
///   3. Return:   L^{-T} as f32 (row-major) — the upper-triangular square root of Σ^{-1}.
///
/// Whitening satisfies  r_w^T · r_w = r^T · Σ^{-1} · r  when  r_w = L^{-T} · r.
pub(crate) fn sqrt_info_from_cov(cov: &[f64], n: usize) -> Vec<f32> {
    // Regularise for numerical stability (handles near-zero covariance entries).
    let mut c = cov.to_vec();
    for i in 0..n {
        c[i + i * n] += 1e-10;
    }

    let l = cholesky_lower(&c, n);
    let linv = invert_lower_tri(&l, n);

    // L^{-T} row-major:  out[i*n + j] = linv_col_major[j + i*n]
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = linv[j + i * n] as f32;
        }
    }
    out
}

/// Apply matrix whitening in-place to a residual and (optionally) its Jacobian.
///
/// `sqrt_info` is an n×n matrix stored **row-major**.
/// `residual` has length n.
/// `jacobian` (if present) has length n × total_cols, stored row-major.
pub(crate) fn whiten_matrix(
    residual: &mut [f32],
    jacobian: Option<&mut [f32]>,
    sqrt_info: &[f32],
    n: usize,
) {
    // r_w = sqrt_info · r
    let r_orig: Vec<f32> = residual[..n].to_vec();
    for i in 0..n {
        residual[i] = (0..n).map(|j| sqrt_info[i * n + j] * r_orig[j]).sum();
    }

    // J_w = sqrt_info · J  (row-wise)
    if let Some(jac) = jacobian {
        let total_cols = jac.len() / n;
        let j_orig = jac.to_vec();
        for i in 0..n {
            for col in 0..total_cols {
                jac[i * total_cols + col] = (0..n)
                    .map(|j| sqrt_info[i * n + j] * j_orig[j * total_cols + col])
                    .sum();
            }
        }
    }
}

// ── Linear-algebra helpers (f64, column-major) ───────────────────────────

/// Cholesky decomposition: mat = L·L^T.  Column-major n×n input/output.
fn cholesky_lower(mat: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            let v = l[j + k * n];
            sum += v * v;
        }
        l[j + j * n] = (mat[j + j * n] - sum).max(1e-20).sqrt();

        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i + k * n] * l[j + k * n];
            }
            l[i + j * n] = (mat[i + j * n] - s) / l[j + j * n];
        }
    }
    l
}

/// Invert a lower-triangular matrix by forward substitution.  Column-major.
fn invert_lower_tri(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0f64; n * n];
    for j in 0..n {
        inv[j + j * n] = 1.0 / l[j + j * n];
        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[i + k * n] * inv[k + j * n];
            }
            inv[i + j * n] = -sum / l[i + i * n];
        }
    }
    inv
}
