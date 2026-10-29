//! The Huber objective the factors share, consistent with the IRLS weight.
//!
//! [`HuberKernel`] gives the weight `w(s) = min(1, √(k/s))` of a residual with
//! squared norm `s = rᵀΩr`. The objective whose derivative that weight is,
//! `½ρ(s)` with `ρ'(s) = w(s)`, is
//!
//! ```text
//!     ρ(s) = s               s ≤ k
//!     ρ(s) = 2√(k s) − k     s > k
//! ```
//!
//! (Ceres' `HuberLoss`), so `∂(½ρ)/∂r = w Ω r`: the gradient the normal
//! equations use. `½ w s` is not that objective (for an outlier it is
//! `½√(k s)`, whose gradient is half the weighted one), and a cost that does
//! not match the step makes LM accept and reject on the wrong numbers.

use kornia_3d::ransac::{HuberKernel, RobustKernel};

/// The Huber weight and cost `½ρ(chi2)` of one residual block with squared
/// (Mahalanobis) norm `chi2`; `None` (or a non-positive `k`, which
/// [`HuberKernel`] treats as no kernel) keeps it quadratic.
pub fn huber_weight_and_cost(chi2: f64, k: Option<f64>) -> (f64, f64) {
    match k {
        Some(k) if k > 0.0 && chi2 > k => {
            let w = HuberKernel.weight(chi2, k);
            (w, (k * chi2).sqrt() - 0.5 * k)
        }
        _ => (1.0, 0.5 * chi2),
    }
}

#[cfg(test)]
mod tests {
    use super::huber_weight_and_cost;
    use kornia_3d::ransac::{HuberKernel, RobustKernel};

    #[test]
    fn quadratic_inside_and_without_a_kernel() {
        assert_eq!(huber_weight_and_cost(4.0, None), (1.0, 2.0));
        assert_eq!(huber_weight_and_cost(4.0, Some(25.0)), (1.0, 2.0));
        assert_eq!(huber_weight_and_cost(25.0, Some(25.0)), (1.0, 12.5));
        assert_eq!(huber_weight_and_cost(100.0, Some(0.0)), (1.0, 50.0));
    }

    #[test]
    fn outlier_cost_is_the_huber_objective() {
        // s = 100, k = 25: ρ = 2·√2500 − 25 = 75, ½ρ = 37.5 (not ½·w·s = 25).
        let (w, c) = huber_weight_and_cost(100.0, Some(25.0));
        assert_eq!(w, 0.5);
        assert!((c - 37.5).abs() < 1e-12);
    }

    /// `d(½ρ(s))/ds = ½ w(s)` on both sides of the threshold and across it,
    /// so the cost's gradient in `r` is the weighted `w Ω r` the solver uses.
    #[test]
    fn cost_derivative_is_half_the_weight() {
        let k = 9.0;
        for s in [0.5, 3.0, 8.9, 9.1, 20.0, 1e3, 3.6e17] {
            let h = 1e-6 * s;
            let d = (huber_weight_and_cost(s + h, Some(k)).1
                - huber_weight_and_cost(s - h, Some(k)).1)
                / (2.0 * h);
            let w = HuberKernel.weight(s, k);
            assert!(
                (d - 0.5 * w).abs() <= 1e-6 * (0.5 * w),
                "s={s}: {d} vs {}",
                0.5 * w
            );
        }
        // Continuous at the threshold.
        let below = huber_weight_and_cost(k * (1.0 - 1e-12), Some(k)).1;
        let above = huber_weight_and_cost(k * (1.0 + 1e-12), Some(k)).1;
        assert!((below - above).abs() < 1e-9);
    }
}
