//! The VIBA velocity gate (card J1): edges whose velocity residual is far
//! above the window's median cost are left out of a second solve.

use kornia_algebra::{SVecF64, Vec3F64};
use kornia_sensors::imu::PreintegratedImu;

/// More than this share of a window is not outliers but the window itself.
const MAX_DROPPED_FRACTION: f64 = 0.25;

/// `evᵀ Σ_vv⁻¹ ev`, with `Σ_vv` the velocity block of the preintegration
/// covariance (rows and columns 3..6 of `[rot, vel, pos]`); `None` when that
/// block is not positive definite or the cost is not finite.
pub(super) fn velocity_cost(ev: Vec3F64, pim: &PreintegratedImu) -> Option<f64> {
    let e = SVecF64::<3>::new(ev.x, ev.y, ev.z);
    let solved = pim
        .covariance
        .fixed_view::<3, 3>(3, 3)
        .into_owned()
        .cholesky()?
        .solve(&e);
    let cost = e.dot(&solved);
    cost.is_finite().then_some(cost)
}

/// Indices of the edges whose cost exceeds `k` times the median, worst first,
/// at most a quarter of `costs`. Edges without a cost are never dropped; a
/// `k` that is not positive drops nothing.
pub(super) fn outlier_edges(costs: &[Option<f64>], k: f64) -> Vec<usize> {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(k > 0.0) {
        return Vec::new();
    }
    let cap = (costs.len() as f64 * MAX_DROPPED_FRACTION).floor() as usize;
    let mut ranked: Vec<(usize, f64)> = costs
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.filter(|c| c.is_finite()).map(|c| (i, c)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let n = ranked.len();
    if n == 0 {
        return Vec::new();
    }
    let median = if n % 2 == 1 {
        ranked[n / 2].1
    } else {
        0.5 * (ranked[n / 2 - 1].1 + ranked[n / 2].1)
    };
    ranked
        .iter()
        .take_while(|(_, c)| *c > k * median)
        .take(cap)
        .map(|(i, _)| *i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat9F64, SVecF64};
    use kornia_sensors::imu::{ImuBias, ImuCalib};

    fn some(costs: &[f64]) -> Vec<Option<f64>> {
        costs.iter().copied().map(Some).collect()
    }

    #[test]
    fn a_gate_of_zero_drops_nothing() {
        let costs = some(&[1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 200.0, 300.0]);
        for k in [0.0, -1.0, f64::NAN] {
            assert!(outlier_edges(&costs, k).is_empty(), "k = {k}");
        }
    }

    #[test]
    fn the_gate_drops_at_most_a_quarter_worst_first() {
        let costs = some(&[1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 200.0, 300.0]);
        assert_eq!(outlier_edges(&costs, 10.0), vec![7, 6]);
    }

    #[test]
    fn fewer_than_four_edges_are_never_gated() {
        let costs = some(&[1.0, 1.0, 1000.0]);
        assert!(outlier_edges(&costs, 10.0).is_empty());
    }

    #[test]
    fn edges_without_a_cost_are_kept() {
        let costs = vec![
            Some(1.0),
            None,
            Some(1.0),
            Some(1.0),
            Some(f64::NAN),
            Some(1.0),
            Some(1.0),
            Some(500.0),
        ];
        assert_eq!(outlier_edges(&costs, 10.0), vec![7]);
    }

    #[test]
    fn velocity_cost_reads_the_velocity_block() {
        let mut pim = PreintegratedImu::new(
            ImuBias::default(),
            ImuCalib {
                gyro_noise: 1e-3,
                accel_noise: 1e-2,
                gyro_bias_noise: 1e-5,
                accel_bias_noise: 1e-4,
            },
        );
        pim.covariance = Mat9F64::from_diagonal(&SVecF64::<9>::from_column_slice(&[
            1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0,
        ]));
        let cost = velocity_cost(Vec3F64::new(2.0, 0.0, 0.0), &pim).unwrap();
        assert!((cost - 1.0).abs() < 1e-12, "{cost}");
    }
}
