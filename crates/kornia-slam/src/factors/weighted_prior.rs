//! Weighted Euclidean prior factor used to anchor velocity / bias variables.
//!
//! Mirrors `kornia_algebra::optim::PriorFactor` (identity Jacobian, residual
//! `x − target`) but scales both residual and Jacobian by a `weight`, so a
//! large weight can act as a near-hard constraint for gauge fixing.

use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};

/// Weighted prior factor for a Euclidean variable (R^n, local_dim == global_dim).
pub struct WeightedPriorFactor {
    target: Vec<f32>,
    weight: f32,
}

impl WeightedPriorFactor {
    pub fn new(target: Vec<f32>, weight: f32) -> Self {
        Self { target, weight }
    }
}

impl Factor for WeightedPriorFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 1 {
            return Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: params.len(),
            });
        }
        let x = params[0];
        if x.len() != self.target.len() {
            return Err(FactorError::DimensionMismatch {
                expected: self.target.len(),
                actual: x.len(),
            });
        }

        let n = x.len();
        let residual: Vec<f32> = x
            .iter()
            .zip(&self.target)
            .map(|(xi, ti)| self.weight * (xi - ti))
            .collect();

        let jacobian = if compute_jacobian {
            let mut jac = vec![0.0f32; n * n];
            for i in 0..n {
                jac[i * n + i] = self.weight;
            }
            Some(jac)
        } else {
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, n))
    }

    fn residual_dim(&self) -> usize {
        self.target.len()
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        self.target.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_residual_at_target() {
        let f = WeightedPriorFactor::new(vec![1.0, 2.0, 3.0], 10.0);
        let x = [1.0f32, 2.0, 3.0];
        let r = f.linearize(&[&x], false).unwrap();
        for v in &r.residual {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn weight_scales_residual_and_jacobian() {
        let f = WeightedPriorFactor::new(vec![0.0, 0.0], 100.0);
        let x = [1.0f32, -2.0];
        let r = f.linearize(&[&x], true).unwrap();
        assert!((r.residual[0] - 100.0).abs() < 1e-4);
        assert!((r.residual[1] + 200.0).abs() < 1e-4);
        let j = r.jacobian.unwrap();
        assert!((j[0] - 100.0).abs() < 1e-4);
        assert!((j[3] - 100.0).abs() < 1e-4);
        assert!(j[1].abs() < 1e-6 && j[2].abs() < 1e-6);
    }
}
