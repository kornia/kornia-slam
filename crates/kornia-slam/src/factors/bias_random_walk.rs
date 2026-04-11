//! Bias random-walk factor.
//!
//! Constrains the change in IMU biases between two consecutive keyframes.
//! The uncertainty grows linearly with integration time (random walk model).
//!
//! Residual (6D): bias_j − bias_i
//! Jacobian (6×12): [−I₆ | I₆]

use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};
use kornia_sensors::imu::PreintegratedImu;

/// Bias random-walk factor between two keyframes.
///
/// Variables:
///   0: bias_i (R⁶, params = [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z], tangent dim = 6)
///   1: bias_j (R⁶, params = [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z], tangent dim = 6)
pub struct BiasRandomWalkFactor {
    /// Square-root information matrix (6×6, row-major).
    sqrt_info: [f32; 36],
}

impl BiasRandomWalkFactor {
    /// Construct from a `PreintegratedImu` whose `bias_covariance` encodes the
    /// accumulated random-walk uncertainty over the integration interval.
    pub fn new(pre: &PreintegratedImu) -> Self {
        let sqrt_vec = super::sqrt_info_from_cov(&pre.bias_covariance, 6);
        let mut sqrt_info = [0.0f32; 36];
        sqrt_info.copy_from_slice(&sqrt_vec);
        Self { sqrt_info }
    }
}

impl Factor for BiasRandomWalkFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 2 {
            return Err(FactorError::DimensionMismatch {
                expected: 2,
                actual: params.len(),
            });
        }

        let bias_i = params[0];
        let bias_j = params[1];

        let mut residual: Vec<f32> = (0..6).map(|i| bias_j[i] - bias_i[i]).collect();

        let jacobian = if compute_jacobian {
            let mut jac = vec![0.0f32; 6 * 12];
            for i in 0..6 {
                jac[i * 12 + i] = -1.0;     // −I block (cols 0-5)
                jac[i * 12 + 6 + i] = 1.0;  // +I block (cols 6-11)
            }
            super::whiten_matrix(&mut residual, Some(jac.as_mut_slice()), &self.sqrt_info, 6);
            Some(jac)
        } else {
            super::whiten_matrix(&mut residual, None, &self.sqrt_info, 6);
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, 12))
    }

    fn residual_dim(&self) -> usize {
        6
    }

    fn num_variables(&self) -> usize {
        2
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        match idx {
            0 | 1 => 6,
            _ => panic!("BiasRandomWalkFactor has 2 variables"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;
    use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};

    fn make_test_pre() -> PreintegratedImu {
        let calib = ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        };
        let mut pre = PreintegratedImu::new(ImuBias::default(), calib);
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81),
        };
        for _ in 0..10 {
            pre.integrate(&m, 0.01);
        }
        pre
    }

    #[test]
    fn zero_residual_same_bias() {
        let pre = make_test_pre();
        let factor = BiasRandomWalkFactor::new(&pre);

        let bias = [0.1f32, -0.05, 0.02, 0.01, -0.02, 0.03];
        let result = factor.linearize(&[&bias, &bias], false).unwrap();

        for i in 0..6 {
            assert!(
                result.residual[i].abs() < 1e-6,
                "residual[{i}] = {} (expected 0)",
                result.residual[i]
            );
        }
    }

    #[test]
    fn jacobian_is_neg_identity_plus_identity() {
        let mut pre = make_test_pre();
        // Identity bias covariance → sqrt_info = I.
        pre.bias_covariance = [0.0; 36];
        for i in 0..6 {
            pre.bias_covariance[i + i * 6] = 1.0;
        }

        let factor = BiasRandomWalkFactor::new(&pre);

        let bias_i = [0.1f32, -0.05, 0.02, 0.01, -0.02, 0.03];
        let bias_j = [0.2f32, 0.0, 0.0, 0.0, 0.0, 0.0];

        let result = factor.linearize(&[&bias_i, &bias_j], true).unwrap();
        let jac = result.jacobian.unwrap();

        for i in 0..6 {
            for j in 0..12 {
                let expected = if j == i {
                    -1.0
                } else if j == i + 6 {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (jac[i * 12 + j] - expected).abs() < 1e-5,
                    "jac[{i},{j}] = {} expected {expected}",
                    jac[i * 12 + j]
                );
            }
        }
    }

    #[test]
    fn numerical_jacobian_check() {
        let mut pre = make_test_pre();
        pre.bias_covariance = [0.0; 36];
        for i in 0..6 {
            pre.bias_covariance[i + i * 6] = 1.0;
        }

        let factor = BiasRandomWalkFactor::new(&pre);

        let bias_i = [0.01f32, -0.02, 0.03, 0.001, -0.002, 0.003];
        let bias_j = [0.02f32, -0.01, 0.01, 0.002, -0.001, 0.001];
        let params: Vec<&[f32]> = vec![&bias_i, &bias_j];

        let result = factor.linearize(&params, true).unwrap();
        let jac = result.jacobian.unwrap();

        let eps = 1e-4f32;
        let dims = [6, 6];
        let mut col_offset = 0;

        for (var_idx, &dim) in dims.iter().enumerate() {
            for d in 0..dim {
                let pp: Vec<f32> = params[var_idx]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v + if i == d { eps } else { 0.0 })
                    .collect();
                let pm: Vec<f32> = params[var_idx]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v - if i == d { eps } else { 0.0 })
                    .collect();

                let mut p_plus: Vec<Vec<f32>> = params.iter().map(|s| s.to_vec()).collect();
                p_plus[var_idx] = pp;
                let pp_refs: Vec<&[f32]> = p_plus.iter().map(|v| v.as_slice()).collect();

                let mut p_minus: Vec<Vec<f32>> = params.iter().map(|s| s.to_vec()).collect();
                p_minus[var_idx] = pm;
                let pm_refs: Vec<&[f32]> = p_minus.iter().map(|v| v.as_slice()).collect();

                let r_plus = factor.linearize(&pp_refs, false).unwrap().residual;
                let r_minus = factor.linearize(&pm_refs, false).unwrap().residual;

                for row in 0..6 {
                    let numerical = (r_plus[row] - r_minus[row]) / (2.0 * eps);
                    let analytical = jac[row * 12 + col_offset + d];
                    assert!(
                        (numerical - analytical).abs() < 1e-2,
                        "jac[{row},{}]: numerical={numerical:.4}, analytical={analytical:.4}",
                        col_offset + d
                    );
                }
            }
            col_offset += dim;
        }
    }
}
