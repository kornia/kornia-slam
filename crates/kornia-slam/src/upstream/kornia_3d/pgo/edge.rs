//! Upstream target: kornia_3d::pgo. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pgo::PgoEdge;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::{CauchyKernel, RobustKernel};
use kornia_algebra::SE3F32;

use super::{RESIDUAL_DIM, SparsePgoError, pose_to_se3, validate_se3};

/// A relative-pose edge with a full 6x6 square-root information `S`. Its residual is
/// `S * log(t_ab_meas^-1 * T_b * T_a^-1)`, ordered (translation, rotation), in camera `b`.
#[derive(Debug, Clone, Copy)]
pub struct InformationEdge {
    pub pose_a: usize,
    pub pose_b: usize,
    /// `T_b * T_a^-1` for world-to-camera `T`, as in [`PgoEdge`].
    pub t_ab_meas: SE3F32,
    /// Row-major; the information is `S^T S`.
    pub sqrt_information: [[f32; RESIDUAL_DIM]; RESIDUAL_DIM],
    /// Cauchy scale `c` on the whitened squared residual `s`, `rho(s) = c^2 ln(1 + s / c^2)`;
    /// `None` is least squares.
    pub cauchy_scale: Option<f32>,
}

pub(crate) mod sealed {
    use kornia_3d::pose::Pose3d;

    use super::{RESIDUAL_DIM, SparsePgoError};

    pub trait EdgeTerm {
        fn pose_a(&self) -> usize;
        fn pose_b(&self) -> usize;
        fn validate(&self) -> Result<(), SparsePgoError>;
        fn residual(
            &self,
            pose_a: &Pose3d,
            pose_b: &Pose3d,
        ) -> Result<[f32; RESIDUAL_DIM], SparsePgoError>;
        /// The IRLS weight at `residual`, `None` for plain least squares.
        fn robust_weight(&self, residual: &[f32; RESIDUAL_DIM]) -> Option<f32>;
        fn accumulate_cost(
            &self,
            residual: &[f32; RESIDUAL_DIM],
            cost: &mut f32,
        ) -> Result<(), SparsePgoError>;
    }
}

/// An edge the sparse pose-graph solver accepts: kornia-3d's scalar-weighted [`PgoEdge`] or an
/// [`InformationEdge`].
pub trait PoseGraphEdge: sealed::EdgeTerm {}

impl PoseGraphEdge for PgoEdge {}
impl PoseGraphEdge for InformationEdge {}

// The PgoEdge path is the pre-information solver verbatim: same residual, same per-component
// cost sum, no robust weight. `optimize_pose_graph_output_is_bit_identical_to_b1be250` holds it.
impl sealed::EdgeTerm for PgoEdge {
    fn pose_a(&self) -> usize {
        self.pose_a
    }

    fn pose_b(&self) -> usize {
        self.pose_b
    }

    fn validate(&self) -> Result<(), SparsePgoError> {
        if !self.weight.is_finite() {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph edge weight must be finite".into(),
            ));
        }
        validate_se3(&self.t_ab_meas)
    }

    fn residual(
        &self,
        pose_a: &Pose3d,
        pose_b: &Pose3d,
    ) -> Result<[f32; RESIDUAL_DIM], SparsePgoError> {
        weighted_relative_residual(pose_a, pose_b, &self.t_ab_meas, self.weight)
    }

    fn robust_weight(&self, _residual: &[f32; RESIDUAL_DIM]) -> Option<f32> {
        None
    }

    fn accumulate_cost(
        &self,
        residual: &[f32; RESIDUAL_DIM],
        cost: &mut f32,
    ) -> Result<(), SparsePgoError> {
        for value in residual {
            *cost += value * value;
            if !cost.is_finite() {
                return Err(SparsePgoError::InvalidInput(
                    "pose-graph cost must be finite".into(),
                ));
            }
        }
        Ok(())
    }
}

impl sealed::EdgeTerm for InformationEdge {
    fn pose_a(&self) -> usize {
        self.pose_a
    }

    fn pose_b(&self) -> usize {
        self.pose_b
    }

    fn validate(&self) -> Result<(), SparsePgoError> {
        if self
            .sqrt_information
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph square-root information must be finite".into(),
            ));
        }
        if let Some(scale) = self.cauchy_scale
            && !(scale.is_finite() && scale > 0.0)
        {
            return Err(SparsePgoError::InvalidInput(
                "Cauchy scale must be finite and positive".into(),
            ));
        }
        validate_se3(&self.t_ab_meas)
    }

    fn residual(
        &self,
        pose_a: &Pose3d,
        pose_b: &Pose3d,
    ) -> Result<[f32; RESIDUAL_DIM], SparsePgoError> {
        let raw = weighted_relative_residual(pose_a, pose_b, &self.t_ab_meas, 1.0)?;
        let whitened: [f32; RESIDUAL_DIM] = std::array::from_fn(|row| {
            (0..RESIDUAL_DIM)
                .map(|col| self.sqrt_information[row][col] * raw[col])
                .sum()
        });
        if whitened.iter().any(|value| !value.is_finite()) {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph residual must be finite".into(),
            ));
        }
        Ok(whitened)
    }

    fn robust_weight(&self, residual: &[f32; RESIDUAL_DIM]) -> Option<f32> {
        let scale = self.cauchy_scale?;
        let squared = residual.iter().map(|v| v * v).sum::<f32>();
        Some(CauchyKernel.weight(f64::from(squared), cauchy_c_sq(scale)) as f32)
    }

    fn accumulate_cost(
        &self,
        residual: &[f32; RESIDUAL_DIM],
        cost: &mut f32,
    ) -> Result<(), SparsePgoError> {
        match self.cauchy_scale {
            // Per component, as PgoEdge sums: LM stops on a relative cost change near f32's
            // summation noise, so a different order stops it elsewhere in a flat valley.
            None => {
                for value in residual {
                    *cost += value * value;
                }
            }
            Some(scale) => {
                let squared = residual.iter().map(|v| v * v).sum::<f32>();
                *cost += CauchyKernel.rho(f64::from(squared), cauchy_c_sq(scale)) as f32;
            }
        }
        if !cost.is_finite() {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph cost must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// `c^2` in f64, as [`CauchyKernel`] takes it.
fn cauchy_c_sq(scale: f32) -> f64 {
    f64::from(scale) * f64::from(scale)
}

pub(crate) fn weighted_relative_residual(
    pose_a: &Pose3d,
    pose_b: &Pose3d,
    measurement: &SE3F32,
    weight: f32,
) -> Result<[f32; RESIDUAL_DIM], SparsePgoError> {
    if !weight.is_finite() {
        return Err(SparsePgoError::InvalidInput(
            "pose-graph edge weight must be finite".into(),
        ));
    }
    validate_se3(measurement)?;
    let error = measurement.inverse() * (pose_to_se3(pose_b)? * pose_to_se3(pose_a)?.inverse());
    let (translation, rotation) = error.log();
    let residual = [
        weight * translation.x,
        weight * translation.y,
        weight * translation.z,
        weight * rotation.x,
        weight * rotation.y,
        weight * rotation.z,
    ];
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(SparsePgoError::InvalidInput(
            "pose-graph residual must be finite".into(),
        ));
    }
    Ok(residual)
}
