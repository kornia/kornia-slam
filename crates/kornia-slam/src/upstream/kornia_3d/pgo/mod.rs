//! Upstream target: kornia_3d::pgo. Kept here until upstreamed; no kornia-slam types.

pub mod edge;
pub mod manifold;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SE3F32, Vec3F64};
use thiserror::Error;

use crate::upstream::kornia_3d::pose_ext::{
    pose_to_se3 as convert_pose_to_se3, se3_to_pose as convert_se3_to_pose,
};

pub use edge::{InformationEdge, PoseGraphEdge};
pub use manifold::{GravityManifold, PoseManifold, Se3Manifold};

pub(crate) const RESIDUAL_DIM: usize = 6;

/// Sparse pose-graph construction or solve failure.
#[derive(Debug, Error)]
pub enum SparsePgoError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("sparse matrix construction failed: {0}")]
    SparseMatrix(String),
    #[error("sparse Cholesky factorization failed: {0}")]
    Factorization(String),
}

pub(crate) fn pose_to_se3(pose: &Pose3d) -> Result<SE3F32, SparsePgoError> {
    validate_pose(pose)?;
    let se3 = convert_pose_to_se3(pose);
    validate_se3(&se3)?;
    Ok(se3)
}

pub(crate) fn se3_to_pose(se3: &SE3F32) -> Result<Pose3d, SparsePgoError> {
    validate_se3(se3)?;
    let pose = convert_se3_to_pose(se3);
    validate_pose(&pose)?;
    Ok(pose)
}

fn normalized_gravity(gravity_axis: Vec3F64) -> Result<Vec3F64, SparsePgoError> {
    if !vec3_f64_is_finite(gravity_axis) {
        return Err(SparsePgoError::InvalidInput(
            "gravity vector must be finite".into(),
        ));
    }
    let norm = gravity_axis.length();
    if !norm.is_finite() {
        return Err(SparsePgoError::InvalidInput(
            "gravity vector magnitude must be finite".into(),
        ));
    }
    if norm <= 1e-9 {
        return Err(SparsePgoError::InvalidInput(
            "gravity vector must be non-zero".into(),
        ));
    }
    Ok(gravity_axis / norm)
}

pub(crate) fn validate_delta<const DOF: usize>(delta: &[f32; DOF]) -> Result<(), SparsePgoError> {
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(SparsePgoError::InvalidInput(
            "pose delta must contain only finite values".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_pose(pose: &Pose3d) -> Result<(), SparsePgoError> {
    let rotation_is_finite = (0..3).all(|column| {
        let value = pose.rotation.col(column);
        vec3_f64_is_finite(value.into())
    });
    if !rotation_is_finite || !vec3_f64_is_finite(pose.translation) {
        return Err(SparsePgoError::InvalidInput(
            "pose must contain only finite values".into(),
        ));
    }
    Ok(())
}

fn validate_se3(se3: &SE3F32) -> Result<(), SparsePgoError> {
    let rotation = se3.r.matrix();
    let rotation_is_finite = (0..3).all(|column| {
        let value = rotation.col(column);
        [value.x, value.y, value.z].into_iter().all(f32::is_finite)
    });
    if !rotation_is_finite || ![se3.t.x, se3.t.y, se3.t.z].into_iter().all(f32::is_finite) {
        return Err(SparsePgoError::InvalidInput(
            "SE(3) transform must contain only finite values".into(),
        ));
    }
    Ok(())
}

fn vec3_f64_is_finite(value: Vec3F64) -> bool {
    [value.x, value.y, value.z].into_iter().all(f64::is_finite)
}
