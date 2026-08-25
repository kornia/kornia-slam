use std::collections::{BTreeMap, HashSet};

use faer::prelude::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};
use kornia_3d::pgo::PgoEdge;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, SO3F64, Vec3AF32, Vec3F64};
use thiserror::Error;

// The primitives become production-reachable as the sparse assembly and
// optimizer are added in Tasks 2-5. Keep the temporary allowances item-local.
#[allow(dead_code)]
const RESIDUAL_DIM: usize = 6;

#[allow(dead_code)]
const NUM_JACOBIAN_EPS: f32 = 1e-3;

#[allow(dead_code)]
pub(crate) trait PoseManifold<const DOF: usize> {
    fn retract(&self, pose: &Pose3d, delta: &[f32; DOF]) -> Result<Pose3d, SparsePgoError>;
}

// Populated by the optimizer driver added in Task 3.
#[allow(dead_code)]
pub(crate) struct SparsePgoResult {
    pub poses: Vec<Pose3d>,
    pub iterations: usize,
    pub converged: bool,
}

#[allow(dead_code)]
struct NormalSystem<const DOF: usize> {
    pose_to_block: Vec<Option<usize>>,
    blocks: BTreeMap<(usize, usize), Vec<f32>>,
    gradient: Vec<f32>,
}

#[allow(dead_code)]
impl<const DOF: usize> NormalSystem<DOF> {
    fn scalar_dim(&self) -> Result<usize, SparsePgoError> {
        if DOF == 0 {
            return Err(SparsePgoError::InvalidInput(
                "pose manifold must have at least one degree of freedom".into(),
            ));
        }
        let block_count = self.pose_to_block.iter().flatten().count();
        let mut seen_blocks = vec![false; block_count];
        for &block in self.pose_to_block.iter().flatten() {
            if block >= block_count || seen_blocks[block] {
                return Err(SparsePgoError::InvalidInput(
                    "pose-to-block map must contain contiguous unique indices".into(),
                ));
            }
            seen_blocks[block] = true;
        }
        let scalar_dim = block_count.checked_mul(DOF).ok_or_else(|| {
            SparsePgoError::InvalidInput("normal-system dimension overflow".into())
        })?;
        let block_len = DOF.checked_mul(DOF).ok_or_else(|| {
            SparsePgoError::InvalidInput("normal-system block dimension overflow".into())
        })?;
        if self.gradient.len() != scalar_dim {
            return Err(SparsePgoError::InvalidInput(
                "normal-system gradient has an invalid dimension".into(),
            ));
        }
        if self.gradient.iter().any(|value| !value.is_finite()) {
            return Err(SparsePgoError::InvalidInput(
                "normal-system gradient must be finite".into(),
            ));
        }
        for (&(block_row, block_col), block) in &self.blocks {
            if block_row >= block_count || block_col >= block_count {
                return Err(SparsePgoError::InvalidInput(
                    "normal-system block index is out of range".into(),
                ));
            }
            if block.len() != block_len {
                return Err(SparsePgoError::InvalidInput(
                    "normal-system block has an invalid dimension".into(),
                ));
            }
            if block.iter().any(|value| !value.is_finite()) {
                return Err(SparsePgoError::InvalidInput(
                    "normal-system Hessian must be finite".into(),
                ));
            }
        }
        Ok(scalar_dim)
    }

    fn solve_damped(&self, damping: f32) -> Result<Vec<f32>, SparsePgoError> {
        let scalar_dim = self.scalar_dim()?;
        validate_damping(damping)?;
        if scalar_dim == 0 {
            return Err(SparsePgoError::InvalidInput(
                "normal system has no free poses".into(),
            ));
        }

        let triplets = self.lower_triangle_triplets(damping)?;
        let matrix =
            SparseColMat::<usize, f32>::try_new_from_triplets(scalar_dim, scalar_dim, &triplets)
                .map_err(|error| SparsePgoError::SparseMatrix(error.to_string()))?;
        let symbolic = SymbolicLlt::try_new(matrix.symbolic(), Side::Lower)
            .map_err(|error| SparsePgoError::SparseMatrix(error.to_string()))?;
        let factor = Llt::try_new_with_symbolic(symbolic, matrix.as_ref(), Side::Lower)
            .map_err(|error| SparsePgoError::Factorization(format!("{error:?}")))?;
        let mut rhs = Mat::from_fn(scalar_dim, 1, |row, _| -self.gradient[row]);
        factor.solve_in_place(&mut rhs);
        let step = (0..scalar_dim).map(|row| rhs[(row, 0)]).collect::<Vec<_>>();
        if step.iter().any(|value| !value.is_finite()) {
            return Err(SparsePgoError::Factorization(
                "sparse solve produced a non-finite step".into(),
            ));
        }
        Ok(step)
    }

    fn lower_triangle_triplets(
        &self,
        damping: f32,
    ) -> Result<Vec<Triplet<usize, usize, f32>>, SparsePgoError> {
        let scalar_dim = self.scalar_dim()?;
        validate_damping(damping)?;
        let mut entries = BTreeMap::new();
        for (&(block_row, block_col), block) in &self.blocks {
            if block_row < block_col {
                continue;
            }
            for local_row in 0..DOF {
                for local_col in 0..DOF {
                    let row = block_row * DOF + local_row;
                    let col = block_col * DOF + local_col;
                    if row >= col {
                        entries.insert((row, col), block[local_row * DOF + local_col]);
                    }
                }
            }
        }
        for diagonal in 0..scalar_dim {
            let value = entries.entry((diagonal, diagonal)).or_insert(0.0);
            *value += damping;
            if !value.is_finite() {
                return Err(SparsePgoError::InvalidInput(
                    "damped Hessian must be finite".into(),
                ));
            }
        }
        Ok(entries
            .into_iter()
            .map(|((row, col), val)| Triplet::new(row, col, val))
            .collect())
    }

    #[cfg(test)]
    fn solve_damped_dense(&self, damping: f32) -> Result<Vec<f32>, SparsePgoError> {
        let scalar_dim = self.scalar_dim()?;
        validate_damping(damping)?;
        if scalar_dim == 0 {
            return Err(SparsePgoError::InvalidInput(
                "normal system has no free poses".into(),
            ));
        }
        let mut matrix = vec![vec![0.0f64; scalar_dim]; scalar_dim];
        for (&(block_row, block_col), block) in &self.blocks {
            for local_row in 0..DOF {
                for local_col in 0..DOF {
                    matrix[block_row * DOF + local_row][block_col * DOF + local_col] =
                        block[local_row * DOF + local_col] as f64;
                }
            }
        }
        for (index, row) in matrix.iter_mut().enumerate() {
            row[index] += damping as f64;
        }
        let mut rhs = self
            .gradient
            .iter()
            .map(|value| -(*value as f64))
            .collect::<Vec<_>>();
        for pivot in 0..scalar_dim {
            let pivot_row = (pivot..scalar_dim)
                .max_by(|&left, &right| {
                    matrix[left][pivot]
                        .abs()
                        .total_cmp(&matrix[right][pivot].abs())
                })
                .unwrap();
            if matrix[pivot_row][pivot].abs() <= f64::EPSILON {
                return Err(SparsePgoError::Factorization(
                    "dense test oracle encountered a singular matrix".into(),
                ));
            }
            matrix.swap(pivot, pivot_row);
            rhs.swap(pivot, pivot_row);
            for row in (pivot + 1)..scalar_dim {
                let multiplier = matrix[row][pivot] / matrix[pivot][pivot];
                let (rows_before, current_and_after) = matrix.split_at_mut(row);
                for (current, pivot_value) in current_and_after[0][pivot..]
                    .iter_mut()
                    .zip(&rows_before[pivot][pivot..])
                {
                    *current -= multiplier * pivot_value;
                }
                rhs[row] -= multiplier * rhs[pivot];
            }
        }
        let mut step = vec![0.0f64; scalar_dim];
        for row in (0..scalar_dim).rev() {
            let known = ((row + 1)..scalar_dim)
                .map(|col| matrix[row][col] * step[col])
                .sum::<f64>();
            step[row] = (rhs[row] - known) / matrix[row][row];
        }
        if step.iter().any(|value| !value.is_finite()) {
            return Err(SparsePgoError::Factorization(
                "dense test solve produced a non-finite step".into(),
            ));
        }
        Ok(step.into_iter().map(|value| value as f32).collect())
    }
}

#[derive(Debug, Error)]
#[allow(dead_code)]
pub(crate) enum SparsePgoError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("sparse matrix construction failed: {0}")]
    SparseMatrix(String),
    #[error("sparse Cholesky factorization failed: {0}")]
    Factorization(String),
}

#[allow(dead_code)]
pub(crate) struct Se3Manifold;

impl PoseManifold<6> for Se3Manifold {
    fn retract(&self, pose: &Pose3d, delta: &[f32; 6]) -> Result<Pose3d, SparsePgoError> {
        validate_delta(delta)?;
        se3_to_pose(&pose_to_se3(pose)?.retract(delta))
    }
}

#[allow(dead_code)]
pub(crate) struct GravityManifold {
    gravity_axis: Vec3F64,
}

#[allow(dead_code)]
impl GravityManifold {
    pub(crate) fn new(gravity_axis: Vec3F64) -> Result<Self, SparsePgoError> {
        Ok(Self {
            gravity_axis: normalized_gravity(gravity_axis)?,
        })
    }
}

impl PoseManifold<4> for GravityManifold {
    fn retract(&self, pose: &Pose3d, delta: &[f32; 4]) -> Result<Pose3d, SparsePgoError> {
        validate_pose(pose)?;
        validate_delta(delta)?;

        let center = pose.inverse().translation
            + Vec3F64::new(delta[0] as f64, delta[1] as f64, delta[2] as f64);
        // Positive world-frame yaw corresponds to a negative right perturbation
        // of the world-to-camera rotation.
        let yaw = SO3F64::exp(self.gravity_axis * -(delta[3] as f64)).matrix();
        let rotation = pose.rotation * yaw;
        let retracted = Pose3d::new(rotation, -(rotation * center));
        validate_pose(&retracted)?;
        Ok(retracted)
    }
}

#[allow(dead_code)]
pub(crate) fn pose_to_se3(pose: &Pose3d) -> Result<SE3F32, SparsePgoError> {
    validate_pose(pose)?;
    let rotation = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose.rotation.col(0).x as f32,
            pose.rotation.col(0).y as f32,
            pose.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(1).x as f32,
            pose.rotation.col(1).y as f32,
            pose.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(2).x as f32,
            pose.rotation.col(2).y as f32,
            pose.rotation.col(2).z as f32,
        ),
    );
    let se3 = SE3F32::new(
        SO3F32::from_matrix(&rotation),
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    );
    validate_se3(&se3)?;
    Ok(se3)
}

#[allow(dead_code)]
pub(crate) fn se3_to_pose(se3: &SE3F32) -> Result<Pose3d, SparsePgoError> {
    validate_se3(se3)?;
    let rotation = se3.r.matrix();
    let pose = Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(
                rotation.col(0).x as f64,
                rotation.col(0).y as f64,
                rotation.col(0).z as f64,
            ),
            Vec3F64::new(
                rotation.col(1).x as f64,
                rotation.col(1).y as f64,
                rotation.col(1).z as f64,
            ),
            Vec3F64::new(
                rotation.col(2).x as f64,
                rotation.col(2).y as f64,
                rotation.col(2).z as f64,
            ),
        ),
        Vec3F64::new(se3.t.x as f64, se3.t.y as f64, se3.t.z as f64),
    );
    validate_pose(&pose)?;
    Ok(pose)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn build_normal_system<const DOF: usize>(
    poses: &[Pose3d],
    edges: &[PgoEdge],
    fixed_pose_indices: &[usize],
    manifold: &impl PoseManifold<DOF>,
) -> Result<NormalSystem<DOF>, SparsePgoError> {
    if DOF == 0 {
        return Err(SparsePgoError::InvalidInput(
            "pose manifold must have at least one degree of freedom".into(),
        ));
    }
    if poses.is_empty() {
        return Err(SparsePgoError::InvalidInput("empty poses".into()));
    }
    for pose in poses {
        validate_pose(pose)?;
    }
    let fixed = fixed_pose_indices.iter().copied().collect::<HashSet<_>>();
    if fixed.iter().any(|&pose_index| pose_index >= poses.len()) {
        return Err(SparsePgoError::InvalidInput(
            "fixed pose index is out of range".into(),
        ));
    }
    let mut pose_to_block = vec![None; poses.len()];
    let mut block_count = 0usize;
    for (pose_index, block) in pose_to_block.iter_mut().enumerate() {
        if !fixed.contains(&pose_index) {
            *block = Some(block_count);
            block_count += 1;
        }
    }
    if block_count == 0 {
        return Err(SparsePgoError::InvalidInput(
            "pose graph has no free poses".into(),
        ));
    }
    let scalar_dim = block_count
        .checked_mul(DOF)
        .ok_or_else(|| SparsePgoError::InvalidInput("normal-system dimension overflow".into()))?;
    let mut normal = NormalSystem {
        pose_to_block,
        blocks: BTreeMap::new(),
        gradient: vec![0.0; scalar_dim],
    };

    for edge in edges {
        if edge.pose_a >= poses.len() || edge.pose_b >= poses.len() {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph edge index is out of range".into(),
            ));
        }
        if edge.pose_a == edge.pose_b {
            return Err(SparsePgoError::InvalidInput(
                "pose-graph edge endpoints must be distinct".into(),
            ));
        }
        let residual = weighted_relative_residual(
            &poses[edge.pose_a],
            &poses[edge.pose_b],
            &edge.t_ab_meas,
            edge.weight,
        )?;
        let jacobian_a = normal.pose_to_block[edge.pose_a]
            .map(|_| edge_endpoint_jacobian(poses, edge, EdgeEndpoint::A, manifold))
            .transpose()?;
        let jacobian_b = normal.pose_to_block[edge.pose_b]
            .map(|_| edge_endpoint_jacobian(poses, edge, EdgeEndpoint::B, manifold))
            .transpose()?;

        if let (Some(block), Some(jacobian)) =
            (normal.pose_to_block[edge.pose_a], jacobian_a.as_ref())
        {
            accumulate_gradient::<DOF>(&mut normal.gradient, block, jacobian, &residual)?;
            accumulate_hessian::<DOF>(&mut normal.blocks, block, block, jacobian, jacobian)?;
        }
        if let (Some(block), Some(jacobian)) =
            (normal.pose_to_block[edge.pose_b], jacobian_b.as_ref())
        {
            accumulate_gradient::<DOF>(&mut normal.gradient, block, jacobian, &residual)?;
            accumulate_hessian::<DOF>(&mut normal.blocks, block, block, jacobian, jacobian)?;
        }
        if let (Some(block_a), Some(block_b), Some(jacobian_a), Some(jacobian_b)) = (
            normal.pose_to_block[edge.pose_a],
            normal.pose_to_block[edge.pose_b],
            jacobian_a.as_ref(),
            jacobian_b.as_ref(),
        ) {
            accumulate_hessian::<DOF>(
                &mut normal.blocks,
                block_a,
                block_b,
                jacobian_a,
                jacobian_b,
            )?;
            accumulate_hessian::<DOF>(
                &mut normal.blocks,
                block_b,
                block_a,
                jacobian_b,
                jacobian_a,
            )?;
        }
    }
    normal.scalar_dim()?;
    Ok(normal)
}

#[derive(Copy, Clone)]
enum EdgeEndpoint {
    A,
    B,
}

fn edge_endpoint_jacobian<const DOF: usize>(
    poses: &[Pose3d],
    edge: &PgoEdge,
    endpoint: EdgeEndpoint,
    manifold: &impl PoseManifold<DOF>,
) -> Result<Vec<f32>, SparsePgoError> {
    let pose_index = match endpoint {
        EdgeEndpoint::A => edge.pose_a,
        EdgeEndpoint::B => edge.pose_b,
    };
    let mut jacobian = vec![0.0; RESIDUAL_DIM * DOF];
    for column in 0..DOF {
        let mut delta_plus = [0.0; DOF];
        let mut delta_minus = [0.0; DOF];
        delta_plus[column] = NUM_JACOBIAN_EPS;
        delta_minus[column] = -NUM_JACOBIAN_EPS;
        let pose_plus = manifold.retract(&poses[pose_index], &delta_plus)?;
        let pose_minus = manifold.retract(&poses[pose_index], &delta_minus)?;
        let (pose_a_plus, pose_b_plus, pose_a_minus, pose_b_minus) = match endpoint {
            EdgeEndpoint::A => (
                &pose_plus,
                &poses[edge.pose_b],
                &pose_minus,
                &poses[edge.pose_b],
            ),
            EdgeEndpoint::B => (
                &poses[edge.pose_a],
                &pose_plus,
                &poses[edge.pose_a],
                &pose_minus,
            ),
        };
        let residual_plus =
            weighted_relative_residual(pose_a_plus, pose_b_plus, &edge.t_ab_meas, edge.weight)?;
        let residual_minus =
            weighted_relative_residual(pose_a_minus, pose_b_minus, &edge.t_ab_meas, edge.weight)?;
        for row in 0..RESIDUAL_DIM {
            let derivative = (residual_plus[row] - residual_minus[row]) / (2.0 * NUM_JACOBIAN_EPS);
            if !derivative.is_finite() {
                return Err(SparsePgoError::InvalidInput(
                    "pose-graph Jacobian must be finite".into(),
                ));
            }
            jacobian[row * DOF + column] = derivative;
        }
    }
    Ok(jacobian)
}

fn accumulate_gradient<const DOF: usize>(
    gradient: &mut [f32],
    block: usize,
    jacobian: &[f32],
    residual: &[f32; RESIDUAL_DIM],
) -> Result<(), SparsePgoError> {
    if jacobian.len() != RESIDUAL_DIM * DOF || (block + 1) * DOF > gradient.len() {
        return Err(SparsePgoError::InvalidInput(
            "Jacobian or gradient has an invalid dimension".into(),
        ));
    }
    for column in 0..DOF {
        let value = (0..RESIDUAL_DIM)
            .map(|row| jacobian[row * DOF + column] * residual[row])
            .sum::<f32>();
        let entry = &mut gradient[block * DOF + column];
        *entry += value;
        if !entry.is_finite() {
            return Err(SparsePgoError::InvalidInput(
                "normal-system gradient must be finite".into(),
            ));
        }
    }
    Ok(())
}

fn accumulate_hessian<const DOF: usize>(
    blocks: &mut BTreeMap<(usize, usize), Vec<f32>>,
    block_row: usize,
    block_col: usize,
    jacobian_row: &[f32],
    jacobian_col: &[f32],
) -> Result<(), SparsePgoError> {
    if jacobian_row.len() != RESIDUAL_DIM * DOF || jacobian_col.len() != RESIDUAL_DIM * DOF {
        return Err(SparsePgoError::InvalidInput(
            "Jacobian has an invalid dimension".into(),
        ));
    }
    let block = blocks
        .entry((block_row, block_col))
        .or_insert_with(|| vec![0.0; DOF * DOF]);
    if block.len() != DOF * DOF {
        return Err(SparsePgoError::InvalidInput(
            "normal-system block has an invalid dimension".into(),
        ));
    }
    for row in 0..DOF {
        for col in 0..DOF {
            let value = (0..RESIDUAL_DIM)
                .map(|residual_row| {
                    jacobian_row[residual_row * DOF + row] * jacobian_col[residual_row * DOF + col]
                })
                .sum::<f32>();
            let entry = &mut block[row * DOF + col];
            *entry += value;
            if !entry.is_finite() {
                return Err(SparsePgoError::InvalidInput(
                    "normal-system Hessian must be finite".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_damping(damping: f32) -> Result<(), SparsePgoError> {
    if !damping.is_finite() || damping < 0.0 {
        return Err(SparsePgoError::InvalidInput(
            "LM damping must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn validate_delta<const DOF: usize>(delta: &[f32; DOF]) -> Result<(), SparsePgoError> {
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(SparsePgoError::InvalidInput(
            "pose delta must contain only finite values".into(),
        ));
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_pose(pose: &Pose3d) -> Result<(), SparsePgoError> {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn vec3_f64_is_finite(value: Vec3F64) -> bool {
    [value.x, value.y, value.z].into_iter().all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use kornia_3d::pgo::PgoEdge;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

    use super::{
        GravityManifold, PoseManifold, Se3Manifold, build_normal_system, pose_to_se3,
        weighted_relative_residual,
    };

    fn assert_near(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_vec3_near(actual: Vec3F64, expected: Vec3F64, tolerance: f64) {
        assert!(
            (actual - expected).length() <= tolerance,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn three_pose_chain() -> (Vec<Pose3d>, Vec<PgoEdge>) {
        let poses = vec![
            Pose3d::IDENTITY,
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.1, 0.1, 0.0)),
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(2.3, -0.1, 0.2)),
        ];
        let expected = [
            Pose3d::IDENTITY,
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, 0.0, 0.0)),
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(2.0, 0.0, 0.0)),
        ];
        let edges = (0..2)
            .map(|pose_a| PgoEdge {
                pose_a,
                pose_b: pose_a + 1,
                t_ab_meas: pose_to_se3(&expected[pose_a + 1]).unwrap()
                    * pose_to_se3(&expected[pose_a]).unwrap().inverse(),
                weight: 1.0,
            })
            .collect();
        (poses, edges)
    }

    fn rotated_gravity_chain() -> (Vec<Pose3d>, Vec<PgoEdge>, GravityManifold) {
        let gravity_axis = Vec3F64::new(0.3, -0.8, 0.4).normalize();
        let manifold = GravityManifold::new(gravity_axis).unwrap();
        let base_rotation = SO3F64::exp(Vec3F64::new(0.2, -0.1, 0.15)).matrix();
        let poses = vec![
            Pose3d::new(base_rotation, Vec3F64::new(0.1, -0.2, 0.3)),
            Pose3d::new(
                base_rotation * SO3F64::exp(gravity_axis * 0.25).matrix(),
                Vec3F64::new(1.0, 0.15, -0.2),
            ),
            Pose3d::new(
                base_rotation * SO3F64::exp(gravity_axis * -0.18).matrix(),
                Vec3F64::new(2.1, -0.25, 0.35),
            ),
        ];
        let expected = [
            poses[0],
            manifold
                .retract(&poses[1], &[-0.08, 0.04, 0.03, 0.05])
                .unwrap(),
            manifold
                .retract(&poses[2], &[-0.12, -0.02, 0.07, -0.04])
                .unwrap(),
        ];
        let edges = (0..2)
            .map(|pose_a| PgoEdge {
                pose_a,
                pose_b: pose_a + 1,
                t_ab_meas: pose_to_se3(&expected[pose_a + 1]).unwrap()
                    * pose_to_se3(&expected[pose_a]).unwrap().inverse(),
                weight: 0.8,
            })
            .collect();
        (poses, edges, manifold)
    }

    fn normal_cost(poses: &[Pose3d], edges: &[PgoEdge]) -> f64 {
        edges
            .iter()
            .map(|edge| {
                weighted_relative_residual(
                    &poses[edge.pose_a],
                    &poses[edge.pose_b],
                    &edge.t_ab_meas,
                    edge.weight,
                )
                .unwrap()
                .into_iter()
                .map(|value| 0.5 * (value as f64).powi(2))
                .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn relative_residual_is_zero_for_matching_measurement() {
        let pose_a = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.05)).matrix(),
            Vec3F64::new(0.3, -0.4, 0.2),
        );
        let pose_b = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.08, 0.12)).matrix(),
            Vec3F64::new(1.4, 0.3, -0.5),
        );
        let measurement = pose_to_se3(&pose_b).unwrap() * pose_to_se3(&pose_a).unwrap().inverse();

        let residual = weighted_relative_residual(&pose_a, &pose_b, &measurement, 0.7).unwrap();

        assert!(residual.into_iter().all(|value| value.abs() <= 1e-5));
    }

    #[test]
    fn se3_retraction_changes_all_six_local_dofs() {
        let pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.12, -0.07, 0.09)).matrix(),
            Vec3F64::new(0.3, -0.5, 1.2),
        );
        let delta = [0.02, -0.03, 0.04, -0.05, 0.06, -0.07];

        let retracted = Se3Manifold.retract(&pose, &delta).unwrap();
        let (translation, rotation) = pose_to_se3(&pose)
            .unwrap()
            .rminus(&pose_to_se3(&retracted).unwrap());
        let actual = [
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
        ];

        for (actual, expected) in actual.into_iter().zip(delta) {
            assert_near(actual as f64, expected as f64, 1e-5);
        }
    }

    #[test]
    fn gravity_retraction_preserves_gravity_in_camera_coordinates() {
        let gravity_axis = Vec3F64::new(0.3, -0.8, 0.4).normalize();
        let rotation = SO3F64::exp(Vec3F64::new(0.2, -0.1, 0.3)).matrix();
        let center = Vec3F64::new(1.0, -2.0, 0.5);
        let pose = Pose3d::new(rotation, -(rotation * center));
        let delta = [0.25, -0.1, 0.4, 0.35];

        let retracted = GravityManifold::new(gravity_axis)
            .unwrap()
            .retract(&pose, &delta)
            .unwrap();

        assert_vec3_near(
            retracted.inverse().translation,
            center + Vec3F64::new(delta[0] as f64, delta[1] as f64, delta[2] as f64),
            1e-9,
        );
        assert_vec3_near(
            retracted.rotation * gravity_axis,
            pose.rotation * gravity_axis,
            1e-9,
        );
    }

    #[test]
    fn normal_system_contains_only_edge_induced_blocks() {
        let (poses, edges) = three_pose_chain();

        let normal = build_normal_system(&poses, &edges, &[0], &Se3Manifold).unwrap();

        assert_eq!(normal.pose_to_block, vec![None, Some(0), Some(1)]);
        assert_eq!(normal.gradient.len(), 12);
        assert_eq!(
            normal.blocks.keys().copied().collect::<Vec<_>>(),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        assert!(normal.blocks.values().all(|block| block.len() == 36));
    }

    #[test]
    fn sparse_and_dense_normal_steps_match() {
        let (poses, edges) = three_pose_chain();
        let normal = build_normal_system(&poses, &edges, &[0], &Se3Manifold).unwrap();
        let damping = 1e-2;

        let sparse = normal.solve_damped(damping).unwrap();
        let dense = normal.solve_damped_dense(damping).unwrap();

        assert_eq!(sparse.len(), dense.len());
        for (sparse, dense) in sparse.into_iter().zip(dense) {
            assert_near(sparse as f64, dense as f64, 1e-4);
        }
    }

    #[test]
    fn gravity_normal_blocks_are_symmetric_and_gradient_matches_cost_change() {
        let (poses, edges, manifold) = rotated_gravity_chain();
        let normal = build_normal_system(&poses, &edges, &[0], &manifold).unwrap();
        let h_ab = &normal.blocks[&(0, 1)];
        let h_ba = &normal.blocks[&(1, 0)];

        for row in 0..4 {
            for col in 0..4 {
                assert_near(h_ab[row * 4 + col] as f64, h_ba[col * 4 + row] as f64, 1e-6);
            }
        }

        let epsilon = 1e-3;
        let mut delta_plus = [0.0; 4];
        let mut delta_minus = [0.0; 4];
        delta_plus[0] = epsilon;
        delta_minus[0] = -epsilon;
        let mut poses_plus = poses.clone();
        let mut poses_minus = poses.clone();
        poses_plus[1] = manifold.retract(&poses[1], &delta_plus).unwrap();
        poses_minus[1] = manifold.retract(&poses[1], &delta_minus).unwrap();
        let cost_derivative = (normal_cost(&poses_plus, &edges)
            - normal_cost(&poses_minus, &edges))
            / (2.0 * epsilon as f64);

        assert_near(normal.gradient[0] as f64, cost_derivative, 2e-4);
    }
}
