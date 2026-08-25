//! Gravity-preserving four-degree-of-freedom pose-graph optimization.

use std::collections::HashSet;

use kornia_3d::pgo::{PgoEdge, PgoParams};
use kornia_3d::pose::Pose3d;
use kornia_algebra::optim::{
    Factor, FactorError, FactorResult, LevenbergMarquardt, LinearizationResult, OptimizerError,
    Problem, ProblemError, TerminationReason, Variable, VariableType,
};
use kornia_algebra::{Mat3AF32, SE3F32, SO3F32, SO3F64, Vec3AF32, Vec3F64};
use thiserror::Error;

const STATE_DIM: usize = 4;
const RESIDUAL_DIM: usize = 6;
const NUM_JACOBIAN_EPS: f32 = 1e-3;

type FactorStates<'a> = (Option<&'a [f32]>, Option<&'a [f32]>);

#[derive(Debug, Error)]
pub(crate) enum GravityPgoError {
    #[error("problem setup error: {0}")]
    Problem(#[from] ProblemError),
    #[error("optimizer error: {0}")]
    Optimizer(#[from] OptimizerError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub(crate) struct GravityPgoResult {
    pub poses: Vec<Pose3d>,
    pub converged: bool,
}

fn normalized_gravity(gravity_world: Vec3F64) -> Result<Vec3F64, GravityPgoError> {
    if ![gravity_world.x, gravity_world.y, gravity_world.z]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err(GravityPgoError::InvalidInput(
            "gravity vector must be finite".into(),
        ));
    }
    let norm = gravity_world.length();
    if norm <= 1e-9 {
        return Err(GravityPgoError::InvalidInput(
            "gravity vector must be non-zero".into(),
        ));
    }
    Ok(gravity_world / norm)
}

fn state_from_pose(pose: &Pose3d) -> Vec<f32> {
    let center = pose.inverse().translation;
    vec![center.x as f32, center.y as f32, center.z as f32, 0.0]
}

fn pose_from_state(
    original: &Pose3d,
    gravity_axis: Vec3F64,
    state: &[f32],
) -> Result<Pose3d, GravityPgoError> {
    if state.len() != STATE_DIM || state.iter().any(|value| !value.is_finite()) {
        return Err(GravityPgoError::InvalidInput(
            "4-DoF pose state must contain four finite values".into(),
        ));
    }
    let gravity_axis = normalized_gravity(gravity_axis)?;
    let center = Vec3F64::new(state[0] as f64, state[1] as f64, state[2] as f64);
    let yaw_world = SO3F64::exp(gravity_axis * -(state[3] as f64)).matrix();
    let rotation = original.rotation * yaw_world;
    Ok(Pose3d::new(rotation, -(rotation * center)))
}

fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
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
    SE3F32::new(
        SO3F32::from_matrix(&rotation),
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    )
}

struct RelPose4DofFactor {
    original_a: Pose3d,
    original_b: Pose3d,
    gravity_axis: Vec3F64,
    measurement_inv: SE3F32,
    weight: f32,
    free_a: bool,
    free_b: bool,
}

impl RelPose4DofFactor {
    fn residual_at(
        &self,
        state_a: Option<&[f32]>,
        state_b: Option<&[f32]>,
    ) -> FactorResult<[f32; RESIDUAL_DIM]> {
        let pose_a = match state_a {
            Some(state) => pose_from_state(&self.original_a, self.gravity_axis, state),
            None => Ok(self.original_a),
        }
        .map_err(|error| FactorError::InvalidParameters(error.to_string()))?;
        let pose_b = match state_b {
            Some(state) => pose_from_state(&self.original_b, self.gravity_axis, state),
            None => Ok(self.original_b),
        }
        .map_err(|error| FactorError::InvalidParameters(error.to_string()))?;
        let error = self.measurement_inv * (pose_to_se3(&pose_b) * pose_to_se3(&pose_a).inverse());
        let (translation, rotation) = error.log();
        Ok([
            self.weight * translation.x,
            self.weight * translation.y,
            self.weight * translation.z,
            self.weight * rotation.x,
            self.weight * rotation.y,
            self.weight * rotation.z,
        ])
    }

    fn states<'a>(&self, params: &'a [&'a [f32]]) -> FactorResult<FactorStates<'a>> {
        match (self.free_a, self.free_b, params) {
            (true, true, [a, b]) => Ok((Some(*a), Some(*b))),
            (true, false, [a]) => Ok((Some(*a), None)),
            (false, true, [b]) => Ok((None, Some(*b))),
            _ => Err(FactorError::DimensionMismatch {
                expected: usize::from(self.free_a) + usize::from(self.free_b),
                actual: params.len(),
            }),
        }
    }
}

impl Factor for RelPose4DofFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let (state_a, state_b) = self.states(params)?;
        let residual = self.residual_at(state_a, state_b)?.to_vec();
        let free_count = usize::from(self.free_a) + usize::from(self.free_b);
        let total_dim = free_count * STATE_DIM;
        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, total_dim));
        }

        let mut jacobian = vec![0.0; RESIDUAL_DIM * total_dim];
        let mut column_base = 0;
        if let Some(state) = state_a {
            numerical_jacobian_block(&mut jacobian, total_dim, column_base, state, |candidate| {
                self.residual_at(Some(candidate), state_b)
            })?;
            column_base += STATE_DIM;
        }
        if let Some(state) = state_b {
            numerical_jacobian_block(&mut jacobian, total_dim, column_base, state, |candidate| {
                self.residual_at(state_a, Some(candidate))
            })?;
        }
        Ok(LinearizationResult::new(
            residual,
            Some(jacobian),
            total_dim,
        ))
    }

    fn residual_dim(&self) -> usize {
        RESIDUAL_DIM
    }

    fn num_variables(&self) -> usize {
        usize::from(self.free_a) + usize::from(self.free_b)
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        STATE_DIM
    }
}

fn numerical_jacobian_block(
    jacobian: &mut [f32],
    total_dim: usize,
    column_base: usize,
    state: &[f32],
    residual: impl Fn(&[f32]) -> FactorResult<[f32; RESIDUAL_DIM]>,
) -> FactorResult<()> {
    for column in 0..STATE_DIM {
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[column] += NUM_JACOBIAN_EPS;
        minus[column] -= NUM_JACOBIAN_EPS;
        let residual_plus = residual(&plus)?;
        let residual_minus = residual(&minus)?;
        for row in 0..RESIDUAL_DIM {
            jacobian[row * total_dim + column_base + column] =
                (residual_plus[row] - residual_minus[row]) / (2.0 * NUM_JACOBIAN_EPS);
        }
    }
    Ok(())
}

pub(crate) fn gravity_pose_graph_optimize(
    poses: &[Pose3d],
    edges: &[PgoEdge],
    fixed_pose_indices: &[usize],
    gravity_world: Vec3F64,
    params: &PgoParams,
) -> Result<GravityPgoResult, GravityPgoError> {
    if poses.is_empty() {
        return Err(GravityPgoError::InvalidInput("empty poses".into()));
    }
    if edges.is_empty() {
        return Err(GravityPgoError::InvalidInput("empty edges".into()));
    }
    let gravity_axis = normalized_gravity(gravity_world)?;
    for edge in edges {
        if edge.pose_a >= poses.len() || edge.pose_b >= poses.len() {
            return Err(GravityPgoError::InvalidInput(
                "edge keyframe index is out of range".into(),
            ));
        }
        if edge.pose_a == edge.pose_b {
            return Err(GravityPgoError::InvalidInput(
                "pose-graph edge endpoints must be distinct".into(),
            ));
        }
    }

    let fixed: HashSet<_> = fixed_pose_indices.iter().copied().collect();
    if fixed.iter().any(|&index| index >= poses.len()) {
        return Err(GravityPgoError::InvalidInput(
            "fixed keyframe index is out of range".into(),
        ));
    }
    let mut is_free = vec![false; poses.len()];
    for edge in edges {
        if !fixed.contains(&edge.pose_a) {
            is_free[edge.pose_a] = true;
        }
        if !fixed.contains(&edge.pose_b) {
            is_free[edge.pose_b] = true;
        }
    }
    if !is_free.iter().any(|&free| free) {
        return Err(GravityPgoError::InvalidInput(
            "pose graph has no free keyframes".into(),
        ));
    }

    let mut problem = Problem::new();
    for (index, pose) in poses.iter().enumerate() {
        if is_free[index] {
            problem.add_variable(
                Variable::new(
                    format!("pose_{index}"),
                    VariableType::Euclidean(STATE_DIM),
                    vec![0.0; STATE_DIM],
                ),
                state_from_pose(pose),
            )?;
        }
    }
    for edge in edges {
        let free_a = is_free[edge.pose_a];
        let free_b = is_free[edge.pose_b];
        if !free_a && !free_b {
            continue;
        }
        let factor = RelPose4DofFactor {
            original_a: poses[edge.pose_a],
            original_b: poses[edge.pose_b],
            gravity_axis,
            measurement_inv: edge.t_ab_meas.inverse(),
            weight: edge.weight,
            free_a,
            free_b,
        };
        let mut variable_names = Vec::with_capacity(2);
        if free_a {
            variable_names.push(format!("pose_{}", edge.pose_a));
        }
        if free_b {
            variable_names.push(format!("pose_{}", edge.pose_b));
        }
        problem.add_factor(Box::new(factor), variable_names)?;
    }

    let optimizer = LevenbergMarquardt {
        lambda_init: params.initial_lambda,
        lambda_max: 1e10,
        lambda_factor: 10.0,
        max_iterations: params.max_iterations,
        cost_tolerance: params.cost_tolerance,
        gradient_tolerance: params.gradient_tolerance,
    };
    let result = optimizer.optimize(&mut problem)?;
    let variables = problem.get_variables();
    let mut optimized_poses = Vec::with_capacity(poses.len());
    for (index, pose) in poses.iter().enumerate() {
        if is_free[index] {
            optimized_poses.push(pose_from_state(
                pose,
                gravity_axis,
                &variables[&format!("pose_{index}")].values,
            )?);
        } else {
            optimized_poses.push(*pose);
        }
    }
    Ok(GravityPgoResult {
        poses: optimized_poses,
        converged: matches!(
            result.termination_reason,
            TerminationReason::CostConverged | TerminationReason::GradientConverged
        ),
    })
}

#[cfg(test)]
mod tests {
    use kornia_3d::pgo::{PgoEdge, PgoParams};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};

    use super::{gravity_pose_graph_optimize, pose_from_state, pose_to_se3};

    fn assert_vec3_near(actual: Vec3F64, expected: Vec3F64, tolerance: f64) {
        assert!(
            (actual - expected).length() <= tolerance,
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn pose_state_identity_preserves_original_pose() {
        let original = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.2, -0.1, 0.3)).matrix(),
            Vec3F64::new(1.0, -2.0, 0.5),
        );
        let center = original.inverse().translation;
        let state = [center.x as f32, center.y as f32, center.z as f32, 0.0];

        let reconstructed =
            pose_from_state(&original, Vec3F64::new(0.0, 1.0, 0.0), &state).unwrap();

        assert_vec3_near(reconstructed.translation, original.translation, 1e-6);
        for column in 0..3 {
            assert_vec3_near(
                reconstructed.rotation.col(column).into(),
                original.rotation.col(column).into(),
                1e-6,
            );
        }
    }

    #[test]
    fn pose_state_changes_center_without_changing_gravity_alignment() {
        let gravity_axis = Vec3F64::new(0.3, -0.8, 0.4).normalize();
        let original = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.2, 0.15, 0.35)).matrix(),
            Vec3F64::new(0.4, -0.7, 1.1),
        );
        let expected_center = Vec3F64::new(2.0, -3.0, 1.5);
        let state = [
            expected_center.x as f32,
            expected_center.y as f32,
            expected_center.z as f32,
            0.7,
        ];

        let reconstructed = pose_from_state(&original, gravity_axis, &state).unwrap();

        assert_vec3_near(reconstructed.inverse().translation, expected_center, 1e-5);
        assert_vec3_near(
            reconstructed.rotation * gravity_axis,
            original.rotation * gravity_axis,
            1e-6,
        );
    }

    fn world_to_camera(center: Vec3F64, yaw: f64, gravity_axis: Vec3F64) -> Pose3d {
        let rotation = SO3F64::exp(gravity_axis.normalize() * -yaw).matrix();
        Pose3d::new(rotation, -(rotation * center))
    }

    fn graph_cost(poses: &[Pose3d], edges: &[PgoEdge]) -> f64 {
        0.5 * edges
            .iter()
            .map(|edge| {
                let error = edge.t_ab_meas.inverse()
                    * (pose_to_se3(&poses[edge.pose_b])
                        * pose_to_se3(&poses[edge.pose_a]).inverse());
                let (translation, rotation) = error.log();
                let weight = edge.weight as f64;
                weight
                    * weight
                    * (translation.x as f64 * translation.x as f64
                        + translation.y as f64 * translation.y as f64
                        + translation.z as f64 * translation.z as f64
                        + rotation.x as f64 * rotation.x as f64
                        + rotation.y as f64 * rotation.y as f64
                        + rotation.z as f64 * rotation.z as f64)
            })
            .sum::<f64>()
    }

    #[test]
    fn optimizer_anchors_first_pose_and_reduces_loop_cost() {
        let gravity = Vec3F64::new(0.2, -0.9, 0.3).normalize();
        let poses = vec![
            world_to_camera(Vec3F64::ZERO, 0.0, gravity),
            world_to_camera(Vec3F64::new(1.0, 0.0, 0.0), 0.0, gravity),
            world_to_camera(Vec3F64::new(2.2, 0.2, -0.1), 0.15, gravity),
        ];
        let true_last = world_to_camera(Vec3F64::new(2.0, 0.0, 0.0), 0.0, gravity);
        let edges = vec![
            PgoEdge {
                pose_a: 0,
                pose_b: 1,
                t_ab_meas: pose_to_se3(&Pose3d::between(&poses[0], &poses[1])),
                weight: 1.0,
            },
            PgoEdge {
                pose_a: 1,
                pose_b: 2,
                t_ab_meas: pose_to_se3(&Pose3d::between(&poses[1], &poses[2])),
                weight: 1.0,
            },
            PgoEdge {
                pose_a: 0,
                pose_b: 2,
                t_ab_meas: pose_to_se3(&Pose3d::between(&poses[0], &true_last)),
                weight: 1.0,
            },
        ];
        let initial_cost = graph_cost(&poses, &edges);

        let result = gravity_pose_graph_optimize(
            &poses,
            &edges,
            &[0],
            gravity * 9.81,
            &PgoParams {
                max_iterations: 100,
                ..PgoParams::default()
            },
        )
        .unwrap();

        assert_eq!(result.poses[0], poses[0]);
        assert!(graph_cost(&result.poses, &edges) < initial_cost);
        for (before, after) in poses.iter().zip(&result.poses) {
            assert_vec3_near(after.rotation * gravity, before.rotation * gravity, 2e-6);
        }
    }

    #[test]
    fn optimizer_rejects_invalid_gravity_and_edge_indices() {
        let poses = vec![Pose3d::IDENTITY, Pose3d::IDENTITY];
        let valid_edge = PgoEdge {
            pose_a: 0,
            pose_b: 1,
            t_ab_meas: pose_to_se3(&Pose3d::IDENTITY),
            weight: 1.0,
        };
        assert!(
            gravity_pose_graph_optimize(
                &poses,
                &[valid_edge],
                &[0],
                Vec3F64::ZERO,
                &PgoParams::default(),
            )
            .is_err()
        );

        let invalid_edge = PgoEdge {
            pose_a: 0,
            pose_b: 2,
            t_ab_meas: pose_to_se3(&Pose3d::IDENTITY),
            weight: 1.0,
        };
        assert!(
            gravity_pose_graph_optimize(
                &poses,
                &[invalid_edge],
                &[0],
                Vec3F64::new(0.0, -9.81, 0.0),
                &PgoParams::default(),
            )
            .is_err()
        );
    }
}
