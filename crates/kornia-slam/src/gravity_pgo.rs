//! Gravity-preserving four-degree-of-freedom pose-graph optimization.

use kornia_3d::pgo::{PgoEdge, PgoParams};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use thiserror::Error;

use crate::sparse_pgo::{GravityManifold, SparsePgoError, sparse_pose_graph_optimize};

#[derive(Debug, Error)]
pub(crate) enum GravityPgoError {
    #[error(transparent)]
    Sparse(#[from] SparsePgoError),
}

pub(crate) struct GravityPgoResult {
    pub poses: Vec<Pose3d>,
    pub converged: bool,
}

pub(crate) fn gravity_pose_graph_optimize(
    poses: &[Pose3d],
    edges: &[PgoEdge],
    fixed_pose_indices: &[usize],
    gravity_world: Vec3F64,
    params: &PgoParams,
) -> Result<GravityPgoResult, GravityPgoError> {
    let manifold = GravityManifold::new(gravity_world)?;
    let result = sparse_pose_graph_optimize(poses, edges, fixed_pose_indices, params, &manifold)?;

    Ok(GravityPgoResult {
        poses: result.poses,
        converged: result.converged,
    })
}

#[cfg(test)]
mod tests {
    use kornia_3d::pgo::{PgoEdge, PgoParams};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SE3F32, SO3F64, Vec3F64};

    use super::gravity_pose_graph_optimize;
    use crate::sparse_pgo::pose_to_se3 as sparse_pose_to_se3;

    fn assert_vec3_near(actual: Vec3F64, expected: Vec3F64, tolerance: f64) {
        assert!(
            (actual - expected).length() <= tolerance,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
        sparse_pose_to_se3(pose).unwrap()
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
    fn wrapper_anchors_first_pose_preserves_gravity_and_reduces_loop_cost() {
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

        assert!(result.converged);
        assert_eq!(result.poses[0], poses[0]);
        assert!(graph_cost(&result.poses, &edges) < initial_cost);
        for (before, after) in poses.iter().zip(&result.poses) {
            assert_vec3_near(after.rotation * gravity, before.rotation * gravity, 2e-6);
        }
    }

    #[test]
    fn wrapper_rejects_invalid_gravity_edge_and_fixed_indices() {
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
                std::slice::from_ref(&valid_edge),
                &[0],
                Vec3F64::ZERO,
                &PgoParams::default(),
            )
            .is_err()
        );

        assert!(
            gravity_pose_graph_optimize(
                &poses,
                std::slice::from_ref(&valid_edge),
                &[2],
                Vec3F64::new(0.0, -9.81, 0.0),
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
