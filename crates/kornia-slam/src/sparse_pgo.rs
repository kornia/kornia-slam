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
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};

    use super::{
        GravityManifold, PoseManifold, Se3Manifold, pose_to_se3, weighted_relative_residual,
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
}
