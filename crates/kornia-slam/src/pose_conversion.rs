use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, Vec3AF32, Vec3F64};

/// Largest [`rotation_defect`] a pose rotation may carry and still count as rigid.
pub(crate) const RIGID_TOLERANCE: f64 = 1e-6;

/// `max|RᵀR - I|`, infinite when `r` has a non-finite entry.
pub(crate) fn rotation_defect(r: &Mat3F64) -> f64 {
    let residual = (r.transpose() * *r - Mat3F64::IDENTITY).to_cols_array();
    if !residual.iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    residual.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

pub(crate) fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    SE3F32::new(
        SO3F32::from_matrix(&mat3_to_f32(pose.rotation)),
        vec3_to_f32(pose.translation),
    )
}

pub(crate) fn se3_to_pose(se3: &SE3F32) -> Pose3d {
    Pose3d::new(mat3_to_f64(se3.r.matrix()), vec3_to_f64(se3.t))
}

pub(crate) fn mat3_to_f32(m: Mat3F64) -> Mat3AF32 {
    Mat3AF32::from_cols(
        m.x_axis.as_vec3a().into(),
        m.y_axis.as_vec3a().into(),
        m.z_axis.as_vec3a().into(),
    )
}

pub(crate) fn mat3_to_f64(m: Mat3AF32) -> Mat3F64 {
    m.as_dmat3().into()
}

pub(crate) fn vec3_to_f32(v: Vec3F64) -> Vec3AF32 {
    Vec3AF32::new(v.x as f32, v.y as f32, v.z as f32)
}

pub(crate) fn vec3_to_f64(v: Vec3AF32) -> Vec3F64 {
    Vec3F64::new(v.x as f64, v.y as f64, v.z as f64)
}

/// Carries a reference-keyframe correction into the current tracking pose while
/// preserving the current camera's pose relative to that reference. Used for
/// both bundle-adjustment and pose-graph corrections.
pub(crate) fn apply_reference_pose_correction(
    current_pose: Pose3d,
    reference_before: Pose3d,
    reference_after: Pose3d,
) -> Pose3d {
    let current_from_reference = Pose3d::between(&reference_before, &current_pose);
    current_from_reference.compose(&reference_after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_casts_are_element_wise_and_column_major() {
        let m = Mat3F64::from_cols_array(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let m32 = mat3_to_f32(m);
        for (c, col) in [m.x_axis, m.y_axis, m.z_axis].into_iter().enumerate() {
            let col32 = m32.col(c);
            assert_eq!(col32.to_array(), [col.x as f32, col.y as f32, col.z as f32]);
        }
        let back = mat3_to_f64(m32);
        assert_eq!(back.to_cols_array(), m32.to_cols_array().map(f64::from));

        let v = Vec3F64::new(0.1, -2.5, 1e-9);
        let v32 = vec3_to_f32(v);
        assert_eq!(v32.to_array(), [v.x as f32, v.y as f32, v.z as f32]);
        assert_eq!(vec3_to_f64(v32).to_array(), v32.to_array().map(f64::from));
    }

    #[test]
    fn rotation_defect_is_zero_for_a_rotation_and_grows_with_scale() {
        assert_eq!(rotation_defect(&Mat3F64::IDENTITY), 0.0);
        let r = kornia_algebra::SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.5)).matrix();
        assert!(rotation_defect(&r) < 1e-15);
        // (1.001)^2 - 1 on the diagonal.
        let scaled = rotation_defect(&(r * 1.001));
        assert!((scaled - 2.001e-3).abs() < 1e-9, "{scaled}");
        let mut broken = r;
        broken.x_axis.x = f64::NAN;
        assert_eq!(rotation_defect(&broken), f64::INFINITY);
    }
}
