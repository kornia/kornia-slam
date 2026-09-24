use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, Vec3AF32, Vec3F64};

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
}
