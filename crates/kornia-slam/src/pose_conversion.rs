use kornia_3d::pose::Pose3d;
use kornia_algebra::SO3F64;

pub(crate) use crate::upstream::kornia_3d::pose_ext::{orthonormalized, pose_to_se3, se3_to_pose};
pub(crate) use crate::upstream::kornia_algebra::cast::{
    mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64,
};

/// Largest [`Pose3d::rotation_defect`] a pose rotation may carry and still count as rigid.
pub(crate) const RIGID_TOLERANCE: f64 = 1e-6;

/// `pose` with its rotation [`orthonormalized`]; the translation is kept.
pub(crate) fn rigid(pose: Pose3d) -> Pose3d {
    Pose3d::new(orthonormalized(pose.rotation), pose.translation)
}

/// Rotation angle between two poses, degrees.
pub(crate) fn angle_deg(from: &Pose3d, to: &Pose3d) -> f64 {
    let relative = Pose3d::between(from, to);
    SO3F64::from_matrix(&relative.rotation)
        .log()
        .length()
        .to_degrees()
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
    rigid(current_from_reference.compose(&reference_after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, SE3F32, SO3F32, Vec3F64};

    fn rotation() -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix()
    }

    fn scaled_pose(scale: f64) -> Pose3d {
        Pose3d::new(rotation() * scale, Vec3F64::new(4.0, -1.5, 12.0))
    }

    /// Angle in radians between two rotations.
    fn angle(a: &Mat3F64, b: &Mat3F64) -> f64 {
        SO3F64::from_matrix(&orthonormalized(a.transpose() * *b))
            .log()
            .length()
    }

    fn assert_pose_close(a: &Pose3d, b: &Pose3d, tolerance: f64) {
        for (x, y) in a
            .rotation
            .to_cols_array()
            .iter()
            .zip(b.rotation.to_cols_array())
        {
            assert!((x - y).abs() <= tolerance, "{x} vs {y}");
        }
        assert!((a.translation - b.translation).length() <= tolerance);
    }

    #[test]
    fn reference_correction_is_rigid_whichever_input_is_scaled() {
        let rigid_pose = scaled_pose(1.0);
        let other = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.1, 0.4, 0.2)).matrix(),
            Vec3F64::new(1.0, 2.0, -3.0),
        );
        for scaled in 0..3 {
            let mut inputs = [rigid_pose, other, other.compose(&rigid_pose)];
            inputs[scaled] =
                Pose3d::new(inputs[scaled].rotation * 1.001, inputs[scaled].translation);
            let corrected = apply_reference_pose_correction(inputs[0], inputs[1], inputs[2]);
            assert!(corrected.rotation_defect() < 1e-12, "input {scaled}");
        }
    }

    #[test]
    fn self_corrections_no_longer_cube_the_scale() {
        let mut pose = scaled_pose(1.0 + 1e-7);
        for _ in 0..20 {
            pose = apply_reference_pose_correction(pose, pose, pose);
            assert!(pose.rotation_defect() < 1e-12);
        }
    }

    #[test]
    fn reference_correction_of_rigid_inputs_is_the_old_formula() {
        let current = scaled_pose(1.0);
        let before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.1, 0.4, 0.2)).matrix(),
            Vec3F64::new(1.0, 2.0, -3.0),
        );
        let after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.12, 0.41, 0.19)).matrix(),
            Vec3F64::new(1.1, 1.9, -3.05),
        );
        let old = Pose3d::between(&before, &current).compose(&after);
        let new = apply_reference_pose_correction(current, before, after);
        assert_pose_close(&new, &old, 1e-12);
    }

    #[test]
    fn se3_round_trip_of_a_scaled_pose_is_rigid() {
        let pose = scaled_pose(1.001);
        let se3 = pose_to_se3(&pose);
        assert!(
            (se3.r.q.length() - 1.0).abs() < 1e-6,
            "{}",
            se3.r.q.length()
        );
        let back = se3_to_pose(&se3);
        assert!(back.rotation_defect() < 1e-12);
        assert!(angle(&back.rotation, &rotation()) < 1.1e-3);
    }

    #[test]
    fn se3_to_pose_of_a_scaled_quaternion_is_rigid() {
        let se3 = pose_to_se3(&scaled_pose(1.0));
        let q = kornia_algebra::QuatF32::from_array(se3.r.q.to_array().map(|v| v * 1.001));
        let scaled = SE3F32::new(SO3F32::from_quaternion(q), se3.t);
        assert!(se3_to_pose(&scaled).rotation_defect() < 1e-12);
    }

    #[test]
    fn se3_round_trip_keeps_a_rigid_pose() {
        let pose = scaled_pose(1.0);
        assert_pose_close(&se3_to_pose(&pose_to_se3(&pose)), &pose, 1e-6);
    }
}
