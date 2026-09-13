use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, QuatF64, SE3F32, SO3F32, SO3F64, Vec3AF32, Vec3F64};

/// Rotation that takes unit vector `from` to unit vector `to`.
pub(crate) fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
    let from = from.normalize();
    let to = to.normalize();
    let dot = from.dot(to).clamp(-1.0, 1.0);
    let cross = from.cross(to);

    // Anti-parallel: pick an arbitrary perpendicular axis.
    if dot < -1.0 + 1e-9 {
        let perp = if from.x.abs() < 0.9 {
            Vec3F64::new(1.0, 0.0, 0.0)
        } else {
            Vec3F64::new(0.0, 1.0, 0.0)
        };
        let axis = from.cross(perp).normalize();
        return SO3F64::from_quaternion(QuatF64::from_array([axis.x, axis.y, axis.z, 0.0]));
    }

    let w = ((1.0 + dot) / 2.0).sqrt();
    let s = 1.0 / (2.0 * w);
    SO3F64::from_quaternion(QuatF64::from_array([
        cross.x * s,
        cross.y * s,
        cross.z * s,
        w,
    ]))
}

pub(crate) fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
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

pub(crate) fn se3_to_pose(se3: &SE3F32) -> Pose3d {
    let rotation = se3.r.matrix();
    Pose3d::new(
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
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_from_to_maps_the_source_onto_the_target() {
        let from = Vec3F64::new(0.0, 0.0, -1.0);
        let to = Vec3F64::new(0.3, 9.6, 1.5).normalize();

        let rotated = rotation_from_to(from, to) * from;

        assert!((rotated - to).length() < 1e-12);
    }

    #[test]
    fn rotation_from_to_handles_anti_parallel_input() {
        for from in [
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ] {
            let rotated = rotation_from_to(from, -from) * from;
            assert!(
                (rotated + from).length() < 1e-9,
                "anti-parallel rotation of {from:?} gave {rotated:?}"
            );
        }
    }
}
