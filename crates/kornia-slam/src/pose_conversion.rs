use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, SE3F32, SO3F32, Vec3AF32, Vec3F64};

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
