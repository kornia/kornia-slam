//! Pinhole reprojection of a fixed world point into a camera pose `T_cw`
//! perturbed on the right, `T_cw ← T_cw · Exp([ρ; ω])`.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SMatF64, SO3F64, SVecF64, Vec3F64};

/// The pixel residual `π(T_cw · p_w) − u` and its Jacobian with respect to the
/// pose, or `None` when the point is not more than `min_depth` in front of the
/// camera. `pixel` is undistorted.
pub fn residual_and_pose_jacobian(
    pose_cw: &Pose3d,
    point_w: &Vec3F64,
    pixel: [f64; 2],
    camera: &PinholeCamera,
    min_depth: f64,
) -> Option<(SVecF64<2>, SMatF64<2, 6>)> {
    let p_c = pose_cw.transform_point(point_w);
    if p_c.z <= min_depth {
        return None;
    }
    let inv_z = 1.0 / p_c.z;
    let residual = SVecF64::<2>::new(
        camera.fx * p_c.x * inv_z + camera.cx - pixel[0],
        camera.fy * p_c.y * inv_z + camera.cy - pixel[1],
    );
    let d_pixel_d_cam = SMatF64::<2, 3>::new(
        camera.fx * inv_z,
        0.0,
        -camera.fx * p_c.x * inv_z * inv_z,
        0.0,
        camera.fy * inv_z,
        -camera.fy * p_c.y * inv_z * inv_z,
    );
    // Exp([ρ; ω])·p ≈ p + ρ + ω×p, so ∂p_c/∂ρ = R and ∂p_c/∂ω = −R·[p_w]×.
    let rotation: SMatF64<3, 3> = pose_cw.rotation.into();
    let rotation_hat: SMatF64<3, 3> = (pose_cw.rotation * SO3F64::hat(*point_w) * -1.0).into();
    let mut jacobian = SMatF64::<2, 6>::zeros();
    jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&(d_pixel_d_cam * rotation));
    jacobian
        .fixed_view_mut::<2, 3>(0, 3)
        .copy_from(&(d_pixel_d_cam * rotation_hat));
    Some((residual, jacobian))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Lcg, perturbed};

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 450.0,
            fy: 455.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let mut rng = Lcg(7);
        let camera = camera();
        let pose = rng.pose(0.3, 0.2);
        let point = pose
            .inverse()
            .transform_point(&Vec3F64::new(0.4, -0.2, 3.0));
        let pixel = [300.0, 250.0];

        let (r0, jacobian) =
            residual_and_pose_jacobian(&pose, &point, pixel, &camera, 1e-3).unwrap();
        let eps = 1e-6;
        for k in 0..6 {
            let (r1, _) =
                residual_and_pose_jacobian(&perturbed(&pose, k, eps), &point, pixel, &camera, 1e-3)
                    .unwrap();
            let numeric = (r1 - r0) / eps;
            for row in 0..2 {
                assert!(
                    (numeric[row] - jacobian[(row, k)]).abs() < 1e-3,
                    "column {k} row {row}: {} vs {}",
                    numeric[row],
                    jacobian[(row, k)]
                );
            }
        }
    }

    #[test]
    fn a_point_behind_the_camera_has_no_residual() {
        let point = Vec3F64::new(0.0, 0.0, -1.0);
        assert!(
            residual_and_pose_jacobian(&Pose3d::IDENTITY, &point, [0.0, 0.0], &camera(), 1e-3)
                .is_none()
        );
    }
}
