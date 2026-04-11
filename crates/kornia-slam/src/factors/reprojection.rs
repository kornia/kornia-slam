//! Monocular reprojection factor (EdgeMono equivalent).
//!
//! Residual: observed_pixel - project(pose * point_world)
//! Pre-whitened by the scale-derived information weight.

use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};
use kornia_algebra::optim::whitening::whiten_scalar;

/// Monocular reprojection factor connecting a 3D map point and a camera pose.
///
/// Variables:
///   0: map point (Euclidean 3D, params = [x, y, z])
///   1: camera pose (SE3, params = [qw, qx, qy, qz, tx, ty, tz], tangent = [upsilon(3), omega(3)])
///
/// The residual is 2D: observed pixel minus projected pixel, pre-whitened
/// by the information weight derived from the feature's pyramid scale.
pub struct ReprojectionFactor {
    /// Observed pixel coordinates [u, v].
    pub observation: [f32; 2],
    /// Camera intrinsics [fx, fy, cx, cy].
    pub intrinsics: [f32; 4],
    /// Information weight = 1 / scale^2 (from image pyramid octave).
    pub information_weight: f32,
}

impl ReprojectionFactor {
    pub fn new(observation: [f32; 2], intrinsics: [f32; 4], scale: f32) -> Self {
        Self {
            observation,
            intrinsics,
            information_weight: 1.0 / (scale * scale),
        }
    }

    /// Transform world point by pose (world-to-camera) and project to pixel.
    /// Returns (u, v, p_cam) where p_cam is the point in camera frame.
    fn project(&self, point: &[f32], pose: &[f32]) -> ([f32; 2], [f32; 3]) {
        let [fx, fy, cx, cy] = self.intrinsics;

        // Parse SE3: [qw, qx, qy, qz, tx, ty, tz]
        let (qw, qx, qy, qz) = (pose[0], pose[1], pose[2], pose[3]);
        let (tx, ty, tz) = (pose[4], pose[5], pose[6]);
        let (px, py, pz) = (point[0], point[1], point[2]);

        // Rotate point: R * p (quaternion rotation)
        // R(q) * v = v + 2*q_w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
        let cx_ = qy * pz - qz * py;
        let cy_ = qz * px - qx * pz;
        let cz_ = qx * py - qy * px;

        let rx = px + 2.0 * (qw * cx_ + qy * cz_ - qz * cy_);
        let ry = py + 2.0 * (qw * cy_ + qz * cx_ - qx * cz_);
        let rz = pz + 2.0 * (qw * cz_ + qx * cy_ - qy * cx_);

        // p_cam = R * p_world + t
        let cam_x = rx + tx;
        let cam_y = ry + ty;
        let cam_z = rz + tz;

        let inv_z = 1.0 / cam_z;
        let u = fx * cam_x * inv_z + cx;
        let v = fy * cam_y * inv_z + cy;

        ([u, v], [cam_x, cam_y, cam_z])
    }
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 2 {
            return Err(FactorError::DimensionMismatch {
                expected: 2,
                actual: params.len(),
            });
        }

        let point = params[0]; // [x, y, z]
        let pose = params[1]; // [qw, qx, qy, qz, tx, ty, tz]

        let ([u, v], [cam_x, cam_y, cam_z]) = self.project(point, pose);

        let mut residual = vec![self.observation[0] - u, self.observation[1] - v];

        let jacobian = if compute_jacobian {
            let [fx, fy, _, _] = self.intrinsics;
            let inv_z = 1.0 / cam_z;
            let inv_z2 = inv_z * inv_z;

            // d(project)/d(p_cam): 2x3 projection Jacobian
            // u = fx * cam_x / cam_z + cx  =>  du/dcam = [fx/z, 0, -fx*x/z²]
            // v = fy * cam_y / cam_z + cy  =>  dv/dcam = [0, fy/z, -fy*y/z²]
            let dp_dx = fx * inv_z;
            let dp_dz_x = -fx * cam_x * inv_z2;
            let dp_dy = fy * inv_z;
            let dp_dz_y = -fy * cam_y * inv_z2;

            // Rotation matrix from quaternion (for Jacobian w.r.t. point)
            let (qw, qx, qy, qz) = (pose[0], pose[1], pose[2], pose[3]);
            let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
            let r01 = 2.0 * (qx * qy - qw * qz);
            let r02 = 2.0 * (qx * qz + qw * qy);
            let r10 = 2.0 * (qx * qy + qw * qz);
            let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
            let r12 = 2.0 * (qy * qz - qw * qx);
            let r20 = 2.0 * (qx * qz - qw * qy);
            let r21 = 2.0 * (qy * qz + qw * qx);
            let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

            // Jacobian w.r.t. point (3 cols): d(residual)/d(point) = -d(proj)/d(p_cam) * R
            // residual = obs - proj, so d(residual)/d(point) = -d(proj)/d(p_cam) * d(p_cam)/d(point)
            // d(p_cam)/d(point) = R
            let j_point = [
                // row 0 (du)
                -(dp_dx * r00 + dp_dz_x * r20),
                -(dp_dx * r01 + dp_dz_x * r21),
                -(dp_dx * r02 + dp_dz_x * r22),
                // row 1 (dv)
                -(dp_dy * r10 + dp_dz_y * r20),
                -(dp_dy * r11 + dp_dz_y * r21),
                -(dp_dy * r12 + dp_dz_y * r22),
            ];

            // Jacobian w.r.t. pose (6 cols): tangent = [upsilon(3), omega(3)]
            // Right perturbation: T_new = T * exp(δξ)
            // p_cam_new = R * exp(δω) * p_w + t + R * δυ  (first order)
            //           ≈ p_cam + R * (δυ + [δω]× * p_w)
            //           = p_cam + R * δυ - R * [p_w]× * δω
            //
            // So d(p_cam)/d(δυ) = R, d(p_cam)/d(δω) = -R * [p_w]×
            // And d(residual)/d(δξ) = -d(proj)/d(p_cam) * [R | -R*[p_w]×]

            let (px, py, pz) = (point[0], point[1], point[2]);

            // -R * [p_w]× where [p_w]× is the skew-symmetric of the world point
            // [p_w]× = [[ 0, -pz,  py],
            //           [ pz,  0, -px],
            //           [-py,  px,  0]]
            // R * [p_w]× gives 3x3, we need columns
            // Compute R * hat(p) directly
            // (R * hat(p))[i,j] = sum_k R[i,k] * hat(p)[k,j]
            // hat(p) = [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
            let rhat00 = r01 * pz - r02 * py;
            let rhat01 = -r00 * pz + r02 * px;
            let rhat02 = r00 * py - r01 * px;
            let rhat10 = r11 * pz - r12 * py;
            let rhat11 = -r10 * pz + r12 * px;
            let rhat12 = r10 * py - r11 * px;
            let rhat20 = r21 * pz - r22 * py;
            let rhat21 = -r20 * pz + r22 * px;
            let rhat22 = r20 * py - r21 * px;

            // d(p_cam)/d(δυ) = R (3x3)
            // d(p_cam)/d(δω) = -R * [p_w]× (3x3)
            // Full: d(p_cam)/d(δξ) = [R | -R*hat(p)] (3x6)
            //
            // d(residual)/d(δξ) = -d(proj)/d(p_cam) * [R | -R*hat(p)]
            let j_pose = [
                // row 0 (du), cols: upsilon(3) then omega(3)
                -(dp_dx * r00 + dp_dz_x * r20),
                -(dp_dx * r01 + dp_dz_x * r21),
                -(dp_dx * r02 + dp_dz_x * r22),
                (dp_dx * rhat00 + dp_dz_x * rhat20),
                (dp_dx * rhat01 + dp_dz_x * rhat21),
                (dp_dx * rhat02 + dp_dz_x * rhat22),
                // row 1 (dv), cols: upsilon(3) then omega(3)
                -(dp_dy * r10 + dp_dz_y * r20),
                -(dp_dy * r11 + dp_dz_y * r21),
                -(dp_dy * r12 + dp_dz_y * r22),
                (dp_dy * rhat10 + dp_dz_y * rhat20),
                (dp_dy * rhat11 + dp_dz_y * rhat21),
                (dp_dy * rhat12 + dp_dz_y * rhat22),
            ];

            // Full Jacobian: [j_point(2x3) | j_pose(2x6)] = 2x9 row-major
            let mut jac = vec![
                j_point[0], j_point[1], j_point[2],
                j_pose[0], j_pose[1], j_pose[2], j_pose[3], j_pose[4], j_pose[5],
                j_point[3], j_point[4], j_point[5],
                j_pose[6], j_pose[7], j_pose[8], j_pose[9], j_pose[10], j_pose[11],
            ];

            // Pre-whiten
            whiten_scalar(&mut residual, &mut jac, self.information_weight);

            Some(jac)
        } else {
            // Still need to whiten the residual for cost computation
            let sqrt_w = self.information_weight.sqrt();
            for r in residual.iter_mut() {
                *r *= sqrt_w;
            }
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, 9))
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        2
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        match idx {
            0 => 3, // map point
            1 => 6, // SE3 pose
            _ => panic!("ReprojectionFactor has 2 variables"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_pose() -> [f32; 7] {
        // [qw, qx, qy, qz, tx, ty, tz]
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn zero_residual_at_projected_point() {
        let fx = 500.0f32;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // Point at [1, 2, 5] in world frame, identity pose
        let point = [1.0f32, 2.0, 5.0];
        let pose = identity_pose();

        // Expected projection
        let u = fx * 1.0 / 5.0 + cx; // 420
        let v = fy * 2.0 / 5.0 + cy; // 440

        let factor = ReprojectionFactor::new([u, v], [fx, fy, cx, cy], 1.0);
        let result = factor
            .linearize(&[&point, &pose], false)
            .unwrap();

        assert!(result.residual[0].abs() < 1e-5);
        assert!(result.residual[1].abs() < 1e-5);
    }

    #[test]
    fn jacobian_numerical_check() {
        let fx = 500.0f32;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;
        let point = [1.0f32, 2.0, 5.0];
        let pose = identity_pose();

        let u = fx * 1.0 / 5.0 + cx;
        let v = fy * 2.0 / 5.0 + cy;

        // Use scale=1 so information_weight=1 (no whitening distortion)
        let factor = ReprojectionFactor::new([u + 0.5, v - 0.3], [fx, fy, cx, cy], 1.0);

        let result = factor
            .linearize(&[&point, &pose], true)
            .unwrap();
        let jac = result.jacobian.unwrap();
        let base_r = result.residual;

        // Numerical Jacobian w.r.t. point
        let eps = 1e-4f32;
        for col in 0..3 {
            let mut perturbed = point;
            perturbed[col] += eps;
            let r_plus = factor
                .linearize(&[&perturbed, &pose], false)
                .unwrap()
                .residual;

            let mut perturbed = point;
            perturbed[col] -= eps;
            let r_minus = factor
                .linearize(&[&perturbed, &pose], false)
                .unwrap()
                .residual;

            for row in 0..2 {
                let numerical = (r_plus[row] - r_minus[row]) / (2.0 * eps);
                let analytical = jac[row * 9 + col];
                let tol = 1e-2 * analytical.abs().max(1.0);
                assert!(
                    (numerical - analytical).abs() < tol,
                    "point jac mismatch at [{row},{col}]: numerical={numerical}, analytical={analytical}"
                );
            }
        }
    }
}
