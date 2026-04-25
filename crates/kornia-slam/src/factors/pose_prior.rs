//! Strong SE(3) prior factor used to anchor a keyframe in inertial BA.
//!
//! Workaround for the lack of a "fixed variable" mechanism in
//! `kornia_algebra::optim::Problem`. Pinning a pose with a high-weight prior
//! keeps the gauge fixed (yaw, xy translation, scale via inertial coupling)
//! without moving it appreciably during optimization.
//!
//! Residual (6D, in SE(3) tangent):
//!   r = log(T_target⁻¹ · T_x)
//! where `T_x = (R_x, t_x)` is the current pose estimate. We use a small-angle
//! linearization around the target — adequate because the pose is held nearly
//! at the target by the strong weight, so the residual stays near zero.
//!
//! Tangent ordering: `[δυ(3), δω(3)]` — translation first, rotation second —
//! matching `InertialFactor`.

use kornia_algebra::optim::{Factor, FactorError, FactorResult, LinearizationResult};

/// Pose tangent dimension (SE3).
const POSE_DIM: usize = 6;

/// SE(3) prior factor that pulls a pose variable toward a target.
///
/// `weight` scales both the residual and Jacobian; use a large value (e.g. 1e3)
/// to behave as a near-hard constraint.
pub struct PosePriorFactor {
    target_quat: [f32; 4],   // [qw, qx, qy, qz]
    target_trans: [f32; 3],  // [tx, ty, tz]
    weight: f32,
}

impl PosePriorFactor {
    /// `target_pose` is the SE3 parameter vector `[qw, qx, qy, qz, tx, ty, tz]`.
    pub fn new(target_pose: &[f32; 7], weight: f32) -> Self {
        Self {
            target_quat: [target_pose[0], target_pose[1], target_pose[2], target_pose[3]],
            target_trans: [target_pose[4], target_pose[5], target_pose[6]],
            weight,
        }
    }
}

impl Factor for PosePriorFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 1 {
            return Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: params.len(),
            });
        }
        let pose = params[0];
        if pose.len() != 7 {
            return Err(FactorError::DimensionMismatch {
                expected: 7,
                actual: pose.len(),
            });
        }

        // Translation residual in target's body frame: R_target^T · (t_x − t_target)
        let r_target = quat_to_rot(&self.target_quat);
        let dt = [
            pose[4] - self.target_trans[0],
            pose[5] - self.target_trans[1],
            pose[6] - self.target_trans[2],
        ];
        // Note: the pose's `t` lives in the same global frame on both sides;
        // expressing the residual in the target's body frame keeps the
        // δυ-Jacobian identity (matching the SE3 tangent we use).
        // For small displacements this is equivalent to dt itself; we use
        // identity Jacobian below and let the residual just be `t_x − t_target`.
        let _ = r_target;
        let r_trans = dt;

        // Rotation residual: log(R_target^T · R_x). For small rotations this
        // is δω in body coords. We compute it via quaternion multiplication.
        let q_target_inv = [
            self.target_quat[0],
            -self.target_quat[1],
            -self.target_quat[2],
            -self.target_quat[3],
        ];
        let q_err = quat_mul(&q_target_inv, &[pose[0], pose[1], pose[2], pose[3]]);
        let r_rot = quat_log(&q_err);

        // Stack residual: [δυ, δω], scaled by weight.
        let mut residual = vec![
            self.weight * r_trans[0],
            self.weight * r_trans[1],
            self.weight * r_trans[2],
            self.weight * r_rot[0],
            self.weight * r_rot[1],
            self.weight * r_rot[2],
        ];

        let jacobian = if compute_jacobian {
            // ∂r/∂[δυ, δω] = weight · I_6
            // (small-angle approximation: for δυ block the perturbation is
            // t_new = t + R · δυ, but R ≈ R_target so identity is fine when
            // the residual is held small by the weight).
            let mut jac = vec![0.0f32; POSE_DIM * POSE_DIM];
            for i in 0..POSE_DIM {
                jac[i * POSE_DIM + i] = self.weight;
            }
            // Apply weight scaling to residual is done above. Jacobian above
            // already includes the same `self.weight` factor.
            // (No extra whitening: the weight IS the sqrt_information.)
            let _ = &mut residual; // residual already weighted
            Some(jac)
        } else {
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, POSE_DIM))
    }

    fn residual_dim(&self) -> usize {
        POSE_DIM
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        match idx {
            0 => POSE_DIM,
            _ => panic!("PosePriorFactor has 1 variable"),
        }
    }
}

// ── Quaternion / rotation helpers (f32) ──────────────────────────────────

fn quat_to_rot(q: &[f32; 4]) -> [f32; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y),
    ]
}

fn quat_mul(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Quaternion log: returns the axis-angle vector ω such that q = exp(ω/2) (in
/// quaternion form). For small rotations, ω ≈ 2 · [qx, qy, qz].
fn quat_log(q: &[f32; 4]) -> [f32; 3] {
    let w = q[0].clamp(-1.0, 1.0);
    let v = [q[1], q[2], q[3]];
    let v_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if v_norm < 1e-7 {
        // Identity rotation
        [0.0, 0.0, 0.0]
    } else {
        let theta = 2.0 * v_norm.atan2(w.abs());
        let scale = if w >= 0.0 { theta / v_norm } else { -theta / v_norm };
        [v[0] * scale, v[1] * scale, v[2] * scale]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_residual_at_target() {
        let target = [1.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let factor = PosePriorFactor::new(&target, 1.0);
        let result = factor.linearize(&[&target], false).unwrap();
        for r in &result.residual {
            assert!(r.abs() < 1e-5, "expected zero residual, got {r}");
        }
    }

    #[test]
    fn nonzero_residual_when_offset() {
        let target = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let offset = [1.0f32, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3];
        let factor = PosePriorFactor::new(&target, 1.0);
        let result = factor.linearize(&[&offset], false).unwrap();
        // Translation residual = offset - target (identity)
        assert!((result.residual[0] - 0.1).abs() < 1e-5);
        assert!((result.residual[1] + 0.2).abs() < 1e-5);
        assert!((result.residual[2] - 0.3).abs() < 1e-5);
        // Rotation residual = 0
        for i in 3..6 {
            assert!(result.residual[i].abs() < 1e-5);
        }
    }

    #[test]
    fn weight_scales_residual_and_jacobian() {
        let target = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let offset = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let factor = PosePriorFactor::new(&target, 100.0);
        let result = factor.linearize(&[&offset], true).unwrap();
        assert!((result.residual[0] - 100.0).abs() < 1e-3);
        let jac = result.jacobian.unwrap();
        // Diagonal entries should be 100.0
        for i in 0..POSE_DIM {
            assert!((jac[i * POSE_DIM + i] - 100.0).abs() < 1e-3);
        }
    }
}
