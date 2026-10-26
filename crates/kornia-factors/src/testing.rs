//! Test helpers shared by the factor modules.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

/// Deterministic uniform samples in [-1, 1).
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    pub(crate) fn vec3(&mut self, scale: f64) -> Vec3F64 {
        Vec3F64::new(self.next(), self.next(), self.next()) * scale
    }

    pub(crate) fn pose(&mut self, t_scale: f64, r_scale: f64) -> Pose3d {
        Pose3d::new(SO3F64::exp(self.vec3(r_scale)).matrix(), self.vec3(t_scale))
    }
}

/// `Exp(ρ, ω)`, translation `J_l(ω)·ρ`.
pub(crate) fn exp(xi: &[f64; 6]) -> Pose3d {
    let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
    let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
    Pose3d::new(
        SO3F64::exp(omega).matrix(),
        SO3F64::left_jacobian(omega) * rho,
    )
}

/// `pose · Exp(eps · e_k)`.
pub(crate) fn perturbed(pose: &Pose3d, k: usize, eps: f64) -> Pose3d {
    let mut xi = [0.0; 6];
    xi[k] = eps;
    pose.compose(&exp(&xi))
}
