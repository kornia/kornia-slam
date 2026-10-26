//! State-estimation factors: residuals, Jacobians, information matrices and
//! robust costs, independent of any map, keyframe store or solver.
//!
//! A factor links two states by index (`from_idx`, `to_idx`); the caller owns
//! the states and the solver. Poses are world-to-camera `T_cw`, perturbed on
//! the right as `T_cw ← T_cw · Exp(ξ)` with tangent order `[ρ; ω]`.

pub mod robust;
pub mod se3;
pub mod stationary;
pub mod wheel_odometry;

#[cfg(test)]
mod testing;
