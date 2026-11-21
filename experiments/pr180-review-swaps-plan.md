# PR #180 review swaps: predictions

The review of PR #180 (2026-11-16) asked for local helpers to be replaced by the kornia-rs APIs that already exist. Most swaps are pure refactors. This card records, before any code, what each swap is expected to change, how much, and what kills it. The validation workflow scores it; nothing here has been measured yet.

## Experiment: PR #180 review swaps
Gap tested: none. These are API swaps, not a LaMAria gap fix; the identity checks below guard that no gap moves.
Base: kornia-slam@`pr180/R` (`b59b0b9`), kornia-rs@`4ada5a7`.   Binary tag: `pr180/S` (swaps without the projection change), `pr180/F8` (with it).

Mechanism, per swap (file:line on the base sha):
- **Step 3, `rotation_defect`:** `pose_conversion.rs:8` duplicates `Pose3d::rotation_defect`. Pure.
- **Step 4, camera centre:** `pose_conversion.rs:29`, `loop_closure/graph/mod.rs:868` and the test-only `sparse_pgo.rs:2096` are three copies of `pose.inverse().translation`. Pure.
- **Step 5, Cauchy loop edges:** `sparse_pgo.rs` carries its own `RobustKernel` enum. `kornia_3d::ransac::CauchyKernel` computes the same weight and rho in f64; ours ran in f32. Not pure on Cauchy loop edges (P7).
- **Step 6, rig PnP:** `rig_pnp.rs` carries `huber_weight` and a pivoted `solve6`. The swap uses `HuberKernel` and the `Mat6F64` Cholesky with the owner's 1e-12 pivot guard. Not pure: `robust_scale_sq <= 0` gave weight 0 everywhere and now gives 1 (default 25.0, so no config is affected).
- **Step 7, pose LM:** `pose_lm.rs` carries `Loss`, `so3_exp`, `se3_exp_parts`, `solve6` and `norm`. The swap uses `RobustKernelKind`, `SO3F64::exp`, `SO3F64::left_jacobian` and the `Mat6F64` Cholesky (P4, P6).
- **Step 8, projection onto SO(3):** `pose_conversion.rs:19` projects through the unit quaternion; `Pose3d::orthonormalized` uses the SVD polar factor. Applied last, as one commit, so it can be dropped (P1-P3).
- **Step 9, `build.rs`:** `keyframes_while_lost` has two sources (`TrackingLossRecoveryPolicy` and the ORB tracking-loss config), and `lost_keyframes = FixedInertial` is accepted even when no keyframe is ever inserted while lost. Not pure for configs: the inert pair is now a `validate()` error. No `configs/*.ron` sets either knob.

Theory: each kornia-rs function computes the same quantity as the local copy. Differences come only from float order (Cholesky against pivoted elimination; f64 kernels cast to f32) and from the projection method in Step 8. ORB-SLAM3 and Basalt both keep rotations on SO(3) through Sophus, which renormalises the quaternion; that is the old body, so Step 8 is the only swap that changes a reference choice.
Knob: none. Steps 3-7 and 9 are refactors or bug fixes; Step 8 is a droppable commit, not a knob.
Baselines: none run yet. The validation workflow records them on `pr180/R` (sha, binary md5, load) before scoring.

Predictions (numbers, each tied to a measurement):
  P1 Projection error of `s·R` at `s = 1.001`, `ω = (0.3, -0.2, 0.9)`: old path 2.65e-4 rad (accept 2.0e-4 to 3.3e-4); SVD below 1e-12.   measured by: the Step 8 test `svd_and_quaternion_projection_agree_within_the_input_defect`.
  P2 Entry noise 1e-9 on a rotation: the two projections differ by at most 1e-8 rad; an exact rotation stays within 1e-14 under both.   measured by: the same test.
  P3 `se3_to_pose` output moves by up to about 1e-7 under Step 8, so the PGO fingerprint test fails; Step 8 is then dropped.   measured by: the `sparse_pgo` fingerprint test on `pr180/F8` against `pr180/F`.
  P4 `pose_lm` Cholesky against the old `solve6`: relative 1e-12 at condition number κ up to 1e4; both return `None` on a non-PD input.   measured by: the Step 7 test-only oracle.
  P5 `rig_pnp` Cholesky against the old pivoted elimination: 1e-10 at κ up to 1e4, 1e-6 at κ about 1e8; both return `None` on a rank-deficient `H`.   measured by: the Step 6 `#[cfg(test)]` oracle.
  P6 `SO3F64::left_jacobian` against the local version: at most `4e-16/θ + 1e-15`. Every other swapped function within relative 1e-15.   measured by: the Step 7 test-only oracles.
  P7 Cauchy weight within 2 f32 ulp of the old f32 body, rho within 4.   measured by: the Step 5 tests in `sparse_pgo.rs`.
  P8 End to end, `pr180/S` against `pr180/R` on EuRoC MH_01 and LaMAria 3_18: ATE and CP score inside the ±50 µs IMU-offset noise band of `pr180/R`.   measured by: the bench harness, 3 runs per tag.
  Mechanism counter: the commit at which trajectories first stop being `cmp` equal to `pr180/R`. Predicted: Steps 3 and 4 keep them equal; the first change comes at Step 5 (only if a Cauchy loop edge fires), Step 6 (map-projection rig PnP) or Step 7 (mono PnP refine, `pnp.rs:284`), from float order in the solve.

P1 dispute (two reviews disagree on why the old path errs at `s = 1.001`):
- Fable: the quaternion projection of a scaled matrix differs from the SVD polar factor.
- DeepSeek: for a uniform scale both return `R` exactly in exact arithmetic, so the error comes from the f32 cast in `pose_to_se3`.
- Test, before Step 8 is judged: project four inputs with both methods and report the angle to `R` for each.
  1. uniform `s = 1.001`, in f64 (`orthonormalized` alone);
  2. uniform `s = 1.001`, through f32 (`se3_to_pose(&pose_to_se3(..))`);
  3. anisotropic `R · diag(1.001, 1.0, 0.999)`, in f64;
  4. the same anisotropic input through f32.
- Reading: if (1) is below 1e-12 for both methods and (2) is near 2.65e-4, the f32 cast holds. If (3) splits the two methods by more than 1e-6, the quaternion-against-SVD difference holds for anisotropic defects. Both can hold at once; the card records which one explains P1.

Kill rules:
  K1 A pure step (3, 4) changes any test result or any `cmp` -> the swap is not equivalent; revert that step and find the difference before any other step lands.
  K2 P4 or P5 misses at κ ≤ 1e4 -> the Cholesky path or the pivot guard is wrong, not float order; fix before the identity gate.
  K3 P7 misses -> the f64 kernel is not the f32 body; check the `c_sq` argument (scale squared, not scale).
  K4 Step 8 changes the PGO fingerprint (P3 as predicted) -> drop the Step 8 commit; the swap is not worth a fingerprint change without an end-to-end gain.
  K5 Trajectories first change at Step 3, 4 or 9 (mechanism counter) -> that step is not pure; revert it and find the difference.
  K6 P8 misses (score outside the noise band) while P4-P7 hold -> float order alone moved the score, so the pipeline is chaotic at that point; record it as a robustness finding, do not revert the swaps for it.
  K7 The P1 dispute test matches neither reading -> the theory of the old error is wrong; record it and keep Step 8 dropped.
Bit-identity: Steps 3, 4 and 9 keep `cmp` equality with `pr180/R` on EuRoC MH_01 and LaMAria 3_18. Steps 6 and 7 change float order in solves that run every frame, so `pr180/S` is not expected to `cmp` equal `pr180/R`. `pr180/S` is the knobs-off baseline for the later lanes (`pr180/F` must `cmp` equal it).
Noise: none for unit-level predictions (deterministic). End to end, the pipeline is deterministic; P8 uses the ±50 µs IMU-offset samples from the LaMAria handoff, 3 per tag.
