# PR #180 card C: solvers (analytic 2x6 refine, PGO manifolds and edges upstream)

The review of PR #180 (2026-11-16) asked for two solver follow-ups:

- **`pose_lm.rs:9`:** "the analytic 2×6 Jacobian would make a good kornia-rs PR, and this file could then go". kornia-3d's `refine_pose_lm` still uses central differences (`pnp/refine.rs:507`).
- **`sparse_pgo.rs:41`:** "`sparse_pgo` is now a general sparse pose-graph solver ... Should this become kornia-3d's `pgo`?"

Lane C builds both under `crates/kornia-slam/src/upstream/`, following the plan's upstream-later convention. Both are moves of code that already runs. They add no behaviour, so the gate is bit-identity, not a score. Nothing here has been measured yet; the validation workflow scores it.

## Experiment: C, solvers
Gap tested: none. No ledger row moves. Lane A's pose-inertial solve (card A, the G-rows it names) calls the Jacobian this lane exposes, so A's gap is where any score effect would show.
Base: kornia-slam@`pr180/S` (`42ee054`), kornia-rs@`4ada5a7`.   Binary tag: `pr180/L-C`.

Mechanism (file:line on `pr180/S`):
- **C1, analytic refine.** `tracking/pose_estimation/pose_lm.rs:1-235` is a pose-only LM solve with a closed-form 2x6 Jacobian. The Jacobian rows sit inline in `Problem::normal_equations` (`pose_lm.rs:105-114`). Lane A cannot call them, and kornia-rs cannot take them, while they are private to kornia-slam. The file also imports `crate::pose_conversion` (`pose_lm.rs:20`), which the upstream boundary forbids.
- **C2, PGO manifolds and edges.** `loop_closure/sparse_pgo.rs` mixes two things: the generic parts (`PoseManifold` `:27`, `InformationEdge` `:43`, the sealed `EdgeTerm` `:55-77`, `PoseGraphEdge` `:81-84`, both edge impls `:88-215`, `cauchy_c_sq` `:218`, `SparsePgoError` `:461`, `Se3Manifold` `:471`, `GravityManifold` `:482-509`, the fallible `pose_to_se3`/`se3_to_pose` `:511-523`, `weighted_relative_residual` `:525-553`, and the validators `normalized_gravity`, `validate_delta`, `validate_pose`, `validate_se3`, `vec3_f64_is_finite` `:1186-1244`) and the faer LM solver. The generic parts reach `crate::pose_conversion::{pose_to_se3, se3_to_pose}` (`sparse_pgo.rs:13-15`), which in turn uses the quaternion `orthonormalized` and the f32/f64 casts (`pose_conversion.rs:10,28-61`).

Theory:
- **Bit-identity.** A move that keeps every floating-point operation and its operand order gives bit-equal outputs.
  - Rust does not contract `a * b + c` into an FMA unless the code calls `mul_add`.
  - Inlining across modules of one crate changes where an operation runs, not its IEEE result.
  - So the only ways to change bits are an edited arithmetic line, a different overload (for example nalgebra indexing in place of a `[f64; 6]` array, if it reorders a sum), or a different conversion path. The mechanism counter below watches for exactly these.
- **Why keep our refine rather than port ORB-SLAM3's.**
  - ORB-SLAM3's pose-only edge (`OptimizableTypes.cpp:49-62`, `EdgeSE3ProjectXYZOnlyPose::linearizeOplus`) takes a g2o left perturbation, ordered (ω, υ), inside `PoseOptimization` (`Optimizer.cc:814`).
  - Ours takes a right perturbation ordered (υ, ω), which is kornia's `SE3F64::retract` (`pose_lm.rs:52-61`). The damped steps therefore follow kornia-3d's generic solver, which `agrees_with_the_generic_solver` (`pose_lm.rs:516`) holds at 1e-3.
  - Both are exact analytic Jacobians, so neither reference converges faster. Matching the kornia-3d solver is what lets the upstream PR replace `refine_pose_lm`'s numerical Jacobian without moving its results.
- **Body-frame chain for Lane A.**
  - ORB-SLAM3's inertial pose edge `EdgeMonoOnlyPose::linearizeOplus` (`G2oTypes.cc:375-395`) is `projectJac(Xc) · Rcb · SE3deriv(Xb)`, with the update in `ImuCamPose::Update` (`G2oTypes.cc:192-219`: `twb += Rwb·ut; Rwb = Rwb·Exp(ur)`).
  - The shared piece is the 2x3 pinhole Jacobian `projectJac` (`CameraModels/Pinhole.cpp:71`). Lane C exposes it alone, plus the 2x6 world-to-camera form that `pose_lm` uses. Lane A composes its own body-frame chain from the 2x3 piece.
- **Why the 4-DoF manifold stays as it is.**
  - `GravityManifold` moves the camera centre in world coordinates and turns about world gravity (`sparse_pgo.rs:494-508`). That is the DoF set of ORB-SLAM3's `VertexPose4DoF` (`G2oTypes.h:155-189`, through `ImuCamPose::UpdateW`, `G2oTypes.cc:222`) in `OptimizeEssentialGraph4DoF` (`Optimizer.cc:5292`).
  - ORB-SLAM3's `Edge4DoF` uses a numerical Jacobian (`G2oTypes.h:817-838`, the comment at `:838`), as `edge_endpoint_jacobian` does here (`sparse_pgo.rs:1061`). The extraction keeps that.
  - ORB-SLAM3 orders the 4-DoF residual (rotation, translation); ours is (translation, rotation) (`sparse_pgo.rs:538-546`). The order is unchanged by design, because changing it would change the bits.
- **Why only manifolds and edges move.** The faer LM solver (`sparse_pgo.rs:222-457, 557-1175`) stays in kornia-slam this round.
  - kornia-3d already depends on faer (`kornia-3d/Cargo.toml:53` at `4ada5a7`), so the solver can follow in a later round.
  - The plan scopes this lane to manifolds and edges only.

Knob: **none.** Both mechanisms are moves, with no new behaviour to switch. Bit-identity against `pr180/S` is the gate, as for Steps 3 and 4 of `pr180-review-swaps-plan.md`. The skeleton therefore adds no RON field, validation or CHANGELOG line for card C. The identity runs below use only RON fields that exist on `pr180/S`:

| Run | RON (path, type, value) | Default on `pr180/S` |
|---|---|---|
| refine path | `configs/euroc.ron` and `configs/lamaria-mono-imu.ron` unchanged; mono PnP refine runs every frame through `map_projection/mod.rs:711`, `pnp.rs:203`, `pnp.rs:284` | as shipped |
| PGO `PgoEdge` path | `configs/euroc-stereo-imu-loop.ron` with `system.pipeline.OrbSlam.loop_closing: Enabled(vocabulary: "../weights/ORBvoc.txt", execution: Synchronous)` (`LoopClosingExecution`) | `execution: FollowMapping`, `graph: (information: ScalarWeights)` |
| PGO `InformationEdge` path | the same, plus `graph: (information: Drift())` (`LoopGraphInformation::Drift`, all fields at their serde defaults) | `ScalarWeights` |

`execution: Synchronous` keeps the loop write-back reproducible, so `cmp` is meaningful. `Asynchronous` writes back at a thread-timed frame boundary.

Baselines: none run yet. The validator records `pr180/S` (sha, binary md5, load) on each run above before scoring `pr180/L-C`.

Predictions (numbers, each tied to a measurement):
  P1 EuRoC MH_01 (`euroc.ron`) and LaMAria 3_18 (`lamaria-mono-imu.ron`): trajectory and log outputs of `pr180/L-C` against `pr180/S`, 0 differing bytes.   measured by: `cmp` on each output file of the two run dirs.
  P2 PGO fingerprint test `optimize_pose_graph_output_is_bit_identical_to_b1be250` (`loop_closure/tests.rs:784`) passes with its constants `0xa4ac_a7f5_0777_b24e` (SE3) and `0x934d_608c_260c_d0b9` (inertial, 4-DoF) unchanged, and `tests.rs` is not edited by the lane (0 diff lines).   measured by: `cargo test --release -j2 -p kornia-slam optimize_pose_graph_output_is_bit_identical` and `git diff --stat pr180/S pr180/L-C -- crates/kornia-slam/src/loop_closure/tests.rs`.
  P3 MH_01 with the two PGO RON variants above: `pr180/L-C` against `pr180/S`, 0 differing bytes on each, given at least 1 accepted loop solve in the `pr180/S` run.   measured by: `cmp`, plus the loop-solve count in the run log.
  P4 Every test that moves from `pose_lm.rs` to `refine_analytic.rs` (the 10 at `pose_lm.rs:370-589`) passes with 0 tolerance literals changed. Every `sparse_pgo.rs` test (`:1510-2296`) passes unedited, apart from `use` lines.   measured by: `cargo test --release -j2 -p kornia-slam refine_analytic sparse_pgo`, and `git diff -U0 pr180/S pr180/L-C | grep -E '^[+-].*(<|<=|assert).*[0-9]e-'` printing 0 lines.
  P5 New test `pose_jacobian_2x6_matches_central_differences`: all 12 entries of `pose_jacobian_2x6` within `1e-6 · (1 + |J_ij|)` of central differences of the projected pixel under `Pose::right_update` (step 1e-6, f64). This is tighter than the existing H/g check at 1e-3 (`pose_lm.rs:371-417`), because it reads J directly and not through `JᵀJ`. Roundoff is about `ε·|uv|/h ≈ 1e-16 · 500 / 1e-6 = 5e-8` px, inside the bound.   measured by: that test.
  P6 Upstream boundary: `grep -rn "crate::" crates/kornia-slam/src/upstream | grep -v "crate::upstream"` prints 0 lines, and every new `upstream/` file starts with the `//! Upstream target:` header.   measured by: that grep, plus `head -1` on each file.
  P7 Per-frame tracking time, MH_01 `euroc.ron`, solo runs: median of `pr180/L-C` within ±3% of `pr180/S`.   measured by: the bench harness timing, 3 solo runs per tag, load within ±2 of each other.
  Mechanism counter: arithmetic-bearing lines in the lane's diff that are not moves. Predicted: 0.
    - Count with `git diff --color=always --color-moved=zebra --color-moved-ws=allow-indentation-change pr180/S pr180/L-C -- crates/kornia-slam/src`.
    - Keep only the plain added and removed lines (green `\e[32m+`, red `\e[31m-`), then keep those that contain an arithmetic operator (`+ - * /`, `sqrt`, `.dot`, `.cross`, `.transpose`) or a numeric literal.
    - The other non-moved lines are expected to be only `use`, `mod` and re-export lines, the `//!` headers, visibility (`pub(crate) fn` to `pub fn`), `Refined` to `RefinedPose` in `pnp.rs`, and one doc sentence in `PoseGraphEdge` that no longer links the solver. The results file lists each with that reason.

Kill rules:
  K1 The mechanism counter is 0, yet P1 or P3 differs (the identity theory is wrong: something other than the operation sequence changed, such as a conversion path or a reordered sum inside a matrix type) -> bisect the lane's commits with `cmp` and revert the first commit that differs. Do not accept the lane until that difference is explained.
  K2 P1 and P3 hold, yet the mechanism counter is above 0 (an arithmetic line changed off every exercised path, so it is untested, not proven equal) -> restore that line verbatim, or add a unit test that pins its old output, before the lane is accepted.
  K3 P2 fails -> the PGO extraction changed f32 operation order. Revert the PGO commit only; the refine commit can stand on its own.
  K4 A moved `pose_lm` test needs a tolerance change to pass (P4) -> not a pure move. Revert the refine commit.
  K5 P6 fails -> boundary violation. Fix it before Phase 4 integration; numerics are unaffected.
  K6 P7 is slower by more than 3% while P1 holds -> inlining was lost across the module boundary. Add `#[inline]` to `pinhole_point_jacobian` and `pose_jacobian_2x6` (no numeric change) and re-time. A loss that remains above 3% blocks adoption of C1.
  K7 The `pr180/S` loop run in P3 has 0 accepted loop solves -> P3 is untested, not passed. Re-run on a sequence that closes loops, and mark P3 provisional until then.
Bit-identity: `pr180/L-C` must `cmp` equal `pr180/S` on all of P1 and P3: MH_01 `euroc.ron`, 3_18 `lamaria-mono-imu.ron`, MH_01 loop `ScalarWeights`, and MH_01 loop `Drift`. With no knob, there is no knob-off case; every run is the identity case.
Noise: P1-P6 are deterministic, so they need no repeats. For P7, use 3 solo runs per tag, with machine load within ±2 between tags. The pipeline repeats bit-for-bit, so timing is the only noisy metric here.

## Design (what the lane builds)

### C1. `upstream/kornia_3d/refine_analytic.rs`

- **Header:** `//! Upstream target: kornia_3d::pnp::refine. Kept here until upstreamed; no kornia-slam types.`
- **Move:** the body of `pose_lm.rs` moves verbatim: constants, `kernel_from_params`, `Pose`, `Problem`, `damped_step`, the LM loop and the tests with their oracles.
- **Jacobian:** `Problem::normal_equations` calls `pose_jacobian_2x6` in place of the inline `jpi`/`jac` block (`pose_lm.rs:106-114`).
  - The operation order inside it is the same: `iz = 1.0 / pc.z`, the two `jpi` rows, then `a = R^T row` and `b = p × a`.
  - The accumulation loop reads `j[(0, i)]` and `j[(1, i)]`. These are copies, with no arithmetic.
- **Casts:** they come from `crate::upstream::kornia_algebra::cast` (below), not from `crate::pose_conversion`.
- **Delete:** `tracking/pose_estimation/pose_lm.rs` and `mod pose_lm;` (`pose_estimation/mod.rs:5`).
- **Callers:** `pnp.rs` calls `crate::upstream::kornia_3d::refine_analytic::{supports_kernel, refine_pose_analytic, RefinedPose}`. `RefinedPose` replaces `Refined`, and the generic fallback (`pnp.rs:287-301`) builds a `RefinedPose`.

### C2. `upstream/kornia_3d/pgo/`

Header for each file: `//! Upstream target: kornia_3d::pgo. Kept here until upstreamed; no kornia-slam types.`

| File | Moved from `sparse_pgo.rs` (verbatim) |
|---|---|
| `pgo/mod.rs` | `RESIDUAL_DIM`, `SparsePgoError`, the fallible `pose_to_se3`/`se3_to_pose`, `normalized_gravity`, `validate_delta`, `validate_pose`, `validate_se3`, `vec3_f64_is_finite`; `pub mod manifold; pub mod edge;` and re-exports |
| `pgo/manifold.rs` | `PoseManifold`, `Se3Manifold`, `GravityManifold` |
| `pgo/edge.rs` | `InformationEdge`, the sealed `EdgeTerm` (now in `pub(crate) mod sealed`, so the kornia-slam solver can call it but the public API stays sealed), `PoseGraphEdge`, both `EdgeTerm` impls, `cauchy_c_sq`, `weighted_relative_residual` |

What stays and what moves around it:
- **Stays in `sparse_pgo.rs`:** `SparsePgoResult`, the solver, its constants (`NUM_JACOBIAN_EPS`, `STEP_SIZE_TOLERANCE`, `LAMBDA_*`), `validate_damping`, `finite_l2_norm`, the test hooks and all tests.
- **Re-exports:** `sparse_pgo.rs` re-exports the moved public items (`pub use crate::upstream::kornia_3d::pgo::{...}`) and `weighted_relative_residual` (`pub(crate) use`). So `loop_closure/mod.rs:47-50`, `graph/mod.rs:22-26`, `pose_graph.rs:8-12`, `loop_closure/tests.rs` and kornia-slam's public paths stay unchanged.
- **Name clash:** inside `upstream/kornia_3d/pgo/`, the external crate is `kornia_3d`. If rustc reports the `kornia_3d` path as ambiguous, write `::kornia_3d::pgo::{PgoEdge, PgoParams}`.

### Shared conversions (moved so `upstream/` never imports `crate::pose_conversion`)

- **Casts:** `upstream/kornia_algebra/cast.rs` (header: `Upstream target: kornia_algebra`) gets `mat3_to_f32`, `mat3_to_f64`, `vec3_to_f32` and `vec3_to_f64` verbatim, with the test `precision_casts_are_element_wise_and_column_major`. kornia-algebra has no named f32/f64 cast at `4ada5a7`; it only has glam's `as_dmat3` through `Deref`.
- **Pose conversions:** `upstream/kornia_3d/pose_ext.rs` gets the quaternion `orthonormalized`, `pose_to_se3` and `se3_to_pose` (the infallible ones, `pose_conversion.rs:10-41`) verbatim.
- **`pose_conversion.rs`:** it replaces those bodies with `pub(crate) use` re-exports, so every caller listed by `grep pose_conversion::` stays unchanged. It keeps `RIGID_TOLERANCE`, `rigid`, `angle_deg`, `apply_reference_pose_correction` and its projection tests.
- **Effect on Phase 4 Step 8:** after this lane, Step 8 edits `upstream/kornia_3d/pose_ext.rs`, not `pose_conversion.rs`. There it deletes the quaternion `orthonormalized`, and `se3_to_pose` becomes `Pose3d::new(...).orthonormalized()`. The rest of Step 8 (`rigid` call sites, the P1-P3 test) is unchanged.

## Shared types and signatures the skeleton (Phase 2c) declares for this lane

These go in `upstream/kornia_3d/refine_analytic.rs`. Each has a `todo!()` body, and no default path reaches it: `pnp.rs` keeps calling `pose_lm` until Lane C switches it. Lane A calls only the two Jacobian functions.

```rust
//! Upstream target: kornia_3d::pnp::refine. Kept here until upstreamed; no kornia-slam types.

use kornia_3d::pnp::LMRefineParams;
use kornia_algebra::{Mat3AF32, Mat3F64, SMatF64, Vec2F32, Vec3AF32, Vec3F64};

/// Rows `d(u)/d(p_c)` and `d(v)/d(p_c)` of a pinhole projection at the camera-frame
/// point `point_cam` (`z > 0`), in pixels per metre.
pub fn pinhole_point_jacobian(fx: f64, fy: f64, point_cam: Vec3F64) -> [Vec3F64; 2];

/// Pixel Jacobian with respect to a right perturbation `T_cw · exp(δ)`, `δ = (υ, ω)` (the
/// order of `SE3F64::retract`), of the world-to-camera pose with rotation `rotation_cw`,
/// at world point `point_world` whose camera-frame position is `point_cam`.
pub fn pose_jacobian_2x6(
    rotation_cw: &Mat3F64,
    point_world: Vec3F64,
    point_cam: Vec3F64,
    fx: f64,
    fy: f64,
) -> SMatF64<2, 6>;

/// Whether [`refine_pose_analytic`] implements `params`' robust kernel (Identity, or Huber
/// with a finite positive scale).
pub fn supports_kernel(params: &LMRefineParams) -> bool;

/// A refined world-to-camera pose and whether a tolerance, not the iteration cap, ended it.
pub struct RefinedPose {
    pub rotation: Mat3AF32,
    pub translation: Vec3AF32,
    pub converged: bool,
}

/// Pose-only LM on undistorted pixels; `None` for an unsupported kernel, fewer than 3
/// points, a start point behind the camera, or a singular damped system.
pub fn refine_pose_analytic(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    rotation: &Mat3AF32,
    translation: &Vec3AF32,
    params: &LMRefineParams,
) -> Option<RefinedPose>;
```

The skeleton also creates the following. Lane C moves code into them, and the skeleton declares no items there, so the old definitions never exist twice:
- `upstream/kornia_3d/pgo/{mod.rs, manifold.rs, edge.rs}`: headers only;
- `upstream/kornia_algebra/{mod.rs, cast.rs}`: `pub(crate) mod cast;`, header only.

Lane A must not import `pose_lm`, which this lane deletes. Lane D may reuse `upstream::kornia_3d::pgo`'s validators. It must not import solver internals from `sparse_pgo.rs`, which stay private.

## Files Lane C touches

- **New or filled:** `crates/kornia-slam/src/upstream/kornia_3d/refine_analytic.rs`, `upstream/kornia_3d/pgo/{mod.rs, manifold.rs, edge.rs}`, `upstream/kornia_algebra/cast.rs`.
- **Edited:**
  - `upstream/kornia_3d/pose_ext.rs`: gains the three conversions;
  - `upstream/kornia_3d/mod.rs` and `upstream/mod.rs`: not edited. The skeleton adds every `mod` line there, `gpnp` (lane B) included (card A, "Skeleton contract"). A missed line is a skeleton fix, not a lane edit, because lane B shares the file;
  - `pose_conversion.rs`;
  - `loop_closure/sparse_pgo.rs`;
  - `tracking/pose_estimation/pnp.rs`;
  - `tracking/pose_estimation/mod.rs`.
- **Deleted:** `tracking/pose_estimation/pose_lm.rs`.
- **Not touched:** `loop_closure/{mod.rs, graph/mod.rs, pose_graph.rs, tests.rs}`, `mapping/vi_ba_schur.rs`, `configs/*.ron`, `CHANGELOG.md` (Phase 6 writes the entry).
- **Commits:** two, so K3 and K4 can revert one without the other.
  - `refactor(tracking): analytic pose refine and its 2x6 Jacobian under upstream/kornia_3d` (C1 plus the casts);
  - `refactor(loop_closure): PGO manifolds and edges under upstream/kornia_3d/pgo` (C2 plus the pose conversions).

## Not in this card

- The speed of the analytic refine against kornia-3d's numerical `refine_pose_lm`. That belongs to the kornia-rs PR's own card, with a criterion bench.
- Moving the faer solver upstream.
- ORB-SLAM3's 4×10-iteration χ² re-classification in `PoseOptimization` (`Optimizer.cc:852`, `deltaMono = sqrt(5.991)`). That is a behaviour change, and it belongs to card A if anywhere.
