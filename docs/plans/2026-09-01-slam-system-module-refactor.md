# SLAM System Module Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the SLAM crate around `system`, `tracking`, and `initialization`, and expose the runtime as `SlamSystem` without changing tracking behavior.

**Architecture:** Move runtime orchestration from `pipeline` into `system`; move tracking-specific estimators, policies, state, and results into `tracking`; move bootstrap and inertial initialization into `initialization`. Preserve the old root-level type names through deprecated aliases while making the new names canonical.

**Tech Stack:** Rust 2024 edition, Cargo workspace, `kornia-slam` library and application crates.

---

### Task 1: Establish the public API expectation

**Files:**
- Modify: `crates/kornia-slam/tests/pipeline_api.rs`

1. Update the integration test to construct `SlamSystem` from `SlamConfig` and `LoopClosingConfig`.
2. Add a compatibility compile check for `SlamPipeline`, `PipelineConfig`, and `PgoPipelineConfig`.
3. Run `cargo test -p kornia-slam --test pipeline_api` and verify that the new names fail to resolve.

### Task 2: Create tracking and initialization ownership

**Files:**
- Create: `crates/kornia-slam/src/tracking/mod.rs`
- Create: `crates/kornia-slam/src/tracking/state.rs`
- Create: `crates/kornia-slam/src/tracking/policy.rs`
- Move: `crates/kornia-slam/src/estimation/optical_flow.rs`
- Move: `crates/kornia-slam/src/estimation/map_projection/`
- Move: `crates/kornia-slam/src/estimation/pnp.rs`
- Create: `crates/kornia-slam/src/tracking/pose_estimation/mod.rs`
- Create: `crates/kornia-slam/src/initialization/mod.rs`
- Move: `crates/kornia-slam/src/estimation/two_view.rs`
- Move: `crates/kornia-slam/src/estimation/imu_init.rs`
- Move: `crates/kornia-slam/src/estimation/inertial_init_factor.rs`
- Delete: `crates/kornia-slam/src/estimation/mod.rs`
- Delete: `crates/kornia-slam/src/system.rs`

1. Move files with `git mv` so history remains visible.
2. Split policies and state/result definitions from the old `system.rs`.
3. Place the shared `Estimate` type in `tracking::pose_estimation`.
4. Repair internal imports without changing algorithms or defaults.
5. Run `cargo check -p kornia-slam` until module ownership compiles.

### Task 3: Rename the runtime facade

**Files:**
- Move: `crates/kornia-slam/src/pipeline/mod.rs` to `crates/kornia-slam/src/system/mod.rs`
- Move: `crates/kornia-slam/src/pipeline/config.rs` to `crates/kornia-slam/src/system/config.rs`
- Modify: `crates/kornia-slam/src/system/mod.rs`
- Modify: `crates/kornia-slam/src/system/config.rs`
- Modify: `crates/kornia-slam/src/lib.rs`

1. Rename `SlamPipeline` to `SlamSystem` throughout the runtime implementation.
2. Rename `PipelineConfig` to `SlamConfig` and `PgoPipelineConfig` to `LoopClosingConfig`.
3. Export the new modules and preferred root-level types.
4. Add deprecated aliases for the old root-level type names.
5. Run `cargo test -p kornia-slam --test pipeline_api` and verify both preferred and compatibility APIs compile.

### Task 4: Migrate workspace consumers and documentation

**Files:**
- Modify: `apps/kornia-slam-app/src/main.rs`
- Modify: Rust documentation and tests found by `rg 'pipeline|estimation|SlamPipeline|PipelineConfig|PgoPipelineConfig'`.

1. Update application imports and construction to the preferred API.
2. Update internal test imports to the new module paths.
3. Update user-facing Rust documentation that names the old facade.
4. Confirm old identifiers remain only in compatibility aliases and their tests.

### Task 5: Verify and prepare the pull request

**Files:**
- Modify: formatting changes produced by `cargo fmt`.

1. Run `cargo fmt --all -- --check`.
2. Run `cargo test -p kornia-slam --lib`.
3. Run `cargo test -p kornia-slam --test pipeline_api`.
4. Run `cargo check --workspace`.
5. Inspect `git diff --check` and the complete diff for accidental behavior changes.
6. Commit the refactor, push `refactor/slam-system`, and open a PR against `develop` summarizing ownership changes, compatibility, and verification.
