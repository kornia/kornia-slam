# Inertial Initialization Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move inertial initialization into `initialization/inertial/` while preserving the public `initialization::*` API and all runtime behavior.

**Architecture:** The crate-level initialization facade keeps both implementation modules private and re-exports their supported API. The new `inertial` child owns the initializer in `mod.rs` and keeps optimizer factors private in `factor.rs`; `SlamSystem` continues to consume only facade exports.

**Tech Stack:** Rust 2024 edition, Cargo workspace, `kornia-slam` library.

---

### Task 1: Guard the public initialization facade

**Files:**
- Modify: `crates/kornia-slam/tests/system_api.rs`

1. Add an integration test that imports `ImuInitConfig`, `ImuInitReject`,
   `ImuInitResult`, `ImuInitializer`, `KeyframeVelocity`, `TwoViewEstimate`, and
   `TwoViewInitConfig` from `kornia_slam::initialization` rather than a source
   submodule.
2. Construct an `ImuInitializer` and the default two-view configuration so the
   test checks real public visibility rather than unused imports.
3. Run `cargo test -p kornia-slam --test system_api` and expect all tests to pass
   before the move, establishing the compatibility baseline.

### Task 2: Nest inertial implementation files

**Files:**
- Move: `crates/kornia-slam/src/initialization/imu.rs` to `crates/kornia-slam/src/initialization/inertial/mod.rs`
- Move: `crates/kornia-slam/src/initialization/inertial_factor.rs` to `crates/kornia-slam/src/initialization/inertial/factor.rs`
- Modify: `crates/kornia-slam/src/initialization/mod.rs`
- Modify: `crates/kornia-slam/src/initialization/inertial/mod.rs`

1. Create the `initialization/inertial/` directory through `git mv` of
   `imu.rs`, then move `inertial_factor.rs` to its `factor.rs` child.
2. Replace `mod imu; mod inertial_factor;` with `mod inertial;` in the facade and
   re-export the same five inertial API types from `inertial`.
3. Declare `mod factor;` in `inertial/mod.rs` and change the factor import to
   `self::factor::{InertialInitFactor, KfConst, WeightedZeroPrior}`.
4. Run `cargo test -p kornia-slam --test system_api` and
   `cargo test -p kornia-slam initialization` and expect both suites to pass.

### Task 3: Verify and publish the PR update

**Files:**
- No additional source files.

1. Run `cargo fmt --all -- --check`.
2. Run `cargo clippy -p kornia-slam --all-targets -- -D warnings`.
3. Run `cargo test --workspace`.
4. Run `cargo check --workspace` and `git diff --check`.
5. Commit with `refactor: group inertial initialization module`.
6. Push `refactor/slam-system` and verify PR #74 recognizes the new commit.
