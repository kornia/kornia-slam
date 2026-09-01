# Initialization Boundary Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make initialization results side-effect-free, self-identifying, diagnosable, and independent of tracking internals.

**Architecture:** `initialization` computes typed visual or inertial results and never mutates tracking state. `SlamSystem` validates and atomically applies inertial results to the live map and tracker, while optimizer factors remain private implementation details.

**Tech Stack:** Rust 2024 edition, Cargo workspace, `kornia-slam` library.

---

### Task 1: Decouple two-view results from tracking

**Files:**
- Modify: `crates/kornia-slam/src/initialization/two_view.rs`
- Modify: `crates/kornia-slam/src/initialization/mod.rs`
- Modify: `crates/kornia-slam/src/system/mod.rs`

1. Add a test that a low-match two-view request returns a typed rejection.
2. Replace the nested tracking `Estimate` with direct pose, match, and inlier fields on `TwoViewEstimate`.
3. Re-export the stable two-view API from `initialization::mod` and make the source module private.
4. Update system call sites and run the initialization and system tests.

### Task 2: Introduce typed inertial initialization results

**Files:**
- Modify: `crates/kornia-slam/src/initialization/imu.rs`
- Modify: `crates/kornia-slam/src/initialization/inertial_factor.rs`
- Modify: `crates/kornia-slam/src/initialization/mod.rs`

1. Add tests for missing extrinsics, insufficient keyframes, invalid configuration, and keyframe IDs in velocity results.
2. Introduce `ImuInitReject` and return `Result<ImuInitResult, ImuInitReject>` from `try_initialize`.
3. Replace positional velocities with `{ keyframe_idx, velocity_world }` records.
4. Make the low-level optimizer private and return an internal named solution instead of a tuple.
5. Remove direct stderr writes and the stale documentation reference.
6. Make inertial factors private behind the initialization facade.

### Task 3: Apply initialization in the system

**Files:**
- Modify: `crates/kornia-slam/src/system/mod.rs`
- Test: `crates/kornia-slam/src/system/mod.rs`

1. Add a failing test using out-of-order keyframe insertion and ID-tagged velocity assignments.
2. Add a system-owned application helper that validates all assignments before mutating the map.
3. Move map scaling, gravity alignment, bias updates, and tracking-state updates out of `ImuInitializer`.
4. Surface typed rejection details through the existing system debug channel.
5. Run system and inertial initialization tests.

### Task 4: Verify and update PR #74

**Files:**
- Modify: `docs/plans/2026-09-01-initialization-boundary-hardening.md`

1. Run `cargo fmt --all -- --check`.
2. Run `cargo clippy -p kornia-slam --all-targets -- -D warnings`.
3. Run `cargo test --workspace`.
4. Run `cargo check --workspace` and `git diff --check`.
5. Commit, push `refactor/slam-system`, and update the PR summary with the hardened initialization boundary.
