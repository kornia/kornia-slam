# CLAUDE.md

Guidance for AI agents and human contributors working in this repository.
Adapted from kornia-rs [`AGENTS.md`](https://github.com/kornia/kornia-rs/blob/main/AGENTS.md).

## Project Overview

**kornia-slam** is a real-time visual-inertial SLAM system written in Rust, built on
[kornia-rs](https://github.com/kornia/kornia-rs). It supports monocular, stereo and
visual-inertial ORB SLAM with local BA, VI-BA, DBoW2 place recognition and loop closure
with pose-graph optimization.

---

## Build & Development Commands

Plain cargo — there is no pixi environment in this repo. These mirror CI
(`.github/workflows/rust_lint.yml`, `rust_test.yml`):

```bash
cargo fmt --all -- --check                          # Formatting
cargo clippy --all-targets -- -D warnings           # Lint: library + app (default features, viz)
cargo clippy --manifest-path apps/kornia-slam-app/Cargo.toml --no-default-features --all-targets -- -D warnings  # App without rerun
cargo test --workspace                              # Full test suite
cargo test -p kornia-slam                           # Library tests only
cargo run -p kornia-slam-app -- --help              # App builds and runs
```

Running the system on data:

```bash
cargo run --release -p kornia-slam-app -- euroc --data /path/to/MH_01_easy                  # mono
cargo run --release -p kornia-slam-app -- euroc --data /path/to/MH_01_easy --stereo --imu --evaluate
```

The `oakd` feature (depthai-sys builds from source) is intentionally not in CI.

---

## Architecture

### Workspace Structure

```
Cargo.toml              ← workspace root (members: crates/*, apps/*)
crates/kornia-slam      ← the SLAM library: runtime + building blocks
crates/kornia-sensors   ← sensor types (IMU)
apps/kornia-slam-app    ← binary `kornia-slam`: CLI, sources, TUI, Rerun viz, evaluation
```

### Library (`crates/kornia-slam/src`)

| Module | Purpose |
|---|---|
| `system/` | `SlamSystem`, `SlamConfig`, inertial runtime state — the orchestrator |
| `tracking/` | Tracker, motion prediction, local-map selection, pose estimation (`MapProjectionEstimator`, PnP), KLT, keyframe policy |
| `initialization/` | Two-view bootstrap, IMU initialization, inertial init factor |
| `mapping/` | `Map`, keyframes, map points, local mapping, map growth and fusion |
| `loop_closure/` | Loop verification, fusion, pose graph |
| `place_recognition.rs` | DBoW2 keyframe database |
| `vi_ba_schur.rs` | Visual-inertial BA (Schur complement) |
| `stereo.rs`, `sensor_rig.rs` | Stereo matching; camera/IMU calibration (`SensorRig`) |
| `pipeline.rs` | Deprecated `SlamPipeline`/`PipelineConfig` aliases — do not extend |

The library owns the runtime. `apps/*` are composition roots (CLI, data sources,
feature extraction, visualization, evaluation) — never a second pipeline.

### Key Patterns

- **Reuse kornia-rs**: geometry, BA, Lie groups and image processing come from
  `kornia-3d`, `kornia-algebra`, `kornia-imgproc`, `kornia-bow`. Don't reimplement them here;
  if something is missing, it likely belongs upstream.
- **Map mutation** goes through the canonical `Map` mutation API — don't poke internal
  collections from outside `mapping/`.
- **Naming**: conversion functions use `<output>_from_<input>` (e.g. `pose_from_matrix`),
  matching kornia-rs. Poses keep their existing frame-named style (`camera_to_body`,
  `pose_world_to_cam`) — just make the direction explicit.
- **ORB-SLAM3 parity**: many algorithms are ports. When porting, cite the ORB-SLAM3
  function you are following and call out deliberate deviations.

---

## Code Conventions

### General

- Rust edition **2024**
- `rustfmt` and `clippy` before every commit — **warnings are denied**
- Prefer **borrowing over cloning**, especially for images, frames and map data
- No new `unwrap()` / `expect()` in library code — propagate errors with `?`
  (existing ones are debt; don't add more)
- Tests live in `#[cfg(test)]` modules; run per-crate with `cargo test -p <crate>`

### Documentation

Every public item **must** have a doc comment: a one-line summary, then units,
coordinate frames and non-obvious behavior. Public functions need `# Arguments`,
`# Returns` and `# Errors` sections. **State the frame convention** for every pose
(which frame maps to which) — frame mix-ups are the most common SLAM bug.

### Safety

- Avoid `unsafe`. Every `unsafe` block must be preceded by a `// SAFETY:` comment
  explaining exactly why it is sound.

### Error Handling

- `thiserror` for library error types — no `anyhow` in library crates
- No `.unwrap()`, `.expect()` or `panic!()` in new library code
- In tests, prefer `#[test] fn ... -> Result<(), Box<dyn Error>>` with `?`
- Error variants should be descriptive and carry context

### Performance

- Avoid allocations in the per-frame hot path (tracking, matching, projection)
- Don't clone frames, keyframes or map points unnecessarily — pass references
- Measure before and after any change to a core stage

---

## Verification

Unit tests are necessary but not sufficient — SLAM regressions show up end-to-end.

- **Default check**: run on a EuRoC easy sequence (e.g. MH_01) with `--max-frames 500`
  and `--evaluate`, and compare ATE / scale / keyframe and map-point counts against the
  base branch.
- **Changes to tracking, IMU, BA or loop closure**: run the benchmark (`/slam-bench` skill)
  before and after, and report the numbers in the PR.
- **Refactors should be behavior-preserving**: diff trajectories/logs against the base
  branch. The first diverging frame usually points straight at the bug.
- **PGO / loop-closure runs are non-deterministic** (unseeded verification RANSAC) —
  seed both arms when A/B testing.

---

## Contributing

### Pull Requests

1. **Keep PRs focused** — one concern per PR. Large refactors land as stacked PRs.
2. **Test locally first** — fmt, both clippy configs and `cargo test --workspace` must pass.
3. **Update documentation** — new or changed public API needs updated doc comments.
4. **Write tests** — new functionality needs unit tests; bug fixes need a regression test.
5. **PR description** — what changed and why, breaking changes, and EuRoC numbers
   (ATE, scale, KFs/points) for anything that can affect accuracy or runtime.

### Commit Style

[Conventional Commits](https://www.conventionalcommits.org) with a module scope:
`refactor(map): …`, `fix(tracking): …`, `feat(loop_closure): …`, `docs(readme): …`.

---

## Quick Reference Checklist (before every PR)

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes (and the `--no-default-features` app config)
- [ ] `cargo test --workspace` passes
- [ ] EuRoC run shows no accuracy regression (numbers in the PR)
- [ ] All new/changed `pub` items have doc comments, including pose frame conventions
- [ ] Every `unsafe` block has a `// SAFETY:` comment
- [ ] No `unwrap()`/`expect()` added to library code
- [ ] Commit messages follow Conventional Commits
