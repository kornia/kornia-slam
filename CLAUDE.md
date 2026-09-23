# CLAUDE.md

Guidance for AI agents and human contributors working in this repository.
Adapted from kornia-rs [`AGENTS.md`](https://github.com/kornia/kornia-rs/blob/main/AGENTS.md).

## Project Overview

**kornia-slam** is a modular SLAM library in Rust, built on
[kornia-rs](https://github.com/kornia/kornia-rs). It provides a SLAM runtime plus the
building blocks it is made of — sensor estimators, odometry, mapping and loop closing —
designed to grow across sensors, map representations and compute targets, and to expose
its map to other systems and agents. See [ROADMAP.md](ROADMAP.md) for direction.

---

## Core Principles

These matter more than anything else in this file.

1. **Don't over-comment.** Code should explain itself through naming and structure.
   Comment only the *why* that isn't obvious — a non-trivial algorithm, a paper or
   ORB-SLAM3 reference, a subtle invariant. No comments that restate the code, narrate
   steps, or describe what changed. Doc comments on public items stay short.
2. **Modular, not monolithic.** Small, focused functions and types with one
   responsibility each. Don't grow long functions that do many things — split them.
   Put code in the module that owns the concern instead of wherever it is convenient.
3. **Don't reinvent.** Before writing anything, search this workspace's crates and
   kornia-rs (`kornia-3d`, `kornia-algebra`, `kornia-imgproc`, `kornia-bow`, …) for an
   existing implementation, and reuse it. If something general is missing, it belongs
   upstream in kornia-rs, not duplicated here.

---

## Build & Development Commands

```bash
cargo fmt --all -- --check                  # Formatting
cargo clippy --all-targets -- -D warnings   # Lint (warnings are denied)
cargo test --workspace                      # Full test suite
cargo test -p <crate>                       # A single crate
```

CI (`.github/workflows/`) is the source of truth for the exact lint and test matrix.

---

## Architecture

```
crates/*    ← library crates
apps/*      ← applications: CLI, data sources, visualization, evaluation
```

- **The library owns the runtime.** Apps are composition roots that wire sources,
  frontends and outputs together — they never become a second pipeline.
- **Subsystems over monoliths.** Keep tracking, mapping, inertial and loop-closing
  concerns separable, with clear ownership of state.
- **Pluggable by design.** Features, sensors, estimators and map representations
  should sit behind interfaces rather than being hard-wired to one implementation.

---

## Code Conventions

### General

- Rust edition **2024**; `rustfmt` and `clippy` before every commit
- Prefer **borrowing over cloning**, especially for images and map data
- No `unwrap()` / `expect()` / `panic!()` in new library code — propagate errors with `?`
- `thiserror` for library error types — no `anyhow` in library crates
- Tests live in `#[cfg(test)]` modules

### Documentation

Public items get a concise doc comment: what it does, plus units and coordinate-frame
conventions where they apply (state them explicitly for poses and transforms).
Add `# Errors` only when the failure modes aren't obvious. Keep it short — see Core Principles.

### Safety

- Avoid `unsafe`. Every `unsafe` block must be preceded by a `// SAFETY:` comment
  explaining exactly why it is sound.

### Performance

- Avoid allocations in per-frame hot paths
- Measure before and after changing a core stage

---

## Verification

Unit tests are necessary but not sufficient — SLAM regressions show up end-to-end.
For changes that can affect accuracy or runtime, run on a real dataset sequence against
the base branch and report the numbers (trajectory error, robustness, timing).
Refactors should be behavior-preserving.

---

## Contributing

1. **Keep PRs focused** — one concern per PR; large changes land as stacked PRs.
2. **Test locally first** — fmt, clippy and tests must pass.
3. **Document** new or changed public API — briefly.
4. **Write tests** — new functionality needs tests; bug fixes need a regression test.
5. **PR description** — what changed and why, breaking changes, and evaluation results
   where relevant.

Commits follow [Conventional Commits](https://www.conventionalcommits.org) with a scope,
e.g. `refactor(mapping): …`, `fix(tracking): …`.
