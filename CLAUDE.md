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
- **Upstream what is general.** Solvers, geometry, camera models and image ops belong in
  kornia-rs; reuse them instead of reimplementing here.

---

## Code Conventions

### General

- Rust edition **2024**; `rustfmt` and `clippy` before every commit
- Prefer **borrowing over cloning**, especially for images and map data
- No `unwrap()` / `expect()` / `panic!()` in new library code — propagate errors with `?`
- `thiserror` for library error types — no `anyhow` in library crates
- Tests live in `#[cfg(test)]` modules

### Documentation

Every public item **must** have a doc comment: a one-line summary, then units,
coordinate frames and non-obvious behavior. Public functions need `# Arguments`,
`# Returns` and `# Errors` sections. State frame conventions explicitly for poses and
transforms.

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
3. **Document** new or changed public API.
4. **Write tests** — new functionality needs tests; bug fixes need a regression test.
5. **PR description** — what changed and why, breaking changes, and evaluation results
   where relevant.

Commits follow [Conventional Commits](https://www.conventionalcommits.org) with a scope,
e.g. `refactor(mapping): …`, `fix(tracking): …`.
