# Changelog

All notable changes to kornia-slam are recorded here, newest first. Entries are
curated for users of the Rust crates and the `kornia-slam` CLI — what changed
and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
 dated section and reset [Unreleased]. Reference the diff range so the changelog
 stays navigable (see the link references at the bottom). -->

## [Unreleased]

## [0.1.0] — 2026-09-26

Early release of kornia-slam. The API will change between minor versions; expect
breaking changes through the 0.x series while we complete the stack and stabilize
interfaces.

- Visual ORB-SLAM pipeline with:
  - Monocular, stereo, and visual-inertial modes
  - Local bundle adjustment, including visual-inertial BA
  - Place recognition (DBoW2) and loop closure with pose-graph optimization
- Sources:
  - EuRoC MAV (ASL format)
  - Hilti-Trimble dataset reader
  - MCAP recordings
  - Live UVC webcams and Luxonis OAK-D (feature-gated)
- Tooling and UX:
  - Terminal UI and Rerun visualization/streaming
  - ATE/RPE evaluation and EuRoC benchmarking helper
  - Pixi environment with standard lint/test tasks
- Architecture and code health:
  - Modular SLAM runtime split into tracking, mapping, inertial, and loop-closing
    subsystems with clear ownership of state
  - Map/covisibility graph ownership clarified and refactored
  - Numerous robustness fixes across initialization, pose estimation and VI-BA

[Unreleased]: https://github.com/kornia/kornia-slam/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kornia/kornia-slam/releases/tag/v0.1.0

