# Changelog

All notable changes to kornia-slam are recorded here, newest first. Each entry is
written for users of the `kornia-slam` and `kornia-sensors` crates and the
`kornia-slam` CLI — what changed and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Keep the link references at the bottom
     in sync so each version links to its diff. -->

## [Unreleased]

**Local BA no longer copies the whole map.** Each local bundle adjustment
captured every keyframe, landmark and IMU factor under the map lock, so its
cost, and the stall it caused tracking, grew with the map. It now captures only
the local window. **Breaking:** `Map::ba_snapshot` takes a `BaWindow`; use
`mapping::bundle_adjustment::local_window` for the standard selection.

## [0.1.0] — 2026-09-27

First public release. The API will change between minor versions through the 0.x
series.

**Visual and visual-inertial SLAM on kornia-rs.** An ORB-based pipeline for
monocular, stereo and stereo-inertial input, built on the kornia-rs 0.2 crates.
The `kornia-slam` crate owns the runtime: tracking, local mapping with bundle
adjustment (including visual-inertial BA with IMU initialization), and loop closing
through DBoW2 place recognition, Sim(3) verification and pose-graph optimization.

**IMU processing in `kornia-sensors`.** IMU preintegration, noise models and
calibration types, usable independently of the SLAM runtime.

**The `kornia-slam` CLI** (`kornia-slam-app`, not published to crates.io) runs the
pipeline on EuRoC, Hilti and MCAP recordings, and live on UVC cameras and Luxonis
OAK-D (behind the `uvc` and `oakd` features). It has a terminal UI, Rerun streaming,
and ATE/RPE evaluation against ground truth. Prebuilt binaries for linux x86_64 and
aarch64 (glibc 2.35+: Ubuntu 22.04 and later, Jetson JetPack 6) are attached to the
GitHub release.

**Integrations.** [INTEGRATIONS.md](INTEGRATIONS.md) shows how to run kornia-slam as a
visual-inertial odometry task inside [Copper](https://github.com/copper-project/copper-rs)
through `cu-kornia-vio`.

**Development.** A Pixi environment with the standard lint and test tasks. The
minimum supported Rust version of the library crates is 1.89, set by kornia-rs;
building the CLI needs 1.91 because of Rerun.

[Unreleased]: https://github.com/kornia/kornia-slam/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kornia/kornia-slam/releases/tag/v0.1.0
