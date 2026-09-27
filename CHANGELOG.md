# Changelog

All notable changes to kornia-slam are recorded here, newest first. Each entry is
written for users of the `kornia-slam` and `kornia-sensors` crates and the
`kornia-slam` CLI — what changed and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Keep the link references at the bottom
     in sync so each version links to its diff. -->

## [Unreleased]

First public release, planned as 0.1.0. The API will change between minor versions
through the 0.x series.

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
and ATE/RPE evaluation against ground truth.

**Development.** A Pixi environment with the standard lint and test tasks. The
minimum supported Rust version is 1.89, set by kornia-rs.

[Unreleased]: https://github.com/kornia/kornia-slam/commits/develop
