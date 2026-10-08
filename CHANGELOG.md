# Changelog

All notable changes to kornia-slam are recorded here, newest first. Each entry is
written for users of the `kornia-slam` and `kornia-sensors` crates and the
`kornia-slam` CLI — what changed and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Keep the link references at the bottom
     in sync so each version links to its diff. -->

## [Unreleased]

**Runs are defined by a configuration file.** The CLI takes one RON run file,
`kornia-slam --config run.ron`, naming the `source` (EuRoC, Hilti, MCAP, OAK-D
or UVC, with its paths, frame range and calibration) and the `system`: the
sensors, ORB settings, keyframe policy, local-mapping execution and loop
closing. `configs/` has an example per source. Relative paths resolve against
the run file. Invalid settings, and sensors
the source cannot provide, are rejected before any data is read. The library
gains `PipelineConfig` (in `kornia_slam::system`, re-exported at the root), with RON loading behind the optional `serde`
feature; the CLI's run files embed it as `system`. **Breaking (CLI):** the
source subcommands (`euroc`, `hilti`, `mcap`, `oakd`, `uvc`) and their flags,
and `--n-keypoints`, `--local-mapping`, `--vocab`, `--apply-pgo`, `--stereo`
and `--imu`, are removed; `euroc --data D --stereo --imu` becomes a run file
with `source: Euroc((data: "D"))` and
`system: (version: 1, sensors: (cameras: Stereo, imu: true))`. `--evaluate` and
`--eval-out` are now global options. `PipelineConfig::resolve_paths` and
`PipelineConfig::check_version` let other file formats embed a configuration.
Loop closing is one setting, `Enabled(vocabulary: …)`, which always applies
pose-graph correction to accepted loops and needs stereo or IMU input; there
is no detection-only mode.

**One configuration, one construction call, one processing call.**
`SlamSystem::build(config, rig)` assembles the system from a `PipelineConfig`
and the source's calibrated `SensorRig`, and `SlamSystem::process` takes a
`SensorFrame` of images and IMU samples. The system now owns ORB extraction,
stereo matching, fisheye keypoint mapping and frame history, so an embedding
no longer reimplements them; `frontend_observation()` exposes the raw-image
keypoints and extraction time for overlays. Pipeline files gain
`frontend.stereo_close_depth` (baselines, metres or disabled); the remaining
former `SlamConfig` settings (two-view initialization, map projection, loss
recovery, loop correction) are algorithm tuning, set from Rust through
`OrbSlamPipeline::tuning` and kept out of the file format. Existing
configuration files resolve to the same settings as before. `kornia-sensors` gains `SensorFrame`,
so sources can produce input without depending on `kornia-slam` (which
re-exports it), and `SensorRig` gains an optional fisheye model for sources
that supply raw fisheye images.
**Breaking (library):**

| Removed | Replacement |
| --- | --- |
| `SlamConfig`, `SlamSystem::new`, `SlamSystem::with_rig` | `SlamSystem::build(PipelineConfig, SensorRig)`, with tuning in `OrbSlamPipeline::tuning` |
| `SlamConfig::debug` | `SlamSystem::set_debug` |
| `SlamSystem::set_vocabulary` | a `loop_closing` branch naming the vocabulary |
| `SlamSystem::set_imu_extrinsics` | IMU calibration on the `SensorRig` |
| `SlamSystem::process_frame` (prepared features) | `SlamSystem::process(SensorFrame)` |
| `LoopClosingConfig::require_imu_initialized` | derived from the rig |
| `PipelineConfig::{slam_config, orb_detector, load_vocabulary}`, `SensorSelection::select_rig` | done by `SlamSystem::build` |
| `kornia_slam::pipeline` | `kornia_slam::system`; common types are re-exported at the crate root |
| `pipeline::{KeyframeConfig, MappingExecution, Stage}` | `KeyframePolicy`, `LocalMappingMode`; the stage list is gone, `Display` describes the pipeline |

`SlamSystem` no longer accepts features computed outside it; such callers pass
images, or compose the tracking and mapping building blocks directly, which
remain public. `TwoViewInitConfig::default()` now carries the triangulation
gates the runtime used (`max_midpoint_gap` 0.25, `max_reprojection_error` 3.0).

**Loop correction replays identically.** Loop-verification RANSAC now uses a
fixed seed, so runs with loop correction are reproducible.

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
