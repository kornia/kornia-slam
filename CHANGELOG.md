# Changelog

All notable changes to kornia-slam are recorded here, newest first. Each entry is
written for users of the `kornia-slam` and `kornia-sensors` crates and the
`kornia-slam` CLI — what changed and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Keep the link references at the bottom
     in sync so each version links to its diff. -->

## [Unreleased]

**Stereo-inertial tracking rides through fast motion instead of resetting.**
Once the IMU has settled, a tracking failure now coasts on the inertial
prediction for up to 5 s (was 1 s, ORB-SLAM3's `time_recently_lost`), keeps
inserting stereo keyframes at the predicted pose every 0.2 s so the map covers
the view the camera turned to, and widens the projection search at once
rather than over several seconds. On EuRoC V2_03 stereo+IMU the system no
longer resets (21 resets before, ATE 1.66 m → 0.88 m with synchronous
mapping); the other ten sequences are unchanged. `MapProjectionConfig`'s
`search_widen_per_sec` and `max_search_scale` are replaced by
`lost_search_scale`, and `TrackingLossRecoveryPolicy` gains
`keyframe_interval_while_lost_sec`.

**EuRoC stereo pairs are matched by timestamp.** The source paired left and
right images by position, which is off by one frame for all of MH_04 and
drifts through V2_03's dropped left frames; stereo results on those two
sequences were invalid.

**Runs are defined by a configuration file.** The CLI takes one RON run file,
`kornia-slam --config run.ron`, with a `source` (EuRoC, Hilti, MCAP, OAK-D or
UVC: paths, frame range, calibration) and a `system` (sensors, ORB settings,
keyframe policy, local-mapping execution, loop closing). `configs/` has an
example per source, and relative paths resolve against the run file. Invalid
settings, and sensors the source cannot provide, are rejected before any data
is read. `--evaluate` refuses a source without ground truth instead of scoring
the trajectory against itself. **Breaking (CLI):** the source subcommands
(`euroc`, `hilti`, `mcap`, `oakd`, `uvc`) and their flags, and
`--n-keypoints`, `--local-mapping`, `--vocab`, `--apply-pgo`, `--stereo` and
`--imu`, are removed: `euroc --data D --stereo --imu` becomes
`source: Euroc((data: "D"))` with
`system: (version: 1, sensors: (cameras: Stereo, imu: true))`.

**One configuration, one construction call, one processing call.**
`SlamSystem::build(config, rig)` assembles the system from a `SystemConfig` and
the source's calibrated `SensorRig`; `SlamSystem::process` takes a `SensorFrame`
of images and IMU samples. The system now owns ORB extraction, stereo matching,
fisheye keypoint mapping and frame history; `frontend_observation()` exposes
the raw-image keypoints and extraction time, and `tracking_duration()` the time
spent in tracking and mapping. `SystemConfig` loads from RON behind the
optional `serde` feature. Algorithm tuning (two-view initialization, map
projection, loss recovery, loop correction) is set from Rust through
`OrbSlamPipeline::tuning` and kept out of the file format. Loop closing is one
setting, `Enabled(vocabulary: …)`: accepted loops are always corrected, and it
needs stereo or IMU input. Stereo systems back-project close keypoints within
35 baselines by default (`frontend.stereo_close_depth`); `SlamConfig` left this
off unless the caller set it. `kornia-sensors` gains `SensorFrame`, so sources
can produce input without depending on `kornia-slam`, and `SensorRig` gains an
optional fisheye model for sources that supply raw fisheye images.
**Breaking (library):**

| Removed | Replacement |
| --- | --- |
| `SlamConfig`, `SlamSystem::new`, `SlamSystem::with_rig` | `SlamSystem::build(SystemConfig, SensorRig)` |
| `SlamConfig::{two_view_init, map_projection, tracking_loss_recovery}` | `OrbSlamPipeline::tuning` |
| `SlamConfig::{keyframe_policy, local_mapping}` | `OrbSlamPipeline::{keyframes, mapping.execution}` |
| `SlamConfig::stereo_close_depth_m` (default off) | `OrbFrontendConfig::stereo_close_depth` (default 35 baselines) |
| `SlamConfig::pgo`, `SlamSystem::set_vocabulary` | `LoopClosingMode::Enabled { vocabulary }`, with `tuning.loop_correction` |
| `SlamConfig::debug` | `SlamSystem::set_debug` |
| `SlamSystem::set_imu_extrinsics` | IMU calibration on the `SensorRig` |
| `SlamSystem::process_frame` (prepared features) | `SlamSystem::process(SensorFrame)` |
| `LoopClosingConfig::require_imu_initialized` | derived from the rig |
| `kornia_slam::initialization::inertial` and its re-exports (`ImuInitializer`, `ImuInitConfig`, `ImuInitResult`, `AlignedTrackingState`) | `kornia_slam::inertial`, which also owns the runtime IMU state |

`SlamSystem` no longer accepts features computed outside it; such callers pass
images, or compose the tracking and mapping building blocks directly, which
remain public. `TwoViewInitConfig::default()` now carries the triangulation
gates the runtime used (`max_midpoint_gap` 0.25, `max_reprojection_error` 3.0).

**Loop correction replays identically.** Loop-verification RANSAC now uses a
fixed seed, so runs with loop correction and synchronous local mapping
reproduce exactly.

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
