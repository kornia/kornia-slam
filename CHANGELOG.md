# Changelog

All notable changes to kornia-slam are recorded here, newest first. Each entry is
written for users of the `kornia-slam` and `kornia-sensors` crates and the
`kornia-slam` CLI — what changed and why it matters, not a raw commit dump.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Keep the link references at the bottom
     in sync so each version links to its diff. -->

## [Unreleased]

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

**Keyframe culling.** The keyframe count no longer has to grow without bound.
`mapping::culling::cull_redundant_keyframes` removes keyframes whose live
landmarks are, by at least 90 %, each seen by three other keyframes at most one
pyramid level coarser (ORB-SLAM3's local-mapping code; set
`scale_level_tolerance = 0` for its paper's "same or finer scale"), never the
first keyframe nor the newest few. Removal is a canonical map operation, `Map::remove_keyframe`:
every landmark record, reference and derived geometry is repaired, and the IMU
edges on each side are merged into one, re-preintegrated from their raw samples
(ORB-SLAM3's `MergePrevious`); a keyframe at an end of the IMU chain is refused.
Wheel-odometry factors, which the caller holds, are composed across a removed
keyframe (`WheelOdomFactor::compose`, with first-order covariance
propagation), or the keyframe is kept. Opt-in: nothing calls it by default.
Landmarks are not bounded by it: a retired landmark keeps its slot in
`Map::map_points` (ids are never compacted or reused) and `remove_keyframe`
walks every slot, so a map that only ever culls still grows by one slot per
landmark created, and each removal costs time in that count. A long-running
caller still needs a whole-map reset (`clear_active`) on some bound, such as
`map_points().len()`. The `max_merged_imu_span_sec` default (3 s) is
ORB-SLAM3's limit for an initialized IMU; cull before initialization only with
0.5 s. In stereo keyframes the redundancy count covers every live landmark
unless `stereo_close_depth_m` is set: set it to the frontend's stereo close
depth for ORB-SLAM3's rule, which counts only close stereo points so that far
and mono-only points cannot cull the keyframe holding the only close ones.
`KeyframeRemoval`, what `remove_keyframe` reports, is `#[non_exhaustive]`.
**Behaviour change for culling callers:** keyframe slots were stable before
(nothing removed a keyframe); `remove_keyframe` shifts every slot after the
removed one down, so a caller that enables culling must not hold slot
positions across it (keyframe ids keep their meaning). Otherwise additive.

**Parallax-based keyframe insertion.** `KeyframePolicy::min_parallax_px`
inserts a keyframe once the median rotation-compensated parallax since the last
one reaches a threshold (`tracking::median_parallax_px` measures it;
`tracking::median_compensated_parallax_px` is the same median over pixel
pairs a caller already has), so with `max_frames_between = usize::MAX` a
parked camera stops adding keyframes. The reference-ratio and weak-tracking
rules still apply. Off by default, so insertion is unchanged; `SlamSystem`
measures parallax only when it is set. In a system file it is
`keyframes.min_parallax_px` (`None` by default, checked finite and positive
when set). `KeyframePolicy::max_interval_sec` forces a keyframe once that long
has passed since the last one (`keyframes.max_interval_sec`, `None` by
default). An inertial system with `max_frames_between = usize::MAX` needs it,
and is refused without it (`ConfigError::UnboundedInertialKeyframes`): IMU
samples are buffered until the next keyframe, so a parked rig with no bound
would buffer and rescan the whole stop and bridge it with one IMU edge.
ORB-SLAM3 forces a keyframe every 0.5 s in its inertial modes.
**Breaking:** `KeyframePolicy` gains two public fields, so a struct literal
needs `..KeyframePolicy::default()`. `should_insert_with` takes the new
`KeyframeEvidence`; `should_insert` is unchanged.

**Local BA no longer copies the whole map.** Each local bundle adjustment
captured every keyframe, landmark and IMU factor under the map lock, so its
cost, and the stall it caused tracking, grew with the map. It now captures only
the local window. `BaUpdate::keyframe_biases` reads the bias an update would
write on each captured keyframe, so a caller can refuse an implausible one
before `Map::apply_ba_update`. **Breaking:** `Map::ba_snapshot` takes a
`BaWindow`; use `mapping::bundle_adjustment::local_window` for the standard
selection.

**Wheel odometry in visual-inertial BA.** A ground robot can now constrain
local VI-BA with the planar base motion its wheels measure between keyframes.
`mapping::wheel_odom::WheelOdomFactor` carries the `(dx, dy, dθ)` delta and its
covariance, holds z, roll and pitch stiff, takes the base-to-camera extrinsic
at each end (a moving head makes it time-varying), and has an optional Huber
kernel. Its Jacobians are analytic. The covariance is over `(dx, dy, dθ)` in
base-i axes; `WheelOdomMeasurement::information` rotates it into the
residual's predicted base-j axes, so the chi² is the Mahalanobis distance of
the planar error whatever the turn between the keyframes. With no factors the
solve is unchanged, bit for bit. `visual_inertial_bundle_adjust` refuses a
factor with a non-finite delta or extrinsic or a covariance that is not
positive definite; `run_local_inertial_ba` drops such a factor
(`WheelOdomFactor::is_valid`) instead of losing the whole window update.
**Breaking:** `run_local_inertial_ba` takes a `wheel_factors` slice (indexed
by `Frame::idx`) and `visual_inertial_bundle_adjust` a `wheel_edges` slice
(indexed by keyframe slot); pass `&[]` to keep the old behaviour. `ViBaError`
gains three variants. The workspace version moves to 0.2.0.

**Stationary edges for parked robots.** Two keyframes of one stop can now be
linked without IMU preintegration. `mapping::stationary::StationaryFactor`
(with `StationaryGyro`) constrains zero relative pose, zero velocity at both
ends, bias continuity with a random walk scaled by the span, and, when the
caller supplies the mean raw gyro rate of the still span, the gyro bias to that
mean (variance `σ_g²/T + σ_bg²·span/12`); there is deliberately no
accelerometer row, as tilt and accelerometer bias are not separable at rest.
Its pose Jacobians are analytic (the wheel factor's with identity extrinsics)
and checked numerically. It has no span limit:
`cull_redundant_keyframes` composes two stationary factors through a removed
keyframe (`StationaryFactor::compose`) however long the parked span, while
`max_merged_imu_span_sec` keeps capping IMU merges only, so a parked map no
longer fills with keyframes whose ~10 s IMU edges cannot merge. A keyframe at
an end of the stationary chain, including the boundary between IMU and
stationary edges, is kept; `KeyframeCullReport::stationary_edges_merged`
counts the merges. Like wheel factors, stationary factors are held by the
caller, keyed by `Frame::idx`; `bundle_adjustment::local_window_with_neighbors`
captures a window in which a stationary neighbour of the active keyframes is
fixed context, and `run_local_inertial_ba` drops an invalid factor
(`StationaryFactor::is_valid`) instead of losing the window update. With no
stationary factors every solve and every cull is unchanged.
**Breaking:** `visual_inertial_bundle_adjust` takes a `stationary_edges` slice
(after `wheel_edges`), `run_local_inertial_ba` a `stationary_factors` slice
and `cull_redundant_keyframes` an `Option<&mut Vec<StationaryFactor>>` (pass
`&[]` / `None` for the old behaviour); `ViBaError` gains
`StationaryFactorOutOfRange` and `InvalidStationaryFactor`. Additive:
`local_window_with_neighbors`, `KeyframeCullReport::stationary_edges_merged`
(the report is `#[non_exhaustive]`), and `kornia_sensors::imu::ImuCalib`
derives `PartialEq`. All of this rides on the already-breaking 0.2.0; none of
it belongs in a 0.1.x.

**The IMU buffer stays bounded while bootstrap fails.** `SlamSystem` dropped
buffered IMU samples only once a map existed or a monocular bootstrap
reference was stored, so a rig that could not bootstrap (a dark room: too few
stereo points, or no keypoints at all) buffered every sample for as long as
that lasted. Samples older than the current frame are now dropped on every
failed stereo bootstrap and whenever no monocular reference is held. While a
monocular reference is held the samples since it are still kept, as the first
edge needs them.

**Inertial initialization while stationary.** A stereo-inertial rig no
longer has to move before the IMU joins the estimate: the motion-based
initializer needs displacement to see gravity, but a stereo map is already
metric, so a still window is enough. `ImuInitializer::initialize_stationary`
takes the window's `ImuWindowStats` (mean and vector spread of the raw
samples) and returns an ordinary `ImuInitResult` for `apply_initialization`:
gravity from the mean specific force through the newest keyframe's attitude
and `T_BC`, gyro bias from the mean rate, zero velocities, and the
accelerometer bias corrected along gravity by `|f| - g` (the across-gravity
part is unobservable at rest and stays at the prior). Deciding that the window
was still is left to the caller. Monocular windows and a mean specific force
far from standard gravity are refused (`StationaryInitError`). Both new types
are `#[non_exhaustive]`. Additive; no existing signature changes.

**Solver diagnostics are opt-in.** The per-solve diagnostic lines of BA,
VI-BA and IMU initialization (including refused IMU
alignments and rejected IMU inits) print to stderr only when the
`KORNIA_SLAM_DEBUG` environment variable is set; it is read once, at the first
diagnostic, so set it before starting. They were unconditional, about a
hundred unbuffered writes a keyframe, which on an embedded host logging to
eMMC blocked the solve thread for seconds. `SlamSystem::set_debug` does not
control them (it never did). Faults still print unconditionally: a panicked
local-mapping worker, and the first singular IMU covariance VI-BA falls back
from (later ones only with the variable set).

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
