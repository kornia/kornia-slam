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
rather than over several seconds (without a settled IMU it still widens
gradually: a wide search around a constant-velocity guess accepts wrong
matches). On EuRoC V2_03 stereo+IMU the system no
longer resets (21 resets before, ATE 1.66 m → 0.88 m with synchronous
mapping); the other ten sequences are unchanged. `MapProjectionConfig::max_search_scale` is renamed
`lost_search_scale`, and `TrackingLossRecoveryPolicy` gains
`keyframe_interval_while_lost_sec`.

**Visual tracking relocalizes against the map.** When tracking fails without a
settled IMU (stereo or mono only, or before inertial initialization), each lost
frame is scored against the keyframe database, matched against the best
candidates, solved with PnP RANSAC and refined against the candidate's local
map; 50 inliers resume tracking where the map already was instead of starting
a new one. It uses the loop closer's vocabulary, so it runs when loop closing
is enabled. On EuRoC V2_03 stereo-only, resets drop from 22 to 18 (ATE
1.72 m → 1.27 m).

**Weak pose estimates no longer teleport the trajectory.** With a settled IMU,
a frame now needs 15 inliers (ORB-SLAM3's inertial threshold) instead of 10
before its visual pose replaces the inertial prediction
(`MapProjectionConfig::min_inliers_with_prediction`), and a relocalized pose
more than 0.5 m from the motion model is refused. On EuRoC stereo+IMU the
largest frame-to-frame jump against ground truth drops from 1.30 m to 0.04 m on
V2_01 and from 1.01 m to 0.51 m on V2_03; jumps above 30 cm on V2_03 go from 15
to 5.

**Monocular initialization no longer waits forever on a stale reference.** A
bootstrap reference that fails to match more than ten frames in a row is
replaced by the current frame. EuRoC MH_02 and MH_05 previously never
initialized in mono or mono+IMU (0% of frames tracked); they now track 95–99%
of frames, and V1_03 mono goes from 10% to 83%.

**TUM-VI sequences.** The app reads TUM-VI's `dataset-*_512_16` exports
(`source: TumVi((data: …))`), monocular on the raw fisheye with the IMU, and
evaluates against motion capture. 16-bit images are reduced to 8 bits, and the
Hilti reader gains the IMU when its images are not rotated.

**Mono+IMU keeps mapping while it coasts too.** Keyframes inserted at the
inertial prediction while lost now include monocular ones, which local mapping
triangulates against their neighbours. On EuRoC mono+IMU, ATE drops from
0.96 m to 0.33 m on V1_02, 1.51 m to 0.62 m on V1_03 and 1.72 m to 0.79 m on
V2_01, with fewer resets; jumps above 30 cm rise on V1_03 and V2_01, where the
coasted pose snaps back as vision returns.

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
gains five variants (`WheelFactorOutOfRange`, `InvalidWheelCovariance`,
`NonFiniteWheelFactor`, `WheelFactorSelfLoop`, `InvalidWheelHuber`): a factor
that links a keyframe to itself, or whose Huber threshold or out-of-plane
sigma is not finite and positive, is refused too. The workspace version moves
to 0.2.0.

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
the boundary between a stationary factor and an IMU edge is removable too: the
stationary factor is dropped and the IMU edge moves onto the still neighbour
(`Map::remove_keyframe_beside_still`, with `StillNeighbour` / `StillSide`),
keeping its interval and samples, so the boundaries every stop leaves no longer
pin keyframes until the map is dropped. Boundaries are judged after every other
candidate, each once per call, so a stop's chain is composed before an IMU edge
takes it over; only a bare chain end is kept.
`KeyframeCullReport::stationary_edges_merged` and
`stationary_boundaries_removed` count both. An IMU edge whose bias link spans
still time outside its samples (a tail edge after a stop, or one moved across
a boundary) records it in `ImuFactor::bias_walk_extra_s`; keyframe merges and
VI-BA repropagation re-preintegrate through `ImuFactor::repreintegrate`, which
adds that walk back (`PreintegratedImu::widen_bias_walk`), instead of
rebuilding a bias link from the samples alone that was orders of magnitude too
stiff. Like wheel factors, stationary factors are held by the
caller, keyed by `Frame::idx`; `bundle_adjustment::local_window_with_neighbors`
captures a window in which a stationary neighbour of the active keyframes is
fixed context, and `run_local_inertial_ba` drops an invalid factor
(`StationaryFactor::is_valid`) instead of losing the window update. With no
stationary factors every solve and every cull is unchanged.
**Breaking:** `visual_inertial_bundle_adjust` takes a `stationary_edges` slice
(after `wheel_edges`), `run_local_inertial_ba` a `stationary_factors` slice
and `cull_redundant_keyframes` an `Option<&mut Vec<StationaryFactor>>` (pass
`&[]` / `None` for the old behaviour); `ViBaError` gains
`StationaryFactorOutOfRange` and `InvalidStationaryFactor`; `map::ImuFactor`
gains the public field `bias_walk_extra_s` (set it to `0.0` for an ordinary
edge) and `KeyframeRemovalError` the variant `InvalidStillNeighbour`. Additive:
`local_window_with_neighbors`, `KeyframeCullReport::stationary_edges_merged`
(the report is `#[non_exhaustive]`), and `kornia_sensors::imu::ImuCalib`
derives `PartialEq`. All of this rides on the already-breaking 0.2.0; none of
it belongs in a 0.1.x.

**A parked window no longer turns a keyframe.** A landmark seen from one
place has no observable depth, and a parked window sees every landmark from
one place. VI-BA inverted each landmark's 3×3 block undamped, so the point
step could fling such a landmark millions of metres out along its ray (or
behind the camera), and the result was written back to the map. When a later
keyframe measured that landmark by stereo, its depth residual was ~1e8 sigma,
and the cheapest way the solver found to lower it was to turn a keyframe by
~90°; downstream this was a VIO pose refused as an 80° jump after hours
parked. Four guards, each with a regression test (the end-to-end one turns a
keyframe by 137° with them reverted):
- the landmark blocks are Marquardt-damped like the keyframe system. This
  bounds the point step (undamped, a parked chain flung landmarks ~1e7 times
  their depth) but does not make the depth observable: over many parked
  windows a depth-free landmark still drifts out along its ray, to tens or
  hundreds of times its depth. The depth gate below is what keeps a landmark
  that got far from turning a keyframe when stereo next measures it;
- a stereo depth row is left out of the solve when, at the start, the
  landmark's depth in that camera is outside `(0, 10·d + 1 m)` for measured
  depth `d` (`ViBaParams::depth_gate_factor`, default 10; NaN, zero and
  negative values are refused with `ViBaError::InvalidDepthGateFactor`,
  `f64::INFINITY` keeps only the behind-the-camera check). A free landmark
  whose every depth row lies beyond that bound is first moved along the ray
  of its best measured observation to the measured depth, and its rows then
  enter the solve: reprojection rows constrain only its bearing, so a gated
  landmark would otherwise stay wrong (and gated) in every later window,
  including the moving ones after the robot drives off. The move needs
  corroboration, so a one-off bad stereo match cannot pull a correct landmark
  onto itself: a second gated depth row that agrees with the moved position
  within three combined sigmas, or a starting depth more than 100 times the
  gate's bound (the ~1e7 m flings). Otherwise the rows are only gated. The
  second row is not required to come from a different viewpoint, so a
  mismatch that repeats in every frame of a parked window (repetitive
  texture, a reflection) corroborates itself and still moves the landmark,
  as it did before corroboration was required;
- a step may not move a landmark from in front of an observing camera to
  behind it: a free landmark whose own step does that keeps its old position
  for that step, and a step whose camera motion does it is refused (λ grows)
  unless every observation that flips is a Huber outlier, so one mismatched
  near-plane association cannot stall a moving window;
- `run_local_inertial_ba` does not write back a landmark the solve left behind
  any observer, or whose distance from its nearest observer grew more than
  tenfold and beyond 50 m; it keeps its old position. This catches a fling
  in one solve, not a slow drift (each window may grow a landmark almost
  tenfold). `cull_landmarks` also retires a landmark farther than 10 km from
  every keyframe observing it (it already retired one behind an observer).

The solve now reports what it did: `ViBaResult` gains `initial_cost`,
`depth_rows_gated`, `points_reseeded`, `cheirality_rejections` and
`cheirality_points_held`, and
the update
`run_local_inertial_ba` returns carries them, with the write-back guard's
count, in `BaUpdate::solve_stats` (`LocalBaStats`; `None` from the visual
solvers).

**The stereo depth row's pose Jacobian is right for a rotated camera.** Its
translation part was `(0, 0, 1/σ)`, which holds only when the camera's
rotation is the identity; under the solver's right perturbation it is row 2
of the rotation, `(r20, r21, r22)/σ`, as the reprojection rows and the
depth row's own point Jacobian already had. With the wrong column, every
rotated keyframe's Gauss-Newton step from a stereo depth row pushed its
translation along the wrong axis. A numeric-differentiation test pins it.
Expect small changes in VI-BA results on stereo-inertial data.

**Huber costs match their weights.** Every robustified VI-BA term (reprojection,
stereo depth, IMU navigation block, wheel, stationary) weighted its normal
equations by Huber's `w = min(1, √(k/s))` but scored the LM step with
`½·w·s`, which for an outlier is `½√(k s)`: not the objective `w` is the
derivative of, so steps were accepted and refused on the wrong numbers. The
cost is now Huber's `½ρ(s)` (`s` inside the threshold, `2√(k s) − k` past it),
whose gradient is the weighted one; tests check the numeric derivative of the
shared Huber objective, and of the IMU, wheel and stationary costs, against
their normal equations. Inlier costs are unchanged; outlier costs
are higher, so reported costs are not comparable with earlier builds.

**Stationary initialization refuses inputs that gave a NaN state.** A
non-finite or negative `max_gravity_norm_error` is refused
(`StationaryInitError::InvalidTolerance`), so is a zero specific force (an
infinite tolerance let it through to a division by zero) and a non-finite mean
gyro rate (`StationaryInitError::NonFiniteGyro`).

**Breaking (all within 0.2.0):** `ViBaParams` gains `depth_gate_factor`, so a
struct literal needs `..ViBaParams::default()`. `ViBaResult` is
`#[non_exhaustive]`. As the 0.2.0 notes promise, the new public types are
`#[non_exhaustive]` too, each with a constructor: `WheelOdomMeasurement`
(`new`, `with_out_of_plane_sigmas`), `WheelOdomFactor` (`new`),
`StationaryFactor` (`new`, `with_gyro`), `StationaryGyro` (`new`),
`KeyframeEvidence` (`new`, `with_median_parallax_px`, `Default`),
`KeyframeCullingConfig` (`Default`) and
`StillNeighbour` (`new`). Code outside the crate can no longer build them with
a struct literal (functional update `..Default::default()` included); it
starts from the constructor and sets the public fields
(`tests/non_exhaustive_api.rs` builds each one that way). `StillSide`, a
two-valued enum, stays exhaustive.

**`merge_imu_edges = false` keeps stationary boundaries.** Culling removed a
keyframe at the boundary between a stationary and an IMU edge, moving its IMU
edge onto the still neighbour, even with `merge_imu_edges` off; it now keeps
it (counted in `kept_for_edges`).

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
