# Roadmap

This roadmap will keep evolving. It sets a direction, not a commitment.

## Deployment realism

Themes derived from ongoing studies of industrial and egocentric deployments. These guide our backlog and evaluation, not parity with MicroAGI microSLAM or any external system, and we do not claim SOTA.

- Dynamic‑object‑aware tracking — https://github.com/kornia/kornia-slam/issues/139
- Long‑sequence consistency — https://github.com/kornia/kornia-slam/issues/140
- Dense/TSDF consistency from fixed poses — https://github.com/kornia/kornia-slam/issues/141
- Monocular scale (already covered) — https://github.com/kornia/kornia-slam/issues/71
- Place recognition / loop features — https://github.com/kornia/kornia-slam/issues/48
- Appearance‑robust relocalization — https://github.com/kornia/kornia-slam/issues/68
- LaMAria / egocentric evaluation — https://github.com/kornia/kornia-slam/issues/78
- Multi‑camera + IMU — https://github.com/kornia/kornia-slam/issues/138

## Progress log

Newest first, one entry per working day. "main" means a merged PR; "#180" means work on the
`feat/global-pose-graph` branch (PR #180), which is not merged. Unless marked validated, branch work
sits behind config knobs that default to the old behaviour, and nothing in it has been benchmarked yet.

### 2026-11-17
- **#180, rebased onto main.** No merge commits. kornia-rs is pinned to main `4ada5a7`. Reviewer comments applied:
  - `rotation_defect`, `HuberKernel`, `CauchyKernel` and `Mat6F64::cholesky` from kornia-rs;
  - one `camera_centre` helper;
  - the `keyframes_while_lost` source-of-truth fix and validation.
- **#180, estimator round** (cards A-D; knobs off, not validated):
  - ORB-SLAM3 estimator port: window 10/25, bias at the window boundary, Huber placement, IMU sample interpolation, per-frame pose-inertial solve (idea from #195), keyframe and local-map rules;
  - IMU covariance trust;
  - G29 snap-back counter;
  - recovery: G19 global rig PnP while lost, and a G28 re-seed with its own window;
  - generic rig calibration and rays;
  - analytic pose refine, and the PGO moved to `upstream/`;
  - offline global VI-BA with calibration states, plus a `--session-out` dump.
- **#180, 1 ms frontend round** (cards E, G; knobs off, not validated):
  - KLT on every rig camera, the windowed u8 LK and `FlowPool` threads;
  - the forward-backward check, Shi-Tomasi refill, gyro guess, anchor guard;
  - analytic flow pose, keyframe-only bookkeeping;
  - online calibration with `imu_t_bc` and the clock offset;
  - `wide` SIMD kernels with OpenCV benches;
  - robot presets `configs/pr180-frontend-1ms.ron` and `configs/pr180-full.ron`.
- **Deferred:** raw Aria fisheye (card F), and keyframe ORB on the mapping thread (`keyframe_orb: Mapping`).
- **Reverted:** Step 8 (SVD `orthonormalized`), because it changed the PGO fingerprint.
- **Generic code** goes in `crates/kornia-slam/src/upstream/`, marked for kornia-rs later.
- **#180, CI fixes:**
  - clippy (rust 1.100), rustfmt and rustdoc;
  - the offline VI-BA test fixture now has observable scale (constant acceleration left a scale and tilt null space);
  - the full local gate is green.

### 2026-11-16
- **main:** #197, inertial factors use kornia-3d `RobustKernel::rho`/`weight`; local `robust.rs` removed.
- **#180:** review received from cjpurackal.

### 2026-11-15
- **main:** #194, inertial factors grouped; `se3` renamed to `relative_pose`.
- **#180:**
  - re-seed with kept inertial state on rigid poses, the gyro attitude and a tilt check;
  - a pose-acceptance gate against the inertial prediction;
  - a VI-BA iteration-cap knob;
  - the LaMAria sprint handoff (`experiments/lamaria-handoff.md`).

### 2026-11-14
- **main:**
  - #185, #191, #192: factors, preintegration and VI-BA on kornia-algebra fixed-size types;
  - #189: build against kornia-rs main;
  - #190: reuse kornia-rs helpers.
- **#180:** every pose kept rigid at the source; keyframes while lost as fixed anchors; non-rigid pose counters.

### 2026-11-13
- **main:**
  - #175: IMU, wheel and stationary factors moved into kornia-factors;
  - #181: loop closures solved on a worker thread;
  - #182: seed after a give-up;
  - #183: LK pyramid reuse;
  - #188: clippy 1.100.
- **#180:** run counters and a `[summary]` line; wheel odometry in the local BA; a keyframe bound; app flags `--trajectory-coasted` and `--no-pace`.

### 2026-11-12
- **#180:**
  - KLT-first tracking between keyframes (mono);
  - LK starts at the predicted pose;
  - fixed-point u8 LK kernel;
  - analytic pose-only LM in `solve_pnp`;
  - frontend replay benchmark;
  - a secondary-camera keypoint budget.

### 2026-11-11
- **#180:** a marginal prior for the VI-BA boundary, carried between windows; an optional state envelope on BA write-back; hold the inertial state of keyframes inserted while lost.

### 2026-11-10
- **#180:**
  - multi-camera rigs: rig declaration, per-camera features and overlap points, per-camera tracking, stereo-inertial initialization;
  - the LaMAria pinhole source;
  - loop-closure fixes (Drift information, inline solve by default);
  - robust kernel in the visual-only BA;
  - octave-scaled visual information.

### 2026-11-09
- **#180:** landmark merge through observation records; loop fusion planned in the worker and applied at write-back; seed after a give-up at the last tracked pose.

### 2026-11-08
- **#180:** one global pose graph for loop closing:
  - sparse PGO with information-weighted Cauchy edges;
  - drift-based loop gate;
  - persistent pose graph owned by `LoopCloser`;
  - off-thread solve with snapshot write-back.

### 2026-11-07
- **#180:** `compact_landmarks` frees retired landmark slots.

### Earlier (main)
- **2026-10-04 to 10-29:**
  - wheel-odometry VI-BA factor and stationary init (#158);
  - tracking-loss recovery and relocalisation (#153);
  - `process_imu`/`process_image` split (#165);
  - wheel kinematics (#174);
  - keyframe eviction at a map bound (#169);
  - configurable pipelines (#149, #150);
  - calibration through `SensorRig` (#147).
- **2026-09-23 to 09-27:**
  - map and system refactors (#105-#125);
  - kornia-rs 0.2.0 on crates.io (#129);
  - Copper integration guide (#130);
  - v0.1.0 release (#136);
  - IROS 2026 workshop paper (#137).

## Now — egocentric accuracy on LaMAria

We are working toward top results on the [LaMAria](https://github.com/cvg/lamaria) egocentric benchmark ([#78](https://github.com/kornia/kornia-slam/issues/78)), using the Aria rig and IMU and no GPS. The work is sequenced by where the benchmark score is lost: each change states its theory and predictions first, ships behind a config knob that defaults to the old behaviour, and is compared window by window against ORB-SLAM3 in the same harness.

Losses and recovery:
- [x] Keep every pose rigid at the source; rotation scale crept in through repeated pose corrections
- [ ] Lost-mode keyframes as fixed anchors (built, knob `keyframes_while_lost`, not yet validated; no gain on its own)
- [ ] Keep gravity, bias and velocity through a tracking loss, so a re-seeded map shares the old world frame (built, knob `tracking_loss.reseed.keep_inertial`, not yet validated; no gain on its own)
- [ ] Reject tracked poses that disagree with the gyro prediction (built, knob `tracking_loss.pose_gate`, not yet validated; no gain on the proxy)
- [ ] Global PnP relocalisation while lost; a re-seeded map with its own local window (built, knobs `tracking_loss.relocalization.while_lost` and `.solver`, `tracking_loss.reseed.own_window_keyframes` and `.relocalize_when_lost`, not yet validated)

Scale and drift while tracking:
- [ ] Longer visual-inertial BA window, and keyframes no longer dragged away from the map by BA (built, knobs `mapping.local_ba.large_window`, `mapping.local_ba.imu_huber`, `tracking.snapback_fix`, not yet validated; `mapping.local_ba.active_keyframes` already existed)
- [ ] Per-frame velocity state and a pose-inertial solve per frame (built, knob `tracking.pose_inertial`, not yet validated)
- [ ] ORB-SLAM3's keyframe insertion, local map and growth rules (built, knobs `keyframes.rules`, `tracking.local_map`, `mapping.growth_neighbours`, not yet validated)
- [ ] IMU bias and noise model, compared across ORB-SLAM3, Basalt and other references (built in part, knob `inertial.sample_interpolation`, not yet validated)
- [ ] IMU covariance trust: trust the inertial prediction by its propagated covariance (built, knob `tracking.imu_trust`, not yet validated)
- [ ] Offline global visual-inertial BA (built, run file `offline_ba` and the `offline-ba` subcommand over a `--session-out` dump, not yet validated)
- [ ] Calibration states in the offline BA: camera rotation and focal drift, gyro intrinsics, clock offset (built, knobs `offline_ba.calibration.*`, not yet validated)

Field of view and speed:
- [ ] Raw Aria fisheye cameras instead of the pinhole crop: deferred to its own round (card F)
- [ ] ORB only on keyframes with KLT in between, on every rig camera (built, knob `klt_rig`, not yet validated; `klt_first` already existed)
- [ ] Optical flow on every rig camera, forward-backward check, refill, gyro guess and flow-only pose (built, knobs `klt_rig`, `forward_backward`, `track_refill`, `gyro_guess` and `flow_pose`, not yet validated)
- [ ] Window-only fixed-point LK at an 11x11 window (built, knobs `klt_kernel: WindowedU8` and `klt_window`, not yet validated)
- [ ] Flow workers across 2-3 cores (built, knob `flow_threads`, not yet validated)
- [ ] Keyframe ORB on the mapping thread (`keyframe_orb: Mapping`): deferred, its design is open (lane T step 12)

## Next — complete the SLAM stack

- [ ] GPU acceleration of the per-frame hot paths (feature extraction, matching, KLT
      tracking), starting with the KLT frontend from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs)
- [ ] Relocalization on tracking loss, robust to appearance change (day/night, lighting) —
      measure the degradation first ([#68](https://github.com/kornia/kornia-slam/issues/68))
- [ ] Metric scale for monocular maps: Sim3 loop closure, and AprilTag anchoring
      ([#71](https://github.com/kornia/kornia-slam/issues/71), [#72](https://github.com/kornia/kornia-slam/pull/72))
- [ ] Match ORB-SLAM3 on trajectory accuracy and tracking robustness
- [ ] Upstream general-purpose code (solvers, camera models, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs). Staged in `crates/kornia-slam/src/upstream/`, one module per target (built, not yet upstreamed):
  - [ ] `kornia_3d::pgo`: PGO manifolds and edges
  - [ ] `kornia_3d::pnp::refine`: analytic 2x6 pose refine
  - [ ] `kornia_3d::pnp::gpnp`: generalised PnP and AP3P on rig rays
  - [ ] `kornia_3d::pose`: `camera_centre` and the `Pose3d`/`SE3` conversions
  - [ ] `kornia_algebra`: f32/f64 casts

## Library structure

- [ ] Pluggable feature frontend for non-ORB and learned features, e.g. XFeat ([#49](https://github.com/kornia/kornia-slam/issues/49))
- [ ] Pluggable place recognition, including learned global descriptors such as DINOv3
      ([#48](https://github.com/kornia/kornia-slam/issues/48))
- [ ] Multi-camera rigs (2+ cameras), ported from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs);
      generic rig calibration file, a ray per rig pixel and a rig-aware AP3P (built, knobs `rig_calibration`
      and `tracking_loss.relocalization.solver: RigAp3p`, not yet validated)
- [ ] Explore [Atlas](https://github.com/kornia/kornia-slam/issues/59), a manifold-native estimation and factor-graph foundation, as
      the base for BA, VI-BA and pose-graph optimization
- [ ] Split `SlamSystem` into tracking, mapping, inertial, and loop-closing subsystems

## Tooling and integrations

- [ ] Pixi environment and tasks for reproducible cross-platform builds
      ([#77](https://github.com/kornia/kornia-slam/issues/77), [#79](https://github.com/kornia/kornia-slam/pull/79))
- [ ] ROS 2 integration via [ros2_rust](https://github.com/ros2-rust/ros2_rust)
      ([#75](https://github.com/kornia/kornia-slam/issues/75))

## Experimental — sensors, maps, agents

- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats ([#141](https://github.com/kornia/kornia-slam/issues/141))
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
