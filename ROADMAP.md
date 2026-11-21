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

### 2026-11-21
- **#180, card K (`experiments/pr180-K-card.md`).** The R_03 failure under `min_cold_start_points: 20` (G32) is fixed behind a knob.
  - **Cause:** while the device is still, keyframe pairs a few millimetres apart grew map points. Only wrong matches can pass the 1° parallax gate from such a short baseline, so the points landed at 2-25 cm instead of 0.8 m and shrank the trajectory to about 1/5 scale.
  - **Fix:** `mapping.growth_min_baseline_ratio` skips a pair whose baseline is below that fraction of the keyframe's median depth (ORB-SLAM3 `CreateNewMapPoints`, mono rule). At 0.0175 (tan 1°): near points 2394 to 11-13, step ratio 1.01, VIBA2 accepted 3/3.
  - **Result with `min_cold_start_points: 20`, 3 runs each:** R_03 784-798/800 at ATE 0.194-0.202 m; R_11 798-800/800 at 0.136-0.170 m; R_01 800/800 at 0.108-0.120 m.
  - **Shipped in `pr180-frontend-1ms`** (owner, 2026-11-21), with both knob defaults in code still off.
- **#180, card F resumed.** The owner decided D1-D6. The raw `.vrs` files for R_01, R_03 and R_11 are downloaded and are being extracted to the ASL layout.

### 2026-11-19
- **#180, first quick evaluation** (2026-11-18, LaMAria 800-frame windows, `--no-pace`, 3 runs per config). This is not the validation protocol; it is a few-minute check on three sequences.
  - EuRoC with the knobs off matches the 79c04bb base.
  - `pr180-frontend-1ms` beats `lamaria-rig-imu` on R_01: ATE 0.10 m against 0.59 m.
  - The full R_01 sequence tracks 100 %, but with ATE 1.18 m and scale 1.136 (drift).
- **#180, card H** (`experiments/pr180-H-card.md`). A bisect of `pr180-full` put its scale collapse on `tracking.pose_inertial: OrbSlam3`, the per-frame solve from card A6 and PR #195. The solve gates the visual matches out: solve inliers / visual inliers are 0.09-0.12 on rejected frames. `pr180-full` now ships it off (f55e997), and is then 3/3 at 100 % tracked on R_01, 0.110-0.122 m. H2 (the `LastKeyframe` variant) is refuted.
- **#180, card I** (`experiments/pr180-I-card.md`), knobs `tracking_loss.reseed.young_map_grace_sec` and `min_cold_start_points`, both off.
  - R_11 loses 132 of 800 frames at start-up: the frame-0 map is reset at frame 3, then the cold start starves below 50 stereo points.
  - `min_cold_start_points: 20` gives R_11 800/800 ×3, but breaks R_03.
  - The grace knob does not help: a held young map does not re-track.
  - The frame-0 map dies at frames 2-4, from the auto-exposure ramp on R_01 and R_03, and from the view moving off the first structure on R_11.
- **#180, card J** (`experiments/pr180-J-card.md`), knobs `inertial.viba_window` and `inertial.viba_velocity_gate`, both off.
  - **Why R_03 breaks:** with the map seeded one frame earlier, VIBA0 is 3.4° off, as measured by VIBA1, against 0.47° in the base. VIBA2 then fits a systematic tilt of about 14° (|ba| 2.3-2.7 m/s²) and is rejected in 10/10 runs, and tracking is lost.
  - **The two fixes:**
    - J4 keeps the keyframes inserted while lost out of the VIBA windows;
    - J1 gates outlier velocity edges per edge;
    - both are refuted as fixes (the error is systematic, not a few edges), and are harmless on R_01, R_11 and the R_03 base.
  - **J3 cancelled before code:** under `min_cold_start_points: 20`, the R_03 visual trajectory is at about 1/5 scale for frames 25-250 (step length ours/GT 0.13-0.23; base 1.00). Rotation matches ground truth to about 0.1°. Stereo VIBA fixes scale at 1, so its velocity residuals are systematic, which explains VIBA0's error and the VIBA2 reject.
  - **Ruled out:** LK (frames 3-45 are tracked by ORB map projection; KLT starts at frame 46), the ORB stereo seed (both configs: 63 points, median 0.80 m), and tracks surviving the reset.
  - **Next:** measure the suspect, pair-growth triangulation across a near-zero baseline while the device is still (frames 0-32). The only guard is `|t| > 1e-8` (`mapping/growth.rs:233`); ORB-SLAM3's `CreateNewMapPoints` skips pairs below the stereo baseline. Then J2 (`FullInertialBA`, kornia-slam#51).
- **#180, still unvalidated.** No preset changed except `pr180-full` (`pose_inertial: Off`). The validation workflow in `experiments/pr180-validation-handoff.md` has not run. The status and how to take over are in `experiments/pr180-status-2026-11-19.md`.

### 2026-11-18
- **#180, theory review.** Four read-only reviews (estimator, offline BA and calibration, geometry and recovery, frontend): 52 findings (5 high, 21 medium, 26 low), all in knob-gated code. The always-on review fixes were found equivalent to what they replaced. Findings and the test for each: `experiments/pr180-theory-review.md`.
- **#180, fixes per lane** (one commit per finding or group, each with a test that fails on the old code; knobs off, not validated):
  - estimator and system:
    - E1/E7/G3: velocity and bias come from the origin keyframe, and velocity is rotated on a pose jump;
    - E2: pixel sigma from the camera-centre covariance;
    - E4/E5: pose-inertial guards, and a solve with too few inliers is rejected and counted;
    - E6: 2-dof gates for every pixel row;
    - E9/E12: stereo evidence counts twice, and the prior gradient is its cost derivative;
    - G2/G7: a lost relocalisation into an older segment lifts the own window, and the coast speed spans the gap;
    - G12: inertial knobs are refused without an IMU;
    - T3/T5/T7: track positions refresh on a keyframe, the gyro guess spans one frame, and the evidence lock wait is timed;
    - T9: an anchor keeps the last keyframe's frames.
  - mapping, VI-BA and calibration:
    - E3: the newest marginal is the undamped system at the solution;
    - D1/D4/D9/D10: the boundary prior at the calibrated extrinsic, chained older priors, a prior that keeps its constant, and its gradient through J_r^-1;
    - E8: a window as large as the map fixes its oldest keyframe;
    - E10/D3: offline edges integrate as the run did, and the clock state is not lag-biased;
    - D5/D6/D14/D8/D2/D12/E13: stereo depth rows are off under a camera state, the clock test motion, the yaw gauge row, the 3 px stage after early convergence, an observable offline walk, a `td_total_ms` report, and the large-window lambda doc.
  - frontend:
    - T1: the LK template is recut when ORB moved the track;
    - T2/T6/T8/T11: refill tracks reach a map point, the WindowedU8 tracks are capped at the refill budget, occupancy is at the tracks' start pixels, and no border corner is spawned that the filter would drop;
    - T4: the analytic flow pose rejects outliers;
    - T9: landmarks are adopted without waiting for a mapping result;
    - T10: stronger flow tests.
  - geometry, loop closing and sensors:
    - G1: a lost relocalisation joins by a loop edge, not an odometry edge;
    - G4/G5: gPnP prefix search and a true pixel residual;
    - G6/G8: loop-edge gravity projection and recentred solve, robust graphs only;
    - G9/G10: rig extrinsic checks and a reflection check;
    - G11: monotone analytic-refine steps behind `tracking.refine_monotone` (off by default; on in both `configs/pr180-*.ron` presets);
    - E14: Midpoint extrapolates past the last IMU sample;
    - E11: the covariance propagation test.
- **#180, not fixed:**
  - G13: unreachable (`node_spacing: Some` only, and `correction()` has no non-test caller);
  - D13: deferred design item (the window-end edge needs samples that do not exist yet when it is made);
  - D7, D11: fixture work, the code was not shown wrong (draft tests are not in the branch);
  - the rest of E11: tests for `solve_pose_inertial`, `marginal_information`, `pixel_sigma`, the G29 counter and `newest_marginal_covariance`, which sit in lane S and M files.
- **#180, still unvalidated:** no eval or benchmark has run. The robot test and the validation workflow come next.
- **2026-11-17 lines corrected:** the estimator-round line now notes that the stereo pose-inertial gates were 3-dof until E6, and the recovery line notes the G1/G2 detach.
- **Card text corrected:** card A (outliers stay in keyframe observations, as in ORB-SLAM3), card D P15 (`|td| ≤ 2 ms`, `td_total_ms`), card E P15 (post-ORB recuts by design), card G (the boundary prior is formed at the calibrated values since D1).

### 2026-11-17
- **#180, rebased onto main.** No merge commits. kornia-rs is pinned to main `4ada5a7`. Reviewer comments applied:
  - `rotation_defect`, `HuberKernel`, `CauchyKernel` and `Mat6F64::cholesky` from kornia-rs;
  - one `camera_centre` helper;
  - the `keyframes_while_lost` source-of-truth fix and validation.
- **#180, estimator round** (cards A-D; knobs off, not validated):
  - ORB-SLAM3 estimator port: window 10/25, bias at the window boundary, Huber placement and pose-inertial gates (the stereo gates were 3-dof until 2026-11-18, E6), IMU sample interpolation, per-frame pose-inertial solve (idea from #195), keyframe and local-map rules;
  - IMU covariance trust;
  - G29 snap-back counter;
  - recovery: G19 global rig PnP while lost, and a G28 re-seed with its own window (a lost relocalisation into an older segment detaches and joins by a loop edge since 2026-11-18, G1/G2);
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

Next, start-up and scale (2026-11-19, from cards I and J; in this order):
- [x] **Baseline and parallax gate on new map points** (card K, 2026-11-21: built, knob `mapping.growth_min_baseline_ratio`; at 0.0175 (tan 1°) with `min_cold_start_points: 20`, R_03 784-798/800, R_11 798-800/800, R_01 800/800; shipped in `pr180-frontend-1ms`).
- [ ] **Raw fisheye images instead of the pinhole crop (card F).** Basalt, OKVIS2, ORB-SLAM3 (`KannalaBrandt8`) and Aria's own SLAM run on the full fisheye view; the wider stereo overlap removes the R_11 start-up starvation (G31) at its source.
- [ ] **Still-device handling.** Detect no motion; skip keyframes or apply a zero-velocity update, and initialise gravity and bias from the still period, as OpenVINS and Basalt do.
- [ ] **IMU carries tracking through visual gaps.** Keep the young map and propagate with the IMU when a frame has too few matches, instead of resetting it at its first lost frame (Basalt, OKVIS2, VINS); ORB-SLAM3 keeps old maps in its Atlas and merges them later.

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
