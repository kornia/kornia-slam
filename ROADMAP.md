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

The experiment log of PR #180 (cards, verdicts, per-day progress) is kept outside the repository. Its EuRoC state is summarised in `experiments/euroc-summary.md`.

## Now — EuRoC top-1, then LaMAria

**EuRoC top-1 comes first (owner, 2026-11-23; summary `experiments/euroc-summary.md`).** LaMAria work waits
behind it. Score: the causal real-time pose (paced, asynchronous, 2 cores), below Basalt on all 11 (M1), then below
ORB-SLAM3 (M2). SYNC runs decide each A/B. In the card's order:
- [ ] **G60:** trace the non-converging loop PGO on S0PEIH MH_04 start 60 (PGO iteration log) before any loop-path change.
- [ ] **2x-4 / 2x-5:** sliding-window VI odometry, Observe mode first; it
  replaces R12 (dropped from this list 2026-11-24, plan section 2).
- [ ] **Drought keyframes** (candidate only; theory and predictions first).
- [ ] **Real-time ladder** (card "Real-time fix ladder F1-F6"): F1 done; F3 kernel done, **2-core system run pending**;
  F2 widened to the flow step, not scored; F4 not promoted, K3 fired, knob off (follow-up: InterruptBA, F5a first, or a
  solve-age refusal bound; owner to choose); F5a built, not scored; F5b after diagnosis D5b; F6 (G61 rig overlap
  partners) in its own lane.
- [ ] **Rig mode**: K2 split (loops off), then rung R-a (`active_keyframes: 10` on
  MH_02, MH_03), then F6.
- [ ] **Card AJ**: code landed behind `tracking.pose_inertial`; arms D/DJ/D10/D10J not run.
- [ ] **Real-time gate on all 11** after F5b (3 runs, load ≤ 6), then the full-sequence LaMAria guard before any knob
  is promoted.

### LaMAria (waits behind EuRoC top-1)

**Experiments take at most 5 min (owner, 2026-11-23):** short unpaced windows or offline runs on saved sessions, at least 3 repeats. The gate below runs once per batch of stacked winners, not per card.

**Regression gate (owner, 2026-11-22).** No knob is promoted into a preset or a code default until it passes the full-sequence gate:
- **Runs:** all 11 validation sequences (R_11_5cp, 1_19, 1_20, 2_11, 2_12, 3_17, 3_18, 4_10, 4_11, 5_11, 5_12), full length, paced at sensor timestamps, on a quiet machine (load ≤ 4, no other SLAM job).
- **Samples:** 2 runs of the candidate against 2 runs of the current baseline.
- **Pass rule:** no sequence's official CP score or tracked frames falls below the baseline's range. Any drop is diagnosed before the knob is promoted.
- **R_11:** with 3 usable control points the CP Sim3 often fails (scored 0), so R_11 is gated on tracked frames and ATE (`ate.py`).
- **Guards:** the 800-frame per-card guards stay as a cheap first check. They do not replace the gate: card AC's Held seed passed them, yet R_11 fell from 55.5 to 26.9 at 370 s.
- **Cleanup card:** it applies the gate to the stacked winners together.

We are working toward top results on the [LaMAria](https://github.com/cvg/lamaria) egocentric benchmark ([#78](https://github.com/kornia/kornia-slam/issues/78)), using the Aria rig and IMU and no GPS. The work is sequenced by where the benchmark score is lost: each change states its theory and predictions first, ships behind a config knob that defaults to the old behaviour, and is compared window by window against ORB-SLAM3 in the same harness.

General purpose, not tuned per deployment (owner, 2026-11-21). A new rig or dataset should need its sensor model only: calibration, IMU noise and frame rate. Knobs are how a change lands and is measured; after each validation round a cleanup card makes the winners the code default and deletes the refuted or redundant knobs.
- [ ] **Card AI: global stage (owner, 2026-11-23: main LaMAria card; planning; waits behind EuRoC top-1).** Every top entry scores from a global solve (SLAM Dunk: global BA about 69, then 82.4 with calibration over time and gyro intrinsics). Two parts:
  - **(a)** a VIO that never resets: IMU coasting with honest covariance, then re-acquiring vision into the same state (Basalt, OKVIS2, OpenVINS). It replaces the give-up and re-seed path.
  - **(b)** a background global graph: IMU bridges and cross-segment matches join the segments, then a bounded global VI-BA runs. Corrections go through a map-to-odom transform. The same code runs as the end-of-sequence batch.
  - **Then the calibration ladder:** camera calibration over time, gyro intrinsics, accel, clock and thermal terms (the `offline_ba.calibration.*` knobs are built).
  - Stage 0: join the segments of saved sessions offline and score CP.
- [ ] **Card AG: drift budget and loop closure on the challenge sequences** (owner, 2026-11-22; stage 0 done 2026-11-23, oracles feed card AI; ORB-SLAM3 reference runs dropped). It splits each sequence's CP loss into breaks, events, smooth drift, coasts and scale, with oracle scores, and checks GT for revisits. Then ORB-SLAM3 reference runs on 1_19, 2_11 and 1_20 (2 each, after the gate), and the #180 loop closer observe-only on 2_11 and 1_20, with 1_19 as the control (on the owner's go). No apply arm: it needs loop and odometry sigmas derived from the sensor model first.
- [ ] **Card AH: the window bias absorbs vision errors** (planned and approved 2026-11-23; stage 0 running; 5-minute windows). In weak vision the local VI-BA moves the gyro bias about 40σ beyond its random walk, and heading steps 10-30° (1_19, 1_20, R_11). First test: card S's `mapping.local_ba.boundary_scales_bias_rows: false` on the event windows. References: ORB-SLAM3 bias-walk edges at full strength, Basalt's marginalisation prior, OpenVINS first-estimate Jacobians. Target: 1_19 heading step ≤ 3°, CP 30-40.
- [ ] **Paused 2026-11-23 (card AI first). Card AE: trusted inertial state for a Held re-seed** (planned and approved 2026-11-22): the median bias over vision-anchored keyframes plus a gyro attitude chain. It fixes the R_11 and 4_11 seed-bias deaths found by the H0 gate.
- [ ] **Paused 2026-11-23, folded into card AI (a). Keep one world through a tracking loss** (owner, 2026-11-22): carry the gyro attitude across a give-up, give a new segment the kept inertial state, refuse visual velocities the IMU contradicts, link or merge the segments (ORB-SLAM3 RECENTLY_LOST and Atlas merge; Basalt and OpenVINS never reset).
- [ ] **Paused 2026-11-23 (card AI first). Card AD: gyro rotation prediction before IMU init** (owner, 2026-11-22). Before VIBA0 the tracker predicts each frame with zero motion (`predict_frame_pose` needs `imu_initialized`; the RotatedRay guess needs a predicted velocity), so a start-up turn of 3-4.4° per frame leaves a 29-32 px PnP residual against the 25 px gate and resets 19-56-point maps at once (1_19 frames 0-13; G31 class). The previous-frame timestamp (`state.last_frame_timestamp_sec`) also stays stale across cold bootstraps. Plan approved 2026-11-22 (knob `tracking.prediction_before_init: Visual | Gyro | GyroHeld`, timestamp refresh unconditional); starts when card AC lands.
- [ ] **Diagnose the first-map resets on R_01, R_03, R_11** (owner, 2026-11-22, read-only): the first map is lost on the 2nd or 3rd frame after a cold bootstrap at under 0.8° per frame, with 2-9 projected matches; not rotation, so card AD does not cover it. Ledger row to add with the diagnosis.
- [ ] **Cleanup after the first validation run** (owner, 2026-11-22: after the loss card, then card F).
  - **Make the fixes the only path:** `dense_parallelism: Sequential`, `merge: PropagateToNewer` (with the stop at a re-seed), `frame_velocity: PropagatedFromKeyframe` (card U), and the coasted-pose output.
  - **Make the reference constants the code defaults:** 10 active keyframes, 7 iterations from λ 1e-4 (Basalt), `growth_min_baseline_ratio` tan 1° (ORB-SLAM3).
  - **Delete the refuted or redundant knobs:** `keyframe_seed` (card R), the full-weight boundary edge (card O), `SnapshotOnly`.
  - **Shrink `pr180-window10`** to dataset and sensor settings.
- [ ] **Derive the remaining hand-set thresholds from the sensor model and the estimate's uncertainty:** depth sigma from f·baseline and disparity, gates from covariance, coast caps from the propagated IMU covariance, `min_cold_start_points` from the pose covariance of the cold start.

Next, start-up and scale (2026-11-19, from cards I and J; in this order):
- [x] **Baseline and parallax gate on new map points** (card K, 2026-11-21: built, knob `mapping.growth_min_baseline_ratio`; at 0.0175 (tan 1°) with `min_cold_start_points: 20`, R_03 784-798/800, R_11 798-800/800, R_01 800/800; shipped in `pr180-frontend-1ms`).
- [ ] **Raw fisheye images instead of the pinhole crop (card F).** Basalt, OKVIS2, ORB-SLAM3 (`KannalaBrandt8`) and Aria's own SLAM run on the full fisheye view; the wider stereo overlap removes the R_11 start-up starvation (G31) at its source.
- [ ] **Still-device handling.** Detect no motion; skip keyframes or apply a zero-velocity update, and initialise gravity and bias from the still period, as OpenVINS and Basalt do. (Card AB: detector and keyframe gate built and scored behind `keyframes.stillness`; stationary edges, stage 2, after the cleanup card.)
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
- [ ] Camera-IMU clock offset from the device record, then estimated online, and ORB-SLAM3's IMU kernel layout (card AA running; the online estimator, `online_calibration.clock_offset`, is built and not yet validated)
- [ ] Window-local frame for the VI-BA solve, so its convergence does not depend on the distance from the world origin (card Y, W2; arithmetic only so far)
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
- [ ] Keyframe ORB on the mapping thread: deferred, its design is open (the unbuilt `keyframe_orb` knob was removed 2026-11-26)

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
