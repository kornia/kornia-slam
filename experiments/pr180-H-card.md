# Card H: the per-frame pose-inertial solve drops good visual matches

Status: card only. Nothing is built or run under it yet; the evidence below comes from the 2026-11-18 bisect.

## Experiment: H, pose-inertial visual collapse

- **Gap tested:** scale and tracking loss of the `pr180-full` estimator bundle on LaMAria (card A, item A6).
- **Base:** kornia-slam `feat/global-pose-graph` @ 6a6a64b, release `kornia-slam-app --no-default-features`.
- **Where A6 comes from:**
  - ORB-SLAM3's `TrackLocalMap` (`Tracking.cc:2970-2992`), with `PoseInertialOptimizationLastFrame` and `...LastKeyFrame` (`Optimizer.cc:4491-5289`);
  - the idea of PR #195 (Astik), whose mono EuRoC A/B improved scale on MH_02-04.

  It entered the cards A-D ORB-SLAM3 estimator port as knob `tracking.pose_inertial`, default `Off`. `configs/pr180-full.ron` turned it on as part of bundle A0.

### Evidence (R_01_easy, 800 frames, unrectified IMU, `pr180-frontend-1ms` base)

Run dirs are under the 2026-11-18 session scratchpad, in `quick/bisect/out/`. ATE is SE3 against pseudo-dense ground truth.

| Config | Runs | Tracked | ATE (m) | Scale | `coast_inertial` |
|---|---|---|---|---|---|
| base | 3 | 100 % | 0.097-0.112 | 1.02-1.04 | 0 |
| base + `pose_inertial: OrbSlam3` | 2 | 90 % | 0.169-0.181 | 1.00-1.05 | 79-80 |
| base + it + `local_map: OrbSlam3` | 2 | 76-89 % | 0.176-1.253 | 1.03-1.33 | up to 181 |
| full | 4 | 68-95 % | 0.25-2.43 | 0.08-0.82 | up to 247 |
| full without it | 3 | 100 % | 0.110-0.122 | 0.99-1.01 | 0 |

### Mechanism, from the debug log of base + `OrbSlam3` (`dbg-a4`)

- **Where frames are lost:**
  - 97 `LowPnpInliers` rejects; the base has none;
  - every loss enters through the pose-inertial inlier count.
- **Typical collapse, frames 140-150:**
  - the solve's inliers go 280, 14, 128, 141, 90, 49, 31, 71, 40, 26, 10;
  - `imu_chi2` falls from 51.6 to 0.004 over the same frames;
  - at frame 150, the visual-only PnP of the same frame (`[predgap] inl=741 m=1106`) keeps 741 of 1106 matches, while the solve keeps 10.
- **Reading:** the solve sides with the inertial prediction and gates the image out. The prior is positive definite every time (`prior_pd=1`, `pose_inertial_prior_dropped=0`), so the 15x15 frame-to-frame prior is always active.
- **IMU chi2 over 745 solves:** p50 0.2, p90 3.2, p99 50.7, max 524; above 100 on 4 frames.

### Theory: three candidate causes. The test design tells them apart.

- **H1, IMU overconfident.**
  - The IMU edge is weighted by `information(preintegrated, 1.0)`, with no inflation (`tracking/pose_estimation/pose_inertial.rs:108`).
  - R_01 has no Aria factory calibration here, so its IMU is not rectified. A biased IMU with an exact information wins over vision.
  - ORB-SLAM3 runs this edge only with rectified IMU data.
- **H2, the frame-to-frame prior runs away.**
  - Under `LastFrame`, each frame's marginal (`pose_inertial.rs:464-475`) is the next frame's prior, with Huber δ 5 (`:32`).
  - After one frame with few inliers, the prior is mostly IMU. That pulls the next pose toward the prediction, fewer matches pass the 5.991 gate (`:25`), and the next prior is IMU-heavier again. The inliers fall monotonically until the rescue (`:35`) is too late.
  - Card A predicted the prior is dropped on ≤ 1 % of frames; it is dropped on 0 %, so nothing breaks the chain.
- **H3, the gate is too tight for these rows.**
  - The rows are KLT or undistorted rig pixels at information `1.2^-2·octave` (`system/estimator.rs:547-565`), gated at chi2 5.991, which is 2.45 px at octave 0.
  - Any pose bias from `t_bc` or the time offset larger than about 2 px pushes good matches out, unlike the RANSAC PnP, which re-picks the inlier set.

### Knobs (all existing; no code needed for H1-H3 discrimination)

- `tracking.pose_inertial: LastKeyframe`: no chained prior. Tests H2.
- A sequence with factory calibration (R_11_5cp; `factory/R_11_5cp.factory.json`): rectified IMU. Tests H1.
- `--debug` `[pose_inertial]` and `[predgap]` lines: the solve's inliers against the visual PnP's inliers on the same frame. Mechanism counter for all three.

Code knobs are added only after a hypothesis survives:
- for H1, an IMU information scale, default 1.0;
- for H2, a prior reset when inliers fall below a share of the visual PnP's inliers;
- for H3, a gate scale, default 1.0.

### Baselines

- `bisect/out/a0-base{,-r2,-r3}`;
- `bisect/out/a4-pose_inertial{,-r2}`;
- `bisect/out/l-no_pose_inertial{,-r2,-r3}`.

All runs used binary `slam-head` (6a6a64b), a laptop with 12 threads, and 3 runs at a time.

### Predictions

Each prediction is checked over 3 runs per config on the 800-frame windows.

- **P1 (mechanism, all hypotheses):** on frames where the solve rejects, solve inliers / visual PnP inliers is ≤ 0.1. Today it is 10/741 at frame 150. Measured by a script over the `[pose_inertial]` and `[predgap]` lines.
- **P2 (H2):** `LastKeyframe` on R_01 brings `LowPnpInliers` from 97 to ≤ 20, tracking to ≥ 98 %, and ATE to 0.10-0.13 m.
- **P3 (H1):** on R_11_5cp with factory calibration, base + `OrbSlam3` tracks within 2 % of the base (677/800 in one run today), and `coast_inertial` is ≤ 10. If R_11 loses as R_01 does, H1 is not the cause.
- **P4 (H3):** on frames that keep ≥ 100 solve inliers, the median reprojection chi2 of the visual PnP inliers is ≤ 2. If it is ≥ 4, the pose bias is large enough that the 5.991 gate cuts good rows.

### Kill rules

- **K1:** P2 holds, but scale still drifts with `local_map: OrbSlam3` added. Then the runaway prior is not the whole story; next, test local_map's own window (card A, A8).
- **K2:** none of P2-P4 holds. Then the collapse has another cause, for example the anchor velocity (E1/E7 fixed only under these knobs). Next, log the anchor velocity against the pseudo-dense ground truth.
- **K3:** P1 fails (the solve keeps most visual inliers on lost frames). Then the loss is not this gating; look at the `LowPnpInliers` threshold path instead.

### Bit-identity and noise

- **Identity:** with `pose_inertial: Off`, nothing changes. The app runs are not deterministic (EuRoC V2_02 ATE 0.14-0.54 m over repeats of one binary), so knobs-off is checked by tests, not by `cmp`.
- **Noise:** 3 runs per config. The base band on R_01 is ±0.01 m ATE.

### Result log

- **2026-11-18:** `configs/pr180-full.ron` sets `pose_inertial: Off` until this card is scored.
