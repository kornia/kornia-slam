# Card M: the local VI-BA window is too short to hold the metric scale

Status: written 2026-11-21, before any run. Only existing knobs are used, so there is no code change.

## Experiment: M, VI-BA window size and the boundary IMU edge

- **Gap tested:** scale drift on full sequences (new ledger row G35). This is the drift behind the Long and Medium challenges.
- **Base:** kornia-slam `feat/global-pose-graph` @ 67a41ef, preset `pr180-frontend-1ms`, which includes card K.
- **Binary:** `92acd29` plus the card K diagnosis print (md5 7d09a8c68f297830aac35ad1dc844ba7). The preset change after it is config only.

### Evidence (full R_01, 2898 frames, one run on the base, 2026-11-21)

- **Tracking is complete:** 2898/2898 tracked, no IMU-only frames, VIBA0-2 accepted.
- **Accuracy is not:** ATE SE3 1.227 m, ATE Sim3 0.899 m, Sim3 scale 1.116 (the estimate is about 11 % too small). Total step length ours/GT is 0.936.
- **The scale wanders.** Step ratio ours/GT per 250 poses: 1.158, 1.034, 1.045, 0.897, 0.991, 0.833, 0.900, 0.887, 0.896, 0.896, 0.925, 0.898. That is a spread of 0.33, with stretches of 1000+ frames 7-17 % short.
- **Local VI-BA, from its `[vi_ba]` lines:**
  - every one of the 594 windows has `active=3`, with 23-30 fixed keyframes;
  - a median 88 % of the window's points are pinned.
- **IMU edges are not the problem:** 90.1 % of the final interior IMU edges have Huber weight 1, and 6.8 % are below 0.5.
- **Map-point depth does not track the error:** grown-point and stereo median depths per bin show no pattern against the step ratio (their ratio runs 0.5-1.8). So card K's mechanism is not this one.

### Mechanism (file:line on 67a41ef)

- `LocalBaConfig::active_keyframes` defaults to 3 (`system/config/mod.rs:596`). ORB-SLAM3's `LocalInertialBA` optimises 10 (`Nd = 10`).
- The fixed past enters through pinned points and through the boundary IMU edge, which is scaled by `boundary_imu_info_scale: 1e-2` (`:597`).
- So each window has 3 free keyframes, about 0.45-0.75 s of motion, and its scale is set by the fixed keyframes' points. The IMU sees too little acceleration in 3 keyframes to move that scale, and the boundary edge that would carry the inertial scale from the past is weighted at 1/100.

### Theory

- **Why the scale drifts.**
  - In a stereo-inertial system, the scale comes from two places: the stereo baseline, through overlap points (here only about 20 per frame, card I), and the accelerometer.
  - With 3 active keyframes, the accelerometer's scale information per window is small. The window inherits the scale of the 88 % pinned points, so scale errors from growth triangulation (monocular-like, from keyframe pairs) are never corrected.
  - The scale therefore random-walks, which is what the per-bin ratios show.
- **Reference choices.**
  - ORB-SLAM3 optimises 10 keyframes, with the previous keyframe's IMU edge at full weight (`Optimizer::LocalInertialBA`).
  - Basalt keeps about 7 keyframes plus a marginalisation prior, so the past's inertial information is kept, not frozen.
  - The best available here without code is the longer window: `active_keyframes: 10`, as ORB-SLAM3, and `large_window: Enabled(active_keyframes: 25)` (card A2). Separately, the boundary edge goes to full weight (`boundary_imu_info_scale: 1.0`).
- **What the theory predicts.** A window of 10-25 keyframes spans 2-5 s of walking acceleration, which is enough for the accelerometer to observe scale to a few percent. The scale then stays near 1 everywhere, instead of wandering.

### Knobs (all existing; their defaults are today's behaviour)

- `mapping.local_ba.active_keyframes` (default 3)
- `mapping.local_ba.large_window` (default `Off`)
- `mapping.local_ba.boundary_imu_info_scale` (default 1e-2)

### Configs

| Name | `mapping.local_ba` |
|---|---|
| M0 | base preset |
| M1 | `active_keyframes: 10` |
| M2 | `active_keyframes: 10, large_window: Enabled(min_tracked_inliers: 100, active_keyframes: 25, max_iterations: 50, initial_lambda: 1e-2)` (the card A0 values) |
| M3 | `boundary_imu_info_scale: 1.0` |

### Predictions

All runs: 3 per config, full R_01 (2898 frames). M1 and M2 are also run on the 800-frame R_03 and R_11 windows for K3.

- **P1, mechanism counter:**
  - every `[vi_ba] WINDOW` line has `active=10` (M1) or up to 25 (M2);
  - the median pinned-point share drops from 0.88 to ≤ 0.70.
- **P2, scale wander:** the spread (max − min) of the per-250-pose step ratio, excluding the first bin (start-up), goes from 0.212 (M0 run 1, bins 250-2750) to:
  - ≤ 0.12 for M1;
  - ≤ 0.08 for M2.
- **P3, global scale:** Sim3 scale from 1.116 to 0.97-1.03 (M1 and M2).
- **P4, accuracy:** ATE SE3 on full R_01 from 1.227 m to:
  - ≤ 0.70 m for M1;
  - ≤ 0.50 m for M2.
- **P5, M3 alone:** spread ≤ 0.15 and Sim3 scale within 0.95-1.05. With 3 active keyframes, the past's inertial scale enters only through the boundary edge, so the effect should be smaller than M1's.
- **P6, cost:** VI-BA time per solve (`ba_solve_ms / ba_solves`) rises by ≤ 3× for M1, and the frame time p50 stays ≤ 20 ms.

### Kill rules

- **K1:** P1 holds (the window is longer), but P2 misses for both M1 and M2. Window size is not the cause. Next suspects:
  - the unrectified R_01 accelerometer: check R_11, which has a factory calibration, on a full run;
  - the boundary edge (M3).
- **K2:** P4 holds, but P2 does not. The ATE moved for another reason (a different trajectory, or luck). Do not ship.
- **K3:** on the 800-frame windows, R_03 or R_11 tracked falls below card K's band (784-798, 798-800), or ATE leaves it. The longer window hurts short-term tracking: keep it off on the online path, and move the test to the offline BA.
- **K4:** P6 fails (cost over 3× or p50 over 20 ms). The window is too expensive for real time. Record it, and test ORB-SLAM3's iteration cap (`max_iterations: 10`).

### Bit-identity and noise

- **Identity:** no code changed. M0 is the base preset.
- **Noise:**
  - 3 runs per config. The app is not deterministic run to run, so P2-P4 are judged on all 3 runs.
  - The M0 band is set from M0's 3 runs. The run in the evidence counts as M0 run 1, and P2's 0.212 is its value; if the M0 band's spread is wider than 0.05, the predictions are read against the band.

### Result log

(empty)
