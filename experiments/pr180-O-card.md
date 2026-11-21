# Card O: a longer VI-BA window drifts in its gauge because its anchor to the past is weak

Status: written 2026-11-21, before any run. Existing knobs only, so there is no code change.

## Experiment: O, an ORB-SLAM3-style boundary for the longer window

- **Gap tested:** G35 (scale drift on full sequences). It is the blocker behind card M, where a longer window was worse, and card N, which was refuted.
- **Base:** kornia-slam @ 9a28785, preset `pr180-frontend-1ms`.
- **Binary:** md5 7d09a8c68f297830aac35ad1dc844ba7, the same as cards M and N.

### Evidence (2026-11-21, full R_01)

- **Card N refuted overconfidence as the main cause.**
  - Accelerometer noise × 10 (N1) brings the Sim3 scale to 0.94-0.98, but ATE gets worse: 1.36-1.66 m against 1.11-1.23 m.
  - With 10 active keyframes (N2), 2 of 3 runs still blow up.
- **Window jumps.** From the `[vi_ba] WINDOW` `shift` field:

| Run | Median shift per solve | Solves moving > 30 cm |
|---|---|---|
| 3 active (base, N0, N1) | 1.1-1.2 cm | 0-1 |
| 10 active (M1 ×2, N2 ×2) | 7.7-13.6 cm | 33-60 |

  - Only 2-20 of those jumps have the boundary IMU edge Huber-weighted below 0.1. The rest happen with the boundary edge at full Huber weight, but still scaled by `boundary_imu_info_scale: 1e-2`.
- **One N2-r1 solve, the blow-up:**
  - `active=10 fixed=9 pinned=212/753 obs_fixed=602/3659`;
  - `shift=3.52 m shift_rot=4.95`, with points moved 7-14 m;
  - the boundary edge has `r_v = 1.99 m/s`, χ² 4.4e5 and hw 0.0062, while the interior edges have χ² 6-20;
  - velocity afterwards is 2.77 m/s.

### Mechanism

- A visual-inertial window with gravity known has 4 unobservable directions: position and yaw. Only the fixed context fixes them, through pinned points and the boundary IMU edge.
- **With 3 active keyframes,** 88 % of points are pinned, so the window holds.
- **With 10 active,** about 28 % are pinned. The boundary edge carries only 1/100 of its information (`system/config/mod.rs:597`), and Huber can cut it further. The window then slides in its gauge on every solve.
- This is what makes every longer-window or stronger-IMU change fail.

### Reference

ORB-SLAM3 `Optimizer::LocalInertialBA`:
- 10 optimisable keyframes;
- the keyframe before them is fixed: pose, velocity and biases;
- its IMU edge to the window is at full information, with Huber on that edge only (`i==N-1 || bRecInit`);
- the covisible keyframes outside the window are also fixed.

Our `boundary_imu_info_scale: 1e-2` is the departure.

### Knobs (existing)

- `boundary_imu_info_scale`
- `imu_huber`
- `boundary_scales_bias_rows`
- `active_keyframes`

### Configs (full R_01)

| Name | `mapping.local_ba` |
|---|---|
| O1 | `active_keyframes: 10, boundary_imu_info_scale: 1.0` |
| O2 | O1 plus `imu_huber: BoundaryOnly, boundary_scales_bias_rows: false` (the ORB-SLAM3 literal) |
| O3 | `boundary_imu_info_scale: 1.0` alone (card M's M3) |

### Predictions (3 runs each)

- **P1, mechanism:**
  - solves moving > 30 cm go from 33-60 (M1) to ≤ 5 (O1, O2);
  - the median shift per solve goes from 7.7-13.6 cm to ≤ 3 cm.
- **P2, accuracy (O1, O2):**
  - ATE SE3 ≤ 0.9 m (base 1.11-1.23 m);
  - Sim3 scale 0.95-1.05;
  - no run above 1.5 m.
- **P3, scale wander:** spread ≤ 0.15 (O1, O2); base 0.18-0.21.
- **P4, O3:** inside the base band (ATE 1.0-1.3 m). With 3 active keyframes the window is already anchored by points, so O3 tests whether the boundary weight matters at all.

### Kill rules

- **K1:** P1 holds (no jumps), but P2 misses. Gauge drift was a symptom, not the drift cause. Next, the Rust ground-truth harness.
- **K2:** P2 holds, but P1 does not (jumps remain). ATE luck. Do not ship.
- **K3:** O2 is better than O1 beyond noise. The interior Huber also mattered, so ship O2 as the ORB-SLAM3 literal.

### Result log

- **2026-11-21, scoring.** Full R_01, 3 runs per config.

| Config | Solves moving > 30 cm | Median shift | ATE SE3 | Scale |
|---|---|---|---|---|
| O1 | 57, 85, 60 | 15.3, 20.3, 13.3 cm | 6.68, 7.07, 8.28 m | 0.62, 0.61, 0.47 |
| O2 | 86, 65, 83 | 25.0, 16.6, 19.0 cm | 5.54, 7.66, 17.20 m | 0.74, 0.54, 0.12 |
| O3 | 8, 10, 3 | 1.9-2.1 cm | 2.64, 2.75, 1.51 m | 0.98, 0.96, 1.09 |

- **Verdict:** P1-P4 missed, all worse. A full-weight boundary edge makes the drift worse, not better, so a weak anchor is not the cause.

### Correction (2026-11-21)

- The reference claim above is wrong. ORB-SLAM3 scales the boundary edge by 1e-2 and puts Huber on it (`Optimizer.cc:2632-2643`: `if(i==N-1) setInformation(information()*1e-2)`, Huber delta `sqrt(16.92)`).
- O1 and O2 (`boundary_imu_info_scale: 1.0`) were therefore not the ORB-SLAM3 literal. The literal is run as Q0b in card Q.
- The predictions above are left as written.
