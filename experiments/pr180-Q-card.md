# Card Q: why a 10-keyframe VI-BA window slides, and the fix

Status: written 2026-11-21, before any run.

## Experiment: Q, diagnosis of the long-window instability, then one fix behind a knob
Gap tested: G35                Base: kornia-slam@fe8be42   Binary tag: q-base (record md5 at build)
Mechanism: not yet known. Candidates C1-C6 with file:line are in the plan table.
  Known departures from ORB-SLAM3 LocalInertialBA: IMU Huber on every edge
  (vi_ba_schur.rs:870-879 vs Optimizer.cc:2632-2643); pose update about the world
  origin in f32 (preintegration.rs:110-152, vi_ba_schur.rs:2584-2587 vs
  G2oTypes.cc:192-200); async merge leaves newer keyframes stale
  (correction.rs:717-739 vs Tracking.cc:1746-1762, 3103-3200); no outlier erase
  (vs Optimizer.cc:2853-2908); zero-mean accel-bias prior (vi_ba_schur.rs:2216-2232).
Theory: to be written for the confirmed candidate after stage 1. ORB-SLAM3 is the
  reference because it is stable at 10 keyframes on the same data (0.068 / 0.247 m).
Knob: stage 1 uses existing knobs only. Stage 2 adds one knob, default = today.
Baselines: full R_01. Rig 3 active: ATE SE3 1.11-1.23 m, scale 1.08-1.12, shift p50
  1.1-1.2 cm, 0-1 solves > 30 cm. Rig 10 active: shift p50 7.7-13.6 cm, 33-85 solves
  > 30 cm. Mono 3 active: ATE 3.54 / 4.39 m, scale 1.52 / 1.37.
  Reference: ORB-SLAM3 mono-inertial 0.068 / 0.247 m, scale 0.999 / 1.031.

Stage 1 predictions (each names the candidate it tests):
  D1 (C1) share of > 30 cm solves with abs(shift - r_v*T)/shift < 0.3: >= 80 %,
          and shift_tilt < 1 deg on them.     measured by: grep of [vi_ba] lines
  D2 (C3) Q0a Synchronous, 10 active: solves > 30 cm 33-85 -> <= 5; p50 <= 3 cm.
  D3 (C6) Q0b literal: solves > 30 cm -> <= 5.
  D4 (C2) W2: result at 100 m and 1 km offset differs from 0 m by > 0.1 mm.
  D5 (C5) W6: accel bias leaves truth by > 0.01 m/s^2.
  D6      Q0c offline global VI-BA on the base session: ATE SE3 <= 0.4 m.
  Mechanism counter: solves > 30 cm per run and shift p50, from [vi_ba] WINDOW.
Stage 2 predictions (knob on, 10 active, 3 runs each):
  P1 solves > 30 cm: 33-85 -> <= 5; shift p50: 7.7-13.6 cm -> <= 3 cm.
  P2 Rig ATE SE3: 1.11-1.23 m -> <= 0.6 m in 3/3; Sim3 scale 1.08-1.12 -> 0.97-1.03.
  P3 Mono ATE SE3: 3.54-4.39 m -> <= 1.0 m; scale 1.37-1.52 -> 0.95-1.05.
  P4 tracked stays 2898/2898; VI-BA time per solve <= 3x base; frame p50 <= 20 ms.
Kill rules:
  K1 P1 holds, P2 and P3 miss -> the slide was a symptom; the drift has another
     cause. Next: Q0c result decides between factor model and window chain.
  K2 P2 or P3 holds, P1 does not -> luck or another cause. Do not ship.
  K3 no stage 1 prediction holds -> the candidate list is incomplete. Go to the
     window replay before any fix.
  K4 two stage 1 predictions hold -> build the cheaper fix first, re-measure P1,
     then stack the second.
  K5 R_03 or R_11 800-frame tracked falls below card K's band (784-798, 798-800)
     with the knob on -> keep the knob off on the online path.
Bit-identity: app runs are not deterministic. Knob off is proven by a unit test:
  visual_inertial_bundle_adjust output equal bit for bit to the pre-change result
  on the W1 and W3 windows, and by the knob-off Rig runs staying in the base band.
Noise: 3 runs per config, full R_01; verdicts on all 3.

## Stage 1 result log (2026-11-21)

Binary: kornia-slam@f5d60ee (card Q debug fields), md5 c116d272c8e9ca533056b2cc9efdd176.
Full R_01, preset `pr180-frontend-1ms`, `MAX_FRAMES=2898`.

### Config runs

| Run | Solves > 30 cm | Shift p50 | Tracked | ATE SE3 | Scale |
|---|---|---|---|---|---|
| Q0a Synchronous, 10 active | 0, 0, 0 | 1.3, 1.3, 1.3 cm | 2898 x3 | 0.575, 0.575, 0.575 m | 0.970 x3 |
| Q0b literal (10 active, BoundaryOrRecentInit, bias rows unscaled, 10 iterations) | 7, 11, 75 | 2.6, 2.9, 4.4 cm | 2747, 2608, 2753 | 6.46, 17.98, 16.28 m | 0.69, 0.15, 0.13 |
| Q0c base online (3 active) | 0 | 1.1 cm | 2898 | 1.058 m | 1.076 |
| Q0c offline global VI-BA on that session | - | - | 2898 | 0.417 m (Sim3 0.150 m) | 0.954 |

The three Q0a runs are identical: the synchronous path is deterministic.

### Forensics on the card M, N, O, P runs (10 active, async)

- Boundary `|r_v| x T` explains < 30 % of the shift on 0-12 of 33-86 large solves per run (0-15 %); `r_v T / shift` p50 0.01-0.36.
- Interior IMU edges at the first iteration of a large solve carry 5-10x the `|r_v|` of a small one (p50 0.03-0.16 against 0.002-0.09 m/s): the window starts inconsistent.
- Shift grows with frames since the previous solve. M1 r1: gap 0-9 frames p50 2.0 cm, 0 large; 10-19 p50 11.9 cm, 17 large; 20-29 p50 14.7 cm, 12 large; 30-39 p50 26.4 cm. The 3-active base and Q0a solve every 4-6 frames.
- Shift against distance from the first pose: r = -0.01 to 0.51.
- Q0b large solves: shift_tilt < 1 deg on 3/7, 7/11, 56/75.

### Synthetic windows (W1-W7, 3 and 10 active, mono and depth)

- W1 green: at truth no step is taken.
- W2 red: at 1 km the result moves 1.3-3.2 mm (100 m: 0.01-0.55 mm), same iteration count.
- W3 green: exact data to < 0.02 mm; 0.5 px error 0.10-0.39 of the solver's own sigma.
- W4 green: from a 0.5 m/s, 2 deg slide back to < 0.012 mm.
- W5: tilt 1 deg moves the newest 1.4-4.2 cm and turns it 0.90-0.96 deg; scale 0.9 moves it 4.4-23.2 cm (10 active mono worst).
- W6 red: the accel bias leaves truth by 0.020, 0.020, 0.010, 0.007 m/s^2 (pose 1.1-9.4 mm); with `accel_bias_prior_weight: 0` it stays exactly.
- W7 red on ATE only: 50 chained windows, ATE 26.1 / 19.7 mm (3 / 10 active), scale 1.0026 / 1.0009, shift p50 1.0 / 2.6 mm, max 2.3 / 9.8 mm.

### Verdicts

- D1 (C1) missed: 0-15 % (Q0b 0-39 %) against >= 80 %.
- D2 (C3) holds: Q0a 0 solves > 30 cm, p50 1.3 cm, in 3/3.
- D3 (C6) missed: Q0b 7, 11, 75.
- D4 (C2) holds, at 1.3-3.2 mm at 1 km; R_01 stays within about 30 m of its origin.
- D5 (C5) holds in 3 of 4 windows, at <= 0.02 m/s^2.
- D6 missed narrowly: 0.417 m against <= 0.4 m.

### Basalt beside ORB-SLAM3 (rows that matter)

| Item | Ours | ORB-SLAM3 | Basalt (`sqrt_keypoint_vio.cpp`, `vio_config.cpp`) |
|---|---|---|---|
| Past information | fixed keyframe, boundary edge x1e-2 | same, Huber on it | square-root marginal prior (`vio_sqrt_marg`), 7 keyframes + 3 frames |
| Precision | f32 pose state and accept test | f64 | f64 instantiation (`template class ...<double>`) |
| Pose update | right perturbation of `T_cw`, about world origin | body-local | translation additive, rotation left-multiplied about the body origin (`incPose`): no lever arm |
| IMU kernel | Huber on every edge | boundary only, or all before VIBA2 | none |
| Visual kernel | Huber 25 px^2, unit sigma | Huber 5.991 / 7.815, octave sigma | Huber 1.0 on whitened residual, sigma 0.5 px |
| Outliers | never erased after local BA | erased each solve | `filterOutliers` not called in VIO (TODO, `:1405`); only the mapper |
| Iterations, damping | 50, `lambda diag(H)` from 1.0 | 10 (4 if large), g2o LM | 7, `lambda diag(H)` from 1e-4, stop at `f_diff < 1e-6` or step < 1e-4 |

ORB-SLAM3 and Basalt disagree on the past (fixed edge against marginal prior), on the IMU kernel (boundary Huber against none), on outlier erasure (each solve against none in VIO) and on iterations (10 against 7) and initial damping (1.0 against 1e-4).

## Owner-session review of stage 1 (2026-11-21)

- **Re-scored independently:**
  - Q0a: ATE SE3 0.575 m, Sim3 0.519 m, scale 0.970, identical in 3/3 runs;
  - Q0b: 6.46, 17.98 and 16.28 m;
  - Q0c base online: 1.058 m.
- **C3 is confirmed as a cause of the slide, not as the whole drift.** Commit 66968bd's subject ("the slide is the async merge") is stronger than the evidence:
  - Q0a does not separate newer keyframes left stale by the merge from a slow solve absorbing many keyframes at once.
  - Q0a's Sim3 0.52 m is still about 8× ORB-SLAM3's 0.05-0.07 m on the same data.
- **Synchronous is not a fix: it is not real time.**
  - Q0a takes 985 ms per VI-BA solve (532 s over 540 solves), with frame time p90 1051 ms and p99 1778 ms.
  - The base (asynchronous, 3 active) takes 120 ms per solve, with frame time p90 108 ms.
- **Next:** a fix plan for C3, taking the best of ORB-SLAM3 (re-predict from the corrected keyframe; insert only while the mapper is idle) and Basalt (one-thread window with a marginal prior). It also has to make the 10-keyframe solve real time. A 3-active synchronous run separates the two C3 readings.
