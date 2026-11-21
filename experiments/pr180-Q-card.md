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
