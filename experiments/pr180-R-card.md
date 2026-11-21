# Card R: keyframes inserted during a slow local VI-BA start the next window inconsistent

Status: written 2026-11-21, before any run. Approved by the owner as drafted, with these
rulings: P5's 100 ms is judged at p50 on single-job runs, p90 recorded too; `schur` stays a
knob (default PerObservation); the Mono config is `scratchpad/mono/cfg/r01full-mono.ron`, and a
synchronous 10-active Mono baseline (Q0a-mono) runs first; no default flip in this card.

## Experiment: R, keyframe seed, merge propagation and a 100 ms 10-keyframe solve
Gap tested: G35                Base: kornia-slam@3fcdd32   Binary tag: r-base (record md5 at build)
Mechanism: with 10 active keyframes a solve takes 985-1356 ms and absorbs 4.5-4.8
  keyframes. (b1) Each was seeded with the reference keyframe's velocity
  (system/mod.rs:1728-1730, 1682-1695, 2891-2894): first-iteration r_v 1.37-1.65 m/s
  on the crossing edge, r_p 0.20-0.30 m on new edges, Huber weight 0.0003. (a) The
  merge corrects snapshot keyframes only (correction.rs:712-734): crossing r_p
  6-9 cm. The solve costs 47 ms per iteration: the Schur pair loop is quadratic in
  observations per point (vi_ba_schur.rs:982-1003, 2086-2111).
Theory: an IMU-consistent start keeps the IMU edges at full weight from iteration 1,
  so the window cannot slide in their null space. Seed: ORB-SLAM3 Tracking.cc:1746-1762.
  Merge: pose carried rigidly (OKVIS2 ViSlamBackend.cpp:1442-1470, ORB-SLAM3
  LoopClosing.cc:2355-2358), velocity re-predicted on the IMU edges, landmarks by
  reference keyframe (LoopClosing.cc:2481-2484). Budget: Basalt 7 iterations, lambda
  1e-4 (vio_config.cpp:73,78), one Schur block per keyframe and point (g2o).
Knob: mapping.keyframe_seed: ReferenceVelocity | ImuPropagated;
  mapping.local_ba.merge: SnapshotOnly | PropagateToNewer;
  mapping.local_ba.schur: PerObservation | PerKeyframe; mapping.local_ba.initial_lambda: 1.0.
  Defaults are today's behaviour.
Baselines: full R_01. Base (async, 3 active): ATE SE3 1.06-1.23 m, scale 1.08-1.12,
  120 ms per solve, frame p50 13 ms, p90 108 ms. Async 10 active (cardM M1 r1-r3):
  33-85 solves > 30 cm, shift p50 7.7-13.6 cm, 848-857 keyframes. Q0a (sync, 10):
  0 solves > 30 cm, p50 1.3 cm, ATE 0.575 m, Sim3 0.519 m, scale 0.970, 985 ms per
  solve, frame p90 1051 ms. Mono base: 3.54 / 4.39 m, scale 1.52 / 1.37.
  Reference: ORB-SLAM3 mono-inertial 0.068 / 0.247 m, Sim3 0.048-0.068 m.

Stage 1 predictions (config only):
  S3    sync, 3 active: ATE 1.0-1.25 m, scale 1.07-1.12 (base has n_new = 1).
  A10-7 async, 10 active, 7 iterations, lambda 1e-4: n_new 4.5-4.8 -> 2.0-2.5;
        shift p50 -> 2-5 cm; solves > 30 cm -> 5-25; crossing r_v p50 >= 1.0 m/s
        on solves with n_new >= 2.
  S10-7 sync, 10 active, same: 985 -> 300-400 ms per solve; ATE 0.50-0.70 m.
  T1    schur share of iteration time at 10 active: >= 75 %.
Stage 2 predictions (async, 10 active, slow solve, 3 runs each):
  seed only:  crossing r_v p50 1.37-1.65 -> <= 0.3 m/s; new-edge r_p p50 0.20-0.30
              -> <= 0.05 m; solves > 30 cm -> 10-30.
  merge only: crossing r_v p50 <= 0.05 m/s; solves > 30 cm <= 8; shift p50 <= 4 cm;
              keyframes 848-857 -> <= 700.
  sync 10 with merge on: trajectory cmp-identical to Q0a.
Final predictions (all knobs, async, 10 active, 3 runs each):
  P1 solves > 30 cm: 33-85 -> <= 3; shift p50: 7.7-13.6 cm -> <= 2 cm.
  P2 n_new mean: 4.5-4.8 -> <= 1.2; ba_stale_keyframes per solve <= 0.2;
     crossing r_v p50 <= 0.05 m/s.
  P3 Rig ATE SE3: 1.06-1.23 m -> 0.50-0.70 m in 3/3; Sim3 <= 0.60 m; scale 0.96-1.01.
     Still 2-8x ORB-SLAM3: this card does not claim that gap.
  P4 Mono ATE SE3: 3.54-4.39 m -> <= 1.5 m; scale 0.90-1.15; within 20 % of Q0a-mono.
  P5 VI-BA <= 100 ms p50 per solve, single job (985 ms); iteration <= 12 ms (47 ms);
     iterations p50 <= 7 (20); frame p50 <= 20 ms, p90 <= 120 ms; tracked 2898/2898.
  Mechanism counters: solves > 30 cm, shift p50, n_new, ba_stale_keyframes,
     crossing r_v p50 (window_edges.py).
Kill rules:
  K1 P1 and P2 hold, P3 misses (ATE > 0.8 m) -> the start state was a symptom.
     Next: offline replay of windows against the global VI-BA result (0.417 m).
  K2 P3 holds, P1 or P2 does not -> luck or another cause. Do not ship.
  K3 seed only already gives <= 5 solves > 30 cm -> (b1) is the slide; keep merge
     for the tracker pose only and say so.
  K4 T1 misses (< 50 %) -> the cost is elsewhere; stop before the Schur change.
  K5 R_03 or R_11 800-frame tracked below card K's band (784-798, 798-800) with the
     knobs on -> knobs stay off on the online path.
  K6 S3 lands <= 0.7 m -> staleness hurts at one keyframe of lag; re-plan the base.
Bit-identity: knobs off, a Q0a-config run must be cmp-identical to
  cardQ/out/r01full-Q0a-r1/traj.txt (the sync path is deterministic). Unit test:
  visual_inertial_bundle_adjust equal bit for bit on W1 and W3.
Noise: 3 runs per config, full R_01; verdicts on all 3. Timing from single-job runs,
  load recorded; async runs are not deterministic.

## Window edges on the earlier runs (window_edges.py, before any card R run)

`python3 -I experiments/quick-eval/window_edges.py <run>/stderr.txt`, full R_01, the card M-Q
run dirs. First-iteration `[vi_ba] edge` lines, classed against the previous solve's newest
keyframe. r_v in m/s, r_p in m, hw is the Huber weight on the nav block.

| Run | Config | Solves | n_new mean | > 30 cm | Shift p50 | Old r_v p50 | Crossing r_v p50 / r_p p50 (p90) / hw p50 | New r_v p50 / r_p p50 / hw p50 |
|---|---|---|---|---|---|---|---|---|
| M1-r1 | async, 10 active | 178 | 4.69 | 33 | 7.7cm | 0.0002 | 1.4789 / 0.0605 (0.3915) / 0.0004 | 0.1262 / 0.2282 / 0.0003 |
| M1-r2 | async, 10 active | 179 | 4.73 | 34 | 10.6cm | 0.0002 | 1.6464 / 0.0836 (0.3889) / 0.0003 | 0.1444 / 0.2971 / 0.0002 |
| M1-r3 | async, 10 active | 174 | 4.82 | 60 | 13.6cm | 0.0003 | 1.3552 / 0.0898 (1.5009) / 0.0004 | 0.1416 / 0.1975 / 0.0003 |
| M2-r1 | async, 10 active, large_window 25 | 88 | 9.12 | 59 | 132.6cm | 0.0016 | 1.4778 / 1.3752 (35.7526) / 0.0001 | 0.1780 / 0.1736 / 0.0004 |
| M2-r2 | async, 10 active, large_window 25 | 57 | 8.61 | 31 | 68.4cm | 0.0023 | 1.6145 / 0.1754 (7.4523) / 0.0003 | 0.1603 / 0.2024 / 0.0003 |
| M2-r3 | async, 10 active, large_window 25 | 62 | 6.74 | 31 | 27.1cm | 0.0025 | 1.3654 / 0.0975 (2.1377) / 0.0004 | 0.1889 / 0.2477 / 0.0003 |
| N0-r1 | base (async, 3) | 579 | 1.01 | 0 | 1.1cm | 0.0002 | 0.2336 / 0.0187 (0.0526) / 0.0029 | 0.1905 / 0.0620 / 0.0008 |
| N0-r2 | base (async, 3) | 586 | 1.01 | 0 | 1.1cm | 0.0002 | 0.2478 / 0.0202 (0.0527) / 0.0027 | 0.3232 / 0.1256 / 0.0006 |
| N1-r1 | acc noise x10 | 560 | 1.01 | 0 | 1.2cm | 0.0045 | 0.2356 / 0.0246 (0.0617) / 0.0251 | 0.1925 / 0.0536 / 0.0095 |
| N1-r2 | acc noise x10 | 564 | 1.01 | 0 | 1.2cm | 0.0049 | 0.2596 / 0.0263 (0.0650) / 0.0234 | 0.1553 / 0.0863 / 0.0084 |
| N1-r3 | acc noise x10 | 567 | 1.00 | 0 | 1.2cm | 0.0051 | 0.2407 / 0.0267 (0.0631) / 0.0238 | 0.3902 / 0.1188 / 0.0063 |
| N2-r1 | N1 + 10 active | 201 | 4.13 | 56 | 12.6cm | 0.0240 | 2.4386 / 0.0899 (0.8237) / 0.0014 | 0.1801 / 0.5601 / 0.0010 |
| N2-r2 | N1 + 10 active | 201 | 4.09 | 37 | 8.0cm | 0.0082 | 3.5837 / 0.0639 (0.4386) / 0.0018 | 0.1439 / 0.7384 / 0.0008 |
| N2-r3 | N1 + 10 active | 187 | 4.55 | 55 | 10.4cm | 0.0098 | 4.5348 / 0.0905 (0.6498) / 0.0008 | 0.1658 / 0.8098 / 0.0007 |
| O1-r1 | 10 active, boundary x1 | 177 | 4.63 | 57 | 15.3cm | 0.0007 | 0.9150 / 0.1354 (1.5373) / 0.0004 | 0.1541 / 0.1448 / 0.0004 |
| O1-r2 | 10 active, boundary x1 | 189 | 4.14 | 85 | 20.7cm | 0.0005 | 1.6220 / 0.1783 (2.5494) / 0.0003 | 0.1688 / 0.2806 / 0.0002 |
| O1-r3 | 10 active, boundary x1 | 176 | 4.73 | 60 | 14.3cm | 0.0008 | 1.4979 / 0.1273 (6.3706) / 0.0004 | 0.1429 / 0.2166 / 0.0003 |
| O2-r1 | O1 + ORB-SLAM3 literal | 176 | 4.61 | 86 | 25.1cm | 0.0004 | 1.6886 / 0.2463 (5.4462) / 1.0000 | 0.1724 / 0.2443 / 1.0000 |
| O2-r2 | O1 + ORB-SLAM3 literal | 188 | 4.34 | 65 | 16.6cm | 0.0005 | 1.3380 / 0.1559 (3.1515) / 1.0000 | 0.1395 / 0.1995 / 1.0000 |
| O2-r3 | O1 + ORB-SLAM3 literal | 184 | 4.41 | 83 | 19.6cm | 0.0006 | 1.5959 / 0.1605 (12.8933) / 1.0000 | 0.1791 / 0.2350 / 1.0000 |
| O3-r1 | boundary x1, 3 active | 631 | 1.01 | 8 | 1.9cm | 0.0010 | 0.2598 / 0.0242 (0.0910) / 0.0018 | 0.1482 / 0.0329 / 0.0014 |
| O3-r2 | boundary x1, 3 active | 635 | 1.02 | 10 | 2.1cm | 0.0011 | 0.2953 / 0.0280 (0.1098) / 0.0014 | 0.1192 / 0.0416 / 0.0014 |
| O3-r3 | boundary x1, 3 active | 621 | 1.02 | 3 | 1.9cm | 0.0009 | 0.2890 / 0.0247 (0.1091) / 0.0016 | 0.1557 / 0.0570 / 0.0010 |
| P1-r1 | disparity sigma | 596 | 1.01 | 0 | 1.1cm | 0.0002 | 0.2596 / 0.0197 (0.0545) / 0.0026 | 0.3869 / 0.0756 / 0.0009 |
| P1-r2 | disparity sigma | 590 | 1.01 | 0 | 1.2cm | 0.0002 | 0.2626 / 0.0200 (0.0538) / 0.0026 | 0.1730 / 0.0572 / 0.0009 |
| P1-r3 | disparity sigma | 595 | 1.02 | 0 | 1.1cm | 0.0001 | 0.2521 / 0.0187 (0.0502) / 0.0027 | 0.1485 / 0.0445 / 0.0011 |
| P2-r1 | P1 + 10 active | 202 | 4.05 | 78 | 19.6cm | 0.0004 | 2.9188 / 0.1404 (3.3236) / 0.0001 | 0.2167 / 0.4766 / 0.0001 |
| P2-r2 | P1 + 10 active | 187 | 4.27 | 60 | 12.0cm | 0.0002 | 1.1949 / 0.1046 (4.5328) / 0.0005 | 0.1439 / 0.1804 / 0.0003 |
| P2-r3 | P1 + 10 active | 185 | 4.57 | 79 | 24.8cm | 0.0004 | 2.2408 / 0.1858 (2.4836) / 0.0001 | 0.1731 / 0.5837 / 0.0001 |
| Q0a-r1 | sync, 10 active | 530 | 1.00 | 0 | 1.3cm | 0.0004 | 0.2043 / 0.0137 (0.0392) / 0.0033 | - |
| Q0a-r2 | sync, 10 active | 530 | 1.00 | 0 | 1.3cm | 0.0004 | 0.2043 / 0.0137 (0.0392) / 0.0033 | - |
| Q0a-r3 | sync, 10 active | 530 | 1.00 | 0 | 1.3cm | 0.0004 | 0.2043 / 0.0137 (0.0392) / 0.0033 | - |
| Q0b-r1 | async, 10 active, ORB-SLAM3 literal, 10 iterations | 358 | 2.10 | 7 | 2.7cm | 0.0009 | 1.5544 / 0.0417 (0.1224) / 1.0000 | 0.1488 / 0.2529 / 1.0000 |
| Q0b-r2 | async, 10 active, ORB-SLAM3 literal, 10 iterations | 364 | 2.10 | 11 | 2.9cm | 0.0009 | 1.4018 / 0.0418 (0.1732) / 1.0000 | 0.1496 / 0.2241 / 1.0000 |
| Q0b-r3 | async, 10 active, ORB-SLAM3 literal, 10 iterations | 411 | 1.97 | 75 | 4.4cm | 0.0013 | 1.5216 / 0.0616 (1.0842) / 1.0000 | 0.1671 / 0.2376 / 1.0000 |
| Q0c-base | base (async, 3) | 571 | 1.01 | 0 | 1.1cm | 0.0002 | 0.2232 / 0.0174 (0.0523) / 0.0030 | 0.2430 / 0.0340 / 0.0015 |

P3-r1 and P3-r2 (`imu: false`) have no VI-BA solves. Shift against n_new (M1 r1-r3): n_new 1-2
p50 1.3 / 2.8 / 2.1 cm with 0 / 1 / 13 large; 3-4 5.6-5.7 cm, 4-11 large; 5-6 11.1-17.5 cm,
11-15 large; >= 7 23.8-29.8 cm, 14-22 large.

Every 10-active asynchronous run at 50 iterations absorbs 4.0-4.8 keyframes per solve (M2,
25 active: 6.7-9.1). Its crossing edge starts at r_v 0.9-4.5 m/s and its new edges at r_p
0.14-0.81 m, with Huber weight 1e-4-1e-3 (O2 and Q0b turn the IMU Huber off: hw 1.0). Q0b,
async at 10 iterations, absorbs 1.97-2.10 and still starts the crossing edge at 1.40-1.55 m/s.
Every 3-active or synchronous run absorbs 1.00-1.02, with crossing r_v 0.20-0.30 m/s.
