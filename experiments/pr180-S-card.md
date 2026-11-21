# Card S: bias random walk at sensor strength across the VI-BA window boundary

Status: 2026-11-21. Stage A run on R_01 (3 pairs): refuted as a score fix (K1 and K3 fired), mechanism confirmed (M1 hit). Knob stays at its default. Stage B not built. Plan: planner (Fable), owner answers 2026-11-21.

## Experiment: S, bias random walk at sensor strength across the window boundary
Gap tested: G35, G6 (bias tie), G18    Base: kornia-slam@81bc775 (pr180c/F1)
Binary tag: stage A = S/F1/bin/kornia-slam-f1 (md5 52e38c46...); stage B = P/bin/kornia-slam-s (md5 at build)
Mechanism: vi_ba_schur.rs:864-875 scales the boundary edge's bias-walk rows by 1e-2
  (defaults :385, bundle_adjustment.rs:128, config/mod.rs:626). The window's biases
  are tied to the past at 10x the calibration walk sigma. Measured: boundary bias
  residual p50 7.8-15 sigma gyro, 3.0-5.5 sigma accel; newest-keyframe bias SD
  3.0-3.4e-3 rad/s and 0.052-0.068 m/s^2 on R_01. Stage B: zero-mean accel prior,
  vi_ba_schur.rs:396, :2328-2348.
Theory: at full weight the bias is a filtered walk with sigma about 6e-4 rad/s and
  0.013 m/s^2, so gyro and accel constrain heading and local scale beyond one window.
  ORB-SLAM3 Optimizer.cc:2632-2657; Basalt sqrt_keypoint_vio.cpp:115-116, :93-108.
Knob: A: mapping.local_ba.boundary_scales_bias_rows (exists; default true = today) -> false.
      B: mapping.local_ba.accel_bias_prior_weight (new; default 1e2 = today) -> 0.0.
Baselines (F1, 3 runs each, S/F1/out; BASE is re-run paired with S1):
  R_01 full: ATE SE3 0.501, 0.295, 0.508 m; Sim3 0.446, 0.227, 0.463 m; scale 0.973-0.977;
    tracked 2898; chord(100-pose) step-ratio SD 0.026-0.038.
  R_03: SE3 0.104-0.117 m, Sim3 0.069-0.075 m, tracked 758-778.
  R_11: SE3 0.157-0.178 m, Sim3 0.112-0.113 m, tracked 792-800.
Predictions, stage A (S1 against paired BASE, 3 runs each):
  M1 (mechanism counter) boundary bias residual p50: gyro 7.8-15 -> <= 2.0 sigma,
     accel 3.0-5.5 -> <= 2.0 sigma, on all three.       measured by: biaswalk.py
  M2 newest-keyframe bias on R_01, per axis: gyro SD 3.0-3.4e-3 -> <= 1.5e-3 rad/s;
     accel SD 0.052-0.068 -> <= 0.030 m/s^2; step RMS 1.1-1.2e-3 -> <= 2.5e-4 rad/s
     and 0.016-0.020 -> <= 6.5e-3 m/s^2.               measured by: biaswalk.py
  P1 R_01 full ATE Sim3: 0.227-0.463 -> 0.12-0.30 m in 3/3.   measured by: ate.py
  P2 R_01 full ATE SE3: 0.295-0.508 -> 0.26-0.40 m in 3/3; scale stays 0.965-0.985.
  P3 R_01 chord step-ratio SD: 0.026-0.038 -> <= 0.022 in 3/3.
  P4 R_03: SE3 0.09-0.12 m, tracked >= 758 in 3/3. R_11: SE3 0.13-0.18 m, tracked >= 792 in 3/3.
  P5 W6 green with the knob at false: accel bias off truth by <= 0.002 m/s^2 in all four shapes.
Predictions, stage B (S1+S2 against S1, new binary):
  M3 [vi_ba] COST bias_prior = 0 in every solve.
  P6 R_01, R_03, R_11 inside the S1 bands; R_01 mean accel bias moves <= 0.01 m/s^2 per axis;
     W6 residual < 1e-6.
Kill rules:
  K1 (mechanism moved, score did not) M1 holds on R_01 and Sim3 is above 0.30 m in 2 of 3:
     bias memory is not what limits R_01's shape. Keep the knob (reference rule), do not
     credit it. Next: split the remaining error into heading and local scale, then the
     visual side (final cost 4-8 px^2 per observation, Huber knee 25, no outlier erase).
  K2 (score moved, mechanism did not) Sim3 <= 0.30 m in 3/3 with gyro p50 still above
     4 sigma: luck or load. Do not credit; 3 more paired runs.
  K3 (harm) any R_01 run above 0.60 m SE3 or below 2898 tracked; R_03 tracked under 758
     in 2/3; a VIBA accepted in BASE and rejected in S1; or vi_ba_solves_3plus_rejects
     above 5 % of solves: the loose bias was absorbing a real model error. Keep off.
     Next: gyro rows only, and compare raw-IMU R_01/R_03 against rectified R_11
     (card N's 1.2 % accelerometer scale error is the suspect).
  K4 P5 misses: the arithmetic is wrong. Stop before any sequence run.
  K5 stage B outside the S1 band (R_01 Sim3 median up by more than 0.05 m, or R_11 mean
     accel bias beyond +/-0.05): the prior was hiding a gravity error. Keep 1e2; open a
     gravity card.
Bit-identity: stage A changes no code. Stage B binary, knobs off: P/baseline/r01full-S10-7.ron,
  RAYON_NUM_THREADS=1, traj md5 d4f77b567b912b5a4caa2af6085a35b1 (cmp with P/baseline/traj.txt);
  unit: r_the_default_solve_keeps_its_bits unchanged.
Noise: async runs differ run to run; 3 per config, BASE and S1 started in pairs under the
  same load (as S/F1/ab.sh). Known R_01 band 0.30-0.50 m (n = 12). Load every 30 s to load.log;
  a pair whose load p50 differs by more than 2 is repeated.

Note (measurement, 2026-11-21): biaswalk.py reproduces the boundary values above (R_01 r1 gyro p50
8.05, accel 3.05; R_03 15.11 / 5.43; R_11 12.32 / 5.38) and the bias SD/step RMS. Its interior gyro
p50 over all interior edges is 0.58-0.59 on R_01 (0.48 R_03, 0.60 R_11), not the 0.72-0.74 quoted in
the diagnosis; interior accel matches. The plan's formula is kept; M1 uses boundary edges only.

## Results (2026-11-21)

### K4: fired

P5 (W6; worst active-keyframe accel bias error vs truth, m/s^2):

| Shape | rows scaled, prior 1e2 (today) | rows full, prior 1e2 (S1) | P5 (<= 0.002) | rows full, prior 0 (S1+S2) |
|---|---|---|---|---|
| active=3 mono | 0.0206 | 0.0006 | hit | 0.0 |
| active=3 depth | 0.0199 | 0.0006 | hit | 0.0 |
| active=10 mono | 0.0102 | 0.0040 | miss | 0.0 |
| active=10 depth | 0.0068 | 0.0034 | miss | 0.0 |

W6 is green at its 0.01 assert. P5 misses in 2 of 4 shapes, so K4 fired. Full rows cut the
error 2-35x; on 10 active keyframes it stays 2x above 0.002.

Owner decision: diagnose the zero-mean accel prior first (test-only third setting,
`accel_bias_prior_weight` 0 on the W6 params). With rows full and the prior at 0 the error is
exactly 0.0 in all four shapes (gyro bias error 0.0 too): the fixture starts at truth with exact
data, so the zero-mean prior is the only force that moved the bias. The theory holds for A+B
together; the stage A miss on 10 keyframes is the prior (stage B's knob), not the walk arithmetic.
On this, stage A runs on R_01 (owner, 2026-11-21). The fingerprint test
`r_the_default_solve_keeps_its_bits` is green after the change.

### Stage A, R_01 full (F1 binary md5 52e38c46, paired rounds, 2026-11-21 22:26-22:33)

BASE and S1 started together in each round (cardS/pairs.sh: one run.sh, JOBS=2, 1 run each).
Both `local BA:` lines checked: BASE `(bias rows scaled)`, S1 `(bias rows full)`.

| Round (load p50) | Config | SE3 m | Sim3 m | scale | chord SD | tracked | bnd gyro / accel z p50 | int gyro / accel z p50 | bg SD rms | bg step rms | ba SD rms | ba step rms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 (5.09) | BASE | 0.488 | 0.434 | 0.973 | 0.037 | 2898 | 7.99 / 2.97 | 0.55 / 0.49 | 3.18e-3 | 1.15e-3 | 0.060 | 0.0176 |
| r1 (5.09) | S1 | 0.367 | 0.314 | 0.977 | 0.034 | 2898 | 1.28 / 1.30 | 0.81 / 0.75 | 1.57e-3 | 5.04e-4 | 0.058 | 0.0138 |
| r2 (7.22) | BASE | 0.601 | 0.573 | 0.978 | 0.037 | 2898 | 8.06 / 3.00 | 0.59 / 0.48 | 3.22e-3 | 1.11e-3 | 0.059 | 0.0191 |
| r2 (7.22) | S1 | 0.658 | 0.630 | 0.977 | 0.040 | 2898 | 1.25 / 1.30 | 0.80 / 0.73 | 1.54e-3 | 5.12e-4 | 0.053 | 0.0135 |
| r3 (7.35) | BASE | 0.353 | 0.340 | 0.988 | 0.031 | 2898 | 7.97 / 2.99 | 0.58 / 0.48 | 3.08e-3 | 1.13e-3 | 0.058 | 0.0172 |
| r3 (7.35) | S1 | 0.399 | 0.390 | 0.990 | 0.034 | 2898 | 1.27 / 1.28 | 0.80 / 0.71 | 1.55e-3 | 5.11e-4 | 0.047 | 0.0133 |

All runs: VIBA0/1/2 accepted, vi_ba_failed 0, vi_ba_solves_3plus_rejects 0, give_ups 2.
Chord SD: chord(100-pose, stride 1) length ratio SD. Bias SD and step are RMS over the three axes.

| Prediction | Result |
|---|---|
| M1 boundary z p50 <= 2.0 | hit: gyro 1.25-1.28, accel 1.28-1.30 |
| M2 bg SD <= 1.5e-3, step <= 2.5e-4; ba SD <= 0.030, step <= 6.5e-3 | miss: bg SD 1.54-1.57e-3 (z axis 2.1-2.2e-3), step 5.0-5.1e-4; ba SD 0.047-0.058, step 0.013-0.014 |
| P1 Sim3 0.12-0.30 in 3/3 | miss 0/3 (0.314, 0.630, 0.390) |
| P2 SE3 0.26-0.40 in 3/3, scale 0.965-0.985 | miss: 2/3 SE3 (0.658 out); scale 0.990 in r3 (BASE r3 0.988) |
| P3 chord SD <= 0.022 | miss (0.034-0.040; BASE 0.031-0.037) |
| K1 M1 holds, Sim3 > 0.30 in 2/3 | fires (3/3) |
| K3 any R_01 run > 0.60 m SE3 | fires: S1 r2 0.658 (its paired BASE r2 is 0.601, also above) |

K3 fired, so R_03 and R_11 were not run; the knob stays off (default true). The mechanism moved
(boundary residual 8 -> 1.3 sigma, gyro wander halved) but the score did not: S1 minus BASE per
pair is SE3 -0.121, +0.057, +0.046 m. Both members of pair r2 sit above the known 0.30-0.50 band
at load 7.2, so the K3 trip is within run-to-run noise as far as n = 3 can tell; it is recorded as
fired, not explained away. Accel wander (ba SD, step) barely moved: the zero-mean prior (stage B)
and the visual side remain the suspects for accel, as the W6 diagnosis above showed in the fixture.

## Diagnosis: why card S missed (2026-11-21, read-only, existing runs)

Scripts: `$S/diagS/` (dec.py, along.py, imufit.py, biascorr.py, vel.py, imuedge.py, toff.py, clockpred.py, diverge.py). Inputs: the 6 card S runs, the F1 runs, ORB-SLAM3 mono-inertial r01-mono-r{1,2}.

- **The R_01 error is along-track scale, not heading.** Accumulated along-track 0.55-0.79 m, cross-track 0.06-0.19 m, vertical 0.07-0.14 m (ORB-SLAM3: 0.05-0.15 / 0.06-0.11 / 0.02 m). Yaw SD 0.35-0.65 deg, unchanged by card S.
- **It is deterministic, not async noise.** 1 s along-track errors correlate 0.53-0.78 between any two of our runs (BASE and S1 alike), -0.10 to +0.02 with ORB-SLAM3. The shared part integrates to 0.58 m RMS; the run-specific part is 0.09-0.42 m. Same shape every run: speed 3-8 % low at 60-85 s, 3-8 % high at 95-135 s.
- **The VI-BA velocity follows the wrong scale** (|v| / GT speed 0.92-1.08), so the IMU is not holding the scale.
- **The accel-bias wander is tilt compensation, not a free walk.** The true bias is constant (GT fit (0.234, 0.203, 0.361) m/s^2, moves < 0.01 between 10 s windows; no accelerometer scale error visible). The wander sits on the horizontal axes and correlates -0.46 to -0.78 with the bias the tilt error predicts. Tying the walk cannot remove it; the zero-mean prior does not bias the mean.
- **Suspected cause (hypothesis, not run-tested): camera-IMU clock offset not applied, plus Huber on every interior IMU edge.** Gyro against GT rate: 0.048 rad/s RMS at offset 0, 0.0077 at +5 ms; our runs log `offset 0 ns`. Interior IMU edge chi2 p90 865-1150, 34-41 % of edges Huber-downweighted (p10 weight 0.12-0.14); |r_R| correlates 0.87-0.88 with the clock-predicted error |w(t1)-w(t0)| x 5 ms. The downweighted edges are the high-dynamics ones carrying the most scale information. ORB-SLAM3 LocalInertialBA puts no robust kernel on interior inertial edges. This may also explain the ledger's "more IMU weight makes it worse" (cards O2, Q0b removed Huber with the clock still wrong).
- **Run-to-run spread** starts at the first IMU init (frame 45-61, keyframe lists split by one frame from async timing) and is amplified by the weakly observable along-track scale.
- **Not like for like with ORB-SLAM3:** it closes a loop per run and writes its refined trajectory; we close none and write online poses. ORB-SLAM3's local VIO still holds scale before its loop.

What the theory got wrong: it assumed heading and a free bias walk; the error is scale, the bias wander is tilt-coupled, and the IMU model (clock, kernel layout) was not checked.

Next card (proposed): clock offset first (`imu_time_offset_ns`, sign to verify; prediction: interior chi2 p90 under 50, downweighted share under 5 %), then ORB-SLAM3's kernel layout (`imu_huber` boundary-only; prediction: 1 s along SD 55-65 -> <= 35 mm, Sim3 < 0.30 m in 3/3). Kill: chi2 clean but along SD > 50 mm means the scale is set by the visual side (next: keyframe span about 4 s per 10 keyframes, loop closure).
