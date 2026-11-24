# Card AA: camera-IMU clock offset, then ORB-SLAM3's inertial kernel layout

Status: 2026-11-22. Stages 0-2 run (F1 binary, config only). Stage 1: mechanism confirmed (M1-M3 on R_01/R_03),
score unchanged, K1 fired. Stage 2: the kernel layout is score-neutral on R_01/R_03, costs 0.01-0.03 m on R_11 (P8).
Knobs stay at their defaults. Stage 3 (code) waits for the owner's verdict. Plan: planner (Fable), owner answers 2026-11-22
(owner-local plan `pr180-clock-card.md`, not in the repo). Card letter AA (Y is reserved for the window-local frame card).

Paths: `P=/home/edgar/dev/split/.pr180-fix`, `AA=$P/cardAA`, `S` = the session scratchpad,
`D=/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data`, `B=/home/edgar/dev/.flux-connect-scratch/bench`.

## Experiment: AA, camera-IMU clock offset, then ORB-SLAM3's inertial kernel layout
Gap tested: G35, A8 (G11), G18     Base: kornia-slam@81bc775 (pr180c/F1), docs at 736f7b2
Binary tag: stages 1-2 = $AA/bin/kornia-slam-f1 (md5 52e38c464d4369df0c9d3f61df17b5aa); stage 3 = kornia-slam-aa (md5 at build)
Mechanism: IMU stamps are 4.2-5.3 ms late against the camera stamps and nothing corrects them
  (datasets/lamaria.rs:127-131 adds imu_time_offset_ns, default 0, source/lamaria.rs:51-53; the factory
  record's TimeOffsetSec_Device_Gyro/Accel are not read, datasets/lamaria.rs:369-385). Hold integration
  (kornia-sensors/src/imu.rs:184-221) hides 0.5 ms of it, so the solver sees 3.7-4.5 ms. An edge's
  rotation error is (w(t1)-w(t0)) x delta: 1.5-5 mrad against sigma 0.08 mrad. Huber on every edge
  (vi_ba_schur.rs:878-887, default :386, knee 16.9 :381) then cuts 34-41 % of interior edges to weight
  0.12-0.14, and those are the high-dynamics edges that carry scale.
Theory: (a) with the stamps corrected the error term vanishes, the edges return to full weight and the
  IMU holds scale. (b) Without a kernel the same error telescopes along the chain (each keyframe shifted
  by w x delta, a consistent time-shifted trajectory), so a quadratic chain tolerates it; ORB-SLAM3 runs
  that way on this data, uncorrected (bench/orbslam3/lamaria_vio.cc:39). ORB-SLAM3 Optimizer.cc:2632-2643:
  Huber sqrt(16.92) only on the boundary edge or while bRecInit, boundary information x1e-2. Basalt
  imu_block.hpp:57-58 and VINS-Fusion estimator.cpp:1059-1060 put no kernel on any IMU factor; they pair
  that with a marginal prior. Our boundary is ORB-SLAM3's fixed keyframe with a scaled edge, so its
  boundary kernel is the one that fits. Clock: VINS-Fusion estimates td online from a config seed
  (estimator.cpp:1042, :1084-1086); OpenVINS seeds calib_camimu_dt from the calibration file
  (VioManagerOptions.h:208, :236). Both seed, then estimate.
Knob: stage 1: source Lamaria.imu_time_offset_ns (exists; default 0 = today) -> the per-sequence value
        from clockscan.py under Hold. Class: sensor-model constant; ships only once read from the device
        record (stage 3), never hand-set.
      stage 2: mapping.local_ba.imu_huber (exists; default AllEdges = today) -> BoundaryOrRecentInit.
        Class: reference constant (ORB-SLAM3).
      Not changed: huber_imu_chi2 16.9 (reference constant), inertial.sample_interpolation Hold (its
        0.5 ms lag is a bug-fix class item; compensated in the offset here, fixed in stage 3).
Arms: BASE (today); C (offset); CK (C + kernel); K0 (kernel at offset 0, control); TD0 (online td at
  offset 0, unscored diagnostic).
Baselines (F1 binary, paired BASE re-run every round):
  R_01 full, card S BASE r1-r3 ($P/cardS/out/r01full/r{1,2,3}/r01full-BASE-r1): SE3 0.488/0.601/0.353,
    Sim3 0.434/0.573/0.340, 1 s along SD 55-65 mm, accumulated along 0.55-0.79 m, chord-100 SD
    0.032-0.041, interior chi2 p50 8.6-12, p90 865-1150, downweighted 34-41 %, slope 2.35-2.40 ms,
    along-error correlation between runs 0.53-0.78. ORB-SLAM3: along SD 20-23 mm, chord SD 0.009-0.013.
  R_03 ($S/F1/out/r03-V1-r{1,2,3}): SE3 0.104-0.117 m, tracked 758-778. Interior chi2 p90 590-895,
    downweighted 23-27 %, slope 0.99-1.33 ms (imuedges.py, stage 0).
  R_11 ($S/F1/out/r11-V1-r{1,2,3}): SE3 0.157-0.178 m, tracked 792-800. Interior chi2 p90 1737-2791,
    downweighted 37-41 %, slope 1.32-1.44 ms (imuedges.py, stage 0).
Predictions, stage 1 (C against paired BASE, 3 rounds):
  M1 interior IMU chi2 p90: R_01 865-1150 -> <= 100; R_03 and R_11 -> <= 150.   measured by: imuedges.py
  M2 interior edges with Huber weight < 1: 34-41 % -> <= 20 % on all three.      measured by: imuedges.py
  M3 slope of |r_R| on |w(t1)-w(t0)|, R_01: 2.35-2.40 ms -> <= 0.6 ms.           measured by: imuedges.py
  P1 R_01 1 s along-track SD: 55-65 mm -> <= 40 mm in 3/3.                       measured by: alongtrack.py
  P2 R_01 chord-100 SD: 0.032-0.041 -> <= 0.022 in 3/3; accumulated along 0.55-0.79 -> <= 0.35 m in 3/3.
  P3 R_01 Sim3: C minus paired BASE <= -0.10 m in >= 2/3 and C <= 0.30 m in >= 2/3; scale 0.98-1.02.
  P4 R_01 correlation of the 1 s along-error series, C runs against BASE runs: 0.53-0.78 -> <= 0.30.
  P5 guards: R_03 SE3 <= 0.12 m and tracked >= 758 in 3/3; R_11 SE3 <= 0.18 m and tracked >= 792 in 3/3.
  D1 (TD0, unscored): median td_ms over ungated solves after frame 300 within 0.7 ms of the Hold offset
     (R_01 -3.94, R_03 -3.66). Card G's hold error is about 1 sigma at 4 ms, so this is a sign and size
     check only.
Predictions, stage 2 (CK against paired C; K0 against paired BASE; 3 rounds):
  M4 `local BA:` line reads `IMU Huber BoundaryOrRecentInit`; interior edges with weight < 1 after
     frame 300: 0.
  M5 CK interior chi2 p90 <= 1.3 x C's; vi_ba_solves_3plus_rejects <= 5 % of solves; VIBA0/1/2 accepted.
  P6 R_01 CK: along SD within 5 mm of C's median and <= 40 mm in 3/3; Sim3 CK minus C within +/-0.05 m
     (median). Expected: no visible change. After stage 1 at most 20 % of interior edges are touched,
     at weight >= 0.5, so the pull changes by at most 2x on few edges. Cards O2 and Q0b removed a kernel
     that held a third of the edges at weight 0.13 (a 7.7x pull toward the time-shifted trajectory), on
     the window solve that cards R and T later fixed; O2 also used boundary scale 1.0.
  P7 R_01 K0: along SD <= 45 mm in >= 2/3 and Sim3 K0 minus BASE <= -0.05 m in >= 2/3. This is the
     reference's own configuration on its own input.
  P8 guards for CK as P5.
Kill rules:
  K1 (mechanism moved, score did not) M1-M3 hold on R_01 and along SD > 50 mm in 2/3: the IMU model is
     clean and the scale is still not held. Do not credit the clock. Diagnose: see "If it misses" 2.
  K2 (score moved, mechanism did not) P1 or P3 holds with p90 > 300 and share > 30 %: luck, load, or the
     offset acting through tracking (gyro_guess, klt_prediction), not VI-BA. Do not credit; diagnose 3.
  K3 (harm) a C or CK run on R_01 above 0.60 m SE3 while its pair is below, tracked < 2898, a VIBA
     accepted in the pair and rejected here, vi_ba_solves_3plus_rejects > 5 %, or a guard missed in 2/3:
     knob stays at its default. Diagnose 4.
  K4 (sign or wiring) after the first C run: `Dataset:` line does not show the offset, or M3 slope
     > 3.0 ms. Stop all runs; re-check with clockscan.py --apply-ns.
  K5 (stage 0) clockscan thirds differ by > 0.5 ms on R_01 or R_03, or d_mid minus the factory gyro
     offset is outside 0 to +1.5 ms: the scan or the sign reading is wrong. Stop before any run.
  K6 (stage 2, mechanism moved, score did not) CK inside C's band (the predicted outcome): the kernel
     adds no score on a clean clock. Do not credit a gain; the decision rests on the reference rule and K0.
  K7 (stage 2, score moved, mechanism did not) CK beats C by > 10 mm along SD in 3/3 while C's
     downweighted share was already <= 5 %: not the kernel. 3 more pairs; check the recent-init span.
  K8 (K0 harm) K0 trips K3's conditions: a quadratic chain does not tolerate the clock error on this
     base. Record it; the clock fix must precede the kernel; judge stage 2 on CK against C only.
Bit-identity: stages 1-2 change no code; the binary md5 is the proof. Stage 3 binary, knobs off:
  $P/baseline/r01full-S10-7.ron, RAYON_NUM_THREADS=1, traj md5 d4f77b567b912b5a4caa2af6085a35b1
  (cmp with $P/baseline/traj.txt); r_the_default_solve_keeps_its_bits unchanged.
Noise: async runs differ run to run; 3 rounds, each variant started together with its base (JOBS=2).
  Known R_01 band 0.30-0.50 m (n = 12). Load every 30 s; a round whose load p50 is more than 2 from the
  other rounds' is repeated.

**If it misses:**
1. *M1-M3 miss.* Check the `Dataset:` line and the slope (doubled means wrong sign). Then split the tail by row
   with imuedges.py (rotation, velocity, position in sigma). Rotation tail that follows |w(t1)-w(t0)| means a
   residual offset: re-scan. Rotation tail that follows |w| itself means gyro scale or shear on the raw IMU of
   R_01/R_03: re-run C with the extracted `factory_calib`. Velocity tail means the 1.0-1.5 ms gyro-accel split,
   which needs stage 3b.
2. *K1.* Run `$AA/diagS/vel.py` logic on the C runs (VI-BA |v| over GT speed, was 0.92-1.08), P4's correlation,
   and `imufit.py` (accel-bias wander against tilt, was -0.46 to -0.78). Shared shape still there and |v| still
   off means the visual side sets the scale: next card is window time span (10 keyframes cover 2 s here) and
   loop closure, as the S card named.
3. *K2.* Compare `[summary] flow_coast` and `[predgap] rot=` between BASE and C; 3 more pairs.
4. *K3 or K8.* List edges with chi2 > 100 (imuedges.py `--tail`) against dt, |w(t1)-w(t0)|, give-ups and
   `flow_refused` frames, to see what the kernel was absorbing.
5. *R_11 guard fails alone.* Its scan is the noisiest. Re-run with the factory value under Hold, -3646000 ns,
   and read TD0.

## Stage 0 (2026-11-22, no SLAM run)

### Factory records (step 2)

Heads of `$D/R_01_easy/raw_data/R_01_easy.vrs` and `R_03_easy.vrs` (4 MiB, deleted after extraction) through
`$B/lamaria/factory/extract_factory_calib.py`, into `$AA/factory/{R_01_easy,R_03_easy}.factory.json`. Not added
to the R_01/R_03 configs (runs stay on the raw IMU).

| Sequence | Serial | imu-right gyro offset ms | imu-right accel offset ms | imu-left gyro / accel ms |
|---|---|---|---|---|
| R_01_easy | 1WM093701G1276 | 4.146 | 3.121 | 1.865 / 1.371 |
| R_03_easy | 1WM093701G1276 | 4.146 | 3.121 | 1.865 / 1.371 |

R_01 and R_03 are the same device as R_11 and sequences 2-5 (1WM093701G1276); sequence_1_19 is 1WM09370081264
(gyro 4.083, accel 2.626).

### Clock scan (`clockscan.py`, step 4)

`python3 -I experiments/quick-eval/clockscan.py $D/<s>/asl/<s> $D/<s>/ground_truth/pseudo_dense/<s>.txt --factory <F>`,
max-dt 0.31 s, 0.25 ms steps, parabola fit; thirds of the sequence in brackets.

| Sequence | pairs (dt p50) | d_mid ms (thirds) | d_hold ms (thirds) | RMS at min / at 0, mrad | imu_time_offset_ns | factory gyro ms | d_mid - factory |
|---|---|---|---|---|---|---|---|
| R_01_easy | 2797 (50 ms) | +4.43 (4.35 / 4.44 / 4.52) | +3.93 (3.85 / 3.94 / 4.02) | 0.19 / 2.17 | -3928097 | 4.146 | +0.28 |
| R_03_easy | 3065 (50 ms) | +4.16 (4.14 / 4.16 / 4.17) | +3.66 (3.64 / 3.67 / 3.67) | 0.17 / 2.20 | -3659335 | 4.146 | +0.01 |
| R_11_5cp | 1137 (150 ms) | +5.04 (4.72 / 5.24 / 5.07) | +4.54 (4.22 / 4.74 / 4.57) | 2.12 / 6.47 | -4542256 | 4.146 | +0.90 |
| sequence_1_19 | 1996 (150 ms) | +4.59 (4.62 / 4.63 / 4.49) | +4.09 | 1.58 / 5.20 | -4088677 | 4.083 | +0.50 |
| sequence_2_11 | 2817 (150 ms) | +5.22 (4.82 / 5.10 / 5.55) | +4.72 | 1.68 / 6.29 | -4724777 | 4.146 | +1.08 |
| sequence_3_18 | 3832 (150 ms) | +5.49 (5.17 / 5.85 / 5.38) | +4.99 | 1.64 / 5.93 | -4992815 | 4.146 | +1.35 |
| sequence_4_11 | 2620 (150 ms) | +4.47 (4.46 / 4.51 / 4.45) | +3.97 | 2.01 / 4.81 | -3974126 | 4.146 | +0.33 |
| sequence_5_11 | no pseudo-dense GT | - | - | - | factory only | 4.146 | - |

Self-check: `--apply-ns -3940000` on R_01 scans to d_hold -0.01 ms (thirds -0.09 / 0.00 / 0.08), RMS 0.18 mrad.

Tool pass against the plan's table (0.1 ms on R_01/R_03, 0.3 ms elsewhere): R_01, R_03, R_11, 1_19, 3_18, 4_11
pass; **sequence_2_11 misses** (5.22 against 4.88, 0.34 ms). It stays at 5.16-5.26 for max-dt 0.16-0.51 s and
its thirds drift 4.82 -> 5.55, so it reads a within-sequence drift, as 3_18 does. No stage 1 run uses it.

**K5: not fired.** Thirds spread 0.17 ms on R_01 and 0.03 ms on R_03 (limit 0.5); d_mid minus the factory gyro
offset is +0.01 to +1.35 ms on every sequence (limit 0 to +1.5).

Stage 1 offsets (`imu_time_offset_ns`, rounded to 10 us as the plan expected): R_01 full -3930000, R_03 -3660000,
R_11 -4540000.

### Tools (step 3)

- `imuedges.py` on card S BASE r1: n 4851, chi2 p50 8.6, p90 885, share 0.340, hw p10 0.14, |r_R| p90 2.55 mrad,
  slope 2.35 ms; S1 r2 (`r2/r01full-S1-r1`): p90 1151, share 0.399, slope 2.40 ms. Pass.
- `alongtrack.py` on the six card S runs and the two ORB-SLAM3 runs equals `dec.py 15` and `along.py` to the
  printed digits (SE3, Sim3, scale, accumulated along, 1 s along mean and SD, chord-100, correlation matrix). Pass.

### Baseline mechanism counters (imuedges.py, skip 300 frames)

| Run | n | chi2 p50 | chi2 p90 | share hw<1 | hw p10 | abs r_R p90 mrad | slope ms |
|---|---|---|---|---|---|---|---|
| R_01 card S BASE r1 | 4851 | 8.6 | 885 | 0.340 | 0.14 | 2.55 | 2.35 |
| R_01 card S BASE r2 | 4833 | 9.9 | 865 | 0.365 | 0.14 | 2.69 | 2.25 |
| R_01 card S BASE r3 | 4770 | 9.0 | 880 | 0.351 | 0.14 | 2.59 | 2.26 |
| R_03 F1 V1 r1 | 1323 | 8.4 | 895 | 0.266 | 0.14 | 2.05 | 1.33 |
| R_03 F1 V1 r2 | 1332 | 7.5 | 590 | 0.234 | 0.17 | 1.68 | 0.99 |
| R_03 F1 V1 r3 | 1278 | 7.6 | 770 | 0.261 | 0.15 | 2.17 | 1.22 |
| R_11 F1 V1 r1 | 963 | 10.4 | 1737 | 0.372 | 0.10 | 3.43 | 1.32 |
| R_11 F1 V1 r2 | 918 | 11.4 | 2636 | 0.411 | 0.08 | 3.97 | 1.44 |
| R_11 F1 V1 r3 | 936 | 10.3 | 2791 | 0.366 | 0.08 | 4.09 | 1.42 |

R_01 interior tail (chi2 > 100, 24.5 % of edges, BASE r1): r_R 13.5 sigma against 1.3 in the rest, |dw| 0.98
against 0.48 rad/s, corr(|r_R|, |dw|) 0.87 against corr(|r_R|, |w|) 0.55: the tail follows the gyro change,
the clock signature. The R_03/R_11 slopes (1.0-1.4 ms) sit well below their 3.7-4.5 ms offsets; not explained
yet (M3 is scored on R_01 only).

## Results, stage 1 (2026-11-22, F1 binary md5 52e38c46, paired rounds)

Runs in `$AA/out/{r01full,r03,r11}-BASE-C/r<k>/`, scores in `$AA/out/score-*.txt` (`$AA/score.sh`). Every C run's
`Dataset:` line reads `offset <value> ns` with the config value; every run exits 0, VIBA0/1/2 accepted in all,
vi_ba_failed 0. Load p50 (1-min average, every 30 s): R_01 r1-r4 3.75 / 5.86 / 7.22 / 8.14; R_03 4.35 / 5.63 /
5.71; R_11 6.71 / 8.61 / 9.76; TD0 5.60. Each set ran back to back, so the 1-min average ramps from idle over the
first round (2.5 min runs, 5 samples). R_01 r1 sat more than 2 below the others and was repeated as r4 (which
started from a loaded machine, 8.14). Verdicts below hold on r1-r3, on r2-r4 and on all four.

R_01 full, C at -3930000 ns (SE3/tracked from ate.py and `[summary]`; the rest from alongtrack.py, Sim3 from 15 s):

| Round (load) | Arm | SE3 m | Sim3 m | scale | 1 s along SD mm | accum along m | chord-100 SD | chi2 p50 / p90 | share hw<1 | slope ms | 3plus rejects / solves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 (3.75) | BASE | 0.329 | 0.267 | 0.9785 | 55.4 | 0.387 | 0.028 | 9.4 / 749 | 0.356 | 2.25 | 1 / 589 |
| r1 (3.75) | C | 0.570 | 0.530 | 0.9785 | 60.8 | 0.558 | 0.038 | 1.8 / 8 | 0.038 | 0.21 | 0 / 592 |
| r2 (5.86) | BASE | 0.381 | 0.293 | 0.9720 | 51.8 | 0.301 | 0.029 | 9.5 / 824 | 0.356 | 2.25 | 1 / 588 |
| r2 (5.86) | C | 0.302 | 0.246 | 0.9769 | 51.7 | 0.536 | 0.030 | 1.6 / 8 | 0.048 | 0.25 | 1 / 583 |
| r3 (7.22) | BASE | 0.420 | 0.414 | 0.9923 | 62.6 | 0.505 | 0.037 | 9.1 / 877 | 0.357 | 2.31 | 0 / 589 |
| r3 (7.22) | C | 0.506 | 0.436 | 0.9745 | 51.8 | 0.418 | 0.027 | 1.7 / 8 | 0.039 | 0.20 | 1 / 586 |
| r4 (8.14) | BASE | 0.557 | 0.531 | 0.9815 | 54.6 | 0.503 | 0.033 | 10.2 / 921 | 0.381 | 2.35 | 0 / 591 |
| r4 (8.14) | C | 0.446 | 0.423 | 0.9865 | 55.4 | 0.361 | 0.027 | 1.7 / 8 | 0.034 | 0.21 | 0 / 584 |

Tracked 2898 in all eight. |r_R| p90 2.45-2.73 mrad (BASE) -> 0.25-0.27 (C); hw p10 0.14 -> 1.00.
Correlation of the 1 s along-track series, C runs against BASE runs (all 16 cross pairs): 0.45-0.82; same-round
pairs 0.80 / 0.45 / 0.68 / 0.76; BASE against BASE 0.54-0.78.

R_03 (C at -3660000) and R_11 (C at -4540000), 800 frames:

| Seq | Round (load) | Arm | SE3 m | Sim3 m | tracked | chi2 p50 / p90 | share hw<1 | slope ms |
|---|---|---|---|---|---|---|---|---|
| R_03 | r1 (4.35) | BASE | 0.100 | 0.069 | 777 | 8.7 / 1280 | 0.286 | 1.30 |
| R_03 | r1 (4.35) | C | 0.108 | 0.070 | 773 | 2.0 / 7 | 0.009 | 0.11 |
| R_03 | r2 (5.63) | BASE | 0.099 | 0.067 | 781 | 8.2 / 1171 | 0.287 | 1.30 |
| R_03 | r2 (5.63) | C | 0.114 | 0.069 | 773 | 1.8 / 6 | 0.015 | 0.11 |
| R_03 | r3 (5.71) | BASE | 0.110 | 0.074 | 763 | 7.0 / 613 | 0.216 | 1.21 |
| R_03 | r3 (5.71) | C | 0.109 | 0.067 | 763 | 1.7 / 7 | 0.013 | 0.12 |
| R_11 | r1 (6.71) | BASE | 0.197 | 0.112 | 796 | 10.9 / 3020 | 0.415 | 1.50 |
| R_11 | r1 (6.71) | C | 0.155 | 0.112 | 799 | 7.2 / 477 | 0.277 | 0.45 |
| R_11 | r2 (8.61) | BASE | 0.187 | 0.112 | 791 | 9.9 / 2188 | 0.378 | 1.39 |
| R_11 | r2 (8.61) | C | 0.156 | 0.111 | 800 | 7.3 / 524 | 0.317 | 0.59 |
| R_11 | r3 (9.76) | BASE | 0.153 | 0.112 | 798 | 9.5 / 1738 | 0.390 | 1.37 |
| R_11 | r3 (9.76) | C | 0.155 | 0.112 | 797 | 7.9 / 748 | 0.324 | 0.65 |

TD0 (online td, sigma 2 ms, offset 0; one run each, `$AA/out/td0`): `[online_calib] ... td_ms=` median over ungated
solves at frame >= 300: R_01 -4.49 ms (p10 -5.02, p90 -4.01, n 528; SE3 0.351 m), R_03 -3.99 (p10 -4.69, p90 -3.56,
n 144; SE3 0.118 m), R_11 -4.24 (p10 -5.22, p90 -3.42, n 103; SE3 0.190 m).

| P/K | prediction | baseline | measured (each sample) | verdict |
|---|---|---|---|---|
| M1 | interior chi2 p90 R_01 <= 100; R_03, R_11 <= 150 | R_01 749-921, R_03 613-1280, R_11 1738-3020 | R_01 8/8/8/8; R_03 7/6/7; R_11 477/524/748 | as predicted on R_01 and R_03; missed on R_11 |
| M2 | share hw<1 <= 20 % on all three | 0.356-0.381 / 0.216-0.287 / 0.378-0.415 | R_01 0.038/0.048/0.039/0.034; R_03 0.009/0.015/0.013; R_11 0.277/0.317/0.324 | as predicted on R_01 and R_03; missed on R_11 |
| M3 | R_01 slope <= 0.6 ms | 2.25-2.35 | 0.21/0.25/0.20/0.21 | as predicted |
| P1 | R_01 1 s along SD <= 40 mm in 3/3 | 51.8-62.6 | 60.8/51.7/51.8/55.4 | missed |
| P2 | R_01 chord-100 SD <= 0.022 in 3/3; accum along <= 0.35 m in 3/3 | 0.028-0.037; 0.30-0.51 | 0.038/0.030/0.027/0.027; 0.558/0.536/0.418/0.361 | missed |
| P3 | R_01 Sim3 C - BASE <= -0.10 in >= 2/3, C <= 0.30 in >= 2/3, scale 0.98-1.02 | Sim3 0.267-0.531 | C - BASE +0.263/-0.047/+0.022/-0.108; C 0.530/0.246/0.436/0.423; scale 0.979/0.977/0.975/0.987 | missed |
| P4 | C-vs-BASE along correlation <= 0.30 | 0.53-0.78 | 0.45-0.82 (same-round 0.80/0.45/0.68/0.76) | missed |
| P5 | R_03 SE3 <= 0.12, tracked >= 758 in 3/3; R_11 SE3 <= 0.18, tracked >= 792 in 3/3 | | R_03 0.108/0.114/0.109, 773/773/763; R_11 0.155/0.156/0.155, 799/800/797 | as predicted |
| D1 | TD0 median td within 0.7 ms of R_01 -3.94, R_03 -3.66 (unscored) | | R_01 -4.49 (0.55 off), R_03 -3.99 (0.33 off); R_11 -4.24 | as predicted (sign and size) |
| K1 | M1-M3 hold on R_01 and along SD > 50 mm in 2/3 | | M1-M3 hold; along SD > 50 in 4/4 | kill fired |
| K2 | P1 or P3 holds with p90 > 300, share > 30 % | | P1 and P3 missed | not fired |
| K3 | C on R_01 > 0.60 SE3 with pair below, tracked < 2898, VIBA reject, 3plus > 5 %, guard missed 2/3 | | max C SE3 0.570; 2898 x4; all accepted; <= 1 / 583; guards pass | not fired |
| K4 | `Dataset:` lacks the offset or slope > 3.0 ms after the first C run | | offset shown; slope 0.21 | not fired |

### Diagnosis, K1 ("If it misses" 2)

`$AA/diagS/diagk1.py` (card S `vel.py` and `biascorr.py` logic on any run dir) on the eight R_01 runs:

| Run | VI-BA abs v / GT speed mean (sd, p5-p95) | accel-bias error corr with tilt-predicted, y / z |
|---|---|---|
| r1 BASE | 1.010 (0.062, 0.930-1.077) | -0.45 / -0.70 |
| r1 C | 1.011 (0.073, 0.924-1.084) | -0.51 / -0.73 |
| r2 BASE | 1.021 (0.052, 0.945-1.086) | -0.58 / -0.78 |
| r2 C | 1.011 (0.071, 0.890-1.077) | -0.63 / -0.70 |
| r3 BASE | 0.999 (0.075, 0.866-1.080) | -0.60 / -0.71 |
| r3 C | 1.008 (0.063, 0.921-1.071) | -0.55 / -0.74 |
| r4 BASE | 1.015 (0.064, 0.922-1.082) | -0.52 / -0.78 |
| r4 C | 1.004 (0.065, 0.923-1.063) | -0.41 / -0.67 |

The per-10 s |v|/GT profile is the same shape in C as in BASE (0.92-0.94 at 65-75 s, 1.04-1.07 at 105-135 s), the
along-track series still correlates 0.45-0.82 with BASE, and the accel-bias wander is still tilt-coupled (-0.41 to
-0.78, card S -0.46 to -0.78). With the clock corrected the IMU edges are at sensor strength (chi2 p50 1.6-1.8,
p90 8, 3-5 % downweighted), yet the VI-BA velocity follows the same wrong scale. So the clock and the all-edge
Huber are not what lets the scale wander on R_01: the visual side sets it. Next card, as the S card named: window
time span (10 keyframes cover 2 s here) and loop closure. The clock correction is real (TD0 finds the same offset
by itself, D1) and costs nothing on the guards, but it is not credited with a score gain.

### Diagnosis, M1/M2 miss on R_11 ("If it misses" 1)

`Dataset:` lines show -4540000 ns; the slope fell 1.37-1.50 -> 0.45-0.65 ms (not doubled, so the sign is right).
`imuedges.py --tail` on the three C runs: 18-22 % of edges above chi2 100 (BASE r1 32 %); tail r_R 11.5-12.8 sigma
(rest 1.1), r_v 6.0-7.1 sigma (rest 0.5), r_p 2.6-3.1 sigma; tail dt 0.3-0.4 s against 0.15 s; corr(|r_R|, |dw|)
0.09-0.30 and corr(|r_R|, |w|) 0.07-0.11 (BASE 0.59 / 0.26); corr(log chi2, dt) 0.42-0.44. The rotation tail
follows neither the gyro change nor the rate, so it is neither a residual offset nor gyro scale (R_11 already runs
the rectified IMU), and `clockscan.py --apply-ns -4540000` on R_11 scans to d_hold 0.00 (thirds -0.32 / 0.20 /
0.03; RMS 2.11 mrad at the minimum against 0.19 on R_01, so the sparse GT limits the scan). The tail sits on long
edges and carries velocity rows at 6-7 sigma: the gyro-accel split (factory 4.146 vs 3.121 ms, about 1.0 ms) that
one offset cannot align is the remaining named cause; it needs stage 3b. Not tested further here.

## Results, stage 2 (2026-11-22, same binary, paired rounds)

Runs in `$AA/out/{r01full-C-CK,r01full-BASE-K0,r03-C-CK,r11-C-CK}/r<k>/`. Every CK and K0 `local BA:` line reads
`IMU Huber BoundaryOrRecentInit`; C and BASE read `AllEdges`. All runs exit 0, VIBA0/1/2 accepted in all, vi_ba_failed
0, vi_ba_solves_3plus_rejects 0-1 per run (<= 0.2 % of solves). Load p50: R_01 C/CK r1-r3 4.74 / 5.67 / 6.60, r4-r6
(diagnosis) 9.41 / 9.84 / 6.01; R_01 BASE/K0 8.15 / 7.87 / 7.41; R_03 8.65 / 8.05 / 6.69; R_11 6.32 / 6.70 / 7.52.
R_01 C/CK r4 and r5 sit more than 2 above r1-r3; they are diagnosis rounds and are reported, not scored.

R_01 full (SE3 from ate.py; Sim3, scale, along, chord from alongtrack.py):

| Round (load) | Arm | SE3 m | Sim3 m | scale | 1 s along SD mm | accum along m | chord-100 SD | chi2 p50 / p90 | share hw<1 | slope ms |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 (4.74) | C | 0.382 | 0.324 | 0.9765 | 50.8 | 0.323 | 0.025 | 1.6 / 7 | 0.031 | 0.21 |
| r1 (4.74) | CK | 0.415 | 0.322 | 0.9684 | 53.3 | 0.384 | 0.028 | 1.7 / 8 | 0.000 | 0.15 |
| r2 (5.67) | C | 0.578 | 0.515 | 0.9754 | 60.9 | 0.531 | 0.034 | 1.7 / 8 | 0.037 | 0.21 |
| r2 (5.67) | CK | 0.295 | 0.273 | 0.9848 | 53.4 | 0.444 | 0.029 | 1.7 / 9 | 0.000 | 0.16 |
| r3 (6.60) | C | 0.484 | 0.413 | 0.9712 | 56.1 | 0.562 | 0.033 | 1.6 / 8 | 0.040 | 0.20 |
| r3 (6.60) | CK | 0.265 | 0.180 | 0.9774 | 45.2 | 0.286 | 0.021 | 1.7 / 8 | 0.000 | 0.15 |
| r4 (9.41) diag | C | 0.336 | 0.251 | 0.9737 | 54.7 | 0.442 | 0.029 | 1.6 / 8 | 0.040 | 0.23 |
| r4 (9.41) diag | CK | 0.452 | 0.385 | 0.9761 | 52.6 | 0.342 | 0.028 | 1.6 / 8 | 0.000 | 0.15 |
| r5 (9.84) diag | C | 0.376 | 0.361 | 0.9890 | 55.3 | 0.471 | 0.031 | 1.6 / 8 | 0.041 | 0.21 |
| r5 (9.84) diag | CK | 0.500 | 0.367 | 0.9625 | 55.1 | 0.394 | 0.030 | 1.8 / 8 | 0.000 | 0.15 |
| r6 (6.01) diag | C | 0.355 | 0.225 | 0.9688 | 54.8 | 0.407 | 0.028 | 1.7 / 7 | 0.038 | 0.24 |
| r6 (6.01) diag | CK | 0.302 | 0.226 | 0.9758 | 50.0 | 0.164 | 0.025 | 1.7 / 8 | 0.000 | 0.15 |
| r1 (8.15) | BASE | 0.350 | 0.270 | 0.9724 | 54.4 | 0.411 | 0.029 | 10.7 / 843 | 0.386 | 2.22 |
| r1 (8.15) | K0 | 0.334 | 0.337 | 0.9909 | 61.3 | 0.590 | 0.038 | 10.6 / 84 | 0.000 | 0.54 |
| r2 (7.87) | BASE | 0.434 | 0.431 | 0.9910 | 66.7 | 0.749 | 0.043 | 9.0 / 928 | 0.338 | 2.30 |
| r2 (7.87) | K0 | 0.359 | 0.334 | 0.9803 | 57.4 | 0.613 | 0.034 | 11.0 / 80 | 0.000 | 0.53 |
| r3 (7.41) | BASE | 0.345 | 0.284 | 0.9778 | 54.1 | 0.371 | 0.027 | 9.9 / 796 | 0.369 | 2.41 |
| r3 (7.41) | K0 | 0.298 | 0.311 | 1.0021 | 65.0 | 0.680 | 0.038 | 10.9 / 72 | 0.000 | 0.51 |

Tracked 2898 in all. Along-track correlation between any two C/CK runs 0.51-0.81; BASE/K0 0.65-0.85.

R_03 and R_11, C against CK:

| Seq | Round (load) | C SE3 / tracked | CK SE3 / tracked | C chi2 p90 / share | CK chi2 p90 / share |
|---|---|---|---|---|---|
| R_03 | r1 (8.65) | 0.118 / 787 | 0.105 / 767 | 7 / 0.016 | 7 / 0.003 |
| R_03 | r2 (8.05) | 0.102 / 782 | 0.109 / 777 | 8 / 0.023 | 7 / 0.002 |
| R_03 | r3 (6.69) | 0.113 / 784 | 0.120 / 790 | 7 / 0.025 | 7 / 0.004 |
| R_11 | r1 (6.32) | 0.159 / 800 | 0.175 / 798 | 536 / 0.306 | 56 / 0.009 |
| R_11 | r2 (6.70) | 0.164 / 800 | 0.167 / 800 | 832 / 0.342 | 47 / 0.007 |
| R_11 | r3 (7.52) | 0.167 / 798 | 0.185 / 798 | 510 / 0.295 | 54 / 0.007 |

| P/K | prediction | baseline | measured (each sample) | verdict |
|---|---|---|---|---|
| M4 | `IMU Huber BoundaryOrRecentInit`; interior hw<1 after frame 300: 0 | C 0.031-0.040 (R_01) | line correct in all CK/K0; R_01 CK 0.000 x6, K0 0.000 x3; R_03 0.003/0.002/0.004; R_11 0.009/0.007/0.007 | as predicted on R_01; missed on R_03 and R_11 |
| M5 | CK chi2 p90 <= 1.3 x C; 3plus rejects <= 5 %; VIBA0/1/2 accepted | | R_01 8/7, 9/8, 8/8; R_03 7/7, 7/8, 7/7; R_11 56/536, 47/832, 54/510; rejects <= 1 / 575; all accepted | as predicted |
| P6 | R_01 CK along SD within 5 mm of C median (56.1) and <= 40 in 3/3; Sim3 CK - C median within +/-0.05 | | along 53.3/53.4/45.2 (none <= 40; 45.2 is 10.9 off); Sim3 CK - C -0.002/-0.242/-0.233, median -0.233 | missed (r1-r3). Diagnosis r4-r6: +0.134/+0.006/+0.001; median over six -0.001 |
| P7 | R_01 K0 along SD <= 45 in >= 2/3; Sim3 K0 - BASE <= -0.05 in >= 2/3 | along 54.1-66.7 | along 61.3/57.4/65.0; Sim3 K0 - BASE +0.067/-0.097/+0.027 | missed |
| P8 | CK guards as P5 | | R_03 0.105/0.109/0.120, 767/777/790; R_11 0.175/0.167/0.185, 798/800/798 | as predicted on R_03; missed on R_11 (0.185 > 0.18 in 1/3) |
| K3 (CK) | as stage 1 | | max CK SE3 on R_01 0.415 (r1-r3), 0.500 (r5 diag, pair 0.376); tracked 2898; all VIBA accepted; guard missed in 1/3 only | not fired |
| K6 | CK inside C's band | C Sim3 0.324-0.515 (r1-r3) | r1-r3: CK 0.180-0.322, below the band in 2/3; r1-r6: CK 0.180-0.385 inside C's 0.225-0.515 | not fired on r1-r3; fired on r1-r6 (diagnosis pairs) |
| K7 | CK beats C by > 10 mm along SD in 3/3 with C share <= 5 % | | CK - C along +2.5/-7.5/-10.9 | not fired |
| K8 | K0 trips K3 | | K0 SE3 0.298-0.359, tracked 2898, all accepted, rejects <= 1 | not fired |

### Diagnoses, stage 2

- **P6 miss (r2/r3 CK gain).** K7's remedy, 3 more pairs (r4-r6): CK - C Sim3 +0.134 / +0.006 / +0.001; over six pairs
  the median is -0.001 m, along SD CK 45.2-55.1 (median 53.0) against C 50.8-60.9 (55.0). The r2/r3 gain was run-to-run
  spread; the kernel adds no score on a clean clock (K6 on six pairs), as the theory predicted.
- **M4 miss on R_03 and R_11.** The downweighted edges sit only in the solves at frames 300-305 (R_03 one solve per run,
  R_11 two), right after VIBA2 is accepted: the `recently_initialized` span, where ORB-SLAM3 keeps the kernel on
  (bRecInit). No downweighting after it. The wording "0 after frame 300" did not leave room for the span; not re-worded.
- **P8 miss on R_11 ("If it misses" 5).** Re-run with the factory value under Hold, -3646000 ns, 3 pairs (`$AA/out/r11-CF-CKF`,
  load 5.68 / 6.76 / 8.76): CF SE3 0.171 / 0.154 / 0.176 (tracked 800 x3), CKF 0.188 / 0.166 / 0.210 (tracked 792 / 798
  / 794), CKF chi2 p90 48-54, share 0.006-0.011. With the factory offset CK misses the guard in 2/3, so the R_11 cost is
  the kernel, not the scan value: removing the interior kernel on R_11 costs 0.01-0.03 m SE3 at either offset. TD0 reads
  -4.24 ms on R_11. R_11 still carries 6-7 sigma velocity rows on long edges with the clock corrected (stage 1 diagnosis),
  and the all-edge kernel was absorbing those. The gyro-accel split (stage 3b) is the named cause.
- **P7 miss (K0, "If it misses" 4 logic, offline).** `imuedges.py --tail` on the K0 runs: chi2 > 100 falls from 24-25 %
  (BASE) to 7.7-8.3 %, p90 from 796-928 to 72-84, slope from 2.22-2.41 to 0.51-0.54 ms. The quadratic chain absorbs most of
  the clock error, as theory (b) said, but by moving the trajectory: the remaining tail is the highest-dynamics edges
  (|dw| 2.0 rad/s against 0.5 in the rest) and the along-track SD gets worse (57-65 mm). On this base the reference's own
  layout does not tolerate the uncorrected clock in score; K8's harm conditions are not met, so it is recorded, not a kill.

## Verdict (stages 1-2)

The clock theory is right on the mechanism and wrong on the score. Correcting the stamps (-3.93 ms on R_01, -3.66 R_03,
-4.54 R_11, all read under Hold) brings the R_01 and R_03 interior IMU edges to sensor strength (chi2 p50 1.6-2.2, p90
6-9, 1-5 % downweighted, slope 0.11-0.25 ms) and online td finds the same offset unaided (TD0). The 1 s along-track error,
its shape (correlation with BASE 0.45-0.82), the VI-BA speed ratio and the tilt-coupled bias wander do not change, so K1
fired: on R_01 the visual side sets the scale, not the IMU clock or the kernel. ORB-SLAM3's kernel layout on top of the
clean clock is score-neutral on R_01 (six pairs) and R_03 and costs 0.01-0.03 m on R_11; at offset 0 it does not help.
Knobs stay at their defaults (no preset change); the clock correction ships only through the device record (stage 3b,
own card). G35 is not closed by this card; next per the S card and K1's diagnosis: window time span and loop closure.
Stage 3 (code) waits for the owner's verdict.
