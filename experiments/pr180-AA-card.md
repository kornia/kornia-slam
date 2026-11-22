# Card AA: camera-IMU clock offset, then ORB-SLAM3's inertial kernel layout

Status: 2026-11-22. Stage 0 done (clock scan, factory records, baselines). Stages 1-2 run on the F1 binary by
config only. Stage 3 (code) waits for the owner's verdict. Plan: planner (Fable), owner answers 2026-11-22
(`~/.claude/plans/pr180-clock-card.md`). Card letter AA (Y is reserved for the window-local frame card).

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
