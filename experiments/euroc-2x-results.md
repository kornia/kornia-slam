# EuRoC 2x: results

Plan and ladder: `experiments/euroc-2x-plan.md`. Paths: `P` = `/home/edgar/dev/split/.pr180-fix`, `E` = `P/euroc-top1`,
`B` = `/home/edgar/dev/.flux-connect-scratch/bench`. All scores are SE(3) ATE RMSE, m, body centre, causal `traj.txt`
unless marked. Lane C scripts and raw outputs are in `E/twox/`.

## 2x-0a: G11 real-time arms scored (2026-11-25), RTbase

Source: `E/gate/scores.jsonl`, last row per run name, only runs whose dir still holds `done` (the coordinator
quarantined runs that ended at 1-min load > 6 as `*.loaded.*` and reran them; 2 stale rows of rerun names are not
counted). Table: `E/twox/g11_valid.txt`. Binary `E/bin-integ2/kornia-slam-euroc` (md5 `e97329d4…`), paced, async,
`taskset -c 0,1`, start 0. Median [min-max] of 3 runs unless the rep list says fewer.

| seq | Basalt | RT3E | RT3EF | RT3EIH |
|---|---|---|---|---|
| MH_01 | 0.067 | 0.054 [0.048-0.055] (0.80x) | 0.051 [0.044-0.065] (0.75x) | 0.083 [0.075-0.091] (1.24x), reps 2-3 |
| MH_02 | 0.048 | 0.031 [0.029-0.032] (0.64x) | 0.033 [0.031-0.033] (0.69x) | 0.040 [0.033-0.046] (0.82x) |
| MH_03 | 0.062 | 0.057 [0.054-0.058] (0.92x) | 0.055 [0.043-0.057] (0.88x) | 0.050 [0.049-0.057] (0.80x) |
| MH_04 | 0.114 | 0.247 [0.243-0.256] (2.17x) | 0.227 [0.212-0.231] (1.99x) | 0.330 [0.291-0.332] (2.89x) |
| MH_05 | 0.144 | 0.082 [0.079-0.102] (0.57x) | 0.085 [0.076-0.103] (0.59x) | 0.141 [0.126-0.143] (0.98x) |
| V1_01 | 0.043 | 0.045 [0.044-0.051] (1.05x) | 0.047 [0.043-0.050] (1.09x), reps 1-2 | 0.048 [0.043-0.050] (1.11x) |
| V1_02 | 0.045 | 0.041 [0.036-0.041] (0.91x) | 0.045 (1.01x), rep 1 | 0.038 [0.035-0.040] (0.85x) |
| V1_03 | 0.053 | 0.146 [0.085-0.152] (2.75x) | 0.117 [0.095-0.140] (2.22x), reps 1-2 | 0.053 [0.047-0.054] (1.00x) |
| V2_01 | 0.039 | 0.044 [0.044-0.055] (1.14x) | 0.051 [0.051-0.051] (1.30x), reps 1-2 | 0.055 [0.053-0.064] (1.41x) |
| V2_02 | 0.049 | 0.093 [0.090-0.096] (1.90x), reps 1-2 | 0.102 [0.098-0.106] (2.08x), reps 1-2 | 0.082 [0.079-0.085] (1.66x) |
| V2_03 | 0.230 | 0.485 [0.483-0.488] (2.11x), reps 1-2 | 0.544 [0.508-0.579] (2.36x), reps 1-2 | 0.512 [0.495-1.151] (2.23x) |
| below Basalt | | 5 of 11 | 4 of 11 | 4 of 11 |

Real-time health (median over runs; latency = output wall time minus scheduled arrival, late = latency > 50 ms):

| seq | RT3E latency p50 / p99 s, late, source p50 ms | RT3EF | RT3EIH |
|---|---|---|---|
| MH_01 | 0.09 / 2.40, 65 %, 17.8 | 0.02 / 1.86, 28 %, 4.2 | 17.2 / 44.9, 100 %, 25.8 |
| MH_02 | 0.10 / 2.22, 67 %, 17.6 | 0.03 / 1.17, 29 %, 4.4 | 15.4 / 41.0, 99 %, 25.9 |
| MH_03 | 0.36 / 2.14, 78 %, 16.4 | 0.06 / 1.94, 53 %, 4.9 | 13.6 / 23.0, 94 %, 23.3 |
| MH_04 | 0.08 / 1.45, 61 %, 17.7 | 0.03 / 0.33, 33 %, 5.1 | 3.4 / 8.9, 86 %, 22.6 |
| MH_05 | 0.05 / 1.03, 51 %, 16.9 | 0.03 / 0.45, 29 %, 5.0 | 8.0 / 21.1, 96 %, 25.4 |
| V1_01 | 0.88 / 25.4, 90 %, 17.8 | 0.42 / 8.81, 68 %, 4.2 | 34.2 / 82.8, 98 %, 27.1 |
| V1_02 | 1.32 / 6.17, 92 %, 19.8 | 0.05 / 1.08, 49 %, 4.0 | 13.0 / 31.8, 96 %, 24.2 |
| V1_03 | 2.14 / 8.09, 83 %, 19.2 | 0.06 / 1.70, 53 %, 3.9 | 25.8 / 57.0, 95 %, 27.4 |
| V2_01 | 0.32 / 3.10, 71 %, 17.5 | 0.03 / 0.50, 33 %, 3.7 | 7.6 / 21.2, 95 %, 23.2 |
| V2_02 | 4.56 / 20.4, 94 %, 18.4 | 0.72 / 7.87, 66 %, 3.8 | 31.2 / 73.8, 98 %, 27.5 |
| V2_03 | 0.50 / 2.25, 73 %, 17.9 | 0.06 / 0.76, 50 %, 3.6 | 18.2 / 30.7, 96 %, 27.2 |

Every run `rc=0`, `give_ups=0`, header clean; `pose_inertial_solves` > 0 on every RT3EIH run (1613-3638).

**Scored against the written predictions (card "Experiment G11" and "G11 extension").**

| prediction | measured | verdict |
|---|---|---|
| P-G1: RT3E below Basalt on MH_01 and V1_02 | 0.054 < 0.067, 0.041 < 0.045 | holds |
| P-G2: RT3E below Basalt on 3-6 of 11; above on MH_04, V1_03, V2_01, V2_02; M1 not met | 5 of 11 (MH_01, MH_02, MH_03, MH_05, V1_02); above on all four named | holds |
| RT3E per-sequence ranges (prediction table) | inside on 9 of 11; MH_01 0.054 above 0.035-0.050, V2_03 0.485 above 0.15-0.45 | 9 of 11 |
| P-G3: RT3EF inside RT3E's range ± max(0.01, 15 %); source p50 3-6 ms against 13-18; final latency ≤ RT3E's | inside on all 11; source 3.6-5.1 ms against 16.4-20.5 (RT3E slightly above 18 on the Vicon rooms); final-latency median lower on all 11 | holds |
| P-G5: rc 0, give_ups 0, headers clean; V2_03 coasts of order tens | all; V2_03 coasts 99-119 (RT3E), 102-110 (RT3EF) | holds (check only) |
| P-G6: RT3EIH below RT3E on V1_03 and V2_02, above on MH_01; below Basalt on 2-5 | 0.053 < 0.146, 0.082 < 0.093, 0.083 > 0.054; 4 of 11 | holds on score |
| RT3EIH per-sequence ranges | V2_02, MH_04, V1_02, V2_01 inside; V1_03 0.053 below 0.08-0.13; MH_01 0.083 above 0.055-0.080 | 4 of 6 |
| P-G4, P-G7 (SYNC arms S0PEI, S0PEIH, S0EIH) | not run yet: the SYNC lanes start only when the lock is free | open |

**RT3EIH is not a real-time arm.** Its score is good on V1_03 and V2_02, but latency p50 is 3-34 s and 86-100 % of
frames are late on every sequence: the per-frame pose-inertial solve on top of the frame work does not fit in the
50 ms period on 2 cores (source p50 also rises to 23-27 ms from 18, from contention with the tracking thread). Its
ATE is that of a pipeline that processes every frame however late. Diagnosis of where the time goes is lane A/B work
(the estimator path, 2x-5, replaces this tracking).

**RT3E is backlogged too.** Its latency p99 is 1-25 s and the final latency of single runs reaches 103 s (V1_01) and
31 s (V2_02). The run-to-run spread is large (MH_01 final latency 1.3 ms to 2.3 s), so part of it is machine
contention during G11 (other lanes' builds on unpinned cores; the gate checks `cargo`/load only at start).

**RTbase = RT3EF** (RT3E + `rectification: FixedPoint`): its scores sit inside RT3E's band on all 11 (P-G3), and it
is the only arm whose median latency p50 is near or below the 50 ms frame period on 10 of 11 sequences. It is the
arm used for 2x-1.

## 2x-0b: keyframe-only 10 s pieces against Basalt

`E/r11b/kfpieces.py` (Sim(3) per 10 s piece, median RMS, keyframes from `session.cbor`, 5 ms association) on
`traj_smoothed.txt`; Basalt on all its frames. Driver `E/twox/kfp_all.sh`, output `E/twox/kfpieces.txt`. The S0E SYNC
runs exist on 6 sequences only (starts 0/1/2, median); the S0EIH runs have not run yet, so the all-11 column is the
G11 RT3E rep-1 run (the RT3E stack contains S0E's erase knob). The V1_02 S0E start-0 value reproduces R11b's 0.0105.

| seq | Basalt all frames | S0E kf-only (÷ Basalt) | RT3E r1 kf-only (÷ Basalt) | RT3E r1 all frames |
|---|---|---|---|---|
| MH_01 | 0.0065 | 0.0060 (0.92) | 0.0050 (0.77) | 0.0058 |
| MH_02 | 0.0067 | - | 0.0051 (0.76) | 0.0061 |
| MH_03 | 0.0153 | - | 0.0151 (0.99) | 0.0153 |
| MH_04 | 0.0177 | 0.0247 (1.40) | 0.0248 (1.40) | 0.0215 |
| MH_05 | 0.0191 | - | 0.0215 (1.13) | 0.0232 |
| V1_01 | 0.0101 | - | 0.0105 (1.04) | 0.0123 |
| V1_02 | 0.0092 | 0.0105 (1.14) | 0.0109 (1.18) | 0.0147 |
| V1_03 | 0.0153 | 0.0116 (0.76) | 0.0131 (0.86) | 0.0420 |
| V2_01 | 0.0050 | 0.0054 (1.08) | 0.0050 (1.00) | 0.0059 |
| V2_02 | 0.0119 | 0.0089 (0.75) | 0.0095 (0.80) | 0.0128 |
| V2_03 | 0.0297 | - | 0.0341 (1.15) | 0.0646 |

- **P (≤ 1.3x Basalt on ≥ 8 of 11): holds.** RT3E 10 of 11 (only MH_04 at 1.40); S0E 5 of the 6 measured.
- **K (> 1.5x on ≥ 4): not fired** (0 sequences above 1.5x on either arm).
- Reading: our keyframes are already at Basalt's local precision almost everywhere. The gap is in the non-keyframe
  frames (RT3E all-frames is 1.2-2.7x the keyframe value on V1_02, V1_03, V2_03) and in drift between pieces, which
  is what the per-frame estimator (2x-4/5) and the global stage (2x-8) target. MH_04 is the one sequence where the
  keyframes themselves are 1.4x worse (the dark-span step, G58).

## 2x-0c: floor test (GT term)

`E/twox/floor.py` (raw `E/twox/floor_S0.json`, text `floor_S0.txt`): Basalt (B,
`B/results/basalt/<seq>/trajectory.txt`), ORB-SLAM3 (O, `f_run.txt`) and our offline global BA (U, `E/r6/S0_<seq>`)
on one common stamp set (Basalt stamps that O, U and GT have within 5 ms). SE(3) Umeyama, position RMSE. GT term
g² = (e_XG² + e_YG² − e_XY²) / 2 per pair. With errors a, b of two estimates and γ of GT, g² = |γ|² − ⟨a,γ⟩ − ⟨b,γ⟩
+ ⟨a,b⟩: it is the GT error when the three are uncorrelated, and it overstates it by any error the two estimates
share (same calibration, same images). Lever arm: lever.py's model E(t) = R_gt(t) l + c fitted on each estimate
against GT, then g recomputed on the corrected trajectories. `E/resid/v13v22/lever.py` itself on V1_01 and V2_01
(`E/twox/lever_V.txt`): no time offset in ±40 ms moves any of the three by more than 0.001.

| seq | target | B vs GT | O vs GT | U vs GT | B-O | B-U | O-U | g (B,O / B,U / O,U) | lever |l| B / O / U, m | g after lever fit |
|---|---|---|---|---|---|---|---|---|---|---|
| MH_01 | 0.034 | 0.067 | 0.046 | 0.031 | 0.041 | 0.045 | 0.024 | 0.049 / 0.041 / 0.035 | 0.18 / 0.13 / 0.08 | 0.041 / 0.034 / 0.029 |
| MH_02 | 0.024 | 0.048 | 0.034 | 0.020 | 0.055 | 0.054 | 0.027 | 0.014 / −0.011 / 0.020 | 0.12 / 0.03 / 0.03 | 0.017 / 0.007 / 0.019 |
| MH_03 | 0.031 | 0.062 | 0.028 | 0.028 | 0.061 | 0.062 | 0.019 | 0.021 / 0.019 / 0.024 | 0.17 / 0.03 / 0.02 | 0.016 / 0.018 / 0.024 |
| MH_04 | 0.057 | 0.113 | 0.046 | 0.073 | 0.106 | 0.104 | 0.063 | 0.042 / 0.061 / 0.042 | 0.50 / 0.21 / 0.36 | 0.032 / 0.053 / 0.037 |
| MH_05 | 0.072 | 0.137 | 0.050 | 0.073 | 0.145 | 0.134 | 0.066 | 0.013 / 0.056 / 0.042 | 0.44 / 0.18 / 0.20 | 0.018 / 0.054 / 0.041 |
| **V1_01** | 0.022 | 0.044 | 0.039 | 0.043 | 0.020 | 0.027 | 0.024 | **0.039 / 0.039 / 0.037** | 0.14 / 0.12 / 0.10 | 0.023 / 0.024 / 0.018 |
| V1_02 | 0.023 | 0.046 | 0.019 | 0.024 | 0.039 | 0.044 | 0.018 | 0.021 / 0.019 / 0.018 | 0.08 / 0.05 / 0.04 | 0.020 / 0.017 / 0.014 |
| V1_03 | 0.027 | 0.054 | 0.026 | 0.065 | 0.047 | 0.069 | 0.063 | 0.026 / 0.034 / 0.021 | 0.03 / 0.01 / 0.02 | 0.025 / 0.033 / 0.019 |
| **V2_01** | 0.020 | 0.038 | 0.031 | 0.041 | 0.022 | 0.026 | 0.028 | **0.031 / 0.035 / 0.031** | 0.07 / 0.05 / 0.08 | 0.031 / 0.035 / 0.030 |
| V2_02 | 0.025 | 0.048 | 0.038 | 0.041 | 0.063 | 0.061 | 0.055 | −0.010 / 0.010 / −0.000 | 0.02 / 0.02 / 0.02 | −0.005 / 0.010 / −0.006 |
| V2_03 | 0.115 | 0.232 | 0.026 | 0.122 | 0.230 | 0.286 | 0.123 | 0.028 / −0.081 / 0.016 | 0.07 / 0.00 / 0.04 | 0.026 / −0.069 / 0.021 |

(Negative entries: g² < 0, printed as −sqrt|g²|; the pair disagrees more than both err against GT, so no shared floor.)

- **P (g ≥ 0.025 on V1_01 and V2_01): holds.** V1_01: 0.037-0.039 on all three pairs; V2_01: 0.031-0.035.
  **K (g ≤ 0.010 there): not fired.** The targets as given (0.022, 0.020) are below the shared floor.
- Why it reads as a floor and not as three similar mistakes: on V1_01 and V2_01 the three systems are closer to each
  other (0.020-0.028) than any of them is to GT (0.031-0.044), with three different estimators (fixed-lag VIO, full
  BA with loops, our offline global BA). On V2_02 and MH_02 the same test gives g ≈ 0, so the method does not
  manufacture a floor.
- V1_01's lever fit is the same vector in all three, l ≈ (0.08-0.12, −0.04, −0.03-0.04) m in the body frame. A
  constant body-frame offset common to every estimator is a GT or calibration term (Vicon-body frame or T_BS), not
  an estimation error. Removing it lowers g to 0.018-0.024, which is still near or above the target. No causal
  estimator can remove it without fitting to GT. On V2_01 the fit does not lower any error (l is poorly observed).
- The MH lever fits (0.2-0.5 m on MH_04, MH_05) absorb drift, not a lever; they are not used.
- MH_01's g (0.035-0.049) is not a floor for us: our offline output is at 0.031 there, below every g estimate. The
  Machine Hall g values carry shared drift and are an upper bound only.
- **For the owner at S-A (open question 1):** with "floor + 10 %" the V1_01 target becomes about 0.043 (= Basalt's
  0.043, so 0.5x Basalt is not reachable there by any system) and V2_01 about 0.036 (0.92x Basalt). The other nine
  targets stand; V1_02 (g ≈ 0.019) and V1_03 (0.021-0.034) are close to their targets (0.023, 0.027) and are worth
  re-checking with the same test once a better causal output exists.

## 2x-1: speed baseline against Basalt

(see below)

## Pause snapshot 2026-11-25 (05:03, owner paused all runs)

State at the pause:
- **2x-1 speed:** no speed run started. The `bench-basalt2-paced` image is built, and its 200-frame smoke test gives p50 28 ms and p99 76 ms with 0 frames dropped.
- **G11:** the real-time arms are done, apart from the queued reruns of loaded runs. The SYNC arms did not run.
- **2x-5 Observe:** 26 of 33 primary VO-ds runs are done. VO-bn, the r2 determinism runs and the remaining 7 primary runs did not run.

2x-5 VO-ds partial results: VIO-only `traj_vio.txt`, SE(3) ATE RMSE in metres, against Basalt cropped to the same stamps. Each cell is ours / Basalt and the ratio.

| seq | s0 | s30 | s60 | median ratio |
|---|---|---|---|---|
| MH_01 | 0.056 / 0.067 (0.83) | 0.051 (0.75) | 0.050 (0.74) | 0.75 |
| MH_02 | 0.145 / 0.048 (3.01) | – | 0.122 (2.53) | 2.77 |
| MH_03 | 0.127 / 0.062 (2.06) | – | 0.140 (2.27) | 2.16 |
| MH_04 | 0.123 / 0.114 (1.08) | – | 0.123 (1.08) | 1.08 |
| MH_05 | 0.152 / 0.145 (1.05) | – | 0.163 (1.13) | 1.09 |
| V1_01 | 0.054 / 0.043 (1.24) | – | 0.053 (1.21) | 1.22 |
| V1_02 | 0.080 / 0.045 (1.76) | 0.080 (1.76) | 0.079 (1.75) | 1.76 |
| V1_03 | 0.101 / 0.053 (1.89) | 0.102 (1.91) | 0.101 (1.89) | 1.89 |
| V2_01 | 0.070 / 0.039 (1.81) | 0.080 (2.06) | – | 1.93 |
| V2_02 | 0.093 / 0.049 (1.90) | 0.100 (2.04) | – | 1.97 |
| V2_03 | 0.921 / 0.230 (4.01) | 3996.6 (diverged) | – | – |

Reading, before any diagnosis:
- **P3 fails.** It predicted a ratio of 0.8-1.3 on at least 9 of 11; only MH_01, MH_04, MH_05 and V1_01 are inside that band.
- **K1 applies** if P1 and P2 hold; they are not scored yet.
- **First step:** the K1 diff against Basalt on MH_01, first 400 frames. The order of suspects is the FEJ form for visual rows, the IMU noise arm, the bias start, and then track counts per frame.
- **V2_03 start 30 diverges.** It needs its own diagnosis. `vio_movement_rejects` (the 60 px cap, risk table in the design card) is the first suspect.
- **MH_04 against the pipeline:** the VIO-only result (1.08x) is better than the pipeline's causal real-time result (RT3E 2.17x). On every other sequence the current pipeline (RT3E) is better.

## 2x-4: sliding-window VI odometry, tests before any run (lane A, `feat/2x-vio`)

Design: `experiments/euroc-2x-vio-design.md`. All tests run in release and are deterministic (no `rand`).

| id | test | prediction | measured | verdict |
|---|---|---|---|---|
| T0 | anchored Jacobians vs central differences, host = −target, ρ = 0, chart round trip | 1e-6 relative | passes on 8 random geometries | pass |
| T1 | 40-state linear chain with landmarks, sequential vs batch | 1e-9 | newest state and prior within 1e-9 | pass |
| T1b | `schur_complement` vs `marginalize_frozen` | 1e-12 | within 1e-12 | pass |
| T2 | sequential prior vs one-shot Schur of the ledger, 60 noisy frames | 1e-9 | H 6.5e-12, b 3.4e-12 (56 priors) | pass |
| T3a | cost under a world translation and 0.7 rad yaw | 1e-9 | 2.7e-15; columns match to 1e-5 at ε = 1e-6 | pass |
| T3b | twin prior x, y, z, yaw after 400 frames | ≤ 1e-9 | final ≤ 7.8e-15, worst over the run 1.5e-13 | pass |
| T3c | roll, pitch ≥ 1e3 × yaw | ≥ 1e3 × | roll 1.6e-4, pitch 7.9e-6 against yaw 3.4e-15 | pass |
| T4a | walk from rest, exact data | accel ≤ 0.01, gyro ≤ 1e-4, tilt ≤ 0.1° at 10 s; position ≤ 0.02 m at 20 s | 0.0041 m/s², 4.3e-5 rad/s, 0.027°; 4.2 mm; no reset | pass |
| T4b | moving start at 5 s, 0.5 px, datasheet IMU noise | accel ≤ 0.03, gyro ≤ 3e-4, tilt ≤ 0.3°; `imu_chi2_nav` 0.5-2.0 | 0.0047, 1.0e-4, 0.020°; `imu_chi2_nav` **0.053** | bias and tilt pass; chi² **miss** |
| T5 | M0-M10 plan rules, keyframe rule, per-frame bookkeeping | exact | pass (kf count ≤ max + 1 between conversions, as M5) | pass |
| T6 | Huber at 2 px = 0.5, λ updates, ≤ 8 attempts, failed factor keeps the prediction | exact | pass; plus the system gradient equals the cost derivative to 1e-5 | pass |
| T7 | stereo link at 12 px disparity, occluded patch, row gate, refill camera 0, determinism | 0.05 px, exact | pass | pass |
| T8 | Observe leaves 30 frames of `TrackingResult` bit-identical; validation; header; RON | exact | pass | pass |

### T4b chi² miss: diagnosis

On the T4b walk the IMU nav chi² per DOF reads 0.053 for the newest edge after the solve (`imu_chi2_nav`), 0.100
for each edge as it enters the marginal prior (`imu_chi2_marg`), and 0.973 for the same edges at the true states.
The noise model and its injection are consistent; the two posterior statistics are biased low by construction. A
50 ms preintegrated edge (σ_p about 1e-5 m, σ_R about 4e-5 rad) carries more information than one frame's vision
(mm and about 1e-4 rad from 0.5 px over about 150 points), so consecutive poses and the free velocities follow the
IMU and the edge's posterior residual stays far below its noise. The test now asserts the truth statistic; both
posterior statistics are logged with no bound. Consequence for 2x-5 (coordinator decision, recorded in the card):
the noise-arm rule cannot run on them, the primary arm is `VO-ds`, `VO-bn` is run on the screening five for the
record. An innovation statistic is a later card.

### Step 12: binary, identity, T9, bench

- Binaries: `bin-2x5-c7` (md5 `08cd9e65b9447f237d67a8640d4a2c0d`, commit 128443a) and `bin-2x5-c11` (md5
  `22dadcc007288253a29f243920127d10`, commit 6c7c18a), under `.pr180-fix/euroc-top1`.
- Identity trio, knob off, on `bin-2x5-c7`: `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` (md5
  `d4f77b567b912b5a4caa2af6085a35b1`). The trio on `bin-2x5-c11` and the Observe identity items 2-4 have not run (see
  below).
- T9 parity: `frontend_opencv.bin` generated locally with OpenCV 5.0.0 (Python) on the kornia-rs `mh01_frame1/2.png`
  (752×480, 200 corners, 197 tracked; md5 `823e6de5fad1f51c5569b1cf82507bf2`, not committed). The three ignored
  parity tests pass (LK within 0.01 px with identical status, goodFeaturesToTrack, cornerSubPix).
- `bench_odometry_window` (P5 window step, includes the nullspace twin): 18.0 / 19.2 / 20.5 ms against a predicted
  p50 ≤ 8 ms. **Provisional, not a valid timing sample:** core 8 at 1.2 GHz, its sibling (core 2) busy, load 5.8.
  To rerun on an idle core pair at load ≤ 3 with governor and frequency read first; if it still misses, profile
  linearize, Schur, Cholesky, marginalization and the twin, and time the step without the twin.
- `bench_upstream_lk_window` with `opencv_bench` (OpenVINO 2021.4 OpenCV 4.5.3, core 8 at 2.4 GHz, load 3.8, under
  `rt.lock`), 48 tracks: our windowed forward solve 141 µs [135, 147], OpenCV prebuilt pyramid 154 µs [144, 163],
  kornia-rs u8 151 µs. Point estimate below OpenCV, confidence intervals overlapping: the "beats OpenCV" half of T9
  is marginal on this sample.
- Identity trio on `bin-2x5-c11`: `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` (md5 `d4f77b567b912b5a4caa2af6085a35b1`).
- Observe identity: S0E V1_02 start 0 knob off (r9) and VO-ds V1_02 start 0 on the same binary: `traj.txt` and
  `traj_smoothed.txt` `cmp`-identical (also identical to the older binary's S0E r1). `expect/VO-ds.header` is
  `S0E.header` plus `  odometry: sliding window, Observe, IMU noise Rig`. Determinism (VO-ds V1_02 and MH_04 twice)
  not run.

## 2x-5: Observe, VO-ds (PARTIAL, PAUSED 2026-11-25 05:03 by the owner)

**Partial and paused.** 26 of the 33 primary runs finished (SYNC, unpaced, 900 s cap, `taskset -c 2-5,8-11`,
`bin-2x5-c11`); the other 7, the determinism repeats, the `VO-bn` record runs and the pinned P5 timing run did not run.
No verdict below is final. Scored by `E/twox/vio_score.py`: `B/eval.py --fmt tum --rpe 1,10` on `traj_vio.txt`,
Basalt cropped to the same stamps (5 ms). SE3 ATE rmse, m.

| seq | start 0: VIO / Basalt = ratio | start 30 | start 60 | resets | landmarks mean | vio_ms p50 / p99 (unpinned) |
|---|---|---|---|---|---|---|
| MH_01 | 0.056 / 0.067 = 0.83 | 0.051 = 0.75 | 0.050 = 0.74 | 0 | 84 | 20.3 / 53.0 |
| MH_02 | 0.145 / 0.048 = 3.01 | not run | 0.122 = 2.53 | 0 | 82 | 19.7 / 49.8 |
| MH_03 | 0.127 / 0.062 = 2.06 | not run | 0.140 = 2.27 | 0 | 72 | 18.8 / 48.2 |
| MH_04 | 0.123 / 0.114 = 1.08 | not run | 0.123 = 1.08 | 0 | 58 | 16.1 / 45.9 |
| MH_05 | 0.152 / 0.144 = 1.05 | not run | 0.163 = 1.13 | 0 | 65 | 16.3 / 45.9 |
| V1_01 | 0.053 / 0.043 = 1.24 | not run | 0.053 = 1.21 | 0 | 56 | 15.2 / 42.2 |
| V1_02 | 0.080 / 0.045 = 1.76 | 0.080 = 1.76 | 0.079 = 1.75 | 0 | 39 | 12.6 / 36.1 |
| V1_03 | 0.101 / 0.053 = 1.89 | 0.102 = 1.91 | 0.101 = 1.89 | 0 | 24 | 10.1 / 32.1 |
| V2_01 | 0.070 / 0.039 = 1.81 | 0.080 = 2.06 | not run | 0 | 59 | 14.7 / 38.9 |
| V2_02 | 0.093 / 0.049 = 1.90 | 0.100 = 2.04 | not run | 0 | 43 | 13.0 / 35.5 |
| V2_03 | 0.921 / 0.230 = 4.01 | **3996.6** (diverged) | not run | 0 | 20 | 8.3 / 25.1 |

Every run: frames = poses (MH_05 start 0: 1 frame without a pose, `errors=1`), `solve_failures=0`, `resets=0`.
Twin nullspace `nullspace_yaw_log10_max` −13.0 to −12.0 on every run. `imu_chi2_nav` 0.02-0.14, `imu_chi2_marg`
0.04-0.21 (logged only; biased low, see 2x-4). `vio_flow_ms` p50 11.4 and `vio_solve_ms` p50 7.1 on MH_01 start 0.

| id | prediction | measured (partial) | reading |
|---|---|---|---|
| P1 | V1_03 HF ≤ 0.008 every start; step p50 ≤ 1.5x Basalt on MH_01, V1_02, V2_01 | HF 0.004 on starts 0/30/60 (Basalt 0.004); step p50 0.95-0.99 / 1.39-1.46 / 0.87-0.90 mm against Basalt 0.80 / 1.36 / 0.67 (1.19-1.24x, 1.02-1.07x, 1.30-1.34x) | holds |
| P2 | MH_04 vertical increment over 63.3-66.7 s within ±0.10 m | −0.03 m (start 0), −0.02 m (start 60) | holds |
| P3 | ratio 0.8-1.3 on ≥ 9 of 11, none above 1.6 | in band on 4 (MH_04, MH_05, V1_01; MH_01 below band at 0.74-0.83, better than Basalt); above 1.6 on 7 (MH_02, MH_03, V1_02, V1_03, V2_01, V2_02, V2_03) | **fails** |
| P4 | V2_03: 0 frames without a pose; ATE ≤ 0.30 | 0 frames without a pose, 0 resets; ATE 0.92 (start 0), 3997 m (start 30) | **fails** |
| P5 | flow + solve p50 ≤ 16 ms, p99 ≤ 35 ms on one core | unpinned SYNC runs, load 4-10: p50 8.3-20.3, p99 25-53; the pinned run did not run; bench 19.2 ms invalid (see step 12) | not measured validly |

Kill rules: **K1 fires** (P1 and P2 hold, P3 fails on 7 ≥ 3). Its card action: diff against Basalt on MH_01, first
400 frames, before anything else; suspects in order: FEJ form for visual rows (open question 1), IMU noise arm, the
uncalibrated bias start, track counts per frame. K2 does not apply (P3 fails; nullspace and HF pass). K3 is not
decided (no valid P5 sample). K4 does not fire (HF 0.004).

Observations for the diagnosis, none tested yet:
- The error is low-frequency, not jitter: V1_03 LF 0.100 against HF 0.004 (Basalt LF 0.053, HF 0.004). The high
  ratios sit where Basalt itself is best (MH_02, the Vicon rooms).
- The worst sequences have the fewest landmarks per frame: 20-43 on V1_02, V1_03, V2_02, V2_03 against 72-84 on
  MH_01-MH_03, with `movement_rejects` 9-14 k per run. MH_02 and MH_03 have many landmarks and still fail, so track
  count alone does not explain it.
- V2_03 start 30 diverged to kilometres with `resets=0` and `errors=0`: the reset policy (owner answer 9) catches only
  non-finite states, so a finite runaway goes undetected. Its cause is not traced.

## 2x-5 K1 diagnosis (card in `experiments/euroc-2x-plan.md`, section "2x-5 K1 diagnosis")

### K1-a, MH_01 start 0 (found 2026-11-25 17:40)
Instruments: ours `VO-ds_MH_01_easy_s0_r7` (`bin-2x5-k1`, md5 `720f91a1`, `--vio-states`; the run hit the 900 s cap
at machine load 30-47, so the first 3479 of 3682 frames are read; its poses are identical to `VO-ds_MH_01_easy_s0_r1`
from `bin-2x5-c11` over those frames). Basalt: `bench-basalt2-states`, unpaced, its biases plus its calibrated
offsets (`euroc_ds_calib.json`). Script: `E/twox/k1/statediff.py`, per 10 s means against GT.

- **Start transient (ours only):** in the first 10 s our accel bias y reads +0.38 m/s² against GT +0.14 (Basalt
  +0.15), z +0.20 against +0.08, and the tilt error is 2.1° against Basalt's 0.5°. Both are gone by 20 s (tilt
  0.12°, accel bias y +0.15). This is the uncalibrated zero-bias start (suspect Pd3; owner answer 5 keeps it).
- **Steady state (20 s on):** gyro bias within 3e-4 rad/s of Basalt and of GT on every axis; accel bias within
  ±0.04 m/s² of GT (Basalt ±0.03); tilt 0.1-0.3° (Basalt 0.1-0.2°); speed error within ±0.01 m/s (Basalt alike).
- **Counts:** 50-114 landmarks and 290-780 left rows per frame.
- MH_01 is our good sequence (0.75-0.83x Basalt), so it cannot show the failing mechanism; MH_02 and V1_02 decide.

### K1-a, MH_02 and V1_02 start 0 (found 2026-11-25 18:00)
Runs `VO-ds_MH_02_easy_s0_r7` (complete; poses identical to r1) and `VO-ds_V1_02_medium_s0_r7` (complete), Basalt
`bench-basalt2-states`. Yaw and position growth from `growth.py` (SE3 fit on the first 10 s of overlap; change from
10 s on), ours against Basalt:

| seq | biases, tilt, speed against Basalt | yaw change from 10 s | vertical error at 70 s | landmarks / left rows / right rows per frame |
|---|---|---|---|---|
| MH_02 | gyro within 2-7e-4 rad/s, accel within 0.03 m/s², tilt 0.16-0.32° (Basalt 0.11-0.54°), speed within 0.02 m/s | +1.5° by 110 s against −0.0° | +0.23 m against +0.14 m | 39-122 / 259-755 / 4-546 (right rows 4 over 10-20 s) |
| V1_02 | gyro within 3-9e-4, accel within 0.03 (our z 0.056-0.094 against GT 0.093, Basalt 0.075-0.087), tilt 0.15-0.59° (Basalt 0.20-0.45°), speed within 0.01 | −0.0° against +0.2° | +0.12 m against +0.03 m | 21-50 / 130-325 / 59-179 |
| MH_01 (above) | same agreement | +0.5° to 90 s, then as Basalt | - | 50-114 / 290-780 |
| V2_02 (trajectory only) | - | +1.2° by 90 s against −0.4° | - | - |
| MH_03 (trajectory only) | - | −2.7° against −3.2° | - | - |

Against the card's predictions:
- **Pd1 (FEJ form):** biases agree within the predicted 0.01 m/s² / 2e-4 rad/s only roughly (accel within 0.03), and
  the yaw drift is ≥ 2x Basalt's on MH_02 and V2_02 but not on V1_02, where the excess is vertical. **Partly shown.**
- **Pd2 (noise arm):** our accel bias wanders ±0.04 against Basalt's ±0.03 (1.3x, not ≥ 2x). **Not shown**; the
  VO-bn record runs are still queued.
- **Pd3 (bias start):** the start transient (MH_01) is over by 20 s on every sequence, while MH_02's excess grows at
  50-110 s. **Not shown** as the main cause.
- **Pd4 (track counts):** Basalt's per-frame track count is not logged, so the 0.5x comparison is open. Our Vicon-room
  left rows (130-325) are under half the Machine Hall ones (290-780), and MH_02 lost its right rows over 10-20 s.
  **Open.**
- **KD1 does not fire** (Pd1 partly shown). Next per the card: A/B 1, `visual_jacobians: Current`, once the fold is
  done; the V1_02 vertical drift and the MH_02 right-row gap are recorded for A/B 4.

### 2x-5 update (2026-11-25 19:10; still partial)
More runs finished after the resume at 17:01 (`bin-2x5-c11`): 31 of the 33 primary VO-ds runs; MH_02 start 30 and
V1_01 start 30 hit the 900 s cap at machine load 30-47 and are in a rerun pass (paused again by the owner).
- Determinism: VO-ds V1_02 and MH_04 start 0, r1 against r2: `traj_vio.txt` `cmp`-identical.
- VO-ds medians over the starts run (ratio to Basalt cropped to each start's stamps): MH_01 0.75, MH_02 3.01 (2
  starts), MH_03 2.27, MH_04 1.08, MH_05 1.13, V1_01 1.24 (2 starts), V1_02 1.76, V1_03 1.89, V2_01 1.97, V2_02 1.90,
  V2_03 4.01 (start 60: 0.321 = 1.39; start 30 diverged). P3 still fails: in band on MH_04, MH_05, V1_01 (MH_01
  better), above 1.6 on 7.
- VO-bn (Basalt's noise values; record only, does not decide the arm), start 0: MH_01 0.043 (0.64), MH_04 0.152
  (1.33), V1_02 0.074 (1.63), V1_03 0.105 (1.96), V2_03 **0.287 (1.25)** against VO-ds 0.921 (4.01). The noise arm
  moves V2_03 by 3x and the others by up to ±20 %: Pd2 is not refuted for the fast sequence; it goes into K1-c.

## Pipeline reference, final tip bench (2026-11-26)

The estimator results above compare against this pipeline baseline: `bin-fold` RT3EF, real time, median of 3 reps. 6 of 11 are below Basalt (best MH_01 0.64x); worst V2_03 2.47x, V1_03 2.38x, V2_02 2.15x, MH_04 2.00x. Speed is not faster than Basalt (G57). Full table: `experiments/euroc-why-not-basalt.md`.
