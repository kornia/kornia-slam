# EuRoC top-1: master card

Status: written 2026-11-23 from the approved EuRoC top-1 plan; the owner took all 10 recommended answers
(listed at the end). Stop S0 (card review) is approved. LaMAria work waits behind this card.

Paths: `P` = `/home/edgar/dev/split/.pr180-fix`, `E` = `P/euroc-top1` (experiment dir, new),
`B` = `/home/edgar/dev/.flux-connect-scratch/bench` (scoring and references), `K` = `crates/kornia-slam/src`,
`A` = `apps/kornia-slam-app/src`. Evidence: `P/euroc/MH04-DIAGNOSIS.md`, `P/euroc/V102-DIAGNOSIS.md`,
`P/euroc/runs/`.

## Goal

One kornia-slam config, stereo+IMU, 100 % tracked, below Basalt on all 11 EuRoC sequences (milestone M1),
then below ORB-SLAM3 (M2). Every step is a rung below, with predictions and kill rules, measured on a
deterministic harness.

## Three findings that change the picture

1. **The harness scored the wrong point (ledger G55).** `P/euroc/report.py` scored `kornia_slam_raw.csv`,
   the camera centre (`A/evaluation.rs:131-138`). GT, Basalt and ORB-SLAM3 are body (IMU) poses. `traj.txt` is
   body (`A/trajectory.rs:18-40`); the two differ by a constant 0.069 m (cam0 lever arm). Scoring `traj.txt`
   with `eval.py --fmt tum` lowers our ATE by 0.003-0.025 m. All numbers in this card are body-centre.
2. **The smoothed output is already below Basalt on 4 sequences** (planner proxy: frames re-expressed on the
   final-map keyframes, translation only, from `session.cbor`): MH_01, MH_02, MH_03, MH_05. No EuRoC run had
   passed the app's `--trajectory-smoothed` before the 2026-11-23 driver (ledger G53).
3. **No single config is healthy on all 11.** Default wins MH_03/04/05; the preset wins the Vicon rooms.
   Default fails every Vicon room (G54: async + `merge: SnapshotOnly`). The preset is worse on MH_03/MH_04
   (window 10 with loose bias rows, G49/G6).

## 1. Gap table

SE(3) ATE RMSE, m, body centre, one async unpaced run each at load 8-24. "sm" = smoothed proxy (planner).
Basalt reference ran blocking (`vio_enforce_realtime: false`) at load 24-27, one run. ORB-SLAM3 README numbers
cover 80-96 % of frames (after IMU-init resets). The harness re-score of the same runs (app smoothed output
where it exists) is in "Results: body-centre re-score" below.

| seq | default causal / sm | preset causal / sm | Basalt | ORB-SLAM3 (fair ref) | best sm ÷ Basalt | best causal ÷ Basalt | cause of the gap | closes it |
|---|---|---|---|---|---|---|---|---|
| MH_01 | 0.080 / 0.041 | 0.083 / 0.034 | 0.067 | 0.046 (96 %); loops-off full 0.055 | 0.51-0.61 | 1.19 | causal: loop jumps stay in output (G53) | R0, R2 |
| MH_02 | 0.064 / 0.027 | 0.052 / 0.025 | 0.048 | 0.034 (93 %) | 0.52 | 1.08 | G53 | R0, R2 |
| MH_03 | 0.048 / 0.050 | 0.071 / 0.085 | 0.062 | 0.028 (85 %) | 0.81 (default) | 0.77 (default) | preset: window-10 gyro-bias freedom (card W) | keep under: R3 before R4 |
| MH_04 | 0.306 / 0.165 | 0.471 / 0.213 | 0.114 | 0.070 loops-off full; README 0.045 on 80 % | 1.45 | 2.7 | dark-span tilt 3.2-3.8° hidden in accel bias (G49); causal loop jump (G53); 4-DoF loop cannot remove tilt | R3, then R5 or R6 |
| MH_05 | 0.178 / 0.081 | 0.164 / 0.109 | 0.144 | 0.050 (82 %) | 0.56 | 1.14 | G53; vertical 0.104 | R2; R3 for causal |
| V1_01 | 0.141 / 0.137 | 0.074 / 0.060 | 0.043 | 0.039 (96 %) | 1.40 | 1.7 | default: G54 at low dose (14 shifts). Residual: **unknown** (10 s-piece error 0.022-0.029 against Basalt 0.013) | R1, then R3/R4/R7 |
| V1_02 | 2.40 / 1.22 | 0.095 / 0.065 | 0.045 | 0.019 (94 %) | 1.44 | 2.1 | default: G54 (84 shifts, 1-10 m jumps; scale 0.345 is an artefact). Residual: **unknown** | R1, then R3/R4/R7 |
| V1_03 | 114.5 / 112 | 0.202 / 0.066 | 0.053 | 0.026 (93 %); loops-off run is broken (0.628, scale 0.857), not a bar | 1.25 | 3.8 | default: G54 + runaway coasts (G39). Residual: **unknown**; flow unavailable on 1711 of 2149 frames (G50/G51) | R1, then R7/R8 |
| V2_01 | 19.1 / 19.0 | 0.105 / 0.079 | 0.039 | 0.031 (96 %) | 2.0 | 2.7 | default: G54 (103 shifts). Residual: **unknown**; error sits at start and end (0.11-0.17) | R1, then unknown |
| V2_02 | pending | pending | 0.049 | 0.037 (96 %) | unknown | unknown | unknown | R0 |
| V2_03 | pending | pending | 0.230 | 0.027 (94 %) | unknown | unknown | unknown (2026-11-10: 0.96) | R0 |

Local accuracy (10 s pieces, Sim3 oracle RMS): ours smoothed 0.008-0.052 against Basalt 0.005-0.026. Vicon
rooms are 1.5-3x behind even smoothed; that is the part with no diagnosis yet.

## 2. Measurement protocol

- **Experiment dir:** `E` (do not write into `P/euroc/`). Binary for R0-R6: `P/euroc/bin/kornia-slam-euroc`,
  md5 `cea7646298a68ddb948b16835472b2e3` (code c21299b). No build.
- **SYNC mode (decides every A/B):** `mapping: ( execution: Synchronous )`, all rayon threads (no pin, see the
  change below), `--no-pace`. Synchronous has no stale keyframes, so G54 cannot hide an effect. It matches how the Basalt
  reference ran (blocking).
- **Protocol change (owner, 2026-11-23, after K0.2):** the `RAYON_NUM_THREADS=1` pin is dropped. The pin cost
  about 1.9x wall (V1_02 projected 347 s pinned against 190 s for the other agent's unpinned T3 run of the same
  config), and T3 matched the pinned run's first 1432 poses byte for byte. Determinism is re-checked unpinned
  (P0.1) before any arm runs.
- **Protocol change (owner, 2026-11-23, after the second K0.2):** SYNC runs get a 900 s cap; every other run
  keeps 300 s. SYNC outputs do not depend on load, only wall time does (every common prefix of pinned, unpinned
  and repeated runs was byte-identical), and at load 15-33 no SYNC run fit 300 s. The laptop can suspend, so
  `score.py` also records the per-frame time sum; where a suspend inflates the wall, that sum is the wall
  figure. P0.5 and K0.2 are read against the cap of the run's mode.
- **ASYNC mode (deployment check, batch gates only):** `--pace-timestamps`, no thread pin, 3 runs per sequence
  for the band. Unpaced async is dropped: it starves BA (1.4-1.75 keyframes inserted per solve) and is neither
  deterministic nor the deployed regime.
- **Command:** `timeout $cap bin/kornia-slam-euroc --config $d/run.ron --no-tui --no-pace
  --trajectory-out $d/traj.txt --trajectory-smoothed $d/traj_smoothed.txt --session-out $d/session.cbor`.
  No `--evaluate --eval-out` (camera-centre csv). `$cap` is 900 s for SYNC arms, 300 s otherwise; `rc=124` means the cap fired: run invalid.
- **Scoring, both outputs:** `B/venv/bin/python -I B/eval.py <euroc>/$seq $d/traj.txt --fmt tum --rpe 1,10`,
  same for `traj_smoothed.txt`. Also ours cropped to ORB-SLAM3's covered span (`B/results/orbslam3/$seq/f_run.txt`
  timestamps). `E/score.py` refuses any other input file.
- **Config self-check:** the err.log lines before `Dataset:` go to `$d/header.txt` and must `diff` clean
  against `E/expect/$arm.header`; non-empty diff makes the run invalid and `done` is not written. The
  `Dataset:` line is checked apart (it carries the sequence length and `start_frame`). The header prints the
  local-BA line only when it differs from the default (`K/system/config/mod.rs:854-879`): SYNC base must show
  `local mapping: synchronous` and no `local BA:` line; E1 must show `(bias rows full)`; the merge arm must show
  `local BA merge: PropagateToNewer`.
- **Determinism check (first):** SYNC base on MH_04 and V1_02, 2 runs each; `cmp` on `traj.txt` and
  `traj_smoothed.txt` must print nothing.
- **Sensitivity band (replaces repeats in SYNC):** 3 runs per arm with `start_frame: 0, 1, 2`. The band of a
  sequence is the min..max of the baseline's 3.
- **Load:** at most 2 SLAM jobs (`ps -eo comm | grep -c '^kornia-slam'`, counts other sessions too). Record
  tag, sha, md5, config, load, wall per run. SYNC results do not depend on load; ASYNC runs are valid only at
  load ≤ 6.
- **Screening set per rung:** MH_03, MH_04, V1_02, V1_03, plus MH_01 as guard. All 11 only at batch gates.

## 3. Fix ladder

Order is expected gain over cost. R0-R6 need no code. Winners stack; the stack becomes the next base. A missed
prediction or fired kill gets a written diagnosis before the next rung. Stacking: R1 is the async base for
everything. R3, then R4 on top. R5 replaces part of R3, never both. R6 and R8 stack on any base. R7 after R3+R4.

### Experiment R0: harness and SYNC baseline

- **Gap tested:** G55 (scored point), G53 (output), G54 (SYNC removes it by construction).
- **Base:** kornia-slam@c21299b, binary md5 `cea7646298a68ddb948b16835472b2e3`.
- **Mechanism:** none changed; this rung builds the instrument.
- **Knob:** arms `S0` = default knobs + `execution: Synchronous`, 11 sequences × 3 starts; `S0-noloop` = S0 with
  `loop_closing: Disabled`, 11 × 1 (VIO against Basalt's VIO, and the size of what loops give).
- **Baselines:** `P/euroc/runs/default_*_r1`, `preset_*_r1` (async unpaced, load 8-24), re-scored body-centre.
- **Predictions:**
  - P0.1 repeats are `cmp`-identical (MH_04, V1_02, 2 runs each; unpinned since the K0.2 protocol change).
  - P0.2 `ba_stale_keyframes=0` and `ba_keyframe_shifts ≤ 5` on all 11; Vicon rooms Sim3 scale 0.98-1.02;
    V1_02 causal 0.05-0.10 (2026-11-10 sync run: 0.075 camera-centre).
  - P0.3 `tracked == frames` on all 11, give-ups 0.
  - P0.4 app smoothed output within 0.005 m of the proxy on the same session.
  - P0.5 every wall ≤ 300 s.
  - P0.6 S0 smoothed below Basalt on MH_01, MH_02, MH_03, MH_05.
  - Mechanism counter: `ba_stale_keyframes`, `ba_keyframe_shifts`.
- **Kill rules:**
  - K0.1 not bit-identical: stop, find the source (loop closing, place recognition); fallback is 3 plain
    repeats per arm.
  - K0.2 any wall > 300 s: stop and report; candidate fix is the Basalt solver budget (R4) in the base.
  - K0.3 a Vicon room fails in SYNC: G54 is not the cause; stop.
- **Bit-identity:** P0.1. **Noise:** the 3-start band.

### Experiment R1: G54, stale keyframes under async

- **Gap tested:** G54, G36. **Base:** c21299b, ASYNC paced.
- **Mechanism:** `LocalBaMerge::SnapshotOnly` (`K/mapping/map/ops/correction.rs:94-106`) leaves keyframes
  inserted during a solve on their seed; stale count at `K/mapping/local_mapping.rs:210`.
- **Theory / reference:** OKVIS2 `ViSlamBackend.cpp:1442-1470` carries the correction to newer states; ORB-SLAM3
  `Tracking.cc:1746-1764` re-predicts from the updated keyframe.
- **Knob:** `mapping.local_ba.merge` (default `SnapshotOnly`) to `PropagateToNewer`.
- **Baselines:** the queued T1-T4 V1_02 runs in `P/euroc/runs/` (read first). T1 passes (shifts < 10,
  ATE < 0.2 m, scale 0.98-1.02): async base `A1` = default + `PropagateToNewer`. T1 fails and T2 or T4 passes:
  add window 10 or the 7-iteration budget. T3 fails: stop.
- **Predictions (`A1` on all 11, ASYNC paced × 3):** shifts ≤ 10 per sequence; Vicon causal V1_02 0.08-0.15,
  V1_03 ≤ 0.30, V2_01 ≤ 0.15; MH within ±15 % of default async; MH_04 no repeated bias state, tilt step over
  64-65.5 s ≤ 0.5°. Mechanism counter: `ba_propagated_keyframes` = `ba_stale_keyframes`, shifts.
- **Kill rules:** mechanism moved (propagated = stale, shifts < 10) and a Vicon room still > 0.5 m: a second
  cause; diagnose. Score moved with stale keyframes unpropagated: another cause; diagnose before stacking.
- **Bit-identity:** in SYNC, S0 + `PropagateToNewer` must be `cmp`-identical to S0.

### Experiment R2: causal output decision

- **Gap tested:** G53. No runs beyond R0. Output exists: `A/main.rs:75-77, 607-614`; reference ORB-SLAM3
  `System.cc:662` (SaveTrajectoryEuRoC).
- **Prediction:** smoothed ≤ causal by 30-50 % on sequences with an applied loop, about 0 without.
- **Decision (owner, 2026-11-23):** smoothed ranks; causal is reported and gated for no regression.

### Experiment R3: bias rows at full weight (card AH arm E1)

- **Gap tested:** G49, G35. **Base:** S0.
- **Mechanism:** boundary-edge bias-walk rows scaled 1e-2 (`K/mapping/vi_ba_schur.rs:864-875`).
- **Reference:** ORB-SLAM3 `Optimizer.cc:2646-2657` (full information on every edge), Basalt
  `imu_block.hpp:73-92`.
- **Knob:** `mapping.local_ba.boundary_scales_bias_rows` (default true) to false. Arm `S1` = S0 + it.
- **Predictions (baseline S0, filled before the run):** accel and gyro walk z p50 ≤ 2.0 on every sequence
  (async default: 11.2 / 4.2 on MH_04). MH_04: tilt at 65-79 s ≤ 1.5°, loop `map_from_odometry` |t_y| ≤ 0.4 m,
  vertical RMS ≤ 0.10 m, causal ≤ 0.20 m, smoothed 0.08-0.13 m. Other sequences within ±15 %. Tracked unchanged.
  Mechanism counter: walk z p50 (`P/euroc/sessbias.py`).
  - **S0 values for these (SYNC, starts 0/1/2, filled 2026-11-23 before the S1 runs, `E/r3/r3metrics.py`,
    `E/r3/s0metrics.jsonl`):** walk z p50 accel / gyro: MH_04 6.25 / 6.25 / 6.39 and 2.54 / 2.54 / 2.44; over the
    six R3 sequences accel 6.3-19.9, gyro 2.0-3.9. MH_04: absolute tilt median 65-79 s 3.74 / 3.74 / 3.39°
    (at 70 s 3.65 / 3.65 / 3.27°); first loop `map_from_odometry` |t_y| 1.079 / 1.079 / 1.104 m; vertical RMS
    after SE(3) alignment causal 0.264 / 0.264 / 0.274 m, smoothed 0.106 / 0.106 / 0.102 m (the prediction is read
    on the causal output); causal 0.302 / 0.302 / 0.316 m; smoothed 0.226 / 0.226 / 0.240 m. Tracked 2032 / 2032 /
    2031 of 2032 / 2032 / 2031.
- **Kill rules:** K1 walk z holds and MH_04 tilt ≥ 3° at 70 s: the tilt is held by map rigidity; go to R5.
  K2 walk z holds, gain < 3 % everywhere: bias memory is not where EuRoC error enters (as card S on R_01).
  K3 any sequence worse > 15 %: knob stays off.
- **Predictions, written before the S1 runs (2026-11-23; the debug agent's recommended arm from D1-D3,
  commit 344feda, plus the R3 rows above; not re-worded after the runs).** Arm `S1` = S0 +
  `mapping: ( local_ba: ( boundary_scales_bias_rows: false ) )`, SYNC, 900 s cap; header must show
  `(bias rows full)` (`E/expect/S1.header`). Runs: V2_01, V1_02, MH_04, MH_01 (guard), V1_03, V2_02 × starts
  0/1/2, plus one DEBUG V2_01 run to frame 450 (`E/debug/D4`). Smoothed and causal are SE(3) ATE RMSE, body
  centre; "band" is the min..max of the 3 starts; tilt is absolute (as D1-D3).

  | id | prediction (S1) | S0 baseline | measured by |
  |---|---|---|---|
  | P3.1 | V2_01 b_a step per keyframe over 19-22.5 s ≤ 0.01 m/s² | 0.02-0.07 (D3) | D4, `[vi_ba] STATE` (`E/debug/solves.py`) |
  | P3.2 | V2_01 b_a error against GT at 22.3 s ≤ 0.3 m/s² | 0.63 (D3) | D4, `E/debug/viba_ba.py` |
  | P3.3 | V2_01 causal tilt median 22-60 s ≤ 2.5° | 5.5-5.8° (D3 agent); 5.67 / 6.80 / 6.05° (`r3metrics.py`) | `r3metrics.py` |
  | P3.4 | V2_01 smoothed ≤ 0.10 m | 0.109 / 0.124 / 0.110 | `score.py` |
  | P3.5 | V1_02 tilt median 50-75 s ≤ 1.5°, smoothed ≤ 0.056 m | 2.2-2.7° (agent); 2.32° × 3; smoothed 0.056 × 3 | `r3metrics.py`, `score.py` |
  | P3.6 | MH_01 smoothed ≤ 0.052 m (0.041-0.045 + 15 %) | 0.045 / 0.041 / 0.045 | `score.py` |
  | P3.7 | MH_04 as R3: tilt 65-79 s ≤ 1.5°, \|t_y\| ≤ 0.4 m, causal vertical RMS ≤ 0.10 m, causal ≤ 0.20 m, smoothed 0.08-0.13 m | see S0 values above | `r3metrics.py`, `score.py` |
  | P3.8 | accel and gyro walk z p50 ≤ 2.0 on all six (mechanism counter) | accel 6.3-19.9, gyro 2.0-3.9 | `sessbias.py` via `r3metrics.py` |
  | P3.9 | V1_03, V2_02 smoothed median within ±15 % of S0 | 0.146, 0.159 | `score.py` |
  | P3.10 | tracked unchanged (± 2 frames) on all six | `E/r0table.md` | `[summary]` |

  Kill rules:
  - KD1 b_a held (P3.1 step ≤ 0.01) and V2_01 tilt still ≥ 4° at 22-40 s: the tilt enters through the visual side
    and the frozen past (G56, §9 b2). Go to the visual log line, then R12.
  - KD2 b_a not held (step > 0.01): the 50-iteration-cap solves dominate; diagnose before R5.
  - K1, K2, K3 as above (K3: any of the six sequences' smoothed median worse than S0's by > 15 %).
  - Mechanism without score (theory incomplete): P3.8 and P3.1 pass but P3.4, P3.5 and P3.7 ATE all miss (K2 form).
    Score without mechanism (luck or another cause): any ATE gain > 15 % with walk z p50 and P3.1 unchanged.

  Recorded with no numeric prediction: χ² of the boundary IMU edges and iterations per solve (D4 against D3, and
  `vi_ba_base_iterations / vi_ba_base_solves` on every run), keyframe shifts, b_a error against GT on the final
  map keyframes (S0 medians: V2_01 0.89-1.05, V1_02 0.19, MH_04 0.15-0.17, MH_01 0.10-0.12, V1_03 0.19-0.30,
  V2_02 0.37-0.38 m/s²).

  Noise: the brief asked for an IMU time shift of ±50 µs per S0/S1 pair. The EuRoC source has no such knob at
  c21299b (`EurocConfig` holds `data`, `start_frame`, `max_frames` only, `deny_unknown_fields`;
  `imu_time_offset_ns` exists for LaMAria only, `A/main.rs:245-248`), so the band is the start offsets only.
  It is degenerate: V1_02 has 1 distinct sample, MH_04 2. Verdicts on V1_02 and MH_04 are provisional.

### Experiment R3b: accel-bias priors (added 2026-11-23, `experiments/vi-ba-vs-basalt.md` §9 b1, c3)

- **Gap tested:** G49 (tilt hidden in the accel bias). **Base:** S0; stacks with R3.
- **Mechanism:** VIBA0/1/2 put a zero-mean accel-bias prior of weight 1e5 (σ about 3 mm/s²) on b_a
  (`K/inertial/schedule.rs:28`, `K/inertial/initialization/solve.rs:166-173`), and each local solve adds 1e2 per
  free keyframe (σ 0.1 m/s², `K/mapping/vi_ba_schur.rs:2845-2854`). EuRoC's true b_a is about 0.15 m/s² and
  horizontal (y/z), so a b_a held at 0 leaves 0.82-0.92° of world tilt, and later solves can only trade
  `g·θ_map` into the bias.
- **Reference:** ORB-SLAM3 frees b_a at VIBA2 (`LocalMapping.cc:200-228`) and follows with `FullInertialBA`
  (kornia-slam#51); Basalt starts from calibrated b_a (within 0.03 m/s² of GT).
- **Knob:** none today; a small code arm: VIBA2 `prior_a` 0 plus local `accel_bias_prior_weight` 0. Card S
  stage B is not built.
- **Predictions (MH_04, V1_02, SYNC):** keyframe tilt median ≤ 0.5°; b_a error median ≤ 0.05 m/s²; ATE
  within the S0 band or better.
- **Kill:** b_a error unchanged with the priors off: the prior is not the force; the map rigidity
  (§9 b2) is. Tilt worse or ATE > band: the b_a/tilt direction is unobservable without the post-init full
  inertial BA; build kornia-slam#51 first.
- **Gate first:** debug run D1 (below).
- **Results D1 and D3 (2026-11-23, `E/debug/D1`, `E/debug/D3`; nothing below re-worded after the run).**
  Both runs exit 0 (13 s and 26 s wall). `traj.txt` is byte-identical to the S0 s0 run over the same prefix
  (400 and 450 poses), so `--debug` does not change the estimate. Scripts: `E/debug/viba_ba.py` (VIBA lines and
  `[vi_ba] STATE` b_a against GT, the frame taken from the last `frame=` tag), `E/debug/tilt_t.py` (absolute
  tilt from `traj.txt`), `E/debug/solves.py` (per-solve cost, interior IMU χ² and hw, b_a).

  | run | stage | frame / t (s) | b_a (m/s²) | GT b_a | error, y/z share (implied tilt) | world rotation | tilt after (deg) |
  |---|---|---|---|---|---|---|---|
  | D1 MH_01 | VIBA0 | 43 / 2.15 | (−0.039, 0.001, 0.015), norm 0.042 | (−0.025, 0.137, 0.076) | 0.149, yz 0.149 (0.87°) | 17.45° (first alignment) | 0.80-0.87 (2.0-6.0 s) |
  | D1 MH_01 | VIBA1 | 102 / 5.10 | (−0.050, 0.000, 0.018), norm 0.053 | same | 0.150 (0.86°) | 0.02° | 0.64-0.83 (5-7 s) |
  | D1 MH_01 | VIBA2 | 303 / 15.15 | (−0.028, 0.014, 0.090), norm 0.095 | same | 0.124 (0.72°) | 0.45° | 0.51-0.61 (15-19 s) |
  | D3 V2_01 | VIBA0 | 104 / 5.20 | (−0.026, 0.000, 0.008), norm 0.028 | (−0.024, 0.121, 0.075) | 0.138 (0.81°) | 15.29° | 0.81-0.89 |
  | D3 V2_01 | VIBA1 | 111 / 5.55 | (−0.031, −0.001, 0.010), norm 0.032 | same | 0.139 (0.81°) | 0.00° | 0.53-0.89 (5.5-10 s) |
  | D3 V2_01 | VIBA2 | 304 / 15.20 | (−0.015, 0.086, 0.130), norm 0.156 | same | 0.065 (0.37°) | 0.88° | 0.21-0.32 (15-18 s) |
  | D2 V1_02 | VIBA0 / 1 / 2 | 97, 123, 324 / 4.85, 6.15, 16.20 | norms 0.034 / 0.029 / 0.277; VIBA2 (−0.037, 0.275, −0.006) | (−0.013, 0.104, 0.093) | 0.134 / 0.127 / 0.198 (0.77 / 0.74 / 1.15°) | 18.58 / 0.17 / 1.94° | 0.62 (5-10 s), 1.24-1.33 (15-30 s) |

  | id | prediction | measured | verdict |
  |---|---|---|---|
  | D1.1 | \|b_a\| ≤ 0.02 at VIBA0/1/2 | 0.042 / 0.053 / 0.095 | missed. At VIBA0/1 b_a y/z stays near 0 (y 0.001 against GT 0.137), so the bias is pinned as the mechanism says, but not to 0.02. At VIBA2 the refutation rule (≥ 0.08) fires: VIBA2 is not held by the prior |
  | D1.2 | tilt over 1 s after VIBA2 0.72-1.12° | 0.51-0.61° | missed (lower). Tilt after VIBA0 is 0.80-0.87° against 0.87° implied by the b_a error: the init tilt is the bias error, to 0.01-0.07° |
  | D1.3 | VIBA1/2 rotate ≤ 1°, scale 0.98-1.02 | 0.02° / 0.45°, scale 1.0000 | as predicted |
  | D1.4 | VIBA2 before frame 400 | frame 303 | as predicted |
  | D3.1 | VIBA2 at 19.0-20.5 s | 15.20 s | refuted |
  | D3.2 | VIBA2 rotates the world ≥ 2° | 0.88°, and it lowers the b_a error 0.139 → 0.065 and the tilt to 0.21-0.32° | refuted |
  | D3.3 | b_a error jumps ≤ 0.15 → ≥ 0.8 within 1 s after VIBA2 | 0.06-0.08 for 3 s after VIBA2; the rise starts at 19.3 s: 0.09 (19.0) → 0.13 (19.35) → 0.18 (20.1) → 0.25 (20.7) → 0.31 (21.3) → 0.48 (21.9) → 0.63 (22.35) | refuted; kill fired (no VIBA event in 19-21 s) |

  **Diagnosis.**
  - The zero-mean prior story (§9 b1) holds for VIBA0 and VIBA1 only: they keep b_a y/z near 0, and the world is
    built with exactly the bias error as tilt (MH_01 0.87° measured against 0.87° implied; V2_01 0.81-0.89°).
    VIBA2 (prior_g 0, 15 s of data) is not prior-bound: |b_a| 0.095 / 0.156 / 0.277, and it moves the tilt both
    ways (MH_01 0.8 → 0.55°, V2_01 0.85 → 0.25°, V1_02 0.62 → 1.24-1.33°, an overshoot of b_a y to 0.275
    against GT 0.104). So the init prior costs about 0.3-0.6° for the first 10-15 s; after VIBA2 the error is
    VIBA2's own noise (one 15 s solve, no full inertial BA after it: kornia-slam#51), not the prior. Turning
    VIBA2's `prior_a` to 0 (half of the R3b arm) is not expected to change much; the local-BA prior half is
    untested.
  - The V2_01 20 s event is a local-BA event, not a VIBA one. From 19.3 s the b_a of each solve walks by
    0.01-0.07 m/s² per keyframe (0.15 s), 0.09 → 0.63 in 3 s. The visual cost rises first: the centre incidence
    bin's mean χ² goes 7 → 19 (20.25 s) → 35 → 49 → 77 px² (21.6 s), and every solve from 20.25 s hits the
    50-iteration cap (`[vi_ba] TIME iters=50`). Only after that, from 21.3 s, the newest interior IMU edge reaches
    χ² 377-4023 and Huber cuts it to hw 0.06-0.25. The tracker keeps 450-480 inliers until 21.0 s and drops to
    236-270 after 21.3 s. Tilt goes 0.2 (17 s) → 0.9 (19.5) → 1.8 (20.5) → 6.4° (21.5). IMU excitation is
    ordinary (gyro peak 0.26-0.65 rad/s, the 22 s peak is 1.04). A bias walk of 0.02-0.07 per 0.15 s is
    about 20-60σ of the datasheet accel random walk (3e-3 m/s³/√Hz) inside the full-weight interior edges, so the freedom comes from the 1e-2
    boundary bias rows (G49) plus the frozen past (§9 b2), with the visual disagreement as the driver.
  - Root candidates: b1 supported for VIBA0/1, small at stake (≤ 0.6° for 10-15 s); b2 + G49 (map rigidity and the
    loose boundary bias rows) supported as the V2_01 mechanism; the VIBA2 timing hypothesis refuted.
  - **Next SYNC arm (recommended): R3 `S1` (`boundary_scales_bias_rows: false`) on V2_01, V1_02, MH_04, MH_01 guard,
    3 starts, plus one DEBUG V2_01 run to frame 450.** Predictions against S0: V2_01 b_a step per keyframe over
    19-22.5 s ≤ 0.01 m/s² (S0 0.02-0.07), b_a error at 22.3 s ≤ 0.3 (S0 0.63), causal tilt median 22-60 s ≤ 2.5°
    (S0 5.5-5.8°), smoothed ≤ 0.10 (S0 band 0.109-0.124); V1_02 tilt median 50-75 s ≤ 1.5° (S0 2.2-2.7°), smoothed
    ≤ 0.056; MH_01 smoothed within 0.041-0.045 + 15 %; MH_04 as R3. Kill: b_a held (step ≤ 0.01) and V2_01 tilt
    still ≥ 4° at 22-40 s means the tilt enters through the visual side and the frozen past (G56, b2): go to
    the visual log line, then R12. b_a not held: the 50-iteration-cap solves dominate; diagnose before R5.
- **Missing from the debug output (no code edited):** the `[imu_init] VIBAn accepted` line
  (`K/system/mod.rs`, `refine_inertial_init`, about line 3313 at c21299b) and `[imu_init] accepted`
  (`K/inertial/initialization/solve.rs:462`) print no frame or timestamp; the frame was inferred from the last
  `frame=` tag (± 1 frame). `[vi_ba] STATE` (`K/mapping/vi_ba_schur.rs`, `solve_vi_ba`, about line 2992) prints the
  window-local `kf=` index, not the keyframe id or timestamp. Tilt is not logged anywhere; it was computed from
  `traj.txt`.

### Experiment R4: reference window and solver budget

- **Gap tested:** G6, G37. Run only on top of R3.
- **Knobs:** `active_keyframes: 10` (ORB-SLAM3 `Optimizer.cc:2387-2394`), `max_iterations: 7`,
  `initial_lambda: 1e-4` (Basalt `data/euroc_config.json`: `vio_max_iterations 7`, `vio_lm_lambda_initial 1e-4`).
  Defaults today: 3, 50, 1.0.
- **Predictions:** with E1 on, MH_03 within +10 % of window 3 (without E1 the preset is +48 %); gyro walk z
  p50 ≤ 2 (preset: 15.2); `vi_ba_solves_3plus_rejects` ≤ 5 % of solves (default V1_03: 32 %); mean smoothed ATE
  over 11: −5 to −20 %.
- **Kill rules:** MH_03 or MH_04 worse > 15 % with walk z ≤ 2: the window-10 harm is not the bias rows; keep
  window 3. Score gain with walk z > 2: luck or another cause; diagnose.

### Experiment R5: boundary model alternatives

- Run if R3 K1 fires or tilt remains. **Gap tested:** G49.
- **Knobs:** `boundary: Marginal(..)` (`K/mapping/marginalization.rs:44-52`; Basalt
  `sqrt_keypoint_vio.cpp:77-109, 552-760`) or `imu_huber: BoundaryOrRecentInit` (`K/mapping/vi_ba_schur.rs:297-306`;
  ORB-SLAM3 `Optimizer.cc:2632`). Pick one; it does not stack with a second boundary model.
- **Prediction:** MH_04 tilt at 65-79 s ≤ 1.5°.
- **Kill:** tilt unchanged: it enters through vision in the dark span; next is the dark-span observation count
  and R8.

### Experiment R6: global inertial BA at end of sequence

- Offline, on saved sessions; no SLAM run. **Gap tested:** A7 (G10); card AI's "refine" step.
- **Mechanism we lack online:** ORB-SLAM3 runs `FullInertialBA` (7 iterations) after every loop
  (`LoopClosing.cc:1206, 2268-2286`, `Optimizer.cc:392`). Our loop correction is 4-DoF and cannot remove tilt;
  MH_04 final-map keyframes keep 3.4° at 60-80 s.
- **Existing code:** `kornia-slam offline-ba --session .. --trajectory-out ..` (`A/main.rs:151, 649-680`,
  `K/mapping/global_vi_ba/`). Knob: the run file's `offline_ba` block; off = today.
- **Predictions:** MH_04 smoothed 0.165 to 0.06-0.10, final-map tilt at 60-80 s ≤ 1°; Vicon rooms smoothed
  −20 to −40 %; MH_01/MH_02 within ±0.005 m; each solve ≤ 60 s.
- **Kill rules:** dropped-observation share > 5 % (G41) or worse on ≥ 3 sequences: the seed and drop-ratchet
  fixes from card AI come first. Unverified: `offline-ba` has never run on a Stereo-mode EuRoC session.
- Affects the smoothed output only. Stacks on everything.

### Experiment R7: per-frame pose-inertial solve (card AJ)

- Code built in the `kornia-slam-aj` worktree (4 commits). **Gap tested:** G13, G30.
- **Knob:** `tracking.pose_inertial` (default Off; `K/system/config/mod.rs:481`); reference ORB-SLAM3
  `Tracking.cc:2982-2990`. Needs R3+R4 first: the solve is only as good as the anchor keyframe's bias.
- **Predictions (card AJ's, re-based on the stack):** per-sequence ATE −30 % to +5 %, mean −5 to −20 %, RPE 1 s
  −20 % on ≥ 6 of 11, rejects ≤ 1 % of frames. Card AJ's K1 and K3 stand. Target: the Vicon-room local error.
- **Bit-identity:** S0 with its knobs off on the AJ binary is `cmp`-identical to the base binary's S0.

### Experiment R8: two-camera flow (EuRoC rig mode)

- Code in progress in the `kornia-slam-eurocrig` worktree. **Gap tested:** G50, G51, G52. Its card stands, arms
  re-based on the stack and run in SYNC. Its own prediction is neutral for the default frontend (ratio
  0.85-1.15), so it is late. It matters most on V1_03/V2_03 (flow unavailable on up to 80 % of frames).

### Experiment R9: loops

- Observe first, from R0/R1 runs. Counters: `[pgo] failed ... not converged after 30 iterations` (4 on default
  V1_03 and V2_01), applied against not applied, S0 against S0-noloop.
- **Predictions on a healthy base:** PGO failures ≤ 1 per sequence; loops-on smoothed ≤ loops-off on ≥ 9 of 11.
- Only if a counter fires: `graph: ( information: Drift, persistent: true )` (`K/loop_closure/closer.rs:56-95`;
  ORB-SLAM3 `LoopClosing.cc:1178`).

### Experiment R10: residual audit against the references

- Python on the stack's runs; no build. Per sequence: 10 s-piece error, RPE 1 s, first-15 s error against
  ORB-SLAM3's VIBA1/VIBA2 timing (`LocalMapping.cc:200-224`), tilt and bias error over time, observations per
  keyframe. It names the next mechanism for the Vicon-room residual, which is unknown today.

### Experiment R11: error model (added 2026-11-23, `experiments/vi-ba-vs-basalt.md` §9 a2, a5, a6, b4)

- **Gap tested:** the 10 s-piece precision gap on healthy SYNC runs (keyframe-only pieces 1.5-2.5x Basalt).
- **Mechanism:** accel information is 64x Basalt's and gyro 2.8x; a Huber at χ² 16.9 sits on every IMU edge
  (Basalt has none); visual σ is 1 px against Basalt's 0.5, visual Huber at 5σ against 2σ; stereo depth σ is
  5 % of depth; Hold integration lags 2.5 ms (A5). When IMU and map disagree, the solve either bends the
  keyframes or switches the IMU edge off.
- **Arms (existing knobs, SYNC, each alone on S0):** `imu_huber: BoundaryOnly`; Basalt EuRoC IMU noise via
  `rig_calibration` (untested with `cameras: Stereo`: check the header); `stereo_depth_sigma: Disparity(px:
  0.5)`; `inertial: (sample_interpolation: Midpoint)`. Visual σ/Huber needs code (`huber_reproj_px_sq` is not a
  knob).
- **Predictions:** keyframe-only 10 s piece median −20 % or more on V1_02 (0.023 → ≤ 0.018) for the winning
  arm; MH_01 guard within the band.
- **Kill:** hw ≥ 0.9 on > 95 % of interior IMU edges (D2) rules out the IMU kernel arm.
- **Gate first:** debug run D2 (below).
- **Results D2 (2026-11-23, `E/debug/D2`, V1_02 full, 1688 poses, rc 0, 137 s; `traj.txt` byte-identical to
  S0 s0).** Script `E/debug/imu_chi2.py` over all 488 local solves (`[vi_ba] FINAL edge` and
  `[vi_ba] INCIDENCE` lines). The same script on D1/D3 is given for scale.

  | run | interior IMU edges: n, χ² p10/p50/p90/p99, hw < 1, hw ≥ 0.9 | boundary IMU edges: n, χ² p50/p90, hw < 1 | visual mean χ² per observation (px², σ 1 px, no Huber) |
  |---|---|---|---|
  | D2 V1_02 | 976; 0.8 / 4.0 / 14.7 / 2469; 8.2 %; 92.1 % | 488; 15.1 / 329; 45.1 % | 25.1 over 3.34 M |
  | D1 MH_01 (400 frames) | 180; 0.6 / 1.9 / 6.2 / 15; 0.0 %; 100 % | 90; 4.1 / 15.8; 8.9 % | 16.4 |
  | D3 V2_01 (450 frames) | 170; 1.4 / 4.8 / 16.8 / 3270; 9.4 %; 91.2 % | 85; 10.8 / 553; 30.6 % | 18.5 |

  | id | prediction | measured | verdict |
  |---|---|---|---|
  | D2.1 | interior hw < 1 on 15-45 % | 8.2 % | missed (lower). Below the report's 10 % action rule, and not the kill (hw ≥ 0.9 on 92.1 %, not > 95 %) |
  | D2.2 | interior χ² median 8-25 | 4.0 | missed (lower): typical interior IMU edges are not over-stiff (a consistent 9-dof model gives 8.3); the down-weighted 8 % are tail events (p99 2469) |
  | D2.3 | visual mean χ² 0.5-1.5 | 25.1 | refuted: the mean sits at the 5σ Huber threshold (25 px²), a heavy tail or a σ far from 1 px |

  **Diagnosis.**
  - The IMU kernel on interior edges is not the general lever: median χ² 4.0, 92 % at full weight. The IMU Huber
    cuts edges in bursts: on V1_02, 7-10 of 32-35 solves per 5 s carry a cut edge at 5-15 s, 45-50 s and
    70-80 s, 0-5 elsewhere. On V2_01 the cuts follow the visual breakdown (R3b results). So `imu_huber: BoundaryOnly` would
    act only in those bursts, where it would force the IMU against a map that has gone wrong.
  - The disagreement sits at the boundary: boundary χ² median 15.1 against 4.0 interior, 45 % down-weighted. The
    boundary edge ties the window to the frozen past, so this supports §9 a1/b2 (frozen context, R4/R12) and
    G49, not the IMU noise level.
  - Visual: a mean of 25 px² per observation is far from the σ = 1 px model and from Basalt's 0.5 px. Whether it
    is a tail (mismatches near the 5 px Huber) or a broad core is not in the log (below). This supports the visual
    half of R11 (σ/Huber), which needs code.
  - V1_02 also shows the tilt growing when b_a drifts: 0.85-1.33° until 45 s, 2.2-2.7° at 50-75 s with the b_a
    error 0.30-0.37, back to 0.2-0.6° after 75 s.
- **Missing from the debug output (no code edited):** the visual residual distribution. `[vi_ba] INCIDENCE`
  (`K/mapping/vi_ba_schur.rs`, `solve_vi_ba`, about lines 2955-2988 at c21299b, `incidence_chi2_line` at 3172)
  prints only the mean whitened χ² per incidence bin, before Huber. The median, the share with
  `chi2_vis > huber_reproj_px_sq` (the inlier flag set at about line 2082 is not printed) and the visual Huber
  weights would need one more line in that block. Visual Huber shares per solve are therefore not measured.
- **Results G56 (2026-11-23, `E/debug/G56a`, `E/debug/G56b`).** The line was added on branch `diag/g56`
  (27a654c, `[vi_ba] VISRES`, debug only, in the INCIDENCE block). Binary md5 `7352d0c311b3509186fd06c037e8d2b8`.
  Runs: S0 config, `--debug`, `KORNIA_SLAM_DEBUG=1`.
  - **Bit-identity:**
    - V1_02 full `traj.txt` is `cmp`-identical to `E/runs/S0_V1_02_medium_s0_r1`.
    - MH_01 (`max_frames: 400`) is `cmp`-identical to `E/debug/D1`.
    - R_01 knobs-off (`P/baseline/r01full-S10-7.ron`, `RAYON_NUM_THREADS=1`, non-debug) gives md5
      d4f77b567b912b5a4caa2af6085a35b1, `cmp`-identical to `P/baseline/traj.txt`.
  - **Script:** `E/debug/visres.py`. Values are at the final iterate, before Huber, at σ 1 px:
    - `obs`, mean χ², `out` (share above 25 px²) and `in_mean` are weighted over all observations of all solves;
    - med, p90 and p99 are the median over solves of each solve's quantile.
  - **Groups:**
    - "young": the oldest keyframe in the solve that sees the landmark is free;
    - "fixed": the observing keyframe is fixed context.

  | run | group | obs (share) | mean χ² | out | in_mean | med / p90 / p99 px |
  |---|---|---|---|---|---|---|
  | V1_02 (488 solves) | all | 3.34 M | 25.11 | 11.8 % | 3.40 | 1.08 / 4.22 / 18.44 |
  | | free pose | 33.5 % | 17.41 | 7.6 % | 2.25 | 0.79 / 3.44 / 18.22 |
  | | fixed pose | 66.5 % | 28.99 | 13.9 % | 4.02 | 1.25 / 4.52 / 18.65 |
  | | young landmark | 12.8 % | 4.15 | 1.7 % | 0.98 | 0.23 / 1.53 / 6.27 |
  | | old landmark | 87.2 % | 28.19 | 13.2 % | 3.80 | 1.20 / 4.62 / 19.56 |
  | | stereo depth rows (σ) | 2.88 M | 2.26 | 2.2 % (> 3σ) | 0.48 | 0.30 / 1.13 / 3.54 |
  | MH_01 400 (90 solves) | all | 0.70 M | 16.44 | 8.4 % | 3.35 | 1.23 / 4.05 / 18.52 |
  | | free pose | 18.9 % | 13.11 | 6.2 % | 2.37 | 0.87 / 3.57 / 17.32 |
  | | fixed pose | 81.1 % | 17.21 | 8.9 % | 3.58 | 1.31 / 4.12 / 19.24 |
  | | young landmark | 5.2 % | 1.61 | 0.8 % | 0.57 | 0.00 / 1.03 / 4.55 |
  | | old landmark | 94.8 % | 17.26 | 8.8 % | 3.51 | 1.31 / 4.18 / 19.44 |
  | | stereo depth rows (σ) | 0.44 M | 1.10 | 0.7 % (> 3σ) | 0.25 | 0.20 / 0.81 / 2.34 |

  - **Other counts:**
    - Observations behind the camera: 0 in every solve.
    - Right-camera rows: none. EuRoC `Stereo` adds a depth row to the left observation
      (`K/mapping/bundle_adjustment.rs:1440-1455`).
- **Verdict G56: the 25 px² is mostly an artefact of the statistic, with a real outlier tail underneath.**
  - **The core is not 10x off.**
    - The per-solve median residual is 1.1-1.2 px.
    - The inlier mean χ² is 3.4 against 2 for a 2-dof σ = 1 px model. That gives a core σ of about 1.3 px.
  - **The mean is carried by a tail.**
    - 8-12 % of observations sit above the 5 px Huber threshold, with p99 of 18-19 px.
    - The tail sits on old landmarks: 13 % out, against 1.7 % on young ones.
    - It sits on fixed-context keyframes: 14 % out, against 7.6 % on free ones.
    - The mean is the mean of a heavy tail. Huber's linear ρ still lets those rows pull: on D2 the final visual
      cost per observation is 7.6, so mean ρ is about 15.
  - **Ruled out by code read:**
    - units: the undistorted pixel uses the feature's own camera (`K/frame.rs:138-177`), and the pinhole
      residual uses the same fx, fy, cx, cy (`K/mapping/vi_ba_schur.rs:685-710`);
    - a double square: χ² is r²/σ² once, and the inlier flag is at about line 2082;
    - stereo rows: there are none;
    - f32 world coordinates: about 1e-4 px at 10 m;
    - cheirality: the behind-camera count is 0.
  - **Stereo depth rows are over-dispersed below 1** (mean 1.1-2.3 in σ, median 0.2-0.3): the 5 %-of-depth σ is
    loose, not tight.
- **Cause: no observation is ever erased for its reprojection error.**
  - **What the local VI-BA does after its solve:**
    - It writes back poses, velocities and biases. Landmarks go back through the guard
      `landmarks_to_write_back` (`K/mapping/bundle_adjustment.rs:941-964, 1204-`). That guard keeps a landmark
      at its old position when it went behind a camera or jumped; it unlinks nothing.
    - The merge (`K/mapping/map/ops/correction.rs:674`) unlinks nothing either.
    - `merge_and_cull` (`K/mapping/local_mapping.rs:287-291`) then runs `cull_landmarks`
      (`K/mapping/culling.rs:28-87`). That removes whole landmarks only for three reasons:
      - a found ratio below 0.20 after 5 views;
      - behind an observing keyframe;
      - farther than 1e4 m.
    - It has no reprojection test.
  - **No production caller unlinks an observation:** `Map::unlink_observation`
    (`K/mapping/map/ops/mutation.rs:323`) is called only from tests.
  - **What ORB-SLAM3 does:** after `LocalInertialBA` it erases every edge above χ² 5.991, or 1.5x that for points
    closer than 10 m, or with depth ≤ 0 (`Optimizer.cc:2691, 2850-2910`). Edges to fixed keyframes are built and
    erased the same way (`:2718-2725`).
  - **Why the tail survives here:**
    - A mismatch that enters a keyframe stays linked for the landmark's life. It joins every later window, first
      on a free keyframe, then as fixed context.
    - So the tail grows with landmark age, and sits mostly on fixed keyframes (66-81 % of rows).
    - The found-ratio cull does not catch it: the landmark is still found by tracking through its good views.
  - **Basalt reaches a low tail by another route** (`BA/data/euroc_config.json`):
    - VIO never calls `filterOutliers` (`sqrt_keypoint_vio.cpp:1405`, TODO).
    - Its KLT frontend keeps only tracks whose round trip is within 0.2 px (`optical_flow_max_recovered_dist2`
      0.04), with an epipolar gate of 0.005.
    - Its Huber is 1 px on the raw residual at σ 0.5 px (`landmark_block_abs_dynamic.hpp:171-174, 478-484`).
    - Its residual distribution is not derivable without a Basalt run that logs it.

### Experiment R11b: post-solve outlier erase (G56; predictions written 2026-11-23 before code)

- **Gap tested:** G56, and through it the 10 s-piece precision gap (keyframe-only pieces 1.5-2.5x Basalt).
- **Mechanism:** about 10 % of local VI-BA reprojection rows are mismatches at 5-20 px that are never unlinked
  (above). Under Huber they keep pulling the window toward the frozen past, at a cost of about 15 per row on
  V1_02.
- **Reference:** ORB-SLAM3 `LocalInertialBA` erase (`Optimizer.cc:2850-2910`), over local and fixed keyframes
  alike. It is chosen over Basalt's frontend gate because our ORB/descriptor matches already enter the map; an
  erase after the solve is the cheaper place to catch them.
- **Knob:** `local_ba: ( outlier_erase: Disabled | Chi2(threshold: 5.991, close_factor: 1.5, close_depth_m: 10.0) )`,
  default `Disabled`.
  - After an accepted merge, unlink (through `Map::unlink_observation`) each window observation whose final χ²
    exceeds the threshold, or whose depth is ≤ 0.
  - Fixed keyframes are included.
  - A landmark left without observations is retired by the existing path.
  - The threshold is at σ = 1 px; the 1.3 px core is the separate octave arm below.
- **Second arm, alone:** `visual_information: OctaveScaled` (existing knob), for the 1.3 px core.
- **Predictions (S0 + knob, SYNC; VISRES on a debug run, scores on the gate protocol):**
  - P1: V1_02 VISRES `out` 11.8 % → ≤ 3 %, mean χ² 25.1 → ≤ 5, `in_mean` 3.4 → 2.0-2.8.
  - P2: MH_01 400 `out` 8.4 % → ≤ 2 %, mean χ² 16.4 → ≤ 4.
  - P3 (mechanism counter): unlinked observations per keyframe 5-15 % of the new keyframe's rows in the first
    solve after insertion, falling below 2 % per solve after.
  - P4: V1_02 keyframe-only 10 s-piece median 0.023 → ≤ 0.019.
  - P5: MH_01 inside its S0 band. No sequence lost or above its S0 band (3 starts).
- **Kill rules:**
  - K1 (mechanism moved, score did not): `out` ≤ 3 % but P4 is not met. The tail is not the precision gap.
    Record G56 as fixed for the statistic and refuted as a score source; go to R12/G49.
  - K2 (score moved, mechanism did not): P4 is met while `out` stays above 6 %. Another cause; do not credit.
  - K3 (over-culling): tracking loss, or the keyframe count up more than 10 %, or map points per keyframe down
    more than 25 % on any sequence. The threshold is too tight at σ 1 px: retry at 2x, or after the octave arm.
- **Bit-identity:** with the knob off, `cmp`-identical on S0 V1_02 (`E/runs/S0_V1_02_medium_s0_r1`) and on R_01
  (`P/baseline/r01full-S10-7.ron`, `RAYON_NUM_THREADS=1`, md5 d4f77b567b912b5a4caa2af6085a35b1).
- **Noise:** SYNC repeats are bit-identical. Scores use 3 starts per sequence (the gate protocol); a full-length
  promotion needs the gate.

### Experiment R12: Basalt-style windowed marginalization (added 2026-11-23)

- Only if R4 on R3 does not close the keyframe-only piece gap. Code project: dense square-root
  visual + inertial prior with FEJ and a non-frozen context (Basalt `sqrt_keypoint_vio.cpp:552-760`), in place
  of 3 keyframes clamped to a frozen past and a 1e-2 boundary edge (`experiments/vi-ba-vs-basalt.md` §4, §9 a1,
  b2, c2). Predictions are written when R4 on R3 has a verdict.

### Debug runs D1-D3 (before R3b and R11; predictions written 2026-11-23 before the runs)

Binary md5 `cea7646298a68ddb948b16835472b2e3`, S0 config (`E/mkcfg.sh S0`) plus `max_frames` where stated,
`--no-tui --no-pace --debug`, `KORNIA_SLAM_DEBUG=1`, 900 s cap; output `E/debug/<id>/`. No score claims: these
runs are instruments. Tilt here is absolute: the angle between our gravity direction in the body frame
(`R_wb^T·(0,1,0)`, our world has gravity on +Y after VIBA0) and GT's (`R_wb,GT^T·(0,0,−1)`), with no
alignment; it carries the 0.16-0.49° hand-eye floor (report §10.2). Source of the VIBA values: the
`[imu_init] accepted ... gravity=(..) bg=(..) ba=(..)` and `[imu_init] VIBAn accepted: scale_correction=..` lines;
the world rotation of a VIBA is the angle between its printed gravity and the current world gravity (0, 9.81, 0).

- **D1, MH_01_easy, `max_frames: 400` (R3b).**
  - D1.1 |b_a| ≤ 0.02 m/s² at VIBA0, VIBA1 and VIBA2 (GT |b_a| 0.158, |b_a,yz| 0.157).
  - D1.2 tilt over the 1 s after VIBA2: 0.72-1.12° (|b_a,GT,yz|/g = 0.92° ± 0.2°).
  - D1.3 VIBA1 and VIBA2 rotate the world by ≤ 1° each; scale correction 0.98-1.02.
  - D1.4 VIBA2 fires before frame 400 (VIBA0 + 15 s). If it does not, D1.2 is read at VIBA1.
  - Refuted if: |b_a| ≥ 0.08 at VIBA2 (the prior does not pin b_a), or tilt ≤ 0.4° (the bias is not forced
    into tilt at init).
- **D2, V1_02_medium, full length (R11).**
  - D2.1 interior IMU edges (`[vi_ba] FINAL edge ... boundary=false`) with hw < 1: 15-45 % (LaMAria R_01:
    34-41 %, mostly clock; EuRoC's clock is better, the noise is datasheet).
  - D2.2 interior IMU χ² median 8-25 (9 dof: a consistent model gives about 8.3).
  - D2.3 visual: mean whitened reprojection χ² per observation (from `[vi_ba] INCIDENCE`) 0.5-1.5 at σ 1 px.
  - Action rule (report): > 10 % with hw < 1 opens the `imu_huber: BoundaryOnly` arm. Kill: hw ≥ 0.9 on > 95 %.
- **D3, V2_01_easy, `max_frames: 450` (R3b, map rigidity).**
  - D3.1 VIBA2 fires at 19.0-20.5 s (sequence time from the first frame).
  - D3.2 VIBA2 rotates the world by ≥ 2°.
  - D3.3 the local-BA b_a error against GT (`[vi_ba] STATE` lines) jumps from ≤ 0.15 to ≥ 0.8 m/s² within 1 s
    after VIBA2.
  - Kill: no VIBA event within 19-21 s means a visual event; then read the window logs around kf 382-418.

## 4. EuRoC gate

- **"The EuRoC baseline":** `EB0` = repo default knobs + `execution: Synchronous` (A/B instrument) and `EA1` =
  default + the R1 merge fix (async, paced). After each passed gate the winner stack becomes `EB(n)`/`EA(n)`.
  Configs live in `E`; `configs/euroc-stereo-imu-loop.ron` changes only on the owner's go.
- **Gate runs (once per batch of stacked winners):** candidate on all 11, SYNC × 3 starts (33 runs) and ASYNC
  paced × 2 (load ≤ 6).
- **Pass rule:** on every sequence, for causal and smoothed, the candidate's median of 3 is not above the
  baseline band's max; ASYNC values are not above the baseline's async range; `tracked == frames`, give-ups 0,
  re-seeds 0; `ba_keyframe_shifts ≤ 10`; config header check clean on every run.
- **Milestones:** M1 = below Basalt on all 11 (smoothed ranks, loops on, SYNC; ASYNC paced published beside it).
  M2 = below the ORB-SLAM3 column on all 11, scored on the full sequence and on ORB-SLAM3's span.

## SLAM Dunk on EuRoC: theoretical comparison (estimate, not a measurement)

Owner, 2026-11-23: no SLAM Dunk run; reasoning only. Nothing here was measured on EuRoC.

- **Its recipe** (from reading the repo, `experiments/lamaria-handoff.md`): a Basalt-based VIO seeds the
  trajectory; an offline global VI-BA over the whole recording (Basalt's square-root solver) refines it; then a
  camera state per 5 s chunk (camera-1-to-camera-0 pose and both focal lengths, random-walk prior) and one full
  3x3 gyro matrix.
- **Seed.** The VIO seed is Basalt-class: our Basalt column (0.039-0.230 m). The global BA starts there.
- **Global BA on EuRoC.** A batch VI-BA over all keyframes of a 1.5-3 min sequence, with loop and covisibility
  constraints, is the same class of solve as ORB-SLAM3's `FullInertialBA` after loops. ORB-SLAM3 stereo-inertial
  reaches 0.019-0.050 m (README, 80-96 % of frames). So the expected SLAM Dunk band after global BA is
  0.02-0.05 m per sequence, mean about 0.03-0.035 m: ORB-SLAM3-with-full-BA class or below. It removes the
  failure our 4-DoF loop correction cannot (MH_04 tilt), and it brings V2_03 from Basalt's 0.230 into that
  band if the seed does not lose track.
- **Time-varying camera calibration: expected gain about 0 on EuRoC.** On LaMAria it gave +8.0 CP because the
  Aria glasses frame flexes and the focal length follows IMU temperature over long recordings. EuRoC is a rigid
  VI-sensor with a Kalibr calibration and sequences of minutes; per-5 s camera states mostly add freedom there.
  Estimate: ≤ 0.002 m, inside our noise band.
- **Gyro 3x3 intrinsics: expected gain small.** The ADIS16448 is factory-calibrated; a residual scale or
  misalignment of about 0.1 % over a few hundred degrees of rotation is a few hundredths of a degree of heading,
  well under a millimetre over EuRoC path lengths. Estimate: ≤ 0.001-0.003 m.
- **The bar it implies for us:** M2 (below ORB-SLAM3) is the SLAM-Dunk-class bar on EuRoC. The rung that buys
  it is R6 (global inertial BA on saved sessions), not calibration states; calibration states (gap A8) stay a
  LaMAria lever.

## Owner answers (2026-11-23)

1. Smoothed ranks (ORB-SLAM3's convention); causal is reported and gated for no regression; "causal below
   Basalt" is a stretch goal after M1.
2. Loops on when comparing with Basalt (system against system), loops-off arm printed beside it.
3. "100 % tracking" = `tracked == frames`, give-ups 0, re-seeds 0. Start-up frames before the first stereo map
   count as misses; `min_cold_start_points` gets its own small rung if S0 skips any.
4. SYNC carries M1; ASYNC paced must pass the gate and is published beside it.
5. ORB-SLAM3 bar: README numbers, ours scored on both spans; rerun ORB-SLAM3 3 times per sequence only when M1
   is done.
6. Basalt is not rerun now; rerun before any public claim.
7. One config for all 11, no per-sequence values.
8. The code default merge stays `SnapshotOnly`; EuRoC configs carry `PropagateToNewer`; the cleanup card changes
   the default after the gate plus a LaMAria guard.
9. The "no kornia-slam merges until LaMAria top-3" rule needs the owner's restatement; this card merges nothing.
10. Card AJ's mono+IMU arms and the queued async `e1` runs (except MH_04) are dropped.

Stop points: S0 card review (approved). S1 after R0. S2 base configs. S3 first batch gate. S4 at M1. Also stop
on any fired kill, any run over 300 s, a non-identical knobs-off binary, and before any push, PR, merge, preset
or code-default change.

## Results

(Filled as runs land; nothing below is re-worded after the run.)

### Results: body-centre re-score of the async unpaced runs (2026-11-23)

`E/rescore_old.py` (python only, read-only on `P/euroc/runs/*_r1`): SE(3) ATE RMSE, m, `eval.py --fmt tum`.
Causal = `traj.txt`. Smoothed: "app" = the run's own `--trajectory-smoothed` output, where the driver passed it;
"proxy" = `E/proxy.py` (a copy of `P/euroc/recover.py`: frames re-anchored on the final keyframe poses of
`session.cbor`). One run per cell, load 8-24. Scale = Sim(3) scale of the causal fit.

| seq | default causal | default smoothed (app / proxy) | default scale | preset causal | preset smoothed (app / proxy) | preset scale | Basalt | ORB-SLAM3 |
|---|---|---|---|---|---|---|---|---|
| MH_01_easy | 0.080 | - / 0.041 | 1.008 | 0.083 | - / 0.034 | 1.006 | 0.067 | 0.046 |
| MH_02_easy | 0.064 | - / 0.027 | 1.002 | 0.052 | - / 0.025 | 0.998 | 0.048 | 0.034 |
| MH_03_medium | 0.048 | - / 0.050 | 1.006 | 0.070 | - / 0.085 | 0.999 | 0.062 | 0.028 |
| MH_04_difficult | 0.306 | - / 0.165 | 1.008 | 0.471 | - / 0.213 | 0.998 | 0.114 | 0.045 |
| MH_05_difficult | 0.178 | - / 0.080 | 1.008 | 0.164 | - / 0.107 | 1.003 | 0.144 | 0.050 |
| V1_01_easy | 0.141 | - / 0.137 | 1.007 | 0.074 | - / 0.059 | 1.016 | 0.043 | 0.039 |
| V1_02_medium | 2.397 | - / 1.005 | 0.369 | 0.095 | - / 0.065 | 1.002 | 0.045 | 0.019 |
| V1_03_difficult | 114.535 | - / 112.044 | 0.003 | 0.203 | - / 0.077 | 0.988 | 0.053 | 0.026 |
| V2_01_easy | 19.073 | - / 18.986 | 0.059 | 0.105 | 0.080 / 0.080 | 1.013 | 0.039 | 0.031 |
| V2_02_medium | 59.607 | 59.947 / 59.947 | 0.005 | 0.103 | 0.055 / 0.055 | 1.028 | 0.049 | 0.037 |
| V2_03_difficult | 242.678 | 242.457 / 242.457 | 0.001 | 0.658 | 0.462 / 0.462 | 0.969 | 0.230 | 0.027 |

- The MH and preset rows reproduce the gap table. Two proxy cells differ from the planner's proxy: default
  V1_02 1.005 (planner 1.22) and preset V1_03 0.077 (planner 0.066); both are far from any bar.
- **Proxy against the app output, same session:** identical to 3 decimals on all 5 runs that have both
  (preset V2_01/V2_02/V2_03, default V2_02/V2_03). This is P0.4's question answered on old async runs; P0.4 itself
  stays open for S0.
- **V2_02 and V2_03 (new):** default fails both (59.6 / 242.7 m, scale 0.005 / 0.001: G54). The preset scores
  0.103 / 0.658 causal and 0.055 / 0.462 smoothed. V2_03 smoothed is 2.0x Basalt (0.230), the worst preset ratio.

### Results: R0 determinism check, stopped by K0.2 (2026-11-23)

Harness `E/drive.sh` (SYNC: `RAYON_NUM_THREADS=1`, `--no-pace`, `timeout 300`), binary md5
`cea7646298a68ddb948b16835472b2e3`. Header check clean on both runs (`E/expect/S0.header`: `local mapping:
synchronous`, no `local BA:` line).

| run | rc | wall s | frames done / total | sum of per-frame ms | frames > 100 ms (time) | load at start |
|---|---|---|---|---|---|---|
| `S0_MH_04_difficult_s0_r1` | 124 | 300 | 1666 / 2033 | 161 s | 377 (145 s) | 6.2 |
| `S0_V1_02_medium_s0_r1` | 124 | 300 | 1479 / 1710 | 187 s | 415 (171 s) | 24.8 |

- **K0.2 fired: both runs hit the 300 s cap.** Linear projection to the full sequence: MH_04 about 366 s,
  V1_02 about 347 s, at load 20-25 on 12 cores. The determinism pairs (r2) were not run; P0.1 has no verdict.
- **Diagnosis (why):** the cost is the synchronous local BA in the frame path: 90-92 % of the logged frame time
  sits in the 23-28 % of frames above 100 ms (p90 336-438 ms, p99 909-927 ms). The rest of the wall (113-139 s)
  is not in the per-frame figure (start-up and work outside the timed span; not split yet). The one-thread pin
  is the difference from the earlier SYNC run: the other agent's T3 run (`P/euroc/runs/t3_V1_02_medium_r1`, the
  same config without the pin and without `start_frame`) finished V1_02 in 190 s (load 14.7 at start, 4.8 at end).
- **Determinism evidence, not P0.1:** the first 1432 poses of `S0_V1_02_medium_s0_r1/traj.txt` are
  byte-identical (`cmp`) to the first 1432 lines of T3's `traj.txt`, which ran with all rayon threads at a
  different load. So up to 84 % of V1_02 the SYNC output does not depend on the thread count.
- **T3 scored body-centre (read only), V1_02 SYNC unpinned:** causal 0.102, smoothed 0.056 (Basalt 0.045),
  Sim(3) scale 1.011, `ba_stale_keyframes=0`, `ba_keyframe_shifts=0`, give-ups 0, tracked 1688 of 1710 (22
  start-up frames skipped). So G54 is gone in SYNC on V1_02 (K0.3 does not fire there), causal sits 0.002 above
  P0.2's 0.05-0.10 range, and P0.3 would fail on the 22 start-up frames (owner answer 3: a
  `min_cold_start_points` rung).
- **Next (owner's call at Stop S1):** the plan's candidate is R4's Basalt solver budget in the base; the T3
  evidence also points at the thread pin itself (dropping it kept the output identical over the compared prefix).

### Results: R0 determinism pairs unpinned, K0.2 fired again (2026-11-23)

Protocol after the change: all rayon threads, synchronous mapping, `--no-pace`, `timeout 300`; two drivers in
parallel (one per pair, the 2-job throttle held). Header check clean on all four runs.

| run | rc | wall s | frames reached / total | poses written | load at start | load at end |
|---|---|---|---|---|---|---|
| `S0_MH_04_difficult_s0_r1` | 124 | 300 | 1095 / 2033 | 1052 | 15.4 | 33.4 |
| `S0_MH_04_difficult_s0_r2` | 124 | 300 | 1300 / 2033 | 1279 | 33.4 | 20.1 |
| `S0_V1_02_medium_s0_r1` | 124 | 301 | 628 / 1710 | 603 | 14.8 | 31.2 |
| `S0_V1_02_medium_s0_r2` | 124 | 300 | 779 / 1710 | 753 | 31.2 | 18.3 |

- **K0.2 fired again; no run finished, so P0.1 has no verdict on full outputs.** `traj_smoothed.txt` is written
  at shutdown, so none exists to `cmp`.
- **Prefix evidence (`cmp` on the common prefix of `traj.txt`, silent in every case):** MH_04 r1 against r2
  over 1052 poses; V1_02 r1 against r2 over 603; each unpinned r1 against the pinned run over the same prefix
  (1052 and 603 poses). No divergence was found anywhere both runs reached.
- **Why the cap fired (diagnosis):** machine load, not the thread count. The machine ran at load 15-33 on 12
  cores, from other sessions (cargo/rustc, flux-web, python, the other EuRoC driver). Unpinned runs reached fewer
  frames in 300 s than the pinned ones did at load 6-25 (MH_04 1095-1300 against 1666; V1_02 628-779 against
  1479): with all threads the process competes for cores it cannot get. T3's 190 s for V1_02 unpinned ran at
  load 14.7 falling to 4.8. So under this load neither variant fits 300 s; the 1.9x pin cost was measured at
  a different load and is not confirmed.
- **Next (owner's call):** a quiet machine (load ≤ 6) for R0, a larger cap for SYNC runs (outputs do not depend
  on load, only wall does), or R4's solver budget in the base.

### Results: P0.1 determinism, 900 s SYNC cap (2026-11-23)

Unpinned SYNC, cap 900 s, load 3-10. Header check clean on all four runs.

| run | rc | wall s | load at start |
|---|---|---|---|
| `S0_MH_04_difficult_s0_r1` | 0 | 116 | 3.1 |
| `S0_MH_04_difficult_s0_r2` | 0 | 116 | 8.8 |
| `S0_V1_02_medium_s0_r1` | 0 | 170 | 3.1 |
| `S0_V1_02_medium_s0_r2` | 0 | 132 | 10.2 |

- **P0.1 as predicted:** `cmp` prints nothing for `traj.txt` and `traj_smoothed.txt` on both pairs. K0.1 does
  not fire.
- At load 3-10 every wall is well under 300 s: the earlier K0.2 stops were caused by load 15-33, not by the
  pipeline. R0 arms started 14:21.

### Results: R0, S0 × 3 starts and S0-noloop × 1 on all 11 (2026-11-23, Stop S1)

SYNC unpinned, cap 900 s, binary md5 `cea7646298a68ddb948b16835472b2e3`, 14:16-15:10, load 3-10 at start.
46 runs, all `rc=0`, header check clean on every run (no `invalid`). Scores: `E/score.py` (`E/scores.jsonl`),
table `E/r0report.py` (`E/r0table.md`). SE(3) ATE RMSE, m, body centre. "ORB span" = ours scored on the frames
ORB-SLAM3 kept. Ratios use the smoothed median of the 3 starts.

| seq | S0 causal s0 / s1 / s2 | S0 smoothed s0 / s1 / s2 | smoothed median | noloop causal / smoothed | Basalt | ORB-SLAM3 (README span) | ours smoothed on ORB span (s0) | sm ÷ Basalt | sm ÷ ORB-SLAM3 |
|---|---|---|---|---|---|---|---|---|---|
| MH_01_easy | 0.090 / 0.086 / 0.094 | 0.045 / 0.041 / 0.045 | 0.045 | 0.116 / 0.116 | 0.067 | 0.046 | 0.045 | 0.67 | 0.98 |
| MH_02_easy | 0.057 / 0.060 / 0.070 | 0.035 / 0.028 / 0.028 | 0.028 | 0.060 / 0.060 | 0.048 | 0.034 | 0.035 | 0.58 | 0.82 |
| MH_03_medium | 0.073 / 0.062 / 0.071 | 0.041 / 0.048 / 0.052 | 0.048 | 0.091 / 0.090 | 0.062 | 0.028 | 0.044 | 0.77 | 1.73 |
| MH_04_difficult | 0.302 / 0.302 / 0.316 | 0.226 / 0.226 / 0.240 | 0.226 | 0.609 / 0.609 | 0.114 | 0.045 | 0.229 | 1.98 | 4.96 |
| MH_05_difficult | 0.174 / 0.231 / 0.178 | 0.082 / 0.088 / 0.079 | 0.082 | 0.295 / 0.295 | 0.144 | 0.050 | 0.089 | 0.57 | 1.64 |
| V1_01_easy | 0.090 / 0.079 / 0.071 | 0.088 / 0.068 / 0.064 | 0.068 | 0.288 / 0.288 | 0.043 | 0.039 | 0.089 | 1.57 | 1.76 |
| V1_02_medium | 0.102 / 0.102 / 0.102 | 0.056 / 0.056 / 0.056 | 0.056 | 0.212 / 0.215 | 0.045 | 0.019 | 0.054 | 1.23 | 2.88 |
| V1_03_difficult | 0.181 / 0.201 / 0.188 | 0.143 / 0.146 / 0.150 | 0.146 | 0.203 / 0.202 | 0.053 | 0.026 | 0.147 | 2.73 | 5.56 |
| V2_01_easy | 0.193 / 0.213 / 0.200 | 0.109 / 0.124 / 0.110 | 0.110 | 0.294 / 0.294 | 0.039 | 0.031 | 0.109 | 2.82 | 3.52 |
| V2_02_medium | 0.176 / 0.162 / 0.161 | 0.097 / 0.160 / 0.159 | 0.159 | 0.179 / 0.178 | 0.049 | 0.037 | 0.096 | 3.23 | 4.24 |
| V2_03_difficult | 0.605 / 0.605 / 0.869 | 0.436 / 0.436 / 0.271 | 0.436 | 1.148 / 1.140 | 0.230 | 0.027 | 0.446 | 1.90 | 16.32 |

Per-run counters (frames, tracked, stale and propagated keyframes, keyframe shifts, scale, loops, PGO
failures): `E/r0table.md`, second table.

| P/K | prediction | baseline | measured | verdict |
|---|---|---|---|---|
| P0.1 | `cmp`-identical repeats | - | MH_04 and V1_02 pairs: `traj.txt` and `traj_smoothed.txt` identical | as predicted |
| P0.2 | stale = 0 and shifts ≤ 5 on all 11; Vicon scale 0.98-1.02; V1_02 causal 0.05-0.10 | async default: stale 346-376, shifts 84-161 in Vicon rooms | stale 0 on all 46 runs. Shifts 0 except V1_03 6/6/6 (noloop 8) and V2_03 9/9/11 (noloop 8). Scale: V1_01 1.015-1.017, V1_02 1.011, V2_01 1.011-1.014 in range; V1_03 0.967-0.975, V2_02 1.017-1.043, V2_03 0.920-1.046 out. V1_02 causal 0.102 ×3 | missed (shifts on V1_03/V2_03, scale on V1_03/V2_02/V2_03, V1_02 causal by 0.002) |
| P0.3 | `tracked == frames` on all 11, give-ups 0 | - | give-ups 0 and re-seeds 0 everywhere. tracked < frames on V1_02 (20-22 start-up `skipped_bootstrap`), V1_03 (1-2 inertial coasts), V2_01 (5-6), V2_03 (89-94) | missed |
| P0.4 | app smoothed within 0.005 m of the proxy | - | 0.0000 difference on all 11 S0 s0 sessions (`E/p04.py`) | as predicted |
| P0.5 | every wall ≤ 300 s | - | S0 79-228 s, noloop 86-179 s (frame-time sum 55-181 s; no suspend in the window) | as predicted |
| P0.6 | S0 smoothed below Basalt on MH_01, MH_02, MH_03, MH_05 | Basalt 0.067 / 0.048 / 0.062 / 0.144 | medians 0.045 / 0.028 / 0.048 / 0.082; every start below too | as predicted |
| K0.1 | not bit-identical: stop | - | identical | not fired |
| K0.2 | any wall > 300 s: stop | - | max 228 s under the 900 s SYNC cap | not fired |
| K0.3 | a Vicon room fails in SYNC: G54 is not the cause | async default fails V1_02-V2_03 | no Vicon room fails: stale 0, scale 0.92-1.05, worst causal 0.869 (V2_03 s2) against 19-243 m async | not fired |

**Diagnosis of the misses (owner rule; the parts marked unverified are hypotheses, not measured):**
- **Shifts on V1_03 and V2_03 are not G54:** `ba_stale_keyframes=0` on those runs, and the loops-off arm shifts
  too (8 and 8). They sit on the two sequences with the most inertial coasts and the most flow-unavailable
  frames (V2_03 218-236, V1_03 76-118). Unverified: the shifts follow the coasts (G39/G50/G51 class).
- **tracked < frames** has two causes. V1_02: 20-22 frames before the first stereo map (owner answer 3 gives it
  a `min_cold_start_points` rung). V1_03/V2_01/V2_03: frames skipped while coasting on the IMU
  (`coast_inertial` 1-94, `coast_visual` 0, no runaway). V2_03 carries 4.6-4.9 % of its frames on coasts.
- **Scale out of band on V1_03/V2_02/V2_03** comes with the largest causal errors (0.16-0.87 m); the V2_03 s2 run
  scale 0.920 is the outlier start. No mechanism measured yet; the R10 audit is the place.
- **V1_02 causal 0.102** is 0.002 over the range; smoothed is 0.056.

**Further findings:**
- **The start-offset band is partly degenerate.** `start_frame: 0` and `1` give byte-identical outputs on
  MH_04 and V2_03 (frame 0 adds nothing; `Dataset: ... (processing 0..2032)` against `1..2033`), and all three
  starts are identical on V1_02 (the offsets fall inside the 20-22 start-up skips). The band has 2 distinct
  samples on MH_04/V2_03 and 1 on V1_02. The owner should pick offsets past the start-up (or the skill's
  ±50 µs IMU time shift) before the band gates anything.
- **Loops help on every sequence** (S0 smoothed against noloop smoothed): 11 of 11, R9's "≥ 9 of 11" holds on
  this base. PGO failures: 1 each on MH_05 s0 and s2, else 0.
- **SYNC is not better than the async arms everywhere.** MH_04 smoothed 0.226 against the async default proxy
  0.165; V1_01-V2_02 smoothed 0.056-0.159 against the preset's 0.055-0.080 (async, window 10). Not diagnosed;
  these are the R3/R4 targets.
- **Against Basalt (smoothed median):** below on MH_01, MH_02, MH_03, MH_05 (0.57-0.77x); above on MH_04
  (1.98x), V1_01 (1.57x), V1_02 (1.23x), V1_03 (2.73x), V2_01 (2.82x), V2_02 (3.23x), V2_03 (1.90x). Against
  ORB-SLAM3: below on MH_02 only (0.82x); MH_01 0.98x.

### Results: R3, arm S1 (`boundary_scales_bias_rows: false`) against S0 (2026-11-23)

SYNC unpinned, 900 s cap, binary md5 `cea7646298a68ddb948b16835472b2e3`, 15:23-15:56, load 6-16 at start.
18 S1 runs (6 sequences × starts 0/1/2), all `rc=0`, walls 87-258 s, header check clean on every run
(`E/expect/S1.header` shows `(bias rows full)`). S1 V1_03 s1 was finished by `E/finish_claim.sh` after its
second driver was stopped to free a slot for G56 (rc from the `[summary]` block). Debug run `E/debug/D4` = S1,
V2_01, `max_frames: 450`, `--debug`, `KORNIA_SLAM_DEBUG=1`, rc 0, 51 s. No IMU time shift (no knob, see the
predictions): the band is the starts only, so V1_02 has 1 sample and MH_04 2. Scores: `E/score.py`
(`E/scores.jsonl`); mechanism: `E/r3/r3metrics.py` (`E/r3/s0metrics.jsonl`, `E/r3/s1metrics.jsonl`), table
`E/r3/r3report.py` (`E/r3/r3table.md`); debug: `E/r3/d4stats.py`, `E/r3/vischi2.py`, `E/debug/viba_ba.py`,
`E/debug/tilt_t.py`. SE(3) ATE RMSE, m, body centre; tilt absolute (as D1-D3).

| seq | arm | causal s0 / s1 / s2 | smoothed s0 / s1 / s2 | smoothed median (band) | ÷ Basalt | tilt median (window) | causal vertical RMS | final-map b_a error median | walk z p50 accel / gyro | iterations per solve | shifts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2_01 | S0 | 0.193 / 0.213 / 0.200 | 0.109 / 0.124 / 0.110 | 0.110 (0.109-0.124) | 2.82 | 5.67 / 6.80 / 6.05 (22-60 s) | 0.151-0.154 | 0.89-1.05 | 19.8 / 3.4 | 25.4-27.5 | 0 |
| V2_01 | S1 | 0.174 / 0.144 / 0.149 | 0.120 / 0.108 / 0.108 | 0.108 (0.108-0.120) | 2.78 | 5.97 / 4.84 / 4.95 | 0.107-0.136 | 0.21-0.23 | 0.7 / 0.02 | 29.1-29.8 | 1 / 0 / 0 |
| V1_02 | S0 | 0.102 × 3 | 0.056 × 3 | 0.056 (1 sample) | 1.24 | 2.32 × 3 (50-75 s) | 0.057 | 0.19 | 12.9 / 2.0 | 29.2 | 0 |
| V1_02 | S1 | 0.127 × 3 | 0.058 × 3 | 0.058 (1 sample) | 1.30 | 1.20 × 3 | 0.041 | 0.22 | 0.2 / 0.02 | 28.6 | 0 |
| MH_04 | S0 | 0.302 / 0.302 / 0.316 | 0.226 / 0.226 / 0.240 | 0.226 (0.226-0.240) | 1.98 | 3.74 / 3.74 / 3.39 (65-79 s) | 0.264-0.274 | 0.15-0.17 | 6.3 / 2.5 | 20.0-20.2 | 0 |
| MH_04 | S1 | 0.327 / 0.327 / 0.308 | 0.245 / 0.245 / 0.235 | 0.245 (0.235-0.245) | 2.15 | 2.46 / 2.46 / 2.26 | 0.232-0.243 | 0.12 | 0.1 / 0.01 | 20.6-21.0 | 0 |
| MH_01 | S0 | 0.090 / 0.086 / 0.094 | 0.045 / 0.041 / 0.045 | 0.045 (0.041-0.045) | 0.67 | 0.49 / 0.56 / 0.67 (all) | 0.036-0.053 | 0.10-0.12 | 11.1 / 2.7 | 18.7-19.9 | 0 |
| MH_01 | S1 | 0.077 / 0.081 / 0.066 | 0.039 / 0.039 / 0.039 | 0.039 (0.039-0.039) | 0.58 | 0.50 / 0.52 / 0.47 | 0.043-0.046 | 0.09 | 0.3 / 0.01 | 19.3-19.7 | 0 |
| V1_03 | S0 | 0.181 / 0.201 / 0.188 | 0.143 / 0.146 / 0.150 | 0.146 (0.143-0.150) | 2.75 | 1.66 / 2.38 / 1.79 (all) | 0.077-0.093 | 0.19-0.30 | 16.6 / 3.8 | 28.7-29.0 | 6 / 6 / 6 |
| V1_03 | S1 | 0.487 / 0.224 / 0.221 | 0.192 / 0.155 / 0.162 | 0.162 (0.155-0.192) | 3.06 | 2.03 / 1.70 / 1.30 | 0.081-0.091 | 0.05-0.09 | 0.2 / 0.01 | 29.6-30.4 | 7 / 10 / 8 |
| V2_02 | S0 | 0.176 / 0.162 / 0.161 | 0.097 / 0.160 / 0.159 | 0.159 (0.097-0.160) | 3.24 | 3.03 / 2.94 / 2.99 (all) | 0.103-0.110 | 0.37-0.38 | 15.6 / 3.0 | 25.3-25.7 | 0 |
| V2_02 | S1 | 0.159 / 0.156 / 0.165 | 0.155 / 0.076 / 0.153 | 0.153 (0.076-0.155) | 3.12 | 1.18 / 1.33 / 1.46 | 0.099-0.101 | 0.19-0.20 | 0.2 / 0.01 | 26.0-26.3 | 0 |

V1_03 S1 s0 is an outlier: causal 0.487, Sim3 scale 0.853. Tracked S1 against S0: equal except V2_01 s1
(2268 against 2273), V1_03 s0/s1 (+2 / +1) and V1_03 s2 (2142 against 2146).

D4 (S1) against D3 (S0), V2_01, frames 0-450 (`E/r3/d4stats.py`, `E/r3/vischi2.py`):

| quantity | D3 S0 | D4 S1 |
|---|---|---|
| newest-keyframe b_a step per solve, 19-22.5 s (median / max, m/s²) | 0.036 / 0.072 | 0.0008 / 0.0012 |
| b_a error against GT at 22.3 s | 0.63 | 0.05 |
| local-BA b_a error, 0.5 s bins 15-22 s | 0.06 → 0.58 | 0.06 → 0.05 (flat) |
| tilt per 1 s bin, 17 / 19 / 20 / 21 / 22 s (deg) | 0.21 / 0.85 / 1.53 / 5.27 / 6.01 | 0.44 / 1.01 / 1.95 / 6.24 / 6.83 |
| visual mean χ² per observation, 20.5 / 21.0 / 21.5 / 22.0 s (px²) | 30.3 / 36.4 / 47.6 / 44.6 | 30.3 / 37.2 / 47.9 / 42.7 |
| boundary IMU edge χ² p50 / p90, 5-19 s | 7.4 / 27.5 | 21.8 / 87.6 |
| boundary IMU edge χ² p50 / p90, 19-22.5 s | 226 / 5816 | 1174 / 17168 |
| interior IMU edge χ² p50 / p90, 19-22.5 s | 8.4 / 1280 | 15.4 / 13037 |
| iterations per solve, 19-22.5 s (mean; solves at the 50 cap) | 44.6 (12 of 23) | 42.9 (9 of 23) |

| P/K | prediction | baseline (S0) | measured (S1, each sample) | verdict |
|---|---|---|---|---|
| P3.1 | V2_01 b_a step per keyframe, 19-22.5 s ≤ 0.01 | 0.02-0.07 | median 0.0008, max 0.0012 (D4) | as predicted |
| P3.2 | V2_01 b_a error at 22.3 s ≤ 0.3 | 0.63 | 0.046 (D4) | as predicted |
| P3.3 | V2_01 tilt median 22-60 s ≤ 2.5° | 5.67 / 6.80 / 6.05 | 5.97 / 4.84 / 4.95 | missed |
| P3.4 | V2_01 smoothed ≤ 0.10 | 0.109 / 0.124 / 0.110 | 0.120 / 0.108 / 0.108 | missed (median 0.108 sits inside the S0 band) |
| P3.5 | V1_02 tilt 50-75 s ≤ 1.5°, smoothed ≤ 0.056 | 2.32°, 0.056 | 1.20°, 0.058 (1 distinct sample) | tilt as predicted; smoothed missed by 0.002 (provisional) |
| P3.6 | MH_01 smoothed ≤ 0.052 | 0.045 / 0.041 / 0.045 | 0.039 × 3 | as predicted (below the S0 band) |
| P3.7 | MH_04 tilt 65-79 s ≤ 1.5°, \|t_y\| ≤ 0.4, causal vertical RMS ≤ 0.10, causal ≤ 0.20, smoothed 0.08-0.13 | 3.74 / 3.74 / 3.39°; 1.079 / 1.079 / 1.104; 0.264-0.274; 0.302-0.316; 0.226-0.240 | 2.46 / 2.46 / 2.26°; 0.918 / 0.918 / 0.834; 0.243 / 0.243 / 0.232; 0.327 / 0.327 / 0.308; 0.245 / 0.245 / 0.235 | missed on all five (tilt −1.1 to −1.3°, ATE worse) |
| P3.8 | walk z p50 ≤ 2.0, accel and gyro, all six | accel 6.3-19.9, gyro 2.0-3.9 | accel 0.1-0.7, gyro 0.01-0.02 | as predicted |
| P3.9 | V1_03, V2_02 smoothed median within ±15 % | 0.146, 0.159 | 0.162 (+11 %), 0.153 (−4 %) | as predicted |
| P3.10 | tracked unchanged ± 2 | - | V2_01 s1 −5, V1_03 s2 −4 | missed |
| KD1 | b_a held and V2_01 tilt ≥ 4° at 22-40 s | 5.71 / 6.90 / 6.13° | b_a held (P3.1); tilt 6.13 / 4.96 / 5.08° | **kill fired** |
| KD2 | b_a not held | - | held | not fired |
| K1 | walk z holds and MH_04 tilt ≥ 3° at 70 s | 3.65 / 3.65 / 3.27° | 2.44 / 2.44 / 2.18° | not fired |
| K2 | walk z holds, gain < 3 % everywhere | - | smoothed median change V2_01 −2 %, V1_02 +4 %, MH_04 +8 %, MH_01 −13 %, V1_03 +11 %, V2_02 −4 % | not fired (MH_01 gains 13 %) |
| K3 | any sequence's smoothed median worse > 15 % | - | worst V1_03 +11 %, MH_04 +8 %; causal median worse > 15 % on V1_02 (+25 %) and V1_03 (+19 %), outside K3's wording | not fired |
| mechanism without score | P3.8 and P3.1 pass; P3.4, P3.5, P3.7 ATE all miss | - | all three ATE predictions miss | **fired: theory incomplete** |
| score without mechanism | ATE gain > 15 % with the mechanism unchanged | - | mechanism moved | not fired |

**Diagnosis (KD1 and "mechanism without score" fired; written before anything else).**
- The 1e-2 bias rows are the conduit of the bias walk: with them at full weight b_a stops moving (step 0.0008
  m/s² per keyframe against 0.036, walk z 0.1-0.7 against 6-20 everywhere), and the final-map b_a error falls
  on five of six sequences (V2_01 0.9-1.05 → 0.21-0.23, V2_02 0.37 → 0.19, V1_03 0.19-0.30 → 0.05-0.09).
- They are not the source of the tilt. On V2_01 the tilt jump at 20-21 s is the same with b_a held: 1.95 → 6.24°
  in D4 against 1.53 → 5.27° in D3. The visual mean χ² rises the same way in both (30 → 48 px² over 20.5-21.5 s),
  so the driver is on the visual side (G56), as KD1 says. With b_a pinned, the disagreement goes into the IMU
  edges instead: boundary χ² p50 226 → 1174 and interior χ² p90 1280 → 13037 over 19-22.5 s. Those edges then
  lose to `IMU Huber AllEdges`, and the frozen past (§9 b2) keeps the tilted map. The IMU is pinned but
  outvoted, not ignored.
- After the event S1 recovers part of the tilt and S0 does not (V2_01 at 60-110 s: S1 3.4-5.9°, S0 7.0-8.7°).
  Tilt falls on every tilted sequence (V1_02 2.32 → 1.20°, MH_04 3.4-3.7 → 2.3-2.5°, V2_02 3.0 → 1.2-1.5°).
  ATE does not follow: MH_04 causal vertical RMS only 0.264 → 0.243 m, and MH_04, V1_02 and V1_03 get worse.
  MH_04's first loop still carries |t_y| 0.83-0.92 m of vertical drift. So most of the vertical error is not
  the steady tilt of the bias. Unverified: it is built during the dark span (57-67 s), when a visual-led
  window drifts and the IMU edges are Huber-cut.
- Ledger effect: G49's "tilt hidden in the accel bias" is refuted as the cause of lost score on EuRoC. The bias
  carries the tilt only while it is free; pinning it moves the same error into Huber-cut IMU edges. The cause
  moves to G56 (visual error model) and §9 b2 (frozen past).
- Knob: stays off for now (mixed: MH_01 −13 %, V2_02 −4 %, V2_01 −2 %; MH_04 +8 %, V1_03 +11 %, V1_02 +4 %
  smoothed, and causal +25 % on V1_02). Full weight is what ORB-SLAM3 and Basalt use, so S1 stays the
  candidate base for the R11/R12 arms. It is not promoted without the full-sequence gate.
- **Next rung (from KD1):** the visual log line (G56: per-solve reprojection χ² median and share above
  `huber_reproj_px_sq`, running in the G56 chain), then R11 (visual σ / Huber). Then R12 (windowed
  marginalization in place of the frozen past). The cheapest config-only arm to price first:
  S1 + `imu_huber: BoundaryOnly` on V2_01 and MH_04. It tests whether the pinned IMU wins once the Huber
  cut on its interior edges is removed (D4: interior χ² p90 13037).
