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
- **Label (owner decision 2026-11-23):** every R6 number is "benchmark-style (non-causal)", a ceiling of a final
  global solve. A robot never gets one. The claim is the causal real-time pose. Results: "Results: R6" below.

#### R6 baselines and S0E predictions (written 2026-11-23, before any offline solve)

- **Runs:** `offline-ba` with the default `offline_ba: ()` block (60 iterations, 10 px pre-drop, 3 px from
  iteration 8, no IMU kernel, no bridge, no calibration states), on the start-0 session of each arm. Binary
  `E/bin-r11b/kornia-slam-euroc` (md5 e67615ad; offline-ba code unchanged from the base). Each session's own
  `run.ron` gives the visual error model. Outputs in `E/r6/`.
- **A/B is clean:** the online smoothed output is the frames re-anchored on the final-map keyframes (P0.4), and
  `reanchor_frames` uses the same `frame_from_kf`. Only the keyframe poses differ between the two outputs.
- **S0E** = S0 + `local_ba.outlier_erase: ChiSquare(())` (R11b). The erase removes map observations above
  χ² 5.991 at σ 1 px (2.45 px) after each local solve, so an S0E session starts with a much smaller tail.

Baselines, start 0 (SE(3) ATE RMSE, m, body centre; tilt absolute, median of 1 s bins of the smoothed output):

| seq | S0 smoothed | S0E smoothed | S0 kf | S0E kf | Basalt | ORB-SLAM3 |
|---|---|---|---|---|---|---|
| MH_01 | 0.045 | 0.036 | 674 | 579 | 0.067 | 0.046 |
| MH_04 | 0.226 (tilt 60-80 s 3.64°) | 0.250 (tilt 60-80 s 3.15°) | 466 | 359 | 0.114 | 0.045 |
| V1_02 | 0.056 | 0.038 | 491 | 407 | 0.045 | 0.019 |
| V1_03 | 0.143 | 0.062 | 605 | 496 | 0.053 | 0.026 |
| V2_01 | 0.109 | 0.047 | 506 | 354 | 0.039 | 0.031 |
| V2_02 | 0.097 | 0.079 | 696 | 562 | 0.049 | 0.037 |

The S0 predictions above stand for S0 sessions. Predictions for S0E sessions:
- **P6.1 (drop share):** S0E total drop (10 px + 3 px) 2-6 % of observations; S0 8-20 % (the G56 tail of about
  10 % at 5-20 px, plus the 3 px cut on a 1.3 px core, about 7 % of a 2D Gaussian beyond 2.3 σ).
  The G41 kill (> 5 %) is likely to fire on S0 by the 3 px cut alone; that is the ratchet, not the data.
- **P6.2 (MH_04):** S0E 0.250 → 0.08-0.15, tilt at 60-80 s 3.15° → ≤ 1°. Full inertial BA over all keyframes
  is the one stage that can rotate the dark span back to gravity.
- **P6.3 (Vicon rooms, S0E):** −10 to −30 % (smaller than the S0 −20 to −40 %, because R11b already took half the
  error): V1_02 0.030-0.034, V1_03 0.043-0.056, V2_01 0.033-0.042, V2_02 0.055-0.071.
- **P6.4 (MH_01, S0E):** within ±0.005 of 0.036.
- **P6.5 (convergence):** S0E `stop=Converged` in ≤ 30 iterations; S0 may run to the 60 cap (the ratchet keeps
  changing the active set after iteration 8). Each solve ≤ 60 s.
- **Kill rules (both arms):** as above (drop > 5 %, worse on ≥ 3 sequences, a solve > 60 s, or offline-ba fails on
  a Stereo session). One extra reading: S0E better but S0 worse means the S0 tail, not the solve, is at fault.

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

#### R11b design and added predictions (2026-11-23, before code)

Base: kornia-slam@b1b7e6b (branch feat/r11b-outlier-erase)   Binary tag: r11b   Baselines: E/runs/S0_*, E/runs/S1_* (md5 cea7646298a68ddb948b16835472b2e3), E/debug/G56a, G56b, D3.
Reference: ORB-SLAM3 LocalInertialBA erase (Optimizer.cc:2690-2693, 2857-2862, 2879, 2887-2902); tracking outliers reach keyframes
for BA to decide (Tracking.cc:2259-2267); a landmark at weighted count <= 2 is discarded (MapPoint.cc:168-201).
Corrections to the sketch above:
- Knob: local_ba.outlier_erase: Off | ChiSquare((reprojection_chi2: 5.991, stereo_chi2: 7.815, close_factor: 1.5, close_range_m: 10.0)).
- Stereo rows (depth row in the solve): reprojection + depth chi2 > 7.815, no close factor. Mono rows: 5.991, 1.5x within 10 m range.
- The existing cull has no observation-count rule (culling.rs:30-70). The knob retires a touched landmark at weighted count <= 2.
- Sigma is the solver's (1 px on S0): per-solve median / 1.177 = 0.92-1.05 px; the 1.3 px figure is a mean truncated at 25 px2.
- Screening set for this rung: V2_01, V1_02, V1_03, V2_02, MH_04, guard MH_01 (the six with S0 and S1 baselines). Band = starts only
  (no IMU time-offset knob); V1_02 has 1 distinct sample, MH_04 2: their verdicts are provisional.
- visual_information: OctaveScaled also scales the tracking PnP gates (system/build.rs:199).
Arms: S0E (decides), S0O (alone), S1E (probe).
Added predictions, S0E, smoothed median of 3 starts, m (S0 band; Basalt):
  P6a V1_02 0.056 (1 sample; 0.045)        -> 0.045-0.054, ratio to Basalt 1.24 -> 1.00-1.20
  P6b V2_01 0.110 (0.109-0.124; 0.039)     -> 0.070-0.100
  P6c V1_03 0.146 (0.143-0.150; 0.053)     -> 0.115-0.146
  P6d V2_02 0.159 (0.097-0.160; 0.049)     -> median <= 0.150, every start <= 0.160
  P6e MH_04 0.226 (0.226-0.240; 0.114)     -> 0.200-0.240
  P6f MH_01 0.045 (0.041-0.045; 0.067)     -> 0.036-0.045
  P6g causal median not above the S0 band max on all six.      measured by: E/score.py
  P7  V2_01 tilt, 1 s bin at 21 s (D3: 5.27 deg) -> <= 3.5 deg; tilt median 22-60 s (5.67-6.80) -> <= 4.0.   measured by: E/debug/tilt_t.py on a
      450-frame debug run of S0E. If it stays >= 5 deg the outlier tail is not the driver of the tilt jump (KD1 narrows to the frozen past, R12).
  P8  S0O alone: V1_02 VISRES out 11.8 % -> 8.0-11.8 % (mismatches do not shrink with octave sigma); smoothed medians within +-10 % of S0.
      If out falls below 6 %, the tail was coarse-octave noise, and G56's cause is partly refuted.
  P9  S1E: V2_01 smoothed 0.108 -> 0.060-0.095; MH_01 <= 0.045; no sequence above max(S0, S1) band.
Mechanism counters (P3): erase_unlinked_newest / erase_tested_newest 5-15 %; (erase_unlinked - erase_unlinked_newest) /
  (erase_tested - erase_tested_newest) < 2 %. Floor from the model itself: about 5 % of correct rows fail a 95 % test once.
K3 measurements: tracked from [summary]; keyframes from [summary]; map points per keyframe = last per-frame pts= / keyframes.
Also reported, no rule: loops_accepted, ba_solve_ms, frame_ms p90, erase_stale_links, erase_landmarks_retired, against S0.
Not in this rung: erase after the visual-only BA, after global/offline BA, in the tracker; OctaveScaled stacked on the erase.

Owner answers (2026-11-23): S0 decides, S1E probe; ORB-SLAM3 weighted-count rule; close rule verbatim for the first arm
(cleanup card ablates close_factor 1.0 if it wins); joint 3-dof stereo test; extra starts 30 and 60 for S0 (new binary,
knob off) and each arm; config-only arm S0P (keyframe_observations: PnpInliers) after the S0E verdict, not stacked; erase
after the visual-only BA and OctaveScaled stacked deferred; P6-P9 accepted as written.

### Experiment R11c: real time with PropagateToNewer (RT1E), and S0P + S0E stacked (S0PE) (predictions written 2026-11-23 before the runs)

Base: kornia-slam@0bf7b6f (branch feat/r11b-outlier-erase, code 71a68f9)   Binary: `E/bin-r11b/kornia-slam-euroc`, md5
`e67615adaf558b4aef7b7d1e35a2fb40` (no rebuild; config-only arms)   Harness: `E/mkcfg.sh` arms RT1, RT1E, S0PE; RT driver
`E/r11c/rt.sh`; SYNC driver `E/drive.sh` (`BIN` override); golden headers `E/expect/RT1.header`, `RT1E.header`, `S0PE.header`.
Baselines: RT0/RT0E (`E/runs/RT0*_s0_r[12]`, R11b real-time check), S0E and S0P (`E/runs/S0E_*`, `S0P_*`).
Sequences: V2_01, V1_02, V1_03, V2_02, MH_04, MH_01.

**Arm 1, RT1E (real time, decides):** RT0E + `mapping: ( local_ba: ( merge: PropagateToNewer ) )`. Asynchronous mapping,
`--pace-timestamps`, `taskset -c 0,1`, 2 runs per sequence, start 0; causal score on `traj.txt`. Control: RT1 = RT0 + the
same merge. One run at a time, started at 1-min load ≤ 6 with fewer than 2 `kornia-slam*` processes.
- Gap tested: G54. In the R11b real-time check each run left 295-424 keyframes stale; T1 (`PropagateToNewer`) removed
  them before (V102-DIAGNOSIS: the preset propagated 602 of 602 on V1_03).
- Predictions:
  - P1 (mechanism): `ba_propagated_keyframes` = `ba_stale_keyframes` on every RT1/RT1E run.
  - P2 (mechanism): `ba_keyframe_shifts` ≤ 10 on every RT1/RT1E run.
  - P3: RT1E causal (mean of r1/r2) within 1.5x of S0E SYNC causal median on each sequence: V2_01 ≤ 0.083, V1_02 ≤ 0.060,
    V1_03 ≤ 0.108, V2_02 ≤ 0.120, MH_04 ≤ 0.47, MH_01 ≤ 0.108.   measured by: `E/score.py`
  - P4: `frame_ms p90` unchanged, inside 54-77 ms (RT0/RT0E range).
- Kill rule K1: P2 holds (shifts ≤ 10) but RT1E causal is still above 2x S0E SYNC on 3 or more sequences (V2_01 > 0.110,
  V1_02 > 0.080, V1_03 > 0.144, V2_02 > 0.160, MH_04 > 0.62, MH_01 > 0.144). Then the real-time loss is not G54: diagnose
  frame drops and mapping lag before the next card.
- Kill rule K2 (score moved, mechanism did not): P3 met on 4 or more sequences while propagated < stale on those runs. Another cause.

**Arm 2, S0PE (SYNC, stacked):** S0 + `outlier_erase: ChiSquare(())` + `keyframe_observations: PnpInliers`. Starts 0/1/2 on
the six, plus starts 30/60 on V1_02 and MH_04 (22 runs). Same SYNC protocol as R11b (unpinned, 900 s cap, 2 at a time).
- Theory: S0P stops most mismatches at tracking; the erase removes those that still enter later. R11b found the arms
  complementary (S0P wins V2_01, MH_04, MH_01; S0E wins V1_02, V1_03, V2_02).
- Predictions:
  - P5: causal median ≤ min(S0E, S0P) on at least 4 of 6 (V2_01 ≤ 0.050, V1_02 ≤ 0.040, V1_03 ≤ 0.072, V2_02 ≤ 0.080,
    MH_04 ≤ 0.254, MH_01 ≤ 0.057; medians over the same starts as R11b).
  - P6: MH_04 causal median ≤ 0.254 (S0P).
  - Mechanism counter (reported, no rule): `erase_unlinked` per run below S0E on the same start (PnpInliers keeps the
    mismatches out first).
- Kill rule K3: causal median worse than both S0E and S0P on 3 or more sequences. Then the two do not stack.

Not in this rung: code changes, the all-11 gate, LaMAria.

### Experiment R11d: mapping lag under real time (RT2E, RT2E-r) and S0PE under real time (RT1PE) (predictions written 2026-11-23 before the runs)

Owner decision (2026-11-23): the score models real time, so the real-time causal score (`traj.txt`, async, paced,
2 cores) is the number that must beat Basalt on every sequence. Base and binary as R11c: `E/bin-r11b/kornia-slam-euroc`,
md5 `e67615adaf558b4aef7b7d1e35a2fb40`, config and environment only. Harness: `E/mkcfg.sh` arms RT2E, RT2E-r, RT1PE;
driver `E/r11d/rt.sh` (sets `RAYON_NUM_THREADS=1` for RT2E and RT2E-r only); golden headers `E/expect/RT2E.header`
(`7 iterations from lambda 0.0001`), `RT2E-r.header` (= `RT1E.header`; the env var does not print), `RT1PE.header`
(`keyframe observations: PnpInliers`). Async, `--pace-timestamps`, `taskset -c 0,1`, start 0, r1 and r2, one RT run at
a time, started at 1-min load ≤ 6 with fewer than 2 `kornia-slam*` processes; load and wall recorded per run.
Sequences: V2_01, V1_02, V1_03, V2_02, MH_04, MH_01. RT2E adds MH_02, MH_03, MH_05, V1_01, V2_03 only if it beats RT1E
(causal mean) on at least 4 of the 6. Baseline: RT1E (`E/runs/RT1E_*_s0_r[12]`), S0E SYNC.

Counters: solves per keyframe = `ba_solves / keyframes`; BA solve mean = `ba_solve_ms / ba_solves`; iterations per
solve = `vi_ba_base_iterations / vi_ba_base_solves`. RT1E measured: solves per keyframe 0.84-0.93 on the Vicon rooms,
1.01 on MH; BA mean 109-226 ms (Vicon), 53-65 ms (MH); 29-32 iterations (Vicon), 19-21 (MH); keyframes 466-657 against
S0E SYNC 353-568 on the Vicon rooms.

**Arm RT2E (decides):** RT1E (erase + `PropagateToNewer`) + `local_ba: ( max_iterations: 7, initial_lambda: 1e-4 )`
(Basalt's budget) + env `RAYON_NUM_THREADS=1`.
- Gaps tested: G57 mapping lag (0.83-0.92 solves per keyframe on V2_01 / V1_03; a solve takes 109-225 ms against a
  keyframe every 150-400 ms) and profile fix 1 (rayon spinning, 0.38 core).
- Theory: a 7-iteration solve at about 6 ms per iteration costs about 40-60 ms, below the keyframe interval, so the
  latest-wins queue stops dropping jobs and every keyframe is solved as newest, as in SYNC. One rayon worker removes
  the spin that competes with the mapping thread for the 2 cores. If lag is what separates RT1E from SYNC, RT2E lands
  near S0E SYNC on the Vicon rooms.
- Predictions:
  - P1 (mechanism): solves per keyframe ≥ 0.97 on every sequence.
  - P2 (mechanism): keyframes within 1.10x of S0E SYNC (start 0) on every sequence (V2_01 ≤ 389, V1_02 ≤ 448,
    V1_03 ≤ 546, V2_02 ≤ 618, MH_04 ≤ 395, MH_01 ≤ 637).
  - P3: BA solve mean ≤ 80 ms on every run.
  - P4: `[summary] frame_ms` p90 ≤ 50 ms on every run.
  - P5: causal mean within 1.2x of S0E SYNC on V2_01 (≤ 0.066) and V1_03 (≤ 0.086), and not worse than RT1E by more
    than its band elsewhere (≤ RT1E's worse run: V1_02 ≤ 0.056, V2_02 ≤ 0.119, MH_04 ≤ 0.332, MH_01 ≤ 0.083).
- Kill rule K1 (lag is not the cause): P1 holds (solves per keyframe ≥ 0.97) but V2_01 causal mean is still ≥ 1.5x
  S0E SYNC (≥ 0.083). Then diagnose before the next card.
- Kill rule K2 (the budget is harmful): the 7-iteration budget worsens synchronous-equivalent accuracy by more than
  15 % on MH. On MH_04 and MH_01 mapping keeps up already (RT1E solves per keyframe 1.01, RT1E = SYNC), so RT2E against
  RT2E-r there isolates the budget: fired if RT2E causal mean > 1.15x RT2E-r on MH_04 or MH_01.

**Arm RT2E-r (control, separates the two changes):** RT1E + env `RAYON_NUM_THREADS=1` only. Reported, no rule:
solves per keyframe, BA mean and causal against RT1E (the rayon share) and RT2E (the budget share).

**Arm RT1PE (stacked candidate under real time):** RT1E + `keyframe_observations: PnpInliers` (the S0PE stack), rayon
default. Prediction P6: causal mean ≤ RT1E on MH_04 and MH_01, where S0PE beat S0E in SYNC (0.227 against 0.325,
0.049 against 0.072). Reported beside RT1E on all six; no kill rule.

Not in this rung: code changes, the keyframe-acceptance rule (next, if K1 does not fire), LaMAria, promotion of any knob
(the full-sequence gate comes first).

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

### Results: R6, offline global inertial BA on S0 and S0E sessions (2026-11-23) — benchmark-style (non-causal)

**Label: benchmark-style (non-causal).** A final global solve over the whole session. It is a ceiling, not the
score: the owner's score is the causal real-time pose (decision below). ORB-SLAM3 and SLAM Dunk report their
EuRoC numbers this way too.

Binary `E/bin-r11b/kornia-slam-euroc` (md5 e67615ad), default `offline_ba: ()` block, start-0 session of each arm,
one thread (faer sequential), slot-gated (≤ 2 `kornia-slam*`). All 17 solves rc 0, every frame written
(`rows == session_frames`), 1 component each. Outputs, scripts and scores: `E/r6/` (`run_r6.sh`, `score_r6.py`,
`scores_r6.jsonl`). Score: `eval.py --fmt tum`, SE(3) ATE RMSE, m, body centre, against the same session's online
`traj_smoothed.txt`. Tilt: `E/r11b/tiltwin.py`, median of 1 s bins, 60-80 s. Drop = observations dropped at
10 px (start state) + at 3 px (from iteration 8), share of all observations.

| seq | S0 online sm | S0 offline | S0E online sm | S0E offline | Basalt | ORB-SLAM3 | S0 drop 10 / 3 px % | S0E drop 10 / 3 px % | iterations S0 / S0E (all `Converged`) | solve s S0 / S0E |
|---|---|---|---|---|---|---|---|---|---|---|
| MH_01 | 0.045 | **0.031** | 0.036 | **0.032** | 0.067 | 0.046 | 4.65 / 9.26 | 0.13 / 0.27 | 16 / 13 | 46.5 / 22.9 |
| MH_02 | 0.035 | **0.020** | - | - | 0.048 | 0.034 | 3.30 / 10.64 | - | 16 | 27.0 |
| MH_03 | 0.041 | **0.027** | - | - | 0.062 | 0.028 | 3.54 / 10.24 | - | 16 | 23.5 |
| MH_04 | 0.226 | **0.070** | 0.250 | **0.075** | 0.114 | 0.045 | 8.94 / 14.16 | 5.16 / 0.51 | 18 / 13 | 13.0 / 4.3 |
| MH_05 | 0.082 | **0.081** | - | - | 0.144 | 0.050 | 10.00 / 15.60 | - | 19 | 21.3 |
| V1_01 | 0.088 | **0.043** | - | - | 0.043 | 0.039 | 10.87 / 9.75 | - | 23 | 48.0 |
| V1_02 | 0.056 | **0.024** | 0.038 | **0.024** | 0.045 | 0.019 | 8.21 / 10.78 | 4.18 / 0.23 | 15 / 12 | 12.2 / 6.6 |
| V1_03 | 0.143 | **0.064** | 0.062 | **0.049** | 0.053 | 0.026 | 6.88 / 17.15 | 0.16 / 0.74 | 20 / 13 | 19.3 / 12.5 |
| V2_01 | 0.109 | **0.043** | 0.047 | **0.035** | 0.039 | 0.031 | 16.04 / 11.91 | 0.36 / 0.32 | 24 / 18 | 26.3 / 12.9 |
| V2_02 | 0.097 | **0.042** | 0.079 | **0.043** | 0.049 | 0.037 | 6.72 / 15.70 | 0.00 / 0.73 | 22 / 14 | 36.2 / 11.1 |
| V2_03 | 0.436 | **0.120** | - | - | 0.230 | 0.027 | 19.14 / 14.64 | - | 26 | 22.6 |
| mean (11) | 0.123 | **0.051** | | | 0.081 | 0.035 | | | | |
| mean (6 with S0E) | 0.113 | 0.046 | 0.085 | **0.043** | 0.061 | 0.034 | | | | |

MH_02/03/05, V1_01 and V2_03 have no S0E session yet. MH_04 tilt at 60-80 s: S0 3.64° → 0.20°, S0E 3.15° → 0.25°.
Mean whitened IMU rotation residual falls 8-65 → 2.4-4.1 on every solve. Peak RSS 54-264 MB.

Against the references (S0 offline, 11 sequences): below Basalt on 9 (V1_01 0.0427 against 0.043 is a tie; above on
V1_03 and V2_01). Below ORB-SLAM3 on MH_01, MH_02, MH_03. Mean 0.051 against Basalt 0.081, ORB-SLAM3 0.035, the
owner's target about 0.03. The S0E six: mean 0.043; above ORB-SLAM3 on all six except MH_01.

Config-only diagnostic, 10 px pre-drop off (`outlier_px: 1000.0`), `E/r6/nopre_*`:

| session | drop 10 / 3 px % | ATE (default → pre-drop off) |
|---|---|---|
| S0E MH_04 | 0 / 0.50 (was 5.16 / 0.51) | 0.0745 → 0.0738 |
| S0E V1_02 | 0 / 0.31 (was 4.18 / 0.23) | 0.0235 → 0.0228 |
| S0 V2_01 | 0 / 24.84 (was 16.04 / 11.91) | 0.043 → 0.0375 |
| S0 V2_03 | 0 / 24.77 (was 19.14 / 14.64) | 0.120 → 0.132 |

| P/K | prediction | baseline | measured | verdict |
|---|---|---|---|---|
| R6 (S0) MH_04 | smoothed 0.06-0.10, tilt 60-80 s ≤ 1° | 0.226 (the 0.165 in the R6 text is the async proxy), 3.64° | 0.070, 0.20° | as predicted |
| R6 (S0) Vicon | −20 to −40 % | 0.056-0.436 | −52 % (V1_01), −58, −56, −60, −57, −72 % (V2_03) | beat (larger gain than predicted) |
| R6 (S0) MH_01/MH_02 | within ±0.005 | 0.045 / 0.035 | 0.031 / 0.020 (−0.014 / −0.015) | missed (better) |
| R6 solve time | each ≤ 60 s | - | 4.3-48.0 s wall, one thread | as predicted |
| P6.1 drop share | S0E 2-6 %, S0 8-20 % | - | S0E 0.40-5.67 % (four below 2 %); S0 13.8-33.8 % | missed (S0E lower, S0 higher) |
| P6.2 MH_04 S0E | 0.08-0.15, tilt ≤ 1° | 0.250, 3.15° | 0.075, 0.25° | beat (below the range) |
| P6.3 Vicon S0E | −10 to −30 % | 0.038 / 0.062 / 0.047 / 0.079 | 0.024 (−39 %) / 0.049 (−21 %) / 0.035 (−27 %) / 0.043 (−46 %) | two in range, two beat |
| P6.4 MH_01 S0E | ±0.005 of 0.036 | 0.036 | 0.032 | as predicted |
| P6.5 convergence | S0E `Converged` ≤ 30 iterations; S0 may hit the cap | - | all 17 `Converged`, 12-26 iterations | as predicted (S0 too) |
| K drop > 5 % (G41) | - | - | fired: all 11 S0 sessions, and S0E MH_04 (5.67 %) | **fired** |
| K worse on ≥ 3 | - | - | worse on 0 of 17 | not fired |
| K solve > 60 s | - | - | max 48.0 s | not fired |
| K Stereo session fails | - | - | works on all 17 | not fired |

**Diagnosis (the G41 kill fired, and three predictions missed).**
- **The drop kill fired without the harm it guards against.** ATE improved on all 17 sessions.
  - On S0E the excess is the 10 px pre-drop (`global_vi_ba/mod.rs:267`) on the start state. The start is the
    final map after 4-DoF loop corrections, so reprojections are bent: cost0 is 6.2e6 against 7.6e4 at the end,
    and IMU rotation residual 14 → 2.4. With the pre-drop off, MH_04 and V1_02 drop only 0.3-0.5 % at 3 px and
    the ATE is the same or better. So the dropped rows were inliers of the solved state.
  - On S0 the 3 px share (9-17 %, 25 % with pre-drop off) is the unculled G56 tail (about 10 % of rows at
    5-20 px), plus the 3 px cut on a 1.3 px core. R11b's online erase removes the tail first: S0E 3 px drop is
    0.2-0.7 %.
  - Pre-drop off helps S0 V2_01 (0.043 → 0.038) and hurts S0 V2_03 (0.120 → 0.132). With a heavy tail
    (V2_03: 89-94 coasted frames, most flow-unavailable frames), a full solve with every outlier active is
    pulled before the 3 px stage starts. So the fix is to drop after a first solve, not on the raw start.
- **The 3 px "ratchet" (`mod.rs:419`) did no visible harm here.** Every solve converged in 12-26 iterations,
  and the 3 px drop on S0E is under 1 %. Card AI's ratchet finding came from LaMAria sessions with 23-76 % drops.
  It stays a risk for heavy-tail sessions, not a cause of loss on EuRoC.
- **MH_01/MH_02 gain more than predicted (−0.014 m).** My S0 prediction assumed the easy sequences had no global
  error left. Sim3 ATE offline is 0.014-0.018 against SE(3) 0.020-0.031, so part of the remaining error is scale
  or a rigid offset. The online map still carries 0.01-0.02 m of globally inconsistent error after 4-DoF loops.
  Unverified: it is the roll/pitch and scale that a 4-DoF pose graph cannot correct.
- **S0 and S0E converge to the same ceiling** (6-sequence mean 0.046 against 0.043). R11b mostly helps the online
  output. Offline it changes V1_03 (0.064 → 0.049) and V2_01 (0.043 → 0.035), and costs MH_04 +0.005.
- **The ceiling is still 1.3-4.4x above ORB-SLAM3 on MH_04, MH_05, V1_02, V1_03, V2_03** (S0; MH_05 gains nothing:
  0.082 → 0.081). R6 removes tilt and loop inconsistency. It does not remove what the observations themselves
  carry: G56, the frontend (flow unavailable on V1_03/V2_03, G50/G51), the 1.3 px core.

**Fix plan for card AI rung 2 (planner/implementer; no code here). Predictions on these 17 sessions:**
1. **Drop schedule:** solve first with every observation under a Huber kernel. Then drop once at `outlier_px` on
   the converged state, re-solve, drop once at `tight_outlier_px`, re-solve. Replaces the pre-drop at `:267` and
   the per-iteration re-drop at `:419`. Config: `outlier_schedule: AfterConvergence | Legacy`.
   - Predict: total drop on S0E ≤ 1 % on all six; S0 drop ≤ the 3 px share with pre-drop off (≈ 25 %) and no S0
     session worse than its default by > 0.003; S0 V2_03 ≤ 0.120; S0 V2_01 ≤ 0.040.
   - Kill: any session worse than its default by > 0.005, or a solve > 60 s.
2. **Seeds:** none needed on EuRoC yet (1 component per session, every solve
   converges; the `inserted_while_lost` count in these sessions is not read). Keep card AI's seed fixes for LaMAria; do not spend on them for EuRoC.
3. **Ceiling stack:** the R6 ceiling then follows the online improvements (R11 error model, R12, frontend), because
   the offline solve cannot fix what the observations carry.

**Background stage in real time (owner's question for card AI (b); estimate, not a measurement).**
- **Cost model, from the 17 solves (one thread):** per LM iteration 0.30-0.91 s for 354-562 keyframes at
  covisibility 24-37 (S0E). MH_01 is 1.7 s at 579 keyframes and covisibility 74, because loop revisits densify the
  reduced system (factor nnz 9.2M against 1.1-2.4M). About 13 iterations per solve. Full solve: 4-23 s CPU (S0E),
  12-48 s (S0).
- **0.25 of a core:** wall time is 4x CPU, so a full end-of-sequence solve lands 16-92 s after it starts.
  - **Full-map stage after each loop** (ORB-SLAM3's pattern): 1-5 runs per sequence. MH_04 (100 s, 4 loops):
    about 4 s CPU, so 16 s wall. The tilt fix would land about 16 s after the loop that triggers it.
  - **Bounded window:** 60 keyframes (about 15-20 s of motion), covisibility ≤ 60. Estimated 50-150 ms per
    iteration, 0.7-2 s CPU per solve, so 3-8 s wall at 0.25 core. It runs every 3-8 s, about every 10-30 keyframes.
- **Share of the offline gain that reaches the causal pose:** unmeasured; three cases.
  - Tilt, gravity and bias (MH_04, Vicon rooms): a correction stays valid going forward. The live pose gets it
    through map-to-odom once it lands, 3-16 s late. Estimate 50-80 % of the smoothed-to-offline gain on poses after
    the first correction, and none before it.
  - Loop consistency (MH_01/MH_02, V1_01): the causal output already loses 0.03-0.04 m to loop jumps (G53, S0E
    causal 0.072 against smoothed 0.036 on MH_01). A background solve does not remove the jump; it moves it.
    Estimate 0-30 %.
  - Refined past keyframes: 0 % causal by definition.
  - Overall guess: 30-50 % of the offline gain reaches the causal score. On the S0E six that would take the
    causal mean from about 0.105 to 0.08-0.09. It needs a measurement.
- **Measurement that settles it (card AI (b)):** a prefix replay, no robot needed. Run offline-ba on the session
  truncated at time t (`--until <t>` on `offline-ba`; a small app change). Re-anchor only frames in
  [t + lag, t + lag + Δ] on that solve, where lag = 4 x CPU time of the solve. Score the stitched causal
  trajectory. Cost: about 10-30 solves per sequence at 4-23 s CPU each.

## Owner decision 2026-11-23: the score models real time

"there's never a final global in real time... we should model this."

- **Primary metric (replaces open-question answer 1, "smoothed ranks"):** the real-time causal pose.
  - Replay paced at sensor rate (`--pace-timestamps`), asynchronous mapping, the process pinned to 2 cores (`taskset -c <2 cores>`, per the 2-core real-time rule).
  - Scored on `traj.txt` (each frame's pose as output at that time), SE(3) body-centre, `eval.py --fmt tum`.
  - M1 = below Basalt on all 11 sequences on this metric. Basalt is itself an online VIO, so the comparison is like for like.
- **Background global stage:** counted only through what it delivers in time. It runs inside the same 2-core budget, and its correction reaches the live pose through map-to-odom from the moment it lands. No retroactive fix of past frames.
- **Synchronous runs:** stay the A/B mechanism instrument (deterministic, fast to compare). They no longer carry the score.
- **Smoothed and offline-global trajectories:** reported only as "benchmark-style (non-causal)", next to ORB-SLAM3 and SLAM Dunk, which also score that way. R6 measures the ceiling of a global solve under this label.
- **Gate:** each batch of winners runs on all 11 sequences in the real-time replay, 3 runs (async runs differ run to run), at load ≤ 6. Pass = causal median below the baseline band max on every sequence, plus tracked frames and give-up counts.

### Results: R11b, post-solve outlier erase (S0E), with S0O, S1E and S0P (2026-11-23)

Code: `feat/r11b-outlier-erase` 71a68f9 (on `feat/global-pose-graph` a0c08d6 + the VISRES commit), knob
`mapping.local_ba.outlier_erase`, default `Off`. Binary `E/bin-r11b/kornia-slam-euroc`, md5
`e67615adaf558b4aef7b7d1e35a2fb40`. SYNC unpinned, 900 s cap, 2 runs at a time, 17:19-19:45, load 3-14.
92 batch runs (S0 starts 30/60 on V1_02 and MH_04 with the new binary, knob off: 4; S0E, S0O, S1E, S0P: 22
each = 6 sequences × starts 0/1/2, plus starts 30/60 on V1_02 and MH_04), all `rc=0`, walls 104-251 s, header
check clean on every run (`E/expect/S0E.header` shows `local BA outlier erase: ChiSquare(...)`, `S1E` adds
`(bias rows full)`, `S0O` `visual information OctaveScaled`, `S0P` `keyframe observations: PnpInliers`).
Debug runs (`--debug`, `KORNIA_SLAM_DEBUG=1`): `E/debug/R11b-V102` (S0E V1_02 full), `R11b-MH01` (S0E MH_01
400), `R11b-V201` (S0E V2_01 450), `R11b-S0O-V102` (S0O V1_02 full), `R11b-S0-MH04` and `R11b-S0E-MH04`
(MH_04 full). Each full debug trajectory is `cmp`-identical to its batch run. Scores: `E/score.py`
(`E/r11b/scores.jsonl`); tables `E/r11b/report.py` (`E/r11b/report_all.md`); P4 `E/r11b/kfpieces.py`; tilt
`E/r11b/tiltwin.py` over `E/debug/tilt_t.py`; MH_04 error by time `E/r11b/errtime.py`. SE(3) ATE RMSE, m, body
centre. Per the owner decision of 2026-11-23 the causal output (`traj.txt`) is the primary column; smoothed is
non-causal (benchmark-style).

**Bit-identity, knob off:** S0 V1_02 (`E/r11b/ident-V1_02`): `traj.txt` and `traj_smoothed.txt` `cmp`-identical
to `E/runs/S0_V1_02_medium_s0_r1`, header identical to `E/expect/S0.header`. R_01 (`E/r11b/ident-r01`,
`RAYON_NUM_THREADS=1`): md5 `d4f77b567b912b5a4caa2af6085a35b1`, `cmp`-identical to `P/baseline/traj.txt`. Both
print `erase_tested=0 erase_unlinked=0`.

**Arms against S0, S1 and Basalt.** Median over starts 0/1/2 and, for V1_02 and MH_04, over all five starts.
Bands are min-max over starts; S0 has 5 starts on V1_02 and MH_04, S1 3.

| seq | Basalt | S0 causal band | S0E causal (med) | S0O causal | S1E causal | S0P causal | S0 smoothed band | S0E smoothed | S0O smoothed | S1E smoothed | S0P smoothed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2_01 | 0.039 | 0.193-0.213 | **0.055** (0.052-0.058) | 0.146 | 0.077 | 0.050 | 0.109-0.124 | 0.047 | 0.098 | 0.043 | 0.044 |
| V1_02 | 0.045 | 0.069-0.387 | **0.040** (0.040-0.151) | 0.067 | 0.055 | 0.063 | 0.050-0.203 | 0.038 | 0.060 | 0.040 | 0.046 |
| V1_03 | 0.053 | 0.181-0.201 | **0.072** (0.069-0.080) | 0.186 | 0.101 | 0.118 | 0.143-0.150 | 0.063 | 0.126 | 0.092 | 0.087 |
| V2_02 | 0.049 | 0.161-0.176 | **0.080** (0.080-0.084) | 0.181 | 0.095 | 0.120 | 0.097-0.160 | 0.079 | 0.166 | 0.092 | 0.115 |
| MH_04 | 0.114 | 0.302-0.324 | **0.325** (0.311-0.342) | 0.361 | 0.362 | 0.254 | 0.226-0.265 | 0.250 | 0.335 | 0.238 | 0.179 |
| MH_01 | 0.067 | 0.086-0.094 | **0.072** (0.072-0.076) | 0.073 | 0.068 | 0.057 | 0.041-0.045 | 0.036 | 0.043 | 0.033 | 0.028 |

S0E causal ÷ Basalt: V2_01 1.41, V1_02 0.89, V1_03 1.34, V2_02 1.63, MH_04 2.84, MH_01 1.07 (S0 causal median of starts 0-2: 5.13, 2.24,
3.52, 3.28, 2.65, 1.34). V1_02 starts 30/60: S0 0.069/0.387, S0E 0.043/0.151 causal.

**Run health (K3 inputs, S0E against S0, same start):** tracked −2 to +1 frames on every run, give-ups 0 on all
88 arm runs. Keyframes 0.69-0.86x of S0 (fewer, not more). Map points per keyframe (last `pts=` ÷ keyframes):
−14 % worst (V1_03 s0 86.2 against 100.2), +8 % best. Loops accepted 2-10 against 2-9. `ba_solve_ms` S0E
28-99 s against 29-109 s. `frame_ms p90` 137-345 ms against 104-348 ms (SYNC, load 6-14: not a real-time
figure). Erase: 4-22 k landmarks retired per run, `erase_stale_links` 0 everywhere (SYNC has no stale update).

**Debug runs (mechanism).**

| run | VISRES out % | mean χ² | in_mean | tested | unlinked newest / tested newest | unlinked older / tested older | tilt bin 21 s |
|---|---|---|---|---|---|---|---|
| V1_02 full, S0 (`G56a`) | 11.8 | 25.11 | 3.40 | - | - | - | - |
| V1_02 full, S0E | 1.1 | 3.62 | 1.63 | 2139862 | 11.6 % | 3.3 % | - |
| V1_02 full, S0O | 8.8 | 17.95 | 2.24 | - | - | - | - |
| MH_01 400, S0 (`G56b`) | 8.4 | 16.44 | 3.35 | - | - | - | - |
| MH_01 400, S0E | 0.6 | 2.45 | 1.68 | 415239 | 11.1 % | 2.1 % | - |
| MH_04 full, S0 | 14.6 | 26.83 | 3.87 | - | - | - | - |
| MH_04 full, S0E | 1.0 | 3.31 | 1.58 | 1192705 | - | - | - |
| V2_01 450, S0 (D3) / S0E | - / 1.5 | - / 3.51 | - / 1.76 | - | - | - | 5.27 / 1.38° |

Per-solve erase share (ERASE lines): V1_02 p50 4.1 %, p90 6.6 %, max 16.3 % (one solve); V2_01 450 p50 8.3 %,
p90 13.0 %, max 16.7 % (kf 431, 434 at 21.5 s); MH_01 p50 2.9 %, max 7.9 %. No mass erase: no solve above 17 %.
`depth_nonpositive` 0 and `guard_skipped` 0 on all three.

| P/K | prediction | baseline | measured | verdict |
|---|---|---|---|---|
| P1 | V1_02 out ≤ 3 %, mean χ² ≤ 5, in_mean 2.0-2.8 | 11.8 %, 25.1, 3.40 | 1.1 %, 3.62, 1.63 | missed on in_mean (below the range); out and mean as predicted |
| P2 | MH_01 400 out ≤ 2 %, mean χ² ≤ 4 | 8.4 %, 16.4 | 0.6 %, 2.45 | as predicted |
| P3 | newest unlinked 5-15 %; older < 2 % | - | debug: newest 11.6 / 11.1 %, older 3.3 / 2.1 % (V1_02 / MH_01). Batch: newest 8.4-16.0 % (above 15 on V2_01, V1_03, V2_02, MH_04), older 1.4-6.1 % (below 2 only on MH_01) | missed |
| P4 | V1_02 keyframe-only 10 s piece median ≤ 0.019 | 0.0226 (s0), 0.0228 (s30), 0.0443 (s60) | 0.0105, 0.0123, 0.0172 | as predicted on every start |
| P5 | MH_01 inside its S0 band; no sequence lost or above its S0 band | smoothed 0.041-0.045 | MH_01 0.034-0.037 smoothed (below the band), 0.072-0.076 causal (below 0.086); no loss; no smoothed median above the S0 band max; causal MH_04 median of 5 0.325 against max 0.324 | missed by letter (MH_01 below, not inside; MH_04 causal +0.0005 m) |
| P6a | V1_02 0.045-0.054, ÷ Basalt 1.00-1.20 | smoothed 0.056 (1 sample); causal 0.102 | smoothed 0.038 (0.84x); causal 0.040 (0.89x) | missed (better than the range), both columns |
| P6b | V2_01 0.070-0.100 | 0.110; causal 0.193-0.213 | smoothed 0.047; causal 0.055 | missed (better), both |
| P6c | V1_03 0.115-0.146 | 0.146; causal 0.181-0.201 | smoothed 0.063; causal 0.072 | missed (better), both |
| P6d | V2_02 median ≤ 0.150, every start ≤ 0.160 | 0.159; causal 0.161-0.176 | smoothed 0.079 (max 0.082); causal 0.080 (max 0.084) | as predicted, both |
| P6e | MH_04 0.200-0.240 | 0.226 (0.226-0.265 over 5 starts); causal 0.302-0.324 | smoothed 0.250 (0.250-0.266); causal 0.311 (0-2) / 0.325 (5) | missed, both (unchanged within the start band) |
| P6f | MH_01 0.036-0.045 | 0.045; causal 0.086-0.094 | smoothed 0.036; causal 0.072 | as predicted on smoothed; missed on causal (0.072 is outside the range, though below S0) |
| P6g | causal median not above the S0 band max, all six | - | V2_01 0.055/0.213, V1_02 0.040/0.387, V1_03 0.072/0.201, V2_02 0.080/0.176, MH_01 0.072/0.094; MH_04 0.3246 against 0.3241 (5 starts), 0.311 against 0.316 (starts 0-2) | missed on MH_04 by 0.0005 m over 5 starts |
| P7 | V2_01 tilt bin 21 s ≤ 3.5°; median 22-60 s ≤ 4.0° | 5.27°; 5.68 / 6.83 / 6.08° | 1.38° (debug); batch 1.38 / 1.08 / 1.40° and 1.30 / 1.23 / 1.48° | as predicted |
| P8 | S0O V1_02 out 8.0-11.8 %; smoothed medians within ±10 % of S0 | 11.8 % | out 8.8 %; smoothed V2_01 −11 %, V1_02 +7 %, V1_03 −14 %, V2_02 +4 %, MH_04 +54 %, MH_01 +2 % | out as predicted; medians missed (V2_01, V1_03, MH_04) |
| P9 | S1E V2_01 smoothed 0.060-0.095, MH_01 ≤ 0.045, no sequence above max(S0, S1) band | - | V2_01 0.043; MH_01 0.033; smoothed all inside; causal MH_04 0.362 against max 0.327 | missed (V2_01 better than the range; MH_04 causal above) |
| K1 | out ≤ 3 % and P4 not met | - | P4 met | not fired |
| K2 | P4 met and out > 6 % | - | out 1.1 % | not fired |
| K3 | tracking loss, keyframes up > 10 %, or map points per keyframe down > 25 % | - | no loss; keyframes 0.69-0.86x; worst map points per keyframe −14 % | not fired |

**Diagnosis of the misses (owner rule; written before anything else; unverified parts marked).**
- **P6a-c, P6b and P9 better than their ranges.** The predictions credited the erase with the keyframe
  precision gap only (the 10 s pieces). It also removes the V2_01 tilt event: the 21 s tilt bin falls from 5.27°
  to 1.08-1.40° and the 22-60 s median from 5.7-6.8° to 1.2-1.5°. The visual χ² rise that KD1 found behind the
  tilt (R3) is this tail: per-solve erase share peaks at 13-17 % on V2_01 at 21.5 s. So the tail drove the drift
  on the Vicon rooms too, and the score moved by the drift term, not only by precision. The largest gains are on
  causal (V2_01 0.200 → 0.055, V1_03 0.188 → 0.072, V2_02 0.162 → 0.080), where drift dominates.
- **P6e, P6g and P5 on MH_04: mechanism moved, score did not.** MH_04 out 14.6 % → 1.0 %, mean χ² 26.8 → 3.3,
  yet causal 0.302 → 0.311 (s0) and smoothed 0.226 → 0.250. The S0 5-start band (0.226-0.265 smoothed,
  0.302-0.324 causal) holds the S0E values, so MH_04 is unchanged within start noise. The error by 10 s bin
  (whole-run SE(3) alignment, causal) is the same in both arms: 0.48/0.45 m at 40-50 s and 0.63/0.65 m at
  70-80 s, after the dark span (57-67 s). The tail is not what builds MH_04's error. S0P lowers every bin
  (0.40 and 0.53 m there; causal 0.254). Unverified: on MH_04 the mismatches do their damage in the newest
  keyframe's first solve, before any erase, and the dark span leaves too few correct rows to absorb it.
- **P6f causal.** The range was written for the smoothed column; causal MH_01 0.072 is 20 % below the S0 causal
  band, not a regression.
- **P1 in_mean below 2.0-2.8.** in_mean is the mean χ² of rows under the 25 px² Huber knee. The erase removes
  the 6-25 px² shoulder as well (the test is at 2.45-3.0 px). With the shoulder gone, a 2-dof χ² at the
  measured σ (median 0.84 px → σ ≈ 0.71 px) gives in_mean about 1.6. The prediction kept the shoulder.
- **P3 above both bounds.** The older-row bound (< 2 %) sat below the floor written next to it in the card
  ("about 5 % of correct rows fail a 95 % test once"). Older rows are tested again in every solve while they are
  in the window (3 free keyframes plus fixed context), so correct rows keep failing at that rate. Measured
  older shares of 1.4-6.1 % lie at that floor. Newest shares of 15.5-16.0 % on four sequences are 0.5-1 point over
  the 15 % bound; they track the tail size of each sequence (MH_01 8.4 %, V2_02 15.8 %). The bound was too
  tight, not the mechanism wrong.
- **P8 medians.** S0O (octave σ) shrinks the tail only to 8.8 % (as predicted: mismatches do not shrink with
  octave σ) but moves scores both ways: −11 to −14 % on V2_01 and V1_03, +54 % on MH_04 (smoothed 0.347;
  causal 0.361). It also scales the tracking PnP gates (system/build.rs:199). Unverified: on MH_04 the looser
  gates for coarse octaves admit more mismatches into tracking. S0O is not pursued alone.

**Further findings.**
- **S0P (`keyframe_observations: PnpInliers`, config only) takes most of the gain without the erase:** causal
  V2_01 0.050, V1_02 0.063, V1_03 0.118, V2_02 0.120, MH_04 0.254, MH_01 0.057. It beats S0E on V2_01, MH_04
  (both columns) and MH_01, and loses on V1_02, V1_03 and V2_02 (causal). The mismatches enter mostly at tracking
  (ORB-SLAM3 keeps them on purpose and erases later; with our 3-keyframe window that ordering loses on MH_04).
  The arms are complementary in where they win; S0P + S0E stacked is the obvious next arm (not run: stacking
  is outside this rung).
- **S1E (full-weight bias rows + erase)** is near S0E on smoothed (V2_01 0.043, MH_01 0.033) but worse on causal
  everywhere except MH_01 (V2_01 0.077, V1_03 0.101, MH_04 0.362). KD1's follow-up: S1 does not add to the erase.
- **Start offsets 30 and 60 widen the S0 band a lot on V1_02:** S0 causal 0.069 (s30) and 0.387 (s60) against
  0.102 at starts 0-2. The single-sample band of R0 understated V1_02's noise. S0E s60 is also the worst S0E
  start (causal 0.151, smoothed 0.101).
- **Real-time check (owner decision 2026-11-23; `E/r11b/rt.sh`, `E/r11b/rtreport.py`, `E/r11b/rt_table.md`).**
  S0 (`RT0`) and S0E (`RT0E`) with asynchronous mapping, `--pace-timestamps`, `taskset -c 0,1`, start 0, 2 runs
  each, one run at a time, started at load 3.3-5.6 with fewer than 2 `kornia-slam*` processes (another job's
  `offline-ba` process shared the machine for part of it). All 12 `rc=0`, headers show `local mapping:
  asynchronous` and, for RT0E, the erase line. Causal (primary) r1 / r2, then smoothed (non-causal):

  | seq | S0 causal | S0E causal | S0E causal mean ÷ Basalt | S0 smoothed | S0E smoothed | stale kf S0 / S0E | keyframes S0 / S0E |
  |---|---|---|---|---|---|---|---|
  | V1_02 | 0.405 / 0.258 | 0.115 / 0.169 | 3.13 (S0 7.30) | 0.114 / 0.131 | 0.081 / 0.069 | 416-421 / 357-374 | 498-500 / 474-476 |
  | V2_01 | 0.610 / 0.616 | 0.078 / 0.087 | 2.12 (S0 15.72) | 0.213 / 0.423 | 0.095 / 0.068 | 399-424 / 295-304 | 560-568 / 488 |
  | MH_04 | 0.329 / 0.339 | 0.322 / 0.312 | 2.78 (S0 2.93) | 0.302 / 0.260 | 0.266 / 0.237 | 163-185 / 50-82 | 480-481 / 374-381 |

  - The SYNC ranking holds in real time: V2_01 causal 0.61 → 0.08, V1_02 0.33 → 0.14 (means), MH_04 unchanged
    (0.33 → 0.32). Tracked 1683-2274 frames, the same in both arms.
  - Real time is worse than SYNC in both arms on V1_02 (S0E 0.115-0.169 against 0.040) and V2_01 (0.078-0.087
    against 0.055): the async worker leaves 295-424 keyframes stale per run (G54), and erases nothing on them
    until their own window. `erase_stale_links` stays 0 (RT0E V1_02 r1: 108161 unlinked).
  - Neither arm keeps the sensor rate on 2 cores: `frame_ms p90` 54-77 ms against the 50 ms frame period, and
    walls 104-180 s against 85-114 s of data (vocabulary load included). Not diagnosed here; it is the same in
    both arms, and S0E is not slower (p90 −1 to −7 ms, walls −14 to −29 s, fewer keyframes).
- **Verdict:** G56 is confirmed as a score source on the Vicon rooms and V1_02 (causal median gains 2.0-3.7x on
  V2_01, V1_02, V1_03, V2_02; V1_02 below Basalt on both columns), refuted on MH_04, neutral-to-better on MH_01. The knob stays off until the gate
  (all 11, real-time replay, 3 runs, load ≤ 6) and the full-sequence LaMAria guard.

## 2-core real-time budget (profile, 2026-11-23)

Question: why RT0 / RT0E (asynchronous mapping, `--pace-timestamps`, `taskset -c 0,1`) miss the 20 Hz sensor rate.
Read-only analysis plus two profile runs; no code change. Outputs in `E/prof/` (`logsplit.py`, `stages.py`,
`threadsamp.py`, `pmp_drive.sh`, `pmp_report.py`, their `.md` outputs). Binary `bin-r11b` (md5 e67615ad…).

- **Runs.** The 12 RT0/RT0E logs (`E/runs/RT0*_s0_r[12]`). Profile 1: `E/prof/RT0E_V1_02_prof_r1`, RT0E V1_02, paced,
  cores 0,1, `--debug --trace-out` (per-stage `trace/timings.csv`), per-thread CPU and run-queue wait from
  `/proc/<pid>/task/*/{stat,schedstat}` at 1 Hz. Load 4.7 at start, 1 other `kornia-slam*` process. Result as in
  the R11b check: wall 122 s, `frame_ms` p90 66.9, 476 keyframes. Profile 2: `E/prof/RT0E_V1_02_pmp_r1`, the same
  run under a gdb stack sampler (`perf` is blocked: `perf_event_paranoid` 4, no sudo), 1 Hz, 137 samples of all
  threads. Each sample stops the process for about 0.2 s, so wall was 160 s. Shares are good to about ±4 points
  (137 samples); absolute times come from profile 1.
- **The headline metric under-reports.** `[summary] frame_ms` is `SlamSystem::tracking_duration`. That
  subtracts the deferred ORB extraction (`system/mod.rs:784-790`). It does not see the work the app does before
  `process_stereo`: PNG decode and stereo rectification (`apps/kornia-slam-app/src/source/mod.rs:246`
  `rectify_pair`, f32-map bilinear `remap_u8` of both images on every frame). The tracking thread's cost per frame is
  `frame_ms + fe` plus that untimed source work.

### Where the time goes

Main (tracking) thread, profile 1, per frame class (ms, p50 / p90 / p99; `stages_RT0E_V1_02.md`; rows that join stages add the per-stage percentiles, so they are approximate):

| stage | keyframe (n=476, 28 %) | ORB frame, no keyframe (n=202) | flow-only frame (n=1010) |
|---|---|---|---|
| ORB left + right | 8.8 + 8.2 (p90 18.9 + 18.0) | 6.9 + 5.9 | 0 |
| stereo pyramids + matching | 8.6 / 19.1 / 30.6 | 5.9 / 15.8 / 23.6 | 0 |
| frontend total (`fe`) | 29.1 / 46.7 / 69.5 | 22.8 / 40.7 / 67.8 | 0 |
| KLT build + track | 4.0 / 10.1 / 15.0 | 2.5 / 8.3 / 14.1 | 4.8 / 11.7 / 18.9 |
| pose: projection search, consistency, PnP, local refine | 5.7 / 14.8 / 34.6 | 4.0 / 14.5 / 29.0 | 0.7 / 1.7 / 4.0 |
| **keyframe insert** (growth + fusion, loop place recognition, BA submit, track refresh) | **40.5 / 71.9 / 257.8** | 0 | 0 |
| other (flow stages, map-lock waits, IMU) | 3.8 / 8.7 / 19.2 | 2.4 / 7.9 / 18.2 | 1.2 / 3.2 / 15.6 |
| `frame_ms` (log) | 57.8 / 94.5 / 268.6 | 10.7 / 29.0 / 41.5 | 7.1 / 16.6 / 26.1 |
| **frame_ms + fe** | **90.3 / 132.0 / 312.1** | 33.3 / 65.0 / 94.1 | 7.1 / 16.6 / 26.1 |

Every frame above 50 ms is a keyframe or an ORB frame. 0-1 flow frames per run exceed 50 ms. For all 12 RT logs
(`logsplit.md`; `frame_ms + fe`):

| seq / arm | KF share | all frames p50 / p90 / p99 | KF frames p50 / p90 / p99 | KF `fe` p50 | flow frames p90 | BA thread (cores) | mean BA solve, iterations |
|---|---|---|---|---|---|---|---|
| V1_02 S0 (r1, r2) | 0.29 | 18/119/227, 17/110/186 | 104/158/306, 97/146/370 | 38, 36 | 20, 20 | 0.74, 0.74 | 259, 242 ms; 31 |
| V1_02 S0E | 0.28 | 16/109/187, 14/103/166 | 97/148/587, 91/130/318 | 34, 32 | 18, 17 | 0.65, 0.69 | 214, 196 ms; 31-32 |
| MH_04 S0 | 0.24 | 7/85/143, 8/89/139 | 81/116/170, 83/119/192 | 23, 26 | 11, 12 | 0.42, 0.44 | 105, 115 ms; 19-20 |
| MH_04 S0E | 0.18-0.19 | 5/78/131, 6/80/145 | 80/120/178, 81/125/224 | 21, 22 | 8, 9 | 0.29, 0.31 | 81, 87 ms; 21 |

(V2_01 is the same as V1_02: KF p50 91-101 ms, BA 0.58-0.67 core.) BA p50/p90 per solve are not in the logs, only
`ba_solve_ms` / `ba_solves`.

Whole process on 2 cores, profile 1 (per-thread `schedstat`; 1710 frames in 83.5 s = 97.6 ms of core time per
frame available):

| thread | CPU s | ms of core per frame | what it does (profile 2 samples) | run-queue wait s |
|---|---|---|---|---|
| main (tracking, source) | 46.1 | 27 | blocked 72 % of samples. Of the main thread's wall: rectify 42 %, PNG decode 12 %, loop closing on the tracking thread 10 % (place recognition, `verify_loop_candidate` Hamming scans, 9 loop fusions at 165-1219 ms each), keyframe growth + fusion 10 % (`neighbor_fusion_links`, descriptor matching), ORB 7 %, KLT 6 %, other tracking 8 %. **No pacing sleep in any sample: the thread is always behind.** | 1.5 |
| 2 rayon workers (unnamed, ORB / pyramid / `remap_u8`) | 79.7 | 47 | non-idle samples: **58 % `sched_yield` spin**, 36 % `remap_u8` (rectification), 6 % ORB / FAST / resize | 103 (51.5 + 51.7) |
| `kornia-local-mapping` | 81.9 | 48 | 64 % local VI-BA (`vi_ba_schur::assemble` alone 36 % of samples), 5 % merge + cull, 30 % idle | 6.0 |
| **all** | **207.7** | **121** | demand is 1.24× the 2 cores | |

Wall 122 s on 83.5 s of data. The tracking thread's timed work is 63.7 s (37 ms per frame). That covers ORB 13.2 s,
stereo 6.6 s, keyframe insert 24.2 s, KLT 9.1 s, pose 5.7 s and other 4.9 s. The untimed source work (rectify +
decode) is about 55 s (about 32 ms per frame), from the wall minus the timed work minus about 3 s of vocabulary load.

### Diagnosis

1. **The machine is oversubscribed: 121 ms of core time demanded per 97.6 ms frame budget.** Three items take
   about 0.75 of the 2 cores and are not SLAM work in the usual budget sense:
   - rayon idle spinning: about 46 s, 0.38 core;
   - stereo rectification by a scalar f32 `remap_u8` (`floorf` per pixel): about 30 s of worker time, plus the
     main thread waiting for it;
   - PNG decode: about 13 s, replay only.
2. **Fork-join on 2 cores convoys.** ORB and remap fan out to 2 rayon workers. The mapping thread keeps one core
   busy (0.67), and the workers spin. So each worker waits in the run queue for 51 s, and the main thread blocks on
   the slowest chunk: main CPU is 46 s against about 118 s busy wall. This is why `fe` p90 is 47 ms for 17 ms of
   ORB p50.
3. **Keyframe frames carry mapping work on the tracking thread.** `keyframe_insert` p50 is 40.5 ms: growth + fusion
   (`system/mod.rs:3221` `grow_keyframe`) and loop place recognition / verification (`:3350`
   `register_place_recognition`, under the map lock). ORB-SLAM3 runs both in its LocalMapping and LoopClosing
   threads. Each loop fusion stalls tracking for 0.17-1.2 s, and those stalls make the p99.
4. **Local BA runs 31 iterations per solve on V1_02 (cap 50, λ 1.0) at about 6.4 ms per iteration.** That is
   0.65-0.74 core. Basalt runs 7 iterations from λ 1e-4. Faster solves would also cut the G54 stale keyframes: the
   latest-wins queue drops jobs while a solve runs.
5. **Keyframe rate: 18-29 % of frames.** `keyframes.min_frames_between: 3` means up to 6.7 keyframes per second, each
   with stereo ORB (2 × 1000 keypoints) and stereo matching on its own pyramids (`stereo_pyramid: Resized`, 8.6 ms).
   Another 12 % of frames (202 on V1_02) run ORB without becoming keyframes (G33's mechanism on EuRoC).

**Compared with Basalt** (paper about 5-8 ms frontend per frame; our bench reproduction is not a quiet-machine number,
`bench/README.md` §Tracking): Basalt never rectifies images (KLT patches on raw images through the camera model). It
has no descriptors (FAST on new keyframe corners only), so it has no per-keyframe ORB × 2 + stereo match. It has no
growth or loop work on the frontend thread, and 7 VIO iterations. Our flow-only frame (7.1 ms p50, KLT about 5 ms)
is already in Basalt's range. Our keyframe frame (90 ms p50) and the untimed source work (about 32 ms per frame) are
the gap.

### Ranked speed fixes

Savings are for RT0E V1_02 on 2 cores. Core time is per frame (97.6 ms available); KF ms is the keyframe-frame
critical path.

| # | fix | expected saving | existing knob | accuracy risk | card / rung |
|---|---|---|---|---|---|
| 1 | Stop rayon spinning and oversubscription: one rayon worker or none on the tracking path | about 27 ms core per frame (0.38 core); `fe` p90 should drop toward ORB + stereo p50 (≈ 47 → 30 ms) because the convoy goes | `RAYON_NUM_THREADS=1` (env, no code); a pinned or sequential pool needs code | none expected (ORB / remap are per-pixel; check `fe` keypoint counts match) | card T (thread placement, as `dense_parallelism: Sequential`) |
| 2 | Rectification: precomputed fixed-point maps + SIMD bilinear (OpenCV `convertMaps` 16SC2 + `remap` class, about 1 ms for both 752×480 images), or rectify keypoints instead of images | about 18-20 ms core per frame, and about 25 ms off the main thread's wall per frame | none | low: ≤ 1 LSB pixel change with 5-bit weights; keypoint-only rectification changes ORB input (medium) | frontend track F/K (kernel must match and beat OpenCV) |
| 3 | Local VI-BA budget `max_iterations: 7`, `initial_lambda: 1e-4` (window 3 kept) | about 36 ms core per frame (BA 0.67 → about 0.15-0.2 core), and fewer stale keyframes (G54) | `local_ba.max_iterations`, `local_ba.initial_lambda` | medium: card V saw no score change at 7 on LaMAria; not run on EuRoC; needs the full-sequence gate | R4 (budget half, without window 10) |
| 4 | Move keyframe growth + fusion and loop place recognition / verification / fusion off the tracking thread | KF frame −35 to −45 ms (p50 90 → about 50); p99 loses the 0.17-1.2 s loop stalls; core time unchanged | none (`keyframe_orb: Mapping` is parsed but `UNBUILT`, `validation.rs:544`) | medium: growth lands one or more frames later (G54-like staleness for new points) | card T lane / R-C6 (deferred keyframe path) |
| 5 | Shared stereo pyramid | 2-4 ms per ORB frame (pyramid builds inside the 8.6 ms) | `frontend.stereo_pyramid: Shared` | low-medium: matches differ (`frontend/mod.rs:176`) | track F |
| 6 | Fewer ORB frames: KLT-only keyframes (FAST corners on keyframes, descriptors on the mapping thread), cut keyframe-due ORB rejects | up to about 30 ms per keyframe frame; 202 ORB non-KF frames × 33 ms | `keyframe_orb: Mapping` (unbuilt); `klt_first` already on; card Z `flow_ratio: TrackSurvival` (not built) | medium-high: loop closing and relocalisation need descriptors | track F/K, card Z (G33) |
| 7 | Harness: decode PNGs off the 2 cores (prefetch thread unpinned or pre-decoded raw) | about 7.6 ms per frame; replay-only cost, not on the robot | none | none | measurement protocol (§2) |
| 8 | Keyframe rate: `min_frames_between` 3 → 5 | KF work × 0.6-0.7 | `keyframes.min_frames_between` | high: per-deployment tuning, against the owner's no-tweak rule | not proposed |

Fixes 1 + 2 + 3 together: about 207.7 → about 70 s of CPU (about 41 ms of core per frame against 97.6). That leaves
the keyframe critical path (90 ms p50) as the only miss against the 50 ms period. Fix 4 then brings it near 50, and
fix 6 brings it toward Basalt's range. The 20 ms-per-Orin-core rule is met today only by flow frames (p90 16.6 ms on
this laptop core).

Measurement fix needed first: `[summary] frame_ms` should report `frame_ms + fe` and the source time, or the RT table
keeps understating. That is a code change and outside this read-only card.

### Results: R11c, RT1E real time with PropagateToNewer, and S0PE stacked (2026-11-23)

Binary `E/bin-r11b/kornia-slam-euroc`, md5 `e67615adaf558b4aef7b7d1e35a2fb40` (unchanged, config-only arms). Predictions
committed in 83b2c98 before the runs. All 46 runs `rc=0`; every header `diff`-identical to its golden file
(`E/expect/RT1.header`, `RT1E.header`, `S0PE.header`: `local BA merge: PropagateToNewer` and/or the erase line).
Scores `E/score.py` (`E/r11b/scores.jsonl`); tables `E/r11c/rtreport.py` (`E/r11c/rt_table.md`) and
`E/r11b/report.py S0PE` (`E/r11c/report_S0PE.md`). SE(3) ATE RMSE, m, body centre; causal (`traj.txt`) primary.

**RT1 / RT1E (real time).** `E/r11c/rt.sh`: async, `--pace-timestamps`, `taskset -c 0,1`, start 0, r1 and r2,
one RT run at a time, 21:01-21:52, started at 1-min load 1.9-5.8 with < 2 `kornia-slam*` processes (a profiling
run of RT0E V1_02 and the rig sweep's last run overlapped the first RT1E V2_01 run). Causal r1 / r2 (mean):

| seq | Basalt | S0E SYNC causal med | P3 bound | RT0 | RT0E | RT1 | RT1E | RT1E ÷ S0E sync | RT1E ÷ Basalt | stale = propagated (RT1E) | kf shifts RT1 / RT1E | frame_ms p90 RT1E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V2_01 | 0.039 | 0.055 | 0.083 | 0.610 / 0.616 | 0.078 / 0.087 | 0.134 / 0.196 (0.165) | 0.122 / 0.128 (**0.125**) | 2.28 | 3.20 | 285, 295 | 0, 0 / 0, 0 | 61.8 / 54.3 |
| V1_02 | 0.045 | 0.040 | 0.060 | 0.405 / 0.258 | 0.115 / 0.169 | 0.073 / 0.057 (0.065) | 0.044 / 0.056 (**0.050**) | 1.24 | 1.10 | 337, 341 | 0, 0 / 0, 0 | 69.5 / 43.6 |
| V1_03 | 0.053 | 0.072 | 0.108 | - | - | 0.195 / 0.242 (0.218) | 0.153 / 0.124 (**0.139**) | 1.94 | 2.62 | 434, 425 | 10, 11 / 10, 11 | 46.7 / 43.5 |
| V2_02 | 0.049 | 0.080 | 0.120 | - | - | 0.154 / 0.155 (0.154) | 0.119 / 0.100 (**0.109**) | 1.36 | 2.22 | 458, 428 | 0, 0 / 0, 0 | 58.1 / 52.8 |
| MH_04 | 0.114 | 0.311 (0-2) | 0.47 | 0.329 / 0.339 | 0.322 / 0.312 | 0.326 / 0.309 (0.317) | 0.332 / 0.299 (**0.316**) | 1.02 | 2.77 | 44, 30 | 0, 0 / 0, 1 | 41.0 / 36.6 |
| MH_01 | 0.067 | 0.072 | 0.108 | - | - | 0.078 / 0.098 (0.088) | 0.066 / 0.083 (**0.075**) | 1.04 | 1.12 | 43, 38 | 0, 0 / 0, 0 | 43.9 / 40.0 |

Smoothed (non-causal) RT1E r1 / r2: V2_01 0.070 / 0.077, V1_02 0.038 / 0.046, V1_03 0.075 / 0.085, V2_02 0.051 / 0.093,
MH_04 0.229 / 0.240, MH_01 0.038 / 0.035. Tracked 2144-3682 frames, within 3 of SYNC on every run; frames read =
dataset length on every run (no frame drops). Walls 87-143 s (Vicon), 103 s (MH_04), 186 s (MH_01).

| P/K | prediction | measured | verdict |
|---|---|---|---|
| P1 | propagated = stale, every RT1/RT1E run | RT1E 12 of 12 equal (30-458). RT1 10 of 12; MH_01 r1 105/104, MH_04 r2 147/146 (`stopped_at_bridge` 0) | missed by letter (2 control runs off by 1) |
| P2 | `ba_keyframe_shifts` ≤ 10, every run | 0-1 on five sequences; V1_03 10 (r1) and 11 (r2) in both arms (RT0 had 32/16 on V2_01, 13/3 on V1_02) | missed (V1_03 r2, 11) |
| P3 | RT1E causal mean within 1.5x of S0E SYNC on each | V1_02 0.050, V2_02 0.109, MH_04 0.316, MH_01 0.075 inside; V2_01 0.125 (> 0.083), V1_03 0.139 (> 0.108) | missed (4 of 6) |
| P4 | frame_ms p90 inside 54-77 ms | 36.6-69.5 ms; 16 of 24 runs below 54 | missed (below the range) |
| K1 | shifts ≤ 10 and > 2x S0E SYNC on ≥ 3 | above 2x only on V2_01 (0.125 > 0.110); V1_03 0.139 < 0.144 | not fired |
| K2 | P3 met on ≥ 4 with propagated < stale | propagated = stale on RT1E | not fired |

**Diagnosis of the misses (owner rule; unverified parts marked).**
- **G54 is fixed in real time; it is not the whole real-time loss.** RT1E propagates every stale keyframe and
  shifts are gone outside V1_03. RT1 against RT0 is the G54 effect alone: V2_01 0.613 → 0.165, V1_02 0.331 → 0.065.
  With the erase on, V1_02 goes 0.142 → 0.050 (1.10x Basalt), but **V2_01 gets worse, 0.083 → 0.125** (both RT1E runs
  above both RT0E runs). By 10 s bin (whole-run SE(3) alignment) RT1E V2_01 is above RT0E at 0-20 s (0.14-0.17
  against 0.08-0.09 m) and 100-130 s (0.17-0.20 against 0.10-0.12). Unverified: the rigid propagation carries the
  window's correction into keyframes whose own solve then never comes (next item), so early, large corrections
  (before and around the inertial init) persist.
- **What remains is mapping lag on the Vicon rooms.** No frames are dropped and tracking is the same as SYNC. The
  measurable difference to SYNC is the mapping worker: on V2_01, V1_02, V1_03 and V2_02 async inserts 16-34 % more
  keyframes than S0E SYNC (V2_01 475-483 against 354; V1_03 593-595 against 496) and runs 0.83-0.92 solves per
  keyframe (SYNC 1.00), so 8-17 % of keyframes never get a solve as the newest keyframe. A solve takes 109-225 ms
  on 2 cores against a keyframe every 3-8 frames (150-400 ms). On MH_04 and MH_01 solves per keyframe are
  1.00-1.01, keyframes +1-3 %, and RT1E equals SYNC (0.316 against 0.311, 0.075 against 0.072). The residual gap
  follows the lag sequence by sequence. Unverified as the cause: more keyframes are inserted because the tracker
  sees fewer refined map points while mapping is behind.
- **P2 on V1_03:** 10-11 shifts in both arms (V1_03 has no RT0 baseline); RT1 and RT1E alike, so not the erase.
  Not traced to keyframes here.
- **P4:** frame time fell, it did not rise. Load at start was 1.9-5.8 (RT0/RT0E: 3.3-5.6 with another job's
  `offline-ba` process for part of it), and per-solve time fell on MH (MH_04 53-59 ms against 79 ms in SYNC under
  load 6-14). The range was a machine-state number, not a property of the arm.

**S0PE (SYNC, S0P + S0E stacked).** `E/drive.sh` with `BIN=bin-r11b/kornia-slam-euroc`, lists `E/lists/r11c_S0PE{a,b}.txt`,
2 at a time, 21:52-22:17, load 2.6-9.2. Causal median (per start) against S0E and S0P on the same starts:

| seq | Basalt | S0E causal | S0P causal | S0PE causal per start | S0PE median | ≤ min(S0E, S0P)? | S0PE ÷ Basalt | smoothed S0E / S0P / S0PE |
|---|---|---|---|---|---|---|---|---|
| V2_01 | 0.039 | 0.055 | 0.050 | 0.049 / 0.056 / 0.066 | 0.056 | no (worse than both, by 0.001 against S0E) | 1.43 | 0.047 / 0.044 / 0.045 |
| V1_02 | 0.045 | 0.040 | 0.063 | 0.036 / 0.036 / 0.036 / 0.042 / 0.110 | 0.036 | yes | 0.80 | 0.038 / 0.046 / 0.030 |
| V1_03 | 0.053 | 0.072 | 0.118 | 0.111 / 0.166 / 0.085 | 0.111 | no (between) | 2.08 | 0.063 / 0.087 / 0.070 |
| V2_02 | 0.049 | 0.080 | 0.120 | 0.099 / 0.091 / 0.091 | 0.091 | no (between) | 1.86 | 0.079 / 0.115 / 0.089 |
| MH_04 | 0.114 | 0.325 | 0.254 | 0.227 / 0.227 / 0.229 / 0.225 / 0.235 | 0.227 | yes | 1.99 | 0.250 / 0.179 / 0.156 |
| MH_01 | 0.067 | 0.072 | 0.057 | 0.042 / 0.050 / 0.049 | 0.049 | yes | 0.73 | 0.036 / 0.028 / 0.029 |

Run health against S0: tracked −6 to +4 frames, give-ups 0; keyframes 0.78-0.92x; map points per keyframe −16 % worst
(V1_02 s0 98.5 against 116.7). Erase `erase_unlinked` (start 0) 0.30-0.60x of S0E (V1_02 30 350 against 95 781,
MH_01 19 561 against 60 724, V2_01 76 663 against 128 734): PnpInliers keeps most mismatches out first, as stated.

| P/K | prediction | measured | verdict |
|---|---|---|---|
| P5 | causal median ≤ min(S0E, S0P) on ≥ 4 of 6 | 3 of 6 (V1_02, MH_04, MH_01) | missed |
| P6 | MH_04 causal median ≤ 0.254 | 0.227 (0.225-0.235 over 5 starts; S0P s60 was 0.419) | as predicted |
| counter | `erase_unlinked` below S0E on the same start | 0.30-0.60x on all six | as predicted |
| K3 | worse than both on ≥ 3 | 1 (V2_01, by 0.001 m; per-start values overlap both bands) | not fired |

**Diagnosis of the P5 miss (unverified parts marked).** The stack wins where S0P beat S0E (MH_04, MH_01) and on V1_02,
and lands between the two where S0E beat S0P (V1_03, V2_02). So what S0P loses on those two is not mismatches the erase
would have caught: the erase can only remove rows, and with PnpInliers it removes 0.30-0.48x as many there. Unverified:
on the fast Vicon sequences the tracking PnP gate rejects correct observations (motion-blurred or prediction-shifted
matches) that S0E keeps and BA then uses. V2_01 is inside noise (per-start 0.049-0.066 overlaps S0E 0.052-0.058).

**Verdict.**
- RT1E: G54 fixed under real time (P1, shifts), real-time ÷ SYNC 1.02-1.36 on V1_02, V2_02, MH_04, MH_01, but 1.94-2.28
  on V1_03 and V2_01. K1 not fired; the remainder is diagnosed as mapping lag (unverified as cause). PropagateToNewer
  is kept for the real-time arms (RT0 → RT1 removes the 0.3-0.6 m failures) but it costs V2_01 with the erase on.
- S0PE: best SYNC arm on V1_02 (0.036, 0.80x Basalt), MH_04 (0.227) and MH_01 (0.049, 0.73x); worse than S0E on V1_03
  and V2_02. Not promoted over S0E; neither knob is promoted (gate and LaMAria guard first).
- Next: a mapping-lag rung under real time before the gate. Config-only first: RT1E with a smaller local-BA iteration
  budget (solve time below the keyframe interval), measuring solves per keyframe and keyframes against SYNC; then, if
  lag is the cause, ORB-SLAM3's rule that tracking inserts a keyframe only when local mapping accepts one
  (`Tracking::NeedNewKeyFrame`, `LocalMapping::AcceptKeyFrames`) behind a knob. RT1PE (S0PE under real time) is run in
  the same rung as the stacked candidate.

## Residual diagnoses (2026-11-23)

Read-only analysis of the existing runs plus one pair of short debug runs. No code change, no build. Binary
`E/bin-r11b/kornia-slam-euroc`, md5 `e67615adaf558b4aef7b7d1e35a2fb40`. Scripts are in `E/resid/`:
- `bins.py`: whole-run SE(3) error per bin, with tilt, piece scale and piece yaw;
- `anch.py`: error per bin under an SE(3) alignment fitted on one span only;
- `cf.py`: ATE with one event's error increment removed;
- `step.py`: per-frame vertical error, split into keyframe and tracking frames;
- `dbg.py`: per-bin debug counters;
- `climb.py`: vertical error rate against GT climb and stereo depth;
- `rotdrift.py`: world-frame yaw and tilt drift, from `R_gt R_est^T` against a reference span;
- `kfrate.py`: keyframes per second.

SE(3) ATE RMSE, m, body centre; causal is `traj.txt`.

### Q1: MH_04 (SYNC causal S0 0.302, S0E 0.311, S0P 0.254, S0PE 0.227; Basalt 0.114)

**Verdict.** Most of the MH_04 causal error comes from one event, not from tilt held in the frozen past. During
63.3-66.7 s (the dark span) the trajectory loses 0.42-0.55 m of height. The size is the same in every arm (S0, S0E,
S0P, S0PE, S1, S1E, S0O) and on every start (S0PE starts 0/1/2/30/60). The offset then stays flat until the loop at
81.5 s. With that one increment removed, S0PE scores 0.110-0.134 on its starts, at Basalt's level (0.114). Online
information can fix it: Basalt keeps the vertical error within 0.07 m through the same span with VIO alone. The
offline global BA (0.075) removes it, but too late for the causal output.

**What the "40-50 s and 70-80 s" error is.** It is the whole-run SE(3) fit spreading one offset. The error per 10 s
bin under the whole-run fit (`bins.py`) shows 0.35-0.49 m at 40-50 s and 0.38-0.55 m at 70-80 s. The absolute tilt
in those two bins is 0.7-1.2° and 2.4-4.0°. A fit on 15-30 s only (`anch.py`, 2.5 s bins) shows when the error
enters:

| span (s) | what happens (S0PE; S0 in brackets) | vertical error change, m |
|---|---|---|
| 25-40 | almost nothing: vertical error +0.01 to −0.06 m (S0 ±0.02), horizontal 0.01-0.08 (S0 0.01-0.13) | −0.06 (S0 0) |
| 40-56 | slow drift; includes the GT step at 46.5 s seen by every system (MH04-DIAGNOSIS) | −0.15 (S0 −0.19) |
| 56-63.3 | dark onset; median stereo depth 10.4-16.7 m against 3.7-7.8 m elsewhere | −0.10 (S0 −0.34) |
| **63.3-66.7** | **the step, while GT climbs at +0.3-0.54 m/s** | **−0.54 (S0 −0.35; −0.42 to −0.55 over all arms)** |
| 66.7-81.5 | flat: −0.73 to −0.86 m held, although the tilt is 2.4-4° (no further vertical drift) | about −0.05 |
| 81.5 | loop to the 38 s map, `map_from_odometry` t_y −0.81 (S0PE) to −1.14 m (S0E), 0.3-1.0°; the error returns to 0.0-0.1 m | +0.8 |

Counterfactual (`cf.py`; the increment ramped out over the span until the 81.5 s loop, then whole-run SE(3)):

| arm (start 0) | base | 40-56 removed | 56-63.3 removed | 63.3-66.7 removed |
|---|---|---|---|---|
| S0 | 0.302 | 0.274 | 0.219 | 0.214 |
| S0E | 0.311 | 0.294 | 0.261 | 0.187 |
| S0P | 0.254 | 0.222 | 0.213 | 0.139 |
| S0PE | 0.227 | 0.202 | 0.202 | **0.118** (starts 1/30/60: 0.123, 0.110, 0.134) |

Basalt (`B/results/basalt/MH_04_difficult/trajectory.txt`, same fit) stays at −0.07 to +0.05 m vertical over 25-100 s
and 0.07-0.18 m per 10 s bin. ORB-SLAM3 without loops dips to −0.12 m at 68-72 s and recovers.

**Mechanism of the step (debug runs `E/debug/R11b-S0-MH04`, `R11b-S0E-MH04`, full MH_04, bit-identical to their batch
runs).**

1. **The drift accumulates on tracking frames, not in the BA solves.** Over 62-67 s the vertical error changes by
   −1.10 m (S0E) and −1.24 m (S0) on non-keyframe frames, and by +0.48 / +0.75 m on keyframe frames (`step.py`). The
   frames form a saw-tooth. Each solve moves the newest keyframe up by 0.13-0.22 m, toward GT. Examples: S0E kf
   1309, 1317, 1325 at 65.4-66.3 s, with `shift` 0.19 / 0.22 / 0.17 m. The next tracked frame then falls back by
   0.14-0.16 m onto the map. `[predgap]` at frame 1310 shows `trans=0.18` m against the tracker's own prediction.
2. **The map, not the solve, holds the offset.** In those windows 512 of 550 landmarks are `pinned` (seen by fixed
   keyframes), and 70 % (S0E) to 90 % (S0) of the rows are fixed context. The 3 free keyframes can move, but the landmarks the
   tracker uses cannot. New landmarks (`grown` 40-120 per keyframe) are triangulated at the low tracked poses, so
   the offset propagates into the new map.
3. **Distant points leave vertical translation and pitch on one ridge.** A camera that climbs by t and pitches by
   θ ≈ t/z gives the same image motion, and the points here sit at z ≈ 11-12 m. Per 1 s bin, the vertical error rate
   follows −0.73 × the GT climb rate where the median depth is above 9 m (n = 8). Below 9 m the slope is −0.01
   (n = 36; `climb.py`): we track climbs where points are near.

   The window rotates along the ridge. The S0E solves at 65.3-66.7 s turn the free keyframes by 0.7-1.5° of tilt
   each (`shift_tilt`), and the absolute tilt goes 1.6 → 3°. At 12 m, 1.4° is about 0.3 m of vertical, the order of
   the step.

   The IMU that would break the ridge is weak there:
   - The tracker is visual-only PnP (G13, `tracking.pose_inertial: Off`).
   - In the local BA, the boundary IMU edge to the past is Huber-cut: hw 0.33-0.40, χ² 103-160 at kf 1317-1333.
   - The accel bias takes the tilt: b_a error 0.12 (60 s) → 0.30 (65 s) → 0.48-0.54 m/s² (67.5-70 s), against
     g·3° = 0.51 (G49).
4. **The dark span degrades the rows the solve sees.** S0E erases 44 % of the newest keyframe's rows in 60-70 s
   (whole run 15.5 %). With 6-7 keyframes per 2.5 s (S0 16-17), keyframes are 8 frames apart, and inliers fall to
   97-117. S0 (no erase) carries 26-33 % of rows above 25 px² in these windows (`VISRES out`).

**Q1 checks asked for.**
- **Scale:** 10 s piece scale is 0.996-1.016 over 20-90 s in all four arms (`bins.py`).
- **Yaw:** piece yaw is within −3.6 to +5.4° of the global fit over 20-90 s. The 60-70 s piece has +3.8 to +5.4°,
  partly the tilt projected onto the fit.
- **Coasts:** `coast_inertial=0 coast_visual=0 give_ups=0`, 2032 / 2032 tracked, in all four arms.
- **Keyframe shifts:** 0 (S0, S0P, S0PE) and 1 (S0E).
- **Visual χ²:** in S0E, the median final visual cost per solve is 9.3-16 k at 57.5-70 s against 1.0-7.2 k
  elsewhere. In S0 it is 130-267 k at 60-70 s.
- **IMU χ²:** median final IMU cost per solve 19-42 at 60-70 s against 3-15 before 57.5 s (S0E).
- **Observations per keyframe:** in the dark the window holds 2.3-5.0 k rows (S0E) and 10-21 k (S0).
- **Loop corrections:** listed in the table above.

**Why PnpInliers helps (S0 → S0P −0.048, S0E → S0PE −0.084).** It does not touch the step: −0.49 / −0.54 against
−0.35 / −0.52 m. It lowers everything else:
- Under the 15-30 s fit, the horizontal error at 42-62 s is 0.09-0.18 m for S0P and 0.05-0.16 for S0PE, against
  0.20-0.35 for S0.
- Whole-run bins at 0-30 s are 0.06-0.07 against 0.09-0.13.
- The 56-63.3 s drift is −0.17 / −0.10 against −0.34 / −0.20.

Mechanism: with `AllMatches`, the tracker links its PnP outliers into the new keyframe. They feed the KLT tracks and
the found counts too (`tracking/pose_estimation/map_projection/mod.rs:131-142, 391-399`). The keyframe's first local
solve, which sets its pose, velocity, bias and new landmarks, sees them before any erase can act. Measured:
- S0E erases 15.5 % of newest-keyframe rows after that first solve (`erase_unlinked_newest / erase_tested_newest`
  = 19305 / 124588), and 44 % in the dark.
- S0PE erases 4.2 % (4628 / 110802).

The erase cleans later windows only. PnpInliers cleans the first solve. MH_04 gains from it and V1_03 / V2_02 lose
(the R11c P5 diagnosis stands).

**Frozen past or online?** Online.
- The step is born in the newest keyframes' tracking, at 63-67 s. The keyframes before 60 s were right (error −0.25
  m and below).
- The loop correction is 4-DoF and comes at 81.5 s, after every causal frame of the offset was written.
- The global solve (R6, `E/r6/S0E_MH_04_difficult`) brings the final map to 0.075. Tilt is ≤ 0.24° per 10 s and
  −0.19 m remains at 65 s. That fixes the smoothed output only.
- S1 (bias rows full) lowers the 65-79 s tilt 3.74 → 2.46° but not the causal error (0.302 → 0.327). The step in S1
  is still −0.42 m.

So the tilt is a by-product of the ridge, not the cause of the score. The fix has to put the IMU on the ridge in
tracking, or free the map.

### Q2: V2_01 real time (RT0E 0.078 / 0.087 → RT1E 0.122 / 0.128)

**Verdict.** PropagateToNewer copies a wrong local-BA correction into keyframes the solve never saw, and that
happens under mapping lag. In V2_01 at 54-55 s (slow, rotation-dominant motion: speed 0.14-0.26 m/s, gyro peak 1.24
rad/s), the solves rotate the window by +5.5° of yaw within 3 solves. In those solves 48-56 % of the newest
keyframe's rows fail χ², and the boundary IMU edge is Huber-cut (hw 0.03-0.2).

RT1E carries that rotation rigidly into the 1-2 newer keyframes per merge and their landmarks
(`mapping/map/ops/correction.rs:781-796`; `propagate_to_newer` and `carry_landmarks` from line 856). The trajectory then holds a +3.0 to +4.0° yaw error from 54 s (both
RT1E runs; RT1 alike, +2.5 to +4.2°). RT0E leaves the newer keyframes on their tracked poses, and its yaw stays within
±1°.

After the 62.6 s loop (4-DoF, 5.5-7.5° in RT1E), RT1E keeps a tilt of 3-5.4° to the end. RT1 without the erase heals
to 0.4-1.7° over 66-106 s. The extra error at "0-20 s and 100-130 s" is the whole-run fit spreading this later tilt and yaw: a fit
on 25-50 s puts RT1E and RT0E within 0.02 m of each other before 50 s (`anch.py`).

**Evidence.**
- World-frame drift against 30-45 s (`rotdrift.py`, 4 s bins, yaw / tilt in degrees):

  | arm | yaw 50 s | yaw 54 s | yaw 58 s | tilt 62 s (loop) | tilt 66-110 s |
  |---|---|---|---|---|---|
  | RT0 r1 / r2 | −3.9 / −6.4 | −3.7 / −7.6 | −1.0 / −8.3 | 4.6 / 17.7 | 0.4-3.6 / 16-21 |
  | RT1 r1 / r2 | −0.4 / −0.1 | +2.5 / +4.2 | +2.2 / +3.7 | 1.5 / 0.8 | 0.4-2.7 |
  | RT0E r1 / r2 | −1.1 / −0.8 | −0.1 / −0.3 | +0.2 / −0.3 | 1.9 / 2.2 | 0.4-2.0 |
  | RT1E r1 / r2 | −0.3 / −0.2 | +3.0 / +3.2 | +3.6 / +3.7 | 4.0 / 5.4 | 3.0-5.3 |
  | S0E SYNC | +0.3 | +1.0 | +0.6 | 0.4 | 0.3-1.2 |

- **Debug pair (new, `E/debug/Q2-RT1E-V201-s850`, `Q2-RT0E-V201-s850`).** V2_01 from frame 850 (42.5 s), 450 frames,
  RT1E / RT0E config, async, `--pace-timestamps`, `--debug`, `taskset -c 6,7`, one at a time, load 3.1 / 4.7, one
  other `kornia-slam` process (the R11d driver on cores 0,1). Both runs rc 0, walls 47 / 41 s, and both hit the same
  loop (local kf 401 ≈ frame 1251 to the 49.3 s map; correction 5.12° and 6.55° in RT1E, 1.73° in RT0E). Both
  reproduce the event:
  - at 54-55 s the tilt rises to 2.2-4.3° in both;
  - RT1E then jumps +4.5° in yaw at 55 s and holds +3.7 to +4.7° to 61 s; RT0E stays at −1.2 to +0.4°;
  - before the loop (57-60 s), RT1E holds a tilt of 2.0-2.8° and RT0E recovers to 0.6-1.3°;
  - after the loop, both rise (5.3 / 3.8°). A start at 42.5 s re-initialises the IMU during the event, so the
    post-loop part does not separate here. It does in the full runs.

  The window solves rotate alike in both arms: shift_yaw sums of +5.5 / +4.3° in the 54 s bin. The output differs
  because of the carry. The erase share per 1 s bin, RT1E against RT0E:

  | bin | all rows | newest keyframe | older free keyframes (the ones earlier merges carried) | fixed context |
  |---|---|---|---|---|
  | 50-53 s | 11.9-14.6 % / 10.3-16.4 % | 26-39 % / 25-36 % | 18-31 % / 21-31 % | 6.4-9.4 % / 5.8-9.5 % |
  | **54 s** | **24.9 % / 12.8 %** | 55.5 % / 48.0 % | **55.8 % / 27.4 %** | **12.2 % / 5.2 %** |
  | 55 s | 25.7 % / 23.7 % | 55.8 % / 51.7 % | 35.8 % / 46.2 % | 18.2 % / 14.0 % |
  | whole run | 12.6 % / 11.2 % | 28.4 % / 27.5 % | 17.5 % / 17.2 % | 8.9 % / 7.5 % |

  In the first second of the event, RT1E unlinks twice the share of rows (older free keyframes and fixed context).
  Those are the rows that disagree with the carried rotation. Unlinked rows are gone for good. Under Huber (RT1)
  they stay and pull the map back after the loop. This "erase locks the carried error in" step is supported by one
  non-deterministic pair: **unverified as the only cause.**
- **The three checks asked for.**
  - Erase counts per solve with propagated keyframes in the window: the table above. They are equal to RT0E outside
    the event and 2x in its first second.
  - Stale-propagated keyframes judged by the erase: never in the merge that carries them. `erase_stale_links=0` on
    every run, and the erase only tests the solved window's links (`mapping/outlier_erase.rs:153-200`). They are
    judged in their next window, as "older free" rows.
  - Mapping lag: solves per keyframe are 0.86-0.87 (RT1E) against 0.82-0.84 (RT0E), and 0.69-0.71 in the debug pair.
    Lag is the precondition: at 49-62 s, RT runs insert 6-7 keyframes per second against 2-5 in SYNC (`kfrate.py`).
    It is not what separates RT1E from RT0E.
- **Post-hoc check on finished R11d runs (looked at after the diagnosis above was formed; mechanism counters only, no
  score read; not predictions).** V2_01 r1:
  - RT2E (7 iterations, `RAYON_NUM_THREADS=1`): `ba_stale_keyframes=0`, 359 keyframes / 363 solves, yaw −1.3 to
    +0.8°, tilt 0.3-1.0° over 46-114 s.
  - RT2E-r (rayon only; 246 stale = propagated): +4.0 / +4.8° yaw at 54 / 58 s and 3.6-4.9° tilt after the loop,
    the RT1E signature.
  - RT1PE (PnpInliers; 86 stale): yaw −0.9 to +0.7°, tilt 0.3-1.3°.

  Removing either precondition removes the signature: the lag (every keyframe solved as newest) or the contaminated
  newest rows (PnpInliers).

### Fix rungs

**R13a (config-only, SYNC, decides Q1): per-frame pose-inertial tracking on S0PE.** Arm `S0PEI` = S0PE +
`tracking: ( pose_inertial: OrbSlam3 )` (`K/system/config/mod.rs:482-493`; ORB-SLAM3 `Tracking.cc:2970-2992`,
`Optimizer.cc` `PoseInertialOptimizationLast{Frame,KeyFrame}`). This moves R7 ahead of R3+R4 on MH_04 only.

Why the anchor bias is good enough here: in the dark the b_a error is 0.12-0.30 m/s². Over the longest gap between
keyframes (0.4 s) it costs ½·0.3·0.4² = 0.024 m, against the 0.14-0.16 m the tracker loses per keyframe interval.
The gyro bias error is 0.5 mrad/s. So the IMU decides along the pitch/vertical ridge.

Runs: MH_04 starts 0/1/2/30/60. Guards: V1_02, V2_01, MH_01 starts 0/1/2. The header does not print
`tracking.pose_inertial` (`K/system/config/mod.rs:800-900`), so the header must `diff` clean against
`E/expect/S0PE.header`, and a run counts only with `pose_inertial_solves` > 0.
- P1 (mechanism): the MH_04 vertical increment over 63.3-66.7 s (`E/resid/cf.py`, fit on 15-30 s) is ≥ −0.25 m on
  every start. S0PE: −0.50 to −0.54.
- P2: MH_04 causal median 0.12-0.17, every start ≤ 0.19 (S0PE 0.225-0.235). The counterfactual floor is 0.110-0.134.
- P3: MH_04 tilt median 65-79 s ≤ 1.5° (S0PE 2.01-2.76°, `r11b/tiltwin.py`).
- P4 (health): tracked = frames, `coast_inertial=0`, `give_ups=0` and `pose_inertial_solves` > 0 on every run.
- P5 (guards): causal median ≤ 1.15x S0PE: V1_02 ≤ 0.041, V2_01 ≤ 0.064, MH_01 ≤ 0.056.
- K1 (card H defect on EuRoC): any run below 100 % tracked or with `coast_inertial` > 0. Stop; card H first.
- K2 (mechanism, no score): P1 holds but the MH_04 causal median is > 0.19. The step is not the score; diagnose.
- K3 (score, no mechanism): MH_04 causal median ≤ 0.17 while the step stays ≤ −0.40 m. Another cause; do not credit.
- If K2 or P1 fails with P4 clean, the map rigidity is the remaining lever: R12 (non-frozen window). R4 window 10 is
  not proposed: the window-10 preset was 0.471 on MH_04.

**R13b (no new run; adds predictions to R11d's runs not yet made, V2_01 r2; written before those runs).**
- RT2E V2_01 r2: `ba_propagated_keyframes` ≤ 5 % of keyframes; yaw drift over 50-62 s within ±1.5°; tilt over
  62-110 s ≤ 1.5° (`E/resid/rotdrift.py V2_01_easy 30 45 4 46 114`).
- RT2E-r V2_01 r2: yaw ≥ +2.5° in the 54-58 s bins, and tilt ≥ 3° after the 62.6 s loop.
- Kill K4: RT2E-r r2 without that signature. The propagation mechanism does not reproduce; the r1 match was luck.
- Kill K5: RT2E r2 with 0 stale keyframes and a yaw step ≥ 3°. Another cause than propagation.

**R13c (code, behind a knob, only if R11d's K1 fires or RT2E misses the 2-core budget on any sequence):** ORB-SLAM3's
keyframe acceptance (`Tracking::NeedNewKeyFrame` with `LocalMapping::AcceptKeyFrames` / `InterruptBA`). Tracking
inserts a keyframe only when local mapping is idle, or interrupts the running solve. Then every keyframe is the newest
of its own solve, and PropagateToNewer has nothing to carry.
- Predictions: solves per keyframe ≥ 0.97 on every real-time run; V2_01 propagated ≤ 5 %; the yaw and tilt bounds of
  R13b; RT causal within 1.2x of S0E SYNC on V2_01 (≤ 0.066).
- Kill: solves per keyframe ≥ 0.97 with the V2_01 signature present means lag is not the precondition; diagnose.
- Alternative if acceptance costs tracking: carry the correction only when the solve kept its boundary IMU edge at
  full weight (hw ≥ 0.9) and erased ≤ 30 % of the newest rows. Otherwise fall back to SnapshotOnly for that merge.
  Theory first, no numbers yet.

Ledger: G58 (MH_04 dark-span step) and G59 (PropagateToNewer carries a wrong correction under lag) in
`experiments/gap-ledger.md`.
