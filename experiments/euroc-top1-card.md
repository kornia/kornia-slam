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
- **ASYNC mode (deployment check, batch gates only):** `--pace-timestamps`, no thread pin, 3 runs per sequence
  for the band. Unpaced async is dropped: it starves BA (1.4-1.75 keyframes inserted per solve) and is neither
  deterministic nor the deployed regime.
- **Command:** `timeout 300 bin/kornia-slam-euroc --config $d/run.ron --no-tui --no-pace
  --trajectory-out $d/traj.txt --trajectory-smoothed $d/traj_smoothed.txt --session-out $d/session.cbor`.
  No `--evaluate --eval-out` (camera-centre csv). `rc=124` means the 5 min cap fired: run invalid.
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
- **Kill rules:** K1 walk z holds and MH_04 tilt ≥ 3° at 70 s: the tilt is held by map rigidity; go to R5.
  K2 walk z holds, gain < 3 % everywhere: bias memory is not where EuRoC error enters (as card S on R_01).
  K3 any sequence worse > 15 %: knob stays off.

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
