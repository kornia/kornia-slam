# Roadmap

This roadmap will keep evolving. It sets a direction, not a commitment.

## Deployment realism

Themes derived from ongoing studies of industrial and egocentric deployments. These guide our backlog and evaluation, not parity with MicroAGI microSLAM or any external system, and we do not claim SOTA.

- Dynamic‑object‑aware tracking — https://github.com/kornia/kornia-slam/issues/139
- Long‑sequence consistency — https://github.com/kornia/kornia-slam/issues/140
- Dense/TSDF consistency from fixed poses — https://github.com/kornia/kornia-slam/issues/141
- Monocular scale (already covered) — https://github.com/kornia/kornia-slam/issues/71
- Place recognition / loop features — https://github.com/kornia/kornia-slam/issues/48
- Appearance‑robust relocalization — https://github.com/kornia/kornia-slam/issues/68
- LaMAria / egocentric evaluation — https://github.com/kornia/kornia-slam/issues/78
- Multi‑camera + IMU — https://github.com/kornia/kornia-slam/issues/138

## Progress log

Newest first, one entry per working day. "main" means a merged PR; "#180" means work on the
`feat/global-pose-graph` branch (PR #180), which is not merged. Unless marked validated, branch work
sits behind config knobs that default to the old behaviour, and nothing in it has been benchmarked yet.

### 2026-11-24

- **Card H fix: the pose-inertial solve's inlier bar becomes a knob; S0PEIH clears the count reject, K1 still fires**
  (`experiments/euroc-top1-card.md` "Results: card H fix"; `feat/cardh-pi-bar` c0342ac, 14 SYNC runs).
  - `tracking.pose_inertial_min_inliers: Visual | WithPrediction` (default `Visual` = 30, bit-identical when off:
    V1_02 S0 `cmp`, R_01 md5 d4f77b56). `WithPrediction` uses 15, ORB-SLAM3's bar after the inertial solve.
  - MH_04 start 0 now tracks 2032 / 2032 with no coast (0.178). K1 (restated) fires on 4 of 14 (S0PEI 7). The
    remaining coasts are visual map-projection rejects in the 75 s match drought, after a keyframe made from a
    15-inlier frame; nothing diverged (`give_ups=0`).
  - MH_04 median 0.178 (1.56x Basalt); start 30 0.273 → 0.173, start 60 0.173 → 0.251. The spread is a loop
    pose-graph solve that does not converge at about 80 s, so the loop is dropped (new gap G60), not the start.
    Guards V1_02 0.035, V2_01 0.046, MH_01 0.049. Nothing promoted. Next: trace G60, then R12.
- **R13a: per-frame pose-inertial tracking (S0PEI) stopped by K1 on its first MH_04 run**
  (`experiments/euroc-top1-card.md` "Results: R13a"; config only, 3 of 14 SYNC runs).
  - One inertial coast at MH_04 frame 1495 (S0, S0E and S0PE have none on any start), so card H comes first.
  - On that run the dark-span step falls from −0.54 to −0.22 m and causal from 0.227 to 0.177. One start only; not
    credited. V1_02 0.035, same health counters as S0PE.
  - Debug and rerun: the coast is a count reject. The pose-inertial gate keeps 29 of 31 PnP inliers, under the
    visual bar of 30 (ORB-SLAM3 uses 15 after its inertial solve); no RON knob reaches it. With K1 restated
    against S0PE, the 14-run rerun fires K1 on 7. The step is halved on every MH_04 start and the median goes 0.227 →
    0.177. Guards hold (V1_02 0.035, V2_01 0.046, MH_01 0.049). Start 30 (0.273) is not traced. Next: the card H fix
    (inlier bar 15), then rerun.
- **RT3E (RT1PE + 7-iteration local BA, default rayon): accuracy holds, lag does not fall**
  (`experiments/euroc-top1-card.md` "Results: RT3E"; 12 real-time runs + 2 controls, on cores 2,3 beside other lanes).
  - Causal V2_01 0.047, V1_02 0.044 (0.98x Basalt), V1_03 0.122, V2_02 0.099, MH_04 0.214 (RT1PE 0.277), MH_01
    0.043 (0.64x Basalt). K2 did not fire: on top of PnpInliers the budget does not hurt MH_04.
  - Lag higher on all six (P4 missed). The lag sits in the tracking thread's keyframe work, not in BA. Next for
    speed: R13c or moving `keyframe_insert` off the tracking thread. Nothing promoted.
- **R11d: a 7-iteration local-BA budget removes the real-time mapping lag; PnpInliers removes it too**
  (`experiments/euroc-top1-card.md` "Results: R11d"; config and env only, binary unchanged; 36 real-time runs).
  - RT2E (RT1E + `max_iterations: 7`, `initial_lambda: 1e-4`, `RAYON_NUM_THREADS=1`): solves per keyframe 1.00-1.02,
    SYNC keyframe counts, 31-56 ms per solve. Causal V2_01 0.057 (RT1E 0.125), V1_02 0.043 (below Basalt 0.045),
    V2_02 0.083, MH_01 0.073; V1_03 0.115 and MH_04 0.431. Kill K2 fired on MH_04 (1.41x RT2E-r): the budget hurts
    where mapping kept up. Extension to the other 5 sequences not run (stop on a fired kill).
  - `RAYON_NUM_THREADS=1` alone (RT2E-r): no score change, and tracking ends 10-59 s behind the sensor. Dropped as a speed fix.
  - RT1PE (RT1E + PnpInliers): V2_01 0.047, MH_04 0.277, MH_01 0.039 (0.59x Basalt); lag-free on MH.
  - Below Basalt in real time: 1 of 6 per arm (RT2E V1_02, RT1PE MH_01). Next: RT3E (RT1PE + 7 iterations, default
    rayon) and R13c (keyframe acceptance). Nothing promoted.

### 2026-11-23

- **R11c: PropagateToNewer fixes G54 in real time; S0P + S0E stacked is the best SYNC arm on V1_02 and MH**
  (`experiments/euroc-top1-card.md` "Results: R11c"; config only, binary unchanged).
  - Real time (async, paced, 2 cores, 2 runs): every stale keyframe is propagated, shifts 0-1. RT1E causal: V1_02
    0.050, V2_02 0.109, MH_04 0.316, MH_01 0.075 (1.02-1.36x SYNC); V2_01 0.125 and V1_03 0.139 (1.9-2.3x). No kill
    fired. The rest of the gap follows mapping lag on the Vicon rooms (8-17 % of keyframes never solved as newest,
    16-34 % more keyframes than SYNC; G57), not G54.
  - S0PE (SYNC): V1_02 0.036 (0.80x Basalt), MH_04 0.227, MH_01 0.049; V1_03 0.111 and V2_02 0.091 stay worse than
    S0E. Predicted "≤ both on 4 of 6" missed (3 of 6). Nothing promoted.
  - Next: a mapping-lag rung under real time (smaller BA iteration budget first, then ORB-SLAM3's keyframe
    acceptance rule), with S0PE under real time beside it; then the all-11 gate.
- **R11b: the local VI-BA outlier erase (G56) closes most of the Vicon-room gap** (`experiments/euroc-top1-card.md`
  "Results: R11b"; branch `feat/r11b-outlier-erase`, knob `mapping.local_ba.outlier_erase`, default off,
  bit-identical when off on V1_02 and R_01).
  - ORB-SLAM3's `LocalInertialBA` rule after each accepted merge: unlink observations whose whitened residual fails
    the solve's own 95 % χ² test, retire landmarks left at weighted count ≤ 2.
  - SYNC, causal median of 3 starts, S0 → S0E: V2_01 0.200 → 0.055, V1_02 0.102 → 0.040 (below Basalt), V1_03
    0.188 → 0.072, V2_02 0.162 → 0.080, MH_01 0.090 → 0.072. MH_04 unchanged (0.302 → 0.311): the tail goes, the
    error built after the dark span stays.
  - The V2_01 tilt event at 21 s goes away (5.3 → 1.4°): the visual tail was the driver KD1 pointed to.
  - Config-only `keyframe_observations: PnpInliers` gets most of the gain on its own and is better on MH_04 (0.254).
    Octave σ alone does not help. No kill rule fired; misses are diagnosed in the card (most were better than the
    predicted ranges).
  - Next: S0P + S0E stacked, then the real-time gate on all 11 (paced, async, 2 cores, 3 runs), then LaMAria.
- **EuRoC top-1 comes before LaMAria** (owner, 2026-11-23; master card `experiments/euroc-top1-card.md`).
  - **Goal:** one stereo+IMU config, 100 % tracked, below Basalt on all 11 EuRoC sequences (M1), then below
    ORB-SLAM3 (M2).
  - **Three findings behind it:**
    - the harness scored the camera centre, not the body (G55; 0.003-0.025 m per sequence);
    - the smoothed output (G53) is already below Basalt on MH_01, MH_02, MH_03 and MH_05 (planner proxy);
    - no single config is healthy on all 11: default fails the Vicon rooms (G54), the preset is worse on MH_03/MH_04 (G49, G6).
  - **Instrument:** synchronous mapping with one thread and no pacing decides every A/B; it must repeat
    bit-for-bit. Async paced runs check deployment at the batch gates.
  - **Ladder:** R0 harness and SYNC baseline, R1 merge fix (G54), R3 full-weight bias rows (G49), R4 window 10 and
    the Basalt solver budget, R6 offline global inertial BA, then R5/R7/R8/R9/R10 as the gate table says.
  - **Owner answers:** smoothed ranks, loops on, SYNC carries M1, one config for all 11. SLAM Dunk is compared
    in theory only (no run); the card's estimate puts it at the ORB-SLAM3-with-full-BA level on EuRoC, so M2 is
    that bar and R6 is the rung that buys it.
  - **R0 stopped at its first kill, K0.2** (`experiments/euroc-top1-card.md` "Results"): SYNC with one rayon
    thread hit the 300 s cap on MH_04 (1666 of 2033 frames) and V1_02 (1479 of 1710). The synchronous local BA
    carries 90-92 % of the frame time. The same config without the thread pin finished V1_02 in 190 s, and
    its first 1432 poses are byte-identical to the pinned run. Owner: drop the pin (it cost about 1.9x wall);
    SYNC keeps synchronous mapping, all threads, no pacing and the 300 s cap. Determinism is re-checked unpinned.
    Unpinned at load 15-33, all four pair runs hit the cap again (K0.2): MH_04 reached 1095-1300 of 2033
    frames, V1_02 628-779 of 1710. Every common prefix is byte-identical (r1 against r2, and against the pinned
    run). The wall is set by machine load; owner decides between a quiet machine, a larger SYNC cap or R4's budget.
    Owner: SYNC runs get a 900 s cap (outputs do not depend on load); all other runs keep 300 s.
  - **R0 done, Stop S1** (46 SYNC runs, all `rc=0`, walls 79-228 s at load 3-10). Repeats are `cmp`-identical
    (P0.1). No stale keyframes anywhere and no Vicon room fails, so G54 is gone in SYNC. Smoothed (median of 3
    starts) is below Basalt on MH_01, MH_02, MH_03, MH_05 (P0.6 as predicted) and 1.2-3.2x above it on MH_04 and
    all six Vicon rooms. Missed: P0.2 (keyframe shifts on V1_03/V2_03 with no stale keyframes, scale out of band on
    V1_03/V2_02/V2_03) and P0.3 (V1_02 start-up skips, IMU coasts on V1_03/V2_01/V2_03). The app smoothed output
    equals the proxy on all 11. Loops lower the smoothed error on 11 of 11. The start-offset band is partly
    degenerate (starts 0 and 1 give identical output on MH_04/V2_03; all three on V1_02).
  - **R3 S1 done: KD1 fired** (`experiments/euroc-top1-card.md` "Results: R3"; 18 SYNC runs plus debug D4).
    - Full-weight boundary bias rows pin the accel bias: walk z p50 drops from 6-20 to 0.1-0.7, and the V2_01 b_a error at 22 s from 0.63 to 0.05 m/s².
    - The V2_01 tilt jump at 20-21 s is unchanged (to 6.2°). Visual χ² rises the same way in both arms. The error moves into Huber-cut IMU edges.
    - Tilt falls on MH_04, V1_02 and V2_02 (by 1.1-1.8°), but smoothed ATE moves only −13 % to +11 % (MH_01 0.039, MH_04 0.245, V2_01 0.108).
    - G49 is refuted as the EuRoC score cause. The knob stays off. Next: the G56 visual log line, then R11, then R12; config-only probe S1 + `imu_huber: BoundaryOnly`.
    - No IMU time-offset knob exists for EuRoC, so the band is the start offsets only (1-3 samples).
  - Body-centre re-score of the async runs: default fails V2_02 and V2_03 too (59.6 and 242.7 m, G54); the preset
    smoothed output reaches 0.055 and 0.462 there. The smoothed proxy equals the app output to 3 decimals on 5 runs.
- **#180 (branch feat/euroc-rig, not pushed), EuRoC as a two-camera rig scored** (`experiments/euroc-rig-results.md`, ledger G50, G51, G52, G61).
  - EuRoC runs as `cameras: Rig`. A distorted pinhole rig now matches its overlap on undistorted keypoints (G52 fix). Stereo is byte-identical to the baseline binary on all 11 sequences (synchronous mapping); LaMAria R_01 is identical.
  - 68 synchronous runs. Two-camera flow (rp48) against one-camera flow (sp48): mean ATE 0.128 against 0.171 m; about 47 % of flow inliers from camera 1. The rig default (rdef) against Stereo (sdef): mean 0.160 against 0.186 m. Better on V1/V2, worse on MH_02, MH_04, MH_05 (rig scale, G61).
  - K2 fires for rdef: it gains without flow, so part of the rig gain is the rig backend. The loop-off split comes next. Best arm 0.128 against Basalt 0.081: no top-1 from this card.
  - Noise (start_frame 0/1/2) is 9-78 %, much wider than the card's 15 % guess; most per-sequence differences are inside it.
- **Re-plan: the global stage comes first, and the loss cards are paused** (owner, 2026-11-23).
  - **Why:**
    - Every top LaMAria method scores from a global solve over the whole sequence, with time-varying calibration (`experiments/lamaria-handoff.md` "The current leader: SLAM Dunk"; bench notes on visloc-rs and the top entries).
    - SLAM Dunk's offline global BA alone gives about 69 on the training set, and 82.4 with camera calibration that drifts over time plus gyro intrinsics.
    - The 2026-11-15 handoff already recommended this first. The online loss cards since then (AB, AC, AC-2b, AE, H0) patched symptoms of a give-up and re-seed design that Basalt, OKVIS2 and MSCKF-style VIO do not have.
  - **Evidence from our runs:**
    - With 60 s pieces aligned, CP reaches 93-96 on 1_19 and 80-85 on 1_20 (card AG). The relative motion is good; global consistency is missing.
    - In both BASE and H0, the best single segment explains only 9-41 % of frames, so one global Sim3 fails either way.
    - Our offline BA (built 2026-11-17) has only run on 7-110 unjoined segments with runaway seeds. It is untested, not refuted.
  - **New main card AI** (planning):
    - **(a)** a VIO that never resets: it coasts on the IMU with honest covariance and re-acquires vision into the same state;
    - **(b)** a background global graph: IMU bridges and cross-segment matches join the segments, then a bounded global VI-BA runs. Corrections reach the live pose as a map-to-odom transform, so the live pose stays real time.
    - The end-of-sequence batch for scoring is the same code, run once more.
    - Ladder after joining: global BA, then camera calibration over time, then gyro intrinsics, then accel, clock and thermal terms.
  - **Paused:** card AC-2b (built and tested, no runs), card AE, the H0 kept-state bias fix, card AD, and "keep one world through a tracking loss" (folded into AI (a)). AH stage 0 continues, because the bias walk at the window boundary matters for the global BA too.
- **#180, H0 gate stopped after round 2 reached 6 of 11 sequences** (owner, 2026-11-23: too slow; 14 h for one verdict). H0 is not promoted. Diagnosis (gate directory, `DIAGNOSIS.md`):
  - **H0's own bug:** after a Held re-seed nothing re-estimates the bias. VIBA0 never re-runs and VIBA1/2 are already spent, so the gyro bias runs to 0.3-1.4 rad/s (a healthy run stays at 0.04). The re-seeded segments then lose tracking at once (4_10: 3-10 s losses 68 against 35).
  - **The Sim3 failures on 2_11, 2_12, 4_10 and 4_11 cannot separate the arms:** both are equally fragmented. The fall in `later_alignments` is not the cause: BASE's later alignments rotate the world by 188-848 m jumps.
  - **1_20 and 3_17 are noise or better under H0:** on 1_20 the knob never fires, and 3_17 has 11 segments against 19.
  - **Load:** every run averaged above 4 (5.3-7.7, 1_20 r1 13).
- **Process: every experiment now takes at most 5 min of wall time** (owner, 2026-11-23). Experiments replay short windows (`start_sec`/`end_sec` on the LaMAria source, commit c21299b) unpaced, 2 in parallel, with at least 3 repeats, or offline runs on saved sessions. The full-sequence gate runs once per batch of stacked winners, not per card. A warm-start knob gets built if a cold window does not reproduce an event (card AH: a cold start at 400 s does not reproduce the 1_19 event).
- **Card AH planned and approved** (bias memory across the VI-BA window boundary). The boundary edge's bias rows are scaled by 1e-2, so the bias walks 10-12σ per keyframe pair (gyro) and 5.8σ (accel) everywhere, not only at events. On 1_19 the accel bias leaves about 100 s before the heading event and hides a 6-14° tilt. Stage 0 (offline) is running.
- **Card AG:** the ORB-SLAM3 reference runs are dropped. Its one existing 1_19 run is the control: heading within ±1°, window RMS 1.46 m against our 10-18 m.
- **Camera input, checked 2026-11-23: no LaMAria run has ever used the raw fisheye.**
  - Every run so far reads the pinhole export shipped with LaMAria: `pinhole_calibrations`, 780×584 at f 242. That is about 116° horizontal and 127° diagonal.
  - The raw Aria SLAM cameras are 640×480 fisheye (Fisheye624) with about a 150° field of view. So the export cuts off the periphery, and it upsamples the centre.
  - **Card F is partly built:**
    - the `Fisheye624` camera model is in (commit e50b8b6);
    - an oracle fixture from projectaria_tools exists for R_11;
    - raw ASL extractions (`asl_raw`) exist for R_01, R_03 and R_11 only (2026-11-21).
  - Not done: the loader, the undistortion path, the validation runs, and the raw data for the 10 challenge sequences.
- **EuRoC shows the same estimator weakness on uncropped images.**
  - The last full table (bench notes, 2026-11-10, config `lws9_async`; older than window 10 and the review fixes) gives SE(3) ATE:

    | | ours | Basalt | ORB-SLAM3 |
    |---|---|---|---|
    | MH_01 | 0.099 m | 0.067 m | |
    | MH_04 | 0.30 m | 0.114 m | |
    | V1_03 | 0.44 m | 0.045 m | |
    | V2_03 | 0.95 m | 0.23 m | |
    | mean | about 0.2 m | 0.065 m | 0.035 m |

  - The window-10 runs (2026-11-15) made MH_03 35-59 % worse, with a gyro-bias error 5-6x larger. That is the same boundary bias-row mechanism as card AH.
  - So the fisheye crop is not the main cause: we are 2-10x behind Basalt without any crop.
  - Not re-measured on the current branch.
- **Top LaMAria entries, researched 2026-11-23** (bench notes `lamaria-top-entries-2026-11-23.md`):
  - Sinanju 78.3, nm-SLAM 77.9 and Sloop 77.5 are anonymous; nothing is public about their methods.
  - The top cluster agrees within about 1 point per group, and dunkVIO calls itself a SLAM Dunk variant. So the leaders are likely SLAM Dunk derivatives (inferred).
  - SLAM Dunk, from its paper page and CHANGELOG:
    - an offline VI-BA over the whole recording, seeded by a separate VIO;
    - a camera state per 5 s chunk: the camera-1-to-camera-0 pose and both focal lengths, with a random-walk prior;
    - one full 3x3 gyro matrix.
  - Its training ablation: time-varying camera +8.0 CP, gyro +2.3, both together 69.1 to 82.4 (ATE 1.02 m to 0.43 m). BA takes about 2 min per recording on 6 threads. Focal length tracks IMU temperature in 22-23 of 23 recordings.
  - None of this needs learned features, GPS or joint solving.
- **Card AH stage 0 self-tests failed.**
  - The heading-event marker M2 flags 12 % of windows with GT as the estimate (target 0) and 26.5 % on ORB-SLAM3 (target at most 5 %).
  - Cause: its noise model has no pose/GT endpoint jitter and no gyro scale term (about 0.4 %). A synthetic 30° event is found exactly (30.1°).
  - Owner, 2026-11-23: add the measured endpoint-noise and scale terms, and count only runs of 2 or more same-sign windows. Re-run in progress.
  - The preview's accel-bias excursion reproduces exactly: 30 s median max 1.31 (r1) and 1.26 (r2) m/s^2 on 1_19 at 300-480 s.
- **EuRoC baseline, first results** (current branch, full length, stereo+IMU; the app's evaluator; `eval.py` SE(3) values to follow): default (knobs off) MH_01 0.081 m, MH_02 0.075 m. The 2026-11-10 table had MH_01 at 0.099 m; Basalt has 0.067 and 0.048 m. The other 9 sequences are running.
- **The first EuRoC preset arm was invalid, and finding out why exposed two gaps** (ledger G50, G51).
  - The `pr180-window10` frontend was copied without `klt_rig: AllCameras`. Optical flow was then "unavailable" on 3198 of 3682 MH_01 frames, and MH_01 came out at 0.120 m.
  - Adding AllCameras still left 3219 unavailable.
  - **Cause:** `cameras: Stereo` runs flow on the left camera only. Multi-camera flow needs `Rig` mode (`system/mod.rs:320-324`). With 48 tracks per camera and `min_tracks: 48`, flow-only tracking works only on frames where every track survives.
  - 400-frame debug runs: 17 flow-only frames at 48 tracks; 141 at `per_camera: 96`. The preset arm is re-running at 96 tracks.
- **EuRoC MH_04 diagnosed** (default 0.297 m against Basalt 0.114 and ORB-SLAM3 loops-off 0.070 over the full sequence; the README's 0.045 for ORB-SLAM3 covers only 80 % of frames, after 3 resets). Full report: `.pr180-fix/euroc/MH04-DIAGNOSIS.md`.
  - **Tracking is clean:** 2032 of 2032 frames, one segment, scale 0.99-1.01, yaw within ±0.8°.
  - **The error is one event plus a reporting convention.**
    - In the dark section (56-68 s) roll and pitch error grows from 0.6° to 3.2-3.8° and stays there until the loop at 79.4 s. Basalt stays at or below 0.35°.
    - The tilt turns horizontal motion into about 1.08 m of vertical drift. The vertical part is 0.265 of 0.306 m SE3, and the loop measures t_y -1.076 m.
    - Cutting the trajectory at 65 s and 80 s takes the ATE from 0.306 to 0.131 m.
  - **Ranked causes:**
    1. The tilt is hidden in the accel bias, as on LaMAria 1_19.
       - The accel-bias error lines up with the bias a tilt that size needs: cos 0.84 on default, 0.94-0.99 on the preset.
       - Accel walk z p50 is 11.2 (about 1.5 expected).
       - This is card AH's mechanism (G49, G35, G6), and card AH's EuRoC arm E1 tests it.
    2. The causal trajectory keeps pre-loop drift (G53). Re-expressing on the final map gives 0.165 m.
    3. Four dark-section keyframes carry a stale state (`ba_stale_keyframes=346`, `ba_propagated_keyframes=0`; G36). Their speed collapses to 0.23-0.26 m/s against GT 0.93-1.12.
    4. The 10-keyframe preset makes cause 1 worse: tilt 4.4-4.9° and gyro walk z 15.2 (card W).
    5. A small event near 43 s, not yet diagnosed.
  - The 4-DoF loop correction cannot remove tilt.
  - G50 and G51 cost nothing on MH_04: 36 frames without flow.
- **EuRoC Vicon-room failure on the default config diagnosed (ledger G54, suspected).** Report: `.pr180-fix/euroc/V102-DIAGNOSIS.md`.
  - **Symptom:** V1_02, V1_03 and V2_01 score 1.36 / 1.53 / 1.99 m, with Sim(3) scale 0.345 / 0.003 / 0.057.
  - **The scale itself is fine:** 1.00-1.04 per 5 s piece while tracking is healthy. The 1-10 m jumps at BA write-backs make the Sim(3) fit shrink the trajectory.
  - **Mechanism:**
    - Mapping falls behind: 296 ms per solve, with a keyframe every 150 ms.
    - `merge: SnapshotOnly` leaves the keyframes inserted during each solve on stale seeds.
    - BA then shifts the newest keyframe by more than 0.3 m: 84-161 shifts per run against 0-2 on MH.
  - **The preset (`PropagateToNewer`) is healthy:** 0.120 / 0.203 m. The 2026-11-10 runs (synchronous mapping, same BA) scored 0.075 m.
  - **Tests:** one V1_02 run per knob: T1 `PropagateToNewer`, T2 window 10, T3 Synchronous, T4 7 iterations.
  - Until then the default arm's Vicon-room numbers are invalid as a measure of the estimator.
- **Every reference tracks both stereo cameras** (Basalt, OpenVINS, VINS-Fusion, ORB-SLAM3, SLAM Dunk; file and line in G50). Our Stereo mode is the outlier, so on EuRoC we track with half the observations they use.
  - Next: a small EuRoC source change, so EuRoC can run as `cameras: Rig`, then Stereo against Rig on all 11 sequences (planning).
  - LaMAria configs already run `Rig` with AllCameras, so there it is a no-regression check.
- **Break-case test suite** (owner, 2026-11-23: planning): turn each known break (1_19 weak spell at 436-448 s, the start-up turn, the 2_11 runaway velocity, the Held re-seed bias runaway) into a red unit test on a real snippet or a dumped state. The first test compares our LK with OpenCV on the same LaMAria frames.

### 2026-11-22
- **#180, card AG planned and approved; stage 0 running** (worktree `kornia-slam-drift`, branch `feat/drift-budget`; python and evaluator only, during the gate). Planner preview, one run per arm:
  - **Revisits from GT:** 7 of 9 sequences with GT have no loop of 50 m or more of path (R_11, 1_19, 2_12, 3_17, 3_18, 4_10, 4_11). Only 2_11 (24.6 % of path enclosed, longest loop 247 m) and 1_20 (11.9 %, 114 m) have one. G10 is confirmed: loop closure cannot reach most of the loss.
  - **1_19 is one event, not smooth drift.** Heading holds within 6° for 0-420 s, then steps +31° (H0) or +16-18° (BASE) at 440-470 s, and stays. 60 s pieces elsewhere fit to 0.1-0.5 m. The step is in the final keyframes too, so it is in the map, not only the causal output. The global scale 1.196 is an artefact of it; local scale is a steady 0.86-0.91. 1_20 has the same pattern (yaw steps of -15° and -10°, scale 1.44-1.55 for 2 minutes).
  - **Mechanism:** in weak vision (26-30 observations), the window moves |bias_gyro| from 0.005 to 0.037 rad/s in 8 s, about 40σ of the sensor's random walk, and |bias_accel| to 0.5-1.6 m/s². Every event window shows such an excursion, and no other window does. It is the same bias failure the H0 gate found at the R_11 re-seeds.
  - **Reference:** ORB-SLAM3 mono-inertial on the same 1_19 input (one run from 2026-11-10, high load) holds heading within ±1° with no loop: CP 38.9, Sim3 RMS 2.85 m, against our 8.8-14.6. ORB-SLAM3's leaderboard mean is 10.0, so it is a floor we are under, not the top-3 bar.
  - **Owner, 2026-11-22:**
    - The loop work stops at an observe-only arm (stage 2, on the owner's go).
    - The 6 ORB-SLAM3 reference runs take the first SLAM slots after the gate.
    - Card AH (the window bias absorbs vision errors; first test card S's `boundary_scales_bias_rows: false` on the event windows) comes next, if round 2 confirms the bias-event link.
    - Order: AG, AH, AE, AD.
- **#180, H0 gate round 1, more sequences** (one run per arm). CP BASE/H0; tracked BASE/H0; give-ups BASE/H0:
  - 2_11: 0.58 / Sim3 failed; 19465 / 19322; 53 / 24.
  - 4_11: 3.89 / Sim3 failed; 14919 / 13106; 114 / 54.
  - 5_12: 0.00 / 0.00; 22716 / 22315; 21 / 12.
  - H0 cuts give-ups everywhere, but its tracked frames are below BASE on R_11, 4_11 and 5_12, and it fails Sim3 on 2_11 and 4_11.
- **#180:** 59 local commits pushed to #180 as a fast-forward (`70c3a45..79015b1`; fmt and clippy checked). Cards AB, AB-1b and AC are integrated after the gate. R_12_10cp and R_13_15cp are downloading (Controlled IV, with 10 and 15 control points; R_11 has only 3 usable).
- **#180, metric check (no bug found).**
  - The official evaluator scores pseudo-dense GT as the estimate at 1_19 69.9, 2_11 80.2 and 4_11 85.2. That is the ceiling a 5 Hz GT sets.
  - `ate.py` recovers a known Sim3 exactly.
  - Our trajectory frame is the evaluator's: rotating it by the 105° GT body rotation drops 1_19 from 14.6 to 3.9 or 8.5.
  - The CP score is rotation-invariant (GT at 40° yaw: 80.17 both ways) but not scale-invariant (GT ×0.5 73.6, ×0.8 76.2, ×1.2 85.1, ×2.5 91.3). The likely cause is the metric camera-IMU lever arm. Scale error moves our score directly; this is not something to exploit.
  - R_11 "Robust Sim3d estimation failed" is a real outcome: only 3 control points have height, and `estimate_sim3d_robust` uses `max_error` 5 m. The gate's BASE run has inter-CP scale ratios of 0.80-1.02. It scores CP 0, so R_11 is gated on tracked frames and ATE.
- **#180, where the score goes.**
  - Gate H0 run, 1_19: one segment over 5-917 s, with per-segment Sim3 RMS 37.1 m over a 1000 m GT path and scale 1.196.
  - 1_20: 21.2 m over 1140 m.
  - So on the long sequences the loss is drift while tracking, not tracking losses.
  - Loop closing is disabled in every validation and gate config (`LoopClosingMode` defaults to `Disabled`).
  - Ledger G35 still has our local VI-BA at 1.5-5× ORB-SLAM3's ATE on full R_01.
- **Order (owner, 2026-11-22): drift first.** Card AG covers the drift budget per sequence, ORB-SLAM3 reference scores on 1_19 and 2_11, and loop closure on the challenge sequences (planned and approved later the same day; see above).
- **#180, H0 regression gate, round 1 (interim, 1 run per arm; 11 sequences × 2 rounds).** CP BASE/H0:

  | sequence | BASE CP | H0 CP | tracked BASE/H0 |
  |---|---|---|---|
  | 1_19 | 14.56 | 8.80 | 17806/17823 |
  | 1_20 | 11.59 | 7.06 | 18933/19162 |
  | 5_11 | 2.33 | 11.22 | 20597/20377 |
  | R_11 | 0 (Sim3 failed) | 0 (Sim3 failed) | 8998/8003 |

  R_11 ATE Sim3: 5.8 m against 29.9 m.
- **#180, R_11 under H0 (diagnosis; card AE planner, from session keyframes).**
  - From 329 s, during a 94-126 °/s turn and still tracked with 138-248 observations, the VI-BA gyro bias leaves its median (about 2e-3 rad/s) and reaches 48e-3 rad/s.
  - The loss at 331.4 s freezes it, and each Held re-seed keeps it (45-57e-3 rad/s).
  - Attitude then tilts at about 3 °/s, so seeds start 5-41° off level. g·sin(7-8°) gives the 1.3-1.4 m/s² speed ramp in every young segment, and 4 young deaths follow (give-ups 8 against 2).
  - It generalises: on 4_11, 35 of 49 seeds sit over 15e-3 rad/s, with segment life 4.9 s against 24.2 s.
  - This corrects the card AC entry below: H0 has no `frame_velocity`, so U is not the only R_11 cause.
- **#180, card AE planned and approved** (`tracking_loss.reseed.inertial_seed: AtGiveUp | Trusted`): the Held seed takes the median bias over a sensor-model window (33.5 s gyro, 58 s accel on R_11) of vision-anchored keyframes, and its attitude from a gyro chain from the last vision-anchored pose. It runs after card AG.
- **#180, card AC-2b (pose innovation gate) being implemented;** its SLAM runs wait for the H0 gate.
- **#180, card AB-1b (`feat/still-device`, 7 commits, unpushed): the IMU still run is carried across a keyframe inserted while still.**
  - The system test now gives lost keyframes at `[60, 120]` against 17 at base, with `vouched_imu` 114.
  - MH_01 frames reading `imu=Pending` in the still window fall from 341 to 18.
  - Every identity check is `cmp`-identical (R_01 md5 d4f77b56, EuRoC Off and Observe), and LaMAria decision counters stay 0.
  - P3 missed as worded (K1): the V1_03 blackout lands before IMU init, where a loss resets the map, so it is a test-design miss. The real-data proof waits for the robot's parked capture.
- **Next (owner, 2026-11-22, superseded later that day by "drift first" above):** the H0 regression gate (Held seed without `frame_velocity`, 11 sequences × 2 paired rounds), and card AC-2b (loss-onset velocity, K4) being planned. Then card AD, then the cleanup card.
- **#180, card AC, H arm (Held seed, `feat/loss-one-world`) on the full validation sequences, one paced run each (interim, not a verdict).** Official CP, F1 against H:
  - 1_19: 4.84 against 6.89. Give-ups 11 against 3, and the 3 are start-up resets.
  - 5_11: 5.39 against 25.86. Give-ups 9 against 3.
  - 2_11: Sim3 failed in F1, 4.06 under H. Give-ups 61 against 21.
  - R_11: 55.54 against 26.92, a regression.
  - Across these runs: no world rotations (`later_alignments=0`), and 3_18 and 4_11 are still running.
  - **R_11 regression (diagnosis, read-only):** both runs lose vision at about 370 s. F1 coasts at 1.45-1.81 m/s (GT 1.50-1.63) and recovers with no give-up. H enters the loss with |ba| 0.55-0.83 m/s² (F1 0.13-0.37), so its coast decelerates to 0.19 m/s and then runs to 2.4 m/s. Projection then finds 0-49 observations, and H gives up at 375.4 s. The Held segment inherits the bad state and dies again. **Cause found (card AC final):** `frame_velocity: PropagatedFromKeyframe` (U), not the Held seed.
    - Keyframe |ba| at 360-370 s is 0.30-0.86 m/s² in every U run, against 0.01-0.54 without U.
    - Without U, the paired runs score BASE 37.0, KI 39.8 and H0 (Held, no U) 30.9, all with 0 re-seeds. With U, U, KIU and H all fail Sim3.
    - R_11_5cp CP spreads 31-55 across runs that behave identically, so one sample per sequence cannot gate it.
- **#180, card AC verdict (`feat/loss-one-world`, `experiments/pr180-AC-card.md` there; 15 commits, unpushed).** Held keeps one world wherever it acts:
  - `later_alignments` is 0 on every Held, KI and KIU run, against 3 per W511 window for BASE (85-94° world rotations).
  - On 2_11 give-ups fall 61 to 21 and young deaths 59 % to 14 %; primary link rotation error is 10.9° against 132.3°.
  - On R_01 with a 6 s blackout, the rotation error across the give-up is 4.1° against 125.6° for BASE.
  - Offline BA on the H sessions starts from cost 1.25e8 (F1 8.1e10) and converges in 34 iterations.
  - K3 fired (R_11 under U) and K4 fired (loss-onset speed 4.4-7.6 m/s against 1.1-1.3 m/s on W211aP-U and W211c-KIU); 3_18 and 4_11 were not run. All knobs stay at their defaults.
  - Metric fixes: `lossev.py` fits the GT body frame once per sequence (104.8-105.9°) and slerps GT; the 4_11 heading error across give-ups is 83°, not 119°.
- **#180, 1_19 start-up resets (diagnosis):** before VIBA0 the frame prediction is the last pose with zero velocity, and the gyro is unused. A start-up turn of 3.0-4.4° per frame gives a 29-32 px PnP residual against the 25 px gate, and 19-56-point maps reset at once. This leads to card AD (planned, approved; it starts after AC).
- **Process:** the full-sequence regression gate (6 sequences, 2 runs each, no sequence below its baseline range) is now required before any knob is promoted (see "Now").
- **Coverage (research, lamaria.ethz.ch and arXiv 2509.26639):**
  - The test set is 63 sequences in 5 leaderboard groups (Short 18, Medium 10, Long 16, Low light 9, Moving platform 10), and we held one training sequence per group.
  - Downloading the second training sequence of each group: sequence_5_12, 3_17, 4_10, 2_12 and 1_20 (59 GB; 43 stale build directories deleted, disk 85 % to 75 % before the download).
  - R_11 is in no leaderboard group.
  - The leaderboard has moved: Sinanju 78.3, nm-SLAM 77.9, Sloop 77.5 (mean of the 5 groups). The top-3 line is now about 77.5, not 66.7.
- **#180, card AC stage 0 (`feat/loss-one-world`, `experiments/pr180-AC-card.md` on that branch): stopped at its reproduction gate (K6 fired); AC-1 not built.** Loss counters (`[summary] loss` line) and the event scripts (worldrot.py, lossev.py, oracle.py) landed and reproduce the F1 diagnosis numbers; knobs-off R_01 stays byte-identical. Only 3 of 6 event windows (W211c, W318, W511) replayed their F1 event; W211a, W211b and W119 spent their lead on start-up give-ups, and the full-prefix 2_11 run reached the same mechanism (primary give-up, then a 31 m, 49° world rotation) 45 s earlier than F1. Why: the machine ran at load 20-40 (F1: 2.5), and with `--pace-timestamps` late frames shift mapping timing, so the same binary and window gave 2 against 0 and 13 against 17 give-ups. One debug sample confirms the later-alignment rotation (VIBA0 gravity 87.5° off, orientation change 88.7°); corrected `worldrot.py` puts the F1 5_11 rotations at 86.5° and 55.1°. R_01 counts its post-reset VIBA as "later" alignments (0.3 m jump), so the counter needs a jump-size split. Owner, 2026-11-22: score on each run's own events, with paired BASE, and run the arms on a quiet machine. Interim scoring found two `lossev.py` metric bugs (the GT body frame is 105° from the trajectory's on 2_11; the GT attitude is the nearest 5 Hz sample, so it reads 10-25° stale in fast turns). The Held seed's apparent 138-165° restart error was the first bug; the seed follows the gyro to 0.1-0.4° per frame through the re-seed. Corrections are in `experiments/pr180-validation-diagnosis.md`; the ~130° base-run heading error across give-ups stands.
- **#180, first LaMAria validation run (tag `pr180c/F1`, preset `pr180-window10`): mean CP score 11.0 online, 10.8 after offline BA.** Six training challenge sequences, one sample each, paced at the sensor timestamps, factory-rectified IMU, no clock offset. Aria SLAM's test mean is 75.9; the top-3 line is 66.7.
  - R_11 55.5 (ATE 4.75 m); 1_19 4.8 (60.6 m); 3_18 0.4 (209 m); 5_11 5.4; 2_11 Sim3 failed (380 m, 76.9 % tracked, 61 give-ups); 4_11 low light Sim3 failed (4392 m, 63.5 % tracked, 134 give-ups).
  - Offline BA stopped at its 60-iteration cap on every sequence; sessions split into 7-110 unjoined components (`bridge_segments` off). It raised 2_11 and 3_18 and lowered 1_19 and 5_11.
  - Speed: S10-7 solve p50 42-43 ms, frame p90 102-104 ms. The card T binary gives the same numbers on the same night and a byte-identical trajectory, so the change from card T's 38 / 87 ms is machine state. Frontend 2.0 ms per frame against a 0.8 ms budget, mostly feature refill (1.2 ms against 40 µs amortised). The OpenCV comparison did not run (OpenCV not installed).
  - Diagnoses of the drift, tracking losses and offline-BA convergence: see the diagnosis entry below.
- **#180, card AA: the clock fix cleans the IMU edges but does not move the score (K1 fired).** With the per-sequence offset (-3.66 to -4.54 ms; factory gyro record 4.146 ms on the R_01/R_03/R_11 device), interior IMU chi2 p90 on R_01 falls 749-921 to 8 and the downweighted share 36-38 % to 3-5 %; online td (card G) converges to -4.0 to -4.5 ms from 0. R_01 along-track SD stays 51-61 mm and Sim3 inside noise, so the visual side sets the scale. ORB-SLAM3's kernel layout adds nothing on R_01/R_03 (median over 6 pairs -0.001 m) and costs 0.01-0.03 m on R_11. All knobs stay at their defaults. The offset still ships later from the device record (own card), because it is correct sensor modelling.
- **#180, validation diagnosis (`experiments/pr180-validation-diagnosis.md`).** The long sequences fail in short events, not smooth drift: weak vision (0-25 observations, near walls) lets a wrong VI-BA velocity in (5-16 m/s against 0.2-0.5), the IMU coast runs away, the give-up restarts at the old attitude after a 120-130° turn on the spot, and the new IMU-less segment dies within 0.5 s (59 % / 86 % of give-ups on 2_11 / 4_11). Between events tracking holds 0.3-0.6 % of path. CP oracle with restarts linked: 1_19 3.2 to 39-49, 3_18 18.9 to 32-45. Offline BA is wrecked by runaway seeds (no IMU kernel) and a 3 px drop repeated every iteration. Next card: keep one world through a loss (owner, 2026-11-22: after AA, before the cleanup card).
- **#180, card AA (`experiments/pr180-AA-card.md`), first entry of the day.** The camera-IMU clock is never corrected: IMU stamps run 4.2-5.3 ms late on every sequence. The Aria factory record carries about 4.1 ms of it (`TimeOffsetSec_Device_Gyro`), and `read_factory_rectification` does not read it. With Huber on every interior IMU edge, the clock error down-weights 34-41 % of them, mostly the fast-motion edges that carry scale. Stage 1 applies the offset (config); stage 2 applies ORB-SLAM3's kernel layout (Huber on the boundary edge only); control arm K0 runs the kernel change alone, as ORB-SLAM3 itself ran on this data.

### 2026-11-21 (evening)
- **#180, full-review fixes (tag `pr180c/F1`, 81bc775).** Six review lanes, 36 commits, gate green, knobs-off R_01 trajectory byte-identical (md5 d4f77b56). They include: no non-finite pose is written; `--pace-timestamps`; the offline-BA stop reason and the faer parallelism restore; the base local-BA budget (7 iterations from λ 1e-4) in `pr180-window10`. R_01 A/B against the card V binary: inside noise (0.33-0.47 against 0.30-0.51 m).
- **#180, card S (`experiments/pr180-S-card.md`): refuted as a score fix.** Bias random-walk rows at full weight across the window boundary cut the boundary bias residual from 8.0 / 3.0 σ (gyro / accel) to 1.3 / 1.3 σ, but R_01 ATE stayed inside noise. Diagnosis: R_01's error is along-track scale, shared across runs (correlation 0.53-0.78), and the accel-bias wander is tilt compensation, not a free walk. That led to card AA.
- **Correction to the entry below:** "the full-sequence drift (G35) is fixed" overstated it. R_01 went from 1.1-1.2 m to 0.30-0.50 m, still 1.5-5× ORB-SLAM3 (0.07-0.25 m), and the long sequences still drift tens to thousands of metres (2026-11-22 validation).

### 2026-11-21
- **#180, cards Q, R and T: the full-sequence drift (G35) is reduced behind knobs** (not fixed; see the evening entry above).
  - **Cause:** with 10 active keyframes, the slow asynchronous solve absorbed 4-5 new keyframes each, starting them from a stale state (crossing velocity error 1.4-1.6 m/s), so the window slid 8-14 cm per solve. A 3-keyframe window avoids the slide but cannot hold the scale.
  - **Fix:** 7 iterations from λ 1e-4 in the 10-keyframe window, `merge: PropagateToNewer` (each solve's correction carried to the keyframes inserted while it ran), and `dense_parallelism: Sequential`. faer's default thread pool made a 150×150 factorisation 3.4 ms idle and 39 ms loaded; on one thread it is 0.12 ms, bit-identical.
  - **Result, full R_01:** ATE 1.11-1.23 m to 0.30-0.40 m (async preset), 0.384 m (sync), mono 3.5-4.4 m to 0.760 m. R_03 800 frames 0.19-0.20 to 0.085-0.112 m, but 762-789 tracked (base 782-800, post-gap velocity drift). R_11 unchanged. Solve 38 ms p50 (sync, idle).
  - **New preset `configs/pr180-window10.ron`.** Code defaults stay off.
- **#180, card L:** `--trajectory-include-coasted` writes the IMU-carried poses into the scored trajectory; R_03 800/800 at the same ATE.
- **#180, card K (`experiments/pr180-K-card.md`).** The R_03 failure under `min_cold_start_points: 20` (G32) is fixed behind a knob.
  - **Cause:** while the device is still, keyframe pairs a few millimetres apart grew map points. Only wrong matches can pass the 1° parallax gate from such a short baseline, so the points landed at 2-25 cm instead of 0.8 m and shrank the trajectory to about 1/5 scale.
  - **Fix:** `mapping.growth_min_baseline_ratio` skips a pair whose baseline is below that fraction of the keyframe's median depth (ORB-SLAM3 `CreateNewMapPoints`, mono rule). At 0.0175 (tan 1°): near points 2394 to 11-13, step ratio 1.01, VIBA2 accepted 3/3.
  - **Result with `min_cold_start_points: 20`, 3 runs each:** R_03 784-798/800 at ATE 0.194-0.202 m; R_11 798-800/800 at 0.136-0.170 m; R_01 800/800 at 0.108-0.120 m.
  - **Shipped in `pr180-frontend-1ms`** (owner, 2026-11-21), with both knob defaults in code still off.
- **#180, card F resumed.** The owner decided D1-D6. The raw `.vrs` files for R_01, R_03 and R_11 are downloaded and are being extracted to the ASL layout.

### 2026-11-19
- **#180, first quick evaluation** (2026-11-18, LaMAria 800-frame windows, `--no-pace`, 3 runs per config). This is not the validation protocol; it is a few-minute check on three sequences.
  - EuRoC with the knobs off matches the 79c04bb base.
  - `pr180-frontend-1ms` beats `lamaria-rig-imu` on R_01: ATE 0.10 m against 0.59 m.
  - The full R_01 sequence tracks 100 %, but with ATE 1.18 m and scale 1.136 (drift).
- **#180, card H** (`experiments/pr180-H-card.md`). A bisect of `pr180-full` put its scale collapse on `tracking.pose_inertial: OrbSlam3`, the per-frame solve from card A6 and PR #195. The solve gates the visual matches out: solve inliers / visual inliers are 0.09-0.12 on rejected frames. `pr180-full` now ships it off (f55e997), and is then 3/3 at 100 % tracked on R_01, 0.110-0.122 m. H2 (the `LastKeyframe` variant) is refuted.
- **#180, card I** (`experiments/pr180-I-card.md`), knobs `tracking_loss.reseed.young_map_grace_sec` and `min_cold_start_points`, both off.
  - R_11 loses 132 of 800 frames at start-up: the frame-0 map is reset at frame 3, then the cold start starves below 50 stereo points.
  - `min_cold_start_points: 20` gives R_11 800/800 ×3, but breaks R_03.
  - The grace knob does not help: a held young map does not re-track.
  - The frame-0 map dies at frames 2-4, from the auto-exposure ramp on R_01 and R_03, and from the view moving off the first structure on R_11.
- **#180, card J** (`experiments/pr180-J-card.md`), knobs `inertial.viba_window` and `inertial.viba_velocity_gate`, both off.
  - **Why R_03 breaks:** with the map seeded one frame earlier, VIBA0 is 3.4° off, as measured by VIBA1, against 0.47° in the base. VIBA2 then fits a systematic tilt of about 14° (|ba| 2.3-2.7 m/s²) and is rejected in 10/10 runs, and tracking is lost.
  - **The two fixes:**
    - J4 keeps the keyframes inserted while lost out of the VIBA windows;
    - J1 gates outlier velocity edges per edge;
    - both are refuted as fixes (the error is systematic, not a few edges), and are harmless on R_01, R_11 and the R_03 base.
  - **J3 cancelled before code:** under `min_cold_start_points: 20`, the R_03 visual trajectory is at about 1/5 scale for frames 25-250 (step length ours/GT 0.13-0.23; base 1.00). Rotation matches ground truth to about 0.1°. Stereo VIBA fixes scale at 1, so its velocity residuals are systematic, which explains VIBA0's error and the VIBA2 reject.
  - **Ruled out:** LK (frames 3-45 are tracked by ORB map projection; KLT starts at frame 46), the ORB stereo seed (both configs: 63 points, median 0.80 m), and tracks surviving the reset.
  - **Next:** measure the suspect, pair-growth triangulation across a near-zero baseline while the device is still (frames 0-32). The only guard is `|t| > 1e-8` (`mapping/growth.rs:233`); ORB-SLAM3's `CreateNewMapPoints` skips pairs below the stereo baseline. Then J2 (`FullInertialBA`, kornia-slam#51).
- **#180, still unvalidated.** No preset changed except `pr180-full` (`pose_inertial: Off`). The validation workflow in `experiments/pr180-validation-handoff.md` has not run. The status and how to take over are in `experiments/pr180-status-2026-11-19.md`.

### 2026-11-18
- **#180, theory review.** Four read-only reviews (estimator, offline BA and calibration, geometry and recovery, frontend): 52 findings (5 high, 21 medium, 26 low), all in knob-gated code. The always-on review fixes were found equivalent to what they replaced. Findings and the test for each: `experiments/pr180-theory-review.md`.
- **#180, fixes per lane** (one commit per finding or group, each with a test that fails on the old code; knobs off, not validated):
  - estimator and system:
    - E1/E7/G3: velocity and bias come from the origin keyframe, and velocity is rotated on a pose jump;
    - E2: pixel sigma from the camera-centre covariance;
    - E4/E5: pose-inertial guards, and a solve with too few inliers is rejected and counted;
    - E6: 2-dof gates for every pixel row;
    - E9/E12: stereo evidence counts twice, and the prior gradient is its cost derivative;
    - G2/G7: a lost relocalisation into an older segment lifts the own window, and the coast speed spans the gap;
    - G12: inertial knobs are refused without an IMU;
    - T3/T5/T7: track positions refresh on a keyframe, the gyro guess spans one frame, and the evidence lock wait is timed;
    - T9: an anchor keeps the last keyframe's frames.
  - mapping, VI-BA and calibration:
    - E3: the newest marginal is the undamped system at the solution;
    - D1/D4/D9/D10: the boundary prior at the calibrated extrinsic, chained older priors, a prior that keeps its constant, and its gradient through J_r^-1;
    - E8: a window as large as the map fixes its oldest keyframe;
    - E10/D3: offline edges integrate as the run did, and the clock state is not lag-biased;
    - D5/D6/D14/D8/D2/D12/E13: stereo depth rows are off under a camera state, the clock test motion, the yaw gauge row, the 3 px stage after early convergence, an observable offline walk, a `td_total_ms` report, and the large-window lambda doc.
  - frontend:
    - T1: the LK template is recut when ORB moved the track;
    - T2/T6/T8/T11: refill tracks reach a map point, the WindowedU8 tracks are capped at the refill budget, occupancy is at the tracks' start pixels, and no border corner is spawned that the filter would drop;
    - T4: the analytic flow pose rejects outliers;
    - T9: landmarks are adopted without waiting for a mapping result;
    - T10: stronger flow tests.
  - geometry, loop closing and sensors:
    - G1: a lost relocalisation joins by a loop edge, not an odometry edge;
    - G4/G5: gPnP prefix search and a true pixel residual;
    - G6/G8: loop-edge gravity projection and recentred solve, robust graphs only;
    - G9/G10: rig extrinsic checks and a reflection check;
    - G11: monotone analytic-refine steps behind `tracking.refine_monotone` (off by default; on in both `configs/pr180-*.ron` presets);
    - E14: Midpoint extrapolates past the last IMU sample;
    - E11: the covariance propagation test.
- **#180, not fixed:**
  - G13: unreachable (`node_spacing: Some` only, and `correction()` has no non-test caller);
  - D13: deferred design item (the window-end edge needs samples that do not exist yet when it is made);
  - D7, D11: fixture work, the code was not shown wrong (draft tests are not in the branch);
  - the rest of E11: tests for `solve_pose_inertial`, `marginal_information`, `pixel_sigma`, the G29 counter and `newest_marginal_covariance`, which sit in lane S and M files.
- **#180, still unvalidated:** no eval or benchmark has run. The robot test and the validation workflow come next.
- **2026-11-17 lines corrected:** the estimator-round line now notes that the stereo pose-inertial gates were 3-dof until E6, and the recovery line notes the G1/G2 detach.
- **Card text corrected:** card A (outliers stay in keyframe observations, as in ORB-SLAM3), card D P15 (`|td| ≤ 2 ms`, `td_total_ms`), card E P15 (post-ORB recuts by design), card G (the boundary prior is formed at the calibrated values since D1).

### 2026-11-17
- **#180, rebased onto main.** No merge commits. kornia-rs is pinned to main `4ada5a7`. Reviewer comments applied:
  - `rotation_defect`, `HuberKernel`, `CauchyKernel` and `Mat6F64::cholesky` from kornia-rs;
  - one `camera_centre` helper;
  - the `keyframes_while_lost` source-of-truth fix and validation.
- **#180, estimator round** (cards A-D; knobs off, not validated):
  - ORB-SLAM3 estimator port: window 10/25, bias at the window boundary, Huber placement and pose-inertial gates (the stereo gates were 3-dof until 2026-11-18, E6), IMU sample interpolation, per-frame pose-inertial solve (idea from #195), keyframe and local-map rules;
  - IMU covariance trust;
  - G29 snap-back counter;
  - recovery: G19 global rig PnP while lost, and a G28 re-seed with its own window (a lost relocalisation into an older segment detaches and joins by a loop edge since 2026-11-18, G1/G2);
  - generic rig calibration and rays;
  - analytic pose refine, and the PGO moved to `upstream/`;
  - offline global VI-BA with calibration states, plus a `--session-out` dump.
- **#180, 1 ms frontend round** (cards E, G; knobs off, not validated):
  - KLT on every rig camera, the windowed u8 LK and `FlowPool` threads;
  - the forward-backward check, Shi-Tomasi refill, gyro guess, anchor guard;
  - analytic flow pose, keyframe-only bookkeeping;
  - online calibration with `imu_t_bc` and the clock offset;
  - `wide` SIMD kernels with OpenCV benches;
  - robot presets `configs/pr180-frontend-1ms.ron` and `configs/pr180-full.ron`.
- **Deferred:** raw Aria fisheye (card F), and keyframe ORB on the mapping thread (`keyframe_orb: Mapping`).
- **Reverted:** Step 8 (SVD `orthonormalized`), because it changed the PGO fingerprint.
- **Generic code** goes in `crates/kornia-slam/src/upstream/`, marked for kornia-rs later.
- **#180, CI fixes:**
  - clippy (rust 1.100), rustfmt and rustdoc;
  - the offline VI-BA test fixture now has observable scale (constant acceleration left a scale and tilt null space);
  - the full local gate is green.

### 2026-11-16
- **main:** #197, inertial factors use kornia-3d `RobustKernel::rho`/`weight`; local `robust.rs` removed.
- **#180:** review received from cjpurackal.

### 2026-11-15
- **main:** #194, inertial factors grouped; `se3` renamed to `relative_pose`.
- **#180:**
  - re-seed with kept inertial state on rigid poses, the gyro attitude and a tilt check;
  - a pose-acceptance gate against the inertial prediction;
  - a VI-BA iteration-cap knob;
  - the LaMAria sprint handoff (`experiments/lamaria-handoff.md`).

### 2026-11-14
- **main:**
  - #185, #191, #192: factors, preintegration and VI-BA on kornia-algebra fixed-size types;
  - #189: build against kornia-rs main;
  - #190: reuse kornia-rs helpers.
- **#180:** every pose kept rigid at the source; keyframes while lost as fixed anchors; non-rigid pose counters.

### 2026-11-13
- **main:**
  - #175: IMU, wheel and stationary factors moved into kornia-factors;
  - #181: loop closures solved on a worker thread;
  - #182: seed after a give-up;
  - #183: LK pyramid reuse;
  - #188: clippy 1.100.
- **#180:** run counters and a `[summary]` line; wheel odometry in the local BA; a keyframe bound; app flags `--trajectory-coasted` and `--no-pace`.

### 2026-11-12
- **#180:**
  - KLT-first tracking between keyframes (mono);
  - LK starts at the predicted pose;
  - fixed-point u8 LK kernel;
  - analytic pose-only LM in `solve_pnp`;
  - frontend replay benchmark;
  - a secondary-camera keypoint budget.

### 2026-11-11
- **#180:** a marginal prior for the VI-BA boundary, carried between windows; an optional state envelope on BA write-back; hold the inertial state of keyframes inserted while lost.

### 2026-11-10
- **#180:**
  - multi-camera rigs: rig declaration, per-camera features and overlap points, per-camera tracking, stereo-inertial initialization;
  - the LaMAria pinhole source;
  - loop-closure fixes (Drift information, inline solve by default);
  - robust kernel in the visual-only BA;
  - octave-scaled visual information.

### 2026-11-09
- **#180:** landmark merge through observation records; loop fusion planned in the worker and applied at write-back; seed after a give-up at the last tracked pose.

### 2026-11-08
- **#180:** one global pose graph for loop closing:
  - sparse PGO with information-weighted Cauchy edges;
  - drift-based loop gate;
  - persistent pose graph owned by `LoopCloser`;
  - off-thread solve with snapshot write-back.

### 2026-11-07
- **#180:** `compact_landmarks` frees retired landmark slots.

### Earlier (main)
- **2026-10-04 to 10-29:**
  - wheel-odometry VI-BA factor and stationary init (#158);
  - tracking-loss recovery and relocalisation (#153);
  - `process_imu`/`process_image` split (#165);
  - wheel kinematics (#174);
  - keyframe eviction at a map bound (#169);
  - configurable pipelines (#149, #150);
  - calibration through `SensorRig` (#147).
- **2026-09-23 to 09-27:**
  - map and system refactors (#105-#125);
  - kornia-rs 0.2.0 on crates.io (#129);
  - Copper integration guide (#130);
  - v0.1.0 release (#136);
  - IROS 2026 workshop paper (#137).

## Now — egocentric accuracy on LaMAria

**Experiments take at most 5 min (owner, 2026-11-23):** short unpaced windows or offline runs on saved sessions, at least 3 repeats. The gate below runs once per batch of stacked winners, not per card.

**Regression gate (owner, 2026-11-22).** No knob is promoted into a preset or a code default until it passes the full-sequence gate:
- **Runs:** all 11 validation sequences (R_11_5cp, 1_19, 1_20, 2_11, 2_12, 3_17, 3_18, 4_10, 4_11, 5_11, 5_12), full length, paced at sensor timestamps, on a quiet machine (load ≤ 4, no other SLAM job).
- **Samples:** 2 runs of the candidate against 2 runs of the current baseline.
- **Pass rule:** no sequence's official CP score or tracked frames falls below the baseline's range. Any drop is diagnosed before the knob is promoted.
- **R_11:** with 3 usable control points the CP Sim3 often fails (scored 0), so R_11 is gated on tracked frames and ATE (`ate.py`).
- **Guards:** the 800-frame per-card guards stay as a cheap first check. They do not replace the gate: card AC's Held seed passed them, yet R_11 fell from 55.5 to 26.9 at 370 s.
- **Cleanup card:** it applies the gate to the stacked winners together.

We are working toward top results on the [LaMAria](https://github.com/cvg/lamaria) egocentric benchmark ([#78](https://github.com/kornia/kornia-slam/issues/78)), using the Aria rig and IMU and no GPS. The work is sequenced by where the benchmark score is lost: each change states its theory and predictions first, ships behind a config knob that defaults to the old behaviour, and is compared window by window against ORB-SLAM3 in the same harness.

General purpose, not tuned per deployment (owner, 2026-11-21). A new rig or dataset should need its sensor model only: calibration, IMU noise and frame rate. Knobs are how a change lands and is measured; after each validation round a cleanup card makes the winners the code default and deletes the refuted or redundant knobs.
- [ ] **Card AI: global stage (owner, 2026-11-23: main card; planning).** Every top entry scores from a global solve (SLAM Dunk: global BA about 69, then 82.4 with calibration over time and gyro intrinsics). Two parts:
  - **(a)** a VIO that never resets: IMU coasting with honest covariance, then re-acquiring vision into the same state (Basalt, OKVIS2, OpenVINS). It replaces the give-up and re-seed path.
  - **(b)** a background global graph: IMU bridges and cross-segment matches join the segments, then a bounded global VI-BA runs. Corrections go through a map-to-odom transform. The same code runs as the end-of-sequence batch.
  - **Then the calibration ladder:** camera calibration over time, gyro intrinsics, accel, clock and thermal terms (the `offline_ba.calibration.*` knobs are built).
  - Stage 0: join the segments of saved sessions offline and score CP.
- [ ] **Card AG: drift budget and loop closure on the challenge sequences** (owner, 2026-11-22; stage 0 done 2026-11-23, oracles feed card AI; ORB-SLAM3 reference runs dropped). It splits each sequence's CP loss into breaks, events, smooth drift, coasts and scale, with oracle scores, and checks GT for revisits. Then ORB-SLAM3 reference runs on 1_19, 2_11 and 1_20 (2 each, after the gate), and the #180 loop closer observe-only on 2_11 and 1_20, with 1_19 as the control (on the owner's go). No apply arm: it needs loop and odometry sigmas derived from the sensor model first.
- [ ] **Card AH: the window bias absorbs vision errors** (planned and approved 2026-11-23; stage 0 running; 5-minute windows). In weak vision the local VI-BA moves the gyro bias about 40σ beyond its random walk, and heading steps 10-30° (1_19, 1_20, R_11). First test: card S's `mapping.local_ba.boundary_scales_bias_rows: false` on the event windows. References: ORB-SLAM3 bias-walk edges at full strength, Basalt's marginalisation prior, OpenVINS first-estimate Jacobians. Target: 1_19 heading step ≤ 3°, CP 30-40.
- [ ] **Paused 2026-11-23 (card AI first). Card AE: trusted inertial state for a Held re-seed** (planned and approved 2026-11-22): the median bias over vision-anchored keyframes plus a gyro attitude chain. It fixes the R_11 and 4_11 seed-bias deaths found by the H0 gate.
- [ ] **Paused 2026-11-23, folded into card AI (a). Keep one world through a tracking loss** (owner, 2026-11-22): carry the gyro attitude across a give-up, give a new segment the kept inertial state, refuse visual velocities the IMU contradicts, link or merge the segments (ORB-SLAM3 RECENTLY_LOST and Atlas merge; Basalt and OpenVINS never reset). See `experiments/pr180-validation-diagnosis.md`.
- [ ] **Paused 2026-11-23 (card AI first). Card AD: gyro rotation prediction before IMU init** (owner, 2026-11-22). Before VIBA0 the tracker predicts each frame with zero motion (`predict_frame_pose` needs `imu_initialized`; the RotatedRay guess needs a predicted velocity), so a start-up turn of 3-4.4° per frame leaves a 29-32 px PnP residual against the 25 px gate and resets 19-56-point maps at once (1_19 frames 0-13; G31 class). The previous-frame timestamp (`state.last_frame_timestamp_sec`) also stays stale across cold bootstraps. Plan approved 2026-11-22 (knob `tracking.prediction_before_init: Visual | Gyro | GyroHeld`, timestamp refresh unconditional); starts when card AC lands.
- [ ] **Diagnose the first-map resets on R_01, R_03, R_11** (owner, 2026-11-22, read-only): the first map is lost on the 2nd or 3rd frame after a cold bootstrap at under 0.8° per frame, with 2-9 projected matches; not rotation, so card AD does not cover it. Ledger row to add with the diagnosis.
- [ ] **Cleanup after the first validation run** (owner, 2026-11-22: after the loss card, then card F).
  - **Make the fixes the only path:** `dense_parallelism: Sequential`, `merge: PropagateToNewer` (with the stop at a re-seed), `frame_velocity: PropagatedFromKeyframe` (card U), and the coasted-pose output.
  - **Make the reference constants the code defaults:** 10 active keyframes, 7 iterations from λ 1e-4 (Basalt), `growth_min_baseline_ratio` tan 1° (ORB-SLAM3).
  - **Delete the refuted or redundant knobs:** `keyframe_seed` (card R), the full-weight boundary edge (card O), `SnapshotOnly`.
  - **Shrink `pr180-window10`** to dataset and sensor settings.
- [ ] **Derive the remaining hand-set thresholds from the sensor model and the estimate's uncertainty:** depth sigma from f·baseline and disparity, gates from covariance, coast caps from the propagated IMU covariance, `min_cold_start_points` from the pose covariance of the cold start.

Next, start-up and scale (2026-11-19, from cards I and J; in this order):
- [x] **Baseline and parallax gate on new map points** (card K, 2026-11-21: built, knob `mapping.growth_min_baseline_ratio`; at 0.0175 (tan 1°) with `min_cold_start_points: 20`, R_03 784-798/800, R_11 798-800/800, R_01 800/800; shipped in `pr180-frontend-1ms`).
- [ ] **Raw fisheye images instead of the pinhole crop (card F).** Basalt, OKVIS2, ORB-SLAM3 (`KannalaBrandt8`) and Aria's own SLAM run on the full fisheye view; the wider stereo overlap removes the R_11 start-up starvation (G31) at its source.
- [ ] **Still-device handling.** Detect no motion; skip keyframes or apply a zero-velocity update, and initialise gravity and bias from the still period, as OpenVINS and Basalt do.
- [ ] **IMU carries tracking through visual gaps.** Keep the young map and propagate with the IMU when a frame has too few matches, instead of resetting it at its first lost frame (Basalt, OKVIS2, VINS); ORB-SLAM3 keeps old maps in its Atlas and merges them later.

Losses and recovery:
- [x] Keep every pose rigid at the source; rotation scale crept in through repeated pose corrections
- [ ] Lost-mode keyframes as fixed anchors (built, knob `keyframes_while_lost`, not yet validated; no gain on its own)
- [ ] Keep gravity, bias and velocity through a tracking loss, so a re-seeded map shares the old world frame (built, knob `tracking_loss.reseed.keep_inertial`, not yet validated; no gain on its own)
- [ ] Reject tracked poses that disagree with the gyro prediction (built, knob `tracking_loss.pose_gate`, not yet validated; no gain on the proxy)
- [ ] Global PnP relocalisation while lost; a re-seeded map with its own local window (built, knobs `tracking_loss.relocalization.while_lost` and `.solver`, `tracking_loss.reseed.own_window_keyframes` and `.relocalize_when_lost`, not yet validated)

Scale and drift while tracking:
- [ ] Longer visual-inertial BA window, and keyframes no longer dragged away from the map by BA (built, knobs `mapping.local_ba.large_window`, `mapping.local_ba.imu_huber`, `tracking.snapback_fix`, not yet validated; `mapping.local_ba.active_keyframes` already existed)
- [ ] Per-frame velocity state and a pose-inertial solve per frame (built, knob `tracking.pose_inertial`, not yet validated)
- [ ] ORB-SLAM3's keyframe insertion, local map and growth rules (built, knobs `keyframes.rules`, `tracking.local_map`, `mapping.growth_neighbours`, not yet validated)
- [ ] Camera-IMU clock offset from the device record, then estimated online, and ORB-SLAM3's IMU kernel layout (card AA running; the online estimator, `online_calibration.clock_offset`, is built and not yet validated)
- [ ] Window-local frame for the VI-BA solve, so its convergence does not depend on the distance from the world origin (card Y, W2; arithmetic only so far)
- [ ] IMU bias and noise model, compared across ORB-SLAM3, Basalt and other references (built in part, knob `inertial.sample_interpolation`, not yet validated)
- [ ] IMU covariance trust: trust the inertial prediction by its propagated covariance (built, knob `tracking.imu_trust`, not yet validated)
- [ ] Offline global visual-inertial BA (built, run file `offline_ba` and the `offline-ba` subcommand over a `--session-out` dump, not yet validated)
- [ ] Calibration states in the offline BA: camera rotation and focal drift, gyro intrinsics, clock offset (built, knobs `offline_ba.calibration.*`, not yet validated)

Field of view and speed:
- [ ] Raw Aria fisheye cameras instead of the pinhole crop: deferred to its own round (card F)
- [ ] ORB only on keyframes with KLT in between, on every rig camera (built, knob `klt_rig`, not yet validated; `klt_first` already existed)
- [ ] Optical flow on every rig camera, forward-backward check, refill, gyro guess and flow-only pose (built, knobs `klt_rig`, `forward_backward`, `track_refill`, `gyro_guess` and `flow_pose`, not yet validated)
- [ ] Window-only fixed-point LK at an 11x11 window (built, knobs `klt_kernel: WindowedU8` and `klt_window`, not yet validated)
- [ ] Flow workers across 2-3 cores (built, knob `flow_threads`, not yet validated)
- [ ] Keyframe ORB on the mapping thread (`keyframe_orb: Mapping`): deferred, its design is open (lane T step 12)

## Next — complete the SLAM stack

- [ ] GPU acceleration of the per-frame hot paths (feature extraction, matching, KLT
      tracking), starting with the KLT frontend from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs)
- [ ] Relocalization on tracking loss, robust to appearance change (day/night, lighting) —
      measure the degradation first ([#68](https://github.com/kornia/kornia-slam/issues/68))
- [ ] Metric scale for monocular maps: Sim3 loop closure, and AprilTag anchoring
      ([#71](https://github.com/kornia/kornia-slam/issues/71), [#72](https://github.com/kornia/kornia-slam/pull/72))
- [ ] Match ORB-SLAM3 on trajectory accuracy and tracking robustness
- [ ] Upstream general-purpose code (solvers, camera models, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs). Staged in `crates/kornia-slam/src/upstream/`, one module per target (built, not yet upstreamed):
  - [ ] `kornia_3d::pgo`: PGO manifolds and edges
  - [ ] `kornia_3d::pnp::refine`: analytic 2x6 pose refine
  - [ ] `kornia_3d::pnp::gpnp`: generalised PnP and AP3P on rig rays
  - [ ] `kornia_3d::pose`: `camera_centre` and the `Pose3d`/`SE3` conversions
  - [ ] `kornia_algebra`: f32/f64 casts

## Library structure

- [ ] Pluggable feature frontend for non-ORB and learned features, e.g. XFeat ([#49](https://github.com/kornia/kornia-slam/issues/49))
- [ ] Pluggable place recognition, including learned global descriptors such as DINOv3
      ([#48](https://github.com/kornia/kornia-slam/issues/48))
- [ ] Multi-camera rigs (2+ cameras), ported from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs);
      generic rig calibration file, a ray per rig pixel and a rig-aware AP3P (built, knobs `rig_calibration`
      and `tracking_loss.relocalization.solver: RigAp3p`, not yet validated)
- [ ] Explore [Atlas](https://github.com/kornia/kornia-slam/issues/59), a manifold-native estimation and factor-graph foundation, as
      the base for BA, VI-BA and pose-graph optimization
- [ ] Split `SlamSystem` into tracking, mapping, inertial, and loop-closing subsystems

## Tooling and integrations

- [ ] Pixi environment and tasks for reproducible cross-platform builds
      ([#77](https://github.com/kornia/kornia-slam/issues/77), [#79](https://github.com/kornia/kornia-slam/pull/79))
- [ ] ROS 2 integration via [ros2_rust](https://github.com/ros2-rust/ros2_rust)
      ([#75](https://github.com/kornia/kornia-slam/issues/75))

## Experimental — sensors, maps, agents

- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats ([#141](https://github.com/kornia/kornia-slam/issues/141))
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
