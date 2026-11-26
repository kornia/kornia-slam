# EuRoC checkpoint, 2026-11-26

**Purpose.** This checkpoint fixes the exact next actions toward the EuRoC goal. It merges two independent read-only reviews (Fable, high effort) of PR #180 with the D5b result.
- **Accuracy review.** Per-sequence causes, the ground-truth floor, and the choice between the pipeline and the estimator.
- **Speed review.** The latency tail and CPU per frame, plus whether our speed measurements are valid.
- **Not yet reviewed.** Neither review was reviewed itself. Figures marked [review] come from the reviewers' inline scripts, which were not saved. A harness script must reproduce them before they are cited as results.
- **Spot-checked so far.** Two review claims, both confirmed:
  - The MH_04 rep at 0.356 has `[pgo] failed kf=1635 ~ kf=756: unusable solve: not converged after 30 iterations` (`FD-RT3E_MH_04_difficult_s0_r3`). The other 5 RT3EF MH_04 runs on the same binary have no PGO failure and score 0.210-0.228.
  - V2_03 has 1922 cam0 images and 2336 cam1 images.

Goal (owner): causal real-time ATE ≤ 0.5x Basalt on all 11 sequences, and faster than Basalt on 2 cores. EuRoC this week; LaMAria from 2026-11-30.

## 1. Where we are

Causal real-time SE(3) ATE (m) on 2 cores. RT3EF is today's single config (median of 3 runs). G57-a is RT3EF plus asynchronous loop closing (1 or 3 runs).

| seq | Basalt | RT3EF | ratio | G57-a | ratio | pooled RT3EF range, n [review] |
|---|---|---|---|---|---|---|
| MH_01 | 0.067 | 0.043 | 0.64 | 0.048 (1) | 0.72 | 0.042-0.058, 6 |
| MH_02 | 0.048 | 0.036 | 0.75 | 0.035 (1) | 0.73 | 0.027-0.040, 6 |
| MH_03 | 0.062 | 0.050 | 0.81 | 0.043 (3) | 0.69 | 0.044-0.054, 9 |
| MH_04 | 0.114 | 0.228 | 2.00 | 0.212 (1) | 1.86 | 0.210-0.356, 6 |
| MH_05 | 0.144 | 0.094 | 0.65 | 0.095 (1) | 0.66 | 0.080-0.103, 6 |
| V1_01 | 0.043 | 0.047 | 1.09 | 0.044 (3) | 1.02 | 0.043-0.054, 9 |
| V1_02 | 0.045 | 0.043 | 0.95 | 0.036 (1) | 0.80 | 0.043-0.057, 6 |
| V1_03 | 0.053 | 0.127 | 2.38 | 0.115 (3) | 2.17 | 0.101-0.185, 9 |
| V2_01 | 0.039 | 0.051 | 1.31 | 0.051 (1) | 1.31 | 0.049-0.111, 6 |
| V2_02 | 0.049 | 0.106 | 2.15 | 0.102 (3) | 2.08 | 0.093-0.115, 9 |
| V2_03 | 0.230 | 0.568 | 2.47 | 0.499 (1) | 2.17 | 0.478-0.577, 6 |

- **Below Basalt, robustly:** 4 of 11 (MH_01, MH_02, MH_03, MH_05).
  - The 2026-11-26 log counted 6. That was too generous: V1_01 is below Basalt on 0 of 9 reps, and V1_02 on 4 of 6, with the median 0.043 against 0.045.
- **At 0.5x:** none.
- **Basalt's MH_01 reference is mostly a scale error** [review]: SE(3) 0.067, Sim(3) 0.024, scale 1.0148. Ours is 0.043 and 0.033 at scale 1.0066. So our 0.64x there is a scale win; in shape, Basalt is ahead.
- **Speed** (G57 full check, same clock): faster than Basalt on 1 of 4 sequences (MH_03). p50 ties or wins elsewhere; p99 and Vicon-room CPU lose.

## 2. What the reviews changed

**Accuracy (corrections to `euroc-why-not-basalt.md` and the ledger).**
1. **MH_04 rep 3 (0.356) is G60, not the G63 fork.** Confirmed above. The ledger's line "on the tip the MH_04 loop no longer fails" holds for SYNC S0PEIH only. The reviewer counts 8 `[pgo] failed` lines in real-time tip-era runs: MH_04 1, MH_05 5, MH_01 2.
2. **V2_03's row says "K1-c is tracing it". That is wrong.** K1-c traces the estimator's divergence, and nothing diagnoses the pipeline coasts. New evidence [review]:
   - 414 of 1920 frame intervals are 100 ms (cam0 drops images). No other sequence has gaps.
   - 34 of 35 coast runs start on the frame after a 100 ms gap, against a base rate of 22 %.
   - The 60 px flow cap is a weak candidate: gyro-predicted motion exceeds it at only 3 of the 35 coast starts.
3. **V1_03: RT3EIH (0.053) does not prove the PnpInliers cause.**
   - It changes two knobs: it drops PnpInliers and adds pose-inertial tracking.
   - It ran without FixedPoint rectification or background loop closing.
   - The SYNC split (S0EIH, S0PEIH on V1_03 and V2_02) that would separate the two knobs never ran.
4. **V2_01: "G59 plus floor" is stale for today's config.** Typical RT3EF reps (0.049-0.051) equal SYNC (S0E 0.055). The 0.111 rep is untraced.
5. **Ground-truth floor** [review]. All three systems have about 1 % scale error against Vicon on V1_01, V1_02, V1_03 and V2_01 (ORB-SLAM3 1.009-1.010, Basalt 1.011-1.016, ours 1.004-1.016).
   - A 1 % scale error alone costs 0.016-0.023 m of SE(3) ATE there.
   - So 0.5x is below the floor on V1_01 (target 0.022) and V2_01 (0.020).
   - It is at the floor estimate on V1_02 (0.023).
   - It is below every result on disk, offline ones included, on V2_02 (0.025) and MH_04 (0.057).
6. **Path: the pipeline, not the estimator, for the EuRoC days.**
   - The estimator cannot drive yet (`Drive` is refused at `K/system/validation.rs:605`).
   - Its VIO-only ratios are 0.65-1.99x, and it diverges on V2_03 with no reset.
   - It beats the pipeline only on MH_04 (0.91 against 1.96).
7. **Rigor problems.**
   - G57-a's 1-run "below the band" claims on V2_03 and V1_02 are inside the pooled spread.
   - V1_03 spreads 0.101-0.185 over 9 reps, so any V1_03 claim needs at least 5 reps.
   - V-c used starts 0/1/2, which the plan calls degenerate; it requires 0/30/60.

**Speed (new findings).**
1. **No p99 verdict is supportable on today's protocol.**
   - The worst tail episodes sit in windows where image decode plus rectification runs 2.5x slower. Those windows fall at different frames in each run.
   - p99 is the 22nd-29th worst frame, so one episode per run decides it. Rep spread is 2-6x for both us and Basalt.
   - The "0.8-3.4 GHz" figure is an idle snapshot taken at block start, not the clock under load.
   - Cores are pinned but not shielded: Chrome sits on cpu6, the SMT sibling of core 0.
2. **The latency definitions differ.**
   - Ours reads and rectifies the image before the pacing sleep (`A/main.rs:424-441`), so decode is outside our latency.
   - Basalt decodes after its sleep (`basalt/paced/vio.cpp:196-197`), so its p50 carries an estimated 3-5 ms of PNG decode.
   - So the V2_02 p50 "tie" (37.8 vs 37.6) is not a win.
3. **The cheapest speed lever has never been built: wasted ORB frames.**
   - 14-29 % of frames on all 11 sequences run ORB and then do not become keyframes. The flow gate compares flow inliers with 0.6x the reference keyframe's map points, and ORB then finds enough (`K/system/mod.rs:2669-2682`, `K/tracking/policy.rs:263-270`; gap G33, card Z).
   - Predicted CPU per frame [review]: V2_02 58.7 → 53-56, V1_03 55.7 → 49-53, V1_01 58.4 → 47-51.
4. **G57-b's lock convoy is 2-3x larger than `map_wait_ms` reports.**
   - The counter is read inside `track_frame` (`K/system/mod.rs:866`, `:874`). The app's later lock calls (`A/main.rs:512-513`) also wait, and are not counted.
   - G57-b-2 must be scored on the queue-replay residual, not on `map_wait_ms` alone.
5. **`num_active_map_points` is an O(N) scan under the map lock on every frame** (`K/mapping/map/mod.rs:156-158`, called at `A/main.rs:513`). It costs 0.2-1.5 ms per frame, growing with map size.
6. **Ranking by queue replay.**
   - The flow-gate fix plus G57-b-2 is the only stack that flips both V2_02 and V1_01 on paper: p50 10-12 ms, p99 64-73 ms.
   - G57-c (loop recognition off the tracking thread) helps V2_02 but not V1_01.

**D5b (background loop closing, `g57-plan.md` "Results: D5b").**
- **Deferred writeback:** writing back one frame late costs 1 of 15 loops (prediction ±1 holds). Loop sets diverge after the first timing difference.
- **Background path:** loses 3 of 15. The extra 2 are late closures to early keyframes.
- **Where the loss happens** (V1_03 debug pair): place recognition returns the same candidate (kf 2065 → kf 72) in both arms. The background arm rejects it at acceptance, on a map corrected by a different earlier loop.
  - It is not a solve cut off at the end of the run.
  - The rejecting stage (verification, odometry gate, temporal gate, episode tracker) is not logged.
  - This is D5d.

## 3. Targets for the owner to restate (stop S-A)

Lowest credible causal target per sequence [review], against the 0.5x goal:

| seq | 0.5x | credible target | why |
|---|---|---|---|
| MH_01 | 0.034 | 0.036-0.038 | smoothed 0.028; offline 0.031; the gap is causality (G53) |
| MH_02 | 0.024 | 0.028 | smoothed 0.021; offline 0.020 |
| MH_03 | 0.031 | 0.036-0.040 | offline 0.027 and ORB-SLAM3 0.028 are the best non-causal results |
| MH_04 | 0.057 | 0.11 (1.0x) | below our own offline result (0.070) |
| MH_05 | 0.072 | 0.075-0.080 | smoothed 0.054 |
| V1_01 | 0.022 | 0.043 (1.0x) | ground-truth floor 0.037-0.039 |
| V1_02 | 0.023 | 0.030-0.034 | floor estimate 0.018-0.021 |
| V1_03 | 0.027 | 0.053 (1.0x) | equal to ORB-SLAM3's full BA |
| V2_01 | 0.020 | 0.036-0.039 | floor 0.031-0.035 |
| V2_02 | 0.025 | 0.049 (1.0x) | below every result on disk |
| V2_03 | 0.115 | 0.12-0.15 | if the coasts go (offline 0.120) |

## 4. Next actions, ranked

Every item is config only or small code unless marked. Each code item goes planner → implementer, theory first.

| # | action | sequences | predicted (basis) | cost | kill |
|---|---|---|---|---|---|
| A1 | **Speed protocol.** On AC power: performance governor, cpus 0, 1, 6, 7 shielded, Chrome off those cores. Move Basalt's decode before its sleep. Gate each run on `source_ms`. ABBA pairs, 3 for a screen and 5 for a verdict. Speed runs without `--session-out` and `--trajectory-smoothed` | all | Our V2_02 p99 173-185 on every rep; Basalt p99 97-135, p50 −3 to −5 ms (decode estimate) [review] | 1-2 h harness, **owner's sudo** | slow source windows still above 2 % of frames after shielding: the cause is ours |
| A2 | **Pose-inertial tracking with FixedPoint and background loops** (config only). SYNC first: S0EIH and S0PEIH, V1_03 and V2_02, starts 0/30/60. Then real time: RT4E plus `tracking: ( pose_inertial: OrbSlam3, pose_inertial_min_inliers: WithPrediction )`, with and without PnpInliers | V1_03, V2_02, MH_04; guards MH_01, MH_05, V1_02 | V1_03 0.129 → 0.053-0.060; V2_02 0.100 → 0.082; MH_04 0.224 → 0.17-0.19 (RT3EIH reps, S0PEIH 0.178-0.185) | 12 SYNC plus 36 paced runs, about 2 h | V1_03 high-frequency error above 0.030 on 2 of 3; any guard above 1.15x; p50 above 50 ms; `pose_inertial_solves` = 0 |
| A3 | **V2_03 coast trace.** One `--debug` S0PE run; read the reject reason at the 35 coast starts; then a card | V2_03 | 0.53 → 0.15-0.25 if coasts fall to ≤ 20 (rig arms; offline 0.120) | 1 run of 5 min | the reject reason is unrelated to the frame gaps |
| A4 | **Flow-gate fix (wasted ORB frames)**, behind a knob. Calibrated form: scale flow inliers by the full-to-flow inlier ratio at the last ORB frame. EuRoC card first | all 11 | CPU V2_02 53-56, V1_03 49-53, V1_01 47-51; V2_02 p99 150-225. Counter: non-keyframe ORB frames 14-29 % → ≤ 5 % | 0.5-1 day code | keyframes change by more than 5 %, more coasts, or ATE leaves its band |
| A5 | **D5d.** Log the rejecting stage in loop acceptance, then rerun the S0E/S0EA debug pair on V1_03 | V1_03, V1_02 | names the stage that rejects kf 2065 → 72 | small code plus 2 runs | none (diagnosis) |
| A6 | **G57-b-2** (planner card ready: T0 trace, then the prepare/apply split). Score it on the queue-replay residual, not `map_wait_ms` alone | all | from RT4EB: V2_02 p99 83-321, V1_01 92-159; stacked on A4: 64 / 73 | medium code, about 650 lines | card KS1-KS6, NH1-NH4 |
| A7 | **Batch gate.** RT4EF plus `accept_capped_solves: true` (G60-b) as the new baseline. 3 reps; 5 on V1_03, V2_01 and MH_04 | all 11 | removes MH_04's 0.356 mode; medians unchanged | about 1.7 h | any `[pgo] failed`; loops −20 % with an ATE cost |
| A8 | **Zero-run reads.** V2_01 rep 2 (0.111); Sim(3) scale table for all three systems; save the reviewers' scripts in the harness | V2_01, S-A | explains the tail and the floor | 0 runs | none |
| A9 | **Cached active-point counter** in the app status line | all | CPU −0.5 to −1 ms per frame | 1 h code | status-line `pts=` differs from today |
| A10 | **V-d2** (carded 2026-11-24, never run): S0E plus a window of 10 at 7 iterations | V2_02 | 0.080 → 0.058-0.072 (SYNC, card) | 6 SYNC runs | card Kd1-Kd3 |
| later | G57-c (loop recognition on the worker), only after A4 and A6; prefix replay for the halls (MH_01, 02, 05); estimator reset (one slot) | | | | |

**Order.**
- Start now, machine-bound, config only: A3, A8, then A2 (SYNC cells first).
- A1 needs the owner's sudo.
- Start code now, through the planner: A4 and A5.
- After A5: A6. After A2, A4 and A6: A7, then the full gate.

## 5. Stop

- **Speed verdicts** taken from today's p99, from ties under 3 %, or from the block-start GHz reading.
- **More G57-b (RT4EB) runs, or promoting it,** before b-2.
- **RT3EIH as an arm** (no FixedPoint); A2 replaces it.
- **V-c2 and the PnpInliers verdict:** they wait for A2's SYNC cells.
- **Estimator this week:**
  - K1-d;
  - more A/B 1 screens;
  - Drive mode (2x-6) and map reuse (2x-7) before 2026-11-30;
  - its one slot is a divergence reset.
- **Rig lanes** R-a, F6 and the K2 split; G60-c this week.
- **Tuning for speed:** `min_frames_between`, `mapping_acceptance`, `RAYON_NUM_THREADS=1` reruns, and more rectification work.
- **Measurement habits:**
  - blocking Basalt runs inside timing blocks;
  - SYNC starts other than 0/30/60;
  - the "floor + 10 %" rule (use the measured floor).

## 6. Owner decisions needed

1. **Targets (S-A).** Restate the targets per sequence as in section 3, or keep 0.5x as a stretch with a "credible" column?
2. **Vicon-room scale.** Accept the common ~1 % Vicon-room scale as part of the floor, or investigate it first?
3. **Speed protocol (A1).** Can you run the sudo steps? Turbo on or off for verdict runs? Should Basalt be re-measured with its decode before the sleep?
4. **Real-time validity.** Which latency bound makes a real-time ATE valid? No run meets p99 ≤ 100 ms today.
5. **V2_03 coasts.** Should coast frames count against us on V2_03, by cropping Basalt to our stamps or by writing coasted poses?
6. **Flow-gate fix (A4).** May the keyframe count change, or must it stay within ±5 %?
7. **Batch gate (A7).** May G57-a and G60-b go through one batch gate together?
8. **Verdict gate.** Is 5 pairs × 11 sequences (about 4 h) acceptable as the speed verdict gate?
