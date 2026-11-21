# Card V: 0-inlier jobs solve with the base VI-BA budget, and a merge carries the old window across a bridged re-seed

Status: written 2026-11-21, before any code. From the global review of feat/global-pose-graph (E-1, E-2).

## Experiment: V, one VI-BA budget for every job (E-1); a propagating merge stops at a re-seeded segment (E-2)

- Gap tested: G35, the real-time side (card T K4, R_03) for E-1. No ledger row for E-2: add G36, "a merge carries
  the old window's velocity and bias across a bridged re-seed", suspected, score at stake unknown.
- Base: kornia-slam feat/global-pose-graph @ 37fa7f6 (code as c3d19d0). Binary cardT/bin/kornia-slam-t-seq,
  md5 32b6d5506ebeaa18fba296abac1e8a4d. New binary tag: v (record md5 at build).
- Baselines: cardT/out3/{r03-W,r11-W,r01full-W}-r{1,2,3}, preset configs/pr180-window10.ron, 2 jobs, load p50 4.2.

### Mechanism (file:line at 37fa7f6)

E-1.
- The preset sets large_window: Enabled(min_tracked_inliers: 0, active 10, 7 iterations, lambda 1e-4).
  large_window_applies tests tracked_inliers > min_tracked_inliers (mapping/bundle_adjustment.rs:151-156), so a job
  with 0 inliers solves with the base settings (mapping/local_mapping.rs:62-74): 10 active, max_iterations 50,
  initial_lambda 1.0.
- initial_lambda is not a RON field: system/build.rs:176 writes LocalBaParams::default().initial_lambda.
- Three jobs carry 0 inliers: a keyframe inserted while lost (system/mod.rs:2071), the first job after VIBA0
  (system/mod.rs:1664-1671), and the job after a pose-graph optimisation (system/mod.rs:3252-3259).
- Lost keyframes are free 15-DOF states with no observation (LostKeyframes::Free, bundle_adjustment.rs:689-696).
- Measured on the baselines ([vi_ba] TIME lines):

  | Sequence | VI solves over 7 iterations | Most iterations | Slowest solve |
  |---|---|---|---|
  | R_03 r1 / r2 / r3 | 11 / 5 / 4 | 29 / 27 / 25 | 867 / 880 / 668 ms |
  | R_11 r1 / r2 / r3 | 1 / 1 / 2 | 44 | 459 / 464 / 398 ms |
  | R_01 full | 0 | 7 | 199-230 ms |

  The longest solve of each run is the job after VIBA0. In the synchronous run cardT/out2/r03-T-r1 the lost
  keyframes 372-399 solve in 13-21 iterations, and [g29] reports keyframe 395 moved 0.384 m.
- Not changed here: the worker takes the latest job before it takes the publication gate
  (local_mapping.rs:489, 494-501), so a job's settings can belong to an older keyframe than the snapshot's newest.
  With one budget for every job this no longer changes the budget.

E-2.
- propagate_to_newer (mapping/map/ops/correction.rs:845-911) stops at an anchor or at a keyframe with no incoming
  IMU edge (:868). A bridged re-seed gives the new segment's first keyframe an incoming edge from the old segment
  (publish_bridge, system/mod.rs:2700-2734, called at :1300-1301).
- A solve still running when the re-seed lands therefore carries its correction over the bridge: the velocity is
  re-predicted over an edge up to max_bridge_sec long, and the bias is the old window's (:877-900).
- system/mod.rs:1039-1042 hands both to the tracker. give_up zeroes an out-of-limit bias
  (system/mod.rs:2643-2653); the merge writes the old bias back.
- Exposure: reseeds=0 and bias_resets=0 in all nine baseline runs. The path was not taken on R_01, R_03 or R_11.

### Theory and reference

E-1.
- A sliding-window solve has one budget. Basalt resets lambda to 1e-4 and caps at 7 iterations on every solve
  (vio_config.cpp:73,78; sqrt_keypoint_vio.cpp:1062,1097), and damps by lambda times diag(H) as this solver does
  (sqrt_keypoint_vio.cpp:1212), so its value transfers.
- ORB-SLAM3 runs 10 iterations from lambda 1 when not bLarge and 4 from 1e-2 when bLarge
  (Optimizer.cc:2387-2393, 2513-2522, 2843). It never runs 50, and its lambda multiplies I, not diag(H).
- Basalt's choice suits us best: it is the budget cards R and T validated, and it bounds the solve time.
- The theory predicts a bounded solve time. It does not predict an accuracy gain: the affected solves are 4-11
  of about 200 on R_03 and at most 2 elsewhere.

E-2.
- A segment re-seeded after a give-up starts from its own state. ORB-SLAM3 starts a new map with a fresh
  preintegration at zero bias and no last keyframe (Tracking.cc:2015-2024, 2683-2690); nothing inertial crosses.
- The bridge edge is published with an inflated covariance because a coast says little. The merge treats it as exact.
- Fix: stop the propagation at a keyframe that starts a re-seeded segment. The correction of the old window then
  sits on the bridge edge as a residual, inside that edge's inflated covariance, for the next solve to weigh.

### Knob

- E-1: mapping.local_ba.initial_lambda, default 1.0 (today). The candidate sets max_iterations: 7 and
  initial_lambda: 1e-4 at the base level and leaves large_window as it is.
- E-2: no knob. The change is inside merge: PropagateToNewer, itself a knob whose default (SnapshotOnly) never
  reaches this code. The function's own comment (correction.rs:838-844) already states the stop at a segment start.
- Instrumentation, no knob: [summary] vi_ba_base_solves, vi_ba_base_iterations, vi_ba_iterations_max,
  ba_propagations_stopped_at_bridge.

### Configs

| Name | Config |
|---|---|
| W | cardT/cfg/{r03-W,r11-W,r01full-W}.ron, new binary |
| V1 | W plus local_ba: ( max_iterations: 7, initial_lambda: 1e-4 ) |

Runs: W and V1 on the R_03 800-frame window, 3 each, 2 jobs. All knobs on (with card U): R_03, R_11, full R_01, 3 each.

### Predictions

- P1 (mechanism, V1): vi_ba_iterations_max 25-29 -> 7 on R_03, 44 -> 7 on R_11, 7 -> 7 on R_01.
  VI solves over 7 iterations: 4-11 -> 0 on R_03, 1-2 -> 0 on R_11.
- P2 (time, V1): slowest solve 668-880 ms -> at most 300 ms on R_03; 398-464 ms -> at most 150 ms on R_11.
- P3 (start-up, V1): the VIBA1 and VIBA2 accepted lines match W in every run; R_11 tracked 798-800.
- P4 (accuracy, V1): no change beyond the band. R_03 ATE SE3 at most 0.12 m (W 0.085-0.112), tracked at least 762
  (W 762-789), scale 1.02-1.04.
- P5 (mechanism, E-2): ba_propagations_stopped_at_bridge = 0 in every run of this card (reseeds = 0). The unit
  test is the proof; no score is claimed.

### Kill rules

- K1 (mechanism moved, score fell): P1 holds, and R_03 ATE is above 0.13 m or tracked below 762 in 2 of 3 runs,
  or R_11 tracked is below 798, or a VIBA2 is rejected. The long solves were needed (the job after VIBA0 ran
  25-44 iterations). Keep the base budget out of the preset; next, a budget for that job alone.
- K2 (score moved, mechanism did not): R_03 tracked 790 or more in 3 of 3 runs, or ATE under 0.08 m in 3 of 3,
  in V1 alone. The theory does not predict it. Do not credit E-1; check the [vel] error at age 1 after a recovery
  (card U) for a link to the lost-keyframe solves before any claim.
- K3 (E-2 exposed): ba_propagations_stopped_at_bridge above 0 in any scoring run. That run differs from the base
  by E-2. Score a sequence with re-seeds (full R_11, give-up window 6239-8241) 3 times before any claim.
- K4: the trajectory of the synchronous identity run is not cmp-identical. Stop; a default changed.

### Bit-identity and noise

- Knobs off: cardR/cfg/r01full-S10-7.ron with the new binary must be cmp-identical to
  cardT/out/T1/r01full-S10-7-r1/traj.txt (md5 d4f77b567b912b5a4caa2af6085a35b1). A synchronous run has no
  newer keyframes, so E-2 cannot act on it.
- Unit tests: the default initial_lambda resolves to 1.0; a propagating merge without a segment start is unchanged
  (the existing test).
- Noise: asynchronous runs differ run to run. All verdicts are on 3 runs. R_03 bands from W: tracked 762-789,
  ATE 0.085-0.112 m.

### Trajectory output fixes in the same round (no estimator change)

- T-1: smoothed.record (main.rs:486-497) is gated like --trajectory-out, so the session dump and offline-ba
  receive the coasted frames under --trajectory-include-coasted. Flag off: the dump is unchanged.
- T-3: a re-seed frame carried on the IMU gets its own SkipKind and is written under the flag.
- T-4: a coast is written under the flag only when the prediction is trusted and not a runaway.

## Result log

- **2026-11-21, scoring.**
  - **Setup:** `feat/global-pose-graph` @ f57c13a, binary `cardV/bin/kornia-slam-v`
    (md5 975ce3b7802abbd03f7540e53ec59604), configs `cardV/cfg/{r03-W,r03-V1,r03-ALL,r11-ALL,r01full-ALL}.ron`
    (ALL = V1 plus card U's `frame_velocity: PropagatedFromKeyframe`), runs in `cardV/out`, `RUN_FLAGS` empty,
    3 runs each, 2 jobs. Load p50 over the batch 6.77; V1 runs 11.8-12.6, ALL runs 5.8-11.8, so no timing
    verdict is drawn. W here is the new binary with the knobs off; `cardT/out3` is the old-binary baseline.
  - **K4 (identity):** `cardR/cfg/r01full-S10-7.ron`, synchronous, `RAYON_NUM_THREADS=1`: `cmp`-identical to
    `cardT/out/T1/r01full-S10-7-r1/traj.txt` (md5 d4f77b567b912b5a4caa2af6085a35b1). Did not fire.

| Run | Tracked | ATE SE3 | Scale | VI solves over 7 it. | Most iterations | Slowest solve | base solves / iterations |
|---|---|---|---|---|---|---|---|
| R_03 W r1 / r2 / r3 | 765 / 772 / 770 | 0.111 / 0.112 / 0.108 m | 1.032 / 1.031 / 1.033 | 8 / 4 / 7 | 23 / 16 / 17 | 702 / 326 / 291 ms | 8 / 127, 4 / 54, 7 / 102 |
| R_03 V1 r1 / r2 / r3 | 776 / 788 / 781 | 0.103 / 0.114 / 0.103 m | 1.033 / 1.037 / 1.032 | 0 / 0 / 0 | 7 / 7 / 7 | 328 / 343 / 137 ms | 6 / 42, 4 / 24, 3 / 19 |
| R_03 ALL r1 / r2 / r3 | 786 / 782 / 783 | 0.111 / 0.101 / 0.115 m | 1.035 / 1.030 / 1.037 | 0 / 0 / 0 | 7 / 7 / 7 | 144 / 150 / 130 ms | 5 / 32, 6 / 39, 5 / 33 |
| R_11 ALL r1 / r2 / r3 | 798 / 800 / 800 | 0.190 / 0.149 / 0.183 m | 0.936 / 0.958 / 0.939 | 0 / 0 / 0 | 7 / 7 / 7 | 90 / 89 / 100 ms | 2 / 14, 1 / 7, 0 / 0 |
| R_01 full ALL r1 / r2 / r3 | 2898 ×3 | 0.322 / 0.384 / 0.332 m | 0.970 / 0.981 / 0.972 | 0 / 0 / 0 | 7 / 7 / 7 | 279 / 284 / 348 ms | 0 / 0 ×3 |

  VIBA1 and VIBA2 were accepted in every run. `reseeds=0` and `ba_propagations_stopped_at_bridge=0` in every
  run of the batch (24 runs).

| P/K | Prediction | Measured | Verdict |
|---|---|---|---|
| P1 | max iterations 7 on R_03, R_11, R_01; VI solves over 7 iterations 0 on R_03 and R_11 | 7 and 0 in every V1 and ALL run (W on this binary: 16-23 and 4-8 on R_03) | as predicted |
| P2 | R_03 slowest solve at most 300 ms; R_11 at most 150 ms | R_03 V1 328 / 343 / 137 ms, ALL 130-150 ms; R_11 ALL 89-100 ms; load p50 5.8-12.6 | not scored (load p50 > 5) |
| P3 | VIBA1 and VIBA2 accepted as in W; R_11 tracked 798-800 | accepted in every run; R_11 ALL 798 / 800 / 800 | as predicted |
| P4 | R_03 ATE at most 0.12 m, tracked at least 762, scale 1.02-1.04 | 0.103-0.114 m, 776-788, 1.032-1.037 | as predicted |
| P5 | `ba_propagations_stopped_at_bridge` = 0 in every run | 0 in all 24 runs | as predicted (no exposure; the unit test `a_propagating_merge_stops_at_a_bridged_segment_start` is the proof) |
| K1 | P1 holds, and R_03 ATE above 0.13 m or tracked under 762 in 2/3, or R_11 tracked under 798, or a VIBA2 rejected | none | did not fire |
| K2 | V1 alone: R_03 tracked 790+ in 3/3 or ATE under 0.08 m in 3/3 | 776-788; 0.103-0.114 m | did not fire |
| K3 | `ba_propagations_stopped_at_bridge` above 0 in a scoring run | 0 | did not fire |
| K4 | identity run not `cmp`-identical | identical | did not fire |

  - **Matrix rows for ALL:** R_11 ATE 0.136-0.170 m predicted, measured 0.190 / 0.149 / 0.183 m (2/3 outside);
    R_01 full at most 0.40 m, no run above 0.45 m, scale 0.95-1.05, tracked 2898: as predicted. The R_01
    contingency (U alone) was run anyway under card U's run list; see card U (K4 fired for U alone).
  - **Reading.** E-1 does what the theory says and no more: every VI-BA solve is capped at 7 iterations and
    the start-up and accuracy are unchanged within the W band. The bound on solve time is not verified: the
    batch ran at load p50 above 5.
  - **Trajectory fixes (verification step 8, T-1):** one R_03 run with `--trajectory-include-coasted
    --session-out` (`cardV/t1/on`): `poses_written=800`, 0 frames without a dumped keyframe, `offline-ba`
    wrote 800 rows. Flag off (`cardV/t1/off`): tracked 785, session and `offline-ba` 785 rows, 0 frames without
    a dumped keyframe.
