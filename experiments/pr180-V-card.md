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
