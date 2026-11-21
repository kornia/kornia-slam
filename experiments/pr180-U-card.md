# Card U: the frame velocity is reset to the keyframe's every frame, so the IMU prediction ages

Status: written 2026-11-21, before any code. It is card T's K4 follow-up (R_03 tracking).

## Experiment: U, propagate the reference keyframe's velocity to the previous frame

- **Gap tested:** card T K4: R_03 tracks 762-789 of 800 with `pr180-window10`, against 782-800 for
  `pr180-frontend-1ms`. It also bears on G13 (no per-frame velocity state) and G29 (PnP snapping back
  after BA).
- **Base:** kornia-slam `feat/global-pose-graph` @ 67eb8e0 (code as c3d19d0). Binary
  `cardT/bin/kornia-slam-t-seq`, md5 32b6d5506ebeaa18fba296abac1e8a4d.
- **Baselines:** `cardT/out3/{r03-W,r11-W,r01full-W}-r{1,2,3}`, preset `configs/pr180-window10.ron`,
  2 jobs, load p50 4.2.

### Evidence (2026-11-21)

- **R_03 gap 2** (card T, K4 follow-up). After the turn at frames 371-387 is bridged, the prediction
  error grows 1.6, 2.1, 3.3, 5.4, 7.6 and 9.5 cm over frames 389-394. No keyframe is inserted in
  those frames: tracking has 231-320 matches. Frame 395 then fails PnP, and 395-401 are lost.
- **The error grows with the frames since the last keyframe.** `[predgap] trans`, median, on tracked
  frames only, binned by frames since the last `[kf]` line:

| Run | 1 frame | 2 | 3-4 | 5 or more |
|---|---|---|---|---|
| r03-W r1 / r2 / r3 | 0.86 / 0.74 / 0.77 cm | 1.21 / 1.17 / 1.27 cm | 1.75 / 1.85 / 1.82 cm | 5.67 / 5.82 / 4.93 cm |
| r11-W r1 | 1.08 cm | 1.62 cm | 2.17 cm | 3.08 cm |
| r01full-W r1 | 1.49 cm | 1.21 cm | 1.54 cm | 2.01 cm |

  A longer run without a keyframe means tracking is going well, so the error should not grow if the
  velocity were right.

### Mechanism (file:line at 67eb8e0)

- With `tracking.pose_inertial: Off` (every pr180 preset), `frame_state_carried` is false
  (`system/estimator.rs:449-452`). So every frame calls `refresh_reference_inertial_state`
  (`system/mod.rs:1772-1774`).
- When the reference is the last keyframe, that call sets `state.velocity_world = kf.velocity_world`
  (`system/mod.rs:1704-1706`). That is the velocity at the keyframe's timestamp t_kf.
- `prediction_origin` then returns `(pose_before, prev_timestamp)` (`system/estimator.rs:509-511`), and
  `predict_frame_pose` integrates only [t_{k-1}, t_k] from that velocity (`system/mod.rs:2389-2415`).
- The frame's velocity therefore lacks g·(t_{k-1} − t_kf) + R_kf·Δv(t_kf → t_{k-1}): all the acceleration
  measured since the keyframe. The position error of the prediction grows with that interval.
- The same stale velocity seeds each new keyframe (`system/mod.rs:2936`), which is card R's (b1). Card
  R's `keyframe_seed: ImuPropagated` fixed the seed only, and alone it made results worse (card R
  ablation). The per-frame prediction was never fixed.
- The `OrbSlam3` and `LastKeyframe` modes do not have this bug: `prediction_origin` integrates from
  the keyframe's own timestamp (`system/estimator.rs:512-525`; test
  `a_keyframe_origin_prediction_starts_from_the_keyframe_velocity`, `system/tests.rs:772`).

### Theory and reference

- A pose predicted from the previous frame needs the velocity at the previous frame. ORB-SLAM3
  `Tracking::PredictStateIMU` either predicts from the last keyframe with the preintegration since
  that keyframe (`mpImuPreintegratedFromLastKF`, when the map was updated), or from the last frame with
  the last frame's own velocity (`mpImuPreintegratedFrame`). It never mixes a keyframe velocity with
  a one-frame integral. Basalt and VINS-Fusion carry a velocity state per frame in the window, so the
  question does not arise. OpenVINS propagates its single state through every IMU sample.
- ORB-SLAM3's choice suits us best: it keeps the BA-refined keyframe velocity (Basalt and VINS need a
  per-frame state we do not have), and it is a propagation, not a new solve.
- Fix: refresh the velocity as v_kf + g·Δt + R_kf·Δv over [t_kf, t_{k-1}], with the keyframe's bias.
  The pose stays the tracked `pose_before`. Card R's `keyframe_seed: ImuPropagated` then becomes
  redundant for tracked keyframes, because the frame velocity is right when a keyframe is cut.

### Knob

`tracking.frame_velocity: KeyframeVelocity | PropagatedFromKeyframe`. The default, `KeyframeVelocity`,
is today's behaviour. It is read only with `pose_inertial: Off`.

### Instrumentation (debug only, no knob)

`[vel] frame=<k> ref_kf=<idx> ref_lost=<0|1> age_s=<t_{k-1} − t_kf> v=(x,y,z)`. A script
(`experiments/quick-eval/velgt.py`) compares `v` against ground truth: the central difference of the
pseudo-dense positions, rotated by the SE3 alignment from `ate.py`.

### Configs

| Name | Config |
|---|---|
| U0 | `pr180-window10` (knob off, instrumented binary) |
| U1 | U0 plus `tracking: ( frame_velocity: PropagatedFromKeyframe )` |

Runs: R_03 and R_11 800-frame windows, and full R_01, 3 each, 2 jobs.

### Predictions

- **P1 (mechanism, U0):** the velocity error against ground truth grows with `age_s`. The median error
  at 5 or more frames is at least 3× the median at 1 frame, on R_03. Over frames 389-394 of R_03 it is
  at least 0.3 m/s.
- **P2 (mechanism, U1):** the per-age `[predgap] trans` medians flatten. The 5-or-more bin falls from
  4.9-5.8 cm to at most 2.0 cm on R_03, and to at most 1.5× the 1-frame bin. The velocity error at
  5 or more frames is at most 1.5× the 1-frame error.
- **P3 (tracking, U1):** R_03 tracked 790-800 in 3/3 (U0: 762-789), and the 395-401 gap is gone in every
  run that has the 371-387 gap.
- **P4 (accuracy, U1):** R_03 ATE SE3 at most 0.11 m (U0 0.085-0.112). R_11 0.136-0.170 m. Full R_01 at
  most 0.40 m (U0 0.296-0.399). Scale 0.95-1.05.
- **P5 (no cost):** frame p50 within 1 ms of U0. The extra preintegration over at most a few keyframe
  intervals is microseconds.

### Kill rules

- **K1:** P1 fails (no growth with age). The velocity error is not the age. Look at the coasted
  velocity of the lost keyframes (`local_ba.lost_keyframes`) instead.
- **K2:** P2 holds but P3 fails. The prediction was not what lost frame 395. Look at map projection
  after the turn (card F's wider field of view).
- **K3:** P3 holds but P2 fails. Luck or another cause. Do not ship.
- **K4:** R_01 ATE above 0.45 m in any run, or scale outside 0.95-1.05. The propagated velocity hurts
  BA seeding. Check the keyframe velocity against card R's crossing r_v.

### Bit-identity and noise

- Knob off: with S10-7 (sync), the U0 binary's trajectory must be `cmp`-identical to
  `cardT/out/T1/r01full-S10-7-r1/traj.txt`. The debug line only adds stderr output.
- Unit test: a constant-acceleration walk, 6 frames after a keyframe. The `PropagatedFromKeyframe`
  frame velocity equals v_kf + a·Δt to 1e-9, and `KeyframeVelocity` keeps today's value.
- Noise: async runs differ run to run. All verdicts are on 3 runs, and the per-age medians are per run.

### Stage 0: staleness or a BA-written velocity (config only, before code)

- Competing cause (frontend review, T-2): VI-BA writes a wrong velocity into the lost keyframes (free states),
  and the tracker copies it every frame.
- Offline, cardT/out2/r03-T-r1 (synchronous): no keyframe is inserted in 388-394, the reference is lost keyframe
  387, and no solve lands in those frames, so the copied velocity is constant. A constant error gives a flat
  predgap. Measured 1.6 -> 9.5 cm. The logged step (1.7-1.9 m/s) and step_rot (1.6-2.2 rad/s) give a centripetal
  3.2-3.9 m/s^2, so staleness alone predicts 0.9, 1.8, 2.7, 3.6, 4.5, 5.4 cm. About 40 % of the slope is unexplained.
- S0a: r03-T (synchronous S10-7) and r03-W, each plus local_ba: ( lost_keyframes: FixedInertial ).
- S0b: r03-T plus tracking_loss: ( keyframes_while_lost: Anchor ). An upper bound only: against an anchor the
  velocity is never reset (system/mod.rs:1702-1716), so this removes staleness and the BA edge together.
- P0a (if the BA edge is the cause): under S0a, predgap/dt stays under 0.3 m/s over the 6 frames after the
  recovery, and the second gap (395-401) is tracked.
- P0b (if staleness is the cause): under S0a, predgap still grows at least 3x from the first to the sixth frame
  after the recovery, with a slope of 2.5-7 m/s^2 against the time since the reference keyframe, and it resets at
  each keyframe.
- K5: P0a holds. The fix for 395-401 is the lost-keyframe state, not this card. Drop P3; keep P1 and P2 for the
  age growth elsewhere; write a card for lost_keyframes: FixedInertial.
- K6: U1 flattens the per-age medians (P2) but the error at age 1 after a recovery stays above 0.3 m/s. Both
  causes act; the remainder is the BA edge.

Mechanism note: the copy at system/mod.rs:1704-1706 also runs for a non-anchor reference that is not the last
keyframe (a relocalization, :2857). The IMU samples before the last keyframe are pruned (:2087-2089), so the knob
falls back to KeyframeVelocity there.

## Result log

- **2026-11-21, stage 0 (config only).**
  - **Setup:** binary `cardT/bin/kornia-slam-t-seq` (md5 32b6d5506ebeaa18fba296abac1e8a4d), configs
    `cardV/cfg/{r03-T-fixed,r03-T-anchor,r03-W-fixed}.ron`, runs in `cardV/out0`. Synchronous runs with
    `RAYON_NUM_THREADS=1` as in `cardT/out2`; W runs as in `cardT/out3`. 2 jobs, load p50 4.44.
  - **Control:** `r03-T` on the same binary (`cardV/out0c/r03-T-r1`) is `cmp`-identical to
    `cardT/out2/r03-T-r1`, so the synchronous S0a run differs from the baseline by the knob alone.

| Run | Tracked | ATE SE3 | Scale | Gaps around the turn | predgap after the first recovery |
|---|---|---|---|---|---|
| r03-T (base) | 776 | 0.110 m | 1.034 | 371-387, 395-401 | 1.6, 2.1, 3.3, 5.4, 7.6, 9.5 cm (389-394) |
| S0a r03-T-fixed | 712 | 3.118 m | 0.432 | 371-412, 414-422, 425-437, 439-462 | none: 414 is lost again |
| S0a r03-W-fixed r1 / r2 / r3 | 545 / 717 / 584 | 32.9 / 1.29 / 26.9 m | 0.044 / 0.737 / 0.067 | 374-398, many after / 374-387, 392-412, ... / 349-367, 371-379, 382-414, ... | r1 9.0, 8.4, 7.8, 7.2, 5.9 cm (400-404); r2 8.9, 8.8, 8.5 cm (389-391); r3 16.8 cm then lost |
| S0b r03-T-anchor | 751 | 0.740 m | 0.950 | 371-412, 416-422 | 4.9, 5.2 cm (414-415) |

| P/K | Prediction | Measured | Verdict |
|---|---|---|---|
| P0a | under S0a, predgap/dt under 0.3 m/s over the 6 frames after the recovery, and 395-401 tracked | sync: no 6 tracked frames after the recovery, 395-401 inside the 371-412 gap. W: 1.0-6.2 m/s on the frames read after the turn-gap recoveries in the table (lowest elsewhere: r2 frame 424, after 414-422, 0.66 m/s), 395-401 lost in 3/3 | refuted |
| P0b | under S0a, predgap grows at least 3x from the first to the sixth frame after the recovery, slope 2.5-7 m/s^2, resets at each keyframe | sync: no 6-frame stretch. W r1 falls 9.0 -> 5.9 cm, r2 flat 8.9 -> 8.5 cm | refuted (not testable as stated: S0a lost the stretch it was to be read on) |
| K5 | P0a holds | P0a refuted | did not fire: card U proceeds |

  - **Reading.** S0a does not isolate the two causes: holding the lost keyframes' inertial state fixed
    makes R_03 lose track for 40-90 more frames and collapses the scale (0.04-0.74) in every run, so the
    stretch where staleness was to be read does not exist under it. Neither hypothesis gained support. The
    BA-written velocity is not shown to be the cause (K5 did not fire), so the plan goes on to U1.
  - **Side finding, not scored by any prediction:** `local_ba.lost_keyframes: FixedInertial` is harmful on
    R_03 with this preset (ATE 1.3-33 m against 0.085-0.112 m). S0b, an upper bound only, also loses the
    395-401 stretch.

- **2026-11-21, scoring (U0 = W with the new binary, U1 = U).**
  - **Setup:** `feat/global-pose-graph` @ f57c13a, binary `cardV/bin/kornia-slam-v`
    (md5 975ce3b7802abbd03f7540e53ec59604), configs `cardV/cfg/{r03,r11,r01full}-{W,U,ALL}.ron`, runs in
    `cardV/out`, `RUN_FLAGS` empty, 3 runs each, 2 jobs. Load p50 over the batch 6.77 (per run 2.2-12.6; only
    W r1/r2 under 5), so no timing verdict is drawn. Velocity error from `quick-eval/velgt.py`.
  - **Identity gate:** `cardR/cfg/r01full-S10-7.ron` with this binary is `cmp`-identical to
    `cardT/out/T1/r01full-S10-7-r1/traj.txt` (md5 d4f77b567b912b5a4caa2af6085a35b1).

| Run | Tracked | ATE SE3 | Scale | predgap 1 / 2 / 3-4 / 5+ (cm) | velocity error 1 / 5+ frames (m/s, ref_lost=0) |
|---|---|---|---|---|---|
| R_03 U0 r1 / r2 / r3 | 765 / 772 / 770 | 0.111 / 0.112 / 0.108 m | 1.032 / 1.031 / 1.033 | 0.79 / 1.24 / 1.77 / 5.46; 0.82 / 1.26 / 1.79 / 5.85; 0.83 / 1.22 / 1.75 / 5.79 | 0.086 / 0.703; 0.079 / 0.799; 0.084 / 0.801 |
| R_03 U1 r1 / r2 / r3 | 781 / 779 / 780 | 0.092 / 0.094 / 0.114 m | 1.030 / 1.026 / 1.037 | 0.54 / 0.55 / 0.53 / 0.40; 0.52 / 0.53 / 0.52 / 0.44; 0.56 / 0.61 / 0.53 / 0.43 | 0.062 / 0.066; 0.059 / 0.064; 0.071 / 0.072 |
| R_11 U1 r1 / r2 / r3 | 800 / 800 / 798 | 0.145 / 0.184 / 0.189 m | 0.961 / 0.939 / 0.937 | 0.53 / 0.39 / 0.48 / 0.40; 0.65 / 0.51 / 0.55 / 0.45; 0.57 / 0.50 / 0.55 / 0.47 | |
| R_11 U0 (`cardT/out3`, old binary) | 800 / 799 / 799 | 0.165 / 0.151 / 0.165 m | 0.949 / 0.957 / 0.949 | 1.08 / 1.62 / 2.17 / 3.08 (r1) | |
| R_01 full U1 r1 / r2 / r3 | 2898 / 2897 / 2898 | 0.399 / 0.411 / 0.498 m | 0.983 / 0.979 / 0.988 | 0.95 / 0.63 / 0.81 / 0.77 (r1) | |
| R_01 full U0 (`cardT/out3`, old binary) | 2898 ×3 | 0.296 / 0.399 / 0.311 m | 0.990 / 0.978 / 0.969 | 1.49 / 1.21 / 1.54 / 2.01 (r1) | |
| R_03 ALL (U1 + card V1) r1 / r2 / r3 | 786 / 782 / 783 | 0.111 / 0.101 / 0.115 m | 1.035 / 1.030 / 1.037 | 0.57 / 0.55 / 0.52 / 0.46 (r1) | |
| R_11 ALL r1 / r2 / r3 | 798 / 800 / 800 | 0.190 / 0.149 / 0.183 m | 0.936 / 0.958 / 0.939 | | |
| R_01 full ALL r1 / r2 / r3 | 2898 ×3 | 0.322 / 0.384 / 0.332 m | 0.970 / 0.981 / 0.972 | | |

| P/K | Prediction | Measured | Verdict |
|---|---|---|---|
| P1 | U0: velocity error at 5+ frames at least 3x the 1-frame error on R_03; at least 0.3 m/s over 389-394 | 8.2x, 10.1x, 9.5x; 0.85-1.79 m/s over 389-394 in 3/3 | as predicted |
| P2 | U1: R_03 5+ predgap bin at most 2.0 cm and at most 1.5x the 1-frame bin; velocity error at 5+ at most 1.5x the 1-frame | 0.40 / 0.44 / 0.43 cm (0.74x, 0.85x, 0.77x the 1-frame bin); velocity 1.06x, 1.08x, 1.01x | as predicted |
| P3 | U1: R_03 tracked 790-800 in 3/3, and 395-401 gone where 371-387 occurs | 781 / 779 / 780; no U1 run has a 395-401 gap (the turn gap is 371-379 to 371-382) | refuted (tracked under the band) |
| P4 | U1: R_03 ATE at most 0.11 m; R_11 0.136-0.170 m; R_01 at most 0.40 m; scale 0.95-1.05 | R_03 0.114 in r3; R_11 0.184 and 0.189, scale 0.939 and 0.937; R_01 0.411 and 0.498 | refuted on all three sequences |
| P5 | U1: frame p50 within 1 ms of U0 | R_03 U0 14.3-15.4 ms, U1 14.7-15.3 ms, but load p50 5.7-9.9 | not scored (load p50 > 5) |
| K1 | P1 fails | P1 holds | did not fire |
| K2 | P2 holds but P3 fails | P2 holds, P3 refuted | **fired** |
| K3 | P3 holds but P2 fails | | did not fire |
| K4 | R_01 ATE above 0.45 m in any run, or scale outside 0.95-1.05 | U1 r3 0.498 m | **fired** |
| K6 | P2 holds but the error at age 1 after a recovery stays above 0.3 m/s | ref_lost=1, bin 1: 0.077-0.111 m/s (U0: 0.72-1.26 m/s) | did not fire |

  - **Reading.** The mechanism is confirmed: the stale keyframe velocity is the source of the per-age
    prediction error, and propagating it removes the growth (predgap flat at about 0.5 cm on R_03 and R_11,
    velocity error flat at 0.06-0.07 m/s). Tracking on R_03 rises from 765-772 to 779-781 but not into the
    band, so the prediction was not the whole cause of the turn losses (K2: look at map projection after the
    turn). U alone hurts full R_01 (K4: 0.399-0.498 m against 0.296-0.399 m), as card R's
    `keyframe_seed: ImuPropagated` did alone; with card V1 (ALL) full R_01 is 0.322-0.384 m. Per K4, check
    the keyframe velocity against card R's crossing r_v before U ships. Do not ship U alone.

## Stage 2: card U2

Status: written 2026-11-21, before any run. From the PR #180 full review, plan `pr180-full-review-fixes.md`
(finding 2, bucket 3). Awaiting owner approval (STOP 1). The text above is unchanged.

Paths: `S` = `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad`,
`P` = `/home/edgar/dev/split/.pr180-fix`, `K` = `crates/kornia-slam/src`.

### Experiment: U2, `frame_velocity: PropagatedFromKeyframe` in the preset

- **Gap tested:** card T K4 (R_03 tracking), G13, G29.
- **Base:** kornia-slam `feat/global-pose-graph` at the gated fix tip, tag `pr180c/F1` (made in phase 5 of
  the plan; not yet cut). Binary tag `x`: `P/bin/kornia-slam-x`, built from `pr180c/F2`, md5 recorded at build.
- **Baselines:** BASE = V1, `P/baseline/{r03,r11,r01full}-V1.ron` (copies of `S/cardV/cfg`), 3 runs R_03 and
  R_11, 5 runs full R_01. Old runs: `S/cardV/out`, `S/cardT/out3`. New runs:
  `P/out/{r03,r11,r01full}-{BASE,U2}-r<n>`.

### Mechanism (file:line at ad443b7; code equals f57c13a)

- The frame velocity is reset to the keyframe's every frame (`K/system/mod.rs:1760`); fix at `:1780-1800`
  (`propagated_frame_velocity`).
- Confirmed by card U P1 and P2 (predgap 5+ bin 5.5-5.9 cm to 0.40-0.44 cm).

### Theory and reference

- ORB-SLAM3 `PredictStateIMU` (`Tracking.cc:1746-1758`), as card U. Not re-argued.
- New fact: on full R_01 the base budget never acts (`vi_ba_base_solves=0` in all 6 runs), so card U's
  "U alone" and card V's "ALL" are one configuration there. K4 fired on 1 of 6 samples.

### Knob

- `tracking.frame_velocity` (exists; default `KeyframeVelocity` = today). Class: bug fix. No parameter.

### Configs

| Name | Config |
|---|---|
| BASE | `P/baseline/{r03,r11,r01full}-V1.ron`, new binary |
| U2 | `P/baseline/{r03,r11,r01full}-ALL.ron` (V1 plus `tracking: ( frame_velocity: PropagatedFromKeyframe )`) |

Runs: 3 per config on R_03 and R_11, 5 per config on full R_01. `--no-pace --debug`, `JOBS=2`, `RUN_FLAGS`
empty, as cards T, U, V. R_01 and R_03 need `ALLOW_RAW_IMU=1` (no factory file). Load sampled every 30 s
into `<out>/load.log`.

### Precondition

- The R-C9 unit test (gravity, rotated keyframe, T_bc) passes:
  `the_frame_velocity_follows_gravity_and_the_lever_arm` (lane sys).

### Predictions

- **P1 (mechanism)** R_03 predgap 5+ bin: 5.5-5.9 cm -> at most 2.0 cm in 3/3. Measured by: `S/cardV/stats.py`.
- **P2** R_03: tracked at least 779 in 3/3 (U-on so far 779-786, n=6); ATE at most 0.115 m.
- **P3** R_11: ATE at most 0.170 m in at least 2/3 and scale at least 0.945 (U-on so far: 4 of 6 above 0.170).
- **P4** full R_01, n=5: median ATE at most BASE median + 0.03 m; no run above 0.45 m.
- **Mechanism counter:** `[vel]` error at 5+ frames at most 1.5x the 1-frame error (`velgt.py`, after R-F4).

### Kill rules

- **K1 (mechanism moved, score fell):** P1 holds, and R_11 ATE above 0.170 m in 2/3, or R_01 median worse
  than BASE by more than 0.05 m, or any run above 0.50 m. U stays out of the preset. Next: card R's crossing
  r_v check on the keyframe velocity (card U K4 follow-up).
- **K2 (score moved, mechanism did not):** R_03 tracked 790+ in 3/3 with the predgap bins unchanged. Do not
  credit U.

### Decision

- U enters `pr180-window10` only if P2, P3 and P4 hold.
- Owner answer 10 (2026-11-21): the validation run does not wait for this verdict.

### Bit-identity and noise

- Knob off: `P/baseline/r01full-S10-7.ron` (copy of `S/cardR/cfg/r01full-S10-7.ron`), synchronous,
  `RAYON_NUM_THREADS=1`, must be `cmp`-identical to `P/baseline/traj.txt` (copy of
  `S/cardT/out/T1/r01full-S10-7-r1/traj.txt`, md5 d4f77b567b912b5a4caa2af6085a35b1).
- Noise: asynchronous runs differ run to run; 3 runs per config, 5 on full R_01. Known bands: R_03 tracked
  776-788, ATE 0.103-0.114 m (V1, n=3); R_11 ATE 0.149-0.190 m (n=6, U-on); R_01 full 0.30-0.50 m
  (n=9, all configs). Pooled U-on full R_01 so far: 0.322, 0.332, 0.384, 0.399, 0.411, 0.498 m (n=6) against
  U-off 0.296, 0.311, 0.399 m (n=3, old binary).
- Load: recorded; no timing verdict above load p50 5.

### Result log (U2)

