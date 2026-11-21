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
