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

## Result log
