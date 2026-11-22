# Card AC-2b: pose innovation gate

Status: 2026-11-22. Card written; stage 0a (jumps.py) next. Plan: planner (Fable), owner answers 2026-11-22. Worktree `/home/edgar/dev/split/kornia-slam-loss`, branch
`feat/loss-one-world`, base 530f2be.

Paths: `P=/home/edgar/dev/split/.pr180-fix`, `O=$P/cardAC2b`.

Owner answers (2026-11-22):
1. Tracker innovation gate only; VI-BA side measured via the `[ba_vel]` log; a write-back refusal only if K1 fires.
2. `frame_velocity: PropagatedFromKeyframe` (U) stays off in the promoted stack; add the R_11 frames 0-7600 BASE-vs-U DEBUG pair to stage 0, read with biaswalk.py.
3. 6-dof NIS, log both parts, stop under K7 if only rotation separates.
4. If K5 passes only with the per-sequence offset: wait for card AA stage 3b.
5. H0 is the base if it passes its gate, else BASE; pair the AC-2b gate with a knob-off run of the same binary.
6. 200-frame median window and floor c >= 1 accepted.
7. Correction note on K4 wording allowed (no re-wording of K4).
8. cargo builds/tests may run now at -j2 with --test-threads=2 during the H0 gate; every SLAM run waits for gateH0 `ALL DONE`.
9. Six samples per arm on W211c.

## Findings from the existing card AC outputs (planner, numpy over `$P/cardAC/out`)

1. **The "4.4-7.6 m/s at loss onset" figure is not the onset.** In `W211aP-U-r1/score.txt` those are the 2-3 s and 3-5 s coast bins (4.40 and 7.63 m/s against GT 1.28 and 1.11). The 0-0.5 s bins are 1.43 against 1.02 m/s (W211aP-U) and 1.28 against 0.68 m/s (W211c-KIU).
2. **Each runaway coast follows single-frame pose jumps, then its speed ramps.** From `traj.txt` finite differences (coast start not cross-checked against status lines):
   - W211c-KIU: steps of 5.98 m and 6.36 m in 50 ms at 29.30 and 29.45 s, then speed 2.3 to 7.2 m/s over 3.5 s (about +1.1 m/s²).
   - W211aP-U: 11.39 m with a 47° turn at 166.75 s, 11.26 m back at 166.95 s, 3.93 m and 3.41 m at 170.6 and 171.1 s, then 4.8 to 12 m/s in 3 s (about +2.4 m/s²).
   - So the coast inherits a wrong velocity and an acceleration error (tilt or bias). A velocity-only fix at onset would leave most of the 20 m.
3. **The jumps are accepted tracked poses, not BA shifts** (six `*-BASEd-r1` DEBUG logs, 12080 `[predgap]` lines).
   - BA newest-keyframe shift is at most 0.95 m (p99 0.19-0.56 m).
   - Accepted poses sit 1.0-3.1 m and up to 86° from the trusted IMU prediction with 16-47 inliers.
   - Raw translation gap at 100+ inliers: p50 0.021 m, p99 0.24 m, p99.9 0.61 m. So raw metres do not separate by 10x; K5 must be measured on the normalised statistic.
   - Interior IMU-edge chi2 at clock offset 0: p50 9.6, p90 1590, 34.6 % above χ²₉(99.9 %) = 27.88 (W211c-BASEd). An edge-chi2 test on VI-BA write-back has no separation without the clock fix.

## Experiment: AC-2b, pose innovation gate
Gap tested: G39, G40, G48 (tracked pose accepted against the IMU)
Base: kornia-slam@530f2be (feat/loss-one-world)   Binary tag: kornia-slam-ac2b (md5 at build)
Mechanism: a tracked pose is accepted on inliers alone, 15 with a trusted IMU (tracking/tracker.rs:773-787);
  pose_gate is Disabled (tracking/pose_gate.rs:18-22). Poses 1.8-11.4 m and up to 47 deg from the IMU
  prediction in one 50 ms frame are accepted with 16-47 inliers. They become keyframes; VI-BA, with Huber on
  every IMU edge (mapping/vi_ba_schur.rs:386), explains the jump with velocity, bias and tilt; the tracker
  re-reads them every frame (system/mod.rs:1817-1849, :1098); the coast integrates them (+1.1 to +2.4 m/s^2).
Theory: the IMU-propagated covariance says how far a pose may sit from the prediction. A pose outside it at
  99.9 % is refused and the frame coasts on the IMU, so it never enters a keyframe. The covariance grows through
  a coast, so recovery poses pass without a speed cap. OpenVINS UpdaterMSCKF.cpp:215-229 (gate),
  ORB-SLAM3 Tracking.cc:1981-1998 (coast the frame). Beats VINS-Fusion estimator.cpp:955-1002 (hand-set, dead
  code) and ORB-SLAM3 LocalMapping.cc:136-145 (a reset).
Knob: tracking_loss.pose_gate: Innovation(( mode: Observe | Reject )); default Disabled = today.
  Derived: chi2_6 at 99.9 % = 22.458 times c, c = max(1, running median NIS / 5.348). Flagged constants, not
  tuned on score: 99.9 % (owner), median window 200 frames (median noise about 5 %), floor c >= 1.
Arms: H0 = keep_inertial + seed: Held (the gate's arm); H = H0 + frame_velocity: PropagatedFromKeyframe (K4
  reproduction only); G = Innovation Reject; OBS = Innovation Observe.
Baselines ($P/cardAC/out, binary ac/ac2, pairs started at load <= 4):
  run            coast err 3-5 s   speed err 3-5 s   kf > 4 m/s   steps > 1 m per frame
  W211c-KIU      20.20 m           5.31 m/s          12           9
  W211c-H        3.83 m            0.77              41           106
  W211c-H-m50    20.64 m           5.49              8            11
  W211c-H-p50    2.40 m            -                 39           80
  W211c-U        0.87 m            0.19              0            10
  W211aP-U       21.32 m           6.53              20           19
  W511-H         no GT             no GT             3            1     (tracked 1023-1035, give-ups 0/0/1)
  BK-H 25:6      1.71 m            -                 0            0     (restart 1.59 m, 4.1 deg)
  Full, H: steps > 1 m: R_11 39, 1_19 56, 2_11 276, 5_11 2. H0 and BASE full-sequence baselines: $P/gateH0/out,
  2 rounds; fill the table from its score.txt before step 7.
  Raw gap, 6 BASEd logs: 100+ inliers trans p50 0.021, p99 0.24, p99.9 0.61 m; rot p50 0.15, p99 0.68 deg.
Predictions, stage 0 (Observe):
  M0  a statistic exists on >= 90 % of trusted tracked frames (pose_innovation_judged against tracked).
  K5m healthy frames' NIS/c p99.9 is under 1/10 of the event frames' p10.      measured by: innov.py
  P0a would-reject share on R_01 full, R_03, R_11 window: <= 1 % of judged frames.
  P0b every primary loss with a coast error over 5 m at 3-5 s has >= 1 would-reject in the 3 s before onset.
Predictions, stage 1 (Reject):
  Mechanism counter: pose_innovation_rejects >= 1 in the 3 s before each primary loss on W211c and W211aP;
      accepted steps over 1 m on judged frames: 0.
  P1  coast error at 3-5 s: W211c H+G <= 2.5 m in 6 of 6 samples (H: over 5 m in 2 of 6 U-family samples);
      W211aP H+G <= 3 m (21.32). Speed error at 3-5 s <= 1.0 m/s (5.3-6.5).
  P2  keyframes over 4 m/s: W211c H+G <= 1 (8-41); W211aP <= 2 (20).
  P3  BK 25:6 H0+G: restart 1.3-1.9 m and <= 5 deg, coast 3-5 s <= 2.2 m, rejects <= 1 % (the gate must not act).
  P4  W511 H0+G: tracked >= 1013, primary give-ups <= 1, later_alignments 0.
  P5  R_11_5cp full, H0+G against H0, 2 pairs: CP and tracked inside the paired H0 range; reseeds no higher.
  P6  full sequences (card AC, unchanged): keyframes over 4 m/s <= 10 % of baseline; coast_runaway_frames
      <= 10 %; 3_18 at 1212 s coast error 10 m -> <= 1.5 m and the next 30 s at scale 0.9-1.1; CP 1_19 39-49,
      3_18 32-45. One sample each is provisional; the regression gate decides promotion.
Kill rules:
  K1 (mechanism moved, score did not) rejects fire and accepted steps over 1 m are 0, but coast error at
     3-5 s stays over 5 m on >= 2 event samples: the growth has another source. Read [ba_vel] and jumps.py
     (tilt against bias), then score the zero-code arms lost_keyframes: FixedInertial and
     imu_huber: BoundaryOrRecentInit.
  K2 (score moved, mechanism did not) P1 or CP improves with 0 rejects before the event, or M0 misses:
     luck or a different trajectory. No credit.
  K3 (harm) identity fails; a guard out of band in 2 of 3; tracked below or give-ups above the paired base;
     rejects over 5 % of judged frames on a guard; a reject streak of 10 frames ending in a give-up the base
     does not have (lock-in): the knob stays off.
  K5 (card AC, as worded) healthy-frame NIS p99.9 within 10x of the event frames' p10: no Reject runs; score
     the HB arm instead.
  K6 K5 passes only with the per-sequence clock offset: the gate depends on card AA stage 3b; stop.
  K7 6-dof fails K5 but the rotation part passes: stop and ask; do not switch the statistic silently.
Bit-identity: r01full-S10-7, RAYON_NUM_THREADS=1, traj md5 d4f77b567b912b5a4caa2af6085a35b1 for knob off,
  H0 knob off, and Innovation Observe.
Noise: paced async runs do not repeat (W211c 13 against 17 give-ups). 6 samples per arm on W211c, 3 on
  W211aP and W511, imu_time_offset_ns 0, +/-50000, +/-100000, +150000. Pairs start at load <= 4 with no other
  kornia-slam process. A one-sample verdict is provisional.
If it misses: M0 -> the keyframe chain in predicted_covariance breaks (count _unavailable by cause);
  K5m -> per-part NIS against inliers and angular rate (clock term w x 4.4 ms); P1 -> K1's path.
Measurement, no prediction (owner answer 2): R_11 frames 0-7600, BASE against U, DEBUG=1, one pair, read with
  biaswalk.py: where U inflates |ba| before any loss.

## Results

### Stage 0a, jumps.py on card AC outputs (offline, 2026-11-22)

`python3 -I experiments/quick-eval/jumps.py $P/cardAC/out/<run> .../sequence_2_11 --body $P/cardAC/body/sequence_2_11.txt`.
A primary loss here is a coast episode of at least 0.5 s in a segment at least 3 s old; times from the window start.

- W211c-KIU, loss 7 (onset frame 21389, 29.45 s, coast 5.05 s): steps 5.98 m / 6.4 deg at 29.30 s (T) and 6.36 m /
  3.9 deg at 29.45 s (the first coasted frame), plus 1.79 m and 2.08 m at 26.25 and 26.50 s. SLAM speed 3.13 ->
  6.87 m/s over the coast, GT 0.50 m/s in the first 0.5 s; speed error first 0.5 s 3.94 m/s. Reproduces finding 2.
- W211aP-U, loss 6 (onset frame 3444, 172.20 s, coast 5.05 s): steps 3.93 m at 170.75 s and 3.41 m at 171.25 s
  inside the 5 s window, plus 16 more steps of 0.5-2.1 m. The 11.39 m / 47.2 deg and 11.26 m / 41.9 deg steps sit at
  166.90 and 167.10 s, 5.30 and 5.10 s before the onset, just outside the 5 s window (a 6 s window shows them). The
  planner's times read 0.15 s earlier (166.75, 170.6, 171.1 s): a different time origin, same frames. SLAM speed
  11.21 -> 12.13 m/s against GT 1.15 m/s; slope of the speed error +1.64 m/s^2 over 3.6 s.
- Slope on W211c-KIU loss 7 reads +0.06 m/s^2, not +1.1: the 0.25 s speed window starts on the 6.36 m step, so the
  first samples already read 3-4 m/s; the start-to-end SLAM speeds (3.1 -> 6.9 m/s) carry the ramp.

