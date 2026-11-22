# Card AG: drift budget and loop closure on the LaMAria challenge sequences

Plan: planner card 2026-11-22, owner answers 2026-11-22 (stage 0 during the H0 gate; observe-only loop arm; AH next if M1 holds on round 2).

```
## Experiment: AG, drift budget and loop closure on the challenge sequences
Gap tested: G10 (A2/A3/A7), G18, G35     Base: kornia-slam@feat/loss-one-world b8e2c11
Binary tag: stage 0/1 none (python, docker); stage 2 cardAG/bin/kornia-slam-ag (md5 at build)
Mechanism: LoopClosingMode defaults to Disabled (system/config/mod.rs:667) and no gate config sets it.
  Whether loops could help is unmeasured. The error is unsplit: 1_19 H0 is one segment over 5-917 s
  with Sim3 RMS 37.1 m (segsim3.py).
Theory: a loop corrects only the path between its two keyframes. On an open path the enclosed share is 0,
  so CP gain is bounded by the share of control points seen inside loop intervals. The rest is VIO drift,
  to be split into events and smooth drift. Reference: ORB-SLAM3 LoopClosing.cc:1164-1183
  (4-DoF essential graph, scale fixed once inertial), :240 (roll/pitch < 0.008 rad, yaw < 0.349 rad).
Knob: stage 2: pipeline.loop_closing: Enabled(vocabulary, graph, execution, correction: Observe).
  New field `correction`, default Apply = today's Enabled behaviour. loop_closing default Disabled is unchanged.
Baselines: /home/edgar/dev/split/.pr180-fix/gateH0/out/{BASE,H0}-<seq>-r{1,2}
  (binary cardAC/bin/kornia-slam-ac2, paced, load 3-6).
  CP r1: 1_19 14.6 / 8.8; 1_20 11.6 / 7.1; 5_11 2.3 / 11.2; 2_11 0.6 / Sim3 fail; 4_11 3.9 / fail; R_11 fail / fail.
Predictions:
  P1 revisits.py, X 3 m, view 30 deg, L 50 m: enclosed path 0 % on R_11, 1_19, 2_12, 3_17, 3_18, 4_10, 4_11;
     10-13 % on 1_20; 20-26 % on 2_11.
  P2 driftbudget.py, 1_19: piece yaw within 6 deg up to 420 s; one step of +28 to +34 deg (H0 r1) and
     +13 to +19 deg (BASE r1) inside 420-480 s; same step on the session's final keyframes.
  P3 oracle CP on 1_19 H0 r1 (arithmetic from piece RMS and the scoring table, capped by GT-as-estimate 69.9):
     O1 segments linked 8-12 (baseline 8.8); O3 split at the event 25-45; O4 60 s pieces 50-70.
  P4 global-scale oracle O5 moves CP by under 3 on 1_19 and 1_20 (uniform 0.86-0.91 is absorbed).
  P5 ORB-SLAM3 mono-inertial, 2 new runs each: 1_19 CP 30-45 with piece yaw within 3 deg; 1_20 15-45;
     2_11 5-25 (leaderboard group mean 10.9).
  Mechanism counter M1: every event window (yaw step > 3 sigma of the piece fit, or piece scale outside
     0.8-1.2) has a bias step z-score |db| / (sigma_walk * sqrt(dt)) > 10 over 5 s; fewer than 10 % of
     non-event 60 s windows do. sigma_walk from imu0 in pinhole_calibrations/<seq>.json.
  Stage 2 (Observe), one paced run each:
  M2 1_19 control: accepted loops with keyframe path gap >= 50 m = 0.
  M3 2_11: GT has 3-4 revisit episodes; accepted in 1-3 of them. 1_20: 3 episodes; accepted 0-2.
  P6 every accepted loop is GT-true (GT distance of its two keyframes <= 3 m + 3 * 0.10 m); its measured
     relative pose is within 0.3 m and 3 deg of the GT relative pose.
  P7 loop correction on 2_11 (587 to 787 s): 5-30 m (120 s piece RMS median 5.6 m, p90 10-24 m).
  P8 CP at stake = share of control points observed inside loop intervals: <= 26 % on 2_11, <= 13 % on 1_20, 0 elsewhere.
Kill rules:
  K0 enclosed share under 10 % on a sequence: no loop run on it. Under 10 % on 7 or more of 9: no Apply arm in AG;
     G10's stake becomes P8's bound.
  K1 O4 (60 s pieces) under 50 on 1_19: the loss is not drift at 60 s or longer. Stop; re-check scorer and lever arm.
  K2 component order differs between BASE and H0, or r1 and r2, on 1_19 or 1_20: the budget is noise; state so, no ranking.
  K3 wrong loop: any accepted loop fails P6's GT test: knob stays off; next is a chi-square gate on the loop residual
     against the chained edge covariance (full-review E-12), not the sim-fitted DriftModel.
  K4 mechanism moved, score did not: true loops accepted but P8's bound is under 5 CP: loop closure parked until AH lands.
  K5 score moved, mechanism did not: tracked frames or give-ups differ from the BASE band with 0 loops accepted:
     BoW cost on the tracking thread changed pacing. Report [loop-time]; the sample is invalid.
  K6 M1 association under 70 % on the remaining sequences: bias excursion is not the event marker; AH needs another counter.
Bit-identity: loop_closing Disabled: R_01 knobs-off trajectory md5 d4f77b56. correction: Observe against Disabled with
  --no-pace on R_01 800 frames: cmp-identical traj.txt.
Noise: stage 0 on every finished gate run (2 arms x 2 rounds). CP band per sequence = min-max of those 4.
  R_11_5cp CP spreads 31-55 on identical behaviour, so it gates nothing. ORB-SLAM3: 2 new runs plus the 2026-11-10 run.
```

## Planner preview (read-only, 2026-11-22, one run per arm, gate machine at load 6; stage 0 must reproduce it)

**Revisits from pseudo-dense GT** (same place = within X m and within 30 deg of view, after at least L m of path):

| seq | path m | start-end m | longest loop, X 5 m | loop-enclosed path, X 3 m, L 50 m |
|---|---|---|---|---|
| R_11_5cp | 475 | 262 | 37 m | 0 % |
| 1_19 | 1004 | 304 | 30 m (48 m any view) | 0 % |
| 1_20 | 1145 | 188 | 114 m (t 718 to 841 s) | 11.9 % |
| 2_11 | 1298 | 390 | 247 m (t 587 to 787 s) | 24.6 % |
| 2_12 | 2076 | 699 | 85 m | 0 % |
| 3_17 | 1996 | 759 | 41 m | 0 % |
| 3_18 | 1699 | 695 | 47 m | 0 % |
| 4_10 | 1450 | 301 | 45 m | 0 % |
| 4_11 | 1100 | 626 | 39 m | 0 % |

- The challenge sequences are open paths. Only 2_11 and 1_20 hold a loop longer than 50 m of path.
- **G10 ("revisit sequences only, not R_11, 1_19") is confirmed and narrowed**: 7 of 9 GT sequences have no loop a closer can use. 5_11 and 5_12 have no GT, so they are unknown.

**Where the 1_19 error is** (60 s Sim3 pieces against GT):

- H0 r1: heading stays within 6 deg for 0-420 s, then steps to +31 deg by 480 s and stays there to the end. BASE r1 shows the same event at +16 to +18 deg.
- Inside 440-470 s our path is 1.6-2.5x the GT path.
- Piece RMS is 0.1-0.5 m everywhere else. The 37 m global RMS is one event, not smooth drift.
- The final keyframe poses in `session.cbor` show the same step (36.8 m, +31 deg). The error is in the map, not in the causal output, so `--trajectory-smoothed` alone will not fix it.
- The global scale of 1.196 is an artefact of that event. Local piece scale is a steady 0.86-0.91 on 1_19 and 1_20 (speed ratio ours/GT 1.11-1.17), which the evaluator's Sim3 absorbs.
- 1_20 H0 has the same pattern: yaw steps at about 240-360 s (-15 deg) and 660 s (-10 deg), and a scale excursion of 1.44-1.55 at 420-540 s.

**Bias state through the events** (H0 `session.cbor` keyframes):

- 1_19, 436-452 s: observations fall to 26-30, a lost keyframe is inserted at 437.4 s, speed drops to 0.13 m/s (GT 1.57) and then runs at 2.5-3.1 m/s (GT 1.7).
- |bias_gyro| climbs from 0.005 to 0.037 rad/s in 8 s. The dataset's own gyro walk (2.44e-4 rad/s/sqrt(s)) allows 6.9e-4, so this is about 40 sigma.
- |bias_accel| is 0.5-1.6 m/s^2. The accel walk allows 0.02 m/s^2 in 10 s.
- All 3 flagged events on 1_19 and 1_20, and the known R_11 loss at 370 s, sit in 60 s windows with |bg| max at least 0.035 rad/s or |ba| median at least 0.45 m/s^2. No other window reaches either.

**An ORB-SLAM3 reference already exists for 1_19**: `/home/edgar/dev/.flux-connect-scratch/bench/results/lamaria/sequence_1_19/mono` (2026-11-10, fixed `lamaria_cfg.py`, one run, load 18-38).

- CP 38.9, R@5m 99.2 %, global Sim3 RMS 2.85 m, scale 0.923.
- Heading within +/-1 deg over all 900 s, with no loop available.
- Ours on the same sequence: CP 8.8 (H0) and 14.6 (BASE), RMS 37.2 and 21.2 m.
- R_11_5cp in the same folder: CP 72.9, RMS 0.97 m. 4_11 failed 3 attempts.
- Caution on the target: ORB-SLAM3 mono-inertial scores 10.0 mean on the leaderboard (`bench/lamaria-published.md`). It is a floor we are under, not the top-3 bar.

**Consequence.** Loop closure cannot reach the loss on 7 of 9 sequences. The loss is yaw and scale events while tracking, inside one segment. Recommended shape: stage 0 and the reference now, the loop work reduced to an observe-only arm on 2_11, 1_20 and a 1_19 control, and the next lever is card AH (section 5).
