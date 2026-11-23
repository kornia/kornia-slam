# Card AH: bias memory across the VI-BA window boundary

Plan: planner card AH, revision 2 (2026-11-23); owner answers 2026-11-23 (all 10 open questions as recommended; on K4, build a warm-start knob). Stage 0 is python only on existing runs (gate H0, cardAC W119, ORB-SLAM3 1_19). Ledger: G49, referenced from G35.

```
## Experiment: AH, bias memory across the VI-BA window boundary
Gap tested: G35, new row (bias chain has no memory across the boundary)
Base: kornia-slam@<commit that adds the Lamaria start_sec/end_sec knobs>   Binary tag: cardAH/bin/kornia-slam-win (md5 at build)
Mechanism: imu_edge_information (mapping/vi_ba_schur.rs:864-875) scales the boundary edge's bias-walk rows by
  boundary_imu_info_scale 1e-2 (defaults vi_ba_schur.rs:385, bundle_adjustment.rs:128, system/config/mod.rs:626).
  A keyframe is last written as the oldest active one, tied to the frozen past only by that edge, so every
  consecutive pair in the map sits at the loosened strength: walk z p50 10.3-12.0 gyro, 5.8 accel (3 sessions).
  Nothing else holds the gyro bias; the accel bias has a zero-mean 0.1 m/s^2 prior (vi_ba_schur.rs:396, :2329-2348),
  overrun to 1.2-1.45 m/s^2. On 1_19 the accel bias leaves at 330 s, hides a 6-14 deg tilt for 100 s, and the
  weak spell at 436-442 s (26-110 observations, 160-186 deg/s) then gives 3-5 m/s and a 27-35 deg yaw step.
Theory: the bias is a random walk of 2.444e-4 rad/s/sqrt(s) and 6.448e-3 m/s^2/sqrt(s) (imu0 in the sequence
  calibration). At dt 0.15 s one edge allows 9.5e-5 rad/s and 2.5e-3 m/s^2. The scale allows 10x: up to
  6.3e-3 rad/s and 0.17 m/s^2 per second one-sided; observed slews are 4.0e-3 and 0.046. At full weight the
  one-sigma bound is 6.3e-4 and 0.017 per second, so a tilt must be corrected in the attitude instead of hidden
  in the accel bias, and the gyro holds heading through a weak spell.
  Reference taken: ORB-SLAM3 Optimizer.cc:2646-2657 (EdgeGyroRW, EdgeAccRW at full information on every edge,
  the boundary included); :2632-2643 scales EdgeInertial only, Huber only on the boundary edge or while bRecInit.
  It holds 1_19 heading within 1 deg on the same input. Same property in Basalt imu_block.hpp:73-92 and OpenVINS
  Propagator.cpp:227-228. ORB-SLAM3's form is taken because our boundary is its fixed keyframe; Basalt and VINS
  pair the walk with a marginal prior we do not use by default.
Knob: A1: mapping.local_ba.boundary_scales_bias_rows (exists; default true = today) -> false.
      A2 = A1 + mapping.local_ba.imu_huber (exists; default AllEdges) -> BoundaryOrRecentInit
              + source imu_time_offset_ns (exists; default 0) -> -(TimeOffsetSec_Device_Gyro x 1e9) + 500000
                from the sequence's factory record (1_19: -3583130). Sensor-model constants, none tuned.
Arms: A0 = gate H0 config + window; A1; A2. Unpaced (--no-pace), asynchronous mapping as in the gate.
Windows (start_sec:end_sec): E19 = 1_19 340:500 or 300:470 (the probe picks); Q19 = 1_19 60:220 (quiet guard);
  E20 = 1_20 380:540; R11 = R_11_5cp 280:460 (counters only). Each run at most 300 s wall.
Baselines: cut from gateH0/out full runs (binary ac2, paced): section 0 table. Per-window A0 baselines come from
  the A0 arm of each pair (3 repeats); fill them into this card after the probe, before stage 1a.
Predictions, stage 0 (existing runs, no SLAM):
  S1 walk z p50 on every H0 session: gyro 8-13, accel 4-7.                                by: biasevent.py
  S2 on 1_19 r1 and r2, 1_20 r1 and r2: |ba30 - ref| passes 3 sigma_a sqrt(dt) at least 10 s before each
     within-segment event starts (1_19: by 80-110 s).
  S3 bias-carried heading: the integral of (b_est - b_true) about gravity over the event is 50-150 % of the
     yaw step on 1_19 r1 and r2.
  S4 marker M2 (5 s gyro-chain yaw innovation over 3 sigma_psi, about 0.9 deg): flags at least 90 % of
     within-segment yaw events on all H0 sessions and at most 10 % of non-event 60 s windows; its sum over
     1_19's event is within 30 % of the step; ORB-SLAM3's 1_19 run has at most 5 % of windows flagged.
Predictions, probe (A0 only, 3 repeats per candidate window):
  R1 E19 reproduces (window Sim3 >= 8 m and yaw step across 436-452 s >= 10 deg) in at least 2 of 3 on
     340:500 or on 300:470. Planner's estimate: 340:500 uncertain, 300:470 more likely (it holds the weak
     spell at 320-336 s where the accel bias leaves).
  R2 E20 reproduces (Sim3 >= 4.5 m and scale >= 1.15) in at least 2 of 3.
  R3 every run's wall time <= 300 s.
Predictions, stage 1a (A1 against paired A0, 3 pairs per window):
  M1 (mechanism counter) walk z p50: gyro 10-12 -> <= 2.0; accel 5.8 -> <= 2.0, on all four windows.
  M3 |ba30 - ref| max: A0 from the probe -> <= 0.25 m/s^2 (3 sigma_a sqrt(160 s)) on E19 and E20.
  M4 5 s median |bg - b_true| max: A0 from the probe -> <= 0.012 rad/s on E19; <= 0.016 on R11 at 329-335 s
     (full-run H0 r1: 48e-3).
  P1 E19 window ATE Sim3: 10.4-18.1 m (full-run cut; A0 must be >= 8) -> <= 4 m in 2 of 3. Cold start without
     the precondition gave 4.3; ORB-SLAM3 1.46.
  P2 E19 yaw step across 436-452 s: 27-35 deg -> <= 5 deg in 2 of 3 (the detector's threshold is 3.9-4.6).
  P3 E19 keyframe speed max over 436-470 s: 3.1-5.2 -> <= 2.3 m/s.
  P4 E20 window ATE Sim3: 6.3-7.3 -> <= 3.5 m; scale 1.21-1.33 -> 0.84-1.00, in 2 of 3.
  P5 guards: Q19 Sim3 <= 1.3 x A0's median (expected 0.29-0.40 m); every window: tracked >= A0 - 1 %,
     give-ups <= A0, vi_ba_solves_3plus_rejects <= 5 % of solves, VIBA0/1/2 accepted, worldrot.py jumps
     over 1 m not more than A0's.
Predictions, stage 1b (A2 against paired A1, same windows, only if the decision table asks):
  M5 header reads `IMU Huber BoundaryOrRecentInit` and the factory offset; M1 and M3 still hold.
  P6 E19 yaw step <= 3 deg and Sim3 <= 2.5 m in 2 of 3; E20 Sim3 <= 2.5 m in 2 of 3.
Batch gate (not a card criterion): expectation for 1_19 CP 8.1-8.8 -> 30-40, recorded for the batch verdict.
Kill rules:
  K0 (stage 0) S2 misses on 3 of 4 sessions, or S3 is under 30 % on both 1_19 rounds: the bias is not the
     carrier. No runs. Report; the event list goes to AC-2b.
  K1 (mechanism moved, score did not) M1 and M3 hold, and P1 or P2 misses in 2 of 3 on E19, or P4 on E20:
     bias memory alone does not hold the event. Run 1b. If P6 also misses with M1, M3, M5 holding: the trigger
     is visual (26-30 observations, pose accepted on inliers, G48). AH closes as refuted for the score.
  K2 (score moved, mechanism did not) P1 or P4 holds with walk z p50 over 4, or with |ba30 - ref| over the
     bound, or the trigger is absent in A1 (observations min over 434-444 s above 150 and no lost keyframe):
     no credit; 3 more pairs on that window.
  K3 (harm) any P5 guard fails in 2 of 3 on a window: knob stays off. Read the IMU nav chi2 in the failing span.
  K4 (reproduction) R1 misses on both starts: 1_19 cannot be tested cold. E20 decides alone if R2 holds.
     R1 and R2 both miss: stop and report; see open question 2.
  K5 S4 misses (under 70 % caught or over 25 % false): M2 is not the marker; report M3 only and say so.
  K6 (unpaced regime) A0's ba_solves / keyframes differs from the gate run's (4609 / 4505) by more than 20 %, or
     tracked share is 3 points under the gate's on the same span: the unpaced window is another regime. Report
     before 1a; fallback is mapping.execution Synchronous on both arms, with +/-50000 ns IMU offset as the repeats.
Bit-identity: A1 and A2 add no code. The window-knob binary with start_sec/end_sec unset: r01full-S10-7,
  RAYON_NUM_THREADS=1, traj md5 d4f77b567b912b5a4caa2af6085a35b1. Must be shown before any AH run.
Noise: unpaced asynchronous runs do not repeat. 3 repeats per arm per window, arms started in pairs under the same
  load; a kill needs 2 of 3. R11 is read on counters only. Windowed ATE on quiet windows spreads 0.29-0.40 m.
If it misses: M1 -> the config did not apply (read the header line); M3 with M1 holding -> the zero-mean accel
  prior, or tilt enters through vision; P1 -> K1; R1 -> K4.
```

## Planner preview (stage 0 must reproduce)

One pass by the planner on the session keyframe caches and the gate trajectories, niced python only; not yet reproduced by `biasevent.py`.

1. **The walk is 10x loose on every keyframe pair, not only at events.** Consecutive-keyframe bias step in the final map, z = |db| / (sigma_w sqrt(3 dt)):

   | session | gyro p50 / p90 | accel p50 / p90 |
   |---|---|---|
   | H0 1_19 r1 (full) | 11.0 / 23.2 | 5.8 / 17.8 |
   | H0 1_19 r2 (full) | 10.3 / 22.8 | 5.8 / 19.1 |
   | cardAC W119-BASE-r1 (cold window 400:620) | 12.0 / 26.7 | 5.8 / 17.0 |

   - Cause: a keyframe is last written as the oldest active one, tied to the frozen past only by the boundary edge, whose bias rows are scaled by 1e-2 (`vi_ba_schur.rs:864-875`). Inside the window the walk rows are at full weight (`vi_ba_schur.rs:2139-2161`).
   - This explains M1's failure. Card S measured the same quantity from debug lines: 8-15, and 1.3 with rows full.
   - It shows in a cold window too, so the mechanism counter works on short runs.
2. **On 1_19 the accel bias leaves about 100 s before the heading event, in both rounds.** 30 s medians of |ba|: 0.10-0.24 m/s^2 up to 330 s, then 0.5-1.3 until 450 s, with 400-600 observations per keyframe and a piece tilt of 6-14 deg. g sin 8 deg = 1.37 m/s^2: the accel bias hides a tilt. 1_20: 0.35-0.62 at 420-570 s. R_11: 0.99-1.63 from 360-390 s.
3. **The 1_19 event repeats in round 2** (-34.6 deg at 449 s, keyframe speed 4.0-5.2 m/s against GT 1.7).
4. **A cold start close to the event does not reproduce it.** W119-BASE-r1 (segment from 414 s): the same weak spell at 436-448 s gives |ba| 0.15-0.35, speed at most 1.77 m/s, 60 s yaw step -2.2 deg, window Sim3 4.29 m. One run. This is why the reproduction probe below is mandatory.
5. **Windowed ATE cut from the existing full runs** (Sim3 RMS, m; 4 runs each: BASE and H0, r1 and r2):

   | window | ours | scale | ORB-SLAM3 |
   |---|---|---|---|
   | 1_19 340:500 (event) | 10.4-18.1 | 0.72-1.49 | 1.46 |
   | 1_19 300:470 (event, earlier start) | 9.1-17.0 | 0.89-1.14 | 1.66 |
   | 1_19 60:220 (quiet) | 0.29-0.40 | 0.88-0.90 | 0.26 |
   | 1_20 380:540 (scale event) | 6.3-7.3 | 1.21-1.33 | not run |
   | 1_20 40:200 (quiet) | 0.74-1.16 | 0.89-0.91 | not run |
   | R_11 280:460 | 1.65-1.94 in 3 runs, 19.7 in H0 r1 | 1.16-3.5 | not run |

   1_20 380:540 is the most repeatable event window. R_11 is chaotic.
6. **Wall time.** Summed frame time in the paced full run: 1_19 340:500 is 159 s for 3200 frames (20 fps); 1_20 380:540 is 164 s; R_11 280:460 is 124 s. But the cold W119 window cost 587 s for 4400 frames (7.5 fps, load unknown). A 160 s window fits 5 min only above 10.7 fps, so stage 1 starts with a timing probe.
7. **Unpaced is close to paced here.** 20 fps unpaced is the sensor rate, so mapping lags tracking about as in the gate. A counter checks it (K6).
8. **The gate configs run with the clock uncorrected and Huber on every IMU edge** (`offset 0 ns`, `IMU Huber AllEdges`). Card AA: 30-41 % of interior edges down-weighted on R_11, mostly at turns. With the bias pinned, the gyro may still lose to vision at the trigger. That is arm A2.

Not verified: the heading carried by the bias error (stage 0); whether a cold start at 340 s or 300 s rebuilds the precondition (probe); BASE's bias state (the gate deletes BASE sessions).

Values stage 0 must reproduce (within 0.3): 1_19 r1 walk z p50 gyro 11.0 / accel 5.8; r2 10.3 / 5.8; W119 12.0 / 5.8. 1_19 |ba| 30 s median max in 300-480 s: 1.31 (r1), 1.26 (r2), within 0.05. `winscore.sh` on the full runs: 1_19 r1 340:500 Sim3 14.82 m, scale 1.242; r2 18.09 m; 1_20 r1 380:540 6.80 m.

## Stage 0 method (`experiments/quick-eval/biasevent.py`, 2026-11-23)

Decided with the owner after the first self-test pass failed (2026-11-23): options (a) and (c) below, and two deviations from the plan's b_true recipe.

- **b_true.** Gyro bias per 60 s piece, from non-overlapping GT pairs about 1 s apart, fitted with the chain's bias Jacobian:
  Log(dR_gyro^T X^T dR_gt X) = J b + K vec(dM), with J = -C_b^T sum_k C_{k+1} dt_k and K = C_b^T sum_k (w_k^T kron C_{k+1}) dt_k.
  Per-piece b and one global 3x3 gyro correction dM are fitted jointly. The camera-IMU clock is scanned from -2 to 10 ms on the bias-only fit, as `clockscan.py` does (1_19: +4.04 ms).
  - Deviation 1: the body rotation X is refitted on the gyro with `clockscan.py`'s constant-bias fit. `lossev.py body`'s C, fitted on SLAM-trajectory pairs, is 0.69-0.71 deg off on 1_19 and doubles the residual at turns: 7.2 against 4.0 mrad on turns over 0.5 rad. C is printed as the check.
  - Deviation 2: the bias Jacobian replaces clockscan's `vi - b dt`, which is first order in the turn. At 160-380 deg/s its error is larger than the bias.
  - The fit is cached per sequence, so window runs read it. Without GT (5_11, 5_12), b_true is the run median of the estimate's 30 s bias medians, and the scale, clock and GT terms are absent.
- **Self-test 1 residual criterion.** It is read as the standard error of b_true per piece (rad/s): the piece's per-pair residual sd divided by sqrt(n pairs). The per-pair rate residual is set by the pseudo-dense GT rotation noise over dt: 2.9 mrad at 0.2 s pairs, 1.2e-2 rad/s. At 1 s pairs it is still 2.1e-3 rad/s, so it cannot meet 1.5e-3.
- **sigma_psi (option a).** sigma_psi^2 = sigma_g^2 tau + sigma_w^2 tau^3/3 + (s_fit tau)^2 + sigma_gt^2 + sigma_run^2 + psi_scale^2.
  - s_fit: the standard error of the piece's b_true.
  - sigma_gt: GT endpoint noise, 1.4826 MAD of the yaw part of Log(dR_gyro(b_true)^T X^T dR_gt X) over GT samples 5 s apart, after the predicted scale yaw is removed (1_19: 0.068 deg).
  - sigma_run: the run's own pose noise, median / 1.538 of the 3D residual of its 1 s hand-eye pairs against the gyro. For a Maxwell norm, median = 1.538 sigma per axis. GT as estimate gives 0.119 deg, ORB-SLAM3 1_19 0.189 deg.
  - psi_scale: the yaw part of C_b^T sum C_{k+1} dM w_k dt over the window, the rotation the fitted gyro scale adds.
  - Every term comes from the data. None is tuned; the self-tests were run once after the change.
- **Flag rule (option c).** A window counts as flagged only inside a run of 2 or more adjacent windows over 3 sigma_psi with the same sign. A single jittered pose shared by two windows flags them with opposite signs (ORB-SLAM3 1_19: +0.95/-1.11, -1.54/+1.20, +1.35/-2.26 deg). Single windows over 3 sigma are still printed.
- **Accel-bias excursion.** ba30 is the trailing 30 s vector median of the keyframe ba on a 5 s grid. The reference is the median of the segment's ba30. The bound is 3 sigma_a sqrt(T), T the span of the segment's nonzero-bias keyframes in the window. The lead of an event is t_ev minus the start of the over-bound run in progress at t_ev; runs merge over gaps of up to 10 s.
- **Bias-carried heading.** -integral of (bg - b_true) . up over the event span [lo, hi], up from the mean specific force over +-2.5 s. Its sign follows M2: a too-large bias turns the estimate the other way. Share = carried x sign(M2 sum) / |yaw step|.
- **Event class.** restart if a segment start lies in [lo - 30 s, t_ev]; else within.

## Self-tests (2026-11-23, outputs in `.pr180-fix/cardAH/s0/selftest/`)

| test | criterion | first pass (sigma_psi without endpoint and scale terms, single-window flags) | final |
|---|---|---|---|
| 1 GT as estimate, 1_19 | 0 flagged; b_true error < 1.5e-3 rad/s | 19/159 flagged; per-pair sd 2.73e-3 | 0/159 flagged (0 single windows over 3 sigma); standard error p50 2.90e-4 rad/s, per-pair sd 2.13e-3: pass |
| 2 GT + 30 deg yaw ramp 436-452 s | M2 sum 28.5-31.5 deg, flagged only there | sum +30.1, plus 19 background flags | M2 sum +30.2 over the event (ramp windows 435-455: +7.48, +9.27, +9.44, +3.69); flagged only 435-455; non-event pieces flagged 0/12: pass |
| 3 ORB-SLAM3 1_19 | <= 5 % of windows flagged | 48/181 (26.5 %) | 0/181 flagged (9 single windows, 5.0 %, all in opposite-sign pairs or alone): pass |

## Stage 0 results

Run 2026-11-23, outputs in `.pr180-fix/cardAH/s0/<run>.txt`: every H0 gate session (r1 and r2), `cardAC/out/W119-BASE-r1`, and M2 only on the BASE `traj.txt` files and ORB-SLAM3 1_19.
- BASE 5_11 and 5_12 have no GT and no session, so there is nothing to fit b_true against and they give no output.
- The planner preview reproduces exactly: 1_19 r1 walk z p50 gyro 11.02 / accel 5.79, r2 10.34 / 5.79, W119 12.04 / 5.77; 1_19 |ba| 30 s median max in 300-480 s 1.31 (r1), 1.26 (r2).
- `winscore.sh` reproduces too: 1_19 r1 340:500 Sim3 14.823 m, scale 1.242; r2 18.094 m; 1_20 r1 380:540 6.802 m.

### Predictions and kill rules

| id | prediction | measured | verdict |
|---|---|---|---|
| S1 | walk z p50 on every H0 session: gyro 8-13, accel 4-7 | Gyro: 7.91-12.34; 16 of 17 sessions in band, 4_11 r2 at 7.91. Accel: 5.43-11.85; 9 of 17 in band. Over 7 on R_11 r1 (7.26), 2_11 (7.65, 7.39), 2_12 (9.05), 4_10 (11.85), 4_11 (9.63, 10.50), 5_12 (7.84): the sequences with many segments | **missed** as stated. The mechanism holds: consecutive pairs sit at 8-12x (gyro) and 5-12x (accel) the sensor walk |
| S2 | 1_19 r1/r2, 1_20 r1/r2: ba30 - ref crosses the bound >= 10 s before each within-segment event (1_19 by 80-110 s) | 1_19 r1: first crossing at 360 s, 92 s before the step (452 s); below the bound 400-435 s, over again 435-455 s (lead of the run in progress 17 s). r2: first crossing 365 s, 84 s before 449 s; below 405-435 s (in-progress lead 14 s). 1_20 r1, r2: never over the bound (max 0.56 and 0.59 against 0.61) | **predicted on 1_19** (both rounds, first crossing 84-92 s before the step; the excursion has a 30-35 s dip), **missed on 1_20** (2 of 4) |
| S3 | bias-carried heading is 50-150 % of the yaw step on 1_19 r1 and r2 | r1: -33.0 deg against a step of -27.2 (121 %); r2: -20.8 against -34.6 (60 %). Same sign as M2 in both | **predicted** |
| S4 | M2 flags >= 90 % of within-segment yaw events on H0 sessions, <= 10 % of non-event 60 s windows; ORB-SLAM3 1_19 <= 5 % | Caught 8 of 11 within-segment yaw events (73 %). Missed 2_11 r1 60-163 s (+132.7 deg, M2 sum -1.5), 3_17 1260-1332 s (-125.7, M2 +0.2), 3_18 1380-1560 s (-1.4). Non-event pieces flagged 22/163 (13.5 %). ORB-SLAM3 0/181 (0 %). M2 sum over 1_19's event: r1 -25.8 against a step of -27.2 (95 %), r2 -14.5 against -34.6 (42 %) | **missed** (73 % caught, 13.5 % false); ORB part predicted |
| S5 | W119-BASE-r1: no M2 flag run > 5 s at 436-452 s; ba30 - ref inside the bound | Flag run 440-450 s (2 windows: -2.13 and -1.00 deg against 3 sigma 1.00). ba30 - ref max 0.29 against a bound of 0.28 (one 5 s point, 475 s) | **missed** narrowly: the yaw is small (-3.1 deg over 10 s against -28 to -36 in the full runs), the ba excursion at the bound |
| K0 | S2 misses on 3 of 4, or S3 < 30 % on both 1_19 rounds | S2 misses 2 of 4; S3 121 % and 60 % | **not fired** |
| K5 | S4 catches < 70 % or flags > 25 % false | 73 % caught, 13.5 % false | **not fired** (close: 3 points over the catch floor) |

### Diagnosis of the misses

- **S1 accel.** The accel walk is over the band on the sequences with 25-67 segments. Each restart begins with a bias near zero that converges inside young windows; those keyframe pairs move several sigma per step. On sequences with at most 14 segments, accel p50 is 5.4-7.3. The prediction did not separate restarts. Gyro z is 10-12 almost everywhere, so the loosened walk is general.
- **S2 on 1_20.** The 1_20 accel bias rises to 0.35-0.62 m/s^2 at 420-570 s, but its bound over a segment of about 1000 s is 0.61: a whole-segment random-walk bound is too loose to see it. The 1_20 events found here are a scale event at 420 s (step -7.5 / -3.4 deg) and, in r2, a yaw event at 645 s. Neither is a heading event of the 1_19 kind. So the accel bias is not shown to carry 1_20's scale event.
- **S4.**
  - The missed events have large position-derived yaw steps (125-133 deg) and an M2 sum near 0. The estimate's attitude agrees with the gyro there, so the step comes from position (scale, origin or alignment of the piece), not from attitude.
  - The third miss (3_18, -1.4 deg) is an event the 60 s detector merged on scale and tilt.
  - M2 is an attitude marker by construction. It cannot see a translation-only heading error.
  - False flags are concentrated in 1_20 r1 (5/15), 3_17 (4/22), 4_10 (4/12) and 2_11 r1 (3/9). They are same-sign runs of 1-1.5 deg, about 1.5x the 3 sigma bound.
- **S5.** The cold start does show a small attitude departure at the same 440-450 s weak spell (-3.1 deg in 10 s). With no 100 s accel-bias lead, it stays an order of magnitude smaller than in the full runs (-28 to -36 deg). This fits the card's reading that the precondition matters.
- **Order of the bias error on 1_19 r1.** The 5 s median |bg - b_true| is 2.0e-2 rad/s at 395 s and 2.7e-2 at 415 s, before the attitude departs from the gyro at 440 s (M2). The largest are 3.5e-2 at 445 s (r1) and 4.1e-2 at 475 s (r2), against p50 5.4-5.7e-3. Vision holds the attitude while the bias is wrong; the departure starts in the weak spell. The share of that early error about gravity was not split out.

### Per session

| run | segs | later_al | walk g p50/p90 | walk a p50/p90 | ba30 max 300-480 | ba30-ref max / bound | over % | |bg-bt| p50/max | M2 flagged | single | events | nonev flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE-R_11_5cp-r1 | 1 | 0 | -/- | -/- | - | - / - | - | -/- | 5/95 (5.3 %) | 14 | 2 | 1/6 |
| BASE-R_11_5cp-r2 | 1 | 0 | -/- | -/- | - | - / - | - | -/- | 5/95 (5.3 %) | 14 | 1 | 0/6 |
| BASE-sequence_1_19-r1 | 1 | 0 | -/- | -/- | - | - / - | - | -/- | 7/182 (3.8 %) | 16 | 1 | 0/12 |
| BASE-sequence_1_19-r2 | 7 | 5 | -/- | -/- | - | - / - | - | -/- | 10/178 (5.6 %) | 26 | 3 | 0/8 |
| BASE-sequence_1_20-r1 | 5 | 1 | -/- | -/- | - | - / - | - | -/- | 6/197 (3.0 %) | 24 | 2 | 2/13 |
| BASE-sequence_1_20-r2 | 3 | 0 | -/- | -/- | - | - / - | - | -/- | 12/200 (6.0 %) | 24 | 2 | 3/14 |
| BASE-sequence_2_11-r1 | 29 | 21 | -/- | -/- | - | - / - | - | -/- | 12/214 (5.6 %) | 33 | 9 | 2/9 |
| BASE-sequence_2_11-r2 | 19 | 18 | -/- | -/- | - | - / - | - | -/- | 15/214 (7.0 %) | 49 | 8 | 1/8 |
| BASE-sequence_2_12-r1 | 60 | 48 | -/- | -/- | - | - / - | - | -/- | 30/255 (11.8 %) | 66 | 14 | 2/8 |
| BASE-sequence_3_17-r1 | 17 | 22 | -/- | -/- | - | - / - | - | -/- | 32/337 (9.5 %) | 62 | 9 | 5/13 |
| BASE-sequence_3_18-r1 | 8 | 6 | -/- | -/- | - | - / - | - | -/- | 4/326 (1.2 %) | 27 | 4 | 0/24 |
| BASE-sequence_4_10-r1 | 108 | 59 | -/- | -/- | - | - / - | - | -/- | 25/204 (12.3 %) | 55 | 18 | 0/6 |
| BASE-sequence_4_11-r1 | 68 | 41 | -/- | -/- | - | - / - | - | -/- | 13/159 (8.2 %) | 34 | 15 | 1/6 |
| BASE-sequence_4_11-r2 | 47 | 42 | -/- | -/- | - | - / - | - | -/- | 13/180 (7.2 %) | 39 | 10 | 2/7 |
| H0-R_11_5cp-r1 | 9 | 0 | 11.79/27.28 | 7.26/31.38 | 1.63 | 1.11 / 0.36 | 16 | 5.74e-03/5.55e-02 | 0/82 (0.0 %) | 7 | 5 | 0/6 |
| H0-R_11_5cp-r2 | 3 | 0 | 11.93/26.12 | 6.96/21.32 | 0.99 | 0.83 / 0.42 | 8 | 5.80e-03/1.52e-02 | 7/94 (7.4 %) | 15 | 1 | 0/6 |
| H0-sequence_1_19-r1 | 6 | 3 | 11.02/23.19 | 5.79/17.78 | 1.31 | 1.14 / 0.58 | 10 | 5.68e-03/3.46e-02 | 7/182 (3.8 %) | 18 | 1 | 0/12 |
| H0-sequence_1_19-r2 | 6 | 0 | 10.34/22.81 | 5.79/19.09 | 1.26 | 1.13 / 0.44 | 11 | 5.42e-03/4.11e-02 | 11/177 (6.2 %) | 22 | 2 | 0/12 |
| H0-sequence_1_20-r1 | 7 | 0 | 10.13/21.18 | 6.05/17.37 | 0.47 | 0.56 / 0.61 | 0 | 5.14e-03/2.45e-02 | 12/200 (6.0 %) | 28 | 1 | 5/15 |
| H0-sequence_1_20-r2 | 7 | 0 | 10.44/21.49 | 6.24/16.16 | 0.35 | 0.59 / 0.61 | 0 | 4.94e-03/2.12e-02 | 3/200 (1.5 %) | 20 | 2 | 0/13 |
| H0-sequence_2_11-r1 | 25 | 0 | 11.42/27.11 | 7.65/27.80 | 0.38 | 2.78 / 0.38 | 35 | 5.89e-03/1.61e-01 | 26/216 (12.0 %) | 54 | 10 | 3/9 |
| H0-sequence_2_11-r2 | 30 | 0 | 10.09/22.98 | 7.39/26.50 | 0.49 | 1.70 / 0.38 | 38 | 5.75e-03/1.89e-01 | 13/206 (6.3 %) | 33 | 14 | 1/7 |
| H0-sequence_2_12-r1 | 46 | 3 | 12.10/28.68 | 9.05/31.07 | 1.78 | 1.55 / 0.27 | 62 | 7.88e-03/1.18e+00 | 12/246 (4.9 %) | 43 | 18 | 0/13 |
| H0-sequence_3_17-r1 | 11 | 0 | 10.40/23.43 | 6.48/22.78 | 0.48 | 2.08 / 0.71 | 21 | 6.04e-03/4.81e-02 | 23/345 (6.7 %) | 54 | 7 | 4/22 |
| H0-sequence_3_18-r1 | 14 | 0 | 11.05/23.36 | 5.72/19.26 | 0.33 | 0.90 / 0.62 | 16 | 5.76e-03/2.43e-02 | 8/334 (2.4 %) | 45 | 4 | 3/22 |
| H0-sequence_4_10-r1 | 67 | 3 | 12.34/32.88 | 11.85/46.49 | 3.07 | 2.16 / 0.18 | 48 | 1.07e-02/1.40e+00 | 23/186 (12.4 %) | 50 | 24 | 4/12 |
| H0-sequence_4_11-r1 | 55 | 0 | 10.88/30.09 | 9.63/40.39 | 0.77 | 1.10 / 0.23 | 36 | 8.83e-03/9.80e-01 | 10/148 (6.8 %) | 32 | 14 | 1/6 |
| H0-sequence_4_11-r2 | 38 | 0 | 7.91/24.50 | 10.50/42.14 | 2.38 | 1.56 / 0.23 | 37 | 8.35e-03/1.63e+00 | 30/164 (18.3 %) | 49 | 15 | 1/8 |
| H0-sequence_5_11-r1 | 6 | 0 | 9.98/21.83 | 5.43/14.49 | 0.38 | 0.74 / 0.60 | 5 | 3.08e-03/5.15e-02 | 0/217 (0.0 %) | 8 | - | -/- |
| H0-sequence_5_11-r2 | 5 | 0 | 10.17/22.16 | 5.53/14.56 | 0.28 | 0.83 / 0.60 | 9 | 2.98e-03/3.58e-02 | 2/220 (0.9 %) | 10 | - | -/- |
| H0-sequence_5_12-r1 | 13 | 0 | 12.15/28.49 | 7.84/27.24 | 1.36 | 3.00 / 0.46 | 42 | 3.82e-03/2.19e-01 | 20/245 (8.2 %) | 32 | - | -/- |
| ORB-sequence_1_19-r0 | 1 | - | -/- | -/- | - | - / - | - | -/- | 0/181 (0.0 %) | 9 | 0 | 0/15 |
| W119-BASE-r1 | 8 | - | 12.04/26.71 | 5.77/16.99 | 0.33 | 0.29 / 0.28 | 2 | 5.08e-03/1.03e-02 | 2/40 (5.0 %) | 3 | 1 | 0/3 |

walk = consecutive-keyframe z p50/p90. ba30 max 300-480 = the 30 s grid median of |ba|. ba30-ref and the bound are in m/s^2. |bg-bt| is in rad/s. M2 flagged counts same-sign runs of 2 or more windows; single = windows over 3 sigma alone. BASE rows are M2 only. Segment counts come from the session for H0 and W119, and from stderr restarts for BASE.

### Events on 1_19, 1_20, R_11 and W119

| run | lo-hi | kind | step | t_ev | class | M2 sum | M2 flagged | carried | share % | ba lead s |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE-R_11_5cp-r1 | 240-300 | scale | +0.2 | 240 | within | -2.4 | 0/12 | +nan | nan | - |
| BASE-R_11_5cp-r1 | 360-420 | scale | -6.4 | 360 | within | -5.4 | 3/12 | +nan | nan | - |
| BASE-R_11_5cp-r2 | 300-420 | yaw+scale | -14.8 | 355 | within | -16.1 | 5/24 | +nan | nan | - |
| BASE-sequence_1_19-r1 | 360-540 | yaw+scale | -14.4 | 453 | within | -13.3 | 7/36 | +nan | nan | - |
| BASE-sequence_1_19-r2 | 360-530 | yaw+scale | -8.8 | 450 | within | -2.4 | 10/32 | +nan | nan | - |
| BASE-sequence_1_19-r2 | 540-660 | yaw+scale | +nan | 554 | restart | -1.8 | 0/24 | +nan | nan | - |
| BASE-sequence_1_19-r2 | 780-900 | yaw | -7.5 | 820 | within | +7.5 | 0/24 | +nan | nan | - |
| BASE-sequence_1_20-r1 | 360-540 | scale | +2.6 | 360 | within | -1.1 | 0/36 | +nan | nan | - |
| BASE-sequence_1_20-r1 | 960-1002 | scale | -9.8 | 960 | within | -0.6 | 0/7 | +nan | nan | - |
| BASE-sequence_1_20-r2 | 420-540 | scale | -7.8 | 420 | within | -0.6 | 0/24 | +nan | nan | - |
| BASE-sequence_1_20-r2 | 960-1020 | scale | -4.2 | 960 | within | +7.3 | 2/10 | +nan | nan | - |
| H0-R_11_5cp-r1 | 300-341 | scale | -10.7 | 300 | within | -1.8 | 0/7 | +2.3 | -22 | - |
| H0-R_11_5cp-r1 | 370-375 | scale | +nan | 370 | restart | +0.0 | 0/0 | +9.0 | nan | - |
| H0-R_11_5cp-r1 | 375-393 | scale | +nan | 375 | restart | +0.5 | 0/2 | -0.2 | nan | - |
| H0-R_11_5cp-r1 | 393-402 | scale | +nan | 393 | restart | +0.0 | 0/0 | -1.3 | nan | - |
| H0-R_11_5cp-r1 | 402-420 | scale | +nan | 402 | restart | -0.3 | 0/3 | -3.8 | nan | - |
| H0-R_11_5cp-r2 | 300-420 | yaw+scale | -10.6 | 375 | within | -11.0 | 7/24 | -31.3 | 294 | - |
| H0-sequence_1_19-r1 | 360-540 | yaw+scale | -27.2 | 452 | within | -25.8 | 7/36 | -33.0 | 121 | 17 |
| H0-sequence_1_19-r2 | 360-509 | yaw+scale | -34.6 | 449 | within | -14.5 | 11/27 | -20.8 | 60 | 14 |
| H0-sequence_1_19-r2 | 509-532 | scale | +nan | 509 | restart | +1.6 | 0/3 | +12.7 | nan | - |
| H0-sequence_1_20-r1 | 420-540 | scale | -7.5 | 420 | within | -0.7 | 0/24 | -1.9 | 25 | - |
| H0-sequence_1_20-r2 | 420-540 | scale | -3.4 | 420 | within | +1.1 | 0/24 | +0.5 | 16 | - |
| H0-sequence_1_20-r2 | 600-720 | yaw | +18.4 | 645 | within | +13.6 | 3/24 | +12.0 | 65 | - |
| W119-BASE-r1 | 420-480 | scale | -2.2 | 420 | restart | -2.3 | 2/12 | -1.1 | 49 | - |

step = yaw step of the 60 s pieces (driftbudget.py; nan when the event has no clean piece before it in the same segment). carried = bias-carried heading in deg (H0 and W119 only; BASE keeps no session). ba lead = t_ev minus the start of the over-bound run in progress at t_ev. The first crossings on 1_19 are in S2.

### Segments and later alignments (multi-segment issue; no segsim3 work)

| sequence / round | H0 segments | H0 later alignments | BASE segments | BASE later alignments |
|---|---|---|---|---|
| R_11_5cp-r1 | 9 | 0 | 1 | 0 |
| R_11_5cp-r2 | 3 | 0 | 1 | 0 |
| sequence_1_19-r1 | 6 | 3 | 1 | 0 |
| sequence_1_19-r2 | 6 | 0 | 7 | 5 |
| sequence_1_20-r1 | 7 | 0 | 5 | 1 |
| sequence_1_20-r2 | 7 | 0 | 3 | 0 |
| sequence_2_11-r1 | 25 | 0 | 29 | 21 |
| sequence_2_11-r2 | 30 | 0 | 19 | 18 |
| sequence_2_12-r1 | 46 | 3 | 60 | 48 |
| sequence_3_17-r1 | 11 | 0 | 17 | 22 |
| sequence_3_18-r1 | 14 | 0 | 8 | 6 |
| sequence_4_10-r1 | 67 | 3 | 108 | 59 |
| sequence_4_11-r1 | 55 | 0 | 68 | 41 |
| sequence_4_11-r2 | 38 | 0 | 47 | 42 |
| sequence_5_11-r1 | 6 | 0 | no output (no GT, no session) | - |
| sequence_5_11-r2 | 5 | 0 | no output (no GT, no session) | - |
| sequence_5_12-r1 | 13 | 0 | no output (no GT, no session) | - |

H0 keeps far fewer later alignments: 0 on 14 of 17 sessions, at most 3 (1_19 r1, 2_12, 4_10). BASE has 0-59, and 18-59 on 2_11, 2_12, 3_17, 4_10 and 4_11. H0 has somewhat fewer segments (4_10: 67 against 108; 2_12: 46 against 60), but 4_10 and 4_11 still have 38-67.
