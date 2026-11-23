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

## Stage 0 results

Pending (`experiments/quick-eval/biasevent.py`, outputs in `.pr180-fix/cardAH/s0/`).
