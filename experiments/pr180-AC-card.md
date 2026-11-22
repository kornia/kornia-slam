# Card AC: one world through a tracking loss

Status: 2026-11-22. AC-0 stopped at R0: K6 fired on W211a, W211b and W119 (BASE does not reproduce the F1 events),
and the box ran at load 20-40 against the F1 validation's 2.5, which breaks the paired-run assumption (W318 BASE and
BASEd differ: 2 and 0 give-ups). Counters (step 4) landed; AC-1 (`tracking_loss.reseed.seed: Held`) not built, per
the stop rule. AC-2b, AC-2c and AC-3 wait for the owner's verdict on AC-0 and AC-1. Plan: planner (Fable), owner answers
2026-11-22. Worktree `/home/edgar/dev/split/kornia-slam-loss`, branch
`feat/loss-one-world`.

Paths: `P=/home/edgar/dev/split/.pr180-fix`, `AC=$P/cardAC`, `V=$P/bench-F1/val/out`.

Owner answers (2026-11-22): S1 and S2 merged into AC-1 through the keep path, no attitude-only variant. `Held` is an
interim, measured policy; the cleanup card must not promote `Held` alone. Restart bug: counters now; fix only if
`later_alignments > 0` beyond 10 m from the origin remain in AC-1 runs, then as a bug fix in the cleanup card. The
R_01 `--blackout` runs join the standing guard set. AC-2b threshold (if built): 99.9 % chi2 with the self-calibrating
consistency factor. Offbridge test marked confounded; rerun on AC-1 sessions later. Full validation after AC-1:
1_19, 5_11, 2_11, R_11 first (one job, paced), then 3_18 and 4_11 if no kill fired.

## Experiment: AC, one world through a tracking loss
Gap tested: G22, G25, G19, G39, G40; new G46 (later-segment alignment rotates earlier segments about the
  origin), G47 (attitude frozen through recovery; IMU-less young segment)
Base: kornia-slam@072e2bf (code = pr180c/F1 81bc775)
Binary tag: AC-0 = kornia-slam-f1 (md5 52e38c464d4369df0c9d3f61df17b5aa); AC-1 = kornia-slam-ac (md5 at build)
Mechanism: section 1 above, links (a)-(e) and findings 1-3.
Theory: gravity, bias and the gyro attitude belong to the device and the world, not to the lost landmarks.
  Kept, the next segment needs no VIBA0, so nothing re-aligns the world, and it has a trusted prediction and
  the 5 s grace from its first frame. A held seed removes the coast's translation from the seed.
Knobs: reseed.keep_inertial (exists, default false); reseed.seed (new, default CoastedIfPlausible);
  reseed.min_seed_points 20 (exists, set equal to min_cold_start_points); tracking.frame_velocity (exists).
Baselines (F1 validation, one sample, $V=/home/edgar/dev/split/.pr180-fix/bench-F1/val/out):
  seq    CP on   tracked  give_ups  runaway_frames  lost_kf_at_runaway  origin jumps
  R_11   55.5    93.8 %   2         0               0                   none
  1_19   4.84    93.7 %   11        247             51                  64.5 m
  2_11   FAIL    76.9 %   61        829             168                 6 of 125-731 m before 730 s
  3_18   0.42    93.7 %   13        44              11                  193.8 + 607.3 m
  4_11   FAIL    63.5 %   134       1467            284                 not listed
  5_11   5.39    91.8 %   9         77              15                  423.5 + 291.9 m
  Young deaths 36 (59 %) on 2_11, 115 (86 %) on 4_11. Rotation error across give-ups, median: 131 / 119 deg.
  Coast error at 3-5 s 6.9-7.1 m, hold 0.6 m. Keyframes over 4 m/s: 155 / 393 / 513 / 37 (1_19/2_11/4_11/5_11).
  CP oracle, restarts linked: 1_19 38.8, 3_18 32.1, R_11 56.3.
Owner change (2026-11-22, after AC-0 round 1, K6): R0 below is retired as a gate. AC-0 and AC-1 are scored on each
  run's own loss events (give-ups after the first VIBA2, coasts, restarts), with BASE and the variant run as a pair
  from the same window or prefix on a quiet machine (load <= 4 before each pair, no other kornia-slam process). F1
  event times are no longer the gate. `later_alignments` counts only alignments in a segment later than the first
  aligned one (start-up resets before any alignment do not count). No distance threshold is needed: a world
  never aligned holds nothing gravity-aligned to rotate, so the first alignment is the world's own levelling
  whatever its size (R_01: 90.4 deg, 0.32 m); any later one rotates aligned keyframes (G46).
Predictions, AC-0 (windows of section 4; paired BASE; DEBUG=1 on one BASE per window):
  R0  (retired 2026-11-22, see the owner change above) each BASE window has >= 1 give-up after the first VIBA2,
      within 10 s of the F1 event time.
  M0e angle between a later segment's printed VIBA0 gravity and (0, 9.81, 0) equals the single-frame
      origin-jump angle within 2 deg.                                  measured by: worldrot.py + [imu_init] lines
  M0a (unscored) at keyframes with |v| > 3 m/s: inliers, [predgap] trans, [vi_ba] edge |r_v|, [vel] src.
  P0u U: coast speed error 1.2-1.9 -> <= 0.6 m/s median; coast error at 3-5 s 6.9-7.1 -> <= 2.5 m;
      [vel] src=prop on >= 95 % of coast frames.                       measured by: lossev.py
  P0k KI: VIBA0 accepted after the first init: 0; origin jumps over 1 m: 0; reseeds = give-ups after init;
      rotation error across give-ups <= 10 deg median; young deaths <= 20 % of give-ups.
  P0s KIU: position error at restart against GT <= 2.5 m median (KI alone: 5-8 m, the old refutation).
  P0b R_01 --blackout 25:6, 1000 frames: BASE position error after restart >= 4 m, KIU <= 1.5 m and <= 3 deg.
Predictions, AC-1 (H = KIU + seed: Held), windows then full sequences:
  M1  later_alignments = 0 on every sequence; later_alignment_jump_m_max = 0.
  M2  seeds_gyro_held = reseeds; rotation error across give-ups <= 10 deg median on 2_11 and 4_11.
  M3  young deaths <= 20 % of give-ups; give_ups 2_11 61 -> <= 30, 4_11 134 -> <= 60.
  M4  per-segment tilt of world Y in the session dump <= 5 deg (baseline 27-86); offline start cost on 1_19
      8.1e10 -> <= 8e9.
  P1  position error at restart <= 1.0 m median on the turn-on-the-spot events.
  P2  tracked: 2_11 76.9 -> >= 82 %; 4_11 63.5 -> >= 70 %.
  P3  official CP: 1_19 4.84 -> 15-39; 5_11 5.39 -> 15-40; 3_18 0.42 -> 15-32; 2_11 and 4_11 Sim3 succeeds.
      One sample each: provisional.
  P4  expected loss: H on the R_01 blackout holds through walking, 5-7 m against KIU's <= 1.5 m. That is
      AC-2c's case, recorded, not a kill.
  Guards: KI and H without U on r01full-S10-7 cmp-identical to the baseline (R_01, R_03 and R_11 windows
      have no give-up after init); with U, R_01 full 0.30-0.57 m; R_03 0.10-0.12 m and 758-790 tracked;
      R_11 0.15-0.20 m; R_11_5cp CP 50-60 with reseeds = 0.
Predictions, AC-2b (only if built): keyframes over 4 m/s <= 10 % of baseline; coast_runaway_frames <= 10 %;
  3_18 at 1212 s: coast error 10 m -> <= 1.5 m and the next 30 s at scale 0.9-1.1; CP 1_19 39-49, 3_18 32-45.
Kill rules:
  K1 (mechanism moved, score did not) M1-M3 hold and P3 misses on >= 2 of 1_19, 3_18, 5_11: the links were
     not the limit. Read the oracle (linked against 60 s pieces): the within-segment events dominate; go to AC-2.
  K2 (score moved, mechanism did not) P3 holds with later_alignments > 0 or reseeds < give-ups after init:
     luck or a different trajectory. No credit; rerun with +/-50 us IMU offset, 3 samples.
  K3 (harm) a guard missed in 2/3; a cmp-identity failure; young deaths above BASE; reseed_stall_frames above
     BASE skipped_bootstrap; bias_resets > 0 on R_11: knobs stay at their defaults.
  K4 (P0u misses) coast error stays above 5 m with U: stale velocity is not the driver; the velocity at loss
     onset is wrong. AC-2b is needed before any coasted seed.
  K5 (AC-2b gate) healthy-frame NIS p99.9 within 10x of the event frames' p10 in the AC-0 logs: no innovation
     threshold separates them. Do not build 2b; score the HB arm instead.
  K6 (R0 misses) a window does not reproduce its event: lengthen the lead to 60 s, then use the full prefix.
Bit-identity: knobs off, r01full-S10-7, RAYON_NUM_THREADS=1, traj md5 d4f77b567b912b5a4caa2af6085a35b1.
Noise: deterministic; 3 samples per arm on W211a and W119 with imu_time_offset_ns 0 and +/-50000.
If it misses: M1 -> later_alignment_* names the segment; M2 -> seed rotation against the gyro integral;
  M3 -> [track] reject reasons in the first 3 s (15-inlier bar on a 20-point seed); P1 -> lost keyframes at
  coasted poses pulling the seed (G28; try own_window_keyframes: 10); P3 -> oracle gap per sequence.

Knob classes: `keep_inertial`: reference mechanism; the only path if AC-1 passes. `seed`: experiment knob; the cleanup card deletes the losers. `frame_velocity`: bug fix (already on the cleanup list). `pose_gate: Innovation`: derived from the sensor model. Hand-set, untouched and flagged for the derive-thresholds item: `timeout_visual_sec` 0.5, `max_coast_speed_mps` 4 (unused under `Held`), `bridge_covariance_scale` 100 (unused under `Held`), `max_tilt_deg` 10, `min_imu_confidence_sec` 2, `min_seed_points`. Reference constant: `timeout_imu_sec` 5 (ORB-SLAM3 `Tracking.cc:48`).


## Mechanism: the chain, link by link

| Link | Verdict | Evidence |
|---|---|---|
| (a) what accepts a contradicted velocity | Open at every step (code). Which step lets 5-16 m/s in is undetermined without a DEBUG replay. | PnP accepts on inliers alone: 15 with a trusted IMU (`tracking/pose_estimation/map_projection/mod.rs:162`, `tracking/tracker.rs:773-788`), 10 otherwise (`pnp.rs:57`). `pose_gate` is `Disabled` (`tracking/pose_gate.rs:18-22`). The keyframe takes the state velocity (`system/mod.rs:3107-3110`). VI-BA puts Huber on every IMU edge at chi2 16.9 (`mapping/vi_ba_schur.rs:381,386,878-887`), so the IMU edge that contradicts a weak pose is the one down-weighted. Write-back is unconditional (`mapping/map/ops/correction.rs:701-717`, envelope `Off` at `:112-114`). Lost keyframes are `Free` (`mapping/bundle_adjustment.rs:136`). The tracker re-reads the keyframe velocity every frame (`system/mod.rs:1923-1927`, `:1793`). |
| (b) coast and the 4 m/s test | Confirmed. | Coast: `system/mod.rs:2056-2073`, prediction `:2556-2589`, `tracking/motion.rs:61-83`. Runaway test: net displacement over 10 s above 4 m/s (`system/recovery.rs:66-67,173-223`), so it needs 40 m net and a 1.5 m/s coast never trips it. Under the validation preset it only sets the `trusted` flag and counters (`system/mod.rs:1257-1266`); lost keyframes are still inserted at runaway poses (`:2221` needs `keep_inertial`, `:2232` only counts). |
| (c) give-up reset | Confirmed, and worse than stated. | `system/mod.rs:2197-2213`, then `give_up` `:2765-2771`, then `system/state.rs:158-162` and `reset()` `:111-128` (velocity zero `:127`, `imu_initialized` false `:125`). The seed is the whole last tracked pose. Each young-segment death re-seeds from the previous seed's attitude, and bootstrap stalls update nothing (`coast_while_reseeding` runs only when `reseeding()`, `system/mod.rs:1200-1202`). So the attitude is frozen for the whole recovery (median 8.9 s, max 35 s). |
| (d) new segment has no IMU state, 0.5 s grace | Confirmed. | `finish_bootstrap` goes to `ImuInit` (`system/mod.rs:1596-1603`). Prediction needs `imu_initialized` (`:2564`), so it is the previous pose with no gyro; the KLT gyro guess is off too (`:2547-2549`). Search radius stays 15 px. Trust needs init plus 2 s (`tracking/policy.rs:287`, `system/mod.rs:2706-2713`). Grace is 0.5 s (`policy.rs:286,301-307`). `map_established` counts keyframes of all segments (`system/mod.rs:2188-2189`), so the young-map rule never applies. |
| (e) restart rotated about the origin | Confirmed real, code and data. It is not yaw. | `ImuInitializer::apply_initialization` (`inertial/initialization/mod.rs:216-223`) calls `Map::apply_inertial_alignment` (`correction.rs:460`); `prepare_world_transform` (`:578-603`) transforms every keyframe and landmark of every segment; `adopt_inertial_initialization` (`state.rs:205-212`) moves the tracking pose. Called at the new segment's VIBA0 (`system/mod.rs:1698,1718-1734`) and VIBA1/2 (`:3286-3329`). The rotation is the minimal one taking the segment's solved gravity to +Y. The comment at `inertial/schedule.rs:95-96` already says a refinement "would re-align the whole world about the origin". |

**Data for (e), from `/home/edgar/dev/split/.pr180-fix/bench-F1/val/out/*/traj.txt`.** Single-frame jumps (dt 0.05 s) that keep the distance from the origin to 0.1 m:

| Seq | Time | Jump | Distance from origin | Angle |
|---|---|---|---|---|
| 5_11 | 974.4 s | 423.5 m | 416.4 to 416.3 m | 61.1° |
| 5_11 | 1113.2 s | 291.9 m | 393.2 to 393.2 m | 43.6° |
| 3_18 | 1265.5 s | 193.8 m | 603.9 to 603.8 m | 18.5° |
| 3_18 | 1356.7 s | 607.3 m | 557.3 to 557.2 m | 66.0° |
| 1_19 | 607.4 s | 64.5 m | 148.2 to 148.2 m | 25.1° |
| 2_11 | 178.4, 386.6, 410.8, 496.8, 615.2, 722.5 s | 142, 358, 731, 125, 703, 697 m | preserved | 12.7-86° |

The axes are generic (5_11: (0.47, −0.59, 0.66); y goes from −27 to −306 m), which resolves the diagnosis's doubt about yaw.

**New finding 1: the session dump's earlier segments are rotated and off gravity.** Origin rotation from the online trajectory to the end-of-run session keyframes, per segment:

| Seq, segment span | Rotation | Tilt of world Y | Fit RMS |
|---|---|---|---|
| 3_18, [2.8, 1264] s | 89.3° | 86.3° | 0.28 m over 6658 keyframes |
| 5_11, [0.2, 966] s | 48.6° | 43.9° | 0.18 m |
| 1_19, [0.7, 596] s | 26.7° | 26.7° | 3.4 m |
| R_11 (control) | 0.1° | 0.1° | 0.29 m |

The dump's `gravity_world` is (0, 9.81, 0) for all segments. Likely cause of the inertial start cost in offline BA (8e10 to 6e15, against R_11's 1.2e8). The parallel offbridge test is confounded by it.

**New finding 2: the restart's gravity solve is right; the seed attitude is wrong.** Per-piece Sim3 to GT gives gravity within 2-5° between long pieces (3_18 4.8°, 1_19 2.2°, R_11 3.8°). So the 18-86° alignment is the tilt error of the frozen seed attitude (inferred). A short segment that only reached VIBA0 is 19.9° off (3_18, 1270-1345 s), a 3.3 m/s² gravity leak into its coasts.

**New finding 3 (code; magnitude unmeasured): coasts replay a stale velocity.** With `pose_inertial: Off` and the default `FrameVelocity::KeyframeVelocity` (`system/config/mod.rs:499-502`), every frame, coasting included, restarts from the reference keyframe's velocity. Between lost keyframes (0.2 s, 4 frames) only 1 frame in 4 of acceleration is integrated. This predicts a coast that keeps the pre-loss walking speed, matching the measured 1.2-1.9 m/s against a true 0.1-0.35. `tracking.frame_velocity: PropagatedFromKeyframe` (card U) is the existing fix, not in `pr180-window10`.

## Design, smallest first

**AC-0: measure with existing knobs, no code (F1 binary).** Arms:
- BASE.
- U: `tracking: (frame_velocity: PropagatedFromKeyframe)`.
- KI: `tracking_loss: (reseed: (keep_inertial: true, min_seed_points: 20, min_cold_start_points: 20))`.
- KIU: both.
- HB, on the two VI-BA velocity windows only: KIU plus `mapping.local_ba.imu_huber: BoundaryOrRecentInit`.

`min_seed_points` defaults to 50 (`recovery.rs:70`) and must be set, or re-seeds stall silently.

**AC-1 (the owner's S1 and S2, merged): keep path with a held seed.** One new knob, `tracking_loss.reseed.seed: CoastedIfPlausible (default, today's keep path) | Held`.
- `Held` skips the 4 m/s plausibility branch in `give_up` (`system/mod.rs:2780-2804`). The seed is `recovery::held_at(coasted, before)` (`recovery.rs:227`): gyro attitude at the last tracked position, zero velocity, no bridge. Stalled re-seeds already keep following the gyro (`system/mod.rs:1304-1305`).
- `Held` keeps the inertial state only once VIBA2 has run (`!schedule.recently_initialized()`, `inertial/schedule.rs:64`; ORB-SLAM3 `Tracking.cc:2149` resets before BA2). Earlier give-ups take today's cold path.
- Everything else is existing code: gravity and bias kept, no VIBA0, IMU prediction and 5 s grace at once, tilt re-level, bias envelope.
- Why merged: an attitude-only restart still re-runs VIBA0, which rotates every older segment (finding 1) and leaves the young segment on VIBA0-level gravity (finding 2).
- References: Basalt (`src/vi_estimator/sqrt_keypoint_vio.cpp:77-108`), OpenVINS (`ov_msckf/src/state/Propagator.cpp:33`) and OKVIS2 (`okvis_multisensor_processing/src/ThreadedSlam.cpp:627-633`) never drop the inertial state. ORB-SLAM3 starts a new Atlas map with a new IMU init (`Tracking.cc:2662-2700`) and relies on `MergeLocal2` (`LoopClosing.cc:1783`); its `Map::ApplyScaledRotation` (`Map.cc:252-264`, called at `LocalMapping.cc:1283`) touches the current map only, the property ours lacks.

**AC-2 (the owner's S3): velocity integrity.**
- 2a, existing knob: `frame_velocity: PropagatedFromKeyframe` (ORB-SLAM3 `Tracking.cc:1746-1758`). Scored in AC-0.
- 2b, built only if AC-0 shows separation: an innovation gate on the tracked pose, new `tracking_loss.pose_gate: Innovation` variant. Statistic: NIS between the PnP pose and the IMU prediction, S = `predicted_covariance` (`system/estimator.rs:265-323`) plus the PnP 6×6 covariance from the analytic refine's normal matrix. Threshold: χ²₆ at 99.9 % times a running consistency factor (median NIS over recent tracked frames / 5.35). Reference: OpenVINS `ov_msckf/src/update/UpdaterMSCKF.cpp:215-229`; beats VINS-Fusion `failureDetection` (`estimator.cpp:955-999`, hand-set) and ORB-SLAM3 `mbBadImu` (`LocalMapping.cc:136-145`, a reset).
- 2c: `seed: ByConsistency`. Coast when innovations up to the loss were consistent, hold otherwise; take card AB's IMU `Quiet` verdict as the zero-velocity case.

**AC-3 (the owner's S4): design only in this card.** With AC-1 the seed is the 4-DoF link (yaw from the gyro, position held). For offline BA and the pose graph, add a bridge edge with rotation from gyro preintegration and a loose zero-mean translation (VINS-Fusion `loop_fusion/src/pose_graph.cpp:434,519` `FourDOFError`; ORB-SLAM3 `Optimizer.cc:5292`). Today's `bridge_segments` uses a full IMU edge, which carries the velocity error. Hook: `KeyframeGap::Reseeded` (`system/mod.rs:2912-2921`). Decide after the offbridge result and AC-1 sessions.

## Measurement

Windows: derive each from `/home/edgar/dev/split/.pr180-fix/bench-F1/cfg/<seq>.ron` by replacing `max_frames: 100000` with `start_frame: N, max_frames: M` (`apps/kornia-slam-app/src/source/lamaria.rs:58-61`). 20 Hz; each window has a 30 s lead for VIBA2.

| Window | Seq | start, frames | Covers |
|---|---|---|---|
| W211a | 2_11 | 2800, 1300 | 171 s turn on the spot (5.05 s loss), restart, 142 m jump at 178 s |
| W211b | 2_11 | 6900, 1500 | losses at 375 and 403 s, young deaths at 382, 385, 409 s, jumps of 358 and 731 m |
| W211c | 2_11 | 20800, 1100 | frames 21400-21900 |
| W119 | 1_19 | 8000, 4400 | 436 s coast, 565-600 s runaways, restart at 605 s |
| W318 | 3_18 | 23000, 5000 | 1212 s wall event, restarts at 1264 and 1355 s |
| W511 | 5_11 | 18600, 1300 | 11.4 s coast at 961 s, restart at 975 s |
| BK | R_01 | 0, 1000 | `--blackout 25:6` (give-up) and `25:2` (coast only); config `bench-F1/cfg/r01full-preset.ron`; GT-backed loss test (`apps/kornia-slam-app/src/blackout.rs`) |

New `[summary] loss` line: `later_alignments`, `later_alignment_deg_max`, `later_alignment_jump_m_max`, `seeds_gyro_held`, `seed_rotation_deg_max`, `young_segment_deaths` (segment under 3 s old), `keeps_refused_before_viba2`, `coast_m_at_give_up_max`, later `pose_innovation_rejects`. Existing: `coast_runaway_frames`, `lost_keyframes_at_runaway`, `reseeds`, `reseeds_held`, `reseed_stall_frames`, `bias_resets`.

Scoring: the official CP score is not comparable below about 25 on a fragmented trajectory. Report per run, never rescale the official number: (1) official CP and R@1m (`bench-F1/val/cpscore.sh`); (2) CP oracle, restarts linked and 60 s pieces (`diagV/drift/oracle.py`, `evalcp.sh`); (3) per-segment Sim3 RMS as share of path and scale where pseudo-dense GT exists (R_11, 1_19, 3_18); (4) link errors: rotation and position error across each give-up. A prediction on official CP counts only where the expected value is above 25; below that, K1 and K2 read the oracle gap closure, (official − as run) / (linked − as run).


## Results

### AC-0, round 1 (2026-11-22, F1 binary md5 52e38c464d4369df0c9d3f61df17b5aa, BASE only)

Runs: `$AC/out/<window>-BASE-r1` (DEBUG=0) and `-BASEd-r1` (DEBUG=1), `PACE=timestamps`, `--trajectory-include-coasted
--session-out`, scored by `$AC/score.sh` (summary, `worldrot.py jumps`, `lossev.py`). Load every 30 s in
`$AC/load/q0.txt`: 1-min load median 25.6, range 19-40 on 12 cores, swap full; the F1 validation ran at median 2.5
(max 6.6). Frontend p50 here 31-108 ms against 10.5 ms in the F1 run (`sequence_1_19-r1`), so the paced runs fell
behind real time.

| Window | F1 event | BASE: give-ups (startup/young/primary), tracked | BASEd | frontend p50 ms | R0 |
|---|---|---|---|---|---|
| W211a | 171 s turn, 5.05 s loss, restart | 2 (2/0/0), 1204/1300; no loss at 171 s | 2 (2/0/0), 1193 | 108 / 105 | miss |
| W211b | give-ups 375, 403 s | 0, 1335/1500; losses 356 s 1.3 s, 370 s 1.4 s only | 0, 1362 | 89 / 81 | miss |
| W211c | frames 21400-21900 (1070-1095 s) | 13 (0/12/1), 541/1100; primary at 1079.9 s after a 5.1 s loss, after VIBA2 | 17 (0/16/1), 511 | 83 / 80 | holds |
| W119 | 436 s coast, 565-600 s runaways, 605 s restart | 7 (7/0/0) at 401-414 s, 3931/4400; no runaway (0 keyframes over 4 m/s) | 7 (7/0/0), 3968 | 99 / 103 | miss |
| W318 | 1212 s wall, restarts 1264, 1355 s | 2 (0/1/1), 4559/5000; primary give-up, segment at 1346.5 s | 0, 4723 | 43 / 39 | holds (BASE), not BASEd |
| W511 | 11.4 s coast at 961 s, restart 975 s | 2 (0/1/1), 1040/1300; 8.8 s loss at 964 s, segment at 972.8 s | 2 (0/1/1), 1034 | 36 / 31 | holds |

M0e (angle of a later segment's VIBA0 gravity to +Y against the origin jump): only W511-BASEd has a later VIBA0 in a
DEBUG run. Gravity (-7.917, 0.426, 5.777), 87.5 deg to +Y; the single-frame origin jump at 45.8 s (5.3 m, distance
from the origin 26.7 m kept) has an orientation change of 88.7 deg: 1.2 deg apart, so M0e holds on its one sample.
The rotation about the origin maps the position before to the one after within 1.25 m. The prediction's "jump
angle" (the angle between the two positions about the origin) reads 11.5 deg there: it only bounds the rotation from
below (equal when the axis is normal to the position), so `worldrot.py jumps` now also prints the orientation change.
On the F1 5_11 run the two origin jumps rotate 86.5 and 55.1 deg (section 1 lists the position angles 61.1 and 43.6).
W318-BASE (DEBUG=0) shows the same mechanism: a 158.3 m origin-preserving jump at 1356.8 s, orientation change
89.4 deg.

Also seen: every window's first VIBA0 sits 72-106 deg off +Y (the bootstrap world is the camera frame), so the
first alignment always rotates the young map a lot; that is the first segment and harmless. The R_01 identity run
reports `later_alignments=3` (max 90.4 deg, jump 0.317 m): R_01's start-up reset (G31) makes the first real segment
segment 1, so its VIBA0/1/2 count as later alignments. The counter follows the plan's definition
(`segments.latest() > 0`); the jump column separates the cases (0.3 m against 158-424 m).

Not run: the U, KI, KIU and HB arms, the BK blackout pairs and the noise samples (P0u, P0k, P0s, P0b, M0a, K5).
Stop point: K6 fired, so step 5 (AC-1) was not built.

### K6 diagnosis (owner rule: why it missed)

1. **The load, first.** The windows ran at 8-16x the F1 validation's load. With `--pace-timestamps` a slow frame is
   not dropped, it is late: frontend p50 31-108 ms against 10.5 ms means 1.5-5 frames of source time per frame of
   work, so local mapping and VI-BA get several times more wall time per keyframe than in the F1 run, and keyframe
   decisions shift. Evidence that the run depends on timing: W318 BASE and BASEd, same window and binary, give 2 and
   0 give-ups (4559 against 4723 tracked); W211c gives 13 and 17. A deterministic-pairing card cannot be scored on
   these runs, so the misses are not evidence against the windows yet.
2. **Mid-sequence starts, second.** W211a, W119 and W211b start with a cascade of start-up give-ups (2-7 in the first
   14 s, while VIBA0 is still unsolved); the 30 s lead is spent on that, and the map at the event is younger and
   different from F1's. W211c, W318 and W511, whose events lie 40-200 s after the window start, reproduce.
3. **K6's remedy, run (BASE, F1 binary).** 60 s lead: W211aL60 (start frame 2200, 1900 frames; load median 17,
   frontend p50 25.5 ms): 3 start-up give-ups at 110-118 s, then losses at 137.6 s (1.5 s) and 143.3 s (4.5 s), no
   loss at 171 s; miss. W119L60 (7520, 4880; frontend p50 56.7 ms): 0 give-ups, no runaway, losses at 444 s (2.2 s)
   and 563 s (1.25 s) only; miss. W211bL60 did not start (config renamed under a queued runner; exit 1). Full
   prefix: W211aP (frames 0-4100, load median 7.6, frontend p50 15.0 ms, the closest to F1's 10.5): 13 start-up
   give-ups in the first 10 s, then a primary give-up after VIBA2, a 5.45 s loss at 120.6 s and a segment at 126.0 s,
   with an origin-preserving jump at 127.3 s (31.1 m, distance from the origin 41.9 m kept, orientation change
   49.0 deg); no loss at 171 s. Coasts at 3-5 s: 15.4 m median error at 4.46 m/s SLAM against 0.12 m/s GT.
   R0 misses at full prefix too: the run reaches the same mechanism (a primary give-up after VIBA2, then the later
   alignment rotating the world) but 45 s before F1's event.
4. **Verdict.** K6 fired and its remedy is exhausted on W211a (30 s, 60 s, full prefix) and W119 (30 s, 60 s). Runs
   of the same binary and window do not repeat (W318 2 against 0 give-ups, W211c 13 against 17), and a window's
   trajectory diverges from F1's within seconds, so "within 10 s of the F1 event time" cannot be met by any window
   while the system is this timing-sensitive. Four runs do produce a scoreable event, a primary give-up after VIBA2
   (W211c 1079.9 s, W318 1346.5 s, W511 972.8 s, W211aP 126.0 s), each followed by the world rotation (W318 89.4 deg,
   W511 83.8-88.7 deg, W211aP 49.0 deg). Whether to score AC-0 on those runs' own events instead of F1's times is
   the owner's call (it changes R0). Per the stop rule AC-1 was not built.
