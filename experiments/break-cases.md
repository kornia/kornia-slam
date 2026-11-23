# Break-case catalogue

Each known LaMAria break becomes a small test whose pass condition comes from ground truth (GT) and the IMU and camera model, so a fix is checked in seconds.
Three tiers: T1 committed snippets in CI, T2 on-demand replays from the data directory, T3 windows of at most 5 minutes.
One rule: a fix lands only with its case flipping to `Fixed`.

Base: `feat/global-pose-graph` (PR #180 head).

## 1. Bounds (sensor model, none tuned)

From `pinhole_calibrations/<seq>.json` imu0. 1_19: sigma_g 1.745e-4 rad/s/sqrt(Hz), sigma_a 7.845e-4 m/s^2/sqrt(Hz), sigma_bg 2.444e-4 rad/s/sqrt(s), sigma_ba 6.448e-3 m/s^2/sqrt(s), f 242.06 px, dt 0.05 s.
All at 3 sigma, or chi-square 99.9 % (the level the owner accepted for AC-2b).

- **B-rot(T)** = 3 sqrt(sigma_g^2 T + sigma_bg^2 T^3/3 + (sigma_b0 T)^2 + (omega_max sigma_t)^2).
  sigma_b0 = robust SD of the gyro bias over vision-anchored keyframes before the event.
- **B-bias(dt)** = 3 sigma_w sqrt(dt) from the reference (median over vision-anchored keyframes).
- **B-walk**: z = |db| / (sigma_w sqrt(3 dt)) per consecutive keyframe pair; p50 <= 1.94 (chi-square 3-dof 99 % point; expected median 0.89).
- **B-vel(T)** = 3 sqrt(sigma_vgt^2 + sigma_a^2 T + sigma_ba^2 T^3/3 + (g sigma_theta(T) T)^2). sigma_vgt from GT differencing.
- **B-pos(T)**: NEES <= 16.27 (chi-square 3-dof 99.9 %) against the system's own covariance when it reports one;
  else 3 sqrt((sigma_v0 T)^2 + sigma_a^2 T^3/3 + sigma_ba^2 T^5/20 + (g sigma_theta0 T^2/2)^2).
- **B-px** = f v_gt,max dt / d_med + 3 f sqrt(sigma_g^2 dt + (sigma_b0 dt)^2). It must also sit under the active gate.
- **B-lk**: status equal and position within 0.01 px of OpenCV, the port's existing tolerance.

## 2. Catalogue

Status is for the PR head. "Red" means the break reproduces today.

| id | sequence, span | stage | symptom today | pass condition | tier | today (stage 0) | bound (stage 0) | status |
|---|---|---|---|---|---|---|---|---|
| BC-01 | 1_19, frames 0-13 | start-up, PnP prior gate | 3.0-4.5 deg/frame; PnP median 29.7 / 30.2 px against the 25 px gate; 3 resets | prior residual at frames 3 and 13 <= B-px; no reset (gyro reference, no GT there) | T1 mini-trace, 4 frames | not recomputed: the saved logs hold no PnP residuals; needs a `--trace-out` capture (S2) | rotation term 0.05-0.16 px + 15.7 px m / d_med (d_med from the trace) | red |
| BC-02 | 1_19, 436-448 s | LK tracks | 19-42 observations per keyframe at 160-186 deg/s; lost keyframe at 437.4 s | B-lk on 3 frames; survivors >= OpenCV's at production settings; report the in-view ceiling | T1 (3 frames + golden), T2 240 frames | F1: 10-226 observations per keyframe (median 71), 13 of 43 lost; W119: 19-292 (median 42), 10 of 46 lost; omega_max 219 deg/s | B-lk needs our kernel on the frames (stage 3) | unknown, stage 0 decides |
| BC-03 | 1_19, 565-600 s | VI-BA velocity, weak vision | keyframe speed 5-16 m/s against GT 0.2-0.5 at 0-25 observations | \|v_kf - v_gt\| <= B-vel | T1 window slice, T2 chain | F1 full run: 51 of 62 keyframes over; worst 14.71 m/s against GT 0.32 at 588.65 s. W119 window: 0 of 167 over, max 0.48 m/s | B-vel 0.68-0.70 m/s (T 0-0.15 s since the last anchor) | red if the slice reproduces |
| BC-04 | 3_18, 1212-1250 s | coast, then scale | 3.9 s coast: 10 m against GT 0.3 m; then speed ratio 4.96 for 30 s | coast error <= B-pos(3.9 s); speed within B-vel | T1 IMU slice + GT, T3 | coast 1215.60-1219.45 s (78 frames): 10.21 m against GT 0.26 m, aligned error growth 9.86 m; path ratio 7.29 over 30 s | B-pos(3.9 s) 3.1-7.1 m (attitude 1-5 deg at 3 sigma) | red |
| BC-05 | 5_11, 961-975 s | coast, give-up | 11.4 s coast, 0-4 observations; restart rotated 86.5 deg | rotation across the gap within B-rot(11.4 s) of the gyro chain | T2 (IMU span over the committed cap) | not computed at S0 (T2) | - | red |
| BC-06 | 2_11 near 1075-1080 s; 4_11 primaries | give-up attitude | restart at the old attitude: 132.3 deg (2_11, n=8), 83.0 deg (4_11, n=9) | within B-rot(5 s) of the gyro chain; <= 5 deg of GT | T1 real gyro slice + synthetic frames through `SlamSystem` | W211c: give-up 1066.2 s, gap 7.65 s, SLAM 0.0 deg against gyro 56.0 deg. F1 full: 2_11 primaries 132.3 deg (n=8), 4_11 83.0 deg (n=9) | B-rot(7.65 s) 4.1 deg (W211c); 6.0-9.7 deg (F1, 5-12 s) | red |
| BC-07 | 2_11 (36), 4_11 (115) young deaths | re-seed, search radius | segment dies the frame after its seed; 1.31-1.53 rad/s = 16-18 px per frame against the 15 px radius | next frame tracked at or above the inlier gate; prediction error <= B-px | T1 mini-trace, seed + next | young deaths 36 (2_11), 113 (4_11), 16 in the W211c window; next-frame residuals need a capture (S2) | B-px needs the trace | red |
| BC-08 | 4_10 (segments 16, 22), 2_12, 4_11; H0 arm | bias after a kept re-seed | \|bg\| 0.3-1.4 rad/s, \|ba\| up to 4.6 m/s^2; healthy 0.04 | B-bias from the last vision-anchored reference | T2 session invariant | not computed at S0 (T2) | - | red only on `feat/loss-one-world`; guard on head |
| BC-09 | every sequence; fixture 1_19 60-70 s | bias walk at the window boundary | z p50 10.3-12.0 gyro, 5.8 accel (`vi_ba_schur.rs:864-875`) | B-walk | T1 window slice, 3-5 chained solves | F1 1_19 60-70 s: z p50 12.65 gyro, 7.98 accel (53 pairs); F1 whole run 9.80 / 5.64; W119 11.95 / 5.49 | p50 <= 1.94 | red |
| BC-10 | 1_19, 330-450 s | accel bias hides tilt | \|ba\| 0.5-1.3 m/s^2 for 100 s, tilt 6-14 deg | B-bias accel | T2 invariant, T3 | F1 1_19: \|ba - ref\| 30 s medians 0.28, 0.56, 0.20, 0.93 m/s^2 (330-450 s) | 0.18-0.26 m/s^2; 3 of 4 bins over | red |
| BC-11 | 1_19, window 340:500 or 300:470 | heading and speed event | yaw step -27 to -35 deg at 449 s; 3-5 m/s against GT 1.7; window ATE 10.4-18.1 m | yaw step <= 3 sigma of the piece fit (3.9-4.6 deg); speed within B-vel | T3 | not computed at S0 (T3) | - | red; a cold window may miss it |
| BC-12 | 1_20, 380:540 | scale event | scale 1.21-1.33, ATE 6.3-7.3 m | piece scale within 3 sigma of the piece-fit noise | T3 | not computed at S0 (T3) | - | red |
| BC-13 | 1_19, 485-495 s and 520-540 s | unknown | r2 arms track 0-17 of 50-100 frames; r1 tracks all | not derivable until the stage is known; stage 0 traces 450:560 | T3 probe | not computed at S0 (T3) | - | open row |
| BC-14 | 2_11, W211c (frames 20800 + 1100) | PnP acceptance | accepted poses jump 5.98 m and 6.36 m in 50 ms with 16-47 inliers; 3.13 to 6.87 m/s against GT 0.50 | accepted pose NIS against the IMU prediction <= 22.458 (chi-square 6-dof 99.9 %), else the frame coasts | T1 mini-trace, T3 | W211c-BASE-r1: largest 50 ms step 0.76 m (15.2 m/s against GT 1.57); other BASE arms up to 16.68 m. NIS needs a covariance (capture) | NIS <= 22.458 | red |
| BC-15 | R_11, 329-335 s | gyro bias in a fast turn, healthy vision | bg 48e-3 rad/s off its median at 94-126 deg/s with 138-248 observations | B-bias gyro | T1 window slice | F1 R_11: \|bg - ref\| max 1.72e-2, median 9.5e-3 rad/s, 28 of 34 keyframes over; D-R11-BASE 1.54e-2, 30 of 33 | 5.96e-3 rad/s (dt 66 s) | red |
| BC-16 | 1_19 session 560-610 s; 5_11 | global join | bridge residual 12 sigma (1_19), 153 sigma (5_11); R_11 0.25 sigma | bridge residual <= 3 sigma; one component | T2 `offline-ba` on a slice | not computed at S0 (T2) | - | red |
| BC-17 | R_11 frames 2-6; R_01 frames 2, 4 | start-up match starvation (G31) | matched 4 and 9 (R_11), 2 and 7 (R_01) against the 15-match gate | matched >= gate | T1 mini-trace, pinhole and raw fisheye | not recomputed: needs a `--trace-out` capture (S2) | 15-match gate | red on pinhole |

## 4. Red today, and the rule

- **Registry, not `#[ignore]` or `#[should_panic]`.** One `const CASES` table in the test-support module: `(id, arm) -> Open | Fixed`.
  - `Open` asserts the pass condition is violated. If the break stops reproducing, the test fails with "BC-xx no longer reproduces: flip to Fixed in the commit that fixes it".
  - `Fixed` asserts the pass condition.
  - A fix behind a knob adds a `(id, knob arm) -> Fixed` row; the default arm stays `Open` until the knob is promoted.
- **Rule**: a fix names its BC id; its commit flips that row; no test and no row is removed; a bound is never edited after a run.
- **Gates.** BC-01, 07, 17: start-up and prediction. BC-02: LK kernel and `RigFlow`. BC-03, 09, 15: local VI-BA. BC-04, 05, 06: coast and loss handling. BC-14: pose acceptance. BC-08, 10: bias model. BC-16: global VI-BA.

## Stage 0 (2026-11-23): values from saved runs

Python only, on runs already on disk. No SLAM run was made for this table.

### Sources

- **F1**: the `bench-F1` validation runs, full sequences, paced, default config, binary `kornia-slam-f1` (md5 52e38c46).
  Used for 1_19, 2_11, 3_18, 4_11 and R_11.
- **W119, W211c, D-R11-BASE**: card AC window runs, BASE arm (default config), binary `kornia-slam-ac` or `-ac2` from `feat/loss-one-world` (e9fe4ad / f48155a; card AC does not say which per run).
  Neither binary is the PR head; with its knobs off the code is expected to match head, but that was not checked.
  W119 is 1_19 frames 8000 + 4400 (400-620 s); W211c is 2_11 frames 20800 + 1100.
- Times are seconds since the sequence's first frame, as `start_sec` counts them.
- Give-up rotations and young deaths come from `lossev.py` (card AC); its 2_11 and 4_11 primary medians match the catalogue (132.3 and 83.0 deg).

### Choices this table makes

- **Vision-anchored keyframe**: at least 100 observations and not inserted while lost.
  Healthy spans hold 138-600 observations and weak ones 0-110, so 100 separates them; no bound was fitted to it.
- **sigma_b0**: robust SD (1.4826 MAD, axis mean) of the gyro bias over anchored keyframes in the 60 s before the span.
  1_19 565 s: 4.43e-3 rad/s (F1), 3.16e-3 (W119), against the plan's 1.16e-3. The keyframe biases carry the BC-09 walk, so this sigma_b0 is inflated (risk "sigma_b0 taken from corrupted keyframes").
  W211c: 0 (the window's 82 anchored keyframes before 1066 s hold one bias value), so the clock term alone sets B-rot there.
- **sigma_t**: 4.8e-3 s, the uncorrected gyro clock offset (runs used offset 0). At 4.9 rad/s it is 4.1 deg at 3 sigma and dominates B-rot(7.65 s).
- **sigma_vgt**: robust SD of successive GT difference velocities / sqrt 2, +/- 30 s around the span: 0.23 m/s (1_19), 0.24 m/s (3_18).
- **BC-03** compares speeds, \|\|v_kf\| - \|v_gt\|\|, a lower bound of \|v_kf - v_gt\| that needs no world alignment. T is the time since the last anchored keyframe.
- **BC-04** error: SE3 fit on the 10 s of poses before the coast (rms 0.17 m), error at the coast's end minus at its start.
  The 30 s speed ratio is path length against path length, so frame jitter raises it (catalogue: 4.96).
- **BC-15** reference: median gyro bias of the anchored keyframes in 269-329 s; dt from 269 s, the loosest choice.
- **BC-10** reference: median accel bias of the anchored keyframes in 270-330 s; dt to each bin's end.

### Verdicts

| id | today against bound | red |
|---|---|---|
| BC-01 | not recomputed; B-px rotation term 0.05-0.16 px, the translation term needs the trace's landmark depths | pending capture (S2) |
| BC-02 | observations collapse to 10-28 per keyframe at 219 deg/s; LK parity not measured | unknown (stage 3) |
| BC-03 | F1: 51 of 62 over, 14.71 against 0.70 m/s | red on the full run |
| BC-03 | W119 window: 0 of 167 over, 0.48 against 0.68 m/s | **not red: the window does not reproduce it** |
| BC-04 | 9.86 m against 3.1-7.1 m | red |
| BC-06 | 56.0 deg against 4.1 deg (W211c); 132.3 and 83.0 deg medians against 6.0-9.7 deg (F1) | red |
| BC-07 | 36 / 113 / 16 young deaths; per-frame residual not recomputed | red by count; bound pending capture |
| BC-09 | 12.65 gyro, 7.98 accel against 1.94 (F1 60-70 s) | red |
| BC-10 | 3 of 4 bins over, up to 0.93 against 0.26 m/s^2 | red |
| BC-14 | one-step speed 15.2 against GT 1.57 m/s; NIS not computable without a covariance | red by jump; NIS pending capture |
| BC-15 | 1.72e-2 against 5.96e-3 rad/s | red |
| BC-17 | not recomputed | pending capture (S2) |

Not computed at stage 0: BC-05, 08, 16 (T2) and BC-11, 12, 13 (T3).

Values that differ from the catalogue's symptom column, reported and not adjusted:
- BC-09 on the F1 whole run: gyro z p50 9.80, under the catalogue's 10.3-12.0 (those came from the gate H0 and W119 sessions).
- BC-14: the 5.98 m and 6.36 m jumps are from the W211c-KIU arm. The BASE arms show 0.76-16.68 m single steps instead.
- BC-15: the largest bias deviation is 1.72e-2 rad/s, not 48e-3, against this reference.
- BC-07: 113 young deaths on F1 4_11, not 115.

## EuRoC cases (BC-E)

Pending baseline: the rows stay pending until the EuRoC baseline of today's head (`.pr180-fix/euroc/RESULTS.md`) exists. EuRoC data stays T2 (InC-NC rights statement, below).

| id | sequence | stage | symptom today | pass condition | tier | status |
|---|---|---|---|---|---|---|
| BC-E1 | MH_03 | gyro bias | gyro-bias error (from the baseline) | derived once the baseline exists | T2 until the licence is settled | pending baseline |
| BC-E2 | V1_03 | tracking loss | loss (from the baseline) | derived once the baseline exists | T2 | pending baseline |
| BC-E3 | V2_03 | tracking loss | loss (from the baseline) | derived once the baseline exists | T2 | pending baseline |

**Licence.** The ETH Research Collection record of the EuRoC MAV datasets (https://doi.org/10.3929/ethz-b-000690084, https://www.research-collection.ethz.ch/handle/20.500.11850/690084) gives "Rights / License: In Copyright - Non-Commercial Use Permitted" (http://rightsstatements.org/page/InC-NC/1.0/).
The ASL dataset pages (https://projects.asl.ethz.ch/datasets/euroc-mav/) state no licence.
This is not CC BY. No EuRoC snippet is committed until the owner decides; EuRoC cases stay T2 (read from the data directory) until then.

## Robot cases (BC-R)

Robot data stays out of the public repository: T2 only.

| id | source | stage | symptom today | pass condition | tier | gates | status |
|---|---|---|---|---|---|---|---|
| BC-R1 | dark-room parked loss: `~/handoff/dark-room-2026-11-22/blackbox_20261122T204322Z_vio_lost.mcap` and `blackbox_20261122T214730Z_vio_lost.mcap` | parked loss, reset, bootstrap | 19 map drops in about 2 h: the 0.5 s give-up fires, then a reset, then a bootstrap on noise frames | 0 map drops; state StationaryHold; no bootstrap on noise frames | T2 (robot data) | card AI part (a) | red |

Clip format: 60 s each, ROS2 CDR.
Topics: `/xle0/head/{left,right}/image_rect/compressed` (about 13 Hz, luma about 7/255), `/xle0/imu/data_raw` (about 200 Hz), `/xle0/odom` (about 75 Hz, twist 0), `/xle0/vio_diag/status`.
