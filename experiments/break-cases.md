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

| id | sequence, span | stage | symptom today | pass condition | tier | status |
|---|---|---|---|---|---|---|
| BC-01 | 1_19, frames 0-13 | start-up, PnP prior gate | 3.0-4.5 deg/frame; PnP median 29.7 / 30.2 px against the 25 px gate; 3 resets | prior residual at frames 3 and 13 <= B-px; no reset (gyro reference, no GT there) | T1 mini-trace, 4 frames | red |
| BC-02 | 1_19, 436-448 s | LK tracks | 19-42 observations per keyframe at 160-186 deg/s; lost keyframe at 437.4 s | B-lk on 3 frames; survivors >= OpenCV's at production settings; report the in-view ceiling | T1 (3 frames + golden), T2 240 frames | unknown, stage 0 decides |
| BC-03 | 1_19, 565-600 s | VI-BA velocity, weak vision | keyframe speed 5-16 m/s against GT 0.2-0.5 at 0-25 observations | \|v_kf - v_gt\| <= B-vel | T1 window slice, T2 chain | red if the slice reproduces |
| BC-04 | 3_18, 1212-1250 s | coast, then scale | 3.9 s coast: 10 m against GT 0.3 m; then speed ratio 4.96 for 30 s | coast error <= B-pos(3.9 s); speed within B-vel | T1 IMU slice + GT, T3 | red |
| BC-05 | 5_11, 961-975 s | coast, give-up | 11.4 s coast, 0-4 observations; restart rotated 86.5 deg | rotation across the gap within B-rot(11.4 s) of the gyro chain | T2 (IMU span over the committed cap) | red |
| BC-06 | 2_11 near 1075-1080 s; 4_11 primaries | give-up attitude | restart at the old attitude: 132.3 deg (2_11, n=8), 83.0 deg (4_11, n=9) | within B-rot(5 s) of the gyro chain; <= 5 deg of GT | T1 real gyro slice + synthetic frames through `SlamSystem` | red |
| BC-07 | 2_11 (36), 4_11 (115) young deaths | re-seed, search radius | segment dies the frame after its seed; 1.31-1.53 rad/s = 16-18 px per frame against the 15 px radius | next frame tracked at or above the inlier gate; prediction error <= B-px | T1 mini-trace, seed + next | red |
| BC-08 | 4_10 (segments 16, 22), 2_12, 4_11; H0 arm | bias after a kept re-seed | \|bg\| 0.3-1.4 rad/s, \|ba\| up to 4.6 m/s^2; healthy 0.04 | B-bias from the last vision-anchored reference | T2 session invariant | red only on `feat/loss-one-world`; guard on head |
| BC-09 | every sequence; fixture 1_19 60-70 s | bias walk at the window boundary | z p50 10.3-12.0 gyro, 5.8 accel (`vi_ba_schur.rs:864-875`) | B-walk | T1 window slice, 3-5 chained solves | red |
| BC-10 | 1_19, 330-450 s | accel bias hides tilt | \|ba\| 0.5-1.3 m/s^2 for 100 s, tilt 6-14 deg | B-bias accel | T2 invariant, T3 | red |
| BC-11 | 1_19, window 340:500 or 300:470 | heading and speed event | yaw step -27 to -35 deg at 449 s; 3-5 m/s against GT 1.7; window ATE 10.4-18.1 m | yaw step <= 3 sigma of the piece fit (3.9-4.6 deg); speed within B-vel | T3 | red; a cold window may miss it |
| BC-12 | 1_20, 380:540 | scale event | scale 1.21-1.33, ATE 6.3-7.3 m | piece scale within 3 sigma of the piece-fit noise | T3 | red |
| BC-13 | 1_19, 485-495 s and 520-540 s | unknown | r2 arms track 0-17 of 50-100 frames; r1 tracks all | not derivable until the stage is known; stage 0 traces 450:560 | T3 probe | open row |
| BC-14 | 2_11, W211c (frames 20800 + 1100) | PnP acceptance | accepted poses jump 5.98 m and 6.36 m in 50 ms with 16-47 inliers; 3.13 to 6.87 m/s against GT 0.50 | accepted pose NIS against the IMU prediction <= 22.458 (chi-square 6-dof 99.9 %), else the frame coasts | T1 mini-trace, T3 | red |
| BC-15 | R_11, 329-335 s | gyro bias in a fast turn, healthy vision | bg 48e-3 rad/s off its median at 94-126 deg/s with 138-248 observations | B-bias gyro | T1 window slice | red |
| BC-16 | 1_19 session 560-610 s; 5_11 | global join | bridge residual 12 sigma (1_19), 153 sigma (5_11); R_11 0.25 sigma | bridge residual <= 3 sigma; one component | T2 `offline-ba` on a slice | red |
| BC-17 | R_11 frames 2-6; R_01 frames 2, 4 | start-up match starvation (G31) | matched 4 and 9 (R_11), 2 and 7 (R_01) against the 15-match gate | matched >= gate | T1 mini-trace, pinhole and raw fisheye | red on pinhole |

## 4. Red today, and the rule

- **Registry, not `#[ignore]` or `#[should_panic]`.** One `const CASES` table in the test-support module: `(id, arm) -> Open | Fixed`.
  - `Open` asserts the pass condition is violated. If the break stops reproducing, the test fails with "BC-xx no longer reproduces: flip to Fixed in the commit that fixes it".
  - `Fixed` asserts the pass condition.
  - A fix behind a knob adds a `(id, knob arm) -> Fixed` row; the default arm stays `Open` until the knob is promoted.
- **Rule**: a fix names its BC id; its commit flips that row; no test and no row is removed; a bound is never edited after a run.
- **Gates.** BC-01, 07, 17: start-up and prediction. BC-02: LK kernel and `RigFlow`. BC-03, 09, 15: local VI-BA. BC-04, 05, 06: coast and loss handling. BC-14: pose acceptance. BC-08, 10: bias model. BC-16: global VI-BA.
