# Diagnosis of the first LaMAria validation run (F1, 2026-11-22)

Run: tag `pr180c/F1`, preset `pr180-window10`, six training challenge sequences, one sample each, paced, factory-rectified IMU, clock offset 0. Mean CP 11.0 online, 10.8 offline BA (ROADMAP 2026-11-22). Outputs: `/home/edgar/dev/split/.pr180-fix/bench-F1/val/`. Read-only analysis, no SLAM runs; scripts in `$S/diagV/drift/` (session parser, piece alignment, event extraction, CP oracle).

## Drift (1_19, 3_18, 5_11, 2_11 between give-ups; R_11 control)

**The long sequences do not drift smoothly. Each has a short catastrophic event:** weak tracking, then a wrong VI-BA velocity, then a coast that runs away, then give-ups, then a restart whose world is rotated about the world origin. One global Sim3 cannot fit the pieces. Between events the trajectories hold 0.3-0.6 % of path RMS, comparable to or better than R_11 (0.8 %).

| Seq | Dur / path | ATE SE3 / Sim3 | CP on / off | Dominant error | Main event(s) |
|---|---|---|---|---|---|
| R_11 | 8 min / 475 m | 4.75 / 3.82 m | 55.5 / 56.1 | smooth; 10 s chord energy along 61 %, cross 32 %, vertical 7 % | none (2 give-ups at t < 1 s, coasts ≤ 2.5 s) |
| 1_19 | 15.3 min / 1004 m | 60.6 / 53.2 m | 4.84 / 3.74 | steps; [0,436) s fits to 2.96 m Sim3 (0.60 %), [610,918) s to 0.97 m | 436 s: 2.6 s coast at 16-43 obs/kf, then 1.45× speed for 40 s; 565-600 s: runaway coasts (est 73-128 m in 4-5 s, GT 0.5-3 m; keyframe \|v\| 5-16 m/s vs GT 0.2-0.5), 11 give-ups, new segment rotated 9.9° |
| 3_18 | 28.2 min / 1699 m | 209 / 179 m | 0.42 / 1.67 | steps; [0,1212) s (1221 m) fits to 3.73 m Sim3 (0.31 %) | 1212 s: 3.9 s coast facing a close wall (depth 1.4 m), est 10 m vs GT 0.3 m; 30 s at 6× speed with healthy tracking (VI-BA \|v\|/v_gt 4.96); restarts at 1264 s (world rotated 161° about vertical) and 1355 s (67°); the 617 m jump equals 2·557·sin(33.5°) |
| 5_11 | 18.7 min | no pseudo-dense GT | 5.39 / FAIL | steps (CP errors 2.7-37 m) | 11.4 s coast at 961 s (obs 0-4); restart at 975 s rotated 61° about the origin at \|p\| 416 m (jump 425 m, predicted 422) |
| 2_11 | 19.8 min | Sim3 failed | FAIL / 3.26 | runaways inside segments: 126 s above 3 m/s, about 6.7 km of fake path | 62 segment runs, 14 low-obs episodes ≥ 4 s |

**Ranked causes:**
1. **Weak tracking gives a wrong VI-BA velocity, then a coast runaway.** Keyframes before runaway coasts have \|v\| 5-16 m/s against GT 0.2-0.5 m/s (1_19), with 0-25 observations. Accel bias there is only 0.03-0.09 m/s², so velocity, not bias, drives it. Velocity failures (\|v_kf − v_gt\| > 3 m/s) on keyframes with < 10 observations: 52 % (1_19), 22 % (3_18), 31 % (2_11). Low-observation keyframes see near-field scenes (median depth 1.6-2.1 m against 4.5 m). Confirm: DEBUG=1 replay of 1_19 400-620 s and 3_18 1150-1400 s; read the VI-BA lines at keyframes with \|v\| > 3 m/s.
2. **A restart is not linked to the old world.** The new world is rotated about the world origin: the position jumps by 2\|p\|sin(θ/2) for θ = 10°, 61°, 47°, 67°, 161°. Candidate: `Map::apply_inertial_alignment` (`mapping/map/ops/correction.rs:460`), a world transform about the origin. Not verified, since gravity-only alignment should not create yaw.
3. **The map is at the wrong scale after a coast:** 3_18 runs 30 s at 6× with healthy tracking (landmark depth 1.4 to 12.4 m); a 10-keyframe window spans about 1.5 s and cannot recover it.
4. **Smooth along-track scale and heading drift** (yaw 1.2-3.3° per minute, scale wanders 3-11 % per minute). Candidates: clock offset + all-edge Huber (card AA), W2 origin dependence (card Y). Worth roughly 32-49 to 76-94 points on the oracle below.
5. **Offline BA is damaged by runaway seeds.** The initial cost is inertial (8e10 on 1_19 against 1.2e8 on R_11; seed visual residuals median 1.3 px). With `imu_kernel: None`, the first 8 iterations pull the geometry towards the runaway states, then the 3 px gate at iteration 8 drops 46-76 % of observations (seed: 17-20 % above 3 px). It improves segments with a clean inertial seed (R_11 pieces 1.25 to 0.55 m) and wrecks those with runaway keyframes (1_19 segment 3: 2.96 to 11.2 m). The 60-iteration cap is not the limit. Fragmentation (7-45 components, `bridged=0`) is a second limit.

**Score at stake (CP oracle: each piece aligned to GT separately, official evaluator):**

| Seq | As run | Restarts linked | Every 60 s piece |
|---|---|---|---|
| 1_19 | 3.2 | 38.8 (48.9 finer) | 76.4 |
| 3_18 | 18.9 | 32.1 (45.0 at 300 s pieces) | 93.8 |
| R_11 | 55.5 | 56.3 | 96.8 |

**Known suspects:**

| Suspect | Verdict | Evidence |
|---|---|---|
| Clock offset + Huber (card AA) | not the cause of the gap; open for the smooth drift | offsets after Hold are similar everywhere (R_11 −4.54, 1_19 −4.16, 2_11 −4.38, 3_18 −4.76 ms), and R_11 scores 55 with the same offset |
| W2 origin dependence (card Y) | not dominant; unresolved | 3_18 [0,1212) s at 380-610 m from the origin holds 0.31 % of path; distance from the origin and time are collinear (partial correlation 0.07); needs an origin-shift run |
| Bias | not the driver of the runaways | \|ba\| ≤ 0.09-0.20 m/s² at runaway onsets while the velocity error is 4.5-16 m/s |
| Fragmentation | supported; dominant for the score | oracle linking: 1_19 3.2 to 39-49, 3_18 18.9 to 32-45 |
| Coasted poses in the scored trajectory | ruled out | scoring without them: no sequence improves; 5_11 gets worse |

**Evaluation artefact (new, unverified cause).** The official CP score is not invariant to a global scale of the estimate: scaling the 3_18 estimate about its origin gives 0.42 at ×1, 24.4 at ×0.8, 17.7 at ×0.567 and 1.1 at ×1.25; rotation-only or SE3 transforms leave it at 0.416, and R_11 is unaffected. On a fragmented trajectory, scores below about 25 are not comparable between runs. Suspect: the robust Sim3 initialisation threshold (`max_error 5`) is applied in the estimate's units.

**Next mechanism (proposed):** keep one consistent world through a tracking loss. Hold the IMU state through weak vision instead of accepting visual velocities that contradict it, bridge short losses with the gyro, and merge the remaining maps (ORB-SLAM3 RECENTLY_LOST IMU-only tracking up to 5 s, Atlas map merging; Basalt's marginal prior for velocity and scale through weak frames). A first step without a SLAM run: offline BA with `bridge_segments: true` and `imu_kernel: Huber` on the existing sessions.

## Tracking loss (2_11, 4_11) and offline-BA convergence

Running (second diagnosis).
