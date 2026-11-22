# Diagnosis of the first LaMAria validation run (F1, 2026-11-22)

Run: tag `pr180c/F1`, preset `pr180-window10`, six training challenge sequences, one sample each, paced, factory-rectified IMU, clock offset 0. Mean CP 11.0 online, 10.8 offline BA (ROADMAP 2026-11-22). Outputs: `/home/edgar/dev/split/.pr180-fix/bench-F1/val/`. Read-only analysis, no SLAM runs; scripts in `$S/diagV/drift/` (session parser, piece alignment, event extraction, CP oracle).

## Drift (1_19, 3_18, 5_11, 2_11 between give-ups; R_11 control)

**The long sequences do not drift smoothly. Each has a short catastrophic event:** weak tracking, then a wrong VI-BA velocity, then a coast that runs away, then give-ups, then a restart whose world is rotated about the world origin. One global Sim3 cannot fit the pieces. Between events the trajectories hold 0.3-0.6 % of path RMS, comparable to or better than R_11 (0.8 %).

| Seq | Dur / path | ATE SE3 / Sim3 | CP on / off | Dominant error | Main event(s) |
|---|---|---|---|---|---|
| R_11 | 8 min / 475 m | 4.75 / 3.82 m | 55.5 / 56.1 | smooth; 10 s chord energy along 61 %, cross 32 %, vertical 7 % | none (2 give-ups at t < 1 s, coasts ≤ 2.5 s) |
| 1_19 | 15.3 min / 1004 m | 60.6 / 53.2 m | 4.84 / 3.74 | steps; [0,436) s fits to 2.96 m Sim3 (0.60 %), [610,918) s to 0.97 m | 436 s: 2.6 s coast at 16-43 obs/kf, then 1.45× speed for 40 s; 565-600 s: runaway coasts (est 73-128 m in 4-5 s, GT 0.5-3 m; keyframe \|v\| 5-16 m/s vs GT 0.2-0.5), 11 give-ups, new segment rotated 9.9° |
| 3_18 | 28.2 min / 1699 m | 209 / 179 m | 0.42 / 1.67 | steps; [0,1212) s (1221 m) fits to 3.73 m Sim3 (0.31 %) | 1212 s: 3.9 s coast facing a close wall (depth 1.4 m), est 10 m vs GT 0.3 m; 30 s at 6× speed with healthy tracking (VI-BA \|v\|/v_gt 4.96); restarts at 1264 s (world rotated 161° about vertical) and 1355 s (67°); the 617 m jump equals 2·557·sin(33.5°) |
| 5_11 | 18.7 min | no pseudo-dense GT | 5.39 / FAIL | steps (CP errors 2.7-37 m) | 11.4 s coast at 961 s (obs 0-4); restart at 975 s rotated 61° about the origin (position angle, a lower bound; orientation change 86.5°, see Corrections) at \|p\| 416 m (jump 425 m, predicted 422) |
| 2_11 | 19.8 min | Sim3 failed | FAIL / 3.26 | runaways inside segments: 126 s above 3 m/s, about 6.7 km of fake path | 62 segment runs, 14 low-obs episodes ≥ 4 s |

**Ranked causes:**
1. **Weak tracking gives a wrong VI-BA velocity, then a coast runaway.** Keyframes before runaway coasts have \|v\| 5-16 m/s against GT 0.2-0.5 m/s (1_19), with 0-25 observations. Accel bias there is only 0.03-0.09 m/s², so velocity, not bias, drives it. Velocity failures (\|v_kf − v_gt\| > 3 m/s) on keyframes with < 10 observations: 52 % (1_19), 22 % (3_18), 31 % (2_11). Low-observation keyframes see near-field scenes (median depth 1.6-2.1 m against 4.5 m). Confirm: DEBUG=1 replay of 1_19 400-620 s and 3_18 1150-1400 s; read the VI-BA lines at keyframes with \|v\| > 3 m/s.
2. **A restart is not linked to the old world.** The new world is rotated about the world origin: the position jumps by 2\|p\|sin(θ/2) for θ = 10°, 61°, 47°, 67°, 161°. These θ are the angle of the position about the origin, a lower bound on the rotation; on 5_11 the orientation changes are 86.5° and 55.1° (see Corrections). Candidate: `Map::apply_inertial_alignment` (`mapping/map/ops/correction.rs:460`), a world transform about the origin. Not verified, since gravity-only alignment should not create yaw.
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

Scripts in `$S/diagV/track/`. DEBUG=0, so no per-frame reject reasons; built from frame status lines, session segment ids, per-keyframe observation counts, GT, IMU and images. Run facts: `keep_inertial: false` (`reseeds=0`), no loop closer (`relocalization_attempts=0` on all six), `gate_rejects=0`, `pose_inertial_solves=0`.

**Loss path.** Visual estimate fails (PnP gate 10 inliers `pnp.rs:57`; 15 with a trusted IMU `tracker.rs:781-788`; reference fallback 15 matches `map_projection/mod.rs:455`; search radius 15 px `:96`, widened ×4 only with an IMU) -> coast -> `gives_up()` (`system/mod.rs:3537`: 5 s with an IMU initialised for 2 s, else 0.5 s; a map with ≤ 10 keyframes resets at once) -> `give_up()` `keep=false` -> `reset_after_loss()` (`state.rs:158-162`): the next segment is seeded at the last tracked pose, **its rotation included**, with zero velocity and no IMU initialisation -> `bootstrap_stereo` needs 20 points.

| Give-up class | 2_11 | 4_11 | Evidence |
|---|---|---|---|
| Start-up, map ≤ 10 keyframes | 13 | 1 | first 9 s of 2_11; frame 0 saturated |
| **Young-segment death loop** (segment < 3 s old) | **36 (59 %)** | **115 (86 %)** | dies at the frame after its seed in 22/36 and 75/115; exactly 10 coast frames = the 0.5 s visual grace; seeds that die rotate 1.53 / 1.31 rad/s (survivors 0.44 / 0.68), about 16-18 px per frame against the 15 px radius |
| **Primary: turn on the spot** (segment ≥ 3 s, 5 s coast expires) | **12** | **18** | gyro turn over the gap median 130° / 118° while GT moves 0.5-0.6 m; gyro agrees with GT within 0.6-4.5°; lost keyframes hold a median 10 / 7 observations against 331 |

- **Every give-up throws the heading away:** the relative rotation across a give-up is wrong by a median 131° (2_11) and 119° (4_11), because the new segment starts at the pre-loss attitude (SLAM rotation across the gap ≈ 0°, so the error equals the turn; this number does not depend on the body-frame and GT-sampling bugs in Corrections). That breaks the Sim3: the best segment holds 4 of 17 control points on 2_11 and 2 of 15 on 4_11 (a Sim3 needs 3).
- **Recovery:** a median 8.9 s from a primary loss to a segment that survives 5 s (max 35 s), a median of 3 seeds (up to 24).
- **Low light (4_11, tracked-frame mean 25 against 78) is rarely the first cause;** it deepens the cascade (115 young deaths against 36).
- **Coasts hurt translation after 0.5 s:** coast error at 3-5 s is 6.9-7.1 m against 0.6 m for "hold still" (coast better in 0 % of frames). Coasted speed 1.2-1.9 m/s while the true speed is 0.1-0.35 m/s. `coast_runaway_frames` 829 / 1467; keyframes faster than 4 m/s 393 / 513 (max 850 m/s). The 4 m/s runaway test misses the typical 1.5 m/s coast. The coast's gyro attitude is good (within a few degrees over 5-20 s).

**Offline BA.**

| Seq | Cost start / end | Dropped at 3 px | Already > 3 px at start | Components | Keyframes > 4 m/s |
|---|---|---|---|---|---|
| R_11 | 1.2e8 / 3.9e5 | 24 % | 28 % | 1 | 0 |
| 1_19 | 8.1e10 / 3.7e6 | 46 % | 20 % | 9 | 155 |
| 2_11 | 6.3e13 / 3.3e7 | 73 % | 28 % | 45 | 393 |
| 4_11 | 5.7e15 / 8.2e8 | 65 % | 42 % | 110 | 513 |
| 5_11 | 3.4e11 / 2.2e7 | 76 % | 18 % | 7 | 37 |

1. **Ratchet outlier drop:** `global_vi_ba/mod.rs:363-366` calls `tighten()` (3 px) on every iteration from 8 on, and dropped observations never return (ORB-SLAM3 and visloc drop once and re-solve).
2. **IMU edges with no robust kernel over corrupt states:** `imu_kernel: None`; edges join every consecutive keyframe pair, including runaway lost keyframes and IMU-uninitialised young segments with v = 0; the solve ignores `inserted_while_lost`. This is the 6e13 / 5.7e15 start cost. Even R_11 starts at an IMU rotation residual of about 30σ.
3. The 60-iteration cap binds everywhere, but raising it under the ratchet drops more. Conditioning is ruled out (4-DOF gauge prior per component, no factorisation failure, no NoDescent).
- 5_11 holds 12 of 14 control points in one segment, so its offline failure is the solve, not fragmentation.

**Ranked causes:**
1. Attitude reset at give-up (`state.rs:158-162` via `system/mod.rs:2769`). Fix: carry the gyro attitude and hold translation (attitude only, not velocity or bias). Prediction: rotation error across give-ups ~130° to < 5°, 2_11 Sim3 succeeds. (Card AC, interim: the Held seed follows the gyro to 0.1-0.4° per frame through the re-seed; the measured 8.6-11° on 2_11 is an upper bound set by the GT sampling, see Corrections.)
2. Young-segment death loop: a new segment has no IMU state, so 0.5 s grace and no prediction. Fix: keep the inertial state, or at least a gyro rotation prior.
3. Turn on the spot into unmapped view: wider overlap (raw fisheye, card F) and per-frame point refill (Basalt).
4. Offline BA: drop outliers once, IMU Huber (ORB-SLAM3 FullInertialBA), leave lost and uninitialised keyframes out of IMU edges (Basalt NFR).
5. Coast translation drift: hold translation, let the IMU carry the attitude only, beyond about 0.5 s (ledger G40).
6. The 20-point cold-start gate: effect unknown (needs a DEBUG replay).

## Combined picture

Both diagnoses point to one chain: weak vision lets a wrong velocity in, the coast runs away, the give-up resets the attitude and starts an IMU-less segment that dies again, and nothing links the pieces. Card AA (2026-11-22) fixed the clock on the IMU edges but did not move R_01's along-track scale (K1), so the smooth-drift term is set by the visual side.

## Offline-BA test on the F1 sessions (2026-11-22, no code)

Arms on the existing sessions (`/home/edgar/dev/split/.pr180-fix/offbridge/`): B0 defaults, H `offline_ba: ( imu_kernel: Huber )`, BR `( bridge_segments: true )`, HB both. B0 reproduces the validation exactly on all three sequences.

**Confounded:** card AC's planner found that the F1 session dumps hold earlier segments rotated 27-89° about the origin and off gravity (a later segment's inertial alignment re-aligns the whole world; `Map::apply_inertial_alignment`). Only R_11 (0.1°) is clean.

| Seq | B0 CP | H CP | BR CP | HB CP | 3 px drop B0 / H | Components B0 / BR |
|---|---|---|---|---|---|---|
| R_11 (clean) | 56.10 | 55.45 | **59.62** (ATE 3.96 to 3.79 m) | 57.69 | 24 / 23 % | 1 / 1 (2 bridges, residual 0.25σ) |
| 1_19 | 3.74 | FAIL | 24.21 | FAIL | 46 / 35 % | 9 / 1 (bridge residual 12σ) |
| 5_11 | FAIL | 8.53 | 3.63 | 0.00 | 76 / 50 % | 7 / 1 (bridge residual 153σ) |

- **H missed its prediction** (3 px drop 14-16 %): it stays at 35-50 % on the tilted sessions, and on R_11 there was little to fix (24 %). On 5_11 the IMU rotation residual rises to 130σ under H: the kernel down-weights the mis-levelled edges, which is what Huber should do there (inference from the report).
- **BR joins every session into one component,** but on the tilted sessions the bridges are forced (residuals 12σ and 153σ). 1_19's CP rise (3.74 to 24.2) is likely partly the scorer: a single connected trajectory lets its robust Sim3 succeed. ATE on 1_19 is unchanged (62.3 to 62.4 m).
- **Clean result:** on R_11, BR alone is best (+3.5 CP, ATE −4 %).
- **Rerun on card AC-1 sessions** (no mis-levelled segments) before any offline-BA default changes.

## Corrections (2026-11-22, during card AC)

Two metric bugs in `experiments/quick-eval/lossev.py` and one in `worldrot.py` were found while scoring card AC. They change how some numbers above should be read.

1. **GT body frame (fixed in `15a0a02` on `feat/loss-one-world`).** The pseudo-dense GT body frame is rotated from the trajectory's body frame (105.4° on 2_11, fitted once on the F1 full run from 6339 tracked 1 s pairs; about 104° on 3_18; 0.2° on R_01). The original script compared relative rotations across that frame difference. A correct rotation about a mis-mapped axis then reads as tens to 165° of error. A per-run fit from a few pairs is off by 4-5° and doubles the error on a 170° turn, so the fit must be done once per sequence.
2. **GT sampling (open, sent to the card AC implementer).** `Gt.rot()` takes the nearest pseudo-dense sample within 0.15 s without interpolation. On 2_11 the samples are 0.20 s apart (median; p90 0.65 s). At the 160-200°/s turns around a give-up, a stale sample reads 10-25° of error. The "tilt jump at the first tracked frame" seen in card AC's first scoring was this artifact: SLAM and gyro steps agree to 0.1-0.4° per frame across the re-seed, while the GT steps go 0, 0, 22.8, 0, 14.5° (frames 21478-21505).
3. **Rotation angles in section 1 and point 2 (fixed in `17baf79`).** `worldrot.py` reported the angle of the position about the origin, which is only a lower bound on the rotation. The true 5_11 orientation changes are 86.5° and 55.1° (not 61.1° and 43.6°). The other θ values were not re-measured.

**What stands.** The ~130° / ~119° heading error across give-ups on 2_11 / 4_11 is unaffected. The base run's SLAM rotation across the gap is about 0°, so the error equals the size of the turn whatever the frame, and the gyro gives the same size. The position and coast numbers use `Gt.pos()`, which interpolates linearly; the sampling bug affects them less, but they are not yet re-checked.

