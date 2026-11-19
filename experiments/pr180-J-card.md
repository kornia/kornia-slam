# Card J: the inertial initialisation absorbs bad edges into gravity and accelerometer bias

Status: written 2026-11-19. Diagnosis step J0 has its predictions here before any code. The fix steps get theirs after J0 is scored.

## Experiment: J, VIBA robustness on R_03 under `min_cold_start_points: 20`

- **Gap tested:** card I next-gap 2. `min_cold_start_points: 20` gives R_11 800/800 tracked, but breaks R_03 (card I, K3 on Ib). The owner's target is 100 % tracking on easy and medium sequences.
- **Base:** kornia-slam `feat/global-pose-graph` @ e6dc68a (binary `slam-cardI`, built from 076faf1; e6dc68a changes docs only). Configs `cardI/r03-I0.ron` and `cardI/r03-Ib.ron` (`pr180-frontend-1ms` plus the knob). LaMAria R_03_easy, 800-frame window.

### Evidence, from `cardJ/out/r03-{I0,Ib}-d*` (`--debug`, `KORNIA_SLAM_DEBUG=1`)

- **Start frames.** I0 seeds the surviving map at frame 3, Ib at frame 2 (card I). Both VIBA0 windows pass the gate at the same point: 10 keyframes and 2.45 s of IMU. Ib's VIBA0 runs at frame 63, with 14 keyframes.
- **The world frame does not explain the VIBA0 gravity difference.** The VIBA0 gravities differ by 2.04° (I0 `(9.801,0.422,0.001)`, Ib `(9.796,0.399,0.350)`, seed-camera frame). The gyro rotates the body 0.6° between frames 2 and 3, so ≥ 1.4° is a real estimate difference.
- **VIBA1's correction measures the VIBA0 error.** After VIBA0, the map is rotated so gravity is `(0, 9.806, 0)`, and VIBA1 (frame ~104) re-solves gravity in that frame:

| Run | VIBA0 ba (m/s²) | VIBA1 gravity | VIBA0 tilt error | VIBA1 ba | VIBA1 cost | VIBA2 (frame ~305) |
|---|---|---|---|---|---|---|
| I0-d1 | (0.145, 0.008, -0.027) | (-0.018, 9.810, -0.078) | 0.47° | (0.199, 0.089, -0.006) \|0.22\| | 8.7e4 | accepted, \|ba\| 0.52, cost 2.4e5 |
| Ib-d1 | (0.096, 0.025, -0.002) | (0.023, 9.792, -0.587) | 3.4° | (0.249, 0.679, 0.220) \|0.76\| | 8.1e5 | **rejected**, \|ba\| 1.55 > 1.0, cost 2.4e7 |
| Ib-d2 | (0.096, 0.025, -0.002) | (0.025, 9.792, -0.588) | 3.4° | (0.248, 0.682, 0.220) \|0.76\| | 8.1e5 | **rejected**, \|ba\| 2.51 > 1.0, cost 1.5e7 |

- **Tilt and bias trade off.** 3.4° of tilt is 9.8 · sin 3.4° = 0.58 m/s² of horizontal specific force. VIBA1's extra `ba_y` in Ib is 0.68 m/s². The solve explains the residual with both.
- **The window carries far more cost under Ib.** VIBA1 cost is 9× I0's and VIBA2 cost is 60-100× I0's, on windows of similar length. With `PRIOR_A` = 1e5 (`inertial/schedule.rs:28`), a bias of 1.5 m/s² costs only 2.4e5 of the 2.4e7. The edges dominate, and the bias goes wherever they push it.
- **The losses follow.** Ib-d2 is lost from frame 322, 17 frames after the VIBA2 reject, and coasts 101 frames. Ib-d1 is lost from frame 269, before VIBA2, and coasts 459 frames.
- **The accelerometer bias is large.** Even I0's VIBA2 |ba| of 0.52 m/s² is large for an Aria IMU.
  - *Corrected 2026-11-19:* this line first said R_03 uses the factory-rectified IMU. It does not. LaMAria has factory calibration only for R_11 and the `sequence_*` sequences, so R_03 and R_01 run on the unrectified IMU, as in card H. An unrectified accelerometer (scale and misalignment) can account for part of the bias, and that is a candidate cause of the systematic VIBA2 tilt.

### Theory

- **(a) A few inconsistent edges dominate the solve.** The VIBA solve (`inertial/initialization/solve.rs:30-174`) is unweighted least squares over every preintegration edge in the window, from `start_idx` on. Every term is quadratic, and no edge is gated or down-weighted. If a few keyframe poses in the window are visually wrong, their edges carry large residuals. The only free parameters that can reduce those residuals across the whole window are gravity direction and `ba`. Under Ib, the extra cost comes from the start of the window: the map is seeded from 45-63 points in the auto-exposure ramp, and KLT refuses frames 48-51.
- **(b) The pipeline lacks ORB-SLAM3's correction step.** ORB-SLAM3 runs `InitializeIMU(priorG, priorA, bFIBA = true)`. After aligning, it runs `Optimizer::FullInertialBA` (`LocalMapping.cc`, `InitializeIMU`), which moves the keyframe poses jointly with the IMU, with a robust visual kernel. Bad visual poses are corrected before VIBA1 and VIBA2 see them. `schedule.rs:26-27` records that this step is missing here (kornia-slam#51). (a) is the symptom, (b) the reason it persists.
- **Not the cause:** the start frame itself, and VIBA0's crude seed (`solve.rs:322-332`). I0's VIBA0 seed is just as crude, and VIBA1 corrects it to 0.47°.

### Step J0: localise the cost (diagnosis only, temporary print, not committed)

- **Change:** a temporary `debug_eprintln!` in `inertial_optimizer`, after the solve. It prints each edge's frame pair, its squared residual, and its share of the total cost.
- **Runs:** R_03 I0 and Ib, one each, at VIBA0, VIBA1 and VIBA2. The edges are a function of the map, so one run per config localises; it does not measure noise.
- **Predictions:**
  - **P1, Ib VIBA1:** the top 3 edges carry ≥ 50 % of the cost, and at least 2 of them have a keyframe in frames 2-60.
  - **P2, I0 VIBA1:** the top 3 edges carry ≤ 30 % of the cost.
  - **P3, Ib VIBA2:** at least one top-3 edge is the same as in P1. The bad edge stays in the window, because the window always starts at `start_idx`.
- **Kill rules:**
  - **K1:** under Ib, the cost is spread out, with the top 3 edges ≤ 30 %. Then (a) is wrong: the whole window disagrees with the IMU, which points at the extrinsic or time offset rather than at bad poses. Next, compare the per-edge rotation residual (gyro against the visual rotation) with the velocity residual.
  - **K2:** the top edges are in frames 100-305, not at the start. Then the start-up is not the cause, and VIBA1 and VIBA2 are being hurt by the mid-sequence tracking. Next, look at that tracking first.

### Fix candidates (each gets predictions and kill rules after J0; each behind a knob, default off)

- **J1, gate the edges (cheap):** after the solve, drop the edges whose residual is above the chi² 95 % bound for their information and re-solve once. This is the two-pass outlier gate that ORB-SLAM3 uses in `PoseInertialOptimization`, but ORB-SLAM3 does not use it in `InertialOptimization`. Only (a) is fixed.
- **J2, an inertial BA after VIBA0 (ORB-SLAM3's `bFIBA`):** fixes (b). It is a larger change, and it touches the mapping thread.
- **J3, a window that starts after start-up:** start the VIBA window ≥ 1 s after the seed, past the exposure ramp. VINS-Fusion initialises over a sliding window of the latest frames (`Estimator::initialStructure`), not from the first frame.

### Bit-identity and noise

- **Identity:** J0 is a print only. Each fix knob defaults to off, with a unit test that the default config is today's.
- **Noise:** the VIBA numbers repeat to 3 digits across Ib-d1 and Ib-d2 up to VIBA1, so J0 needs one run per config. The scoring of the fixes uses 3 runs per config. The R_03 I0 band is tracked 777-780; the Ib band is 341-747.

### Result log

- **2026-11-19, J0 scoring.**
  - **Setup:** binary 076faf1 plus a temporary per-edge print in `inertial_optimizer` (reverted, not committed). One run each of `r03-I0` and `r03-Ib`, in `cardJ/out/r03-{I0,Ib}-j0`.
  - **Measurement:** each edge's whitened squared residual, split into rotation, velocity and position blocks. The blocks are approximate, because `sqrt_info` is not block-diagonal.
  - **Run tracking:** I0 tracked 786; Ib tracked 747, lost frames 254-289, VIBA2 rejected (|ba| 2.47).

| P/K | Prediction | Measured | Verdict |
|---|---|---|---|
| P1, Ib VIBA1 | top 3 ≥ 50 %, ≥ 2 with a keyframe in frames 2-60 | top 3 = 35.5 % (40→45, 54→57, 45→48), all in frames 40-57 | missed on share; location as predicted |
| P2, I0 VIBA1 | top 3 ≤ 30 % | top 3 = 77 % (3→6 40 %, 18→26 31 %, 26→34 7 %) | missed |
| P3, Ib VIBA2 | a top-3 edge is the same as in P1 | top 8 are all in frames 223-289 (264→269, 226→234, 269→274, ...) | missed |
| K1 | Ib cost spread out, top 3 ≤ 30 % | 35.5 % | not fired |
| K2 | top edges in frames 100-305 | VIBA1: no. VIBA2: yes, frames 223-289 | **fired for VIBA2** |

- **What the share hid: the rotation block.**
  - Almost all of the top cost in both configs sits in the rotation block: I0 3→6 has 28k of 32k there. The visual relative rotation and the gyro disagree, by roughly 0.5-1° over ~0.15 s at the gyro noise density.
  - Gravity and `ba` do not enter the rotation residual (`factor.rs:153-154`), so this cost cannot tilt gravity. The concentrated rotation cost is real but harmless, which is why P2 missed while I0 still works.
- **The block that moves gravity and `ba` is velocity.**
  - At VIBA1, Ib's velocity residuals are about 10× I0's. Ib's top-8 edges carry 15-53k each, and I0's top 4 carry 0.2-3.7k.
  - The large Ib edges sit in frames 40-86: the stretch where KLT refuses frames 48-51 on the map seeded at frame 2.
  - (a) holds for VIBA1 in a weaker form: it is not 3 edges but ~8 start-up edges, about half of the cost. They are visually poor poses, and the solve absorbs them as 3.4° of tilt and 0.76 m/s² of `ba`.
- **VIBA2 fails for a different reason: coast keyframes.**
  - Ib is lost at frames 254-289, before VIBA2 (frame ~305). Its top-8 VIBA2 edges are the keyframes inserted at the inertial prediction while lost (259→289) and the edges just before the loss (223→234).
  - Their velocity blocks are 2.7-8.9e5 each. Those are 30-100× the VIBA1 edges, and they hold most of the 1.34e7 total.
  - A coast keyframe's pose is the IMU prediction under the VIBA1 state. Re-solving the IMU state while that pose stays fixed makes it inconsistent by construction. The window always starts at `start_idx` and keeps every keyframe, so these edges enter it.
- **Theory status:**
  - (a) is supported for VIBA1 (start-up velocity residuals), and refuted for VIBA2 as worded.
  - New (c): coast keyframes poison the refinement window.
  - (b) is untested.
  - I0 survives because its VIBA2 runs before any loss: it is lost first at frame 374, after VIBA2 at ~304.
- **Next steps, both behind knobs that default to off:**
  - **J4: keep coast keyframes out of the VIBA windows.** Their poses carry no visual information. This is not ORB-SLAM3's choice: with an IMU, ORB-SLAM3 also inserts keyframes while `RECENTLY_LOST` (`Tracking.cc`, `mInsertKFsLost`). It does not need to exclude them, because its `FullInertialBA` after each VIBA moves those poses as well, which is (b) again. VINS-Fusion initialises once, over a window of frames that are tracked visually. J4 is our own stop-gap for the missing (b). It is cheap, and it addresses the VIBA2 reject directly.
  - **J1: gate the velocity block per edge**, for the start-up edges of VIBA1.
  - The predictions for J4 and J1 are written before code, in the next entry.

## Steps J4 and J1: predictions (written 2026-11-19, before code)

- **One more fact from the d-runs:** Ib-d2 rejected VIBA2 at frame 305 *before* its first loss, at frame 322. Its window held no coast keyframe, yet |ba| was 2.51. So J4 alone cannot fix every Ib run, and J1 is needed for the rest.

### Knobs (in `inertial`, `InertialConfig`; the defaults keep today's behaviour)

- **`viba_window: VibaWindow`, default `All`.** `VisualOnly` leaves out of every VIBA solve the keyframes with `inserted_while_lost` (J4). Their velocities stay as they are, and the alignment still moves them with the map.
- **`viba_velocity_gate: f64`, default `0.0` (off).** With `k > 0`, after the solve, each edge's velocity residual is normalised by its preintegration velocity covariance. An edge whose cost is above `k` × the median edge cost is dropped, and the problem is solved once more. At most 25 % of the edges are dropped, the worst first (J1). The test runs use `k = 10`.
- **Debug line:** `[imu_init] gate dropped N of M edges: a->b, ...`, the mechanism counter.

### Configs (3 runs each, `--debug`, `KORNIA_SLAM_DEBUG=1`)

| Name | Base config | Knobs |
|---|---|---|
| J4 | r03-Ib | `viba_window: VisualOnly` |
| J1 | r03-Ib | `viba_velocity_gate: 10.0` |
| J41 | r03-Ib | both |
| J41-I0 | r03-I0 | both |
| J41-R01 | r01-Ib | both |
| J41-R11 | r11-Ib | both |

### Predictions

- **P1, J4 on R_03:**
  - in each run lost before VIBA2, VIBA2 cost ≤ 3e6 (Ib 1.3-2.4e7) and VIBA2 is accepted;
  - `coast_inertial` ≤ 60 in each run (Ib band 53-459).
- **P2, J1 on R_03, mechanism:**
  - the gate drops 2-8 VIBA1 edges, all with a keyframe in frames 40-86;
  - VIBA1's gravity correction is ≤ 1.2° (Ib 3.4°, I0 0.47°);
  - VIBA1 |ba| is ≤ 0.40 m/s² (Ib 0.76, I0 0.22).
- **P3, J41 on R_03:**
  - VIBA2 accepted in 3/3 runs;
  - tracked ≥ 777 (I0 band 777-786, Ib band 341-747);
  - ATE SE3 ≤ 0.25 m (I0 band 0.207-0.211 m).
- **P4, J41 on the other configs:** each stays inside its base band.
  - J41-I0: tracked ≥ 777, ATE ≤ 0.23 m.
  - J41-R01: 800/800, ATE 0.097-0.12 m.
  - J41-R11: 800/800, ATE ≤ 0.33 m, scale 0.95-1.16 (Ib band 1.08-1.16).

### Kill rules

- **K1, mechanism moved, score did not:** P2 holds, but J41 R_03 runs are still lost around frames 250-290. Then the loss is not caused by the IMU state. Next, look at the visual tracking at frame 250.
- **K2, score moved, mechanism did not:** tracking improves, but the gate drops 0 edges and the VIBA costs stay in the Ib band. That is luck or another cause, and the knob does not ship.
- **K3:** J41-I0 or J41-R01 leaves its base band. Then the knobs hurt the configs that work, and do not ship in a preset.
- **K4:** the gate hits its 25 % cap in any VIBA. Then it is over-firing and the threshold or the normalisation is wrong; it does not ship.

### Ship rule

If P3 and P4 hold, `pr180-frontend-1ms` gets `min_cold_start_points: 20` together with J4 and J1. That would give R_11 800/800 without breaking R_03, which is card I's goal.

## Result log, J4 and J1 (2026-11-19)

- **Setup:**
  - binary c21243f (knobs from 846bdf2), configs in `cardJ/cfg/`, run dirs `cardJ/out/<config>-r{1,2,3}`;
  - `--debug` and `KORNIA_SLAM_DEBUG=1`, 3 runs per config, 3 at once on the laptop;
  - ATE is SE3 against pseudo-dense ground truth (`ate.py`); tracked and `coast_inertial` come from `[summary]`.
- **Knobs-off identity:** the app is not deterministic run to run, so trajectory `cmp` cannot pass. As the proxy, `r03-Ib-off` (new binary, knobs off) repeats the Ib VIBA0 and VIBA1 numbers to all printed digits:
  - gravity `(9.796,0.399,0.350)`, then `(0.026,9.792,-0.588)`;
  - VIBA1 ba `(0.248,0.681,0.218)`.

  The unit tests and the implementer's golden dump also showed the solve bit-identical with the knobs off.
- **Implementation note:** the gated VIBA0 re-solve starts from the same seed, and that seed (finite-difference velocities and the summed gravity direction) is built from every edge, including the dropped ones. On the refinement solves the seed does not depend on the edges.

| P/K | Prediction | Baseline | Measured (each run) | Verdict |
|---|---|---|---|---|
| P1, J4 R_03 | VIBA2 cost ≤ 3e6 and accepted in runs lost first; coast ≤ 60 | VIBA2 cost 1.3-2.4e7, rejected; coast 53-459 | every run was lost first (frames 247-272). VIBA2 cost 8.6e6, 6.7e6, 1.05e7; rejected ×3, \|ba\| 2.38, 2.43, 2.32. Coast 44, 476, 458 | missed |
| P2, J1 R_03 | gate drops 2-8 VIBA1 edges in frames 40-86; VIBA1 correction ≤ 1.2°; VIBA1 \|ba\| ≤ 0.40 | 3.4°, 0.76 | gate dropped **0** edges in VIBA0 and VIBA1 in all 6 Ib-based runs (J1, J41). VIBA1 gravity `(0.026,9.792,-0.588)` = 3.4° ×6; ba 0.76 ×6. It fired only in VIBA2, on frames 183-307 | missed |
| P3, J41 R_03 | VIBA2 accepted 3/3, tracked ≥ 777, ATE ≤ 0.25 m | Ib tracked 341-747 | VIBA2 rejected ×3 (\|ba\| 2.70, 2.70, 2.47); tracked 365, 668, 657; ATE 77.4, 1.44, 4.22 m | missed |
| P4, J41-I0 | tracked ≥ 777, ATE ≤ 0.23 m | 777-786, 0.179-0.211 m | 788, 784, 788; 0.210, 0.198, 0.205 m. Gate drops 3→6 and 18→26 in VIBA0 and VIBA1 | as predicted |
| P4, J41-R01 | 800/800, ATE 0.097-0.12 m | 800 ×3, 0.105-0.111 m | 800 ×3; 0.102, 0.121, 0.119 m | inside noise (r2 0.001 m over the bound) |
| P4, J41-R11 | 800/800, ATE ≤ 0.33 m, scale 0.95-1.16 | 800 ×3, 0.214-0.333 m, 1.08-1.16 | 800, 800, 797; 0.230, 0.312, 0.221 m; scale 1.09-1.14 | as predicted, except r3 at 797 |
| K1 | P2 holds, loss remains | | P2 did not hold | not fired |
| K2 | score moves, mechanism does not | | no score moved on R_03 | not fired |
| K3 | J41-I0 or J41-R01 leave their band | | both inside | not fired |
| K4 | gate hits its 25 % cap | | at most 5 of 70 edges | not fired |

- **Theory (a) is refuted.** The Ib start-up edges are not outliers against the window median. At k = 10, the gate finds nothing in VIBA0 or VIBA1 under Ib, while it does find the two I0 rotation-heavy edges (3→6, 18→26). The whole early Ib window carries a high velocity cost, so its error is systematic, not a few bad edges.
- **Theory (c) is only part of it.** Without the coast keyframes, the VIBA2 cost falls about 40 %, from 1.3-2.4e7 to 0.67-1.05e7. VIBA2 is still rejected in all 9 Ib-based runs.
- **The VIBA2 reject is systematic.** The rejected `ba` points the same way in every one of the 10 Ib runs, whatever the knobs: `(≈0, 1.55-1.94, 1.52-2.00)`, |ba| 2.3-2.7 m/s². 9.8 · sin 14° = 2.4 m/s², so the solve is fitting a gravity tilt of about 14° that its `prior_g = 0` re-solve cannot reach through gravity. Robust gating cannot fix a systematic misfit.
- **What is known now:**
  - The R_03 Ib collapse starts at VIBA0: its 3.4° error, measured by VIBA1, is reproduced to 3 digits in every run.
  - Tracking is then lost at frames 176-272, and VIBA2 meets an inconsistent window.
  - The only difference from I0 is the seed one frame earlier (frame 2 against 3), inside the auto-exposure ramp (card I, gap 1).
- **Outcome:** neither knob ships in a preset. Both stay in the code, off by default. They are harmless on I0, R_01 and R_11, and R_11 holds 800/800 in 2 of 3 runs with `min_cold_start_points: 20`. The card I ship rule stays unmet: `min_cold_start_points: 20` still breaks R_03.
- **Next gaps, in order:**
  1. **J3:** start the VIBA window ≥ 1 s after the seed, past the exposure ramp. This tests whether frames 2-20 are what make VIBA0 wrong. It is cheap, and one knob in `ImuInitializer::in_solve_window`.
  2. **J2:** ORB-SLAM3's `FullInertialBA` after each VIBA (theory b, kornia-slam#51). It is the larger change, and it is also the one reference mechanism the pipeline lacks.
  3. **Why VIBA2 fits a 14° tilt:** compare the VIBA2 window's visual trajectory with ground truth around frames 176-307. Also check how much of it is the unrectified IMU: R_03 has no factory calibration, and R_11, which has one, never hit this reject.

## Step J3: a VIBA window that starts after the exposure ramp (written 2026-11-19, before code)

- **Gap:** G32. **Base:** 46376a8 (binary c21243f plus J3), configs from `experiments/quick-eval/make_config.sh`.
- **Mechanism, restated:**
  - Under Ib, the R_03 map is seeded at frame 2, in the auto-exposure ramp (card I gap 1: mean intensity 151 → 66 over the first frames, in steps every 2 frames).
  - VIBA0 then solves over frames 2-63 and is 3.4° off, as VIBA1 measures. I0's window starts at frame 3 and is 0.47° off.
  - J1 showed that the error is not a few outlier edges: nothing in the window stands out against its median.
- **Theory:**
  - The poses of the first keyframes are biased together: they are tracked on a map seeded under a changing exposure, from 45-63 points.
  - A biased start block shifts the finite-difference velocities and the gravity seed coherently (`solve.rs`, the `!already_initialized` branch), and no per-edge gate sees a coherent shift.
  - VINS-Fusion initialises over a sliding window of the latest frames (`Estimator::initialStructure`), so the first frames of a run fall out of its window. ORB-SLAM3 uses all the map's keyframes, but its map does not start inside a ramp with a 1-frame-old seed.
  - Starting the window after the ramp removes the biased block, and the solve sees only poses tracked under steady exposure.
- **Knob:** `inertial.viba_window_delay_sec: f64`, default `0.0`.
  - The VIBA window (VIBA0, VIBA1 and VIBA2, and `ready()`) starts at the first keyframe whose accumulated IMU-factor time since `start_idx` is ≥ the delay.
  - IMU time is used because a `Keyframe` has no timestamp.
  - The resume keyframe and the alignment are unchanged: the whole map is still aligned.
  - Debug line: `[imu_init] window starts at keyframe N (delay D s)`.

### Configs (3 runs each)

| Name | Base | Knob |
|---|---|---|
| J3 | r03 + `min_cold_start_points: 20` (Ib) | `viba_window_delay_sec: 1.0` |
| J3b | Ib | `viba_window_delay_sec: 2.0` |
| J3-I0 | r03 base (I0) | `1.0` |
| J3-R01 | r01 Ib | `1.0` |
| J3-R11 | r11 Ib | `1.0` |

### Predictions

- **P1, J3 on R_03, mechanism:**
  - the VIBA0 window starts at a keyframe in frames 20-26;
  - VIBA1's gravity correction is ≤ 1.2° (Ib 3.4°, I0 0.47°), and VIBA1 |ba| ≤ 0.40 m/s² (Ib 0.76).
- **P2, J3 on R_03:**
  - VIBA2 accepted 3/3, with |ba| ≤ 1.0 (Ib rejected 10/10 at 2.3-2.7);
  - tracked ≥ 777 (I0 band 777-788; Ib 341-747);
  - ATE SE3 ≤ 0.25 m (I0 0.179-0.211 m).
- **P3, J3b on R_03:** as P1 and P2. If J3 misses and J3b holds, the ramp effect lasts longer than 1 s.
- **P4, J3 on the configs that work:**
  - J3-I0: tracked ≥ 777, ATE ≤ 0.23 m;
  - J3-R01: 800/800, ATE 0.097-0.125 m;
  - J3-R11: ≥ 797/800, ATE ≤ 0.33 m, scale 0.95-1.16.

### Kill rules

- **K1, mechanism moved, score did not:** VIBA1's correction is ≤ 1.2°, but VIBA2 is still rejected or tracking is still lost at frames 170-310. Then the start block was one cause, and the systematic VIBA2 tilt has another. Next, J2, and the unrectified-IMU check.
- **K2, score moved, mechanism did not:** R_03 tracks ≥ 777, but VIBA1's correction stays > 2°. That is luck, for example because a later VIBA0 changes the tracking path, and the knob does not ship.
- **K3:** J3-I0, J3-R01 or J3-R11 leaves its band. The knob does not ship in a preset.
- **K4:** VIBA1's correction is still ≥ 3° under both 1.0 and 2.0 s. Then the theory is refuted: the error is not in the first seconds but in the whole Ib trajectory.

### Bit-identity and noise

- **Identity:** default `0.0` gives an effective start equal to `start_idx`. A unit test, and a knobs-off run whose VIBA numbers repeat the Ib ones, check that nothing else changes.
- **Noise:** 3 runs per config. The Ib VIBA numbers repeat to 3 digits, so the mechanism counters are not noisy; tracking and ATE are.

### Ship rule

If P2 and P4 hold, `pr180-frontend-1ms` gets `min_cold_start_points: 20` and `viba_window_delay_sec: 1.0`.

## J3 cancelled before code: the visual trajectory is at 1/5 scale (2026-11-19)

The owner asked to make sure the gap is not an LK or ORB issue. Comparing our per-step motion with ground truth (`relmotion.py`, `scaleprof.py`, `stepwise.py` over the card J run dirs) answered it before J3 was built. The planner was stopped and J3 has no code.

- **Rotation is right.** Over frames 16-120, the median per-step rotation error is 0.04-0.12° in I0 and in Ib's frames 16-40. The gyro agrees with ground truth to 0.03-0.06°.
- **Translation is shrunk under Ib.** Median ratio of our step length to ground truth, per 25-frame bin:

| Run | 25 | 50 | 75-125 | 150-250 | after the re-seed (~270) |
|---|---|---|---|---|---|
| I0-j0 | 0.75 | 1.01 | 1.00 | 0.99-1.00 | 1.00 |
| Ib-j0 | 0.13 | 0.18 | 0.18 | 0.20-0.23 | 0.98-1.00 |
| Ib-d2 | 0.13 | 0.21 | 0.17-0.19 | 0.19-0.23 | 1.0 (after 400) |

- **The VIBA errors follow from this.** For stereo, VIBA fixes the scale at 1, and the finite-difference velocities are then about 1/5 of the IMU's. The velocity residuals are systematic, which is why the edge gate (J1) found nothing and VIBA2 fits a large tilt and bias. G32 is a symptom, not the gap.
- **Not LK.** Under Ib, frames 3-45 are all tracked by ORB map projection (`[track] ok`, 160-230 matches and inliers, no `[klt-first]` line until frame 46). When the device starts to move at frame ~33 (ground-truth steps 0.3-1.2 cm), our steps are 0.03-0.14 cm. I0 runs `klt-first` from frame 27, and its steps match the ground truth.
- **Not the ORB stereo seed.** Both maps are seeded from the same stereo overlap: 63 points, median depth 0.80 m, at frames 2 and 3. The per-frame overlap depth stays at 0.79-0.80 m.
- **Not stale tracks across the reset.** The 53 flow tracks at frame 2 ran before the reset; frame 3 starts with 20 fresh tracks.
- **Suspect: map points triangulated across a near-zero baseline.**
  - The device is almost static for frames 0-32 (ground-truth steps < 1 mm). Under Ib, keyframes 5, 16, 24 and 32 grow 26-51 points each by pair triangulation (`mapping/growth.rs`).
  - The only baseline guard is `|t| > 1e-8` (`growth.rs:233`). ORB-SLAM3's `CreateNewMapPoints` skips a pair whose baseline is below the stereo baseline (stereo) or 1 % of the median depth (mono).
  - Points that are too close make a real translation look 1/5 as large, with rotation unaffected, which is what the trajectory shows.
  - This is a mapping (triangulation) gate, not the LK or ORB frontend. It is not yet measured.
- **Next diagnosis (temporary print, before any card):**
  - print, per keyframe from 2 to 60, the median depth of its map points against the median stereo depth of the same frame, for I0 and Ib;
  - print the source of each grown point (stereo, or a pair with its baseline).

  If the Ib map points are about 5× closer than the stereo depth and come from pairs with a sub-millimetre baseline, the fix is ORB-SLAM3's baseline gate behind a knob.
