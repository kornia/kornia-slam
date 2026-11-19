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
- **A plausible accelerometer bias is small.** R_03 uses the factory-rectified IMU. Even I0's VIBA2 |ba| of 0.52 m/s² is large for an Aria IMU.

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
