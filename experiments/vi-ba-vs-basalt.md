# kornia-slam local VI-BA against Basalt's VIO, and SLAM Dunk in theory

Status: 2026-11-23. This is a read-only analysis. There were no builds, no SLAM runs and no code edits. The only Python was run (at nice 19) on existing outputs: the `S0` / `S0-noloop` synchronous runs and the bench Basalt trajectories. SLAM Dunk was read, not run, and none of its code is copied here (AGPL).

Owner question: where is everyday precision lost? Healthy synchronous runs are still 1.2-3x behind Basalt in the EuRoC Vicon rooms. Over 10 s pieces, our smoothed error is 0.008-0.052 m against Basalt's 0.005-0.026 m (`experiments/euroc-top1-card.md:51`).

Path prefixes:
- `K` = `crates/kornia-slam/src` (branch `feat/global-pose-graph`, HEAD d080e7e).
- `F` = `crates/kornia-factors/src`.
- `SN` = `crates/kornia-sensors/src`.
- `BA` = `/home/edgar/dev/.flux-connect-scratch/bench/src/basalt`.
- `O3` = `/home/edgar/dev/.flux-connect-scratch/bench/src/ORB_SLAM3/src`.
- `E` = `/home/edgar/dev/split/.pr180-fix/euroc-top1` (S0 runs, binary md5 `cea7646298a68ddb948b16835472b2e3`).

Scripts used for the new numbers below are in this session's scratchpad (`pieces.py`, `kfpieces.py`, `kfbias.py`, `kfbias_t.py`, `rot2.py`, `spikes.py`, `tsteps.py`). They are not committed. Each one reads `traj*.txt`, `session.cbor` and the EuRoC GT through `bench/eval.py`'s loaders, with 5 ms association.

## 0. Answer in one paragraph

The gap is not in the solver arithmetic: Basalt even runs in f32. It is not mainly in per-frame tracking jitter either. On 6 of 7 Vicon/MH sequences, the 10 s-piece error of the keyframes alone is within 15 % of the all-frames figure (section 10.1). The keyframe trajectory itself is 1.5-2.5x Basalt's.

What the keyframe solve lacks compared with Basalt is information carried across windows:
- Our window is 3 keyframes, about 0.5-1.2 s.
- It is clamped to a frozen, infinitely confident past: fixed covisible keyframes, their landmarks, and a 1e-2 boundary IMU edge.
- Basalt keeps 7 keyframe poses and 3 full frame states. Everything older lives in a dense square-root marginal prior with first-estimate Jacobians: vision, IMU and bias history at full weight.

The same structure explains the tilt. Two zero-mean accel-bias priors bake about 0.8-0.9° of tilt into the map at initialization: σ 3 mm/s² in every VIBA, plus σ 0.1 m/s² per keyframe in every local solve. EuRoC's true accel bias is about 0.15 m/s², horizontal in the body frame. After that, nothing releases the map's roll and pitch; the accel bias absorbs the difference instead.

Measured on S0, tilt error is 2-7x Basalt's on every sequence. The accel-bias error is 0.10-0.89 m/s² and lies in the horizontal plane (section 10.3).

## 1. State held by each window

| | kornia-slam local VI-BA | Basalt `SqrtKeypointVioEstimator` |
|---|---|---|
| Optimized states | The newest `active_keyframes` = **3** keyframes. Each has 15 DOF: pose `[ρ;ω]`, world velocity, b_g, b_a (`K/system/config/mod.rs:622`, `K/mapping/vi_ba_schur.rs:11-14, 166-181`). | The newest **3 frame states** (`vio_max_states 3`), each 15 DOF (pose, velocity, b_g, b_a). Plus up to **7 keyframe poses**, 6 DOF (`vio_max_kfs 7`; `BA/data/euroc_config.json`; `BA/src/vi_estimator/sqrt_keypoint_vio.cpp:118-119, 559-613`). |
| Frames vs keyframes | Only keyframes are states. They come every 3..=8 frames (S0 header): MH_01 has 674 keyframes in 3682 frames, one per 5.5 frames. Frames between keyframes are tracked by visual PnP and never enter the solve. | **Every frame** becomes a 15-DOF state and is optimized (`measure` → `optimize_and_marg`, `sqrt_keypoint_vio.cpp:322-330, 482`). A frame becomes a keyframe when the tracked fraction drops below 0.7 and more than 5 frames have passed since the last one (`:369-374`). |
| Older states | Every keyframe that observes a window landmark, or shares an IMU edge with the window, enters as **fixed** context. Its pose, velocity and bias are frozen, and its reprojection rows stay in at full weight (`K/mapping/bundle_adjustment.rs:242-270, 693`). The boundary IMU edge from the newest fixed keyframe is scaled by 1e-2 (`vi_ba_schur.rs:864-876`). | Marginalized (section 4). When a frame state leaves the window, its velocity and bias are marginalized. If it is a keyframe, its pose stays (`:594-603, 795-806`). |
| Landmarks | World XYZ, 3 DOF, every live landmark seen by an active keyframe (`bundle_adjustment.rs:253`; `vi_ba_schur.rs:31-36`). No host frame. | Inverse distance plus a stereographic bearing, **hosted** on a keyframe's cam0 (`BA/include/basalt/vi_estimator/landmark_database.h:62-66`). Created only on keyframes, by two-view triangulation with baseline ≥ 0.05 m and distance > 0.33 m (`inv_dist < 3`, `sqrt_keypoint_vio.cpp:383-455`). Lost landmarks are marginalized (`:471-480`). |
| Window time span | 3 keyframes at 0.15-0.4 s spacing: about 0.5-1.2 s of IMU. | Frame states span 0.1 s. Keyframe poses span several seconds of visual baseline. The IMU history before that is in the prior. |
| Gauge | Fixed by the fixed context and its landmarks: 88 % of points pinned at window 3, 28 % at window 10 (card O). | Fixed by the initial prior (position and yaw at 1e8, `:87-92`), then by the marginal prior. |

## 2. IMU factor

| | ours | Basalt |
|---|---|---|
| Preintegration | Forster-style ΔR/Δv/Δp, Euler update per sample (`SN/imu.rs:321-412`). Default `SampleInterpolation::Hold`: each sample is held over the interval that ends at its stamp, a right Riemann sum (`SN/imu.rs:50-61, 184-221`; `K/system/config/mod.rs:524-531`). `Midpoint` (trapezoid, `:245-288`) exists but is not validated (gap A5). | `IntegratedImuMeasurement::integrate` per sample between frame stamps; the last sample is stretched to the frame time (`sqrt_keypoint_vio.cpp:216-249`). **The integration scheme itself (basalt-headers) is not on disk and was not verified.** |
| Bias Jacobians, re-linearization | First-order `delta_*_with_bias` (`SN/imu.rs:291-310`). The edge is re-integrated when the from-keyframe bias moves more than 0.02 (m/s² or rad/s) from its linearization bias (`K/mapping/bundle_adjustment.rs:736-760`). For gyro, 0.02 rad/s is about 1000 walk sigmas, so in practice gyro edges are never re-integrated. | First-order bias correction inside `residual` (`BA/include/basalt/linearization/imu_block.hpp:42-51`). No re-integration. Jacobians are taken at the **first-estimate (linearized) state** once a state is part of the prior (`getStateLin`, `isLinearized`, `imu_block.hpp:44-51`; `setLinTrue` at `sqrt_keypoint_vio.cpp:907`). |
| Covariance | 9×9 navigation covariance propagated per sample, `A C Aᵀ + B N Bᵀ`, with discrete noise `σ²/dt` (`SN/imu.rs:366-387`). The full 9×9 inverse is the information, with a diagonal fallback if singular (`F/inertial/preintegration.rs:187-209`). | Square-root inverse covariance from the same kind of propagation (`imu_block.hpp:55-67`, `get_sqrt_cov_inv`). |
| EuRoC noise | Datasheet values (`SN/rig.rs:206-213`): gyro 1.6968e-4 rad/s/√Hz, accel 2.0e-3 m/s²/√Hz, gyro walk 1.9393e-5, accel walk 3.0e-3. ORB-SLAM3 uses the same values. | `BA/data/euroc_ds_calib.json:187-206`: accel noise **0.016** (8x ours, so 64x less accel information), gyro noise **2.82e-4** (1.66x ours), accel walk **1e-3** (3x tighter than ours), gyro walk **1e-4** (5x looser). Converted to discrete time with `dicrete_time_*_noise_std` (`sqrt_keypoint_vio.cpp:160-162`). |
| Bias random walk | Per edge, diagonal, `σ_rw² · dt` (`SN/imu.rs:164-171`). Weight 1 on interior edges. **1e-2 on the boundary edge**, because `boundary_scales_bias_rows: true` (`vi_ba_schur.rs:864-876`, config `:626`). | `σ_rw / √dt` square-root weight, always at full weight (`imu_block.hpp:73-92`). |
| Robust kernel | Huber on the 9-dim nav χ² at **16.9 on every edge** (`ImuHuber::AllEdges`, config `:628`; `vi_ba_schur.rs:878-888, 2141-2160`; `F/inertial/preintegration.rs:168-181`). The bias rows stay quadratic. | **None** on IMU (`imu_block.hpp:54-67`). ORB-SLAM3 also puts no kernel on interior edges (`O3/Optimizer.cc:2632-2643`). |
| Extra bias priors | A zero-mean accel-bias prior, weight 1e2 (σ 0.1 m/s²), **on every free keyframe in every solve** (`vi_ba_schur.rs:233-253, 396, 2329-2348, 2845-2854`). It is not a config knob. | Only an initial prior on the first state, centred on the offline-calibrated biases: σ_bg 0.316 rad/s and σ_ba 0.1 m/s². The diagonal segments are swapped relative to the comment (`:93-108`; printed `marg_H` in `bench/results/basalt/*/stdout.log`). After that, the prior's information decays through the walk. |
| Gravity | Fixed world vector per solve (`ViBaParams::gravity`, `vi_ba_schur.rs:200-201`). It is re-aligned only by VIBA0/1/2 world rotations. Roll and pitch of fixed context keyframes are frozen. | Fixed `g` along world z (`sqrt_keypoint_vio.cpp:65`). Roll and pitch are part of every pose state and are held by the prior, never frozen. |
| Precision | f64 IMU blocks. | **f32**: the bench ran `basalt_vio` without `--use-double` (`BA/src/vio.cpp:226, 255`; `results/basalt/*/time.txt`). |

## 3. Visual factor

| | ours | Basalt |
|---|---|---|
| Residual | Pixel reprojection of world XYZ into a pinhole, f32 internally (`vi_ba_schur.rs:685-765, 2072-2100`). Plus a **stereo depth residual** on camera z (`:780-830, 2101-2115`). | Pixel reprojection of a host-anchored inverse-distance landmark into each target camera, including the right camera (`BA/include/basalt/linearization/landmark_block_abs_dynamic.hpp:160-181`). Stereo enters as right-camera pixel rows. |
| Noise σ | **1 px at every octave** (`VisualInformation::Unit`, `K/mapping/visual_information.rs:11-27`, config `:630`). Depth σ is **5 % of depth** (`StereoDepthSigma::RelativeDepth(0.05)`, config `:645-656`). | **0.5 px** (`vio_obs_std_dev`, `:72`). |
| Robust kernel | Huber at χ² 25, i.e. **5σ = 5 px** (`vi_ba_schur.rs:382, 844-853`). Depth Huber at 3σ (`:383`). | Huber at **1 px raw = 2σ** (`vio_obs_huber_thresh 1.0`; `landmark_block_abs_dynamic.hpp:474-486`). ORB-SLAM3: σ scaled by octave, Huber √5.991 (mono) and √7.815 (stereo) (`O3/Optimizer.cc:2690-2692`). |
| Outliers | Huber only, plus the depth gate, the re-seed and the cheirality guards (`vi_ba_schur.rs:458-667`). | Optical-flow forward-backward check at 0.2 px (`optical_flow_max_recovered_dist2 0.04`, `BA/include/basalt/optical_flow/frame_to_frame_optical_flow.h:220-228`). Stereo epipolar check 0.005 (`:367-378`). The VIO calls no outlier filter (`sqrt_keypoint_vio.cpp:1405` TODO). |
| Observations | ORB, 1000 keypoints, KLT-first. Every observation of a window landmark by any active or fixed keyframe. Left-camera flow only in Stereo mode (G50). | Frame-to-frame KLT on a 50 px grid in **both** cameras (`frame_to_frame_optical_flow.h:160-165`). Every frame in the window and every keyframe observes its tracked landmarks. |

Net weighting: relative to vision, our accel-driven IMU rows carry about 64 × 4 = **256x** Basalt's information ratio, and our gyro rows about 11x. The 4x comes from σ 1 px against 0.5 px. Huber then switches IMU edges off whenever their χ² exceeds 16.9. Basalt's IMU is never down-weighted. This is a code-read ratio, not a measured effect.

## 4. Old states leaving the window

**Basalt.** It uses square-root marginalization into a dense prior on the states that remain (`sqrt_keypoint_vio.cpp:552-1000`).
- **Selection.** States beyond `max_states` are marginalized. A non-keyframe state goes completely; a keyframe state loses only velocity and bias (`:562-613`).
- **Which keyframe goes.** When there are more than 7 keyframes, the first one whose tracked share falls under 10 % is marginalized. Otherwise the DSO distance score picks one, never one of the newest 2 (`:615-685`).
- **What the prior holds.** Linearization runs over every IMU factor between kept or marginalized states, every landmark hosted in a marginalized keyframe, and every lost landmark. QR then gives the square-root prior (`:694-735, 860-895`).
- **First-estimate Jacobians.** The prior is linearized once. The kept states are marked linearized (`setLinTrue`, `:907`). The prior's residual is re-expressed through `delta` (`:958-990`).
- **What survives.** The whole visual and inertial history survives as a consistent Gaussian: velocity, bias and roll/pitch information from every past frame. That includes the bias walk, which ties the current bias to all past IMU data at its true strength.
- **Consistency.** FEJ keeps the 4-DOF nullspace (x, y, z, yaw) unobservable (`checkMargNullspace`, `:509-523`).

**Ours, `ScaledEdge` (default).** Older keyframes are simply frozen (ORB-SLAM3 style, `vi_ba_schur.rs:174-176`).
- **What is kept.** Visual rows from fixed context keyframes, at full weight against frozen poses. One boundary IMU edge at 1e-2, with nav and bias rows both scaled (`:864-876`).
- **What is lost:**
  - all IMU information older than the boundary edge;
  - 99 % of the boundary edge's own velocity and bias information;
  - all uncertainty of the frozen poses, which are treated as exact.
- **Consistency.** Roll, pitch and position of the frozen past are pinned with infinite confidence. That is over-confident: whatever tilt the past has, the window has to inherit. The yaw/position gauge is fixed by rigidity, not by a prior.
- **Why 1e-2.** ORB-SLAM3 does the same: 1e-2 plus Huber √16.92 on the edge `i==N-1` (`O3/Optimizer.cc:2632-2643`). But ORB-SLAM3 keeps its gyro and accel walk edges at full information (`:2646-2657`), and runs `FullInertialBA` after every IMU initialization (`O3/LocalMapping.cc:1304-1309`). We do neither by default.

**Ours, `Marginal(..)` (option, R5).** It is an IMU-only Schur complement of the frozen keyframe's state, the full-weight boundary edge and a pose pseudo-prior of 2 mm / 1 mrad, giving a 15×15 prior on the oldest active keyframe. It uses FEJ, and **vision never enters** (`K/mapping/marginalization.rs:1-14, 43-72`). It is dropped once it has moved more than 0.5 m or 10° (`:36-37`).
- **What it fixes:** the boundary's bias and velocity information.
- **What it does not fix:**
  - the frozen poses, still pinned at σ 1 mrad (0.057°), well below the 0.5-5° tilt errors measured below;
  - the fixed context keyframes, which remain rigid;
  - IMU history beyond one edge.

So `Marginal` is not Basalt's marginalization. It changes the IMU boundary, not map rigidity.

## 5. Solver

| | ours | Basalt |
|---|---|---|
| Linear system | Normal equations with the bipartite Schur complement over 3×3 point blocks. Dense 15P Cholesky (faer) on the reduced keyframe system (`vi_ba_schur.rs:1-27, 2206-2400, 2664`). | ABS_QR: landmarks are eliminated in place by QR per landmark block, then a dense reduced H and LDLT (`sqrt_keypoint_vio.cpp:1110-1120, 1195-1230`). Square-root prior. |
| LM | λ₀ **1.0** (config `:625`). Marquardt diagonal scaling with a 1e-6 floor (`:2648-2653`). Accept if the cost falls: λ/3. Reject: λ×10 (`:2873-2905`). Rejects count against the budget (G37). | λ₀ **1e-4** at every solve (`:1062`). Diagonal × λ with min 1e-6 and max 1e2 (`:1210-1216`). Nielsen-style update on gain ratio (`:1350-1355`). |
| Iterations, convergence | Cap **50** (config `:624`). Stops at relative cost decrease < 1e-6, with the cost compared **in f32** (`:2631, 2869, 2873-2895`). S0 means: MH_01 18.7 iterations per solve, V1_02 29.2, V1_01 29.6. Solves with 3+ rejects: 13-47 per run (`E/scores.jsonl`). | Cap **7** (`vio_max_iterations`). Stops at f_diff < 1e-6 or ‖inc‖∞ < 1e-4 (`:1360-1363`). |
| Precision | f64 system. f32 pose retraction (`:2717-2719`) and f32 reprojection. Both are far below mm level: f32 ulp at 5 m is 0.5 µm. | f32 throughout (section 2). |
| Cost, cadence | Once per **keyframe**, synchronous in S0. Mean 74 ms per solve on MH_01, 190 ms on V1_02, 185 ms on V1_01 (`ba_solve_ms / ba_solves`, `E/scores.jsonl`). | Once per **frame**, about 40 ms per frame for the whole pipeline at load 20-30 (`bench/results/euroc_timing.md`). |

Solver precision and budget are not where the error is. Card V found no score change from a 7-iteration budget on LaMAria. Basalt reaches its accuracy in f32 with 7 iterations.

## 6. Frame tracking against the window

- **Basalt.** No separate tracker pose: every frame is a VIO state with its IMU factor and all its observations (`:322-330`). The output pose is the solved state, so velocity and bias are re-estimated at 20 Hz.
- **Ours.**
  - **Per frame:** visual-only PnP / flow pose. The IMU only seeds and vetoes it (`tracking.pose_inertial: Off`, `K/system/config/mod.rs:481-494`).
  - **Velocity for the prediction:** the keyframe's (`FrameVelocity::KeyframeVelocity`, `:499-509`).
  - **Keyframes:** the keyframe BA then rewrites the keyframe. In SYNC this happens before the frame returns; ASYNC is subject to G54 (`K/mapping/map/ops/correction.rs:93-104`).
- **Measured on S0** (this report, section 10.2):
  - Per-frame translation step error, non-keyframe frames: p50 1.45-4.3 mm against Basalt 0.67-1.76 mm (2-3x).
  - On keyframe frames: 3.8-10.5 mm (4-6x). That is the snap when BA moves the keyframe away from where tracking put it.
  - Rotation steps: p50 2-4x Basalt's. Bursts of 1-2° per 0.05 s occur in the Vicon rooms, against a Basalt maximum of 0.25-0.33°. 54-59 % of the steps over 0.5° fall on keyframe frames, which are 22-29 % of frames.
- **Effect.** Velocity and bias update only per keyframe, and from a 3-keyframe window. Latency is fine in SYNC. In ASYNC the per-frame state is stale by up to one solve (G54).

## 7. Initialization

- **Basalt** (`sqrt_keypoint_vio.cpp:167-213`):
  - The offline calibration biases from `euroc_ds_calib.json:161-185` are subtracted from every raw sample. These are b_a (−0.003, 0.120, 0.067) and b_g (−0.0022, 0.0204, 0.0767), within about 0.03 m/s² and 1 mrad/s of the EuRoC GT means computed here (section 10.3).
  - Attitude comes from one accel sample (`FromTwoVectors(accel, z)`) with velocity 0, so it assumes a static start, which EuRoC has.
  - The VIO then refines everything under the initial priors.
  - So Basalt starts with nearly true biases. That is an unfair advantage against a from-scratch estimator, and it is exactly the quantity our tilt problem is about.
- **Ours:** ORB-SLAM3-style VIBA0/1/2 (`K/inertial/schedule.rs:76-93`). VIBA0 at readiness, VIBA1 after 5 s, VIBA2 after 15 s; each is a world re-alignment.
  - The accel-bias zero prior is **1e5 in all three** (`schedule.rs:24-41, 130-132`; `K/inertial/initialization/solve.rs:161-176`). That is σ = 3.2 mm/s² toward zero.
  - ORB-SLAM3 uses 1e5 for VIBA0/1 but **0 for VIBA2** (`O3/LocalMapping.cc:184-224`), and follows each init with `FullInertialBA` (100 iterations, `:1304-1309`).
  - The comment at `schedule.rs:24-27` says ours keeps 1e5 because the full BA is missing (kornia-slam#51).
- **Consequence (theory, partly measured).** EuRoC's true |b_a| is 0.14-0.16 m/s², almost all on the IMU's y/z axes, which are horizontal for EuRoC's x-up IMU. A prior that forces b_a → 0 makes the solve explain that acceleration as gravity. The result is a tilt of |b_a,yz|/g = **0.82-0.92°** (computed from GT, section 10.3).
- **Evidence.** MH04-DIAGNOSIS found "a constant ~0.9 deg start-up tilt", 88 % hidden in the bias, before any dark span.

## 8. SLAM Dunk in theory (no run; ideas only)

Sources: `experiments/lamaria-handoff.md` (section "The current leader: SLAM Dunk"), `bench/lamaria-top-entries-2026-11-23.md`, and a reading today of the public repo: README, CHANGELOG, `configs/{full,baseline}.json`, `slam_dunk/{frontend,vio,backend,config}.py`, `slam_dunk/native_defaults.json`. The C++ under `slam_dunk/engine/native/` was **not** read today. Line numbers were not available from the fetch.

### Pipeline

1. **VIO seed.**
   - A native binary `aria_vio` (`frontend.py`, `vio.py`) runs with a start-up recovery loop.
   - The frontend docstring names "Basalt flow, RootSIFT, dual IMU and triangulation".
   - Our 2026-11-15 notes say the tracker is OpenCV LK with Basalt vendored for the estimator. These two readings disagree on the flow, so whether the seed is a Basalt `sqrt_keypoint_vio` fork is **not verified**. "Basalt-class seed" is the working assumption.
   - Keyframes: fixed time spacing `BA_STEP = max(0.6, duration/1500)` s (`frontend.py`). SIFT 900 per image, RootSIFT, min 3 observations.
2. **Offline global BA.**
   - Native `batch_joint`, starting from the VIO seed (`backend.py`): 250 iterations × 2 outer loops, a 1200 s budget, `WHITEN_VISUAL`, soft adaptive Cauchy loss (`native_defaults.json`).
   - The README credits Basalt's square-root solver. Today's fetch could not confirm the solver identity or the exact state set.
   - Per the README, this stage alone gives about 69 on the LaMAria training set.
3. **Time-varying camera.** `CHUNK_DT 5`: per 5 s chunk, camera 1 relative to camera 0 plus both focal pairs. Camera 0 is the fixed reference (CHANGELOG 0.1.0). There is a focal random walk `FOCAL_RW .035`.
4. **Gyro response.** `GYRO_REFINE` (`config.py`, flag `gyro_response`). Our notes say it is one full 3×3 matrix; today's fetch could not confirm the parameterization.
5. **Other terms.** Accelerometer alignment, scale and cross-axis (`ACC_ALIGN/SCALE/CROSS`), clock (`TIME_REFINE`), thermal bias (`THERMAL`, `BIAS_THERMAL`). Training score goes from 69.1 to 82.4 with the camera and gyro terms (README).
6. **IMU noise in its defaults.** `SIGMA_G 5e-4`, `SIGMA_A 3e-3`, `BIAS_RW 2e-5`, `ACC_BIAS_RW 5e-3` (units not stated). Their accel white noise is 1.5x ours, not Basalt's 8x.

### What each part would fix of our measured errors

| SLAM Dunk part | our error it addresses | expected effect on EuRoC (estimate) |
|---|---|---|
| Basalt-class VIO seed | the 10 s-piece gap (sections 10.1-10.2) and tilt (10.3), by construction: it starts from Basalt's 0.005-0.030 m pieces and 0.13-0.63° tilt | large; this is the whole Basalt-vs-us gap |
| Offline global VI-BA over all keyframes, IMU at full weight, robust loss | baked map tilt: a batch solve re-levels every keyframe against all IMU data (MH_04 final-map 3.4°, G49). Bias walk at full weight over the whole run (G49/card S). Causal loop jumps (G53), since the output is the batch result. Per-frame jitter, if the dense export interpolates on the IMU (`fuse.py`, not read) | large: Basalt 0.081 m mean → ORB-SLAM3-with-full-BA class |
| 5 s camera chunks (relative extrinsic, focal) | none we measure on EuRoC. EuRoC is a rigid, Kalibr-calibrated stereo rig, minutes long | ≤ 0.002 m; mostly added freedom |
| Gyro 3×3 | none measured. Our gyro-bias error is 0.5-1.2 mrad/s on healthy sequences (section 10.3); a 0.1 % scale error is sub-millimetre over EuRoC paths | ≤ 0.001-0.003 m |
| Accel scale and misalignment, clock, thermal | an accel misalignment would mimic tilt, but EuRoC GT shows a constant bias and no scale error is visible (card S diagnosis on LaMAria; not checked on EuRoC). The clock is Kalibr-calibrated | ≤ 0.003 m |

**Estimate (labelled as an estimate, not a measurement).** SLAM Dunk's design on EuRoC should land at about **0.02-0.04 m ATE per sequence, mean about 0.03 m**. That is ORB-SLAM3 stereo-inertial class (0.019-0.050 m on 80-96 % of frames) or slightly better, with 10 s-piece RMS at or below Basalt's 0.005-0.030 m. Most of the gain comes from the seed plus the batch BA. The calibration terms that give it +13 LaMAria points should be worth about 0 on EuRoC.

For us, that makes R6 (offline global inertial BA on saved sessions) the SLAM-Dunk-class lever for the smoothed output. It does not touch the causal output or the online window.

## 9. Ranked differences

Each item gives the mechanism, the evidence from our runs, the cheapest test, and the rung. "Rung" refers to the master-plan ladder in `experiments/euroc-top1-card.md` §3.

### (a) The 10 s-piece precision gap on healthy SYNC runs

1. **The window carries no history: 3 keyframes clamped to a frozen past, against Basalt's 7 keyframes, 3 frames and a dense FEJ prior** (sections 1 and 4).
   - **Mechanism.** Each solve sees about 1 s of IMU. Its pose is pulled onto landmarks fixed by keyframes that are never revised. Every frozen error passes forward, VO-style. IMU and bias information across windows survives only through a 1e-2 edge. Basalt's prior keeps the IMU history at full weight, with consistent uncertainty.
   - **For:**
     - The **keyframe-only** 10 s pieces are 1.5-2.5x Basalt's, about as much as all frames (section 10.1). So the gap sits in the keyframe estimate, not in the tracker.
     - Window 10 lifted LaMAria R_11 10 s scale from 46-63 % to 95-98 % (G6).
     - The preset arm, which has window 10, beats the default on V1_01 smoothed (0.059 against 0.137), but that comparison is confounded by G54.
   - **Against:**
     - Async T2 (window 10 alone) was catastrophic, but that was G54 + async.
     - Card W: window 10 with loose bias rows makes MH_03 +35-59 % worse (item c1).
   - **Cheapest test (about 10-15 min, SYNC).** S0 plus `local_ba: (boundary_scales_bias_rows: false, active_keyframes: 10, max_iterations: 7, initial_lambda: 1e-4)` on V1_02, then `kfpieces.py`.
     - Prediction: keyframe-only piece median 0.023 → ≤ 0.015 m; MH_03 within +10 %.
     - Kill: no change in the keyframe pieces means the window length is not the lever. Go to item 2.
   - **Rung:** R4 on R3. If R4 alone is not enough, a new rung "R12: Basalt-style windowed marginalization (visual + inertial prior, FEJ, non-frozen context)", which is a code project.

2. **Error model: IMU stiff and switched off by Huber; vision soft and barely robust** (sections 2-3).
   - **Mechanism:**
     - Accel information is 64x Basalt's and gyro 2.8x, while visual σ is 2x Basalt's.
     - Every IMU edge gets Huber at χ² 16.9. Visual Huber sits at 5σ against Basalt's 2σ.
     - So whenever the IMU and the map disagree (tilt, bias, vibration, the 2.5 ms Hold lag), the solve sees large IMU χ². It then either bends the keyframes toward the IMU or discards the IMU edge altogether. Mismatched tracks up to 5 px pull quadratically.
   - **For:**
     - On LaMAria R_01, 34-41 % of interior IMU edges were Huber-downweighted (card S diagnosis). The cause there was mostly the clock (card AA).
     - The S0 keyframe-frame snaps are 4-6x Basalt's (section 10.2).
   - **Against:**
     - Card N: accel noise ×10 on LaMAria improved scale but worsened ATE.
     - Card AA: fixing the chi² did not fix the R_01 scale.
     - Not measured on EuRoC: S0 has no DEBUG chi² lines.
   - **Cheapest test (analysis plus one run, about 3 min).** A DEBUG SYNC V1_02 run to frame 800 for the `[vi_ba]` per-edge `hw` and χ² distribution.
     - If more than 10 % of interior edges have hw < 1, run S0 + `imu_huber: BoundaryOnly` (existing knob) on V1_02 and V2_01.
     - Then a `rig_calibration` RON with Basalt's EuRoC IMU noise. This is an existing run-file field, but untested with `cameras: Stereo`, so verify the header.
     - Prediction: keyframe pieces −20 % or more.
     - Kill: hw ≥ 0.9 on more than 95 % of edges rules out the kernel.
   - **Rung:** new "R11: error model (IMU kernel layout, visual σ/Huber, stereo depth σ)". The visual σ/Huber change needs code; `huber_reproj_px_sq` is not a knob.

3. **Tilt and accel bias wrong by 0.5-5° and 0.1-0.9 m/s²** (item b1; sections 7 and 10.3).
   - **Mechanism.** A constant tilt is absorbed by the per-piece Sim(3). A tilt that changes (dark span, VIBA2, loops) is not, and through `g·Δθ` it bends velocity and position inside a piece.
   - **For:**
     - The sequences with the largest bias error have the largest piece ratios: V2_01 (0.89 m/s², 2.4x), V1_01 (0.47, 1.8x), V2_02 (0.38, 1.3-1.8x), V2_03 (0.50, 2.3x).
     - The sequences with the smallest bias error are at parity: MH_01 and MH_02 (0.10-0.13, 1.0x).
   - **Against:** V1_03 has a moderate bias error (0.19) but the largest ratio (4.3x). The tracker dominates there (item 4).
   - **Cheapest test:** the b1 test below. Re-read the pieces afterwards.
   - **Rung:** new rung (b1) plus R3.

4. **Per-frame pose is visual-only, and the keyframe snap** (section 6).
   - **Mechanism.** Non-keyframe poses are PnP with no gyro fusion. Keyframe frames jump to the BA solution.
   - **For:**
     - Per-frame step error is 2-3x Basalt's (non-keyframe frames) and 4-6x (keyframe frames).
     - On V1_03 and V2_02 the all-frames pieces are 1.7x and 1.4x the keyframe-only pieces (section 10.1). There the tracker *is* the gap.
   - **Against:** on MH_03, MH_04, V1_01 and V2_01 the keyframe-only pieces equal the all-frames ones, so the tracker is not the main term there.
   - **Cheapest test (about 3-4 min per sequence).** S0 + `tracking: (pose_inertial: OrbSlam3)` on V1_03 and V2_02. The knob exists in this binary; card AJ's newer version is in the `kornia-slam-aj` worktree.
     - Prediction: non-keyframe step p50 −30 %; all-frames piece median approaches keyframe-only (V1_03 0.064 → ≤ 0.045).
     - Kill: G30-style gating (inliers lost, scale break) means card AJ's fix is needed first.
   - **Rung:** R7.

5. **Stereo depth σ = 5 % of depth** (section 3).
   - **Mechanism.** EuRoC Vicon-room points are at 1-4 m. With 0.5 px disparity noise, σ_z/z is 1-4 % (`z·0.5/bf`, bf = 50.35). The 5 % model underweights the metric anchor of near points by up to 25x in information, and overweights points beyond 5 m. Basalt and ORB-SLAM3 use right-camera pixel residuals.
   - **For:** none measured on EuRoC.
   - **Against:** card P on LaMAria found disparity weighting did not help (G35), but that was a different rig.
   - **Cheapest test (about 3 min, existing knob).** S0 + `local_ba: (stereo_depth_sigma: Disparity(px: 0.5))` on V1_02.
     - Prediction: keyframe pieces −10 % or more.
   - **Rung:** R11.

6. **Hold integration lags 2.5 ms (half the 5 ms IMU period).** `SN/imu.rs:50-61`.
   - **Mechanism.** The lag gives a rotation error of ω·2.5 ms, which is 0.14° at 1 rad/s, and it grows with the dynamics (V1_03, V2_03).
   - **For:** card AA showed that a 4 ms clock error inflates LaMAria IMU χ² 100x.
   - **Against:** Basalt's own sample handling looks similar (right-held) but is not verified.
   - **Cheapest test (about 3 min, existing knob).** S0 + `inertial: (sample_interpolation: Midpoint)` on V1_03.
   - **Rung:** A5, folded into R11.

7. **Solver budget, λ and f32 convergence check.**
   - **Against:** Basalt is f32 at 7 iterations; card V saw no score change. Low priority.
   - **Rung:** R4. Already covered.

### (b) Tilt and accel-bias absorption (MH_04 3.2-3.8°, LaMAria 1_19)

1. **Zero-mean accel-bias priors bake the tilt in at initialization, and every solve keeps pulling.** VIBA0/1/2 use `PRIOR_A 1e5` (σ 3 mm/s²); local BA applies 1e2 per free keyframe per solve (σ 0.1 m/s²) (sections 2 and 7).
   - **Mechanism.** True b_a is about 0.15 m/s² and horizontal. A prior that forces b_a toward 0 leaves a world tilt of 0.82-0.92°. The map is then built tilted. Later, local BA can only make b_a absorb `g·θ_map`.
   - **For:**
     - MH04-DIAGNOSIS: a ~0.9° constant start-up tilt with 88 % in the bias.
     - S0: accel-bias error 0.10-0.19 m/s² on MH and V1_02/V1_03, almost all in y/z (`kfbias.py`, section 10.3).
     - ORB-SLAM3 frees b_a at VIBA2 and runs a full inertial BA; Basalt starts from calibrated b_a.
   - **Against:**
     - LaMAria card S: "the zero-mean prior does not bias the mean" (R_01, different bias).
     - The S0 MH b_a norm is 0.12-0.14, not 0, so the local solves do move it. The init prior is not the only force.
   - **Cheapest test.**
     - Analysis first (0 runs): a DEBUG S0 MH_01 run to frame 400 (about 1 min) logs the VIBA0/1/2 b_a and gravity rotation. Prediction: |b_a| ≤ 0.02 at each VIBA, and the tilt after VIBA2 is within 0.2° of |b_a,GT,yz|/g.
     - Then a code arm: VIBA2 `prior_a` 0 (ORB-SLAM3), plus `accel_bias_prior_weight` 0. Neither is a knob today; card S stage B is not built.
     - Prediction for MH_04 and V1_02 SYNC: tilt median ≤ 0.5°, b_a error ≤ 0.05, ATE no worse than the band.
   - **Rung:** new "R3b: bias priors (VIBA2 free b_a, no zero-mean local prior)", next to R3. It also needs kornia-slam#51: ORB-SLAM3's post-init `FullInertialBA` (online R6-lite).

2. **Map rigidity: fixed context keyframes and landmarks freeze roll and pitch forever. Only loops, which are 4-DOF, ever correct the past** (section 4).
   - **For:**
     - MH_04: tilt held 14 s in bright frames until the loop.
     - Final-map keyframes keep 3.4°.
     - V2_01 S0: b_a jumps from 0.07 to 0.85 m/s² error at about 20 s and stays at 0.8-1.3 for the remaining 100 s (`kfbias_t.py`), which implies about 5° of map tilt. A burst of 1-2° frame rotation steps occurs at 19.65-20.65 s (`spikes.py`).
     - The jump time matches VIBA2's 15 s schedule after window start, but that is not verified: S0 has no VIBA log.
   - **Against:** none so far.
   - **Cheapest test (about 2 min).** A DEBUG SYNC V2_01 run to frame 450, logging the VIBA2 time, world rotation and b_a.
     - Prediction: VIBA2 fires at 19-20.5 s and rotates the world by ≥ 2°.
     - Kill: no VIBA event there means a visual event. Then read the window logs around kf 382-418.
   - **Rung:** R6 (offline) for the smoothed output. Online: R3b's post-init full inertial BA, and R12 for the window.

3. **Bias rows at 1e-2 on the boundary (G49).**
   - **Mechanism.** The bias is re-chosen per window, so the tilt/bias trade is cheap.
   - **For:**
     - Walk z 4-11 on MH_04 async.
     - Card S: full rows bring walk z to 1.3 (mechanism confirmed).
   - **Against:**
     - Card S: no R_01 score gain.
     - E1 async MH_04 was +85 % worse (one irreproducible async run).
   - **Cheapest test (about 2 min).** R3 arm S1 on MH_04 and V2_01 SYNC, with `tilt2.py` / `kfbias.py`.
     - Prediction: MH_04 tilt at 65-79 s ≤ 1.5°.
     - Kill K1 (tilt ≥ 3°) confirms item 2.
   - **Rung:** R3.

4. **IMU Huber on every edge.**
   - **Mechanism.** The edges with high dynamics, which carry the tilt information, are the ones that get downweighted.
   - **For:** LaMAria (card S diagnosis).
   - **Against:** not measured on EuRoC.
   - **Test:** the item a2 DEBUG run.
   - **Rung:** R5 (`imu_huber`) or R11.

### (c) Why window 10 with loose bias rows hurts

1. **A longer window turns the shared bias into a free per-window parameter.**
   - **Mechanism.** The interior walk edges tie the 10 keyframes' biases together over 2-4 s, at full weight. The tie to the past is only 1/10 σ. So the window has one free bias vector that can absorb a rotation or tilt error spread over 10 keyframes. With window 3 the visual lever is shorter, and this matters less.
   - **For:**
     - Card W: gyro-bias error 5-6x on MH_03 at window 10.
     - MH_04 preset: gyro walk z p50 15.2, and a gyro-bias range of 4-8 mrad/s against 0.01-0.02 in GT.
     - Card S fixture: full rows cut the active=10 accel-bias error 2-35x.
     - Basalt keeps bias rows at full weight (`imu_block.hpp:73-92`).
   - **Against:** none.
   - **Cheapest test.** R4 on R3: window 10 + full rows on MH_03 SYNC (about 5-10 min).
     - Prediction: MH_03 within +10 % of window 3; gyro walk z p50 ≤ 2.
   - **Rung:** R4 on R3.

2. **Fewer pinned points, so the gauge slides** (card O).
   - **Mechanism.** At window 10 only about 28 % of points are fixed, against 88 % at window 3. With a 1e-2 boundary and Huber, the window rotates and slides as a whole. Loose bias rows then let tilt and bias co-rotate.
   - **For:** card O's shift counts (median 7.7-13.6 cm per solve at 10 against 1.1-1.2 cm at 3).
   - **Against:** card O: a full-weight boundary edge (O1/O2) was even worse, though not as the ORB-SLAM3 literal.
   - **Cheapest test:** the same R4 run, reading the `[vi_ba] WINDOW shift` field (DEBUG).
   - **Rung:** R4, then R12.

3. **The zero-mean accel prior scales with the number of free keyframes.**
   - **Mechanism.** The prior is 1e2 per free keyframe (`vi_ba_schur.rs:2845-2854`). With the biases tied by the walk, the effective pull grows from 3e2 to 1e3 (σ from 0.058 to 0.032 m/s²). That pushes more of the true 0.15 m/s² bias into tilt at window 10.
   - **For:** code read only. The MH_04 preset tilt (4.5-4.9°) is above the default's (3.3-3.8°).
   - **Against:** IMU information on b_a should dominate the prior inside a window (theory), so the effect may be small.
   - **Cheapest test:** the R3b code arm at window 10 on MH_04.
   - **Rung:** R3b.

## 10. New measurements on the S0 runs (body frame, SE(3), 5 ms association)

### 10.1 10 s pieces, Sim(3)-aligned per piece, median RMS in m

Source: `pieces.py` and `kfpieces.py` on `E/runs/S0_*_s0_r1` and the bench Basalt trajectories.

| seq | Basalt | ours smoothed, all frames | ours smoothed, keyframes only | ours causal | ours no-loop |
|---|---|---|---|---|---|
| MH_01 | 0.007 | 0.007 | - | 0.010 | 0.008 |
| MH_02 | 0.007 | 0.007 | - | 0.010 | 0.008 |
| MH_03 | 0.015 | 0.018 | 0.016 | 0.025 | 0.017 |
| MH_04 | 0.018 | 0.031 | 0.031 | 0.027 | 0.027 |
| MH_05 | 0.019 | 0.028 | - | 0.030 | 0.026 |
| V1_01 | 0.010 | 0.018 | 0.017 | 0.020 | 0.015 |
| V1_02 | 0.009 | 0.027 | 0.023 | 0.034 | 0.034 |
| V1_03 | 0.015 | 0.064 | 0.037 | 0.067 | 0.067 |
| V2_01 | 0.005 | 0.011 | 0.012 | 0.015 | 0.013 |
| V2_02 | 0.012 | 0.022 | 0.016 | 0.026 | 0.018 |
| V2_03 | 0.030 | 0.068 | - | 0.122 | 0.039 (partial) |

- The keyframe-only pieces use 8-14 pieces with at least 12 keyframes each.
- An every-5th-frame control matches the all-frames median within 0.003 m, so keyframe density does not bias the comparison.

### 10.2 Per-frame steps (0.05 s), errors against GT

Source: `tsteps.py` and `spikes.py`.

| seq | translation step p50 / p90 (mm): Basalt | ours, non-keyframe frames | ours, keyframe frames | rotation step p50 / p99 (deg): Basalt | ours |
|---|---|---|---|---|---|
| MH_01 | 0.80 / 1.87 | 1.45 / 4.69 | 3.79 / 9.39 | 0.013 / 0.055 | 0.023 / 0.178 |
| MH_04 | 1.76 / 4.21 | 4.30 / 17.4 | 10.5 / 38.9 | - | 0.046 / 0.493 |
| V1_02 | 1.36 / 2.87 | 3.74 / 13.1 | 7.67 / 26.9 | 0.025 / 0.145 | 0.092 / 1.196 |
| V2_01 | 0.67 / 1.39 | 1.84 / 7.21 | 4.86 / 17.6 | 0.020 / 0.119 | 0.046 / 1.147 |

- **Startup re-rotation.** Each of our runs has one 15-23° rotation step at 1-4 s. That is the causal output crossing the inertial-initialization world re-alignment. It inflates RPE-1 s yaw RMS (eval: ours 0.25-2.3° against Basalt 0.06-0.26°). Medians are the fair figure.
- **Rotation at 1 s.** Excluding that step, the 1 s rotation error p50 is 0.10 / 0.34 / 0.21° (MH_01, V1_02, V2_01) against Basalt's 0.07 / 0.16 / 0.13°. The p90 is 0.20 / 1.31 / 1.67° against 0.31 / 0.31 / 0.25°.
- **Frame convention.** Hand-eye between our body frame and GT is 0.16-0.49°, so the frame convention is right.

### 10.3 Tilt and biases

- **Tilt error:** the angle between the body-frame "up" of ours and GT after a rotation-only alignment of the whole run (`pieces.py`). Median / p90, deg:
  - Basalt 0.13-0.71 / 0.30-1.06 on every sequence except V1_01 (1.69 / 3.08; the V1_01 GT orientation looks odd for every system).
  - Ours smoothed:

    | seq | MH_01 | MH_02 | MH_03 | MH_04 | MH_05 | V1_02 | V1_03 | V2_01 | V2_02 | V2_03 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | tilt median / p90 | 0.39/0.58 | 0.40/0.65 | 0.26/0.50 | 0.78/2.40 | 1.20/1.48 | 0.72/1.43 | 1.19/2.33 | 2.17/5.16 | 0.78/3.60 | 2.23/4.14 |

- **Keyframe accel-bias error against GT,** median, m/s², with the implied tilt `|Δb_a,yz|/g` (`kfbias.py`):

  | seq | MH_01 | MH_02 | MH_03 | MH_04 | MH_05 | V1_01 | V1_02 | V1_03 | V2_01 | V2_02 | V2_03 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | error | 0.10 | 0.13 | 0.10 | 0.17 | 0.17 | 0.47 | 0.19 | 0.19 | 0.89 | 0.38 | 0.50 |
  | implied tilt (deg) | 0.53 | 0.73 | 0.57 | 0.96 | 0.93 | 2.69 | 1.06 | 1.05 | 5.18 | 2.14 | 2.93 |

  The y/z share of the error is 95-99 % everywhere.
- **Gyro-bias error,** median: 0.5-1.2 mrad/s, except V1_03 (8.5), V2_02 (2.4) and V2_03 (4.5).
- **EuRoC GT accel bias:**
  - MH_01: (−0.026, 0.137, 0.076), |yz| 0.157 m/s² (0.92° if forced into gravity).
  - MH_04: 0.87°.
  - V1_02: 0.82°.
  - V2_01: 0.85°.
- **Basalt's starting calibration** b_a is (−0.003, 0.120, 0.067), within 0.03 m/s² of GT.

## 11. Explicit uncertainty

What I could not verify:
- **Basalt preintegration.** basalt-headers is not in the bench tree, so the integration scheme, its sample hold and its covariance details are unverified.
- **SLAM Dunk internals:**
  - whether the seed is Basalt `sqrt_keypoint_vio` (the two sources disagree on the flow);
  - the global BA solver's identity (Basalt square-root per the README, not confirmed in code);
  - whether the global BA states include all frames or keyframes only;
  - the IMU factor form;
  - the gyro 3×3 parameterization;
  - loop closure in the base entry;
  - the units of its noise values.
  The C++ was not read, and there are no EuRoC numbers for it anywhere. Section 8's EuRoC band is an estimate.
- **The EuRoC IMU χ² and Huber-weight distribution.** It needs a DEBUG run, so items a2 and b4 are code-read plus LaMAria evidence only.
- **The V2_01 20 s event.** Its timing matches VIBA2, but S0 has no VIBA log line.
- **Whether `rig_calibration` works with `cameras: Stereo` on EuRoC.** This matters for the IMU-noise arm of a2.
- **Whether `tracking.pose_inertial: OrbSlam3` in binary `cea7646…` is card AJ's version.** It is probably the older card H code, which has the G30 gating problem.
- **Tilt metric.** It uses one rotation alignment over the whole run. It does not separate a constant from a drifting tilt, and V1_01's GT gives about 1.7° even for Basalt.
- **Statistics.** S0 is one deterministic run per sequence (start frame 0), with no 3-start band yet. Per-sequence differences under about 15 % are not distinguishable from start-frame sensitivity.
- **Basalt reference.** It is one blocking run at load 20-30. Not re-run.
- **Fairness.** Basalt's offline-calibrated biases give it a start our estimator never gets. Any "beat Basalt" claim should note it.
