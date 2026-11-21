# PR #180 card D: offline global VI-BA with calibration states (A7, A8)

The LaMAria handoff ("Recommended way forward", step 0) names offline global visual-inertial BA plus calibration states as the main lever. The evidence is SLAM Dunk's README, whose offline BA alone reaches about 69 on training and 82.4 with drifting camera calibration and gyro intrinsics, and visloc-rs's batch BA. Lane D builds three things, each behind a knob that is off by default:
- a session dump of the online run;
- an `offline-ba` subcommand that solves one sparse VI-BA over the whole recording;
- three calibration states: camera drift, gyro intrinsics and clock.

Nothing here has been measured yet. The validation workflow scores this card.

Licences: SLAM Dunk is AGPL-3.0 and visloc-rs is ideas only (owner, 2026-11-10). Neither repo's code was read or ported for this card. Every number quoted from them comes from their docs, as cited.

## Experiment: D, offline global VI-BA with calibration states
Gap tested: G10 (A7, global BA absent) and G11 (A8, fixed calibration), on the scratch ledger `slam-viba/audit/gap-ledger.md`. The score link is G18: drift and scale, not coverage, is the score gap. The segment bridge also tests G1: a loss ends in a disconnected trajectory. G24 (causal against smoothed output) sets the baseline; see Baselines.
Base: kornia-slam@`pr180/S` (`42ee054`), kornia-rs@`4ada5a7`. The lane starts from the skeleton tip after Lane C.   Binary tag: `pr180/L-D` (lane), `pr180/F` (integrated).

Mechanism (file:line on `pr180/S`):
- **No global solve exists.** VI-BA is a window of `mapping.local_ba.active_keyframes` = 3 keyframes (`system/config/mod.rs:283,309`), solved by a dense Schur system (`mapping/vi_ba_schur.rs:1-30`, `:1257`).
  - The past enters only through the boundary edge, scaled 1e-2 (`config/mod.rs:285,310`).
  - Nothing ever re-solves a keyframe once it leaves the window. So the online scale wander (0.55-4x, G4) and the per-CP error growth (3_18: 4.7 m rising to 627 m, scale 1.285; 5_11: up to 317 m; G18) stay in the written trajectory.
- **Segments stay disconnected.** A give-up re-seeds the same `Map` at `pose_before_loss` (audit A1). With `tracking_loss.reseed.keep_inertial: false`, the default (`system/recovery.rs:47-50`), no IMU edge spans the gap. Only the kept-inertial path publishes a bridge (`system/mod.rs:1966-1985`).
  - Give-ups on the rg1 base: R_11 5, 1_19 3, 2_11 26, 3_18 14, 5_11 8 (`s5/out/rg1/*/stderr.log`, `[summary]`).
  - Sim3 fails on R_11 and 2_11. The R_11 splice prices the give-up disconnect as failing the Sim3 on its own.
- **Calibration is fixed.** The camera extrinsics (`RigCamera::primary_from_camera`, `kornia-sensors/src/rig.rs:44`; `ImuCalibration::camera_to_body`, `:125`) are constants, and so are the pinhole intrinsics and the IMU timeline (`source/lamaria.rs:33-35`, `imu_time_offset_ns`).
  - The preintegrator applies no gyro scale or misalignment (`kornia-sensors/src/imu.rs:211-215`: `gyro = measurement.gyro - bias.gyro`).
- **The inputs a global solve needs are already in memory.** They are lost at exit:
  - keyframes with their velocity and bias (`mapping/map/keyframe.rs:9-23`);
  - landmark associations;
  - IMU edges that keep their raw samples (`mapping/map/imu_factor.rs:5-27`, `repreintegrate` at `:30-41`);
  - the smoothed-trajectory records (`apps/kornia-slam-app/src/trajectory.rs:48-90`).
  - `--dump-map` writes landmarks only (`apps/kornia-slam-app/src/map_dump.rs`).

Theory:
- **Global VI-BA (A7).** One LM solve has every keyframe's pose, velocity and biases, every landmark, and one IMU edge per consecutive keyframe pair at full information. It observes metric scale everywhere: accelerometer excitation from walking, against the reprojection terms. The online solve sees scale only inside a 3-keyframe window tied to a 1e-2 boundary.
  - So the 10 s and 60 s scale ratios should approach what a long window gives. On R_11, window 10 already took the 10 s in-band share from 46-63 % to 95-98 % (handoff, "Results so far").
  - Heading drift is a flat direction. The solve keeps its starting drift: visloc found the BA ends near the start's heading drift on sequence_1_20 (-5.3° in, -4.6° out), and the cost barely separates drift-free from drifted solutions (`visloc-rs/docs/lamaria_batch_ba.md:846-886`). So this card predicts a scale gain, not a heading gain (K1 separates the two).
- **Segment bridge (G1).** With `offline_ba.bridge_segments`, the last keyframe of a segment and the first of the next are joined by an IMU edge integrated from the raw stream over the gap.
  - The edge is built only for gaps up to 10 s; VINS-Fusion skips IMU factors with `sum_dt > 10.0` (`estimator.cpp:1057-1058`).
  - Gravity, scale and biases then hold across the gap, and the problem becomes one connected component. That is the property the Sim3 needs and the online system lacks.
  - Over 8 s the edge is weak in position: a 0.05 m/s² bias error gives `0.05 · 8² / 2 = 1.6 m`. It is strong in rotation: gyro noise 1.7e-4 rad/s/√Hz × √8 s = 0.03°.
- **Calibration states (A8).**
  - **Camera drift, rotation.** Each camera's IMU-camera rotation, per time slice, with a random-walk prior between slices. visloc: 8 slices at 0.05°.
    - On 1_19, cam0 is 0.58° off and cam1 0.04°. On device 1WM093701G1276, both cameras turn together by about 0.1° over a recording (`lamaria_batch_ba.md:86-151`).
    - Five-track mean: 61.41 to 64.36 (`:153-200`).
  - **Camera drift, focal.** Per camera, per time slice, one focal scale. visloc names the ratio of image angular scale to gyro scale as the cause of its scale drift (`:285-384`).
    - Its estimate by device: +0.26 to +0.57 % on 1WM093701G1276 by day, -0.06 to -0.11 % on 1WM09370081264.
    - Five-track mean: 66.24 to 69.96. It loses on the moving platform (5_11: 57.15 to 48.02), where the focal absorbs the IMU-vision conflict.
  - **Gyro intrinsics.** `ω ← (I + S) ω` with `S` symmetric: three axis scales and three non-orthogonality terms. The antisymmetric part is a pure rotation of the gyro frame, which the camera-rotation states and the fixed IMU-camera rotation absorb.
    - visloc fitted this form against the pseudo-GT (`:1256-1310`). The z scale is -0.12 to -0.15 % on 1WM093701G1276 and +0.19 to +0.22 % on 1WM09370081264; non-orthogonality is -0.3 to -1.1 mrad.
    - Its fixed correction hurt its chain (mean 72.61 to 70.08), while SLAM Dunk credits an in-solve gyro response, together with drifting camera calibration, with 69.1 to 82.4. That conflict is why P14 has a wide score range.
  - **Clock.** One IMU-camera time offset `td` per sequence, as in VINS-Fusion (`estimator.cpp:1042-1043`, `1084-1086`). visloc's sweep finds the minimum within 0.5 ms of its device table (`:822-842`), so the prediction is a small value and a small effect.
  - **Re-preintegration from raw IMU.**
    - With any calibration state on, every IMU edge is re-preintegrated from the raw stream at the start of each LM iteration: at the current bias, gyro correction and `[t_i - td, t_j - td]`. This is ORB-SLAM3's `Preintegrated::Reintegrate` (`ImuTypes.cc:168-175`).
    - With all calibration off, the solve uses the first-order bias correction, as `FullInertialBA` does after `SetNewBias` (`Optimizer.cc:491`).
- **Best from each reference.**
  - **Problem shape: ORB-SLAM3 `FullInertialBA`** (`Optimizer.cc:392-811`).
    - States: per-keyframe body pose, velocity and gyro and accel bias vertices (`:423-453`).
    - IMU terms: `EdgeInertial` plus `EdgeGyroRW` and `EdgeAccRW`, whose information is the preintegrated covariance blocks (`:532-563`).
    - Landmarks are marginalised (`:606`), which is the Schur complement. Per-octave information and Huber √5.991 on reprojection (`:592`, `:643-649`). Sparse linear solver (`:402`).
    - Body-pose vertex with per-camera `Tcb` (`G2oTypes.cc:25-68`, update `:192-219`). We take body-frame states from it: then every camera's extrinsic, the primary's included, can carry a calibration state.
  - **Robustness and stopping: visloc.** Outlier drop at 10 px, then at 3 px from iteration 8; a landmark damping floor of 1 % of its largest Hessian diagonal; 60 iterations or a relative cost drop below 1e-7 (`lamaria_batch_ba.md:30-84`, `:758-801`).
    - visloc measured each of these choices; ORB-SLAM3 runs a fixed 7 iterations after a loop (`LoopClosing.cc:2286`) and 100 at IMU init (`LocalMapping.cc:1307-1309`).
  - **IMU kernel.** ORB-SLAM3's global BA puts Huber √16.92 on every inertial edge (`Optimizer.cc:540-542`). GTSAM as visloc uses it has no kernel. Default: no kernel (visloc's measured 64.36), with ORB-SLAM3's choice selectable. This also corrects the audit's R1(a) claim that no reference puts a kernel on inner IMU edges; that claim holds for local BA only, and card A owns that question.
  - **Linear solver: faer supernodal sparse Cholesky,** f64, AMD ordering, symbolic factorisation once, as `sparse_pgo.rs:303-338` does in f32.
    - visloc uses a banded LAPACK Cholesky and needs PCG once revisits break the band (`:567-650`). A general sparse factor needs no band assumption.
    - The audit's A13 says f64 is required; no reference runs f32.
  - **Calibration parametrisation.**
    - visloc for the per-slice rotation and focal, and for the symmetric gyro form.
    - OpenVINS uses a triangular 6-parameter `Dw` plus a gyro rotation (`ov_msckf/src/state/State.h:91-100`), each behind its own `do_calib_*` flag (`StateOptions.h:47-56`). We take the flag-per-state structure; the symmetric form avoids a second rotation gauge.
    - VINS-Fusion for one scalar `td`.
    - SLAM Dunk only for the list of terms (`temporal_camera`, `gyro_response`, clock), from its README.
- **Gauge.** Visual-inertial BA leaves 4 DoF unobservable: position and yaw. The first keyframe of each connected component gets a prior on position (σ 1 mm) and yaw (σ 1e-4 rad), and roll and pitch stay free, because gravity observes them.
  - visloc pins the full rotation at 1e-4 rad and found that loosening it halves the IMU cost but splits the CP scores both ways (`:1044-1058`).
  - Pinning roll and pitch would be wrong for a component whose online world was never gravity-aligned: a segment re-seeded cold.

Knob: see the knob table in Design. Every field is in a new run-file section `offline_ba`, which only the `offline-ba` subcommand reads. The online system never reads it, so a run file with or without it gives the same online run. The calibration states and the bridge default to off. `--session-out` is an output-only CLI flag.

Baselines:
- **Dumps.** For each of the 5 proxy sequences (3_18, 1_19, 2_11, 5_11, R_11) and EuRoC MH_01, the validator runs `pr180/F` with knobs off and `--session-out`, `--trajectory-out` and `--trajectory-smoothed`.
  - On LaMAria this is 3 runs per sequence: base, +50 µs and -50 µs IMU offset.
  - Record sha, binary md5 and load per run.
- **Score baseline.** The baseline for every score prediction below is the **smoothed** trajectory of the same run, scored by the official evaluator. It is not the causal one, so the offline gain is not credited with the G24 re-anchoring gain.
  - The rg1 causal numbers are context only: CP 7.37 (1_19), 4.94 (3_18), 5.68 (5_11); Sim3 fails on 2_11 and R_11; proxy mean 3.60.
- **Size of the inputs on rg1** (`[summary]` and the last `pts=` in each stderr.log):

| sequence | frames | keyframes | landmarks | online RSS |
|---|---|---|---|---|
| R_11 | 9,547 | 1,620 | 45,789 | 0.56 GB |
| 1_19 | 18,352 | 4,310 | 158,615 | 1.37 GB |
| 2_11 | 23,748 | 4,134 | 144,349 | 1.42 GB |
| 3_18 | 33,878 | 8,160 | 360,594 | 2.54 GB |
| 5_11 | 22,483 | 5,391 | 232,979 | 1.69 GB |

Predictions (numbers, each tied to a measurement):
  P1 Online outputs with `--session-out` against without, on MH_01 `euroc.ron` and 3_18 `lamaria-rig-imu.ron`: 0 differing bytes in every other output file.   measured by: `cmp` per file of the two run dirs.
  P2 Round trip: `offline-ba` with `offline_ba.max_iterations: 0` writes, for every frame of `--trajectory-smoothed`, a pose within 1e-9 m and 1e-9 rad of it. It writes the same timestamps, in the same order. It is not expected to `cmp` equal, because it re-anchors in the body frame, so float order differs.   measured by: a pose-diff script over the two files; the `[offline_ba] frames=` count equals the smoothed file's line count.
  P3 Peak RSS of `offline-ba` with all calibration on: 3_18 0.8-2.5 GB, never above 4 GB; R_11 0.15-0.8 GB.   measured by: `/usr/bin/time -v`, "Maximum resident set size".
    Arithmetic for 3_18:
    - states `8,160 × 15 = 122,400` DoF;
    - observations, at 300-700 per keyframe, are `2.4-5.7 M × 32 B = 80-180 MB`;
    - landmarks `360,594 × 120 B = 43 MB`;
    - the reduced lower triangle has `8,160 × (120 diag + 225 IMU + c × 36 visual)` entries, with one-sided covisibility `c = 15-60`, so `7-20 M` entries × 16 B (value and row index) = `0.11-0.33 GB`;
    - the Cholesky factor fills near the band: `122,400 × 15c = 27-110 M` entries × 8 B = `0.22-0.88 GB`;
    - the raw IMU is `1,694 s × 1 kHz × 56 B = 95 MB`;
    - the calibration border is `122,400 × 55 × 8 B = 54 MB`.
    The sum is 0.6-1.6 GB, and the range adds allocator and dump-decoding overhead. It holds only if the observation pattern is assembled into a block pattern fixed once, not as faer triplets per iteration: triplets would be `Σk² × 36 × 16 B` and over 10 GB.
  P4 Linear-system counters on 3_18: median one-sided covisible keyframes per keyframe `c` 15-60; `nnz(L) / nnz(M)` 1.5-8 with AMD.   measured by: `[offline_ba] covis_median= nnz_m= nnz_l=`.
  P5 Time on 3_18, single thread (`faer::Par::Seq`, which also keeps the solve deterministic):
    - calibration off: 2-30 s per LM iteration (factorisation `n·b² = 122,400 × (225-900)² = 6e9-1e11` flop at 5-10 GFLOP/s, plus a 1-10 s Schur assembly);
    - calibration on: up to 20 s more per iteration (`1.69 M samples × 15` re-integrations, for 1 nominal, 2 `td` and 12 gyro central differences);
    - whole solve at most 60 min on 3_18 and at most 10 min on R_11.
    measured by: `[offline_ba] iter_ms=` lines and `time -v` wall time.
  P6 Convergence, core solve (calibration off, bridge off) on all 5 proxy sequences: final cost at most 0.5x the initial cost, observations dropped by the 3 px stage 1-5 % (visloc dropped 1-2 % of SIFT observations; ORB associations carry more mismatches), and no LM step accepted that raised the cost (0 by construction).   measured by: `[offline_ba] cost0= cost= dropped_10px= dropped_3px=`.
  P7 Mechanism, scale: the 10 s in-band scale share (the handoff's `r11_series.py` metric).
    - R_11 core solve: from the smoothed baseline (window 3; handoff range 46-63 %) to 90-99 %.
    - 1_19, 3_18 and 5_11: at least 85 % each, and at least 15 points above each one's own smoothed baseline.
    measured by: `slam-viba/r11_series.py` on the offline and smoothed trajectories of the same dump.
  P8 Mechanism, connection:
    - Bridge off: connected components = (give-ups + 1) minus the segments the online map re-joined through shared landmarks. Predicted: R_11 2-6, 2_11 5-27, 3_18 3-15.
    - Bridge on: exactly 1 component on every sequence whose gaps are all at most 10 s; otherwise 1 plus the number of gaps over 10 s.
    - The whitened rotation residual of the bridged edges, mean over edges after the solve, is at most 3.
    measured by: `[offline_ba] components= bridged= gap_max_s= bridge_rot_whitened=`.
  P9 CP score, core solve (bridge off, calibration off) against the smoothed baseline: 1_19, 3_18 and 5_11 each gain +3 to +30 points (absolute CP 8-40); 2_11 and R_11 still fail the Sim3.   measured by: `evaluate_wrt_control_points`, 3 samples (dumps) per sequence.
  P10 CP score, core solve plus `bridge_segments: true`: the Sim3 passes on at least 1 of R_11 and 2_11, and that sequence scores 5-50. The 5-sequence proxy mean (a failed Sim3 counts 0) goes from the smoothed baseline (rg1 causal context: 3.60) to 8-30.   measured by: the proxy runner, 3 samples per sequence.
  P11 The reference gain is not predicted for this build. visloc's 59.70 to 64.35 (and 56.42 to 56.99 over ten sequences, online plus one pass; `lamaria_batch_ba.md:1675-1711`) rests on three things: revisit matching, `loop-correct` (a pose graph over revisit PnP edges) and a second solve that adds the revisit observations (`:567-650`). It also starts from an online result near 56, not 3.6.
    - Without revisit matches, visloc's loop stage is worth +1.7 of that (61.41 to 63.12).
    - Revisit matching is a prerequisite and this lane does not build it, so any gain of the reference's size is out of scope here.
    measured by: none in this round; the results file records it as "prerequisite missing".
  P12 Camera rotation per slice (8 slices), on top of core plus bridge:
    - 1_19: cam0's mean rotation 0.35-0.8° and cam1's at most 0.2°.
    - R_11, 2_11, 3_18 and 5_11: both cameras 0.08-0.35°, at most 45° apart in axis; range of the common rotation over the slices at most 0.2°.
    - CP change per scored sequence: -5 to +8.
    measured by: `[offline_ba] cam<i> slice<s> rot_deg=` and the CP runner.
  P13 Focal scale per slice, on top of core plus bridge: mean per camera +0.2 to +0.6 % on 2_11, 3_18 and R_11; -0.2 to 0 % on 1_19; +0.4 to +0.7 % on 5_11. 5_11 loses 2-12 CP points against the solve without focal. The other scored sequences change by -3 to +8.   measured by: `[offline_ba] cam<i> slice<s> focal_pct=` and the CP runner.
  P14 Gyro intrinsics, on top of core plus bridge:
    - z scale -0.08 to -0.20 % on R_11, 2_11, 3_18 and 5_11, and +0.14 to +0.27 % on 1_19;
    - non-orthogonality xy and xz each -1.4 to -0.2 mrad;
    - mean whitened IMU rotation residual per edge falls 10-50 %;
    - CP change per scored sequence -5 to +5.
    measured by: `[offline_ba] gyro_scale_pct= gyro_shear_mrad= imu_rot_whitened0= imu_rot_whitened=` and the CP runner.
  P15 Clock, on top of core plus bridge: `|td| ≤ 2 ms` on all 5 sequences (the reported `td_ms` is the residual on top of the offset the dump already applied; `td_total_ms` = applied offset + `td_ms` is the new offset; theory review D12). The 4 sequences of 1WM093701G1276 agree within 0.7 ms (compare `td_total_ms`). The CP change lies inside the 3-sample band of the solve without `td`.   measured by: `[offline_ba] td_ms=` and the CP runner.
  Mechanism counters: the P7 scale share, the P8 component count, and the calibration estimates in P12-P15. If the theory is right, each must move as predicted before its score prediction counts.

Kill rules:
  K1 (mechanism moved, score did not) P7 holds on 1_19, 3_18 and 5_11, yet P9 gains less than +2 on all three, on all 3 samples -> scale is fixed and the score is limited elsewhere. Measure the heading error per 120 s window and the per-CP error against the path before adding any calibration state. The flat heading direction is the first suspect (`lamaria_batch_ba.md:846-886`).
  K2 (score moved, mechanism did not) P9 or P10 gains +5 or more on a sequence, yet its P7 share moved less than 10 points and its P8 component count is unchanged -> the gain comes from another cause: outlier drop, re-anchoring or landmark re-triangulation. Attribute it with the `max_iterations: 0` round trip (P2) and a solve without the 3 px stage before crediting A7.
  K3 Memory: P3 above 4 GB, or `nnz(L)` above 3e8 on 3_18 -> the reduced system is denser than the band theory says (long-range re-observations). Do not raise the limit (laptop OOM cap). Add a landmark observation span cap, visloc's `--band 64`, as a new knob in a new card.
  K4 Divergence: on any sequence the cost rises over the solve, or a keyframe that is not bridged moves more than 50 m from its start -> the start is outside the basin. That can come from a bad accepted pose (R_11 frame 5437, 1096 m, G21) seeding landmarks. Re-run that sequence from a dump recorded with `tracking_loss.pose_gate` on, and record which start converges.
  K5 Bridge: with `bridge_segments` on, a sequence whose Sim3 passed without it loses more than 5 CP points -> the bridged edge drags the next segment (cold-restart velocity and bias). Keep the bridge off. Next: seed the bridged segment's velocity from the edge's preintegrated Δv before the solve.
  K6 Identifiability: a calibration estimate leaves its P12-P15 range by more than 2x (rotation above 2°, focal above 1.5 %, gyro scale above 0.5 %, `|td|` above 3 ms, or a same-device spread of `td` above 1.5 ms) -> the state absorbs another error (association, scale drift). Keep that knob off; do not tune its prior to pass.
  K7 A calibration state moves as P12-P15 predict, yet the CP change is inside the 3-sample band -> the device model is right, but it is not the error that limits our trajectories now. Record it as "built, not adopted"; re-test after K1's limiter is fixed.
  K8 Round trip: P2 misses -> the dump or the re-anchoring is wrong. No offline score is trusted until P2 holds.
Bit-identity:
- P1: `--session-out` changes no online output.
- `pr180/F` with knobs off must still `cmp` equal `pr180/S` (the plan's identity gate). Lane D cannot break it: the online path only gains the eviction copy, which runs only when a map bound and recording are both set, and neither the app nor any config sets a bound.
- P2: the offline solve with 0 iterations equals the smoothed output within 1e-9.
Noise:
- The offline solve is deterministic given a dump (`faer::Par::Seq`, single-threaded assembly), so repeating it measures nothing.
- The samples are the 3 dumps per LaMAria sequence from the online ±50 µs runs. That measures how the offline result depends on its online start.
- Known online band from the ledger (G3, G20): 1_19 CP rg1 0.29-7.37, 3_18 0.49-6.65.
- With 3 samples per config on the 5 proxy sequences, every score verdict is provisional until the deciding sequences have 3 samples on both sides.

## Design (what the lane builds)

### Knobs

All fields are in the run file's top-level section `offline_ba` (`RunConfig::offline_ba`, `apps/kornia-slam-app/src/config.rs:24-29`). Only `offline-ba` reads them. "Old behaviour" means no offline pass at all. Every behaviour switch below defaults to off, and the numeric fields are the defaults of a pass that someone asked for.

| RON path (run file) | type | default | reference |
|---|---|---|---|
| `offline_ba` | `OfflineBaConfig` | absent: `OfflineBaConfig::default()` | |
| `offline_ba.max_iterations` | `usize` | `60` | visloc `--iterations 60` (`:758-801`); `0` = round trip (P2) |
| `offline_ba.relative_cost_tolerance` | `f64` | `1e-7` | visloc `--rel-tol` |
| `offline_ba.outlier_px` | `f64` | `10.0` | visloc |
| `offline_ba.tight_outlier_px` | `f64` | `3.0` | visloc |
| `offline_ba.tighten_from_iteration` | `usize` | `8` | visloc |
| `offline_ba.landmark_damping_floor` | `f64` | `0.01` | visloc (1 % of the largest diagonal entry) |
| `offline_ba.imu_kernel` | `OfflineImuKernel` = `None` \| `Huber` | `None` | `Huber` = ORB-SLAM3 `FullInertialBA` δ √16.92 (`Optimizer.cc:540-542`) |
| `offline_ba.bridge_segments` | `bool` | `false` | VINS-Fusion 10 s IMU-edge limit (`estimator.cpp:1057-1058`) |
| `offline_ba.calibration.time_slices` | `usize` | `8` | visloc `--ext-segments 8` |
| `offline_ba.calibration.camera_rotation` | `Option<RotationSlicePrior { step_sigma_deg: f64, value_sigma_deg: f64 }>` | `None` | visloc 0.05° step (`:120-151`); value prior 1.0° (ours) |
| `offline_ba.calibration.camera_focal` | `Option<FocalSlicePrior { step_sigma: f64, value_sigma: f64 }>` | `None` | visloc 0.0005 step, 0.01 value (`:345-348`), relative to the calibrated focal |
| `offline_ba.calibration.gyro_intrinsics` | `Option<GyroIntrinsicsPrior { sigma: f64 }>` | `None` | `S` entries, unitless; suggested 0.005 |
| `offline_ba.calibration.clock_offset` | `Option<ClockOffsetPrior { sigma_ms: f64 }>` | `None` | VINS-Fusion `td`; suggested 2.0 |

- **Error model.** The visual error model is not a new knob. It is the run file's `system.pipeline.OrbSlam.mapping.local_ba.visual_information` and `stereo_depth_sigma`, with the Huber of `vi_ba_schur.rs:789` (`visual_huber`), so offline and online weight observations the same way.
- **Session dump.** `--session-out <file>` is a CLI flag next to `--dump-map`, not a RON field. It only writes output.
- **`validate()`** returns a `ConfigError` for:
  - a non-finite or non-positive sigma, tolerance or pixel threshold;
  - `tight_outlier_px > outlier_px`;
  - `tighten_from_iteration == 0`;
  - `landmark_damping_floor` outside `(0, 1]`;
  - `time_slices` outside `1..=64`.
- **CHANGELOG.** One line per knob, written by the skeleton.

### Solver (`mapping/global_vi_ba/`)

- **States.**
  - Per keyframe: body pose `T_bw` (world to body; the IMU residual takes it through `preintegration::residual_and_jacobians` with `t_bc = None`, `kornia-factors/src/inertial/preintegration.rs:62-80`), velocity, gyro bias and accel bias.
  - Per landmark: its world position, eliminated by the Schur complement.
  - The calibration border: `2 × slices × 3` rotation, `2 × slices` focal, 6 gyro and 1 `td`, at most 55 columns.
- **Start.**
  - The dump's final keyframe poses are converted to body frame with the factory `camera_to_body`, with the dump's velocity, bias and landmarks.
  - A keyframe whose velocity is exactly zero (its segment never initialised) starts at `(p_{k+1} - p_{k-1}) / (t_{k+1} - t_{k-1})` with zero bias.
- **IMU edges.** One per consecutive keyframe pair in a segment, built from the raw stream. It does not reuse the online chain, so every segment, including its pre-init keyframes, gets the same edges. Across a segment boundary an edge is added only with `bridge_segments`. No edge is built over a gap longer than 10 s.
- **Reprojection.**
  - The body-frame chain `T_cw = T_cb(slice) · T_bw`, through Lane C's `pinhole_point_jacobian` (2x3), as ORB-SLAM3's `EdgeMono` does through `Rcb` (`G2oTypes.cc:349`).
  - The focal column is `∂u/∂s = fx · x/z` and `∂v/∂s = fy · y/z`.
  - Stereo depth rows reuse `vi_ba_schur`'s depth residual (made `pub(crate)`).
- **Linear system.**
  - The block pattern of `M = A - B C⁻¹ Bᵀ` is built once from the observation and edge lists, as a faer `SparseColMat<usize, f64>` lower triangle plus a block-to-value-offset map.
  - The values are refilled in place each iteration, and the symbolic `SymbolicLlt` is built once. An observation dropped as an outlier keeps its pattern slot, as a zero.
  - The calibration border is solved by a Schur complement on the sparse factor (55 right-hand sides), so the factor never sees a dense row.
- **Calibration Jacobians.**
  - Rotation and focal: analytic, through the projection.
  - Gyro and `td`: central differences of re-preintegration, with steps `1e-6` (S) and `0.25 ms` (`td`, a quarter of the 1 kHz sample spacing).
  - The cost is bounded by P5. Analytic `∂Δ/∂S` would have to be added to `PreintegratedImu::integrate`, the hot path of every online user.
- **Output.**
  - Every tracked frame is re-anchored in the body frame: `T_bw(frame) = (T_bc · frame_from_kf · T_cb) · T_bw(kf)`, with factory extrinsics, because the online relative motion was measured with them.
  - Written in the LaMAria format through the existing `TrajectoryWriter`. Coasted frames are not written, as online.
- **Report.** `[offline_ba]` lines on stderr with every counter named in P2-P15, plus `--report <file.json>` with the same fields.

### Session dump

- **Library.** `mapping/session.rs` holds the dump types and `snapshot_map`.
- **App.** `apps/kornia-slam-app/src/session.rs` records:
  - the IMU samples as fed to the system (after factory rectification and `imu_time_offset_ns`);
  - the smoothed-trajectory records.
  At exit it writes CBOR through `ciborium`, which is already a dependency (`apps/kornia-slam-app/Cargo.toml:37`).
- **No-evict recording.** `SlamSystem::set_session_recording(true)` makes `enforce_map_bound` (`system/mod.rs:286-310`) copy each keyframe it removes into the recorder before removal. The dump then holds every keyframe ever made. The online map behaves exactly as without it.
  - Nothing sets a map bound in this repo, so on LaMAria and EuRoC the copy never runs. The hook exists for bounded maps (the robot).
- **Size.** About 200 MB for 3_18: 95 MB of IMU, 80-180 MB of observations, 9 MB of landmarks. Descriptors are not dumped, because revisit matching is not in this card.

### CLI

- **Online run, unchanged:** `kornia-slam --config <run.ron> [--session-out <file>] ...`.
- **Offline pass:** `kornia-slam --config <run.ron> offline-ba --session <file> --trajectory-out <file> [--report <file>]`.
  - It is an optional `argh` subcommand. argh 0.1.19 supports `Option` subcommand fields (`argh_derive-0.1.19/src/lib.rs:138`), so every existing invocation parses as before.
  - With the subcommand given, `main` runs only the offline pass. `--config` supplies the `offline_ba` section and the visual error model.

## Shared types and signatures the skeleton (Phase 2c) declares for this lane

`todo!()` bodies, reachable only through `--session-out` or the `offline-ba` subcommand.

```rust
// crates/kornia-slam/src/mapping/session.rs
pub const SESSION_DUMP_VERSION: u32 = 1;

/// Rotation as a unit quaternion `[x, y, z, w]` and translation, metres.
pub struct SessionPose { pub rotation_xyzw: [f64; 4], pub translation: [f64; 3] }

pub struct SessionCamera {
    pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64,
    pub width: u32, pub height: u32,
    /// `T_PC`, as `RigCamera::primary_from_camera`.
    pub primary_from_camera: SessionPose,
}

pub struct SessionRig {
    pub cameras: Vec<SessionCamera>,
    /// Primary camera to body (IMU), `X_b = T_bc · X_c`.
    pub camera_to_body: SessionPose,
    /// `[gyro_noise, accel_noise, gyro_bias_noise, accel_bias_noise]`, as `ImuCalib`.
    pub imu_noise: [f64; 4],
    /// Already applied to `SessionDump::imu`; recorded for the report.
    pub imu_time_offset_ns: i64,
}

pub struct SessionObservation {
    pub landmark: u32, pub camera: u8, pub octave: u8,
    /// Undistorted pixel in `camera`.
    pub u: f32, pub v: f32,
    /// Stereo depth (m) when the frame had one.
    pub depth: Option<f32>,
}

pub struct SessionKeyframe {
    pub kf_idx: u64, pub timestamp_sec: f64,
    /// World to primary camera, `T_cw`.
    pub pose_cw: SessionPose,
    pub velocity_world: [f64; 3], pub bias_gyro: [f64; 3], pub bias_accel: [f64; 3],
    pub segment: SegmentId,
    pub inserted_while_lost: bool, pub anchor: bool,
    /// Removed from the live map by a bound before the dump.
    pub evicted: bool,
    pub observations: Vec<SessionObservation>,
}

pub struct SessionLandmark { pub idx: u32, pub position_world: [f64; 3] }

/// One tracked frame as `SmoothedTrajectory::record` saw it.
pub struct SessionFrame { pub timestamp_sec: f64, pub reference_kf: u64, pub frame_from_kf: SessionPose }

pub struct ImuSample { pub t: f64, pub gyro: [f64; 3], pub accel: [f64; 3] }

pub struct SessionDump {
    pub version: u32,
    pub rig: SessionRig,
    pub keyframes: Vec<SessionKeyframe>,
    pub landmarks: Vec<SessionLandmark>,
    pub frames: Vec<SessionFrame>,
    pub imu: Vec<ImuSample>,
}

/// Keyframes (live, then `evicted`) and live landmarks of `map`, tagged by `segment_of`.
pub fn snapshot_map(
    map: &Map,
    rig: &SensorRig,
    evicted: &[Keyframe],
    segment_of: impl Fn(usize) -> SegmentId,
) -> (Vec<SessionKeyframe>, Vec<SessionLandmark>);
```

All of these derive `serde` behind the crate's `serde` feature. Plain arrays avoid needing serde on kornia-rs types.

```rust
// crates/kornia-slam/src/mapping/global_vi_ba/mod.rs
pub struct OfflineBaConfig { /* the knob table */ }
impl OfflineBaConfig { pub fn validate(&self) -> Result<(), ConfigError>; }

pub struct OfflineBaResult {
    /// Per dump keyframe, in dump order: the solved `T_bw`.
    pub body_poses: Vec<Pose3d>,
    pub calibration: CalibrationEstimate,
    pub report: OfflineBaReport,
}

#[derive(Debug, thiserror::Error)]
pub enum OfflineBaError { /* NoImu, VersionMismatch(u32), CholeskyFailed(String), InvalidConfig(..) */ }

pub fn offline_visual_inertial_ba(
    session: &SessionDump,
    visual: &VisualInformation,
    stereo_depth_sigma: StereoDepthSigma,
    config: &OfflineBaConfig,
) -> Result<OfflineBaResult, OfflineBaError>;
```

```rust
// crates/kornia-sensors/src/imu.rs
/// Symmetric gyro correction `ω ← (I + S) ω`: `scale` on the diagonal, `shear` = (xy, xz, yz).
pub struct GyroIntrinsics { pub scale: Vec3F64, pub shear: Vec3F64 }
impl GyroIntrinsics { pub fn apply(&self, gyro: Vec3F64) -> Vec3F64; }
```

```rust
// crates/kornia-slam/src/system/mod.rs
impl SlamSystem {
    pub fn set_session_recording(&mut self, on: bool);
    /// Keyframes a map bound removed while recording, oldest first.
    pub fn evicted_keyframes(&self) -> &[Keyframe];
    pub fn segment_of(&self, kf_idx: usize) -> SegmentId;
}
```

From other lanes:
- **`SegmentId`** (skeleton, card A "Skeleton contract"): `pub type SegmentId = u32;` in `mapping/map/mod.rs`, re-exported as `crate::mapping::SegmentId`. `mapping/session.rs` imports it from mapping, never from `system`. A `u32` alias needs no serde derive of its own. The skeleton also ports `Segments` with its bodies from `4a2ee7b`, the `segments: Segments` field, the `segments.begin` call in `finish_bootstrap` and `segment_of`, so the tagging runs with Lane B's knobs off and before Lane B lands (integration order C, D, B, A). Tagging is bookkeeping only.
- **Lane C:** `upstream::kornia_3d::refine_analytic::pinhole_point_jacobian(fx, fy, point_cam) -> [Vec3F64; 2]` (card C). Lane D composes its own body-frame chain from it, as Lane A does.
- **faer:** used directly in f64. Lane D does not import `sparse_pgo` internals (card C).

## Files Lane D touches

- **New:**
  - `crates/kornia-slam/src/mapping/session.rs`;
  - `crates/kornia-slam/src/mapping/global_vi_ba/{mod.rs, problem.rs, reduced.rs, imu_edges.rs, calibration.rs, report.rs}`;
  - `apps/kornia-slam-app/src/session.rs`.
- **Edited:**
  - `system/mod.rs`: the eviction copy in `enforce_map_bound` (body only);
  - `crates/kornia-sensors/src/imu.rs`: the `GyroIntrinsics::apply` body;
  - `apps/kornia-slam-app/src/{main.rs, config.rs}`: bodies of `--session-out`, the `offline-ba` subcommand and `RunConfig::offline_ba`;
  - `apps/kornia-slam-app/src/trajectory.rs`: a `SmoothedTrajectory` accessor for its records.
- **Skeleton-owned (review round 3b'; lane D fills bodies only):**
  - `mapping/mod.rs`: the `session` and `global_vi_ba` `mod` lines;
  - `mapping/vi_ba_schur.rs`: `depth_residual_and_jacobian` and `visual_huber` made `pub(crate)`, no other change;
  - `system/mod.rs`: `set_session_recording`, `evicted_keyframes`, `segment_of` and their fields;
  - `crates/kornia-sensors/src/imu.rs`: the `GyroIntrinsics` declaration;
  - `apps/kornia-slam-app/src/{main.rs, config.rs}`: `RunConfig.offline_ba`, `--session-out` and the optional `offline-ba` subcommand, with `todo!()` bodies;
  - `mapping/map/mod.rs`: `SegmentId`.
- **Not touched:**
  - `configs/*.ron`;
  - the online VI-BA solve (`vi_ba_schur.rs` bodies);
  - `kornia-factors` (the IMU residual is reused as is);
  - `loop_closure/*`;
  - `CHANGELOG.md` (the skeleton and Phase 6 write it).
- **Commits:** two, so K8 can block the solver without the dump.
  - `feat(app): session dump of keyframes, landmarks, IMU and frames (--session-out)`;
  - `feat(mapping): offline global VI-BA with calibration states (offline-ba)`.

## Open for the card review (choices this card made that the plan does not fix)

1. **Where the offline knobs live.** They are in the run file's top-level `offline_ba` (app `RunConfig`), not in `SystemConfig`. The online system never reads them, which keeps P1 and the identity gate trivially safe. The alternative is `system.pipeline.OrbSlam.offline_ba`.
2. **What "no-evict recording" means.** Here it means the recorder copies removed keyframes and online behaviour stays the same. It does not mean that recording disables eviction. The latter would change bounded-map runs.
3. **Body-frame states and a 4-DoF gauge** (position plus yaw), in place of visloc's full first-pose pin.
4. **Numeric Jacobians for gyro intrinsics and `td`,** in place of analytic preintegration Jacobians in kornia-sensors.
5. **One IMU edge per consecutive keyframe pair, rebuilt from the raw stream,** in place of reusing the online `imu_factors` chain.

## Not in this card

- **Revisit matching, `loop-correct` and the second solve** (visloc `:567-650`). They are the prerequisite for P11's reference gain. Next card, after K1 is read.
- **The yaw deformation mode** (visloc `--modes yaw`, `:887-970`). It is the fix for the flat heading direction if K1 fires on heading.
- **Accelerometer intrinsics, thermal terms and the final refinement pass** (SLAM Dunk's further terms).
- **Coasted frames in the offline output** (G9), and per-track choices of knobs. The owner wants as few tweaks as possible.
- **Moving the sparse Schur solver to `upstream/kornia_3d/`.** It is generic, but the plan's upstream module list does not include it this round.

## Sources

- ORB-SLAM3, `/home/edgar/dev/.flux-connect-scratch/bench/src/ORB_SLAM3/`:
  - `src/Optimizer.cc:392-811` (`FullInertialBA`);
  - `src/G2oTypes.cc:25-68, 192-219, 349`;
  - `src/ImuTypes.cc:168-175, 263-274`;
  - `src/LoopClosing.cc:2283-2286`;
  - `src/LocalMapping.cc:1307-1309`.
- VINS-Fusion `vins_estimator/src/estimator/estimator.cpp:1042-1043, 1057-1058, 1084-1086`.
- OpenVINS `ov_msckf/src/state/State.h:91-100`, `StateOptions.h:47-56`.
- visloc-rs docs (ideas only): `/home/edgar/dev/.flux-connect-scratch/bench/visloc-rs/docs/lamaria_batch_ba.md` (lines cited above).
- SLAM Dunk: README findings as summarised in `experiments/lamaria-handoff.md`; code not read.
- Our evidence:
  - `slam-viba/audit-2026-11-13.md` (R13, A13);
  - `slam-viba/audit/gap-ledger.md` (G1, G10, G11, G18, G24);
  - `slam-viba/audit/window-results.md`;
  - `slam-viba/estimator-plan.md` (S8);
  - `bench/lamaria-plan.md` (S8);
  - rg1 run logs under `slam-viba/s5/out/rg1/`.
