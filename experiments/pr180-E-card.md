# PR #180 card E: the 1 ms frontend (rig KLT, LK tricks, threading)

Round 2 of PR #180 (plan `pr180-frontend-1ms.md`, sections 2-6, 8 and 10). Lane T builds it on top of the kernels from lane K:
- KLT on every rig camera, so a flow-only rig frame keeps every camera's correspondences;
- the LK tricks read from SLAM Dunk: fewer well-spread Shi-Tomasi tracks, level 0 from the predicted start, a forward-backward check, a gyro-rotated start ray and an optional anchor guard;
- the speed items: window-only gradients, no eager pyramid, no image copy or compare, an analytic pose in place of the epipolar filter plus RANSAC PnP, visibility bookkeeping at keyframes only, and keyframe ORB on the mapping thread;
- one camera per core through long-lived worker threads.

Every behaviour sits behind a knob whose default is today's behaviour. Nothing here has been measured. The validation workflow scores this card on the laptop proxy, R_11 and `nvidia-orin01`.

Licences: SLAM Dunk is AGPL-3.0, so ideas only. Its numbers here come from `experiments/lamaria-handoff.md:169-187` (read 2026-11-15), not from its code. VINS-Fusion, Basalt and ORB-SLAM3 are cited from `/home/edgar/dev/.flux-connect-scratch/bench/src/`.

`$R` = this repo, `$K` = `/home/edgar/.cargo/git/checkouts/kornia-rs-0edb86845af2bca0/4ada5a7/crates`, `$S` = `/home/edgar/dev/.flux-connect-scratch`. Every line number is on `c0a1dc3`, and the paths without a prefix are under `$R/crates/kornia-slam/src/`.

## Experiment: E, 1 ms rig frontend
Gap tested: two new rows, which the docs phase adds to `experiments/gap-ledger.md`. Card E does not edit the ledger.
- **E-rt (runtime):** a rig frame costs about 49.6 ms on one Orin core (plan section 2), against a 15 Hz real-time budget on one core and a 1 ms goal on 2 cores. This is not a score row; its "score at stake" is `none (runtime)`.
- **E-rig (the flow-only rig frame drops the secondary cameras):** KLT-first on R_11 tracked 61.0 % against 71.1 %. It was lost at frame 4850, right after a keyframe, and stayed in Bootstrap until 6866 (`$S/slam-viba/frontend-speedup.md:579-581`, commit `3d84baf`). Score at stake: unknown. R_11 fails the Sim3 either way, so the effect is on tracked share, not on CP.
- The forward-backward check and the track spread also touch G18 (drift, through outlier tracks into PnP), but card E makes no score claim for G18.

Base: kornia-slam@`c0a1dc3`, kornia-rs@`4ada5a7`. Lane T starts from the round-2 skeleton tag `pr180b/K`.   Binary tag: `pr180b/L-T` (lane), `pr180b/F` (integrated).

Mechanism (file:line on `c0a1dc3`):
- **KLT-first is off for rigs.** `system/build.rs:151-156` filters `klt_min_tracks` out with `!rig.is_multi_camera()`, so every rig frame runs ORB on every camera: `frontend/mod.rs:171-172` calls `timed_orb` per camera, and that runs `detect_and_extract_u8` at `:223` with 1000 keypoints. The overlap match follows at `:182`. Plan section 2 prices this at 14.8 ms plus 1.5 ms on x86.
- **A flow frame is the primary camera alone.**
  - `tracking/tracker.rs:270-324` (`flow_frame`) builds `OrbFeatures` from the primary survivors and leaves `rig: RigFeatures::default()` at `:321`. The frame then has no `camera_ranges`, so `view_grids` (`tracking/pose_estimation/map_projection/mod.rs:1024`) treats it as mono, and the rig solve at `:794` never sees another camera.
  - `tracker.rs:434-435` snaps flow seeds against the primary keypoints only.
  - That is the theory of the R_11 loss at 4850 (E-rig). It was never verified frame by frame.
- **The flow set only shrinks between ORB frames** (303 against 372 matches, p50), and **no forward-backward check** exists (`$S/slam-viba/audit/klt-first-diagnosis.md:29-37`).
  - `TrackSet::spawn` (`tracking/optical_flow.rs:122`) is called only from ORB reconciliation.
  - `KltTracker::survivors` / `accepts` (`optical_flow.rs:554-609`) filter on status, `max_error` 20 (`:256`), movement and border, never on a round trip.
- **Per-frame work the 1 ms budget cannot carry:**
  - `system/mod.rs:711-719` copies the whole primary image into `previous_image` on every frame. `optical_flow.rs:527` clones `current` into the cache, and `:694-713` byte-compares the cached image against `previous` to validate it. Plan section 2 estimates the three passes at 90 µs.
  - `optical_flow.rs:491-529` builds a full u8 pyramid with whole-image i16 Scharr derivatives for every frame (`LkPyramidU8::new(..., true)` / `rebuild`): 1,100 µs measured, of which pyramid 400 and Scharr 300.
  - `optical_flow.rs:521` collects a fresh `Vec` of previous pixels per call, and `tracker.rs:244-268` (`predicted_pixels`) collects another one, reading the map under the lock (`system/mod.rs:1532`).
  - LK forward is 23 µs per track at 21×21 over levels 0-1 (9,300 µs for about 410 tracks): `$S/slam-viba/frontend-speedup.md:708-712`.
  - `estimate_from_seeds` (`map_projection/mod.rs:468-512`) runs the fundamental-matrix RANSAC (`:486-488`, `ransac_fundamental_8point` at `:573`, primary only at `:551`) and then `solve_pnp`: 300 µs plus 700 µs.
  - Visibility bookkeeping (`system/mod.rs:1676-1705`) locks the map, selects the local map and projects it into the frustum on every tracked frame, flow frames included.
- **The LK kernel on Orin is scalar.** `$K/kornia-imgproc/src/optical_flow_pyr_lk/fixed_point.rs` has SSE2/AVX2 paths only behind `#[cfg(target_arch = "x86_64")]` (`:33-36`, `:418`, `:430`, `:526`, `:536`, `:684`, `:694`, `:726`). aarch64 runs the scalar fallback.
- **The window is fixed at 21 for the f32 kernel.** `$K/kornia-imgproc/src/optical_flow_pyr_lk.rs:39-44` says that only `win_size == 21` is supported, and `:1472`, `:1505` enforce it. The fixed-point kernel accepts any odd window of 3 or more (`fixed_point.rs:291-292`).
- **Keyframe ORB is on the tracking thread.** At `FlowOnly::KeyframeDue` (`system/mod.rs:1894-1899`), `tracking_step` calls `resolve` (`:1574`), and that runs `extract` (`:629-638`): ORB on both cameras, 29.6 ms on Orin. `insert_keyframe` (`:2344`) then runs on the tracking thread. The async mapping worker (`mapping/local_mapping.rs:336-338`) receives only the BA job (`KeyframeJob`, `:42-52`).
- **No worker threads for flow.** kornia-slam has no rayon dependency (`crates/kornia-slam/Cargo.toml`). The only long-lived workers are local mapping (`local_mapping.rs:336`) and loop closing (`loop_closure/worker.rs:169`).

Theory:
- **Rig KLT (changes 1-2).**
  - With a `RigFeatures { geometry, camera_ranges }` flow frame, the flow-only rig frame gives `solve_rig_pnp` (`tracking/pose_estimation/rig_pnp.rs:59`) the secondary camera's rays. This is the same residual ORB-SLAM3 uses for a multi-camera body (`EdgeSE3ProjectXYZOnlyPoseToBody`, `rig_pnp.rs:3-7`).
  - The two Aria cameras look about 90° apart. A pose from two non-parallel fields of view is constrained in the directions where one camera alone is weak: translation along its optical axis, and rotation against translation about its image axes. So a flow frame should hold where today's primary-only flow frame lost tracking at 4850.
  - ORB then runs only at keyframes. With about 21.5 % keyframes (the traced schedule, `frontend-speedup.md:712`), the 29.6 ms ORB cost leaves 78.5 % of the frames.
- **Well-spread tracks (change 2).** LK cost scales with the number of tracks, while pose information saturates once the field of view is covered.
  - 48 tracks per camera at a 23 px minimum spacing cover 640×480 about as evenly as SLAM Dunk's 160 do. Its spacing and refill rule are taken (23 px, refill below 80 %, 9 px border), but not its count, because the 1 ms budget pays for 48.
  - 48 tracks per camera × 2 cameras is 96 rays, against about 300 on today's mono flow frames. The analytic GN needs at least `min_inliers` of them, and K6 counts the frames where it falls back.
- **Level 0 from the predicted start, 11×11 (change 3).** With a trusted IMU prediction, the start pixel is within about 1-2 px of the answer (plan section 4: 2.5 iterations on average). Coarse levels only buy capture range. The 11×11 window costs 0.94 µs per iteration against 2.6 at 21×21 on x86 (plan section 2), at a smaller convergence basin. That is why `klt_retry: CoarseLevel { max_level: 1 }` reruns the tracks that failed at level 0 after the join, as VINS-Fusion does: with a prediction it tracks at level 1, and below 10 successes it retries with 3 levels (`VINS-Fusion/vins_estimator/src/featureTracker/feature_tracker.cpp:118-133`).
- **Window-only gradients, no eager pyramid, no copy (change 4).** About 96 windows of 11×11 (plus a 1 px border for Scharr) touch about 1.3 % of the two images' level-0 pixels. Scharr on those patches alone replaces the whole-image passes at `optical_flow.rs:491-529`. A frame-id key (the system's own frame counter) replaces the byte compare at `:694-713`. The tracking core then makes one padded level-0 copy per camera, because `process_rig` borrows the images (`system/mod.rs:572-576`); it does not also copy into `previous_image` at `:711-719`.
- **Forward-backward on the cached template (change 5).** The forward solve ends at the current pixel, and the template there is exactly the next frame's forward template. Solving it backward into the previous frame therefore costs 1.5 iterations and no extraction of its own. A round trip above 0.6 px marks a track that LK pulled onto a different patch: an occlusion edge, a repeated texture or a specularity. Dropping it before the pose solve removes the outliers that today reach PnP and stay in the track set (`klt-first-diagnosis.md:33-36`).
- **Analytic Gauss-Newton pose (change 6).** From the IMU prediction, with outliers already removed by the round trip, the pose needs no RANSAC hypothesis. Three Huber GN iterations from a start within a few pixels converge, as ORB-SLAM3's `PoseOptimization` does from its motion-model start (`ORB_SLAM3/src/Optimizer.cc:814`, `:1001-1012`: 4 rounds of 10 iterations at χ² 5.991). The epipolar filter is redundant with the round trip for the outliers it catches, and it relates primary views only (`map_projection/mod.rs:551`). RANSAC PnP stays as the counted fallback below `min_inliers`.
- **Bookkeeping at keyframes only (change 7).** On a flow frame no projection search runs, so ORB-SLAM3 counts no visibility either (`Tracking.cc:3357,3379`, cited at `klt-first-diagnosis.md:28`). Counting visibility there culled 35 % more landmarks (`klt-first-diagnosis.md:26-31`). Skipping `system/mod.rs:1676-1705` on flow frames removes that bookkeeping lock and the bias in one step. Patch A fixed only the counter; this removes the work. The other lock on a flow frame is in `flow_frame` (`tracker.rs:274-324`), which reads every survivor's descriptor and culled flag under the map `Mutex` (`system/mod.rs:1532`, `:1579`); lane T step 10 removes it on the preset path, so a flow frame no longer waits on a mapping merge.
- **Keyframe ORB on the mapping thread (section 6).** ORB-SLAM3's tracking thread only creates the keyframe and hands it over (`Tracking.cc:3216`, `:3335`). Its mapping thread processes keyframes from a FIFO (`LocalMapping.cc:284-287` push, `:83` `ProcessNewKeyFrame`, `:92` culling, `:101` `CreateNewMapPoints`). Card E does the same, and moves ORB itself there too, so a keyframe frame costs the tracking thread 980 + 120 µs.
- **One camera per core (change 10).** Shards share nothing but read-only `Arc<LkFrame>`s and write their own buffers. So the parallel section scales with the shard count, minus the sync cost (plan section 3: 15 µs on 2 cores, 50 µs on 3). Basalt parallelises the same way, over cameras (`basalt/include/basalt/optical_flow/frame_to_frame_optical_flow.h:126`, `:147`) and over points (`:236`).
- **Best from each reference, per mechanism:**

| Mechanism | Taken from | Why it beats the alternatives |
|---|---|---|
| Round-trip threshold 0.6 px, on `max_error` 20 | SLAM Dunk (0.6 px; error cap 35) | VINS-Fusion uses 0.5 px (`feature_tracker.cpp:135-146`) and Basalt 0.3 px (`optical_flow_max_recovered_dist2 = 0.09`, `basalt/src/utils/vio_config.cpp:51`, used at `frame_to_frame_optical_flow.h:214-226`). Basalt's SE2 patch on a u16 pyramid converges finer than a translation-only fixed-point window, so its 0.3 px would reject correct tracks here. SLAM Dunk runs our LK. Our existing cap of 20 is tighter than its 35 and stays, with no knob (plan section 8). |
| Backward solve reuses the next frame's template | ours | All three references re-run a full pyramidal LK backward (`feature_tracker.cpp:139`, `frame_to_frame_optical_flow.h:219`); the cached template avoids that extraction. |
| Shi-Tomasi, spacing, empty cells, refill below 80 % | SLAM Dunk (23 px, 80 %, 9 px border, half resolution, `cornerSubPix`) | VINS-Fusion masks circles of `MIN_DIST` around tracks and runs `goodFeaturesToTrack` on the whole image (`feature_tracker.cpp:55-81`, `:184`). Basalt detects in 50 px grid cells (`vio_config.cpp:50`). SLAM Dunk's half-resolution, empty-cells-only form is the cheapest that keeps the spacing. |
| Coarse level only on retry | VINS-Fusion (`feature_tracker.cpp:118-133`) | SLAM Dunk always runs 3 levels; Basalt runs all `optical_flow_levels` (`frame_to_frame_optical_flow.h:251`). A retry pays coarse levels only for the failed tracks. |
| Gyro-rotated start ray | SLAM Dunk | VINS-Fusion predicts from the full pose (`hasPrediction`), and we predict from the map point at the IMU pose (`tracker.rs:244-268`). A track without a map point gets no prediction today; the rotated ray gives every track one, with no map lock. |
| Analytic GN pose from the prediction | ORB-SLAM3 `PoseOptimization` (`Optimizer.cc:814-1012`) | No reference runs RANSAC on frame-to-frame flow tracks against a map. The round trip has removed the outliers RANSAC exists for. |
| Keyframe processing on the mapping thread, FIFO | ORB-SLAM3 (`LocalMapping.cc:284-287`) | Basalt and VINS-Fusion have no descriptor keyframes. |
| Per-camera parallel flow | Basalt (`frame_to_frame_optical_flow.h:126,147`) | VINS-Fusion is single-threaded. |
| Long-lived named workers, not a pool | ours (`local_mapping.rs:336` pattern) | Basalt uses `tbb::parallel_for`. `rayon::scope` allocates per frame, cannot pin and shares the global pool with ORB (plan section 3). |

Knob: 18 fields under `system.pipeline.OrbSlam.frontend`; see "Knobs" in Design. Each defaults to today's behaviour, and the Orin presets set them.

Baselines:
- **Identity base:** `c0a1dc3` with all knobs off, on EuRoC MH_01 (`euroc.ron`) and LaMAria R_11 (`lamaria-rig-imu.ron`). Record sha, binary md5, config and load.
- **Tracked-share base for K4:** `pr180/S` (`42ee054`) on R_11, rig, 3 samples (0 and ±50 µs IMU offset). The 71.1 % figure is from an older binary (`frontend-speedup.md:536`) and is context only; the validator records the `pr180/S` value.
- **Timing base:** `kornia-slam-frontend-bench` at `c0a1dc3`, `--mode today` and `--mode klt-first`, on R_11 frames, laptop i7-9750H, `taskset -c 8`, `RAYON_NUM_THREADS=1`, at load within ±2 of the knob-on run. On Orin, the same bench at `c0a1dc3` on one core.
- **Inlier-ratio base for P10:** flow-frame inliers over flow seeds on mono EuRoC MH_01 with `klt_first` at `c0a1dc3`. The plan's 84 % is the value this card predicts against; the validator re-measures it.

Predictions (numbers, each tied to a measurement):
  P1 With every card-E knob at its default: `cmp` shows 0 differing bytes against `c0a1dc3` in every output file on MH_01 and R_11.   measured by: `cmp` per file of the two run dirs.
  P2 Orin 2-core preset on R_11 (48 tracks per camera), run with `flow_threads` 1, 2 and 3 on the same machine: every output file is `cmp`-identical across the three runs, the timing CSV excepted. The 64-track preset repeats this with its own three runs.   measured by: `cmp`; and `tests/flow_zero_alloc.rs`, second test.
  P3 After 10 warm-up frames: 0 allocations per `FlowPool::step` for 1, 2 and 3 threads.   measured by: `cargo test --release -j2 -p kornia-slam --test flow_zero_alloc` (counting allocator).
  P4 Laptop proxy, `--mode klt-rig --flow-threads 1`, 48 tracks per camera, on the 758×572 pinhole export (lane F is deferred, owner 2026-11-17): critical-path median at most 800 µs. Only the per-pixel rows scale with the area ratio over 640×480 (1.41): copies and refill. Template, LK, filter and pose rows cost per track and do not scale. The pinhole unproject replaces the fisheye one. The re-derived x86 rows: IMU 30, copies 56, start pixels 20, templates 164, forward 226, backward 135, filter 4, unproject 10 (pinhole, estimated), refill 56, pose 68, bookkeeping 30; about 800 µs in all. Each stage's median is at most 1.0x its row.   measured by: the bench's per-stage µs CSV.
  P5 Orin, 2 cores, 48 tracks per camera, pinhole 758×572 input: non-keyframe critical-path median 900-1,000 µs, p95 at most 2,000 µs. Keyframe frame 950-1,100 µs; the frame that adopts landmarks 1,050-1,185 µs. Off-nominal: refill frame at most 1,180, retry at most 1,380, both at once at most 1,580 µs.   measured by: the bench's CSV on `nvidia-orin01`, `taskset` to 2 cores.
  Margin note (P4, P5): on pinhole input the x86 sum (about 800 µs) sits on the K1 proxy gate and the Orin 2-core estimate (about 999 µs) on K1, with a margin of about 0. K1 and K1 proxy are unchanged, so K1's existing cut list (40 tracks per camera first) is the expected first response.
  P6 Orin, 3 cores, 64 tracks per camera: median 850-1,000 µs, the same as P5 within 50 µs, at 16 more tracks per camera.   measured by: as P5, 3 cores.
  P7 Sync cost: `flow_dispatch + flow_join` median 10-30 µs on 2 cores and 30-70 µs on 3 cores.   measured by: those two columns.
  P8 Mechanism, flow frames: share of R_11 frames tracked from flow alone, from 0 % (rigs never defer, `build.rs:151-156`) to 70-85 %.   measured by: `flow_frames / frames` in `[summary]`.
  P9 Mechanism, secondary cameras: share of flow-frame pose inliers from cameras 1 and up, from 0 % (`tracker.rs:321`) to 35-55 % over R_11, and at least 30 % within frames 4850-6866.   measured by: `flow_secondary_inliers / flow_inliers`, overall and on that window (per-frame `[klt-first]` debug lines).
  P10 Mechanism, inlier ratio: flow-frame pose inliers over flow seeds, from 84 % to 90-97 %.   measured by: `flow_inliers / flow_seeds`.
  P11 Mechanism, round trip: the forward-backward check rejects 3-8 % of tracks per flow frame (median over frames).   measured by: `flow_fb_rejects / flow_tracks_tried`.
  P12 R_11 frames 4850-6866: 0 frames in Bootstrap (KLT-first rig, `3d84baf`: 2,016).   measured by: per-frame status in the trajectory log.
  P13 R_11 tracked share: within -1 to +3 points of `pr180/S` (mean of 3 samples).   measured by: `tracked / frames` in `[summary]`.
  P14 `flow_pose_fallbacks`: 0.5-3 % of flow frames.   measured by: the counter over `flow_frames`.
  P15 `flow_reconcile_clears` = 0 on R_11 and MH_01; `flow_template_rebuilds` at most 2 % of track-frames after warm-up.   measured by: the counters.
  P16 Keyframe latency under `keyframe_orb: Mapping`: a deferred keyframe is inserted 1-2 frames after it was due (median 1), and the deferred queue never holds more than 3.   measured by: `deferred_keyframe_lag_frames` (sum) and `deferred_keyframe_queue_max`.
  P17 With `flow_pose: AnalyticGn` and `flow_bookkeeping: KeyframesOnly`: `flow_map_wait` = 0 on every `FlowOnly::Tracked` frame (the flow path takes no map lock).   measured by: the `flow_map_wait` column of the bench CSV, on those frames.
  Mechanism counters: P8, P9, P10 and P11. If the theory is right, each must move as predicted before P13 counts.

Kill rules (K1-K8 from plan section 10, verbatim; the skill's two required kinds are marked):
  K1 (Orin, 2 cores) The critical-path median goes above 1,000 µs, or the p95 above 2,000 µs -> next cuts, in order: 40 tracks per camera, then the forward-backward check on alternate frames only (`every_n_frames: 2`).
  K1 proxy (laptop, `--flow-threads 1`) The median must be at most 800 µs, and every stage at most half its Orin row -> a pass here is provisional until the Orin bench.
  K2 Any stage costs more than 1.5x its row -> re-profile before tuning.
  K3 A keyframe frame goes above 2,000 µs on Orin with `keyframe_orb: Mapping` -> the hand-off copies or the queue are on the critical path; profile the hand-off before any other change.
  K4 (mechanism moved, score did not) R_11 rig tracked share falls more than 1 point below `pr180/S` -> response: 3 cores with 64 tracks. If P8 and P9 hold while K4 fires, the theory that the secondary cameras carry the hard segments is incomplete: read the per-frame inliers at the first lost frame before adding tracks.
  K5 (score moved, mechanism did not) The secondary cameras supply under 30 % of flow-frame inliers in frames 4850-6866 while tracked share still recovers -> the recovery has another cause (the keyframe timing or the IMU prediction), not the rig flow frame. Do not credit change 1; bisect with `klt_rig: PrimaryOnly` and the rest of the preset on.
  K6 `flow_pose_fallbacks` exceeds 5 % of frames -> GN from the prediction is outside its basin too often; read `imu_trust` on the fallback frames before raising `max_iter`.
  K7 Dispatch plus join costs more than 60 µs on 2 cores or 100 µs on 3, or 3 cores is no faster than 2 -> the park/wake path is too slow; measure the wake latency alone and try `pin_threads` before redesigning the pool.
  K8 Any allocation in `step` after warm-up, or survivors that differ across thread counts -> a bug, not a tuning question. No timing number is trusted until K8 passes.
Bit-identity:
- P1: every card-E knob off gives byte-identical output to `c0a1dc3`. That covers `flow_threads: 1`, `klt_rig: PrimaryOnly`, `klt_kernel: FixedPointU8`, `klt_window: 21`, today's `klt_prediction`, and every new enum at its first variant.
- P2: the output does not depend on `flow_threads` at a fixed track count.
- With `klt_kernel: WindowedU8` alone (everything else at default), the lane K oracle test requires bit-equal LK results to `calc_optical_flow_pyr_lk_u8` (`fixed_point.rs:1054`) for the same window, level and iterations. A whole run is still expected to differ, because the window-only Scharr changes the pyramid build path. Card E does not claim that run is identical.
- `stereo_pyramid: Shared` changes output by design (plan step 14), so it is outside P1 when on.
Noise:
- **Timing:** criterion medians over at least 200 frames per configuration; a run whose mean load differs from its baseline's by more than ±2 is not a timing sample. The laptop proxy is single-threaded only; laptop multi-core timings are not used for K1 or K7.
- **Tracked share and inliers:** 3 samples per configuration on R_11 (0 and ±50 µs IMU offset). With fewer, every K4/K5 verdict is provisional.
- **P2 and P3** are deterministic; one run each.

## Design (what lane T builds)

### Knobs

All of these are fields of `OrbFrontendConfig` (`system/config/mod.rs:124-141`, default at `:210-221`), RON path `system.pipeline.OrbSlam.frontend.<field>`. The table is plan section 8, restricted to the knobs card E covers.

| RON field | type | default (today) | Orin 2-core preset (3-core) | Plan step |
|---|---|---|---|---|
| `flow_threads` | `usize`, at least 1 | `1` | `2` (`3`) | T4 |
| `flow_first_core` | `Option<usize>`; used only with cargo feature `pin_threads` | `None` | set by the deployment | T4 |
| `klt_rig` | `KltRig { PrimaryOnly, AllCameras }` | `PrimaryOnly` | `AllCameras` | T1-T2 |
| `klt_kernel` | `LkKernel { F32, FixedPointU8, WindowedU8 }` (adds `WindowedU8`) | `FixedPointU8` | `WindowedU8` | T3 |
| `klt_window` | `usize`, odd | `21` | `11` | T3 |
| `klt_prediction` | existing `KltPrediction` | `Enabled { max_level: 1, max_iter: 10 }` | `Enabled { max_level: 0, max_iter: 8 }` | T3 |
| `klt_retry` | `KltRetry { Off, CoarseLevel { max_level: usize } }` | `Off` | `CoarseLevel { max_level: 1 }` | T4 |
| `forward_backward` | `ForwardBackward { Off, Enabled { max_round_trip_px: f32, max_iter: usize, every_n_frames: usize } }` | `Off` | `Enabled { max_round_trip_px: 0.6, max_iter: 4, every_n_frames: 1 }` | T6 |
| `track_refill` | `TrackRefill { Off, Enabled { per_camera: usize, min_spacing_px: f32, refill_below: f32, border_px: usize } }` | `Off` | `Enabled { per_camera: 48, min_spacing_px: 23.0, refill_below: 0.8, border_px: 9 }` (`per_camera: 64`) | T7 |
| `adopt_landmarks` | `bool` | `false` | `true` | T8 |
| `gyro_guess` | `GyroGuess { Off, RotatedRay }` | `Off` | `RotatedRay` | T9 |
| `anchor_guard` | `AnchorGuard { Off, Enabled { every_n_frames: usize, max_drift_px: f32 } }` | `Off` | `Off` | T13 |
| `flow_pose` | `FlowPose { RansacPnp, AnalyticGn { max_iter: usize } }` | `RansacPnp` | `AnalyticGn { max_iter: 3 }` | T10 |
| `flow_bookkeeping` | `FlowBookkeeping { EveryFrame, KeyframesOnly }` | `EveryFrame` | `KeyframesOnly` | T11 |
| `keyframe_orb` | `KeyframeOrb { Tracking, Mapping }` | `Tracking` | `Mapping` | T12 |
| `stereo_pyramid` | `StereoPyramid { Resized, Shared }` | `Resized` | not on the rig path | T14 |

- The LK error cap needs no knob: `SurvivorFilterConfig::max_error` is already 20 (`optical_flow.rs:256`).
- `klt_first` (`config/mod.rs:182-194`, default `Enabled { min_tracks: 80 }`) is unchanged. `klt_rig: AllCameras` only removes the rig filter at `build.rs:153-156`, so rig deferral still needs `klt_first` enabled.
- Not in card E: `fisheye_weight` and the LaMAria `images` source (card F), and `mapping.local_ba.online_calibration` (card G).
- **Presets.** Two RON files for the validation workflow, written by the docs phase: `configs/lamaria-rig-imu-orin2.ron` and `configs/lamaria-rig-imu-orin3.ron`. Each is `lamaria-rig-imu.ron` plus the preset column, with `mapping.execution: Asynchronous` (section 6 condition). Lane T does not edit `configs/`.
- **Validation** (`OrbFrontendConfig::validate`, `system/validation.rs:262-290`). The skeleton writes these, and each returns `ConfigError`:
  - `flow_threads == 0`, or above `MAX_FLOW_THREADS` (4; see open item 3);
  - `flow_threads > 1` with `klt_rig: PrimaryOnly`, or without a rig (see open item 3);
  - `klt_window` even or below 3;
  - `klt_window != 21` with `klt_kernel: F32` (`optical_flow_pyr_lk.rs:39-44`);
  - `klt_retry`, `forward_backward`, `track_refill`, `anchor_guard` or `flow_threads > 1` without `klt_kernel: WindowedU8`, because they need its template cache (see open item 4);
  - `forward_backward.max_round_trip_px` not finite or not above 0, `max_iter == 0` or `every_n_frames == 0`;
  - `track_refill.per_camera == 0`, `min_spacing_px` not finite or not above 0, `refill_below` outside `(0, 1]`;
  - `anchor_guard.every_n_frames == 0`, `max_drift_px` not finite or not above 0;
  - `flow_pose: AnalyticGn { max_iter: 0 }`;
  - `klt_retry: CoarseLevel { max_level: 0 }`;
  - `keyframe_orb: Mapping` with `klt_first: Disabled` (no flow frame would ever defer);
  - `flow_first_core: Some(_)` in a build without `pin_threads`: a warning on stderr, not an error, so one RON file serves both builds.
- **CHANGELOG.** One line per knob, written by the skeleton.

### Threading (plan section 3)

- **Per frame, on the tracking core:**
  1. IMU prediction, trust and gate (as today, `system/mod.rs:1496-1515`).
  2. The level-0 padded copy of each camera into its spare `LkFrame` (`LkFrame::rebuild`), keyed by the system frame index. This replaces `retain_previous_image` (`:711-719`) on this path; the previous frame's `LkFrame` is kept as `previous`.
  3. `ShardGuess` filled in place, then `FlowPool::step`.
  4. After the join: the pose (`flow_pose`), the track advance, the counts and the keyframe decision.
  5. `klt_retry` as a second `step` over the failed tracks only, after the coarse levels are built (`LkFrame::build_coarse`).
- **Shards.** `shard` is fixed when a track spawns and stored on `Track`:
  - `flow_threads == 1`: shard 0 for every track, run inline.
  - `flow_threads == cameras`: shard = camera. Shard 0 runs on the tracking core; shard `c` on worker `klt-flow-<c>`.
  - `flow_threads == cameras + 1`: per camera `c`, a spawn counter `k` puts the track on shard 0 (tracking core) when `k % 3 == 0`, otherwise on shard `c + 1`. Worker `c + 1` gets two thirds of camera `c`; the tracking core gets the remaining third of each.
  - Any other combination is a validation error (open item 3).
- **Output order.** Survivors merge in `(camera, track id)` order, so the output does not depend on the thread count (P2).
- **Pool.** `FlowPool::new` spawns `threads - 1` workers with `thread::Builder::new().name(format!("klt-flow-{n}"))`, as `local_mapping.rs:336`. Each worker owns two `sync_channel(1)`s: one carries a `FlowJob` in, the other returns it. A job moves the `FlowShard`, the frames' `Arc<LkFrame>` clones in reused `Vec`s, its own reused `ShardGuess` and its own `ShardTimings`, and comes back with them. After warm-up nothing allocates (P3).
- **Join.** A `try_recv` spin of at most 100 µs, then a blocking `recv`.
- **Pinning.** Behind feature `pin_threads` (`core_affinity`): worker `n` goes to core `flow_first_core + n`. Default: no pinning; the deployment confines the process with `taskset`.
- **Coarse levels and `Arc`.** A worker drops its `Arc<LkFrame>` clones before it returns the job, so the tracking core holds the only reference at join time, and `Arc::get_mut` succeeds for `build_coarse` and the next `rebuild`. `FlowPool::step` returns `FlowPoolError::FrameShared` if that fails, and never clones a frame.

### Per-step design (lane T order, plan section 9)

1. **`FlowShard` plus the inline path; rig filter.** With `klt_rig: AllCameras`, the skeleton passes `klt_min_tracks` through for rigs (`build.rs:153-156`). Lane T moves the per-track loop of `KltTracker::track_cached_u8` (`optical_flow.rs:491-529`) into `FlowShard::run` for `WindowedU8`. `F32` and `FixedPointU8` keep today's code path unchanged (P1).
2. **Rig flow frame.** `flow_frame` (`tracker.rs:270-324`) groups survivors by `Track::camera` and fills `RigFeatures { geometry: Some(rig), camera_ranges, left_to_right: vec![], stereo_points: vec![] }`. `reconcile` (`:400-417`) runs per camera range. `estimate` (`:434-435`) snaps each camera's survivors against that camera's ORB keypoints, not only the primary's.
3. **`WindowedU8`.** The template of each track is cached in its shard with the frame index it came from. The forward solve starts at level 0 from the predicted pixel. The clone and compare (`optical_flow.rs:527`, `:694-713`) and the previous-image copy (`system/mod.rs:711-719`) are not used on this path.
4. **`FlowPool`.** As in "Threading" above. `klt_retry` is the second phase.
5. **`tests/flow_zero_alloc.rs`.** A `#[global_allocator]` counting wrapper, 10 warm-up frames of synthetic shifted squares on 2 cameras, then 50 counted `step`s for 1, 2 and 3 threads. The second test compares the survivors of 1, 2 and 3 threads for `==`.
6. **Forward-backward.** The current-frame template at the forward result is extracted (it is the next frame's forward template) and solved back into `previous` from the track's previous pixel, with at most `max_iter` iterations. The track is dropped when the round trip exceeds `max_round_trip_px`. With `every_n_frames > 1` the check runs on frames where `frame_idx % every_n_frames == 0`.
7. **Refill inside the shard.** One camera per frame, round robin (`frame_idx % cameras`). When that camera held fewer than `refill_below × per_camera` tracks at the start of the frame, `good_features_to_track` runs on its half-resolution empty cells, then `corner_sub_pix`.
   - The occupied pixels are the start-of-frame pixels of every track of that camera, across shards, passed in `ShardGuess::refill_occupied`. So the spawns do not depend on how the camera's tracks are split (P2).
   - It runs on the camera's worker shard: shard `c` when `flow_threads == cameras`, shard `c + 1` when `cameras + 1`, and shard 0 inline.
   - New tracks get no map point. They join the map at the next keyframe, through `adopt_landmarks` or through ORB reconciliation.
8. **`adopt_landmarks`.** In `apply_local_mapping_results` (`system/mod.rs:863`, called at `:799`), each newly created map point observed by the latest keyframe spawns a track at its keypoint. The spawn uses the keyframe's stored plane, so its template is cut from that plane and tracked forward to the current frame. With `keyframe_orb: Mapping`, the plane is the one in the `DeferredKeyframe`.
   - **Position refresh.** `Track::position` is a copy, so every map change makes it stale, and a stale start at level 0 with an 11×11 window falls outside its few-pixel capture range without any warning (it shows up only as retries, fallbacks and losses). Each time `apply_local_mapping_results` runs, under the map lock that path already holds, `TrackSet::refresh_positions` rewrites `position` for every associated track from `map_points()[idx].position`, and sets it to `None` for a culled point. `apply_loop_closing_results` (`system/mod.rs:2614`) does the same under its lock at `:2624`, after a loop correction or a map merge. Each rewritten position counts in `flow_position_refreshes`.
   - Unit test: move a map point, apply a local-mapping result, and check that the track's `position` equals the new value; cull the point and check that it is `None`.
9. **`gyro_guess: RotatedRay`.** For a track without a map point, or without a trusted prediction: unproject the previous pixel, rotate it by `R_cur_prev(c) = R_cb · ΔR_gyroᵀ · R_bc` for camera `c` through its extrinsic, then project. Tracks with a map point keep the map-point prediction (`tracker.rs:244-268`) when the prediction is trusted.
10. **`flow_pose: AnalyticGn`.** From the IMU-predicted pose, on the round-trip survivors:
    - rig: `solve_rig_pnp` (`rig_pnp.rs:59`) with `lm_max_iterations = max_iter` and `outlier_rounds = 1`, from the prediction (see open item 1);
    - mono: `refine_pose_analytic` (`upstream/kornia_3d/refine_analytic.rs:179`).
    
    On the `WindowedU8` + `AnalyticGn` path the `RigObservation`s come from `Track::position` and `Track::camera` (kept fresh by step 8), not from the map. The `Frame` carries map-point indices only; descriptors are read from the map only when an ORB or keyframe path needs them. So on `FlowOnly::Tracked` frames with `KeyframesOnly` the flow path takes no map lock, and `flow_map_wait` (time blocked on the map `Mutex` on the flow path) must be 0 (P17).

    Below `min_inliers` it falls back to today's `estimate_from_seeds` and counts `flow_pose_fallbacks`.
11. **`flow_bookkeeping: KeyframesOnly`.** On `FlowOnly::Tracked` frames, skip `system/mod.rs:1676-1705`. Frames that run ORB keep it.
12. **`keyframe_orb: Mapping`.** At `FlowOnly::KeyframeDue` (`system/mod.rs:1894-1899`) the frame is accepted with its flow pose, and a `DeferredKeyframe` is queued in place of calling `resolve` (`:1574`). The `DeferredKeyframe` holds the level-0 planes from a reused pool, the pose, the tracks, the inliers and the matches. The mapping worker then runs, in FIFO order:
    1. ORB on every camera with `detect_and_extract_u8_into` and an `OrbScratch` (`$K/kornia-imgproc/src/features/orb/extractor.rs:67`, `:1092`);
    2. the overlap match;
    3. the projection search seeded by the flow pose;
    4. the snap of tracks to ORB keypoints (`snap_unique`, `optical_flow.rs:718`);
    5. the visibility counts and the insert;
    6. then today's BA job.
    
    The result returns through `apply_local_mapping_results`. With `mapping.execution: Synchronous` the same job runs inline (no budget claim).
13. **`anchor_guard`.** Each track keeps its first-sighting template. Every `every_n_frames` frames it is tracked into the current frame from the current pixel, and the track is dropped if the two differ by more than `max_drift_px`.
14. **`stereo_pyramid: Shared`.** `frontend/mod.rs:134-135` (the two `build_pyramid` calls of `:271-287`) is replaced by `detect_and_extract_u8_from_pyramid` (`extractor.rs:1017`) on the detector's own pyramid. This changes output.
15. **Frontend bench.** `apps/kornia-slam-frontend-bench` gains mode `klt-rig` (`pipeline.rs:35-50`) and flag `--flow-threads`. It writes a per-stage µs CSV whose columns are the `TrackingTimings` flow fields plus the per-shard fields, and a `critical_path_us` column = the tracking-core stages + `flow_dispatch` + the slowest shard + `flow_join`. Both Orin presets are read from the preset RON files.

### Counters (in `RunCounters`, `system/counters.rs:22`; printed in `[summary]`)

| counter | meaning | used by |
|---|---|---|
| `flow_frames` | frames tracked from flow alone | P8 |
| `flow_seeds` | flow survivors with a map point, summed over flow frames | P10 |
| `flow_inliers` | pose inliers on flow frames | P9, P10 |
| `flow_secondary_inliers` | of those, from cameras 1 and up | P9, K5 |
| `flow_tracks_tried` | tracks entering the forward solve | P11 |
| `flow_fb_rejects` | tracks dropped by the round trip | P11 |
| `flow_retries` | tracks re-run at the coarse level | P5 off-nominal |
| `flow_refill_spawns` | tracks spawned by refill | P5 off-nominal |
| `flow_anchor_drops` | tracks dropped by the anchor guard | T13 |
| `flow_pose_fallbacks` | AnalyticGn frames that fell back to RANSAC PnP | P14, K6 |
| `flow_template_rebuilds` | templates re-extracted because the cached one had another frame index | P15 |
| `flow_reconcile_clears` | silent track-set clears in `reconcile` (`tracker.rs:409-415`) | P15 |
| `flow_position_refreshes` | track positions rewritten after a local-mapping result or a loop or map correction (step 8) | step 8 |
| `deferred_keyframe_lag_frames` | frames between a deferred keyframe being due and its insert, summed | P16 |
| `deferred_keyframe_queue_max` | largest deferred-queue length seen | P16 |

Per-shard track counts go in the debug line `[flow] frame= shard= tracks=` (plan risk "shard imbalance").

## Skeleton contract

The skeleton declares every item below, with `todo!()` bodies reachable only through a card-E knob that is not at its default. Lane T fills the bodies.

### `system/config/mod.rs` (skeleton)

```rust
pub const MAX_FLOW_THREADS: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRig { #[default] PrimaryOnly, AllCameras }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KltRetry { #[default] Off, CoarseLevel { max_level: usize } }

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForwardBackward {
    #[default]
    Off,
    Enabled { max_round_trip_px: f32, max_iter: usize, every_n_frames: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrackRefill {
    #[default]
    Off,
    Enabled { per_camera: usize, min_spacing_px: f32, refill_below: f32, border_px: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GyroGuess { #[default] Off, RotatedRay }

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnchorGuard { #[default] Off, Enabled { every_n_frames: usize, max_drift_px: f32 } }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowPose { #[default] RansacPnp, AnalyticGn { max_iter: usize } }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowBookkeeping { #[default] EveryFrame, KeyframesOnly }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframeOrb { #[default] Tracking, Mapping }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StereoPyramid { #[default] Resized, Shared }

// OrbFrontendConfig gains, in this order after `klt_kernel`:
//   flow_threads: usize, flow_first_core: Option<usize>, klt_rig: KltRig,
//   klt_window: usize, klt_retry: KltRetry, forward_backward: ForwardBackward,
//   track_refill: TrackRefill, adopt_landmarks: bool, gyro_guess: GyroGuess,
//   anchor_guard: AnchorGuard, flow_pose: FlowPose, flow_bookkeeping: FlowBookkeeping,
//   keyframe_orb: KeyframeOrb, stereo_pyramid: StereoPyramid
// Defaults: 1, None, PrimaryOnly, 21, then every enum's #[default], false.
```

- `OrbFrontendConfig` is `Copy` today (`config/mod.rs:121`), and every type above is `Copy`, so it stays `Copy`.
- The `Display` summary (`config/mod.rs:532-548`) gains one line per knob that is not at its default.
- `config/tests.rs` gets a RON round trip of both presets and one rejection test per validation rule.

### `system/build.rs` (skeleton)

- `SystemSettings` gains a `frontend_flow: FlowSettings` field, filled from the knobs above. `klt_min_tracks` (`:153-156`) keeps the `!rig.is_multi_camera()` filter only for `klt_rig: PrimaryOnly`.
- `BuildError` (`:24`) gains `FlowPool(#[from] FlowPoolError)`. `SlamSystem::build` creates the `FlowPool` and passes it to `assemble`; with `flow_threads: 1` that spawns nothing and cannot fail.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlowSettings {
    pub threads: usize,
    pub first_core: Option<usize>,
    pub rig: KltRig,
    pub window: usize,
    pub retry: KltRetry,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    pub adopt_landmarks: bool,
    pub gyro_guess: GyroGuess,
    pub anchor_guard: AnchorGuard,
    pub pose: FlowPose,
    pub bookkeeping: FlowBookkeeping,
    pub keyframe_orb: KeyframeOrb,
}
```

`stereo_pyramid` goes to `OrbFrontend` (`frontend/mod.rs`) through a `with_stereo_pyramid(StereoPyramid)` builder, as `with_secondary_keypoints` does (`system/mod.rs:229`).

### `tracking/optical_flow.rs` (declarations by the skeleton; bodies by lane T)

```rust
pub enum LkKernel {
    #[default]
    F32,
    FixedPointU8,
    /// Window-only fixed-point LK on `wide`, from the predicted start.
    WindowedU8,
}

pub struct Track {
    id: TrackId,
    pixel: [f32; 2],
    map_point_idx: Option<usize>,
    age: u32,
    /// The associated map point's world position, copied when the track is
    /// associated and refreshed after every map change, so shards predict
    /// without the map lock; `None` once the point is culled.
    position: Option<Vec3F64>,
    /// Rig camera the track lives in; 0 outside a rig.
    camera: u8,
    /// Flow shard fixed at spawn (see `FlowPool`).
    shard: u8,
}
impl Track {
    pub fn position(&self) -> Option<Vec3F64>;
    pub fn camera(&self) -> usize;
    pub fn shard(&self) -> usize;
}
impl TrackSet {
    /// As `spawn`, in `camera`, on `shard`.
    pub fn spawn_in(&mut self, pixel: [f32; 2], camera: usize, shard: usize) -> TrackId;
    /// Rewrites every associated track's `position` from `map`, `None` for a
    /// culled point; returns how many it rewrote (`flow_position_refreshes`).
    pub fn refresh_positions(&mut self, map: &Map) -> u64;
}
```

`Track` stays `Copy`. `spawn` keeps its signature and means camera 0, shard 0, so today's callers and tests are unchanged.

### `tracking/timings.rs` (skeleton)

```rust
pub const MAX_FLOW_SHARDS: usize = MAX_FLOW_THREADS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShardTimings {
    pub flow_template: Duration,
    pub flow_forward: Duration,
    pub flow_backward: Duration,
    pub flow_project: Duration,
    pub flow_refill: Duration,
}

// TrackingTimings gains:
//   flow_prepare, flow_dispatch, flow_parallel, flow_join, flow_pose, flow_bookkeeping,
//   flow_map_wait: Duration,
//   shards: [ShardTimings; MAX_FLOW_SHARDS],
// flow_map_wait is the time blocked on the map Mutex on the flow path; the other
// fields exclude it.
// total() adds the seven tracking-core fields only (the shards run inside flow_parallel).
impl TrackingTimings { pub fn critical_path(&self) -> Duration; }
```

`TrackingTimings` stays `Copy + Eq`. With every knob off the new fields stay zero, so `total()` is unchanged (P1).

### `system/counters.rs` (skeleton)

`RunCounters` (`#[non_exhaustive]`, `:20-22`) gains the 15 `u64` fields of the counter table. Lane T increments them.

### `tracking/mod.rs` (skeleton)

`pub(crate) mod flow_tracker;` and `pub(crate) mod flow_pool;` (open item 5: `tracking/mod.rs` is in no lane's list).

### `tracking/flow_tracker.rs` (new; skeleton declares, lane T fills)

```rust
/// One camera's model for a flow step, refreshed in place each frame.
#[derive(Debug, Clone)]
pub(crate) struct FlowCamera {
    pub camera: PinholeCamera,
    /// `T_CP`; `None` for the primary camera.
    pub camera_from_primary: Option<Pose3d>,
    /// `R_cur_prev` of this camera from the gyro (`GyroGuess::RotatedRay`).
    pub gyro_rotation: Option<Mat3F64>,
}

/// What every shard needs to start its tracks; filled by the tracking core.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShardGuess {
    pub frame_idx: u64,
    /// World to primary camera at the IMU prediction, when trusted.
    pub pose_world_to_primary: Option<Pose3d>,
    pub cameras: Vec<FlowCamera>,
    pub max_level: usize,
    pub max_iter: usize,
    pub epsilon: f32,
    pub window: usize,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    /// Camera refilled this frame, if any.
    pub refill_camera: Option<usize>,
    /// Start-of-frame pixels of every track of `refill_camera`, all shards.
    pub refill_occupied: Vec<[f32; 2]>,
    pub anchor_guard: AnchorGuard,
    /// Retry pass: only `retry` tracks, from `max_level`.
    pub retry: bool,
}

/// One shard's tracks and every buffer it reuses; moved to a worker and back.
#[derive(Default)]
pub(crate) struct FlowShard {
    pub index: usize,
    /// Input, in (camera, track id) order.
    pub tracks: Vec<Track>,
    /// Output, in the order of `tracks`.
    pub survivors: Vec<FlowSurvivor>,
    /// Tracks that failed at level 0, for `klt_retry`.
    pub retry: Vec<TrackId>,
    pub spawned: Vec<(usize, [f32; 2])>,
    pub fb_rejects: u32,
    pub anchor_drops: u32,
    pub template_rebuilds: u32,
    templates: Vec<(TrackId, u64, LkTemplate)>,
    anchors: Vec<(TrackId, LkTemplate)>,
    starts: Vec<[f32; 2]>,
    features: GoodFeaturesScratch,
    occupied: Vec<[f32; 2]>,
}

impl FlowShard {
    /// Tracks `self.tracks` from `previous[c]` into `current[c]`.
    pub(crate) fn run(
        &mut self,
        previous: &[Arc<LkFrame>],
        current: &[Arc<LkFrame>],
        guess: &ShardGuess,
        timings: &mut ShardTimings,
    );
}
```

- `FlowShard` holds no guess and no timings, so `run` borrows both from its caller and never aliases `&mut self`.
- The guess and timings a worker uses live on that worker's `FlowJob` (see `flow_pool.rs`).

### `tracking/flow_pool.rs` (new; skeleton declares, lane T fills)

```rust
/// The frames of one flow step, one entry per rig camera.
pub(crate) struct FlowFrames {
    pub previous: Vec<Arc<LkFrame>>,
    pub current: Vec<Arc<LkFrame>>,
}

#[derive(Debug, thiserror::Error)]
pub enum FlowPoolError {
    #[error("flow worker spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("flow worker {0} is gone")]
    WorkerGone(usize),
    #[error("core {0} could not be pinned")]
    Pin(usize),
    #[error("a flow frame is still shared after the join")]
    FrameShared,
}

/// One worker's reused job; moved to the worker and back every step.
struct FlowJob {
    shard: FlowShard,
    previous: Vec<Arc<LkFrame>>,
    current: Vec<Arc<LkFrame>>,
    guess: ShardGuess,
    timings: ShardTimings,
}

pub(crate) struct FlowPool { /* workers, channels, one FlowJob per worker */ }

impl FlowPool {
    /// `threads - 1` named workers `klt-flow-<n>`; `threads == 1` spawns nothing.
    pub(crate) fn new(threads: usize, first_core: Option<usize>) -> Result<Self, FlowPoolError>;
    /// Runs `shards[0]` inline and `shards[n]` on worker `n`, then joins.
    pub(crate) fn step(
        &mut self,
        shards: &mut [FlowShard],
        frames: &FlowFrames,
        guess: &ShardGuess,
    ) -> Result<(), FlowPoolError>;
    pub(crate) fn threads(&self) -> usize;
}

impl Drop for FlowPool { /* closes the channels and joins the workers */ }
```

`FlowPoolError` is `pub`, because `BuildError::FlowPool` wraps it.

- Before it sends a job, the pool swaps `shards[n]` into `job.shard`, fills the job's frame `Vec`s and `clone_from`s `guess` into `job.guess`, which does not allocate once the capacity is reached.
- The worker calls `job.shard.run(&job.previous, &job.current, &job.guess, &mut job.timings)`.
- Shard 0 runs inline with the caller's `guess` and a timings value owned by the pool.
- After the join, the pool swaps each shard back and copies `job.timings` into `TrackingTimings::shards[index]`.

### `mapping/keyframe_mapping.rs` and `mapping/local_mapping.rs` (`keyframe_orb: Mapping`)

```rust
// mapping/keyframe_mapping.rs (skeleton declares the three structs; lane T fills only the functions)
pub(crate) struct DeferredKeyframe {
    pub frame_idx: usize,
    pub timestamp_sec: f64,
    pub pose_world_to_cam: Pose3d,
    /// One level-0 plane per rig camera, from a reused pool.
    pub planes: Vec<Image<u8, 1>>,
    pub tracks: Vec<Track>,
    pub tracked_inliers: usize,
    pub matches: Vec<(usize, usize)>,
}

pub(crate) struct DeferredKeyframeResult {
    pub frame_idx: usize,
    pub keyframe_idx: Option<usize>,
    /// Map points created at this keyframe, for `adopt_landmarks`.
    pub new_map_points: Vec<usize>,
    /// The planes, returned to the pool.
    pub planes: Vec<Image<u8, 1>>,
}

/// What the mapping worker needs to finish a deferred keyframe.
pub(crate) struct DeferredKeyframeContext {
    /// The primary camera's detector (`OrbFrontend::detector`, `frontend/mod.rs:44`).
    pub detector: OrbDetector,
    /// The other rig cameras' detector (`OrbFrontend::secondary`, `:46`).
    pub secondary: OrbDetector,
    /// The overlap match reads only the rig (`match_overlap`, `frontend/overlap.rs:52-56`).
    pub rig: Arc<RigGeometry>,
}

// mapping/local_mapping.rs (skeleton; one commit shared with card G)
impl LocalMapping {
    /// Enables the deferred path; the build calls it only with `keyframe_orb: Mapping`.
    pub(crate) fn with_deferred(self, context: DeferredKeyframeContext) -> Self;
    /// Queues `job` (FIFO, never replaced, unlike `submit`'s latest-wins job).
    pub(crate) fn submit_deferred(&self, job: DeferredKeyframe) -> bool;
    pub(crate) fn drain_deferred(&self) -> Vec<DeferredKeyframeResult>;
}
```

- **`keyframe_mapping.rs`.** The skeleton declares `DeferredKeyframe`, `DeferredKeyframeResult` and `DeferredKeyframeContext` with the fields above, so `submit_deferred` compiles at `pr180b/K`. Lane T writes only the worker-side functions (open item 2).
- **`LocalMapping::new` keeps its signature** (`local_mapping.rs:206`; callers `system/mod.rs:243` and the 4 tests at `:513`, `:526`, `:594`, `:617`). `LocalMapping` and `new` are `pub`, and a `pub(crate)` context type in a `pub` signature would break the public API and trip `private_interfaces` under `clippy -D warnings`. `system/build.rs` calls `with_deferred` only when `keyframe_orb` is `Mapping`.
- **Where the context lives.** Synchronous: the backend gains `deferred: Option<DeferredKeyframeContext>`. Asynchronous: the worker is already running when `with_deferred` is called, so the context goes into the shared `PendingState` as `deferred_context: Option<Arc<DeferredKeyframeContext>>`, before any job is submitted.
- **One skeleton commit for `local_mapping.rs`, covering cards E and G:**
  - card G's calibration slot and signatures: `solve_snapshot` (`:75`) and `solve_counted` (`:164`) take `calibration: &OnlineCalibrationState`, `Synchronous` holds `calibration: Mutex<OnlineCalibrationState>`, and the async worker (`:326-400`) owns a local one (card G, Skeleton contract);
  - `PendingState` (`:291-294`) gains `deferred: VecDeque<DeferredKeyframe>` next to `latest`; `WorkerExit` and `LocalMappingHandle::drop` clear it with `latest`;
  - the worker's wait (`:348`) also wakes on a non-empty `deferred`. A branch takes the front deferred job before `latest`, with a `todo!()` body, reachable only when the context is `Some`. `submit_deferred` returns `false` without a context.
- **The deferred path's BA job** (step 12.6) calls `solve_counted(…, &calibration)` with the same `OnlineCalibrationState` as `submit`'s path, and stores the returned state only when `merge_and_cull` returns `Ok`, as card G's C6 says. So `keyframe_orb: Mapping` never skips online calibration.
- Lane T fills the deferred bodies in `local_mapping.rs` (the branch, `with_deferred`, `submit_deferred`, `drain_deferred`). Lane C does not touch `local_mapping.rs`, so the hunks do not conflict.

### From lane K (`upstream/kornia_imgproc/`, plan section 8)

```rust
// lk_window.rs
pub struct LkFrame { /* padded level-0 plane, coarse levels on request, frame id */ }
impl LkFrame {
    pub fn new(image: &Image<u8, 1>, win: usize, frame_id: u64) -> Result<Self, LkWindowError>;
    pub fn rebuild(&mut self, image: &Image<u8, 1>, frame_id: u64) -> Result<(), LkWindowError>;
    pub fn build_coarse(&mut self, max_level: usize) -> Result<(), LkWindowError>;
    pub fn frame_id(&self) -> u64;
    pub fn levels(&self) -> usize;
}
pub struct LkTemplate { /* win×win i16 values and Scharr gradients, Q14 */ }
pub struct LkTrack { pub pixel: [f32; 2], pub status: bool, pub error: f32, pub iterations: u32 }
pub fn extract_template(frame: &LkFrame, pixel: [f32; 2], level: usize, win: usize, out: &mut LkTemplate) -> bool;
pub fn track_template(template: &LkTemplate, next: &LkFrame, start: [f32; 2], max_iter: usize, epsilon: f32) -> LkTrack;

// good_features.rs
pub struct GoodFeaturesParams { pub max_corners: usize, pub quality: f32, pub min_spacing_px: f32, pub border_px: usize, pub half_resolution: bool }
#[derive(Default)] pub struct GoodFeaturesScratch { /* response, cell occupancy, candidates */ }
pub fn good_features_to_track(image: &Image<u8, 1>, occupied: &[[f32; 2]], params: &GoodFeaturesParams, scratch: &mut GoodFeaturesScratch, out: &mut Vec<[f32; 2]>);

// corner_subpix.rs
pub fn corner_sub_pix(image: &Image<u8, 1>, corners: &mut [[f32; 2]], half_win: usize, max_iter: usize, epsilon: f32);
```

- The plan fixes the function signatures. Card E adds `LkFrame`'s methods, the `LkTrack` field types, `GoodFeaturesParams`' fields and `LkWindowError`, because lane T calls them. Lane K may rename private fields, but not these.
- `good_features_to_track` reads the level-0 plane through `LkFrame::plane(&self) -> &Image<u8, 1>`, which `LkFrame` also exposes.
- `track_template` with `level > 0` tracks on `next`'s coarse level and scales the result to level 0; lane K documents it.

### Reused unchanged

- `solve_rig_pnp`, `RigObservation`, `PnpConfig` (`tracking/pose_estimation/rig_pnp.rs`, `pnp.rs`);
- `refine_pose_analytic` (`upstream/kornia_3d/refine_analytic.rs:179`);
- `snap_unique` (`optical_flow.rs:718`);
- `detect_and_extract_u8_into`, `OrbScratch`, `detect_and_extract_u8_from_pyramid` (kornia-rs `4ada5a7`).

## Files lane T touches

- **New:**
  - `tracking/flow_tracker.rs` and `tracking/flow_pool.rs` (bodies);
  - `crates/kornia-slam/tests/flow_zero_alloc.rs`.
- **Edited (bodies):**
  - `tracking/optical_flow.rs`: the `WindowedU8` path, `Track` accessors, `spawn_in`;
  - `tracking/tracker.rs`: `flow`, `predicted_pixels`, `flow_frame`, `reconcile`, `estimate` snap per camera;
  - `system/mod.rs`: the pool field, the `LkFrame` pair per camera, the flow path in `tracking_step`, `track_flow_only`, the bookkeeping skip, the deferred keyframe, `adopt_landmarks` in `apply_local_mapping_results`;
  - `frontend/mod.rs`: `stereo_pyramid: Shared`;
  - `mapping/keyframe_mapping.rs`: the worker-side keyframe steps (the skeleton declares the three deferred structs);
  - `mapping/local_mapping.rs`: the deferred bodies only (the worker branch, `with_deferred`, `submit_deferred`, `drain_deferred`);
  - `apps/kornia-slam-frontend-bench/src/{main.rs,pipeline.rs}`: mode `klt-rig`, `--flow-threads`, the CSV.
- **Skeleton-owned** (lane T fills bodies only): `system/config/mod.rs`, `config/tests.rs`, `system/build.rs`, `system/validation.rs`, `tracking/timings.rs`, `system/counters.rs`, `tracking/mod.rs`, `crates/kornia-slam/Cargo.toml` (`wide`, `criterion`, features `bench_internals`, `opencv_bench`, `pin_threads` with optional `core_affinity`), `CHANGELOG.md`, the three deferred structs in `mapping/keyframe_mapping.rs`, and in `mapping/local_mapping.rs` the `with_deferred` / `submit_deferred` / `drain_deferred` declarations, `PendingState::deferred` and the worker branch (one commit with card G's calibration slot).
- **Not touched:** `configs/*.ron` (docs phase), `upstream/*` (lane K), `tracking/pose_estimation/map_projection/mod.rs` (lane F owns its search radius; lane T only calls `estimate_from_seeds` and `solve_rig_pnp`), `rig_pnp.rs`, `frame.rs`, `crates/kornia-sensors/*`, and the mapping BA files.
- **Commits** (one concern each; K8 can block the pool without blocking the rig flow frame):
  1. `feat(tracking): KLT on every rig camera and a rig flow frame (klt_rig)`;
  2. `feat(tracking): window-only fixed-point LK with cached templates (WindowedU8)`;
  3. `feat(tracking): flow worker pool with per-camera shards (flow_threads)`;
  4. `feat(tracking): forward-backward check, refill, gyro guess and anchor guard`;
  5. `feat(tracking): analytic flow pose and keyframe-only bookkeeping`;
  6. `feat(mapping): keyframe ORB on the mapping thread and landmark adoption`;
  7. `feat(frontend): shared stereo pyramid`;
  8. `feat(bench): klt-rig mode and per-stage CSV`.

## Open for the card review (choices this card made that the plan does not fix)

1. **`AnalyticGn` on a rig uses `solve_rig_pnp`, not `refine_pose_analytic`.** Plan step 10 names `refine_pose_analytic`. That is a single-camera solve: one `K`, f32 inputs (`refine_analytic.rs:179-187`), and two `Vec` collects per call (`:191-195`). It cannot take the secondary camera's rays. `solve_rig_pnp` already is a Huber LM over every camera from a start pose with no RANSAC (`rig_pnp.rs:1-7`, `:59-104`), so card E calls it from the IMU prediction on rigs and keeps `refine_pose_analytic` for mono.
2. **The deferred keyframe queue is FIFO and lives on the mapping worker.** Today's worker keeps only the latest BA job (`PendingState::latest`, `local_mapping.rs:291-294`). A latest-wins slot would silently drop a deferred keyframe. ORB-SLAM3 queues them (`LocalMapping.cc:284-287`), and the plan's own word is "enqueue". Resolved in the card review: the skeleton declares the queue, the structs and the methods, in one `local_mapping.rs` commit with card G's calibration slot; lane T fills the deferred bodies, and lane C does not touch the file (Skeleton contract).
3. **Thread counts other than 1, `cameras` and `cameras + 1` are rejected, with `MAX_FLOW_THREADS = 4`.** Plan section 3 defines shards for 2 and 3 cores on a 2-camera rig only. Section 8 says "at least 1" with no upper bound, but the fixed per-shard timing array needs one. Mono with `flow_threads > 1` is rejected for the same reason.
4. **`klt_retry`, `forward_backward`, `track_refill`, `anchor_guard` and `flow_threads > 1` require `klt_kernel: WindowedU8`.** They are built on its template cache, as plan steps 3 and 6 describe. Porting them to `F32` / `FixedPointU8` would duplicate the work on a path the presets do not use.
5. **Files no lane owns.** `tracking/mod.rs` (the two `mod` lines) and `system/counters.rs` (14 counters) are not in plan section 7. Card E assigns their declarations to the skeleton.
6. **`kornia_algebra::simd` exists, but only for f32.** Plan section 1 says that kornia-algebra has no SIMD layer. At `4ada5a7` it has `F32x4` / `F32x8` on glam, with NEON on aarch64 (`$K/kornia-algebra/src/simd.rs:1-13`), but no i16/i32 lanes. The Q14 fixed-point LK needs integer lanes, so lane K still uses `wide` there. The f32 good-features response may use `kornia_algebra::simd` instead; that is lane K's call.
7. **`gyro_guess` applies only to tracks without a trusted map-point prediction.** SLAM Dunk rotates every ray. Card E keeps the map-point prediction where it exists, because it also carries translation.

## Not in this card

- Raw fisheye input and magnification weights (card F; lane F is deferred to its own round, owner 2026-11-17, so this round runs on the 758×572 pinhole export); online calibration (card G).
- Multi-camera VIO beyond the rig PnP (deferred, owner 2026-11-08).
- The kernel parity and OpenCV benches themselves: lane K, plan step K5. Card E only consumes the kernels.
- Changing `LkKernel`'s own `#[default]` (`optical_flow.rs:288-289`, `F32`): it stays, and only the handoff line is corrected (plan section 14).

## Sources

- Plan: `/home/edgar/.claude/plans/pr180-frontend-1ms.md`, sections 2-6, 8, 10 (cost model, threading, budget, ordered changes, cuts, knobs, kill rules).
- Our evidence:
  - `$S/slam-viba/frontend-speedup.md:536`, `:579-581`, `:708-712`;
  - `$S/slam-viba/audit/klt-first-diagnosis.md:26-37`;
  - `experiments/lamaria-handoff.md:155-187`.
- VINS-Fusion: `$S/bench/src/VINS-Fusion/vins_estimator/src/featureTracker/feature_tracker.cpp:55-81`, `:118-146`, `:184`.
- Basalt:
  - `$S/bench/src/basalt/include/basalt/optical_flow/frame_to_frame_optical_flow.h:126`, `:147`, `:214-236`, `:251`;
  - `$S/bench/src/basalt/src/utils/vio_config.cpp:50-51`.
- ORB-SLAM3:
  - `$S/bench/src/ORB_SLAM3/src/Optimizer.cc:814`, `:1001-1012`;
  - `Tracking.cc:3216`, `:3335`, `:3357`, `:3379`;
  - `LocalMapping.cc:83`, `:92`, `:101`, `:284-287`.
- kornia-rs `4ada5a7`:
  - `kornia-imgproc/src/optical_flow_pyr_lk.rs:39-58`, `:1472`, `:1505`;
  - `optical_flow_pyr_lk/fixed_point.rs:33-36`, `:291-292`, `:1054`;
  - `features/orb/extractor.rs:67`, `:1017`, `:1092`;
  - `kornia-algebra/src/simd.rs:1-13`.
- SLAM Dunk: README and `main.cpp` as summarised in `experiments/lamaria-handoff.md:169-187`; code not read for this card.
