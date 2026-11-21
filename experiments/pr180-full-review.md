# PR #180 full review, 2026-11-21 (950aa17..70c3a45)
Six read-only Fable lanes. Reports verbatim below. Lane letters: A VI-BA core, B map/inertial/factors, C system, D tracking/sensors, E loop closure/upstream, F apps/configs/scripts. The earlier delta review (79c04bb..d7c2bf3) findings E-1..E-6 and T-1..T-8 are in experiments/pr180-U-card.md, pr180-V-card.md (E-1, E-2, T-1, T-3, T-4 fixed in bca842d..70c3a45).

---

## Lane A

# Lane A report: VI-BA core (PR #180, 950aa17..70c3a45)

Read-only review. No builds, no runs. Line numbers are at 70c3a45. Nothing here was executed; every finding is from reading code, and unproven ones are marked "suspect".

## Verdict for the validation run

Lane A does not block the run. No high-severity defect found.

The `configs/pr180-window10.ron` path through this lane is: `ScaledEdge` boundary, no online calibration, no marginal prior, rig observations, linearization cache, `DenseLlt` Sequential, 7 iterations from λ 1e-4, then offline BA. On that path the solver math matches the pre-PR solver. I checked the knob-off equivalence hunk by hunk (cost sum order, boundary scaling, Huber selection, whitening by 1.0, point damping, Cholesky); it holds except for the items in A-2 and A-3.

Two things can mis-score the run: A-1 (offline hard outlier drop before any iteration) and A-4 (rejected steps eat the 7-iteration budget). Watch `dropped_10px` in the `[offline_ba]` line and `rejects=` in `[vi_ba] TIME`.

**Top 3 fixes**
1. A-1: do not drop offline observations at 10 px on the unoptimized online state, or at least report the dropped share and refuse above a bound.
2. A-4: count accepted steps (linearizations) against `max_iterations`; cap rejects separately.
3. A-5: keep stereo depth rows when the online calibration gate is closed (knob-on only, not in the preset).

## Theory-review fixes: do they hold?

Checked in code: E3, E8, E10, E13, D1, D2, D3, D4, D5, D6, D8, D9, D10, D12, D14 all hold as described in `experiments/pr180-theory-review.md`.
- E3: `undamped_information` (vi_ba_schur.rs:2398-2415) rebuilds at the accepted state with λ = 0 and drops rank-deficient points; used for both `newest_marginal` and the calibration sigma (3007-3024).
- D1: `boundary_under_calibration` (1360-1382) used before the solve with carried values and after it with solved values (bundle_adjustment.rs:863-893, 968-992).
- D10: `SE3F32::retract` is `T · Exp` with the full SE3 exponential (kornia-algebra `se3.rs:96-109`), so `J_r⁻¹(d)` in `accumulate_prior` is the exact Jacobian.

Still open, as the review says: D7, D11, D13. Not re-reported.

## Findings

### A-1, medium, suspect: offline BA drops 10 px observations before the first iteration, for good
- **Where:** `global_vi_ba/mod.rs:238-241`; `global_vi_ba/problem.rs:393-406`; `Observation::active` "never comes back" (problem.rs:116-117).
- **Defect:** `drop_outliers(&start, …, outlier_px)` runs on the dumped online state. A dropped observation is never re-admitted.
- **Failure scenario:** a landmark moved by later local windows is still linked to old, now-fixed keyframes. Those old observations exceed 10 px at the dump state. They are exactly the constraints that tie old keyframes to new ones. Dropped, the offline solve cannot pull the drift back, and `observers()` / `Pattern` are then built without those links (mod.rs:243, 260). Offline ATE then matches online instead of improving.
- **Evidence:** the Huber kernel (knee 25, problem.rs:330) already bounds such rows; ORB-SLAM3 `FullInertialBA` has no pre-solve drop. I did not measure how many observations exceed 10 px in a real dump, hence "suspect".
- **Minimal fix:** run the first stage Huber-only and apply the 10 px drop after the first convergence (the D8 `tightened` machinery already supports a staged drop). Cheaper stop-gap: check `dropped_10px / observations` in the validation logs.

### A-2, low, default-path change: the E8 clamp is unconditional and can trigger at the default window
- **Where:** `bundle_adjustment.rs:315-321`, used at 674.
- **Defect:** `inertial_active_start` works on the snapshot's keyframes, not the map's. The card's claim "knobs-off unchanged, needs a map of at most 3 keyframes" covers only the map case.
- **Failure scenario (suspect):** a re-seeded segment with no IMU edge across the jump and no landmark shared with older keyframes gives a snapshot of exactly 3 keyframes with no context. `start == 0`, no anchor, so the oldest is now fixed. Pre-PR all three were free.
- **Evidence:** `local_window_of` adds context only through shared landmarks or IMU links (256-269).
- **Fix:** none needed for correctness (the new behaviour is the right one). Record it as an intended default-path change, or put it behind a knob to satisfy the owner rule.

### A-3, low, default-path change: `rigid()` on visual-BA write-back
- **Where:** `bundle_adjustment.rs:445`, `545`.
- **Defect:** `run_initial_ba` and `run_local_ba` now orthonormalize the solved rotation. No knob. Bits change whenever kornia-3d returns a slightly non-orthonormal rotation.
- **Fix:** record as intended (it matches the `RIGID_TOLERANCE` check in `system/mod.rs:796`).

### A-4, low to medium, suspect: rejected steps consume the 7-iteration budget
- **Where:** `vi_ba_schur.rs:2596-2598` (`iters_done += 1` on every pass), reject paths 2660-2668, 2759-2768, 2893-2899.
- **Defect:** a Cholesky failure, a cheirality refusal and a cost-increasing step each cost one of `max_iterations`. With the cache (2607-2608) a reject is only a re-assemble, yet it counts like a full iteration.
- **Failure scenario:** preset starts at λ 1e-4, near Gauss-Newton. A window with depth-free landmarks or a poor seed needs λ ≈ 1e-1: three rejects, four accepted steps at most. With 50 iterations this did not matter; with 7 it does.
- **Evidence:** pre-existing semantics, but the preset newly depends on it. Card T's R_01 numbers already include this effect; the challenge sequences are untested.
- **Fix:** count linearizations (accepted steps) against `max_iterations`, with a separate reject cap.

### A-5, medium, knob-on only: D5 fix removes the stereo metric anchor from every solve
- **Where:** `vi_ba_schur.rs:2546-2548`.
- **Defect:** with `camera_rotation` or `camera_focal` on, `depth_on.fill(false)` applies to every solve, gate open or closed.
- **Failure scenario:** stereo run with online camera calibration, robot slow or parked. The gate is closed, the states are constants, and yet the depth rows are gone. The file's own comment (772-775) records about 25 % scale drift without them.
- **Fix:** when the gate is closed, keep the rows with the measurement corrected by the carried focal scale, `(1 + s)·d_meas`. When open, add the focal column the theory review offered as the alternative.
- Not on the preset path (no calibration, rig frames carry no `frame.depth`).

### A-6, low: `newest_marginal` reports variance 1.0 for a held keyframe
- **Where:** `vi_ba_schur.rs:1073-1094`, `2380`, `3018-3024`.
- **Defect:** `hold_inertial_state` writes identity rows. With `LostKeyframes::FixedInertial` and a newest keyframe inserted while lost, the returned 15×15 has v and bias blocks equal to I, cross terms zero.
- **Failure scenario:** the IMU trust consumer reads a velocity sigma of 1 m/s that is a placeholder, not an estimate.
- **Fix:** return `None` when the newest keyframe is `inertial_fixed`.

### A-7, low: offline solve stops silently
- **Where:** `global_vi_ba/mod.rs:409-411`.
- **Defect:** a Cholesky failure after iteration 1 does `break`; `Ok(None)` also ends the solve. The report has no field saying why it stopped. `report.iterations` is documented "accepted LM steps" but stores the index of the last accepted iteration (400).
- **Fix:** add a stop reason to `OfflineBaReport`; count accepted steps.

### A-8, low: library function mutates a process global
- **Where:** `global_vi_ba/mod.rs:222`, `faer::set_global_parallelism(Par::Seq)`.
- **Defect:** it is never restored. `DenseParallelism::Global` documents the side effect (dense_llt.rs:21-24) instead of removing it.
- **Failure scenario:** any process that runs offline BA and then online mapping silently loses the threaded dense solve.
- **Fix:** save and restore the previous value around the solve.

### A-9, low: offline weighting is not "as the online VI-BA does"
- **Where:** `global_vi_ba/problem.rs:283`; doc claim at `mod.rs:209-213`.
- **Defect:** offline uses `visual.sigma_px(octave)` only. Online also applies `fisheye_weight` (bundle_adjustment.rs:1385-1388). Offline also ignores the dump's `anchor`, `evicted` and `inserted_while_lost` flags (problem.rs:236-258).
- **Failure scenario:** a run with `FisheyeWeight::Magnification` gets different weights offline. Evicted keyframes enter with the pose they had at eviction (suspect: wrong if the world epoch changed later; depends on `session.rs`, another lane).
- Not on the preset path for the weight part.

### A-10, low: project rules
- **Monoliths:** `solve_vi_ba` is about 665 lines (vi_ba_schur.rs:2463-3128); `run_local_inertial_ba_inner` about 360 (bundle_adjustment.rs:647-1009).
- **Triplicated code:** the per-observation view construction appears three times (2056-2067, 2788-2799, 2962-2973); the trial cost calls `residual_and_jacobians` and discards the Jacobians (2800).
- **Duplicated across online/offline:** the re-integration slice (vi_ba_schur.rs:1523-1564 against imu_edges.rs:124-153), the calibrated projection Jacobian (1712-1736 against problem.rs:605-633), `MIN_DAMPING_FLOOR`, `MIN_Z`/`MIN_DEPTH`.
- **Inconsistent constant:** IMU Huber χ² is 16.9 online (vi_ba_schur.rs:381) and 16.92 offline (imu_edges.rs:23).
- No `unwrap`/`expect`/`panic!`/`unsafe` in non-test code of the lane. `synthetic_walk` is `#[cfg(test)]`.

### A-11, low: test gaps
- Window tests W1-W5 use `LocalBaParams::default()` (50 iterations, λ 1), not the preset's 7 and 1e-4 (vi_ba_window_tests.rs:92-108). W5 asserts nothing.
- No end-to-end solve through the rig path; only the Jacobian test `a_rig_observation_differentiates_the_primary_pose`.
- Offline tests have one camera, one segment, one component: no rig, no bridge, no multi-anchor gauge.

## Theory gaps against the references

1. **Zero-mean accel-bias prior on every free keyframe, every solve** (vi_ba_schur.rs:2329-2344, weight 1e2). No reference system has it. It is re-applied in each window, so it is not a prior in the Bayesian sense; with 10 active keyframes it is 10 pulls toward zero per solve. Pre-existing, but the larger window multiplies it.
2. **`newest_marginal` is a conditional covariance.** Fixed keyframes are treated as exact, so it is the uncertainty relative to the window boundary. OKVIS2 and Basalt get the absolute one from the marginal prior.
3. **Marginal prior replaces vision by an isotropic pseudo-prior** (marginalization.rs:266-269, 2 mm and 1 mrad). OKVIS and Basalt marginalise the real visual factors. In a feature-poor scene the prior is overconfident. The IMU edge enters unrobustified (239).
4. **LM without a gain ratio, cost compared in f32** (2627, 2865, 2869), poses stored as `SE3F32` (2713-2716). At 500-1000 m from the origin f32 translation resolution is about 6e-5 m, near the IMU `r_p` sigma. Pre-existing; LaMAria trajectories are long.
5. **Preset keeps the non-reference defaults** `boundary_scales_bias_rows: true` and `ImuHuber::AllEdges`, though the ORB-SLAM3-faithful settings exist.
6. **Huber knees:** 25 on whitened χ² (5σ) against ORB-SLAM3's 5.991; depth 9 against 3.84 for one row.
7. **Offline IMU kernel defaults to `None`** while the module cites `FullInertialBA`, which uses Huber 16.92; edges up to 10 s enter at full weight.

## Knob table

| Knob (default) | Class | Derivation or note |
|---|---|---|
| E8 oldest-keyframe clamp (no knob) | bug fix | already the only path |
| `merge` SnapshotOnly / PropagateToNewer | bug fix | PropagateToNewer should be the only path in async mode |
| `visual_kernel` L2 / Huber | bug fix | Huber only |
| `boundary_scales_bias_rows` (true) | bug fix | false is ORB-SLAM3; make it the only path |
| `depth_sigma` Relative(0.05) / Disparity | bug fix | Disparity is the sensor model: σ_z = z²·σ_d/(f·b) |
| `visual_information`, `fisheye_weight` | bug fix | sensor model: octave scale and lens magnification |
| `lost_keyframes` Free / FixedInertial | bug fix | or derive: hold v and bias when their information block is rank-deficient |
| `imu_huber` (AllEdges) | reference constant | `BoundaryOrRecentInit` is ORB-SLAM3 |
| `huber_imu_chi2` 16.9 | reference constant | χ²₉ at 95 % |
| `boundary_imu_info_scale` 1e-2 | reference constant | ORB-SLAM3; `Marginal` removes the need |
| `LargeWindow` 100 / 25 / 4 / 1e-2 | reference constant | ORB-SLAM3; preset uses Basalt's 7 and 1e-4 |
| `GATE_MIN_SPEED_MPS` 0.2 | reference constant | VINS-Fusion |
| `MAX_EDGE_GAP_SEC` 10 | reference constant | VINS-Fusion |
| offline `outlier_px` 10, `tight_outlier_px` 3, `tighten_from_iteration` 8, `landmark_damping_floor` 0.01 | reference constant | visloc; thresholds could be χ² quantiles × σ_px |
| `active_keyframes` (3) | tunable | from observability: grow until the scale or velocity marginal falls below a bound, or a fixed time span |
| `max_iterations`, `initial_lambda` | tunable | from the time budget (keyframe interval) and a gain-ratio λ rule |
| `huber_reproj_px_sq` 25, `huber_depth_sigma_sq` 9 | tunable | χ² quantile on whitened residuals; needs a real σ_px from the tracker |
| `accel_bias_prior_weight` 1e2 | tunable | from the IMU turn-on bias spec, applied once; or drop when `Marginal` carries bias information |
| `FrozenPoseSigma` 2 mm, 1 mrad | tunable | the frozen keyframe's own pose marginal (same machinery as `newest_marginal`) |
| `STALE_TRANSLATION_M` 0.5, `STALE_ROTATION_RAD` 10° | tunable | Mahalanobis distance of `d` under the prior's H |
| `depth_gate_factor` 10, margin 1 m, re-seed factor 100, agreement 3σ | tunable | kσ gate from `depth_sigma`; 3σ is a quantile |
| online and offline calibration prior sigmas, `time_slices` 8 | tunable | factory calibration tolerances; slice count from the thermal time constant |
| `CLOCK_STEP_SEC` 0.25 ms, `GYRO_STEP`, `ROTATION_STEP_RAD` | numeric | clock step is hard-coded to a 1 kHz IMU; derive from the stream's sample period |
| `GAUGE_POSITION_SIGMA`, `GAUGE_YAW_SIGMA`, `MIN_DAMPING_FLOOR`, `UNDAMPED_POINT_CONDITION`, λ bounds | numeric | not deployment parameters |
| `newest_marginal`, `dense_parallelism` | not behaviour | output flag; threading with bit-identical result |
| `offline imu_kernel`, `bridge_segments` | tunable | kernel should be on (reference); bridging from gap length and preintegration covariance |

## Coverage

| File | Hunks | Read | Notes |
|---|---|---|---|
| mapping/vi_ba_schur.rs | 34 | 34 | Lines 60-3208 read in full at head; removed lines read from the diff. Tests 3209-3848 read with blank lines stripped. Header doc 1-59 not opened (no hunk there). |
| mapping/bundle_adjustment.rs | 26 | 26 | Whole file read at head, plus the diff. |
| mapping/bundle_adjustment_tests.rs | 12 | 12 | Diff read condensed (blank and closing-bracket lines stripped). |
| mapping/calibration_states.rs | 1 | 1 | Full. |
| mapping/dense_llt.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/imu_edges.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/mod.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/problem.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/reduced.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/report.rs | 1 | 1 | Full. |
| mapping/global_vi_ba/tests.rs | 1 | partial | Outline only: doc comments, signatures, asserts. Fixture bodies (22-139) not read line by line; time went to the solver files. |
| mapping/marginalization.rs | 1 | 1 | Full, tests condensed. |
| mapping/online_calibration.rs | 1 | 1 | Full. |
| mapping/synthetic_walk.rs | 1 | 1 | Full; test-only module. |
| mapping/vi_ba_calibration_tests.rs | 1 | partial | Outline only: doc comments, signatures, asserts. Generator and helpers (32-480) not read line by line; same reason. |
| mapping/vi_ba_flung_landmark_tests.rs | 11 | 11 | Mechanical: `ObservationViews::PRIMARY`, `inertial_fixed`, `Gauss` moved. |
| mapping/vi_ba_prior_tests.rs | 1 | partial | Outline only; fixture bodies not read line by line. |
| mapping/vi_ba_schur.rs tests | (above) | (above) | |
| mapping/vi_ba_stationary_tests.rs | 5 | 5 | `inertial_fixed: false` only. |
| mapping/vi_ba_wheel_tests.rs | 1 | 1 | `inertial_fixed: false` only. |
| mapping/vi_ba_window_tests.rs | 1 | 1 | Committed version from `git show 70c3a45`; working-tree change ignored. |

Cross-lane notes for whoever owns them:
- `local_mapping.rs:63-65`: with the preset's `min_tracked_inliers: 0` the test is strict `>`, so a keyframe tracked with 0 inliers solves with 10 keyframes at 50 iterations and λ 1.0, the slow solve card R measured at about 1 s.
- `reanchor_frames` (global_vi_ba/mod.rs:287-302) writes only frames in `dump.frames` whose reference keyframe is dumped. Whether coasted frames are in the dump decides if the offline trajectory has the same coverage as the online one under `--trajectory-include-coasted`.
---

## Lane B

# Lane B review: mapping ops, inertial, initialization, factors (PR #180, 950aa17..70c3a45)

Read-only review. No builds, no runs. All claims come from reading code; none is test-verified.

## Verdict for the validation run

No blocker found in lane B. The run can go. `PropagateToNewer`, the baseline gate, rig seeds and the graph-correction writeback are correct as read. Three things can change the score or timing: B-1 (preset omits card V's base budget), B-2 (hot-path cost), B-3 (O(map) work under the map lock on long sequences).

Top 3 fixes:
1. B-1: add `max_iterations: 7, initial_lambda: 1e-4` at base level in `configs/pr180-window10.ron`.
2. B-2: make `observer_keyframes` linear again for single-camera records.
3. B-3: have `carry_landmarks` walk the carried keyframes' feature slots, not every map point.

## Findings

### B-1, medium: validation preset leaves 0-inlier VI jobs on the 50-iteration, λ = 1.0 budget
- **Where:** `crates/kornia-slam/src/mapping/local_mapping.rs:63-75` (`large_window`, `ba_params`); `configs/pr180-window10.ron:43`; `bundle_adjustment.rs:151` (`tracked_inliers > min_tracked_inliers`).
- **Defect:** with `min_tracked_inliers: 0`, a job with `tracked_inliers == 0` never takes the large window. It solves with the base `LocalBaParams`: default `max_iterations: 50`, `initial_lambda: 1.0`, 10 keyframes. The preset sets 7 iterations and 1e-4 only inside `large_window`.
- **Sources of 0-inlier jobs:** the job after IMU init (`system/mod.rs:1716`), the job after PGO (`:3384`), and every keyframe inserted "when not tracked" (`:3032`).
- **Scenario:** challenge sequences lose tracking. Each lost or coasted keyframe runs a solve of up to 50 iterations on the async worker. Card V measured this preset on R_03: 4-8 solves over 7 iterations, up to 23 iterations, slowest 702 ms. Latest-wins then skips jobs and more keyframes need propagation.
- **Evidence:** card V line 69 says "the candidate sets max_iterations: 7 and initial_lambda: 1e-4 at the base level". K1 did not fire, and the ALL runs were neutral or better. The preset was not updated.
- **Fix:** set the base `max_iterations: 7, initial_lambda: 1e-4` in the preset. Or state in the preset header that the base budget is left at 50 on purpose.

### B-2, medium (performance, default path): `observer_keyframes` is now quadratic per landmark
- **Where:** `crates/kornia-slam/src/mapping/map/map_point.rs:168-178`.
- **Defect:** the dedupe scans the prefix for every record, so it costs O(n²) per landmark. Before it was a plain map.
- **Callers:** `tracking/local_map.rs:126, 188, 209` (per frame, per matched landmark); `bundle_adjustment.rs:258` (every snapshot, under the map lock); `geometry.rs:63` (every merge, every live landmark); `queries.rs:33`.
- **Scenario:** a landmark with 100 observers costs about 5,000 compares. At 1,000 matched landmarks that is about 5M compares per frame, a few ms. It hits non-rig runs too, where no dedupe is needed. The file's own comment notes a parked map gives every landmark one observer per keyframe.
- **Fix:** dedupe only when a record has `camera != 0`. Or keep a keyframe's records adjacent and compare with the previous one.

### B-3, low-medium: `carry_landmarks` scans every map point under the map lock and publication gate
- **Where:** `crates/kornia-slam/src/mapping/map/ops/correction.rs:927-947`; the `in_snapshot` HashSet is built at `:781`.
- **Defect:** each propagating merge walks all `map_points`, culled slots included, and builds a HashSet of the snapshot's landmarks.
- **Scenario:** on long sequences the cost grows with the map, on every merge. `landmarks_observed_by` already does the same (pre-existing), and B-2 makes that one quadratic per point.
- **Fix:** iterate `map_point_by_desc_idx` of the carried keyframes and keep points whose `keyframe_idx` is carried and that are not in the snapshot. The snapshot's `landmark_ids` is sorted, so `binary_search` replaces the HashSet.

### B-4, low: `todo!()` in library code on the worker thread
- **Where:** `local_mapping.rs:355, 364, 522`.
- **Defect:** three `todo!()` panics. They are reachable only after `with_deferred`, which has no caller (`#[allow(dead_code)]`).
- **Scenario:** not reachable in the validation run. A later caller that wires `with_deferred` before lane T lands would panic the mapping worker; mapping then stops with one stderr line.
- **Fix:** return `false` or an empty Vec, and skip the pop, until the path is built. This also meets the CLAUDE.md rule against panics in library code.

### B-5, low (suspect): propagation stops at an anchor instead of skipping it
- **Where:** `correction.rs:877`.
- **Defect:** `live.anchor || incoming.is_empty()` breaks the loop. Anchors are "left out of the IMU chain" (`keyframe.rs`), so an edge N → N+2 can exist around anchor N+1.
- **Scenario:** N+2 keeps its stale pose and velocity. Its edge from the corrected N is at full information, not inflated like a bridge. Knob-gated: anchors exist only with the lost-keyframe anchor knob.
- **Related:** if the map's newest keyframe at capture is an anchor held as fixed context, `newest_correction` (`:770`) is `None` and nothing propagates.
- **Fix:** `continue` past anchors. Take the newest *active* snapshot keyframe as the carry reference.
- **Evidence gap:** I did not confirm that the IMU chain skips anchors with a spanning edge. The test `a_propagating_merge_does_not_cross_an_anchor` encodes the current behaviour as intended.

### B-6, low: unlinking a secondary-camera observation at the reference keyframe moves the reference
- **Where:** `crates/kornia-slam/src/mapping/map/ops/mutation.rs:346-356`.
- **Defect:** `was_reference` tests the keyframe only. With a rig, removing the camera-1 link of the reference keyframe calls `adopt_reference`, although the camera-0 reference feature is still there. `adopt_reference` picks the smallest `(keyframe, feature)`, usually an older keyframe.
- **Effect:** `keyframe_idx` and `reference_octave` change. `apply_graph_correction` and `carry_landmarks` then move the point with another keyframe.
- **Reach:** `unlink_observation` has no non-test caller today.
- **Fix:** adopt a new reference only when no record of the reference keyframe remains.

### B-7, low: baseline gate uses camera-0 depth for every rig camera
- **Where:** `growth.rs:142-144`; `queries.rs:72-90`.
- **Defect:** `scene_median_depth` takes all the keyframe's landmarks, secondary-camera ones too, as z in the camera-0 frame, and drops z ≤ 0. The result gates each camera's pair.
- **Scenario:** on Aria's diverging cameras the median mixes views. A side camera's pair is gated on a depth that is not its own. The preset turns the gate on (0.0175).
- **Fix:** take the median per camera, in that camera's frame, over `camera_range(c)`.

### B-8, low: `VibaWindow::VisualOnly` and anchors leave stale velocities after a re-alignment
- **Where:** `inertial/initialization/mod.rs:216-223`; `correction.rs` `apply_inertial_alignment`.
- **Defect:** keyframes left out of the solve get no velocity assignment. The world is still scaled and rotated, so their `velocity_world` stays in the old frame. This already held for keyframes before `start_idx`.
- **Scenario:** VIBA1 or VIBA2 with a non-trivial rotation, with the knob on. The test `visual_only_writes_velocities_by_keyframe_and_resumes_at_the_newest` asserts the sentinel velocity is kept, so tracking resumes from an unrotated velocity when the newest keyframe was a lost one.
- **Fix:** rotate and scale the velocity of every unassigned keyframe inside `apply_inertial_alignment`.

### B-9, low: default-path numeric changes with no knob
All look like bug fixes or refactors. Flagged under the owner rule; none is proven bit-identical to 950aa17.
- `pose_conversion.rs:35`: `apply_reference_pose_correction` now orthonormalizes. Its test allows 1e-12, not bit equality. `pose_to_se3` / `se3_to_pose` are also rigid now.
- `geometry.rs:194`: `scale_power` (repeated multiply) replaces `powi` for the distance bounds. The new test compares new code with new code.
- `inertial/initialization/mod.rs` `window_is_mono`: `is_metric()` replaces `is_stereo()`. Same for stereo and mono; rig is new.
- `culling.rs:486-505`: `is_redundant` counts distinct keyframes. Same result without a rig.
- `ready()` and `initialize_stationary` skip anchors. Same result with no anchors.

### B-10, low (suspect): `rig_seeds` has no finite check
- **Where:** `growth.rs:95`.
- **Defect:** `point.z > max_depth` is false for NaN, so a NaN point becomes a seed. `apply_insertion` then refuses the whole batch with `NonFinitePosition`, and the keyframe is not inserted.
- **Evidence gap:** `frontend/overlap.rs:216` rejects z ≤ 0 but has no explicit NaN test. I did not trace whether `triangulate_n_view_linear` can return NaN.
- **Fix:** add `!point.z.is_finite()` to the skip condition.

## Checked and found correct
- **Propagation math (`propagate_to_newer`):** `v_j = v_i + g·dt + R_wb,i·Δv(b_i)` matches the factor's `r_v`. `body_frame` returns `R_bw`. The rotation fallback `R_afterᵀ·R_before` is the world rotation. The pose carry and the landmark carry use the same world transform.
- **Merge ordering:** the worker takes its snapshot after the previous merge, so no correction is applied twice.
- **Card V E-2:** the segment-start stop is in place and has a unit test.
- **`apply_graph_correction`:** `C = solved⁻¹·snapshot`, `T' = T·C⁻¹`, landmarks by `C`, velocity by `C.rotation`. Epoch is checked before any write, and everything is validated first. The PGO runs on a gravity manifold (`closer.rs:1354`), so gravity alignment holds.
- **Marginal prior:** it carries a world epoch, and `ba_snapshot` filters on it. It is cleared when its keyframe or edge is removed, and stored only on an accepted merge. With `boundary: ScaledEdge` (default, and the preset) it is not used.
- **`propagate_state_covariance`:** residual order `[R v p]` against state order `[ρ ω v]` is handled through `J_j`; the noise blocks match `information`. Two Monte Carlo tests cover it, one on a rotated rig.
- **Midpoint window and pruning:** bracketing samples are kept, and `interpolation` is carried through merge and `repreintegrate`.
- **`keyframe_slot`:** the binary search falls back to a linear scan, so an out-of-order list is still correct.
- **Map invariants with two cameras:** `apply_insertion`, `check_link` and `merge_landmarks` key on (landmark, keyframe, camera). Keyframe removal drops every record of that keyframe.
- **`unsafe`:** none in lane B.

## Knob table

| Knob | Class | Note |
|---|---|---|
| `local_ba.merge: PropagateToNewer` | bug fix | Async mapping without it leaves newer keyframes inconsistent. Make it the only path after validation. |
| segment-start stop (card V E-2, no knob) | bug fix | Already unconditional inside the merge. |
| `local_ba.state_envelope` (5 m/s, 0.5 m/s², 0.3 rad/s) | tunable | Derive from the prior: k·σ of bias walk over elapsed time plus the init covariance. Speed from the platform or the marginal covariance. |
| `large_window` (inliers, keyframes, iterations, λ) | tunable | Iterations: stop on cost decrease or a time budget. λ: from the last accepted solve. Size: from the compute budget. |
| `local_ba.active_keyframes` | tunable | From the time budget, or the span needed for bias and scale observability. |
| `local_ba.max_iterations`, `initial_lambda` | tunable | As for `large_window`. One budget for all jobs (B-1). |
| `growth_min_baseline_ratio` (0.0175) | tunable | Derivable: depth error σ_z/z ≈ z·σ_px/(f·b), so ratio = k·σ_px/f. At f = 241 and 1 px, 0.0175 is about 4σ. ORB-SLAM3 uses 0.01. |
| `min_cold_start_points` (default 50, preset 20) | tunable | Derive from the pose covariance of the seeded PnP instead of a count. |
| `GrowthNeighbours::OrbSlam3 { covisible }` | reference constant | 10 (30 mono), 15 shared points, 20 fusion targets: ORB-SLAM3 values. |
| `KeyframeSeed::ImuPropagated` | bug fix candidate | ORB-SLAM3 behaviour; the seed is then consistent with the edge. |
| `VibaWindow::VisualOnly` | tunable (policy) | Better: weight lost keyframes by their pose covariance, not exclude them. See B-8. |
| `viba_velocity_gate` (k × median, cap 25 %) | tunable | Replace with a χ² test on the 3-DOF velocity residual (7.81 at 95 %); the covariance is already used. |
| `VisualInformation::OctaveScaled` | reference constant | ORB-SLAM3 `invSigma2`. Should be the only path once scored. |
| `FisheyeWeight::Magnification` | bug fix | Follows from the camera model; no parameter. |
| `ImuTrustMode::Covariance { search_sigmas }` | tunable | Already uncertainty-based; `search_sigmas` is a confidence level (3). |
| `SampleInterpolation::Midpoint` | bug fix | `Hold` lags by dt/2 (prior review D3). Make it the only path. |
| `LostKeyframes::FixedInertial`, anchors | tunable (policy) | Could follow from observability: fix states whose marginal information is below a floor. |
| `VisualKernel::Huber` | reference constant | Knees 5.991 / 7.815 are χ² values. |
| `keyframe bound (max_keyframes)` | tunable | Memory budget; a deployment limit, not an estimator parameter. |
| `MAX_PLAUSIBLE_ACCEL_BIAS` 1.0, init `min_keyframes` 10, `min_time` 1 s | reference constants | The last two are ORB-SLAM3's; the bias cap could come from the IMU datasheet. |
| `FUSE_SEARCH_RADIUS_PX` 7, `FUSE_MAX_HAMMING` 50, `MIN_GROWTH_MATCHES` | reference constants | Pre-existing. The radius could scale with the projected pose covariance. |

## Coverage

| File | Hunks | Read | Notes |
|---|---|---|---|
| mapping/culling.rs | 4 | 4 | |
| mapping/growth.rs | 14 | 14 | Also read the unchanged body of `pair_growth_in_camera`. |
| mapping/keyframe_culling_tests.rs | 4 | 4 | Tests; added lines only. |
| mapping/keyframe_mapping.rs | 20 | 20 | |
| mapping/local_mapping.rs | 16 | 16 | |
| mapping/map/imu_factor.rs | 2 | 2 | |
| mapping/map/keyframe.rs | 2 | 2 | |
| mapping/map/map_point.rs | 9 | 9 | |
| mapping/map/mod.rs | 6 | 6 | |
| mapping/map/ops/correction.rs | 16 | 16 | Plus surrounding context of `apply_ba_update` and the alignment. |
| mapping/map/ops/geometry.rs | 3 | 3 | |
| mapping/map/ops/graph_correction_tests.rs | 1 | partial | New test file: test names and assertions only. |
| mapping/map/ops/mod.rs | 1 | 1 | |
| mapping/map/ops/mutation.rs | 12 | 12 | |
| mapping/map/ops/mutation_tests.rs | 5 | 5 | Added lines only. |
| mapping/map/ops/queries.rs | 2 | 2 | |
| mapping/map/ops/removal.rs | 3 | 3 | |
| mapping/map/ops/removal_tests.rs | 2 | 2 | |
| mapping/map/ops/snapshot.rs | 7 | 7 | |
| mapping/map/tests.rs | 3 | 3 | |
| mapping/mod.rs | 1 | 1 | |
| mapping/session.rs | 1 | 1 | New file, read whole. |
| mapping/visual_information.rs | 1 | 1 | New file, read whole. |
| kornia-factors/src/inertial/preintegration.rs | 3 | 3 | Also read the unchanged residual and Jacobians. |
| inertial/initialization/factor.rs | 2 | 2 | |
| inertial/initialization/gate.rs | 1 | 1 | New file, read whole. |
| inertial/initialization/mod.rs | 8 | 8 | |
| inertial/initialization/solve.rs | 7 | 7 | Read the whole file at HEAD. |
| inertial/initialization/stationary.rs | 9 | 9 | |
| inertial/initialization/tests.rs | 11 | partial | Tests: function names, doc lines and assertions only, not the bodies. |
| inertial/mod.rs | 9 | 9 | |
| inertial/schedule.rs | 3 | 3 | |
| inertial/trust.rs | 1 | 1 | New file. |
| initialization/bootstrap.rs | 7 | 7 | |
| initialization/two_view.rs | 1 | 1 | |
| lib.rs | 2 | 2 | |
| pose_conversion.rs | 2 | 2 | Read the whole file at HEAD and the removed lines. |
| stereo.rs | 1 | 1 | Test fixture field only. |

Out of lane, read only to judge a hunk: `bundle_adjustment.rs` (`LocalBaParams`, `local_window_of`), `system/mod.rs` (job construction), `frontend/overlap.rs` (`triangulate`), `loop_closure/closer.rs` (`write_back`), `frame.rs` helpers, card V. The uncommitted change in `vi_ba_window_tests.rs` was ignored.
---

## Lane C

# Lane C review: system crate (PR #180, 950aa17..70c3a45)

Read-only. No builds, no runs. Line numbers are at 70c3a45. Paths are under `/home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/system/` unless given in full.

## Verdict for the validation run

Lane C does not block the run mechanically, but it can mis-score it. Three things matter:

- **Runaway coasts can be written as scored poses (C-1).** The T-4 runaway check is off in `pr180-window10` because it is tied to `reseed.keep_inertial`, which the preset leaves false.
- **Every give-up shifts the rest of the trajectory (C-3).** The preset has no card B knobs, so a give-up re-seeds at the pose from before the loss.
- **Card U is inert in the preset.** `frame_velocity` is not set, and the knob-off path is the old code, so U cannot affect the run.

No panic path is reachable from the preset. `keyframe_orb` stays `Tracking`, and all `configs/*.ron` are parse-tested in `apps/kornia-slam-app/src/config/tests.rs:304`.

Top 3 fixes:
1. C-1: make the runaway check for coast trust (and for lost-keyframe insertion) independent of `keep_inertial`.
2. C-3: decide the re-seed policy for the run, or at least read `give_ups` per sequence and score those sequences separately.
3. C-2: bound the age of a scored coast, or ship U together with V1.

## Findings

### C-1, high: the T-4 runaway check is inert without `keep_inertial`
- **Where:** `mod.rs:1227`, `mod.rs:2138-2145`, `mod.rs:2189`.
- **Defect:** `settle_coast_trust` computes `runaway = self.recovery.keep_inertial && self.runaway.runaway()`. The detector is also only fed inside `if self.recovery.keep_inertial`. With the preset's default (`keep_inertial: false`), `trusted` is just `imu_confident`, which is 2 s since IMU init.
- **Failure scenario:** a loss on a challenge sequence coasts for up to `timeout_imu_sec` = 5 s (`tracking/policy.rs:285`). Every one of those frames is `Coasting { inertial: true, trusted: true }`, and `writes_scored_pose` (`apps/kornia-slam-app/src/main.rs:668-682`) writes it under `--trajectory-include-coasted`. One diverged pose in the file breaks the alignment for the whole sequence.
- **Evidence:**
  - Commit 74f158b's message records "the runaway inertial coast into the seed (poses at 5.8e29 m)" on LaMAria sequence_5_11.
  - The same gate at `:2189` means lost keyframes are still inserted at runaway poses in this preset.
  - The unit test `a_coast_is_trusted_only_when_confident_and_not_a_runaway` builds its system with `keep_inertial = true`, so it does not cover the default.
  - Card L's 2026-11-21 note says the flag no longer writes a coast "the runaway check flags". That is only true with `keep_inertial`.
- **Minimal fix:** record the runaway track unconditionally at `:2138` and drop `keep_inertial &&` at `:1227`. This changes only the skip kind, not the estimator. Add a finite-pose check in the app before writing. Consider dropping the gate at `:2189` too, as a bug fix.

### C-2, medium (magnitude is suspect): scored coasts run on a nearly frozen velocity
- **Where:** `mod.rs:1760`, `mod.rs:3061-3064`, `mod.rs:2026-2033`.
- **Defect:** with `frame_velocity` off, each coast frame resets the velocity to the reference keyframe's and adds one frame of Δv. A lost keyframe is seeded with that same value. So across a 0.2 s lost-keyframe interval of N frames, the velocity chain keeps only about 1/N of the measured acceleration.
- **Failure scenario:** a coast through a turn drifts about ½·a·t²·(1−1/N) between BA merges. Card U measured 0.70-0.80 m/s velocity error at 5 or more frames, on tracked frames. These poses are scored under the flag.
- **What bounds it:** the async VI-BA merge re-anchors the pose and velocity (`:1051-1067`). Lost-keyframe jobs carry 0 inliers, so they take the base budget of 50 iterations from λ 1.0, which card V measured at up to 880 ms per solve.
- **Counter-evidence:** card L saw no ATE cost on short R_03 gaps. The size on long challenge gaps is a derivation, not a measurement.
- **Minimal fix:** score a coast only while `lost_for` is under about 1 s, or under a covariance bound. Otherwise ship U with V1 in the preset.

### C-3, medium: a default give-up re-seeds at the pose before the loss, with no knob
- **Where:** `mod.rs:2719-2725`, `state.rs:158-162`. Introduced by 910ac20.
- **Defect:** main kept the coasted pose (`state.reset()`). HEAD seeds the new map at `pose_before_loss`. That is a default-path change with no knob. The intended path (`keep_inertial`: coasted seed plus a bridge edge) is not in `pr180-window10`.
- **Failure scenario:** after a 5 s grace period at walking speed, the new segment starts about 5-7 m behind the device and stays offset for the rest of the sequence. The written file shows coasted poses moving forward, then a `KeyframeAccepted` pose that jumps back.
- **Related:** `pose_before_loss` is deliberately not moved by BA corrections (revert 74f158b). A cold re-seed keyframe is not marked `segment_start` (`:1303-1307` sets it only when `reseeding()`), so mapping tells the segments apart only by the missing IMU edge. I did not check how the 10-keyframe window handles that (suspect, lane M).
- **Minimal fix:** for the run, report `give_ups` per sequence and treat any sequence with give-ups as exposed. Longer term, either put the seed choice behind a knob or make the kept-inertial path the only one.

### C-4, medium: frontend defaults changed against main
- **Where:** `config/mod.rs:346`, `:370-373`, `:311`.
- **Defect:** `OrbFrontendConfig::default()` now has `klt_first: Enabled(min_tracks: 80)`, `klt_kernel: FixedPointU8` (the enum's own default is `F32`), and `klt_prediction: Enabled(1, 10)`. At 950aa17 ORB ran on every frame. The test `the_flow_knobs_default_to_todays_frontend` pins the new values as "today".
- **Failure scenario:** any config that relies on defaults, such as `configs/euroc.ron`, behaves differently from main.
- **Not in the validation path:** the preset overrides all three.
- **Minimal fix:** the owner confirms these are intended, or the defaults go back to `Disabled` / `F32`.
- **Other unknobbed default changes:** the relocalization velocity rotation (`velocity_across_jump`, `mod.rs:2976`), which is a bug fix, and C-3.

### C-5, medium (knob configs only): a BA result overwrites a carried velocity with a stale one
- **Where:** `mod.rs:1021-1025`, `:1064-1067`, against `:1748-1772`.
- **Defect:** when the reference is not the last keyframe (an anchor, `local_map: OrbSlam3`, `while_lost`, `relocalize_when_lost`), the refresh keeps the propagated velocity. But `apply_local_mapping_results` still sets `velocity_world = correction.velocity_world`, the last keyframe's velocity at t_kf. Nothing refreshes it afterwards.
- **Failure scenario:** under `pr180-full` with anchors, a merge that lands 1 s into a coast at 3 m/s² injects about 3 m/s of velocity error, which stays until the next tracked keyframe.
- **Not reachable in `pr180-window10`:** there the reference is always the last keyframe.
- **Minimal fix:** in the decoupled case apply the velocity delta, or re-propagate from the corrected keyframe with `propagate_keyframe_velocity` over `[t_kf, t_prev]`.

### C-6, low: a `todo!()` is reachable from a RON file
- **Where:** `mod.rs:276-278`.
- **Defect:** `frontend.keyframe_orb: Mapping` passes `validate()` (`validation.rs:404-407` only needs `klt_first` on) and then `SlamSystem::build` panics.
- **Minimal fix:** return `ConfigError::FrontendKnobNeeds`.

### C-7, low: T-3 edge cases
- **Where:** `mod.rs:2177-2181`, `:1240-1254`, `:1199-1203`.
- **Defect 1:** on the give-up frame of a kept re-seed, `last_frame_timestamp_sec` is already the frame's own time. The coast window is empty, `predict_pose` falls to the visual model, and the frame is reported as `Bootstrap` although its pose is the coasted seed. One frame per give-up is not written.
- **Defect 2:** `Reseeding` frames skip the IMU-confidence check that T-4 applies to coasts. Only the runaway and starved holds apply.
- **Minimal fix:** treat the give-up frame as carried when the seed is the coasted pose; pass `imu_confident` into the re-seed classification.

### C-8, low: validation gaps (G12 class)
- **Where:** `validation.rs:239-260`, `:168-203`.
- **Defects:**
  - `tracking.frame_velocity: PropagatedFromKeyframe` is accepted without an IMU and with `pose_inertial` on, where it is inert (`mod.rs:1781-1787`).
  - `mapping.keyframe_seed: ImuPropagated` is accepted without an IMU.
  - `reseed.max_bridge_sec` has no upper bound. Above `MAX_BUFFERED_SEC` = 30 s (`inertial/mod.rs:29`) the bridge edge is integrated from a truncated buffer with no error.
- **Minimal fix:** three more rejections.

### C-9, low: card U gaps
- **Where:** `estimator.rs:504-508`, `mod.rs:1782`, `tests.rs` module `anchors`.
- **Defect 1:** card U says the `LastKeyframe` mode does not have the stale velocity. But `prediction_origin` returns the previous frame for `LastKeyframe`, and the U knob is skipped when `pose_inertial != Off`. A frame without a valid prior (rejected solve, after a relocalization, window over 1 s) still copies the keyframe velocity.
- **Defect 2:** the U unit tests use zero gravity, identity rotations and no `T_bc`, so a sign or frame error could not fail them. Risk is low because the path reuses `predict_pose_imu`.
- **Checked correct:** the math is v_kf + g·Δt + R_kf·Δv with the keyframe's bias, over `[t_kf, t_prev]`. It falls back when `t_prev <= t_kf`, when the reference is not the last keyframe, and before IMU init. The knob-off path is the old code.

### C-10, low: allocations and copies
- `mod.rs:455-456`: with a map bound and session recording, `map.keyframes().to_vec()` clones every keyframe on each insert once the map is at the bound. The app sets no bound, so the validation run is not affected.
- `mod.rs:2391`: `seeded.clone()` on every flow frame.
- `mod.rs:1793`: under U, a second `preintegrate_window` per frame, which allocates a sample `Vec` that grows with the frames since the keyframe.

### C-11, low, suspect: an accepted flow pose is thrown away when a keyframe is due
- **Where:** `mod.rs:1951-1977`, `:2017-2035`.
- **Defect:** `FlowOnly::KeyframeDue(pose)` keeps only the pose as a seed. If the ORB estimate then rejects, the frame becomes a coast, although flow had a pose with enough inliers.
- **Why it may matter:** G33 says a keyframe is due on about 76 % of frames. R_03 frame 395 (`matched=25 LowPnpInliers(4)`) could be this case; not proven.
- **Minimal fix:** keep the flow estimate and fall back to it as `Tracked`, without a keyframe.

### Project-rule notes
- `tracking_step` is about 340 lines (`mod.rs:1878-2217`), `insert_keyframe` about 180, `track_flow_only` about 145.
- `unwrap()`/`expect()` in `mod.rs` library code went from 28 to 42, almost all `map.lock().unwrap()`. Newer code uses the poison-tolerant form, so the file is inconsistent.
- No `unsafe` in the lane.
- `wheel.rs` hand-rolls 3×3 and 4×4 products that kornia-algebra has.

## Earlier findings: do the fixes hold?

- **Hold in lane C files:** E2, E4, E5, E7, E9, G2, G7, G12 (three knobs), T3, T5, T7, T9 (system part).
- **E1 and G3 hold only behind knobs.** The refresh is decoupled only with `local_map: OrbSlam3`, `while_lost`, `relocalize_when_lost` or an anchor. The default still copies a stale reference's velocity and bias, pinned by the test `without_the_knobs_the_reference_state_is_copied`. Not reachable in `pr180-window10`: no loop closer means no relocalization, and there are no anchors.
- **D1's validation stop-gap is absent.** That is fine: D1 was fixed in mapping (`boundary_under_calibration`).
- **Async hand-off checked:** the worker sends its merge result while holding the publication gate (`mapping/local_mapping.rs:540-553`), so a frame cannot track against a merged map and then be corrected twice. Results that arrive during bootstrap are drained and dropped; the map merge itself has already happened.

## Knob classification (lane C)

| Knob | Class | Note |
|---|---|---|
| `tracking.frame_velocity` | bug fix | ORB-SLAM3 `PredictStateIMU`; make it the only path once K4 is settled with V1 |
| `mapping.keyframe_seed: ImuPropagated` | bug fix, redundant | covered by `frame_velocity`; remove |
| `tracking.refine_monotone` | bug fix | only path |
| `local_ba.merge: PropagateToNewer` | bug fix | only path for async mapping |
| `local_ba.dense_parallelism: Sequential` | bug fix (speed) | bit-identical; only path |
| `inertial.sample_interpolation: Midpoint` | bug fix | only path |
| `mapping.growth_min_baseline_ratio` 0.0175 | reference constant | tan(1°); derive from the triangulation `min_parallax_deg`, one source |
| `local_ba.initial_lambda`, `max_iterations`, `large_window.*` | reference constants | Basalt 1e-4 and 7; one budget for every job, drop `min_tracked_inliers` (card V) |
| `local_ba.active_keyframes` | real tunable | compute budget; could follow the measured solve time |
| `reseed.keep_inertial` | design choice | its runaway and tilt guards should be unconditional (C-1) |
| `reseed.max_coast_speed_mps` 4 | real tunable | platform speed; unify with `state_envelope.max_speed_mps` |
| `reseed.runaway_window_sec` 10 | reference constant | visloc |
| `reseed.bridge_covariance_scale` 100 | real tunable | derive: propagate the bias marginal through the bridge instead of a scale |
| `reseed.max_bridge_sec` 10 | real tunable | derive: stop when the bridge position sigma passes a bound |
| `reseed.min_seed_points`, `min_cold_start_points` | real tunable | derive: information of the seed (point count weighted by depth sigma and spread) |
| `reseed.max_tilt_deg` 10 | real tunable | derive: chi-square on the gravity direction from accel noise over the averaging window |
| `reseed.young_map_grace_sec` | real tunable | derive from the covariance trust |
| `reseed.relocalize_when_lost`, `own_window_keyframes` | feature switches | ORB-SLAM3 Atlas behaviour |
| `relocalization.max_jump_m` 0.5 | real tunable | derive: k·σ of the predicted pose covariance |
| `relocalization.while_lost`, `.solver` | feature switches | |
| `tracking_loss.pose_gate` limits | real tunable | derive: Mahalanobis gate on the predicted covariance |
| `tracking.imu_trust: Covariance` | reference constant (3σ) | the principled replacement for `min_imu_confidence_sec` 2 s and `timeout_imu_sec` 5 s |
| `tracking.pose_inertial`, `snapback_fix`, `local_map` | feature switches | ORB-SLAM3 ports; `pose_inertial` is off per G30 |
| `tracking_loss.keyframes_while_lost` | feature switch | |
| `inertial.viba_window`, `viba_velocity_gate` | refuted experiments (card J) | remove |
| `frontend.klt_first.min_tracks` | real tunable | derive from `track_refill.per_camera` × cameras |
| `frontend.klt_kernel`, `flow_threads`, `flow_first_core` | compute choices | lane F |
| Wheel config (`translation_noise`, `rotation_noise`, floors, `clock_sigma_s`) | real tunables | per-robot sensor model |
| Constants `BA_SHIFT_M`, `SNAPBACK_RATIO`, `MAX_FIXED_WINDOW_SEC`, `VISUAL_AFTER_RELOCALIZATION_SEC`, `TILT_WINDOW_SEC`, `MIN_RUNAWAY_SPAN_SEC` | reference constants | diagnostics or ORB-SLAM3 values |

## Coverage

| File | Hunks in diff | Hunks read | Notes |
|---|---|---|---|
| `build.rs` | 6 | 6 | full file at HEAD plus removed lines |
| `config/mod.rs` | 12 | 12 | full diff |
| `config/tests.rs` | 21 | 21 by outline, about 6 in full | not fully read: test-only; read the display, flow-default, rig `klt_first` and G12 tests, the rest by test name |
| `counters.rs` | 1 | 1 | new file, read in full |
| `estimator.rs` | 1 | 1 | new file, read in full |
| `input.rs` | 4 | 4 | changed lines read |
| `mod.rs` | 48 | 48 | whole file at HEAD (3562 lines) plus every removed line |
| `recovery.rs` | 1 | 1 | new file, read in full |
| `state.rs` | 7 | 7 | whole file at HEAD |
| `tests.rs` | 9 | 9 | all added tests read except the body of the fixture at diff lines 986-1030, skimmed |
| `trace.rs` | 1 | 1 | new file, read in full |
| `validation.rs` | 11 | 11 | whole non-test file at HEAD; its test module not read (test-only) |
| `wheel.rs` | 1 | 1, non-test part | lines 1-969 read; the test module (970-1580) only by test name, as wheels are not in the validation run |
---

## Lane D

# Lane D report: tracking, frontend, kornia-sensors, rig.rs, frame.rs (PR #180, 950aa17..70c3a45)

Read-only review. Nothing built or run; every finding is derived from code. No high-severity bug found on the `pr180-window10` path. 3 medium, 7 low.

Paths are relative to `/home/edgar/dev/split/kornia-slam-graph/`. `T/` = `crates/kornia-slam/src/tracking/`. Lines at 70c3a45.

## Findings

### D-1 — medium (mechanism certain, size suspect)
- **Where:** `T/optical_flow.rs:329-347` (`TrackSet::thin`), with `T/tracker.rs:974-992` (`settle`) and `T/flow_tracker.rs:947-975` (`plan_refill`).
- **Defect:** `thin` keeps tracks oldest first, with no regard for map association. Refill tracks without a map point outlive every ORB frame (`advanced_unassociated` then `restore`), so they out-rank the new associated tracks (age 1) that `reconcile` creates.
  - An ORB match on an existing map point never merges with a refill track sitting on the same feature. It becomes a new track, and `thin` drops it as "crowded" (under 23 px) by the older unassociated one.
  - Only `adopt` (new landmarks of the last keyframe, 3 px) can give a refill track a map point.
- **Scenario:** camera 0 holds 20 associated and 28 refill survivors, all age ≥ 2. An ORB frame matches 200 points. `thin(48, 23)` keeps the 48 old tracks and drops every new associated one as full or crowded. Associated tracks stay at 20, so `flow_frame_windowed` misses `min_tracks: 48` and ORB runs again on the next frame.
- **Evidence:** sort key `(Reverse(age), id)` at `optical_flow.rs:332`. The test `thinning_keeps_the_oldest_tracks_within_the_budget` asserts that three unassociated tracks evict three adopted landmarks.
- **Effect:** speed (G33 class) and fewer pose observations on flow frames. No direct mis-score.
- **Fix:**
  1. Change the sort key to `(associated first, Reverse(age), id)`.
  2. In `reconcile_in_cameras` on WindowedU8, try `associate_nearest(pixel, camera, 3 px)` for a match with no retained track before making a new one.

### D-2 — medium, speed only (certain when the reference keyframe has over 160 points)
- **Where:** `T/policy.rs:178-183`, fed by `system/mod.rs:2584-2625` and `:2445-2456`.
- **Defect:** this is the mechanism of G33. On a flow frame `tracked_inliers` is capped at 2 × 48 = 96 by `track_refill`, but it is compared with `n_ref_map_points * 0.6` of the uncapped reference keyframe.
- **Scenario:** with `n_ref` ≥ 161 the threshold is ≥ 96. Every flow frame past `min_frames_between` returns `KeyframeDue`, and ORB runs. This matches the ledger's 357 "keyframe due" against 161 keyframes.
- **Fix:** on flow frames, test the fraction of associated tracks that survive since the last ORB frame. Alternatively, scale `n_ref` by the tracks followed at that frame.

### D-3 — medium, knob-gated (`pose_inertial` not Off, with WindowedU8 and AnalyticGn)
- **Where:** `T/tracker.rs:629` (flow-frame descriptors are `[0; 32]`), with `system/mod.rs:1980-1986` and the Skipped branch at `:2105-2200`.
- **Defect:** `flow_frames_lock_free` assumes a lock-free flow frame "cannot become a keyframe". If `refine_pose_inertial` rejects a flow-tracked estimate (`system/estimator.rs:408-411`), the frame becomes `Skipped` while `frame` is still the flow frame. That frame has at most 96 keypoints, zero descriptors and no stereo points.
- **Scenario:** the Skipped branch then passes it to three places:
  - `relocalize(&frame)`;
  - `insert_keyframe(&frame, …, Lost or Anchor)`, giving a keyframe whose descriptors all match each other at Hamming distance 0;
  - `bootstrap_step(frame)` on give-up.
- **Reach:** not in `pr180-window10` or `pr180-full` (both have `pose_inertial: Off`).
- **Fix:** when a flow-tracked estimate is rejected after the fact, resolve the deferred source (run ORB) before the Skipped branch. Alternatively, apply `refine_pose_inertial` inside `track_flow_only` and return `Refused`.

### D-4 — low (error path, silent wrong association)
- **Where:** `T/flow_tracker.rs:808-852`, with `system/mod.rs:774-786`.
- **Defect:** a failed `prepare` sets `frames = None` and only logs. The track set keeps pixels of frame k-1. The next `prepare` makes `previous` and `current` both frame k+1.
- **Scenario:** templates are recut at stale pixels in the wrong image. Flow is zero, the round trip passes (identical images), and map points get bound to the wrong patches. The same happens when `process_prepared` returns `Err` after `prepare` (`self.resolve(source)?`).
- **Fix:** store in `TrackSet` the frame id its pixels refer to. In `RigFlow::step`, clear the tracks when it differs from `previous.frame_id()`. Also call `reset_tracks()` on a prepare error.

### D-5 — low (project rule: allocations in the per-frame hot path)
- **Where:** `T/tracker.rs:424` (`output.survivors.clone()`), `:589-594` and `:596-606` (HashMap plus 6 Vecs in `flow_frame_windowed`), `:826-842` (HashSet, HashMap and Vec in `accept_flow_only`), `:681-682` (two Vecs in `estimate_flow_analytic`).
- **Defect:** `tests/flow_zero_alloc.rs` covers only `RigFlow::step`. The tracking-core half of a flow frame makes about 12 allocations per frame.
- **Fix:** keep scratch buffers on `Tracker`. `Track` already holds its `map_point_idx` and position, so index tracks by a sorted id instead of a HashMap.

### D-6 — low to medium, theory gap (suspect on score)
- **Where:** `crates/kornia-slam/src/frontend/overlap.rs:33` (`MIN_PARALLAX_COS = cos 0.25°`).
- **Defect:** ORB-SLAM3 rejects fisheye-stereo triangulations with `cosParallaxRays > 0.9998`, which is 1.15° (`KannalaBrandt8::TriangulateMatches`). Here points pass down to 0.25°, about 25 m on the 11 cm baseline.
- **Derivation:** ray noise is about 1 px / f. At f ≈ 240 px (the value in the lane's Aria test fixtures; the pinhole export's f was not checked) that is 0.24°. Relative depth error is about √2 · 0.24° / parallax, so over 100 % at the gate and about 30 % at 1.15°. The 2 px reprojection gate cannot reject these points, because any depth along the ray passes at low parallax.
- **Scenario:** these points enter `rig.stereo_points` as metric depth. They feed cold-start seeding (`min_cold_start_points: 20`) and scale. This is a candidate contributor to G35 (scale 1.08-1.12); it is not tested.
- **Fix:** gate on the predicted depth sigma, `σ_z / z = z·σ_px / (f·b)`, under a bound. Alternatively, use the reference 0.9998 and give the far points bearing-only use.

### D-7 — low
- **Where:** `T/pose_estimation/rig_pnp.rs:79-98` and `:108-142`, as used by `T/tracker.rs:677-680`.
- **Defect:** each outlier round restarts from `pose_init`, and rejected LM steps consume iterations. With `AnalyticGn(max_iter: 3)` the final pose is at most 3 LM steps from the IMU prediction, and round 1's work is discarded.
  - If all 3 steps are rejected, `refine` returns the start pose as `Some`. The prediction then counts as a visual solve when ≥ 30 points fall within 3 px of it. This is the case the comment at `pnp.rs:204-206` warns about.
- **Fix:** start round n+1 from round n's pose, and return `None` when no step was accepted.
- **Also:** `config.robust` is ignored (always Huber), and the mono path `refine_mono` runs no outlier round.

### D-8 — low (the G10 fix is incomplete)
- **Where:** `crates/kornia-sensors/src/rig_calibration.rs:129-153`, against `:118-121`.
- **Defect:** `validate` checks each camera's `primary_from_camera` for finiteness, rigidity and reflection. `camera_to_body` is applied unchecked.
- **Scenario:** a rounded, mirrored or NaN `T_BC` in a calibration file goes straight into every IMU residual.
- **Fix:** run the same four checks on `camera_to_body`.

### D-9 — low, knob-gated (`inertial.sample_interpolation: Midpoint`)
- **Where:** `crates/kornia-sensors/src/imu.rs:465-482`.
- **Defect:** past the last sample, `reading_at` extrapolates along the last two samples with no bound. ORB-SLAM3 extrapolates within one sample period only.
- **Scenario:** an IMU gap of 0.5 s at 200 Hz gives f ≈ 100. The sample-to-sample noise difference is amplified 100 times at `t1`, and the trapezoid integrates half of it.
- **Fix:** hold the last sample when `t - b.timestamp` exceeds one sample period.
- **Also:** this path makes 2 Vec allocations per preintegration.

### D-10 — low, latent
- **Where:** `crates/kornia-slam/src/frame.rs:161-172`.
- **Defect:** the uncached fallback of `undistorted_xy` undistorts a secondary-camera feature with the primary model. `ensure_undistorted` (`:138-156`) uses the feature's own view.
- **Reach:** no effect on LaMAria, where the pinhole exports have k = 0 and the cache is filled.
- **Same class:**
  - `prepare_rig` matches the overlap on raw keypoints for pinhole rigs (`frontend/mod.rs:289-299`);
  - `RigView::unproject_raw` and `overlap_box` ignore pinhole distortion (`rig.rs:86-98`, `:243-258`).
  
  All three are wrong on a distorted pinhole rig such as EuRoC radtan.
- **Fix:** `self.secondary_view(i).map_or(camera, |v| &v.camera)` in the fallback, and fill the undistortion cache before `match_overlap` for every rig.

## Fixes from the 2026-11-18 review: do they hold

- **Hold:** T1 (source pixel in the template cache, `NO_TEMPLATE` on a round-trip reject), T2 a and b, T4 (Huber 5.991, 2 rounds, inlier-only matches), T5 (`gyro_rotation` from candidate and `pose_before`), T6 (`thin`), T8, T9 (`keep_keyframe` only for non-anchor keyframes; adoption on the next frame), T11, G5 (axis and `pixel_scale` in `RayMatch`), G9 (point-based tests), G10 for cameras only (see D-8), E12 (`minus_jacobian`), E14.
- **T3:** refreshes are present at `system/mod.rs:1699`, `:3180` and `:3208`.
- **T6 side effect:** D-1.

## Checked and found correct

- **Fisheye624:** projection against the projectaria formula, the tangential and thin-prism 2×2 Jacobian, the radial derivative, `ds/dr` and `ds/dz`, and the magnification `(sec²θ/θ_d', tanθ/θ_d)`.
- **Overlap:** `F = K1⁻ᵀ [t]× R K0⁻¹`, the parallax rays, the one-to-one claim.
- **Rig PnP:** the left-perturbation Jacobian and the sigma scaling.
- **Pose-inertial:** the prior Jacobian, the Huber weights and the marginal.
- **Flow pool:** workers drop their frame Arcs before the send; the join and Drop order are sound.
- **Retry pass:** bookkeeping of `spawned`, `survivors` and `fb_rejects` across the two passes.
- **Frame ids:** `prepare`, `keep_keyframe` and `adopt` agree.
- **`grow_keyframe`:** runs synchronously, so adoption on the next frame sees the new landmarks.
- **No `unsafe`** and no new `unwrap`/`expect`/`panic!` outside tests in the lane.

## Knob-off equivalence (default path)

| Change | Default behaviour | Verdict |
|---|---|---|
| `pnp.rs:293` LM swapped to `refine_pose_analytic` for Huber and L2 | same objective, not bit-identical | always on; intended (review-swaps plan). Confirm the owner accepts it without a knob |
| `motion.rs:53,82` `rigid()` on predictions | re-orthonormalises; changes last bits | always on; a fix |
| `parallax.rs`, consistency filter, `solve_pnp` primary range, `snap`/`reconcile` on `camera_range(0)`, relocalisation camera filter | no-ops without a rig (main has no rig frames) | equivalent |
| `keypoint_grid.rs` `col_min`/`row_min` clamp | pinhole query centres are inside the image, so `col_min ≤ n_cols-1` already | equivalent |
| KLT `track` to `track_cached`, `track_impl` split | bit-identical by tests | equivalent, plus one image clone and one memcmp per frame |
| stereo right image on the `secondary` detector | same settings | equivalent |
| `LocalMapRules::Kornia`, `KeyframeRules::Kornia`, `PoseGate::Disabled`, `PrimaryEpnp`, `AllMatches`, `Hold`, `Unit`, `Uniform` | old code paths | equivalent |

## Verdict for the validation run

Lane D does not block the run. No bug found that would crash or mis-score the 6 challenge sequences under `pr180-window10`. D-1 and D-2 cost speed, not accuracy, because the fallback is ORB. D-6 is the only item with a plausible score effect (scale), and it is unproven.

Top 3 fixes:
1. **D-2 with D-1:** the flow-frame keyframe ratio and association-first thinning. Together they decide whether the 1 ms path runs at all (G33).
2. **D-6:** stereo-point parallax or depth-sigma gate. Run it as a slam-experiment with a prediction on G35 scale.
3. **D-3 and D-4:** descriptor-less or stale-track frames reaching keyframe insertion or flow. Fix both before any preset turns `pose_inertial` on.

## Knob classification

| Knob or constant | Class | Note / derivation |
|---|---|---|
| `tracking.refine_monotone` | bug fix | make it the only path (G11) |
| `adopt_landmarks` | bug fix | only path on WindowedU8 (T2) |
| `gyro_guess: RotatedRay` | bug fix / feature | only path with an IMU |
| `klt_rig: AllCameras` | feature | only path for rigs |
| `flow_bookkeeping: KeyframesOnly` | reference (ORB-SLAM3 `Tracking.cc:3357`) | only path |
| `keyframe_observations: PnpInliers` | reference (ORB-SLAM3 nulls outliers) | only path |
| `visual_information: OctaveScaled` | reference | only path |
| `inertial.sample_interpolation: Midpoint` | bug fix (half-sample lag) | only path after D-9 |
| `relocalization.solver` | `RigAp3p` is a fix for rigs; AP3P against EPnP is a reference choice | pick by rig type, no knob |
| `fisheye_weight: Magnification` | derived from the lens model | only path for raw views |
| `klt_kernel` | implementation | WindowedU8 only once validated |
| `klt_first(min_tracks)` | tunable | derive: accept the flow pose when its GN covariance (H⁻¹) is under a bound, not on a count |
| `klt_window`, `klt_prediction(level, iter)`, `klt_retry` | tunable | derive the level per track from the predicted-pixel sigma (IMU covariance over depth): `level = ceil(log2(3σ_px / (win/2)))`; the retry then goes away |
| `forward_backward(0.6 px, 4, 1)` | reference-like (VINS 0.5 px) | derive from LK noise: 3·√2·σ_LK |
| `track_refill(per_camera, spacing, refill_below, border)` | tunable | spacing = k·√(W·H/N); border = win/2 + sub-pixel half window (9 conflicts with the 10 px margin today) |
| `flow_pose: AnalyticGn(max_iter)` | tunable | stop on the step tolerance, not a count |
| `pose_gate` limits (5°, 6 m/s, 10 m/s) | tunable | innovation chi² from the preintegration and PnP covariances (OpenVINS) |
| `keyframes.rules: OrbSlam3{max_frames}` | reference constant | derive from the frame rate in the timestamps |
| `local_map: OrbSlam3` | reference choice | — |
| `keyframes_while_lost` | design choice | `Anchor` is the safe one |
| `flow_threads`, `flow_first_core`, `stereo_pyramid`, secondary keypoints | deployment / compute | not accuracy |
| `MIN_PARALLAX_COS` (0.25°) | constant off the reference (1.15°) | derive from σ_z/z (D-6) |
| `EPIPOLAR_BAND_PX` 2, `MAX_REPROJECTION_PX` 2, Huber 5.991 | reference (chi²) | — |
| `MAX_HAMMING` 100, `NN_RATIO` 0.9, `ADOPTION_RADIUS_PX` 3, `REFILL_QUALITY` 0.01, `SurvivorFilter` (20, 60, 10), `RAW_MAX_INCIDENCE_DEG` 80 | constants | the 60 px movement cap could come from the prediction sigma |

## Coverage

| File | Hunks | Read | Notes |
|---|---|---|---|
| crates/kornia-sensors/src/camera_fisheye624.rs | 1 | 1 | new file; code in full, tests by name and assert |
| crates/kornia-sensors/src/imu.rs | 6 | 6 | |
| crates/kornia-sensors/src/lib.rs | 1 | 1 | |
| crates/kornia-sensors/src/rig_calibration.rs | 1 | 1 | code in full, tests by name and assert |
| crates/kornia-sensors/src/rig.rs | 5 | 5 | |
| crates/kornia-slam/src/frame.rs | 7 | 7 | |
| crates/kornia-slam/src/frontend/mod.rs | 9 | 9 | |
| crates/kornia-slam/src/frontend/overlap.rs | 1 | 1 | code in full; tests (lines 300-485) by name and assert |
| crates/kornia-slam/src/frontend/tests.rs | 1 | 1 | test file; names and asserts only |
| crates/kornia-slam/src/rig.rs | 1 | 1 | full |
| T/flow_pool.rs | 1 | 1 | full |
| T/flow_tracker.rs | 1 | 1 | full, tests included |
| T/local_map.rs | 6 | 6 | tests by name and assert |
| T/mod.rs | 1 | 1 | |
| T/motion.rs | 5 | 5 | |
| T/optical_flow.rs | 22 | 22 | last 3 test hunks by name and assert |
| T/parallax.rs | 1 | 1 | |
| T/policy.rs | 9 | 9 | |
| T/pose_estimation/map_projection/keypoint_grid.rs | 2 | 2 | |
| T/pose_estimation/map_projection/mod.rs | 31 | 31 | |
| T/pose_estimation/mod.rs | 2 | 2 | |
| T/pose_estimation/pnp.rs | 15 | 15 | |
| T/pose_estimation/pose_inertial.rs | 1 | 1 | code in full; tests (lines 520-638) by name and assert |
| T/pose_estimation/rig_pnp.rs | 1 | 1 | lines 1-290 in full; the test tail 290-448 was not read |
| T/pose_gate.rs | 1 | 1 | code in full; tests past line 150 not read |
| T/relocalization.rs | 11 | 11 | new tests by name and assert |
| T/timings.rs | 1 | 1 | full |
| T/tracker.rs | 8 | 8 | full, tests included |

Read outside the lane, to judge hunks: `system/mod.rs` flow call sites (700-810, 1005-1165, 1878-2520, 3060-3215, 3390-3500), `system/estimator.rs:348-429`, and the header of `upstream/kornia_3d/refine_analytic.rs`.

Not checked: `lk_window.rs` internals (coarse-level template coordinates, whether `rebuild` reuses coarse levels), `wide` semantics, and any dataset behaviour.
---

## Lane E

# Lane E review: loop closure + upstream/ (PR #180, 950aa17..70c3a45)

Read-only. No builds, no runs. All line numbers at 70c3a45. Paths relative to `/home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/` unless absolute.

## Verdict for the validation run

**No blocker in lane E.** `configs/pr180-window10.ron` sets no `loop_closing`, and `LoopClosingMode` defaults to `Disabled` (`system/config/mod.rs:667-670`). So nothing under `loop_closure/` runs in the validation run.

Lane E code that does run: `upstream/kornia_imgproc/{lk_window,pyrdown_u8,good_features,corner_subpix,lanes}.rs`, `upstream/kornia_3d/refine_analytic.rs` (flow pose + PnP refine, `refine_monotone: true`), `pose_ext.rs`, `cast.rs`.

Checked by derivation on that code, found correct:
- **LK (`lk_window.rs`)**: every SIMD read is in-plane under `interior` (`:215`) and `window_inside` (`:505`); border path uses reflect-101 for any index. i32 lane sums cannot overflow before flush (`TENSOR_FLUSH` 64, `MISMATCH_FLUSH` 32). Status, epsilon, oscillation half-step, min-eigen and error formulas match OpenCV `LKTrackerInvoker`. NaN start gives `corner() == None`, status false. Bit-tested against kornia-rs level 0 and level 1.
- **pyrdown**: u16 sums max 65408; kept lanes never read past `w + 3`.
- **good_features**: all Sobel/block reads stay inside the image (regions clipped to `SUPPORT = 3`); tensor sums exact in f32 (max 9.4e6 < 2^24).
- **corner_subpix**: matches OpenCV update, stop and "moved more than half_win" rules; patch reads in bounds.
- **refine_analytic**: 2x6 Jacobian `[Rᵀrow | p × Rᵀrow]` matches the right retraction; NaN input ends in `None`, never a NaN pose; monotone rule holds (G11 fix holds).
- No `unsafe` in the lane. No `unwrap/expect/panic` in new library code (the two `expect`s at `loop_closure/closer.rs:563,568` predate the PR).

Prior-review fixes that hold: G4 (`confidence: 1.0`; kornia treats 1.0 as "no adaptive stop", `kornia-3d/src/ransac/config.rs:79`), G5 (image-plane residual with `axis`), G6 (gravity-twist-only rotation block on robust loop edges, `graph/mod.rs:917-923`), G8 (recentring, `graph/mod.rs:852-861`), G1 (jump = loop edge, `closer.rs:1307-1327`), G11. G13 still open as documented (unreachable).

**Top 3 fixes (none gate the validation run; all matter the day loop closing is turned on for LaMAria):**
1. E-1: make `LoopOdometryGate` default `Disabled` (owner rule), or record it as an approved exception.
2. E-2: default snapshot graph stops closing loops for the rest of the run once any loop keyframe is culled. Make `persistent` the only path.
3. E-3: `sync_graph` is O(N·E) per solve on the tracking thread. One-pass refresh.

## Findings

### E-1 | medium | `loop_closure/odometry_gate.rs:61-68`, used at `closer.rs:579`
**Defect:** `LoopOdometryGate::default()` is `Enabled { drift: DriftModel::default(), k_sigma: 3.0 }`. `LoopClosingConfig` derives `Default`, so every `loop_closing: Enabled(...)` config now rejects loops whose correction exceeds 3σ of a drift model fitted on one 61 m sim run (1.9 %/m + 0.10 m; 0.026°/m + 1°). At base there was no gate. Only the app CLI turns it off (`apps/kornia-slam-app/src/main.rs:171`).
**Scenario:** `configs/pr180-full.ron` and `configs/euroc-stereo-imu-loop.ron` enable loops. On a drifting LaMAria sequence (ledger G18: 3_18 error 627 m at scale 1.285) the true loop's correction is far outside 3σ, so the loops that would fix the worst drift are refused; knob-off runs differ from main.
**Evidence:** the equivalence tests themselves must disable it: `tests.rs:599`, `:1125`, and `default_closer_matches_optimize_pose_graph_through_a_run_with_culling` ("With the gate off and the default configuration…").
**Fix:** `#[default] Disabled` on `LoopOdometryGate`; enable it in the presets that want it.

### E-2 | medium | pre-existing, now pinned by a test | `closer.rs:1170-1172` (was `pose_graph.rs` `MissingLoopKeyframe`)
**Defect:** default graph (`persistent: false`) sends every verified loop to every solve and refuses the whole solve if any earlier loop's keyframe left the map. `verified_loops` is never pruned (`forget_keyframes`, `closer.rs:239-244`, erases only the BoW entry and the gap).
**Scenario:** long run with keyframe culling. Loop (2, 22) accepted, keyframe 2 culled. Every later loop returns `PgoFailed: loop edge references a keyframe outside the map snapshot`, `applied: false`, for the rest of the run. A map reset (`clear_active`) does the same.
**Evidence:** `tests.rs` `default_closer_matches_optimize_pose_graph_through_a_run_with_culling` asserts `[(true,true),(true,true),(true,true),(false,false)]`, "the last loop needs keyframe 2, culled after the third".
**Fix:** make `persistent: true` the only path (it is a bug fix, see knob table). Minimal alternative: in `forget_keyframes`, drop verified loops whose end was removed.

### E-3 | medium (time) | `closer.rs:1293-1295` with `graph/mod.rs:371-401`
**Defect:** `sync_graph` calls `graph.refresh_odometry(kf, pose)` once per keyframe, and each call walks all edges (`self.edges.values_mut().filter(...)`, a `BTreeMap`). Cost is N·E per solve; the old `optimize_pose_graph` was O(N). The job also clones the whole graph and gap map each solve (`closer.rs:690-691`).
**Scenario:** default `Synchronous` execution, 10 000 keyframes: about 1e8 edge visits on the tracking thread per accepted loop (estimate: 0.5-1 s, not measured), before the solve starts.
**Fix:** set all node `odometry`/`pose` first, then re-measure odometry edges in one pass over the edges. The result is identical (each edge ends measured from both refreshed ends).

### E-4 | medium, suspect | `closer.rs:239-244`, `:1307-1327`
**Defect:** gap information is lost on culling, which re-opens G1.
- Gap keyframe k culled: `forget_keyframes` removes `gaps[k]`. In the snapshot graph k+1 then chains to k-1 by an ordinary odometry edge (σ = 1 in `ScalarWeights`, `sqrt(walk·path)` in `Drift`) across the teleport or the unmeasured re-seed gap.
- `found_kf` culled: `gap_link` returns `None` (`node_of_keyframe(found_kf)?`, `in_map.get(&found_kf)?`), so the jump becomes an odometry edge. This is the documented deviation, but in a persistent graph the node still exists and could be used.
**Scenario:** relocalisation into an old segment; the relocalised keyframe is highly redundant with the old segment and is a likely culling target. A later true loop across the jump is then opposed by a false odometry edge.
**Evidence:** code path only; no run.
**Fix:** when a gap keyframe is forgotten, move its gap to the next keyframe in map order (sigmas combined in quadrature); for a persistent graph resolve `found_kf` through the graph's forgotten node and its stored pose.

### E-5 | low-medium, suspect | `closer.rs:1256-1273`, `graph/mod.rs:815-817`
**Defect:** the persistent graph never learns of a world transform. `sync_graph` detects a new world only by "no shared keyframe". Inertial alignment/refinement (`mapping/map/ops/correction.rs:614`, epoch +2) moves and rescales every live keyframe, but forgotten nodes stay frozen in the old world, and their edges to live neighbours are re-measured across two worlds.
**Scenario:** `persistent: true`; a loop is closed and a keyframe culled before the last inertial refinement (scale 0.99 at 20 m from origin gives a 0.2 m false odometry measurement at σ ≈ 0.085 m).
**Fix:** apply the same similarity to graph nodes and edge translations when the map commits a world transform, or reset the graph then.

### E-6 | low | `closer.rs:971-974`
**Defect:** a finished background solve is thrown away when `context.current_keyframe_idx` is `None` or not in the map at that frame boundary. Its loops report `Accepted { applied: false }` and are retried only when another loop is accepted. The world-change path re-queues (`:899-904`); this path does not.
**Scenario:** `Asynchronous` execution, tracking lost on the frame the solve ends.
**Fix:** keep the outcome and retry the write-back at the next boundary with a reference, or re-queue as the discard path does.

### E-7 | low | `fusion_plan.rs:215-229` against `fusion.rs:107-109`
**Defect:** the synchronous fusion skips source features of non-primary cameras; the planned (background) search does not. A landmark first met through a side-camera feature is kept with that feature, so `Link` ops are produced that the synchronous path never makes, and its `Merge` always fails the `reprojects_onto` check (keypoint is `None`). The doc says the plan is "`fuse_loop_direction` on the copy".
**Scenario:** rig + `Asynchronous`/`Deferred`: fusion differs from the synchronous closure on the same input.
**Fix:** skip `keyframe.keypoints[feature].is_none()` sources in `Search::direction`, before `seen.insert`.

### E-8 | low | unknobbed default-path changes (likely intended; record them)
- **PnP refine solver**: `tracking/pose_estimation/pnp.rs:293` routes Identity/Huber (the default kernel, `pnp.rs:59`) to `refine_pose_analytic`; main called kornia's `refine_pose_lm`. Introduced by 634d96f as a perf change (45.8 to 0.2 ms), "same objective", not bit-identical to main. No knob.
- **Rigid conversions**: `upstream/kornia_3d/pose_ext.rs:20-34` now normalises the quaternion and re-orthonormalises (28fe10a). The PGO golden was re-captured: 22 SE(3) iterations where b1be250 took 9. The test is still named `optimize_pose_graph_output_is_bit_identical_to_b1be250` (`loop_closure/tests.rs:785`); b1be250 is not an ancestor of the merge base, and the fingerprint is no longer b1be250's.
- **Primary-camera-only fusion and verification**: `fusion.rs:107-109,150-152`, `verification.rs:107-110`.
- **Unusable solve now emits `PgoFailed`** (`closer.rs:789-792`); it was silent before. Events only.
**Fix:** list these as approved exceptions in the PR text, and rename the golden test.

### E-9 | low | `upstream/kornia_3d/refine_analytic.rs:195-199`, `:32`
- Two `Vec` allocations per call (`points`, `pixels`) in the per-frame flow-pose and PnP path. CLAUDE.md asks for none; `tests/flow_zero_alloc.rs` covers the flow tracker only. Fix: convert inside the loops, or take a caller scratch.
- A non-finite Huber scale silently becomes the Identity (L2) kernel. Fix: return `None` so the caller sees it.

### E-10 | low, suspect | `upstream/kornia_imgproc/good_features.rs:106-116`
**Defect:** the quality threshold is `quality × max response over the empty cells only` (documented), and that max includes the 1 px grown ring. When the only empty cells are low-texture, the threshold collapses and noise corners are accepted.
**Scenario:** refill (`per_camera: 48`, validation preset) with the remaining empty cells on sky or a wall: weak tracks spend the 48-track budget.
**Fix:** add an absolute floor derived from image noise (see knob table), or keep the previous call's image-wide max.

### E-11 | low | `loop_closure/graph/mod.rs:671`
`loop_candidates` builds its gate from `min_sigma_m + drift_m_per_m × path` and ignores the random-walk terms the edges use (`:726-733`). With a pure random-walk config the gate is 3 × min_sigma and never opens. Unreachable from the closer (`loop_max_candidates: 0`); it is public API. Fix: use the same sigma as `odometry_since_anchor`.

### E-12 | low, theory | `closer.rs:1391-1411`, `odometry_gate.rs:117-135`
The gate's `path_m` is the whole keyframe path between candidate and query in map order. It includes re-seed and relocalisation jumps, and it ignores earlier corrections, so σ only grows: on the third lap the gate allows two laps of drift although the last closure removed it. Two different drift models also exist for one quantity (`DriftModel` linear, `OdometryRandomWalk` square-root). Fix: one test, Mahalanobis distance of the loop residual against the chained odometry-edge covariance of the graph path since the last accepted loop.

### E-13 | low, theory | `loop_closure/sparse_pgo.rs:18`, `upstream/kornia_3d/pgo/edge.rs:194-219`
PGO is f32 with central-difference Jacobians (ε 1e-3) and stops on a relative cost change at f32 summation noise (the code says so, `edge.rs:163-164`). Iteration count is therefore noise-driven (9 against 22 on the same lap, E-8). References (g2o in ORB-SLAM3, Ceres in VINS-Fusion) use f64 and analytic or auto-diff Jacobians. Fix belongs upstream: f64 residual with the SE(3) adjoint Jacobian.

Other theory gaps against references, for the record:
- No covisibility edges (ORB-SLAM3 essential graph); the chain plus loops is the VINS-Fusion 4-DoF design, which is consistent with the gravity manifold.
- No global BA after a loop (ledger A7).
- Loop and odometry information are constants, not estimator covariances (knob table).

## What should go upstream to kornia-rs

| here | upstream target | note |
|---|---|---|
| `upstream/kornia_algebra/cast.rs` | `kornia_algebra` `From` impls f32/f64 for Mat3/Vec3 | trivial |
| `upstream/kornia_3d/pose_ext.rs` | `Pose3d::camera_centre()`, `Pose3d::orthonormalized()`, `From<&Pose3d> for SE3F32` and back | upstream `Pose3d::inverse` is a transpose; the renormalisation invariant belongs there |
| `upstream/kornia_3d/refine_analytic.rs` | `kornia_3d::pnp::refine`: replace the numeric f32 Jacobian of `refine_pose_lm` | also fix the uphill-step rule (G11) upstream as the only path |
| `upstream/kornia_3d/gpnp.rs` | `kornia_3d::pnp::gpnp` | upstream RANSAC bug to file: `GenericCap` assumes the sampler draws from all `n` (G4); a sampler should report its population |
| `upstream/kornia_3d/pgo/*` and `loop_closure/sparse_pgo.rs` | `kornia_3d::pgo` | `sparse_pgo.rs` (1955 lines) has no kornia-slam types but lives outside `upstream/`; upstream `pose_graph_optimize` already exists and this is a second solver. Merge them: sparse LLT, `InformationEdge`, Cauchy, manifolds |
| `upstream/kornia_imgproc/lk_window.rs` | `kornia_imgproc::optical_flow_pyr_lk` | window-only variant; bit-equal to upstream per level |
| `upstream/kornia_imgproc/pyrdown_u8.rs` | `kornia_imgproc::pyramid` | upstream `pyrdown_u8` is SIMD on x86_64 only; this one should replace it |
| `good_features.rs`, `corner_subpix.rs` | `kornia_imgproc::features` | no upstream equivalent at rev 4ada5a7 |
| `lanes.rs`, `opencv_fixture.rs` | `kornia_imgproc::simd`, test data | |

Local worktree `/home/edgar/dev/split/kornia-rs-frontend-simd` (branch `feat/frontend-simd`) holds none of these kernels yet, so nothing is duplicated there.

## Knob classification

| knob / constant | class | note |
|---|---|---|
| `loop_closing.graph.persistent` | bug fix, should be the only path | E-2 |
| `tracking.refine_monotone` | bug fix, should be the only path | and upstream |
| primary-camera-only fusion/verification; rigid `pose_to_se3` | bug fix, already the only path | unknobbed, E-8 |
| `loop_closing.execution` (Synchronous / Deferred / Asynchronous / FollowMapping) | real tunable (deployment) | `Deferred` is a test aid. The code says the background path accepts fewer loops "for reasons not yet explained" (`system/config/mod.rs:688-690`); open defect, not a tuning matter |
| `graph.information: ScalarWeights` | legacy | `w·I` has no noise model; retire once `Drift` is validated |
| `OdometryRandomWalk` (0.0145 m²/m, 2.7e-6 rad²/m, floors) | tunable, per-deployment today | fitted to EuRoC MH worst case. Derive from the VI-BA marginal covariance of the relative pose between consecutive keyframes |
| `loop_sigma_m` 0.10, `loop_sigma_rad` 1° | tunable | derive from the verification PnP: `(JᵀWJ)⁻¹` at the inliers (depth and inlier count enter by themselves) |
| `loop_cauchy_scale` 3.0 | reference constant | 3σ in whitened units |
| `odometry_gate` `DriftModel` (4 values), `k_sigma` 3 | tunable, per-deployment today | fitted on a 61 m sim run. Replace by a χ² test of the loop residual against the chained edge covariance (E-12); then only the χ² quantile remains |
| `temporal_consistency` (min_detections, 0.3 m, 3°) | tunable | tolerances are "three verification floors"; derive from the loop sigma above |
| `MAX_DISCARDS` 3 | reference constant | |
| `VIEWED_NEIGHBOURS` 10, `VIEWED_MIN_COVISIBILITY` 15 | should not be separate | must equal tracking's local-map selection; read them from that config |
| `VIEWED_AGREEMENT_M` 0.01 | tunable | derive from the loop sigma |
| `PoseGraphConfig::node_spacing`, `loop_gate_sigmas`, `loop_max_yaw_rad`, `loop_min_path_m` | tunable, unused by the closer | library API only |
| PGO `NUM_JACOBIAN_EPS` 1e-3, LM λ factors | reference constant | disappears with analytic f64 Jacobians (E-13) |
| gpnp `threshold_px`, `max_iterations`, `confidence: 1.0` | tunable / bug fix | threshold from feature σ (χ² 2-dof); the fixed 300 draws could be a prefix-only adaptive cap |
| LK `W_BITS`, `FLT_SCALE`, `OSCILLATION_TOL`, `MIN_EIGEN_THRESHOLD` 1e-4 | reference constants (OpenCV) | |
| `klt_window` 11, `max_iter` 8, `epsilon` | tunable | window from expected residual flow after the gyro/pose prediction (prediction σ in px) |
| good-features `quality` | tunable | replace by a noise floor: min-eigenvalue of the 3×3-block Sobel tensor under pixel noise σ_n scales with σ_n² |
| good-features `min_spacing_px`, `per_camera`, `border_px`, `half_resolution` | tunable | spacing follows from image area / track budget; border from LK half-window + pyramid level |
| corner_subpix `half_win`, `max_iter`, `epsilon`; `MAX_HALF_WIN`, `MAX_ITERS` | reference constants (OpenCV) | |
| refine `LAMBDA_*`, `STEP_TOLERANCE` | reference constants | mirror kornia LM |

## Coverage

| file | hunks | read | notes |
|---|---|---|---|
| loop_closure/closer.rs | 11 | 11 | whole diff; old acceptance loop read at base for equivalence |
| loop_closure/fusion_plan.rs | 1 | 1 | all 604 lines |
| loop_closure/fusion.rs | 3 | 3 | |
| loop_closure/graph/gate.rs | 1 | 1 | |
| loop_closure/graph/mod.rs | 1 | 1 | all 1016 lines |
| loop_closure/graph/tests.rs | 1 | 1 (outline) | test names and helpers only; bodies not read line by line |
| loop_closure/graph/verify.rs | 1 | 1 | |
| loop_closure/mod.rs | 1 | 1 | |
| loop_closure/odometry_gate.rs | 1 | 1 | library part in full; tests by name |
| loop_closure/pose_graph.rs | 2 | 2 | |
| loop_closure/sparse_pgo.rs | 19 | 19 | mostly moves and generics; new test module read |
| loop_closure/tests.rs | 6 | 6 (hunk 6 partly) | 1371 added test lines: doc comments and names of all, bodies of the golden, scalar-information and default-closer equivalence tests; the rest not line by line |
| loop_closure/verification.rs | 1 | 1 | |
| loop_closure/worker.rs | 1 | 1 | tests skimmed |
| upstream/kornia_3d/gpnp.rs | 1 | 1 | library in full; tests partly |
| upstream/kornia_3d/mod.rs | 1 | 1 | |
| upstream/kornia_3d/pgo/edge.rs | 1 | 1 | |
| upstream/kornia_3d/pgo/manifold.rs | 1 | 1 | |
| upstream/kornia_3d/pgo/mod.rs | 1 | 1 | |
| upstream/kornia_3d/pose_ext.rs | 1 | 1 | |
| upstream/kornia_3d/refine_analytic.rs | 1 | 1 | library (1-250) in full; tests 250-748 by name plus the oracle module |
| upstream/kornia_algebra/cast.rs | 1 | 1 | |
| upstream/kornia_algebra/mod.rs | 1 | 1 | |
| upstream/kornia_imgproc/corner_subpix.rs | 1 | 1 | library in full; tests by name |
| upstream/kornia_imgproc/good_features.rs | 1 | 1 | library in full; tests by name |
| upstream/kornia_imgproc/lanes.rs | 1 | 1 | |
| upstream/kornia_imgproc/lk_window.rs | 1 | 1 | library (1-650) in full; tests by name and assertions |
| upstream/kornia_imgproc/mod.rs | 1 | 1 | |
| upstream/kornia_imgproc/opencv_fixture.rs | 1 | 1 (first 30 of 63 lines) | test-only fixture reader; parser body not read |
| upstream/kornia_imgproc/pyrdown_u8.rs | 1 | 1 | library in full; tests by name |
| upstream/mod.rs | 1 | 1 | |

Not checked: kornia-rs internals beyond `ransac/config.rs` confidence handling; `Map::apply_graph_correction` and the system-side callers of `note_keyframe_gap` / `forget_keyframes` / `on_frame` (other lanes); any dataset behaviour. E-3 timing and E-4/E-5 scenarios are derived from code, not observed.
---

## Lane F

# Lane F review: apps, configs, manifests, benches, scoring scripts (950aa17..70c3a45)

Read-only. No builds, no runs. Every lane file opened; only Cargo.lock was read as a summary (see coverage).

`ate.py` is correct: Umeyama SE3 and Sim3 math, reflection guard, scale, RMSE, ns-vs-ns stamps and nearest-stamp association all check out. The biases are in what `score.sh` hides, in `make_config.sh`'s silent fallback, and in `velgt.py` / `steps.py` on sparse ground truth.

## Findings

### F-1, medium (high if the offline-BA trajectory is the one scored)
**File:** `apps/kornia-slam-app/src/main.rs:488-505`

- **Defect:** the session and smoothed frames are not gated like `--trajectory-out`, although the comment says they are. `smoothed.record` also needs `system.current_keyframe_idx()` and `keyframe_pose()` to be `Some`.
- **Evidence:** `SystemState::reset()` sets `current_keyframe_idx = None` (`system/state.rs:113`). So every `SkipKind::Reseeding` frame is written to `traj.txt` under `--trajectory-include-coasted` but never recorded. The same happens to any frame whose reference keyframe was culled.
- **Second drop:** `SessionRecorder::write` (`session.rs:65-73`) and `reanchor_frames` (`global_vi_ba/mod.rs:287`) drop frames whose parent chain has no dumped keyframe.
- **Failure:** the offline-BA trajectory has fewer poses than the online one, with holes at the re-seed stalls. The only signal is one stderr line, "N frame(s) without a dumped keyframe".
- **Reach on the validation preset:** `pr180-window10.ron` has `keep_inertial` off, so `Reseeding` cannot occur there. The culled-reference path and the chain drop still apply.
- **Fix:** record unanchored frames with their causal pose, and have `offline-ba` write them (interpolated or causal) so the pose count equals `poses_written`. Fail loudly on a mismatch.

### F-2, medium, validation run
**File:** `experiments/quick-eval/make_config.sh:11-14,23`

- **Defect:** silent fallback to the raw IMU. The preset's `factory_calib:` line is always deleted. It is re-added only if `$LAMARIA_FACTORY/$seq.factory.json` exists.
- **Evidence:** the local factory directory holds R_11_5cp and sequence_1_19, 2_11, 3_18 and 5_11. It has no sequence_4_11, R_01 or R_03. The data directory does hold sequence_4_11.
- **Failure:** a validation config for sequence_4_11 runs unrectified with no error. Only the source summary line shows it ("IMU raw").
- **Fix:** exit non-zero when `LAMARIA_FACTORY` is set and the file is missing, or require `ALLOW_RAW_IMU=1`. Have `score.sh` print the "IMU raw/rectified" token.

### F-3, medium
**File:** `experiments/quick-eval/score.sh:8-11`, `ate.py:8,18`

- **Defect:** `score.sh` strips `poses=`, `tracked=`, `path=` and `ate_sim3=` from `ate.py`'s output. ATE covers only associated poses.
- **Evidence:** R_11's pseudo-dense ground truth has 154 rows inside the first 800 frames (1627 rows for 9547 frames, gaps 0.2-0.6 s). An R_11 "800-frame" ATE therefore rests on at most 154 poses.
- **Failure 1:** a run that loses track, or omits coasted frames, is scored on fewer and easier poses. Its ATE is not comparable with a run that tracked everything.
- **Failure 2:** `2>/dev/null` plus the `[0-9.]+` grep turns a crash, NaN or empty trajectory into a blank field. The `exit N` that `run.sh` appends to `stdout.txt` is never read, so a partial trajectory from a crashed run is scored as normal.
- **Fix:** print `poses=` and `poses_written`, print the exit code, and do not discard stderr.

### F-4, medium for cards U, J and K; not on the validation path
**File:** `experiments/quick-eval/velgt.py:45-46`, `steps.py:17-29`

- **Defect:** both scripts assume dense 20 Hz ground truth.
- **velgt:** the central difference `pg[gi+1] - pg[gi-1]` has no limit on `dt`. On R_11 it spans 0.3-1.2 s, and on `sequence_*` 0.8 s (ground truth at 2.5 Hz). The "truth" is a smoothed velocity, which inflates the error and the age slope.
- **Padded starts:** R_01 and R_03 begin with 16 identical ground-truth poses, so `vgt = 0` there. `ate.py` also scores against those 16 rows.
- **steps:** `FROM..TO` (50..250) indexes associated poses, not frames. On R_11 the slice ends silently at about 153. Runs that drop different frames compare different time spans.
- **steps, stationary bias:** a sum of step norms is biased high by jitter when the device is still.
- **Fix:** skip rows with `tg[gi+1] - tg[gi-1]` above about 0.11 s and drop repeated ground-truth rows. In `steps.py`, select by timestamp or frame index and print the pose count used.

### F-5, medium, validity of all card numbers
**File:** `experiments/quick-eval/run.sh:7`, `main.rs:356`

- **Defect:** scores depend on machine timing. Runs use `--no-pace --debug`, `KORNIA_SLAM_DEBUG=1`, `JOBS=3` and `execution: Asynchronous`.
- **Evidence:** frames arrive as fast as the frontend allows, far faster than the 20 Hz sensor. The mapping worker then gets less wall time per frame than in deployment, and that share changes with load. Card R records load averages of 4.4-35.
- **Debug cost:** debug mode adds per-edge `eprintln` calls and the `debug_window_line` medians on the mapping thread.
- **Unproven part:** I did not check whether a back-pressure gate bounds this.
- **Related, older than this PR:** paced mode sleeps to 30 fps (`main.rs:258`), not to the timestamps, so even "paced" is 1.5x real time on LaMAria.
- **Fix:** run the validation paced by timestamps (or with deterministic mapping hand-off), without `--debug`, one job at a time. Otherwise report it as load-dependent.

### F-6, low-medium, suspect
**File:** `apps/kornia-slam-app/src/trajectory.rs:76-82,148-164`

- **Defect:** `walk_parents` can loop forever. The comment "Parents always precede their children" is false for the first recorded reference keyframe A: it has no entry until the reference returns to it.
- **Evidence:** the reference can go back through relocalization (`system/mod.rs:2983`) or `most_voted_keyframe` (`mod.rs:1997`). Then `parents[A] = B` while `parents[B] = A`.
- **Failure:** if neither A nor B is found (culled, not in the dump), shutdown hangs in `smoothed.write` or `anchored_frames`, and no session file is written.
- **Reach:** not reachable on `pr180-window10` (no relocalization, default keyframe rules). Reachable on `pr180-full`.
- **Fix:** never insert a parent for a keyframe that already appeared as a reference, or cap the walk with a visited set.

### F-7, medium, test gap
**File:** `crates/kornia-slam/tests/flow_zero_alloc.rs:98-116,200-207,227`

- **Defect:** the zero-allocation test does not cover the preset's hot path.
  - It runs `settings()` with refill, retry, gyro guess and anchor guard all off. The preset uses refill, `CoarseLevel` retry and `RotatedRay`.
  - Counting wraps only `flow.step`. `flow.prepare` (the pyramid build) is outside it.
  - Nothing asserts that survivors are non-empty, so the test passes vacuously if all tracks die during warm-up.
- **Fix:** count `prepare` plus `step` under `preset_settings` with `jumpy_shift`, and assert survivors > 0 on the last frame.

### F-8, medium, owner rule "match and beat OpenCV"
**File:** `crates/kornia-slam/benches/opencv_frontend_fixtures.py`; `src/upstream/kornia_imgproc/{lk_window.rs:896, good_features.rs:647, corner_subpix.rs:289}`

- **Defect:** the three OpenCV parity tests are `#[ignore]`, and `tests/fixtures/upstream/frontend_opencv.bin` is not in the tree. Parity of the `WindowedU8` LK, refill and sub-pixel kernels is unverified in CI. All three are on the validation preset's path.
- **Bench caveat:** `forward_48` leaves template extraction untimed. The fair arm against `opencv_prebuilt_48` is `templates_and_forward_48`.
- **Fix:** commit the fixture (about 0.7 MB) and remove the ignores.

### F-9, low
**File:** `experiments/quick-eval/window_edges.py:29-36,41-45`

- **Defect:** edge lines leak between solves. `[vi_ba] edge` is printed by every `vi_ba_schur` first iteration. `[vi_ba] WINDOW` is printed only after a successful local solve (`bundle_adjustment.rs:954`).
- **Failure:** edges from a failed solve, or from any solve with no WINDOW line, join the next window. `max(j)` and `first_new` then misclassify old, crossing and new edges.
- **Second assumption:** the script assumes the new keyframes occupy the top contiguous slots.
- **Fix:** reset `edges` on a "local inertial BA failed" line and tag edge lines with a solve id.

### F-10, low
**File:** `apps/kornia-slam-app/src/summary.rs:8-17`, `main.rs:392`

- **Defect:** `FrameTimes` grows by 16 bytes per frame forever on the default path, including live OAK-D and UVC runs. This is new default-path state with no knob.
- **Fix:** a fixed-size histogram or reservoir.

### F-11, low
**File:** `apps/kornia-slam-app/src/main.rs:164-190`

- **Defect:** `KORNIA_SLAM_LOOP_GATE` and `KORNIA_SLAM_LOOP_TEMPORAL` change the pipeline from the environment, outside the RON file. A stray variable changes a run and only one stderr line records it.
- **Fix:** move both into the run file.

### F-12, low
**File:** `apps/kornia-slam-frontend-bench/src/pipeline.rs:273-280,812-870`, `trace.rs:94-112`

- **Defect:** the bench does not time what the system runs.
  - Secondary cameras get `camera_from_primary = IDENTITY`. The comment "the trace has no extrinsics" is stale: `rig.csv` now carries them, and `trace.rs` ignores columns 12-23.
  - Secondary tracks carry no landmarks, so the `klt-rig` timing omits the secondary prediction and projection work the system does under `adopt_landmarks`.
  - `match_projection` and its gate constants are a hand copy of the library's, which can drift (CLAUDE.md reuse rule).

### F-13, low, documentation
**File:** `apps/kornia-slam-app/src/trajectory.rs:179-183`

- **Defect:** the comment says the ns round trip is exact "for recordings shorter than days". It actually depends on the absolute stamp. EuRoC epoch stamps (1.4e18 ns) are off by up to about 120 ns. LaMAria stamps (about 1e12 ns) are exact.
- **Impact:** harmless for `ate.py`. It breaks an evaluator that matches timestamps exactly.
- **Fix:** carry the source's `u64` ns through `FrameItem`.

## Checked and correct

- **`ate.py`:** estimate and ground truth are both in ns. `MAX_FRAMES` only feeds the `poses=N/n` text.
- **`trajectory.rs`:** `T_WB = T_WC · T_CB`; `between` and the parent-chain composition; quaternion order xyzw.
- **`offline-ba` output:** writes `T_bw` through a writer with no extrinsic, which gives `T_wb`.
- **Official evaluator:** uses `--corresponding_sensor imu`, which matches the body frame written.
- **Live-pose consistency:** `apply_local_mapping_results` moves the live pose with its reference keyframe, so `frame_from_kf` stays consistent.
- **`score.sh` greps:** match the `[summary]` line and the `[imu_init]` strings.
- **`velgt.py`:** the regex matches the `[vel]` format, and the 0.05 s bin is right for 20 Hz LaMAria.
- **Earlier review, finding 12** (`imu_time_offset_ns` unread): fixed, it now feeds `td_total_ms`.
- **Refactors:** the `datasets/euroc.rs` and `source/mcap.rs` changes are pure moves.
- **Manifests:** all seven kornia-rs crates sit at the same rev, 4ada5a7 (upstream #1322). `opencv` and `core_affinity` are optional.

## Verdict for the validation run

Lane F does not block the run. Scores can be mis-stated in three ways: missing factory calibration (F-2), offline-BA pose holes (F-1), and timing dependence (F-5). `pr180-window10.ron` parses consistently with the brief.

Top 3 fixes:
1. **F-2:** make `make_config.sh` fail on a missing factory calibration, and check sequence_4_11 before the run.
2. **F-1:** make the offline-BA trajectory carry every scored frame, and assert its pose count equals `poses_written`.
3. **F-3 and F-5:** have `score.sh` print `poses=` and the exit code, and run the validation without `--debug`, one job at a time, paced or with the load recorded.

## Knob table

| Knob | Class | Note |
|---|---|---|
| `--trajectory-include-coasted` | bug fix, should be the only path | a pose per frame is what the benchmark wants; keep the trusted flag in the file |
| `--trajectory-coasted`, `--trajectory-smoothed`, `--dump-map`, `--trace-out`, `--blackout`, `--session-out` | diagnostics | not tunables |
| `--no-pace` | tunable, evaluation only | replace with timestamp pacing (F-5) |
| `KORNIA_SLAM_LOOP_GATE` / `KORNIA_SLAM_LOOP_TEMPORAL` (env) | tunable, hidden | move to RON (F-11) |
| `Lamaria.factory_calib` | bug fix | rectification should be mandatory when the record exists (F-2) |
| `Lamaria.imu_time_offset_ns` | tunable | estimate td online or offline; `offline-ba` already reports `td_total_ms` |
| `Lamaria.camera`, `images` | deployment choice | none |
| `rig_calibration` | deployment input | none |
| `klt_window: 11`, `klt_prediction (0, 8)`, `klt_retry CoarseLevel(1)` | tunable | derive the level from predicted-flow uncertainty (gyro and pose covariance) against the window half-width |
| `forward_backward max_round_trip_px: 0.6` | tunable | k·σ of the LK localisation noise (from the template Hessian) |
| `klt_first min_tracks: 48`, `track_refill (48, 23 px, 0.8, 9)` | tunable | derive from image area and the inliers the pose covariance needs; spacing = sqrt(area/N); border = window/2 + margin |
| `flow_pose AnalyticGn(3)` | reference constant | iterate to convergence under a cap |
| `gyro_guess RotatedRay`, `adopt_landmarks`, `flow_bookkeeping`, `keyframe_orb` | design switches | become the only path once validated |
| `tracking.refine_monotone: true` | bug fix | never accept an uphill LM step |
| `growth_min_baseline_ratio: 0.0175` | tunable | derive from the depth σ target: baseline/depth ≥ σ_px/(f·rel_depth_err) |
| `reseed.min_cold_start_points: 20` (default 50) | tunable | derive from pose observability (information matrix rank and conditioning), not a count |
| `local_ba.active_keyframes: 10` | tunable | budget from solve time against the keyframe period |
| `large_window: Enabled(min_tracked_inliers: 0, active_keyframes: 10, max_iterations: 7, initial_lambda: 1e-4)` | misuse | a gate set to 0 with the same window size only selects iterations and λ; expose those as `local_ba.max_iterations` / `initial_lambda`. Iterations should end on convergence, and λ should come from the LM trust region |
| `dense_parallelism: Sequential` | deployment (core budget) | none |
| `merge: PropagateToNewer` | bug fix | a correction must reach the keyframes inserted during the solve |
| `ate.py` 10 ms association | reference constant | half the frame period would be the derived value |
| `velgt.py` 0.05 s bin | reference constant | read the frame period from the data |

## Coverage

| File | Hunks | Read | Notes |
|---|---|---|---|
| apps/kornia-slam-app/Cargo.toml | 1 | 1 | |
| apps/kornia-slam-app/src/blackout.rs | 1 | 1 | |
| apps/kornia-slam-app/src/config.rs | 4 | 4 | |
| apps/kornia-slam-app/src/config/tests.rs | 1 | 1 | |
| apps/kornia-slam-app/src/datasets/euroc.rs | 2 | 2 | pure move |
| apps/kornia-slam-app/src/datasets/lamaria.rs | 1 | 1 | Aria rectification direction not checked against projectaria source |
| apps/kornia-slam-app/src/datasets/mod.rs | 1 | 1 | |
| apps/kornia-slam-app/src/main.rs | 14 | 14 | |
| apps/kornia-slam-app/src/map_dump.rs | 1 | 1 | |
| apps/kornia-slam-app/src/session.rs | 1 | 1 | |
| apps/kornia-slam-app/src/source/euroc.rs | 1 | 1 | |
| apps/kornia-slam-app/src/source/hilti.rs | 1 | 1 | |
| apps/kornia-slam-app/src/source/lamaria.rs | 1 | 1 | |
| apps/kornia-slam-app/src/source/mcap.rs | 4 | 4 | pure move |
| apps/kornia-slam-app/src/source/mod.rs | 9 | 9 | |
| apps/kornia-slam-app/src/source/oakd/device.rs | 1 | 1 | |
| apps/kornia-slam-app/src/source/uvc/device.rs | 1 | 1 | |
| apps/kornia-slam-app/src/summary.rs | 1 | 1 | |
| apps/kornia-slam-app/src/trace_out.rs | 1 | 1 | |
| apps/kornia-slam-app/src/trajectory.rs | 1 | 1 | |
| apps/kornia-slam-frontend-bench/Cargo.toml | 1 | 1 | |
| apps/kornia-slam-frontend-bench/src/main.rs | 1 | 1 | |
| apps/kornia-slam-frontend-bench/src/pipeline.rs | 1 | 1 | read in full, tool code, lighter scrutiny |
| apps/kornia-slam-frontend-bench/src/trace.rs | 1 | 1 | |
| Cargo.lock | 20 | summary | package names, versions and sources only; not line by line (generated file) |
| Cargo.toml | 1 | 1 | |
| configs/lamaria-mono-imu.ron | 1 | 1 | |
| configs/lamaria-rig-imu.ron | 1 | 1 | |
| configs/pr180-frontend-1ms.ron | 1 | 1 | |
| configs/pr180-full.ron | 1 | 1 | |
| configs/pr180-window10.ron | 1 | 1 | |
| crates/kornia-slam/benches/bench_upstream_corner_subpix.rs | 1 | 1 | |
| crates/kornia-slam/benches/bench_upstream_good_features.rs | 1 | 1 | |
| crates/kornia-slam/benches/bench_upstream_lk_window.rs | 1 | 1 | |
| crates/kornia-slam/benches/opencv_frontend_fixtures.py | 1 | 1 | |
| crates/kornia-slam/Cargo.toml | 1 | 1 | |
| crates/kornia-slam/tests/flow_zero_alloc.rs | 1 | 1 | |
| experiments/quick-eval/ate.py | 1 | 1 | math derived by hand |
| experiments/quick-eval/make_config.sh | 1 | 1 | |
| experiments/quick-eval/run.sh | 1 | 1 | |
| experiments/quick-eval/score.sh | 1 | 1 | greps checked against the emit sites |
| experiments/quick-eval/steps.py | 1 | 1 | |
| experiments/quick-eval/velgt.py | 1 | 1 | |
| experiments/quick-eval/window_edges.py | 1 | 1 | |

**Outside the brief's tool list:** I used `awk`, `cut`, `sort`, `uniq`, `ls` and `find` to read LaMAria ground-truth and camera timestamps under `/home/edgar/dev/.flux-connect-scratch/bench/lamaria`, plus `git log`/`for-each-ref` in the cargo git cache for the kornia-rs rev. All read-only; nothing was written or run.