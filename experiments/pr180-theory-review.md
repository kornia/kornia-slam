**PR #180 theory-review findings (kornia-slam, branch feat/global-pose-graph at 79c04bb)**

Four read-only reviews (E estimator, D offline BA and calibration, G geometry and recovery, T frontend). Reports are verbatim; each finding gets an ID heading (E1-E14, D1-D14, G1-G13, T1-T11) in report order. Plan: `/home/edgar/.claude/plans/show-me-the-previous-cosmic-meteor.md`.

## Owner decisions (2026-11-18)

- G13: skipped, not fixed. Unreachable: only with `node_spacing: Some`, and `correction()` has no non-test caller.
- G5: the rig relocalisation residual becomes a true pixel residual. `RayMatch` gains the optical axis; the residual is on the image plane and `INFINITY` behind the camera. Test with a 4 px off-axis error.
- G1: a lost relocalisation detaches. No odometry edge across the jump; add an `EdgeKind::Loop` edge from `found.keyframe_idx` to the new keyframe with loop sigmas (coasted re-seed: sigmas from the bridge covariance; held re-seed: `sigma_m = max_coast_speed_mps * gap_sec`). Merging that segment later stays a roadmap item.
- Rest: fix all findings, each with a test that fails on today's code.

## Lane assignment

| Lane | Owns | Findings |
|---|---|---|
| S, system and estimator | `system/mod.rs`, `system/estimator.rs`, `system/recovery.rs`, `system/validation.rs`, `tracking/pose_estimation/pose_inertial.rs`, `tracking/pose_gate.rs` | E1, E2, E4, E5, E6, E7, E9, E12; G2, G3, G7, G12; T3, T5, T7, system part of T9 |
| M, mapping, VI-BA and calibration | `mapping/vi_ba_schur.rs`, `bundle_adjustment.rs`, `marginalization.rs`, `global_vi_ba/*`, `online_calibration.rs`, `calibration_states.rs`, `vi_ba_calibration_tests.rs` | E3, E8, E10, E13; D1-D14 |
| F, frontend | `tracking/flow_tracker.rs`, `flow_pool.rs`, `optical_flow.rs`, `tracker.rs`, `upstream/kornia_imgproc/*` tests, `tests/flow_zero_alloc.rs`, `configs/pr180-*.ron` | T1, T2, T4, T6, T8, T10, T11, flow-tracker part of T9 |
| G, geometry, loop closing and sensors | `upstream/kornia_3d/gpnp.rs`, `pgo/*`, `refine_analytic.rs`, `loop_closure/*`, `kornia-sensors/src/rig.rs`, `rig_calibration.rs`, `imu.rs`, `kornia-factors/src/inertial/preintegration.rs` tests, `tracking/relocalization.rs` | G1, G4, G5, G6, G8, G9, G10, G11; E11, E14 |

Cross-lane: E1, E7, G3 are one design in lane S (velocity and bias from the origin keyframe, velocity rotated on a pose jump). G1 call site is one hunk in `system/mod.rs` written by lane G. T9 splits between S (`keep_keyframe`) and F (adopt without waiting).

## Tests that catch each finding

(Lanes fill this in, one line per finding ID: test name and file.)

### Lane G

- G1: `a_lost_relocalisation_joins_by_a_loop_edge_not_odometry`, `a_reseeded_keyframe_edge_takes_the_gap_sigmas` (`crates/kornia-slam/src/loop_closure/tests.rs`); the bridge sigmas in `the_bridge_edge_spans_the_gap_with_an_inflated_covariance` (`system/tests.rs`). Both graph tests failed against a stub `sync_graph` that ignored the gaps (today's chaining). Deviation: a relocalisation whose keyframe has left the map (snapshot graph) or the graph falls back to the ordinary odometry edge, since a loop edge needs both ends.
- G4: `a_mediocre_pose_does_not_cut_the_prefix_search_short` (`crates/kornia-slam/src/upstream/kornia_3d/gpnp.rs`); found the truth in 4 of 40 seeds before the fix (predicted about 11 %), needs 24.
- G5: `an_off_axis_error_reads_in_pixels` (`crates/kornia-slam/src/upstream/kornia_3d/gpnp.rs`); read 1.98 px for the 4 px radial error at 45 deg before the fix (axis field added first, old residual kept). `a_point_behind_the_ray_has_no_residual` gained the behind-the-camera case.
- G6: `a_tilted_loop_measurement_leaves_no_residual_on_the_gravity_manifold` (`crates/kornia-slam/src/loop_closure/graph/tests.rs`); a 3-sigma roll tilt left 4.35 whitened sigmas against 2.54 untilted before the fix, 2.54 after. Deviation: the projection applies only to robust (Cauchy) loop edges, since least squares (`ScalarWeights`, the default) must stay `optimize_pose_graph`'s solve (`pose_graph_with_scalar_information_matches_optimize_pose_graph` failed when it applied to every edge). Doc of `InformationEdge` now says camera `a`.
- G8: `a_graph_far_from_the_origin_solves_as_at_the_origin` (`crates/kornia-slam/src/loop_closure/graph/tests.rs`); the same arc 2.5 km from the origin solved to centres up to 5.7 mm from the at-origin solve before the fix, needs 0.1 mm. Deviation: the recentring is in `PoseGraph::solve` for robust (Cauchy, `Drift`) graphs only, so least squares (`ScalarWeights`, the default) stays `optimize_pose_graph`'s solve bit for bit; the cancellation there predates the lanes.
- G9: `each_rig_camera_has_its_body_extrinsic` (`crates/kornia-sensors/src/rig.rs`) and `body_extrinsics_compose_through_the_primary` (`crates/kornia-sensors/src/rig_calibration.rs`, needs `--features serde`) now check a point. The implementation is right, so both pass on today's code; with the composition order swapped in `camera_to_body_of` and `rig_cameras_from_body_extrinsics` both fail (mutation run, reverted).
- G10: `transforms_that_are_not_a_rig_are_rejected` (`crates/kornia-sensors/src/rig_calibration.rs`, needs `--features serde`); the mirrored camera returned `Ok(())` before the fix. New variants `NotFinite` and `Reflection`.
- G11: fixed behind knob `tracking.refine_monotone` (`PnpConfig::refine_monotone`), off by default. `the_monotone_rule_refuses_an_uphill_step_inside_the_cost_tolerance` (`crates/kornia-slam/src/upstream/kornia_3d/refine_analytic.rs`): Huber with the gross outlier, a start whose first damped step raises the cost 80954 to 83594 (3.3 %) under a loose `cost_tolerance` of 0.05; with the knob on and today's acceptance rule it kept the uphill step, with the fix the cost does not rise. `without_the_monotone_rule_an_uphill_step_inside_the_cost_tolerance_is_kept` (same file) pins knobs-off acceptance (the uphill step is still kept, 1 % to 5 % rise). `refine_monotone_reaches_the_pnp_refine` (`crates/kornia-slam/src/system/config/tests.rs`) checks the default and the wiring. Deviation: the 3.3 % step needs a loose tolerance to be visible; the rule is the same at the default 1e-6. The generic kornia-3d fallback (kernels other than Identity and Huber) is not covered by the knob.
- E11 (T1 part): `propagated_camera_pose_spread_matches_monte_carlo_on_a_rotated_rig` (`crates/kornia-factors/src/inertial/preintegration.rs`): rotated, turning rig, `t_bc`, anisotropic start covariance; checks the translation, rotation and velocity blocks and the whitened 9x9 (cross terms carry the signs). The propagation is right, so it passes on today's code; a sign flip of the velocity-attitude Jacobian block fails it (whitened entry off by 75). An extrinsic-rotation error cannot show here: a right perturbation of `T_cw` is the same of `T_bw`, so the propagated covariance does not depend on it (checked by mutation; passes either way). Not done in lane G: the tests for `solve_pose_inertial` and `marginal_information` (`pose_inertial.rs`, lane S), `pixel_sigma` and the G29 counter (`estimator.rs`, lane S, with E2/E5) and `newest_marginal_covariance` (`vi_ba_schur.rs`, lane M, with E3), since those files belong to the other lanes.
- E14: `midpoint_extrapolates_past_the_last_sample` (`crates/kornia-sensors/src/imu.rs`); a linear gyro rate to a stamp 2.5 ms past the last sample read 0.155 rad against 0.15503 exact before the fix (3.1e-5 rad held away). `Midpoint` only, so `Hold` (the default) is unchanged. The `imu_tail_holds` counter (`system/mod.rs`, lane S) now counts extrapolated tails; its doc still says "held".

### Lane F

- T1: `a_track_rewritten_between_frames_survives_its_round_trip`, `crates/kornia-slam/src/tracking/flow_tracker.rs` (the review's numbers: flow (100.4, 80.7), keypoint (101, 80)). Card E's P15 re-statement is left to the docs stage.
- T2: `adoption_associates_the_nearest_refill_track` (fix a) and `an_orb_frame_keeps_the_refill_survivors_on_windowed_flow` (fix b), `crates/kornia-slam/src/tracking/tracker.rs`. Fixes (a) and (b) are applied, so the interim `track_refill: Off` is not.
- T4: `the_analytic_flow_pose_drops_its_outliers`, `crates/kornia-slam/src/tracking/tracker.rs` (the review's 80 inliers and 16 coherent outliers; inlier shift 1.012 px before the fix, bound 0.7 px; mono path, so `outlier_rounds: 2` is exercised only on a rig).
- T6: `an_orb_frame_keeps_the_tracks_within_the_refill_budget` and `thinning_keeps_the_oldest_tracks_within_the_budget`, `crates/kornia-slam/src/tracking/tracker.rs`. The cap reaches the tracker by `Tracker::with_track_refill` in `SlamSystem::assemble`, and `thin_tracks` runs at the end of `adopt_landmarks`: two one-line hunks in `system/mod.rs` (lane S's file) the plan did not list. Spacing is an exact distance check, not a grid. Both presets set `klt_first: Enabled(min_tracks: 48)`, half the 2 x 48 cap; the review left the value open.
- T8: `refill_occupancy_is_where_the_tracks_start`, `crates/kornia-slam/src/tracking/flow_tracker.rs` (7 degrees of yaw, about 25 px of flow at f = 200 px).
- T9 (adopt without waiting, lane F): `landmarks_are_adopted_without_waiting_for_a_mapping_result`, `crates/kornia-slam/src/system/tests.rs`. The wait sits in `SlamSystem::adopt_landmarks` (`system/mod.rs`), not in `flow_tracker.rs` as the plan's split says, so this part is a `system/mod.rs` hunk: the `mapping_result_applied` gate and field are removed. The `keep_keyframe` part stays with lane S.
- T10 (test-coverage finding; the strengthened tests pass on today's code, which is their point: they now cover the paths): `survivors_do_not_depend_on_the_thread_count` in `crates/kornia-slam/tests/flow_zero_alloc.rs` now runs refill on, `CoarseLevel(1)` and a 6 px jump every eighth frame, compares the refill corners too, and asserts retries, refills and last-frame survivors; `a_yaw_moves_the_ray_by_f_tan_yaw_against_the_turn` and `adoption_tracks_a_landmark_from_its_keyframe_into_the_previous_frame` in `crates/kornia-slam/src/tracking/flow_tracker.rs`; the round trip after a pixel rewrite is T1's test; `response_matches_a_scalar_reference` in `crates/kornia-slam/src/upstream/kornia_imgproc/good_features.rs` now checks the closed-form eigenvalue in f64. The `#[ignore]` OpenCV parity tests are unchanged (the fix does not ask for them).
- T11: `refill_spawns_no_corner_the_survivor_filter_would_drop`, `crates/kornia-slam/src/tracking/flow_tracker.rs` (before the fix, every border corner was spawned at 9.56 px; the filter variant was applied, not `border_px: 15`).

### Lane M

- E3: `newest_marginal_is_the_undamped_system_at_the_solution`, `crates/kornia-slam/src/mapping/vi_ba_schur.rs`. Red at 8f23e47: sigma 5.77e-3 m from λ 1e-2 against 6.24e-3 m from λ 1. The fix also takes the calibration `marginal_sigma` from the same undamped build, since `border_marginal_sigma` documents the same undamped system and had the same defect.
- D1: `the_boundary_prior_is_formed_at_the_calibrated_extrinsic`, `crates/kornia-slam/src/mapping/bundle_adjustment_tests.rs`. Red at 8f23e47: fallback prior gradient 42.7 against 9.5e-3 fixed, data exact at θ = 0.2°. Covers the fallback path after the solve; the carried path before it goes through the same `boundary_under_calibration`.
- E8: `a_window_as_large_as_the_map_fixes_its_oldest_keyframe`, `crates/kornia-slam/src/mapping/bundle_adjustment_tests.rs`. Red at 8f23e47: keyframe 0 moved in a 5-keyframe window of 5. The clamp acts only in the inertial solve and only when the snapshot has neither a fixed context keyframe nor an anchor; with the default window of 3 that needs a map of at most 3 keyframes after IMU init, which needs 10 (culling keeps 21), so knobs-off is unchanged.
- E10: `offline_edges_integrate_as_the_run_did`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs`. The run's mode reaches the offline solve as a new `interpolation` argument of `offline_visual_inertial_ba` (the app passes `inertial.sample_interpolation`, one line in `apps/kornia-slam-app/src/main.rs`, outside the lane table). Red with the mode ignored (the 8f23e47 behaviour, always `Hold`): start cost 1.47 under both modes; fixed, 2.1e-4 with `Midpoint`.
- D3: offline `the_clock_state_does_not_absorb_the_integration_lag`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs` (red at 8f23e47: the IMU cost over td was least at +2.45 ms on a 200 Hz stream read at the stamps; fixed, at 0); online GT1 `online_td_recovers_injected_offset`, `crates/kornia-slam/src/mapping/vi_ba_calibration_tests.rs`, now on instantaneous samples and `Hold` edges (red at 8f23e47: −5 ms came back −4.43 ms). Both solvers integrate `Midpoint` whenever the clock state is on, whatever the run's mode. GT0 moved from the clock layout to the gyro one, since the clock layout no longer re-integrates `Hold` edges as `Hold`.
- D6: `online_td_recovers_from_the_gyro_alone`, `crates/kornia-slam/src/mapping/vi_ba_calibration_tests.rs` (GT1 with translation off). Red at the old 1 Hz motion: 3 ms came back 0.93 ms. The card G motion is now 0.7 Hz (`FREQ`), so keyframes no longer sit at zero rate and zero acceleration; GT0-GT7 still pass.
- D5: `stereo_depths_do_not_hold_the_focal_at_the_factory_value`, `crates/kornia-slam/src/mapping/vi_ba_calibration_tests.rs`. Red at 8f23e47: a true +0.5 % focal came back +0.413 % with quarter-pixel stereo depths measured at the factory focal; fixed, +0.495 %. Depth rows are dropped whenever a rotation or focal state is on (`Layout::camera_states`), online and offline, gated or not; the gyro and clock states keep them. The offline filter has no separate test.
- D14: `the_yaw_gauge_row_is_its_residual_derivative`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs`. Red at 8f23e47: at a 0.4 rad tilt the row's ω_x entry was 0 against a central difference of −1.47e3.
- D8: `the_tight_stage_runs_even_after_an_early_convergence`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs`. Red at 8f23e47: a 5 px outlier, solve converged before iteration 50, `dropped_tight_outlier` 0; fixed, 1. A solve that stops at `max_iterations` before `tighten_from_iteration` still skips the stage, as the review's fix specifies (convergence and `Ok(None)` only).
- D2: `a_flipped_gravity_is_not_absorbed_by_the_accel_bias` and `a_perturbed_rotating_start_converges_to_the_truth`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs`, on a new two-axis rotating walk (`Sampling::Rotating`, interval-mean samples turned as `Hold` integrates them). Red on the old walk that never turns: gravity flipped reached 1.14e-8 against 1.13e-8 at the truth; rotating, 6.5e2 against 1.5e-8. Deviation: the review's final `|b_a| < 1e-3`, `|b_g| < 1e-5` asserts are not added, since `OfflineBaResult` does not return biases; the flipped-gravity cost is the check instead.
- D12: `the_report_adds_the_applied_clock_offset`, `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs`. New report field `td_total_ms` = `imu_time_offset_ns`·1e-6 + `td_ms`, printed next to `td_ms`; the field did not exist at 8f23e47, so the test does not build there. Card D P15's rewording is left to the docs stage.
- E13: no test; documentation only. The `LargeWindow::Enabled` doc now says this solver's λ scales diag(H), so 1e-2 is not g2o's damping and about 1e-6 is nearer it. The `configs/pr180-*.ron` value is lane F's file and is left unchanged.
- D9: `a_marginal_prior_is_never_negative`, `crates/kornia-slam/src/mapping/marginalization.rs`. Red at 8f23e47: the least value of a three-edge chained prior was −15.6. `MarginalPrior` gains `constant`, the joint cost at linearization minus ½ m₁ᵀH₁₁⁻¹m₁, carried through the chain.
- D10: `the_prior_gradient_is_its_cost_derivative`, `crates/kornia-slam/src/mapping/marginalization.rs`. Red at 8f23e47: at a 10° tangent the system gradient on ρ_x was 2.644 against a cost derivative of 2.665. `accumulate_prior` now applies `blkdiag(J_r⁻¹(d), I₉)` to the gradient and to H (kornia-algebra's `SE3F64::right_jacobian_inverse`).
- D4: `an_older_prior_is_chained_onto_the_boundary` and `a_boundary_older_than_the_prior_keeps_the_prior`, `crates/kornia-slam/src/mapping/bundle_adjustment_tests.rs`. Red at 8f23e47: a prior two keyframes behind the boundary fell back (`prior_fallback`), and a boundary older than the prior replaced it with a fresh one (`next_prior` set). The chain walks the snapshot's IMU edges between captured keyframes; a prior whose keyframe is not captured, or an edge missing between, still falls back. Under online calibration the intermediate edges keep their stored preintegration (only the boundary edge is re-integrated, D1).
- D7: not fixed. Four injected-state offline tests were written (gyro y scale 0.3 %, td 3 ms, camera rotation 0.3° about x, focal +0.5 %; one state on, one slice, on the rotating walk of D2, td on the instantaneous walk). All four fail at the lane-M head, and not for a solver defect that was found: each converges to a low cost with the state held near its prior (gyro 0.037 % for 0.3 %, final cost 2.2e-2; td 0.10 ms for 3 ms, cost 6.2e-3; rotation (0.10, −0.11, −0.01)°; focal 0.389 %). The synthetic walk does not excite these states enough to beat their priors, so a fixture with real excitation (faster rotation, longer span) is needed before these tests can pass; that is fixture design, not the review's fix. Draft tests: `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad/D7-injected-state-tests.patch`. GT3's online part is not touched.
- D11: not fixed. The re-anchoring needs a fixture where a 3 px drop splits a component. Two attempts (one and three link landmarks, 5 and 8 px off, segments unbridged) did not split: the second part's free position and yaw, and the links' depths along nearly collinear rays, absorbed the offsets (final cost 2.0e-3, nothing dropped). Draft test: `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad/D11-draft-test.patch`.
- D13: not fixed. Samples beyond a window-end edge's ends must be stored when the edge is made (`crates/kornia-slam/src/inertial/mod.rs`) or read from the IMU buffer the system holds (`system/mod.rs`); both are outside lane M's files, and the newest edge's later samples do not exist when it is made, so it needs a design choice (deferred extension or a buffer handed to the mapping job).

### Lane S

- E1: `a_most_voted_reference_leaves_the_inertial_state_to_the_last_keyframe` (`system/tests.rs`, mod `inertial_origin`); prediction part shared with E7. Deviation: with the reference not the last keyframe, the refresh keeps the propagated velocity and takes the last keyframe's bias (G3's rule, one design for E1/E7/G3) instead of copying the last keyframe's velocity; the BA write-back copy at `:1009` reads the last keyframe's correction as E1 says. Both are gated on the knobs that let the reference differ (`local_map: OrbSlam3`, `relocalization.while_lost`, `reseed.relocalize_when_lost`): a default relocalization (loop closing on, IMU not yet trusted) still copies the reference's state, as before; `without_the_knobs_the_reference_state_is_copied` pins it.
- E7: `a_keyframe_origin_prediction_starts_from_the_keyframe_velocity` (`system/tests.rs`, mod `anchors`).
- G3: `a_relocalized_reference_leaves_the_inertial_state_to_the_last_keyframe` (`system/tests.rs`, mod `inertial_origin`) and `a_pose_jump_keeps_the_camera_frame_velocity` (`system/recovery.rs`).

# E estimator

Review complete: 14 findings, three high. The core math is right; the bugs are in what feeds it and what reads it. Line numbers are at HEAD `79c04bb`, paths under `/home/edgar/dev/split/kornia-slam-graph/crates/`. Two numpy checks ran in memory with no files written, because plan mode barred creating `review-A/`.

## Findings

### E1

1. `high | certain | kornia-slam/src/system/mod.rs:1784-1786 (with :1652-1672, :1009-1010; estimator.rs:476-499)` | A8 sets `current_keyframe_idx` to the most-voted keyframe, but that index also selects the keyframe whose velocity and bias are copied to the frame. The copy then takes v(t_ref) from an older keyframe as the current velocity. With A6 `OrbSlam3`, the prediction starts at the `last_keyframe_idx` pose and time but uses the most-voted keyframe's velocity. Error is (v_ref − v_kf)·(t − t_kf): 0.5 m/s over a 2 s coast gives 1 m. A0 minus A6 copies it every frame. It bites when the newest keyframe has few points, which is A8's own motivating case. | Read IMU state from `last_keyframe_idx` (ORB-SLAM3's `mpLastKeyFrame`) in `refresh_reference_inertial_state` and in the velocity/bias copy at `:1009-1010`. Keep the pose composition on the reference. Make `prediction_origin` return the origin keyframe's velocity and bias with its pose and time.

### E2

2. `high | certain | kornia-slam/src/system/estimator.rs:312-324` | `pixel_sigma` uses λmax of the ρ block as camera-centre variance. Under `T_cw·Exp(ρ,ω)`, δc = −ρ + [c]×ω, so Σ_ρρ contains [c]×Σ_ω[c]×ᵀ and grows with distance from the world origin. Numeric check (σ_θ 2 mrad, σ_c 5 mm, z 3 m, f 458): code gives 1.20 / 2.87 / 26.06 px at |c| = 0 / 8.5 / 85 m; Monte Carlo truth is 1.20 px at all three. Trust ends early and the search scale saturates far from the origin. | Pass the predicted pose; with `c = camera_centre(pose)` and `A = [−I | hat(c)]`, use `λmax(A·Σ[0..6,0..6]·Aᵀ)` in place of `largest(0)`. The fixed formula reproduced 1.20 px at all three distances.

### E3

3. `high | certain | kornia-slam/src/mapping/vi_ba_schur.rs:2111, :2303-2305, :2627-2630` | `last_information` is not the undamped system the card specifies. `damp_point_blocks(&mut vis.c, lambda)` runs before the Schur complement, so the saved matrix is A − B(C+λ·diag C)⁻¹Bᵀ. It is also taken at the pre-step state of the last iteration. Numeric check on a metric two-view fixture: pose σ shrinks to 0.56x at λ = 1e-2, 0.43x at λ = 1, 0.39x at λ = 1e10. An unobservable direction (σ = ∞) became 7 mm at λ = 1e-3. λ starts at 1.0 and grows 10x per reject, so solves ending on the cap or on rejects are worst. Σ_k is overconfident by construction, which is KT1. | Extract steps 1-3b plus `hold_inertial_state` into one function. After the loop, when `params.newest_marginal`, call it once at the accepted `kfs/xyz/calib` with λ = 0, then `newest_marginal_covariance`. Drop the rows of any point whose undamped C is singular.

### E4

4. `medium | likely | kornia-slam/src/system/estimator.rs:357-376` | The solve has no guard after relocalisation. ORB-SLAM3 stays visual for `mnFramesToResetIMU = mMaxFrames` frames (`Tracking.cc:2973`); here `last_relocalization_frame` only feeds keyframe evidence. After a relocalisation or coast, the anchor is the pre-loss keyframe, held exact, with white-noise-only information over the whole gap. σ_p = σ_a·√(T³/3) is 1.3 cm at 5 s (EuRoC noise); card T's own arithmetic gives 0.35 m, so the anchor is 27x overconfident. | Return early with `frame_prior = None` while `frame.idx <= last_relocalization_frame + max_frames`, and clear `frame_prior` in `relocalize`. For a `Fixed` window longer than about 1 s, skip the solve or add the keyframe marginal through `propagate_state_covariance`.

### E5

5. `medium | certain | kornia-slam/src/system/estimator.rs:406-409` | When the solve keeps fewer than `min_inliers`, the visual pose stays, silently and uncounted. That is the strong snap-back case, so the G29 fix is inert where it should act. ORB-SLAM3 fails the frame instead. No `LowPnpInliers` reject can come from the solve, so kill rules KG4 and A6's last kill cannot fire. | Reject the frame as `LowPnpInliers` and coast on the prediction. At minimum add a `pose_inertial_rejected` counter and point both kill rules at it.

### E6

6. `medium | certain | kornia-slam/src/tracking/pose_estimation/pose_inertial.rs:116-120, :133-137, :141-146` | Every row is 2-dof, but metric frames use ORB-SLAM3's 3-dof stereo gates {15.6, 9.8, 7.815, 7.815} plus the 1.5x close factor, which ORB-SLAM3 applies only to mono edges. P(χ²₂ > 7.815) = 2.0 % and P(χ²₂ > 11.72) = 0.28 %, against the intended 5 % and 1.1 %. Rescue at 24 is also the 3-dof value. All EuRoC stereo and LaMAria rig configs take this path. | Use the mono tables for every row: `CHI2_KEYFRAME_MONO` for `Fixed`, `CHI2_FRAME_MONO` for `Prior`, rescue 18. Remove the stereo tables until a 3-dof right-u row exists.

### E7

7. `medium | certain (only with KeyframesWhileLost::Anchor) | kornia-slam/src/system/mod.rs:1658-1666 with estimator.rs:476-499` | With an anchor as reference, the refresh keeps the propagated velocity. Keyframe-origin prediction then re-integrates the whole [t_kf, t] window onto it every coast frame: v̂_n = v̂_{n−1} + D_n in place of v_kf + D_n. The error is Σ D_m, a runaway. It applies to `OrbSlam3` and to a pending `snapback_fix`. | Same fix as finding 1: velocity and bias come from the origin keyframe.

### E8

8. `medium | likely | kornia-slam/src/mapping/bundle_adjustment.rs:287-298` | `active_start` returns 0 when the map has no more non-anchor keyframes than the window, so no keyframe is fixed and there is no boundary edge. The 4-dof gauge is held only by damping. ORB-SLAM3 uses `Nd = min(KFs−2, maxOpt)` and always fixes one. Window 3 never hits this after IMU init (10 keyframes); window 25 hits it for about 15 solves per map. The newest marginal is then singular, giving `None` or a huge covariance and a spurious untrusted verdict. | For IMU-initialised jobs only (keeps defaults bit-identical), clamp the window to non-anchor keyframes − 1.

### E9

9. `medium | certain | kornia-slam/src/system/estimator.rs:107-125` | `ref_points_min_obs` counts distinct observer keyframes ≥ 3. ORB-SLAM3 counts 2 per stereo observation (`MapPoint.cc:162-165`), so a stereo point seen by 2 keyframes passes there and fails here. nRefMatches is undercounted on EuRoC stereo, c2 fires less, and the interval drifts to the 0.5 s of c3. | Weight an observation 2 when the keyframe is not a rig and that feature has stereo depth; compare the sum with `min_observers`.

### E10

10. `medium | certain | kornia-slam/src/mapping/global_vi_ba/imu_edges.rs:112-124` | Offline re-preintegration always calls `from_measurements` (Hold) without bracketing samples. The card requires the run's mode. Under `Midpoint`, offline edges get the half-sample lag back: 0.25-1.25 mrad per EuRoC edge. | Carry `SampleInterpolation` in the offline problem. Slice `lo−1..hi+1` and call `from_measurements_with`, as `vi_ba_schur.rs:1489-1512` does.

### E11

11. `low | certain | kornia-factors/src/inertial/preintegration.rs:341-439; pose_inertial.rs (no test module)` | T1 uses identity rotations, no `t_bc`, isotropic Σ, and checks only the position trace, so a frame or sign error cannot fail it. There is no test for `solve_pose_inertial`, `marginal_information`, `newest_marginal_covariance`, `pixel_sigma`, or the G29 counter. | Rotate T1's poses, add `t_bc` and an anisotropic Σ, and compare the rotation block. Add tests for the five untested items; findings 2 and 3 would have failed them.

### E12

12. `low | certain | pose_inertial.rs:360-369` | The prior residual uses an identity Jacobian. The exact one is blkdiag(R_meanᵀR, Jr⁻¹(r_ω), I₉), which ORB-SLAM3 uses. The error is O(|r|), about 0.05 % at 1 mrad. | Apply that Jacobian to the prior's Hessian and gradient.

### E13

13. `low | certain | bundle_adjustment.rs LocalBaParams::large(); vi_ba_schur.rs:2327-2330` | `initial_lambda: 1e-2` is not ORB-SLAM3's value in effect. g2o adds λ·I; this solver adds λ·diag(H), so 1e-2 damps every DOF by 1 %, orders of magnitude more. | Document it, or use about 1e-6 for arm A0-orb.

### E14

14. `low | suspect | kornia-sensors/src/imu.rs:464-477` | Without a sample after t1, the tail holds the last sample. ORB-SLAM3 extrapolates linearly (`Tracking.cc:1700-1712`). The error is at most about 7.5e-5 rad per edge. If the app feeds IMU only up to the frame stamp, `imu_tail_holds` is near 100 %, not ≤ 1 per 1000. | Extrapolate from the last two samples when no later one exists.

## Checked and found correct

- **Midpoint:** the trapezoid, end interpolation and dt handling match `Tracking.cc:1684-1720`. Covariance stays Σσ²·dt and the bias Jacobians stay linear in the averaged reading. U1 is a real test.
- **IMU factor:** residual, gravity sign, `T_bc` direction, state order and both Jacobians under `T_cw·Exp`, with a non-circular central-difference test.
- **Pose-inertial solve:** `plus`/`minus`, `pose_jacobian_2x6` including rig views, gradient sign, the Schur marginalisation, Huber knees (5.991, 25) on whitened χ², kernel dropped in round 4, and bias-walk information 1/(σ²dt).
- **Window:** the boundary 1e-2 on the 9×9 block only, the kernel on scaled χ², and `BoundaryOrRecentInit`.
- **Trust:** `propagate_state_covariance`, and the BA marginal is used as a covariance, not as information.
- **G29:** the counter formula and the snap-back flag lifecycle.
- **Rules:** A7 conditions c2, c3, c4 and the 0.25 s pre-init rule; the A8 and A9 list construction.
- **Outliers:** they stay in keyframe observations, which matches ORB-SLAM3 (`Tracking.cc:2259-2267`); the card's "known differences" line states the opposite.

## Could not check

- Whether the app delivers an IMU sample after each frame stamp.
- How often the most-voted keyframe differs from the last one.
- The λ distribution at the final VI-BA iteration.
- The `SE3F32::retract` convention and f32 pose precision in VI-BA (pre-existing).
- Loop-closure and epoch interplay.
- Any dataset behaviour.

# D offline BA and calibration

Review complete: 14 findings, 1 high. Read-only; no files written, no cargo. Numeric checks ran as inline numpy (plan mode blocked scratch files). All paths are under `/home/edgar/dev/split/kornia-slam-graph/`.

## Findings (most severe first)

### D1

**1. high | certain | `crates/kornia-slam/src/mapping/bundle_adjustment.rs:826` and `:914-920` (via `marginalization.rs:210-217`)**
- **Wrong:** With `boundary: Marginal` and online calibration both on, the boundary prior is built with the factory `vi_params.imu_t_bc` and the stored preintegration. The solve it feeds uses `imu_t_bc · Exp(−θ_0)` and td/gyro re-integrated edges (`vi_ba_schur.rs:2022-2023`).
- **Size:** θ_0 = 0.2° gives `g|θ|·dt = 0.034 × 0.5 = 0.017 m/s` against σ(Δv) = 5.5e-4, about 31σ. In rotation it is 1.4 mrad against 1.2e-4, about 12σ. td = 2 ms gives `|Δω|·td` = 0.6-2 mrad, 5-16σ. All at full information.
- **Card conflict:** card G line 68 says `next_prior` "is formed at the solve's θ_0". The code does not do that.
- **Fix:** In `run_local_inertial_ba_inner`, when `calibration` is `Some`, build the prior inputs from the calibration values. Use `imu_t_bc_with_offset(imu_t_bc, θ_0)` and the boundary edge re-integrated at those values (expose `CalibratedSolve::reintegrate`). Pass them to `carried_prior` with the carried values, and to the fallback `marginalize_frozen` at `:917` with the solved values. Stop-gap: `validation.rs` rejects `Marginal` together with `online_calibration.is_enabled()`.

### D2

**2. medium | certain (numeric) | `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs:14-21`, `:100`, `:179-206`**
- **Wrong:** The sinusoidal fixture fixed scale only. Roll and pitch are still an exact null space, because the body never rotates.
- **Derivation:** Tilt the world by `R_δ` about keyframe 0 and set constant `b_a = R_δᵀg − g`. Then `r_v = (g − R_δᵀg + b_a)·dt = 0`, and `r_p`, `r_R`, pixels and the yaw gauge are unchanged.
- **Numeric:** Tilts of 0.05, 0.2 and 0.5 rad give the same IMU residual as truth (4.45e-7) and 1e-13 px, with positions moved 0.012 to 0.85 m. A gravity sign flip in the solver is absorbed by `b_a = −2g` at zero cost.
- **Consequence:** The test cannot detect gravity or tilt errors. The comment "makes them observable" is false for roll and pitch.
- **Fix:** Give the fixture a two-axis rotation (non-zero gyro samples, rotated accel). Also assert final `|b_a| < 1e-3` and `|b_g| < 1e-5`.

### D3

**3. medium | certain (numeric) | `crates/kornia-slam/src/mapping/global_vi_ba/imu_edges.rs:124`; `crates/kornia-sensors/src/imu.rs:203-216`; online `vi_ba_schur.rs:1505` when the edge is `Hold`**
- **Wrong:** `Hold` is a right Riemann sum, so it integrates the signal advanced by `dt_imu/2`. The td state absorbs that lag instead of the device offset.
- **Numeric:** With instantaneous samples the best td is +0.500 ms at 1 kHz and +2.475 ms at 200 Hz. The EuRoC value is larger than the 2 ms prior sigma and near K6's 3 ms kill limit.
- **Why GT1 misses it:** Its samples are interval means (`vi_ba_calibration_tests.rs:97-113`).
- **Second defect:** The `Hold` tail uses the previous sample, so Δ jumps by `(x_k − x_{k−1})·dt` when `t1 − td` crosses a stamp (measured −8.09e-6 against −8.09e-6 predicted). The cost is a sawtooth in td.
- **Fix:** Offline, integrate with `from_measurements_with(…, SampleInterpolation::Midpoint)` on `samples[lo.saturating_sub(1)..(hi+1).min(len)]`. Online, reject `clock_offset` unless the edges are `Midpoint`, or force `Midpoint` in `reintegrate`.

### D4

**4. medium | likely | `crates/kornia-slam/src/mapping/bundle_adjustment.rs:1007-1024`, `:914-920`**
- **Wrong:** The map holds one prior. Any solve whose boundary is not the stored keyframe or its successor falls back, then overwrites the chain with `marginalize_frozen(None, …)`.
- **Triggers:** Large-window solves (`LocalBaParams::large`), and the async worker skipping a keyframe job.
- **Consequence:** A fresh prior has rank ≤ 6 (`H' = 0` without the pose pseudo-prior), so inertial history never accumulates. A large window also frees keyframes that were already marginalised.
- **Fix:** In `carried_prior`, when the stored keyframe is older than `from`, chain `marginalize_frozen` over the intermediate fixed keyframes and their snapshot IMU edges. When the solve's boundary is older than the stored keyframe, use the scaled edge and leave `update.next_prior = None` so the map keeps its prior.
- **Not checked:** whether the snapshot carries the fixed-to-fixed edges.

### D5

**5. medium | likely | `crates/kornia-slam/src/mapping/global_vi_ba/problem.rs:284-288`, `:366-373`, `:486-496`; `vi_ba_schur.rs:2084-2096`**
- **Wrong:** Stereo depth rows are measurements derived under the factory calibration, yet they stay active with zero calibration columns when focal or rotation states are on.
- **Derivation:** The true depth is `(1+s)·z_meas`. A relative rotation φ between cameras biases the depth by `z²·f·φ/(f·B)`.
- **Size:** 0.1° at z = 5 m, f = 241, B = 0.1 m is 0.42 px of disparity, or 0.44 m. Every row pulls the same way, toward the factory value.
- **Fix:** Drop the depth rows when the layout has rotation or focal columns (offline `depth = None`; online skip the block when `calibration.is_some()`). Or model `p_c.z − (1+s_0)·z_meas` with column `−z_meas/σ`, which covers focal only.
- **Not checked:** how rig depth is computed.

### D6

**6. medium | certain | `crates/kornia-slam/src/mapping/vi_ba_calibration_tests.rs:18-27`, `:64-66`, `:79-89`**
- **Wrong:** Keyframes sit at `t = k/2` with f = 1 Hz. There `α̇ = sin 2πt = 0` and `p̈ = 0`, so `∂ΔR/∂td ≈ ω(t0) − ω(t1) = 0` and `∂Δv/∂td = 0` to first order.
- **Consequence:** GT1 sees td only through `Δp ∝ v(t1) − v(t0) = ±0.63 m/s`. Card G's mechanism (Δω, "Δa about 2 m/s²") is not exercised; Δa at the edge ends is 0.
- **Fix:** Set `EDGE_SEC = 0.3` or `FREQ = 0.7`. Add a GT1 variant with translation off that recovers td from the gyro alone.

### D7

**7. medium | certain (offline), likely (GT3) | `crates/kornia-slam/src/mapping/global_vi_ba/tests.rs:100`, `:208-239`; `vi_ba_calibration_tests.rs:585-607`**
- **Wrong:** No test can fail on the gyro-intrinsics columns.
  - Offline, the gyro stream is zero, so `(I+S)·0 = 0`. The columns are identically zero and only the prior holds S.
  - GT3 is consistent with S = 0. Fixed landmarks with two depths fix the focal from vision, so the "trade" is never forced.
- **Also untested:** Offline td and the offline rotation offset are never injected. Truth equals the prior mean in the only offline calibration test.
- **Fix:** Inject each state and assert recovery with sign.
  - Gyro: divide the stream by `(1 + 0.003)` on one axis.
  - td: shift the stamps by 3 ms.
  - Rotation: generate pixels through `Exp(θ)`.
  - Focal: scale the true focal.

**Lane M (2026-11-18): not fixed.** See "Tests that catch each finding": the injected-state tests fail on weak excitation, not on a found defect.

### D8

**8. low | certain | `crates/kornia-slam/src/mapping/global_vi_ba/mod.rs:351-398`**
- **Wrong:** The loop breaks on `decrease < relative_cost_tolerance`, or on `Ok(None)`, before `tighten_from_iteration`. The 3 px stage then never runs.
- **Consequence:** The result keeps outliers of 3 to 10 px, and P6's `dropped_3px` reads 0.
- **Fix:** Keep a `tightened` flag. On convergence or `Ok(None)` while it is false, run `drop_outliers(tight_outlier_px)`, recompute the cost, reset λ and continue. Break only once it is true.

### D9

**9. low | certain | `crates/kornia-slam/src/mapping/marginalization.rs:136-142`; `vi_ba_schur.rs:2549-2553`**
- **Wrong:** The prior cost `½dᵀHd + bᵀd` drops its constant. Its minimum is `−½bᵀH⁺b < 0`.
- **Consequence:** The relative-decrease test is skewed. If the total is ≤ 1e-12, `rel = 0` and the solve reports converged after one accepted step.
- **Fix:** Add `constant: f64` to `MarginalPrior`: the joint cost at linearisation minus `½ m₁ᵀH₁₁⁻¹m₁`, which is ≥ 0. Add it in `cost()`.

### D10

**10. low | certain | `crates/kornia-slam/src/mapping/marginalization.rs:172-191`**
- **Wrong:** `accumulate_prior` uses a unit Jacobian. The exact gradient in the solver's step is `J_r⁻ᵀ(d)(b + Hd)`, with `J_r⁻¹ ≈ I + ½ad(d)` on the pose block.
- **Size:** At the 10° stale limit the gradient is off by about 9 %, so it disagrees with the cost used for acceptance.
- **Fix:** Apply the SE(3) `J_r⁻¹(d)` to the pose block of the gradient and of H. Or tighten the stale limits to 0.05 m and 2°.

### D11

**11. low | likely | `crates/kornia-slam/src/mapping/global_vi_ba/mod.rs:236-239`, `:353-361`; `problem.rs:521-556`**
- **Wrong:** Gauge anchors are computed once, after the 10 px drop. A 3 px drop that splits a component leaves the new part with no position or yaw anchor. Only LM damping holds its 4 free directions.
- **Fix:** Recompute `observers()` and `gauge_anchors()` after any tight drop with `dropped > 0`. Anchor new roots at their current pose.

**Lane M (2026-11-18): not fixed.** No fixture yet in which a 3 px drop splits a component; see "Tests that catch each finding".

### D12

**12. low | certain | `crates/kornia-slam/src/mapping/global_vi_ba/report.rs:91-92`; `session.rs:61-62`; card D P15**
- **Wrong:** The reported td is the residual on top of the offset the dump already applied. P15 compares it with "the dump's applied offset".
- **Sign:** td has the same sign as `imu_time_offset_ns`, so the new offset is `applied + td·1e9`. It is opposite to VINS-Fusion.
- **Also:** `SessionRig::imu_time_offset_ns` is documented "for the report" but never read.
- **Fix:** Print `td_total_ms = applied_ns·1e-6 + td_ms` next to `td_ms`. Reword P15 to "|td| ≤ 2 ms".

### D13

**13. low | likely | `crates/kornia-slam/src/mapping/vi_ba_schur.rs:1488-1497`**
- **Wrong:** Once td is carried, every solve holds the last sample over `|td|` past the stream on a window-end edge. For td < 0 that is the newest edge.
- **Size:** The error is `ω̇·td²/2`. At 5 ms and 10 rad/s² it is 1.25e-4 rad, about 1σ(ΔR), and it repeats every solve. Card G bounds this only for |td| ≤ 2 ms.
- **Fix:** Store raw samples out to `±3·sigma_ms` beyond each edge end, or read the IMU buffer for the window.

**Lane M (2026-11-18): not fixed.** Needs samples past the edge ends from outside lane M's files; see "Tests that catch each finding".

### D14

**14. low | certain | `crates/kornia-slam/src/mapping/global_vi_ba/problem.rs:572-583`**
- **Wrong:** The yaw gauge row uses `∂(up·φ)/∂ω = up`. The exact row is `J_r⁻ᵀ(φ)·up`.
- **Size:** Roll and pitch are free, so a cold re-seeded segment can tilt by degrees. The row error is about `½|φ_tilt|`.
- **Fix:** `let up = SO3F64::right_jacobian(phi).inverse().transpose() * up;` before filling the row.

## Checked and found correct

- **Visual Jacobians, offline.** Pose under `T_bw·Exp([ρ;ω])` (ρ part `R_cwᵀa`, ω part `p_w × R_cwᵀa`), rotation offset `J_l(θ)ᵀ(p_c × a)`, focal `fx·x/z`. Numeric errors 3e-8, 1e-8, 7e-9 on values near 260.
- **Gauge position rows.** `−ρ + [t_wb]×ω`, numeric error 8e-11.
- **Online calibrated view.** Same algebra for secondary cameras. D1 sign: `T_bw = imu_t_bc·Exp(−θ)·T_cw` matches offline's `Exp(θ)·T_cb·T_bw`, and the camera centre is unchanged.
- **Numeric columns.** The central differences for gyro, td and primary rotation use the same residual function, state and bias as the cost, so step and sign are consistent.
- **Clock sign.** `[t0−td, t1−td]` matches the loader adding the offset to IMU stamps. Both ends shift. Visual rows correctly have no td term.
- **Offline reduced system.**
  - `W = J_xᵀJ_l`, and the rhs is `b_x − WC⁻¹b_l`.
  - Each block pair is subtracted once.
  - The border is `E −= WC⁻¹W_c`.
  - The second Schur complement and both back-substitutions have the right signs.
  - Damping goes on C and on the pre-elimination diagonal, before elimination.
  - The slot arithmetic for mixed 15×15 and 6×6 blocks is right.
- **Online border.** Assembly, point elimination, IMU columns, priors and back-substitution are correct.
- **Robust weights.** √w is applied to J and r for vision. For IMU the weight scales the 9×9 navigation block of Ω in both H and g. The costs use ½ρ.
- **LM.** There is no gain ratio in either solver. Acceptance is strict cost decrease, which is valid.
- **Prior weights.** All are 1/σ with the right units (deg to rad, ms to s).
- **Marginalisation.** The Schur complement onto `next`, the sign of `b`, `m −= b + Hd`, and `⊖` matching the solver's retraction are correct. The prior keeps its first-estimate point. The edge is counted once.
- **Gate.** A gated solve adds no column and no prior row, uses the carried values as constants, and returns the state unchanged.

## Not checked

- Any dataset behaviour.
- faer's symbolic and AMD internals.
- `PreintegratedImu` covariance propagation (it predates these commits).
- `vi_ba_prior_tests.rs` and the app-side session recorder.
- Whether GT2's two-axis motion gives the claimed 0.01° data sigma; both angles share one α(t), so the axis is nearly fixed.

# G geometry and recovery

No high-severity bug found: the always-on review fixes (scope 3) are mathematically equivalent to what they replaced. I found 13 issues, four of them medium, all in knob-gated code. Nothing was built or run; two numpy checks ran from stdin with no files written.

## Findings

### G1

1. `medium | likely | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/loop_closure/graph/mod.rs:651-675 (via closer.rs sync_graph, around :1233) and system/mod.rs:1152-1158 | A relocalisation "whatever the jump" into an older segment becomes a false odometry edge.` After `resume_after_lost_relocalization`, the next keyframe is chained to the previous keyframe in map order, which is the last keyframe of the given-up segment. The edge measurement is `T_new * T_prev^-1` from map poses, so it equals the teleport, which is the accumulated drift D. Its sigma is ordinary odometry: `sqrt(walk*path)` in `Drift`, 1 in `ScalarWeights`. Nothing measured that relative pose. A later true loop between the given-up segment and the older one has residual D and is opposed by the path (teleport edge + consistent new loops). In `Drift` mode the Cauchy kernel then down-weights the true loop. The same holds for a held re-seed (`held_at`): the edge states zero translation after seconds of motion. `| Fix:` carry a per-keyframe gap flag into `sync_graph`.
   - Teleport: add no odometry edge. Add an `EdgeKind::Loop` edge from the node of `found.keyframe_idx` to the new keyframe, measurement `T_new * T_found^-1`, with loop sigmas.
   - Coasted re-seed: keep the edge, but take `sigma_m`/`sigma_rad` from the position and rotation blocks of the bridge preintegration covariance (`publish_bridge`, already inflated).
   - Held re-seed: `sigma_m = max_coast_speed_mps * gap_sec`.

### G2

2. `medium | likely | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/system/mod.rs:2414-2431 (update_seeded_window), :1127-1131, and system/recovery.rs:88-99 | The own window blocks the older segment that was just relocalised into.`
   - Lost path: `resume_after_lost_relocalization` does not change `segments`. On the next frame `latest_start()` is still the given-up segment's start. If that segment holds fewer than `own_window_keyframes`, the window is `Some(start)`, and `filter_to_window` drops every point observed only by keyframes `< start`, which is the whole older segment.
   - `while_lost` path: `relocalize` runs inside `tracking_step` with the window already set. `estimate_pose` (map_projection/mod.rs:305, :607) refines against the filtered local map, so an older-segment candidate cannot reach `min_inliers = 50`.
   - `Segments::of` also tags keyframes created after such a relocalisation with the given-up segment's id.
   
   `| Fix:`
   - Add `window_segment_start: Option<usize>` on `SlamSystem`. Set it in `finish_bootstrap` when `segments.latest() > 0`. Clear it in `relocalize()` on adoption and when `held >= own`. `update_seeded_window` reads this field.
   - In `tracking_step`, call `self.tracker.set_window_start(None)` before the `while_lost` relocalise, as `bootstrap_step` already does.
   - Store `(start_kf, SegmentId)` pairs in `Segments`. On a lost relocalisation into segment `s`, push `(frame.idx, s)`.

### G3

3. `medium | likely | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/system/mod.rs:1667-1670 (refresh_reference_inertial_state) with :2646-2652 (relocalize) | After relocalisation the inertial state is replaced by a stale keyframe's.` `relocalize` sets `current_keyframe_idx = found.keyframe_idx`. On the next frame `frame_state_carried` is false, so the refresh copies that keyframe's `velocity_world` and `imu_bias`, which can be minutes old. The IMU prediction is then off by `|v_now - v_kf| * dt` per frame (3 m/s × 0.05 s = 0.15 m, about 15 px at 4 m with fx = 400) until a new keyframe becomes the reference. With `while_lost` or `relocalize_when_lost` the IMU is trusted, so that prediction narrows the search and feeds the pose gate. Also, the pose jump does not rotate the world velocity. Camera-frame velocity must be invariant: `R_new v_new = R_old v_old`. `| Fix:`
   - In `relocalize()` before writing the pose: `let world = found.estimate.pose.inverse().compose(&self.state.pose_world_to_cam); self.state.velocity_world = world.rotation * self.state.velocity_world;` (same rule as closer.rs:727).
   - In `refresh_reference_inertial_state`, copy velocity and bias only when `Some(kf_idx) == self.state.last_keyframe_idx`. Otherwise keep the propagated velocity and take the bias from `last_keyframe_idx` when present.

### G4

4. `medium | certain (formula checked numerically) | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/upstream/kornia_3d/gpnp.rs:191-204 | The adaptive stop uses the wrong population.` kornia's `GenericCap` (driver.rs:172) computes the all-inlier probability as `C(support,3)/C(n,3)` with `n` and `support` over all cameras. `PrefixSampler` draws only from the hypothesis camera. Counter-example: prefix 40 matches with 8 inliers, 160 other matches. A mediocre pose with 100/200 support caps the loop at 53 draws, and the true pose at 8. The prefix needs 1216 draws for 0.999, and the chance of a clean sample in 52 draws is 0.26. This bites whenever the prefix inlier ratio is below the overall ratio. `| Fix:` set `confidence: 1.0` in the `RansacConfig` of `ransac_lifted_ap3p`, which disables adaptive stopping (300 draws × ≤4 poses × N residuals is cheap). Alternative: an own cap from prefix-only inliers, `C(k_prefix,3)/C(prefix,3)`.

### G5

5. `low | certain (numeric) | gpnp.rs:102-111 (doc at :16) | ray_residual_sq is angular, not px².` `fx * tan(angle)` under-reports pixel error by cos²θ radially and cosθ tangentially. A 4 px error reads 3.19 px at u_n = 0.5, 1.99 px at u_n = 1.0, and 1.31 px at u_n = 1.43. The 4 px threshold is therefore 8 px at 45° off axis, and `RigAp3p` inlier counts are not comparable with `PrimaryAp3p`. The test `on_one_camera_the_rig_solver_agrees_with_primary_ap3p` (relocalization.rs:792) uses noise-free data and cannot see this. `| Fix:` add `axis: Vec3F64` (optical axis in the rig frame) to `RayMatch`. Residual becomes `(pixel_scale * |p/(p·axis) - d/(d·axis)|)²`, `INFINITY` when `p·axis <= 0`. If angular is intended, rename the threshold and fix the doc. Add a test with a 4 px off-axis error.

### G6

6. `low | certain (derivation) | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/upstream/kornia_3d/pgo/edge.rs:11-12, :154-158 and loop_closure/graph/mod.rs:806-829 | On the gravity manifold a loop edge carries an irreducible roll/pitch residual that the Cauchy weight sees.` The retraction keeps `R·g` for every node, so `T_b T_a^-1` always maps `g_a = R_a ĝ` to `g_b`. If the measurement does not, with angle ε, then `|ω| >= ε` for every reachable state. The whitened square gains `(ε/σ_rad)²` and the weight drops to `1/(1 + s/c²)` permanently. Example: ε = 3σ, c = 3 gives weight ≤ 0.5. Also, the residual `log(M⁻¹ T_b T_a⁻¹)` is a twist in camera a's frame, not camera b's as documented; this only matters for anisotropic information. `| Fix:` for `GraphManifold::Gravity`, set the rotation block of `sqrt_information` to `(1/σ_rad) * g_a g_aᵀ` with `g_a = pose(from).rotation * ĝ`. Correct the doc to "in camera a".

### G7

7. `low | certain | system/mod.rs:2456-2463 | The give-up speed check divides by a gap that is one frame short.` `pose_before_loss` is the pose one frame before `lost_since`, but `coast_is_plausible` gets `lost_for_sec = t - lost_since`. Speed is over-estimated by `(L+dt)/L`: 10% at a 0.5 s grace and 20 Hz, so the effective limit is 3.6 m/s, not 4. The pose gate does it correctly (`lost_for_sec + dt_sec`, :2064). `| Fix:` store the previous frame's timestamp with the pose in `mark_lost`, and pass `timestamp_sec - that_timestamp`.

### G8

8. `low | certain (numeric) | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/loop_closure/sparse_pgo.rs:18 (NUM_JACOBIAN_EPS) and pgo/edge.rs:204 | The f32 residual cancels far from the origin.` `t_b - R_b R_aᵀ t_a` has magnitude `|centre|`. At 500 m the residual noise reaches 8e-5 m and the 1e-3 central-difference translation Jacobian is off by 0.6% median, 1.5% max. At 2000 m it is 2.4% median, 6% max. This predates the lanes, but information edges now make LM acceptance depend on it. `| Fix:` before the solve, subtract node 0's camera centre from all poses and add it back after. This is exact: `T_b W (T_a W)⁻¹ = T_b T_a⁻¹`, and the gravity retraction commutes with a translation.

### G9

9. `low | certain | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-sensors/src/rig.rs:296-310 and rig_calibration.rs:371-394 | Two tests compare a formula against itself.` `each_rig_camera_has_its_body_extrinsic` expects `camera_to_body.compose(primary_from_camera)`, and `body_extrinsics_compose_through_the_primary` expects `T_BP⁻¹.compose(T_BC)`. Both are the implementation's own expression, so a wrong order would pass. `| Fix:` assert on a point. For rig.rs: `camera_to_body_of(1).transform_point(x) == camera_to_body.transform_point(&primary_from_camera.transform_point(x))`. For rig_calibration.rs: `T_BC[0].transform_point(x_c0) == T_BP.transform_point(&rig[1].primary_from_camera.transform_point(x_c0))`.

### G10

10. `low | certain | rig_calibration.rs:119-136 | validate accepts a reflection and a non-finite translation.` `rotation_defect` only tests `RᵀR = I`, so `diag(1,1,-1)` passes; rays are then mirrored. NaN in the translation also passes. `| Fix:` also require `determinant() > 0` and a finite translation, with a new error variant or `NotRigid`.

### G11

11. `low | likely | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/upstream/kornia_3d/refine_analytic.rs:230-242 | A step that raises the cost can be kept.` `relative_change` uses `abs()`, so a candidate with `new_cost > cost` inside `cost_tolerance` is adopted and reported as converged. It mirrors the generic solver and is bounded by the tolerance (1e-6 default). `| Fix:` adopt the candidate only when `new_cost <= cost`; still set `converged = true`.


   Lane G: fixed behind knob `tracking.refine_monotone`, off by default; see "Tests that catch each finding".
### G12

12. `low | likely | /home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/system/validation.rs:235-248 | The inert-knob rule is incomplete.` The `(FixedInertial, Anchor | Disabled)` rejection is correct. But lost keyframes need `imu_confident`, so `FixedInertial` on a rig without IMU is equally inert and passes. The same holds for `pose_gate: Enabled` (it needs an inertial prediction) and `reseed.keep_inertial`. `| Fix:` pass `sensors.imu` into the validation and return `InertialKnobWithoutImu` for these three.

### G13

13. `low | suspect | loop_closure/graph/mod.rs:313-320 | Correction on the gravity manifold keeps the full translation with a yaw-only rotation.` If `full.rotation` has swing (possible after `close_by_reanchor`, where `offset` is a loop measurement), `p_map = R_yaw p + t_full` moves the anchor camera centre. This is only reachable with `node_spacing: Some`; the closer uses `None`, and `correction()` has no non-test caller. `| Fix:` `t = c_map - R_yaw * c_odo`, with the centres of the anchor's estimate and odometry reference.

## Checked and found correct

- **2x6 Jacobian (refine_analytic.rs):** `[Rᵀrow | p × Rᵀrow]` matches the retraction `R·exp(ω)`, `t + R·V(ω)·υ` in (υ, ω) order. `SO3F64::left_jacobian` is V. The upper triangle is mirrored to the lower one that nalgebra's Cholesky reads. Step sign `-(H+λI)⁻¹g`, weights on both H and g, `rho` for acceptance, and squared arguments are all consistent.
- **rig_pnp.rs:** the left-perturbation Jacobian `J_proj·R_CP·[I | -[X_P]×]` matches `apply_step`. `g = -ΣwJᵀr` with `step = H⁻¹g` is consistent. Huber gets `r²` and `c_sq = robust_scale_sq` (validated positive). The Cholesky `l_dd²` guard is equivalent to the old pivot guard.
- **Kernel call sites:** every `HuberKernel`/`CauchyKernel` call in the workspace passes a squared residual (or chi²) and a squared scale. Cauchy uses `c² = scale²`, and IRLS scales residual and Jacobian by `sqrt(w)`.
- **Keyframes-while-lost commit:** config is the single source and there is no second reader.
- **AP3P lift:** `R = SᵀR'`, `t = Sᵀ(s·t' - R'c) + o` derived. `rotation_taking_to_z` is proper. Floyd sampling is correct. NaN poses score no inliers. Thresholds are squared on both sides.
- **Rig transforms:** `camera_ray`, `camera_to_body_of` (`T_BP∘T_PC`), `rig_cameras_from_body_extrinsics` (`T_BP⁻¹∘T_BC`), `b_from_a`, and the row-major `matrix_of`/`pose_of` are correct.
- **PGO:** `Se3Manifold` is a right retraction. `GravityManifold` keeps `R·g` and the centre semantics. Numeric Jacobians are consistent with the retraction by construction.
- **Applying a correction:** `apply_pose_graph_correction` and `apply_graph_correction` use `C = after⁻¹∘before` for landmarks through the reference keyframe, velocity rotation, the `T∘C⁻¹` rule and the frontier rule. Graph composition in `estimate`, `odometry_since_anchor`, `close_by_reanchor` and `twist_about` is correct.
- **Recovery:** `held_at`, the `TiltMonitor` sign (specific force is up), `rotation_from_to` direction, `leveled` about the camera centre, `inflated_noise` (√scale on densities), and bias and velocity carry into the seed keyframe are correct.
- **Pose gate:** the angle is `|log(R_to R_fromᵀ)|`, which is scale-invariant through `atan2`. NaN is rejected, and the lost branch uses the correct gap.

## Not checked

- kornia-rs internals: `solve_ap3p_multi`, `solve_pnp_ransac`, the faer sparse solve.
- VI-BA and the pose-inertial solver, the loop worker and episode bookkeeping, and the closer's rollback when a solve is unusable.
- The LaMAria source wiring and config parsing.
- Any dataset behaviour. Findings 1-3 are derived from the code paths, not observed.

## Questions for the owner

- Finding 5: is the angular residual intended for wide-FOV rigs? If so, only the doc and the threshold name change.
- Finding 1: should a lost relocalisation eventually merge the given-up segment (Atlas-style), or is detaching it from the odometry chain enough for now?

# T frontend

**Theory review: optical-flow frontend, kornia-slam-graph @ 79c04bb**

No numeric check ran: the scratch directory `review-T` does not exist and plan mode forbade creating it. Every "certain" below is a derivation from the code, not a measurement. No cargo, no edits.

Root for all paths: `/home/edgar/dev/split/kornia-slam-graph/crates/kornia-slam/src/`

## Findings

### T1

1. `high | certain | tracking/flow_tracker.rs:326-334, 429-449; tracking/optical_flow.rs:368-374; tracking/tracker.rs:925-927`
   - **Wrong:** the template cache is valid by `(track id, frame id)` only, but the source pixel can change under it.
     - On every ORB frame (keyframe due, or flow pose refused), `settle` calls `reconcile`, which rewrites a retained track's pixel to the ORB keypoint. That keypoint is up to 3 px from the flow pixel (snap radius), or anywhere if the match came from the projection search.
     - `keep` already cut the template at the flow result with `frame_id = k`. At frame k+1, `frame_id == previous.frame_id()`, so the stale-source template is reused.
     - Forward converges to the flow feature. The backward solve starts at `track.pixel()` (the ORB keypoint) and converges to the flow pixel. Round trip = `|p_flow - p_orb|`.
     - Counter-example: flow (100.4, 80.7), keypoint (101, 80) gives 0.92 px > 0.6, so a perfect track is rejected.
     - Second path: on a forward-backward reject, `keep` returns with the slot holding the template cut at the rejected result and `frame_id = current`. If ORB re-matches that map point, the next frame tracks from the wrong patch.
     - Effect: after each ORB frame, forward-backward kills long-lived tracks, and `flow_fb_rejects` (P11) spikes on post-keyframe frames.
   - **Fix:**
     - Store the source pixel in the cache entry: `(TrackId, u64, [f32; 2], LkTemplate)`.
     - `keep` writes `result.pixel`.
     - `forward` recuts from `previous` when `frame_id != previous.frame_id() || source != track.pixel()`.
     - On a forward-backward reject, set `frame_id = NO_TEMPLATE`.
     - Re-state P15: recuts now happen on post-ORB frames by design.
     - Add a test: rewrite a track pixel by 1 px between two steps; the track must survive the round trip.

### T2

2. `medium | certain | tracking/optical_flow.rs:239, 365-391; tracking/tracker.rs:423-434; tracking/flow_tracker.rs:487`
   - **Wrong:** refill tracks can never get a map point.
     - `map_point_idx` is set only in `spawn_associated` (a new track) and in `reconcile_in_cameras` (matches only).
     - `reconcile_in_cameras` rebuilds the set from matches, so every unassociated track dies at each ORB frame.
     - `Tracker::adopt` always spawns a new track at the ORB keypoint.
     - `flow_frame_windowed` ignores tracks without a map point.
     - So `track_refill` adds Shi-Tomasi, sub-pixel and LK cost and zero pose observations. The card's "they join the map at the next keyframe" holds only under `keyframe_orb: Mapping`, which is not built.
     - Unassociated tracks also count in `refill_occupied.len()`, so a camera looks full while it has few usable tracks.
   - **Fix:**
     - (a) In `Tracker::adopt`, before `spawn_associated`, look for the nearest unassociated track of the same camera within 3.0 px of the adoption pixel. If found, set its `map_point_idx` and `position` and keep its id and template.
     - (b) In the `Ok` branch of `settle` on WindowedU8, carry the unassociated survivors past `reconcile`, as `accept_flow_only` does.
     - Interim: `track_refill: Off` in both presets until step 12.4.

### T3

3. `medium | likely | system/mod.rs:1601-1613, 2836-2838, 972-976`
   - **Wrong:** `Track::position` is a copy, and inertial alignment does not refresh it.
     - `apply_inertial_initialization` (VIBA0 and the refinements inside `insert_keyframe`) rescales and rotates the map. It calls neither `refresh_track_positions` nor `reset_tracks`.
     - `apply_local_mapping_results` refreshes only when results are non-empty. With `execution: Asynchronous` (both presets) the list is empty right after `submit`.
     - Until the next mapping result, flow frames use old-world points with a new-world prediction.
     - A small correction (1 % scale at 5 m is 5 cm) passes the 25 px prior gate, so Gauss-Newton returns an old-world pose and start pixels are biased. A large one (VIBA0 gravity rotation) empties the gate, so every frame falls back.
     - The fuse and map-bound culls at the end of `insert_keyframe` are stale the same way.
   - **Fix:** call `self.refresh_track_positions()` as the last statement before `Ok(applied)` in `apply_inertial_initialization`, and before both `return true` paths of `insert_keyframe` (:2829, :2855).

### T4

4. `medium | likely | tracking/tracker.rs:633-686; tracking/pose_estimation/pnp.rs:56; tracking/pose_estimation/rig_pnp.rs:78-97`
   - **Wrong:** AnalyticGn has no outlier rejection, and outliers stay in the track set.
     - `outlier_rounds = 1` means the `retain` in `solve_rig_pnp` never runs.
     - The Huber scale is inherited: `robust_scale_sq = 25`, so delta = 5 px at `sigma_px = 1`. That is above the 3 px final inlier test and twice ORB-SLAM3's 2.45.
     - Everything within 25 px of the prediction enters the solve. At the optimum the pull is bounded but not small: shift ≈ 5·n_out/n_in px. Example: 16 coherent outliers against 80 inliers gives about 1 px.
     - `Estimate.matches` holds every finite observation, not the inliers. So `accept_flow_only` keeps outlier tracks until the next ORB frame, and `refine_pose_inertial` and the keyframe evidence receive them.
     - The round trip does not remove wrong-depth landmarks or moving objects, which is what the card's "no RANSAC needed" argument assumes.
     - KeyframesOnly makes this last longer: found-ratio culling needs 4 more ORB frames, about 4 keyframes.
   - **Fix:** in `estimate_flow_analytic` set `config.robust_scale_sq = 5.991` and `config.outlier_rounds = 2`, and build `matches` only from observations that pass `inlier`. Trade-off: a track dropped this way returns only at the next ORB frame.

### T5

5. `medium | certain | system/mod.rs:2283-2289; system/estimator.rs:482-498`
   - **Wrong:** the gyro guess can cover the wrong time window.
     - `gyro_rotation` comes from the preintegration over `[prediction_sec, t]`. When the prediction is keyframe-anchored, that window starts at the last keyframe, not at the previous frame.
     - Keyframe anchoring happens with `pose_inertial: OrbSlam3` on the first frame after each map change (as in `configs/pr180-full.ron`), and with `snapback_fix: AnchorNextFrame`.
     - The rotation is then applied to previous-frame pixels.
     - Counter-example: keyframe 5 frames back, 30°/s, 15 Hz. ΔR spans 10° instead of 2°. The 8° error is about 400·tan 8° ≈ 56 px (assumed f = 400 px, as in the test rig), outside level-0 and level-1 capture.
     - The 1 ms preset is not affected (`pose_inertial: Off`).
   - **Fix:** remove the assignment from `predict_frame_pose`. In `tracking_step`, after the prediction, set `self.gyro_rotation = (RotatedRay && predicted_velocity.is_some()).then(|| candidate_pose.rotation * pose_before.rotation.transpose())`. This equals `R_CB·ΔRᵀ·R_BC` when the origin is the previous frame.

### T6

6. `medium | certain (no cap) / suspect (budget impact) | tracking/optical_flow.rs:365-391; system/mod.rs:1045-1074; tracking/flow_tracker.rs:929-953`
   - **Wrong:** `per_camera: 48` is only a refill floor; nothing bounds the track count.
     - `reconcile` makes one track per ORB match, and `adopt_landmarks` one per new landmark of the keyframe. Neither has a count or spacing limit.
     - P4, P5 and K1 are costed at 48 tracks per camera.
     - If a cap is added, the default `klt_first` floor of 80 map-point survivors is 83 % of 96 and will bind.
   - **Fix:** add `TrackSet::thin(per_camera, min_spacing_px)` (oldest first, same 23 px grid) and call it after `reconcile` and after `adopt_landmarks` on WindowedU8. Set `klt_first: Enabled(min_tracks: …)` explicitly in the preset.

### T7

7. `medium | certain | system/mod.rs:2213-2214, 2326; system/estimator.rs:108`
   - **Wrong:** P17 cannot fail. Every flow-only frame calls `keyframe_evidence`, which locks the map once, and a second time in `with_orb_slam3_evidence` under OrbSlam3 keyframe rules. `flow_map_wait` times only the two locks in the fallback branch, so it reads 0 while the flow path can block on the mapping thread.
   - **Fix:** add those lock waits to `flow_map_wait`. Or cache `n_ref_map_points` and `ref_points_min_obs` at keyframe insert and in `apply_local_mapping_results`, and build the evidence without the lock.

### T8

8. `low | certain | tracking/flow_tracker.rs:940-946, 500-506`
   - **Wrong:** refill occupancy uses previous-frame pixels, but detection runs on the current frame. Cell marks and the 23 px spacing are off by the inter-frame flow. At 25 px of flow a new corner can land on an existing track.
   - **Fix:** fill `refill_occupied` with `start_pixel(track, model, guess)`. It is still independent of the shard split.

### T9

9. `low | likely | system/mod.rs:2798-2815, 1030-1041; tracking/flow_tracker.rs:747-750`
   - **Wrong:** two adoption gaps, both safe (no wrong pixel) but lossy.
     - `keep_keyframe` also runs for Anchor and Lost keyframes, which do not advance `last_keyframe_idx`. The kept plane id then differs from the adopted keyframe, so every carried landmark returns `None`.
     - If keyframe B is inserted before A's mapping result arrives, A's landmarks are never adopted.
   - **Fix:** call `keep_keyframe` only in the `!anchor` branch (:2811-2815). Adopt on the frame after insertion without waiting for `mapping_result_applied`: the landmarks exist after `grow_keyframe`, positions refresh at :975, and `previous_idx == kf_idx` then always takes the direct keypoint path.

### T10

10. `low | certain | ../tests/flow_zero_alloc.rs:98-116, 201-215; upstream/kornia_imgproc/good_features.rs:536-563`
    - **Wrong:** tests that prove less than they claim.
      - The thread-count test runs with retry Off, refill Off, gyro Off and no map points. It skips the only shard-dependent paths: `refill_shard`, `keep_retries`, `templates.find` after the sort, and `build_coarse` after a join. It asserts only that some frame has survivors.
      - `flow_tracker.rs` and `flow_pool.rs` have no test module. The gyro sign, round-trip rejection, retry, adoption, the template cache and AnalyticGn have no behavioural test, only config-parsing tests.
      - The Shi-Tomasi scalar reference calls the module's own `min_eigen`, so the eigenvalue formula is compared with itself.
      - The three OpenCV parity tests are `#[ignore]`.
    - **Fix:**
      - Run the thread test with the preset knobs (refill on, `CoarseLevel(1)`, a shift large enough to force retries), compare `spawned` too, and assert survivors on the last frame.
      - Add unit tests for `rotated_ray_pixel` (known yaw gives a known pixel shift and sign), for the round trip after a pixel rewrite, and for adopt.
      - Write the closed-form eigenvalue in the test.

### T11

11. `low | certain | tracking/flow_tracker.rs:234, 496; tracking/optical_flow.rs:427`
    - **Wrong:** refill keeps corners from 9 px of the edge, and `corner_sub_pix` can move them 5 px. The survivor filter drops anything under 10 px, so those spawns die on their first step.
    - **Fix:** drop refined corners outside `border_margin_px` before the `spawned.extend`, or set `border_px` to at least 15.

## Checked and found correct

- **LK fixed point:**
  - Q14 weights round half to even, and w11 is the remainder.
  - Sample shift 9 (I·32) and gradient shift 14, both with rounding.
  - The i16 Scharr bound is 4080.
  - `FLT_SCALE` 2⁻²⁰ applies to A and b alike.
  - The update sign is δ = −A⁻¹b.
  - The minimum eigenvalue is divided by 2·win², against 1e-4 and `det ≥ ε`.
  - Epsilon is squared, the oscillation half step and the status rules match OpenCV, and the error is sum / (32·win²).
- **Overflow:** 2·4080²·64 and 2·8160·4080·32 both equal 2,130,739,200, under i32 max. A 21×21 window has 63 chunks.
- **SIMD against scalar:** both are exact integers, and padded lanes are zeroed by the mask or by zero gradients.
- **Borders:** interior and border paths agree, and all reads stay in bounds.
- **Pyramid:** x·0.5 matches the pyrDown centred at 2x. The coarse retry propagates the start as OpenCV does.
- **Forward-backward:** template from the current frame at the forward result, solved into the previous frame from the previous pixel, in level-0 pixels (VINS scheme).
- **Gyro:** `R_BCᵀ·ΔRᵀ·R_BC` is previous-to-current, given `R_{k+1} = R_k·ΔR`. The per-camera form `R_CP·R·R_CPᵀ` is right. Unproject and project are ideal pinhole plus a local distortion offset. Ignoring translation only costs a failed start.
- **Rig frame:** ranges tile, each track uses its own camera's intrinsics and `T_CP`, and the `rig_pnp` Jacobian is consistent with the left perturbation.
- **AnalyticGn:** the start is the candidate pose, the inlier floor counts over all observations, and the fallback path works.
- **Shi-Tomasi:** response formula, f32 exactness (≤ 9.4M < 2²⁴), 3×3 non-maximum suppression with valid neighbours, spacing grid, half-resolution mapping 2x + 0.5, border mask, deterministic sort.
- **cornerSubPix:** normal equations, mask, patch centre and stop rules match OpenCV.
- **Threading:** no shared mutable state, workers drop their `Arc` clones before the send, the merge is sorted by (camera, id), and ids and spawns do not depend on the split.
- **KeyframesOnly bookkeeping:** the found ratio is unbiased, only slower.
- **Adoption:** right plane, right frame id and right previous-frame pose.

## Could not check

- `wide` semantics (`dot` as pairwise madd, `mul_widen`, `from_i32x8_truncate`, `round_float`) are assumed from their names.
- The reference `fixed_point.rs` was only skimmed for constants and status rules. Bit equality rests on the reported-passing tests.
- Not read: `refine_pose_analytic` internals (mono path), `stereo_pyramid`, the bench app.
- How many matches and landmarks a LaMAria keyframe yields, which decides the size of findings 2 and 6.
