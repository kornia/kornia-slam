# PR #180 card G: online calibration in the local VI-BA (Lane C, round 2)

Plan `pr180-frontend-1ms.md` section 9 (Lane C) asks the local VI-BA to estimate four calibration states online: the IMU-camera clock offset, a rotation offset per camera, a focal scale per camera, and the gyro intrinsics. The offline solve (card D, `global_vi_ba/`) already has all four. Lane C moves its layout and prior code to `mapping/calibration_states.rs` and reuses it in `vi_ba_schur.rs`, behind `mapping.local_ba.online_calibration`, with every state off by default.

This lane saves no tracking time. The solve runs on the mapping thread, so card E's budget does not see it. It tests ledger row A8 (G11), fixed calibration, in the online system.

Nothing here has been measured. The validation workflow scores it.

**Owner decisions (2026-11-17).**
- **D1:** the primary camera's rotation offset lives on the IMU side, as a rotation of the camera-to-IMU extrinsic `imu_t_bc` inside the IMU residual, as VINS-Fusion does. Keyframe poses stay in the frame tracking estimates. The clock offset is unchanged.
- Card F (raw fisheye) is deferred to its own round. Nothing in this card depends on it.
- D2-D4 are settled below ("Decisions") with the smallest consistent choice. No point blocks the lane.

## Experiment: G, online calibration
Gap tested: A8 (G11), fixed calibration (`experiments/gap-ledger.md:16`), in the online system. The score link is G18: drift and scale, not coverage, is the score gap (`gap-ledger.md:11`). Card D tests the same row offline. This card asks whether a window of 3 keyframes, or 25 in the large window, observes the same states.
Base: kornia-slam@`c0a1dc3`, kornia-rs@`4ada5a7`. Skeleton tag `pr180b/K` (not cut yet).   Binary tag: `pr180b/L-C`.

Mechanism (file:line on `c0a1dc3`, `crates/kornia-slam/src/` unless named):
- **The online VI-BA has no calibration column.**
  - `ViBaParams` holds a fixed IMU-camera extrinsic `imu_t_bc` (`mapping/vi_ba_schur.rs:181`) and fixed `RigObservationCamera::camera_from_primary` for the secondary cameras (`:286-289`).
  - `residual_and_jacobians` projects with each camera's fixed `fx, fy` (`:651-654`, `:665-666`).
  - The reduced system is `15 × n_free_kf` wide (`:1477-1479`). Nothing beyond the keyframe states enters it.
- **The IMU residual takes the extrinsic per call.**
  - `preintegration::residual_and_jacobians(i, j, pim, gravity, t_bc)` forms the body pose as `t_bc ∘ T_cw` (`crates/kornia-factors/src/inertial/preintegration.rs:62-65,72-82`).
  - The online solve passes `params.imu_t_bc` at three calls: linearisation (`mapping/vi_ba_schur.rs:1527`), trial cost (`:1832`) and the debug norms (`:1900`).
  - Its pose columns hold for any fixed `t_bc` (`preintegration.rs:70-71`), so an offset that is constant within one linearisation needs no new pose term.
- **The IMU interval is the camera interval.**
  - A map edge spans `[t0, t1]`, the two keyframe stamps (`mapping/map/imu_factor.rs:17-18`).
  - The window re-integrates an edge only when the bias has moved past 0.02, and always over the same interval (`mapping/bundle_adjustment.rs:646-669`).
  - The LaMAria source adds one constant `imu_time_offset_ns` to every IMU stamp (`apps/kornia-slam-app/src/source/lamaria.rs:35-37`). The factory record rectifies the IMU samples once (`apps/kornia-slam-app/src/datasets/lamaria.rs:39-48,112-116`). Nothing estimates what remains.
- **The offline solve has every piece the online one needs.**
  - Layout, one flag per state: `global_vi_ba/calibration.rs:14-79`.
  - Value priors and the random walk between slices: `calibration.rs:153-219`.
  - Projection with the rotation offset and focal scale: `global_vi_ba/problem.rs:335-363`. Offline states are body poses, and the offset sits on the left of `camera_from_body` (`:337-345`).
  - Calibration columns: `problem.rs:453-460`.
  - Rotation-offset Jacobian `J_l(θ)ᵀ (p_c × a)`, and the focal column: `problem.rs:618-628`.
  - IMU integration over `[t0 - td, t1 - td]` with the gyro correction: `global_vi_ba/imu_edges.rs:105-128`.
  - Central differences along the gyro and clock columns: `imu_edges.rs:203-236`.
  - Border assembly and Schur elimination with calibration columns: `global_vi_ba/reduced.rs:219-293,328-350,355-375,417-429,449-458,466-487`.
  - All of it is `pub(super)`, private to `global_vi_ba`.
- **Raw IMU samples are already kept.**
  - Every map edge stores `raw_samples` over `[t0, t1]` (`map/imu_factor.rs:12-16`). Under `Midpoint` it also stores the samples just outside (`:26-28`). The edge is filled at insertion (`inertial/mod.rs:174-184`).
  - Plan section 14 ("kept only while a state re-integrates") therefore needs no new storage. The samples exist on every edge today.

Theory:
- **The states and their observability in one window** (plan section 9 table, made quantitative). LaMAria IMU noise: gyro 1.75e-4 rad/s/√Hz, accel 7.8e-4 m/s²/√Hz (`R_11_5cp.json`, `imu0`). Over a 0.5 s edge this gives σ(ΔR) ≈ 1.2e-4 rad and σ(Δv) ≈ 5.5e-4 m/s. The keyframe spacing is assumed, not measured.
  - **Clock offset `td`.**
    - A shift `δ` of the IMU interval changes the preintegrated rotation by about `(ω(t1) − ω(t0)) δ` and the velocity by `(a(t1) − a(t0)) δ`. At constant rate and acceleration it changes nothing, so `td` is observable only while the motion changes (Li and Mourikis 2014).
    - Walking head motion gives `|Δω|` of 0.3-1 rad/s and `|Δa|` of 1-3 m/s² across an edge. One millisecond then moves ΔR by 0.3-1 mrad, an SNR of 2.5-8 per edge.
    - A window of 3-4 edges should give a per-solve σ(td) of 0.05-0.3 ms. A constant gyro bias cannot absorb the shift, because the shift scales with `Δω` and a constant bias does not.
  - **Camera rotation.**
    - *Primary camera, the IMU-camera rotation, on the IMU side (D1).* It enters only the IMU residual, as the rotation in `imu_t_bc`. This is hand-eye calibration: a relative rotation about axis `n` constrains the offset only perpendicular to `n` (Kelly and Sukhatme 2011). LaMAria motion is mostly yaw, so the offset about gravity is weak, and only the value prior holds it. The other two axes see an SNR of about 12 per edge: a 0.2° offset on a 0.4 rad relative rotation is a 1.4 mrad residual against σ(ΔR) = 1.2e-4. Per window that is σ_data ≈ 0.01°, below the plan's 0.05° step prior, so one solve moves about 0.96 of the way to the window's estimate (gain below).
    - *Gravity in the primary rotation.* The velocity and position residuals rotate gravity into the body (`preintegration.rs:96-100`). A tilt of the offset therefore looks like an accel bias of `g |θ|`, 0.034 m/s² at 0.2°, until the body turns. GK3's correlation with the accel-bias norm detects this absorption.
    - *Secondary cameras.* The offset sits on `camera_from_primary`. Landmarks seen by both cameras observe it from vision alone, in every axis.
  - **Focal scale.**
    - Under rotation, the image moves by `f · θ` and the gyro measures `θ`, so the focal is observable through the gyro scale (visloc `lamaria_batch_ba.md:285-384`, cited by card D).
    - With the gyro intrinsics on as well, the two trade 1:1 about each rotation axis. In translation, the focal trades with landmark depth. The plan predicts no gain.
  - **Gyro intrinsics.**
    - The factory record already rectifies the gyro (`datasets/lamaria.rs:112-116`). The remaining scale is a residual well under 0.5 %. In a window it trades with the focal and with the bias, so the plan predicts the knob fails its kill rule.
    - The knob is unseeded (D3): it starts at identity, and the value prior pulls to identity.
- **Primary rotation on the IMU side: frames and sign (D1).**
  - *Frame.* Offline, physical camera `k` is `Exp(θ_k) · camera_from_body_k · T_bw` (`problem.rs:337-345`). Online, the state `T_cw` is the physical primary camera, the frame tracking's PnP estimates. The body pose is then `T_bw = imu_t_bc · Exp(−θ_0) · T_cw`, where `Exp(−θ_0)` is a pure rotation about the camera centre.
  - *Sign.* `θ_0` keeps offline's meaning and sign, so card D's primary estimate compares directly (GP6).
  - *Translation.* The translation of `imu_t_bc` is not changed. The body origin stays at the same point relative to the camera.
  - *Consistency.* The primary projection carries no calibration rotation. A keyframe is written back in the frame tracking estimates, so tracking's next PnP agrees with it (GP7). The fixed-context keyframes are in the same frame, so they do not hold the estimate back.
  - *Secondary cameras.* Camera `k ≥ 1` projects through `Exp(φ_k) · camera_from_primary_k · T_cw`. The offline offset is relative to the factory `camera_from_body_k`, and the online one to the physical primary. To first order, `φ_k ≈ θ_k − R_{c_k c_0} θ_0`. GP6 compares through this relation. `φ_k` changes no keyframe's frame.
  - *Cost.* The primary rotation needs no re-integration. Its 3 columns are central differences of `residual_and_jacobians` at `imu_t_bc · Exp(−(θ_0 ± h e_a))`: 6 residual evaluations per edge per linearisation.
  - *Marginal prior.* `next_prior` is formed at the solve's `θ_0`. The next solve's `θ_0` differs by that solve's change, `k (θ_data − θ_prev)` (gain below): near the window noise once converged, larger in the first few solves. This card accepts that linearisation mismatch.
- **What tracking still uses.** The estimates reach only the next local BA. Tracking keeps the factory calibration; none of the residual differences change a pose's frame:
  - IMU prediction with the factory `imu_t_bc`: the error is `|θ_0|` times the inter-frame rotation, about 0.2° × 0.05 rad (1 rad/s at 20 Hz) ≈ 2e-4°;
  - secondary pixels with the factory `camera_from_primary`: `f |φ_k|` ≈ 0.8 px at 0.2° and f = 241.6;
  - `td`: `td · ω` of rotation;
  - focal: up to `s · f · x/z` ≈ 1.5 px at the image edge for `s` = 0.5 %.
- **Prior structure: an exponential filter, not a marginal.**
  - Each per-camera state has one slice online. It gets a value prior to the factory value (0) and a step prior to the previous solve's estimate (plan section 9 Lane C step 2).
  - The gyro and clock states keep the offline form, a value prior only (`calibration.rs:212-217`). The previous estimate only seeds their start.
  - Consecutive windows share most of their data, so the step prior counts information again. The estimate is a smoother with a memory set by `σ_step` against the window's information, not a posterior.
  - **Gain per solve.** A Gaussian step prior does not cap a jump. Per component, the solve minimises `(θ − θ_prev)²/σ_step² + (θ − θ_data)²/σ_data²`, so it moves a fraction `k = σ_step² / (σ_step² + σ_data²)` of the way from `θ_prev` to the window's estimate `θ_data`. The value prior (σ 1°) shrinks the result by `σ_data²/σ_value²` ≤ 0.25 %, negligible.
    - σ_data = 0.01° (theory above): `k` = 0.96. One solve reaches 96 % of a step change.
    - σ_data = 0.05° (a short or slow window): `k` = 0.5. The residual `(1 − k)^n` falls under 10 % after 4 solves.
    - So 90 % of a step change is reached in about 1 to 10 ungated solves (GP6). An unobserved axis (σ_data far above σ_step, the primary yaw) has `k` ≈ 0 and stays where the value prior holds it.
    - Per-solve jitter is about `k σ_data`, 0.01-0.025°, inside GP6's 0.05° interquartile bound.
- **Clock offset by re-integration, not by feature velocity.**
  - VINS-Fusion moves each normalised point by `(td − td_i) · velocity_i` (`factor/projectionTwoFrameOneCamFactor.cpp:57-61`). That needs a per-observation pixel velocity, and `BaObservation` has none.
  - OpenVINS propagates the IMU to `t_cam + calib_dt` (`ov_msckf/src/state/Propagator.cpp:59-64`). That is the interval shift this card uses, and the offline solve already implements it (`imu_edges.rs:116`).
  - Reusing it keeps one factor for both solves (plan section 9 Lane C step 1).
  - **Sign:** in our convention the camera stamp `t` is read on the IMU clock at `t − td` (`calibration.rs:111`). VINS-Fusion reads IMU time `t_cam + td` (`estimator/estimator.cpp:280`), and OpenVINS `t_imu = t_cam + calib_dt` (`Propagator.cpp:59`). So our `td` = −VINS `td` = −OpenVINS `calib_dt`.
- **Window ends.**
  - A shifted interval of the oldest edge (`td > 0`) or the newest edge (`td < 0`) reaches up to `|td|` past the samples the window holds. `from_measurements` then holds the nearest sample over the gap (`kornia-sensors/src/imu.rs:203-216`).
  - That is right to first order in `td`. The error is about `|ω̇| td² / 2`: 2e-5 rad at `td` = 2 ms and `ω̇` = 10 rad/s², below σ(ΔR). The predicted bias on `td` is under 0.2 ms for `|td|` ≤ 2 ms (GT1).
- **Gate.**
  - VINS-Fusion holds `td` constant while the oldest window frame moves below 0.2 m/s (`estimator.cpp:1042-1043`). It opens the extrinsic once the window is full and moving, and latches it (`:1029-1032`).
  - The plan applies both conditions to every state: hold below 0.2 m/s, or before the window is full (Lane C step 4). A held solve applies the carried estimate as constants and adds no column.
- **References compared.**
  - ORB-SLAM3 and Basalt have no online calibration (plan section 9).
  - OpenVINS has one flag per state (`ov_msckf/src/state/StateOptions.h:47-56`), the same split as our layout.
  - VINS-Fusion gives the gate, the scalar `td`, and the placement of the IMU-camera rotation (D1): its states are body poses, and the extrinsic `ric` enters between body and camera, so the camera poses stay where vision puts them. Its feature-velocity `td` factor loses to re-integration here, for the reason above.
  - visloc gives the per-camera rotation and focal parametrisation, and the 0.05° and 5e-4 steps.

Knob: `mapping.local_ba.online_calibration` (plan section 8), all `None` by default, which is today's behaviour. The prior types are the offline ones (`global_vi_ba/mod.rs:108-136`).

| RON path | Type | Default | Experiment value (this card) | Source of the value |
|---|---|---|---|---|
| `mapping.local_ba.online_calibration` | `OnlineCalibrationConfig { camera_rotation: Option<RotationSlicePrior>, camera_focal: Option<FocalSlicePrior>, gyro_intrinsics: Option<GyroIntrinsicsPrior>, clock_offset: Option<ClockOffsetPrior> }` | all `None` | per run below | plan section 8 |
| `.camera_rotation` | `Option<RotationSlicePrior { step_sigma_deg: f64, value_sigma_deg: f64 }>` | `None` | `Some((step_sigma_deg: 0.05, value_sigma_deg: 1.0))` | step: plan section 9 table; value: card D (`pr180-D-card.md:180`) |
| `.camera_focal` | `Option<FocalSlicePrior { step_sigma: f64, value_sigma: f64 }>` | `None` | `Some((step_sigma: 5e-4, value_sigma: 0.01))` | step: plan section 9 table; value: card D (`:181`) |
| `.gyro_intrinsics` | `Option<GyroIntrinsicsPrior { sigma: f64 }>` | `None` | `Some((sigma: 0.005))` | card D (`:182`) |
| `.clock_offset` | `Option<ClockOffsetPrior { sigma_ms: f64 }>` | `None` | `Some((sigma_ms: 2.0))` | card D (`:183`) |

- **Online meaning of the shared fields:**
  - `step_sigma*`: the random walk from the previous solve's estimate, per solve;
  - `value_sigma*`, `sigma` and `sigma_ms`: the prior to the factory value (0, or identity for the gyro).
  - `camera_rotation` on camera 0 is the IMU-side offset of `imu_t_bc` (D1); on camera `k ≥ 1` it is the offset of `camera_from_primary_k`. One knob covers both, with the same priors.
- **Gate:** no knob. Plan section 8 lists none. Constants in `mapping/online_calibration.rs`: `GATE_MIN_SPEED_MPS = 0.2` (VINS-Fusion `estimator.cpp:1042`), and "window full" means the solve has `active_keyframes` free keyframes.
- **Validation:** see the skeleton contract. A state on without an IMU is refused.

Run configs (rig `configs/lamaria-rig-imu.ron`, mono `configs/lamaria-mono-imu.ron`, EuRoC `configs/euroc.ron`; `mapping: (execution: Synchronous)` for every `cmp` run):

| Name | `online_calibration` |
|---|---|
| `off` | default (all `None`) |
| `td` | `clock_offset` only |
| `rot` | `camera_rotation` only |
| `td+rot` | both |
| `focal` | `td+rot` plus `camera_focal` |
| `gyro` | `td+rot` plus `gyro_intrinsics` |
| `pinned` | all four, every sigma `1e-9`: the states stay at 0, but the IMU path re-integrates, because `clock_offset` and `gyro_intrinsics` are on (control for GK2) |

Baselines: none run yet. The validator records `pr180b/K` (sha, binary md5, load) for each run config at `off`, before scoring `pr180b/L-C`. Card D's offline estimates on the same dumps (`[offline_ba] td_ms=`, `cam<i> slice<s> rot_deg=`, `focal_pct=`, `gyro_scale_pct=`) are the references for GP5, GP6, GP8 and GP9. Without them, those predictions fall back to the stated absolute ranges.

Predictions (numbers, each tied to a measurement):
  GP0 Knob off: `pr180b/L-C` against `pr180b/K`, 0 differing bytes on MH_01 `euroc.ron`, 3_18 `lamaria-mono-imu.ron` and R_11 `lamaria-rig-imu.ron`. `pr180b/K` against `c0a1dc3` on the same three runs, 0 differing bytes: this proves the skeleton's `run_local_inertial_ba_inner` refactor changes no output. The `offline-ba` subcommand on a 3_18 session dump, with all four calibration states on (card D's config), gives 0 differing bytes after the move.   measured by: `cmp` on every output file of each pair of run dirs.
  GT0 Re-integration at `td = 0` and identity gyro, through the window stream (Design C4), is bit-equal to `map::ImuFactor::repreintegrate(bias)` (`map/imu_factor.rs:34-45`). This holds for `Hold` and for `Midpoint`, on an interior edge and on both window-end edges.   measured by: unit test `window_reintegration_matches_repreintegrate`.
  GT1 Synthetic window from the varying-motion generator (below), one solve. IMU stamps shifted so that `td = +5 ms`: estimate 4.5 to 5.5 ms, and positive. At `td = −5 ms`: −5.5 to −4.5 ms. At `td = 2 ms` on a window-end edge: bias under 0.2 ms. `td` has a value prior only (σ 2 ms), which shrinks a 0.3 ms-σ estimate by at most 2 %, so one solve is enough.   measured by: unit test `online_td_recovers_injected_offset`.
  GT2 Synthetic, from the varying-motion generator, each case a single solve starting at 0 (so `θ_prev` = 0). The test first asserts the window's σ_data, from the solve's information on the rotation columns, is at most 0.01° on each observed axis, so `k` ≥ 0.96 (theory) and one solve reaches at least 0.288°. Tolerance 0.03° throughout.
    - Two-axis rotation, a 0.3° offset about the camera x axis on the secondary camera: recovered within 0.03°.
    - Two-axis rotation, a 0.3° offset injected in the primary's `imu_t_bc` about the camera x axis: recovered within 0.03°, with offline's sign (`+0.3°`, not `−0.3°`).
    - Yaw-only motion with a 0.3° primary offset about the yaw axis: the estimate stays under 0.03° (under 10 % recovered, held by the value prior). The other two axes recover within 0.03°.
    measured by: unit test `online_rotation_observability`.
  Varying-motion generator (GT1, GT2, GT6), a new test helper in `vi_ba_schur.rs`. The `spin_*` helpers (`vi_ba_schur.rs:2201-2263`) are not reused for the IMU: they spin at the constant `SPIN_RATE`, so `Δω` = 0 and `td` is unobservable, and `spin_edge` builds a preintegrated `ImuFactor` with no map edge and no `raw_samples`, so nothing can re-integrate. `spin_vision` is reused unchanged: it reads only the keyframe poses.
    - *Motion:* 5 keyframes, 4 edges of 0.5 s, `active_keyframes` = 4. Body rotation `R(t) = Exp(α_x(t) e_x) · Exp(α_y(t) e_y)`, each `α(t) = (A / 2πf)(1 − cos 2πf t)` with A = 1 rad/s, f = 1 Hz; the yaw-only case uses one axis. Position `(v0 t, 0, 0) + B sin(2πf t) e_y` with v0 = 0.5 m/s and B = 0.05 m, so the speed stays above the 0.2 m/s gate and `|Δa|` across an edge is about 2 m/s² (theory).
    - *IMU:* noise-free analytic samples at 1 kHz: body angular rate `ω_b = Exp(α_y e_y)ᵀ α̇_x e_x + α̇_y e_y`, specific force `R(t)ᵀ (p̈(t) − g)`, with LaMAria's noise model (gyro 1.75e-4 rad/s/√Hz, accel 7.8e-4 m/s²/√Hz) as the weights. A sample taken at camera-clock time `τ` is stamped `τ − td` (`calibration.rs:111`).
    - *Edges:* per edge, a `MapImuFactor` with `raw_samples` over `[t0, t1]` plus the nearest sample outside each end, `t0` and `t1` the keyframe stamps, `Hold`, `bias_walk_extra_s` = 0. Its `preintegrated` is `from_measurements_with` over `[t0, t1]` at zero bias, and the solver's `ImuFactor` wraps the same preintegration. These map edges are `OnlineCalibrationInput::map_edges`.
    - *Cameras:* the true camera poses are `T_cw = Exp(θ_0) · imu_t_bc⁻¹ · T_bw`, so the solver's `imu_t_bc · Exp(−θ_0) · T_cw` is the true body (theory, D1). The secondary camera's pixels are projected through `Exp(φ_1) · camera_from_primary_1`.
  GT3 Synthetic pure rotation, focal +0.5 % injected, focal and gyro both on: the sum of the focal scale and the gyro scale about the rotation axis recovers 0.4 to 0.6 %. Neither alone is within 0.1 % of its injected value unless the priors are unequal.   measured by: unit test `focal_and_gyro_scale_trade`.
  GT4 `online_prior_rows`: for one slice, the value rows equal `prior_rows`'s slice-0 rows bit for bit. Each step row's residual is `(current − previous) / step_sigma`, and its two coefficients are `±1/step_sigma`.   measured by: unit test `online_prior_rows_are_value_and_step_rows`.
  GT5 Gate: a synthetic window whose oldest free keyframe moves at 0.19 m/s adds 0 calibration columns and returns the carried estimate unchanged. At 0.21 m/s with a full window it adds every column of the layout. At 0.21 m/s with one free keyframe short of `active_keyframes`, it adds 0 columns.   measured by: unit test `online_calibration_gate`.
  GT6 Frame consistency (D1): the GT2 primary case, with the true camera poses as the start. The solved free-keyframe orientations stay within 0.005° of the true camera poses, and their centres within 1 mm. The body poses `imu_t_bc · Exp(−θ_0) · T_cw` match the true body poses within 0.03°.   measured by: unit test `online_imu_side_rotation_keeps_camera_frame`.
  GT7 Re-integration rule (D2): `Layout::online` with only `camera_rotation`, only `camera_focal`, or both, gives `imu_reintegrates() == false`. With `clock_offset` or `gyro_intrinsics` on, it gives `true`. In a calibrated solve with only `camera_rotation` on, no edge is re-integrated (a counter is 0).   measured by: unit test `online_reintegration_only_for_imu_states`.
  GP5 `td` on LaMAria (`td`, rig, 5 proxy sequences 1_19, 2_11, 3_18, 5_11, R_11):
    - median of the per-solve `td` over ungated solves within ±1.0 ms of card D's offline `td_ms` on the same sequence (fallback: `|median|` ≤ 2 ms);
    - interquartile range over ungated solves 0.1 to 1.0 ms;
    - the 4 sequences of device 1WM093701G1276 (card D, P15) agree within 1.0 ms.
    measured by: `[online_calib]` lines (Design C7) and `offline-ba` `td_ms=`.
  GP6 Rotation on LaMAria (`rot`, rig, 5 proxy sequences):
    - *Secondary camera (cam1, rig-relative `φ_1`):* final magnitude 0.03 to 0.3°, the median over the last half of ungated solves. Per-axis interquartile range at most 0.05°. Each axis within 0.1° of card D's `θ_1 − R_{c_1 c_0} θ_0` (theory, D1).
    - *Primary camera (IMU side):* the component about gravity under 0.03° from 0. The other two reach 60 to 140 % of card D's offline primary-slice mean, with the same sign (fallback: 0.05 to 0.35°). The fixed context no longer holds the estimate back (theory, D1).
    - *Convergence:* 90 % of the final value within 1 to 10 ungated solves on the secondary camera and the primary's two tilt axes: the gain per solve is `k` = 0.5 to 0.96 for σ_data 0.05 to 0.01° (theory, prior structure). Slower convergence means the real per-window σ_data is above 0.05°; record the implied σ_data from the solve count.
    measured by: `[online_calib] cam<i>_rot_deg=` lines.
  GP7 Frame check (`rot` against `off`, rig, R_11 and 3_18): the median `[predgap] rot=` (deg) on the first tracked frame after each keyframe write-back changes by under 0.02°. On all other frames it changes by under 0.02°. With `td` alone it changes by under 0.02° on both.   measured by: `[predgap]` lines (`system/mod.rs:1844`), split by "frame after a keyframe write-back".
  GP8 Focal (`focal` against `td+rot`): mean per camera +0.1 to +0.6 % on 2_11, 3_18 and R_11 (card D P13's band). CP score change within the 3-sample band of `td+rot` on 4 of 5 sequences; 5_11 loses 0 to 8 points (moving platform, `pr180-D-card.md:46`).   measured by: `[online_calib] cam<i>_focal_pct=`, CP runner.
  GP9 Gyro (`gyro` against `td+rot`): every per-axis scale estimate at most 30 % of card D's offline `gyro_scale_pct` (fallback: `|scale|` ≤ 0.1 %). Shear at most 1 mrad. CP change inside the band. This is the plan's prediction that the knob fails (GK6).   measured by: `[online_calib] gyro_scale_pct=`, `gyro_shear_mrad=`.
  GP10 Score (`td+rot` against `off`, rig, 3 samples each):
    - CP score on 1_19, 3_18 and 5_11: each −2 to +4 points;
    - R_11 and 2_11 tracked share within ±1 point;
    - EuRoC MH_01 and V1_03 (`euroc.ron`, stereo-IMU) ATE −10 % to +5 %;
    - MH_03 ATE no worse than +5 % (the absorption check, GK3).
    measured by: `evaluate_wrt_control_points`, the bench `eval.py` ATE.
  GP11 Solve time, as the mean of `ba_solve_ns / ba_solves` (`mapping/local_mapping.rs:94-102`) against `off` on R_11. Arithmetic, with E = 4 edges, 200-1000 samples per edge at 1 kHz, 0.3-1.0 µs per sample integration (card D's budget gives at most 0.8 µs), 5-20 LM iterations, and a VI-BA base of 85-250 ms (plan section 2):
    - `rot`: +2 to +20 %. No re-integration (D2). The primary columns cost 6 residual evaluations per edge per linearisation, under 0.1 ms per solve. The 3 border columns of each point seen by camera 1 widen its Schur update, and the dense system grows by 6.
    - `td`: +5 to +150 % (3 integrations per edge per linearization, plus about 1.5 LM trials).
    - `gyro`: +20 to +550 % (15 integrations per edge per linearization).
    measured by: run counters.
  GP12 Tracking time (`td+rot`, Asynchronous, with `keyframe_orb: Tracking`, R_11): median per-frame tracking time within ±2 % of `off`. The solve is on the mapping thread. Local-BA solves per inserted keyframe drop by at most 10 % (the worker keeps only the latest job, `local_mapping.rs:400-413`).   measured by: tracking timings, `ba_solves` against the keyframe count.
  GP13 Gate share (`td+rot`, R_11 and 3_18): gated solves are 1 to 25 % of all VI-BA solves (the start, plus stops at doors and turns). The gated count equals the count of solves whose logged oldest-free-keyframe speed is below 0.2 m/s or whose window is short, exactly.   measured by: `[online_calib] gated=` and `speed=`.
  Mechanism counters: the estimates in GP5, GP6, GP8 and GP9, the GP7 `[predgap]` split and the GP13 gate count. Each must move as predicted before its score prediction counts.
  Noise mechanism: with `td` on, shifting `imu_time_offset_ns` by +50 µs and by −50 µs (the noise method, `lamaria-handoff.md:19`) moves the median `td` by −100 ± 60 µs between the two runs. In our convention a +δ shift of the IMU stamps gives `td − δ`.   measured by: GP5's median in the two noise runs.

Kill rules:
  GK1 (mechanism moved, score did not) GP5 and GP6's secondary-camera rows hold, yet GP10's CP change is inside the 3-sample band on all of 1_19, 3_18 and 5_11 -> the device model is right, but online calibration is not what limits the score now. Record "built, not adopted"; re-test after card A's or D's limiter moves.
  GK2 (score moved, mechanism did not) GP10 moves beyond the band, yet every estimate stays near 0 (median `|td|` < 0.2 ms, every rotation axis < 0.02°) -> the IMU path change or the gate moved the score, not the calibration. Run `pinned`: if `pinned` reproduces the change, the gain belongs to re-integration, which runs only with `clock_offset` or `gyro_intrinsics` on (D2). Do not credit calibration.
  GK3 (absorption, the MH_03 mechanism, plan section 13) any of the following -> the state absorbs bias or scale error. Keep that knob off; do not tune its prior to pass.
    - an estimate leaves 2x its range: rotation above 0.7°, focal above 1.2 %, `|td|` above 4 ms;
    - the per-solve `td` or a rotation axis correlates with the newest keyframe's accel-bias norm at `|ρ|` > 0.5 over ungated solves (the primary tilt axes are the expected case, theory);
    - MH_03 ATE worsens by more than 5 % with any state on.
  GK4 (sign) GT1 recovers the wrong sign, or the noise mechanism moves `td` by +100 µs instead of −100 µs -> a sign error in the interval shift. Block the lane until GT1 passes. The same rule holds for the primary rotation: GT2 recovering `−0.3°`, or GP6's primary axes opposite in sign to card D, is a sign error in `Exp(−θ_0)`.
  GK5 (frame mismatch, D1) GT6 fails, or GP7 rises by at least 0.05° on frames after a write-back -> the offset leaks into the keyframe frame, so it is composed on the wrong side or with the wrong sign. `camera_rotation` stays off whatever GP10 says, until GT6 passes. A rise between 0.02° and 0.05° is recorded and checked against GP6's secondary row, the only camera-side rotation left.
  GK6 (gyro, predicted to fire) GP9 holds -> the gyro intrinsics are unobservable in the window, as the plan predicts. The knob stays off. If it does not fire (the estimates track the offline values within 30 %), the observability table is wrong: re-run GT3 with the LaMAria noise values.
  GK7 (identity) GP0 or GT0 fails -> bisect the lane's commits with `cmp` and revert the first that differs. A GP0 failure on `offline-ba` alone points at the move (Design C1). A failure of `pr180b/K` against `c0a1dc3` points at the skeleton's `run_local_inertial_ba_inner` refactor.
  GK8 (cost) the mean solve time goes above GP11's upper bound, or GP12's solves per keyframe drop more than 10 % -> the calibration starves the mapping thread. `rot` and `focal` do not re-integrate (D2), so an overrun there points at the border, not the IMU. For `td`, the next step is an analytic column in a later card.
  GK9 (gate) GT5 fails, or GP13's exact count differs by even 1 -> the gate is not the plan's rule. Fix it before any score is read.
Bit-identity: `off` must `cmp` equal `pr180b/K` on GP0's three runs, `pr180b/K` must `cmp` equal `c0a1dc3` on the same runs, and `offline-ba` must `cmp` equal across the move. `pinned` is not an identity run, because re-integration replaces the first-order bias correction.
Noise: GT0-GT7 and GP0 are deterministic, so they need no repeats. For GP5-GP13, run 3 samples per config with the IMU offset at 0 and ±50 µs, and judge against `off`'s 3-sample band. A `td`-on config absorbs part of the ±50 µs perturbation (the noise mechanism), so its own band is narrower. Use `off`'s band for verdicts. Timing runs are solo, with load within ±2.

## Design (what the lane builds)

### C1. Move the offline calibration code
- `git mv crates/kornia-slam/src/mapping/global_vi_ba/calibration.rs crates/kornia-slam/src/mapping/calibration_states.rs`, then:
  - visibility `pub(super)` becomes `pub(in crate::mapping)` on `Layout`, its methods, `Calibration`, its methods, `PriorRow` and its fields, and `prior_rows`;
  - `use super::{CalibrationEstimate, OfflineCalibrationConfig}` becomes `use super::global_vi_ba::{...}`;
  - `PriorRow::new` stays private.
- `mapping/mod.rs`: the same commit adds `mod calibration_states;` (D4).
- `global_vi_ba/mod.rs`: delete `mod calibration;` and import from `crate::mapping::calibration_states`. Do the same for `problem.rs`, `reduced.rs` and `imu_edges.rs`. These are imports only (plan section 7).
- **Shared IMU-calibration pieces.** `GYRO_STEP`, `CLOCK_STEP_SEC` and the column list of `ImuModel::calibration_jacobian` (`imu_edges.rs:22-25,212-219`) move to `calibration_states.rs`:
  - `pub(in crate::mapping) fn imu_calibration_columns(layout: &Layout) -> Vec<(usize, f64)>`, the gyro and clock columns and their steps;
  - `pub(in crate::mapping) fn corrected_samples(samples: &[ImuMeasurement], gyro: &GyroIntrinsics) -> Vec<ImuMeasurement>`.
  - `imu_edges.rs` calls both, with the same operations in the same order, so the offline output keeps its bits (GP0). No factor is duplicated (plan Lane C step 1).

### C2. Online layout and priors (`calibration_states.rs`)
- `Layout::online(config: &OnlineCalibrationConfig, cameras: usize) -> Layout`: `slices = 1`, columns in the order of `Layout::new`: rotation `cameras × 3`, focal `cameras`, gyro 6, clock 1. A 2-camera rig has at most 15 columns. Camera 0's rotation columns hold `θ_0`, the IMU-side offset (D1).
- `online_prior_rows(layout, config, current, previous) -> Vec<PriorRow>` (plan section 8):
  - per camera: a value row `current / value_sigma` and a step row `(current − previous) / step_sigma` per component;
  - gyro and clock: value rows only, as `prior_rows` builds them (`calibration.rs:212-217`).
  - Rotation sigmas are converted to radians, as `prior_rows` does.
- `Layout::imu_reintegrates(&self) -> bool`: true when the gyro or clock columns exist (D2). Only the online solve calls it. `Layout::reintegrates` and its offline callers are unchanged (GP0).
- `pub(in crate::mapping) fn imu_t_bc_with_offset(imu_t_bc: Option<&Pose3d>, theta: Vec3F64) -> Pose3d`: `imu_t_bc · Exp(−θ)`, the rotation applied about the camera centre, translation unchanged (theory, D1). `None` is the identity extrinsic.
- `pub(in crate::mapping) const ROTATION_STEP_RAD: f64 = 1e-6`, the central-difference step of the primary rotation columns, the same as `GYRO_STEP`.

### C3. Calibration border in the online solver (`vi_ba_schur.rs`)
- **System size:** the dense reduced system grows from `15 P` to `15 P + L` (`vi_ba_schur.rs:1477`). It is dense already, so the border needs no separate block, unlike offline's sparse factor.
- **Observations:**
  - camera `k = observation_views[o]` (0 is the primary) uses its focal `fx (1 + s_k)`, `fy (1 + s_k)`;
  - camera `k ≥ 1` also uses `R_off(k) = Exp(φ_k)` on the left of its composed pose (`views.camera_pose`, `:328-338`), with the rotation columns `J_l(φ)ᵀ (p_c × a)`, `a` the whitened rows `∂r/∂p_c` (`problem.rs:618-625`);
  - camera 0 has no rotation in its projection (D1);
  - the focal column is `(fx x/z, fy y/z)`, whitened (`problem.rs:468-469`);
  - the projection stays in f32 (`:645-736`). The calibration columns are formed in f64 from its `p_c`.
- **Point elimination:** each point's B block gains its calibration columns, accumulated per point as `reduced.rs:252-293` does. `M −= B C⁻¹ Bᵀ` then covers the keyframe-calibration and calibration-calibration blocks (`reduced.rs:466-487`).
- **IMU edges:**
  - every `residual_and_jacobians` call of the calibrated solve (linearisation, trial cost, debug norms; `:1527`, `:1832`, `:1900`) passes `imu_t_bc_with_offset(params.imu_t_bc, θ_0)` in place of `params.imu_t_bc`;
  - the primary rotation columns are central differences of that residual along `θ_0`, step `ROTATION_STEP_RAD`, on the edge's current preintegration;
  - the gyro and clock columns are central differences of the re-integrated residual (`imu_edges.rs:221-235`);
  - all of them couple to both endpoint keyframes (`reduced.rs:417-429`).
- **Priors:** `online_prior_rows` adds rows to the border (`reduced.rs:328-335`).
- **Damping and steps:** the existing Marquardt diagonal (`vi_ba_schur.rs:1682-1685`) covers the border columns, because they are inside `dim`. Point back-substitution includes the calibration step (`reduced.rs:355-375`). `Calibration::retracted` applies it.
- **Cost:** the trial cost (`:1805-1880`) uses the trial calibration: the trial `θ_0` in the extrinsic, and a re-integration at the trial gyro and clock only when `imu_reintegrates()`.

### C4. IMU re-integration from the window stream
- **When:** only with `imu_reintegrates()` (D2). Otherwise each edge keeps today's preintegration and first-order bias correction.
- **Input:** the map edges the window solves over, parallel to `imu_edges` (skeleton contract). For an edge `[t0, t1]` at `td`:
  - samples are the time-sorted union of the window edges' `raw_samples`, deduplicated by stamp (as `map/ops/removal.rs:448-456` merges two edges);
  - the slice taken is the samples in `[t0 − td, t1 − td]`, plus the nearest sample on each side;
  - the result is `PreintegratedImu::from_measurements_with(bias, edge.preintegrated.calib, &corrected_samples(slice, gyro), t0 − td, t1 − td, edge.interpolation)` followed by `widen_bias_walk(edge.bias_walk_extra_s)`.
- **Identity at zero:** at `td = 0` and identity gyro, the slice is what the edge stores (`imu_factor.rs:26-28`), and `Hold` filters to `[t0, t1]` (`imu.rs:193-196`). So GT0's bit-equality is expected.
- **Edge without samples:** an edge with empty `raw_samples` keeps its preintegration and contributes no gyro or clock column. It still contributes its primary rotation columns. The lane counts these edges.

### C5. Gate (`online_calibration.rs`)
- **Rule:** the gate is open when `‖v‖` of the oldest free keyframe is at least 0.2 m/s and the window holds `active_keyframes` free keyframes.
- **Closed:** the solve uses the carried estimate as constants: `θ_0` in the extrinsic, the secondary rotation offsets, the focal and the gyro, plus `td` in the interval shift. It adds no column and returns the estimate unchanged.
- **Before IMU initialisation,** the visual-only BA runs (`local_mapping.rs:75-90`). It neither uses nor changes the estimate.

### C6. The carried estimate
- **Owner:** the skeleton's slot in `local_mapping.rs` holds one `OnlineCalibrationState` per run. It starts at the factory values (all 0) and is replaced only when `merge_and_cull` accepts the update that carries it. A refused or stale merge leaves it unchanged, as `next_prior` is stored only on an applied update (`map/ops/snapshot.rs:58-60`).
- **Lifetime:** it is a device property, so it is kept across tracking losses and map segments.
- **Use:** the next solve starts from it, and the step priors pull toward it.

### C7. Log
- **Line:** one `debug_eprintln!` line per VI-BA solve with the knob on, from the calibrated entry:
  `[online_calib] solve=<n> gated=<0|1> speed=<m/s> free=<k> td_ms=<f> cam<i>_rot_deg=(<x>,<y>,<z>) cam<i>_focal_pct=<f> gyro_scale_pct=(..) gyro_shear_mrad=(..) ba_norm=<f> edges_without_samples=<n> reintegrated=<n>`.
- **Fields:** only the states that are on are printed. `cam0_rot_deg` is `θ_0`, the IMU-side offset; `cam<k>_rot_deg` for `k ≥ 1` is `φ_k`. `ba_norm` is the newest free keyframe's accel-bias norm, for GK3's correlation. `reintegrated` counts the edges re-integrated in the solve, for GT7 and GK2.

## Skeleton contract

Every item below is reachable only with `online_calibration.is_enabled()`. With all `None`, no new code runs (GP0). The `todo!()` bodies belong to Lane C. D1 changes nothing here: the calibrated entry already reads `imu_t_bc` from `ViBaParams`.

**`system/config/mod.rs`:**
```rust
use crate::mapping::global_vi_ba::{
    ClockOffsetPrior, FocalSlicePrior, GyroIntrinsicsPrior, RotationSlicePrior,
};

/// Calibration states the local VI-BA estimates online; each is off when `None`.
/// Priors: `value_*`/`sigma*` to the factory value, `step_*` to the previous solve.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct OnlineCalibrationConfig {
    pub camera_rotation: Option<RotationSlicePrior>,
    pub camera_focal: Option<FocalSlicePrior>,
    pub gyro_intrinsics: Option<GyroIntrinsicsPrior>,
    pub clock_offset: Option<ClockOffsetPrior>,
}

impl OnlineCalibrationConfig {
    /// Whether any state is on.
    pub fn is_enabled(&self) -> bool;   // a real body, not todo!()
}
```
`LocalBaConfig` (`:362`) gains `pub online_calibration: OnlineCalibrationConfig`, set to `OnlineCalibrationConfig::default()` in `Default` (`:389`).

**`system/validation.rs`:**
- In `validate_local_ba` (`:321`), mirror the offline checks (`:703-723`): every sigma passes `POSITIVE`, under `mapping.local_ba.online_calibration.<state>.<field>`.
- In `OrbSlamPipeline::validate` (`:108`): `online_calibration.is_enabled() && !sensors.imu` gives `ConfigError::InertialKnobWithoutImu("mapping.local_ba.online_calibration")` (`:75`).

**`system/config/tests.rs`:**
- the default is all `None`;
- each state parses, for example `(mapping: (local_ba: (online_calibration: (clock_offset: Some((sigma_ms: 2.0))))))`;
- an unknown field is refused;
- a sigma of `0.0` is refused;
- a state on without an IMU is refused.

**`system/build.rs`:** `LocalBaConfig::params` (`:95-119`) copies `online_calibration` into `LocalBaParams`. `LocalBaParams::large` (`mapping/bundle_adjustment.rs:144`) keeps it through `..*self`.

**`mapping/mod.rs`** (D4): add `pub(crate) mod online_calibration;`. The skeleton does not add `mod calibration_states;`: the skeleton must not create that file (below), and a declaration without its file would not build `pr180b/K`. Lane C adds that one line in its move commit.

**`mapping/online_calibration.rs` (new; declared by the skeleton, filled by Lane C):**
```rust
use crate::mapping::global_vi_ba::CalibrationEstimate;
use crate::mapping::map::ImuFactor as MapImuFactor;
use crate::system::OnlineCalibrationConfig;

/// Oldest free keyframe speed (m/s) below which the states are held (VINS-Fusion
/// `estimator.cpp:1042`).
pub(crate) const GATE_MIN_SPEED_MPS: f64 = 0.2;

/// The calibration estimate carried from one local VI-BA to the next, relative to the
/// factory calibration; `Default` is the factory calibration.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OnlineCalibrationState { /* Lane C: values in `Layout::online` order */ }

impl OnlineCalibrationState {
    /// The solved values in the offline report's shape (one slice per camera);
    /// `None` before any ungated solve.
    pub(crate) fn estimate(&self) -> Option<CalibrationEstimate>;   // todo!()
}

/// What one calibrated solve reads besides `ViBaParams`.
pub(crate) struct OnlineCalibrationInput<'a> {
    pub config: OnlineCalibrationConfig,
    pub start: &'a OnlineCalibrationState,
    /// The map edge of each entry of `imu_edges`, in the same order (raw samples,
    /// `t0`, `t1`, interpolation, bias walk).
    pub map_edges: Vec<&'a MapImuFactor>,
    /// `LocalBaParams::active_keyframes` of this solve, for the "window full" gate.
    pub active_keyframes: usize,
}
```

**`mapping/vi_ba_schur.rs`:** a new entry point. The skeleton declares it with a `todo!()` body, and Lane C fills it. `visual_inertial_bundle_adjust` (`:1275`) is unchanged.
```rust
/// [`visual_inertial_bundle_adjust`] with the calibration states of `calibration`:
/// the result, and the state the next solve starts from (the input's when the
/// gate held the states).
#[allow(clippy::too_many_arguments)]
pub(crate) fn visual_inertial_bundle_adjust_calibrated(
    keyframes: &[ViBaKeyframe],
    points: &[Vec3F64],
    observations: &[BaObservation],
    imu_edges: &[ImuFactor],
    wheel_edges: &[WheelOdomFactor],
    stationary_edges: &[StationaryFactor],
    camera: &PinholeCamera,
    params: &ViBaParams,
    calibration: &OnlineCalibrationInput<'_>,
) -> Result<(ViBaResult, OnlineCalibrationState), ViBaError>;
```

**`mapping/bundle_adjustment.rs`** (D4):
- `LocalBaParams` (`:41`) gains `pub online_calibration: OnlineCalibrationConfig`, default all `None`.
- The body of `run_local_inertial_ba_with` (`:564-855`) moves into one private function, so the two entries share it and cannot drift apart:
  ```rust
  #[allow(clippy::too_many_arguments)]
  fn run_local_inertial_ba_inner(
      snapshot: BaSnapshot,
      camera: &PinholeCamera,
      imu_t_bc: Option<Pose3d>,
      gravity_world: Vec3F64,
      wheel_factors: &[WheelOdomFactor],
      stationary_factors: &[StationaryFactor],
      params: &LocalBaParams,
      calibration: Option<(&OnlineCalibrationState, &OnlineCalibrationConfig)>,
  ) -> BaUpdate
  ```
- Both public functions are thin wrappers around it:
  - `run_local_inertial_ba_with` (`:564`) keeps its signature and passes `None`, so its callers and tests are untouched;
  - a new `pub(crate) fn run_local_inertial_ba_calibrated(snapshot, camera, imu_t_bc, gravity_world, wheel_factors, stationary_factors, params: &LocalBaParams, calibration: &OnlineCalibrationState) -> BaUpdate` passes `Some((calibration, &params.online_calibration))` when `params.online_calibration.is_enabled()`, and `None` otherwise.
- In the inner function, with `calibration` `Some`:
  - build `map_edges` inside the same `filter_map` at `:671-688`, so the two vectors stay aligned;
  - call `visual_inertial_bundle_adjust_calibrated` in place of `:773`, and handle its `ViBaResult` as `:773` does today;
  - put the returned state on the update.
- With `None`, the inner function runs today's code. GP0's `pr180b/K` against `c0a1dc3` `cmp` proves the refactor changes no output.

**`mapping/map/ops/snapshot.rs`** (D4):
- `BaUpdate` (`:52`) gains `pub(crate) online_calibration: Option<OnlineCalibrationState>`, set by the calibrated run and `None` otherwise. Its doc says "stored only if this update is applied", as `next_prior`'s does.
- `LocalBaStats` (`:93`, `#[non_exhaustive]`) gains `pub calibration_gated: bool` and `pub calibration_columns: usize`.

**`mapping/local_mapping.rs`** (the plan's calibration-state slot):
- `solve_snapshot` (`:75`) and `solve_counted` (`:164`) take `calibration: &OnlineCalibrationState`, and call `run_local_inertial_ba_calibrated` when `job.ba_params().online_calibration.is_enabled()`.
- `LocalMappingBackend::Synchronous` holds `calibration: Mutex<OnlineCalibrationState>`. The asynchronous worker (`:326-400`) owns an `OnlineCalibrationState` local.
- Both take `update.online_calibration` out before `merge_and_cull` (`:199`) and store it only when the merge returns `Ok`.
- **One skeleton commit with card E.** The same `local_mapping.rs` commit adds card E's deferred keyframe path: `PendingState` (`:291-294`) gains `deferred: VecDeque<DeferredKeyframe>`, and the worker loop gains a branch with a `todo!()` body, reachable only when the `DeferredKeyframeContext` is `Some` (`keyframe_orb: Mapping`). See card E, Skeleton contract.
- **The deferred path keeps calibration on.** Its BA job (card E step 12.6) calls `solve_counted(…, &calibration)` with the same `OnlineCalibrationState` as the `latest` path, under the same store-on-`Ok` rule. So `keyframe_orb: Mapping` never skips online calibration. With `keyframe_orb: Mapping` the worker also runs every deferred keyframe in FIFO order, so GP12's "latest job only" holds only with `keyframe_orb: Tracking`.
- Lane T fills the deferred bodies. Lane C does not touch `local_mapping.rs`, so the hunks do not conflict.

**`CHANGELOG.md`:** one line for `mapping.local_ba.online_calibration`, off by default.

**Not in the skeleton:**
- `calibration_states.rs`: Lane C creates it by `git mv`, so the skeleton must not create it, and the old definitions never exist twice;
- `Layout::online`, `Layout::imu_reintegrates`, `online_prior_rows`, `imu_calibration_columns`, `corrected_samples`, `imu_t_bc_with_offset` and `ROTATION_STEP_RAD`: these are internal to `crate::mapping`.

## Files Lane C touches

- **Moved:** `mapping/global_vi_ba/calibration.rs` becomes `mapping/calibration_states.rs`, and gains C1's helpers and C2.
- **One line:** `mapping/mod.rs` gains `mod calibration_states;` in the move commit (D4).
- **Filled:**
  - `mapping/online_calibration.rs`: the bodies, the gate, the state values and the log line;
  - `mapping/vi_ba_schur.rs`: the calibrated entry, the border, the IMU-side rotation, the re-integration and the tests GT0-GT7.
- **Imports only:** `mapping/global_vi_ba/{mod.rs, problem.rs, reduced.rs, imu_edges.rs}`. `imu_edges.rs` also calls the two moved helpers, with the same operations.
- **Not touched:** `system/*`, `mapping/{bundle_adjustment.rs, local_mapping.rs, map/*}`, the rest of `mapping/mod.rs`, `crates/kornia-factors/*`, tracking, `configs/*.ron`, `CHANGELOG.md`.
- **Commits:** two, so GK7 can revert the move alone.
  - `refactor(mapping): calibration states shared by the offline and online solves` (C1, offline outputs bit-identical);
  - `feat(mapping): online calibration states in the local VI-BA` (C2-C7).

## Decisions

- **D1 (owner, 2026-11-17): the primary camera's rotation offset lives on the IMU side.** It is a rotation of `imu_t_bc` inside the IMU residual, `imu_t_bc · Exp(−θ_0)`, as VINS-Fusion places its extrinsic between body and camera. The primary projection carries no calibration rotation, so keyframe poses and tracking share one frame (theory, GT6, GP7, GK5). The column is a central difference of `preintegration::residual_and_jacobians`, with no re-integration. The secondary cameras keep the rig-relative offset on `camera_from_primary`, which changes no keyframe's frame. Tracking keeps the factory calibration. The clock offset is unchanged.
- **D2: re-integrate only when a state needs it.** Online, edges are re-integrated only when `clock_offset` or `gyro_intrinsics` is on (`Layout::imu_reintegrates`, C2). `camera_rotation` and `camera_focal` do not enter the preintegration, so they keep today's first-order bias correction. The offline `Layout::reintegrates` is unchanged, so `offline-ba` keeps its bits (GP0). This removes the re-integration cost from GP11's `rot` row and the re-integration confound from a `rot` or `focal` score change (GK2).
- **D3: the gyro knob stays unseeded.** `GyroIntrinsicsPrior { sigma }` is shared with the offline config (`global_vi_ba/mod.rs:127-129`); a seed field would change the offline RON format. The online gyro state starts at identity, and its value prior pulls to identity. The plan's "seed from the offline result" (section 9 table) is not built in this round. The knob is predicted to fail its kill rule anyway (GP9, GK6).
- **D4: files outside every lane go to the skeleton.** `mapping/bundle_adjustment.rs`, `mapping/map/ops/snapshot.rs` and `mapping/mod.rs` are in no lane's list in plan section 7, so the skeleton owns them. One exception: Lane C adds the line `mod calibration_states;` to `mapping/mod.rs` in its move commit, because the skeleton must not create the file and the skeleton tag must build. With Lane F deferred, no other lane edits `bundle_adjustment.rs` this round.
- **Gyro and clock step priors (a reading, not a blocker).** The plan's section 8 types give the gyro and clock one sigma each, which this card reads as a value prior (as offline). Their information therefore does not carry from solve to solve beyond the start value. GP5's spread assumes that.

## Not in this card

- Feeding the estimates to tracking (projection, `camera_from_primary`, IMU prediction) and to the visual-only BA. D1 keeps tracking on the factory calibration; the theory bounds what that costs.
- A marginal prior for the calibration states (VINS-Fusion's marginalisation). The step prior stands in for it.
- An analytic `td` column (VINS-Fusion's feature velocity, or `∂Δ/∂td` inside `PreintegratedImu::integrate`). It is the GK8 fallback, for a later card.
- An analytic primary-rotation column in `kornia-factors`. The central difference costs under 0.1 ms per solve (GP11).
- Seeding the gyro intrinsics from the offline result (D3).
- Thermal and accelerometer intrinsics (`lamaria-handoff.md:42`).
- The raw-fisheye path (card F), deferred by the owner on 2026-11-17 to its own round.
