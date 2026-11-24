# Design: sliding-window VI odometry (2x-4)

Status: design approved by the owner at stop S-B, 2026-11-24 (all 12 open questions as recommended; see the end). Written read-only: nothing below is measured unless a source is cited. Implementation on branch `feat/2x-vio`; results in `experiments/euroc-2x-results.md`.

**Goal.** A Basalt-style fixed-lag visual-inertial smoother in a new module `crates/kornia-slam/src/odometry/`, behind `pipeline.odometry`, default `MapTracking` (today's behaviour, bit-identical).
Rung 2x-5 runs it in `Observe` mode beside the pipeline and writes `traj_vio.txt`; `Drive` parses and is rejected until 2x-6.

Paths: `R` = `/home/edgar/dev/split/kornia-slam-graph`, `K` = `R/crates/kornia-slam/src`, `F` = `R/crates/kornia-factors/src`, `SN` = `R/crates/kornia-sensors/src`, `A` = `R/apps/kornia-slam-app/src`, `P` = `/home/edgar/dev/split/.pr180-fix`, `E` = `P/euroc-top1`, `B` = `/home/edgar/dev/.flux-connect-scratch/bench`, `BA` = `B/src/basalt`, `VIO` = `BA/src/vi_estimator/sqrt_keypoint_vio.cpp`, `OF` = `BA/include/basalt/optical_flow/frame_to_frame_optical_flow.h`.

## 0. State found (read-only)

- `R` HEAD is f26560a (5 ahead of origin); the plan is now in the repo as `R/experiments/euroc-2x-plan.md`. `K/mapping/vi_ba_window_tests.rs` is still dirty: not touched by this card, never staged.
- Lane A worktree `/home/edgar/dev/split/kornia-slam-2x-a` (branch `feat/2x-vio`) does not exist yet.
- `basalt-headers` is not on disk (`BA/thirdparty` has only apriltag, ros, vcpkg). Basalt's preintegration scheme and its `StereographicParam` are unverified; we use our own preintegration and the standard stereographic formula, each covered by a test.
- `tests/fixtures/upstream/frontend_opencv.bin` is not in the tree. The three OpenCV parity tests are `#[ignore]` (`K/upstream/kornia_imgproc/lk_window.rs:896`, `good_features.rs:647`, `corner_subpix.rs:289`).
- `RigFlow::step` returns early on an empty track set (`K/tracking/flow_tracker.rs:886-888`), so refill never runs on frame 0. `plan_refill` rotates over cameras (`:956`). There is no cross-camera track. Three small additions are needed (section 6).
- The survivor filter caps per-frame movement at 60 px from the previous pixel, not from the gyro-rotated start (`K/tracking/optical_flow.rs:494-510, 846-875`; called at `flow_tracker.rs:373-381`). 60 px is 2.76 rad/s at f = 435 px and 20 Hz. This is a risk for V2_03 (section 12).
- `marginalization.rs` is a fixed 15×15, IMU-only Schur on `ViBaKeyframe` (`K/mapping/marginalization.rs:221-317`). Its conventions are reusable; its function is not general.
- Basalt's Pattern51 is Pattern52 scaled by 0.5: 52 samples within ±3.5 px (`BA/include/basalt/optical_flow/patterns.h:110-156`).

## 1. Module layout, types, knob

### Files (`K/odometry/`, all new; each under about 400 lines)

| file | responsibility | reuses |
|---|---|---|
| `mod.rs` | `SlidingWindowOdometry` facade: IMU queue, frontend, window, counters, timings; `OdometryOutput` | - |
| `config.rs` | `Odometry`, `SlidingWindowConfig`, `OdometryMode`, `OdometryImuNoise`, `OdometryTuning` (constants, each with its Basalt citation) | `SN/imu.rs:37` `ImuCalib` |
| `state.rs` | `FrameId`, `StateWithLin` (15 DOF), `PoseWithLin` (6 DOF), backup / restore | `FrameNavState` and its `plus` (`K/tracking/pose_estimation/pose_inertial.rs:44-48, 190-205`) |
| `landmarks.rs` | `Landmark`, `LandmarkStore`: host, bearing, inverse distance, observations; creation at keyframes; removal rules | `kornia_3d::pose::triangulate_n_view_linear` (used at `K/frontend/overlap.rs:214`) |
| `init.rs` | first state from the first accelerometer sample; initial prior | `PreintegratedImu::predict` (`SN/imu.rs:429-450`) |
| `keyframes.rs` | keyframe rule; `plan_marginalization` (which states and keyframes leave) | - |
| `linearize.rs` | variable order; IMU blocks; visual rows; prior block; per-landmark Schur; cost | `kornia_factors::inertial::preintegration::{residual_and_jacobians, information}` (`F/inertial/preintegration.rs:72, 187`); new `kornia_factors::reprojection::anchored_*` built on `residual_and_pose_jacobian` (`F/reprojection.rs:11`); `HuberKernel` (`kornia_3d::ransac`) |
| `solve.rs` | LM loop, back-substitution, `SolveReport` | `DenseLlt` with `Par::Seq` (`K/mapping/dense_llt.rs:40-80`) |
| `marginalize.rs` | `MargPrior`; apply a `MargPlan`; FEJ flags; re-centring; nullspace check | new generic `schur_complement` in `K/mapping/marginalization.rs` |
| `window.rs` | `Window`: the per-frame sequence (predict, add observations, keyframe, landmarks, optimize, marginalize) | all of the above |
| `frontend.rs` | `FlowFrontend`: two-camera KLT adapter over `RigFlow`, stereo link, row gate | `RigFlow`, `TrackSet` (`K/tracking/flow_tracker.rs:675`, `K/tracking/optical_flow.rs:146`) |
| `synthetic.rs` | `#[cfg(any(test, feature = "bench_internals"))]` stereo-inertial walk | `HeadWalk`, `Gauss` (`K/mapping/synthetic_walk.rs:101-207`) |
| `tests/{marginalize,window,frontend}.rs` | section 9 | - |

System glue, one new file: `K/system/odometry.rs` (`OdometryRunner`: owns `SlidingWindowOdometry`, the last output, error counter). `K/system/mod.rs` gains only the field, the build line and three call sites.

### Public types and signatures

```rust
// config.rs (re-exported from `system` and the crate root)
pub enum Odometry { MapTracking /* default */, SlidingWindow(SlidingWindowConfig) }
pub struct SlidingWindowConfig { pub mode: OdometryMode, pub imu_noise: OdometryImuNoise }   // serde(default, deny_unknown_fields)
pub enum OdometryMode { Observe /* default */, Drive }
pub enum OdometryImuNoise { Rig /* default: SensorRig::imu_noise() */, Explicit(ImuCalib) }
pub struct OdometryTuning { /* Rust only, inside OrbTuning (serde(skip)); table in section 3 */ }

// mod.rs
pub struct OdometryOutput { pub timestamp_sec: f64, pub pose_world_to_cam: Pose3d, pub velocity_world: Vec3F64,
                            pub bias: ImuBias, pub keyframe: bool, pub optimized: bool }
#[non_exhaustive] pub struct OdometryCounters { /* section 7 */ }
pub(crate) struct SlidingWindowOdometry { .. }
impl SlidingWindowOdometry {
    pub(crate) fn new(config: &SlidingWindowConfig, tuning: &OdometryTuning, rig: &SensorRig,
                      interpolation: SampleInterpolation) -> Result<Self, OdometryError>;
    pub(crate) fn push_imu(&mut self, samples: &[ImuMeasurement]);
    pub(crate) fn process_pair(&mut self, timestamp_sec: f64, left: &Image<u8, 1>, right: &Image<u8, 1>)
        -> Result<OdometryOutput, OdometryError>;
    pub(crate) fn counters(&self) -> &OdometryCounters;
}

// window.rs: the estimator below the frontend; the synthetic tests drive this
pub(crate) struct Observation { pub landmark: LandmarkId, pub camera: u8, pub pixel: [f64; 2] }
pub(crate) struct FrameInput<'a> { pub id: FrameId, pub imu: Option<PreintegratedImu>, pub observations: &'a [Observation] }
impl Window {
    pub(crate) fn new(tuning: &OdometryTuning, cameras: [StereoView; 2], t_bc: Pose3d, gravity: Vec3F64) -> Self;
    pub(crate) fn initialize(&mut self, id: FrameId, first_accel: Vec3F64);
    pub(crate) fn add_frame(&mut self, input: FrameInput<'_>) -> Result<FrameReport, OdometryError>;
    pub(crate) fn newest(&self) -> &StateWithLin;
}

// linearize.rs / solve.rs / marginalize.rs / keyframes.rs
pub(crate) fn linearize(window: &WindowView<'_>, order: &Order, scratch: &mut LinearSystem) -> f64;       // returns the cost
pub(crate) fn cost(window: &WindowView<'_>, order: &Order) -> f64;
pub(crate) fn optimize(window: &mut Window, scratch: &mut SolveScratch) -> SolveReport;
pub(crate) fn plan_marginalization(view: &WindowView<'_>, connected: &KeyframeCounts) -> Option<MargPlan>;
pub(crate) fn marginalize(window: &mut Window, plan: &MargPlan, lost: &[LandmarkId]) -> Result<(), OdometryError>;
pub(crate) fn nullspace_residual(prior: &MargPrior, view: &WindowView<'_>, gravity: Vec3F64) -> [f64; 6]; // x, y, z, roll, pitch, yaw

// SlamSystem (K/system/mod.rs)
pub fn odometry_output(&self) -> Option<&OdometryOutput>;
pub fn odometry_counters(&self) -> Option<&OdometryCounters>;
```

New in `F/reprojection.rs` (kornia-factors owns factors):

```rust
pub struct AnchoredLandmark { pub bearing: [f64; 2], pub inverse_distance: f64 }   // stereographic bearing, 1/m, host camera frame
pub struct AnchoredRow { pub residual: SVecF64<2>, pub d_target: SMatF64<2, 6>, pub d_landmark: SMatF64<2, 3> } // d_host = -d_target
pub fn anchored_residual(host_cw: &Pose3d, target_cw: &Pose3d, lm: &AnchoredLandmark, pixel: [f64; 2],
                         camera: &PinholeCamera, min_depth: f64) -> Option<SVecF64<2>>;
pub fn anchored_residual_and_jacobians(/* same */) -> Option<AnchoredRow>;
```

### Knob: exact RON shape

Field `odometry: Odometry` on `OrbSlamPipeline` (`K/system/config/mod.rs:105-118`; the struct is already `serde(default, deny_unknown_fields)`, so existing files parse unchanged).

```ron
pipeline: OrbSlam((
    odometry: MapTracking,                                  // default; may be omitted
    odometry: SlidingWindow(( mode: Observe )),             // 2x-5, rig (datasheet) IMU noise
    odometry: SlidingWindow(( mode: Observe, imu_noise: Explicit(( gyro_noise: 2.82e-4, accel_noise: 1.6e-2,
              gyro_bias_noise: 1.0e-4, accel_bias_noise: 1.0e-3 )) )),   // Basalt's values, BA/data/euroc_ds_calib.json:187-206
    odometry: SlidingWindow(( mode: Drive )),               // parses; rejected until 2x-6
))
```

Validation (`K/system/validation.rs`, new `validate_odometry`, called from `OrbSlamPipeline::validate` `:150`):

| rule | error |
|---|---|
| `SlidingWindow` without `sensors.imu` | `ConfigError::InertialKnobWithoutImu("odometry")` (`:86`) |
| `SlidingWindow` with `sensors.cameras != Stereo` | new `ConfigError::OdometryNeeds { needs: "sensors.cameras: Stereo (a rectified pair)" }` |
| `mode: Drive` | `OdometryNeeds { needs: UNBUILT }` (`:588`) |
| `imu_noise: Explicit`: any of the four values not finite or ≤ 0 | `check("odometry.imu_noise.<field>", v, POSITIVE)` (`:1087`) |

Header (`Display`, `config/mod.rs:818`): one line only when not default, `  odometry: sliding window, Observe, IMU noise Rig`. Knob off prints nothing, so `HEADER_OK` holds.

## 2. State

**Conventions are ours, so the factors are reused unchanged.** Pose is `T_cw` (world to left camera), perturbed on the right, `T_cw · Exp([ρ; ω])`. Tangent layout `[ρ ω v b_g b_a]`, velocity in the world (`F/inertial/preintegration.rs:1-16`). World gravity is `(0, 0, −9.81)`; the world origin is the first body position.

| item | value | Basalt |
|---|---|---|
| frame states | at most 3, each 15 DOF, keyed by timestamp in ns | `vio_max_states 3`; `VIO:118, 328` |
| keyframe poses | at most 7 keyframes in total, counting keyframes still held as frame states; 6 DOF once they leave the states | `vio_max_kfs 7`; `VIO:119, 385, 612` |
| variable order | keyframe poses by time, then frame states by time; the prior is the leading block | `VIO:1033-1055, 936-951` |
| landmarks | host = camera 0 of a keyframe; stereographic bearing (2) + inverse distance (1); never shared with the map | `BA/include/basalt/vi_estimator/landmark_database.h:55-66`; `VIO:387, 446-451` |
| reduced system | at most 6·7 + 15·3 = 87 | - |

**First-estimate state** (`state.rs`), as `BA/include/basalt/utils/imu_types.h:112-138`: `{ lin: FrameNavState, delta: [f64; 15], linearized: bool }`. Free state: an increment moves `lin`. Linearized state: `delta += inc`, value = `lin ⊞ delta`. `PoseWithLin` is the 6-DOF head of the same (`imu_types.h:209-216`).

**Landmark creation, keyframes only** (`VIO:380-465`): for each camera-0 track with no landmark, collect its observations in every window frame and camera. Triangulate against the first one whose camera centre is at least 0.05 m from the host camera (`:412-439`); the same-frame right camera qualifies at once (baseline 0.11 m). Accept when finite and `0 < inverse_distance < 3.0` (`:444-445`). Then add all its observations (`:458-461`), the host's own pixel included.

**Which states are marginalized when** (`keyframes.rs::plan_marginalization`):

| # | rule | Basalt |
|---|---|---|
| M0 | No optimization until more than 4 frame states exist; no marginalization before the first optimization | `VIO:1017, 555` |
| M1 | Trigger each frame: keyframe poses > 7, or frame states ≥ 3 | `VIO:559` |
| M2 | `remove = states − 3 + 1`. The state after those is `last_state_to_marg`: it stays, becomes the prior's 15-DOF state, and is marked linearized | `VIO:562-566, 905-907` |
| M3 | An older state that is a keyframe loses velocity and bias (9 DOF); its pose stays as a keyframe pose | `VIO:590-592, 802-806, 916-923` |
| M4 | An older state that is not a keyframe is marginalized whole (15 DOF) | `VIO:593-595, 799-801, 910-914` |
| M5 | While keyframes > 7 and a keyframe state is converting this step: drop the first keyframe (oldest first, never the newest 2) whose tracked share `connected / created` is under 0.1 | `VIO:612-631`; `vio_kf_marg_feature_ratio` |
| M6 | Else drop the keyframe with the lowest score `sqrt(d(i, newest kf)) · Σ_j 1 / (d(i, j) + 1e-5)`, never the newest 2 | `VIO:638-671` |
| M7 | Enter the marginalization: the old prior; every IMU factor whose two states are both in the order; every landmark hosted in a dropped keyframe; every lost landmark (no observation in the current frame, any camera) | `VIO:718-731, 470-482`; `BA/src/linearization/linearization_abs_qr.cpp:117-127` |
| M8 | Of those landmarks, only observations whose frame is in the order enter; the newest state's are dropped | `BA/include/basalt/linearization/landmark_block_abs_dynamic.hpp:72-77` |
| M9 | Afterwards: delete those landmarks; delete every observation in a removed frame; delete landmarks left with fewer than 2 observations; erase the IMU factors of removed states | `BA/src/vi_estimator/landmark_database.cpp:70-93`; `landmark_database.h:153`; `VIO:910-934` |
| M10 | New prior order: remaining keyframe poses, then `last_state_to_marg` | `VIO:936-951` |

Observations that surviving landmarks had in a removed non-keyframe state are dropped without entering the prior (M9). That loses information and stays consistent.

## 3. Factors

**Constants** (`OdometryTuning`, initial values are Basalt's; they move to sensor-derived values in the cleanup card):

| constant | value | source |
|---|---|---|
| pixel σ | 0.5 px | `vio_obs_std_dev`, `BA/src/utils/vio_config.cpp:68`; `VIO:72` |
| visual Huber | 1.0 px on the raw residual norm (χ² knee 4 at σ 0.5) | `vio_obs_huber_thresh`, `vio_config.cpp:69`; `landmark_block_abs_dynamic.hpp:474-486` |
| IMU kernel | none | `BA/include/basalt/linearization/imu_block.hpp:54-67` |
| triangulation baseline | ≥ 0.05 m | `vio_min_triangulation_dist`, `VIO:412-439` |
| inverse distance bound | < 3.0 (0.33 m) | `VIO:444-445` |
| keyframe rule | tracked share < 0.7 and more than 5 frames since the last | `VIO:370-373`; `vio_config.cpp:63-64` |
| LM | 7 iterations, λ 1e-4, min 1e-6, max 1e2 | `vio_config.cpp:78-80`; `VIO:1062, 1097` |
| initial gauge weight | 1e8 on position and yaw | `VIO:87-91`; `vio_config.cpp:84` |
| initial bias σ | gyro 0.316 rad/s, accel 0.1 m/s² (Basalt's effective values; its two diagonal segments are swapped against its comments) | `VIO:93-97`; `vio_config.cpp:85-86` |
| round trip | ≤ 0.2 px | `optical_flow_max_recovered_dist2 0.04`; `OF:222-228` |
| row gate | `0.005 · fy` px (2.2 px at fy 435) | `optical_flow_epipolar_error 0.005` on a unit-baseline essential matrix, `OF:367-378`; `BA/include/basalt/utils/keypoints.h:73-79` |
| detection | grid 50 px: 752·480 / 50² = 144 cells | `optical_flow_detection_grid_size`, `OF:320-321` |
| KLT | 4 levels (0..=3), 5 iterations per level | `optical_flow_levels 3`, `optical_flow_max_iterations 5`, `OF:251-306` |

**IMU factor.** One per consecutive state pair. Residual and Jacobians from `residual_and_jacobians(i, j, pim, gravity, Some(&t_bc))`; information from `information(pim, 1.0).omega`: the full inverse of the 9×9 navigation covariance plus the diagonal bias walk `1 / (σ_rw² · dt)` (`F/inertial/preintegration.rs:187-209`). Basalt weights the same way (`imu_block.hpp:55-105`). `weight_and_cost` is not called.
- **Integration.** `PreintegratedImu::from_measurements_with(bias_of_start_state, noise, samples, t0, t1, interpolation)` (`SN/imu.rs:224`) when the frame arrives, at the start state's current bias (`VIO:227-229`). `Hold` matches Basalt's sample handling (`VIO:242-256`).
- **Bias re-linearization.** First order, inside the residual: `delta_*_with_bias(&i.bias)` (`SN/imu.rs:291-310`). No re-integration, as Basalt (`imu_block.hpp:42-51`): a factor lives 3 frames, then it is in the prior.
- **FEJ.** Jacobians from one call with every linearized state at `lin` (bias included). If either state is linearized, the residual comes from a second call at the current states (`imu_block.hpp:44-54`).

**Visual factor.** Rectified pinhole pixels, both cameras with `rig.camera`; the right camera's `camera_from_primary` is `(I, [−b, 0, 0])`, `b = rig.stereo_baseline_m`. For a landmark `(u, v, ρ)` hosted in keyframe `h`, observed in frame `t`, camera `c`, at pixel `z`:

```text
bearing(u, v) = [2u, 2v, 1 − u² − v²] / (1 + u² + v²)
q_w = R_w,h · bearing + ρ · t_w,h                 (ρ times the world point; finite at ρ = 0)
r   = π( R_c,w(t) · q_w + ρ · t_c,w(t) ) − z      with T_c,w(t) = T_c,c0 · T_cw(t)
∂r/∂ξ_t = dπ · [ ρ R_c,w , −R_c,w [q_w]× ]        ∂r/∂ξ_h = −∂r/∂ξ_t
∂r/∂(u, v) = dπ · R_c,h · ∂bearing/∂(u, v)        ∂r/∂ρ = dπ · t_c,h
```

`anchored_residual_and_jacobians` gets this from `residual_and_pose_jacobian` called with the pose `(R_c,w, ρ t_c,w)` and the point `q_w`; its first three Jacobian columns are scaled by ρ.

| row kind | `t`, `c` | constrains |
|---|---|---|
| host | `t = h`, 0 | bearing only (pose-free) |
| stereo | `t = h`, 1 | inverse distance (pose-free: host and target columns cancel) |
| left temporal | `t ≠ h`, 0 | host pose, target pose, landmark |
| right temporal | `t ≠ h`, 1 | the same, through the fixed extrinsic |

- **Weight.** `w = HuberKernel.weight(|r|² / σ², 4.0) / σ²`, cost `½ HuberKernel.rho(...)`. Weights are fixed within one linearization.
- **FEJ.** Jacobians with linearized poses at `lin`, the landmark at its current value; residual at the current poses. Basalt takes the relative-pose value at the current state and only the relative-to-absolute Jacobians at the lin point (`linearization_abs_qr.cpp:206-221`); equal to first order (open question 1).
- **Cheirality.** A row whose projection is invalid is skipped in the linearization and in the cost (`landmark_block_abs_dynamic.hpp:162`), and counted.

**Outlier handling.** No erase and no chi² gate in the window: Basalt's VIO calls no outlier filter (`VIO:1405`). Outliers stop at the frontend, before any pose exists: round trip ≤ 0.2 px on every temporal and stereo track (`OF:214-228`), then the row gate on every left-right pair of the frame (`OF:346-386`). Inside the window only Huber acts.

## 4. Marginalization

1. **Joint system** over the order of M2-M7: prior block `H_p`, `b_p + H_p δ` (`BA/src/vi_estimator/ba_base.cpp:408-476`); the IMU factors of M7; the visual rows of M7-M8 with Huber weights at the current residuals (`VIO:710-733`).
2. **Landmarks first:** 3×3 Schur per landmark into the pose system.
3. **Schur complement into a dense prior:** `H' = H_kk − H_km H_mm⁻¹ H_mk`, `b' = b_k − H_km H_mm⁻¹ b_m`, symmetrized (as `K/mapping/marginalization.rs:288-294`). `H_mm` by faer Cholesky. If it is not positive definite: pseudo-inverse from a symmetric eigendecomposition, cut at `n · ε · λ_max`, counted in `marg_pinv` (Basalt always uses a pseudo-inverse, `BA/src/vi_estimator/marg_helper.cpp:103-104`).
4. **First-estimate Jacobians.** `last_state_to_marg` gets `linearized = true` with `delta = 0` (`VIO:905-907`). Every variable in the prior is linearized from then on; an assertion checks it (`ba_base.cpp:319-327`). Kept keyframe poses keep the lin point they got as `last_state_to_marg`.
5. **Re-centre:** `b' −= H' δ` (`VIO:959-984`), so the stored prior is `½ δᵀH'δ + b'ᵀδ` in `δ = state ⊟ lin`.
6. **Storage** (`MargPrior`): `order: Vec<(FrameId, Pose | State, offset)>`, `h: faer::Mat<f64>` (at most 57 × 57 = 6·7 + 15), `b: Vec<f64>`. Lin points live only in the states. Buffers are allocated once at 87 × 87 and reused.
7. **Initial prior** on the first state (`VIO:76-110, 141-143, 203-209`): information 1e8 on the body position rows `[−I₃ 0]` (body origin at 0, `F/inertial/preintegration.rs:110-111`) and on the yaw row `ĝᵀω`; bias diagonals `1/0.316²` and `1/0.1²`; `b = 0`. The first state is linearized from the start. No prior on roll, pitch or velocity.

**4-DOF nullspace check.** In our parameterization a world transform is the same right increment on every pose, so the nullspace columns are simple:

```text
translation along e_k:  ρ = −e_k on every pose and state
yaw about ĝ through the centroid m of the lin positions:  ρ = ĝ × m,  ω = −ĝ on every pose and state;  δv = ĝ × v_lin on every state
biases and landmarks: 0 (landmarks are host-anchored)
```

- A twin prior `nullspace_prior` starts at zero (no gauge rows) and goes through the same marginalizations (`VIO:82-84, 823-880`).
- Reported per direction: `nᵀHn / (‖n‖² ‖H‖_F)` and `|nᵀb| / (‖n‖ ‖b‖)`, after `BA/src/vi_estimator/sqrt_ba_base.cpp:43-228`.
- Pass: x, y, z, yaw ≤ 1e-9; roll and pitch ≥ 1e3 times the yaw value.
- Visual rows keep the nullspace at any evaluation point (`∂r/∂ξ_h = −∂r/∂ξ_t`). The IMU factor keeps it only if `v_lin` is one value per state across all factors: that is what FEJ protects.
- The twin runs in tests and in `Observe`; its time is reported apart from `vio_ms`.

## 5. Solver

Information form, f64, landmarks eliminated by 3×3 Schur, the reduced system (≤ 87) factored by `DenseLlt::new(dim, Par::Seq)`.

| step | rule | Basalt |
|---|---|---|
| start | λ = 1e-4 at every solve; λ_vee = 2 | `VIO:1062`; `BA/include/basalt/vi_estimator/sqrt_keypoint_vio.h:251-252` |
| budget | attempts counted while `it ≤ 7`: at most 8, accepted or rejected; Jacobians are rebuilt only after an accept | `VIO:1097, 1159, 1357, 1389` |
| damping | `H_red + max(λ · diag(H_red), 1e-6)`; landmarks undamped | `VIO:1212-1214` |
| step | `inc_p = −H_d⁻¹ b_red`; `inc_l = −H_ll⁻¹ (b_l + H_lp inc_p)` | `VIO:1216-1217`; `landmark_block_abs_dynamic.hpp:275` |
| model decrease | `l = −b_redᵀ inc_p − ½ inc_pᵀ H_red inc_p + ½ Σ b_lᵀ H_ll⁻¹ b_l`, with the undamped `H_red` | `imu_block.hpp:210-222` |
| cost | Huber vision + IMU + bias walk + prior at δ | `VIO:1269-1291` |
| accept | `l > 0` and `(F − F_new) / l > 0`; then `λ ← max(1e-6, λ · max(1/3, 1 − (2q − 1)³))`, λ_vee ← 2 | `VIO:1320-1321, 1350-1355` |
| converged | `0 < F − F_new < 1e-6`, or `‖inc_p‖∞ < 1e-4` | `VIO:1360-1361, 1266` |
| reject | restore; `λ ← λ · λ_vee`; `λ_vee ← 2 λ_vee`; stop when `λ > 1e2` | `VIO:1382-1396` |

**Fallbacks.**
- Cholesky fails or the step is not finite: raise λ as a reject, at most 3 times per attempt (`VIO:1207-1227`). Then the solve ends and the states keep the IMU prediction; `solve_failures += 1`.
- A landmark whose `H_ll` fails Cholesky after a relative ridge of `1e-9 · max diag`: held fixed for this solve (its pose rows stay), counted in `landmarks_fixed`; at marginalization it is dropped without contributing.
- No visual rows (blackout): the IMU factor and the prior carry the frame. This is the coast.
- An inter-frame interval without IMU samples (`pim.dt ≤ 0`), or a non-finite state or prior: reinitialize as on frame 0; `resets += 1`. Any reset on EuRoC is a finding.

**Time budget.** Solve p50 ≤ 8 ms on one laptop core, flow p50 ≤ 8 ms (plan section 5). Estimate, not measured: about 150 landmarks × at most 20 observations is 3000 rows, about 1 ms per linearization, 2-4 linearizations typical. There is no wall-clock exit: it would break run-to-run identity. The budget is held by the 8-attempt cap and the bounded track count, and measured by P5.

## 6. Frontend adapter (`frontend.rs`)

`FlowFrontend` owns its own `RigFlow` and `TrackSet`, whatever `frontend.klt_kernel` says, so the main path's flow is untouched.

**Settings** (`RigFlowSettings`, `K/tracking/flow_tracker.rs:583-600`): `threads: 1` (no worker is spawned, `K/tracking/flow_pool.rs:80`), `window: 11`, `unpredicted: (3, 5)`, `epsilon` as `PyrLKParams::default()`, `retry: Off`, `forward_backward: Enabled { max_round_trip_px: 0.2, max_iter: 5, every_n_frames: 1 }`, `refill: Enabled { per_camera: 144, min_spacing_px: 40.0, refill_below: 1.0, border_px: 10 }`, `anchor_guard: Off`, `gyro_guess: RotatedRay`, new `refill_cameras: PrimaryOnly`. Cameras: `[left, right]` as in section 3.

**Three additions to `flow_tracker.rs`** (legacy defaults unchanged; guarded by `IDENT_R01_OK`, whose preset runs this file):
1. `RigFlowSettings.refill_cameras: RefillCameras { RoundRobin /* today */, PrimaryOnly }`, read in `plan_refill` (`:956`). Four literal constructors get the field: `flow_tracker.rs:1176`, `K/system/mod.rs:3714`, `R/crates/kornia-slam/tests/flow_zero_alloc.rs:99, 154`; `from_frontend` sets `RoundRobin`.
2. `RigFlow::seed(&mut self, camera: usize) -> Result<&[(usize, [f32; 2])], FlowPoolError>`: runs `FlowShard::refill` (`:478-534`) on the current frame with no occupied pixels. Needed because `step` returns early on an empty set (`:886-888`).
3. `RigFlow::track_across(&mut self, from: usize, to: usize, pixel: [f32; 2], start: [f32; 2]) -> Option<[f32; 2]>`: template of `pixel` in the current frame of `from`, coarse-to-fine into the current frame of `to` (the loop of `forward`, `:350-370`), `KltTracker::accepts`, then the round trip back at 0.2 px (the check of `keep`, `:442-460`). Modelled on `adopt` (`:757-790`). It calls `build_coarse` first (idempotent, `K/upstream/kornia_imgproc/lk_window.rs:90-113`).

**Per frame:**
1. `prepare(&[left, right], frame_idx, ..)` (`:808`): the only image copy, into the level-0 planes.
2. `step(&tracks, &FlowStepInput { frame_idx, pose_world_to_primary: None, primary_rotation })` (`:878`). `primary_rotation` is `R_cur_prev` of the left camera from the window's own gyro preintegration at the current bias. This gives the left temporal and right temporal tracks, each with its round trip (`OF:161-165, 207-232`).
3. `tracks.advance(survivors)` (`K/tracking/optical_flow.rs:362`). A right track whose left partner died is removed.
4. New left corners: `output.spawned` (camera 0 only), or `seed(0)` when the set is empty. Each gets `tracks.spawn_on(pixel, 0)`.
5. Stereo link for each new left track: `track_across(0, 1, pixel, pixel)`; on success `tracks.spawn_on(right_pixel, 1)` and `partner[right_id] = left_id` (`OF:337-343`). Existing tracks are not re-linked, as Basalt.
6. Row gate on every live left-right pair: `|v_l − v_r| > 0.005 · fy` removes the right track only (`OF:370-385`).
7. Output: `Vec<Observation>` sorted by (camera, landmark id); the landmark id is the left track id.

**Keyframe selection** is in the window, on this output (`VIO:340-373`): `connected0` = camera-0 observations whose track has a landmark, `unconnected0` = the rest. Keyframe when `connected0 / (connected0 + unconnected0) < 0.7` and more than 5 frames passed since the last keyframe. The first frame is a keyframe (`VIO:63`).

**How the F3 remap feeds it.** The source rectifies each pair (`A/source/mod.rs:262-300`, `Rectification::FixedPoint` is `StereoRemap`, `K/rectify.rs`, the same bytes as `RemapF32`). `process_stereo` receives the rectified pair; the estimator reads those same borrowed buffers. No second remap and no raw images in v1 (owner answer 8).

## 7. Observe mode

- **Where.** `SlamSystem::process_imu` (`K/system/mod.rs:681`) also calls `runner.push_imu(samples)`: the estimator keeps its own sample queue, because the main buffer prunes at keyframes (`K/inertial/mod.rs:212`). `process_stereo` (`:718`) first computes the main result with `track_capture`, then calls `runner.observe(timestamp_sec, left, right)`, then returns the main result unchanged. An odometry error is counted and logged, never returned.
- **Threads.** The calling (tracking) thread only. No worker thread, no rayon, no flow workers (`threads: 1`), faer `Par::Seq`. The module never calls `faer::set_global_parallelism`.
- **Guarantee.** With `mapping.execution: Synchronous`, Observe-on `traj.txt` and `traj_smoothed.txt` are byte-identical to Observe-off on the same binary. Reason: the estimator reads only its own state and the two image borrows; it writes nothing the main path reads. The hook is outside `track_frame`, so `tracking_duration` is unchanged too.
- **Not guaranteed.** Under real time (paced, asynchronous) Observe adds its cost to the tracking thread, so an Observe-on real-time `traj.txt` is not a valid real-time arm. 2x-5 scores Observe in SYNC only.
- **Output.** `--trajectory-vio <path>` in the app (`A/main.rs`, beside `--trajectory-out` `:60`): one `TrajectoryWriter` line per frame from `system.odometry_output()`, the body pose, from the first frame on. The flag without the knob is an error at start-up.
- **Summary** (`A/summary.rs`), fixed-point decimals only, because `E/score.py` parses `(\w+)=(-?[\d.]+)`:
  - `[summary] vio frames= poses= keyframes= solves= solve_failures= resets= iterations_mean= rejected_steps= landmarks_mean= rows_left_mean= rows_right_mean= stereo_links= fb_rejects= row_gate_rejects= movement_rejects= landmarks_fixed= marg_pinv= imu_chi2_nav= bias_walk_chi2= visual_chi2= nullspace_yaw_log10_max=`
  - `[summary] vio_ms p50= p90= p99=` (flow + solve + marginalize), `vio_flow_ms`, `vio_solve_ms`, `vio_nullspace_ms`.
  - `imu_chi2_nav` is the mean of `r_navᵀ Ω r_nav / 9` over the newest IMU edge after each solve; `visual_chi2` the mean Huber cost per row DOF.

## 8. Initialization: recommend Basalt's, not VIBA0

**Recommended (Basalt, `VIO:185-220`).** At the first frame: attitude from the first accelerometer sample at or after the frame stamp, rotating the measured specific force onto world +z; velocity 0; biases 0; the initial prior of section 4.7. Optimization starts at the fifth state (`VIO:1017`). No offline-calibrated biases (owner answer 5); Basalt subtracts them (`VIO:167-168`).

Why not VIBA0:
- VIBA0 arrives after about 7 s (V1_03 frame ≈ 147, card "V1_03 / V2_02 diagnosis"). The estimator would have no pose before it, against P4, and those frames carry the pre-gravity problem the plan lists as a risk.
- It would tie the estimator to the pipeline it replaces in `Drive`.
- Roll, pitch and velocity have no prior in Basalt's scheme, so a start in motion is corrected by the first solves.
- VIBA0 holds a zero-mean accel-bias prior of σ 3.2 mm/s² (`R/experiments/vi-ba-vs-basalt.md` section 7), the mechanism behind the 0.8-0.9° tilt.

Known cost: the first state is linearized at a guess that can be about 1° off on a moving start, and its FEJ Jacobians stay there until it is marginalized (5 frames). Test T4b starts in motion to bound it.

## 9. Tests before any run

All in release, deterministic (`Gauss`, no `rand`). Thresholds are predictions: a miss gets a diagnosis, not a looser bound.

| id | test | pass |
|---|---|---|
| T0 | `F/reprojection.rs`: anchored Jacobians against central differences (target, host, landmark); host = −target; a landmark at ρ = 0 projects; stereographic round trip | 1e-6 relative; exact sign |
| T1 | **Batch equivalence, linear.** `marginalize.rs` on a random linear chain: 40 states of 15, 3-dim landmarks seen by 2-6 states, unit information, oldest state and finished landmarks marginalized each step | newest state's mean and its prior `(H', b')` equal the dense batch solve and the batch Schur complement to 1e-9 relative |
| T1b | New `schur_complement` reproduces `marginalize_frozen` on that file's existing test inputs | 1e-12 |
| T2 | **Ledger equivalence, real factors.** The window records (test only) every row and IMU block it absorbs, with the Jacobians used. After 60 frames of the synthetic walk, the sequential prior equals the one-shot Schur complement of the ledger | 1e-9 relative on `H'` and `b'` |
| T3 | **Nullspace.** (a) The full cost is invariant under a finite world translation and yaw, with velocities rotated: this validates the columns independently. (b) After 400 frames, the twin prior on x, y, z, yaw. (c) Roll and pitch | (a) 1e-9 relative; (b) ≤ 1e-9; (c) ≥ 1e3 × yaw |
| T4a | **Synthetic walk.** `HeadWalk` with `accel_bias (−0.026, 0.137, 0.076)`, `gyro_bias (−0.002, 0.020, 0.077)`, 20 s at 1 kHz, frames at 20 Hz, stereo f 435 / b 0.11 on 752×480, points 2-8 m around the path, exact pixels, started from zero biases | at 10 s: accel bias error ≤ 0.01 m/s², gyro ≤ 1e-4 rad/s, tilt ≤ 0.1°; position error at 20 s ≤ 0.02 m; no reset |
| T4b | T4a from t = 5 s (moving start), with pixel noise 0.5 px and IMU noise at the datasheet densities | at 10 s after start: accel ≤ 0.03 m/s², gyro ≤ 3e-4 rad/s, tilt ≤ 0.3°; `imu_chi2_nav` in 0.5-2.0 |
| T5 | Bookkeeping: M0-M10 one by one; no observation refers to a frame outside the window after any step; the keyframe rule | exact sets |
| T6 | Solver: Huber weight at 2 px is 0.5; λ after an accept and after a reject; the 8-attempt cap; the Cholesky-failure path leaves the prediction | exact |
| T7 | Frontend on synthetic shifted pairs: a stereo link finds the set disparity; an occluded patch fails the round trip; the row gate; refill only in camera 0; `seed` on an empty set; the same output on two runs | 0.05 px; exact counts |
| T8 | System: `Observe` leaves every `TrackingResult` of a 30-frame synthetic stereo-inertial run bit-identical to knob off; `Drive` is rejected; the four validation rules; the header line; RON round trip | exact |
| T9 | **KLT parity with OpenCV**, existing fixtures: the three ignored tests on a locally generated `frontend_opencv.bin` (11×11, level 0: the estimator's window). Then `bench_upstream_lk_window` with `opencv_bench` | LK within 0.01 px with identical status; our arm faster than OpenCV's |

## 10. Experiment card for 2x-5 (Observe)

To be written into `R/experiments/euroc-2x-plan.md` before code; results go to `R/experiments/euroc-2x-results.md`.

- **Gaps.** G13, G58, G62, G50 (`R/experiments/gap-ledger.md`).
- **Mechanism.** Frames are visual-only PnP on a frozen map (`K/system/config/mod.rs:519-530`). The window is 3 keyframes clamped to a fixed past; flow is left-only in Stereo mode.
- **Theory.** Every frame holds pose, velocity and bias under an IMU factor at full weight, with a dense first-estimate prior for the past. The gyro then constrains the rotation-translation ridge on every frame (V1_03 high-frequency error, MH_04 dark-span step), and right-camera rows keep the frame constrained when the left tracks thin out (V2_03).
- **Knob.** `odometry: SlidingWindow(( mode: Observe ))` on S0E.
- **Arms** (`E/mkcfg.sh`, outside the repo): `VO-ds` = S0E + the knob (rig, datasheet noise); `VO-bn` = the same with `imu_noise: Explicit((..))`, Basalt's values (section 1).
- **Noise-arm rule, written before the runs (owner answer 5).** Screening: MH_01, V1_02, V1_03, MH_04, V2_03, start 0, both arms (10 runs). The primary arm is the one whose `imu_chi2_nav` is closer to 1 in `|log|` on at least 3 of 5; a tie goes to `VO-ds`. ATE does not enter the choice. The other arm is reported, not scored.
- **Protocol.** SYNC, `--no-pace`, 900 s cap, at most 2 `kornia-slam*` processes, the command of the card's section 2 plus `--trajectory-vio $d/traj_vio.txt`. Primary arm on all 11 sequences × starts 0, 30, 60 (33 runs, about 4 min each). Each run is within the 5-minute rule.

**Predictions (as in the plan), with the measurement.**

| id | prediction | measured by |
|---|---|---|
| P1 (mechanism) | V1_03 HF ≤ 0.008 on every start (`hf.py`; ours 0.041-0.054, Basalt 0.004). Per-frame translation step p50 ≤ 1.5x Basalt on MH_01, V1_02, V2_01. | `E/resid/v13v22/hf.py V1_03_difficult - VO=traj_vio.txt`; new `E/twox/tsteps.py` (the earlier `tsteps.py` was never committed) |
| P2 | MH_04 vertical increment over 63.3-66.7 s within ±0.10 m (`E/resid/cf.py`; S0PEIH −0.22..−0.29). | `E/resid/cf.py MH_04_difficult traj_vio.txt 81.5 63.3:66.7:v`, the `d=(..)` z value |
| P3 | VIO-only ATE ÷ Basalt 0.8-1.3 on ≥ 9 of 11, none above 1.6. | new `E/twox/vio_score.py`: `B/eval.py --fmt tum --rpe 1,10` on `traj_vio.txt`, and on `B/results/basalt/<seq>/trajectory.txt` cropped to the same stamps (5 ms); `E/score.py` gains `traj_vio.txt` in `ALLOWED` |
| P4 | V2_03 frames without a pose 0; ATE ≤ 0.30. | `[summary] vio frames=` against `poses=`; `resets=0` |
| P5 | flow plus solve p50 ≤ 16 ms, p99 ≤ 35 ms on one core (criterion bench and a `[summary]` line). KLT kernel matches and beats OpenCV on the existing fixtures. | `[summary] vio_ms` from one run pinned with `taskset -c 8` under `flock P/rt.lock`; `bench_odometry_window`; T9 |

**Kill rules (as in the plan).**

| id | rule |
|---|---|
| K1 (mechanism, no score) | P1-P2 hold and P3 fails on ≥ 3. Diff against Basalt on MH_01, first 400 frames, before anything else. |
| K2 (score, no mechanism) | P3 holds with a failed nullspace test or HF > 0.02. Not credited. |
| K3 | P5 misses (p50 > 25 ms). Square-root f32 solve and `wide` SIMD before Drive. |
| K4 | V-a's K1 fired and Observe HF > 0.02. The "IMU on every frame" theory is wrong; stop the estimator. |

For K1's diff, the order of suspects is: open question 1 (FEJ form), the IMU noise arm, the bias start (Basalt's is calibrated), then track counts per frame.

**Bit-identity.**
1. Knob off: `BIN=E/bin-2x5-<tag>/kornia-slam-euroc E/integ/ident.sh` prints `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` (md5 `d4f77b567b912b5a4caa2af6085a35b1`).
2. Knob `Observe`, SYNC S0E V1_02: `traj.txt` and `traj_smoothed.txt` `cmp`-identical to knob off on the same binary.
3. Header: `E/expect/VO-ds.header` = `S0E.header` plus the odometry line, `diff`-clean.
4. Determinism: VO-ds V1_02 and MH_04 twice; `cmp traj_vio.txt` prints nothing.

**Noise.** `traj_vio.txt` is deterministic in SYNC, so the band is the start band: min..max over starts 0, 30, 60 per sequence (starts 0/1/2 are degenerate on MH_04, V2_03, V1_02). P3 is read on the median of the three, each start against Basalt cropped to its own stamps. Basalt is one run at load 24-27; the quiet rerun of rung 2x-1 is printed beside it and the targets do not change.

**Stop S-C** after the verdict: go or no-go on Drive. A miss or a fired kill rule gets a written diagnosis first.

## 11. Implementation steps (Opus implementer)

Worktree first: `git -C R worktree add /home/edgar/dev/split/kornia-slam-2x-a -b feat/2x-vio feat/global-pose-graph`. Work only there. Stage by path; never `git add -A`. Conventional commits with a scope. No push without the owner's go.

Cost is an estimate, not a measurement: new lines including tests, and Opus working time excluding the `-j2` builds.

| # | commit | files | reuses | cost |
|---|---|---|---|---|
| 0 | `docs(experiments): 2x-4 design card and the 2x-5 block` | `experiments/euroc-2x-vio-design.md` (this card), `experiments/euroc-2x-plan.md` | - | 0.5 h |
| 1 | `feat(factors): host-anchored inverse-distance reprojection` | `F/reprojection.rs` | `residual_and_pose_jacobian` `:11`; `testing::{Lcg, perturbed}` | 220 lines, 1.5 h; T0 |
| 2 | `feat(system): pipeline.odometry knob, validated, Drive not built` | new `K/odometry/{mod,config}.rs`; `K/lib.rs`; `K/system/config/mod.rs` (`:105-118`, `:806-815`, `:818`); `K/system/validation.rs`; `K/system/config/tests.rs`; `CHANGELOG.md` | `UNBUILT` `:588`; `check` `:1087` | 280 lines, 2 h; the T8 config parts |
| 3 | `feat(mapping): size-generic Schur complement` | `K/mapping/marginalization.rs`: new `pub(crate) fn schur_complement(h, b, keep, marg)`; `marginalize_frozen` is not rewired | the algebra at `:271-297` | 160 lines, 1.5 h; T1b |
| 4 | `feat(odometry): first-estimate states and the marginal prior` | `K/odometry/{state,marginalize}.rs`, `tests/marginalize.rs`; `pose_inertial.rs`: `plus` and `nav_state` become `pub(crate)` (visibility only) | commit 3; `FrameNavState` | 450 lines, 3 h; T1 |
| 5 | `feat(odometry): landmarks, linearization and the LM solve` | `K/odometry/{landmarks,linearize,solve}.rs` | commit 1; `preintegration::{residual_and_jacobians, information}`; `DenseLlt`; `triangulate_n_view_linear` | 650 lines, 5 h; T6 |
| 6 | `feat(odometry): the window: initialization, keyframes, marginalization plan` | `K/odometry/{init,keyframes,window,synthetic}.rs`, `tests/window.rs`; `K/mapping/mod.rs:30` cfg to `any(test, feature = "bench_internals")` | `HeadWalk`, `Gauss`; `PreintegratedImu::{from_measurements_with, predict}` | 800 lines, 6 h; T2, T3, T4, T5 |
| 7 | `feat(tracking): primary-only refill, seeding and cross-camera tracks in RigFlow` | `K/tracking/flow_tracker.rs`; `K/system/mod.rs:3714`; `tests/flow_zero_alloc.rs:99, 154` | `FlowShard::refill`, `forward`, `keep`, `adopt` | 220 lines, 2.5 h; identity trio here |
| 8 | `feat(odometry): two-camera flow frontend` | `K/odometry/frontend.rs`, `tests/frontend.rs` | commit 7; `TrackSet` | 380 lines, 3 h; T7 |
| 9 | `feat(system): Observe mode beside the pipeline` | `K/odometry/mod.rs`; new `K/system/odometry.rs`; `K/system/mod.rs` (field, build `:303`, `process_imu` `:681`, `process_stereo` `:718`, two accessors); `K/system/tests.rs` | `SensorRig::{camera_to_body, imu_noise, stereo_baseline_m}` | 380 lines, 3 h; T8 |
| 10 | `feat(app): --trajectory-vio and the vio summary lines` | `A/main.rs`, `A/summary.rs` | `TrajectoryWriter` (`A/trajectory.rs`) | 150 lines, 1.5 h |
| 11 | `test(bench): criterion bench of one window step` | `R/crates/kornia-slam/benches/bench_odometry_window.rs`; `R/crates/kornia-slam/Cargo.toml` (`[[bench]]`, `required-features = ["bench_internals"]`) | `synthetic.rs` | 130 lines, 1 h |
| 12 | Binary, identity, T9. No commit | `E/bin-2x5-<tag>/`; fixture generated locally, not committed | `E/integ/ident.sh` | 1.5 h plus run time |

Total, estimate: about 3800 lines, 32 h of implementer time, about 14 verification cycles.

Ledger and ROADMAP: at each verdict, update `R/experiments/gap-ledger.md` and the ROADMAP progress log; drop "R12 inside the local BA" from the ROADMAP "Now" list with commit 0. No path under the user's home config directory goes into repo docs.

### Verification (run from `/home/edgar/dev/split/kornia-slam-2x-a`; at most 2 cargo builds on the machine)

After every commit:

```bash
cargo fmt --all -- --check
cargo +1.100.0 test --release -j2 -p kornia-factors
cargo +1.100.0 test --release -j2 -p kornia-slam --features serde
cargo +1.100.0 clippy --release -j2 --all-targets -- -D warnings
cargo +1.100.0 clippy --release -j2 -p kornia-slam --all-targets -- -D warnings
cargo +1.100.0 clippy --release -j2 --manifest-path apps/kornia-slam-app/Cargo.toml --no-default-features --all-targets -- -D warnings
cargo +1.100.0 clippy --release -j2 --manifest-path apps/kornia-slam-app/Cargo.toml --features uvc --all-targets -- -D warnings
git status --short        # only the intended paths
```

Pass: fmt prints nothing; every test passes (801 passed and 4 ignored before this work, plus the new ones); the four clippy jobs (the CI matrix, `R/.github/workflows/rust_lint.yml:111-155`) are clean. If the `uvc` job lacks `libv4l-dev`, report it; do not install.

After commits 9 and 10 also: `cargo +1.100.0 test --release -j2 --manifest-path apps/kornia-slam-app/Cargo.toml --no-default-features` (72 passed before, plus the flag tests).

Module rule, after commits 4-9: `grep -rn "rayon\|set_global_parallelism\|unwrap()\|expect(" crates/kornia-slam/src/odometry --include=*.rs | grep -v "/tests/"` prints nothing.

After commit 7 and at step 12:

```bash
cargo +1.100.0 build --release -j2 -p kornia-slam-app
mkdir -p /home/edgar/dev/split/.pr180-fix/euroc-top1/bin-2x5-<tag>
cp target/release/kornia-slam-app /home/edgar/dev/split/.pr180-fix/euroc-top1/bin-2x5-<tag>/kornia-slam-euroc
BIN=/home/edgar/dev/split/.pr180-fix/euroc-top1/bin-2x5-<tag>/kornia-slam-euroc /home/edgar/dev/split/.pr180-fix/euroc-top1/integ/ident.sh
```

Pass: the output holds `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` and the md5 `d4f77b567b912b5a4caa2af6085a35b1`. No SYNC run starts while `P/rt.lock` is held. Confirm the built binary's name before the copy.

Step 12, parity and speed:

```bash
python3 crates/kornia-slam/benches/opencv_frontend_fixtures.py \
  ~/.cargo/git/checkouts/kornia-rs-0edb86845af2bca0/4ada5a7/tests/data/mh01_frame1.png \
  ~/.cargo/git/checkouts/kornia-rs-0edb86845af2bca0/4ada5a7/tests/data/mh01_frame2.png
cargo +1.100.0 test --release -j2 -p kornia-slam --features serde -- --ignored \
  matches_opencv_within_a_hundredth_of_a_pixel matches_opencv_good_features_to_track matches_opencv_corner_sub_pix
RAYON_NUM_THREADS=1 taskset -c 8 cargo +1.100.0 bench -j2 -p kornia-slam --features bench_internals,opencv_bench --bench bench_upstream_lk_window
RAYON_NUM_THREADS=1 taskset -c 8 cargo +1.100.0 bench -j2 -p kornia-slam --features bench_internals --bench bench_odometry_window
```

Pass: 3 passed; our LK time below the OpenCV arm; window step p50 ≤ 8 ms. If Python lacks `cv2` or the `opencv` crate does not build, report and stop; do not install.

Step 12, Observe identity (section 10, items 2-4): `cmp` prints nothing on each pair; the header `diff` is empty.

## 12. Risks and open questions

### Risks: what fails silently, and what detects it

| risk | detection |
|---|---|
| The nullspace columns have a sign error, so the nullspace test passes on wrong vectors | T3a validates the columns on the full cost first; T3c requires roll and pitch to be observable |
| A linearized state is read at its current value in some factor: fine on short runs, yaw drifts on long ones | T3b over 400 frames; `nullspace_yaw_log10_max` on every Observe run; K2 |
| Prior re-centring `b' −= H'δ` has the wrong sign, or an IMU factor stays in the window after it entered the prior (double count) | T1 and T2 at 1e-9 |
| An observation of a removed frame survives | T5; an error return, not a panic |
| Observe changes the main path through shared state (faer global parallelism, rayon, the IMU buffer, the tracker's flow) | T8; identity items 2 and 3; the module grep |
| The `flow_tracker.rs` additions change the legacy `WindowedU8` arms | `IDENT_R01_OK` (its preset runs that file); `flow_zero_alloc` |
| Right-camera extrinsic sign wrong: stereo rows push depth the wrong way | T0 and T7 with a known point; `landmarks_created` near `stereo_links` |
| The Huber knee is applied in the wrong unit (px against χ²): weights off 4x | T6 |
| The 60 px movement cap, measured from the previous pixel, drops tracks above 2.76 rad/s (V2_03, V1_03) | `movement_rejects` and rows per frame against gyro rate. If P4 misses with this counter high, the fix is its own small card (measure the cap from the start pixel) |
| Disparity above 60 px (depth under 0.80 m) gets no stereo link | `stereo_links / landmarks_created` per sequence; such points still triangulate by motion |
| Solve time grows with track count; there is no wall-clock exit | P5, K3, `bench_odometry_window` |
| New landmarks only at keyframes, at most every 6 frames: rows thin out under fast rotation | `rows_left_mean`, `rows_right_mean`; the per-frame minimum in the results |
| The first state's FEJ point is a poor attitude guess on a moving start | T4b; starts 30 and 60 in the 2x-5 band |
| The dirty window-test file gets staged | the new worktree is clean; `git status --short` before each commit |
| `basalt-headers` is absent: our preintegration or bearing formula may differ from Basalt's | T0 and T4 test ours on their own terms; K1's diff is where a difference would show |

### Open questions, with the recommended answer

1. **FEJ form for visual rows.** Strict (Jacobians at the lin poses) or Basalt's (relative-pose value at the current state, `linearization_abs_qr.cpp:206-221`)? Recommend strict: one rule for every factor; both keep the nullspace. Align with Basalt only if K1's diff points there.
2. **Gyro-rotated start pixels.** Basalt has none; `GyroGuess::RotatedRay` exists in `RigFlow`. Recommend on: it is existing code, and the round trip guards a bad guess.
3. **LK window 11.** Basalt's patch spans ±3.5 px; our default is 21; 11 is the window the OpenCV fixture and the existing bench validate. Recommend 11.
4. **Observe runs inline on the tracking thread, with no worker.** Then an Observe-on real-time run is not a valid real-time arm. Recommend accept: 2x-5 scores SYNC, and in Drive this is the tracking path itself.
5. **Initialization.** Recommend Basalt's first-sample attitude with zero-mean bias priors (σ 0.316 rad/s, 0.1 m/s²), no VIBA0, no calibrated biases (section 8).
6. **Home of the anchored factor.** Recommend `kornia-factors` (it owns factors). It adds public API to a published crate; say if it should stay private to `kornia-slam` instead.
7. **`schur_complement` beside `marginalize_frozen`, not replacing its body.** Recommend yes for now: `BoundaryPrior::Marginal` arms stay bit-identical; rewire in the cleanup card.
8. **IMU-noise arm rule** (section 10): `imu_chi2_nav` closest to 1 on 3 of 5 screening sequences, tie to the datasheet arm. Recommend as written.
9. **Reset policy.** Recommend reset only on a non-finite state or an IMU gap; no velocity or bias envelope in v1.
10. **The OpenCV fixture.** Recommend generating it locally and not committing it, as today; record its md5 in the results.
11. **Every frame is written to `traj_vio.txt`,** including the four frames before the first solve. Recommend yes: P4 counts frames without a pose.
12. **The nullspace twin runs in every Observe run** (a second marginalization per frame, timed apart). Recommend yes for 2x-5, off by default in Drive.

## Owner answers (2026-11-24, stop S-B)

- Design approved; all 12 open questions answered as recommended.
