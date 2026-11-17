# PR #180 card B: recovery and rig

Lane B of the PR #180 round adds five things:
- a generic rig calibration file;
- a ray per rig pixel;
- relocalisation while lost, with a rig-aware AP3P hypothesis (G19);
- a re-seeded map that tracks its own window first (G28);
- the port of `4a2ee7b` (map segments, relocalisation once lost).

This card records the theory, the numeric predictions and the kill rules before any code. The validation workflow scores it. Nothing here has been measured yet.

Base: kornia-slam@`pr180/S` (`42ee054`), kornia-rs@`4ada5a7`. Binary tags: `pr180/L-B` (lane), `pr180/F` (all lanes, knobs off).
All file:line references are on `42ee054`, unless they name ORB-SLAM3 (`/home/edgar/dev/.flux-connect-scratch/bench/src/ORB_SLAM3/`) or kornia-rs at `4ada5a7`.

## Knobs (all off by default; defaults are today's behaviour)

| RON path (run file) | type | default (old behaviour) | experiment |
|---|---|---|---|
| `rig_calibration` (top level, next to `source` and `system`) | `Option<PathBuf>`, relative to the run file | `None`: the source's own calibration | B1 |
| `system.pipeline: OrbSlam((tracking_loss: (relocalization: (while_lost: ..))))` | `bool` | `false`: relocalise only while the IMU is not trusted (`system/mod.rs:1576-1583`) | B3 |
| `tracking_loss.relocalization.solver` | enum `RelocalizationSolver { PrimaryEpnp, PrimaryAp3p, RigAp3p }` | `PrimaryEpnp`: EPnP RANSAC on primary-camera matches (`relocalization.rs:123-124, 175-187`) | B3 |
| `tracking_loss.relocalization.max_jump_m` | `f64`, > 0 | `0.5`: today's `RELOCALIZATION_MAX_JUMP_M` (`system/mod.rs:2414`) | B3 |
| `tracking_loss.reseed.relocalize_when_lost` | `bool` | `false` (`4a2ee7b` shipped `true`; re-cut off, per the plan) | B5 |
| `tracking_loss.reseed.own_window_keyframes` | `usize` | `0`: off. With `N > 0`, a segment after the first tracks and grows only against its own keyframes until it holds `N` of them. | B4 |

CLI (not a knob, never changes a run): `--write-rig-calibration <path>` writes the opened source's `SensorRig` as a rig calibration file, then exits. It produces the files B1's identity check needs.

`validate()`:
- New `ConfigError::RelocalizationWithoutVocabulary(&'static str)`, with the message "`{0}` needs loop closing's vocabulary: enable `pipeline.loop_closing`".
- It is returned when `pipeline.loop_closing` is `Disabled` and any of these is set:
  - `relocalization.while_lost` is `true`;
  - `relocalization.solver` is not `PrimaryEpnp`;
  - `reseed.relocalize_when_lost` is `true`.
- Each of these is inert without the loop closer (`system/mod.rs:2001-2003` returns `false`). This follows Step 9's rule of rejecting an inert knob.
- `relocalization.max_jump_m` is checked with `check(.., POSITIVE)`, at `tracking_loss.relocalization.max_jump_m`.
- No `configs/*.ron` sets any of these fields, and the parse stays `deny_unknown_fields`.

## Counters (new `RunCounters` fields; 0 with the knobs off unless stated)

The skeleton declares these fields and prints them on one new summary line shared with card A's counters (card A, "Skeleton contract"). The existing first summary line and its test (`summary.rs:91`) do not change. Lane B only increments them.

| field | counts |
|---|---|
| `relocalization_attempts` | calls of `SlamSystem::relocalize` that reached the candidate query, in every mode. Knob off: the visual-mode calls of today, which are not 0. |
| `relocalizations` | adopted relocalisations, in every mode. |
| `relocalizations_while_lost` | adopted ones that only `while_lost` allowed: the IMU was confident (the G19 mechanism). |
| `relocalization_refused_jump` | candidate poses refused by the jump rule. |
| `relocalizations_older_segment` | adopted ones into an older segment once lost (`4a2ee7b`). |
| `relocalization_without_vocabulary` | frames where a relocalisation was due but no loop closer exists. `validate()` keeps it at 0 for RON builds. It is the runtime proof that the guard holds for systems built some other way. |
| `relocalization_ns` | wall time inside `relocalize`, in ns. |
| `seeded_window_frames` | frames tracked with the B4 window active. |
| `seeded_window_dropped_points` | local-map points that the B4 window filtered out at the observation-count selection (`system/mod.rs:1551`). |

## Shared types and signatures the skeleton (Phase 2c) must add

Bodies are `todo!()`. They can only be reached through the knobs above.

**kornia-sensors, `src/rig.rs`.** This is the ray pair from the plan's upstream convention. It has no dependency on kornia-slam.
```rust
/// Ray through an undistorted pixel of `camera`, in the primary-camera frame:
/// `(origin, unit direction)`, origin = `primary_from_camera.translation`.
pub fn camera_ray(camera: &PinholeCamera, primary_from_camera: &Pose3d, pixel_undist: [f64; 2]) -> (Vec3F64, Vec3F64);
impl RigCamera { pub fn ray(&self, pixel_undist: [f64; 2]) -> (Vec3F64, Vec3F64); }
impl SensorRig {
    /// `T_BC` of rig camera `camera`: `camera_to_body ∘ primary_from_camera`.
    pub fn camera_to_body_of(&self, camera: usize) -> Option<Pose3d>;
}
```

**kornia-sensors, new `src/rig_calibration.rs`.** It is behind the existing `serde` feature, which the app already enables through kornia-slam. Add `cfg_attr(feature = "serde", derive(Serialize, Deserialize))` to `ImuCalib`.
```rust
pub struct RigCalibration {
    pub cameras: Vec<RigCameraCalibration>,      // primary first
    pub camera_to_body: Option<[[f64; 4]; 4]>,   // T_BC of the primary, row-major, last row 0 0 0 1
    pub imu_noise: Option<ImuCalib>,
}
pub struct RigCameraCalibration {
    pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64,
    #[serde(default)] pub k1: f64, #[serde(default)] pub k2: f64,
    #[serde(default)] pub p1: f64, #[serde(default)] pub p2: f64,
    pub width: usize, pub height: usize,
    pub primary_from_camera: [[f64; 4]; 4],      // T_PC (ORB-SLAM3 `Stereo.T_c1_c2`); identity for cameras[0]
}
pub enum RigCalibrationError { NoCameras, PrimaryNotIdentity, NotRigid { camera: usize, defect: f64 }, NotHomogeneous { camera: usize } }
impl RigCalibration {
    pub fn from_sensor_rig(rig: &SensorRig) -> Self;
    /// Replaces `rig`'s camera geometry, plus `camera_to_body` and the noise when `rig.imu` is `Some`.
    /// A rig without IMU stays without one: the IMU selection belongs to the system config.
    pub fn apply_to(&self, rig: SensorRig) -> Result<SensorRig, RigCalibrationError>;
}
/// Rig cameras from per-camera body extrinsics `T_BC` (Kalibr, LaMAria, EuRoC):
/// `T_PC = T_BC[primary]⁻¹ ∘ T_BC[c]`, the primary first.
pub fn rig_cameras_from_body_extrinsics(cameras: &[(PinholeCamera, ImageSize, Pose3d)], primary: usize) -> Vec<RigCamera>;
```

**kornia-slam, new `src/upstream/kornia_3d/gpnp.rs`.**
- Header: `//! Upstream target: kornia_3d::pnp::gpnp. Kept here until upstreamed; no kornia-slam types.`
- Imports: kornia-3d, kornia-algebra and `std` only. That rules out `rand`, so the sampler carries its own splitmix64 state.
```rust
/// A 2D-3D match as a ray of a multi-camera rig; everything in the rig (primary) frame.
#[derive(Debug, Clone, Copy)]
pub struct RayMatch {
    pub origin: Vec3F64,      // camera centre, metres
    pub direction: Vec3F64,   // unit
    pub point_world: Vec3F64,
    /// Pixels per unit of tangent-plane error (the camera's fx): residuals come out in px².
    pub pixel_scale: f64,
}
/// World-to-rig poses (up to 4) from three matches that share one origin (bit-equal):
/// AP3P in a frame at that origin, lifted to the rig. Empty when the origins differ.
pub fn ap3p_lifted(matches: &[RayMatch; 3]) -> Vec<Pose3d>;
/// `(pixel_scale · tan∠(p, d))²` with `p = R X + t − origin`; `f64::INFINITY` when `p·d ≤ 0`.
pub fn ray_residual_sq(pose_world_to_rig: &Pose3d, m: &RayMatch) -> f64;
/// `kornia_3d::ransac::Estimator<Sample = RayMatch, Model = Pose3d>`, SAMPLE_SIZE 3.
pub struct LiftedAp3pEstimator;
/// `kornia_3d::ransac::Sampler` that draws only from `[0, prefix)`.
pub struct PrefixSampler { state: u64, prefix: usize }
/// RANSAC (`kornia_3d::ransac::run`, `ThresholdConsensus`): hypotheses from
/// `matches[..hypothesis_count]`, scored on all matches. Returns the pose and the inlier mask.
pub fn ransac_lifted_ap3p(matches: &[RayMatch], hypothesis_count: usize, threshold_px: f64, max_iterations: u32, seed: u64) -> Option<(Pose3d, Vec<bool>)>;
```
The skeleton adds `pub(crate) mod gpnp;` to `upstream/kornia_3d/mod.rs`; lane B does not edit that file.

**kornia-slam, the rest.**
- `rig.rs`: `RigView` gains `pub primary_from_camera: Pose3d`, stored from the `RigCamera` as given (not the inverse of the inverse). Add `RigView::ray(&self, pixel_undist: [f64; 2]) -> (Vec3F64, Vec3F64)`, which calls `kornia_sensors::camera_ray`.
- `tracking/relocalization.rs`: `pub enum RelocalizationSolver { #[default] PrimaryEpnp, PrimaryAp3p, RigAp3p }`, with serde, plus a field `solver` in `RelocalizationConfig`. The tracker keeps a `RelocalizationConfig` set from the settings, in place of `RelocalizationConfig::default()` at `tracker.rs:139`.
- `system/config/mod.rs`: `pub struct LostRelocalization { pub while_lost: bool, pub solver: RelocalizationSolver, pub max_jump_m: f64 }`, with `serde(default, deny_unknown_fields)`. Add the field `relocalization: LostRelocalization` to `TrackingLossConfig`. Carry it through `SystemSettings` (`build.rs:131-134`).
- `system/recovery.rs`:
  - `RecoveryConfig` gains `relocalize_when_lost: bool` (false) and `own_window_keyframes: usize` (0).
  - The segment id is not defined here: `pub type SegmentId = u32;` lives in `mapping/map/mod.rs` (re-exported as `crate::mapping::SegmentId`), so card D's `mapping/session.rs` can use it without mapping importing a system type. `recovery.rs` imports it.
  - `pub(crate) struct Segments { starts: Vec<usize> }`, with `begin(&mut self, first_kf_idx)`, `of(&self, kf_idx) -> SegmentId`, `latest(&self) -> SegmentId` and `latest_start(&self) -> Option<usize>`, as in `4a2ee7b`. `starts` keeps keyframe ids as `usize`; `of` and `latest` return the segment index as `SegmentId`.
  - The skeleton ports `Segments` with its four bodies (bookkeeping only), the `segments: Segments` field on `SlamSystem`, the `segments.begin` call in `finish_bootstrap` and `SlamSystem::segment_of`, so lane D does not depend on lane B's body. Lane B fills the relocalisation parts.
  - `pub(crate) fn relocalization_adopted(segments, kf_idx, jump_m, max_jump_m, lost) -> bool`, as in `4a2ee7b`.
- `system/mod.rs`, the relocalise trigger hook, which Lane B owns:
  - `fn relocalization_due(&self, timestamp_sec: f64) -> bool` replaces `!self.imu_confident(timestamp_sec)` at `system/mod.rs:1580`. It returns `!imu_confident || relocalization.while_lost`, so the knob-off expression is unchanged.
  - `fn relocalize(&mut self, frame: &Frame, lost: bool) -> bool`.
  - `fn seeded_window_start(&self) -> Option<usize>`.
- `tracking/local_map.rs` (signatures shared with lane A, card A "Skeleton contract"):
  - `LocalMapSelectionConfig` gains `rules: LocalMapRules` and `window_start_kf: Option<usize>`;
  - `select_local_map_points(map, tracked_matches, current_keyframe, config) -> LocalMapSelection { points, dropped_by_window, most_voted_keyframe }`;
  - lane B fills `pub(crate) fn filter_to_window(map: &Map, points: &mut Vec<usize>, window_start_kf: usize) -> usize`, which `select_local_map_points` calls after lane A's rule (fallback included) when `window_start_kf` is `Some`. It returns the number dropped. There is no separate `select_seeded_window_points`.
- `mapping/keyframe_mapping.rs`: `KeyframeGrowthPlan { keyframe_idx, triangulation_neighbours, fusion_neighbours }` and `prepare_keyframe_growth(map, keyframe_idx, rule: GrowthNeighbours)` (skeleton). Lane B fills `pub(crate) fn prepare_seeded_window_growth(map: &Map, keyframe_idx: usize, rule: GrowthNeighbours, window_start_kf: usize) -> KeyframeGrowthPlan`: it calls `prepare_keyframe_growth` with the same rule and filters both lists to keyframes `>= window_start_kf`. The skeleton writes the dispatch at `system/mod.rs:2121`.
- `tracking/pose_estimation/map_projection/mod.rs`: a field `window_start_kf: Option<usize>`, default `None`, on `MapProjectionEstimator`. Set it through `pub(crate) fn set_window_start(&mut self, Option<usize>)`, reached from the system through `Tracker::set_window_start`. `local_map_config()` builds the one config used at `:280`, `:577`, and, through `Tracker::local_map_config()`, at `system/mod.rs:1551` and `system/trace.rs:60`. All of this is the skeleton's; lane B edits no call site there.

## Experiment B1: generic rig calibration file
Gap tested: none (an input format); it unblocks rig runs from any calibration source. Guards: B3 on the rig.
Base: kornia-slam@`pr180/S` (`42ee054`).   Binary tag: `pr180/L-B`.
Mechanism:
- Rig geometry can only be built in code, per dataset. For example, LaMAria composes `T_PC = T_BC0⁻¹ ∘ T_BCi` inline (`apps/kornia-slam-app/src/source/lamaria.rs:134-163`).
- `SensorRig` stores `RigCamera::primary_from_camera` for each camera (`kornia-sensors/src/rig.rs:38-45`) and one `ImuCalibration::camera_to_body` (`rig.rs:119-126`).
- There is no file that a new rig, or a recalibration, can be described in.
Theory:
- Store exactly what `SensorRig` stores, so loading is a copy and not a recomputation. That means `primary_from_camera` for each camera, and `camera_to_body` for the primary only.
- This is ORB-SLAM3's choice: `Stereo.T_c1_c2` plus `IMU.T_b_c1` (`src/Settings.cc:342, 422`). Kalibr-style per-camera `T_BC` is converted once, by `rig_cameras_from_body_extrinsics`.
- The LaMAria source calls that function. It keeps the same expression, `primary.camera_to_body.inverse().compose(&cameras[c].camera_to_body)`, so its output stays bit-equal.
- RON writes f64 with Rust's shortest round-trip formatting, so a write followed by a read returns the same bits.
Knob: `rig_calibration: None`. CLI `--write-rig-calibration`.
Baselines: `pr180/S` LaMAria 3_18 with `configs/lamaria-rig-imu.ron` re-pointed at 3_18. The validation workflow records the sha, binary md5 and load.
Predictions (numbers, each tied to a measurement):
  B1-P1 Round trip `from_sensor_rig` → RON → `apply_to` on a two-camera rig (yaw 70°, 0.11 m) with IMU: 0 differing bits over all f64 fields.   measured by: unit test in `rig_calibration.rs`.
  B1-P2 `rig_cameras_from_body_extrinsics` against the old inline LaMAria expression: 0 differing bits.   measured by: unit test in `source/lamaria.rs`.
  B1-P3 LaMAria 3_18 rig, run with `rig_calibration` written by `--write-rig-calibration` from the same sequence's JSON: the trajectory file is `cmp`-identical to the JSON-only run.   measured by: `cmp` of the two trajectory files.
  B1-P4 `apply_to` rejects a non-identity `cameras[0]` (`PrimaryNotIdentity`), a rotation defect of 1e-3 (`NotRigid`), and a last row other than `0 0 0 1` (`NotHomogeneous`). It accepts a defect of 1e-9.   measured by: unit tests.
  Mechanism counter: none at run time. Here the bits are the mechanism (P1-P3).
Kill rules:
  K1 P3 fails while P1 and P2 hold -> the app applies the file at a different point than the source's rig (for example after `RigGeometry` is built, or the noise is lost); fix before any rig number of lane B is trusted.
  K2 P1 or P2 fails -> the format or the composition order is not exact; keep the source path as the only path and fix the record.
  (No score kill: a format change must not move any score. Any score change with `rig_calibration` set is K1.)
Bit-identity: with `rig_calibration: None`, `pr180/L-B` `cmp` equals `pr180/S` on EuRoC MH_01 and LaMAria 3_18, both mono and rig. That includes the LaMAria refactor through `rig_cameras_from_body_extrinsics`.
Noise: none (deterministic, bit-level).

## Experiment B2: rig rays
Gap tested: none on its own; it is the input of B3's `RigAp3p`.
Base: `pr180/S`.   Binary tag: `pr180/L-B`.
Mechanism: rig PnP consumes pixels plus a camera and `T_CP` (`tracking/pose_estimation/rig_pnp.rs:17-29`). A generalised solver needs camera-independent rays, and no function produces them.
Theory: one implementation, `kornia_sensors::camera_ray`:
- direction = `R_PC · normalise([(u−cx)/fx, (v−cy)/fy, 1])`;
- origin = `t_PC`.
`RigCamera::ray` and `RigView::ray` both call it, so kornia-slam does not re-implement it (upstream convention). Rays carry no camera type, so `upstream/kornia_3d/gpnp.rs` stays free of kornia-slam and kornia-sensors types.
Knob: none. A pure addition, called only by B3 with `solver: RigAp3p`.
Predictions:
  B2-P1 Project `origin + 5·direction` back through camera `c` (by `T_CP`): the result is within 1e-9 px of the input pixel, for 100 pixels over a 640×480 grid, on both cameras of the 70° rig.   measured by: unit test in `kornia-sensors/src/rig.rs`.
  B2-P2 The primary camera ray has origin exactly `Vec3F64::ZERO`. Two pixels of one camera give bit-equal origins, which `ap3p_lifted`'s same-origin test relies on.   measured by: unit test.
  Mechanism counter: n/a (pure function).
Kill rules:
  K1 P1 misses by > 1e-9 px -> wrong frame convention (`T_PC` used as `T_CP`); fix before B3.
  K2 P2 fails -> origins come from a recomputed inverse; store `primary_from_camera` as given (see the `RigView` field above).
Bit-identity: no run calls it with the knobs off.
Noise: none.

## Experiment B3: relocalisation while lost, AP3P lifted to the rig (G19)
Gap tested: G19 ("Global PnP while lost"; ledger: `/home/edgar/dev/.flux-connect-scratch/slam-viba/audit/gap-ledger.md:77`). It also touches G1 and G22 through fewer give-ups.
Base: `pr180/S`.   Binary tag: `pr180/L-B`, scored on `pr180/F`.
Mechanism:
1. Tracking PnP is local only. Every match passes a 25 px prior gate around the prediction (`tracking/pose_estimation/pnp.rs:48, 118-133`), widened by `search_scale` up to 100 px.
   - On LaMAria, coasts often start from a bad tracked pose: 13 of 38 loss episodes on R_11 rg1 had more than 5° or more than 0.5 m error (ledger G21).
   - A rotation error of about 21° (95 px at f = 242) empties the gate, so the IMU coast never re-enters it.
2. The only global recovery, `SlamSystem::relocalize` (`system/mod.rs:1999-2048`), is skipped once the IMU is confident (`system/mod.rs:1576-1583`). So a confident inertial coast can only end in re-entering the gate or in a give-up after 5 s (`tracking/policy.rs:222`).
3. The relocalisation hypothesis has two weaknesses:
   - It uses primary-camera matches only (`relocalization.rs:123-124`).
   - It uses EPnP, which `solve_pnp_ransac` samples 5 at a time (kornia-rs `crates/kornia-3d/src/pnp/ransac.rs:310-321`), with 300 iterations (`relocalization.rs:52`).
   - At inlier ratio w = 0.2, P(at least one clean sample) = 1 − (1 − 0.2⁵)³⁰⁰ = 0.09.
   - With the 3-point AP3P it is 1 − (1 − 0.2³)³⁰⁰ = 0.91.
   - On the LaMAria rig, the second camera's matches never count.
Theory:
- **Trigger (`while_lost`).** Run the existing relocalisation on every lost frame before a give-up, with the IMU confident too.
  - ORB-SLAM3 does not do this in inertial mode. RECENTLY_LOST only predicts with the IMU for `time_recently_lost` = 5 s (`src/Tracking.cc:48, 1985-1997`), and relocalises only in visual modes (`src/Tracking.cc:2001-2009`). Basalt and OpenVINS have no relocalisation.
  - ORB-SLAM3 can afford this because its tracking rarely accepts the bad poses that start our coasts (G21). Ours does accept them, so the IMU alone does not bring us back.
  - The procedure is ORB-SLAM3's `Tracking::Relocalization` (`src/Tracking.cc:3609-3780`): BoW candidates, at least 15 matches, RANSAC, pose optimisation plus search by projection, and at least 50 inliers. We already have it in `relocalization.rs:66-101`. Only when it runs changes.
  - A relocalised pose inside the segment being tracked must lie within `max_jump_m` of the coasted pose (`system/mod.rs:2020`). A pose from a drifted part of the map would otherwise teleport the trajectory.
  - The IMU chain from the last keyframe is kept. It measures real motion, and the relocalised pose is closer to the truth than the coast, so the next keyframe's IMU edge stays valid.
- **Window of use.** The gate covers 25 px; at f = 242 px and 3 m depth that is about 0.3 m of lateral error. The jump rule allows 0.5 m.
  - G19 can only recover losses whose coast error at the time of matching lies between about 0.3 m and 0.5 m, or whose rotation error empties the gate while the position is still within 0.5 m.
  - With the tilt error behind a bad last pose (3.66 m/s² misprojected gravity, ledger G19), the position error passes 0.5 m after √(2·0.5/3.66) = 0.52 s.
  - So refusals by the jump rule are predicted to be a large share (B3-P3). That is why `max_jump_m` is a knob: the next experiment can widen it without code.
- **Solver.**
  - `PrimaryAp3p`: today's path with `PnPMethod::AP3PDefault` in place of `EPnPDefault`. It isolates the effect of a minimal set of 3 instead of 5.
  - `RigAp3p` steps:
    1. Keep the matches from all cameras (drop the `camera_of == 0` filter).
    2. Turn each match into a `RayMatch` (B2), with `pixel_scale` = the camera's fx.
    3. Sort the matches of the camera with the most of them first; ties go to the lower index. That is the hypothesis camera.
    4. Run `ransac_lifted_ap3p`: AP3P on three rays of that camera, lifted by `primary_from_camera`, scored on every rig ray at the same 4 px threshold and the same 300 iterations.
    5. Refine with `solve_rig_pnp` (`rig_pnp.rs:60`), using the estimator's PnP config and the gate centred on the hypothesis.
    6. Hand the result to the existing `estimate_pose` with the inliers pre-seeded (`relocalization.rs:82-93`).
  - The lift:
    - Rotate the three bearings by `R_s`, which takes their mean to +z (a pinhole camera keeps them in front).
    - Centre the three world points on their mean `c` and scale them by 1/s (RMS radius). AP3P in kornia is f32 (`pnp/ap3p.rs:708`).
    - Solve, then undo the scaling: `X_cam = R (X − c) + s t'`.
    - Then `R_PW = R_sᵀ R` and `t_PW = R_sᵀ (s t' − R c) + origin`.
  - ORB-SLAM3's MLPnP builds bearings from `F.mpCamera` only, and skips keypoints past `mvKeysUn.size()` (`src/MLPnPsolver.cpp:55-78`). It scores inliers in camera 0 only (`src/MLPnPsolver.cpp:262-290`), and its minimal set is 6 (`src/Tracking.cc:3657`).
  - So on a non-rectified rig, ORB-SLAM3 ignores the second camera until `PoseOptimization`, which does add it (`EdgeSE3ProjectXYZOnlyPoseToBody`, `src/Optimizer.cc:960-975`).
  - `RigAp3p` is better than both: it has the smallest minimal set (3), and its consensus is on all cameras.
  - A true generalised P3P (Kneip's gp3p in OpenGV) would also take mixed-camera triples. kornia-rs has none, so that stays an upstream follow-up.
  - Residual: `ray_residual_sq` is a tangent-plane error times fx. It equals the pixel error near the optical axis and is up to 1/cos²θ looser at angle θ off it. It is used only to score hypotheses. The final inliers are counted in pixels by `solve_rig_pnp` and `estimate_pose`.
Knob: `tracking_loss.relocalization.while_lost: false`, `.solver: PrimaryEpnp`, `.max_jump_m: 0.5`.
Baselines:
- L0 = the proxy config of each sequence with `pipeline.loop_closing: Enabled((vocabulary: <ORBvoc>))` and every card-B knob at its default.
- Rig sequences use `configs/lamaria-rig-imu.ron` re-pointed. This config does not enable loop closing today, and without it B3 is rejected by `validate()`.
- Recorded by the validation workflow on `pr180/F` (sha, binary md5, load).
Predictions (numbers, each tied to a measurement):
  B3-P1 (unit) AP3P lifted to the rig: on the 70° / 0.11 m rig of `rig_pnp.rs:401-447`, three camera-1 matches give one of up to 4 solutions within 2e-5 rad and 1e-4 m of the truth.   measured by: unit test in `gpnp.rs`.
  B3-P2 (unit) RANSAC success at inlier ratio 0.2 (100 matches, 20 inliers at 0.5 px), 300 iterations, seeds 0..99. Success means within 5° and inliers ≥ 15. `PrimaryAp3p` / `RigAp3p`: 80-98 of 100. `PrimaryEpnp`: 3-20 of 100. Theory: 91 and 9.   measured by: unit test in `relocalization.rs`.
  B3-P3 (unit) Starved primary: 10 matches on the primary and 80 on camera 1, 25 % outliers at 20 px. `PrimaryEpnp` and `PrimaryAp3p` return `None` (fewer than 15 correspondences). `RigAp3p` returns a pose within 0.01 m and 1e-3 rad of the truth, with at least 50 inliers.   measured by: unit test.
  B3-P4 (unit) Mono frame: `RigAp3p` and `PrimaryAp3p` give the same inlier count, and poses within 1e-6 m of each other after `estimate_pose`.   measured by: unit test.
  B3-P5 R_11 rig, L0 + `while_lost` + `RigAp3p`, 3 samples: `relocalization_attempts` / `coast_inertial` in 0.9-1.0. The trigger fires on almost every coasted frame.   measured by: the run summary counters.
  B3-P6 `relocalizations_while_lost`, 3 samples each: L0 0 on R_11 and 2_11 (by definition). Knob on: 3-30 per run on R_11, and 5-60 per run on 2_11.   measured by: counters.
  B3-P7 Jump-rule share `relocalization_refused_jump` / (`relocalization_refused_jump` + `relocalizations_while_lost`) on R_11: 0.2-0.8.   measured by: counters.
  B3-P8 2_11 `give_ups` (27 on the pc1 base; re-measured on L0): down by 10-40 % with the knobs on.   measured by: counters, 3 samples.
  B3-P9 Tracked share: 2_11 +2 to +8 points, R_11 +0.5 to +5 points, against L0.   measured by: the bench harness tracked share, 3 samples.
  B3-P10 Score: the 5-sequence proxy CP mean stays within ±3 points of L0. Sim3 stays FAIL on R_11 and 2_11, because the disconnects of G1 and G22 remain.   measured by: the harness proxy (`eval.py`), ±50 µs samples.
  B3-P11 Solver split, with `while_lost` on (R_11 and 2_11 rig, 3 samples): adopted relocalisations rank `RigAp3p` ≥ `PrimaryAp3p` ≥ `PrimaryEpnp`. `RigAp3p` / `PrimaryEpnp` is 1.2-3.0.   measured by: `relocalizations`.
  B3-P12 EuRoC MH_01 with `--blackout 40:1`, `40:2` and `40:4`, one sample each plus ±50 µs.
    - When the ground-truth coast error at the end of the blackout is 0.3-0.5 m: the knob-on run tracks again within 2 frames of the images returning. The L0 run needs ≥ 10 frames or gives up.
    - When it is below 0.3 m: both runs track again within 2 frames, and the knob-on run is `cmp`-identical to L0 (tracking succeeds before any relocalisation is due).
    - When it is above 0.5 m: `relocalization_refused_jump` > 0 and no gain.
    measured by: ground-truth coast error from `eval.py`; frames to re-acquire from the trace.
  B3-P13 Every adopted relocalisation on the EuRoC blackout runs lies within 0.10 m of ground truth.   measured by: `eval.py` per-frame error at each `[reloc]` frame.
  B3-P14 EuRoC MH_01, MH_03, MH_04, V1_03, V2_02 without a blackout: every run with `relocalizations` = 0 is `cmp`-identical to L0. The rest have ATE within ±5 % of L0.   measured by: `cmp`, `eval.py`.
  B3-P15 Cost: `relocalization_ns` / `relocalization_attempts` ≤ 15 ms. R_11 total tracking wall time +3 % at most against L0.   measured by: counters, solo runs only (load within ±2).
  Mechanism counter: `relocalizations_while_lost` (P6), and the jump share (P7).
Kill rules:
  K1 (mechanism moved, score did not) P6 is in range, but the tracked share on 2_11 and R_11 moves by less than +1 point (inside noise) -> the adopted poses lose tracking again at once. The theory is missing the state after a relocalisation (the velocity is reset to `None` at `system/mod.rs:2043`; the coast velocity and the tilt are not corrected). Inspect the 10 frames after each `[reloc]`, then plan a velocity re-seed.
  K2 (score moved, mechanism did not) Tracked share or CP moves outside the ±50 µs band on a sample with `relocalizations_while_lost` = 0 -> not G19. Run the identity gate (P14 and the knob-off `cmp`); the cause is a leak or float order.
  K3 The jump share in P7 is above 0.8 -> the 0.5 m bound blocks G19, as the tilt arithmetic predicts. Do not adopt. The next experiment is `max_jump_m` 1.0 and 2.0, both scored with P13.
  K4 The solver split in P11 is inverted (`RigAp3p` < `PrimaryAp3p`) -> the lifted scoring or the hypothesis-camera rule is wrong. Re-check P1 to P3 before any further runs.
  K5 Any adopted relocalisation more than 0.5 m from ground truth (P13) -> a perceptual alias was accepted. Raise `min_inliers` or put the pose gate on relocalised poses before any adoption.
  K6 A run with the knobs off is not `cmp`-identical to `pr180/S`, or a knob-on run with `relocalization_attempts` = `relocalizations` = 0 after IMU confidence differs from L0 -> a knob leaks. Find it before any other B3 number.
Bit-identity:
- With all three knobs at their defaults, `pr180/L-B` and `pr180/F` `cmp` equal `pr180/S` on EuRoC MH_01 and LaMAria 3_18.
- A failed relocalisation has no side effects: `tracker.rs:142-144` drops tracks only on success, and the candidate query is read-only. So any knob-on run with no adopted relocalisation equals L0.
Noise: 3 samples (base, ±50 µs) on R_11 and 2_11. One sample elsewhere, marked provisional. The CP band on 1_19 moves 7-21 points under ±50 µs (ledger G20), so no single-sample score verdict.

## Experiment B4: a re-seeded segment tracks its own window first (G28)
Gap tested: G28 (ledger `gap-ledger.md:52`). It feeds G1 and G22.
Base: `pr180/S`.   Binary tag: `pr180/L-B`.
Mechanism:
- After a re-seed with the inertial state kept, the new segment's first keyframes grow and fuse against the 10 most recent keyframes (`mapping/keyframe_mapping.rs:130-155`). Those are the pre-loss keyframes and the lost-mode keyframes, which carry the drift of the loss.
- `SearchInNeighbors`-style fusion links the seed's landmarks to old ones (`keyframe_mapping.rs:200-215`).
- Tracking's local map then follows the votes of those fused points into the old keyframes (`tracking/local_map.rs:52-87`). The whole-map fallback also opens when fewer than 4 points are selected (`local_map.rs:111-118`).
- So a 10-16 point seed tracks against the kept map. Ledger evidence:
  - ir1p1 frame 6853: a 16-point seed, then 6-17 m steps per frame at 6891-6897 with 47-97 inliers;
  - pg1rsp2: 5.2-12.7 m at 6891-6903, with PnP 2-12 m off the IMU prediction.
Theory:
- ORB-SLAM3 starts a new Atlas map when LOST (`src/Tracking.cc:2011-2031`). `CreateMapInAtlas` clears `mpLastKeyFrame` and `mpReferenceKF` (`src/Tracking.cc:2662-2700`, at 2690 and 2693).
- As a result, the new map's keyframes find neighbours only by the new map's covisibility (`src/LocalMapping.cc:388-405`). Its inertial `mPrevKF` padding stops at the new map's first keyframe.
- Tracking votes only reach keyframes of the new map (`src/Tracking.cc:3457-3550`). Relocalisation candidates come from the current map only (`src/KeyFrameDatabase.cc:834`).
- The old map rejoins only through a verified merge (`src/LoopClosing.cc:1215`, `MergeLocal`).
- We keep one map, with segments (`4a2ee7b`, B5). With `own_window_keyframes = N`, from the second segment on and while that segment holds fewer than `N` keyframes:
  - growth and fusion neighbours, both lists, are filtered to keyframes at or after the segment's first keyframe (`prepare_seeded_window_growth`, which post-filters `prepare_keyframe_growth` under the run's `GrowthNeighbours` rule);
  - the local map keeps only points observed by such a keyframe (`filter_to_window`, applied inside `select_local_map_points` after lane A's rule, whole-map fallback included);
  - the observation-count selection (`system/mod.rs:1551-1564`) uses the same config, so old points that are not searched are not counted as "visible, not found" and culled.
- Post-filters leave Lane A's rule bodies untouched (plan, shared-file owners). The skeleton owns the shared signatures and call sites.
- Window end. With N = 10 (`MAX_COVIS_KFS`, `keyframe_mapping.rs:134`), once the segment holds 10 keyframes the recency neighbours are all inside the segment anyway. The two rules then agree, and lifting the window changes nothing abruptly.
- After that, old landmarks come back only through fusion with the segment's own keyframes, loop closing, or B5.
- The first segment never has an older one, so the knob does nothing before the first give-up.
- It acts after cold give-ups as well as kept ones, because the map survives a give-up in both cases (`system/state.rs:145-167`).
Knob: `tracking_loss.reseed.own_window_keyframes: 0`. Scored at 10.
Baselines: L1 = L0 + `tracking_loss: (reseed: (keep_inertial: true, min_seed_points: 10), pose_gate: Enabled(()))`. These are the pg1rs settings of pose-gate Exp 2, where G28 reproduced. 3 samples on R_11 and 2_11.
Predictions (numbers, each tied to a measurement):
  B4-P1 (unit) `select_local_map_points` with `window_start_kf: Some(s)` returns only points observed by a keyframe at or after `s`, including on the fallback path. `prepare_seeded_window_growth` returns only such neighbours, in both lists. With `window_start_kf: None`, the selection equals today's exactly.   measured by: unit tests in `local_map.rs` and `keyframe_mapping.rs`.
  B4-P2 Mechanism: `seeded_window_dropped_points` ≥ 100 summed over each R_11 sample that re-seeds. The old map was in reach of the seed.   measured by: counters.
  B4-P3 The largest per-frame step of the written trajectory in the 50 frames after each kept re-seed on R_11: L1 5-17 m (ledger), knob on ≤ 1.0 m, on all 3 samples.   measured by: `slam-viba/audit/posegate/stepclass.py`.
  B4-P4 `[predgap]` PnP-versus-IMU position gap in those 50 frames: L1 up to 2-12 m, knob on ≤ 0.5 m.   measured by: the debug log `[predgap]` lines.
  B4-P5 The re-seeded segment reaches 10 keyframes without a further give-up on at least 2 of 3 R_11 samples.   measured by: `[recover] segment` lines and `give_ups`.
  B4-P6 The join error at the re-seed (`slam-viba/audit/reseed/joins.py`; ir1 median 9.8-12.4 m) stays within ±2 m of L1. B4 does not change the seed pose, only what the seed then locks onto.   measured by: `joins.py`.
  B4-P7 Score: R_11 Sim3 stays FAIL. The proxy CP mean is within ±3 points of L1.   measured by: `eval.py`.
  Mechanism counter: `seeded_window_dropped_points` (P2) with the steps of P3.
Kill rules:
  K1 (mechanism moved, score did not) P2 and P3 hold, but the join error (P6) and the coverage after the re-seed do not improve -> isolation removes the lock-on but the seed pose is itself wrong (G21, G25). The theory is incomplete; next comes B5 or a seed-pose fix, not a longer window.
  K2 (score moved, mechanism did not) The steps fall to ≤ 1 m while `seeded_window_dropped_points` = 0 -> the old map was not the source of the steps. Re-open G29 for these frames.
  K3 P5 fails (a further give-up inside the window on 2 of 3 samples) -> the old map was holding the 16-point seed up. Try N = 5, or B5 together with B4, before any verdict.
Bit-identity: with `own_window_keyframes: 0`, `cmp` equality with `pr180/S` (identity gate). With N > 0, runs without a give-up are `cmp`-identical to L1 (EuRoC MH_01 and LaMAria 3_18, if no give-up there).
Noise: 3 samples on R_11 (base, ±50 µs). The re-seed frame moves between samples, so P3 to P5 are read per sample.

## Experiment B5: map segments and relocalisation once lost (port of `4a2ee7b`)
Gap tested: G1 and G22 (a disconnected trajectory after a give-up), via the light Atlas.
Base: `pr180/S`. The port is re-cut on the new tip from `4a2ee7b` only. Do not take `a95f8eb` or `aa0b2bf`: `fc6c8c5` reverted them on evidence.   Binary tag: `pr180/L-B`.
Mechanism:
- After a give-up with the inertial state kept, the system re-seeds and coasts (`system/mod.rs:827-847, 854-890`).
- No relocalisation runs in that state: the bootstrap path never calls `relocalize`. In tracking, `relocalize` is skipped while the IMU is confident (`system/mod.rs:1576-1583`).
- So the camera can never find the older map again. Moreover, a match there would be refused, because its drift against the coast exceeds 0.5 m (`system/mod.rs:2020`).
Theory (`4a2ee7b`'s commit message):
- Keyframes are tagged by segment. A new segment begins at each seed of the map, a cold start or a re-seed (`Segments::begin` in `finish_bootstrap`, `system/mod.rs:1159`).
- While re-seeding, with `relocalize_when_lost`, the system calls `relocalize(frame, lost = true)`:
  - a match in an older segment is adopted whatever the jump;
  - a match in the segment just lost keeps the `max_jump_m` bound (`relocalization_adopted`).
- Tracking resumes at the relocalised pose with no IMU edge across the jump (`resume_after_lost_relocalization`). Loop closing corrects the lost segment when it next closes a loop with it.
- This is the jump-version of ORB-SLAM3's Atlas merge. ORB-SLAM3 merges the maps after a verified common region (`src/LoopClosing.cc:324, 1215`) instead of moving the camera. We adopt the older map's frame, because its drift is unknown either way.
- The `[recover] segment N begins` debug line prints only when `relocalize_when_lost` or `own_window_keyframes` is set, so knob-off logs stay comparable.
Knob: `tracking_loss.reseed.relocalize_when_lost: false`. It has effect only with `keep_inertial: true`, because re-seeding needs it (`system/state.rs:171-173`), and needs a vocabulary (validate).
Baselines: L1 (B4), 3 samples on R_11 and 2_11.
Predictions:
  B5-P1 (unit) `4a2ee7b`'s two tests pass unchanged: `keyframes_belong_to_the_segment_they_were_seeded_in` and `a_lost_camera_adopts_any_jump_into_an_older_segment_only`.   measured by: `system/recovery.rs` tests.
  B5-P2 Mechanism: `relocalizations_older_segment`: L1 0. Knob on: 0-3 per R_11 sample and 1-10 per 2_11 sample (2_11 gives up 27 times on pc1, so it has more re-seeds that face a mapped place).   measured by: counters.
  B5-P3 `reseed_stall_frames` on R_11 (ir1: 178-191): with P2 ≥ 1, it falls by at least the number of stall frames after the first adopted relocalisation. With P2 = 0, the run is `cmp`-identical to L1.   measured by: counters, `cmp`.
  B5-P4 2_11 tracked share +1 to +6 points against L1.   measured by: harness, 3 samples.
  B5-P5 Score: proxy CP mean within ±3 of L1. Sim3 on 2_11 can flip to PASS on at most 1 of 3 samples.   measured by: `eval.py`.
  Mechanism counter: `relocalizations_older_segment` (P2).
Kill rules:
  K1 (mechanism moved, score did not) P2 ≥ 1, but the tracked share and the joins do not improve -> the older-segment pose teleports the trajectory as far as the coast was off. A merge that corrects the segment (a PGO edge, as `MergeLocal` does) is needed instead of a jump. Not adopted.
  K2 (score moved, mechanism did not) The score moves outside the band with P2 = 0 -> another cause: the segment bookkeeping leaked into the output. Check the knob-off `cmp`.
  K3 P2 = 0 on all 5 proxy sequences -> no re-seed faced a mapped place within its stall. Record "no opportunity", not "refuted"; re-test with B4, which shortens nothing but keeps the seed from locking on.
Bit-identity: with `relocalize_when_lost: false`, `cmp` equality with `pr180/S`. `Segments` bookkeeping runs always and never feeds an output with the knobs off.
Noise: 3 samples on R_11 and 2_11. One sample elsewhere, marked provisional.

## Files to touch (lane B)

| file | change |
|---|---|
| `crates/kornia-sensors/src/rig.rs` | `camera_ray`, `RigCamera::ray`, `SensorRig::camera_to_body_of`, tests B2-P1/P2 |
| `crates/kornia-sensors/src/rig_calibration.rs` (new; the skeleton adds the `lib.rs` line and the serde derive on `ImuCalib` in `imu.rs`) | B1 types, `from_sensor_rig`, `apply_to`, `rig_cameras_from_body_extrinsics`, tests B1-P1/P4 |
| `crates/kornia-slam/src/upstream/kornia_3d/gpnp.rs` (the skeleton adds the `mod` line in `upstream/kornia_3d/mod.rs`) | B3 lifted AP3P, residual, estimator, sampler, RANSAC, test B3-P1 |
| `crates/kornia-slam/src/rig.rs` | `RigView::primary_from_camera`, `RigView::ray` |
| `crates/kornia-slam/src/tracking/relocalization.rs` | `RelocalizationSolver`; all-camera correspondences for `RigAp3p`; `PrimaryAp3p`; the `RigAp3p` path through `gpnp` and `solve_rig_pnp`; tests B3-P2/P3/P4 |
| `crates/kornia-slam/src/tracking/tracker.rs` | a `RelocalizationConfig` from the settings (`set_window_start` and `local_map_config` are the skeleton's) |
| `crates/kornia-slam/src/tracking/local_map.rs` | `filter_to_window` body only (Lane A owns the rule; the skeleton owns the signatures); test B4-P1 |
| `crates/kornia-slam/src/mapping/keyframe_mapping.rs` | `prepare_seeded_window_growth` body only; test B4-P1 |
| `crates/kornia-slam/src/system/recovery.rs` | `relocalize_when_lost`, `own_window_keyframes`, `relocalization_adopted`; tests B5-P1 (`SegmentId` and `Segments` come from the skeleton) |
| `crates/kornia-slam/src/system/mod.rs` | trigger hook `relocalization_due`; `relocalize(.., lost)` with `max_jump_m`; `4a2ee7b` hunks (`segments.begin`, the re-seeding call, `resume_after_lost_relocalization`); the `seeded_window_start` body, set each frame; counter increments. The skeleton writes the declarations, the `segments.begin` call, the growth dispatch at `:2121` and the config at `:1551`. Lane A owns IMU trust in this file. |
| `crates/kornia-slam/src/system/config/mod.rs`, `build.rs` | `LostRelocalization`, `TrackingLossConfig.relocalization`, settings plumbing |
| `crates/kornia-slam/src/system/validation.rs`, `config/tests.rs` | `RelocalizationWithoutVocabulary`, `max_jump_m` check; tests: each of the 3 knobs × loop closing off/on, plus the defaults |
| `apps/kornia-slam-app/src/config.rs`, `main.rs` | bodies only (the skeleton declares the field and the flag): `RunConfig.rig_calibration` (resolved against the run file, applied to `source.rig()` before `SlamSystem::build`; error if its camera count differs from the source's rig); `--write-rig-calibration` |
| `apps/kornia-slam-app/src/source/lamaria.rs` | `rig()` through `rig_cameras_from_body_extrinsics`; test B1-P2 |
| `CHANGELOG.md` | one line per knob (written by the skeleton) |

## Benchmark protocol for lane B (for the validation handoff)

1. **Identity gate first:** B1 to B5 knobs off, `pr180/F` against `pr180/S`, `cmp` on EuRoC MH_01 and LaMAria 3_18 (mono and rig). Nothing below counts until this passes.
2. **B1-P3:** LaMAria 3_18 rig, run with the file written by `--write-rig-calibration`, `cmp` against the JSON run.
3. **`--blackout` series (B3-P12/P13):** EuRoC MH_01 at `40:1`, `40:2` and `40:4`, L0 against L0 + B3, one sample each plus ±50 µs. This is the controlled test with ground truth.
4. **B3 on L0:** R_11 and 2_11 rig, 3 samples each. The other 3 proxy sequences get 1 sample. EuRoC MH_01, MH_03, MH_04, V1_03 and V2_02 get 1 sample each (P14).
   - Solver split (P11) on R_11 and 2_11: `PrimaryEpnp`, `PrimaryAp3p`, `RigAp3p`, all with `while_lost` on.
5. **B4 and B5 on L1:** R_11 and 2_11, 3 samples each. Read the R_11 give-up window, 6239-8241 on rg1 (the frames move with the base), for P3 to P5 of B4 and P2 and P3 of B5.
   - Then B3+B4+B5 together on L1, as the lane bundle.
6. **Timing (B3-P15):** solo runs only, load within ±2 of the baseline.

## Open points for the card review

- **G19 goes beyond ORB-SLAM3:** ORB-SLAM3 does not relocalise during an inertial RECENTLY_LOST. B3 is justified by G21, the bad last poses ORB-SLAM3 does not accept. K3 expects the 0.5 m bound to block much of it.
- **Two knob choices are this card's, not the plan's:**
  - the `PrimaryAp3p` variant, which separates the minimal-set effect from the rig effect;
  - `max_jump_m`, which exposes today's constant so K3's follow-up needs no code.
- **B4 post-filters Lane A's selection:** `filter_to_window` and `prepare_seeded_window_growth` run on the output of lane A's rule, through signatures the skeleton declares (card A, "Skeleton contract"). If Lane A changes the output set, B4 inherits that change.
- **Skeleton-owned in lane B's files (review round 3b'):** `RunCounters` fields, the shared summary line (`summary.rs`), `RunConfig.rig_calibration` and `--write-rig-calibration`, `Tracker::set_window_start`, the `local_map.rs`, `keyframe_mapping.rs` and `map_projection/mod.rs` signatures and call sites, the `imu.rs` derive, and the `gpnp` `mod` line. Lane B fills bodies only.
- **Not covered:** an EuRoC source that yields raw rig images, so a rig calibration file could drive EuRoC as a non-rectified rig. B1's file format allows it, but no source here supplies those images.
