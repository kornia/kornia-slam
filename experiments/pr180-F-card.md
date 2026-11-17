# PR #180 card F: raw Aria fisheye on both rig cameras

DEFERRED by the owner on 2026-11-17 to its own round; blocking decisions D1-D6 below stay open.

Plan: `/home/edgar/.claude/plans/pr180-frontend-1ms.md`, Lane F (section 9), knobs from section 8, defaults from section 14. The roadmap and `lamaria-handoff.md:116,187` call this item "A9". Card A already uses A9 for `mapping.growth_neighbours` (`pr180-A-card.md:47,346`), so this card uses the id **F** and the gap id **G12**.

Today LaMAria runs on the pinhole ASL export: 758×572 per camera, f = 241.6 (`apps/kornia-slam-app/src/datasets/lamaria.rs:1-2,185-214`). That export crops each Aria SLAM camera from about 150° to about 115° horizontally. Lane F feeds the raw 640×480 fisheye images to the rig instead. Keypoints are mapped into a virtual pinhole, and the BA weights and search radius are scaled by the pinhole magnification. Nothing here has been measured, and the raw images are not on disk. Every number below is arithmetic on the R_11 Aria calibration (`$S/bench/lamaria/data/R_11_5cp/aria_calibrations/R_11_5cp.json`) or a prediction for the validation workflow.

## Blocking: what the plan does not settle

This card stops here for a decision. Six points in Lane F cannot be built as the plan writes them, or need a choice the plan does not make. Each one lists the options, with no choice made. The rest of the card is written so it holds under any of the options, unless a line names a decision.

**D1. `RigCamera.raw_model: Option<Fisheye624Camera>` is a crate cycle.**
- `RigCamera` lives in kornia-sensors (`crates/kornia-sensors/src/rig.rs:37-46`).
- `Fisheye624Camera` lives in kornia-slam, in `upstream/kornia_3d/camera_fisheye624.rs` (plan section 8, Lane K).
- kornia-slam depends on kornia-sensors (`crates/kornia-slam/Cargo.toml:22`). kornia-sensors depends only on kornia-3d, kornia-algebra and kornia-image (`crates/kornia-sensors/Cargo.toml`). Plan step F1 cannot compile.
- Options:
  - (a) Put `Fisheye624Camera` in kornia-sensors (for example `kornia-sensors/src/camera_fisheye624.rs`), marked for kornia-3d later. Lane K then owns a kornia-sensors file, not an `upstream/` file. The `upstream/` convention allows this, because kornia-sensors is a workspace crate and is edited directly.
  - (b) Keep the type in `upstream/` and store it only on `crate::rig::RigView`. The app hands it over through a new kornia-slam API, for example `SlamSystem::build` options. That is a skeleton-owned signature.
  - (c) In kornia-sensors, store a model-agnostic `Option<Arc<dyn kornia_3d::camera::CameraModel + Send + Sync>>` (the trait is at `$K/kornia-3d/src/camera/model.rs:9`). This needs a `Debug` wrapper, because `RigCamera` derives `Debug` and the trait does not require it. The magnification (D3) then cannot reach the distortion coefficients.

**D2. The virtual pinhole has no image extent, so the plan's design would lose field of view.**
- In virtual-pinhole pixels with f = 241.6, the raw pixels at θ < 80° span x ∈ [−1309, 1285] and y ∈ [−966, 964] around the principal point: a 2594×1931 box. The raw image is 640×480.
- Three places bound by the raw image size:
  - the projection search bounds by `view.image_size` through `project_to_image` (`map_projection/mod.rs:906-909`);
  - the keypoint grid clamps keypoints into the same size (`map_projection/mod.rs:1016-1023,1031,1050-1053`; `keypoint_grid.rs:17-28`);
  - the rig overlap box samples `view.size` (`rig.rs:116-117`).
- Camera 0's `image_size` is `primary.size()`, which is the raw size (`frontend/mod.rs:197`, Lane T's file).
- `RigCamera.size` must stay the raw image size, because `system/input.rs:58` rejects an image of any other size.
- With raw 640×480 bounds, the usable half-angles in virtual pixels are atan(320/241.6) = 52.9° and atan(240/241.6) = 44.8°. Today's crop has 57.7° and 49.9°. **The plan as written would give a smaller field of view than today.**
- Options:
  - (a) A separate virtual canvas per camera (size, plus a shifted cx', cy'), sized to the 80° cone. `RigView` gains it, and camera 0's frame `image_size` becomes the canvas. That changes `frontend/mod.rs:197` (Lane T) and every consumer of `Frame::image_size` (23 non-test lines in 22 files).
  - (b) Bounds from the unprojected raw border. This is ORB-SLAM3's `Frame::ComputeImageBounds` for distorted pinholes (`ORB_SLAM3/src/Frame.cc:782-813`), which undistorts the four image corners. Here the border is clipped at 80°, plus a validity mask, so that a canvas corner the lens never sees does not count as "in view". Without the mask, the visibility counters (`system/mod.rs:1676-1704`, Lane T) count invisible points and cull good map points.
  - (c) Bound in raw pixels instead: project through `Fisheye624Camera` and test against 640×480. That is the camera-generic path ORB-SLAM3 takes for `KannalaBrandt8` (`Frame.cc:749-752` keeps `mvKeysUn = mvKeys`). Plan section 6 rejects that path for BA, but it could be used for the bounds test alone.

**D3. `pinhole_magnification(theta: f64) -> (f64, f64)` has no model input, and the sigma is a scalar.**
- The magnification from raw to virtual pixels is `(radial, tangential) = (sec²θ / θ_d'(θ), tanθ / θ_d(θ))`. With Aria's k0-k5, the radial term at 80° is 43.3. The equidistant form `1/cos²θ` gives 33.2. The tangential term is 4.08 against 4.06. A free function of θ alone can only be the equidistant approximation.
- `observation_sigma_px` is one `f32` per observation (`vi_ba_schur.rs:260`), but the magnification is anisotropic. At 70°, the radial term is 9.06 and the tangential term 2.21.
- Two choices are open:
  - (i) a method `Fisheye624Camera::pinhole_magnification(&self, theta)` or the free equidistant function;
  - (ii) the scalar: the radial term (conservative, under-weights the tangential direction), the geometric mean `sqrt(radial · tangential)` (keeps the information determinant), or a 2×2 information (exact, but it changes the BA residual weighting API).

**D4. The sigma and the knob must reach call sites that no lane owns.**
- The per-observation sigma is computed at `mapping/bundle_adjustment.rs:1142-1145`, and only when `!information.is_unit()`. It is then passed on at `:613,757`.
- Other sigma sites: `global_vi_ba/problem.rs:283` (Lane C, imports only) and `tracking/relocalization.rs:339-350` (no lane). The PnP refine is a further site.
- `bundle_adjustment.rs` and `relocalization.rs` are in no lane's "Owns" list (plan section 7). `vi_ba_schur.rs` is Lane C's.
- Options: the skeleton adds a `fisheye_weight: FisheyeWeight` field to `BundleAdjustmentParams` (next to `visual_information`, `bundle_adjustment.rs:54,120`) and the scaled-sigma call at `:1142-1145`, and Lane F fills the body. Or `bundle_adjustment.rs` is added to Lane F's files. The relocalisation and offline-BA sites also need a decision: in scope, or left on unit weight with the reason recorded.

**D5. The rig overlap match triangulates raw pixels as pinhole pixels.**
- `frontend/overlap.rs:74,81-93` reads `features.keypoints_xy` with each view's pinhole `K`. It runs inside `prepare_rig` (`frontend/mod.rs:182`), before `ensure_undistorted`.
- With raw fisheye images, `keypoints_xy` stays raw (plan step F3 maps only the undistortion cache). The overlap's stereo points, and the metric scale they give a rig frame (`frame.rs:66-71`), would be wrong.
- `overlap.rs` is in no lane, and `frontend/mod.rs` is Lane T's.
- Options: `match_overlap` reads the undistortion cache (the skeleton moves `ensure_undistorted` ahead of it in `prepare_rig`, and `overlap.rs` joins Lane F), or `prepare_rig` remaps keypoints as the mono `FisheyeMapping` does (`frontend/mod.rs:290-335`). The second option breaks Lane T's raw-pixel KLT.

**D6. The raw image layout is unknown.**
- The fetch scripts download only `asl_folder/training/<seq>.zip` (the pinhole export), the calibrations and the ground truth (`$S/bench/lamaria/fetch.sh:7-11`). Where and in what format the raw 640×480 frames come (VRS, or an extracted ASL folder) is not on disk.
- Reading VRS needs a dependency that plan section 8 does not list.
- Options: assume an ASL-layout extraction (`<data>/aria/{cam0,cam1}/data.csv` plus PNGs, made by the validation workflow with projectaria_tools), or make the image root a separate path inside `RawFisheye`.

Minor point, settled by the plan's struct: `Fisheye624Camera` has one `f`, and the Aria JSON has `fx` and `fy` (equal on R_11: 241.6048925567284 for both). The loader rejects `|fx − fy| > 1e-9 · fx`.

## Experiment: F, raw Aria fisheye
Gap tested: **G12**, "pinhole crop of about 110-116° instead of the 150° fisheye" (`slam-viba/audit/gap-ledger.md:38`, status suspected, score at stake unknown). The row is not yet in `experiments/gap-ledger.md`. The docs step should add it.
Base: kornia-slam@`pr180b/K` (the skeleton commit, not yet tagged; today `c0a1dc3`), kornia-rs@`4ada5a7`.   Binary tag: `pr180b/L-F`.

Mechanism (file:line on `c0a1dc3`):
- **F-M1, the crop.** The loader accepts only `PINHOLE` JSON (`datasets/lamaria.rs:185-198`) and reads the ASL export (`:84-96`). The export's half-angles at f = 241.6 are 57.7° (horizontal) and 49.9° (vertical), or 2.80 sr of solid angle per camera. The raw camera at θ < 80° covers 4.24 sr: **1.52×** (sampled on the R_11 cam0 model, radial terms only; tangential and thin-prism terms are about 1e-3).
- **F-M2, image pixels outside the crop.** 36.3 % of the raw pixels at θ < 80° lie outside the crop's frustum. By θ bin, the raw pixel shares are: 0-30° 17.4 %, 30-50° 32.2 %, 50-60° 21.2 %, 60-70° 17.0 %, 70-80° 12.2 %. 6.2 % of the raw pixels are past 80° (0.8 % of them lie outside the lens circle, which ends at 376.9 px).
- **F-M3, no per-camera raw model.** A rig camera is a pinhole only (`kornia-sensors/src/rig.rs:39-46`, `kornia-slam/src/rig.rs:33-45`). Raw fisheye exists only for the single-camera `SensorRig::fisheye` (Kannala-Brandt, `kornia-sensors/src/rig.rs:27-29`). Validation rejects it on a rig (`system/validation.rs:768-773`).
- **F-M4, uniform weights.** A virtual-pinhole residual at θ is a raw residual times the magnification (D3). With `VisualInformation::Unit`, a 75° observation keeps the same weight as a central one, while its pixel noise is about 15× larger radially. The weight is `1/σ` at `vi_ba_schur.rs:1435-1438` and `:1809-1811`.
- **F-M5, fixed search radius.** `ProjectionMatchConfig::search_radius` (15 px, `map_projection/mod.rs:83,92`; 30 px local, `:150`) is applied in virtual pixels at every θ (`:932-934`). A rotation error δ moves a projected point by f·δ·sec²θ radially. At 75°, a radius that is right at the centre is about 15× too small.

Theory:
- **Field of view.** Tracking is lost in fast turns and close-range scenes (R_11 frames 4850-6866, `pr180-frontend-1ms.md` section 1). There the landmark count falls below the PnP minimum. A 1.52× solid angle keeps landmarks in view for longer during a yaw. Each Aria SLAM camera adds about ±17° horizontally (from 57.7° to 74.5-76.1° at the raw image edges), which also widens the overlap between the two cameras. That gives more stereo points, and so more metric scale, per rig frame (`frame.rs:66-71`), once D5 is fixed.
- **Why a virtual pinhole and not camera-generic BA.** The plan settles this (sections 6 and 14): about 190 `PinholeCamera` sites across 29 files. ORB-SLAM3 is camera-generic for fisheye: `KannalaBrandt8` keeps raw keypoints and projects through the model (`ORB_SLAM3/src/Frame.cc:749-752`; Newton unproject at `CameraModels/KannalaBrandt8.cpp:116-135`). The virtual pinhole is exact for geometry, because a bearing is a bearing. It is wrong only in the noise model, and F-M4/F-M5 correct that to first order.
- **Why the cap is 80°, not 88° as on mono** (`frontend/mod.rs:22`). The pinhole radius grows as tanθ: 1370 px at 80°, 6919 px at 88°. The Aria radial magnification at 80° is 43×, so a point there carries about 1/43² = 5e-4 of a central point's information, even with the right weight. Rays from 80° to the lens limit hold 5.4 % of the pixels and add almost nothing. Above about 100°, the Aria polynomial folds (θ_d stops growing at θ = 1.745 rad), so the unprojection must stop below that anyway.
- **The magnification weight.** Whitening a virtual residual by `σ_raw · m(θ)` makes its χ² match a raw-pixel χ² in the direction `m` describes. That is the first-order covariance `J Σ Jᵀ` with `J = d(virtual)/d(raw)`. D3 decides the scalar or the matrix. Prediction FP5 tests it per θ bin, the detector named in plan section 13.
- **The search radius.** Multiplying the radius by the radial magnification keeps the pose-error bound in angle constant at every θ. The cost is more candidates per query in the outer bins (FK6).
- **Speed.** 640×480 has 0.709× the pixels of 758×572, so ORB detection and pyramid cost about 0.71× per camera. The per-keypoint Newton unprojection is new cost (FP7).

Knob (plan section 8; every default is today's behaviour):

| Knob | RON path | Type | Default | Preset |
|---|---|---|---|---|
| `fisheye_weight` | `system.pipeline.OrbSlam.frontend.Orb.fisheye_weight` (a field of `OrbFrontendConfig`, `system/config/mod.rs:124`) | `FisheyeWeight { Uniform, Magnification }` | `Uniform` | `Magnification` |
| `images` | LaMAria source `images` (a field of `LamariaConfig`, `apps/kornia-slam-app/src/source/lamaria.rs:27-46`) | `LamariaImages { Pinhole, RawFisheye { calib: PathBuf } }` | `Pinhole` | `RawFisheye(calib: ".../aria_calibrations/<seq>.json")` |

- `fisheye_weight` acts only on observations of a rig camera whose view has a raw model, and it scales both the BA sigma and the projection search radius. With `Uniform`, or with no raw model, the code paths are unchanged.
- `RawFisheye.calib` is the Aria calibration JSON. The existing `calib` (the pinhole JSON) stays required, because it carries `T_b_s` and the IMU noise, which are equal in both files on R_11 (see the JSON). The loader checks that both files give bit-equal `T_b_s`, and returns an error if they differ.
- No knob for the 80° cap. It is a constant (`RAW_MAX_INCIDENCE_DEG = 80.0`) beside `ensure_undistorted`.

Baselines: none run yet. The validator records `pr180b/K` (sha, binary md5, load) on R_11 and on 3_18 with `configs/lamaria-rig-imu.ron` on `images: Pinhole`. FP3-FP6 also need the raw-image download (plan section 12, after a disk check).

Predictions (numbers, each tied to a measurement):
  FP0 Identity: with `images: Pinhole` and `fisheye_weight: Uniform`, `pr180b/L-F` against `pr180b/K` has 0 differing bytes on R_11 rig, MH_01 `euroc.ron` and the Hilti mono fisheye path (`FisheyeMapping`, untouched).   measured by: `cmp` on each output file.
  FP1 Geometry unit test (synthetic, no data): for 2000 bearings at θ ∈ [0°, 79°] on the R_11 cam0 model, `Fisheye624Camera::project`, then `ensure_undistorted`, gives the virtual pixel `f·(x/z, y/z) + c'` within 1e-3 px, and the `RigView::unproject_raw` bearing within 1e-9 rad. Bearings at θ ≥ 80° come back NaN, keeping their slot.   measured by: `cargo test --release -j2 -p kornia-slam raw_fisheye`.
  FP2 Magnification unit test: `pinhole_magnification` against central differences of raw-to-virtual at θ ∈ {10°, 45°, 70°, 79°}, within 1e-4 relative (model-aware variant), or the equidistant closed form within 1e-12 (free-function variant, D3).   measured by: that test.
  FP3 Mechanism counter, coverage: on R_11 `RawFisheye`, the share of keyframe keypoints whose virtual pixel lies outside the crop frustum (|x/z| > 382.4/241.6 or |y/z| > 286.5/241.6) is 30-40 % (pixel arithmetic: 36.3 %, ORB's grid spreads keypoints about uniformly). The share dropped by the 80° cap is 3-7 % (pixel arithmetic: 6.2 %; ORB finds fewer corners near the lens edge).   measured by: a script over the `--session-out` keyframe dump (undistorted keypoints written at `mapping/session.rs:237`).
  FP4 Mechanism counter, use: the share of keyframe map-point observations whose virtual pixel lies outside the crop frustum is 15-30 %. It is lower than FP3, because grazing rays triangulate less often and, with `Magnification`, weigh less.   measured by: the same dump, observations only.
  FP5 Weight, per incidence bin: with `Uniform`, the median squared virtual-pixel residual of BA inliers in the 70-80° bin is at least 20× that of the 0-30° bin. With `Magnification`, the median whitened χ² ratio of those two bins is in [0.5, 2.0] for the geometric-mean scalar, or in [0.2, 1.0] for the radial scalar (D3).   measured by: a per-θ-bin χ² dump of the last local-BA iteration, added next to the VI-BA `debug_eprintln!` lines (`vi_ba_schur.rs:882,1568`), off by default.
  FP6 R_11 rig tracked share: `pr180b/K` Pinhole baseline -> +2 to +8 points with `RawFisheye` plus `Magnification`; no Bootstrap entries in frames 4850-6866 on at least 2 of 3 samples. Low confidence: this is the theory's main claim, and it has no prior measurement.   measured by: the bench harness tracked share, 3 samples at IMU offsets −50, 0 and +50 µs.
  FP7 Timing: the `orb_per_camera` median ratio of `RawFisheye` to `Pinhole` is in [0.65, 0.85]. `ensure_undistorted` on a raw rig frame takes ≤ 1.0 ms per camera for 1000 keypoints on x86 (Newton on a degree-13 odd polynomial, up to 10 iterations).   measured by: `FrontendObservation::orb_per_camera`, plus a criterion bench of `ensure_undistorted` (`benches/bench_upstream_fisheye624.rs`, Lane K).
  FP8 LaMAria score on R_11 and 3_18: change in [0, +6] points against the Pinhole baseline, with no sequence losing more than its noise band.   measured by: `$S/bench/eval.py`, 3 samples per config.
  Mechanism counters: FP3 and FP4 (field of view in use) and FP5 (weight). FP6 and FP8 are score.

Kill rules (ids FK, so they do not collide with card E's K1-K8):
  FK1 FP3 and FP4 hold (the wider view is in use), but FP6 gains under 1 point and FP8 stays inside noise -> mechanism moved, score did not. Field of view is not the binding gap on R_11. Record G12 as refuted for R_11, and test 3_18 and 5_11 before dropping it.
  FK2 FP6 or FP8 improves beyond noise, but FP4 is under 10 % -> score moved, mechanism did not. The gain comes from something else: the smaller image (ORB density per pixel), or the overlap change of D5. Do not adopt it. Bisect with `fisheye_weight: Uniform`, then compare per-frame inliers inside the crop region only.
  FK3 FP0 fails -> a default path reaches the raw model or the magnification. Revert the lane commit that differs (bisect with `cmp`). Nothing knob-on is trusted until it passes.
  FK4 FP5 is outside its band with `Magnification` -> the weight is inverted, or the scalar is wrong (plan section 13, "magnification weight inverted"). Fix it before any score run. If the band misses on the tangential axis only, D3's 2×2 option is the next card.
  FK5 Overlap scale: on R_11 `RawFisheye`, the median depth of overlap stereo points differs from the Pinhole run on the same timestamps by more than ±5 %, or the overlap-matched count drops below 0.8× -> the overlap still reads raw pixels (D5). Block adoption, whatever the score.
  FK6 Search cost: the median keypoint-grid candidates per query in the 70-80° bin exceed 6× the 0-30° bin, or tracking time per frame rises by more than 10 % against Pinhole -> the radius scale is too generous. Cap the scale (for example at 4×) in a follow-up card. Do not tune it in this one.
  FK7 FP7 timing: the ratio is above 1.0, or `ensure_undistorted` takes more than 1.5 ms per camera -> re-profile the Newton unprojection (iteration count, seed from the equidistant inverse) before any score run.
  FK8 `Magnification` scores below `Uniform` beyond noise while FP5 holds -> the first-order weight is incomplete (anisotropy, or bias at grazing angles). Keep `Uniform` as the preset and record it.
Bit-identity: FP0's three runs, knobs at their defaults. `RawFisheye` has no identity case, because its input images differ.
Noise: FP0-FP2 and FP5 are deterministic. FP6 and FP8 use 3 samples per config at IMU offsets −50, 0 and +50 µs (skill section 2), on R_11 and 3_18. FP7 uses 3 solo runs per tag, with load within ±2.

## Design (what the lane builds; subject to D1-D6)

### F1. Per-camera raw model
- `kornia_sensors::RigCamera` gains `raw_model` (type per D1). `RigCamera::ray` (`kornia-sensors/src/rig.rs:67-72`) keeps taking undistorted virtual pixels, so the gpnp boundary (`pr180-A-card.md`, "Skeleton contract") is unchanged.
- `crate::rig::RigView` mirrors it (`rig.rs:33-45`), copied in `RigGeometry::from_rig` (`rig.rs:72-82`).
- `overlap_box` (`rig.rs:112-151`) samples the valid region of D2 instead of the raw `view.size` when a raw model is set. Without a raw model it is unchanged, and the rig overlap tests (`rig.rs:206-221`) stay as they are.

### F2. Loader (`datasets/lamaria.rs`, `source/lamaria.rs`)
- `LamariaImages::RawFisheye { calib }` reads the Aria JSON: model `RAD_TAN_THIN_PRISM_FISHEYE`, 16 params `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`, resolution 640×480.
- The parameter order follows projectaria_tools. `lamaria-gap.md:18` marks it unchecked, and the projectaria fixture in validation (plan section 14) is its oracle.
- The virtual pinhole keeps the export's `f` per camera (241.6049 cam0, 241.3273 cam1), with the principal point and extent from D2.
- `LamariaCamera` gains `raw: Option<Fisheye624Camera>`. `camera` (`:22-30`) stays the pinhole that geometry uses.
- `LamariaConfig::validate` (`source/lamaria.rs:48-59`) rejects `RawFisheye` unless the cameras are `Rig`. The mono path has no Fisheye624 frontend, and the mono `FisheyeMapping` is Kannala-Brandt only.

### F3. `Frame::ensure_undistorted` (`frame.rs:127-145`)
- Per feature: take the view of `camera_of(i)`. With a raw model, unproject the raw pixel to a bearing, drop it to NaN if `bearing.z ≤ cos 80°`, and write `f·(x/z, y/z) + c'`. Otherwise use today's `undistort_keypoint`.
- The primary's raw model comes from `rig.geometry.view(0)`, because `secondary_view` returns `None` for camera 0 (`frame.rs:90-96`).
- The signature stays the same. `undistorted_xy`'s fallback (`frame.rs:150-159`) uses the same per-feature function, so a frame built outside the pipeline is not undistorted as a pinhole.
- This generalises `FisheyeMapping::map_keypoints` (`frontend/mod.rs:300-335`). The difference is that `keypoints_xy` stays raw, for Lane T's KLT and for colour sampling, and a dropped keypoint keeps its slot as NaN.

### F4. `fisheye_weight: Magnification`
- `visual_information.rs` gains `FisheyeWeight` (the skeleton declares the enum and its serde) and `FisheyeWeight::sigma_scale(self, view, pixel_virtual) -> f32`. It is 1.0 for `Uniform` or for no raw model, and otherwise the D3 scalar at `θ = atan(|p − c'| / f)`.
- The BA sigma becomes `VisualInformation::sigma_px(octave) · sigma_scale`. The sites are listed in D4.
- Projection search: in `match_by_projection` (`map_projection/mod.rs:931-934`), the query radius is `config.search_radius · radial_magnification(θ)` for a view with a raw model. This is the only edit to that file, as the plan's ownership allows ("search radius only").

### F5. Validation
- `system/validation.rs:768-773` keeps rejecting `rig.fisheye` on a rig. A per-camera `raw_model` does not trip that check, so nothing needs relaxing.
- New check: a `raw_model` on any rig camera with `cameras != Rig` is an error (a new `ConfigError` variant; the skeleton owns `validation.rs`).

## Skeleton contract

Every item below is declared by the skeleton (Phase 1), with `todo!()` bodies reachable only through the knobs. Items marked with a D number take their final form from that decision.

**Types and fields**
- `kornia_sensors::RigCamera { camera, size, primary_from_camera, raw_model: <D1 type> }`. The skeleton fixes every struct literal with `raw_model: None`:
  - `kornia-sensors/src/rig_calibration.rs:158,215,255,260`;
  - `kornia-sensors/src/rig.rs:251,256`;
  - `kornia-slam/src/rig.rs:173`;
  - `tracking/relocalization.rs:711,716`;
  - `system/input.rs:120`;
  - `frontend/tests.rs:280,334`;
  - `system/config/tests.rs:733`;
  - `frontend/overlap.rs:257,262`;
  - `apps/kornia-slam-app/src/config/tests.rs:266`.

  `RigCameraCalibration` (`rig_calibration.rs:24-45`) gets no raw-model field this round: its file format stays pinhole-only, and `rig_camera()` writes `None`.
- `crate::rig::RigView` gains `raw_model: <D1 type>` and the D2 extent (`virtual_size: ImageSize`, plus the principal-point shift, or a validity mask).
- `pub enum FisheyeWeight { #[default] Uniform, Magnification }`: `Debug, Clone, Copy, Default, PartialEq, Eq`, with serde under `feature = "serde"`, in `mapping/visual_information.rs`. Re-exported where `VisualInformation` is.
- `OrbFrontendConfig::fisheye_weight: FisheyeWeight`, `#[serde(default)]` (`system/config/mod.rs:124-219`). Its default is `Uniform`, plus a `Display` line only when it is not the default (`:529-551`), and a RON round-trip test in `config/tests.rs`.
- `MapProjectionConfig` gains `fisheye_weight: FisheyeWeight`, set by `system/build.rs` from the frontend config. `BundleAdjustmentParams` gains `fisheye_weight: FisheyeWeight` (D4).
- `ConfigError::RawFisheyeOutsideRig` (`system/validation.rs`), with the message "raw fisheye rig cameras need the `Rig` camera selection".
- CHANGELOG lines: `frontend.fisheye_weight`, and the LaMAria `images` source option.

**From Lane K** (`upstream/kornia_3d/camera_fisheye624.rs`, or its D1 location):
```rust
//! Upstream target: kornia_3d::camera. Kept here until upstreamed; no kornia-slam types.

/// Aria `RAD_TAN_THIN_PRISM_FISHEYE` (FisheyeRadTanThinPrism, "Fisheye624"), in raw image
/// pixels: one focal length, principal point, radial k0..k5, tangential p0, p1, thin prism s0..s3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fisheye624Camera {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: [f64; 6],
    pub p: [f64; 2],
    pub s: [f64; 4],
}

impl kornia_3d::camera::CameraModel for Fisheye624Camera { /* project, project_with_jacobian, unproject (Newton) */ }

/// (radial, tangential) magnification from raw to virtual-pinhole pixels at incidence `theta` (rad).
pub fn pinhole_magnification(theta: f64) -> (f64, f64); // D3: or a `&self` method
```

**Lane F provides, for Lane T** (declared by the skeleton in `rig.rs` with `todo!()`, filled by Lane F):
```rust
impl RigView {
    /// Whether this camera's images are raw fisheye.
    pub fn is_raw(&self) -> bool;
    /// Raw image pixel to virtual-pinhole pixel; `None` past the 80° cap or outside the lens.
    pub fn raw_to_virtual(&self, raw: [f32; 2]) -> Option<[f32; 2]>;
    /// Unit bearing of a raw pixel in this camera's frame; `None` as above.
    pub fn unproject_raw(&self, raw: [f32; 2]) -> Option<Vec3F64>;
    /// Raw pixel of a camera-frame point; `None` behind the camera or past the cap.
    pub fn project_raw(&self, p_cam: Vec3F64) -> Option<[f32; 2]>;
}
```
- Lane T's shard calls `raw_to_virtual` in its "fisheye unproject" step (plan section 4, 40 µs per camera), and `unproject_raw` with `project_raw` for `gyro_guess: RotatedRay` (Lane T step 9).
- Lane T never touches `Fisheye624Camera` directly.
- Without a raw model, `raw_to_virtual` is the identity and `project_raw` is the pinhole projection, so Lane T's rig code has one path.

**Lane F needs from Lane T** (`frontend/mod.rs`, D2 and D5): camera 0's `Frame::image_size` (`:197`) follows the D2 decision, and `prepare_rig` calls `ensure_undistorted` before `match_overlap` (D5). These are the only Lane T lines this card depends on.

## Files Lane F touches

- **Edited:**
  - `apps/kornia-slam-app/src/datasets/lamaria.rs`: raw JSON, `LamariaCamera::raw`, tests for the parameter order and the fx = fy check;
  - `apps/kornia-slam-app/src/source/lamaria.rs`: `LamariaImages`, `validate`, the rig with raw models;
  - `crates/kornia-sensors/src/rig.rs`: field docs and tests only; the skeleton adds the field;
  - `crates/kornia-slam/src/frame.rs`: `ensure_undistorted`, the `undistorted_xy` fallback, `RAW_MAX_INCIDENCE_DEG`, tests;
  - `crates/kornia-slam/src/rig.rs`: the `RigView` method bodies and the `overlap_box` valid region;
  - `crates/kornia-slam/src/mapping/visual_information.rs`: `FisheyeWeight::sigma_scale`;
  - `crates/kornia-slam/src/tracking/pose_estimation/map_projection/mod.rs`: the radius scale in `match_by_projection` only.
- **Pending D4 and D5:** `mapping/bundle_adjustment.rs:1142-1145` and `frontend/overlap.rs:74-93`.
- **Not touched:** `system/mod.rs` (Lane T only, plan section 7), `system/config/*`, `system/validation.rs`, `system/build.rs` (skeleton), `vi_ba_schur.rs` (Lane C; it already reads `observation_sigma_px`), `frontend/mod.rs` (Lane T), `configs/*.ron`, `CHANGELOG.md`.
- **Commits:** two, so FK4 or FK6 can revert the weight without the loader:
  - `feat(lamaria): raw Aria fisheye rig cameras behind images: RawFisheye` (F1, F2, F3, F5);
  - `feat(mapping): magnification weights and search radius for raw fisheye views` (F4).

## Not in this card

- Camera-generic BA (Fisheye624 residuals in raw pixels): deferred by plan section 14.
- The projectaria_tools and COLMAP oracle fixtures: made in validation (plan section 14). This lane ships the round-trip and Jacobian tests only (FP1, FP2).
- Raw fisheye on the mono path (`SensorRig::fisheye` with Fisheye624). F2 rejects it.
- Online calibration of the fisheye intrinsics: card G (`mapping.local_ba.online_calibration`).
