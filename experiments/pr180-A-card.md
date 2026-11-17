# PR #180 card A: estimator (ORB-SLAM3 port, IMU covariance trust, G29)

Lane A ports ORB-SLAM3's estimator choices as separate knobs. It also adds IMU covariance trust (review follow-up on `system/mod.rs:1375`) and the G29 counter with its fix. This card records the theory, numbers and kill rules before any code. The validation workflow scores it; nothing here has been measured on this branch.

- Paths are under `crates/kornia-slam/src/` unless stated. Line numbers are on `pr180/S` (`42ee054`).
- `ORB/` is `/home/edgar/dev/.flux-connect-scratch/bench/src/ORB_SLAM3/src/`. `OV/` is `.../bench/src/open_vins/ov_msckf/src/`.
- `S` is `/home/edgar/dev/.flux-connect-scratch/slam-viba`.
- Every knob defaults to today's behaviour. With all of them off, `pr180/F` must `cmp` equal `pr180/S`.

## Baselines this card quotes

No run exists on `pr180/S` yet. The numbers below come from knob-off binaries whose defaults behave the same: WB1 `exp/vi-window` 75630c9 (card G P1: identical to RG1/rg1) and RG1 for EuRoC. The source is `S/audit/window-results.md`. The swaps card (P8) predicts that `pr180/S` lands inside their noise band. The validator measures the `pr180/S` baselines first. If one falls outside the band below, the predictions here are read as ratios to the new baseline, and the results file says so.

| measure | base (0 / +50 µs / −50 µs) | window 10 (w10, card W) |
|---|---|---|
| R_11 tracked | 67.3 / 78.0 / 73.9 % | 88.1 / 88.2 / 87.8 % |
| R_11 10 s scale in 0.8-1.25, rows to 4800 | 46.0 / 52.3 / 63.2 % | 97.5 / 94.6 / 95.8 % |
| R_11 D1 share (newest IMU edge first-iteration \|r_p\| > 0.3 m) | 12.8 / 5.4 / 9.0 % (200 / 97 / 152) | 4.2 / 4.0 / 5.2 % |
| R_11 D1 followed by a next PnP step of 0.5-2x | 131 / 45 / 92 | not counted |
| R_11 written non-init steps > 0.5 m | 344 / 113 / 241 | 80 / 70 / 108 |
| R_11 `[predgap]` lines over 6 m/s · dt | 374 / 141 / 273 | not counted |
| R_11 interior IMU edges with proxy chi2 > 16.9 | 21.1 / 13.9 / 16.3 % | not counted |
| R_11 splice D(W23..W26) | 70.3 / 97.5 / 107.7 | 8.0 / 14.1 / 64.0 |
| R_11 BA ms per solve; iterations mean | 84.7 / 89.9 / 101.1; 31.0 / 27.6 / 28.1 | 248.1 / 248.8 / 252.6; 31.5 |
| R_11 frame_ms p50 | 8.4 / 8.6 / 8.6 | 9.5 / 9.6 / 9.8 |
| R_11 full Sim3 | FAIL x3 | FAIL x3, give-up near frame 7830 |
| R_11 keyframe interval | 0.20 s (1620 per 6423 tracked frames); ORB-SLAM3 0.39 s | |
| EuRoC ATE (RG1, one sample) | MH_01 0.1337, MH_03 0.0928 (3 samples 0.0803-0.0941), V1_03 0.2225, V2_02 0.1689; MH_04 not measured on this line (0.594 loop-off on 91e279d) | MH_01 0.1250, MH_03 0.1272-0.1476, V1_03 0.2005, V2_02 0.1157 |
| MH_03 \|bg − gt\| in tenth 8 | 0.0004-0.0005 rad/s | 0.0024-0.0028 rad/s |
| Proxy mean (5 sequences, a failed Sim3 counts 0) | 3.6 (handoff) | not run |
| Reference: ORB-SLAM3 mono-inertial, R_11, same harness | 99.8 % tracked, CP 72.9 | |

## Knobs

RON paths are relative to `system.pipeline.OrbSlam(( ... ))`. Every field is `serde(default)`, so existing files load unchanged.

| id | RON path | type | default (today) | new code |
|---|---|---|---|---|
| A1 | `mapping.local_ba.active_keyframes` | `usize` | `3` | no |
| A2 | `mapping.local_ba.large_window` | `LargeWindow` | `Off` | yes |
| A3 | `mapping.local_ba.boundary_scales_bias_rows` | `bool` | `true` | no |
| A4 | `mapping.local_ba.imu_huber` | `ImuHuber` (+ variant `BoundaryOrRecentInit`) | `AllEdges` | yes (variant) |
| A5 | `inertial.sample_interpolation` (new section `InertialConfig`) | `SampleInterpolation` (kornia-sensors) | `Hold` | yes |
| A6 | `tracking.pose_inertial` (new section `TrackingConfig`) | `PoseInertial` | `Off` | yes |
| A7 | `keyframes.rules` | `KeyframeRules` | `Kornia` | yes |
| A8 | `tracking.local_map` | `LocalMapRules` | `Kornia` | yes |
| A9 | `mapping.growth_neighbours` | `GrowthNeighbours` | `RecentTen` | yes |
| A10 | `mapping.local_ba.max_iterations` | `usize` | `50` | no; not ported (see A10) |
| T | `tracking.imu_trust` | `ImuTrustMode` | `TimeSinceInit` | yes |
| G29 | `tracking.snapback_fix` | `SnapbackFix` | `Off` | yes |
| G29c | none: counter `ba_snapbacks`, always on | | | yes (counter only) |

```rust
/// ORB-SLAM3's larger window when tracking is strong (`LocalMapping.cc:148`, `Optimizer.cc:2389-2393`).
pub enum LargeWindow {
    #[default]
    Off,
    /// Used for a keyframe tracked with more than `min_tracked_inliers` inliers.
    /// ORB-SLAM3: 100 (75 monocular), 25 keyframes, 4 iterations, λ₀ 1e-2.
    Enabled { min_tracked_inliers: usize, active_keyframes: usize, max_iterations: usize, initial_lambda: f64 },
}

pub enum ImuHuber {
    #[default]
    AllEdges,
    BoundaryOnly,
    Off,
    /// The boundary edge, plus every IMU edge until VIBA2 has run (`Optimizer.cc:2632`, `i==N-1 || bRecInit`).
    BoundaryOrRecentInit,
}

/// kornia-sensors: how a preintegration turns samples into intervals.
pub enum SampleInterpolation {
    /// Each sample held over the interval that ends at its stamp (`imu.rs:162-174`).
    #[default]
    Hold,
    /// The mean of consecutive samples, the end samples interpolated to the frame stamps
    /// (`Tracking.cc:1684-1720`). The integrator stays Euler.
    Midpoint,
}

pub enum PoseInertial {
    #[default]
    Off,
    /// `PoseInertialOptimizationLastKeyFrame` on every frame: the last keyframe held fixed.
    LastKeyframe,
    /// ORB-SLAM3's choice per frame: `...LastFrame` under the previous frame's 15×15 prior;
    /// `...LastKeyFrame`, predicted from the keyframe, on the first frame after the map changed.
    OrbSlam3,
}

pub enum KeyframeRules {
    #[default]
    Kornia,
    /// `Tracking::NeedNewKeyFrame` (inertial branches). `max_frames` is ORB-SLAM3's `mMaxFrames`, the camera rate.
    /// The other `KeyframePolicy` fields are ignored.
    OrbSlam3 { max_frames: usize },
}

pub enum LocalMapRules {
    #[default]
    Kornia,
    /// `Tracking::UpdateLocalKeyFrames` (inertial), without children and parent (no spanning tree here).
    OrbSlam3,
}

pub enum GrowthNeighbours {
    /// The 10 most recent keyframes for triangulation and fusion (`keyframe_mapping.rs:134`).
    #[default]
    RecentTen,
    /// `CreateNewMapPoints` and `SearchInNeighbors` lists; `covisible` is 10 (30 monocular).
    OrbSlam3 { covisible: usize },
}

pub enum ImuTrustMode {
    /// `imu_confident()`: the IMU has been initialised for `min_imu_confidence_sec`.
    #[default]
    TimeSinceInit,
    /// Trusted while `search_sigmas` predicted pixel sigmas fit inside the widest search
    /// (`lost_search_scale · projection.search_radius`, 60 px by default).
    Covariance { search_sigmas: f64 },
}

pub enum SnapbackFix {
    #[default]
    Off,
    /// On the first frame after a BA write-back, a `LastKeyframe` pose-inertial solve
    /// anchored at the corrected keyframe (Tracking.cc:2982-2991 without the rest of A6).
    AnchorNextFrame,
}
```

`validate()` errors the skeleton adds (names are suggestions):
- `LargeWindowNotLarger`: `large_window.active_keyframes < active_keyframes`, or `max_iterations == 0`, or `initial_lambda` not finite and positive.
- `InertialKnobWithoutImu(&'static str)`: `pose_inertial`, `imu_trust: Covariance` or `snapback_fix` without `sensors.imu`.
- `SnapbackFixWithPoseInertial`: `snapback_fix: AnchorNextFrame` with `pose_inertial` not `Off` (A6 already contains it).
- `NonPositiveSearchSigmas`: `search_sigmas` not finite and positive.
- `ZeroMaxFrames`, `ZeroCovisible`.

## Experiment A0: the ORB-SLAM3 estimator as one bundle
Gap tested: G6, G4, G5, G13, G29 (ledger `S/audit/gap-ledger.md`)   Base: kornia-slam@`pr180/S` (`42ee054`), kornia-rs@`4ada5a7`   Binary tag: `pr180/F` (knobs set in RON)
Mechanism: our online estimator differs from ORB-SLAM3, which scores 72.9 on R_11 in this harness, in A1-A9 below. Each one is cited there.
Theory: the handoff's "port in one step, then bisect". One bundle run tells whether the reference's estimator moves our score at all. Leave-one-out then attributes it. Single-mechanism cards one after another took about an hour each and did not close the gap.
Knob: the bundle sets
```ron
mapping: (local_ba: (active_keyframes: 10, max_iterations: 50,
    large_window: Enabled(min_tracked_inliers: 100, active_keyframes: 25, max_iterations: 50, initial_lambda: 1e-2),
    boundary_scales_bias_rows: false, imu_huber: BoundaryOrRecentInit),
  growth_neighbours: OrbSlam3(covisible: 10)),
inertial: (sample_interpolation: Midpoint),
tracking: (pose_inertial: OrbSlam3, local_map: OrbSlam3),
keyframes: (rules: OrbSlam3(max_frames: 20)),
```
Monocular configs use `min_tracked_inliers: 75` and `covisible: 30` (ORB-SLAM3's mono values). `max_frames` is the camera rate of the sequence (EuRoC 20). `imu_trust` and `snapback_fix` are not in the bundle, because ORB-SLAM3 has neither.
Baselines: `pr180/S` on EuRoC MH_01, MH_03, MH_04, V1_03 and V2_02, and LaMAria R_11 plus the 5-sequence proxy. Each is recorded with sha, binary md5, config and load.
Predictions (numbers, each tied to a measurement):
  P1 EuRoC ATE: ≤ 1.10x `pr180/S` on 5 of 5; MH_03 ≤ 0.1021 m on ≥ 2 of 3 samples; mean change over MH_01, MH_03, V1_03 and V2_02 −15 to −45 %.   measured by: `eval.py` on `kornia_slam_raw.csv`
  P2 MH_03 \|bg − gt\| in tenth 8: ≤ 0.0010 rad/s on ≥ 2 of 3 (W10 0.0024-0.0028).   measured by: `S/audit/window/eurocstate.py`
  P3 EuRoC RPE over 1 s (translation): −20 % or more on ≥ 3 of 5.   measured by: `eval.py --rpe 1,10`
  P4 R_11 tracked ≥ 88 % on 3 of 3.   measured by: run summary
  P5 R_11 10 s scale in band ≥ 94 % on ≥ 2 of 3.   measured by: `r11_series.py` / `scalewin.py`
  P6 R_11 splice D(W23..W26) ≤ 20 on ≥ 2 of 3.   measured by: `S/audit/r11_splice.py --full`
  P7 R_11 full Sim3 FAILs on ≥ 2 of 3: the give-up near frame 7830 is a recovery gap (lane B). At most 1 established give-up per sample. If a Sim3 passes, the CP score is 20-60.   measured by: the LaMAria evaluator
  P8 Proxy mean 3.6 -> 4-20. Lane A alone does not reach the gate of 70.   measured by: the proxy runner, ±50 µs samples
  Mechanism counter M1: D1 share on R_11 ≤ 3.0 % on ≥ 2 of 3.   measured by: `S/audit/window/winlines.py`
  Mechanism counter M2: R_11 `[predgap]` lines over 6 m/s · dt ≤ 0.4x base (≤ 150 / 56 / 109).   measured by: `[predgap]` grep
  Mechanism counter M3: `ba_snapbacks` ≤ 0.4x its `pr180/S` value.   measured by: run summary (G29c)
  Mechanism counter M4: VI-BA iterations mean 15-25 (w10 31.5). This tests card C's K1: a better seed needs fewer iterations.   measured by: WINDOW lines
  Mechanism counter M5: R_11 keyframe interval p50 0.35-0.5 s (base 0.20 s).   measured by: keyframe rows
  Cost C1: R_11 BA ms per solve 400-800 (w10 248, w20 531); `ba_solves` −30 to −50 %; wall 15-35 min per R_11 run; frame_ms p50 +0.3 to +2.0 ms.   measured by: summary, timing from solo runs
Kill rules:
  K1 Mechanism moved, score did not: M1-M3 as predicted, but P4-P6 no better than w10 on ≥ 2 of 3 -> the per-frame pose and the keyframe snap-back are not where R_11 drift enters. Go to G4 scale (marginal prior) and G11 calibration; do not tune this bundle.
  K2 Score moved, mechanism did not: P1, P5 or P6 as predicted while M1, M2 and M3 all miss -> the gain comes from a part off this theory (most likely A1 alone, known from card W). Attribute by leave-one-out before any claim.
  K3 Any EuRoC sequence above 1.10x `pr180/S` on all 3 samples -> do not adopt; leave-one-out, starting with A3 for MH_03.
  K4 Any `[rigid]`, `nonrigid_*`, `vi_ba_failed`, `NonPositivePivot` or `ba_updates_rejected` count above 0 -> stop.
  K5 Wall above 45 min per R_11 run, or BA mean above 1000 ms per solve -> drop A2 from the bundle and rerun one sample before leave-one-out.
  K6 Bundle below w10 alone on P5 by more than 5 points on ≥ 2 of 3 -> a part hurts; leave-one-out.
Bit-identity: every knob at its default: `pr180/F` `cmp` equal to `pr180/S` on EuRoC MH_01 and LaMAria 3_18 (raw.csv, body.tum, coasted file). The G29c counter must not change any output file.
Noise: ±50 µs IMU time offset, 3 samples on R_11 and MH_03 (shifted EuRoC copies as in `S/audit/window/euroc/`). One sample on the other EuRoC sequences is a screen only.

### Leave-one-out (each arm is A0 minus one mechanism; screen first: R_11 base sample plus 5 EuRoC; 3 samples only for arms that move more than the noise band)

| arm | prediction against A0 |
|---|---|
| −A1 (window 3) | P5 ≤ 75 %: the window is the largest scale lever (card W: 46-63 % -> 95-98 %) |
| −A2 | P5 within ±3 points; BA cost −20 to −40 % |
| −A3 | MH_03 ATE ≥ 0.1272 m: card W's K6 regression returns, if card B's theory holds |
| −A4 | P5 within ±5 points; EuRoC ATE within ±5 % |
| −A5 | EuRoC ATE within ±5 %; R_11 inside noise |
| −A6 | M1 and M2 back to ≥ 0.7x base; P6 at least doubles; M4 back to ≥ 28 |
| −A7 | `ba_solves` +40 to +90 %; P5 −0 to −10 points |
| −A8 | tracking ms −0.5 to −2.0; P4 −0 to −5 points |
| −A9 | EuRoC ATE within ±5 % |

Predicted order of contribution: A1 > A6 > A3 > A7 > A4 > A8 > A2 ≈ A9 ≈ A5. If the measured order puts any of A5, A9 or A2 in the top three, record it as a theory miss.

Two extra single-sample arms:
- **A0-i10:** A0 with both iteration caps at 10, ORB-SLAM3's `opt_it`. Prediction: P5 within −10 points of A0 (card C lost 38.9 points without A6). Kill: P5 falls more than 10 points -> A6 does not fix the seed; the cap stays at 50.
- **A0-orb:** A0 with `large_window.max_iterations: 4`, ORB-SLAM3 literal. Prediction: P5 ≤ 75 %, the card C i10 result again. Kill: P5 within 3 points of A0 -> our solve needs fewer iterations than measured; re-run card C.

## A1. Window of 10 keyframes (existing knob)
- **Ours:** `system/config/mod.rs:283,309` (3). The window is the newest non-anchor keyframes (`mapping/bundle_adjustment.rs:223-235`).
- **ORB-SLAM3:** `maxOpt=10` and `Nd = min(KeyFramesInMap-2, maxOpt)`, a temporal `mPrevKF` chain (`ORB/Optimizer.cc:2387-2412`), with the previous keyframe fixed (`:2435-2447`).
- **Why ORB-SLAM3 and not Basalt (3 frames + 7 keyframes + sqrt marginal prior):** it is the reference that scores 72.9 in this harness. Basalt's prior is already the knob `boundary: Marginal`.
- **Predictions:** card W's measurements (table above) stand as the single-knob result. Card W's K6 (MH_03 +35 to +59 % on 3 of 3) fired, so A1 is never adopted without A3.

## A2. Window 25 when tracking is strong
- **Ours:** none. **ORB-SLAM3:** `bLarge = inliers > 75` (mono) or `> 100` (`ORB/LocalMapping.cc:148-149`); `maxOpt=25, opt_it=4` (`ORB/Optimizer.cc:2389-2393`); `setUserLambdaInit(1e-2)` (`:2513-2517`).
- **Theory:** ORB-SLAM3 trades iterations for keyframes: 25 keyframes at 4 iterations cost about as much as 10 at 10. Our solve needs 31.5 iterations at window 10 (card C), so A2 keeps our cap (50). It is then a pure accuracy test of a longer window when tracking is strong.
- **Code:** `KeyframeJob` (`mapping/local_mapping.rs:42-49`) gains `tracked_inliers`. `capture_local_window` (`:51-53`) and the `ViBaParams` fill (`mapping/bundle_adjustment.rs:686-697`) take the large values when it exceeds `min_tracked_inliers`.
- **Predictions (with A1 on, against w10):**
  - large-solve share 70-90 % on R_11 (PnP inliers p10/p50/p90 73/196/415) and ≥ 90 % on EuRoC stereo;
  - BA ms per large solve 450-700;
  - P5 within ±3 points; D1 within ±1 point.
- **Mechanism counter:** `large_window_solves`.
- **Kills:** P5 up more than 5 points while the large share is below 30 % -> luck. Cost above 2.5x w10 with no P5 or P6 gain -> drop.

## A3. Boundary 1e-2 on the 9×9 nav block only; bias-walk rows unscaled and quadratic (existing knob)
- **Ours:**
  - `boundary_scales_bias_rows: true` scales the whole 15×15 (`mapping/vi_ba_schur.rs:809-820`);
  - the Huber weight already touches only the nav rows (`:1526-1537`; `kornia-factors/src/inertial/preintegration.rs:168-182`).
  - So "bias-walk edges unscaled, with no kernel" needs only `false` here; no new code.
- **ORB-SLAM3:** `vei[i]->setInformation(information()*1e-2)` on `EdgeInertial` (9×9) only. `EdgeGyroRW` and `EdgeAccRW` keep `C.block(9,9)` and `C.block(12,12)` inverted, with no kernel (`ORB/Optimizer.cc:2632-2657`).
- **Theory (card W MH_03 check):** at window 10, gyro-bias error in tenth 8 is 5-6x larger. The bias reaches the past only through the boundary edge, whose bias rows are weighted 1e-2. At full weight the bias walk ties the window to the fixed past.
- **Predictions (card B addendum, written 2026-11-15, never run):**
  - PB1 w10b: \|bg − gt\| in tenth 8 ≤ 0.0010 rad/s on ≥ 2 of 3.
  - PB2 b3: tenth 8 stays 0.0003-0.0007 on ≥ 2 of 3.
  - PB3 w10b: median \|ba − gt\| over tenths 6-9 ≤ 0.15 m/s² on ≥ 2 of 3 (W10 0.12-0.22).
  - PB4 W10B_MH_03 ATE ≤ 0.1021 m on ≥ 2 of 3; 10 s drift in tenths 8-9 ≤ 0.09 m.
  - Card B P2: 10 s in band, b3 +5 to +20 points over base; w10b +0 to +10 over w10.
- **Kills:**
  - KB1: PB1 holds and PB4 misses -> the gyro bias is not what drifts MH_03.
  - KB2: PB4 holds and PB1 misses -> another cause.
  - Card B K4: w10b more than 5 points below w10, or D1 share above 1.5x w10 -> do not stack.

## A4. Huber only where ORB-SLAM3 puts it
- **Ours:** `ImuHuber::AllEdges` (`mapping/vi_ba_schur.rs:266-272,823-829`); `BoundaryOnly` exists.
- **ORB-SLAM3:** Huber δ = √16.92 on the boundary edge, and on every inertial edge while `bRecInit = !GetIniertialBA2()` (`ORB/Optimizer.cc:2632-2643`; `ORB/LocalMapping.cc:149`). VIBA2 runs 15 s into the map (`:217-225`).
- **New variant:** `BoundaryOrRecentInit` reads `recently_initialized` from the inertial schedule (`inertial/schedule.rs:15,70-89`). It is passed `KeyframeJob` -> `LocalBaParams` -> `ViBaParams`.
- **Theory:** before VIBA2, velocity and scale are rough and a kernel guards every edge. After it, ORB-SLAM3 trusts every interior edge fully. On R_11 at window 3, 13.9-21.1 % of interior edges have proxy chi2 > 16.9, so `AllEdges` discards metric constraint on one edge in five.
- **Predictions (with A1 on):**
  - interior edges down-weighted after VIBA2: 0 (by construction);
  - interior final chi2 > 16.92 share ≤ 10 %;
  - 10 s in band within ±5 points of w10; EuRoC ATE −10 to +5 %;
  - STATE \|v\| > 4 m/s outside detach episodes stays 0.00 %.
- **Mechanism counter:** interior final nav chi2 share (`S/audit/imuedges.py`).
- **Kills:**
  - `vi_ba_failed` above 0, or more \|ba\| steps above 1 m/s² -> the kernel was hiding real outliers after init; do not adopt.
  - ATE gains while the interior chi2 > 16.92 share stays 14-21 % -> not the kernel.
- **Not ported:** δ² 16.92 against our 16.9 (`ViBaParams` default). The difference is 0.1 %.

## A5. IMU sample averaging with end interpolation
- **Ours:** `PreintegratedImu::from_measurements` (`kornia-sensors/src/imu.rs:142-176`) holds each sample over the interval that ends at its stamp (`:162-168`) and the last one to `t1` (`:170-174`). That is a right Riemann sum.
  - Callers: `inertial/mod.rs:104-117` (frame prediction and keyframe edges) and `mapping/map/imu_factor.rs:29-40` (repropagation).
- **ORB-SLAM3:** `PreintegrateIMU` keeps the sample before the previous frame and the one after the current frame (`mImuPer` 1 ms, `ORB/Tracking.cc:607,1648-1663`). It integrates the mean of consecutive samples, with the end samples linearly interpolated to the frame stamps (`:1684-1720`). `IntegrateNewMeasurement` is Euler (`ORB/ImuTypes.cc:177-235`), as ours (`imu.rs:211`).
- **Why ORB-SLAM3:** OpenVINS does the same, averaging `data_minus` and `data_plus` and cutting at the ends (`OV/state/Propagator.cpp:269-349,408-425`). Two references agree; Euler stays.
- **Theory:** a right Riemann sum minus the integral ≈ (dt/2)(ω(t1) − ω(t0)). That equals an IMU time shift of half a sample period: 2.5 ms on EuRoC (200 Hz) and 0.5 ms on LaMAria (`imu-right`, 1 kHz).
  - On EuRoC, \|Δω\| of 0.1-0.5 rad/s between keyframes gives 0.25-1.25 mrad of systematic rotation error per edge.
  - That is 3-15x the gyro noise over 0.25 s (1.7e-4 · √0.25 = 8.5e-5 rad), so it enters the nav chi2.
  - It follows angular acceleration, so a constant bias cannot absorb it.
- **Code:**
  - `SampleInterpolation` and `PreintegratedImu::from_measurements_with(..)` in kornia-sensors; `from_measurements` stays `Hold`, bit for bit.
  - `preintegrate_window` keeps one bracketing sample on each side when `Midpoint`. If the sample after `t1` is not buffered yet, it holds the last sample, counted in `imu_tail_holds`.
  - `ImuFactor` keeps the mode so `repreintegrate` matches. Lane D's re-preintegration calls the same function with the run's mode.
- **Predictions:**
  - U1 unit test: ω = (0, 0, sin 2πt) rad/s, 200 Hz IMU, frames every 50 ms offset 1.3 ms from the IMU grid. Max ΔR angle error over the intervals: `Hold` 7.0e-4 to 8.5e-4 rad (theory 7.8e-4); `Midpoint` ≤ 1e-5 rad (theory 4.1e-6).
  - EuRoC MH_01 interior IMU edges: median final nav chi2 −20 to −60 %; rotation residual p50 −30 to −70 %.
  - LaMAria R_11: the same measures change by ≤ 15 %.
  - ATE: EuRoC −8 to +3 %; R_11 inside noise.
  - `imu_tail_holds` ≤ 1 per 1000 frames on dataset replay.
- **Mechanism counter:** interior rotation residual p50 (`S/audit/imuedges.py`).
- **Kills:**
  - U1 misses -> the interpolation is wrong; fix before any run.
  - The chi2 falls as predicted but EuRoC ATE does not move -> velocity and bias absorb the shift. Keep it only if the bundle needs it.
  - ATE moves more than 10 % while the chi2 does not -> another cause, likely the tail handling.

## A6. Pose-inertial solve per frame, two variants, 15×15 prior (idea of PR #195, credit Astik)
- **Ours:**
  - the per-frame pose is visual-only PnP (`tracking/pose_estimation/map_projection/mod.rs:563-600`, rig `:767`); the IMU only seeds and vetoes it;
  - the velocity is copied from the reference keyframe every frame (`system/mod.rs:1334-1355`, called at `:1368`; G13);
  - the prediction runs from the last frame (`:1764-1791`).
- **ORB-SLAM3:**
  - `TrackLocalMap` uses `PoseInertialOptimizationLastFrame` when the map did not change, and `...LastKeyFrame` when it did (`ORB/Tracking.cc:2970-2992`). It stays visual for `mMaxFrames` frames after a relocalisation (`:2973-2977`).
  - `LastKeyFrame` (`ORB/Optimizer.cc:4491-4872`): the keyframe is fixed (`:4655-4670`); one inertial and two bias-walk edges (`:4672-4695`); mono chi2 {12, 7.5, 5.991, 5.991}, stereo {15.6, 9.8, 7.815, 7.815}; 4 rounds of 10 iterations (`:4698-4712`); kernels off from round 3 (`:4750`); below 30 inliers, rescue at chi2 18 / 24 (`:4796-4822`); its prior is the current frame's Hessian (`:4831-4870`).
  - `LastFrame` (`:4875-5289`): the previous frame is free under `EdgePriorPoseImu`, Huber δ 5 (`:5085-5093`); mono chi2 5.991 x4 (`:5098`); the 30×30 Hessian is marginalised to 15×15 (`:5232-5285`).
  - Prediction after a map change starts from the last keyframe's state (`ORB/Tracking.cc:1746-1761`), otherwise from the last frame (`:1764-1779`).
- **Code (#195 for the idea, not the code):**
  - new `tracking/pose_estimation/pose_inertial.rs`. The residual and Jacobians come from `kornia_factors::inertial::preintegration::{residual_and_jacobians, information}`; the reprojection Jacobian from Lane C's `pose_jacobian_2x6`; kernels from kornia-3d `HuberKernel`.
  - A rig observation passes `R_c'w = R_c'p R_pw` and `p_c'`. A right perturbation of `T_pw` is the same right perturbation of `T_c'w`, so no extra term is needed.
  - Stereo depth rows are not used; ORB-SLAM3 uses a right-image u residual instead. This is a known difference.
  - The anchor is `state.last_keyframe_idx`, as ORB-SLAM3's `mpLastKeyFrame`.
  - With A6 on, the frame's velocity and bias come from the solve, and the keyframe copy runs only after a map change.
  - A give-up or re-seed drops the frame prior.
- **Theory:** the IMU edge from the anchor ties each frame to the inertial state. A visual pose that jumps against it costs (\|Δp\| / σ_p)², with σ_p a few mm over 0.05-0.2 s, so a 0.3 m snap-back is rejected and its matches reclassify as outliers. The per-frame velocity removes the keyframe seed error (G13: 0.13-0.19 m/s) that card C showed the window spends iterations on.
- **Reference A/B (PR #195, mono, its own base, window 10 plus this solve):** MH_03 0.3312 -> 0.1256 m and MH_04 2.8765 -> 0.6570 m, mostly scale; MH_01 +8 %, MH_05 +11 %. Our configs are metric from stereo or the rig, so the gain should be smaller.
- **Predictions (`OrbSlam3` variant, A1 on, against w10):**
  - EuRoC RPE 1 s −20 % or more on ≥ 3 of 5; ATE −30 to +5 %;
  - R_11 `[predgap]` lines over 6 m/s · dt ≤ 0.4x base; D1 share ≤ 3.0 %; steps > 0.5 m ≤ 0.5x w10;
  - newest keyframe first-iteration \|r_v\| median −40 % or more; VI-BA iterations mean 15-25;
  - tracked −3 to +8 points; frame_ms p50 +0.2 to +1.5 ms;
  - prior dropped (marginal not positive definite) on ≤ 1 % of frames;
  - `LastKeyframe` against `OrbSlam3`: RPE 1 s 0 to 15 % worse.
- **Mechanism counters:** `pose_inertial_solves`, `pose_inertial_keyframe_anchored` (expected about the keyframe rate), `pose_inertial_prior_dropped`. Debug line `[pose_inertial] frame= anchor=kf|frame inliers= imu_chi2= prior_pd=`.
- **Kills:**
  - `[predgap]` and D1 fall as predicted, but neither 10 s in band nor ATE improves -> drift does not enter through the per-frame pose; it is in BA (G4/G6).
  - ATE improves while `[predgap]` counts stay ≥ 0.7x base -> the gain is the velocity seed, not the pose. Run `LastKeyframe` and A6 with the keyframe copy kept to separate them.
  - Healthy-frame `imu_chi2` p99 above 100 -> the IMU information or `T_bc` is wrong; stop.
  - LowPnpInliers rejects up by more than 50 % -> the IMU edge pulls good matches out; check the chi2 schedule before scoring.

## A7. Keyframe insertion rules
- **Ours:** `KeyframePolicy::should_insert_with` (`tracking/policy.rs:130-158`; defaults `:39-48`: every 3-8 frames, ratio 0.6); `n_ref_map_points` counts all associated points (`system/mod.rs:1805-1810`). R_11 inserts one keyframe per 0.20 s, MH_01 692 keyframes.
- **ORB-SLAM3 `NeedNewKeyFrame` (`ORB/Tracking.cc:3064-3214`), inertial branches:**
  - before IMU init, a keyframe every 0.25 s and nothing else (`:3066-3074`);
  - none within `mMaxFrames` of a relocalisation (`:3091`);
  - `nRefMatches` counts points seen ≥ 3 times (2 while ≤ 2 keyframes) (`:3097-3100`);
  - stereo close points: tracked < 100 and untracked > 70 (`:3106-3127`);
  - `thRefRatio` 0.75; mono-inertial 0.75 above 350 inliers, else 0.90 (`:3130-3153`);
  - c1a every `mMaxFrames`; c1b always true when mapping is synchronous (`mMinFrames` 0); c2 ratio or close, and inliers > 15; c3 0.5 s since the last keyframe; c4 (mono-inertial only) 15 < inliers < 75 or `RECENTLY_LOST` (`:3156-3185`). Insert if `((c1a||c1b||c1c)&&c2)||c3||c4` (`:3187`).
  - Our `RECENTLY_LOST` is the first tracked frame after a loss.
- **Theory:** fewer, better-spaced keyframes make a 10-keyframe window span 3.5-5 s instead of 2 s, which observes scale and bias better per solve. It also halves BA calls.
- **Code:** `KeyframeEvidence` (`tracking/policy.rs:56-70`, `#[non_exhaustive]`) gains `ref_points_min_obs`, `tracked_close`, `untracked_close`, `recently_lost`, `imu_initialized`, `mono_inertial`, `secs_since_last_kf` and `frames_since_reloc`.
- **Predictions (alone):**
  - MH_01 keyframes 692 -> 370-550; R_11 interval p50 0.20 -> 0.35-0.5 s; `ba_solves` −30 to −50 %;
  - EuRoC ATE within ±10 %; R_11 10 s in band (window 3) +0 to +15 points;
  - c4 fires on 5-15 % of keyframes on `lamaria-mono-imu.ron`.
- **Kills:** keyframe interval inside the prediction but in-band unchanged -> time span is not the lever. Score up while the interval stays below 0.3 s -> another part of the rule; check c4.

## A8. Tracking local map
- **Ours:** `select_local_map_points` (`tracking/local_map.rs:44-118`): the current keyframe plus the 10 most-voted keyframes (`:30-38`). It is called with the default config at four sites: `map_projection/mod.rs:280` (initial projection search), `:577` (local refine), `system/mod.rs:1551` (visibility count) and `system/trace.rs:60` (debug trace). The reference keyframe is the last inserted one (`system/mod.rs:2164`).
- **One rule for all four sites:** every site takes the same `LocalMapSelectionConfig` (see "Skeleton contract"). If the visibility count at `:1551` kept the old rule while the search used the new one, the found/visible ratios, and so culling, would shift silently.
- **ORB-SLAM3 `UpdateLocalKeyFrames` (`ORB/Tracking.cc:3457-3606`):**
  - once the IMU is initialised, votes come from the last frame's points (`:3461-3503`); every voted keyframe is admitted (`:3514-3528`);
  - then, up to 80 keyframes, each one's first new best-10 covisible neighbour (`:3532-3555`);
  - then the 20 most recent keyframes (`:3585-3600`);
  - the reference keyframe is the most voted (`:3602-3606`).
  - Children and the parent (`:3557-3583`) are left out: this map has no spanning tree. A known difference.
- **Theory:**
  - The temporal 20 put the BA window's landmarks into every frame's search, so the next PnP sees the keyframes BA just moved (G29).
  - The most-voted reference cannot be a coast keyframe with 7 points. That removes step 2 of the loss cascade (`S/lamaria-estimator-plan.md` §1(b): 80-83 % of LowReferenceCorrespondences frames had a lost reference).
- **Predictions (alone):**
  - local keyframes per frame p50 ≤ 11 -> 25-60; landmarks searched x1.5-3; tracking ms +0.5 to +2.0;
  - lost-reference share of LowReferenceCorrespondences frames ≤ 30 %; their runs median ≤ 5 frames (15-100 on the s2marg line);
  - D1 follow rate (snap-back share) −10 points or more.
- **Kills:** tracking ms p50 up 3 ms or more -> cost; stop. Lost-reference share falls but reject runs stay ≥ 15 frames -> persistence is visual (lost-mode projection, lane B).

## A9. Growth and fusion neighbours
- **Ours:** the 10 most recent keyframes serve triangulation and fusion alike (`mapping/keyframe_mapping.rs:134,142-153`, used at `:176` and `:206`).
- **ORB-SLAM3:**
  - `CreateNewMapPoints`: best-10 covisible (30 mono), then the `mPrevKF` chain while ≤ 10 (`ORB/LocalMapping.cc:388-407`);
  - `SearchInNeighbors`: best-10 (30), each one's best 20, then temporal keyframes up to 20 targets (`:714-763`).
- **Code:** `KeyframeGrowthPlan` carries `triangulation_neighbours` and `fusion_neighbours` (skeleton). `RecentTen` puts today's list in both, so the plan is bit-identical. `prepare_keyframe_growth` takes the `GrowthNeighbours` rule; lane B's seeded-window wrapper takes the same rule and filters both lists.
- **Theory:** recency never fuses with an old pass, so revisits make twin landmarks (estimator plan S3b).
- **Predictions (alone):** `observations_added` per keyframe x1.3-2; `landmarks_added` per keyframe −10 to −40 %; EuRoC ATE −10 to +5 %, with the largest gain on MH_01 (revisit).
- **Kill:** fusion links up as predicted but ATE flat on MH_01 -> duplicates do not cost accuracy here.

## A10. Iteration cap: not ported
ORB-SLAM3 runs 10 iterations, or 4 when large (`ORB/Optimizer.cc:2388,2392,2843`). Card C measured that 10 iterations at window 10 lose 38.9 points of 10 s scale (100 % of solves stop on the cap), so our 50 stays. A0-i10 re-tests the cap once A6 gives a better seed.

## Known differences not in this card (listed so a bisect does not miss them)
- `initial_lambda` 1.0 (`mapping/vi_ba_schur.rs:336`) against 1e-2 when large only (`ORB/Optimizer.cc:2516`). A2 carries the large value.
- `accel_bias_prior_weight` 1e2 (`vi_ba_schur.rs:229,356`); ORB-SLAM3's `LocalInertialBA` has no bias prior (`EdgePriorAcc` appears only at `ORB/Optimizer.cc:579,3105,3279`).
- Visual information: invSigma2 per octave with Huber √5.991 (`ORB/Optimizer.cc:2690-2693,2752-2753`) against our 1 px and 5 px. The existing knob is `visual_information: OctaveScaled`.
- Tracking outliers are nulled before keyframe insertion (`ORB/Tracking.cc:3011-3023`). The existing knob is `mapping.keyframe_observations: PnpInliers`.
- At most 200 fixed covisible keyframes (`ORB/Optimizer.cc:2481`); ours has no cap.

## Experiment T: IMU covariance trust
Gap tested: G2 (coast divergence), G21 (the pose gate reads the trusted prediction), and a new row for Phase 6: "the IMU prediction is trusted by time, not by covariance"   Base: `pr180/S`   Binary tag: `pr180/F`
Mechanism: `imu_confident()` (`system/mod.rs:1851-1858`) checks only that the IMU was initialised `min_imu_confidence_sec` (2.0, `tracking/policy.rs:199,224`) ago. It gates three things:
- the tracking prediction, the pose-gate velocity and the KLT guess (`:1377-1384`);
- relocalisation, which runs only when not confident (`:1580-1582`, lane B's trigger);
- the 5 s grace before give-up (`:1604-1605`, `policy.rs:238-244`).
Also, the search scale jumps to `lost_search_scale` 4 as soon as a trusted coast starts (`map_projection/mod.rs:180-189`). A coast is trusted the same at 4.9 s as at 0.1 s.
Theory:
- **Propagation:** let Σ_k be the newest keyframe's 15×15 marginal from the VI-BA reduced system. Σ_k is propagated through the preintegration from that keyframe: Σ_t = J_j⁻¹ (J_i Σ_k J_iᵀ + Ω⁻¹) J_j⁻ᵀ, with J_i, J_j from `preintegration::residual_and_jacobians` (`kornia-factors/src/inertial/preintegration.rs:72`) and Ω from `information` (`:187-206`).
- **What it covers:** this includes the starting velocity and bias uncertainty that the preintegration covariance alone leaves out (the review's point). That uncertainty grows fastest in a coast: σ_p ≈ σ_v t + ½ σ_ba t² + ½ \|g\| σ_tilt t².
- **Pixel sigma:** σ_px = f · √(σ_θ² + (σ_p / z)²), from the largest eigenvalues of the rotation and translation blocks and z the median depth of the reference keyframe's landmarks.
- **Decisions:** trusted while `search_sigmas` · σ_px ≤ `lost_search_scale` · `projection.search_radius` (4 · 15 = 60 px). The search scale is clamp(`search_sigmas` · σ_px / 15, 1, 4). Relocalise and give up when trust ends.
- **Reference:** OpenVINS propagates the full filter covariance (`OV/state/Propagator.cpp:33-130`) and gates on the marginal chi2 (`OV/state/StateHelper.cpp:459-470`). ORB-SLAM3 uses a fixed 5 s `time_recently_lost` (`ORB/Tracking.cc:48,1993`). OpenVINS's choice wins because the timer treats a coast from a well-estimated state and from a bad one alike. ORB-SLAM3's `LastKeyFrame` prior also treats the keyframe as exact (`ORB/Optimizer.cc:4831-4837`); Σ_k does not.
Knob: `tracking.imu_trust: Covariance(search_sigmas: 3.0)`; default `TimeSinceInit`.
Code:
- `ViBaParams.newest_marginal: bool`, set only when T is on. `ViBaResult.newest_marginal: Option<Mat15F64>`: one extra Cholesky of the undamped reduced system at the final state, plus 15 solves (`mapping/vi_ba_schur.rs:1641-1680`). Off, the solve is unchanged.
- The keyframe stores it; `preintegration::propagate_state_covariance` propagates it per frame.
- `SlamSystem::imu_trust(timestamp) -> ImuTrust`. `imu_confident()` returns `imu_trust(..).trusted`, so all three callers switch together. With `TimeSinceInit`, `ImuTrust` is the old rule bit for bit.
- Lane B's relocalise hook (`relocalization_due`, card B) reads `!imu_confident()`, which is `!trusted` under T. Lane A does not edit that hook.
- Debug line: `[trust] frame={idx} t={:.6} trusted={0|1} sigma_px={:.2} scale={:.2} cov6=[36 values of the pose block, row-major]`.
NEES (for the validator): ε = δᵀ Σ_pose⁻¹ δ with δ = `SE3F64::log`(T_cw,gt⁻¹ · T_cw,pred) in (υ, ω) order. Σ_pose is the 6×6 pose block of Σ_t from the same frame's `[trust]` line. T_cw,gt is the ground-truth body pose through `T_bc`, aligned by one SE3 Umeyama fit (no scale) over the run's tracked frames. For a consistent estimate, E[ε] = 6 (95 % one-sample bound 12.59); the 3-dof rotation and position parts have mean 3. Reader to write: `S/audit/trust_nees.py <run dir> <gt>`.
Baselines: `pr180/S` MH_01, V1_03, R_11; and `--blackout 60:2` and `60:5` on MH_01 and V1_03 at 0 / ±50 µs (12 runs).
Predictions:
  T1 unit test: on a seeded synthetic constant-velocity trajectory with EuRoC noise and Σ_k = diag(1e-6 ×3, 1e-6 ×3, 1e-4 ×3, 1e-8 ×3, 1e-4 ×3), σ_p at 5 s matches a 2000-sample Monte Carlo within 10 %.   measured by: unit test
  T2 MH_01 healthy tracking: the decision differs from `TimeSinceInit` on ≤ 0.5 % of frames; search scale 1.0 on ≥ 99 % of tracked frames; ATE within the ±50 µs band of `pr180/S`.   measured by: `[trust]` lines, `eval.py`
  T3 one-frame rotation NEES (3 dof, `[predgap]` rot against the trust line's rotation block) on healthy MH_01 frames: p50 1.2-4.7 (χ²₃ median 2.37), p95 ≤ 16.   measured by: `trust_nees.py`
  T4 blackouts: position NEES at the gap's end, mean of 12 samples in [1, 9]; trust ends 2.5-4.5 s into every 5 s gap and never inside a 2 s gap. Arithmetic for MH_01: σ_v 0.02 m/s and σ_ba 0.02 m/s² give σ_p 0.35 m at 5 s, about 50 px at z = 3 m.   measured by: `trust_nees.py`, `[trust]`
  T5 R_11: coasted frames still trusted whose position error against pGT exceeds 3σ ≤ 5 %. With lane B on, the first relocalisation attempt comes ≤ 0.5 s after trust ends.   measured by: `trust_nees.py` on R_11 pGT
  Mechanism counter: `imu_trust_lost` (trusted-to-untrusted transitions) and the trust duration per coast.
Kill rules:
  KT1 T4 mean NEES above 9 -> Σ_k is overconfident: VI-BA treats the fixed context as exact. Do not adopt. Inflating it needs its own card.
  KT2 T4 mean below 1 -> underconfident: searches too wide, give-ups too early. Do not adopt.
  KT3 Mechanism moved, score did not: trust and search scale change as predicted (T4), but coast error, give-ups and CP do not move -> harmless and of no use for the score. Keep it off.
  KT4 Score moved, mechanism did not: ATE or give-ups change while T2 says the decisions match the time rule -> float order or another cause. Check `cmp` with T on but its outputs unused.
  KT5 EuRoC give-ups rise by 1 or more on any sequence -> trust is too strict.
Bit-identity: `TimeSinceInit` keeps `cmp` equality; `newest_marginal` is not computed.
Noise: ±50 µs on every blackout run; the NEES uses 12 samples.

## Experiment G29: BA shift against the next PnP step (counter first, then the fix)
Gap tested: G29   Base: `pr180/S`   Binary tag: `pr180/F`
Mechanism:
- `apply_local_mapping_results` (`system/mod.rs:774-806`) composes the reference keyframe's BA correction onto the tracked pose and copies the keyframe's velocity and bias.
- The next frame's PnP matches mostly landmarks outside the BA window. Card G found that the window's landmarks move with the keyframe (median 0.9-1.9x), so the snap-back comes from outside the window.
- Vision-only PnP returns to the old place.
Counter G29c (always on, no knob):
- At a BA write-back, s = the newest keyframe's camera-centre shift (`upstream::kornia_3d::pose_ext::camera_centre`, after minus before).
- On the next tracked frame, q = accepted centre − predicted centre, and ρ = −(q · s) / \|s\|².
- Counters: `ba_keyframe_shifts` (\|s\| > 0.3 m) and `ba_snapbacks` (ρ in [0.5, 2]).
- Debug line: `[g29] kf={} shift={:.3} next_step={:.3} ratio={:.2}`.
Fix: `tracking.snapback_fix: AnchorNextFrame`. On the first tracked frame after a write-back, run the A6 `LastKeyframe` solve anchored at the corrected keyframe. Otherwise tracking stays visual. This is ORB-SLAM3's `mbMapUpdated` branch (`ORB/Tracking.cc:1746-1761,2982-2991`; `ORB/Optimizer.cc:4655-4695`) without the rest of A6, so G29 is tested apart from G5 and G13.
Theory: with an IMU edge of full information from the moved keyframe, a 0.3-4 m snap-back costs chi2 of order 1e4-1e6. Tracking then follows BA, and the outside-window matches that disagree become outliers. Which side is right decides whether the fix helps.
Predictions:
  G0 offline, zero runs, done first: on the existing R_11 logs (pg1off*), for D1 keyframes, the BA keyframe pose is closer to pGT than the next frame's PnP pose on ≥ 60 %. If not, BA is the wrong side; do not run the fix (KG1).   measured by: a reader over `[vi_ba]` STATE / `[predgap]` lines and pGT
  G1 counter at defaults: `ba_keyframe_shifts` matches the D1 counts within ±25 % (R_11 200 / 97 / 152 at window 3; 81 / 78 / 101 at window 10). `ba_snapbacks` matches the follow counts within ±25 % (131 / 45 / 92 at window 3). EuRoC MH_01 has ≤ 5 shifts.   measured by: run summary
  G2 fix on, window 3: `ba_snapbacks` −60 % or more on ≥ 2 of 3; longest detach episode ≤ 10 keyframes (77 / 25 / 64); written steps > 0.5 m ≤ 0.5x base.   measured by: summary, `winlines.py`, `stepclass.py`
  G3 fix on: R_11 10 s in band +5 to +25 points at window 3; splice D(W23..W26) ≤ 0.6x base; EuRoC ATE within ±5 % (few shifts there).   measured by: `r11_series.py`, `r11_splice.py`, `eval.py`
Kill rules:
  KG1 G0 misses -> the outside landmarks hold the truth and the keyframe was flung by BA. The fix would lock tracking onto the error. Move to the BA side (A3, G4).
  KG2 Mechanism moved, score did not: snap-backs −60 % but 10 s in band and splice inside noise -> G29 is not a score gap. Mark it refuted as a gap.
  KG3 Score moved, mechanism did not: G3 holds while snap-backs fall less than 30 % -> luck, or the anchored solve helps through the velocity. Attribute with A6 `LastKeyframe`.
  KG4 LowPnpInliers rejects up by more than 50 % -> the anchored solve throws out good matches; stop.
Bit-identity: the counter does not change raw.csv, body.tum or the coasted file (`cmp` against `pr180/S`). `snapback_fix: Off` is unchanged.
Noise: ±50 µs, 3 samples on R_11.

## Files lane A touches
- **kornia-sensors:** `crates/kornia-sensors/src/imu.rs` (`SampleInterpolation`, `from_measurements_with`; `from_measurements` unchanged).
- **kornia-factors:** `crates/kornia-factors/src/inertial/preintegration.rs` (`propagate_state_covariance`).
- **Config (bodies; the skeleton declares the fields):** `system/config/mod.rs`, `system/validation.rs`, `system/build.rs`, `system/config/tests.rs`.
- **Mapping:**
  - `mapping/local_mapping.rs`: `KeyframeJob.tracked_inliers` and `recently_initialized`;
  - `mapping/bundle_adjustment.rs`: large window, marginal request;
  - `mapping/vi_ba_schur.rs`: `BoundaryOrRecentInit`, `newest_marginal`;
  - `mapping/keyframe_mapping.rs`: `GrowthNeighbours`; lane B's seeded-window function is not touched;
  - `mapping/map/imu_factor.rs`: the interpolation mode.
- **Inertial:** `inertial/mod.rs` (bracketing samples), `inertial/schedule.rs` (`recently_initialized`), new `inertial/trust.rs` (`ImuTrust` body).
- **Tracking:**
  - new `tracking/pose_estimation/pose_inertial.rs`; `tracking/pose_estimation/mod.rs`;
  - `tracking/pose_estimation/map_projection/mod.rs`: the pose-inertial hook at the final refine. The local-map config at `:280` and `:577` comes from the skeleton's `local_map_config()`; lane A does not edit those call sites;
  - `tracking/tracker.rs`: the search-scale override body, nav anchor in and out;
  - `tracking/local_map.rs`: the `LocalMapRules::OrbSlam3` body and `most_voted_keyframe` inside `select_local_map_points` (not `filter_to_window`, lane B); `tracking/policy.rs` (`KeyframeRules`, evidence fields).
- **System:**
  - `system/mod.rs`: `imu_trust`, the `imu_confident` body, frame nav state, map-changed flag, keyframe evidence, G29 counter and fix, the A8 reference keyframe at `:2164`. Not the relocalise trigger (lane B).
  - `system/state.rs`: frame nav state and prior.
  - `system/counters.rs`: increments only; the skeleton declares the fields.
- **Not touched:** `tracking/pose_estimation/pose_lm.rs` (lane C deletes it), `configs/*.ron`, `CHANGELOG.md` (skeleton and Phase 6), `apps/kornia-slam-app/src/summary.rs` (the skeleton owns it, see "Skeleton contract").

## Shared types and signatures lane A needs from the skeleton (Phase 2c)

From lane C (card C, `upstream/kornia_3d/refine_analytic.rs`), called only:
```rust
pub fn pinhole_point_jacobian(fx: f64, fy: f64, point_cam: Vec3F64) -> [Vec3F64; 2];
/// Right perturbation `T_cw · exp(δ)`, `δ = (υ, ω)`.
pub fn pose_jacobian_2x6(rotation_cw: &Mat3F64, point_world: Vec3F64, point_cam: Vec3F64, fx: f64, fy: f64) -> SMatF64<2, 6>;
```

Declared by the skeleton, filled by lane A:
```rust
// kornia-sensors/src/imu.rs
pub enum SampleInterpolation { #[default] Hold, Midpoint }
impl PreintegratedImu {
    /// `Midpoint` needs the samples just before `t0` and just after `t1` in `samples`;
    /// without the latter it holds the last sample to `t1`.
    pub fn from_measurements_with(bias: ImuBias, calib: ImuCalib, samples: &[ImuMeasurement],
        t0: f64, t1: f64, interpolation: SampleInterpolation) -> Self;
}

// kornia-factors/src/inertial/preintegration.rs
/// Covariance of state `j` given state `i`'s, through the edge's linearisation:
/// `J_j⁻¹ (J_i Σ_i J_iᵀ + Ω⁻¹) J_j⁻ᵀ`, layout `[ρ ω v b_g b_a]`. `None` if `J_j` is singular.
pub fn propagate_state_covariance(state_i: NavState<'_>, covariance_i: &Mat15F64, state_j: NavState<'_>,
    preintegrated: &PreintegratedImu, gravity: &Vec3F64, t_bc: Option<&Pose3d>) -> Option<Mat15F64>;

// kornia-slam/src/inertial/trust.rs (lane B reads `trusted`)
/// How far the current frame's inertial prediction can be trusted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuTrust {
    /// The prediction may seed tracking, the pose gate and the KLT guess; relocalise when false.
    pub trusted: bool,
    /// Projection search scale; `None` keeps `search_scale_for`.
    pub search_scale: Option<f32>,
    /// One-sigma reprojection spread at the reference depth, px; `None` under `TimeSinceInit`.
    pub pixel_sigma: Option<f64>,
    /// Predicted state covariance, layout `[ρ ω v b_g b_a]`; `None` under `TimeSinceInit`.
    pub covariance: Option<Mat15F64>,
}
// system/mod.rs
fn imu_trust(&self, timestamp_sec: f64) -> ImuTrust;   // imu_confident() == imu_trust(..).trusted

// inertial/schedule.rs
pub(crate) fn recently_initialized(&self) -> bool;     // a window is open and VIBA2 has not run

// mapping/vi_ba_schur.rs
pub struct ViBaParams { /* ... */ pub newest_marginal: bool, pub imu_recently_initialized: bool }
pub struct ViBaResult { /* ... */ pub newest_marginal: Option<Mat15F64> }

// mapping/bundle_adjustment.rs, mapping/local_mapping.rs
pub struct LocalBaParams { /* ... */ pub large_window: LargeWindow, pub recently_initialized: bool, pub newest_marginal: bool }
pub struct KeyframeJob   { /* ... */ pub tracked_inliers: usize }

// tracking/pose_estimation/pose_inertial.rs (todo!() bodies, reachable only with A6 or G29 on)
pub(crate) struct FrameNavState { pub pose_cw: Pose3d, pub velocity: Vec3F64, pub bias: ImuBias }
pub(crate) struct NavPrior { pub mean: FrameNavState, pub information: Mat15F64 }
pub(crate) enum Anchor<'a> { Fixed(&'a FrameNavState), Prior(&'a NavPrior) }
pub(crate) struct PoseInertialObservation { pub point_w: Vec3F64, pub pixel: Vec2F64, pub information: f64,
    pub view: u8 /* 0 primary, k = rig view k-1 */ }
pub(crate) struct PoseInertialSolution { pub state: FrameNavState, pub prior: Option<NavPrior>,
    pub inliers: Vec<bool>, pub imu_chi2: f64 }
pub(crate) fn solve_pose_inertial(anchor: Anchor<'_>, preintegrated: &PreintegratedImu, gravity_world: Vec3F64,
    camera_to_body: Option<Pose3d>, cameras: &[RigObservationCamera], primary: &PinholeCamera,
    initial: FrameNavState, observations: &[PoseInertialObservation], monocular: bool) -> PoseInertialSolution;

// tracking/policy.rs: KeyframeEvidence (#[non_exhaustive]) gains
pub ref_points_min_obs: usize, pub tracked_close: usize, pub untracked_close: usize, pub recently_lost: bool,
pub imu_initialized: bool, pub mono_inertial: bool, pub secs_since_last_kf: Option<f64>, pub frames_since_reloc: Option<usize>,
```

Counters (`system/counters.rs` fields declared by the skeleton; printed on the shared new summary line, see "Skeleton contract"): `ba_keyframe_shifts`, `ba_snapbacks`, `large_window_solves`, `pose_inertial_solves`, `pose_inertial_keyframe_anchored`, `pose_inertial_prior_dropped`, `imu_trust_lost`, `imu_tail_holds`.

Config types `LargeWindow`, `PoseInertial`, `KeyframeRules`, `LocalMapRules`, `GrowthNeighbours`, `ImuTrustMode` and `SnapbackFix` are given in "Knobs". So are the new sections `OrbSlamPipeline.tracking: TrackingConfig { pose_inertial, imu_trust, local_map, snapback_fix }` and `OrbSlamPipeline.inertial: InertialConfig { sample_interpolation }`, each `serde(default, deny_unknown_fields)`.

Lane D: re-preintegration from raw IMU calls `from_measurements_with` with the run's `sample_interpolation`, so offline and online edges agree.

## Skeleton contract (card review round 3b')

The skeleton commit declares every item below, with today's behaviour as the body. Lanes fill bodies only and change none of these signatures. Cards B, C and D state the same lines for their parts; this section is the full list.

**Local map (`tracking/local_map.rs`): one selection for every caller.**
```rust
pub struct LocalMapSelectionConfig {
    /* the four existing fields, same defaults */
    pub rules: LocalMapRules,            // A8, default `Kornia`
    pub window_start_kf: Option<usize>,  // B4, default `None`
}
/// One local-map selection.
pub struct LocalMapSelection {
    pub points: Vec<usize>,
    /// Points the B4 window removed; 0 when `window_start_kf` is `None`.
    pub dropped_by_window: usize,
    /// The most-voted keyframe (A8's reference keyframe); `None` without votes or under `Kornia`.
    pub most_voted_keyframe: Option<usize>,
}
pub fn select_local_map_points(map: &Map, tracked_matches: &[(usize, usize)],
    current_keyframe: Option<&Keyframe>, config: LocalMapSelectionConfig) -> LocalMapSelection;
/// Lane B's body: keeps points observed by a keyframe at or after `window_start_kf`; returns the count removed.
pub(crate) fn filter_to_window(map: &Map, points: &mut Vec<usize>, window_start_kf: usize) -> usize;
```
- Order inside `select_local_map_points`: the rule (lane A) picks keyframes and points, whole-map fallback included; then `filter_to_window` (lane B) runs when `window_start_kf` is `Some`.
- Skeleton body: today's selection, `dropped_by_window: 0`, `most_voted_keyframe: None`.
- Call sites, all moved by the skeleton to `.points` and to one config:
  - `map_projection/mod.rs:280` and `:577`: `MapProjectionEstimator::local_map_config()`, built from the fields `local_map_rules: LocalMapRules` (from settings) and `window_start_kf: Option<usize>` (setter `set_window_start`);
  - `system/mod.rs:1551`: the same config through `Tracker::local_map_config()`, so the visibility count uses the rule the search used;
  - `system/trace.rs:60`: the same config, passed into `begin`, so the trace shows what tracking searched.
- `Estimate` gains `most_voted_keyframe: Option<usize>`, filled from the `:577` selection; lane A reads it at `system/mod.rs:2164`.
- `select_seeded_window_points` (card B's first draft) is not added: the window is a config field.

**Growth (`mapping/keyframe_mapping.rs`).**
```rust
pub(crate) struct KeyframeGrowthPlan {
    keyframe_idx: usize,
    triangulation_neighbours: Vec<usize>,  // today's `neighbor_keyframe_indices`, read at `:176`
    fusion_neighbours: Vec<usize>,         // read at `:206`; equal to the above under `RecentTen`
}
pub(crate) fn prepare_keyframe_growth(map: &Map, keyframe_idx: usize, rule: GrowthNeighbours) -> KeyframeGrowthPlan;
/// Lane B's body: `prepare_keyframe_growth(map, keyframe_idx, rule)`, both lists filtered to `>= window_start_kf`.
pub(crate) fn prepare_seeded_window_growth(map: &Map, keyframe_idx: usize, rule: GrowthNeighbours,
    window_start_kf: usize) -> KeyframeGrowthPlan;
```
- Skeleton body: today's recent-ten list copied into both lists. The existing tests (`:397-559`) pass `GrowthNeighbours::RecentTen`.
- The dispatch at `system/mod.rs:2121` is written by the skeleton: `seeded_window_start()` `Some(s)` calls the seeded wrapper, `None` calls `prepare_keyframe_growth`, both with the run's rule. `seeded_window_start` returns `None` until lane B fills it.

**Segment id.** `pub type SegmentId = u32;` in `mapping/map/mod.rs`, re-exported as `crate::mapping::SegmentId`. `mapping/session.rs` (D) and `system/recovery.rs` (B) import it; mapping never imports a system type. `Segments { starts: Vec<usize> }` keeps keyframe ids as `usize`; `of` and `latest` return the segment index as `SegmentId`. The skeleton ports `Segments` with its four bodies from `4a2ee7b` (bookkeeping, no output), the `segments: Segments` field on `SlamSystem`, the `segments.begin` call in `finish_bootstrap`, and `segment_of`. So lane D does not wait on lane B's body.

**Files more than one lane edits: the skeleton declares, lanes fill.**
- `crates/kornia-sensors/src/imu.rs`: `SampleInterpolation` and `PreintegratedImu::from_measurements_with` (A); the `cfg_attr(feature = "serde", derive(Serialize, Deserialize))` on `ImuCalib` (B; the skeleton adds the derive itself); `GyroIntrinsics` and `apply` (D).
- `system/mod.rs`: `imu_trust` (A); `relocalization_due`, `relocalize(.., lost)`, `seeded_window_start` (B); `set_session_recording`, `evicted_keyframes`, `segment_of` and their fields (D). The eviction copy in `enforce_map_bound` is D's body.
- `tracking/tracker.rs`: `Tracker::set_window_start(Option<usize>)` (B), `Tracker::local_map_config()`, and the `ImuTrust` search-scale input `Tracker::set_search_scale_override(Option<f32>)` (A; `None` keeps `search_scale_for`).
- `tracking/pose_estimation/map_projection/mod.rs`: the fields, setters and `local_map_config()` above.
- `system/counters.rs`: every `RunCounters` field of cards A and B.
- `apps/kornia-slam-app/src/summary.rs`: one new summary line, after the existing ones, that holds both lanes' counters in a fixed order, plus a test pinning it. The first line and its test (`summary.rs:91`) do not change. No lane edits this file.
- `apps/kornia-slam-app/src/{config.rs, main.rs}`: `RunConfig.rig_calibration` and `--write-rig-calibration` (B); `RunConfig.offline_ba`, `--session-out` and the optional `offline-ba` subcommand (D). Each body is `todo!()`, reached only when the field or flag is given.
- `mapping/vi_ba_schur.rs`: `ViBaParams.newest_marginal`, `ViBaParams.imu_recently_initialized`, `ViBaResult.newest_marginal` (A); `depth_residual_and_jacobian` and `visual_huber` made `pub(crate)` (D).
- `mod` lines: `upstream/mod.rs`, `upstream/kornia_3d/mod.rs` (`pose_ext`, `gpnp`, `refine_analytic`, `pgo`), `upstream/kornia_algebra/mod.rs`, `mapping/mod.rs` (`session`, `global_vi_ba`), `inertial/mod.rs` (`trust`), `tracking/pose_estimation/mod.rs` (`pose_inertial`), kornia-sensors `lib.rs` (`rig_calibration`).
