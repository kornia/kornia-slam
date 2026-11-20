# Card K: no map points from keyframe pairs too close for their parallax gate

Status: written 2026-11-20, before any code. Roadmap item "baseline and parallax gate on new map points" (2026-11-19).

## Experiment: K, baseline-to-depth gate on pair growth

- **Gap tested:** G32 (R_03 breaks under `min_cold_start_points: 20`), which blocks G31 (R_11 start-up starvation).
- **Base:** kornia-slam `feat/global-pose-graph` @ df45adf, preset `pr180-frontend-1ms`, LaMAria windows.
- **Binary tag:** `cardK`.

### Evidence (diagnosis, 2026-11-20)

- **How it was measured:** a temporary print in `pair_growth_in_camera` (`mapping/growth.rs`), not committed. For each keyframe pair, it printed:
  - the baseline;
  - the match count;
  - the number triangulated;
  - the depth p10, p50 and p90 of the new points in the current camera.
- **Setup:** R_03, 320 frames, 3 runs each of I0 (base) and Ib (`min_cold_start_points: 20`). The patch is kept with the run directories.

| Config | Keyframes | Points grown | From baseline < 10 mm | In pairs with p50 depth < 0.4 m |
|---|---|---|---|---|
| I0 (×3) | 0-60 | 2502-2566 | 55 | 82 |
| I0 (×3) | 61-250 | 8027-8399 | 0 | 2-4 |
| Ib (×3) | 0-60 | 2399 | 1637 | **2399** |
| Ib (×3) | 61-250 | 8850-9153 | 2060-2100 | **8633-8913** |

- **The true scene depth is about 0.8 m:** the stereo seed has median 0.80 m, and per-frame stereo depth stays at 0.79-0.80 m.
- **Ib's early keyframes:** grown points have p50 0.02-0.25 m. The pairs have baselines of 0.6-8 mm, at keyframes 5, 8, 16, 24 and 32, while the device is nearly still.
- **I0's early keyframes:** they show the same near points (keyframes 6-10, baselines 3-9 mm, p50 0.05-0.25 m), but only 82 of them. The pairs at keyframes 18-26 have 30-50 mm baselines and add hundreds of points at p50 0.9-1.2 m, which dominate.

### Mechanism (file:line on df45adf)

- **The only baseline guard is `rel.translation.length() <= 1e-8`** (`mapping/growth.rs:233`).
- **The per-point parallax gate selects wrong points at a small baseline.**
  - Growth uses `TwoViewInitConfig`'s triangulation settings, `min_parallax_deg` 1.0 (the kornia-3d default; `initialization/two_view.rs:58-63`, `system/config/mod.rs:713-717`).
  - A correctly matched point at depth d seen across baseline b has parallax at most about b/d radians.
  - So when b/d < tan(1°) = 0.0175, no correct match passes the gate. Only a wrong or noisy match, whose disparity is not from the baseline, can reach 1°. It then triangulates at depth ≤ b / tan(1°).
  - At b = 5 mm that is ≤ 0.29 m, which is what the table shows.
- **Why it sticks.**
  - The near points enter the local map next to the 0.8 m stereo points.
  - When the device starts moving (about frame 33), map projection fits the translation to the near points as well, so the estimated step shrinks. Rotation is unaffected, matching card J.
  - Later pairs triangulate consistently at the shrunk scale. That is why keyframes 61-250 are still near (8633-8913 points) and the step ratio stays at 0.18-0.23 until the re-seed near frame 270.

### Theory

- **The gate.** Skip a keyframe pair when `baseline / median_scene_depth(curr_kf) < min_baseline_ratio`. With the ratio at tan(min_parallax_deg), every pair skipped is one from which only wrong points could pass the parallax gate, so no correct point is lost.
- **Reference: ORB-SLAM3 `LocalMapping::CreateNewMapPoints`.**
  - Stereo: skip if `baseline < pKF2->mb`.
  - Monocular: skip if `baseline / pKF2->ComputeSceneMedianDepth(2) < 0.01`.
- **Why the ratio form, not the stereo form, is the best of the references here.**
  - The Aria rig's stereo baseline is about 0.138 m (R_03 calibration, |t_cam0 − t_cam1|).
  - Skipping every pair below it would drop nearly all growth at walking speed.
  - Growth matters here because the pinhole-cropped overlap gives only about 21 stereo points per frame (card I).
- **Median depth.** The median of the current keyframe's map points, depth in its camera 0 (ORB-SLAM3 `ComputeSceneMedianDepth`).
  - At the seed, that median comes from the 0.8 m stereo points. So the gate removes the first near points, before they can shrink the scale.
  - A map already at the wrong scale is self-consistent, so the gate cannot repair it. That is not this card's claim.

### Knob

`mapping.growth_min_baseline_ratio: f64`, default `0.0`, which keeps today's behaviour (only the `1e-8` guard). When it is greater than 0, a pair is skipped if `|t_rel| < ratio · median_depth(curr_kf)`. A keyframe with no map points uses no gate. Validation: finite and ≥ 0.

### Configs

| Name | Knobs |
|---|---|
| I0 | base (`pr180-frontend-1ms`) |
| Ib | `min_cold_start_points: 20` |
| KA | Ib + `growth_min_baseline_ratio: 0.0175` (tan 1°) |
| KB | Ib + `growth_min_baseline_ratio: 0.01` (ORB-SLAM3 mono) |
| KC | I0 + `growth_min_baseline_ratio: 0.0175` |

### Predictions

All runs are 3 per config. The tracking counts are 800-frame windows. The counters come from the diagnosis patch applied on top of the `cardK` binary.

- **P1, mechanism counter (KA, R_03):**
  - points grown in pairs with p50 < 0.4 m, keyframes 0-60: from 2399 (Ib) to ≤ 150 (the I0 level is 82);
  - total grown in keyframes 0-60: 800-2600.
- **P2, scale (KA, R_03):** the step-length ratio ours/GT over frames 50-250 goes from 0.18-0.23 (Ib) to 0.95-1.05 (I0: 1.00).
- **P3, tracking (KA, R_03):**
  - tracked goes from 677-747 (Ib) to ≥ 777 (I0 band 777-788);
  - `coast_inertial` goes from 53-123 to ≤ 10;
  - VIBA2 is accepted in 3/3 runs.
- **P4, inertial initialisation (KA, R_03):** VIBA0's gravity error, measured as VIBA1's correction, goes from 3.4° (Ib) to ≤ 1.0° (I0: 0.47°).
- **P5, R_11 (KA):** tracked 800/800 in 3/3 runs, and ATE SE3 ≤ 0.35 m (Ib: 0.214-0.333 m).
- **P6, R_01 (KA):** tracked 800/800, and ATE ≤ 0.13 m (base band 0.097-0.112 m).
- **P7, the gate on the base (KC, R_03):** tracked in 777-788, and ATE ≤ 0.25 m (base 0.207-0.211 m).
- **P8, KB:** if KA passes P1-P3, KB passes them too. It is run only on R_03.

### Kill rules

- **K1:** P1 holds (near points ≤ 150), but P3 fails. The early near points are not the whole cause. Next, look at the keyframes inserted while still (roadmap item "still-device handling").
- **K2:** P3 holds, but P1 fails. Tracking moved for another reason (luck, or the re-seed). Do not ship.
- **K3:** P5 or P6 regresses (tracked falls, or ATE is out of its band). The gate starves growth. Try KB. If KB also regresses, gate per point by stereo depth instead.
- **K4:** P2 holds, but P4 fails. VIBA0's error has a second source; the unrectified R_03 IMU is next.

### Bit-identity and noise

- **Identity:**
  - with the knob at 0.0, a unit test shows the gate function admits every pair with a baseline above 1e-8;
  - on R_03, I0 and Ib repeat card J's VIBA numbers to 3 digits (the app is not deterministic run to run).
- **Noise:** 3 runs per config. Bands: R_03 I0 tracked 777-788, ATE 0.207-0.211 m; R_01 0.097-0.112 m; R_11 Ib ATE 0.214-0.333 m.

### Result log

(empty)
