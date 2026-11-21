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

- **2026-11-21, scoring.**
  - **Setup:** binary `92acd29` plus the diagnosis print (md5 7d09a8c68f297830aac35ad1dc844ba7), preset `pr180-frontend-1ms`, 800-frame windows, `--no-pace --debug`, 3 runs per config, 2 at a time (an extraction job ran alongside).
  - **Measurements:**
    - tracking counts from `[summary]`;
    - ATE SE3 from `quick-eval/ate.py`;
    - the step ratio from the new `quick-eval/steps.py`, over poses 50-250;
    - the near-point counter from the diagnosis print, keyframes 0-60, points in pairs with p50 < 0.4 m.

| P/K | Prediction | Baseline (this binary) | Measured (each run) | Verdict |
|---|---|---|---|---|
| P1 (KA, R_03) | near ≤ 150; total 800-2600 | Ib: near 2394, total 2394 | near 11, 13, 13; total 3039, 3036, 3037 | near as predicted; total missed, above the range (growth not starved) |
| P2 (KA, R_03) | step ratio 0.95-1.05 | Ib 1.300, 0.718, 0.235; I0 1.051, 1.048 | 1.012, 1.012, 1.010 | as predicted |
| P3 (KA, R_03) | tracked ≥ 777, coast ≤ 10, VIBA2 accepted 3/3 | Ib tracked 705, 715, 542, coast 85-128, VIBA2 rejected 3/3; I0 787, 786, 776 | tracked 790, 798, 784; coast 10, 2, 16; VIBA2 accepted 3/3; ATE 0.202, 0.194, 0.196 m | as predicted except coast in run 3 (16) |
| P4 (KA, R_03) | VIBA0 gravity error ≤ 1.0°, by VIBA1's correction | Ib VIBA0 gravity z 0.350 | VIBA1's gravity is not printed by this binary (card J's print was temporary); VIBA0 gravity z 0.062-0.073, and VIBA2 is accepted 3/3 | not measured; proxy consistent |
| P5 (KA, R_11) | 800/800 3/3, ATE ≤ 0.35 m | Ib (card I) 800 ×3, 0.214-0.333 m | 800, 799, 798; ATE 0.136, 0.156, 0.170 m; scale 0.947-0.970 | tracked missed by 1-2 frames; ATE better than predicted |
| P6 (KA, R_01) | 800/800, ATE ≤ 0.13 m | base 0.097-0.112 m | 800 ×3; ATE 0.120, 0.108, 0.119 m | as predicted |
| P7 (KC, R_03) | tracked 777-788, ATE ≤ 0.25 m | I0 787, 786, 776; 0.208-0.216 m | 797, 788, 789; ATE 0.214, 0.209, 0.218 m; ratio 1.045; near 16 ×3 | as predicted (tracked at or above the band) |
| P8 (KB, R_03) | passes P1-P3 if KA does | as above | near 10, 26, 10; ratio 1.017, 1.012, 1.017; tracked 775, 790, 776; coast 25, 10, 24; VIBA2 accepted 3/3 | P1-P2 as predicted; P3 missed in 2 runs (775, 776) |
| K1 | near falls but tracking does not | n/a | tracking rose with the counter | not fired |
| K2 | tracking rises but near does not fall | n/a | near fell 2394 to 11-13 | not fired |
| K3 | R_01 or R_11 regresses | n/a | R_11 ATE improved, R_01 inside 0.13 m | not fired |
| K4 | P2 holds but P4 fails | n/a | P4 not measured | open |

- **The theory was right.**
  - Under `min_cold_start_points: 20`, R_03's early keyframes grew points only from pairs too short for the 1° parallax gate. Those points sat at 2-25 cm instead of 0.8 m and shrank the trajectory to about 1/5 scale.
  - The baseline-to-depth gate removes them (2394 to 11-13). It does not starve growth: total growth in keyframes 0-60 rose to about 3040.
  - With the gate, the scale comes out right (ratio 1.01), and VIBA2 is accepted in 3/3 runs, which card J's two fixes could not achieve.
  - R_03 tracks 784-798/800 with ATE 0.194-0.202 m, better than the base (0.208-0.216 m).
- **G31 unblocked.** With the gate, `min_cold_start_points: 20` no longer breaks R_03.
  - R_11 tracks 798-800/800 with ATE 0.136-0.170 m (Ib alone: 0.214-0.333 m).
  - R_01 stays 800/800 at 0.108-0.120 m, slightly above the base band of 0.097-0.112 m; the 0.13 m bound held.
- **The gate on the base (KC) is neutral to slightly better.** The near-point counter was only 82 there to begin with.
- **The ORB-SLAM3 mono ratio 0.01 (KB) also fixes the scale,** but it tracks less on R_03 (775-790). tan 1° = 0.0175 is the better setting, as the theory predicts: it is the ratio below which a correct match cannot pass the parallax gate.
- **Next:**
  - ship `growth_min_baseline_ratio: 0.0175` together with `min_cold_start_points: 20` in `pr180-frontend-1ms`. This needs the owner's go, and the validation workflow should confirm it on the other sequences.
  - P4 is open until a VIBA1 gravity print exists. Roadmap item "still-device handling" is next.
- **2026-11-21, shipped.** By the owner's decision, `configs/pr180-frontend-1ms.ron` now sets `mapping.growth_min_baseline_ratio: 0.0175` and `tracking_loss.reseed.min_cold_start_points: 20`. The knob defaults in code stay off.
