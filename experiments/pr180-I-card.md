# Card I: a young map is not thrown away at its first lost frame

Status: written 2026-11-18, before any code.

## Experiment: I, young-map grace and cold-start stereo minimum

- **Gap tested:** tracking on R_11_5cp. The owner's target is 100 % tracking on easy and medium sequences.
- **Base:** kornia-slam `feat/global-pose-graph` @ 9f8e2e0, preset `pr180-frontend-1ms`, LaMAria 800-frame windows.

### Mechanism, from `cardH/out/r11-base-r1` (`--debug`)

- **First map lost at once:** frame 0 seeds a map from 90 stereo points. Frame 3 fails with `LowReferenceMatches`. A map with ≤ `min_keyframes_for_grace` (10) keyframes is reset on its first lost frame, with no grace (`system/mod.rs:1978-1981`).
- **Cold start then starves:** it needs `MIN_STEREO_POINTS` = 50 stereo points (`initialization/bootstrap.rs:35`, used at `system/mod.rs:1245-1249`). Frames 3-130 have 12/21/31 at p10/p50/p90, at most 49.
- **The next maps die young too:** the maps seeded at frames 71 and 76 have 55 and 52 points and are reset at frames 75 and 78. The map from frame 131 holds.
- **Cost:** 122 frames are skipped in bootstrap and there are 3 give-ups; 668 of 800 frames are tracked (3 runs: 667, 669, 668).
- **Same reset on the easy sequences:** R_01 and R_03 also reset at frames 2-4. They recover in the next frame because their overlap gives more than 50 points.

### Theory

- **(a) The reset destroys the only usable structure.** The 90-point map seen at frames 0-2 is still in view at frame 3, which failed on reference matches, not on view. If the map is kept for a short grace, map projection can re-track it from the motion model, and the next keyframes grow it. A new map would instead need 50 overlap points, which these frames rarely have.
  - ORB-SLAM3 resets a map with fewer than 10 keyframes (`Tracking.cc`, `KeyFramesInMap() < 10`). Its stereo start-up uses the full stereo field of view, where a reset costs one frame. On the pinhole-cropped Aria rig, one reset costs about 70 frames.
- **(b) The 50-point minimum is ours.** ORB-SLAM3's `StereoInitialization` asks for more than 500 keypoints, not for a count of stereo points. A smaller minimum starts a map within a few frames of a loss. The risk is a thin map that dies again, which (a) then holds.

### Knobs (both in `tracking_loss.reseed`, `RecoveryConfig`; the defaults keep today's behaviour)

- `young_map_grace_sec: f64`, default `0.0`: a map with ≤ `min_keyframes_for_grace` keyframes is reset only after it has been lost this long. `0.0` resets at once, as today.
- `min_cold_start_points: usize`, default `50` (`MIN_STEREO_POINTS`): the fewest stereo points that seed a map from a cold start.

### Configs

| Name | Knobs |
|---|---|
| I0 | base (`pr180-frontend-1ms`) |
| Ia | `young_map_grace_sec: 0.5` |
| Ib | `min_cold_start_points: 20` |
| Iab | both |

### Predictions (3 runs each, `[summary]` counts)

- **P1, Ia on R_11:**
  - `skipped_bootstrap` goes from 122 to ≤ 40 and `give_ups` from 3 to ≤ 1;
  - tracked goes from 668 to ≥ 740.
- **P2, Ib on R_11:**
  - `skipped_bootstrap` goes from 122 to ≤ 30;
  - `give_ups` goes from 3 to 3-8, because the thin maps die;
  - tracked goes from 668 to ≥ 720.
- **P3, Iab on R_11:**
  - tracked ≥ 780/800, the owner's bar within 2.5 %;
  - ATE SE3 ≤ 0.25 m (base band 0.158-0.239 m).
- **P4, Iab on R_01 and R_03:**
  - tracked: R_01 stays at 800/800, R_03 is ≥ 790/800;
  - ATE inside the base bands (R_01 0.097-0.112 m; R_03 from 3 base runs taken with this card).
- **Mechanism (P1, P3):** a new debug line `[lost] frame= young map held`. Under Ia, the first R_11 give-up at frame 3 becomes held frames, and the frame-0 map reaches 10 keyframes.

### Kill rules

- **K1:** `skipped_bootstrap` falls, but tracked does not rise by at least half the frames saved. The held map cannot re-track, so (a) is incomplete. Next, look at why frame 3 fails `LowReferenceMatches`.
- **K2:** tracked rises, but R_11 ATE goes above 0.30 m or scale leaves 0.95-1.05. Thin or held maps corrupt the trajectory. Keep only the knob that does not.
- **K3:** R_01 or R_03 tracking falls, or ATE leaves the base band. The knob hurts easy sequences and does not ship in the presets.

### Bit-identity and noise

- **Identity:** with defaults, nothing changes. The app is not deterministic run to run, so this is checked by unit tests: the default config equals today's, and the give-up rule with grace 0.0 matches the old one.
- **Noise:** 3 runs per config. The R_11 base band is tracked 667-669 and ATE 0.158-0.239 m.

### Result log
- **2026-11-18, first scoring:**
  - **Setup:** binary 9f8e2e0 plus the card I knobs, `pr180-frontend-1ms`, 800-frame windows, `--debug`, 3 runs per config.
  - **Measurements:** tracked, `skipped_bootstrap` and `give_ups` come from `[summary]`; "held" counts the `young map held` lines. ATE is SE3 against pseudo-dense ground truth, which on R_11 covers part of the window only.

| P/K | Prediction | Baseline (I0) | Measured (each repeat) | Verdict |
|---|---|---|---|---|
| P1 (Ia, R_11) | skipped ≤ 40, give-ups ≤ 1, tracked ≥ 740 | skipped 122, give-ups 3, tracked 667-671 | skipped 104 ×3, give-ups 2 ×3, tracked 667, 674, 670; held 20 | missed |
| P2 (Ib, R_11) | skipped ≤ 30, give-ups 3-8, tracked ≥ 720 | as above | skipped 0 ×3, give-ups 2 ×3, tracked **800 ×3**; ATE 0.307, 0.333, 0.214 m; scale 1.08-1.16 | tracking better than predicted; give-ups below range |
| P3 (Iab, R_11) | tracked ≥ 780, ATE ≤ 0.25 m | ATE 0.166-0.214 m | tracked 783, 783, 786; ATE 1.060, 0.996, 0.954 m; scale 1.71-1.85 | **K2 fired** |
| P4 (Iab, R_01 / R_03) | R_01 800/800, R_03 ≥ 790, ATE in the base bands | R_01 800/800, 0.097-0.112 m; R_03 777-778, 0.207-0.211 m | R_01 790 ×3, 0.353-0.441 m. R_03 482, 767, 464 (give-ups 24, 1, 25), 0.32-0.65 m | **K3 fired** |
| K1 (Ia) | `skipped_bootstrap` falls but tracked does not rise | n/a | skipped 122 to 104, tracked unchanged | **fired** |
| K3 (Ib alone, R_01 / R_03) | easy sequences unchanged | as above | R_01: 800 ×3, 0.105-0.111 m, unchanged. R_03: 747, 677, 685; `coast_inertial` 53-123; ATE 1.04, 4.28, 4.06 m; scale 0.20-0.90 | **fired on R_03** |

- **(a) is wrong as built.** A held young map does not re-track:
  - its held frames end in the same reset, 20 frames later;
  - under Iab, holding the R_01 frame-0 map costs 10 frames and moves ATE from 0.10 to 0.35-0.44 m.

  The failure at frame 3 is not a single bad frame. Next, look at why `LowReferenceMatches` hits the frame-0 maps on every sequence (frames 2-4).
- **(b) solves R_11 tracking but breaks R_03 indirectly.**
  - On R_03, the map starts at frame 0 (45 points) instead of frame 1 (51 points), and is reset at frame 2 instead of 3, so the map that survives starts one frame earlier.
  - The VIBA0 that follows estimates gravity z = 0.350 instead of 0.001. Tracking is then lost at frame 271, VIBA2 is rejected, and the run coasts for 53-123 frames.
  - The thin map is not the direct cause. The inertial initialisation is sensitive to its starting frame, which is a separate gap.
- **Outcome:** neither knob ships in a preset. Both stay in the code, off by default.
- **Next gaps, in order:**
  1. the frame-0 map dying at frames 2-4 on every sequence;
  2. VIBA0's sensitivity to the start frame on R_03;
  3. card F (raw fisheye), for the overlap.
