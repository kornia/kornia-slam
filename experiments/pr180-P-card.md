# Card P: far stereo depths are trusted far beyond their disparity

Status: written 2026-11-21, before any run. Existing knob only, so there is no code change.

## Experiment: P, stereo depth sigma from disparity

- **Gap tested:** G35 (scale drift on full sequences). It is the common cause behind cards H, M, N and O, where every increase in IMU weight made results worse.
- **Base:** @ 9a28785, preset `pr180-frontend-1ms`.
- **Binary:** md5 7d09a8c68f297830aac35ad1dc844ba7.

### Evidence (2026-11-21, full R_01)

- **Ruled out against ground truth:**
  - IMU data: scale 0.997, residual 3 mm;
  - camera-IMU clock: about +5 ms;
  - camera-IMU rotation: 0.37°, ground truth 0.33°;
  - lever arm: no consistent offset;
  - preintegration noise discretisation: density²/dt, correct.
- **Stereo only (`imu: false`, 2 runs):** ATE SE3 4.18 and 4.58 m; Sim3 scale 0.76 and 0.91; step ratio per 250 poses 0.67-1.96. Stereo does not hold the scale.
- **Base run (stereo + IMU): step ratio against the median stereo depth of the same 250-pose bin:**
  - near scenes (2.1-3.0 m) run long, 1.03-1.16;
  - far scenes (7.0-11.1 m) run short, 0.83-0.90.

### Mechanism (file:line on 9a28785)

- `StereoDepthSigma::default()` is `RelativeDepth(0.05)` (`system/config/mod.rs:623-626`), so σ_z = 0.05·z.
- The physical σ from disparity noise is σ_z = z²·σ_d/(f·b), with f = 241.6 and b = 0.138 m. At σ_d = 1 px that gives:
  - 0.085 m at 1.7 m (equal to the 5 % model there);
  - 0.27 m at 3 m;
  - 3.0 m at 10 m, against 0.50 m trusted, i.e. 36× too much information.
- Far stereo points have only 3-4 px of disparity, so any disparity bias δ (calibration or rectification between the two pinhole crops, or the matcher) changes their depth by about −δ/d: −9 % at 10 m for δ = 0.3 px.
- Those over-weighted far points set the scale. It then wanders with the scene depth, and the IMU fights them: more IMU weight gives more conflict, as cards H, M, N and O showed.

### Reference

ORB-SLAM3 weights a stereo observation as a third image coordinate `uR`, with the keypoint's pixel σ (`EdgeStereoSE3ProjectXYZ`, `invSigma2` per octave). That is equivalent to the disparity model. Basalt uses stereo only through two-view reprojection, also in pixels.

### Knob (existing)

`mapping.local_ba.stereo_depth_sigma`: `RelativeDepth(0.05)` (default) or `Disparity(px: ...)`.

### Configs (full R_01)

| Name | Change |
|---|---|
| P1 | `stereo_depth_sigma: Disparity(px: 1.0)` |
| P2 | P1 plus `active_keyframes: 10` |
| P3 | P1 with `imu: false` |

### Predictions (P1 and P2: 3 runs; P3: 2 runs)

- **P1, scale (P1):** the Sim3 scale moves from 1.08-1.12 into 0.95-1.05, and the per-bin step ratio spread falls from 0.18-0.21 to ≤ 0.12.
- **P2, accuracy (P1):** ATE SE3 ≤ 0.8 m (base 1.11-1.23 m).
- **P3, the window flips (P2):** P2's ATE is ≤ P1's in at least 2 of 3 runs, with solves moving > 30 cm ≤ 10 per run. Card M had 33-60 such jumps.
- **P4, mechanism (P1):** the correlation of the per-bin step ratio with the bin's median stereo depth goes from the base's (near long, far short) to no trend: the far-bin and near-bin means within 0.05 of each other.
- **P5, stereo only (P3):** step ratio spread ≤ 0.6 (base stereo-only 0.94-1.26). It stays worse than with the IMU: monocular-like scale drift remains without the IMU.

### Kill rules

- **K1:** P4 holds (no depth trend), but P1 and P2 miss. Stereo weighting was not the drift cause. Next: measure the disparity bias directly, from stereo points and their BA'd depths.
- **K2:** P1 and P2 hold, but P4 does not. Scale improved for another reason. Do not ship until it is understood.
- **K3:** tracking drops (tracked < 2890) or a VIBA is rejected. Weaker stereo starves the start-up. Keep `RelativeDepth` before VIBA2, and test a hybrid.

### Result log

(empty)
