# Card N: the IMU edges are overconfident against the real accelerometer error

Status: written 2026-11-21, before any run. Config only (a calibration copy), no code change.

## Experiment: N, accelerometer noise that covers the unmodelled error

- **Gap tested:** scale drift on full sequences (G35), and card M's result that a longer VI-BA window makes it worse.
- **Base:** kornia-slam `feat/global-pose-graph` @ 9a4fb6f, preset `pr180-frontend-1ms`.
- **Binary:** `92acd29` plus the card K diagnosis print (md5 7d09a8c68f297830aac35ad1dc844ba7). The commits after it do not touch the pinhole rig path's estimator.

### Evidence (2026-11-21)

- **Card M (full R_01):**
  - with 10 active keyframes, ATE SE3 is 3.96, 1.63 and 4.77 m (base 1.23 m), and the Sim3 scale 0.81-1.01;
  - with the 25-keyframe large window, run 1 collapses (ATE 31.3 m, scale 0.047), and `|v|` reaches 6-8 m/s on a walk at about 1.2 m/s.
  - More IMU constraint makes the estimate worse.
- **The IMU data is right.** A least-squares fit of `f_body = M·a_meas + c`, plus gravity, against ground-truth motion uses p(t+T) − 2p(t) + p(t−T) with T = 0.5 s (`scratchpad/imucheck/accfit.py`):
  - R_01: M's singular values 0.988, 0.996, 1.004 and rotation under 0.22°; residual 2.97 mm; `M = s·I` gives s = 0.997; `T_b_s` is identity (Rz(180) does not fit).
  - R_03: the same (0.993-1.003, 0.1°, 2.97 mm).
  - The bias is about 0.45 m/s² in total. Our VI-BA's estimate matches it: about (0.22, 0.20, 0.39) against the fitted (0.24, 0.20, 0.35) m/s².
- **But the IMU information assumes white noise only.**
  - `acc_noise_density` is 7.8e-4 m/s²/√Hz, and the position block of a 0.15 s edge reaches Ω ≈ 5.8e9 (σ ≈ 13 µm; `[vi_ba]` lines).
  - The fit shows up to 1.2 % scale and cross-axis error: R_01 and R_03 have no factory rectification. On gravity that is about 0.1 m/s², which changes with head orientation, so a constant bias cannot absorb it.

### Mechanism

- `information()` (`kornia-factors/src/inertial/preintegration.rs:187`) inverts the preintegration covariance, which comes only from the white-noise densities.
- An unmodelled, orientation-dependent accelerometer error of about 0.1 m/s² gives a position error over 0.15 s of about ½·0.1·0.15² ≈ 1.1 mm. That is about 90× the modelled σ of 13 µm, so the edge is overconfident by about 10⁴ in χ².
- The VI-BA then bends velocities and scale to satisfy these edges, against the visual evidence. A longer window adds more such edges and more freedom, which is what card M measured.

### Theory and reference

- **Basalt and OpenVINS** inflate the IMU noise above the datasheet for real devices; the Kalibr IMU-noise guidance is to multiply the Allan-variance densities by 10 or more.
- **ORB-SLAM3** reads the noise from its settings file, and its EuRoC and TUM-VI files use inflated values.
- So the best fix from the references is an accelerometer noise model that covers the real error. The first test needs no code: a calibration copy with `acc_noise_density` × 10.
- If the theory holds, the inflated noise makes the IMU edges consistent, the scale stops wandering, and a longer window helps instead of hurting.

### Configs (full R_01, 2898 frames)

| Name | Change |
|---|---|
| N0 | base preset (the R_01 run in card M's evidence is N0 run 1) |
| N1 | `acc_noise_density` × 10 (7.84e-3) in a copy of the pinhole calibration JSON |
| N2 | N1 plus `mapping.local_ba.active_keyframes: 10` (card M's M1) |

### Predictions (3 runs each; N0 has 2 new runs plus run 1)

- **P1, mechanism (N1):** the median final chi2 per interior IMU edge in `[vi_ba] FINAL` lines falls by ≥ 10×.
- **P2, scale wander:** the spread of the per-250-pose step ratio over bins 250-2750 goes from 0.212 (N0 run 1) to:
  - ≤ 0.12 for N1;
  - ≤ 0.10 for N2.
- **P3, global scale:** Sim3 scale from 1.116 into 0.95-1.05 (N1), and 0.97-1.03 (N2).
- **P4, accuracy:** ATE SE3 from 1.23 m to:
  - ≤ 0.8 m for N1;
  - ≤ 0.6 m for N2.
- **P5, the window flips:** N2's ATE is below N1's in at least 2 of 3 runs. The reverse of card M, where the longer window made it worse.

### Kill rules

- **K1:** P1 holds (the chi2 drops), but P2-P4 do not move. Overconfidence is not what drives the drift. Next, run our IMU factor on ground truth (a Rust harness: χ²/dof per edge with GT poses, GT velocity and the fitted bias), to look for a factor bug.
- **K2:** ATE improves, but N2 is still worse than N1. Inflated noise helps, but the window still hurts, so a second cause sits in the window solve: card M's velocity blow-up, the boundary prior, or the marginal.
- **K3:** N1 loses tracking (tracked < 2890) or VIBA is rejected. The IMU is now too weak for initialisation. Inflate only in local BA, which needs a knob and a code change.

### Noise

3 runs per config, full R_01.

### Result log

(empty)
