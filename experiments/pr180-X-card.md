# Card X: parallax gate on overlap-triangulated stereo points (R-D6)

Status: written 2026-11-21, before any code. From the PR #180 full review (R-D6), plan
`pr180-full-review-fixes.md`, bucket 3. Awaiting owner approval (STOP 1).

Paths: `S` = `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad`,
`P` = `/home/edgar/dev/split/.pr180-fix`, `K` = `crates/kornia-slam/src`.

## Experiment: X, gate overlap-triangulated seeds on their predicted relative depth error

- **Gap tested:** G35 (scale), G31 (cold start).
- **Base:** kornia-slam `feat/global-pose-graph` at the gated fix tip, tag `pr180c/F1` (made in phase 5 of
  the plan; not yet cut). Binary tag `x`: `P/bin/kornia-slam-x`, built from `pr180c/F2`, md5 recorded at build.
- **Baselines:** BASE = V1, configs `P/baseline/{r03,r11,r01full}-V1.ron` (copies of `S/cardV/cfg`;
  `pr180-window10` plus base budget 7 / 1e-4) on the new binary. Old BASE runs exist for R_03 only
  (`S/cardV/out/r03-V1-r{1,2,3}`); R_11 and full R_01 BASE runs are made in the scoring phase.
  New runs: `P/out/{r03,r11,r01full}-{BASE,X}-r<n>`.

### Mechanism (file:line at ad443b7; code equals f57c13a)

- `K/frontend/overlap.rs:33` accepts triangulations down to 0.25 deg of ray parallax
  (`MIN_PARALLAX_COS = 0.999_990_5`, tested at `:222`).
- With f = 241.6 px (R_01 pinhole calibration) one pixel is 0.237 deg, so the relative depth error
  sqrt(2)·sigma_px/(f·theta) is about 134 % at 0.25 deg and 29 % at 1.15 deg.
- The 2 px reprojection gate cannot reject them.
- The stereo bootstrap and every cold re-seed take all of them as metric landmarks
  (`K/initialization/bootstrap.rs:112`, `rig_seeds` with `max_depth = f64::INFINITY`).
- Keyframe insertion is already bounded at 35 baselines (about 3.85 m;
  `K/mapping/keyframe_mapping.rs:122-125`, `stereo_close_depth`), so only seeds are affected.

### Theory and reference

- ORB-SLAM3 `KannalaBrandt8::TriangulateMatches` rejects `cosParallaxRays > 0.9998` (1.146 deg).
- Better than a fixed angle: gate on the predicted relative depth error,
  theta_min = sqrt(2)·sigma_px/(f·rho_max).
- With sigma_px = 1 and rho_max = 0.3 this gives 1.12 deg here, ORB-SLAM3's value, and it follows the lens
  on other devices.

### Knob

- `frontend.overlap_parallax: MinAngle(deg: 0.25)` (default = today) | `DepthSigma(max_relative: 0.3)`.
- Class: reference constant replaced by a sensor-model rule. sigma_px comes from `visual_information`, f from
  the camera; rho_max is one dimensionless number, not per deployment.
- Instrumentation (no knob): `[summary] bootstrap_seeds`, `bootstrap_seeds_low_parallax` (seeds under
  theta_min, counted with the knob off too).

### Configs

| Name | Config |
|---|---|
| BASE | `P/baseline/{r03,r11,r01full}-V1.ron`, new binary, knob off |
| X | BASE plus `frontend: ( overlap_parallax: DepthSigma(max_relative: 0.3) )` |

- **Stage 0** (knob off, one BASE run per sequence): the share of seeds under theta_min. Apply K0 before any
  X run.
- Runs: 3 per config on R_03 and R_11, 3 on full R_01. `--no-pace --debug`, `JOBS=2`, `RUN_FLAGS` empty, as
  cards T, U, V. R_01 and R_03 need `ALLOW_RAW_IMU=1` (no factory file). Load sampled every 30 s into
  `<out>/load.log`.

### Predictions

- **P0** R_03 share at most 10 % (card J: 63 seeds, median 0.80 m). R_11 and R_01: measured in stage 0.
- **P1 (mechanism, knob on):** seeds per seed event fall by the stage-0 share +/- 2 points; no seed under
  theta_min.
- **P2** R_11: tracked at least 798 in 3/3 (BASE 798-800); `skipped_bootstrap` within +5 frames of BASE.
- **P3** ATE inside the BASE band on all three; full R_01 |scale - 1| at most BASE median + 0.01.

Measured by: `bash experiments/quick-eval/score.sh <out>` and the `[summary]` counters.

### Kill rules

- **K0:** stage-0 share under 5 % on all three sequences: the quick sequences cannot test X. Do not score it
  here; score it as an A/B in the validation run on the two sequences with the most give-ups.
- **K1 (mechanism moved, score fell):** at least 10 % of seeds dropped and R_11 tracked under 798 in 2/3 or
  `skipped_bootstrap` up by 20+. The gate starves the cold start (G31). Keep off. Next: keep far points as
  bearing-only observations instead of dropping them.
- **K2 (score moved, mechanism did not):** R_03 ATE under 0.09 m in 3/3 or R_01 under 0.28 m in 3/3 with
  under 5 % of seeds dropped. Not this mechanism; do not credit.

### Bit-identity and noise

- Knob off: `P/baseline/r01full-S10-7.ron` (copy of `S/cardR/cfg/r01full-S10-7.ron`), synchronous,
  `RAYON_NUM_THREADS=1`, must be `cmp`-identical to `P/baseline/traj.txt` (copy of
  `S/cardT/out/T1/r01full-S10-7-r1/traj.txt`, md5 d4f77b567b912b5a4caa2af6085a35b1). The new counters only
  add `[summary]` output.
- Unit tests (lane front-x): `the_default_gate_is_0_25_degrees`, `depth_sigma_gate_matches_the_formula`
  (1.12 deg at f 241.6), `a_seed_under_the_gate_is_counted`.
- Noise: asynchronous runs differ run to run; 3 runs per config. Known bands: R_03 tracked 776-788, ATE
  0.103-0.114 m (V1, n=3); R_11 ATE 0.149-0.190 m (n=6, U-on); R_01 full 0.30-0.50 m (n=9, all configs).
- Load: recorded; no timing verdict above load p50 5.

## Result log

