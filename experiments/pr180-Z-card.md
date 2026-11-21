# Card Z: flow-frame keyframe test and track thinning (R-D2, R-D1; G33)

Status: written 2026-11-21, before any code. From the PR #180 full review (R-D1, R-D2), plan
`pr180-full-review-fixes.md`, bucket 3. Awaiting owner approval (STOP 1).

Paths: `S` = `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad`,
`P` = `/home/edgar/dev/split/.pr180-fix`, `K` = `crates/kornia-slam/src`.

## Experiment: Z, compare flow inliers with a count of the same kind (Z1); keep associated tracks first (Z2)

- **Gap tested:** G33 (speed).
- **Base:** kornia-slam `feat/global-pose-graph` at the gated fix tip, tag `pr180c/F1` (made in phase 5 of
  the plan; not yet cut). Binary tag `x`: `P/bin/kornia-slam-x`, built from `pr180c/F2`, md5 recorded at build.
- **Baselines (debug logs, `S/cardV/out`):**
  - full R_01 ALL r1 (`r01full-ALL-r1`): 1449 keyframe-due on 2898 frames, 593 keyframes, 2074 ORB frames (72 %).
  - R_11 ALL r1 (`r11-ALL-r1`): 313 due, 181 keyframes.
  - R_03 W r1 (`r03-W-r1`): only 343 of 800 frames reach a flow decision (144 tracked, 157 due, 42 refused);
    206 keyframes; frontend p50 8.2 ms under load.
  - BASE = V1 on the new binary, configs `P/baseline/{r03,r11,r01full}-V1.ron` (copies of `S/cardV/cfg`).
    R_11 and full R_01 BASE runs are made in the scoring phase. New runs:
    `P/out/{r03,r11,r01full}-{BASE,Z1,Z1Z2}-r<n>`.

### Mechanism (file:line at ad443b7; code equals f57c13a)

- **Z1:** `K/tracking/policy.rs:178-183` compares flow inliers, capped at 2 x 48 = 96 by `track_refill`, with
  0.6 x the reference keyframe's map points. From 161 reference points on, every flow frame is "due".
- **Z2:** `K/tracking/optical_flow.rs:329-347` thins oldest first, so unassociated refill tracks evict the newly
  associated tracks of each ORB frame, and flow frames fall under `min_tracks: 48`.

### Theory and reference

- ORB-SLAM3 `NeedNewKeyFrame` compares two counts of the same kind (tracked points against the reference
  keyframe's tracked points).
- Z1: on a flow frame, compare the inliers with `ref_ratio` x the associated tracks alive at the last ORB frame.
- Z2: keep associated tracks first, and give an ORB match with no track the refill track within
  `ADOPTION_RADIUS_PX` (3 px) instead of a new track.

### Knob

- `tracking.keyframes.flow_ratio: ReferencePoints` (default = today) `| TrackSurvival`.
- `frontend.track_thinning: OldestFirst` (default = today) `| AssociatedFirst`.
- Class: bug fixes (a unit mismatch and an eviction order). No new number: Z1 reuses `ref_ratio`, Z2 reuses
  the adoption radius.
- Instrumentation (no knob): `[summary] flow_frames`, `flow_keyframe_due`, `flow_refused`, `flow_unavailable`.

### Configs

| Name | Config |
|---|---|
| BASE | `P/baseline/{r03,r11,r01full}-V1.ron`, new binary, knobs off |
| Z1 | BASE plus `flow_ratio: TrackSurvival` |
| Z1+Z2 | Z1 plus `track_thinning: AssociatedFirst` |

Runs: 3 per config on R_03, R_11 and full R_01. `--no-pace --debug`, `JOBS=2`, `RUN_FLAGS` empty, as cards
T, U, V. R_01 and R_03 need `ALLOW_RAW_IMU=1` (no factory file). Load sampled every 30 s into `<out>/load.log`.

### Predictions

- **P1 (Z1)** full R_01: `flow_keyframe_due` 1449 -> 600-800; ORB-frame share 72 % -> 45-55 %.
  R_11: due 313 -> 180-230.
- **P2 (Z1+Z2)** R_03: frames with a flow decision 343 -> at least 550 of 800; refused 42 -> at most 25.
- **P3** keyframes within +/- 5 % of BASE on each sequence (593, 181, 206).
- **P4** tracked and ATE inside the BASE bands on all three.
- **P5** frontend p50 on R_03 8.2 -> at most 6 ms; scored only at load within +/- 2 of BASE.

Measured by: the `[summary] flow_*` counters, `bash experiments/quick-eval/score.sh <out>`, and the
frontend timing lines.

### Kill rules

- **K1 (mechanism moved, score fell):** ORB share down 10+ points and ATE above the band in 2/3 on any
  sequence, or R_03 tracked under 776 in 2/3. Flow frames cost accuracy. Keep off; look at the flow pose (R-D7).
- **K2 (score moved, mechanism did not):** ATE or tracked beyond the band with the ORB share moved under
  5 points. Do not credit.
- **K3:** keyframes down more than 10 %: the flow test delays real keyframes. Try scaling n_ref by the track
  budget instead.

### Bit-identity and noise

- Knobs off: `P/baseline/r01full-S10-7.ron` (copy of `S/cardR/cfg/r01full-S10-7.ron`), synchronous,
  `RAYON_NUM_THREADS=1`, must be `cmp`-identical to `P/baseline/traj.txt` (copy of
  `S/cardT/out/T1/r01full-S10-7-r1/traj.txt`, md5 d4f77b567b912b5a4caa2af6085a35b1). The new counters only
  add `[summary]` output.
- Unit tests (lane flow-z): `a_flow_frame_is_not_due_while_its_tracks_survive`,
  `the_reference_test_is_unchanged_by_default`, `associated_tracks_survive_thinning`; the existing
  `thinning_keeps_the_oldest_tracks_within_the_budget` still passes with the default.
- Noise: asynchronous runs differ run to run; 3 runs per config. Known bands: R_03 tracked 776-788, ATE
  0.103-0.114 m (V1, n=3); R_11 ATE 0.149-0.190 m (n=6, U-on); R_01 full 0.30-0.50 m (n=9, all configs).
- Load: recorded; no timing verdict above load p50 5 (P5 also needs load within +/- 2 of BASE).

## Result log

