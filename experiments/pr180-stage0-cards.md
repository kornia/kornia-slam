# Stage-0 cards: counters only (PR #180 full review)

Status: written 2026-11-21, before any code. From the PR #180 full review, plan `pr180-full-review-fixes.md`,
bucket 3 (S0). Awaiting owner approval (STOP 1).

Paths: `S` = `/tmp/claude-1000/-home-edgar/978f5be7-3122-4eb6-bf8e-8ea40ff0cac6/scratchpad`,
`P` = `/home/edgar/dev/split/.pr180-fix`, `K` = `crates/kornia-slam/src`, `Q` = `experiments/quick-eval`.

## Common to all stage-0 cards

- **What a stage-0 card is:** a counter lands in the fix phase. No behaviour change, no knob. The card is
  decided from the validation run. A knob is built only if the threshold is passed; it then gets a full card
  (theory, knob, predictions and both kill rules) before any code. Owner answer 9 (2026-11-21): no knob for
  any of these in this workflow.
- **Base:** kornia-slam `feat/global-pose-graph` at the gated fix tip, tag `pr180c/F1` (made in phase 5 of
  the plan; not yet cut). Binary: the validation run's binary from `pr180c/F1` or later, sha and md5 recorded
  at build.
- **Knob:** none (default = today, by construction). Class: instrumentation only.
- **Baselines:** the existing debug logs in `S/cardV/out` (R_03 W / V1 / ALL, R_11 ALL, R_01 full ALL) for
  exposure on the quick sequences; the validation run (the six sequences of
  `experiments/pr180-validation-handoff.md`, section "pr180-window10 run") for the decision.
- **Kill rules (same shape for every card):**
  - **K-none (mechanism not exposed):** the counter stays under the threshold on every validation sequence.
    The suspected gap is not exposed on this data: no knob, ledger row marked refuted for the validation set.
  - **K-exposed:** the threshold is passed on any validation sequence. Write a full card (with a
    "mechanism moved, score did not" and a "score moved, mechanism did not" kill rule) before any knob.
- **Bit-identity:** every counter is output only. `P/baseline/r01full-S10-7.ron` (copy of
  `S/cardR/cfg/r01full-S10-7.ron`), synchronous, `RAYON_NUM_THREADS=1`, must be `cmp`-identical to
  `P/baseline/traj.txt` (copy of `S/cardT/out/T1/r01full-S10-7-r1/traj.txt`,
  md5 d4f77b567b912b5a4caa2af6085a35b1).
- **Noise:** the counters are read on one validation run per sequence (paced, one job). A full card that
  follows sets its own repeats.

## Summary

| Card | Ledger | Mechanism | Counter | Build a knob if |
|---|---|---|---|---|
| S0-A4 | G37 | Rejected LM steps use the 7-iteration budget (`vi_ba_schur.rs:2596`). Basalt does the same; g2o does not | `vi_ba_rejected_steps`, `vi_ba_solves_3plus_rejects` | 3+ rejects in over 5 % of solves on any validation sequence (easy sequences: under 1 %) |
| S0-C11 | G38 | A flow pose is dropped when a keyframe is due and ORB then rejects (`mod.rs:1951-1977`) | `flow_due_orb_rejected` | over 1 % of frames on any sequence (quick sequences: 0) |
| S0-C1b | G39 | A lost keyframe is inserted at a runaway pose without `keep_inertial` (`mod.rs:2189`) | `coast_runaway_frames`, `lost_keyframes_at_runaway` | any value above 0 |
| S0-C2 | G40 | Scored coasts run on a stale velocity (`mod.rs:1760`) | none in code: `coasts.py` on `traj.txt` and `coasted.txt` gives run lengths and ATE with and without coasts | ATE with coasts worse than without by over 10 % on any sequence |
| S0-A1 | G41 | Offline 10 px drop before the first iteration (`global_vi_ba/mod.rs:238-241`) | `[offline_ba] observations=` | dropped share above 5 %. The knob would be a staged drop on the existing `tightened` path |
| S0-B7 | G42 | The baseline gate uses the camera-0 median depth for every rig camera (`growth.rs:142-144`) | debug line: per-camera median against the mixed median | they differ by over 10 % in over 10 % of keyframes on R_03 or R_11 |
| S0-D7 | G43 | Each rig-PnP outlier round restarts from the prediction; an unmoved pose counts as a visual solve (`rig_pnp.rs:79-98`) | `flow_pose_unmoved` (lands with card Z's lane) | over 1 % of flow frames |
| S0-E10 | G44 | Refill threshold relative to the empty cells only (`good_features.rs:106-116`) | `flow_refill_spawns` against spawns dead within 2 frames | over 50 % die within 2 frames |

## S0-A4: rejected LM steps count against the VI-BA iteration budget (R-A4)

- **Gap tested:** G37 (new, suspected).
- **Mechanism (ad443b7):** `K/mapping/vi_ba_schur.rs:2596` (`for _iter in 0..params.max_iterations`) counts a
  rejected LM step as an iteration (`:2660-2668`), so with the preset's 7-iteration budget a solve with
  several rejects stops with fewer accepted steps.
- **Theory and reference:** Basalt, the source of the preset's 7 / 1e-4, also counts rejected steps as
  iterations (`S/basalt-src/src/vi_estimator/sqrt_keypoint_vio.cpp:1157-1159`, `it++` at `:1388`); g2o does
  not. The review finding is therefore not a defect against the reference.
- **Exposure (`S/cardV/out`):** 2-6 of about 200 solves have a reject on R_03, 2-5 of about 170 on R_11, 5-21 of
  about 580 on R_01. Only one run (`r01full-ALL-r2`) has solves with 3 or more rejects (4).
- **Counter:** `vi_ba_rejected_steps`, `vi_ba_solves_3plus_rejects` (lane map: `rejected_steps` in the solve
  outcome and `MappingCounts`; wired to `RunCounters` and `[summary]` in phase 3).
- **Prediction:** easy sequences: 3+ rejects in under 1 % of solves.
- **Build a knob if:** 3+ rejects in over 5 % of solves on any validation sequence.

## S0-C11: a flow pose is dropped when a keyframe is due and ORB then rejects (R-C11)

- **Gap tested:** G38 (new, suspected).
- **Mechanism (ad443b7):** on `FlowOnly::KeyframeDue` the frame is resolved and ORB tracking runs from the flow
  pose (`K/system/mod.rs:1951-1977`); if ORB rejects, the flow pose is not kept (`:2017-2035`).
- **Theory:** a flow pose that passed its own checks is a valid track; a later ORB reject should not erase it.
  No reference cited; the full card names one if the threshold is passed.
- **Exposure:** in 9 runs (`S/cardV/out`, R_03 W / V1 / ALL, R_11 ALL, R_01 ALL) no frame has a
  `[klt-first] keyframe due` line followed by a `[proj]` reject.
- **Counter:** `flow_due_orb_rejected` (lane sys).
- **Prediction:** quick sequences: 0.
- **Build a knob if:** over 1 % of frames on any sequence.

## S0-C1b: a lost keyframe is inserted at a runaway pose (R-C1b)

- **Gap tested:** G39 (new, suspected).
- **Mechanism (ad443b7):** the runaway check that blocks a lost keyframe at a coasted pose runs only with
  `keep_inertial` (`K/system/mod.rs:2189`: `due && role.is_some() && self.recovery.keep_inertial &&
  self.runaway.runaway()`). Without it, a lost keyframe can be inserted at a runaway pose. Default-path
  estimator change, so counter first; the fix lane does not touch `mod.rs:2189`.
- **Theory:** a keyframe at a pose the detector calls a runaway seeds the map with a known-bad pose.
  No reference cited; the full card names one if the threshold is passed.
- **Counter:** `coast_runaway_frames`, `lost_keyframes_at_runaway` (lane sys; the detector call moves out of
  the `keep_inertial` block, `record_tilt` stays inside).
- **Build a knob if:** any value above 0.

## S0-C2: scored coasts run on a stale velocity (R-C2)

- **Gap tested:** G40 (new, suspected).
- **Mechanism (ad443b7):** while coasting, the frame velocity is the keyframe's (`K/system/mod.rs:1760`), and
  coasted poses written under `--trajectory-include-coasted` are scored.
- **Theory:** a coast integrated from a stale velocity drifts with the coast length; scoring it costs ATE that a
  dropped pose would not. Priced offline from the validation files, no code.
- **Counter:** none in code. `Q/coasts.py` (lane scripts) on `traj.txt` and `coasted.txt` gives run lengths
  and ATE with and without coasts.
- **Build a knob if:** ATE with coasts worse than without by over 10 % on any sequence.

## S0-A1: offline 10 px drop before the first iteration (R-A1)

- **Gap tested:** G41 (new, suspected).
- **Mechanism (ad443b7):** `K/mapping/global_vi_ba/mod.rs:238-241` drops every observation above
  `outlier_px` (10 px) before the first iteration (`problem.drop_outliers`), from the online start. On R_03
  today 3084 are dropped; the total is not printed.
- **Theory:** a drop before any iteration removes residuals that the solve could have brought in; a staged
  drop (after the first solve) keeps them. No reference cited; the full card names one if the threshold is
  passed.
- **Counter:** `[offline_ba] observations=` (lane offline, bucket 1), so the dropped share is visible.
- **Build a knob if:** dropped share above 5 %. The knob would be a staged drop on the existing `tightened`
  path.

## S0-B7: the baseline gate uses camera 0's median depth for every rig camera (R-B7)

- **Gap tested:** G42 (new, suspected).
- **Mechanism (ad443b7):** `K/mapping/growth.rs:142-144` takes `map.scene_median_depth(curr_kf_idx)` once and
  applies it to every rig camera's pairs. On the preset path (`growth_min_baseline_ratio: 0.0175`).
- **Theory:** the baseline-to-depth ratio gates parallax per camera pair; a camera looking at a nearer or
  farther scene gets the wrong gate. No reference cited; the full card names one if the threshold is passed.
- **Counter:** a debug line (lane map): per-camera median against the mixed median.
- **Build a knob if:** they differ by over 10 % in over 10 % of keyframes on R_03 or R_11.

## S0-D7: rig-PnP outlier rounds restart from the prediction (R-D7)

- **Gap tested:** G43 (new, suspected).
- **Mechanism (ad443b7):** `K/tracking/pose_estimation/rig_pnp.rs:79-98` refines each outlier round from
  `pose_init`, not from the previous round's pose, and an unmoved pose counts as a visual solve. On the preset
  path.
- **Theory:** each round should start from the last estimate so the inlier set and the pose converge
  together; a pose that never moves from the prediction is not visual evidence. No reference cited; the full
  card names one if the threshold is passed.
- **Counter:** `flow_pose_unmoved` (lands with card Z's lane, flow-z).
- **Build a knob if:** over 1 % of flow frames.

## S0-E10: refill threshold relative to the empty cells only (R-E10)

- **Gap tested:** G44 (new, suspected).
- **Mechanism (ad443b7):** `K/upstream/kornia_imgproc/good_features.rs:106-116` takes the corner threshold as
  `quality` times the strongest response in the unoccupied cells only, so on a refill in a weak region the
  threshold falls and weak corners spawn. On the preset path (refill).
- **Theory:** a threshold relative to the empty cells alone lets weak corners spawn where the image is weak,
  and weak spawns die quickly. No reference cited; the full card names one if the threshold is passed.
- **Counter:** `flow_refill_spawns` against spawns dead within 2 frames.
- **Build a knob if:** over 50 % die within 2 frames.

## Result log

