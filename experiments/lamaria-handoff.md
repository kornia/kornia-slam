# LaMAria sprint: handoff (2026-11-15)

This note is for whoever picks up the LaMAria accuracy work on this branch (PR #180). It says where the score stands, what we learned, why progress has been slow, and what we recommend doing next.

## Where we stand

| | score |
|---|---|
| Goal: rank #1 (Aria SLAM, test mean) | 75.9 |
| Top-3 line | > 66.7 |
| Our internal gate (5-sequence proxy mean) | ≥ 70 |
| ORB-SLAM3 mono-inertial, our harness, sequence R_11 | 72.9 |
| **Ours, proxy mean (5 sequences, a failed Sim3 counts 0)** | **3.6** |

- **Proxy sequences:** 3_18, 1_19, 2_11, 5_11 and R_11.
- **Sim3:** the fit passes on 3_18, 1_19 and 5_11 and fails on 2_11 and R_11.
- **Noise:** our pipeline is deterministic, so the noise is measured by shifting the IMU time offset by ±50 µs. One run per sequence is never enough to judge a change.

## The current leader: SLAM Dunk

SLAM Dunk, by Jacob Van Meter and Ayush Paul (code: https://github.com/jacobvm04/slam-dunk, AGPL-3.0), is at the top of the [leaderboard](https://www.lamaria.ethz.ch/leaderboard), level with Aria's SLAM. Its "+Retrieval" variant, an unofficial derivative, scores above Aria on Medium, Long and Low light. The findings below come from reading the repo on 2026-11-15; we have not run it.

| | Short | Medium | Long | Low light | Moving platform |
|---|---|---|---|---|---|
| Aria's SLAM | 90.7 | 78.5 | 70.9 | 84.2 | 55.0 |
| SLAM Dunk+Retrieval | 90.1 | 80.1 | 74.5 | 87.1 | 52.4 |
| SLAM Dunk | 89.6 | 78.9 | 71.1 | 85.4 | 52.3 |

**Features:** classical only; we found no learned features in the repo.
- **Frame-to-frame tracking:** Basalt's patch optical flow (vendored from Basalt), with detection at half resolution.
- **Keyframe features:** OpenCV SIFT turned into RootSIFT. They use 900 features per image and a very low contrast threshold (0.006). Keyframe pairs are matched with an essential-matrix RANSAC.
- **Loop closure (+Retrieval):** VLAD over the RootSIFT descriptors, with 48 words, power and L2 normalisation and IDF weighting. There is no vocabulary tree.
- **Low light:** the camera gain metadata is read, and dark-adapted SIFT is available as an option.

**What gives the score.** The README says it is the estimator, not the features:
- **Offline global bundle adjustment** on Basalt's square-root solver. By their account, this alone reaches about 69 on the training set, the level of strong public entries.
- **Two terms that standard VIO leaves out** take the training score from 69.1 to 82.4:
  - camera calibration that drifts over time (`temporal_camera`);
  - gyroscope intrinsic calibration (`gyro_response`).
- **Further model terms:** accelerometer, clock, thermal and a final refinement pass. Each one has an ablation config in `configs/`.
- **How it runs:** an offline pipeline (Python and C++, 8 cores, 8 GB) packaged as a Docker container. The leaderboard lists 62 FPS for the base entry.

**What this means for us.**
- A good classical frontend combined with offline global VI-BA (gap A7) and calibration states (A8) is enough to reach the top. Tuning the online estimator alone will not get there.
- AGPL-3.0 means we can study the ideas but must not copy code into kornia, which is Apache-licensed.
- Their 82.4 is a training-set number. Their test-set result is the leaderboard row above.

## What is on this branch

Every change sits behind a RON knob that defaults to the old behaviour. With the knobs off, the output is byte-identical to the base.

| change | knob | result |
|---|---|---|
| Rigid poses at the source (gap G20) | none, it is a fix | Non-rigid poses on R_11: 3365 before, 0 after. The coast blow-ups are gone. |
| Lost-mode keyframes as fixed anchors | `keyframes_while_lost: Anchor` | No gain on its own. |
| Keep gravity, bias and velocity through a tracking loss (G22, G25) | `tracking_loss.reseed.keep_inertial` | The 140 m world re-rotation is gone. Not adopted: the seed is fed by bad accepted poses. |
| Reject tracked poses that disagree with the gyro prediction (G21) | `tracking_loss.pose_gate` (rotation only, 5 deg) | It rejects the 1096 m bad pose on R_11, and on EuRoC V1_03 ATE goes from 0.2225 to 0.1799 m. The proxy shows no gain: without a re-seed, each reject becomes a cold restart. |

## Work in flight, not yet on this branch

Branch `exp/vi-window` (two commits, both behaviour-neutral with knobs off):
- the VI-BA window, nav chi2 and newest state are printed in debug builds only;
- the VI-BA iteration cap becomes the knob `local_ba.max_iterations`, default 50.

The window size is already a knob, `mapping.local_ba.active_keyframes`. It defaults to 3, while ORB-SLAM3 uses 10 or 25.

**Results so far (window 3 against window 10):**
- **R_11:**
  - tracked share goes from 67–78 % to 88 %;
  - 10 s scale within band goes from 46–63 % to 95–98 %;
  - splice cost of the drift block goes from 70–108 points to 8–64.
- **R_11 still fails:** the full run still fails Sim3, at a give-up near frame 7830.
- **EuRoC:** MH_01 −6.5 %, V1_03 −9.9 %, V2_02 −31.5 %, but **MH_03 is 35–59 % worse**, on all 3 samples.
- **MH_03 cause:**
  - What we measured: the gyro-bias error late in the sequence is 5–6x larger.
  - Likely mechanism: the past ties into the window only through the boundary IMU edge, whose bias rows are weighted 1e-2 (`mapping/vi_ba_schur.rs`, the boundary edge rows).
  - With 10 free keyframes, the bias soaks up rotation error.
- **Cost:** BA is about 2.9x more expensive at window 10. Capping iterations at 10 or 20 cuts the cost but loses too much of the gain.
- **Running now:** card B, which gives the boundary bias rows full weight, at windows 10 and 3. It is the direct test of the MH_03 mechanism.

## What we learned (gap list, by score at stake)

1. **Drift and scale while tracking (G6, G18):** the 3-keyframe window is too short. A longer window helps a lot, but the bias then needs a stronger tie to the past.
2. **Keyframes dragged away by BA (G29):** BA moves a keyframe to satisfy its IMU edge, then the next PnP snaps back.
   - Window landmarks move with the keyframe, so the snap-back must come from landmarks outside the window.
   - This is untested.
3. **Global PnP while lost (G19):** there is no relocalisation while lost, so every reject or give-up costs coverage.
4. **Re-seeded map locks onto the old map (G28):** a new 10-point seed tracks against the whole kept map, which gives 5–13 m pose jumps.
5. **Per-frame velocity and a pose-inertial solve per frame (G13):** today the IMU only seeds and vetoes the pose solve.
6. **Offline global visual-inertial BA (A7):** not built yet. Comparable systems gain from this.

**Ruled out:** KLT-first tracking patch A, anchors alone, the re-seed without the gate, and the initial scale offset (G23).

## Why progress has been slow

- One R_11 run takes 12–18 min, and a laptop runs 3 at a time.
- Each experiment card needs about 10–12 runs (±50 µs samples on two sequences, plus 4 EuRoC sequences), so about an hour before scoring starts.
- Cards ran one after another, each waiting for full scoring and a decision.
- We fixed one mechanism at a time. That will not close a gap of about 66 points to ORB-SLAM3 in reasonable time.

## Recommended way forward

0. **Consider offline global VI-BA with calibration states first.** SLAM Dunk (above) shows this is the main lever:
   - global BA over the whole recording;
   - camera calibration that drifts over time;
   - gyro intrinsics;
   - clock and thermal terms.

   This is gaps A7 and A8. It can run on top of our current online trajectory, so it does not wait on the online fixes below. Measure their ablation ladder (`configs/ladder_*.json`) on the training split before choosing the order.
1. **Port ORB-SLAM3's estimator settings in one step, then bisect.** It scores 72.9 in the same harness, so it is the closest working reference. Change everything we know differs, together:
   - VI-BA window of 10 (or 25), with ORB-SLAM3's handling of bias between consecutive keyframes and at the window boundary;
   - its robust kernels, applied only where ORB-SLAM3 applies them (it uses Huber only on the boundary edge, not on every IMU edge);
   - midpoint IMU integration instead of zero-order hold;
   - a pose-inertial solve per frame, with a velocity state;
   - its keyframe insertion and local-map selection rules.

   If the score jumps, bisect back to find which parts carry it. This takes a few large runs instead of dozens of small cards. It follows the rule "take the best from each reference": cite each mechanism's ORB-SLAM3 file and line.
2. **Screen, then confirm.**
   - Screen each candidate on one sample, using EuRoC (1–2 min per run) and the splice harness scored on the drift-block window only.
   - Run the 3-sample proof and the 5-sequence proxy only for candidates that pass the screen.
   - This should be about 3x faster per round.
3. **More compute if possible:** a second machine or cloud runners, so more than 3 LaMAria runs can go at once.
4. **Then the recovery side:** G19 global PnP while lost, and G28 a seed with its own local map. Recovery work only pays off once tracking drift is under control.

## Speed: the LK pyramid

These findings come from reading the code only; none of them is measured in the full pipeline.

**Already in place** (`tracking/optical_flow.rs`, `KltPyramidCache`):
- Each frame's pyramid is kept and reused as the next frame's previous pyramid. A byte compare guards against a stale pyramid. So each frame builds one pyramid, not two.
- The Scharr gradients are computed once per image, only when it becomes the previous frame.
- With an IMU prediction, LK starts at the predicted pixel and searches only the fine levels.

**Still open, largest first:**
1. **Make the u8 fixed-point kernel the default.** `LkKernel::FixedPointU8` (u8 pyramid, i16 Scharr, Q14 windows, OpenCV's scheme) already exists. The default is still `F32`, which every frame casts the full image u8→f32 and runs an f32 pyrdown and an f32 Scharr. The earlier frontend plan predicted about 1.2 ms to build the u8 pyramid. The output changes slightly, so this needs an experiment card with an accuracy check. It is the first one to try, because it is a single knob.
2. **Compute gradients only where tracks are.** Scharr runs over the whole image at every level. With about 300 tracks and a 21×21 window, that is about 10 % of the L0 pixels, so gradients computed lazily per window would save most of the cost.
3. **Build coarse levels only when needed.** With a good IMU prediction, LK uses only levels 0–1, but levels 2–3 are still built for the cache. The saving is small: those levels hold about 1/16 of L0's area.
4. **Stop building duplicate pyramids:**
   - The stereo path builds two extra resize pyramids that duplicate ORB's own pyramid (`frontend/mod.rs`).
   - The ORB pyramid (scale 1.2) and the LK pyramid (scale 2) cannot share levels. Running ORB only on keyframes, with KLT in between (the KLT-first design), works on mono but lost tracked share on the rig.
5. **Rig cameras:** KLT runs only on the primary camera today. Adding the other cameras adds pyramids, which makes 1 and 2 matter more.

**Priority:** low for the LaMAria score. The median tracking time per frame on R_11 is 8.4–8.6 ms, while VI-BA costs 85–250 ms per solve. This work matters for the Orin and for the rig.

## How we work

- Follow the `slam-experiment` skill (`.claude/skills/slam-experiment/SKILL.md`):
  - write the theory, numeric predictions and kill rules before any code;
  - put every new behaviour behind a knob that defaults to today's behaviour;
  - prove byte-identity with `cmp` when the knob is off.
- All LaMAria development goes into this branch (PR #180). Commits can be grouped later to trace issues.
- No GPS, no learned features, and no test-set submission without the owner's go.

## Gaps in this handoff

- **The scoring harness is not in the repo yet.** That means the splice harness (our trajectory spliced into the ORB-SLAM3 trajectory and scored with the official evaluator), the window and series readers, and the proxy runner.
- **The detailed plans and results files** for each experiment also live outside the repo.
- Ask the owner for them, or move them into `experiments/` as the first step.
