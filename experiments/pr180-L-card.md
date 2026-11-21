# Card L: IMU-carried poses go in the scored trajectory

Status: written 2026-11-21, before any code. This is roadmap item "IMU carries tracking through visual gaps" (2026-11-19).

## Experiment: L, one pose per frame through visual gaps

- **Gap tested:** coverage on the easy and medium windows. The owner's target is 100 % tracking. There is no ledger row yet; the docs step adds G34.
- **Base:** kornia-slam `feat/global-pose-graph` @ 60bfb1a, with card K's knobs on (`min_cold_start_points: 20`, `mapping.growth_min_baseline_ratio: 0.0175`).
- **Binary tag:** `cardL`.

### Evidence (2026-11-21)

- **Where card K's runs still lose frames.**
  - **The start:** the frame-0 map is reset at frames 2-6, from the exposure ramp, or on R_11 the view moving off the first map. A new map is seeded in the same frame, so no frame is lost.
  - **R_03, frames 371-387 (run KA-r3):** a turn of 6-8° per frame.
    - The stereo overlap drops to 0-9 points and map matches to 59-63.
    - Map projection then fails with `LowReferenceCorrespondences` or `LowPnpInliers`.
    - The IMU prediction carries the pose (`coast_inertial` 10-16 frames per run), and vision recovers at frame 388 with a 1.9° rotation error.
- **Those poses are not scored.**
  - The app writes them to a separate file, `--trajectory-coasted` (`apps/kornia-slam-app/src/main.rs:61-64`, "kept apart so the scored file is unchanged").
  - `--trajectory-out` holds only visually tracked frames, so `[summary] tracked` and the benchmark coverage both miss them.
- **Splice on one R_03 KA run, with `--trajectory-coasted`:**
  - scored file alone: 768/800 poses, ATE SE3 0.207 m, median tracked error 0.092 m;
  - merged with the 32 carried poses: 800/800, ATE SE3 0.206 m;
  - carried-pose error: median 0.129 m, max 0.318 m.

### Theory

- **The pose already exists.** In a tightly coupled VIO (Basalt, VINS-Fusion, OKVIS2), the IMU-propagated state is the output whenever vision is weak, so every frame gets a pose. This pipeline already computes that prediction and keeps the map through the gap: `coast_inertial` counts the frames, and the keyframes inserted at the inertial prediction keep the map growing. Only the output drops it.
- **Its accuracy is bounded by the gap length.**
  - With VIBA2's biases, IMU-only drift over under 1 s is centimetres. The measured median of 0.129 m is mostly the alignment error of the surrounding track, which is 0.092 m for tracked poses.
  - Long gaps would grow quadratically. A cap therefore keeps the output honest: past `max_coast` the existing give-up rule applies, and no pose is written.
- **Why the output, and not the tracker, first.** The frames are already carried. Changing the tracker (wider field of view in fast turns) is card F's claim, and its gain should be measured on the full trajectory, not on a file that drops the frames card F is meant to recover.

### Knob

- **App flag:** `--trajectory-include-coasted`, off by default. With it on, `--trajectory-out` also writes each frame carried by the IMU prediction, in time order and in the same format. Off, the file is byte-identical to today's.
- **Summary:** a `[summary]` counter, `poses_written`, is added. `tracked` keeps its meaning (visually tracked).
- **Where:** app only. No library change, unless the coasted pose is not available per frame through the existing coasted-trajectory path.

### Configs

| Name | Setup |
|---|---|
| L0 | card K's KA config (`pr180-frontend-1ms` plus both card K knobs), flag off |
| L1 | L0 plus `--trajectory-include-coasted` |

### Predictions (3 runs each, 800-frame windows)

- **P1, coverage:**
  - R_03: poses in `--trajectory-out` rise from 768-798 (L0) to 800/800 in 3/3 runs;
  - R_11 and R_01 stay at 798-800 and 800.
- **P2, accuracy:** ATE SE3 on each sequence within ±0.01 m of L0 in the same run. Both files come from one run, so this is a paired comparison.
- **P3, mechanism counter:** `poses_written − tracked` equals `coast_inertial` in every run.
- **P4, carried-pose error:** the median error of the carried poses, after the SE3 alignment on tracked poses, is ≤ 2× the median tracked-pose error (splice: 0.129 against 0.092 m).

### Kill rules

- **K1:** P1 holds, but P2 fails (ATE rises by more than 0.01 m). The carried poses are worse than the theory says, so the bias or velocity at the gap is wrong. Do not ship. Look at the velocity at the coast start, which is card I's young-map item.
- **K2:** P3 fails (the counts disagree). Some frames are written that the IMU did not carry, or the other way round. Fix the counting before trusting P1.
- **K3:** a run writes a pose past the coast cap. The cap is not applied to the output.

### Bit-identity and noise

- **Identity:** with the flag off, `--trajectory-out` is `cmp`-identical to the coasted-free file of the same run. The app is not deterministic run to run, so identity is checked within a run: the L1 file, minus the coasted rows, is `cmp`-identical to the L0 file written by the same process (a test writes both).
- **Noise:** 3 runs per sequence. ATE is compared paired, within one run.

### Result log

(empty)
