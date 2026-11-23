# EuRoC rig card: results (stop point S3)

Card: `experiments/euroc-rig-plan.md`. Branch feat/euroc-rig, binary md5 344317b26a368a528f5ab1bb9bf8be21.
Sweep 2026-11-23 15:09-23:02 (paused three times for top-1 rungs R3/G56, R11b, R11c; no run was cut).
68 runs, all rc=0, none at the 900 s cap. Run dirs `.pr180-fix/eurocrig/sync/<arm>_<seq>_s<start>_r1`;
report `sync/report.md` from `report_sync.py`.

Instrument (changed from the card after S2, see the card's "Decisions after S2"):
- every arm runs `mapping: ( execution: Synchronous )`, all rayon threads, unpaced, 900 s cap;
- ATE = SE(3) RMSE of body-frame `traj.txt` through `euroc-top1/score.py` (G55), not the `--evaluate` csv;
- noise = start_frame 0/1/2 on MH_01, MH_03, V1_02.
The card's absolute ATE predictions were written against the async, camera-centre baseline
(MH_01 sdef 0.083, sp48 0.097). Ratios between arms are the fair reading. Absolute ranges are scored
as written, with that caveat.

## Identity
- B2, synchronous: `sdef_*_s0` is byte-identical to top-1's R0 `S0_*_s0_r1` on all 11 sequences. Top-1's binary
  is the baseline binary (md5 cea7646298a68ddb948b16835472b2e3, code c21299b), so Stereo is unchanged.
- B1 (LaMAria R_01): identical (S2 report in the card).

## ATE SE(3), m, start_frame 0

| seq | sdef | rdef | sp48 | rp48 | rdef/sdef | rp48/sp48 | Basalt | ORB-SLAM3 |
|---|---|---|---|---|---|---|---|---|
| MH_01_easy | 0.090 | 0.061 | 0.120 | 0.067 | 0.68 | 0.56 | 0.067 | 0.046 |
| MH_02_easy | 0.057 | 0.069 | 0.037 | 0.084 | 1.21 | 2.28 | 0.048 | 0.034 |
| MH_03_medium | 0.073 | 0.071 | 0.076 | 0.045 | 0.97 | 0.59 | 0.062 | 0.028 |
| MH_04_difficult | 0.302 | 0.610 | 0.413 | 0.569 | 2.02 | 1.38 | 0.114 | 0.045 |
| MH_05_difficult | 0.174 | 0.243 | 0.187 | 0.136 | 1.39 | 0.73 | 0.144 | 0.050 |
| V1_01_easy | 0.090 | 0.057 | 0.088 | 0.077 | 0.63 | 0.88 | 0.043 | 0.039 |
| V1_02_medium | 0.102 | 0.061 | 0.041 | 0.067 | 0.60 | 1.63 | 0.045 | 0.019 |
| V1_03_difficult | 0.181 | 0.095 | 0.111 | 0.123 | 0.52 | 1.10 | 0.053 | 0.026 |
| V2_01_easy | 0.193 | 0.147 | 0.127 | 0.043 | 0.76 | 0.34 | 0.039 | 0.031 |
| V2_02_medium | 0.176 | 0.135 | 0.088 | 0.068 | 0.77 | 0.77 | 0.049 | 0.037 |
| V2_03_difficult | 0.605 | 0.216 | 0.592 | 0.123 | 0.36 | 0.21 | 0.230 | 0.027 |
| mean (n) | 0.186 (11) | 0.160 (11) | 0.171 (11) | 0.128 (11) | 0.90 | 0.95 | 0.081 | 0.035 |

Basalt / ORB-SLAM3: `bench/results/euroc_systems.json`.

## Noise (ATE at start_frame 0 / 1 / 2; spread = (max - min) / mean)

| seq | sdef | rdef | sp48 | rp48 |
|---|---|---|---|---|
| MH_01 | 0.090 0.086 0.094 (0.09) | 0.061 0.075 0.065 (0.21) | 0.120 0.073 0.080 (0.52) | 0.067 0.072 0.060 (0.17) |
| MH_03 | 0.073 0.062 0.071 (0.16) | 0.071 0.067 0.065 (0.09) | 0.076 0.050 0.064 (0.41) | 0.045 0.042 0.046 (0.09) |
| V1_02 | 0.102 0.102 0.102 (not measured) | 0.061 0.043 0.065 (0.39) | 0.041 0.106 0.107 (0.78) | 0.067 0.047 0.040 (0.52) |

- The band is wide: 9-21 % for the default arms on MH, 39-78 % for the 48-track arms and on V1_02. The card's
  "under 15 % is noise" guess was too tight. Most per-sequence differences below are inside noise.
- sdef on V1_02 is not perturbed: start 0/1/2 give byte-identical trajectories (md5 0665a842). The Stereo default
  bootstraps at the same later frame every time, so the 1-2 frame offset is absorbed. Its noise is unknown;
  another perturbation is needed (for example start_frame past the bootstrap frame).
- sp48 is the noisiest arm. Its single-camera 48-track gate passes on only 1-13 % of frames (G51), so which
  frames go flow-only flips between starts.

## Predictions

| id | prediction | result | verdict |
|---|---|---|---|
| P1-P5 | 400-frame mechanism and calibration | see the card (S2) | hold |
| P6 | rp48 MH_01 flow_unavailable <= 800, flow_frames 900-1600; every seq unavailable/frames <= 0.30; secondary/inliers 0.30-0.60 | MH_01 394 and 1303; unavailable/frames 0.07-0.21 on 9 seqs, 0.43 V1_03, 0.45 V2_03; secondary/inliers 0.45-0.49 on all 11 | holds except the ratio on V1_03 and V2_03 (fast motion: tracks die faster than refill) |
| P7 | rp48 MH_01 0.070-0.095; mean ratio rp48/sp48 0.75-1.00 | MH_01 0.067 (below the range; sp48 itself 0.120 at s0, 0.073-0.120 over starts); mean ratio 0.95, per-seq 0.21-2.28 | mean holds; MH_01 better than predicted; per-seq spread is mostly noise |
| P8 | rdef MH_01 0.070-0.100, MH_02 0.062-0.090, MH_03 0.050-0.075; mean ratio 0.85-1.15 | 0.061, 0.069, 0.071; mean ratio 0.90 (0.36-2.02 per seq) | MH_01 below (better), MH_02/03 hold, mean holds (neutral on average, not per sequence) |
| P9 | best arm mean >= 0.060; beats Basalt on <= 4 seqs; MH_02 >= 0.055 | best arm rp48 mean 0.128; beats Basalt on MH_03, MH_05, V2_03 (ties MH_01, 0.067 / 0.067); rp48 MH_02 0.084 | holds; no top-1 from this card |
| P10 | MH_01 Sim3 scale 0.99-1.01, rdef and rp48 | rdef 1.0072, rp48 0.9965 (sdef 1.0080) | holds. The 400-frame watch (1.03) did not persist on MH_01; G61 has the rig scale on MH_02-04 |
| P11 | MH_01: rp48 frontend p50 12-25, frame p50 <= 12; rdef frontend p50 18-35, wall 300-450 | rp48 9.6 / 10.0; rdef 11.2, wall 418 | frame p50 and rdef wall hold; both frontends faster than predicted. Synchronous mapping and shared load (see per-run table) make time a rough read |

## Kill rules

- K1 (mechanism moved, score did not): P6 mostly holds, and the mean ratio rp48/sp48 is 0.95 (<= 1.00).
  rp48 mean 0.128 against sdef 0.186. Does not fire.
- K2 (score moved, mechanism did not) **fires for rdef**. rdef has no flow-only frames at all
  (flow_frames 0, `klt_rig: PrimaryOnly` extracts ORB on both cameras every frame), yet beats sdef beyond the
  measured noise on V1_02 (0.043-0.065 against 0.102), V1_03 (0.095 / 0.181), V2_03 (0.216 / 0.605) and
  MH_01 (0.061-0.075 against 0.086-0.094). Mean ratio 0.90, not below 0.85.
  So part of the rig gain is the rig backend, not flow: raw images, ORB overlap depth, second-camera BA rows,
  ORB on both cameras every frame. The card's next step is to split it first: re-run MH_01 with
  loop_closing Disabled in both modes. Not run (stop at S3).
  For rp48 the mechanism moved (secondary/inliers 0.45-0.49, unavailable down 2-9x), so K2 does not apply.
- K3 (correctness): P4, P5, P10 hold; MH_01 tracked = frames on every arm. Does not fire.
- K4 (regression): rdef > 1.3 x sdef on MH_04 (2.02) and MH_05 (1.39) only, 2 of 11 (< 6). Does not fire.
  MH_04 is the worst rig sequence: rdef 0.610, rp48 0.569, scale 1.022 / 1.009 (G61's sequence list).
- K5 (cap): longest run rdef V2_02 618 s, under the 900 s cap. Does not fire. Under the card's original 300 s
  rule, 18 of 44 s0 runs would fire: rdef 8, rp48 7, sp48 2, sdef 1 (synchronous mapping, shared load). Rig mode runs 1.5-3x slower than Stereo (rdef
  frame p50 11-40 ms against sdef 2-12 ms).

## Reading (with reasoning)

- **Two-camera flow (G50) works as a mechanism and helps on average.** rp48 against sp48 gives 9-13x the flow-only frames on
  MH_01-03 and V1_01, and 2-9x fewer flow_unavailable frames. About 47 % of flow inliers come from camera 1. Mean ATE is 0.128 against 0.171,
  best on 7 of 11. The per-sequence ratio swings 0.21-2.28 inside a 41-78 % sp48 noise band, so no
  single-sequence claim holds. The gain fits the theory that more tracks behind a fixed gate (G51) and a
  second lever arm help. With the noise this wide, the size of the gain is not measured.
- **The rig backend helps on V1/V2 and hurts on MH_04/MH_05, without flow.** rdef against sdef: better on 8 of 11,
  worse on MH_02, MH_04, MH_05. This matches G61's diagnosis: the rig BA holds scale with half of Stereo's
  metric rows, so scale floats on the 3-keyframe window (rdef scale 1.010-1.022 on MH_02-04). On V1/V2,
  the rig tracks more frames (V2_03 1903 against 1832; V1_02 1710 against 1688), so its gain there is
  coverage and fewer bad bootstraps, not depth quality. That is a hypothesis: not split yet (K2 action).
- **Best arm rp48 mean 0.128 against Basalt 0.081 and ORB-SLAM3 0.035.** This card closes none of the top-1 gap.
- **Not decided here:** whether Stereo should gain right-camera flow (design note Option A). K1 did not fire, so the
  card does not rule it out; K2's split and G61's R-a rung (owner approval pending) come first.

## Next (owner)
1. K2 split: MH_01 with loop_closing Disabled, Stereo and Rig, before any claim about rig depth.
2. G61 rung R-a (rdef with `local_ba: ( active_keyframes: 10 )` on MH_02, MH_03): proposed, not run.
3. A perturbation that moves the sdef bootstrap on V1_02, so its noise is measured.
