# EuRoC: where kornia-slam stands against Basalt (2026-11-26)

**Goal (owner).** Causal real-time SE(3) ATE ≤ 0.5x Basalt on all 11 EuRoC sequences, and faster than Basalt on 2 cores. One configuration per dataset.

**Protocol.**
- `traj.txt`: each pose is written once, when its frame is tracked, and never revised.
- Paced at sensor rate on `taskset -c 0,1`.
- Scored against GT with SE(3) alignment.
- The Basalt reference is its blocking VIO run, without loop closing.

## Configuration

`configs/euroc-stereo-imu-loop.ron`:
- local mapping asynchronous, with PnP-inlier keyframe observations;
- local BA at 7 iterations from λ = 1e-4;
- `PropagateToNewer` merge and χ² outlier erase;
- loop closing on;
- fixed-point rectification.

## Accuracy (m; median of 3 paced runs unless marked)

| seq | Basalt | ours | ratio | with asynchronous loop closing | ratio |
|---|---|---|---|---|---|
| MH_01 | 0.067 | 0.043 | 0.64 | 0.048 (1 run) | 0.72 |
| MH_02 | 0.048 | 0.036 | 0.75 | 0.035 (1 run) | 0.73 |
| MH_03 | 0.062 | 0.050 | 0.81 | 0.043 | 0.69 |
| MH_04 | 0.114 | 0.228 | 2.00 | 0.212 (1 run) | 1.86 |
| MH_05 | 0.144 | 0.094 | 0.65 | 0.095 (1 run) | 0.66 |
| V1_01 | 0.043 | 0.047 | 1.09 | 0.044 | 1.02 |
| V1_02 | 0.045 | 0.043 | 0.95 | 0.036 (1 run) | 0.80 |
| V1_03 | 0.053 | 0.127 | 2.38 | 0.115 | 2.17 |
| V2_01 | 0.039 | 0.051 | 1.31 | 0.051 (1 run) | 1.31 |
| V2_02 | 0.049 | 0.106 | 2.15 | 0.102 | 2.08 |
| V2_03 | 0.230 | 0.568 | 2.47 | 0.499 (1 run) | 2.17 |

- **Below Basalt beyond run spread:** 4 of 11 (MH_01, MH_02, MH_03, MH_05). None is at 0.5x.
- **The 0.5x target is below the GT floor on V1_01 and V2_01.** All three systems carry a 1.0-1.6 % scale error against Vicon there. That alone costs 0.019-0.023 m, the size of the target.
- **The target is reachable on V1_02 and V1_03 in principle:** ORB-SLAM3 (non-causal, full BA) scores 0.019 and 0.026 there.

## Why not below Basalt

| seq | cause | status |
|---|---|---|
| MH_04 | A height step in a dark span: frames there are visual-only PnP on a frozen map, so the IMU constrains nothing. One rep also failed its loop solve (PGO not converged after 30 iterations) | located; pose-inertial tracking halves the step in SYNC |
| V1_03, V2_02 | Frame-to-frame jitter: high-frequency error 0.041 against Basalt's 0.004, with no gyro term in frame tracking. PnP-inlier keyframe observations add duplicates. V2_02 also has an early under-scale in the window | partly explained; pose-inertial tracking is the next test |
| V2_01 | GT floor; one outlier rep is a 3 m excursion over 5 coast frames after a loop correction | floor; excursion under diagnosis |
| V2_03 | cam0 drops frames: 414 of 1920 intervals are 100 ms. 34 of 35 coast runs start right after such a gap, where matching gets too few inputs | located; the next card scales the match search with the predicted motion |
| V1_01 | GT floor (0.037-0.039 m) | target to be restated |

## Speed (same clock, 3 paced runs, median p50 / p99 latency in ms, CPU ms per frame)

| seq | ours, asynchronous loop closing | Basalt paced |
|---|---|---|
| V2_02 | 43 / 244 / 58.7 | 38 / 539 / 58.7 |
| V1_03 | 36 / 244 / 55.7 | 35 / 174 / 53.3 |
| V1_01 | 40 / 910 / 58.4 | 39 / 278 / 65.9 |
| MH_03 | 27 / 158 / 47.9 | 41 / 348 / 69.0 |

- **Asynchronous loop closing removes every loop stall:** no frame of 300 ms or more, against 2-14 per run before.
- **We are faster than Basalt on MH_03 only.** The rest of the tail comes from keyframe-dense bursts: in those spans 26-35 % of frames are keyframes, at 82-161 ms each.
- **The protocol is not yet fair.** The cores are not shielded, and image decode is inside Basalt's latency but outside ours. No p99 verdict holds until both are fixed.

## Next

1. **Pose-inertial tracking** with the current configuration (V1_03, V2_02, MH_04).
2. **A V2_03 match search** that scales with the predicted motion on frame gaps.
3. **Fewer wasted ORB frames.** 14-29 % of frames run ORB without becoming keyframes. Cutting them saves 3-11 ms CPU per frame.
4. **Two more speed changes:**
   - growth on the mapping thread, with a short apply lock;
   - loop recognition on the loop worker.
5. **A fair speed protocol**, then one batch gate on all 11 sequences.
