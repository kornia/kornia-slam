# EuRoC: why we do not beat Basalt, per sequence

Owner rule (2026-11-25): every sequence where an arm is not below Basalt gets a reason here, with evidence, or "unknown" plus the diagnosis that is running. It is updated at every benchmark result. It feeds the next review round.

Score: causal real-time SE(3) ATE (`traj.txt`, paced, asynchronous, `taskset -c 0,1`), except where marked SYNC or VIO-only. Basalt references: `B/results/basalt/<seq>/trajectory.txt` (blocking VIO, no loops). The goal is 0.5x Basalt on all 11 (`experiments/euroc-2x-plan.md`). Gap ids refer to `experiments/gap-ledger.md`.

## Open conditions that affect every real-time number (2026-11-25)

- **CPU held at 800 MHz** on the pinned cores, from at least 19:00. The battery reads 0 % and "Not charging" on AC. Measured under load on cpu1: 800 MHz. Throttle counts: package 24499, core 9492; 55-63 °C, so it is not thermal. In that state, MH_01 on the tip had 3675 of 3682 frames late and a lag of 99 s at the end of the run. Every real-time number taken in this state is labelled "800 MHz". The causal score suffers from the lag, so these runs are not a fair comparison with Basalt's blocking reference. The G11 latency numbers (p99 1-25 s) were probably partly this, not only load.
- **G57, oversubscription.** The real-time arms are behind the sensor even unthrottled: G11 RT3E has latency p99 of 1-25 s, and no run met p99 ≤ 100 ms. A pose computed late has drifted further.
- **G63, numeric sensitivity.** MH_04 moves 0.06-0.09 m per start from a 1e-8 rounding change. A difference smaller than that is noise unless it holds across starts.

## Final tip bench (95bc33c, RT3EF, real time, 3 reps, 2026-11-25/26)

Binary `bin-fold` (95bc33c; a4a2f4b only adds an off-by-default knob). Rep 1 ran at 0.8-1.2 GHz (lag 47-207 s at the end of the run). Reps 2 and 3 ran at the normal clock (lag 0-16 s, except V2_02 at 9-10 s on every rep). The median is over the 3 reps.

| seq | Basalt | 0.5x target | reps | median | ratio | smoothed | status |
|---|---|---|---|---|---|---|---|
| MH_01 | 0.067 | 0.034 | 0.043, 0.058, 0.042 | 0.043 | 0.64 | 0.028 | below Basalt |
| MH_02 | 0.048 | 0.024 | 0.040, 0.036, 0.033 | 0.036 | 0.75 | 0.021 | below Basalt |
| MH_03 | 0.062 | 0.031 | 0.050, 0.044, 0.054 | 0.050 | 0.81 | 0.033 | below Basalt |
| MH_04 | 0.114 | 0.057 | 0.210, 0.228, 0.356 | 0.228 | 2.00 | 0.137 | G58 (row below); r3 0.356 is the G63 fork |
| MH_05 | 0.144 | 0.072 | 0.103, 0.089, 0.094 | 0.094 | 0.65 | 0.054 | below Basalt |
| V1_01 | 0.043 | 0.022 | 0.047, 0.049, 0.043 | 0.047 | 1.09 | 0.043 | GT floor (row below) |
| V1_02 | 0.045 | 0.023 | 0.043, 0.057, 0.043 | 0.043 | 0.95 | 0.032 | below Basalt |
| V1_03 | 0.053 | 0.027 | 0.145, 0.123, 0.127 | 0.127 | 2.38 | 0.101 | G62 + jitter (row below) |
| V2_01 | 0.039 | 0.020 | 0.051, 0.111, 0.049 | 0.051 | 1.31 | 0.047 | G59 + floor; r2 0.111 is an outlier, cause not yet traced |
| V2_02 | 0.049 | 0.025 | 0.094, 0.108, 0.106 | 0.106 | 2.15 | 0.101 | G62 + early under-scale; also 9-10 s lag on every rep (G57) |
| V2_03 | 0.230 | 0.115 | 0.521, 0.577, 0.568 | 0.568 | 2.47 | 0.295 | unknown, 99-117 coasts per run; K1-c |

Result: 6 of 11 are below Basalt (MH_01, MH_02, MH_03, MH_05, V1_02, and V1_01 within 0.004). None is at 0.5x. The best ratio is 0.64x on MH_01. The `bin-integ2` medians (G11, all at the normal clock) agree within 0.01 on every sequence, except V1_03 (0.097 there).

## Pipeline, real time (G11, RT3E / RT3EF, `bin-integ2`, median of 3)

| seq | Basalt | ours (best RT arm) | ratio | why not below Basalt | evidence | status |
|---|---|---|---|---|---|---|
| MH_04 | 0.114 | 0.227 (RT3EF) | 1.99 | **Dark-span height step (G58).** The vertical increment over 63.3-66.7 s is −0.22 to −0.29 m (S0PEIH). Frames are visual-only PnP on a frozen map, so the IMU constrains nothing there. On the same span the new estimator (IMU on every frame) gives −0.03 m. Also G60 (loop PGO stopped at the cap) and the G63 noise. | `E/resid/cf.py MH_04_difficult … 63.3:66.7:v`; card "R13a"; `experiments/euroc-2x-results.md` 2x-5 P2 | explained; fix path = the sliding-window estimator in Drive |
| V1_01 | 0.043 | 0.045 (RT3E) | 1.05 | **Ground-truth floor plus lag.** The GT term is 0.037-0.039 m (three-system floor test), so every system is within 0.01 of the floor. All three fit the same lever arm of about (0.08-0.12, −0.04, −0.03) m. The 0.022 target (0.5x) is below the floor. | `E/twox/floor.py`, `floor_S0.json`, `lever_V.txt` | explained; target to be restated by the owner (stop S-A) |
| V1_02 | 0.045 | 0.041 (RT3E); 0.044 (RT3EF) | 0.91-0.98 | Below Basalt on RT3E. RT3EF is at 0.98, within run spread. | G11 table | below Basalt |
| V1_03 | 0.053 | 0.097 (RT3EF), 0.146 (RT3E) | 1.83-2.75 | **(a) PnpInliers duplicate landmarks (G62).** Dropping PnpInliers (RT3EIH) gives 0.053, equal to Basalt. V-c removes the duplicates but recovers only about 40 % (SYNC 0.111 → 0.094 against S0E 0.072). **(b) Frame jitter.** High-frequency error is 0.041-0.054 against Basalt's 0.004, because there is no gyro term in frame tracking (G13). The new estimator reaches 0.004. | `E/resid/v13v22/hf.py`; card "V-c"; G11 RT3EIH | explained; partial fixes (V-c, estimator) |
| V2_01 | 0.039 | 0.044 (RT3E) | 1.14 | **Propagation under mapping lag (G59), plus the GT floor.** The floor GT term is 0.031-0.035, which leaves only 0.004-0.008 above it. The rest is lag-dependent: SYNC S0E scores 0.046-0.055. | floor test; G59 row | partly explained; target to be restated (stop S-A) |
| V2_02 | 0.049 | 0.092 (RT3E) | 1.88 | **(a) PnpInliers duplicates (G62).** RT3EIH 0.082; V-c SYNC 0.092 → 0.086 against S0E 0.080. **(b) Early under-scale in the window at 26-28.5 s** (V-d trace). | card "V1_03 / V2_02 diagnosis", "V-d" | partly explained; the under-scale cause is open |
| V2_03 | 0.230 | 0.485 (RT3E) | 2.11 | **Unknown.** It has 99-173 coasts per run on every arm, under fast rotation and motion blur. The cause of the coasts is not isolated. Candidates: the 60 px per-frame flow movement cap (2.76 rad/s at f = 435), and visual-only frames. | G11 per-run counters | unknown; diagnosis K1-c (lane A) is tracing V2_03 |
| MH_01, MH_02, MH_03, MH_05 | 0.067, 0.048, 0.062, 0.144 | 0.051, 0.031, 0.055, 0.082 | 0.57-0.92 | below Basalt | G11 | below Basalt; none at 0.5x |

## New estimator, VIO only (2x-5 VO-ds, SYNC, `traj_vio.txt`, median over starts)

The predicted mechanisms hold: V1_03 high-frequency error is 0.004 (equal to Basalt), per-frame steps are 1.02-1.34x Basalt, and the MH_04 dark-span step is −0.03 m. The misses are slow drift. K1-a compared our states with Basalt's logged states. From 20 s on, biases, tilt and speed agree.

| seq | ratio | why not below Basalt | evidence | status |
|---|---|---|---|---|
| MH_02 | 2.5-3.0 | **Yaw drift:** +1.5° by 110 s against Basalt's about 0°. The right-camera rows fall to about 4 per frame over 10-20 s. | K1-a (`E/twox/k1/statediff.py`) | cause candidate: FEJ form (Pd1); A/B 1 `visual_jacobians: Current` is coded, waiting for its runs |
| V2_02 | 1.9 | **Yaw drift:** +1.2° against −0.4°. | K1-a | same as MH_02 |
| V1_02 | 1.76 | **Vertical drift:** +0.12 m at 70 s against +0.03 m. Accel bias z reads low (0.056-0.094 against GT 0.093; Basalt 0.075-0.087). | K1-a | open |
| MH_03 | 2.2 | Unknown. It has 72-82 landmarks per frame, so track count alone does not explain it. | 2x-5 counters | diagnosis K1-d queued |
| V1_01, V1_03, V2_01 | 1.2-2.0 | Not diagnosed per sequence yet. Same drift class assumed, not shown. | 2x-5 | open |
| MH_04, MH_05 | 1.08-1.13 | Close to Basalt. Not diagnosed. | 2x-5 | open |
| V2_03 | 4.0; diverged at start 30 | **Divergence with no reset:** the state stays finite, so the reset rule never fires. With Basalt's IMU noise values (VO-bn) it scores 0.287 (1.25x) against 0.921: the noise model matters here. | 2x-5, VO-bn record runs | diagnosis K1-c running |
| MH_01 | 0.75 | below Basalt | | |

## Speed against Basalt

`twox/speed_all.sh` on 95bc33c, 3 paced reps per sequence, 2 cores (`taskset -c 0,1`). Rep 1 was throttled for both systems, so the medians use mostly reps 2-3.

- **CPU per frame:** ours 48-57 ms on MH against Basalt's 74-80 ms (0.65-0.72x). On the Vicon rooms we are equal or slower: 62-71 ms against 58-74 ms.
- **Latency:** we are faster only on MH_02 (p50 44 vs 49 ms, p99 1355 vs 1719 ms). On every other sequence our p99 is higher, at 535 ms to 11 s against Basalt's 105-1270 ms. V2_02 p50 is 1.1 s, a growing backlog.
- **Verdict:** not faster than Basalt (1 of 11). The cause is G57. The per-frame mean cost is lower on MH, but mapping and loop work arrive in bursts. Tracking queues behind them on 2 cores, and the backlog never drains on the Vicon rooms. Fix path: bound the mapping work per keyframe, or move it off the tracking core; the estimator in Drive mode would also remove the PnP-on-map frames.

## Log

- 2026-11-25: file created from the G11 interim results, the 2x-5 partial results, K1-a, V-c and the floor test. Tip (95bc33c) RT3EF at 800 MHz, first runs: MH_01 0.0435 (0.65x), MH_02 0.0404 (0.84x); both below Basalt, with 82-99 s of lag at the end of the run.
- 2026-11-25: tip RT3EF rep 1 at 0.8-1.2 GHz, 8 of 11: MH_03 0.0500 (0.81x), MH_04 0.2101 (1.84x, G58 as logged), MH_05 0.1035 (0.72x), V1_01 0.0466 (1.08x, GT floor), V1_02 0.0432 (0.96x), V1_03 0.1449 (2.73x, G62 + jitter as logged). 0 give-ups; 2 coasts per Vicon-room run; 47-207 s of lag at the end of the run.
- 2026-11-26: final tip bench done (3 reps RT3EF plus speed pairs). 6 of 11 below Basalt, none at 0.5x; speed is not faster (1 of 11, G57). Tables above.
