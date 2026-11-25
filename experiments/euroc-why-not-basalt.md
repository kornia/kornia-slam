# EuRoC: why we do not beat Basalt, per sequence

Owner rule (2026-11-25): every sequence where an arm is not below Basalt gets a reason here, with evidence, or "unknown" plus the diagnosis that is running. It is updated at every benchmark result. It feeds the next review round.

Score: causal real-time SE(3) ATE (`traj.txt`, paced, asynchronous, `taskset -c 0,1`), except where marked SYNC or VIO-only. Basalt references: `B/results/basalt/<seq>/trajectory.txt` (blocking VIO, no loops). The goal is 0.5x Basalt on all 11 (`experiments/euroc-2x-plan.md`). Gap ids refer to `experiments/gap-ledger.md`.

## Open conditions that affect every real-time number (2026-11-25)

- **CPU held at 800 MHz** on the pinned cores, from at least 19:00. The battery reads 0 % and "Not charging" on AC. Measured under load on cpu1: 800 MHz. Throttle counts: package 24499, core 9492; 55-63 °C, so it is not thermal. In that state, MH_01 on the tip had 3675 of 3682 frames late and a lag of 99 s at the end of the run. Every real-time number taken in this state is labelled "800 MHz". The causal score suffers from the lag, so these runs are not a fair comparison with Basalt's blocking reference. The G11 latency numbers (p99 1-25 s) were probably partly this, not only load.
- **G57, oversubscription.** The real-time arms are behind the sensor even unthrottled: G11 RT3E has latency p99 of 1-25 s, and no run met p99 ≤ 100 ms. A pose computed late has drifted further.
- **G63, numeric sensitivity.** MH_04 moves 0.06-0.09 m per start from a 1e-8 rounding change. A difference smaller than that is noise unless it holds across starts.

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

Not measured. The 2x-1 runs on the tip started at 800 MHz. The one invalid sample (MH_01 at 800 MHz) gives 109 ms of CPU per frame for ours, against 183-199 ms for Basalt. Neither kept up with real time. No verdict until the clock is normal.

## Log

- 2026-11-25: file created from the G11 interim results, the 2x-5 partial results, K1-a, V-c and the floor test. Tip (95bc33c) RT3EF at 800 MHz, first runs: MH_01 0.0435 (0.65x), MH_02 0.0404 (0.84x); both below Basalt, with 82-99 s of lag at the end of the run.
