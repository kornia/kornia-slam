# Card T: the local VI-BA dense solve runs on every core, and it pays for it under load

Status: written 2026-11-21, before any run. Phase 1 is env only (no code change). Phase 2 is card R's
step 8 scoring with the phase 1 winner.

## Experiment: T, a sequential dense solve for the 10-keyframe window

- **Gap tested:** G35 (scale drift on full sequences), the real-time side. Card R found S10-7 (sync,
  10 active, 7 iterations, λ 1e-4) at ATE 0.384 m, but at 405-502 ms per solve under load.
- **Base:** kornia-slam `feat/global-pose-graph` @ e3c9b90. Binary `cardR/bin/kornia-slam-r-5a`
  (md5 22ef5566cdcf170ac14d967e9f5e15a3, built at 2fad071; e3c9b90 adds docs only).
- **Baselines:**
  - `cardR/out-5a-time/r01full-Q0a-off-r1`: Q0a config (20 iterations), single job, idle (load p50 2.7).
    Solve p50 170 ms, per iteration 8.44 ms p50, chol 30.3 % (about 3.0 ms per iteration).
  - `cardR/out-prof/...` (loaded, load p50 15.1): per iteration 55.1 ms, chol 71.7 % (39 ms).
  - `cardR/out/r01full-S10-7-r{1,2,3}`: ATE SE3 0.384 m ×3, scale 0.971, tracked 2898; 2 jobs, load ~18.

### Mechanism (file:line on e3c9b90)

- `vi_ba_schur.rs:2655` factors the reduced system with `m_mat.llt(faer::Side::Lower)`. That call
  uses faer's global parallelism, whose default is `Par::rayon(0)`, every core
  (faer-0.22.6 `src/lib.rs:1108-1119`).
- Only the offline global VI-BA sets `faer::set_global_parallelism(faer::Par::Seq)`
  (`global_vi_ba/mod.rs:222`), and only when it runs, at the end.
- The reduced system is 150×150 (10 free keyframes × 15). Its LLT is about 1.1 MFLOP, about 0.1 ms on
  one core. Splitting it over 12 rayon workers costs more in hand-off than it saves. When other
  processes hold the cores, each hand-off waits for a scheduler slot: 3 ms idle becomes 39 ms loaded.
- faer is the only rayon user in the app (`flow_pool.rs:2` uses its own pinned threads), so
  `RAYON_NUM_THREADS=1` makes faer sequential without touching anything else.

### Theory and reference

- Small dense systems in a real-time loop are factored on the calling thread. Basalt solves its
  local BA reduced system with Eigen `LDLT` on the calling thread (`sqrt_ba_base`, no parallel
  factorisation). ORB-SLAM3 runs g2o's dense/Cholmod solver single-threaded in the local mapping
  thread. Both beat a thread pool here because the matrix is too small to amortise the hand-off.
- The fix is also what our own offline path already does.

### Knob

- Phase 1: environment `RAYON_NUM_THREADS=1`, no code change.
- If phase 1 holds, a later code change makes the local VI-BA pass `Par::Seq` explicitly (behind
  `mapping.local_ba.dense_parallelism: Global | Sequential`, default `Global` = today).
  That change gets its own plan.

### Configs

| Name | Config | Env |
|---|---|---|
| T0 | `cardR/cfg/r01full-S10-7.ron` | default |
| T1 | same | `RAYON_NUM_THREADS=1` |

Timing runs are single job on an idle laptop (load p50 ≤ 3, recorded). T0 ×1, T1 ×1 single job,
then T1 ×2 more at 2 jobs for ATE noise.

### Predictions

- **P1 (mechanism), chol per iteration, T1 single job:** about 3.0 ms (T0 or the Q0a idle baseline)
  to ≤ 0.5 ms. Measured by `cardR/vitime.py` (chol share × per-iteration time).
- **P2, solve p50, T0 single job idle:** 55-75 ms (7.5 iterations × 8.4 ms).
- **P3, solve p50, T1 single job idle:** ≤ 60 ms, and ≤ T0 − 10 ms. Card R's P5 (≤ 100 ms p50) met.
- **P4, T1 at 2 jobs:** chol per iteration ≤ 1 ms. The sequential solve does not inflate with load.
- **P5, frame p90, T1 single job:** 670-687 ms (S10-7, loaded) to ≤ 150 ms.
- **P6, accuracy, T1 ×3:** ATE SE3 0.33-0.45 m, scale 0.95-1.00, tracked 2898. A parallel LLT sums
  in another order, so the trajectory need not be `cmp`-identical to T0.

### Kill rules

- **K1:** P1 holds, P3 misses (solve p50 > 100 ms). The cost is in schur and trial, so build card R's
  step 5b.
- **K2:** P1 misses (chol ≥ 2 ms per iteration sequential). It is not faer threading. Profile with
  `perf` before any change.
- **K3:** any T1 run above 0.6 m ATE. The factorisation order matters numerically; look at the
  damping and conditioning before shipping the knob.

### Bit-identity and noise

- T1 runs must be `cmp`-identical to each other (sync path, one thread).
- Timing: one single-job run per config, load recorded; a sample with load p50 > 3 is not a gate.

## Phase 2: card R step 8 with the phase 1 winner

Run only if phase 1 has no kill. Same binary and env.

| Name | Config | Runs |
|---|---|---|
| T-mono | S10-7 with `Mono` input (from `mono/cfg/r01full-mono.ron`) | 3 |
| T-am | A10-7 plus `merge: PropagateToNewer` (async, the real-time candidate) | 3 |
| T-r03, T-r11 | S10-7 on the 800-frame windows | 3 each |

Predictions (card R's, unchanged where they apply):

- **P7, mono (card R P4):** ATE SE3 ≤ 1.5 m (Q0a-mono 4.95 m). Unverified: S10-7 helped stereo 1.5×
  over Q0a, so the likely result is 3-4 m and a miss. The prediction stays as card R wrote it.
- **P8, T-am:** ATE SE3 ≤ 0.6 m in 3/3 runs, solves moving > 30 cm ≤ 5 (A10-7 had 12/4/1 and
  ATE 2.24/0.36/0.28 m; merge only had 13-24 at 20 iterations).
- **P9, R_03 and R_11 (card R K5):** tracked within card K's band (R_03 784-798, R_11 798-800 of 800);
  ATE SE3 ≤ card K's (R_03 0.194-0.202 m, R_11 0.136-0.170 m).

Kill rules:

- **K4:** P9 tracked below card K's band. 10 active keyframes do not hold on the short windows; keep
  3 active there.
- **K5:** P8 holds on ATE but solves > 30 cm stays above 5. Luck; do not pick async.

## Result log

- **2026-11-21, phase 1.** Binary `kornia-slam-r-5a` (md5 22ef5566cdcf170ac14d967e9f5e15a3), config
  `cardR/cfg/r01full-S10-7.ron`, full R_01. Run dirs `cardT/out/{T0,T1,T1n}`. Load average (1 min)
  0.9-4.5 over the whole phase; p50 3.2 for T0. Timing from `cardR/vitime.py`.

| Run | Jobs | Solve p50 / p90 / mean | Per iteration p50 | chol share (ms per iteration) | Frame p50 / p90 ms | ATE SE3, scale |
|---|---|---|---|---|---|---|
| T0, default threads | 1 | 52.2 / 129.8 / 67.6 ms | 7.46 ms | 35.5 % (3.39) | 10.9 / 115.4 | 0.384 m, 0.971 |
| T1, `RAYON_NUM_THREADS=1` | 1 | 38.0 / 60.5 / 41.3 ms | 5.43 ms | 2.1 % (0.12) | 4.6 / 87.4 | 0.384 m, 0.971 |
| T1n r1, r2 | 2 | 50.0, 50.6 / 90.0, 88.7 ms | 7.16, 7.28 ms | 2.0, 2.1 % (0.16) | 5.2 / 114.8, 113.4 | 0.384 m, 0.971 |

All four trajectories are `cmp`-identical, to each other and to `cardR/out/r01full-S10-7-r1`.

| P/K | Prediction | Baseline | Measured | Verdict |
|---|---|---|---|---|
| P1 chol per iteration, T1 | ≤ 0.5 ms | 3.0 ms (Q0a idle); 3.39 ms (T0) | 0.12 ms | as predicted |
| P2 solve p50, T0 | 55-75 ms | - | 52.2 ms | missed (below the band) |
| P3 solve p50, T1 | ≤ 60 ms and ≤ T0 − 10 ms | 52.2 ms | 38.0 ms | as predicted |
| P4 chol per iteration, T1 at 2 jobs | ≤ 1 ms | 39 ms (loaded, parallel) | 0.16 ms | as predicted |
| P5 frame p90, T1 | ≤ 150 ms | 670-687 ms (S10-7, loaded) | 87.4 ms | as predicted |
| P6 ATE, T1 ×3 | 0.33-0.45 m, scale 0.95-1.00, tracked 2898 | 0.384 m | 0.384 m ×3, 0.971, 2898 | as predicted |
| K1-K3 | | | | none fired |

- **Reading.** faer's default rayon split was the whole `chol` cost: 3.4 ms of a 7.5 ms iteration
  idle, and 39 ms loaded in card R. On one thread the factorisation is 0.12 ms, and the result is
  bit-identical, so the parallel path never changed the arithmetic. S10-7 now solves in 38 ms p50,
  inside card R's P5 target (≤ 100 ms), without card R's step 5b. Schur is now 70 % of the
  iteration (3.8 ms), the next cost if one is needed.
- **Not tested:** a truly loaded laptop (P4 ran at load 2-4.5, caused by the two runs themselves).
- **Next:** phase 2 with `RAYON_NUM_THREADS=1`. The code knob (`dense_parallelism`) gets its own plan.

- **2026-11-21, phase 2.** Same binary, `RAYON_NUM_THREADS=1`, 2 jobs, load p50 2.9 (max 4.4). Configs
  `cardT/cfg/{r03-T,r11-T,r01full-Tam,r01full-Tmono}.ron`; run dirs `cardT/out2`. R_03 and R_11 are
  800-frame windows; R_01 is the full 2898 frames. Synchronous runs are identical in 3/3.

| Config | ATE SE3 (each run) | Scale | Tracked | Coasted | Solves > 30 cm | Solve p50 |
|---|---|---|---|---|---|---|
| r03-T (S10-7) | 0.110 m ×3 | 1.034 | 776 | 24 | - | 52.6 ms |
| r11-T (S10-7) | 0.145 m ×3 | 0.961 | 800 | 0 | - | - |
| Tam (A10-7 + merge, async) | 0.398, 0.462, 0.404 m | 0.967-0.982 | 2898 ×3 | 0 | 0, 0, 1 | 79.4 ms |
| Tmono (S10-7, Mono) | 0.760 m ×3 | 1.048 | 2796 | 68 | - | - |

| P/K | Prediction | Baseline | Measured | Verdict |
|---|---|---|---|---|
| P7 mono | ATE SE3 ≤ 1.5 m | Q0a-mono 4.95 m; base mono 3.54-4.39 m | 0.760 m ×3, scale 1.048 | as predicted (the "likely miss" note was wrong) |
| P8 Tam | ATE ≤ 0.6 m 3/3, solves > 30 cm ≤ 5 | A10-7 2.24/0.36/0.28 m, 12/4/1 | 0.398/0.462/0.404 m, 0/0/1 | as predicted |
| P9 R_03 | tracked 784-798, ATE ≤ 0.194-0.202 m | card K | 776, 0.110 m | ATE as predicted; tracked missed |
| P9 R_11 | tracked 798-800, ATE ≤ 0.136-0.170 m | card K | 800, 0.145 m | as predicted |
| K4 | R_03 tracked below card K's band | 784-798 | 776 | kill fired |
| K5 | Tam ATE holds but > 5 large solves | - | 0-1 | not fired |

- **Reading.** Ten active keyframes, 7 iterations and λ 1e-4 fix the drift on every input tested:
  R_01 stereo 1.11-1.23 m to 0.384 m, R_01 mono 3.5-4.4 m to 0.760 m, R_03 0.19-0.20 m to 0.110 m.
  The async version with the merge knob holds too (0.40-0.46 m, 0-1 jumps), at 79 ms p50 per solve,
  so it is a real-time candidate.
- **K4 fired on R_03 tracking:** 776 visually tracked against 784-798, with 24 frames carried by the
  IMU, so `--trajectory-include-coasted` (card L) still writes 800. Per the kill rule, 10 active
  keyframes are not yet proven on short windows. The cause of the extra coasting is not looked at.
- **Next:** R_03 coasting (which frames, and why the 10-active window loses them), then the code knob
  for the sequential solve, then a preset candidate for the validation workflow.
