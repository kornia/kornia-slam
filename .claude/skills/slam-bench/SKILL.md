---
name: slam-bench
description: Run, score and compare the EuRoC stereo-inertial benchmark for kornia-slam, optionally against ORB-SLAM3. Use when asked to benchmark SLAM accuracy, check whether a change improved or regressed tracking, measure ATE/rotation drift/RPE on EuRoC, run ablations, or produce a benchmark report. Also use before and after any change to tracking, IMU, BA, or loop closure.
---

# SLAM benchmark

Answers one question repeatably: **did this change make kornia-slam better or worse?**

Every trajectory — kornia-slam's and ORB-SLAM3's alike — is scored through the same
evaluator rather than each system's own reporting, which is what makes the numbers
comparable at all.

## Before running

- **Dataset**: the directory holding the EuRoC sequence folders. Read from
  `KORNIA_BENCH_EUROC`, else `data/euroc` under the repo root:
  ```bash
  export KORNIA_BENCH_EUROC=/path/to/kornia-slam-datasets/euroc
  ```
- **Binary**: `cargo build --release -p kornia-slam-app`. Override with `KORNIA_BENCH_BIN`.
- **Vocabulary**: `extra/ORB_SLAM3/Vocabulary/ORBvoc.txt`, needed by any config with loop
  closure. Override with `KORNIA_BENCH_VOCAB`.
- **ORB-SLAM3** (optional, only for `--orb3`): a built `stereo_inertial_euroc`.
  Override with `KORNIA_BENCH_ORB3`.

Sequences run at roughly real time, so a full 11-sequence pass is ~40 min per repeat.
Start with one sequence while iterating.

## The loop

```bash
S=.claude/skills/slam-bench

# 1. Quick check on one sequence while iterating (~7 min)
python3 $S/scripts/bench.py run --out /tmp/bench-wip --seqs MH_01_easy
python3 $S/scripts/bench.py score /tmp/bench-wip \
    --baseline $S/baselines/2026-08-27-develop.json

# 2. Full sweep once the change looks right (~2 h with 3 repeats)
python3 $S/scripts/bench.py run --out /tmp/bench-full --runs 3
python3 $S/scripts/bench.py compare /tmp/bench-full \
    --baseline $S/baselines/2026-08-27-develop.json

# 3. Report with figures, and a PDF if Chrome is installed
python3 $S/scripts/report.py /tmp/bench-full \
    --baseline $S/baselines/2026-08-27-develop.json --pdf
```

`run` is resumable — a cell that already has a trajectory is skipped, so an interrupted
sweep continues where it stopped. `compare` exits non-zero when more sequences regressed
than improved, so it works as a gate.

## Reading the result

**The noise floor is ±19% of ATE.** Measured over three runs of both systems on all
eleven sequences. Both systems are non-deterministic; a 10% "improvement" from a single
run is nothing. Use `--runs 3` for any claim worth making, and treat differences under
19% as unchanged — `compare` already applies this.

Metrics, and what each is for:

| metric | what it catches |
|---|---|
| `ate_rmse` | overall accuracy. Rigid fit, deliberately **not** Sim(3) — a scale term would absorb trajectory-length error and hide it |
| `ate_rot_rmse` | attitude error, after removing the constant frame offset a camera-frame estimate carries against body-frame truth |
| `rpe_rot_rmse` | rotation drift over 1 s. Local error no global alignment can flatter |
| `path_ratio` | estimated trajectory length vs truth. Robust stand-in for scale error — Sim(3) scale is meaningless once a trajectory diverges |
| `coverage` | fraction of the ground-truth span the estimate covers |

If ATE moves, check the others to learn *why*: rotation drift up means the front end got
noisier; `path_ratio` up means noise is inflating the path; coverage down means tracking
is being lost.

## Ablations

`--configs si_loop si_noloop s_only mi_loop` attributes error to a subsystem instead of
guessing. `si_loop` is the default and the one baselines are keyed on.

| config | what it isolates |
|---|---|
| `si_loop` | stereo-inertial + loop closure — the shipping configuration |
| `si_noloop` | what loop closure is contributing |
| `s_only` | what the IMU is contributing |
| `mi_loop` | whether stereo is pulling its weight |

## Baselines

`baselines/2026-08-27-develop.json` is the reference: 84 runs on `develop` at
commit `6ec653a`, 11 sequences × 3 repeats plus ablations. Save a new one after a
change that legitimately shifts the numbers:

```bash
python3 $S/scripts/bench.py baseline /tmp/bench-full --save $S/baselines/<date>-<branch>.json
```

Keep old baselines — they are the record of what actually improved.

## What the reference baseline found

Context for interpreting a rerun, valid as of 2026-08-27:

- kornia-slam is ~2× ORB-SLAM3's ATE on the five easier sequences, ~12× on the harder six,
  at 2.6 cores against 8.1 and 0.38 GB against 1.00.
- **Most of the gap is per-frame attitude noise**: 0.41° at a 0.1 s interval against
  ORB-SLAM3's 0.029°. It *plateaus* with interval length (1.02° at 1 s, 1.09° at 5 s),
  which is the signature of noise rather than drift. Per-frame translation is only 1.4×
  worse. Rotation is the defect; the rest follows from it.
- Removing the IMU costs 2–22×, so inertial fusion is carrying the system.
- Loop closure gains 28–54% where the map is sound, and is neutral-to-harmful on V1_03
  and V2_03 where the estimate has already diverged.
- V1_03 and V2_03 (8.6° and 29.8° attitude error) have left the regime the others are in
  — treat them as a separate bug, not the tail of one distribution.
