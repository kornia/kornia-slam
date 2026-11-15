---
name: slam-experiment
description: Use before and after any SLAM/VIO change or experiment in kornia-slam (LaMAria, EuRoC, sim). Forces theory, numeric predictions and kill rules before code, scores results against them with a noise check, and keeps the gap ledger of where benchmark score is lost.
---

# SLAM experiment discipline

**Rule: write the theory and the predictions before any SLAM change.** A better number is not a fix. We do not yet know for certain where the benchmark gaps are, so every experiment must also say which gap it tests.

Where the files go (paths relative to the repo root unless your team agrees on another place):
- **Gap ledger:** `experiments/gap-ledger.md`. Create it if missing, in the format below.
- **One plan per change:** `experiments/<topic>-plan.md`.
- **Results:** `experiments/<topic>-results.md`.

## 1. Before code: the experiment card

Write this block into the plan file. Do not build or run anything until it exists and has been reviewed.

```
## Experiment: <name>
Gap tested: <ledger row id(s)>         Base: <repo>@<sha>   Binary tag: <tag>
Mechanism: <what is wrong, with file:line on the base sha>
Theory: <why the change fixes it; cite a reference implementation file:line if one exists>
Knob: <RON field, default = today's behaviour>
Baselines: <run dirs, with sha, binary hash, machine load>
Predictions (numbers, each tied to a measurement):
  P1 <metric> on <seq>: <baseline> -> <expected range>     measured by: <script/command>
  P2 ...
  Mechanism counter: <counter that must move if the theory is right>
Kill rules:
  K1 <condition> -> <what it means, what to do next>
Bit-identity: <which runs must stay cmp-identical with the knob off>
Noise: <repeats per config; known noise band for this metric>
```

Rules:
- **Predictions are numbers.** Each one is a number or range with its baseline, never a direction.
- **At least one prediction is on the mechanism** (a counter), not only on ATE or the benchmark score.
- **Two kill rules are required:**
  - one for "mechanism moved but score did not": the theory is incomplete;
  - one for "score moved but mechanism did not": luck, or another cause.
- **New behaviour goes behind a RON knob** whose default is today's behaviour. With the knob off, outputs must be byte-identical to the base. Prove it with `cmp` before trusting any knob-on number.
- **Best from each reference.** Name the reference system each mechanism comes from (ORB-SLAM3, Basalt, VINS-Fusion, OpenVINS, OKVIS2, ...) with file:line, and say why it beats the other references' choice.
- **Winners stack.** After a step passes, its winner becomes the base for the next experiment, and the stack is re-checked together.

## 2. Running

- **Record per run:** tag, commit sha, binary hash, config path, cores, machine load and run dir.
- **Load.** A run whose mean load differs from its baseline's by more than ±2 is not a valid timing sample.
- **Noise.**
  - Deterministic pipelines repeat bit-for-bit, so repeats do not measure noise. Perturb the input instead; for example, shift the IMU time offset by ±50 µs.
  - LaMAria trajectories diverge after the first behavioural difference, so one run per sequence is never enough to call a kill on the benchmark score.
  - Use at least 3 samples per config on the sequences that decide, or mark the verdict provisional.
- **Run within the machine's limits.** Prefer pricing an idea offline on existing runs before spending new runs.

## 3. After: score the card

Append this table to the results file:

| P/K | prediction | baseline | measured (each sample) | verdict |
|---|---|---|---|---|

- **Verdicts:** `as predicted`, `missed`, `kill fired`, `inside noise`.
- **Then one paragraph:** was the theory right, what is now known about the gap, and what comes next.
- **Never re-word a prediction after the run.** If a kill rule fires on its wording but the evidence says otherwise (for example, ground truth shows the "false" rejects were correct), record "fired, overruled", with the evidence. Do not edit the rule.

## 4. Gap ledger

One row per suspected source of lost benchmark score. Update it after every experiment.

| id | gap | evidence (run dirs) | sequences hit | est. score at stake | status | tested by |
|---|---|---|---|---|---|---|

- **Status:** `suspected`, `confirmed`, `refuted`, `fixed` or `partly fixed`.
- **Score at stake** must say how it was estimated: a reference run, arithmetic, a splice (our trajectory spliced into a reference trajectory and re-scored), or `unknown`.
- **Next experiment:** rank open rows by score at stake, and propose the next experiment from the top row whose cost is known.
