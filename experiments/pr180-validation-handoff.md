# PR #180 validation handoff

For the separate validation workflow. The 2026-11-17 round rebased `feat/global-pose-graph` on main and built cards A to D plus the review swaps. It compiled with `cargo check -j2` only. No test, clippy run, identity gate, benchmark or review has run on it, and this file makes no score claim.

Plan: `/home/edgar/.claude/plans/show-me-the-previous-cosmic-meteor.md`. Cards: `experiments/pr180-{A,B,C,D}-card.md` and `experiments/pr180-review-swaps-plan.md`. Gap rows: `experiments/gap-ledger.md`. Background: `experiments/lamaria-handoff.md`.

## Tags

| tag | commit | what |
|---|---|---|
| `pr180/R` | `b59b0b9` | rebased on main `950aa17`, builds (`cargo check`), no review swaps, no new lanes |
| `pr180/S` | `42ee054` | `pr180/R` plus the review swaps (Appendix A Steps 2-7, 9); the knobs-off baseline |
| `pr180/K` | `a816b53` | cards commit `9e56edf` plus the skeleton |
| `pr180/L-C` | `dafe244` | lane C (analytic refine, PGO under `upstream/`) on K |
| `pr180/L-D` | `f6d5067` | lane D (session dump, offline BA) after C |
| `pr180/L-B` | `01730da` | lane B (recovery, rig) after D; includes the `fix(check)` duplicate-test fix |
| `pr180/F` | `cbf192b` | all four lanes (lane A last), knobs off |
| `pr180/F8` | `d82d4dc` | `pr180/F` plus Step 8 (SVD projection onto SO(3)); the branch tip when this file was written, before this docs commit |

Two things the tags do not show:
- `pr180/S` is not expected to `cmp` equal `pr180/R`: Steps 6 and 7 change the solve float order on every frame. `pr180/S` is the baseline for everything after it.
- Commits inside L-B (`0b7d399..99f8ea2`) and inside F (`a437f75..0fcbf28`) fail their test targets alone; the fix commits sit on top of each layer. Do not bisect through them with `cargo test`.
- Lane layers are cherry-picks, so each lane's commits are in order but not on separate branches (the `lane/*` branches and worktrees were removed).

## Verification list

Run in this order on `pr180/F8`, and on `pr180/F` if Step 8 is dropped. Laptop limits: `-j2`, at most 3 heavy jobs.

```bash
cargo fmt --all -- --check
cargo clippy --release -j2 --all-targets -- -D warnings
cargo clippy --release -j2 --manifest-path apps/kornia-slam-app/Cargo.toml --no-default-features --all-targets -- -D warnings
cargo test --release -j2 --workspace
RUSTDOCFLAGS="-D warnings" cargo doc -j2 --locked --no-deps -p kornia-slam -p kornia-sensors -p kornia-factors
python3 scripts/check_version_pins.py
grep -c -E "c2bcce1|11b4951" Cargo.toml Cargo.lock   # both 0
grep -c 4ada5a7554ffe4d373b39c611e55ddc47bcd3294 Cargo.toml   # 7
git log --merges --oneline origin/main..HEAD
```

Pass means:
- fmt and clippy are silent;
- every test binary prints `test result: ok` with 0 failed;
- `cargo doc` gives no warnings;
- the pins script prints `version pins OK (0.2.0)`;
- both greps give 0 and the second gives 7;
- the merges log is empty.

Known before the run:
- `rustfmt --check` reported diffs in `mapping/vi_ba_schur.rs` (near `:1537` and `:2010`) that were already there; run `cargo fmt --all` and commit it separately if it still fires.
- The only warning `cargo check` left was the `binrw v0.12.1` future-incompat note.
- Oracle tests added by the swaps and by Step 8 (`cauchy_weight_and_rho_match_the_f32_bodies`, the rig_pnp and pose_lm oracles, `svd_and_quaternion_projection_agree_within_the_input_defect`) have never been run; a failure there is a finding, not a flake.
- Not run locally: CI's `app-uvc` job and the MSRV 1.89 job.

## Identity gate

With every knob off, `pr180/F` must `cmp` equal `pr180/S` on the raw trajectory, body TUM and coasted files of:
- EuRoC MH_01;
- LaMAria 3_18, mono and rig.

Also required by the cards:
- card C: `pr180/L-C` equals `pr180/S` on MH_01 `euroc.ron`, 3_18 `lamaria-mono-imu.ron`, and MH_01 loop with `ScalarWeights` and with `Drift` (use `execution: Synchronous`; at least 1 accepted loop solve or mark P3 provisional);
- card B: B1 rig-calibration-file run on 3_18 rig equals the JSON run;
- card A: the G29 counter `ba_snapbacks` changes no output file.

Nothing below counts until the gate passes. If it fails, bisect with `cmp` by layer (`pr180/L-C`, `L-D`, `L-B`, `F`) and read the card's kill rule before reverting.

## Step 8 rule

Step 8 is the last commit (`d82d4dc`). Run the PGO fingerprint test:

```bash
cargo test --release -j2 -p kornia-slam optimize_pose_graph_output_is_bit_identical
```

If the fingerprints (`0xa4ac_a7f5_0777_b24e`, `0x934d_608c_260c_d0b9`) or any identity cmp change on `pr180/F8` while `pr180/F` is clean, drop `d82d4dc` with a rebase, and re-check. Its test oracle bounds are P1 3.3e-4, P2 1e-8, exact 1e-14 (card C line 107 and the commit message).

## Review dimensions

One review pass each, on the diff `pr180/R..pr180/F8`:
- correctness: conflict resolutions in `system/mod.rs`, `tracking_step` (A6 and B's `update_seeded_window`), the `fix(rebase)` commits, and the cards' open interactions (below);
- simplify: duplicated helpers, knobs declared and never read;
- upstream boundary: `grep -rn "crate::" crates/kornia-slam/src/upstream | grep -v "crate::upstream"` prints 0 lines, every `upstream/` file starts with `//! Upstream target:`, nalgebra only through `kornia_algebra` aliases.

Declared but possibly unwired, check each has a consumer (the skeleton listed them as unwired and lanes were to wire them): `large_window`, `pose_inertial`, `snapback_fix`, `relocalization.solver`, `max_jump_m`, `relocalize_when_lost`, `own_window_keyframes`, `sample_interpolation`, `KeyframeJob.tracked_inliers`, `LocalBaParams.recently_initialized`, `newest_marginal`.

Open points from the lanes:
- B4 with B5: after an adopted older-segment relocalisation, the lost segment's window can still filter out the older map (card B does not settle it).
- B3-P3: the RigAp3p part is measured at the hypothesis and rig-PnP stage, not after `estimate_pose`.
- `from_sensor_rig` on a non-rig source writes width and height 0.
- A7: the `nKFs<=2` and `nKFs<2` clauses of `NeedNewKeyFrame` are not ported.
- A9: the previous keyframe's covisibility stands in for the new keyframe's.
- D: SlamSystem stamps frames while recording; frames whose parent chain is gone are left out of the dump; the calibration border is 71 wide at 2 cameras and 8 slices (card said 55); depth rows use their own f64 chain.
- D: five choices left open for review in the card (knob location, no-evict meaning, body-frame states and 4-DoF gauge, numeric calibration Jacobians, IMU edges rebuilt from the raw stream).
- Revisit matches for the offline BA are a missing prerequisite (card D P11); the reference gain is not predicted for this build.

## Benchmark protocol

Record sha, binary md5, config and load for each run. Pipeline output is deterministic; the noise measure is a ±50 µs shift of the IMU time offset (shifted EuRoC copies as in `S/audit/window/euroc/`).

1. **Baselines:** `pr180/S` first, on EuRoC MH_01, MH_03, MH_04, V1_03, V2_02 and LaMAria R_11 plus the 5-sequence proxy (3_18, 1_19, 2_11, 5_11, R_11). If a baseline falls outside the WB1/RG1 band quoted in card A, read the card predictions as ratios to the new baseline and say so.
2. **Lane A:** as a bundle first (the A0 knob set in card A), then leave-one-out (arms in the card's table, screen first, 3 samples only for arms that move more than the noise band), plus the two single-sample arms A0-i10 and A0-orb. Card T and G29 are scored separately; `imu_trust` and `snapback_fix` are not in the bundle.
3. **Lane B:** identity gate, then the `--blackout` series on EuRoC MH_01 at `40:1`, `40:2`, `40:4` (L0 against L0 + B3, one sample plus ±50 µs); B3 on R_11 and 2_11 rig (3 samples each, solver split `PrimaryEpnp`, `PrimaryAp3p`, `RigAp3p` with `while_lost` on); B4 and B5 on R_11 and 2_11 reading the R_11 give-up window (6239-8241 on rg1; frames move with the base); then B3+B4+B5 as the lane bundle.
4. **Lane C:** the identity runs above, plus P7 timing on MH_01 (median within ±3 % of `pr180/S`).
5. **Lane D:** dump a session with `--session-out`, then `offline-ba` with every calibration state off (round trip, `max_iterations: 0`, P2), then states one at a time, then together; memory cap 4 GB on 3_18; scores via the splice harness and the proxy.
6. **Sequences:** EuRoC MH_01, MH_03, MH_04, V1_03, V2_02 (1 sample screen, 3 on MH_03); the 5-sequence LaMAria proxy with ±50 µs samples (a failed Sim3 counts 0); R_11 against ORB-SLAM3's R_11 72.9.
7. **Timing:** from solo runs only, load within ±2 of the baseline. R_11 runs take 12-18 min each, so at most 3 at once and at most 4 SLAM runs on the laptop.

The scoring harness (splice harness, series readers, proxy runner) is not in the repo; see "Gaps in this handoff" in `experiments/lamaria-handoff.md`.

## Predictions to score

Score each card's predictions as written. Never re-word a prediction after the run; if a kill rule fires on its wording but the evidence disagrees, record "fired, overruled" with the evidence.

| card | predictions | kill rules | where |
|---|---|---|---|
| A (A0 bundle) | P1-P8, M1-M5, C1; leave-one-out table; predicted contribution order A1 > A6 > A3 > A7 > A4 > A8 > A2 ≈ A9 ≈ A5 | K1-K6 | `pr180-A-card.md` "Experiment A0", A1-A9, T, G29 |
| B (recovery, rig) | B1-B5 predictions including B3-P11 to P15 | per experiment | `pr180-B-card.md` |
| C (solvers) | P1-P7, mechanism counter 0 | K1-K7 | `pr180-C-card.md` |
| D (offline BA) | P1-P15 | K1-K8 | `pr180-D-card.md` |
| swaps | P1-P8 | per card | `pr180-review-swaps-plan.md` |

Fill the table `P/K | prediction | baseline | measured (each sample) | verdict` per the `slam-experiment` skill, then one paragraph per card: was the theory right, what is now known about the gap, what is next. Update `experiments/gap-ledger.md` after each card.

Known theory flag: ORB-SLAM3 does not relocalise in inertial RECENTLY_LOST, so card B's K3 predicts the 0.5 m jump bound blocks much of G19.

## Prune rule

Plan decision 6: every new behaviour has a knob, off by default, and pruning happens after validation. Proposed rule, for the validator to confirm with the owner:
- **Adopted:** the card's predictions hold and no kill rule fires on at least 2 of 3 samples. The knob default flips (or the knob is removed and the behaviour made unconditional), the ROADMAP line becomes `[x]`, and the CHANGELOG line is rewritten.
- **Not adopted, kept:** inside the noise band, or a kill rule fires but the mechanism moved. The knob stays, ROADMAP reads "(built, knob `name`, not adopted: reason)".
- **Pruned:** a kill rule fires and the evidence agrees. Delete the code, the knob, its validation and its CHANGELOG line in one commit per mechanism.
- Moves with no knob (card C, Step 8) are accepted on the identity gate, not on a score.
- Nothing is pruned on a single sample, and a better number alone is not a fix (`slam-experiment` skill).

## Draft replies to Christian

Not written yet and never posted by this workflow. After validation, write them from results, per review comment:
- rebased on main and the swaps (what each swap did, the identity result);
- follow-ups built here behind knobs under `upstream/` (analytic pose refine and general PGO, IMU covariance trust, generic rig calibration, rig rays and generalised PnP), with the measured outcome;
- remaining items deferred.
