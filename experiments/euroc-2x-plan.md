# EuRoC 2x plan: 0.5x Basalt on all 11, faster than Basalt (owner-approved 2026-11-24)

Copy of the approved plan, ladder and decisions, for the repo. Results go to `experiments/euroc-2x-results.md`; the
ledger (`experiments/gap-ledger.md`) and the ROADMAP progress log are updated at each verdict. The master card for
the legacy stack stays `experiments/euroc-top1-card.md`.

Paths: `R` = `/home/edgar/dev/split/kornia-slam-graph`, `K` = `R/crates/kornia-slam/src`, `P` = `/home/edgar/dev/split/.pr180-fix`, `E` = `P/euroc-top1`, `B` = `/home/edgar/dev/.flux-connect-scratch/bench`, `BA` = `B/src/basalt`.

## Goal
Causal real-time ATE at most 0.5x Basalt on all 11 EuRoC sequences, and faster than Basalt, on 2 cores (paced, async, `taskset`, scored on `traj.txt`).
All work lands behind knobs on `feat/global-pose-graph` (PR #180); nothing is promoted before the gates.

## State found (read-only)
- `R` HEAD is 8a7fdde, 3 local commits ahead of origin (7fa9b92, 73873f4, 8a7fdde), not 1.
- `K/mapping/vi_ba_window_tests.rs` is dirty in the working tree. Stage files by path only; never `git add -A`.
- The G11 batch is still running under `P/rt.lock`: `E/gate/scores.jsonl` has 27 real-time runs (RT3E, RT3EF), 1-2 per sequence. V-a arms (S0EIH, RT3EIH) have not landed. I read these values; I did not score them against their predictions.
- Stock `basalt_vio` cannot run `vio_enforce_realtime`: `BA/src/vio.cpp:276-284` forces it to false, and `feed_images` (`:165-194`) pushes frames unpaced. The fair speed test needs a bench-side patch (section 5).
- A KLT round-trip check already exists (`K/tracking/flow_tracker.rs:443-454`), but only under `klt_kernel: WindowedU8` (`K/system/validation.rs:560-575`). EuRoC Stereo runs `FixedPointU8`, so it has never run there.
- Loop PGO: the default solve caps at 30 iterations (`K/loop_closure/pose_graph.rs:26-35`), and `unusable_reason` (`:59-92`) discards any solve that hit the cap, even one whose cost fell (`sparse_pgo.rs:354-361`).

## 1. Gap arithmetic

Target mean is 0.041 m. Reference bars on disk (`B/README.md:75-88`): ORB-SLAM3, non-causal with full BA, misses the target on 5 of 11 in our harness (MH_01, MH_02, V1_01, V2_01, V2_02) and on 6 of 11 by its paper. Our offline global BA (non-causal) misses it on 8. No system on disk meets all 11 targets even non-causally.

"Quad" = sqrt(best² − target²), the independent error that must be removed.

| seq | target | best causal today | G11 in flight, RT3E / RT3EF (unscored) | our offline ceiling | ORB-SLAM3 harness / paper | quad | what must remove it | verdict |
|---|---|---|---|---|---|---|---|---|
| MH_01 | 0.034 | 0.039 RT (SYNC 0.049) | 0.048-0.054 / 0.051-0.065 | 0.031 | 0.046 / 0.036 | 0.02-0.04 | pre-loop drift: S0PE causal 0.049 against smoothed 0.029. Map reuse in the window, then global stage | edge |
| MH_02 | 0.024 | not in the cards | 0.029-0.031 / 0.031-0.033 | 0.020 | 0.034 / 0.033 | 0.020 | same as MH_01 | within reach |
| MH_03 | 0.031 | 0.045 (rig, SYNC) | 0.057-0.058 / 0.057 | 0.027 | 0.028 / 0.035 | 0.048 | causal must reach the ceiling; keyframe pieces are already at Basalt parity (0.016 / 0.015) | edge, hard |
| MH_04 | 0.057 | 0.177 SYNC, 0.214 RT | 0.247 / 0.212 | 0.070 | 0.045 / 0.051 | 0.17-0.21 | dark-span step: 0.177 → 0.11-0.13 (counterfactual floor). G60: 0.05-0.07 on 1 of 5 starts. 0.12 → 0.057 has no identified mechanism | out of reach today |
| MH_05 | 0.072 | 0.136 (rig, SYNC) | 0.102 / 0.076 | 0.081 | 0.050 / 0.082 | 0.03-0.07 | MH_04's class; the ceiling itself must fall 11 % | edge |
| V1_01 | 0.022 | 0.057 (rig, SYNC) | 0.045 / 0.043 | 0.043 | 0.039 / 0.038 | 0.038 | none known. Basalt 0.043, ORB-SLAM3 0.038, ours offline 0.043 agree; Basalt's own tilt against GT is 1.69° | suspected GT floor |
| V1_02 | 0.023 | 0.035 SYNC, 0.043 RT | 0.041 / 0.045 | 0.024 | 0.019 / 0.014 | 0.026-0.036 | frame jitter and loop jumps (S0PEIH smoothed 0.025) | edge |
| V1_03 | 0.027 | 0.072 SYNC | 0.085 / 0.140 | 0.049 | 0.026 / 0.024 | 0.067 | HF 0.041 → Basalt's 0.004 gives 0.057. LF 0.057 → 0.026 needs the ceiling halved | out of reach today |
| V2_01 | 0.020 | 0.046 SYNC, 0.047 RT | 0.055 / 0.051 | 0.035 | 0.031 / 0.032 | 0.041 | none known below 0.031 | suspected GT floor |
| V2_02 | 0.025 | 0.080 SYNC, 0.083 RT | 0.096 / 0.098 | 0.042 | 0.037 / 0.014 | 0.076 | early event holds 64 % of E²: → 0.045-0.055. The rest needs 1.6x better than Basalt after 28 s | out of reach today |
| V2_03 | 0.115 | 0.123 (rig, SYNC) | 0.488 / 0.579, 102-119 coasts | 0.120 | 0.027 / 0.024 | 0.04 from the rig arm | coverage: two-camera flow plus IMU carry | edge |

Honest reading:
- **A Basalt-class estimator alone scores about 1.0x Basalt.** It is necessary (MH_04 step, V1_03 jitter, V2_03 coverage, speed) and not sufficient. Everything below 1.0x has to come from map reuse and a global stage delivered causally. That already shows: our map-tracking pipeline is at 0.58x on MH_01 and about 0.6x on MH_02.
- **My estimate after the whole ladder:** at most 1.0x Basalt on all 11 is likely; 0.5x on 4-7 sequences; 0.5x on all 11 is unlikely. V1_01 and V2_01 depend on the floor test (rung 2x-0c). MH_04, V1_03 and V2_02 need the ceiling itself to move, which only better observations do (two-camera sub-pixel rows, outlier policy).

## 2. Architecture changes, ranked by expected gain

| rank | change | expected gain (evidence) | reference | decision |
|---|---|---|---|---|
| 1 | (a) Fixed-lag VI smoother: every frame holds pose, velocity, bias; marginal prior with first-estimate Jacobians | V1_03 HF 0.041 → ≤ 0.008 (0.072 → about 0.057); MH_04 step −0.22..−0.29 → within ±0.10 m (0.177 → 0.11-0.13); V2_03 coasts 102-119 → 0; live pose no longer depends on mapping lag (G54, G57, G59); keyframe frames 68-95 ms → about 20 ms | `BA/src/vi_estimator/sqrt_keypoint_vio.cpp:306-330` (state per frame), `:369-374` (keyframe rule), `:552-685` (what is marginalized), `:860-907` (prior, `setLinTrue`), `:540` (nullspace check), `:1012-1062` (7 iterations, λ 1e-4); `BA/data/euroc_config.json` | **New module behind a knob.** See below |
| 2 | (d) Map reuse and background global stage, delivered through `map_from_odometry` | The only source of < 1.0x. Card estimate: 30-50 % of the offline gain reaches the causal pose (unmeasured). MH_01 0.049 → about 0.035-0.040 | ORB-SLAM3 `LoopClosing.cc:1206, 2268-2286` (global BA thread after a loop); ours `K/mapping/global_vi_ba/mod.rs:240, 314`, `K/loop_closure/graph/mod.rs:131-134`, `K/loop_closure/closer.rs:1010` | Measure by prefix replay first, then build |
| 3 | (b) Two-camera tracking in Stereo mode | V2_03 0.49-0.58 → about 0.12 (rig rp48 0.123); V1_03 HF → 0.015-0.020 (rig arms); mean 0.171 → 0.128 (rp48 against sp48, inside noise) | VINS-Fusion `feature_tracker.cpp:212-223`; OpenVINS `TrackKLT.cpp:202, 260-263`; Basalt `frame_to_frame_optical_flow.h:161-165` | **Build it as the estimator's frontend, not as Option A on the legacy PnP path.** The estimator takes right-camera pixel rows natively; the legacy path would need G61 partner linking first |
| 4 | (c) Outlier policy at the frontend: round trip ≤ 0.2 px plus stereo epipolar gate, before any pose exists | Keeps PnpInliers' MH gain (MH_04 0.311 → 0.227, MH_01 0.072 → 0.049) without its Vicon loss (LF +0.02..+0.04, duplicate seeds) | Basalt `frame_to_frame_optical_flow.h:180-228` (`max_recovered_dist2 0.04`), `:367-378` (epipolar 0.005); ours `K/tracking/flow_tracker.rs:443-454` | Reuse the existing round-trip code inside the estimator frontend. Legacy path: rung V-c only |
| 5 | (e) Loop PGO robustness (G60) | MH_04 spread 0.173-0.251 → ≤ 0.03 wide; 17 `[pgo] failed` lines unsorted | `K/loop_closure/pose_graph.rs:26-35, 59-92`; `K/loop_closure/sparse_pgo.rs:354-361` | Trace first, then a knob |
| 6 | (f) Speed: F5a and F2-wide scoring, F5b, keyframe ORB on the mapping thread | No accuracy; needed for the gate and for "faster than Basalt" | `K/system/validation.rs:577-583` (`keyframe_orb: Mapping` is unbuilt); `K/system/config/mod.rs:744-758` (loop solve inline by default) | F5a, F2-wide, F5b proceed as carded. F4 follow-up and F6 are parked: the estimator removes their cause |

**(a): new estimator, not R12 plus per-frame pose-inertial.** Reasons:
1. Cadence and budget differ: a per-frame solve on the tracking thread (≤ 8 ms) against a per-keyframe solve on the mapping thread (31-56 ms).
2. The local VI-BA's landmarks are world points shared with loop closing, culling, erase and merge (`K/mapping/bundle_adjustment.rs:242-270`, `K/mapping/vi_ba_schur.rs:876-896`). A landmark marginalized into a prior cannot be re-observed without double counting. A window-owned, host-anchored landmark can.
3. `vi_ba_schur.rs` (4075 lines) also carries wheel, stationary and calibration rows for LaMAria. A separate module keeps all of that bit-identical.
4. It removes G54, G57 and G59 instead of patching them.

The existing pieces are reused, not rewritten:
- `kornia_factors::inertial::preintegration` (`residual_and_jacobians`, `information`) and `kornia_factors::reprojection::residual_and_pose_jacobian`.
- `K/mapping/marginalization.rs` (Schur complement, first-estimate prior, staleness rule; today IMU-only, 15×15).
- `K/tracking/pose_estimation/pose_inertial.rs:42-85` (`FrameNavState`, `NavPrior`): this is the window-of-one case of the same estimator.
- `K/tracking/flow_tracker.rs` `RigFlow` (per-camera shards, template cache, round trip, refill), `K/upstream/kornia_imgproc/{lk_window,good_features,corner_subpix}.rs` with their OpenCV fixtures and benches.
- `K/mapping/synthetic_walk.rs` for synthetic tests.

Per-frame pose-inertial (`tracking.pose_inertial: OrbSlam3`) stays as the bridge: V-a is already running and tests the theory cheaply. R12 inside the local BA is dropped from the ROADMAP "Now" list.

## 3. Ladder

Rule for every rung: write the full slam-experiment block into `R/experiments/euroc-2x-plan.md` before code; results go to `R/experiments/euroc-2x-results.md`; ledger and ROADMAP log updated at each verdict. Knob default = today's behaviour. Identity trio on every new binary. SYNC noise band = starts 0/30/60 (starts 0/1/2 are degenerate on MH_04, V2_03, V1_02). No path under the user's home config directory goes into repo docs.

**2x-0. Baseline and floors (no repo code, no new SLAM runs).**
- a. Score G11 and V-a against their written predictions. Pick the real-time base arm `RTbase` from them.
- b. Keyframe-only 10 s pieces on the S0E and S0EIH runs, all sequences (`E/r11b/kfpieces.py`), against Basalt. P: ≤ 1.3x Basalt on ≥ 8 of 11 (V1_02 measured 0.0105 against 0.009). K: > 1.5x on ≥ 4 means the frozen window still costs keyframe precision.
- c. Floor test: pairwise SE(3) ATE among `B/results/basalt/<seq>/trajectory.txt`, `B/results/orbslam3/<seq>/f_run.txt` and `E/r6/` offline outputs, and each against GT; lever-arm fit with `E/resid/v13v22/lever.py`. GT term g² = (e_AG² + e_BG² − e_AB²) / 2, three estimates per sequence. P: g ≥ 0.025 on V1_01 and V2_01. K: g ≤ 0.010 there means no floor; the targets stand.
- **Stop S-A:** owner reviews the refreshed table and the floor.

**2x-1. Speed baseline against Basalt (bench only).** Section 5. P and K are written after the first quiet Basalt sample; no number is assumed.

**2x-2. G60 loop PGO.**
- Gap G60. Step 1: sort the 17 `[pgo] failed` lines by sequence from existing logs. Step 2: additive debug line with per-iteration cost; debug run of S0PEIH MH_04 start 60.
- Knob (after the trace, one of): accept a capped solve whose cost and loop residual fell; or the information-weighted persistent graph (`LoopGraphSettings`, `K/loop_closure/closer.rs:56-62`, config only).
- P: MH_04, 5 starts: `[pgo] failed` 0; start-60 error at 80-90 s 0.48-0.56 → ≤ 0.30; max − min over starts ≤ 0.03. Guards V1_02, V2_01, MH_01 inside band, loops applied ≥ baseline.
- K1: failures gone and start 60 still > 0.22: G60 is not the spread. K2: spread closes with failures unchanged: not credited.

**2x-3. Bank the legacy stack (cards exist in `R/experiments/euroc-top1-card.md`).** V-c (PnpInliers without duplicate seeds), F5a and F2-wide scoring, D5b then F5b. Run them as written. They stay useful: the mapping thread hosts growth, loop work and local BA in both architectures.

**2x-4. Estimator design card and synthetic tests. Stop S-B before any estimator code.**
- New module `K/odometry/` (small files: state, landmarks, linearize, marginalize, solve, frontend adapter, tests). Knob `pipeline.odometry: MapTracking | SlidingWindow(( mode: Observe | Drive, .. ))`, default `MapTracking`.
- Spec v1: 3 frame states of 15 DOF, 7 keyframe poses; host-anchored inverse-distance landmarks; two-camera KLT on the rectified pair (left temporal, right temporal, round trip ≤ 0.2 px, row gate); 7 iterations from λ 1e-4; f64 information form with a dense Cholesky (faer). Initial constants are Basalt's, each cited; they move to sensor-derived values in the cleanup card.
- Tests before any run: marginalized solve equals the batch solve on a linear problem (1e-9); the prior's 4-DOF nullspace (x, y, z, yaw) stays unobservable; a synthetic walk recovers bias and gravity.

**2x-5. Estimator, Observe mode.** The estimator runs beside the pipeline and writes `traj_vio.txt` (new flag `--trajectory-vio`; `E/score.py` gains that input).
- Gaps G13, G58, G62, G50. Mechanism: frames are visual-only PnP on a frozen map (`K/system/config/mod.rs:519-530`).
- P1 (mechanism): V1_03 HF ≤ 0.008 on every start (`hf.py`; ours 0.041-0.054, Basalt 0.004). Per-frame translation step p50 ≤ 1.5x Basalt on MH_01, V1_02, V2_01.
- P2: MH_04 vertical increment over 63.3-66.7 s within ±0.10 m (`E/resid/cf.py`; S0PEIH −0.22..−0.29).
- P3: VIO-only ATE ÷ Basalt 0.8-1.3 on ≥ 9 of 11, none above 1.6.
- P4: V2_03 frames without a pose 0; ATE ≤ 0.30.
- P5: flow plus solve p50 ≤ 16 ms, p99 ≤ 35 ms on one core (criterion bench and a `[summary]` line). KLT kernel matches and beats OpenCV on the existing fixtures.
- K1 (mechanism, no score): P1-P2 hold and P3 fails on ≥ 3. Diff against Basalt on MH_01, first 400 frames, before anything else.
- K2 (score, no mechanism): P3 holds with a failed nullspace test or HF > 0.02. Not credited.
- K3: P5 misses (p50 > 25 ms). Square-root f32 solve and `wide` SIMD before Drive.
- K4: V-a's K1 fired and Observe HF > 0.02. The "IMU on every frame" theory is wrong; stop the estimator.
- Identity: knob off, the trio. Knob `Observe`, SYNC S0E V1_02: `traj.txt` `cmp`-identical to knob off on the same binary.
- **Stop S-C:** go or no-go on Drive.

**2x-6. Estimator, Drive mode.** Output pose = `map_from_odometry` ∘ VIO pose. The mapper takes VIO keyframes; keyframe ORB runs on the mapping thread (this builds `keyframe_orb: Mapping`); local BA refines the map and no longer rewrites the live pose.
- P1 (mechanism): no ORB on the tracking thread; `total_ms_keyframe` p50 ≤ 30 ms (today 68-95); late frames ≤ 1 %.
- P2: RT ÷ SYNC 0.9-1.15 on all 11.
- P3: causal ≤ 1.0x Basalt on ≥ 9 of 11; ≤ 0.75x on MH_02, V1_02, V2_03.
- K1: worse than `RTbase` by > 15 % on ≥ 3 of MH_01, MH_02, MH_03, MH_05, V1_02. Mid-term reuse is missing; go to 2x-7 before any gate.
- K2: tracked < frames − 3 or any give-up.
- **Stop S-D.**

**2x-7. Map landmarks in the window, with their covariance.** Knob inside `SlidingWindow`.
- P: MH_01 ≤ 0.040, MH_02 ≤ 0.028, V1_02 ≤ 0.030 (the smoothed values of today's map). Counter: 20-50 % of window rows on map landmarks; mean normalized χ² of those rows 0.8-1.3.
- K1: χ² > 2: the map is over-confident (double counting). No credit even if ATE falls. K2: counters fine and < 10 % gain on ≥ 3: map error dominates; 2x-8.

**2x-8. Background global stage (card AI (b)).**
- First the measurement the master card names: prefix replay with `offline-ba --until <t>` on saved sessions, re-anchoring only frames after t + 4 × solve CPU. P: 30-50 % of the offline gain reaches the causal score (MH_04 0.177 → 0.12-0.15; V2_02 0.080 → 0.06-0.07). K: < 15 % on ≥ 4 of 6: do not build it.
- Then the build: knob `mapping.global_stage: Off | Background(..)`, a bounded window of about 60 keyframes, on the worker pattern of `K/loop_closure/worker.rs:102-143`, correction through `map_from_odometry`. The carded drop-schedule fix (`outlier_schedule: AfterConvergence`) goes in first.
- 2-core budget: tracking ≤ 0.45 core, mapper ≤ 0.9, loop plus global ≤ 0.35, headroom 0.3.

**2x-9. Gates. Stop S-E before any push or promotion.** All 11 × 3 real-time runs per arm under the lock; the paced Basalt comparison; then the full-sequence LaMAria guard.

## 4. Parallel lanes and machine limits

| lane | worktree (new, from `feat/global-pose-graph`) | rungs | builds |
|---|---|---|---|
| A | `/home/edgar/dev/split/kornia-slam-2x-a`, branch `feat/2x-vio` | 2x-4 → 2x-5 → 2x-6 → 2x-7, in order | yes |
| B | `/home/edgar/dev/split/kornia-slam-2x-b`, branch `feat/2x-loop-rt` | 2x-2, 2x-3 | yes |
| C | none (scripts under `E/` and `B/`) | 2x-0, 2x-1, the 2x-8 prefix replay | Docker only |

- 2x-0 and 2x-1 start now. 2x-2 and 2x-3 run beside lane A. 2x-8's build waits for S-D.
- Limits: at most 3 heavy jobs, at most 2 cargo builds (`-j2`), at most 2 `kornia-slam*` SYNC runs. Every pinned run goes through `flock P/rt.lock`, one at a time, load ≤ 6, no cargo running; no SYNC run starts while the lock is held. New real-time runs queue behind G11.
- Integration: cherry-pick onto `feat/global-pose-graph` (no merge commits), identity trio on each integrated binary.

## 5. Speed comparison against Basalt

- **Measure A (no Basalt change, do first): CPU per frame.** Basalt blocking in Docker with `--cpuset-cpus 0,1` and `--num-threads 2`; ours paced through `E/gate/rt_one.sh`. Compare process CPU ms per frame (`/usr/bin/time` user + sys over frames) and peak RSS. CPU per frame does not depend on pacing when no work is dropped.
- **Measure B (the owner's test): paced latency.** Bench-side patch of about 30 lines in the Basalt Docker image, outside the repo: pace `feed_images` and `feed_imu` to dataset timestamps; remove the forced disable at `BA/src/vio.cpp:276-284`; log wall time per output state. Then `vio_enforce_realtime: true` drops late frames (`sqrt_keypoint_vio.cpp:173-176`). Latency = output wall time minus scheduled arrival, the same definition as our `latency_ms`.
- **Protocol:** cores 0,1 for both; interleave ours and Basalt per sequence; 3 runs each; the shared lock; load ≤ 6; record `scaling_cur_freq`; one warm-up read of the dataset; PNG decode counts for both.
- **Report per sequence:** latency p50 / p90 / p99, final latency, CPU ms per frame, frames dropped (Basalt) and late frames (ours never drops), and the ATE of that paced Basalt run.
- **"Faster" (proposed):** on every sequence our median latency p50 and p99 and our CPU ms per frame are below Basalt's, with 0 frames dropped by us.
- **Budget:** unknown until Measure A. On disk: Basalt 40 ms mean wall on 3.2 cores at load 22-30 (not a quiet number), paper 7.8 ms. Ours today: 30-41 ms mean total, 47-74 ms CPU per frame, keyframe frames 68-95 ms. Planning budget for the estimator path: source ≤ 5 ms, two-camera flow ≤ 8 ms, solve ≤ 8 ms.

## Risks (what fails silently, and the check)

- The shadow estimator perturbs the main path. Check: Observe-on `traj.txt` `cmp`-identical in SYNC.
- Wrong first-estimate handling works on short runs and drifts yaw on long ones. Check: the nullspace test and the batch-equivalence test in 2x-4.
- Map landmarks counted twice in 2x-7. Check: the χ² counter and K1 there.
- Frames before inertial init are written in a pre-gravity frame. Check: score on the same associated frame set as Basalt.
- A concurrent lane contaminates a pinned run. Check: `E/gate/pinwatch.log` and the `contaminated` marker.
- The targets derive from one Basalt run at load 24-27. Check: report the quiet rerun beside them; the targets do not change.
- The dirty window-test file gets staged. Check: `git status --short` shows only intended paths before each commit.

## Verification (release only)

- `cargo fmt --all -- --check`
- `cargo +1.100.0 test --release -j2 -p kornia-slam --features serde` — all pass.
- Clippy with `-D warnings` on `kornia-slam --all-targets` and on the app with `--no-default-features --all-targets` — clean.
- `cargo build --release -j2 -p kornia-slam-app`, then copy the binary to `E/bin-<tag>/kornia-slam-euroc`.
- `BIN=E/bin-<tag>/kornia-slam-euroc E/integ/ident.sh` — prints `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` (md5 `d4f77b567b912b5a4caa2af6085a35b1`).
- Real time: `BIN=… flock P/rt.lock E/gate/rt_one.sh <ARM> <SEQ> <REP>` — a `done` file, no `invalid`, no `contaminated`, header `diff`-clean.
- Score: `E/score.py` (body centre, `B/eval.py --fmt tum --rpe 1,10`). A rung passes when every P holds and no K fires; anything else gets a written diagnosis first.

## Open questions (with my recommendation)

1. **V1_01 0.022 and V2_01 0.020 are below every result on disk, non-causal included.** Keep them, or replace them by the measured floor plus 10 % if 2x-0c shows one? Recommend: decide at S-A, from the floor numbers.
2. **Milestones.** Recommend three gates: ≤ 1.0x on all 11, then ≤ 0.75x, then 0.5x.
3. **New estimator module, R12 dropped.** Recommend yes (section 2).
4. **Square-root form.** Recommend f64 information form in v1; square-root f32 only if K3 of 2x-5 fires.
5. **Initial biases and IMU noise.** Basalt starts from offline-calibrated biases and inflated accel noise (8x datasheet). Recommend: no calibrated biases; run Basalt's noise values as a reference arm in Observe, datasheet as the second arm, and choose by χ² consistency, not by ATE.
6. **Patching Basalt in the bench Docker image for the paced test.** Recommend yes; it stays outside the repo.
7. **Basalt rerun.** Recommend: targets stay as given; rerun Basalt quiet, 3 times, only for the speed baseline, and print its ATE beside.
8. **Estimator input.** Recommend the rectified pair in v1 (3-5 ms with F3); raw images through the camera model later, with the fisheye work.
9. **ORB-SLAM3 causal reference** (dump the pose at each `TrackStereo` return, bench side). It would show whether any system reaches 0.5x Basalt causally. Recommend yes, lane C.
10. **Parking the F4 follow-up, F6 and Option A on the legacy path.** Recommend yes until S-C.
11. **Pushes.** The three local commits and all new work wait for your go, as the master card's stop rule says. Confirm.

## Owner answers (2026-11-24)

- Plan approved; all 11 open questions answered as recommended.
- **New input:** G60 is diagnosed (card section "Traces V-d … and G60", commit 8ba53ba). The f32 PGO solve converges, then is dropped by the 30-iteration `!converged` rule. Rungs G60-a (config: `information: Drift`), G60-b (accept a capped solve whose cost and loop residual fell) and G60-c (f64 with analytic Jacobians).
- **V-d:** V2_02 is short by 2-7 % (under-scale), carried by the 3-keyframe window. Rung V-d2: window 10 + 7 iterations from λ 1e-4.

## 2x-1 protocol as run (lane C, written 2026-11-24 before the speed runs)

- Scripts: `E/twox/speed_all.sh` → `E/twox/speed_seq.sh ARM SEQ` under `flock P/rt.lock`, one block per sequence
  (MH_01, V1_02, V2_01, MH_04). In each block, after one warm-up read of the PNGs, for rep 1..3: ours, Basalt A,
  Basalt B. Every run starts only with no `cargo`/`rustc`, no `kornia-slam*`, no `basalt_vio`, 1-min load ≤ 6; a
  1 s sampler marks a run contaminated (rerun, at most 3 tries) when any of those appears during it.
- Ours: `E/twox/rt_one_speed.sh` (copy of `E/gate/rt_one.sh` that also records peak RSS), `RTbase`, paced,
  `taskset -c 0,1`, run dirs `E/runs/SP-<ARM>_<SEQ>_s0_r<rep>`.
- Basalt A: stock image `bench-basalt2` (commit 0f3b2b52), `docker run --cpuset-cpus 0,1`, `--num-threads 2`,
  blocking (as fast as it can), no `--marg-data`. `/usr/bin/time -v` inside the container.
- Basalt B: image `bench-basalt2-paced` = `bench-basalt2` + `B/basalt/paced/vio.cpp` (patch of `src/vio.cpp` only,
  35 diff lines, `B/basalt/paced/vio.orig.cpp` is the original). With `BASALT_PACE=1`: `feed_images` and
  `feed_imu` sleep until `t0_wall + (t − t0_data)` before decoding / pushing (t0_data = first image or IMU stamp),
  the forced `vio_enforce_realtime = false` is replaced by `true`, and each output state writes
  `t_ns latency_ms` (pop time minus scheduled arrival) to `latency.txt`. Frames dropped = images − output states.
- Recorded per run: `scaling_governor`, `scaling_cur_freq` of cpu0 and cpu1, load, at start and end. Cores 0 and 1
  are two physical cores (siblings 0,6 and 1,7).
- **Faster (definition approved with the plan):** on every sequence our latency p50 and p99, and our CPU ms per
  frame, below Basalt's, with 0 frames dropped by us.
- No P/K numbers are assumed before the first quiet Basalt sample (plan rung 2x-1).

## 2x-5 experiment block (Observe), written 2026-11-24 before any estimator code

Design: `experiments/euroc-2x-vio-design.md` (approved at S-B). This block is its section 10, unchanged; results go to `experiments/euroc-2x-results.md`.

- **Gaps.** G13, G58, G62, G50 (`R/experiments/gap-ledger.md`).
- **Mechanism.** Frames are visual-only PnP on a frozen map (`K/system/config/mod.rs:519-530`). The window is 3 keyframes clamped to a fixed past; flow is left-only in Stereo mode.
- **Theory.** Every frame holds pose, velocity and bias under an IMU factor at full weight, with a dense first-estimate prior for the past. The gyro then constrains the rotation-translation ridge on every frame (V1_03 high-frequency error, MH_04 dark-span step), and right-camera rows keep the frame constrained when the left tracks thin out (V2_03).
- **Knob.** `odometry: SlidingWindow(( mode: Observe ))` on S0E.
- **Arms** (`E/mkcfg.sh`, outside the repo): `VO-ds` = S0E + the knob (rig, datasheet noise); `VO-bn` = the same with `imu_noise: Explicit((..))`, Basalt's values (section 1).
- **Noise-arm rule, written before the runs (owner answer 5).** Screening: MH_01, V1_02, V1_03, MH_04, V2_03, start 0, both arms (10 runs). The primary arm is the one whose `imu_chi2_nav` is closer to 1 in `|log|` on at least 3 of 5; a tie goes to `VO-ds`. ATE does not enter the choice. The other arm is reported, not scored.
- **Protocol.** SYNC, `--no-pace`, 900 s cap, at most 2 `kornia-slam*` processes, the command of the card's section 2 plus `--trajectory-vio $d/traj_vio.txt`. Primary arm on all 11 sequences × starts 0, 30, 60 (33 runs, about 4 min each). Each run is within the 5-minute rule.

**Predictions (as in the plan), with the measurement.**

| id | prediction | measured by |
|---|---|---|
| P1 (mechanism) | V1_03 HF ≤ 0.008 on every start (`hf.py`; ours 0.041-0.054, Basalt 0.004). Per-frame translation step p50 ≤ 1.5x Basalt on MH_01, V1_02, V2_01. | `E/resid/v13v22/hf.py V1_03_difficult - VO=traj_vio.txt`; new `E/twox/tsteps.py` (the earlier `tsteps.py` was never committed) |
| P2 | MH_04 vertical increment over 63.3-66.7 s within ±0.10 m (`E/resid/cf.py`; S0PEIH −0.22..−0.29). | `E/resid/cf.py MH_04_difficult traj_vio.txt 81.5 63.3:66.7:v`, the `d=(..)` z value |
| P3 | VIO-only ATE ÷ Basalt 0.8-1.3 on ≥ 9 of 11, none above 1.6. | new `E/twox/vio_score.py`: `B/eval.py --fmt tum --rpe 1,10` on `traj_vio.txt`, and on `B/results/basalt/<seq>/trajectory.txt` cropped to the same stamps (5 ms); `E/score.py` gains `traj_vio.txt` in `ALLOWED` |
| P4 | V2_03 frames without a pose 0; ATE ≤ 0.30. | `[summary] vio frames=` against `poses=`; `resets=0` |
| P5 | flow plus solve p50 ≤ 16 ms, p99 ≤ 35 ms on one core (criterion bench and a `[summary]` line). KLT kernel matches and beats OpenCV on the existing fixtures. | `[summary] vio_ms` from one run pinned with `taskset -c 8` under `flock P/rt.lock`; `bench_odometry_window`; T9 |

**Kill rules (as in the plan).**

| id | rule |
|---|---|
| K1 (mechanism, no score) | P1-P2 hold and P3 fails on ≥ 3. Diff against Basalt on MH_01, first 400 frames, before anything else. |
| K2 (score, no mechanism) | P3 holds with a failed nullspace test or HF > 0.02. Not credited. |
| K3 | P5 misses (p50 > 25 ms). Square-root f32 solve and `wide` SIMD before Drive. |
| K4 | V-a's K1 fired and Observe HF > 0.02. The "IMU on every frame" theory is wrong; stop the estimator. |

For K1's diff, the order of suspects is: open question 1 (FEJ form), the IMU noise arm, the bias start (Basalt's is calibrated), then track counts per frame.

**Bit-identity.**
1. Knob off: `BIN=E/bin-2x5-<tag>/kornia-slam-euroc E/integ/ident.sh` prints `IDENT_V102_OK`, `HEADER_OK`, `IDENT_R01_OK` (md5 `d4f77b567b912b5a4caa2af6085a35b1`).
2. Knob `Observe`, SYNC S0E V1_02: `traj.txt` and `traj_smoothed.txt` `cmp`-identical to knob off on the same binary.
3. Header: `E/expect/VO-ds.header` = `S0E.header` plus the odometry line, `diff`-clean.
4. Determinism: VO-ds V1_02 and MH_04 twice; `cmp traj_vio.txt` prints nothing.

**Noise.** `traj_vio.txt` is deterministic in SYNC, so the band is the start band: min..max over starts 0, 30, 60 per sequence (starts 0/1/2 are degenerate on MH_04, V2_03, V1_02). P3 is read on the median of the three, each start against Basalt cropped to its own stamps. Basalt is one run at load 24-27; the quiet rerun of rung 2x-1 is printed beside it and the targets do not change.

**Stop S-C** after the verdict: go or no-go on Drive. A miss or a fired kill rule gets a written diagnosis first.

