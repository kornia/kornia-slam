# G57: real time on 2 cores (latency against Basalt)

Status: card written 2026-11-26 and reviewed by the owner (answers below). The screen is running. Paths: `K` = `crates/kornia-slam/src`; the harness lives outside the repo.

## Evidence (final tip bench, reps 2-3 of the FOLDF-RT3E runs, per-frame `[i] status … tot= lat=` lines in err.log)

- **The end lag is the sum of loop-closure stalls on the tracking thread.** Every `[loop-closure] accepted` is followed by one keyframe frame of 0.25-3.7 s.

  | Seq | Loops | Total stall (s) | End lag |
  |---|---|---|---|
  | V2_02 | 5-6 | 9.8-10.5 | 10.0-11.0 s |
  | V1_01 | 11-12 | 12.6-14.6 | 9.8-14.1 s |
  | V1_03 | 3 | 5.4-8.0 | 1.7-4.2 s |
  | MH_03 | 6-7 | 6.8 | p99 1.3-1.7 s |

  All frames with `tot` ≥ 300 ms are loop frames, except 2 frames in 8 runs. The stall grows with the map: on V2_02 it is 0.97 s at 30k points and 2.6-2.8 s at 41-49k.
- **Code path.**
  - Keyframe insertion calls `register_place_recognition` (`K/system/mod.rs:3526-3528`, `:3622-3648`), which holds the map lock through `on_keyframe`.
  - With the default `execution: Synchronous` (`K/system/config/mod.rs:747-754`), `handle_accepted` calls `close_loop` (`K/loop_closure/closer.rs:626-662`). That solves, writes back and runs `fuse_verified_loop` inline (`closer.rs:758-775`).
  - Fusion is a brute-force scan of every target descriptor per source point (`K/loop_closure/fusion.rs:118-176`).
- **Split of the stall.** Only unpaced SYNC debug runs log `[loop-time]`; the paced runs do not, so the split on 2 paced cores is not measured.

  | Step | Time |
  |---|---|
  | `fuse_ms` | 55-1166 ms of `on_keyframe_ms` 130-1262 ms |
  | solve | 3-145 ms |
  | writeback | 2-43 ms |
  | detect + verify | 5-20 ms |

- **The backlog cannot drain.**
  - Tracking-thread wall time ÷ sensor time:

    | Seq | Ratio | Without loop frames |
    |---|---|---|
    | V2_02, V1_01 | 0.94-0.97 | 0.86-0.88 |
    | V1_03 | 0.83-0.84 | |
    | MH_03 | 0.80 | |

  - A keyframe frame costs 85-100 ms p50, against a 50 ms frame period:

    | Step | ms |
    |---|---|
    | ORB | 24-26 |
    | growth | 23-37 |
    | loop recognition | 12-17 |
    | source | 4-6 |
    | rest | tracking |

  - Keyframes are 27-28 % of frames on V2_02 and V1_03.
- **Not the cause:**
  - core contention: the process uses 1.13-1.30 of 2 cores; local BA uses 0.13-0.24 core, at 31-52 ms per solve;
  - the map lock: `map_wait_ms` p99 is 0.0;
  - relocalisation: 0 attempts;
  - dropped frames: 0, because the pacer never drops.
- **Basalt.**
  - It has no loop closing.
  - It uses a fixed-lag window of 3 states plus 7 keyframes at 7 iterations, optimised inline per frame. Its flow runs on its own thread.
  - Paced, it drops queued frames: 1-8 per run on the Vicon rooms and 36-46 on MH_03.
- **Offline price.** A queue replay of the logged `tot`, with loop frames replaced by a normal keyframe plus 60 ms (or plus 25 % of the stall), gives the ranges in P1-P4. The replay reads the baseline 10-25 % low, so the ranges are wide.

## Experiment: G57-a, loop solve and fusion off the tracking thread
```
Gap tested: G57 (also G59: lag)      Base: kornia-slam@b231a87 (code = 95bc33c)   Binary tag: bin-fold (md5 aefd2aca…), arm RT4EF
Mechanism: an accepted loop is solved, written back and fused inside the accepting keyframe, on the tracking thread, under the
  map lock (K/system/mod.rs:3526-3528, :3622-3648; K/loop_closure/closer.rs:626-662, :758-775; fusion.rs:118-176). Each loop
  stalls tracking 0.25-3.7 s. Tracking runs at 0.80-0.97 of real time, so the lag never drains.
Theory: the background path already exists. The worker runs the solve and the fusion search on a snapshot (worker.rs:54-82);
  tracking keeps only the capture (closer.rs:819-838) and the writeback at a frame boundary (system/mod.rs:1060, closer.rs:967-1027).
  Reference: VINS-Fusion, one solve in flight on its own thread, later loops coalesce (loop_fusion/src/pose_graph.cpp:52, :434).
  ORB-SLAM3 also uses a loop thread (System.cc:214, LoopClosing.cc:274, :1140) but stops local mapping (:975), which costs BA
  on 2 cores. Basalt has no loop work at all, so it has no correction to deliver.
Knob: loop_closing: Enabled(vocabulary: …, execution: Asynchronous). Exists (config/mod.rs:740, :747-761); default Synchronous = today.
  No library code. Harness only: arm RT4E (RT3E + execution), expected header adds ", asynchronous".
Baselines: FOLDF-RT3E_{V2_02_medium,V1_01_easy,MH_03_medium,V1_03_difficult}_s0_r{2,3} (95bc33c, md5 aefd2aca…, load 2.9-4.0,
  governor powersave). Rep 1 was throttled: not a baseline. Rerun one RT3EF control beside each RT4EF rep (same clock).
Predictions:
  P1 V2_02: latency p50 1100-1296 -> 150-550 ms; p99 11.1-12.0 -> 1.3-3.5 s; end lag 10.0-11.0 -> 0.1-2.3 s; ATE inside 0.093-0.108
  P2 V1_01: p50 306-420 -> 100-330 ms; p99 12.2-14.8 -> 1.2-3.6 s; end lag 9.8-14.1 -> 0.0-2.6 s; ATE inside 0.043-0.054
  P3 MH_03: p50 72-75 -> 47-56 ms; p99 1.34-1.72 -> 0.25-0.61 s; end lag 0.0-0.5 -> <= 0.05 s; ATE inside 0.044-0.054
  P4 V1_03: p50 106-638 -> 57-70 ms; p99 3.9-5.4 -> 0.3-1.0 s; end lag 1.7-4.2 -> <= 0.1 s; ATE inside 0.101-0.165
  P5 CPU per frame within +-3 ms of 69.6-71.1 / 69.0-70.9 / 56.3-56.8 / 64.2-65.3; loops applied within +-1 of 5-6 / 11-12 / 6-7 / 3;
     0 dropped; frames = dataset length; solves per keyframe >= 0.97.
     measured by: [summary] latency_ms, final_latency_ms, loop_closure lines in err.log; speed_report.py FOLDF-RT4E (ATE, cpu_ms)
  Mechanism counter: frames with tot >= 300 ms and their sum: 6-7 / 10.3-10.5 s (V2_02), 9-11 / 11.8-13.7 s (V1_01),
     6-8 / 6.5-6.6 s (MH_03), 3 / 5.3-7.9 s (V1_03) -> at most 1 frame and <= 0.5 s on each; no loop frame above 250 ms.
Kill rules:
  K1 every loop frame <= 250 ms but end lag > 3 s on V2_02 or V1_01 -> the loops were not the whole backlog; the rest is the
     0.86-0.88 keyframe load. Stack candidate 2; do not tune the worker.
  K2 end lag falls > 50 % while a loop frame still costs >= 500 ms, or loops applied fall > 20 % -> the gain is lost loops or clock
     luck (known defect, config/mod.rs:750-752: 18-22 against 47 on the sim recording). Not credited. Run D5b (SYNC,
     Deferred(frames: 1) against Synchronous on MH_01 and V1_02) and explain the difference first.
  K3 any rep's ATE leaves its band by more than max(0.01 m, 15 %) -> stop; read where the late correction lands (G59).
Bit-identity: no code change and the default is unchanged, so knob-off is the same binary. Check the RT3EF control header
  against the RT3E expected header. If an app counter is added later: IDENT_V102_OK and HEADER_OK.
Noise: 3 paced reps per sequence, taskset -c 0,1, one run at a time under rt.lock, load within +-2 of the control; record f0/f1.
  Known bands (reps 2-3): end lag V2_02 10.0-11.0 s, V1_01 9.8-14.1 s; V1_03 p50 106-638 ms. Paced runs are not bit-repeatable.
```
Run order:
1. **Screen:** V2_02 and V1_03, one RT4EF rep each (2.2 and 1.9 min).
2. **Full:** 3 reps on each of the 4 sequences, each with an RT3EF control.

**Stated limit.** At its predictions this rung does not pass the goal alone:
- Basalt paced reads p50 39-50 ms and p99 97-546 ms on these four sequences.
- CPU per frame stays above Basalt on V2_02 (70 against 63 ms) and on V1_03 (65 against 58 ms).

**Next rungs (they stack):**
- **G57-b:** `mapping: ( keyframe_growth: Mapping )`.
  - The knob already exists (F5a, `K/system/mod.rs:3475-3498`, default `Tracking`); no result section has been found for it.
  - It moves 23-37 ms p50 per keyframe to the mapping thread.
  - Tracking load on V2_02 falls from about 0.87 to about 0.67.
- **G57-c:** loop verification off the tracking thread (12-17 ms p50, 46 ms p90 per keyframe; card F5b).

## Owner answers (2026-11-26)
1. Asynchronous loop closing is acceptable: a correction may land 1-4 s after acceptance.
2. "0 dropped" stays a rule for us. The bar is Basalt's paced latency as measured, with its drops.
3. Screen first: V2_02 and V1_03, one rep per arm, with an RT3EF control at the same clock. The full 3 × 4 check follows only if the screen passes.
4. G57-b stacks in this card after G57-a. It must not hurt the other tests (rules below).

## Experiment: G57-b, keyframe growth on the mapping thread (stacked on G57-a)
```
Gap tested: G57      Base: G57-a (arm RT4EF)      Binary tag: bin-fold (md5 aefd2aca…), arm RT4EBF
Mechanism: after G57-a, a keyframe frame still costs 85-100 ms p50 against a 50 ms period. Growth (triangulation and forward fusion,
  K/mapping/keyframe_mapping.rs:330-397) is 23-37 ms of it, run inside insert_keyframe on the tracking thread (K/system/mod.rs:3475-3498).
  Keyframes are 27-28 % of frames on V2_02 and V1_03, so tracking load without loops is 0.86-0.88 of real time.
Theory: ORB-SLAM3 runs CreateNewMapPoints and SearchInNeighbors on the local-mapping thread, outside mMutexMapUpdate
  (LocalMapping.cc:101-108). The F5a lock audit (euroc-top1-card.md, "F5a lock audit") shows every append is safe against tracking.
Knob: mapping: ( keyframe_growth: Mapping ). Exists (F5a); default Tracking = today. No library code.
Baselines: the RT4EF screen runs (same clock), and the RT3EF control.
Predictions:
  P1 keyframe-frame tot p50: 85-100 -> 55-75 ms (V2_02, V1_03)              measured by: per-frame [i] lines, keyframe frames
  P2 tracking wall / sensor time without loop frames: 0.86-0.88 -> 0.62-0.75 (V2_02)
  P3 V2_02 latency p50 -> 45-150 ms, end lag <= 0.5 s; V1_03 p50 -> 45-65 ms, end lag <= 0.1 s
  P4 map_wait_ms p90 <= 5 ms; growth_lag_frames_max <= 3
  Mechanism counter: keyframe_insert p50 (or the growth share of tot) falls by 15-30 ms on the tracking thread.
Kill rules:
  KB1 keyframe-frame tot falls >= 15 ms but end lag or p50 does not move beyond the RT4EF screen run -> the rest of the backlog
     is elsewhere (ORB on keyframes, loop recognition 12-17 ms: G57-c). Keep the knob only if the "no harm" rules hold.
  KB2 latency improves but keyframe-frame tot is unchanged (< 10 ms) -> clock luck or another cause; not credited.
  KB3 map_wait p90 > 5 ms -> lock convoy; split prepare from apply (F5a K1).
"No harm" rules (owner: it must not hurt other tests). Any one fails -> G57-b is not promoted:
  NH1 tracked frames drop by more than 3, or keyframes rise more than 10 %, or any give-up or extra coast frame (F5a K2).
  NH2 local-BA solves per keyframe fall below the RT4EF run (F5a K3).
  NH3 ATE inside its band on the screen; then, before promotion, on all 11 sequences: ATE inside the RT3EF 3-rep band
      (euroc-why-not-basalt.md, final table) by max(0.01 m, 15 %).
  NH4 knob off is the same binary and the same default config, so the identity trio (IDENT_V102_OK, HEADER_OK, IDENT_R01_OK)
      and the repo tests do not change; rerun the trio once on the promoted config's binary.
```

## Open questions for the owner (answered above)
1. With `Asynchronous`, a loop correction lands 1-4 s after acceptance, and the map jumps at a frame boundary. Is that acceptable for the causal score and for consumers?
2. Basalt paced drops 1-131 frames per run. Does "0 dropped" bind only us, and is Basalt's latency with drops still the bar?
3. The full check is about 27 min of serial paced runs plus controls. Does it count as a batch gate, or must it fit the 5-minute rule (screen only)?
4. May G57-b stack in this card after the screen, or does it need its own card after the verdict?

## Results: screen (2026-11-26 08:35-08:47, `bin-fold`, paced, `taskset -c 0,1`, one run per arm, alone on the machine)

**Clock.** The CPU now runs at about 3.6 GHz; the final bench's reps 2-3 ran lower. The RT3EF control at the same clock already reads V2_02 p50 351 ms and end lag 76 ms, against 1100 ms and 10 s in the baseline reps. So the predictions, which were written against the baseline reps, mix the fix with the clock. The clean comparison is against the control. Basalt must also be re-measured at this clock before any "faster than Basalt" claim (the full check does this).

| run | ATE | p50 / p90 / p99 ms | end lag ms | loops accepted | frames ≥ 300 ms (sum) | keyframe tot p50 | keyframe_insert p50 | map_wait p90 | growth lag max | keyframes / tracked / coasts | CPU ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2_02 RT3EF control r4 | 0.115 | 351 / 1526 / 1986 | 76 | 5 | 6 (7.0 s) | 87.3 | 46.1 | 0.0 | 0 | 657 / 2347 / 1 | 65.0 |
| V2_02 RT4EF (G57-a) | 0.102 | 43 / 114 / 244 | 99 | 6 | 0 | 78.8 | 39.4 | 0.0 | 0 | 646 / 2348 / 0 | 58.4 |
| V2_02 RT4EBF (G57-a+b) | 0.103 | 37 / 97 / 251 | 6 | 5 | 0 | 54.2 | 13.1 | 19.1 | 5 | 643 / 2347 / 1 | 58.1 |
| V1_03 RT3EF control r4 | 0.129 | 38 / 129 / 2362 | 537 | 3 | 2 (3.3 s) | 67.6 | 26.4 | 0.0 | 0 | 572 / 2145 / 4 | 56.0 |
| V1_03 RT4EF (G57-a) | 0.096 | 34 / 87 / 161 | 69 | 2 | 0 | 68.0 | 27.8 | 0.0 | 0 | 574 / 2147 / 2 | 55.3 |
| V1_03 RT4EBF (G57-a+b) | 0.104 | 30 / 76 / 142 | 19 | 2 | 0 | 51.4 | 9.1 | 8.9 | 2 | 568 / 2146 / 3 | 55.5 |

Basalt paced (final bench, measured at a lower clock): V2_02 p50 41.6, p99 132.9 ms, 63.5 ms/frame; V1_03 39.6, 113.7, 58.5.

| P/K | prediction | baseline | measured | verdict |
|---|---|---|---|---|
| G57-a mechanism: frames ≥ 300 ms | ≤ 1 and ≤ 0.5 s; no loop frame > 250 ms | 6-7 / 10.3-10.5 s; 3 / 5.3-7.9 s | 0 and 0 (control 6 and 2) | as predicted |
| G57-a P1 V2_02 p50 / p99 / end lag | 150-550 ms / 1.3-3.5 s / 0.1-2.3 s | 1100-1296 / 11.1-12.0 s / 10-11 s | 43 / 0.24 s / 0.099 s | better than the range (the clock adds to it) |
| G57-a P1 V2_02 ATE | inside 0.093-0.108 | | 0.102 | as predicted |
| G57-a P4 V1_03 p50 / p99 / end lag | 57-70 / 0.3-1.0 s / ≤ 0.1 s | 106-638 / 3.9-5.4 s / 1.7-4.2 s | 34 / 0.16 s / 0.069 s | better than the range (the clock adds to it) |
| G57-a P4 V1_03 ATE | inside 0.101-0.165 | | 0.096 | below the band (better) |
| G57-a P5 CPU per frame within ±3 ms of the control | | 65.0 / 56.0 | 58.4 / 55.3 | missed on V2_02 (−6.6 ms, lower) |
| G57-a K2: loops accepted fall > 20 % | | V1_03 control 3 | 2 | fired on its wording (3 → 2, small count). V2_02 went 5 → 6. Decided on the full check's 3 reps |
| G57-b P1 keyframe tot p50 | 55-75 ms | 78.8 / 68.0 (RT4EF) | 54.2 / 51.4 | as predicted (at the low edge) |
| G57-b mechanism: keyframe_insert p50 falls 15-30 ms | | 39.4 / 27.8 | 13.1 / 9.1 (−26 / −19) | as predicted |
| G57-b P3 p50 / end lag | V2_02 45-150 / ≤ 0.5 s; V1_03 45-65 / ≤ 0.1 s | | 37 / 0.006 s; 30 / 0.019 s | as predicted or better |
| G57-b P4 map_wait p90 ≤ 5 ms; growth lag ≤ 3 | | 0.0 | 19.1 and 8.9 ms; 5 and 2 | **KB3 fired** (lock convoy); growth lag missed on V2_02 |
| NH1 tracked −3, keyframes +10 %, coasts | no extra coast | 0 / 2 coasts (RT4EF) | tracked −1 / −1, keyframes −0.5 / −1 %, coasts 1 / 3 | fired on its wording (+1 coast each). The control had 1 / 4; decided on 3 reps |
| NH2 BA solves per keyframe ≥ RT4EF | | 1.005 / 1.003 | 1.003 / 1.002 | inside noise (−0.002) |
| NH3 ATE in band | | | 0.103 / 0.104 | holds |

**Reading.**
- **G57-a works as theorised.** The loop stalls disappear and the queue drains. ATE is equal or better.
- **G57-b moves 19-26 ms per keyframe off the tracking thread**, as predicted. It brings p50 below Basalt's on both sequences. But tracking now waits up to 19 ms (p90) on the map lock: the lock convoy F5a's K1 foresaw.
- **Both** are at or below Basalt's p50 and CPU per frame on these two sequences. They are not below Basalt's p99 (142-251 ms against 114-133). Basalt has not yet been re-measured at this clock.

**Next.**
1. Full check: 3 reps × {V2_02, V1_01, MH_03, V1_03} × {RT3EF control, RT4EF, RT4EBF}, plus Basalt paced at the same clock. This decides K2, NH1 and NH2 on 3 samples.
2. G57-b before promotion: split prepare from apply (F5a K1), to remove the map_wait p90.
