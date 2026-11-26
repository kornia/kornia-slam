# G57: real time on 2 cores (latency against Basalt)

Status: card written 2026-11-26, not yet reviewed by the owner, nothing run. Paths: `K` = `crates/kornia-slam/src`; the harness lives outside the repo.

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

## Open questions for the owner
1. With `Asynchronous`, a loop correction lands 1-4 s after acceptance, and the map jumps at a frame boundary. Is that acceptable for the causal score and for consumers?
2. Basalt paced drops 1-131 frames per run. Does "0 dropped" bind only us, and is Basalt's latency with drops still the bar?
3. The full check is about 27 min of serial paced runs plus controls. Does it count as a batch gate, or must it fit the 5-minute rule (screen only)?
4. May G57-b stack in this card after the screen, or does it need its own card after the verdict?
