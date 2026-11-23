# Card: EuRoC as a two-camera rig (Stereo vs Rig)

Status: 2026-11-23. Plan by planner (Fable); owner approved it with all 8 recommended answers on 2026-11-23.
Code base moved from ece6c5f to 82f61a0 (docs-only commits in between). Steps 0-6 in this run; the
44-run sweep (step 7) waits for the stop-point S2 report.

Paths: `P=/home/edgar/dev/split/.pr180-fix`, `E=$P/eurocrig` (experiment dir), baseline `$P/euroc`.

## Experiment: EuRoC as a two-camera rig (Stereo vs Rig)
Gap tested: G50, G51            Base: kornia-slam@82f61a0   Binary tag: eurocrig
Mechanism: cameras: Stereo runs flow on the left camera only (system/mod.rs:320-324, rig_flow_cameras :3512);
  flow_threads > 1 is refused outside Rig (validation.rs:504-508). With per_camera 48 and min_tracks 48 the
  flow-only gate passes only when every track survives (G51): MH_01 flow_unavailable 3219 of 3682.
  Second mechanism found while planning: prepare_rig matches the overlap on distorted pixels for a pinhole rig
  (frontend/mod.rs:292-299).
Theory: a second camera doubles the tracks behind the same gate and adds rays with a 0.11 m lever arm, so
  flow-only frames return and each flow pose has more inliers.
  References: Basalt KLT in every camera (frame_to_frame_optical_flow.h:161-165), epipolar check on unprojected
  rays (:367-375); OpenVINS TrackKLT.cpp:202,259-266; VINS-Fusion feature_tracker.cpp:202-225.
  Basalt's choice (independent KLT per camera plus an epipolar filter) is the one RigFlow already implements.
  Not predicted to help: accuracy of the default frontend, which already has 2220 flow frames on MH_01 and
  sub-pixel SAD depth; the rig's overlap depth has no sub-pixel step.
Knob: sensors.cameras: Rig on an EuRoC source (default Stereo; Rig on EuRoC is an error on the base).
Arms: sdef, rdef, sp48, rp48 (+ baseline preset = Stereo with 96 tracks, equal total tracks).
Baselines: /home/edgar/dev/split/.pr180-fix/euroc/runs/{default,preset,preset48all}_<seq>_r1, code c21299b,
  binary md5 cea7646298a68ddb948b16835472b2e3, load 8-16. Only MH_01-03 exist at card time; other baselines
  are filled from sdef/sp48 runs before any Rig number is read. ATE = bench/eval.py --fmt kornia, SE(3) RMSE.
Predictions:
  400 frames, MH_01, --debug:
  P1 rp48 flow-only frames ([klt-first] .. tracked): 17 (sp48) -> 110-220; flow_unavailable 346 -> <= 110
  P2 rp48 median tracks, [flow] shard=0 and shard=1: 30-48 each (today shard 1 does not exist)
  P3 rp48 mean secondary= on flow-only frames: >= 12 (today 0)
  P4 rdef and rp48 median [overlap] triangulated= per ORB frame: >= 150
  P5 rig file: cameras[1].primary_from_camera |t| = 0.1101 +- 0.0005 m, t.x > 0.109; 752x480; k1 -0.2834/-0.2837
  Full length:
  P6 rp48 MH_01: flow_unavailable 3219 -> <= 800; flow_frames 111 -> 900-1600;
     every sequence: flow_unavailable / frames <= 0.30; flow_secondary_inliers / flow_inliers 0.30-0.60
  P7 rp48 vs sp48 ATE: MH_01 0.097 -> 0.070-0.095; mean of per-sequence ratios rp48/sp48 over 11: 0.75-1.00
  P8 rdef vs sdef ATE: MH_01 0.083 -> 0.070-0.100; MH_02 0.075 -> 0.062-0.090; MH_03 0.060 -> 0.050-0.075;
     mean ratio over 11: 0.85-1.15 (neutral)
  P9 top-1 check: best arm's 11-sequence mean >= 0.060 (ORB-SLAM3 0.035, Basalt 0.081 with V2_03, 0.066
     without); best arm beats Basalt (0.067/0.048/0.062/0.114/0.144/0.043/0.045/0.053/0.039/0.049/0.23) on
     at most 4 sequences; MH_02 stays >= 0.055. This card does not reach top-1 alone.
  P10 scale: MH_01 Sim(3) scale 0.99-1.01 on rdef and rp48 (sdef 1.004)
  P11 time, MH_01: rp48 frontend_ms p50 12-25 (sp48 18.3), frame_ms p50 <= 12 (sp48 9.9);
      rdef frontend_ms p50 18-35 (sdef 0.0), rdef wall 300-450 s (sdef 262 s)
  Mechanism counter: flow_unavailable, flow_secondary_inliers, [flow] shard=1 tracks.
Kill rules:
  K1 mechanism moved, score did not: P6 holds but mean ratio rp48/sp48 > 1.00, or rp48 mean > sdef mean.
     Second-camera flow is not where EuRoC accuracy is lost. G50 stays a speed item; do not build right-camera
     flow in Stereo mode; take the next ledger row.
  K2 score moved, mechanism did not: an R arm beats its S arm by more than the noise band while
     flow_secondary_inliers / flow_inliers < 0.10 or flow_unavailable is not reduced; or rdef/sdef < 0.85.
     The gain is the rig backend (raw images, ORB overlap depth, second-camera BA rows), not flow. Split it
     before any claim; first re-run MH_01 with loop_closing Disabled in both modes.
  K3 correctness, before any ATE: P4 median < 80, or P5 fails, or P10 outside 0.98-1.02, or tracked < frames
     on MH_01. Calibration, extrinsic or undistortion is wrong. Stop; no 11-sequence run.
  K4 regression: rdef > 1.3 x sdef on 6 or more sequences. The rig path loses to rectified stereo on a
     parallel pair (first suspect: no sub-pixel disparity in overlap.rs). EuRoC stays Stereo; diagnose first.
  K5 owner rule: any run over 300 s wall. Stop and ask.
Bit-identity:
  B1 LaMAria R_01: /home/edgar/dev/split/.pr180-fix/baseline/r01full-S10-7.ron, RAYON_NUM_THREADS=1,
     traj cmp-identical to /home/edgar/dev/split/.pr180-fix/baseline/traj.txt
     (md5 d4f77b567b912b5a4caa2af6085a35b1)
  B2 EuRoC sdef: traj.txt cmp-identical to .pr180-fix/euroc/runs/default_<seq>_r1/traj.txt for every
     finished baseline run
Noise: 3 samples per arm on MH_01, MH_03, V1_02 (repeats for the asynchronous preset arms; start_frame 0/1/2
  for the deterministic default arms). Band = spread of those samples; until measured, a difference under 15%
  on one sequence is "inside noise". Verdicts on the other 8 sequences are provisional.
  Timing samples are valid only against runs of the same session with load within +-2.

## Results 2026-11-23: steps 0-6 (stop point S2, before the 44-run sweep)

Commits on feat/euroc-rig (base 82f61a0): 3331f5f card, 1bbd821 EuRoC source as a rig, f791b1d overlap
undistortion fix (G52), 9cfba6e summary flow_tracks line. Binary `$E/bin/kornia-slam-eurocrig`,
md5 344317b26a368a528f5ab1bb9bf8be21, toolchain 1.100.0, release.

Build gates: fmt clean; the four clippy runs (workspace, kornia-slam, app no-default, app uvc) give no
warning; `test --workspace` and `test -p kornia-slam` pass, the app's new tests included (largest
binary: 778 passed, 4 ignored under --workspace; 755 passed, 4 ignored under -p kornia-slam). The first test run failed in the new `radtan_overlap_needs_undistorted_pixels`: one of
the four raw corner pairs matched. The plan's points sat only 1.8-2.7 px off the epipolar line in raw
pixels, inside the 2 px band. Moved to corner points 1.6-1.9 m away (3.0-3.5 px off); the test passes.
Finding from that arithmetic: on EuRoC's near-parallel pair, distortion moves a corner pair only a few
pixels off the row, so many raw pairs pass the band. The larger error is disparity: at 2.5 m near a corner
the raw disparity is 13.6 px against 20.2 px undistorted, so a raw match gives ~1.5x depth. Without the fix
G52 would have biased depth more than it removed matches.

### Identity
- B1 LaMAria R_01 (`r01full-S10-7.ron`, RAYON_NUM_THREADS=1): traj md5 d4f77b567b912b5a4caa2af6085a35b1,
  `cmp` silent. Pass.
- B2 as written fails, and the cause is not the code: the baseline binary does not repeat itself.
  The default arm runs local mapping asynchronously. Baseline default r1 against r2 differs on all 5
  sequences with both; with RAYON_NUM_THREADS=1, baseline MH_01 run twice still differs (byte 8944).
  All 11 `sdef_*_r1` differ from `default_*_r1` (first diff at lines 8-84).
- B2 on a deterministic variant (sdef + `mapping: ( execution: Synchronous )`, RAYON_NUM_THREADS=1,
  `$E/id/b2sync`): baseline binary x2 and the new binary give one md5 per sequence, MH_01
  c197ef84d7fdb3eee0b4c72e881ce1c2, V1_02 (unequal camera counts) 0665a842b7897b239a851c29e01cce16.
  Stereo is bit-identical. Pass on that variant.
- Side finding (baseline, not this card): the async default arm is far from repeatable on V1/V2. ATE SE3
  r1 / r2: V1_03 121 / 270 m, V2_02 59.4 / 5.95 m, V2_03 250 / 2.7 m; sdef r1 here 17.8, 3.36, 9.75 m.
  Per-sequence verdicts on those sequences need many samples, or the synchronous mode.

### 400-frame MH_01 (`$E/q400`, --debug, max_frames 400)
Two rounds: r1 at load 12-16, r2 at load 2-3.5.

| arm | r1 / r2 flow-only frames | flow_unavailable | median tracks shard 0 / 1 | secondary inliers per flow frame | overlap triangulated p50 | wall s r1 / r2 | ATE SE3 r1 / r2 | Sim3 scale r1 / r2 |
|---|---|---|---|---|---|---|---|---|
| sdef | - / 0 (flow_frames 218) | - / 3 | - | 0 | - | - / 8 | - / 0.0119 | - / 1.010 |
| sp48 | 15 / 14 | 355 / 357 | 48 / none | 0 | - | 61 / 10 | 0.0125 / 0.0124 | 1.010 / 1.009 |
| rp48 | 151 / 147 | 78 / 92 | 48 / 48 | 23.7 / 23.9 | 316 / 318 | 64 / 12 | 0.0115 / 0.0114 | 1.017 / 1.016 |
| rdef | 0 / 0 | 0 / 0 | - | - | 316 / 316 | 85 / 16 | 0.0124 / 0.0126 | 1.029 / 1.030 |

All arms: tracked = frames = 400, no give-up. rp48 flow_secondary_inliers / flow_inliers = 0.46 / 0.47.

- P1 holds: rp48 flow-only 151 / 147 (pred 110-220), flow_unavailable 78 / 92 (pred <= 110). sp48 matches the
  card's base (15 / 14 against 17; 355 / 357 against 346).
- P2 holds: both shards median 48 (pred 30-48). At the 48 cap: refill keeps both cameras full.
- P3 holds: 23.7 / 23.9 secondary inliers per flow-only frame (pred >= 12).
- P4 holds: rdef and rp48 median 316-318 overlap points per ORB frame (pred >= 150).
- P5 holds: `$E/mh01-rig.ron`: camera 1 t = (0.11007, -0.00016, 0.00089), |t| = 0.11008 m; 752x480 both;
  k1 -0.28340811 / -0.28368365; camera_to_body = cam0 T_BS.
- K3 does not fire on its listed tests: P4 >= 80, P5 holds, tracked = frames. P10 is a full-length test.
  Watch item: rdef Sim3 scale is 1.029-1.030 in both rounds, against 1.009-1.010 for the Stereo arms and
  1.016-1.017 for rp48. The window has 7.5 m of path but only 0.37 m maximum excursion, so scale is weakly
  observed. Still, the order repeats. Hypothesis, unverified: the overlap depth has no sub-pixel step, and
  inverse-disparity noise biases depth upward (E[1/d] > 1/E[d]); rectified SAD with sub-pixel refinement has
  less noise. That is K4's first suspect. Full-length P10 (0.98-1.02) decides.
- K5 / wall extrapolation (`wall_400 x 3682 / 400`): r1 (load 12-16) sp48 562 s, rp48 589 s, rdef 782 s, all
  over 300 s. r2 (load 2-3.5) sdef 74 s, sp48 92 s, rp48 110 s, rdef 147 s, all under. The linear rule
  also overstates: the first 400 frames hold about one keyframe in 4 frames (98-105 keyframes). Calibrated
  on sp48 r1 against the baseline preset48all full run (61 s -> 221 s, factor 3.6): rp48 about 232 s, rdef
  about 308 s at load 12-16. So the gate passes at low load and fails at the shared-machine load. The sweep
  needs a quiet machine, or the owner's call per plan answer 2 (rdef on 4 sequences only).
- Time (P11 is full length, not scored): at load 2-3.5, frontend_ms p50 sp48 6.8, rp48 8.5, rdef 8.9;
  frame_ms p50 sp48 4.5, rp48 8.5, rdef 11.2, sdef 2.7.

Run times: the laptop suspended from 09:33:25 to 13:42:01 (lid closed, 14916 s). B1 and sdef MH_01 r1 ran
across it. Their recorded walls (15738 s, 15042 s) include it; corrected about 822 s and 126 s.

### Decisions after S2 (coordinator, 2026-11-23, per the owner-approved EuRoC top-1 master plan)
- B2 accepted on the synchronous variant. Synchronous local mapping is the A/B instrument, because the
  async default arm does not repeat itself (above).
- Sweep mode changes from the card: every arm runs with `mapping: ( execution: Synchronous )`, all rayon
  threads, unpaced, 900 s cap per run (owner raised the cap for synchronous runs only). sp48/rp48 switch
  their `execution: Asynchronous` to `Synchronous`; nothing else in the arms changes.
  Arms sdef, rdef, sp48, rp48 on all 11 sequences (rdef on all 11 too).
- Noise: synchronous runs are deterministic, so plain repeats are useless; start_frame 0/1/2 on MH_01,
  MH_03, V1_02 for every arm (24 extra runs).
- K5 for this sweep is the 900 s cap.
- The sweep waits for the top-1 harness's R0 baseline (`.pr180-fix/euroc-top1/lists/r0.txt`), then runs at most
  2 kornia-slam processes. Driver `$E/drive_sync.sh sweep.txt`, run dirs `$E/sync/<arm>_<seq>_s<start>_r1`.
- Scoring follows G55: body-frame `traj.txt` (and `traj_smoothed.txt`) through `euroc-top1/score.py`, never the
  `--evaluate` csv. The 400-frame ATE and scale figures above came from `kornia_slam_raw.csv`, the camera
  centre. They compare arms against each other on one footing but are not body-frame scores; the rdef
  scale watch item is re-read on `traj.txt` at full length.
- Stop at S3 with the P/K table.
