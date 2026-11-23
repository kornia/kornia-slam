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

## Results 2026-11-23: diagnosis of the rdef scale bias and ATE gap (read-only)

Question: why does `cameras: Rig` (rdef) score worse than Stereo (sdef) on the Machine Hall sequences, with a
Sim3 scale of 1.01-1.04? Inputs: the synchronous sweep runs `$E/sync/<arm>_<seq>_s0_r1` (smoothed body-frame
trajectories; these runs have no session dump) and the 400-frame `$E/q400/*_r2/session.cbor` dumps. Scripts live
in the session scratchpad, not in the repo. No new run: the sweep held both kornia-slam slots the whole time.

### Verdict
The rig's overlap depth is not biased. The rig BA holds the metric scale less firmly than Stereo, so the map
and the trajectory shrink together by about 1.3 %. The overlap match of a primary keypoint that is already
tracked never becomes a camera-1 observation of its landmark, so a rig keyframe carries half as many metric
rows as a Stereo keyframe, and each row is noisier. The default 3-keyframe VI-BA window (G6) then leaves scale
to the other terms. On MH_02 and MH_03 scale accounts for the whole ATE gap. MH_04 and MH_05 also lose
accuracy that scale does not explain.

### Evidence
1. The gap on MH_02/03 is scale only (smoothed, eval.py SE3 / Sim3 RMSE in metres, then the scale):

   | seq | sdef | rdef | rp48 |
   |---|---|---|---|
   | MH_01 | 0.045 / 0.026 / 1.009 | 0.047 / 0.027 / 1.009 | 0.034 / 0.031 / 1.003 |
   | MH_02 | 0.035 / 0.035 / 1.001 | 0.057 / **0.029** / 1.011 | 0.037 / 0.030 / 0.995 |
   | MH_03 | 0.041 / 0.038 / 1.004 | 0.065 / 0.040 / 1.015 | 0.047 / 0.046 / 0.998 |
   | MH_04 | 0.226 / 0.191 / 1.018 | 0.552 / 0.470 / 1.044 | 0.437 / 0.406 / 1.024 |
   | MH_05 | 0.082 / 0.082 / 1.001 | 0.122 / 0.122 / 1.000 | 0.106 / 0.106 / 0.999 |

   With scale removed, rdef beats sdef on MH_02 and ties on MH_01/03. MH_01 and MH_05 show no scale
   difference, so "biased on every sequence" does not hold. The scale loss hits MH_02/03/04. MH_04/05 also
   have a shape loss.
2. The scale error is a level offset, not drift. Sim3 scale per 20 s window (Umeyama, smoothed trajectory against
   GT): rdef 1.013 / 0.998 / 1.014 / 1.008 / 1.012 / 1.012 / 1.015 on MH_01 (after the take-off window), 1.007-1.025
   on MH_02, 1.012-1.020 on MH_03. sdef sits at 1.001-1.010 and rp48 at 0.995-1.004. rp48 is a rig too, and its scale
   is better than sdef's on MH_01-03. The rig geometry is not the fault.
3. H1 (overlap depth biased by the missing sub-pixel step) is refuted as the cause of the bias. The test matched the
   same scene points in the rdef and sdef sessions (ray matching with a fitted rectifying rotation, median
   0.08 deg) and re-triangulated the rig pairs by DLT from the dumped undistorted observations:
   - KF0, raw overlap depth over raw stereo depth: median 0.986, mean 1.026, IQR 0.938-1.044 (n 135).
   - Early keyframes: 0.995 (n 410). Keyframes after 5 s: 1.0006 (n 481).
   Wrong sign, too: the E[1/d] > 1/E[d] effect makes depth too large, which gives a Sim3 scale below 1, and the
   runs show a scale above 1. The high tail (mean above the median) is that effect, but BA follows the median.
   What the missing sub-pixel step costs is noise. Raw overlap over BA depth has an IQR of 0.913-1.058 at KF0 and
   0.978-1.027 over 400 frames. Stereo's raw over BA is 0.988-1.010 at KF0 and 0.987-1.008 over 400 frames.
   That is 2.5-6x the stereo spread.
4. The BA shrinks the rig map below its own pair depths, and Stereo's BA does not. Same matched points, keyframes
   after 5 s (`q400` r2):
   - rig raw over stereo raw: 1.0006.
   - rig BA over stereo BA: 0.9878 (0.9859 for points nearer than 1.5 m, 0.9944 beyond).
   - rig raw over rig BA: 1.0065. Stereo raw over stereo BA: 1.0000.
   Over all cam0-observed landmarks of those keyframes, the rig map ranges are 0.983-0.985 of Stereo's. That holds
   for landmarks paired in the keyframe, for landmarks seen by cam1 elsewhere, and for cam0-only landmarks alike, so
   it is a uniform scale. At KF0 the maps agree (0.996). The 400-frame trajectory scale is 1.029 against 1.009, a 2 %
   gap that matches the map.
5. H6 (mono rows only) is confirmed. Metric rows per keyframe, from the q400 sessions:
   - rdef: 109 landmarks with cam0 and cam1 observations in the same keyframe (median), out of 486 cam0 and 470
     cam1 observations. rp48: 127.
   - Stereo depth rows: sdef 239 of 363, sp48 261 of 405.
   The frontend triangulates 316 overlap pairs per ORB frame (P4). But `frame.rig.left_to_right` is read only by
   `rig_seeds` (`mapping/growth.rs:76-118`), for features that are still unclaimed (`mapping/keyframe_mapping.rs:122-127`
   and the bootstrap, `initialization/bootstrap.rs:111-112`). A tracked primary keypoint's overlap partner is
   dropped. Its landmark gets a camera-1 row only when the tracker's own camera-1 projection search happens to
   pick the same landmark. Growth then runs per camera (`growth.rs:120-200`), which builds duplicate single-camera
   landmarks. Full runs: 73k active landmarks for rdef against 27k for sdef. These duplicates carry no metric
   information beyond the 0.11 m lever arm.
   The reference does it differently, as I recall it (line not checked): ORB-SLAM3's non-rectified stereo assigns
   a tracked map point to the keypoint's `mvLeftToRightMatch` partner as well.
6. The BA window makes it worse. rdef and sdef run the default VI-BA: 3 active keyframes, 50 iterations, λ 1.0
   (`vi_ba_base_solves` 782 on rdef MH_01). rp48 solves a 10-keyframe large window every time
   (`large_window_solves` 767). A 10-keyframe solve sees about 10 x 127 metric pairs; a default solve sees
   3 x 109. Stereo's 3 x 239 depth rows hold the scale, and the rig's 3 x 109 do not. This is G6's mechanism on a
   second sensor setup. rp48 also adds KLT sub-pixel tracks in both cameras, so the window and the sub-pixel tracks
   are confounded in rp48-vs-rdef.
7. H2 (calibration) is refuted. `$E/mh01-rig.ron` has cam1 `primary_from_camera` = inv(T_BS cam0) * T_BS cam1, recomputed
   from `sensor.yaml`. Intrinsics and radtan match `sensor.yaml` for both cameras. `camera_to_body` = cam0 T_BS.
   The session dumps write the same baseline for both modes (0.110078 m). Stereo uses fx 457.418 on the rectified
   pair. The rig uses 458.654 / 457.587 per raw camera, which is consistent.
8. H3 (undistortion mismatch after G52) is refuted. `ensure_undistorted` undistorts each feature with its own camera
   (`frame.rs:138-156`). The overlap F, P and reprojection gate use the ideal pinhole (`overlap.rs:93-105`;
   `reprojection_error_sq_cam` ignores distortion). The dumped BA observations are undistorted pixels (u from -83
   to 794). The rdef KF0 BA residual with a no-distortion projection is flat across the radius: median 1.44 / 1.21
   / 1.06 / 1.26 px for r_n < 0.2 / 0.2-0.4 / 0.4-0.6 / > 0.6. A model mismatch would grow like r³ (about 16 px at
   r_n 0.5).
9. H4 (IMU scale) is partly tested. VIBA0 accepts `scale=1.0000` and VIBA1 `scale_correction=1.0000` in rdef, sdef
   and rp48 alike (q400 err.log), so rig mode does not accept a different initial scale. Which term pulls the
   loosely held scale toward a short estimate is not identified. sdef's own scale is at or above 1.000 on all 5 MH
   sequences, which points to a pull both modes share that Stereo's depth rows mostly resist. A `--debug` run is
   needed to read per-solve scale.
10. H5 (epipolar band, wrong matches) is minor. The disparity excess of rig over stereo (px, f·b(1/z_rig - 1/z_st))
    has a median of +0.04 px overall. Mixed-octave pairs show opposite signs that cancel (1/2: +0.47, 2/1: -0.63).
    Far points (stereo range > 6 m, n 136) come out 4-7 % short, which fits mismatches at small disparity: along
    the epipolar line the 2 px reprojection gate cannot reject them (`overlap.rs:226-240`). There are too few of
    them to move the median.

### Ranked causes
1. Weak metric anchor in the rig BA (G61, new; G57 on feat/euroc-rig): tracked keypoints' overlap partners are dropped, so a keyframe has
   109-127 cross-camera pairs against Stereo's 239-261 depth rows. Combined with the default 3-keyframe window (G6),
   scale floats by 1-2 % on MH_02/03 and 4 % on MH_04. Evidence: 4, 5, 6, 2.
2. Pair noise without a sub-pixel step: 2.5-6x the stereo depth spread. It adds to the weak anchor but has the wrong
   sign to cause the bias (3).
3. MH_04/05 accuracy beyond scale (Sim3 ATE 0.470 against 0.191, and 0.122 against 0.082) is not explained here.
   Candidates: dark frames with fewer overlap pairs, and far-point mismatches (10). Open.
Refuted: H2, H3, H1 as a bias. H4: initial scale is the same in both modes; the pulling term is open.

### Next rung (cheapest first; nothing runs until two kornia-slam slots are free)
R-a, config only: rdef with `local_ba: ( active_keyframes: 10 )`, nothing else changed, synchronous, MH_02 and MH_03
full length, 2 runs each (start_frame 0/1).
- Predictions: Sim3 scale 1.011 to 1.000-1.006 on MH_02, 1.015 to 1.000-1.008 on MH_03; SE3 ATE 0.057 to 0.035-0.045,
  0.065 to 0.040-0.050; per-window scale at or below 1.008.
- Kill: scale stays above 1.009 on both. Then the window is not the lever; go to R-b directly.
- Cost: about 2 x 2 x 300 s.

R-b, code fix sketch: at keyframe insertion (`mapping/keyframe_mapping.rs:106-116`), for each accepted tracked claim
(landmark L, primary feature i) with `left_to_right[i] = Some(j)`, an overlap point and j unclaimed, push
`ObservationLink { (kf, j) -> Existing(L) }`. Optionally do the same in tracking's pose rows. This is ORB-SLAM3's
left-to-right propagation. Run it with the default window, so the result separates from R-a.
- Predictions, 400 frames MH_01: metric pairs per keyframe 109 to at least 220; rig raw over rig BA (late keyframes)
  1.0065 to within ±0.003; Sim3 scale (`traj.txt`) 1.029 to 1.008-1.018.
- Predictions, full length: Sim3 scale MH_02 and MH_03 at or below 1.007; SE3 ATE rdef/sdef at or below 1.15 on MH_02
  and MH_03; MH_05 unchanged within ±10 % (no scale loss there to recover); LaMAria R_01 B1 not bit-identical (the
  rig path changes) but inside its noise band.
- Kill: pairs per keyframe double but the 400-frame scale stays above 1.022. Then the anchor count is not the lever,
  and the pull term (H4) needs the `--debug` per-solve scale read first.

The order also follows the owner rules: R-a is a 1-knob config test with no code. R-b changes the rig path, so it
needs the full-sequence regression gate before any promotion. Sub-pixel overlap refinement (K4's first suspect) drops
to third. It would cut the noise in cause 2, but it cannot fix a scale that BA does not hold.
