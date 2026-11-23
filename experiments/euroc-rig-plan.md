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
