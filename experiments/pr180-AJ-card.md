# Card AJ: per-frame pose-inertial solve on EuRoC (PR #195 port)

Status: card only. Code steps C1-C4 land on `feat/pose-inertial-aj`; no run is scored yet.
Baseline numbers are filled from `RESULTS.md` before the first knob-on run, and never after.
The A/B tables quoted from PR #195 are Astik's.

```
## Experiment: AJ, per-frame pose-inertial solve on EuRoC (PR #195 port)
Gap tested: G13, G30        Base: kornia-slam feat/global-pose-graph @ <sha after C4>   Binary tag: slam-aj
Mechanism: frame pose is visual-only and velocity is copied from the keyframe (G13). The solve is as good as
  the anchor keyframe's velocity and bias (card H; PR #195 b32474e: joint 3.14 m at window 3, 0.81 m at 10).
Theory: ORB-SLAM3 TrackLocalMap with PoseInertialOptimizationLastFrame/LastKeyFrame. A 10-keyframe VI-BA
  constrains the anchor state, so the IMU edge stops gating good matches out.
Knob: tracking.pose_inertial (default Off) with mapping.local_ba.active_keyframes (default 3)
Arms, stereo+IMU, on the baseline `default` config: D (baseline), DJ (+pose_inertial: OrbSlam3),
  D10 (+active_keyframes: 10), D10J (both). Mono+IMU: M, MJ, M10, M10J (sensors: (cameras: Mono, imu: true)).
Baselines: /home/edgar/dev/split/.pr180-fix/euroc/RESULTS.md (D); the other controls are run here.
Predictions:
  P1 mechanism, D10J and M10J: pose_inertial_solves >= 90 % of tracked frames after IMU init;
     keyframe_anchored / solves 0.15-0.35; prior_dropped <= 1 % of solves; rejects <= 1 % of frames.
  P2 imu_chi2 on solved frames: p50 0.1-5, p99 <= 100.
  P3 stereo scale: |s-1| <= 0.01 on all 11 in every arm; J moves it by <= 0.005.
  P4 stereo ATE SE3, D10J vs D10: -30 % to +5 % per sequence; mean over 11 -5 % to -20 %.
  P5 stereo RPE 1 s, D10J vs D10: -20 % or more on >= 6 of 11.
  P6 window dependence: DJ rejects >= 3x D10J rejects; DJ mean ATE +10 % to +80 % against D.
  P7 tracked share, D10J vs D10: within +/-0.5 points on every sequence.
  P8 frame_ms p50: +0.2 to +1.5 ms (not PR #195's -21 ms: the visual stage still runs).
  P9 mono scale, M10J vs M10: where M10 has |s-1| > 0.05, |s-1| at least halves; elsewhere it stays <= 0.03.
  P10 mono ATE Sim3, M10J vs M10: MH_03 and MH_04 -30 % to -75 % if P9's condition holds there;
      MH_01, MH_05, V1_03 -10 % to +15 %.
  P11 LaMAria R_01, 800-frame window, window 10, flow_pose: RansacPnp, knob on vs off, 3 runs each:
      tracked >= 98 %, scale 0.97-1.03, ATE within +/-0.02 m of the control.
Kill rules:
  K1 P1 and P2 hold but stereo mean ATE moves less than -5 % -> the per-frame pose is not where stereo
     EuRoC error enters; it is in BA and bias (card AH). Knob stays Off.
  K2 ATE improves >= 10 % while solves < 50 % of frames or keyframe_anchored > 80 % -> the gain is the
     velocity carry, not the joint pose. Run LastKeyframe to separate them.
  K3 rejects > 1 % of frames or tracked share down > 0.5 points on any D10J sequence -> G30 reproduces
     with a rectified IMU. Stop and diagnose (card H next step: IMU information scale).
  K4 imu_chi2 p99 > 100 -> IMU information or T_bc is wrong. Stop.
  K5 no mono sequence with |s-1| > 0.05 halves -> PR #195's scale gain does not transfer to this base.
  K6 P11 fails -> the knob stays Off in every LaMAria preset; it is not general purpose until G30 is fixed.
Bit-identity: /home/edgar/dev/split/.pr180-fix/baseline/r01full-S10-7.ron, RAYON_NUM_THREADS=1, cmp-identical
  to /home/edgar/dev/split/.pr180-fix/baseline/traj.txt (md5 d4f77b567b912b5a4caa2af6085a35b1).
Noise: asynchronous runs differ run to run (V2_02 0.14-0.54 m on one binary). 1 run per arm for the
  screen; 3 runs on every sequence that decides a P or K. Record load; discard timing if load differs by > 2.
```
