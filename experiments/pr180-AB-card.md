# Card AB: no keyframes from a still device

Status: 2026-11-22. Card written; stages 0-1 (Observe, Gate) to be built and scored on branch `feat/still-device`.
Stage 2 (StationaryEdges) waits for the cleanup card. Plan: planner (Fable), owner answers 2026-11-22. Ledger row G45.

Paths: `P=/home/edgar/dev/split/.pr180-fix`, `O=$P/cardAB`, `Q=experiments/quick-eval`.

Owner answers (2026-11-22): orin01 runs cu-xlerobot-slam over `SlamSystem`, so the fix reaches the robot through
`keyframes.stillness` in its config (no robot change in this card). Stages 0-1 now; stage 2 after the cleanup card.
Port c95203b and replace its hand-set gates; heartbeat 3 s / 0.5 s in stage 1, no tail edge until stage 2; LaMAria is
a guard only; a vibrating rig in the dark without wheels still adds lost keyframes. P8 (stage 2) runs on a robot
capture the owner records (10 min parked from boot, then a 1 min drive; stereo, IMU, wheels), replayed here.

## Findings behind the predictions

Numbered as in the plan (findings 1 and 2 are about the c95203b port and the robot stack, covered above).

3. **The LaMAria start-up is not still by any sensor model.** GT speed is under 3 cm/s for 1-2 s, but the head rotates: gyro spread 0.06-0.18 rad/s against 0.0096 predicted from the noise density (6-19x), and net rotation 7-85 mrad per 8-frame interval (about 2-20 px at f near 240 px). R_11's GT stops (187-194 s, 306-308 s) read the same (3-10x). So this card predicts zero effect on LaMAria, which is a guard only. Card K's baseline gate stays the fix for the mm-baseline pairs. In the R_01 identity run the first keyframes are at frames 0 2 4 7 15 23 30: re-seeds and the ratio rule, not the frame bound.

4. **A real still segment exists on disk: EuRoC MH_01_easy frames 430-770.** IMU-quiet at 21.5-31.0 s and 34.0-38.5 s, with a rotor test on the ground at 31.5-33.5 s. Measured spread is 0.004-0.007 rad/s and 0.06-0.08 m/s² against 0.0042 and 0.049 predicted. Net rotation per 0.4 s after bias removal is 0.06-0.40 mrad against 0.19 predicted. The gyro bias there is 0.082 rad/s, which is why the hand-set 0.02 gate never fired. Shorter quiet spans: MH_05 0-3 s, V1_03 0-3 s, MH_03 10.5-13 s, V1_01 144.5-146 s. V1_01's start is not quiet (accel spread 0.2-0.8).

## Experiment: AB, no keyframes from a still device
Gap tested: G45 (new: keyframes accumulate while still)   Base: kornia-slam@28f03b6 (pr180c/F1 81bc775 + docs); built on 072e2bf (docs only since)
Binary tag: cardAB (md5 at build)
Mechanism: KeyframeRules::Kornia inserts at max_frames_between whatever the motion (tracking/policy.rs:170-172);
  the time bound (:195-205), the OrbSlam3 0.25/0.5 s rules (:218-220, :233) and the lost-interval rule
  (system/mod.rs:2220-2229, policy.rs:288) do the same. Each keyframe adds stereo seeds
  (mapping/keyframe_mapping.rs:118-128), an IMU edge, a BA job and a place entry. The parked fix c95203b was
  never ported; set_map_bound (system/mod.rs:401) then evicts the oldest keyframes of a parked rig.
Theory: a keyframe taken while the image motion against the last keyframe is inside pixel noise adds no view;
  withholding it loses nothing but the IMU edge bound, which a heartbeat keeps (stage 1) or a stationary factor
  replaces (stage 2). Test: OpenVINS UpdaterZeroVelocity.cpp:150-152, :206-214, :228-231, :256 with its hand
  multipliers replaced by sigma_px (visual_information.rs:22) and by max(model, measured) IMU variance.
  Gate: Basalt sqrt_keypoint_vio.cpp:370-373, VINS feature_manager.cpp:95-117, OKVIS2 Frontend.cpp:1147-1255.
  Heartbeat bound: ORB-SLAM3 LocalMapping.cc:1022-1024 (3 s initialised, 0.5 s before). ORB-SLAM3's own rule
  (Tracking.cc:3166-3179) keeps 1 keyframe per 3 s forever and is not followed.
Knob: keyframes.stillness: Off (default = today) | Observe | Gate | StationaryEdges.
  Class: bug fix (a reference mechanism the pipeline lacks); the only path once validated.
  Inside it, no per-deployment number: confidence 99.9 % (design constant), chunk 0.5 s (design constant,
  cu-kornia-vio parked.rs:73), 20 features (reference constant, OpenVINS :231), heartbeat 3 s / 0.5 s
  (reference constants, ORB-SLAM3), sigma_px and IMU densities (sensor model). Existing, not changed here:
  wheel floors PARKED_PATH_M 5 mm / PARKED_PATH_RAD 0.01 (system/wheel.rs:439-442), a real tunable to derive
  from encoder quantisation later.
Arms: BASE (Off); OBS (Observe: detector and counters, no suppression); G (Gate); SE (StationaryEdges, stage 2).
Still set: EuRoC MH_01_easy frames 0-900, window W = 430-770 (IMU-quiet 21.5-31.0 s and 34.0-38.5 s, rotor
  test between); stereo + IMU, Synchronous, loop closing off, so counts are exact.
Baselines: MH_01 BASE on the cardAB binary, paired every round (no F1 EuRoC run exists).
  LaMAria F1 ($S/F1/out, card AA): R_03 SE3 0.104-0.117 m, tracked 758-778; R_11 0.157-0.178 m, 792-800;
  R_01 full 0.30-0.51 m (n = 12). Identity run: 2898 frames, 545 keyframes.
Predictions, stage 0 (OBS):
  M1 still_frames inside W >= 196 (70 % of the 280 IMU-quiet frames); outside W <= 6 (1 % of 560).
     measured by: [summary] stillness + still.py against the IMU-quiet spans
  M2 still_frames <= 1 % of frames on R_01, R_03, R_11 (800-frame windows) and on R_01 full.
  M3 OBS trajectory cmp-identical to BASE on MH_01 and on R_01 S10-7 (md5 d4f77b567b912b5a4caa2af6085a35b1).
  M4 BASE keyframes inserted in W: 42-60 (340 / 8 plus ratio keyframes); still_quality_keyframes in W recorded.
Predictions, stage 1 (G):
  P1 keyframes inserted in W: M4's value -> 6-12; inside a still run longer than 6 s, 20 +/- 2 per minute
     (today 150 at 20 fps).                                           measured by: still.py on the frame log
  P2 keyframe_asks_withheld > 0 only on still frames; heartbeat_keyframes 5-7 in W.
  P3 landmarks created in W fall by >= 60 % against BASE.             measured by: [kf_stereo] close_points + grown
  P4 MH_01 900 frames: tracked equal to BASE; ATE SE3 within BASE +/- 10 %; vi_ba_failed and
     vi_ba_solves_3plus_rejects not above BASE.
  P5 quiet starts (MH_05 and V1_03 frames 0-60, before init): keyframes within +/- 1 of BASE (heartbeat 0.5 s
     against the 0.4 s bound); ATE inside the paired band.
  P6 LaMAria guards, 3 rounds: R_03 SE3 <= 0.117 m and tracked >= 758; R_11 SE3 <= 0.178 m and tracked >= 792;
     R_01 full <= 0.51 m; keyframe_asks_withheld = 0; and R_01 S10-7 with Gate on is cmp-identical to the
     baseline trajectory.
Predictions, stage 2 (SE, scored before it is enabled anywhere):
  P7 live map keyframes at frame 770 minus at frame 430: <= 3 (BASE 42-60); stationary_edges >= 1, tail_edges >= 1.
  P8 parked from boot, 30 min (8097 sim or the robot): live keyframes within +/- 5, evictions 0, pose rate at
     minute 30 within 5 % of minute 1 (c95203b reported K 3-5 and 1.04x).
  P9 guards as P4 and P6.
  Mechanism counter: still_frames, keyframe_asks_withheld, heartbeat_keyframes, stationary_edges.
Kill rules:
  K1 (mechanism moved, score did not) keyframe_asks_withheld > 0 on >= 70 % of W's still frames and keyframes
     in W still above 20: the frame bound is not the main source. Read still_quality_keyframes, give_ups and
     lost_keyframes_withheld; next is the ratio rule (card Z's unit mismatch, G33) or the re-seed loop.
  K2 (score moved, mechanism did not) keyframes in W fall with keyframe_asks_withheld = 0 or still_frames under
     50 % of W: tracking was lost or frames were skipped. Do not credit; compare tracked and give_ups.
  K3 (harm) any P4 or P6 guard missed in 2/3, a VIBA accepted in BASE and rejected here, or a tracked count
     below BASE: the knob stays Off.
  K4 (false still) M2 fails, or still.py finds a withheld keyframe while GT speed is above 5 cm/s: the
     detector is unsound. Stop before stage 1.
  K5 (false moving) M1 fails: read runs_closed_* by voter. Do not add a multiplier; fix the model (KLT against
     ORB pixel noise, or the chunk statistic).
  K6 (side effect) M3 fails, or P6's identity fails with keyframe_asks_withheld = 0: Observe or Gate changes
     state on a frame it did not act on. Bisect the hooks.
Bit-identity: Off: $P/baseline/r01full-S10-7.ron, RAYON_NUM_THREADS=1, cmp with $P/baseline/traj.txt
  (md5 d4f77b567b912b5a4caa2af6085a35b1). Observe on the same config: identical too (M3).
Noise: Synchronous runs repeat bit-for-bit, so counters need 1 run. ATE guards on the async presets: 3 rounds,
  each arm started with its BASE (JOBS=2); a round whose load p50 is more than 2 from the others is repeated.
If it misses: (1) M1 low: [still] lines give the closing voter, statistic and threshold per run.
  (2) K1: split keyframes in W by ask (bound, quality, lost, seed) from the frame log.
  (3) P4 ATE: imuedges.py on the heartbeat edges (chi2 against dt); if the 3 s edges carry the tail, test
  heartbeat 0.5 s as a diagnostic only. (4) P6 identity: list the first frame whose [still] state is not moving.

New `[summary] stillness` line: `still_frames still_runs keyframe_asks_withheld heartbeat_keyframes still_quality_keyframes lost_keyframes_withheld runs_closed_imu runs_closed_image runs_closed_wheels vouched_image vouched_imu vouched_wheels stationary_edges stationary_merged tail_edges`.

Data: EuRoC `/home/edgar/data/euroc/MH_01_easy` (also MH_05, V1_03). LaMAria `/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data`. Robot: `kornia-slam-app`'s `Mcap` source has no IMU (`apps/kornia-slam-app/src/source/mcap.rs:63`), so no robot bag replays here. P8 runs on the 8097 sim or the robot. I did not check `/home/edgar/blackbox_20261019T090607Z_fusion_vision_rejected.mcap` for a parked span.
