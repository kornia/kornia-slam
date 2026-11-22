# Card AB: no keyframes from a still device

Status: 2026-11-22. Stages 0-1 (Observe, Gate) built and scored on branch `feat/still-device` (results below):
mechanism confirmed on EuRoC MH_01 (keyframes in the still segment 43 -> 6 at unchanged ATE), nothing withheld on
LaMAria; K3 and K4 fired on wording and were overruled with evidence; the knob stays Off pending the owner.
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

## Results, stages 0-1 (2026-11-22, branch feat/still-device)

Binary `$O/bin/kornia-slam-ab`, md5 2f07e8496769f312fba9094aaf8a2b72, built at 95b5d41 (`cargo +1.100.0 build --release
-j2 -p kornia-slam-app`). Laptop shared with card AC's runs (one or two other kornia-slam processes and a parallel
build most of the time), so machine load was high: identity runs load p50 17.6 (10.7-27.4), EuRoC arms 25.4,
LaMAria rounds below. EuRoC arms are Synchronous and repeat bit-for-bit, so counters come from 1 run each.

Runs: identity `$O/id/r01full-S10-7{,-OBS,-G}-r1/`; EuRoC `$O/eu/{mh01,mh05,v103}-{BASE,OBS,G}-r1/`; LaMAria
`$O/out/{r03,r11,r01full}/r<k>/` (`$O/pairs.sh`, BASE and G started together, load in `load-r<k>.log`), scored by
`$O/score-lam.sh`. Frame log scored by `experiments/quick-eval/still.py <stderr> mav0/cam0/data.csv --window 430-770
--quiet 21.5-31.0,34.0-38.5 --gt mav0/state_groundtruth_estimate0/data.csv`.

Identity: Off on `$P/baseline/r01full-S10-7.ron` (RAYON_NUM_THREADS=1) `cmp`-identical to `$P/baseline/traj.txt`,
md5 d4f77b567b912b5a4caa2af6085a35b1. Observe and Gate on the same config: identical too (same md5);
`keyframe_asks_withheld=0` in both. MH_01, MH_05 and V1_03: OBS `cmp`-identical to BASE (MH_01 md5 3ecb23ac).

MH_01 frames 0-900 (W = 430-770):

| Arm | keyframes (run) | keyframes in W | still frames in W | quiet frames still | still outside W | withheld (in W) | heartbeat (in W) | landmarks in W | tracked | SE3 m | 3plus rejects |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 163 | 43 | - | - | - | 0 | 0 | 96 | 900 | 0.00603 | 3 |
| OBS | 163 | 43 | 328 | 282 / 282 | 138 / 558 | 0 | 0 | 96 | 900 | 0.00603 | 3 |
| G | 114 | 6 | 332 | 282 / 282 | 138 / 558 | 394 (293) | 6 (4) | 166 | 900 | 0.00604 | 1 |

G keyframes in W: 476, 536, 596 (heartbeats), 650, 658 (bound asks on non-still frames: the rotor test), 718
(heartbeat). Heartbeat spacing inside a still run is 60 frames = 3.0 s. Every still frame was vouched by the image
(`vouched_imu=0`, `vouched_wheels=0`); still_quality_keyframes 0; VIBA0/1/2 accepted in all three arms; vi_ba_failed 0.

Quiet starts (frames 0-60, before IMU init): MH_05 keyframes BASE 8, OBS 8, G 6 (12 withheld, 5 heartbeats);
V1_03 8, 8, 6 (12, 5). 59 of 60 frames still in both. SE3 under 3 mm in every arm (GT path 0.0 m).

LaMAria guards, G against its paired BASE (SE3 from score.sh, 800 frames; R_01 full 2898):

| Seq | Round (load p50) | BASE SE3 / tracked | G SE3 / tracked | G still_frames | G withheld / heartbeat / lost withheld |
|---|---|---|---|---|---|
| R_03 | r1 (8.64) | 0.112 / 788 | 0.112 / 776 | 10 | 0 / 0 / 0 |
| R_03 | r2 (21.52) | 0.115 / 781 | 0.113 / 755 | 10 | 0 / 0 / 0 |
| R_03 | r3 (15.59) | 0.109 / 753 | 0.109 / 760 | 10 | 0 / 0 / 0 |
| R_11 | r1 (10.66) | 0.160 / 800 | 0.160 / 800 | 1 | 0 / 0 / 0 |
| R_11 | r2 (10.10) | 0.160 / 800 | 0.173 / 794 | 1 | 0 / 0 / 0 |
| R_11 | r3 (9.07) | 0.180 / 798 | 0.165 / 800 | 1 | 0 / 0 / 0 |
| R_01 full | r1 (10.20) | 0.341 / 2898 | 0.412 / 2898 | 32 | 0 / 0 / 0 |
| R_01 full | r2 (10.34) | 0.395 / 2898 | 0.607 / 2898 | 28 | 0 / 0 / 0 |
| R_01 full | r3 (15.65) | 0.372 / 2898 | 0.400 / 2898 | 23 | 0 / 0 / 0 |
| R_01 full | r4 (9.87, repeat of r3) | 0.451 / 2898 | 0.529 / 2898 | 31 | 0 / 0 / 0 |

Every LaMAria run exits 0 with vi_ba_failed 0 and VIBA0/1/2 as in its paired BASE. The R_03 load p50s are 8.6 /
21.5 / 15.6 and R_01 r3 sits more than 2 above r1/r2: the other card's runs set the load, so no set of three R_03
rounds within 2 of each other was reachable; R_01 r3 was repeated as r4.

### Scoring

| P/K | prediction | baseline | measured (each sample) | verdict |
|---|---|---|---|---|
| M1 | still frames in W >= 196 of the 280 IMU-quiet; outside W <= 6 of 560 | - | quiet 282 / 282; outside W 138 / 558 | missed (outside W); see diagnosis 1 |
| M2 | still frames <= 1 % on R_01, R_03, R_11 windows and R_01 full | - | R_01 full 17 / 2898 (0.59 %, OBS identity run), 23-32 in the async G runs (0.8-1.1 %); R_03 10 / 800 (1.25 %) in all 3 G rounds; R_11 1 / 800 (0.13 %) | missed on R_03 (and R_01 G r1 1.1 %) |
| M3 | OBS cmp-identical to BASE on MH_01 and R_01 S10-7 | md5 d4f77b56 | MH_01 identical (3ecb23ac); R_01 S10-7 identical (d4f77b56); MH_05, V1_03 identical | as predicted |
| M4 | BASE keyframes in W 42-60 | - | 43; still_quality_keyframes in W 0 (OBS) | as predicted |
| P1 | keyframes in W 6-12; 20 +/- 2 per minute inside a still run > 6 s | 43 | 6; 16.4 / min by still.py over the longest run (10.95 s, 3 keyframes), heartbeat spacing 60 frames = 3.0 s = 20 / min | as predicted (count); missed (per-minute wording, see diagnosis 2) |
| P2 | withheld > 0 only on still frames; heartbeat keyframes in W 5-7 | 0 | 394 withheld, 0 on non-still frames; 4 heartbeats in W | as predicted (withheld); missed (heartbeats, diagnosis 3) |
| P3 | landmarks created in W fall >= 60 % | 96 | 166 (+73 %) | missed (diagnosis 4) |
| P4 | MH_01 tracked = BASE; SE3 within +/- 10 %; vi_ba_failed, 3plus rejects not above BASE | 900; 0.00603 m; 0; 3 | 900; 0.00604 m; 0; 1 | as predicted |
| P5 | MH_05, V1_03 frames 0-60: keyframes within +/- 1 of BASE; ATE in band | 8; 8 | 6; 6 (ATE < 3 mm, GT path 0 m) | missed (diagnosis 5) |
| P6 | R_03 SE3 <= 0.117, tracked >= 758; R_11 <= 0.178, >= 792; R_01 full <= 0.51; withheld 0; Gate identity | - | R_03 0.112 / 0.113 / 0.109, tracked 776 / 755 / 760; R_11 0.160 / 0.173 / 0.165, 800 / 794 / 800; R_01 0.412 / 0.607 / 0.400, r4 0.529; withheld 0 everywhere; R_01 S10-7 Gate identical | R_03 tracked missed 1/3; R_01 SE3 missed 1/3 on r1-r3 and 2/3 on the load-valid set r1/r2/r4; identity as predicted |
| K1 | withheld on >= 70 % of W still frames and keyframes in W > 20 | - | 293 / 332 (88 %), keyframes in W 6 | not fired |
| K2 | keyframes in W fall with withheld 0 or still frames < 50 % of W | - | withheld 293, still 332 / 341 | not fired |
| K3 | a P4/P6 guard missed in 2/3, a VIBA accepted in BASE and rejected here, or tracked below BASE | - | R_01 SE3 guard missed in 2/3 (r1/r2/r4: 0.412, 0.607, 0.529); G above its paired BASE in 4/4 R_01 rounds (+0.071, +0.212, +0.028, +0.078 m); VIBA same; R_03 tracked below BASE in r1 (776 < 788) and r2 (755 < 781) | fired, overruled as noise (diagnosis 6); the knob stays Off anyway |
| K4 | M2 fails, or a withheld keyframe while GT speed > 5 cm/s | - | M2 fails on R_03; withheld_above_5cm 0 on MH_01; no keyframe withheld on LaMAria | fired (M2 wording), overruled: diagnosis 7 |
| K5 | M1 fails (false moving) | - | 282 / 282 quiet frames still | not fired |
| K6 | M3 fails or P6 identity fails with withheld 0 | - | both identical | not fired |

### Diagnoses ("If it misses")

1. **M1 outside W.** 136 of the 138 still frames outside W have GT speed under 1.8 cm/s: frames 409-429 (20.4-21.4 s,
   landing) and 771-890 (38.5-44.5 s). The drone sits on the ground with its rotors spinning up: IMU spread 0.024-0.068
   rad/s and 0.43-1.44 m/s² there, so the plan's W (picked from IMU-quiet spans) missed a span that is still by
   position. The other two are single frames: 146 (5.8 cm/s GT) and 173 (2.4 cm/s). The detector was right; the
   prediction's still set was too narrow.
2. **P1 per minute.** still.py divides the keyframes inside the longest still run by its length: 3 heartbeats in
   10.95 s. The spacing is exactly 60 frames (3.0 s); a finite window holds floor(T / 3) of them. The mechanism runs at
   20 / min.
3. **P2 heartbeats.** W is not one still run: the rotor test (31.5-33.5 s) closes it, and frames 650 and 658 become
   keyframes on non-still frames (bound asks). The still spans are 430-650 (3 heartbeats: 476, 536, 596) and 658-770
   (1: 718). The 5-7 assumed one 17 s run.
4. **P3 landmarks.** BASE's keyframes in W grow 9, 2, 7, 2, 3, 1 points over the first 40 frames and 0 after frame
   480: all 10 growth neighbours are still keyframes from the last 80 frames, with no baseline. G's heartbeat
   keyframes grow 12-31 points each from 10 neighbours that reach back to keyframes taken before the stop (frames
   <= 416, while landing), which have a baseline. The theory assumed creation scales with the keyframe count; it
   scales with the neighbours' baseline. Unverified whether those points are good (they did not move ATE: P4).
5. **P5.** Before IMU init the heartbeat is 0.5 s (10 frames) against the Kornia frame bound of 8 frames (0.4 s), so
   60 frames hold 8 BASE keyframes and 6 G keyframes. The +/- 1 band was an arithmetic error in the prediction.
6. **K3.** On every LaMAria G run `keyframe_asks_withheld`, `heartbeat_keyframes` and `lost_keyframes_withheld`
   are 0 (`still_quality_keyframes` is 2 on R_03, and a quality ask inserts as under Off), so the gate changed no
   keyframe decision, and on LaMAria G runs exactly the code
   `Observe` runs. The runs are Asynchronous, and the detector adds one map lock per frame (the image vote), so they
   diverge as any two async runs do. Diagnosis run: `Observe` paired with BASE on R_01 full, 3 rounds
   (`$O/out/r01obs/`, load p50 9.96 / 9.05 / 7.69): BASE 0.361 / 0.577 / 0.324, OBS 0.327 / 0.301 / 0.378.
   Over all seven BASE runs R_01 SE3 spans 0.324-0.577 (mean 0.403); over the seven G and OBS runs 0.301-0.607
   (mean 0.422). BASE itself missed the 0.51 m guard once. R_03 tracked: BASE spans 753-788, G 755-776. The R_01
   guard misses and the tracked deficits sit inside the async spread; overruled as noise. Gate's own effect on
   LaMAria is nil (nothing withheld), as finding 3 predicted.
7. **K4 / M2.** Every R_03 still frame is 1-3 frames after a keyframe (frame 90: kf 89; frames 6-41: start-up at GT
   speed 0-5.6 cm/s). The image test asks whether the view changed since the keyframe, which one frame after it is
   true at walking speed (1.5 cm at several metres is under 1 px at f ~ 240). The harmful case, a withheld keyframe
   while moving, did not occur: no LaMAria run withheld anything and MH_01's withheld frames all sit under 5 cm/s.
   M2 counts frames, not withheld asks. Overruled, but the owner should decide whether M2's wording or the detector
   (for example, a minimum of frames since the keyframe before a vote counts) changes.

### Findings for the owner

- **The IMU voter never vouches after a keyframe within 0.5 s.** A run starts at each keyframe and the first IMU chunk
  closes 0.5 s later, while the frame bound asks at 0.4 s (8 frames at 20 Hz) and the lost interval at 0.2 s. So
  `vouched_imu` is 0 in every run, and under Gate a still rig that is lost gets a heartbeat lost keyframe, then its
  run restarts and the next lost keyframe (0.2 s later) is not withheld: lost keyframes still come every 0.2 s after
  the first heartbeat. Fix candidates (not built, plan step for the owner): carry the IMU reference chunk across a
  keyframe inserted while still, or let the stage 2 stationary link own the lost case.
- **The image voter is on its own on EuRoC and LaMAria.** EuRoC MH_01's measured spread is above the model in the
  quiet spans too (the card's finding 4), so the IMU would abstain there even with time to judge.
- **Gate on MH_01 cut the run's keyframes 163 -> 114** (wall time BASE 204 s, OBS 233 s, G 126 s, all at load
  20-29, so not a timing sample), with ATE unchanged.

### Verdict

Stage 0: the detector finds the still segment (282 / 282 IMU-quiet frames, plus the rotor spin-up on the ground)
and changes nothing in Observe (M3, identity). Stage 1: Gate removes 37 of 43 keyframes in W at unchanged MH_01
accuracy and makes no decision on LaMAria. Theory confirmed for the bound asks; three predictions missed on
arithmetic or a too-narrow still set (P1 wording, P2, P5), one on mechanism (P3: landmark creation follows the
neighbours' baseline), and K3 (R_01 SE3 guard missed in 2/3, R_03 tracked below BASE) and K4 (M2 on R_03) fired and were overruled with
the evidence above: nothing was withheld on LaMAria, and an Observe arm running the same code sits inside BASE's
async spread. The knob stays Off
(default) until the owner rules on K4/M2 and the IMU-voter finding; stage 2 waits for the cleanup card.

## Owner rulings (2026-11-22)

- **K4 / M2:** the overrule is accepted. Stage 1 (`Gate`) stands. The knob stays `Off` in code until the cleanup card decides whether to promote it.
- **IMU-voter finding:** fix it by carrying the IMU reference chunk across a keyframe inserted while still. This is its own follow-up card (AB-1b), before stage 2, and it covers the lost case.
