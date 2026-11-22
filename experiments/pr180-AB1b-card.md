# Card AB-1b: the IMU still run survives a keyframe inserted while still

Status: 2026-11-22. Card written before code. Plan: planner (Fable), owner answers 2026-11-22: Q1 accept (no multiplier; price the robot's parked capture before stage 2),
Q2 the carried vote comes from the newest chunk, Q3 lag tolerance in `quiet_at` in this card (own commit, unit test),
Q4-Q7 as recommended. Ledger row G45.

Paths: `R=/home/edgar/dev/split/kornia-slam-still`, `P=/home/edgar/dev/split/.pr180-fix`, `O=$P/cardAB`,
`O1=$P/cardAB1b`, `Q=$R/experiments/quick-eval`.

## Experiment: AB-1b, the IMU still run survives a keyframe inserted while still
Gap tested: G45 (lost-interval part; the IMU voter)   Base: kornia-slam@3327238   Binary tag: cardAB1b (md5 at build)
Mechanism: StillnessState::sync builds a new StillRun whenever the last keyframe changes
  (system/stillness.rs:101-111, :106). A new run has no reference and votes Pending (inertial/still.rs:168-177);
  its first chunk closes RUN_CHUNK_SEC = 0.5 s later (:19, :219) and the IMU vouches only when Quiet (:199-201,
  stillness.rs:183). The frame bound asks at 0.4 s (tracking/policy.rs:170-172) and the lost interval at 0.2 s
  (policy.rs:361, system/mod.rs:2226-2228, :3067-3076), and each insert moves the last keyframe (mod.rs:3217-3218).
  So after any keyframe the IMU is Pending when the next ask comes: vouched_imu = 0 in every card AB run, and
  under Gate a lost still rig gets one heartbeat lost keyframe, then one every 0.2 s. The frame's prune
  (mod.rs:2255-2257) also drops the samples before the keyframe, so a run kept as it is would close on a hole.
Theory: rest is a property of the device, not of the keyframe. Constancy against one reference chunk is
  transitive, so chunks before and after a keyframe that both match the reference match each other; with the
  samples kept from the judged end, no interval is unjudged. A keyframe inserted on a frame judged still
  therefore keeps the reference and the judged end. What is measured against the keyframe restarts: the gyro
  view rotation, the non-IMU closure, and the vote (from the newest chunk, as a restart cleared Unknown before).
  Reference: OpenVINS UpdaterZeroVelocity.cpp:93-97 (the IMU window runs from the last state to this image and
  no clone resets it), :241-243 (reset only when the test fails), :327-328 (the zero-velocity state is carried
  over consecutive frames); master, fetched 2026-11-22. It beats cu-kornia-vio parked.rs:40 (restart at each
  keyframe, sound at its 30 s refresh, not at 0.2-3 s) and Basalt sqrt_keypoint_vio.cpp:370-373 / VINS
  feature_manager.cpp:95-117 / ORB-SLAM3 Tracking.cc:3166-3179, which decide from the image only and are blind
  when lost.
Knob: none new. The change lives inside keyframes.stillness Observe | Gate; Off (default) builds no run.
  Base arm = the card AB binary ($O/bin/kornia-slam-ab, md5 2f07e849, code-identical to 3327238).
Thresholds: no new number. Unchanged and from the sensor model: per-axis var(mean) = max(density^2 * rate,
  measured) / n; quiet band 1 + 3.09 * sqrt(2 / (3 (n - 1))) = 1.254 at n = 100; chi2_3(99.9 %) = 16.27; view
  bound from sigma_px. Expected cost of an unbounded reference, from the model: test false alarms 0.2 % per
  chunk = one false close per 250 s of rest (1 keyframe each; 1 per 300 s before); gyro bias walk
  1.9393e-5 * sqrt(T) reaches the mean-difference sigma 3.4e-4 rad/s at T = 307 s. The yaml accel walk (3.0e-3)
  would reach it at 1.8 s, which MH_01 Gate contradicts (0 closures over 386 judged frames), so it is not used
  to predict. Each closing line logs ref_age so stage 2 can price OpenVINS's Q_bias term (:194-196).
Offline pricing (imuquiet.py, 0.5 s chunks): MH_01 21.5-31.0 s and 34.0-38.5 s 0 / 28 chunks quiet (accel
  variance 1.35-10.5x the model); MH_05 0-3 s 0 / 6; V1_03 0-3 s 6 / 6 (0.62-0.97x). All 11 EuRoC sequences:
  V1_03 0-3 s is the only in-band span >= 1.5 s. So the IMU can vouch on V1_03's start and in the system test,
  and is predicted not to on MH_01 (it reads Unknown: constant mean, spread above the model).
Baselines: $O/eu/{mh01,mh05,v103}-{BASE,OBS,G}-r1 (Synchronous, bit-for-bit); $P/baseline/traj.txt
  (md5 d4f77b567b912b5a4caa2af6085a35b1); blackout runs paired with the old binary; LaMAria BASE paired, pooled
  with card AB's (R_01 full 0.324-0.577 m, n = 7; R_03 0.109-0.115 m, tracked 753-788; R_11 0.160-0.180 m).
Predictions:
  T1 system test, rig at rest, lost from frame 21 to 140 (20 fps, Gate, settled): lost keyframes
     60, 64, 68, ... (>= 16; 5 / s after the first heartbeat) -> exactly 60 and 120 (0.33 / s); vouched_imu
     0 after frame 60 -> >= 110; runs_carried 2; runs_closed_imu 0.            measured by: cargo test
  P1 MH_01 Observe, W = 430-770: frames with imu=Pending 341 / 341 -> <= 51 (15 %); imu=Unknown >= 273 (80 %);
     runs_carried in W 36-43.                                       measured by: still.py on the frame log
  P2 V1_03 Observe frames 0-59: imu=Quiet 0 -> 40-50; runs_carried 5-7. MH_05 Observe: imu=Unknown 0 -> 40-50,
     Quiet <= 5.
  P3 V1_03 Gate with --blackout 1.05 | 1.10 | 1.15 : 0.35 (7 black frames inside 0.5 s of the heartbeat at
     frame 20): vouched_imu 0 -> 5-7 in each; give_ups equal to the old binary.
  P4 MH_01 Gate with --blackout 24.0:3.0: lost_keyframes_withheld 0 -> 0, vouched_imu 0 -> 0, lost keyframes in
     the blackout equal to the old binary (13-15 expected). The IMU reads Unknown there.
  P5 MH_01 Gate, no blackout: keyframes in W 6 -> 6-8, heartbeats in W 4, tracked 900, SE3 0.00604 m +/- 10 %,
     vi_ba_failed 0, runs_closed_imu 0 -> 0-3, runs_carried 6-8; trajectory cmp-identical to $O/eu/mh01-G-r1
     when runs_closed_imu = 0. MH_05 and V1_03 Gate: keyframes 6 -> 6.
  P6 LaMAria guards (R_03, R_11 800 frames, R_01 full), Gate against paired BASE: keyframe_asks_withheld,
     heartbeat_keyframes, lost_keyframes_withheld, vouched_imu all 0; SE3 R_03 <= 0.117 m, tracked >= 753;
     R_11 <= 0.180 m, tracked >= 792; R_01 <= 0.61 m.
  Mechanism counter: runs_carried (new), vouched_imu, the imu= state per frame, lost keyframes per second
     while still.
Kill rules:
  K1 (mechanism moved, score did not) runs_carried > 0 and P2 holds, but P3's vouched_imu stays 0 or T1 still
     shows a lost keyframe every 0.2 s: the reference is not what blocks the vouch. Read the black frames'
     [still] lines: closed by=imu_hole (the prune), or t beyond the judged end + 0.5 s (sample lag). Fix that,
     not a threshold.
  K2 (score moved, mechanism did not) lost keyframes or keyframes in W fall with runs_carried = 0 or
     vouched_imu = 0: the blackout did not apply, tracking never lost, or a give-up. Do not credit.
  K3 (harm) P5 misses on keyframes in W, tracked, SE3 or vi_ba_failed; or a LaMAria decision counter is above 0
     with SE3 above the pooled BASE range in 2 / 3 rounds: revert the carry.
  K4 (false still) still.py withheld_above_5cm > 0 on any EuRoC run, or vouched_imu > 0 on a LaMAria run: the
     carried Quiet vouches for a moving device. Stop; the vote at the carry is unsound.
  K5 (false moving) runs_closed_imu in MH_01's W above 3 under Observe: the reference ages faster than the
     noise test allows. Read ref_age on the closing lines; next is the Q_bias term, not a multiplier.
  K6 (side effect) Off not cmp-identical on any run, Observe not identical to BASE, or Gate on R_01 S10-7 not
     d4f77b56 with the decision counters at 0: the prune floor or the carry changes state. Bisect, prune first.
  K7 (pricing wrong) P4 shows lost_keyframes_withheld > 0: a chunk read Quiet where imuquiet.py says none can.
     Compare the chunk phases before crediting it.
Bit-identity: Off: R_01 S10-7 (RAYON_NUM_THREADS=1) against $P/baseline/traj.txt; MH_01, MH_05, V1_03 against
  $O/eu/<seq>-BASE-r1/traj.txt; the MH_01 blackout Off run, new against old binary. Observe: the same four.
  Gate: R_01 S10-7 only.
Noise: Synchronous runs repeat bit-for-bit, so the blackout start is perturbed by one frame (3 phases) for P3.
  LaMAria is asynchronous and a guard only: 1 paired round, 3 rounds if a decision counter is not 0. Load
  recorded per run; no run starts while card AC's queue runs.
If it misses: (1) P1 / P2 low: list the frames after each carried line and their imu= state. (2) P3: K1's
  reading. (3) K6: first frame whose [still] line differs between the binaries.
