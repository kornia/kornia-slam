# Gap ledger

Where benchmark score is lost, in the format of `.claude/skills/slam-experiment/SKILL.md` section 4. This file holds only the rows the PR #180 round touches. The full ledger lives outside the repo (`slam-viba/audit/gap-ledger.md`, cited by the cards); its other rows are not copied.

Status stays at what the evidence shows. "Built" below means the code exists behind a knob; no run has scored it, so no status moved because of the 2026-11-17 round.

| id | gap | evidence (run dirs) | sequences hit | est. score at stake | status | tested by |
|---|---|---|---|---|---|---|
| G6 | Online window of 3 keyframes is too short for scale and drift | `S/audit/window-results.md`: window 10 takes R_11 10 s scale in band from 46-63 % to 95-98 %; MH_03 35-59 % worse (bias tie) | R_11, MH_03, MH_01, V1_03, V2_02 | unknown (R_11 still fails Sim3 at window 10) | confirmed as a mechanism; score effect unknown | card A, A0 bundle and leave-one-out (knobs `local_ba.active_keyframes`, `large_window`, `imu_huber`, `boundary_scales_bias_rows`, `inertial.sample_interpolation`, `tracking.local_map`, `keyframes.rules`, `mapping.growth_neighbours`, `tracking.imu_trust`): built, not validated |
| G13 | No per-frame velocity state or pose-inertial solve; the IMU only seeds and vetoes | ORB-SLAM3 `Tracking.cc` per-frame inertial solve (card A6) | R_11 | unknown | suspected | card A6, `tracking.pose_inertial`: built, not validated |
| G18 | Drift and scale, not coverage, is the score gap | handoff: 3_18 error 4.7 m rising to 627 m at scale 1.285; 5_11 up to 317 m | 3_18, 5_11, R_11, 2_11 | arithmetic: proxy mean 3.6 against the 70 gate | confirmed | cards A and D together; no single card: open |
| G19 | No relocalisation while lost, so every reject or give-up costs coverage | `system/mod.rs` relocalisation gate (card B3) | R_11, 2_11, 3_18, 5_11 | unknown; card B3 K3 predicts the 0.5 m jump bound blocks part of it | confirmed | card B3 (`relocalization.while_lost`, `.solver`, `.max_jump_m`) and B5: built, not validated |
| G28 | A re-seeded 10-point map locks onto the whole old map, 5-13 m jumps | handoff, R_11 give-up window rg1 6239-8241 | R_11, 2_11 | unknown | confirmed | card B4 (`reseed.own_window_keyframes`): built, not validated |
| G29 | BA moves a keyframe to satisfy its IMU edge, the next PnP snaps back | `[predgap]` lines on R_11 (374 / 141 / 273 at base); no run isolates the cause | R_11 | unknown | suspected | card A, G29 counter `ba_snapbacks` and `tracking.snapback_fix`: built, not validated |
| A7 (G10) | No global VI-BA: nothing re-solves a keyframe after it leaves the window | card D mechanism; SLAM Dunk README (offline BA about 69 train, reading only) | all LaMAria | unknown; SLAM Dunk's number is a training figure, not ours | suspected | card D (`offline_ba`, `offline-ba`): built, not validated |
| A8 (G11) | Calibration is fixed: no camera drift, gyro intrinsics or clock states | visloc `lamaria_batch_ba.md` (device offsets); SLAM Dunk reading (69.1 to 82.4 train) | all LaMAria | unknown | suspected | card D calibration states (`offline_ba.calibration.*`): built, not validated |

## Next experiment

Rank by score at stake once the validation workflow has numbers. Until then the order is the protocol in `experiments/pr180-validation-handoff.md`: identity gate, then Lane A bundle (G6, G13, G29), then Lane B, then Lane D.
