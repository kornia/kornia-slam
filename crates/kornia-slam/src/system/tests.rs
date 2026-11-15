use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

use super::*;
use crate::frontend::OrbFrontend;
use crate::frontend::tests::synthetic_pair;
use crate::loop_closure::place_recognition::{Vocabulary, VocabularyLoadError};
use crate::mapping::LocalMappingMode;

#[test]
fn formats_compact_imu_init_gate() {
    assert_eq!(
        format_imu_init_gate(12, Some(12), Some(32), 7, 10, 1.05, 1.0),
        "[imu_init_gate] start_idx=12 first_idx=Some(12) last_idx=Some(32) kfs=7/10 imu_time=1.05/1.0s"
    );
}

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 435.0,
        fy: 435.0,
        cx: 188.0,
        cy: 120.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

fn stereo_imu_rig() -> SensorRig {
    SensorRig::new(camera())
        .with_stereo_baseline(0.11)
        .with_imu(Pose3d::IDENTITY)
}

fn config(cameras: CameraSelection, imu: bool, loop_closing: LoopClosingMode) -> SystemConfig {
    let mut config = SystemConfig {
        sensors: SensorSelection { cameras, imu },
        ..SystemConfig::default()
    };
    let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
    orb.mapping.execution = LocalMappingMode::Synchronous;
    orb.loop_closing = loop_closing;
    config
}

fn mono() -> SystemConfig {
    config(CameraSelection::Mono, false, LoopClosingMode::Disabled)
}

/// Deterministic texture of 4x4 blocks of noise, shifted left by `shift` pixels.
fn textured(shift: usize) -> Image<u8, 1> {
    synthetic_pair(376, 240, shift).1
}

/// Capture time of the `idx`-th frame, at 20 Hz.
fn stamp(idx: usize) -> f64 {
    idx as f64 * 0.05
}

fn imu_samples() -> Vec<ImuMeasurement> {
    (0..10)
        .map(|i| ImuMeasurement {
            timestamp: i as f64 * 0.005,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81),
        })
        .collect()
}

/// A vocabulary file, removed on drop.
struct SavedVocabulary(PathBuf);

impl Drop for SavedVocabulary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A vocabulary trained on the texture's descriptors, saved where a
/// configuration can name it.
fn saved_vocabulary(name: &str) -> SavedVocabulary {
    let features = OrbFrontend::new(mono().orb_detector(), &SensorRig::new(camera()))
        .prepare_image(0, &textured(0))
        .unwrap()
        .features;
    let descriptors: Vec<_> = features
        .descriptors
        .iter()
        .map(kornia_bow::orb_slam3::pack_orb_descriptor)
        .collect();
    let path = std::env::temp_dir().join(format!(
        "kornia-slam-system-{}-{name}.bin",
        std::process::id()
    ));
    Vocabulary::train(&descriptors, 1)
        .unwrap()
        .save(path.to_str().unwrap())
        .unwrap();
    SavedVocabulary(path)
}

#[test]
fn build_validates_programmatic_configs() {
    let config = SystemConfig {
        version: 2,
        ..mono()
    };
    assert!(matches!(
        SlamSystem::build(config, stereo_imu_rig()),
        Err(BuildError::Config(ConfigError::UnsupportedVersion { .. }))
    ));
}

#[test]
fn build_rejects_sensors_the_rig_lacks() {
    let stereo = config(CameraSelection::Stereo, false, LoopClosingMode::Disabled);
    assert!(matches!(
        SlamSystem::build(stereo, SensorRig::new(camera())),
        Err(BuildError::Config(ConfigError::MissingSensor(
            "stereo cameras"
        )))
    ));
}

#[test]
fn build_restricts_the_rig_to_selected_sensors() {
    let system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    assert!(system.rig().stereo_baseline_m.is_none());
    assert!(system.rig().imu.is_none());
}

#[test]
fn build_reports_a_missing_vocabulary() {
    let missing = PathBuf::from("/nonexistent/ORBvoc.bin");
    let enabled = config(
        CameraSelection::Stereo,
        false,
        LoopClosingMode::Enabled {
            vocabulary: missing.clone(),
            execution: LoopClosingExecution::default(),
            graph: Default::default(),
        },
    );
    match SlamSystem::build(enabled, stereo_imu_rig()) {
        Err(BuildError::Vocabulary(VocabularyLoadError { path, .. })) => {
            assert_eq!(path, missing)
        }
        other => panic!("expected a vocabulary error, got {:?}", other.err()),
    }
}

#[test]
fn loop_closing_constructs_only_when_enabled() {
    let disabled = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    assert!(disabled.loop_closer.is_none());

    let vocabulary = saved_vocabulary("loop-closing");
    let enabled = |imu| {
        let config = config(
            CameraSelection::Stereo,
            imu,
            LoopClosingMode::Enabled {
                vocabulary: vocabulary.0.clone(),
                execution: LoopClosingExecution::default(),
                graph: Default::default(),
            },
        );
        SlamSystem::build(config, stereo_imu_rig()).unwrap()
    };
    let stereo = enabled(false).loop_closer.unwrap();
    assert!(!stereo.correction_requires_imu());
    assert!(enabled(true).loop_closer.unwrap().correction_requires_imu());
}

#[test]
fn disabled_imu_input_is_ignored() {
    let samples = imu_samples();
    let image = textured(0);

    let mut visual = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    visual.process_imu(&samples);
    visual.process_image(stamp(0), &image).unwrap();
    assert_eq!(visual.inertial.pending_sample_count(), 0);

    let inertial = config(CameraSelection::Mono, true, LoopClosingMode::Disabled);
    let mut inertial = SlamSystem::build(inertial, stereo_imu_rig()).unwrap();
    inertial.process_imu(&samples);
    inertial.process_image(stamp(0), &image).unwrap();
    assert_eq!(inertial.inertial.pending_sample_count(), samples.len());
}

#[test]
fn stereo_input_without_a_right_image_changes_nothing() {
    let stereo = config(CameraSelection::Stereo, true, LoopClosingMode::Disabled);
    let mut system = SlamSystem::build(stereo, stereo_imu_rig()).unwrap();
    let samples = imu_samples();
    let image = textured(0);
    system.process_imu(&samples);
    let err = system.process_image(stamp(0), &image).unwrap_err();
    assert!(matches!(err, ProcessError::StereoRequired));
    assert!(system.previous_image.is_none());
    assert!(system.frontend_observation().keypoints_xy.is_empty());

    let right = textured(8);
    assert!(system.process_stereo(stamp(0), &image, &right).is_ok());
    assert!(system.frontend_observation().stereo_matched.unwrap() > 0);
}

#[test]
fn a_mono_pipeline_rejects_a_stereo_pair() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    let (left, right) = (textured(0), textured(8));
    let err = system.process_stereo(stamp(0), &left, &right).unwrap_err();
    assert!(matches!(err, ProcessError::MonoPipeline));
    assert!(system.previous_image.is_none());
    assert!(system.frontend_observation().keypoints_xy.is_empty());
}

/// A rig that never bootstraps (a dark room) inserts no keyframe, yet must
/// not buffer IMU samples for as long as that lasts.
#[test]
fn imu_buffer_stays_bounded_while_bootstrap_keeps_failing() {
    let dark = Image::from_size_val(
        ImageSize {
            width: 376,
            height: 240,
        },
        0u8,
    )
    .unwrap();
    let frame_samples = |idx: usize| -> Vec<ImuMeasurement> {
        // 10 samples at 200 Hz ending at the frame's stamp.
        (0..10)
            .map(|k| ImuMeasurement {
                timestamp: idx as f64 * 0.05 - 0.045 + k as f64 * 0.005,
                gyro: Vec3F64::ZERO,
                accel: Vec3F64::new(0.0, 0.0, 9.81),
            })
            .collect()
    };
    // Stereo: keypoints on the left, none matched on the right, so the
    // single-frame stereo bootstrap fails on too few stereo points. Mono:
    // no keypoints at all, so no bootstrap reference is ever held.
    let left = textured(0);
    for cameras in [CameraSelection::Stereo, CameraSelection::Mono] {
        let stereo = matches!(cameras, CameraSelection::Stereo);
        let config = config(cameras, true, LoopClosingMode::Disabled);
        let mut system = SlamSystem::build(config, stereo_imu_rig()).unwrap();
        for idx in 0..400 {
            system.process_imu(&frame_samples(idx));
            let result = if stereo {
                system.process_stereo(stamp(idx), &left, &dark)
            } else {
                system.process_image(stamp(idx), &dark)
            }
            .unwrap();
            assert_eq!(result.status, TrackingStatus::Skipped);
            assert!(
                system.inertial.pending_sample_count() <= 10,
                "{cameras:?} frame {idx}: {} samples buffered",
                system.inertial.pending_sample_count()
            );
        }
    }
}

#[test]
fn image_history_follows_successful_frames_only() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    let (first, second, third) = (textured(0), textured(2), textured(4));

    let result = system.process_image(stamp(0), &first).unwrap();
    // The first frame only becomes the bootstrap reference, yet still
    // becomes the previous image.
    assert_eq!(result.status, TrackingStatus::Skipped);
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        first.as_slice()
    );

    assert!(system.process_image(f64::NAN, &second).is_err());
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        first.as_slice()
    );

    system.process_image(stamp(2), &third).unwrap();
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        third.as_slice()
    );
}

#[test]
fn frames_are_numbered_in_call_order_and_failures_keep_the_number() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    let image = textured(0);
    assert!(system.process_image(f64::NAN, &image).is_err());
    assert_eq!(system.next_frame_idx, 0);
    for expected in 1..=3 {
        system.process_image(stamp(expected), &image).unwrap();
        assert_eq!(system.next_frame_idx, expected);
    }
}

#[test]
fn a_fresh_systems_first_frame_is_a_bootstrap_skip() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    assert_eq!(system.last_skip(), None);
    let result = system.process_image(stamp(0), &textured(0)).unwrap();
    assert_eq!(result.status, TrackingStatus::Skipped);
    assert_eq!(system.last_skip(), Some(SkipKind::Bootstrap));
    let counters = system.run_counters();
    assert_eq!(counters.frames, 1);
    assert_eq!(counters.skipped_bootstrap, 1);
}

#[test]
fn every_frame_is_counted_as_exactly_one_kind() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    assert!(system.process_image(f64::NAN, &textured(0)).is_err());
    let frames = 30;
    for idx in 0..frames {
        let result = system
            .process_image(stamp(idx), &textured(idx * 2))
            .unwrap();
        assert_eq!(
            result.status == TrackingStatus::Skipped,
            system.last_skip().is_some(),
            "frame {idx}"
        );
    }
    let c = system.run_counters();
    assert_eq!(c.frames, frames as u64, "a refused input is not a frame");
    assert_eq!(
        c.tracked + c.skipped_bootstrap + c.coast_inertial + c.coast_visual,
        c.frames
    );
    assert!(c.keyframes <= c.tracked);
}

mod keyframe_evidence {
    use super::*;
    use crate::tracking::parallax::tests::{
        all_matches, camera, frame_seeing_points, map_with_keyframe_at,
    };
    use kornia_algebra::{Mat3F64, SO3F64};

    /// A system whose map holds keyframe 0 at the identity and keyframe 1, the
    /// tracking reference, at `ref_pose` observing nothing, with the camera
    /// tracked at `tracked` after the last keyframe 0.
    fn system_tracked_at(min_parallax_px: Option<f64>, tracked: Pose3d) -> SlamSystem {
        let mut config = mono();
        let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
        orb.keyframes.min_parallax_px = min_parallax_px;
        let mut system = SlamSystem::build(config, SensorRig::new(camera())).unwrap();
        let mut map = map_with_keyframe_at(Pose3d::IDENTITY);
        let ref_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));
        map.insert_keyframe(Keyframe::from_frame(frame_seeing_points(1, ref_pose, &[])))
            .unwrap();
        *system.map.lock().unwrap() = map;
        system.state.last_keyframe_idx = Some(0);
        system.state.current_keyframe_idx = Some(1);
        system.state.pose_world_to_cam = tracked;
        system
    }

    #[test]
    fn measures_parallax_against_the_last_keyframe_at_the_tracked_pose() {
        // The head yawed 0.08 rad without moving: about 32 px of raw pixel
        // motion, zero parallax once the rotation is compensated.
        let yaw = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.08, 0.0)).matrix(),
            Vec3F64::ZERO,
        );
        let points = map_with_keyframe_at(Pose3d::IDENTITY)
            .map_points()
            .iter()
            .map(|mp| mp.position)
            .collect::<Vec<_>>();
        // The frame's own pose is the pre-tracking one, still at the keyframe;
        // the keypoints are where the tracked (yawed) camera sees the points.
        let mut frame = frame_seeing_points(7, yaw, &points);
        frame.pose_world_to_cam = Pose3d::IDENTITY;

        let system = system_tracked_at(Some(10.0), yaw);
        let evidence = system.keyframe_evidence(&frame, 9, &all_matches());
        assert_eq!(evidence.curr_idx, 7);
        assert_eq!(evidence.last_keyframe_idx, Some(0));
        assert_eq!(evidence.tracked_inliers, 9);
        assert_eq!(evidence.n_ref_map_points, 0, "the reference is keyframe 1");
        let parallax = evidence
            .median_parallax_px
            .expect("measured against keyframe 0, which observes the matches");
        assert!(parallax < 1e-3, "{parallax}");

        let off = system_tracked_at(None, yaw);
        assert_eq!(
            off.keyframe_evidence(&frame, 9, &all_matches())
                .median_parallax_px,
            None,
            "not measured unless the policy reads it"
        );
    }
    /// A parked camera earns no keyframe from the motion policy; the time
    /// bound still forces one, measured from the last keyframe's timestamp.
    #[test]
    fn the_time_bound_forces_a_keyframe_on_a_parked_camera() {
        let mut config = mono();
        let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
        orb.keyframes.max_frames_between = usize::MAX;
        orb.keyframes.min_parallax_px = Some(10.0);
        orb.keyframes.max_interval_sec = Some(0.5);
        let mut system = SlamSystem::build(config, SensorRig::new(camera())).unwrap();
        system.state.last_keyframe_idx = Some(100);
        system.inertial.last_keyframe_timestamp_sec = Some(10.0);
        let parked = KeyframeEvidence {
            curr_idx: 1100,
            last_keyframe_idx: Some(100),
            tracked_inliers: 80,
            n_ref_map_points: 100,
            median_parallax_px: Some(0.3),
        };
        assert!(!system.keyframe_policy.should_insert_with(&parked));
        assert!(!system.keyframe_due(&parked, 10.49));
        assert!(system.keyframe_due(&parked, 10.5));
        system.keyframe_policy.max_interval_sec = None;
        assert!(!system.keyframe_due(&parked, 1e6), "no bound unless set");
    }
}

mod wheels {
    use super::*;
    use crate::system::wheel::{WheelError, WheelOdomConfig, WheelSample};
    use kornia_algebra::Mat3F64;

    fn system() -> SlamSystem {
        SlamSystem::build(
            config(CameraSelection::Stereo, false, LoopClosingMode::Disabled),
            stereo_imu_rig(),
        )
        .expect("builds")
    }

    fn straight(seconds: u64) -> Vec<WheelSample> {
        (0..=seconds * 50)
            .map(|i| WheelSample {
                stamp_ns: i * 20_000_000,
                x: i as f64 * 0.002,
                y: 0.0,
                theta: 0.0,
            })
            .collect()
    }

    fn enabled() -> SlamSystem {
        let mut s = system();
        s.enable_wheel_odometry(
            WheelOdomConfig::new(Pose3d::IDENTITY, 0.05, 0.01).expect("valid config"),
        );
        s
    }

    #[test]
    fn without_wheels_nothing_is_kept_and_jobs_carry_no_factors() {
        let mut s = system();
        assert_eq!(s.process_wheel_odometry(straight(1)), 0);
        assert!(s.push_camera_mount(0, Pose3d::IDENTITY).is_ok());
        s.note_wheel_keyframe(Some((0, 0.0)), 5, 1.0);
        assert!(s.wheel_factors().is_empty());
    }

    #[test]
    fn a_covered_keyframe_interval_becomes_one_edge_between_those_keyframes() {
        let mut s = enabled();
        assert!(s.process_wheel_odometry(straight(2)) > 0);
        s.note_wheel_keyframe(None, 0, 0.0);
        s.note_wheel_keyframe(Some((0, 0.0)), 5, 1.0);
        let factors = s.wheel_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!((factors[0].from_idx, factors[0].to_idx), (0, 5));
    }

    #[test]
    fn an_interval_past_the_newest_reading_waits_for_the_readings() {
        let mut s = enabled();
        s.process_wheel_odometry(straight(1));
        s.note_wheel_keyframe(Some((0, 0.5)), 9, 3.0);
        assert!(s.wheel_factors().is_empty(), "not covered yet");
        s.process_wheel_odometry((51..=200).map(|i| WheelSample {
            stamp_ns: i * 20_000_000,
            x: i as f64 * 0.002,
            y: 0.0,
            theta: 0.0,
        }));
        s.note_wheel_keyframe(Some((9, 3.0)), 12, 3.5);
        let ends: Vec<_> = s
            .wheel_factors()
            .iter()
            .map(|f| (f.from_idx, f.to_idx))
            .collect();
        assert!(ends.contains(&(0, 9)), "the waiting edge landed: {ends:?}");
    }

    #[test]
    fn a_non_rigid_mount_is_refused() {
        let mut s = enabled();
        let skew = Pose3d::new(Mat3F64::IDENTITY * 2.0, Vec3F64::ZERO);
        assert!(matches!(
            s.push_camera_mount(0, skew),
            Err(WheelError::MountNotRigid { .. })
        ));
    }
}

mod anchors {
    use super::*;
    use crate::tracking::parallax::tests::{frame_seeing_points, map_with_keyframe_at};
    use kornia_sensors::imu::ImuBias;

    const ACCEL_X: f64 = 0.3;

    /// A monocular-inertial system tracking against keyframe 0 (inserted at
    /// t = 0), with an initialized IMU and 2 s of constant-acceleration
    /// samples buffered.
    fn system() -> SlamSystem {
        let mut config = config(CameraSelection::Mono, true, LoopClosingMode::Disabled);
        let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
        orb.tracking_loss.keyframes_while_lost = KeyframesWhileLost::Anchor;
        let rig = SensorRig::new(camera()).with_imu(Pose3d::IDENTITY);
        let mut system = SlamSystem::build(config, rig).unwrap();
        *system.map.lock().unwrap() = map_with_keyframe_at(Pose3d::IDENTITY);
        system.state.current_keyframe_idx = Some(0);
        system.state.last_keyframe_idx = Some(0);
        system.state.imu_initialized = true;
        system.inertial.last_keyframe_timestamp_sec = Some(0.0);
        system.inertial.gravity_world = Vec3F64::ZERO;
        let samples: Vec<ImuMeasurement> = (0..=400)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * 0.005,
                gyro: Vec3F64::ZERO,
                accel: Vec3F64::new(ACCEL_X, 0.0, 0.0),
            })
            .collect();
        system.process_imu(&samples);
        system
    }

    fn frame(idx: usize) -> Frame {
        frame_seeing_points(idx, Pose3d::IDENTITY, &[])
    }

    #[test]
    fn the_config_selects_anchors_and_defaults_to_lost_keyframes() {
        assert_eq!(
            TrackingLossRecoveryPolicy::default().keyframes_while_lost,
            KeyframesWhileLost::Enabled
        );
        assert_eq!(
            system().tracking_loss_recovery.keyframes_while_lost,
            KeyframesWhileLost::Anchor
        );
    }

    #[test]
    fn an_anchor_stays_out_of_the_imu_chain_and_the_next_keyframe_bridges_it() {
        let mut system = system();
        assert!(system.insert_keyframe(&frame(4), 0.2, &[], KeyframeRole::Anchor));
        {
            let map = system.map.lock().unwrap();
            let anchor = map.get_keyframe(4).unwrap();
            assert!(anchor.anchor && anchor.inserted_while_lost);
            assert!(map.imu_factors().is_empty());
        }
        assert_eq!(system.state.current_keyframe_idx, Some(4));
        assert_eq!(system.state.last_keyframe_idx, Some(0));
        assert_eq!(system.inertial.last_keyframe_timestamp_sec, Some(0.0));
        assert_eq!(system.state.last_anchor_timestamp_sec, Some(0.2));

        assert!(system.insert_keyframe(&frame(8), 0.4, &[], KeyframeRole::Tracked));
        let map = system.map.lock().unwrap();
        let [edge] = map.imu_factors() else {
            panic!("one edge, got {}", map.imu_factors().len());
        };
        assert_eq!((edge.prev_kf_idx, edge.curr_kf_idx), (0, 8));
        assert_eq!((edge.t0, edge.t1), (0.0, 0.4));
        assert!(!map.get_keyframe(8).unwrap().anchor);
        assert_eq!(system.state.last_keyframe_idx, Some(8));
    }

    #[test]
    fn anchors_are_spaced_from_the_newest_anchor_and_tracking_clears_it() {
        let mut system = system();
        assert!(system.keyframe_due_while_lost(0.2));
        assert!(system.insert_keyframe(&frame(4), 0.2, &[], KeyframeRole::Anchor));
        assert!(!system.keyframe_due_while_lost(0.3));
        assert!(system.keyframe_due_while_lost(0.4));
        system.state.mark_tracked();
        assert_eq!(system.state.last_anchor_timestamp_sec, None);
        system.state.last_anchor_timestamp_sec = Some(1.0);
        system.state.reset();
        assert_eq!(system.state.last_anchor_timestamp_sec, None);
    }

    /// Against an anchor the velocity keeps propagating through the loss
    /// (`v + a k dt`) and the bias is the last tracked keyframe's, whatever
    /// the anchor carries.
    #[test]
    fn an_anchor_reference_keeps_the_propagated_velocity_and_the_tracked_bias() {
        let mut system = system();
        let tracked_bias = ImuBias::default();
        system
            .map
            .lock()
            .unwrap()
            .edit_keyframes_for_test(|_, kf| kf.imu_bias = tracked_bias);
        system.state.velocity_world = Vec3F64::new(1.0, 0.0, 0.0);
        assert!(system.insert_keyframe(&frame(4), 0.2, &[], KeyframeRole::Anchor));
        system.map.lock().unwrap().edit_keyframes_for_test(|_, kf| {
            if kf.anchor {
                kf.velocity_world = Vec3F64::new(-5.0, 0.0, 0.0);
                kf.imu_bias.accel = Vec3F64::new(1.0, 0.0, 0.0);
            }
        });
        system.inertial.bias.accel = Vec3F64::new(2.0, 0.0, 0.0);

        let dt = 0.05;
        let k = 10;
        let mut pose = Pose3d::IDENTITY;
        for i in 0..k {
            system.refresh_reference_inertial_state();
            let (t0, t1) = (0.2 + i as f64 * dt, 0.2 + (i + 1) as f64 * dt);
            let (preintegrated, _) =
                system
                    .inertial
                    .preintegrate_window(system.rig.imu_noise(), t0, t1);
            let (next, velocity) = predict_pose(
                pose,
                None,
                system.state.velocity_world,
                Some(InertialPrediction {
                    preintegrated: &preintegrated,
                    camera_to_body: system.rig.camera_to_body(),
                    gravity_world: system.inertial.gravity_world,
                }),
            );
            pose = next;
            system.state.velocity_world = velocity.unwrap();
        }
        assert_eq!(system.inertial.bias.accel, tracked_bias.accel);
        let expected = 1.0 + ACCEL_X * k as f64 * dt;
        let v = system.state.velocity_world;
        assert!((v.x - expected).abs() < 1e-9, "{v:?} vs {expected}");
        assert!(v.y.abs() < 1e-12 && v.z.abs() < 1e-12);
    }
}

mod recovery_after_loss {
    use super::*;
    use crate::tracking::parallax::tests::frame_seeing_points;

    /// World-to-camera pose of a camera centred at `(x, 0, 0)`.
    fn centred_at(x: f64) -> Pose3d {
        Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-x, 0.0, 0.0),
        )
    }

    /// A stereo-inertial system with anchors and the inertial state kept
    /// across a give-up, tracked to keyframe 40 at t = 10 s, lost
    /// from t = 10 s at `centred_at(1.0)` and coasted to `centred_at(coast_x)`.
    fn lost_metric_system(coast_x: f64) -> SlamSystem {
        let mut config = config(CameraSelection::Stereo, true, LoopClosingMode::Disabled);
        let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
        orb.tracking_loss.keyframes_while_lost = KeyframesWhileLost::Anchor;
        orb.tracking_loss.reseed.keep_inertial = true;
        orb.tracking_loss.reseed.min_seed_points = 10;
        let mut system = SlamSystem::build(config, stereo_imu_rig()).unwrap();
        system.state.mode = SystemMode::Tracking;
        system.state.imu_initialized = true;
        system.state.imu_init_timestamp_sec = Some(2.0);
        system.state.last_keyframe_idx = Some(40);
        system.state.current_keyframe_idx = Some(40);
        system.inertial.last_keyframe_timestamp_sec = Some(10.0);
        system.state.mark_lost(10.0, centred_at(1.0));
        system.state.pose_world_to_cam = centred_at(coast_x);
        system.state.velocity_world = Vec3F64::new(1.2, 0.0, 0.0);
        system
    }

    fn stereo_frame(idx: usize) -> Frame {
        let mut frame = frame_seeing_points(idx, Pose3d::IDENTITY, &[]);
        frame.depth = vec![2.0; frame.features.keypoints_xy.len().max(1)];
        frame
    }

    #[test]
    fn a_plausible_coast_seeds_the_next_segment_there_and_keeps_the_inertial_state() {
        let mut system = lost_metric_system(7.0);

        system.give_up(&stereo_frame(140), 5.0);

        let state = &system.state;
        assert!(state.reseeding(), "the IMU stays initialized");
        assert_eq!(state.imu_init_timestamp_sec, Some(2.0));
        assert_eq!(state.pose_world_to_cam, centred_at(7.0));
        assert_eq!(state.velocity_world, Vec3F64::new(1.2, 0.0, 0.0));
        assert_eq!(
            state.bridge_anchor,
            Some(BridgeAnchor {
                kf_idx: 40,
                timestamp_sec: 10.0
            })
        );
    }

    #[test]
    fn a_runaway_coast_seeds_at_the_last_tracked_pose_at_rest_without_a_bridge() {
        let mut system = lost_metric_system(-8.1e57);

        system.give_up(&stereo_frame(140), 5.0);

        let state = &system.state;
        assert!(state.reseeding());
        assert_eq!(state.pose_world_to_cam, centred_at(1.0));
        assert_eq!(state.velocity_world, Vec3F64::ZERO);
        assert_eq!(state.bridge_anchor, None);
    }

    /// A coast whose average over the gap looks like walking but whose last
    /// 10 s ran away is no seed either.
    #[test]
    fn a_runaway_over_the_window_overrides_a_plausible_gap_average() {
        let mut system = lost_metric_system(7.0);
        system.runaway.record(3.0, Vec3F64::ZERO);
        system.runaway.record(13.0, Vec3F64::new(60.0, 0.0, 0.0));

        system.give_up(&stereo_frame(140), 5.0);

        assert_eq!(system.state.pose_world_to_cam, centred_at(1.0));
        assert_eq!(system.state.bridge_anchor, None);
        assert!(!system.runaway.runaway(), "the runaway history is dropped");
    }

    #[test]
    fn a_monocular_frame_or_the_knob_off_starts_over_as_before() {
        let mut mono_frame = lost_metric_system(7.0);
        mono_frame.give_up(&frame_seeing_points(140, Pose3d::IDENTITY, &[]), 5.0);
        assert!(!mono_frame.state.imu_initialized);
        assert_eq!(mono_frame.state.pose_world_to_cam, centred_at(1.0));

        let mut off = lost_metric_system(7.0);
        off.recovery.keep_inertial = false;
        off.give_up(&stereo_frame(140), 5.0);
        assert!(!off.state.imu_initialized);
    }

    /// The bridge is the IMU edge from the anchor to the new keyframe with
    /// its covariance scaled by `bridge_covariance_scale`.
    #[test]
    fn the_bridge_edge_spans_the_gap_with_an_inflated_covariance() {
        let mut system = lost_metric_system(7.0);
        let samples: Vec<ImuMeasurement> = (0..=1000)
            .map(|i| ImuMeasurement {
                timestamp: 10.0 + i as f64 * 0.005,
                gyro: Vec3F64::new(0.0, 0.01, 0.0),
                accel: Vec3F64::new(0.0, 0.0, 9.81),
            })
            .collect();
        system.inertial.latest_image_timestamp_sec = Some(15.0);
        system.process_imu(&samples);
        {
            let mut map = system.map.lock().unwrap();
            for (idx, x) in [(40, 1.0), (141, 7.0)] {
                map.insert_keyframe(Keyframe::from_frame(frame_seeing_points(
                    idx,
                    centred_at(x),
                    &[],
                )))
                .unwrap();
            }
        }
        let anchor = BridgeAnchor {
            kf_idx: 40,
            timestamp_sec: 10.0,
        };

        system.publish_bridge(anchor, 141, 15.0);

        let plain = system
            .inertial
            .keyframe_edge(system.rig.imu_noise(), 40, 141, 10.0, 15.0)
            .unwrap();
        let map = system.map.lock().unwrap();
        let [bridge] = map.imu_factors() else {
            panic!("one bridge edge, got {}", map.imu_factors().len());
        };
        assert_eq!((bridge.prev_kf_idx, bridge.curr_kf_idx), (40, 141));
        let scale = system.recovery.bridge_covariance_scale;
        crate::system::recovery::tests::assert_scaled(
            &plain.preintegrated.covariance,
            &bridge.preintegrated.covariance,
            scale,
        );
        let replayed = bridge.repreintegrate(bridge.preintegrated.bias);
        assert_eq!(replayed.covariance, bridge.preintegrated.covariance);
    }

    /// A system re-seeding after a plausible give-up at t = 15 s, with level
    /// static IMU samples buffered to t = 16 s.
    fn reseeding_system() -> SlamSystem {
        let mut system = lost_metric_system(7.0);
        system.give_up(&stereo_frame(140), 5.0);
        system.state.last_frame_timestamp_sec = 15.0;
        system.inertial.latest_image_timestamp_sec = Some(15.0);
        let samples: Vec<ImuMeasurement> = (0..=200)
            .map(|i| ImuMeasurement {
                timestamp: 15.0 + i as f64 * 0.005,
                gyro: Vec3F64::ZERO,
                accel: Vec3F64::new(0.0, 0.0, 9.81),
            })
            .collect();
        system.process_imu(&samples);
        system
    }

    #[test]
    fn a_stalled_reseed_coasts_on_the_kept_inertial_state() {
        let mut system = reseeding_system();

        system.coast_while_reseeding(141, 15.05);

        let centre = recovery::camera_centre(&system.state.pose_world_to_cam);
        assert!((centre.x - (7.0 + 1.2 * 0.05)).abs() < 1e-6, "{centre:?}");
        assert_eq!(system.state.last_frame_timestamp_sec, 15.05);
        assert!(system.state.bridge_anchor.is_some());
    }

    #[test]
    fn a_runaway_while_reseeding_holds_the_pose_at_rest_and_drops_the_bridge() {
        let mut system = reseeding_system();
        system.runaway.record(14.0, Vec3F64::new(-100.0, 0.0, 0.0));

        system.coast_while_reseeding(141, 15.05);

        assert_eq!(system.state.pose_world_to_cam, centred_at(7.0));
        assert_eq!(system.state.velocity_world, Vec3F64::ZERO);
        assert_eq!(system.state.bridge_anchor, None);
    }

    #[test]
    fn a_bridge_older_than_max_bridge_sec_is_dropped() {
        let mut system = lost_metric_system(7.0);
        system.give_up(&stereo_frame(140), 5.0);
        assert!(system.live_bridge_anchor(19.9).is_some());
        assert!(system.live_bridge_anchor(20.1).is_none());
        assert!(system.state.bridge_anchor.is_none());
    }

    #[test]
    fn the_inertial_state_is_kept_only_when_asked() {
        assert!(!RecoveryConfig::default().keep_inertial);
        assert_eq!(RecoveryConfig::default().min_seed_points, MIN_STEREO_POINTS);
    }

    /// The first loss of a re-seeded segment gets its first anchor within
    /// `keyframe_interval_while_lost_sec` of the seed keyframe: the old
    /// segment's anchor spacing does not carry over.
    #[test]
    fn the_first_anchor_after_a_kept_give_up_comes_within_the_lost_interval() {
        let mut system = lost_metric_system(7.0);
        system.state.last_anchor_timestamp_sec = Some(14.9);
        assert!(system.give_up(&stereo_frame(140), 5.0));
        assert_eq!(system.state.last_anchor_timestamp_sec, None);
        system.finish_bootstrap(141, 15.0);
        assert!(system.state.imu_initialized);
        let interval = system
            .tracking_loss_recovery
            .keyframe_interval_while_lost_sec;
        assert!(!system.keyframe_due_while_lost(15.0 + interval / 2.0));
        assert!(system.keyframe_due_while_lost(15.0 + interval * 1.01));
    }
}
