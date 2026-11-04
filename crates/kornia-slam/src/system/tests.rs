use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_sensors::imu::ImuBias;

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

mod keyframe_evidence {
    use super::*;
    use crate::mapping::keyframes::parallax::tests::{
        all_matches, camera, frame_seeing_points, map_with_keyframe_at,
    };
    use kornia_algebra::{Mat3F64, SO3F64};

    /// A system whose map holds keyframe 0 at the identity and keyframe 1, the
    /// tracking reference, at `ref_pose` observing nothing, with the camera
    /// tracked at `tracked` after the last keyframe 0.
    fn system_tracked_at(min_parallax_px: Option<f64>, tracked: Pose3d) -> SlamSystem {
        let mut config = mono();
        let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
        orb.keyframes.insertion = min_parallax_px
            .map(
                |min_parallax_px| crate::mapping::KeyframeInsertionStrategy::Parallax {
                    min_parallax_px,
                },
            )
            .unwrap_or(crate::mapping::KeyframeInsertionStrategy::FrameBased);
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
        orb.keyframes.insertion = crate::mapping::KeyframeInsertionStrategy::Parallax {
            min_parallax_px: 10.0,
        };
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

    #[test]
    fn a_culled_tracking_reference_is_replaced_before_the_next_insertion() {
        let mut system = SlamSystem::build(mono(), SensorRig::new(camera())).unwrap();
        let mut map = map_with_keyframe_at(Pose3d::IDENTITY);
        map.insert_keyframe(Keyframe::from_frame(frame_seeing_points(
            1,
            Pose3d::IDENTITY,
            &[],
        )))
        .unwrap();
        map.insert_keyframe(Keyframe::from_frame(frame_seeing_points(
            2,
            Pose3d::IDENTITY,
            &[],
        )))
        .unwrap();
        map.remove_keyframe(1).unwrap();
        *system.map.lock().unwrap() = map;
        // Relocalization can make an older, subsequently redundant keyframe
        // the tracking reference; it is not necessarily the most recent
        // keyframe, which remains a valid insertion reference.
        system.state.current_keyframe_idx = Some(1);
        system.state.last_keyframe_idx = Some(2);
        let tracked_pose = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(0.3, -0.2, 0.1),
        );
        let tracked_velocity = Vec3F64::new(1.0, -2.0, 3.0);
        let tracked_bias = ImuBias {
            gyro: Vec3F64::new(0.01, 0.02, 0.03),
            accel: Vec3F64::new(-0.1, 0.2, -0.3),
        };
        system.state.pose_world_to_cam = tracked_pose;
        system.state.velocity_world = tracked_velocity;
        system.inertial.bias = tracked_bias;

        system.forget_keyframes(&[1]);

        assert_eq!(system.state.current_keyframe_idx, Some(2));
        assert_eq!(system.state.last_keyframe_idx, Some(2));
        assert_eq!(system.state.pose_world_to_cam, tracked_pose);
        assert_eq!(system.state.velocity_world, tracked_velocity);
        assert_eq!(system.inertial.bias.gyro, tracked_bias.gyro);
        assert_eq!(system.inertial.bias.accel, tracked_bias.accel);
        let next = frame_seeing_points(3, Pose3d::IDENTITY, &[]);
        assert!(system.insert_keyframe(&next, 0.1, &[]));
        assert!(system.map.lock().unwrap().get_keyframe(3).is_some());
    }
}
