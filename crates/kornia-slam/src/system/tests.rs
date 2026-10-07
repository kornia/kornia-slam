use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

use super::*;
use crate::frontend::OrbFrontend;
use crate::loop_closure::place_recognition::VocabularyLoadError;
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

fn config(cameras: CameraSelection, imu: bool, loop_closing: LoopClosingMode) -> PipelineConfig {
    let mut config = PipelineConfig {
        sensors: SensorSelection { cameras, imu },
        ..PipelineConfig::default()
    };
    let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
    orb.mapping.execution = LocalMappingMode::Synchronous;
    orb.loop_closing = loop_closing;
    config
}

fn mono() -> PipelineConfig {
    config(CameraSelection::Mono, false, LoopClosingMode::Disabled)
}

/// Deterministic texture of 4x4 blocks of noise, shifted right by `shift` pixels.
fn textured(shift: usize) -> Image<u8, 1> {
    let (width, height) = (376, 240);
    let value = |x: usize, y: usize| -> u8 {
        let mut s = ((x / 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((y / 4) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (s ^ (s >> 32)) as u8
    };
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| value(x + shift, y)))
        .collect();
    Image::from_size_slice(ImageSize { width, height }, &pixels).unwrap()
}

fn input<'a>(idx: usize, image: &'a Image<u8, 1>, imu: &'a [ImuMeasurement]) -> SensorFrame<'a> {
    SensorFrame {
        idx,
        timestamp_sec: idx as f64 * 0.05,
        image,
        right_image: None,
        imu_samples: imu,
    }
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

/// A vocabulary trained on the texture's descriptors, saved where a pipeline
/// file can name it.
fn saved_vocabulary(name: &str) -> PathBuf {
    let features = OrbFrontend::new(mono().orb_detector(), &SensorRig::new(camera()))
        .prepare(&input(0, &textured(0), &[]))
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
    path
}

#[test]
fn build_validates_programmatic_configs() {
    let config = PipelineConfig {
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
    let detect = config(
        CameraSelection::Mono,
        false,
        LoopClosingMode::DetectOnly {
            vocabulary: missing.clone(),
        },
    );
    match SlamSystem::build(detect, stereo_imu_rig()) {
        Err(BuildError::Vocabulary(VocabularyLoadError { path, .. })) => {
            assert_eq!(path, missing)
        }
        other => panic!("expected a vocabulary error, got {:?}", other.err()),
    }
}

#[test]
fn loop_branches_construct_only_what_they_enable() {
    let disabled = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    assert!(!disabled.loop_closer.has_vocabulary());
    assert!(!disabled.loop_closer.corrects_loops());

    let vocabulary = saved_vocabulary("branches");
    let detect = config(
        CameraSelection::Stereo,
        false,
        LoopClosingMode::DetectOnly {
            vocabulary: vocabulary.clone(),
        },
    );
    let detect = SlamSystem::build(detect, stereo_imu_rig()).unwrap();
    assert!(detect.loop_closer.has_vocabulary());
    assert!(!detect.loop_closer.corrects_loops());

    let correct = |imu| {
        let config = config(
            CameraSelection::Stereo,
            imu,
            LoopClosingMode::DetectAndCorrect {
                vocabulary: vocabulary.clone(),
            },
        );
        SlamSystem::build(config, stereo_imu_rig()).unwrap()
    };
    let stereo = correct(false);
    assert!(stereo.loop_closer.has_vocabulary());
    assert_eq!(stereo.loop_closer.correction_requires_imu(), Some(false));
    assert_eq!(
        correct(true).loop_closer.correction_requires_imu(),
        Some(true)
    );
    std::fs::remove_file(vocabulary).unwrap();
}

#[test]
fn disabled_imu_input_is_ignored() {
    let samples = imu_samples();
    let image = textured(0);

    let mut visual = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    visual.process(input(0, &image, &samples)).unwrap();
    assert_eq!(visual.inertial.pending_sample_count(), 0);

    let inertial = config(CameraSelection::Mono, true, LoopClosingMode::Disabled);
    let mut inertial = SlamSystem::build(inertial, stereo_imu_rig()).unwrap();
    inertial.process(input(0, &image, &samples)).unwrap();
    assert_eq!(inertial.inertial.pending_sample_count(), samples.len());
}

#[test]
fn stereo_input_without_a_right_image_changes_nothing() {
    let stereo = config(CameraSelection::Stereo, true, LoopClosingMode::Disabled);
    let mut system = SlamSystem::build(stereo, stereo_imu_rig()).unwrap();
    let samples = imu_samples();
    let image = textured(0);
    let err = system.process(input(0, &image, &samples)).unwrap_err();
    assert!(matches!(err, ProcessError::MissingRightImage { idx: 0 }));
    assert!(system.previous_image.is_none());
    assert!(system.frontend_observation().keypoints_xy.is_empty());
    assert_eq!(system.inertial.pending_sample_count(), 0);

    let right = textured(8);
    let pair = SensorFrame {
        right_image: Some(&right),
        ..input(0, &image, &samples)
    };
    assert!(system.process(pair).is_ok());
    assert!(system.frontend_observation().stereo_matched.unwrap() > 0);
}

#[test]
fn image_history_follows_successful_frames_only() {
    let mut system = SlamSystem::build(mono(), stereo_imu_rig()).unwrap();
    let (first, second, third) = (textured(0), textured(2), textured(4));

    let result = system.process(input(0, &first, &[])).unwrap();
    // The first frame only becomes the bootstrap reference, yet still
    // becomes the previous image.
    assert_eq!(result.status, TrackingStatus::Skipped);
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        first.as_slice()
    );

    let invalid = SensorFrame {
        timestamp_sec: f64::NAN,
        ..input(1, &second, &[])
    };
    assert!(system.process(invalid).is_err());
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        first.as_slice()
    );

    system.process(input(2, &third, &[])).unwrap();
    assert_eq!(
        system.previous_image.as_ref().unwrap().as_slice(),
        third.as_slice()
    );
}
