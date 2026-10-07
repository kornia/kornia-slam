use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_sensors::SensorRig;

use super::*;

fn orb_mut(config: &mut PipelineConfig) -> &mut OrbSlamPipeline {
    let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
    orb
}

fn with_loop_closing(mode: LoopClosingMode, sensors: SensorSelection) -> PipelineConfig {
    let mut config = PipelineConfig {
        sensors,
        ..PipelineConfig::default()
    };
    orb_mut(&mut config).loop_closing = mode;
    config
}

fn detect_and_correct() -> LoopClosingMode {
    LoopClosingMode::DetectAndCorrect {
        vocabulary: PathBuf::from("ORBvoc.txt"),
    }
}

fn sensors(cameras: CameraSelection, imu: bool) -> SensorSelection {
    SensorSelection { cameras, imu }
}

fn rig() -> SensorRig {
    SensorRig::new(PinholeCamera {
        fx: 458.0,
        fy: 457.0,
        cx: 367.0,
        cy: 248.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    })
}

#[test]
fn default_is_valid_orb_without_branches() {
    let config = PipelineConfig::default();
    assert_eq!(config.validate(), Ok(()));
    assert_eq!(
        config.stages(),
        [
            Stage::OrbFrontend,
            Stage::Tracking,
            Stage::KeyframeSelection,
            Stage::LocalMapping,
        ]
    );
}

#[test]
fn rejects_unsupported_version() {
    let config = PipelineConfig {
        version: 2,
        ..PipelineConfig::default()
    };
    assert_eq!(
        config.validate(),
        Err(ConfigError::UnsupportedVersion {
            found: 2,
            supported: PIPELINE_CONFIG_VERSION,
        })
    );
}

#[test]
fn keypoint_budget_is_bounded() {
    let range = OrbFrontendConfig::N_KEYPOINTS_RANGE;
    for (n_keypoints, ok) in [
        (0, false),
        (*range.start() - 1, false),
        (*range.start(), true),
        (*range.end(), true),
        (*range.end() + 1, false),
    ] {
        let mut config = PipelineConfig::default();
        orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig { n_keypoints });
        assert_eq!(config.validate().is_ok(), ok, "n_keypoints {n_keypoints}");
    }
}

#[test]
fn keyframe_gaps_must_be_positive_and_ordered() {
    for (min, max, ok) in [(0, 8, false), (9, 8, false), (8, 8, true), (1, 2, true)] {
        let mut config = PipelineConfig::default();
        let keyframes = &mut orb_mut(&mut config).keyframes;
        keyframes.min_frames_between = min;
        keyframes.max_frames_between = max;
        assert_eq!(config.validate().is_ok(), ok, "gaps {min}..={max}");
    }
}

#[test]
fn keyframe_ratio_must_be_finite_unit_interval() {
    for (ratio, ok) in [
        (0.0, true),
        (1.0, true),
        (-0.1, false),
        (1.5, false),
        (f64::NAN, false),
        (f64::INFINITY, false),
    ] {
        let mut config = PipelineConfig::default();
        orb_mut(&mut config).keyframes.ref_ratio = ratio;
        assert_eq!(config.validate().is_ok(), ok, "ratio {ratio}");
    }
}

#[test]
fn vocabulary_path_must_not_be_empty() {
    let config = with_loop_closing(
        LoopClosingMode::DetectOnly {
            vocabulary: PathBuf::new(),
        },
        SensorSelection::default(),
    );
    assert_eq!(config.validate(), Err(ConfigError::EmptyVocabularyPath));
}

#[test]
fn correction_requires_metric_input() {
    let mono = with_loop_closing(detect_and_correct(), sensors(CameraSelection::Mono, false));
    assert_eq!(
        mono.validate(),
        Err(ConfigError::CorrectionWithoutMetricScale)
    );
    for metric in [
        sensors(CameraSelection::Stereo, false),
        sensors(CameraSelection::Mono, true),
    ] {
        assert_eq!(
            with_loop_closing(detect_and_correct(), metric).validate(),
            Ok(())
        );
    }
    let detect_only = LoopClosingMode::DetectOnly {
        vocabulary: PathBuf::from("ORBvoc.txt"),
    };
    assert_eq!(
        with_loop_closing(detect_only, sensors(CameraSelection::Mono, false)).validate(),
        Ok(())
    );
}

#[test]
fn stages_follow_sensors_and_loop_branches() {
    let detect_only = LoopClosingMode::DetectOnly {
        vocabulary: PathBuf::from("ORBvoc.txt"),
    };
    let config = with_loop_closing(detect_only, sensors(CameraSelection::Stereo, false));
    assert_eq!(
        config.stages(),
        [
            Stage::OrbFrontend,
            Stage::StereoDepth,
            Stage::Tracking,
            Stage::KeyframeSelection,
            Stage::LocalMapping,
            Stage::PlaceRecognition,
        ]
    );

    let config = with_loop_closing(detect_and_correct(), sensors(CameraSelection::Mono, true));
    assert_eq!(
        config.stages(),
        [
            Stage::OrbFrontend,
            Stage::ImuIntegration,
            Stage::Tracking,
            Stage::KeyframeSelection,
            Stage::LocalMapping,
            Stage::PlaceRecognition,
            Stage::LoopCorrection,
        ]
    );
}

#[test]
fn display_lists_enabled_stages_with_settings() {
    let config = with_loop_closing(detect_and_correct(), sensors(CameraSelection::Stereo, true));
    let description = config.to_string();
    for line in [
        "frontend: ORB, 1000 keypoints",
        "stereo depth",
        "IMU integration",
        "keyframes: every 3..=8 frames, ref ratio 0.6",
        "local mapping: asynchronous",
        "place recognition: ORBvoc.txt",
        "loop correction",
    ] {
        assert!(
            description.contains(line),
            "missing {line:?} in\n{description}"
        );
    }
    assert!(
        !PipelineConfig::default()
            .to_string()
            .contains("place recognition")
    );
}

#[test]
fn rig_must_provide_selected_sensors() {
    let mono = rig();
    let stereo_imu = rig().with_stereo_baseline(0.11).with_imu(Pose3d::IDENTITY);

    let stereo = sensors(CameraSelection::Stereo, false);
    let imu = sensors(CameraSelection::Mono, true);
    assert_eq!(
        stereo.validate_rig(&mono),
        Err(ConfigError::MissingSensor("stereo cameras"))
    );
    assert_eq!(
        imu.validate_rig(&mono),
        Err(ConfigError::MissingSensor("an IMU"))
    );
    assert_eq!(SensorSelection::default().validate_rig(&stereo_imu), Ok(()));
    assert_eq!(
        sensors(CameraSelection::Stereo, true).validate_rig(&stereo_imu),
        Ok(())
    );
}

#[test]
fn stereo_rejects_raw_fisheye_images() {
    let fisheye = kornia_3d::camera::FisheyeCamera {
        fx: 460.0,
        fy: 460.0,
        cx: 367.0,
        cy: 248.0,
        k1: 0.03,
        k2: -0.02,
        k3: 0.003,
        k4: -0.0005,
    };
    let rig = rig().with_stereo_baseline(0.11).with_fisheye(fisheye);
    assert_eq!(
        sensors(CameraSelection::Stereo, false).validate_rig(&rig),
        Err(ConfigError::FisheyeStereo)
    );
    assert_eq!(SensorSelection::default().validate_rig(&rig), Ok(()));
}

mod runtime_settings {
    use super::*;
    use crate::mapping::LocalMappingMode;
    use crate::tracking::KeyframePolicy;

    fn stereo_imu_rig() -> SensorRig {
        rig().with_stereo_baseline(0.11).with_imu(Pose3d::IDENTITY)
    }

    #[test]
    fn select_rig_keeps_only_selected_sensors() {
        let mono = SensorSelection::default()
            .select_rig(stereo_imu_rig())
            .unwrap();
        assert!(mono.stereo_baseline_m.is_none() && mono.imu.is_none());

        let all = sensors(CameraSelection::Stereo, true)
            .select_rig(stereo_imu_rig())
            .unwrap();
        assert_eq!(all.stereo_baseline_m, Some(0.11));
        assert!(all.imu.is_some());

        assert_eq!(
            sensors(CameraSelection::Mono, true)
                .select_rig(rig())
                .unwrap_err(),
            ConfigError::MissingSensor("an IMU")
        );
    }

    #[test]
    fn default_maps_to_current_runtime_defaults() {
        let config = PipelineConfig::default();
        let slam = config.slam_config(&rig());
        let policy = KeyframePolicy::default();
        assert_eq!(
            slam.keyframe_policy.min_frames_between,
            policy.min_frames_between
        );
        assert_eq!(
            slam.keyframe_policy.max_frames_between,
            policy.max_frames_between
        );
        assert_eq!(slam.keyframe_policy.ref_ratio, policy.ref_ratio);
        assert_eq!(slam.local_mapping, LocalMappingMode::Asynchronous);
        assert_eq!(slam.stereo_close_depth_m, None);
        assert!(slam.pgo.is_none());
        assert!(!slam.debug);
        assert_eq!(config.orb_detector().n_keypoints, 1000);
    }

    #[test]
    fn stage_settings_reach_runtime() {
        let mut config = PipelineConfig::default();
        let orb = orb_mut(&mut config);
        orb.frontend = FrontendConfig::Orb(OrbFrontendConfig { n_keypoints: 3000 });
        orb.keyframes = KeyframeConfig {
            min_frames_between: 2,
            max_frames_between: 5,
            ref_ratio: 0.8,
        };
        orb.mapping.execution = MappingExecution::Synchronous;

        let slam = config.slam_config(&rig());
        assert_eq!(slam.keyframe_policy.min_frames_between, 2);
        assert_eq!(slam.keyframe_policy.max_frames_between, 5);
        assert_eq!(slam.keyframe_policy.ref_ratio, 0.8);
        assert_eq!(slam.local_mapping, LocalMappingMode::Synchronous);
        assert_eq!(config.orb_detector().n_keypoints, 3000);
    }

    #[test]
    fn stereo_close_depth_derives_from_baseline() {
        let selected = sensors(CameraSelection::Stereo, false)
            .select_rig(stereo_imu_rig())
            .unwrap();
        let slam = PipelineConfig::default().slam_config(&selected);
        assert_eq!(slam.stereo_close_depth_m, Some(0.11 * 35.0));
    }

    #[test]
    fn only_correction_branch_configures_pgo() {
        let detect_only = LoopClosingMode::DetectOnly {
            vocabulary: PathBuf::from("ORBvoc.txt"),
        };
        let stereo = sensors(CameraSelection::Stereo, false);
        let stereo_rig = stereo.select_rig(stereo_imu_rig()).unwrap();
        assert!(
            with_loop_closing(detect_only, stereo)
                .slam_config(&stereo_rig)
                .pgo
                .is_none()
        );

        let pgo = with_loop_closing(detect_and_correct(), stereo)
            .slam_config(&stereo_rig)
            .pgo
            .unwrap();
        assert!(!pgo.require_imu_initialized);

        let mono_imu = sensors(CameraSelection::Mono, true);
        let imu_rig = mono_imu.select_rig(stereo_imu_rig()).unwrap();
        let pgo = with_loop_closing(detect_and_correct(), mono_imu)
            .slam_config(&imu_rig)
            .pgo
            .unwrap();
        assert!(pgo.require_imu_initialized);
    }

    #[test]
    fn vocabulary_loads_only_for_enabled_branches() {
        assert!(
            PipelineConfig::default()
                .load_vocabulary()
                .unwrap()
                .is_none()
        );

        let missing = PathBuf::from("/nonexistent/ORBvoc.txt");
        let config = with_loop_closing(
            LoopClosingMode::DetectOnly {
                vocabulary: missing.clone(),
            },
            SensorSelection::default(),
        );
        assert_eq!(config.load_vocabulary().err().unwrap().path, missing);
    }
}

#[cfg(feature = "serde")]
mod ron_files {
    use std::path::Path;

    use super::*;

    const PLAN_EXAMPLE: &str = r#"
        (
            version: 1,
            sensors: ( cameras: Mono, imu: false ),
            pipeline: OrbSlam((
                frontend: Orb(( n_keypoints: 1000 )),
                keyframes: ( min_frames_between: 3, max_frames_between: 8, ref_ratio: 0.6 ),
                mapping: ( execution: Asynchronous ),
                loop_closing: Disabled,
            )),
        )
    "#;

    fn orb(config: &PipelineConfig) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &config.pipeline;
        orb
    }

    fn parse(text: &str) -> Result<PipelineConfig, LoadError> {
        PipelineConfig::from_ron_str(text, Path::new(""))
    }

    #[test]
    fn explicit_defaults_equal_default() {
        assert_eq!(parse(PLAN_EXAMPLE).unwrap(), PipelineConfig::default());
        assert_eq!(parse("(version: 1)").unwrap(), PipelineConfig::default());
    }

    #[test]
    fn partial_file_keeps_other_defaults() {
        let config =
            parse("(version: 1, pipeline: OrbSlam((frontend: Orb((n_keypoints: 3000)))))").unwrap();
        let mut expected = PipelineConfig::default();
        orb_mut(&mut expected).frontend =
            FrontendConfig::Orb(OrbFrontendConfig { n_keypoints: 3000 });
        assert_eq!(config, expected);
    }

    #[test]
    fn version_is_required() {
        assert!(matches!(parse("()"), Err(LoadError::Parse(_))));
    }

    #[test]
    fn other_version_is_reported_before_its_fields() {
        let err = parse("(version: 2, pipeline: Future((stages: [])))").unwrap_err();
        assert!(matches!(
            err,
            LoadError::Invalid(ConfigError::UnsupportedVersion { found: 2, .. })
        ));
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_level() {
        for text in [
            "(version: 1, extra: 1)",
            "(version: 1, sensors: (lidar: true))",
            "(version: 1, pipeline: OrbSlam((tracker: 1)))",
            "(version: 1, pipeline: OrbSlam((frontend: Orb((n_levels: 8)))))",
            "(version: 1, pipeline: OrbSlam((keyframes: (gap: 3))))",
            "(version: 1, pipeline: OrbSlam((mapping: (window: 3))))",
            "(version: 1, pipeline: OrbSlam((loop_closing: DetectOnly(vocabulary: \"v.txt\", pgo: true))))",
        ] {
            assert!(matches!(parse(text), Err(LoadError::Parse(_))), "{text}");
        }
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let err = parse("(version: 1, pipeline: OrbSlam((frontend: Orb((n_keypoints: 0)))))")
            .unwrap_err();
        assert!(matches!(
            err,
            LoadError::Invalid(ConfigError::KeypointsOutOfRange { value: 0, .. })
        ));
        let err = parse(
            "(version: 1, pipeline: OrbSlam((loop_closing: DetectAndCorrect(vocabulary: \"v.txt\"))))",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            LoadError::Invalid(ConfigError::CorrectionWithoutMetricScale)
        ));
    }

    #[test]
    fn vocabulary_resolves_against_base_dir() {
        let text = |path: &str| {
            format!(
                "(version: 1, pipeline: OrbSlam((loop_closing: DetectOnly(vocabulary: {path:?}))))"
            )
        };
        let base = Path::new("/configs");
        let relative = PipelineConfig::from_ron_str(&text("../weights/ORBvoc.txt"), base).unwrap();
        assert_eq!(
            orb(&relative).loop_closing.vocabulary(),
            Some(Path::new("/configs/../weights/ORBvoc.txt"))
        );
        let absolute = PipelineConfig::from_ron_str(&text("/weights/ORBvoc.bin"), base).unwrap();
        assert_eq!(
            orb(&absolute).loop_closing.vocabulary(),
            Some(Path::new("/weights/ORBvoc.bin"))
        );
    }

    #[test]
    fn file_paths_resolve_against_the_file_directory() {
        let dir = std::env::temp_dir().join(format!("kornia-slam-pipeline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orb.ron");
        std::fs::write(
            &path,
            "(version: 1, pipeline: OrbSlam((loop_closing: DetectOnly(vocabulary: \"ORBvoc.txt\"))))",
        )
        .unwrap();
        let loaded = PipelineConfig::from_ron_file(&path);
        std::fs::remove_dir_all(&dir).unwrap();

        let config = loaded.unwrap();
        assert_eq!(
            orb(&config).loop_closing.vocabulary(),
            Some(dir.join("ORBvoc.txt").as_path())
        );
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = PipelineConfig::from_ron_file("/nonexistent/orb.ron").unwrap_err();
        assert!(matches!(err, LoadError::Io { .. }));
    }

    #[test]
    fn serialization_round_trips() {
        let mut config = with_loop_closing(
            LoopClosingMode::DetectAndCorrect {
                vocabulary: PathBuf::from("/weights/ORBvoc.bin"),
            },
            sensors(CameraSelection::Stereo, true),
        );
        let orb = orb_mut(&mut config);
        orb.frontend = FrontendConfig::Orb(OrbFrontendConfig { n_keypoints: 2000 });
        orb.keyframes.ref_ratio = 0.75;
        orb.mapping.execution = MappingExecution::Synchronous;

        let text = config.to_ron_string().unwrap();
        assert_eq!(parse(&text).unwrap(), config, "{text}");
    }
}
