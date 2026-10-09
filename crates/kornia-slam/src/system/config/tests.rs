use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_sensors::SensorRig;

use super::*;
use crate::mapping::LocalMappingMode;
use crate::system::ConfigError;
#[cfg(feature = "serde")]
use crate::system::LoadError;
use crate::tracking::KeyframePolicy;

fn orb_mut(config: &mut SystemConfig) -> &mut OrbSlamPipeline {
    let PipelineDefinition::OrbSlam(orb) = &mut config.pipeline;
    orb
}

fn with_loop_closing(mode: LoopClosingMode, sensors: SensorSelection) -> SystemConfig {
    let mut config = SystemConfig {
        sensors,
        ..SystemConfig::default()
    };
    orb_mut(&mut config).loop_closing = mode;
    config
}

fn loop_closing() -> LoopClosingMode {
    LoopClosingMode::Enabled {
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
fn display_describes_the_pipeline() {
    let stereo = SystemConfig {
        sensors: sensors(CameraSelection::Stereo, false),
        ..SystemConfig::default()
    };
    let mut inertial_loop = with_loop_closing(loop_closing(), sensors(CameraSelection::Mono, true));
    orb_mut(&mut inertial_loop).mapping.execution = LocalMappingMode::Synchronous;
    for (config, expected) in [
        (
            SystemConfig::default(),
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: asynchronous
",
        ),
        (
            stereo,
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  stereo depth: rectified pair, close within 35 baselines
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: asynchronous
",
        ),
        (
            inertial_loop,
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  IMU integration
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: synchronous
  loop closing: ORBvoc.txt
",
        ),
    ] {
        assert_eq!(config.to_string(), expected);
    }
}

#[test]
fn frontend_and_keyframe_settings_are_range_checked() {
    let keypoints = |n_keypoints| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints,
            ..OrbFrontendConfig::default()
        });
        config.validate()
    };
    let range = OrbFrontendConfig::N_KEYPOINTS_RANGE;
    for n in [*range.start(), *range.end()] {
        assert_eq!(keypoints(n), Ok(()), "n_keypoints {n}");
    }
    for n in [0, *range.start() - 1, *range.end() + 1] {
        assert!(
            matches!(keypoints(n), Err(ConfigError::KeypointsOutOfRange { value, .. }) if value == n),
            "n_keypoints {n}"
        );
    }

    let gaps = |min, max| {
        let mut config = SystemConfig::default();
        let keyframes = &mut orb_mut(&mut config).keyframes;
        keyframes.min_frames_between = min;
        keyframes.max_frames_between = max;
        config.validate()
    };
    assert_eq!(gaps(8, 8), Ok(()));
    assert_eq!(gaps(1, 2), Ok(()));
    for (min, max) in [(0, 8), (9, 8)] {
        assert_eq!(
            gaps(min, max),
            Err(ConfigError::InvalidKeyframeGaps { min, max })
        );
    }

    let ratio = |ref_ratio| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).keyframes.ref_ratio = ref_ratio;
        config.validate()
    };
    assert_eq!(ratio(0.0), Ok(()));
    assert_eq!(ratio(1.0), Ok(()));
    for ref_ratio in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(ratio(ref_ratio), Err(ConfigError::InvalidRefRatio(_))),
            "ratio {ref_ratio}"
        );
    }
}

fn invalid_setting(config: &SystemConfig) -> Option<String> {
    match config.validate() {
        Err(ConfigError::InvalidSetting { setting, .. }) => Some(setting),
        _ => None,
    }
}

#[test]
fn settings_are_range_checked() {
    let mut config = SystemConfig::default();
    let orb = orb_mut(&mut config);
    orb.frontend = FrontendConfig::Orb(OrbFrontendConfig {
        stereo_close_depth: StereoCloseDepth::Metres(-1.0),
        ..OrbFrontendConfig::default()
    });
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("frontend.stereo_close_depth")
    );

    let mut config = SystemConfig::default();
    orb_mut(&mut config)
        .tuning
        .initialization
        .match_config
        .nn_ratio = 0.0;
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tuning.initialization.match_config.nn_ratio")
    );

    let mut config = SystemConfig::default();
    orb_mut(&mut config)
        .tuning
        .map_projection
        .local_projection
        .search_radius = f32::NAN;
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tuning.map_projection.local_projection.search_radius")
    );

    // Loop-correction tuning only matters, and is only checked, with loop
    // closing enabled.
    let mut config = with_loop_closing(loop_closing(), sensors(CameraSelection::Stereo, false));
    orb_mut(&mut config)
        .tuning
        .loop_correction
        .verification
        .pnp_ransac
        .confidence = 1.0;
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tuning.loop_correction.verification.pnp_ransac.confidence")
    );
    orb_mut(&mut config).loop_closing = LoopClosingMode::Disabled;
    assert_eq!(config.validate(), Ok(()));
}

#[test]
fn vocabulary_path_must_not_be_empty() {
    let config = with_loop_closing(
        LoopClosingMode::Enabled {
            vocabulary: PathBuf::new(),
        },
        SensorSelection::default(),
    );
    assert_eq!(config.validate(), Err(ConfigError::EmptyVocabularyPath));
}

#[test]
fn loop_closing_requires_metric_input() {
    let mono = with_loop_closing(loop_closing(), sensors(CameraSelection::Mono, false));
    assert_eq!(
        mono.validate(),
        Err(ConfigError::LoopClosingWithoutMetricScale)
    );
    for metric in [
        sensors(CameraSelection::Stereo, false),
        sensors(CameraSelection::Mono, true),
    ] {
        assert_eq!(with_loop_closing(loop_closing(), metric).validate(), Ok(()));
    }
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

    fn stereo_imu_rig() -> SensorRig {
        rig().with_stereo_baseline(0.11).with_imu(Pose3d::IDENTITY)
    }

    #[test]
    fn stage_settings_reach_runtime() {
        let mut config = SystemConfig::default();
        let orb = orb_mut(&mut config);
        orb.frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints: 3000,
            ..OrbFrontendConfig::default()
        });
        orb.keyframes = KeyframePolicy {
            min_frames_between: 2,
            max_frames_between: 5,
            ref_ratio: 0.8,
        };
        orb.mapping.execution = LocalMappingMode::Synchronous;
        orb.tuning.initialization.acceptance_config.min_inliers = 40;
        orb.tuning.map_projection.pnp.min_inliers = 12;
        orb.tuning.loss_recovery.timeout_imu_sec = 2.0;

        let slam = config.settings(&rig());
        assert_eq!(slam.keyframe_policy, orb_keyframes(&config));
        assert_eq!(slam.local_mapping, LocalMappingMode::Synchronous);
        assert_eq!(slam.two_view_init.acceptance_config.min_inliers, 40);
        assert_eq!(slam.map_projection.pnp.min_inliers, 12);
        assert_eq!(slam.tracking_loss_recovery.timeout_imu_sec, 2.0);
        assert_eq!(config.orb_detector().n_keypoints, 3000);
    }

    fn orb_keyframes(config: &SystemConfig) -> KeyframePolicy {
        let PipelineDefinition::OrbSlam(orb) = &config.pipeline;
        orb.keyframes
    }

    #[test]
    fn stereo_close_depth_resolves_against_the_baseline() {
        let selected = sensors(CameraSelection::Stereo, false)
            .select_rig(stereo_imu_rig())
            .unwrap();
        let close_depth = |depth| {
            let mut config = SystemConfig::default();
            orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
                stereo_close_depth: depth,
                ..OrbFrontendConfig::default()
            });
            config.settings(&selected).stereo_close_depth_m
        };
        assert_eq!(close_depth(StereoCloseDepth::default()), Some(0.11 * 35.0));
        assert_eq!(close_depth(StereoCloseDepth::Metres(2.5)), Some(2.5));
        assert_eq!(close_depth(StereoCloseDepth::Disabled), None);
    }
}

#[cfg(feature = "serde")]
mod ron_files {
    use std::path::Path;

    use super::*;

    const EXPLICIT_DEFAULTS: &str = r#"
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

    fn orb(config: &SystemConfig) -> &OrbSlamPipeline {
        let PipelineDefinition::OrbSlam(orb) = &config.pipeline;
        orb
    }

    fn parse(text: &str) -> Result<SystemConfig, LoadError> {
        SystemConfig::from_ron_str(text, Path::new(""))
    }

    fn parse_error(text: &str) -> ron::Error {
        match parse(text) {
            Err(LoadError::Parse(error)) => error.code,
            other => panic!("{text}: expected a parse error, got {other:?}"),
        }
    }

    fn rejects_field(text: &str, field: &str) -> bool {
        matches!(parse_error(text), ron::Error::NoSuchStructField { found, .. } if found == field)
    }

    /// Tuning has no `PartialEq` and no file representation, so configurations
    /// are compared through their serialized form.
    fn ron(config: &SystemConfig) -> String {
        config.to_ron_string().unwrap()
    }

    #[test]
    fn omitted_settings_take_their_defaults() {
        let default = ron(&SystemConfig::default());
        assert_eq!(ron(&parse(EXPLICIT_DEFAULTS).unwrap()), default);
        assert_eq!(ron(&parse("(version: 1)").unwrap()), default);

        let config =
            parse("(version: 1, pipeline: OrbSlam((frontend: Orb((n_keypoints: 3000)))))").unwrap();
        let mut expected = SystemConfig::default();
        orb_mut(&mut expected).frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints: 3000,
            ..OrbFrontendConfig::default()
        });
        assert_eq!(ron(&config), ron(&expected));
    }

    /// Tuning is not part of the file format, so a file cannot set it.
    #[test]
    fn files_cannot_set_tuning() {
        assert!(rejects_field(
            "(version: 1, pipeline: OrbSlam((tuning: ())))",
            "tuning"
        ));
        let mut config = SystemConfig::default();
        orb_mut(&mut config).tuning.loss_recovery.timeout_imu_sec = 9.0;
        let reloaded = parse(&ron(&config)).unwrap();
        assert_eq!(
            orb(&reloaded).tuning.loss_recovery.timeout_imu_sec,
            OrbTuning::default().loss_recovery.timeout_imu_sec
        );
    }

    #[test]
    fn version_is_required() {
        assert!(matches!(
            parse_error("()"),
            ron::Error::MissingStructField {
                field: "version",
                ..
            }
        ));
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
        for (text, field) in [
            ("(version: 1, extra: 1)", "extra"),
            ("(version: 1, sensors: (lidar: true))", "lidar"),
            ("(version: 1, pipeline: OrbSlam((tracker: 1)))", "tracker"),
            (
                "(version: 1, pipeline: OrbSlam((frontend: Orb((n_levels: 8)))))",
                "n_levels",
            ),
            (
                "(version: 1, pipeline: OrbSlam((keyframes: (gap: 3))))",
                "gap",
            ),
            (
                "(version: 1, pipeline: OrbSlam((mapping: (window: 3))))",
                "window",
            ),
            (
                "(version: 1, pipeline: OrbSlam((loop_closing: Enabled(vocabulary: \"v.txt\", pgo: true))))",
                "pgo",
            ),
        ] {
            assert!(rejects_field(text, field), "{text}");
        }
    }

    #[test]
    fn loaded_files_are_validated() {
        let err = parse("(version: 1, pipeline: OrbSlam((frontend: Orb((n_keypoints: 0)))))")
            .unwrap_err();
        assert!(matches!(
            err,
            LoadError::Invalid(ConfigError::KeypointsOutOfRange { value: 0, .. })
        ));
    }

    #[test]
    fn vocabulary_resolves_against_base_dir() {
        let text = |path: &str| {
            format!(
                "(version: 1, sensors: (cameras: Stereo), pipeline: OrbSlam((loop_closing: Enabled(vocabulary: {path:?}))))"
            )
        };
        let base = Path::new("/configs");
        let relative = SystemConfig::from_ron_str(&text("../weights/ORBvoc.txt"), base).unwrap();
        assert_eq!(
            orb(&relative).loop_closing.vocabulary(),
            Some(Path::new("/configs/../weights/ORBvoc.txt"))
        );
        let absolute = SystemConfig::from_ron_str(&text("/weights/ORBvoc.bin"), base).unwrap();
        assert_eq!(
            orb(&absolute).loop_closing.vocabulary(),
            Some(Path::new("/weights/ORBvoc.bin"))
        );
    }

    #[test]
    fn file_paths_resolve_against_the_file_directory() {
        let dir = std::env::temp_dir().join(format!("kornia-slam-system-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orb.ron");
        std::fs::write(
            &path,
            "(version: 1, sensors: (cameras: Stereo), pipeline: OrbSlam((loop_closing: Enabled(vocabulary: \"ORBvoc.txt\"))))",
        )
        .unwrap();
        let loaded = SystemConfig::from_ron_file(&path);
        std::fs::remove_dir_all(&dir).unwrap();

        let config = loaded.unwrap();
        assert_eq!(
            orb(&config).loop_closing.vocabulary(),
            Some(dir.join("ORBvoc.txt").as_path())
        );
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = SystemConfig::from_ron_file("/nonexistent/orb.ron").unwrap_err();
        assert!(matches!(err, LoadError::Io { .. }));
    }

    #[test]
    fn serialization_round_trips() {
        let mut config = with_loop_closing(
            LoopClosingMode::Enabled {
                vocabulary: PathBuf::from("/weights/ORBvoc.bin"),
            },
            sensors(CameraSelection::Stereo, true),
        );
        let settings = orb_mut(&mut config);
        settings.frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints: 2000,
            stereo_close_depth: StereoCloseDepth::Metres(4.0),
        });
        settings.keyframes.ref_ratio = 0.75;
        settings.mapping.execution = LocalMappingMode::Synchronous;

        let text = ron(&config);
        let parsed = parse(&text).unwrap();
        assert_eq!(ron(&parsed), text);
        assert_ne!(text, ron(&SystemConfig::default()));
        assert_eq!(
            orb(&parsed).frontend,
            FrontendConfig::Orb(OrbFrontendConfig {
                n_keypoints: 2000,
                stereo_close_depth: StereoCloseDepth::Metres(4.0),
            })
        );
    }
}
