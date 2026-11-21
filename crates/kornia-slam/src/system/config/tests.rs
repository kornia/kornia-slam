use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_sensors::SensorRig;

use super::*;
use crate::mapping::LocalMappingMode;
use crate::mapping::bundle_adjustment::{DepthSigma, LocalBaParams};
use crate::mapping::map::{StateEnvelope, StateEnvelopeLimits};
use crate::mapping::marginalization::{BoundaryPrior, FrozenPoseSigma};
use crate::mapping::visual_information::{FisheyeWeight, VisualInformation};
use crate::system::ConfigError;
#[cfg(feature = "serde")]
use crate::system::LoadError;
use crate::tracking::optical_flow::LkKernel;
use crate::tracking::{KeyframePolicy, KeyframeRules};

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
        execution: LoopClosingExecution::default(),
        graph: Default::default(),
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
    let mut parallax = SystemConfig::default();
    let keyframes = &mut orb_mut(&mut parallax).keyframes;
    keyframes.max_frames_between = usize::MAX;
    keyframes.min_parallax_px = Some(10.0);
    for (config, expected) in [
        (
            SystemConfig::default(),
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  klt kernel: fixed-point u8
  klt-first: ORB only when a keyframe is due or fewer than 80 tracks survive
  klt prediction: levels 0..=1, 10 iterations when trusted
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: asynchronous
",
        ),
        (
            stereo,
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  klt kernel: fixed-point u8
  klt-first: ORB only when a keyframe is due or fewer than 80 tracks survive
  klt prediction: levels 0..=1, 10 iterations when trusted
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
  klt kernel: fixed-point u8
  klt-first: ORB only when a keyframe is due or fewer than 80 tracks survive
  klt prediction: levels 0..=1, 10 iterations when trusted
  IMU integration
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: synchronous
  loop closing: ORBvoc.txt (pose graph: w*I, snapshot)
",
        ),
        (
            parallax,
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  klt kernel: fixed-point u8
  klt-first: ORB only when a keyframe is due or fewer than 80 tracks survive
  klt prediction: levels 0..=1, 10 iterations when trusted
  tracking
  keyframes: every 3..=unbounded frames, ref ratio 0.6, parallax 10 px
  local mapping: asynchronous
",
        ),
    ] {
        assert_eq!(config.to_string(), expected);
    }
}

#[test]
fn the_viba_knobs_default_off_and_the_gate_is_range_checked() {
    assert_eq!(
        InertialConfig::default(),
        InertialConfig {
            sample_interpolation: SampleInterpolation::Hold,
            viba_window: VibaWindow::All,
            viba_velocity_gate: 0.0,
        }
    );
    let gate = |viba_velocity_gate| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).inertial.viba_velocity_gate = viba_velocity_gate;
        config.validate()
    };
    assert_eq!(gate(0.0), Ok(()));
    assert_eq!(gate(10.0), Ok(()));
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                gate(value),
                Err(ConfigError::InvalidSetting { setting, .. })
                    if setting == "inertial.viba_velocity_gate"
            ),
            "gate {value}"
        );
    }
}

#[test]
fn the_growth_baseline_ratio_defaults_off_and_is_range_checked() {
    assert_eq!(MappingConfig::default().growth_min_baseline_ratio, 0.0);
    let ratio = |growth_min_baseline_ratio| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).mapping.growth_min_baseline_ratio = growth_min_baseline_ratio;
        config.validate()
    };
    assert_eq!(ratio(0.0), Ok(()));
    assert_eq!(ratio(0.0175), Ok(()));
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                ratio(value),
                Err(ConfigError::InvalidSetting { setting, .. })
                    if setting == "mapping.growth_min_baseline_ratio"
            ),
            "ratio {value}"
        );
    }
}

#[test]
fn display_names_the_growth_baseline_ratio_only_when_set() {
    let mut config = SystemConfig::default();
    orb_mut(&mut config).mapping.growth_min_baseline_ratio = 0.0175;
    assert!(
        config
            .to_string()
            .contains("  growth: pairs under baseline/depth 0.0175 skipped\n")
    );
}

#[test]
fn display_names_the_fisheye_weight_only_when_set() {
    let mut config = SystemConfig::default();
    assert!(!config.to_string().contains("fisheye weight"));
    orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
        fisheye_weight: FisheyeWeight::Magnification,
        ..OrbFrontendConfig::default()
    });
    assert!(
        config
            .to_string()
            .contains("  fisheye weight: Magnification\n")
    );
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

    let parallax = |min_parallax_px| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).keyframes.min_parallax_px = min_parallax_px;
        invalid_setting(&config)
    };
    assert_eq!(parallax(None), None);
    assert_eq!(parallax(Some(10.0)), None);
    for px in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            parallax(Some(px)).as_deref(),
            Some("keyframes.min_parallax_px"),
            "parallax {px}"
        );
    }

    let interval = |max_interval_sec| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).keyframes.max_interval_sec = max_interval_sec;
        invalid_setting(&config)
    };
    assert_eq!(interval(Some(0.5)), None);
    for secs in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            interval(Some(secs)).as_deref(),
            Some("keyframes.max_interval_sec"),
            "interval {secs}"
        );
    }
}

/// A parked inertial rig with no keyframe bound would buffer IMU samples, and
/// grow the next IMU edge, for the whole stop.
#[test]
fn an_inertial_system_needs_a_keyframe_bound() {
    let parked = |imu, max_frames_between, max_interval_sec| {
        let mut config = SystemConfig {
            sensors: sensors(CameraSelection::Stereo, imu),
            ..SystemConfig::default()
        };
        let keyframes = &mut orb_mut(&mut config).keyframes;
        keyframes.max_frames_between = max_frames_between;
        keyframes.min_parallax_px = Some(10.0);
        keyframes.max_interval_sec = max_interval_sec;
        config.validate()
    };
    assert_eq!(
        parked(true, usize::MAX, None),
        Err(ConfigError::UnboundedInertialKeyframes)
    );
    assert_eq!(parked(true, usize::MAX, Some(0.5)), Ok(()));
    assert_eq!(parked(true, 30, None), Ok(()));
    assert_eq!(parked(false, usize::MAX, None), Ok(()));
}

fn invalid_setting(config: &SystemConfig) -> Option<String> {
    match config.validate() {
        Err(ConfigError::InvalidSetting { setting, .. }) => Some(setting),
        _ => None,
    }
}

#[test]
fn pose_gate_limits_must_be_positive_and_finite() {
    use crate::tracking::pose_gate::{PoseGate, PoseGateLimits};
    let mut config = SystemConfig {
        sensors: sensors(CameraSelection::Stereo, true),
        ..SystemConfig::default()
    };
    orb_mut(&mut config).tracking_loss.pose_gate = PoseGate::Enabled(PoseGateLimits {
        max_translation_gap_mps: f64::INFINITY,
        ..PoseGateLimits::default()
    });
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tracking_loss.pose_gate.max_translation_gap_mps")
    );
    orb_mut(&mut config).tracking_loss.pose_gate = PoseGate::Enabled(PoseGateLimits {
        max_rotation_gap_deg: 0.0,
        ..PoseGateLimits::default()
    });
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tracking_loss.pose_gate.max_rotation_gap_deg")
    );
    orb_mut(&mut config).tracking_loss.pose_gate = PoseGate::Enabled(PoseGateLimits::default());
    assert!(config.validate().is_ok());
    assert!(config.to_string().contains("pose gate: 5 deg"));
}

#[test]
fn fixed_inertial_lost_keyframes_need_keyframes_inserted_as_full_states() {
    use crate::mapping::bundle_adjustment::LostKeyframes;
    use crate::tracking::KeyframesWhileLost;
    let with = |lost_keyframes, keyframes_while_lost| {
        let mut config = SystemConfig {
            sensors: sensors(CameraSelection::Stereo, true),
            ..SystemConfig::default()
        };
        orb_mut(&mut config).mapping.local_ba.lost_keyframes = lost_keyframes;
        orb_mut(&mut config).tracking_loss.keyframes_while_lost = keyframes_while_lost;
        config.validate()
    };
    for keyframes_while_lost in [
        KeyframesWhileLost::Enabled,
        KeyframesWhileLost::Anchor,
        KeyframesWhileLost::Disabled,
    ] {
        assert_eq!(with(LostKeyframes::Free, keyframes_while_lost), Ok(()));
    }
    assert_eq!(
        with(LostKeyframes::FixedInertial, KeyframesWhileLost::Enabled),
        Ok(())
    );
    for inert in [KeyframesWhileLost::Anchor, KeyframesWhileLost::Disabled] {
        assert_eq!(
            with(LostKeyframes::FixedInertial, inert),
            Err(ConfigError::LostKeyframesWithoutEffect(inert))
        );
    }
}

#[test]
fn the_reseed_tilt_limit_must_be_positive() {
    let mut config = SystemConfig::default();
    orb_mut(&mut config).tracking_loss.reseed.max_tilt_deg = 0.0;
    assert_eq!(
        invalid_setting(&config).as_deref(),
        Some("tracking_loss.reseed.max_tilt_deg")
    );
}

#[test]
fn relocalization_knobs_are_off_by_default_and_need_loop_closing() {
    use crate::tracking::relocalization::RelocalizationSolver;
    let mut defaults = SystemConfig::default();
    let tracking_loss = &orb_mut(&mut defaults).tracking_loss;
    assert!(!tracking_loss.relocalization.while_lost);
    assert_eq!(
        tracking_loss.relocalization.solver,
        RelocalizationSolver::PrimaryEpnp
    );
    assert_eq!(tracking_loss.relocalization.max_jump_m, 0.5);
    assert!(!tracking_loss.reseed.relocalize_when_lost);
    assert_eq!(tracking_loss.reseed.own_window_keyframes, 0);

    type Knob = (&'static str, fn(&mut OrbSlamPipeline));
    let knobs: [Knob; 3] = [
        ("tracking_loss.relocalization.while_lost", |orb| {
            orb.tracking_loss.relocalization.while_lost = true;
        }),
        ("tracking_loss.relocalization.solver", |orb| {
            orb.tracking_loss.relocalization.solver = RelocalizationSolver::RigAp3p;
        }),
        ("tracking_loss.reseed.relocalize_when_lost", |orb| {
            orb.tracking_loss.reseed.relocalize_when_lost = true;
        }),
    ];
    let stereo = sensors(CameraSelection::Stereo, false);
    for (knob, set) in knobs {
        let mut without = with_loop_closing(LoopClosingMode::Disabled, stereo);
        set(orb_mut(&mut without));
        assert_eq!(
            without.validate(),
            Err(ConfigError::RelocalizationWithoutVocabulary(knob))
        );
        let mut with = with_loop_closing(loop_closing(), stereo);
        set(orb_mut(&mut with));
        assert_eq!(with.validate(), Ok(()), "{knob}");
    }
    for mode in [LoopClosingMode::Disabled, loop_closing()] {
        assert_eq!(with_loop_closing(mode, stereo).validate(), Ok(()));
    }
}

#[test]
fn the_relocalization_jump_bound_must_be_positive() {
    for bad in [0.0, -1.0, f64::INFINITY] {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).tracking_loss.relocalization.max_jump_m = bad;
        assert_eq!(
            invalid_setting(&config).as_deref(),
            Some("tracking_loss.relocalization.max_jump_m")
        );
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

    for (local_ba, setting) in [
        (
            LocalBaConfig {
                active_keyframes: 0,
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.active_keyframes",
        ),
        (
            LocalBaConfig {
                boundary_imu_info_scale: f64::NAN,
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.boundary_imu_info_scale",
        ),
        (
            LocalBaConfig {
                max_iterations: 0,
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.max_iterations",
        ),
        (
            LocalBaConfig {
                boundary: BoundaryPrior::Marginal(FrozenPoseSigma {
                    frozen_pose_sigma_m: 0.0,
                    ..FrozenPoseSigma::default()
                }),
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.boundary",
        ),
        (
            LocalBaConfig {
                state_envelope: StateEnvelope::On(StateEnvelopeLimits {
                    max_accel_bias: f64::NAN,
                    ..StateEnvelopeLimits::default()
                }),
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.state_envelope",
        ),
        (
            LocalBaConfig {
                stereo_depth_sigma: StereoDepthSigma::Disparity { px: 0.0 },
                ..LocalBaConfig::default()
            },
            "mapping.local_ba.stereo_depth_sigma",
        ),
    ] {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).mapping.local_ba = local_ba;
        assert_eq!(invalid_setting(&config).as_deref(), Some(setting));
    }
}

#[test]
fn loop_graph_information_is_range_checked() {
    use crate::loop_closure::OdometryRandomWalk;
    let with = |information| {
        with_loop_closing(
            LoopClosingMode::Enabled {
                vocabulary: PathBuf::from("ORBvoc.txt"),
                graph: LoopGraphSettings {
                    information,
                    persistent: true,
                },
                execution: LoopClosingExecution::default(),
            },
            sensors(CameraSelection::Stereo, false),
        )
    };
    let drift = |edit: fn(&mut OdometryRandomWalk, &mut [f64; 3])| {
        let LoopGraphInformation::Drift {
            mut odometry,
            loop_sigma_m,
            loop_sigma_rad,
            loop_cauchy_scale,
        } = LoopGraphInformation::drift()
        else {
            unreachable!()
        };
        let mut loop_values = [loop_sigma_m, loop_sigma_rad, loop_cauchy_scale];
        edit(&mut odometry, &mut loop_values);
        let [loop_sigma_m, loop_sigma_rad, loop_cauchy_scale] = loop_values;
        invalid_setting(&with(LoopGraphInformation::Drift {
            odometry,
            loop_sigma_m,
            loop_sigma_rad,
            loop_cauchy_scale,
        }))
    };
    let at = |field: &str| Some(format!("pipeline.loop_closing.graph.information.{field}"));
    assert_eq!(drift(|_, _| {}), None);
    assert_eq!(with(LoopGraphInformation::ScalarWeights).validate(), Ok(()));
    assert_eq!(
        drift(|walk, _| walk.translation_m2_per_m = -1e-3),
        at("odometry.translation_m2_per_m")
    );
    assert_eq!(
        drift(|walk, _| walk.rotation_rad2_per_m = f64::NAN),
        at("odometry.rotation_rad2_per_m")
    );
    assert_eq!(drift(|walk, _| walk.floor_m = 0.0), at("odometry.floor_m"));
    assert_eq!(
        drift(|walk, _| walk.floor_rad = f64::INFINITY),
        at("odometry.floor_rad")
    );
    assert_eq!(drift(|_, values| values[0] = 0.0), at("loop_sigma_m"));
    assert_eq!(drift(|_, values| values[1] = -1.0), at("loop_sigma_rad"));
    assert_eq!(drift(|_, values| values[2] = 0.0), at("loop_cauchy_scale"));
}

#[test]
fn vocabulary_path_must_not_be_empty() {
    let config = with_loop_closing(
        LoopClosingMode::Enabled {
            vocabulary: PathBuf::new(),
            execution: LoopClosingExecution::default(),
            graph: Default::default(),
        },
        SensorSelection::default(),
    );
    assert_eq!(config.validate(), Err(ConfigError::EmptyVocabularyPath));
}

/// Loop closing solves inline unless a configuration asks for the worker, whatever the mapping
/// execution.
#[test]
fn default_loop_closing_execution_is_synchronous() {
    for mapping in [
        LocalMappingMode::Synchronous,
        LocalMappingMode::Asynchronous,
    ] {
        assert_eq!(
            LoopClosingExecution::default().resolve(mapping),
            LoopClosingExecution::Synchronous
        );
    }
    assert_eq!(
        LoopClosingExecution::default().resolve(LocalMappingMode::default()),
        LoopClosingExecution::Synchronous
    );
}

#[test]
fn follow_mapping_resolves_to_the_mapping_execution() {
    use LoopClosingExecution::*;
    assert_eq!(
        FollowMapping.resolve(LocalMappingMode::Synchronous),
        Synchronous
    );
    assert_eq!(
        FollowMapping.resolve(LocalMappingMode::Asynchronous),
        Asynchronous
    );
    assert_eq!(
        Deferred { frames: 2 }.resolve(LocalMappingMode::Asynchronous),
        Deferred { frames: 2 }
    );
    assert_eq!(
        Synchronous.resolve(LocalMappingMode::Asynchronous),
        Synchronous
    );
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

fn raw_fisheye_rig() -> SensorRig {
    let primary = rig().camera;
    let view = |primary_from_camera| kornia_sensors::RigCamera {
        camera: primary.clone(),
        size: kornia_image::ImageSize {
            width: 640,
            height: 480,
        },
        primary_from_camera,
        raw_model: Some(kornia_sensors::Fisheye624Camera {
            f: primary.fx,
            cx: primary.cx,
            cy: primary.cy,
            k: [-0.025, 0.096, -0.063, 0.0062, 0.0035, -0.00073],
            p: [0.0009, 0.00036],
            s: [-0.0014, -0.0001, -0.00046, -0.0001],
        }),
    };
    rig().with_rig_cameras(vec![
        view(Pose3d::IDENTITY),
        view(Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
        )),
    ])
}

#[test]
fn raw_fisheye_rig_cameras_need_the_rig_selection() {
    let rig = raw_fisheye_rig();
    for cameras in [CameraSelection::Mono, CameraSelection::Stereo] {
        let rig = rig.clone().with_stereo_baseline(0.11);
        assert_eq!(
            sensors(cameras, false).validate_rig(&rig),
            Err(ConfigError::RawFisheyeOutsideRig)
        );
    }
    assert_eq!(
        sensors(CameraSelection::Rig, false).validate_rig(&rig),
        Ok(())
    );
}

#[test]
fn raw_fisheye_weight_reaches_local_ba_and_projection_on_raw_rigs_only() {
    let mut config = SystemConfig::default();
    orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
        fisheye_weight: FisheyeWeight::Magnification,
        ..OrbFrontendConfig::default()
    });
    let rig_selection = sensors(CameraSelection::Rig, false);
    let raw = config.settings(&rig_selection.select_rig(raw_fisheye_rig()).unwrap());
    assert_eq!(raw.local_ba.fisheye_weight, FisheyeWeight::Magnification);
    assert_eq!(
        raw.map_projection.fisheye_weight,
        FisheyeWeight::Magnification
    );
    let pinhole = config.settings(&rig());
    assert_eq!(pinhole.local_ba.fisheye_weight, FisheyeWeight::Uniform);
    assert_eq!(
        pinhole.map_projection.fisheye_weight,
        FisheyeWeight::Uniform
    );
    let default =
        SystemConfig::default().settings(&rig_selection.select_rig(raw_fisheye_rig()).unwrap());
    assert_eq!(default.local_ba.fisheye_weight, FisheyeWeight::Uniform);
}

#[test]
fn raw_fisheye_rig_rejects_klt_first() {
    let selected = sensors(CameraSelection::Rig, false)
        .select_rig(raw_fisheye_rig())
        .unwrap();
    let with_klt = |klt_first| {
        let mut config = SystemConfig::default();
        orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
            klt_first,
            ..OrbFrontendConfig::default()
        });
        config
    };
    assert_eq!(
        with_klt(KltFirst::default()).validate_raw_fisheye(&selected),
        Err(ConfigError::FrontendKnobNeeds {
            knob: "klt_first",
            needs: "pinhole rig images",
        })
    );
    assert_eq!(
        with_klt(KltFirst::Disabled).validate_raw_fisheye(&selected),
        Ok(())
    );
    assert_eq!(
        with_klt(KltFirst::default()).validate_raw_fisheye(&rig()),
        Ok(())
    );
}

/// The Orin preset of card E: 2 threads with 48 tracks per camera, or 3
/// with 64 (`configs/lamaria-rig-imu-orin{2,3}.ron`).
fn orin_preset(threads: usize, per_camera: usize) -> SystemConfig {
    let mut config = SystemConfig {
        sensors: sensors(CameraSelection::Rig, true),
        ..SystemConfig::default()
    };
    let orb = orb_mut(&mut config);
    orb.mapping.execution = LocalMappingMode::Asynchronous;
    orb.frontend = FrontendConfig::Orb(OrbFrontendConfig {
        klt_kernel: LkKernel::WindowedU8,
        flow_threads: threads,
        klt_rig: KltRig::AllCameras,
        klt_window: 11,
        klt_prediction: KltPrediction::Enabled {
            max_level: 0,
            max_iter: 8,
        },
        klt_retry: KltRetry::CoarseLevel { max_level: 1 },
        forward_backward: ForwardBackward::Enabled {
            max_round_trip_px: 0.6,
            max_iter: 4,
            every_n_frames: 1,
        },
        track_refill: TrackRefill::Enabled {
            per_camera,
            min_spacing_px: 23.0,
            refill_below: 0.8,
            border_px: 9,
        },
        adopt_landmarks: true,
        gyro_guess: GyroGuess::RotatedRay,
        flow_pose: FlowPose::AnalyticGn { max_iter: 3 },
        flow_bookkeeping: FlowBookkeeping::KeyframesOnly,
        keyframe_orb: KeyframeOrb::Mapping,
        ..OrbFrontendConfig::default()
    });
    config
}

/// A config whose frontend is `frontend` on `cameras`, with the IMU.
fn with_frontend(frontend: OrbFrontendConfig, cameras: CameraSelection) -> SystemConfig {
    let mut config = SystemConfig {
        sensors: sensors(cameras, true),
        ..SystemConfig::default()
    };
    orb_mut(&mut config).frontend = FrontendConfig::Orb(frontend);
    config
}

#[test]
fn the_flow_knobs_default_to_todays_frontend() {
    let frontend = OrbFrontendConfig::default();
    assert_eq!(frontend.flow_threads, 1);
    assert_eq!(frontend.flow_first_core, None);
    assert_eq!(frontend.klt_rig, KltRig::PrimaryOnly);
    assert_eq!(frontend.klt_kernel, LkKernel::FixedPointU8);
    assert_eq!(frontend.klt_window, 21);
    assert_eq!(frontend.klt_prediction, KltPrediction::default());
    assert_eq!(frontend.klt_retry, KltRetry::Off);
    assert_eq!(frontend.forward_backward, ForwardBackward::Off);
    assert_eq!(frontend.track_refill, TrackRefill::Off);
    assert!(!frontend.adopt_landmarks);
    assert_eq!(frontend.gyro_guess, GyroGuess::Off);
    assert_eq!(frontend.anchor_guard, AnchorGuard::Off);
    assert_eq!(frontend.flow_pose, FlowPose::RansacPnp);
    assert_eq!(frontend.flow_bookkeeping, FlowBookkeeping::EveryFrame);
    assert_eq!(frontend.keyframe_orb, KeyframeOrb::Tracking);
    assert_eq!(frontend.stereo_pyramid, StereoPyramid::Resized);
    assert_eq!(frontend.fisheye_weight, FisheyeWeight::Uniform);
    for threads in [2, 3] {
        let preset = orin_preset(threads, 48);
        assert_eq!(preset.validate(), Ok(()), "{threads} threads");
        assert!(preset.to_string().contains("flow threads"));
    }
}

#[test]
fn flow_knob_ranges_are_checked() {
    let windowed = OrbFrontendConfig {
        klt_kernel: LkKernel::WindowedU8,
        ..OrbFrontendConfig::default()
    };
    for (frontend, setting) in [
        (
            OrbFrontendConfig {
                flow_threads: 0,
                ..windowed
            },
            "frontend.flow_threads",
        ),
        (
            OrbFrontendConfig {
                flow_threads: MAX_FLOW_THREADS + 1,
                ..windowed
            },
            "frontend.flow_threads",
        ),
        (
            OrbFrontendConfig {
                klt_window: 10,
                ..windowed
            },
            "frontend.klt_window",
        ),
        (
            OrbFrontendConfig {
                klt_window: 1,
                ..windowed
            },
            "frontend.klt_window",
        ),
        (
            OrbFrontendConfig {
                klt_retry: KltRetry::CoarseLevel { max_level: 0 },
                ..windowed
            },
            "frontend.klt_retry.max_level",
        ),
        (
            OrbFrontendConfig {
                forward_backward: ForwardBackward::Enabled {
                    max_round_trip_px: 0.0,
                    max_iter: 4,
                    every_n_frames: 1,
                },
                ..windowed
            },
            "frontend.forward_backward.max_round_trip_px",
        ),
        (
            OrbFrontendConfig {
                forward_backward: ForwardBackward::Enabled {
                    max_round_trip_px: f32::NAN,
                    max_iter: 4,
                    every_n_frames: 1,
                },
                ..windowed
            },
            "frontend.forward_backward.max_round_trip_px",
        ),
        (
            OrbFrontendConfig {
                forward_backward: ForwardBackward::Enabled {
                    max_round_trip_px: 0.6,
                    max_iter: 0,
                    every_n_frames: 1,
                },
                ..windowed
            },
            "frontend.forward_backward.max_iter",
        ),
        (
            OrbFrontendConfig {
                forward_backward: ForwardBackward::Enabled {
                    max_round_trip_px: 0.6,
                    max_iter: 4,
                    every_n_frames: 0,
                },
                ..windowed
            },
            "frontend.forward_backward.every_n_frames",
        ),
        (
            OrbFrontendConfig {
                track_refill: TrackRefill::Enabled {
                    per_camera: 0,
                    min_spacing_px: 23.0,
                    refill_below: 0.8,
                    border_px: 9,
                },
                ..windowed
            },
            "frontend.track_refill.per_camera",
        ),
        (
            OrbFrontendConfig {
                track_refill: TrackRefill::Enabled {
                    per_camera: 48,
                    min_spacing_px: 0.0,
                    refill_below: 0.8,
                    border_px: 9,
                },
                ..windowed
            },
            "frontend.track_refill.min_spacing_px",
        ),
        (
            OrbFrontendConfig {
                track_refill: TrackRefill::Enabled {
                    per_camera: 48,
                    min_spacing_px: 23.0,
                    refill_below: 0.0,
                    border_px: 9,
                },
                ..windowed
            },
            "frontend.track_refill.refill_below",
        ),
        (
            OrbFrontendConfig {
                track_refill: TrackRefill::Enabled {
                    per_camera: 48,
                    min_spacing_px: 23.0,
                    refill_below: 1.5,
                    border_px: 9,
                },
                ..windowed
            },
            "frontend.track_refill.refill_below",
        ),
        (
            OrbFrontendConfig {
                anchor_guard: AnchorGuard::Enabled {
                    every_n_frames: 0,
                    max_drift_px: 1.0,
                },
                ..windowed
            },
            "frontend.anchor_guard.every_n_frames",
        ),
        (
            OrbFrontendConfig {
                anchor_guard: AnchorGuard::Enabled {
                    every_n_frames: 5,
                    max_drift_px: 0.0,
                },
                ..windowed
            },
            "frontend.anchor_guard.max_drift_px",
        ),
        (
            OrbFrontendConfig {
                flow_pose: FlowPose::AnalyticGn { max_iter: 0 },
                ..windowed
            },
            "frontend.flow_pose.max_iter",
        ),
    ] {
        let config = with_frontend(frontend, CameraSelection::Rig);
        assert_eq!(invalid_setting(&config).as_deref(), Some(setting));
    }
}

#[test]
fn flow_knobs_are_refused_without_the_knobs_they_need() {
    let needs = |knob: &'static str, needs: &'static str| -> Result<(), ConfigError> {
        Err(ConfigError::FrontendKnobNeeds { knob, needs })
    };
    let windowed = OrbFrontendConfig {
        klt_kernel: LkKernel::WindowedU8,
        ..OrbFrontendConfig::default()
    };
    let threaded = OrbFrontendConfig {
        flow_threads: 2,
        klt_rig: KltRig::AllCameras,
        ..windowed
    };
    let on_rig = |frontend| with_frontend(frontend, CameraSelection::Rig).validate();
    assert_eq!(on_rig(threaded), Ok(()));
    assert_eq!(
        on_rig(OrbFrontendConfig {
            klt_rig: KltRig::PrimaryOnly,
            ..threaded
        }),
        needs("flow_threads", "klt_rig AllCameras on a rig")
    );
    assert_eq!(
        with_frontend(threaded, CameraSelection::Mono).validate(),
        needs("flow_threads", "klt_rig AllCameras on a rig")
    );
    assert_eq!(
        on_rig(OrbFrontendConfig {
            klt_window: 11,
            klt_kernel: LkKernel::F32,
            ..OrbFrontendConfig::default()
        }),
        needs(
            "klt_window",
            "klt_kernel FixedPointU8 or WindowedU8 unless it is 21"
        )
    );
    assert_eq!(
        on_rig(OrbFrontendConfig {
            klt_window: 11,
            ..OrbFrontendConfig::default()
        }),
        Ok(())
    );
    let fixed_point = OrbFrontendConfig::default();
    for (frontend, knob) in [
        (
            OrbFrontendConfig {
                klt_retry: KltRetry::CoarseLevel { max_level: 1 },
                ..fixed_point
            },
            "klt_retry",
        ),
        (
            OrbFrontendConfig {
                forward_backward: ForwardBackward::Enabled {
                    max_round_trip_px: 0.6,
                    max_iter: 4,
                    every_n_frames: 1,
                },
                ..fixed_point
            },
            "forward_backward",
        ),
        (
            OrbFrontendConfig {
                track_refill: TrackRefill::Enabled {
                    per_camera: 48,
                    min_spacing_px: 23.0,
                    refill_below: 0.8,
                    border_px: 9,
                },
                ..fixed_point
            },
            "track_refill",
        ),
        (
            OrbFrontendConfig {
                anchor_guard: AnchorGuard::Enabled {
                    every_n_frames: 5,
                    max_drift_px: 1.0,
                },
                ..fixed_point
            },
            "anchor_guard",
        ),
        (
            OrbFrontendConfig {
                klt_kernel: LkKernel::FixedPointU8,
                ..threaded
            },
            "flow_threads",
        ),
    ] {
        assert_eq!(
            on_rig(frontend),
            needs(knob, "klt_kernel WindowedU8"),
            "{knob}"
        );
    }
    assert_eq!(
        on_rig(OrbFrontendConfig {
            keyframe_orb: KeyframeOrb::Mapping,
            klt_first: KltFirst::Disabled,
            ..OrbFrontendConfig::default()
        }),
        needs("keyframe_orb", "klt_first Enabled")
    );
}

#[test]
fn flow_threads_need_a_shard_layout_for_the_rig() {
    for (threads, cameras, ok) in [
        (1, 0, true),
        (1, 2, true),
        (2, 2, true),
        (3, 2, true),
        (4, 2, false),
        (2, 0, false),
    ] {
        let frontend = OrbFrontendConfig {
            flow_threads: threads,
            ..OrbFrontendConfig::default()
        };
        let expected = if ok {
            Ok(())
        } else {
            Err(ConfigError::FlowThreadsForRig { threads, cameras })
        };
        assert_eq!(
            frontend.validate_flow_threads(cameras),
            expected,
            "{threads} threads, {cameras} cameras"
        );
    }
}

#[test]
fn online_calibration_is_off_by_default_and_checked() {
    use crate::mapping::global_vi_ba::{ClockOffsetPrior, RotationSlicePrior};
    let default = LocalBaConfig::default().online_calibration;
    assert_eq!(default, OnlineCalibrationConfig::default());
    assert!(!default.is_enabled());
    assert_eq!(default.camera_rotation, None);
    assert_eq!(default.camera_focal, None);
    assert_eq!(default.gyro_intrinsics, None);
    assert_eq!(default.clock_offset, None);

    let with = |online_calibration, imu| {
        let mut config = SystemConfig {
            sensors: sensors(CameraSelection::Mono, imu),
            ..SystemConfig::default()
        };
        orb_mut(&mut config).mapping.local_ba.online_calibration = online_calibration;
        config
    };
    let td = OnlineCalibrationConfig {
        clock_offset: Some(ClockOffsetPrior { sigma_ms: 2.0 }),
        ..OnlineCalibrationConfig::default()
    };
    assert!(td.is_enabled());
    assert_eq!(with(td, true).validate(), Ok(()));
    assert_eq!(
        with(td, false).validate(),
        Err(ConfigError::InertialKnobWithoutImu(
            "mapping.local_ba.online_calibration"
        ))
    );
    let zero = OnlineCalibrationConfig {
        camera_rotation: Some(RotationSlicePrior {
            step_sigma_deg: 0.0,
            value_sigma_deg: 1.0,
        }),
        ..OnlineCalibrationConfig::default()
    };
    assert_eq!(
        invalid_setting(&with(zero, true)).as_deref(),
        Some("mapping.local_ba.online_calibration.camera_rotation.step_sigma_deg")
    );
}

mod runtime_settings {
    use super::*;

    /// `klt_rig: AllCameras` lets a rig defer ORB to flow frames; the default
    /// keeps ORB on every rig frame.
    #[test]
    fn klt_first_reaches_a_rig_only_with_every_camera_tracked() {
        let primary = rig().camera;
        let view = |primary_from_camera| kornia_sensors::RigCamera {
            camera: primary.clone(),
            size: kornia_image::ImageSize {
                width: 640,
                height: 480,
            },
            primary_from_camera,
            raw_model: None,
        };
        let two_cameras = rig().with_rig_cameras(vec![
            view(Pose3d::IDENTITY),
            view(Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
            )),
        ]);
        let selected = sensors(CameraSelection::Rig, false)
            .select_rig(two_cameras)
            .unwrap();
        let klt_min_tracks = |klt_rig| {
            let mut config = SystemConfig::default();
            orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
                klt_rig,
                ..OrbFrontendConfig::default()
            });
            config.settings(&selected).klt_min_tracks
        };
        assert_eq!(klt_min_tracks(KltRig::PrimaryOnly), None);
        assert_eq!(klt_min_tracks(KltRig::AllCameras), Some(80));
        assert_eq!(
            SystemConfig::default().settings(&rig()).klt_min_tracks,
            Some(80)
        );
    }

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
            min_parallax_px: Some(12.0),
            max_interval_sec: Some(0.5),
            rules: KeyframeRules::Kornia,
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

    /// The default knobs are the solver's constants, bit for bit; a disparity
    /// sigma resolves against focal length times baseline.
    #[test]
    fn local_ba_settings_resolve_against_the_rig() {
        let selected = sensors(CameraSelection::Stereo, false)
            .select_rig(stereo_imu_rig())
            .unwrap();
        assert_eq!(
            SystemConfig::default().settings(&selected).local_ba,
            LocalBaParams::default()
        );
        assert_eq!(LocalBaParams::default().max_iterations, 50);
        let mut config = SystemConfig::default();
        orb_mut(&mut config).mapping.local_ba = LocalBaConfig {
            active_keyframes: 10,
            max_iterations: 10,
            stereo_depth_sigma: StereoDepthSigma::Disparity { px: 1.0 },
            visual_information: VisualInformation::OctaveScaled,
            dense_parallelism: DenseParallelism::Sequential,
            ..LocalBaConfig::default()
        };
        let settings = config.settings(&selected);
        assert_eq!(
            settings.map_projection.pnp.visual_information,
            VisualInformation::OctaveScaled
        );
        let local_ba = settings.local_ba;
        assert_eq!(local_ba.visual_information, VisualInformation::OctaveScaled);
        assert_eq!(local_ba.active_keyframes, 10);
        assert_eq!(local_ba.max_iterations, 10);
        assert_eq!(local_ba.dense_parallelism, DenseParallelism::Sequential);
        let fb = selected.camera.fx * 0.11;
        assert_eq!(
            local_ba.depth_sigma,
            DepthSigma::Disparity {
                px_over_focal_baseline: (1.0 / fb) as f32
            }
        );
    }

    /// G11: off by default; when set it reaches the PnP refine.
    #[test]
    fn refine_monotone_reaches_the_pnp_refine() {
        let selected = rig();
        let monotone = |c: &SystemConfig| c.settings(&selected).map_projection.pnp.refine_monotone;
        let mut config = SystemConfig::default();
        assert!(!monotone(&config));
        orb_mut(&mut config).tracking.refine_monotone = true;
        assert!(monotone(&config));
    }

    #[test]
    fn only_the_rig_selection_keeps_the_rig_cameras() {
        let primary = rig().camera;
        let view = |primary_from_camera| kornia_sensors::RigCamera {
            camera: primary.clone(),
            size: kornia_image::ImageSize {
                width: 640,
                height: 480,
            },
            primary_from_camera,
            raw_model: None,
        };
        let two_cameras = rig().with_rig_cameras(vec![
            view(Pose3d::IDENTITY),
            view(Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
            )),
        ]);
        let selected = sensors(CameraSelection::Rig, false)
            .select_rig(two_cameras.clone())
            .unwrap();
        assert_eq!(selected.cameras.len(), 2);
        let mono = sensors(CameraSelection::Mono, false)
            .select_rig(two_cameras)
            .unwrap();
        assert!(mono.cameras.is_empty());
        assert!(matches!(
            sensors(CameraSelection::Rig, false).select_rig(rig()),
            Err(ConfigError::MissingSensor(_))
        ));
    }
}

#[cfg(feature = "serde")]
mod ron_files {
    use std::path::Path;

    use super::*;
    use crate::loop_closure::OdometryRandomWalk;

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
    fn the_viba_knobs_parse_and_round_trip() {
        let config = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((inertial: \
             (viba_window: VisualOnly, viba_velocity_gate: 10.0))))",
        )
        .unwrap();
        assert_eq!(orb(&config).inertial.viba_window, VibaWindow::VisualOnly);
        assert_eq!(orb(&config).inertial.viba_velocity_gate, 10.0);
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let default = parse("(version: 1)").unwrap();
        assert_eq!(orb(&default).inertial, InertialConfig::default());
    }

    #[test]
    fn the_growth_baseline_ratio_parses_and_round_trips() {
        let config = parse(
            "(version: 1, pipeline: OrbSlam((mapping: (growth_min_baseline_ratio: 0.0175))))",
        )
        .unwrap();
        assert_eq!(orb(&config).mapping.growth_min_baseline_ratio, 0.0175);
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let default = parse("(version: 1)").unwrap();
        assert_eq!(orb(&default).mapping.growth_min_baseline_ratio, 0.0);
    }

    #[test]
    fn the_dense_parallelism_parses_and_round_trips() {
        let default = parse("(version: 1)").unwrap();
        assert_eq!(
            orb(&default).mapping.local_ba.dense_parallelism,
            DenseParallelism::Global
        );

        let config = parse(
            "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (dense_parallelism: Sequential)))))",
        )
        .unwrap();
        assert_eq!(
            orb(&config).mapping.local_ba.dense_parallelism,
            DenseParallelism::Sequential
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));
        assert_eq!(config.validate(), Ok(()));
        assert!(matches!(
            parse_error("(version: 1, pipeline: OrbSlam((mapping: (local_ba: (dense_parallelism: Bogus)))))"),
            ron::Error::NoSuchEnumVariant { found, .. } if found == "Bogus"
        ));
    }

    #[test]
    fn the_fisheye_weight_parses_and_round_trips() {
        let config = parse(
            "(version: 1, pipeline: OrbSlam((frontend: Orb((fisheye_weight: Magnification)))))",
        )
        .unwrap();
        let FrontendConfig::Orb(frontend) = orb(&config).frontend;
        assert_eq!(frontend.fisheye_weight, FisheyeWeight::Magnification);
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let default = parse("(version: 1)").unwrap();
        let FrontendConfig::Orb(frontend) = orb(&default).frontend;
        assert_eq!(frontend.fisheye_weight, FisheyeWeight::Uniform);
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

        let config = parse(
            "(version: 1, pipeline: OrbSlam((keyframes: (max_frames_between: 18446744073709551615, min_parallax_px: Some(10.0), max_interval_sec: Some(0.5)))))",
        )
        .unwrap();
        assert_eq!(orb(&config).keyframes.max_frames_between, usize::MAX);
        assert_eq!(orb(&config).keyframes.min_parallax_px, Some(10.0));
        assert_eq!(orb(&config).keyframes.max_interval_sec, Some(0.5));

        let config = parse(
            "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (active_keyframes: 10, imu_huber: BoundaryOnly, stereo_depth_sigma: Disparity(px: 1.0))))))",
        )
        .unwrap();
        let local_ba = orb(&config).mapping.local_ba;
        assert_eq!(local_ba.active_keyframes, 10);
        assert_eq!(local_ba.imu_huber, ImuHuber::BoundaryOnly);
        assert_eq!(
            local_ba.stereo_depth_sigma,
            StereoDepthSigma::Disparity { px: 1.0 }
        );
        assert!(local_ba.boundary_scales_bias_rows);
        assert_eq!(local_ba.boundary, BoundaryPrior::ScaledEdge);
        assert_eq!(local_ba.max_iterations, 50);

        let config =
            parse("(version: 1, pipeline: OrbSlam((mapping: (local_ba: (max_iterations: 10)))))")
                .unwrap();
        assert_eq!(orb(&config).mapping.local_ba.max_iterations, 10);
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let config = parse(
            "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (boundary: Marginal((frozen_pose_sigma_m: 0.002, frozen_pose_sigma_rad: 0.001)))))))",
        )
        .unwrap();
        assert_eq!(
            orb(&config).mapping.local_ba.boundary,
            BoundaryPrior::Marginal(FrozenPoseSigma::default())
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));
        assert!(matches!(
            parse_error("(version: 1, pipeline: OrbSlam((mapping: (local_ba: (boundary: Bogus)))))"),
            ron::Error::NoSuchEnumVariant { found, .. } if found == "Bogus"
        ));

        assert_eq!(
            orb(&parse("(version: 1)").unwrap())
                .mapping
                .local_ba
                .state_envelope,
            StateEnvelope::Off
        );
        let config = parse(
            "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (state_envelope: On((max_speed_mps: 5.0, max_accel_bias: 0.5, max_gyro_bias: 0.3)))))))",
        )
        .unwrap();
        assert_eq!(
            orb(&config).mapping.local_ba.state_envelope,
            StateEnvelope::On(StateEnvelopeLimits::default())
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let config = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((mapping: (local_ba: (lost_keyframes: FixedInertial)))))",
        )
        .unwrap();
        assert_eq!(
            orb(&config).mapping.local_ba.lost_keyframes,
            LostKeyframes::FixedInertial
        );
        assert_eq!(
            orb(&parse("(version: 1)").unwrap())
                .mapping
                .local_ba
                .lost_keyframes,
            LostKeyframes::Free
        );

        let config =
            parse("(version: 1, pipeline: OrbSlam((mapping: (local_ba: (visual_kernel: Huber)))))")
                .unwrap();
        assert_eq!(
            orb(&config).mapping.local_ba.visual_kernel,
            VisualKernel::Huber
        );
        assert_eq!(
            orb(&parse("(version: 1)").unwrap())
                .mapping
                .local_ba
                .visual_kernel,
            VisualKernel::L2
        );
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
                "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (window: 3)))))",
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
    fn loop_graph_defaults_to_the_snapshot_and_drift_is_opt_in() {
        let base = Path::new("/configs");
        let text = |graph: &str| {
            format!(
                "(version: 1, sensors: (cameras: Stereo), pipeline: OrbSlam((loop_closing: Enabled(vocabulary: \"v.txt\"{graph}))))"
            )
        };
        let default = SystemConfig::from_ron_str(&text(""), base).unwrap();
        assert_eq!(
            orb(&default).loop_closing.graph(),
            Some(LoopGraphSettings {
                information: LoopGraphInformation::ScalarWeights,
                persistent: false,
            })
        );
        let drift = SystemConfig::from_ron_str(
            &text(", graph: (information: Drift(loop_cauchy_scale: 2.0), persistent: true)"),
            base,
        )
        .unwrap();
        let LoopGraphInformation::Drift {
            odometry,
            loop_sigma_m,
            loop_cauchy_scale,
            ..
        } = orb(&drift).loop_closing.graph().unwrap().information
        else {
            panic!("Drift expected");
        };
        assert_eq!(odometry, OdometryRandomWalk::default());
        assert_eq!(loop_sigma_m, 0.10);
        assert_eq!(loop_cauchy_scale, 2.0);
        assert!(orb(&drift).loop_closing.graph().unwrap().persistent);
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
    fn loop_closing_execution_is_optional_and_parses() {
        let text = |execution: &str| {
            format!(
                "(version: 1, sensors: (cameras: Stereo), pipeline: OrbSlam((loop_closing: Enabled(vocabulary: \"v.txt\"{execution}))))"
            )
        };
        let execution = |config: &SystemConfig| match &orb(config).loop_closing {
            LoopClosingMode::Enabled { execution, .. } => *execution,
            LoopClosingMode::Disabled => panic!("loop closing is enabled"),
        };
        assert_eq!(
            execution(&parse(&text("")).unwrap()),
            LoopClosingExecution::Synchronous
        );
        assert_eq!(
            execution(&parse(&text(", execution: FollowMapping")).unwrap()),
            LoopClosingExecution::FollowMapping
        );
        assert_eq!(
            execution(&parse(&text(", execution: Deferred(frames: 4)")).unwrap()),
            LoopClosingExecution::Deferred { frames: 4 }
        );
        assert_eq!(
            execution(&parse(&text(", execution: Asynchronous")).unwrap()),
            LoopClosingExecution::Asynchronous
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
                execution: LoopClosingExecution::default(),
                graph: Default::default(),
            },
            sensors(CameraSelection::Stereo, true),
        );
        let settings = orb_mut(&mut config);
        settings.frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints: 2000,
            stereo_close_depth: StereoCloseDepth::Metres(4.0),
            klt_first: KltFirst::Enabled { min_tracks: 120 },
            klt_prediction: KltPrediction::Disabled,
            secondary_n_keypoints: Some(500),
            klt_kernel: LkKernel::F32,
            ..OrbFrontendConfig::default()
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
                klt_first: KltFirst::Enabled { min_tracks: 120 },
                klt_prediction: KltPrediction::Disabled,
                secondary_n_keypoints: Some(500),
                klt_kernel: LkKernel::F32,
                ..OrbFrontendConfig::default()
            })
        );
    }

    #[test]
    fn tracking_loss_selects_anchors_and_defaults_to_lost_keyframes() {
        use crate::tracking::KeyframesWhileLost;
        assert_eq!(
            orb(&parse("(version: 1)").unwrap())
                .tracking_loss
                .keyframes_while_lost,
            KeyframesWhileLost::Enabled
        );
        let config = parse(
            "(version: 1, pipeline: OrbSlam((tracking_loss: (keyframes_while_lost: Anchor))))",
        )
        .unwrap();
        assert_eq!(
            orb(&config).tracking_loss.keyframes_while_lost,
            KeyframesWhileLost::Anchor
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let config = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((tracking_loss: (keyframes_while_lost: Anchor, reseed: (keep_inertial: true, min_seed_points: 10)))))",
        )
        .unwrap();
        let reseed = orb(&config).tracking_loss.reseed;
        assert!(reseed.keep_inertial);
        assert_eq!(reseed.min_seed_points, 10);
        assert_eq!(
            reseed.max_coast_speed_mps,
            crate::system::RecoveryConfig::default().max_coast_speed_mps
        );
        assert!(
            !orb(&parse("(version: 1)").unwrap())
                .tracking_loss
                .reseed
                .keep_inertial
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));
    }

    #[test]
    fn the_pose_gate_defaults_off_round_trips_and_takes_default_limits() {
        use crate::tracking::pose_gate::{PoseGate, PoseGateLimits};
        let old = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((tracking_loss: (reseed: (keep_inertial: true)))))",
        )
        .unwrap();
        assert_eq!(orb(&old).tracking_loss.pose_gate, PoseGate::Disabled);
        assert!(!config_display_mentions_gate(&old));

        let config = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((tracking_loss: (pose_gate: Enabled((max_rotation_gap_deg: 6.5))))))",
        )
        .unwrap();
        let PoseGate::Enabled(limits) = orb(&config).tracking_loss.pose_gate else {
            panic!("the gate is enabled");
        };
        assert_eq!(limits.max_rotation_gap_deg, 6.5);
        assert_eq!(
            limits.max_translation_gap_mps,
            PoseGateLimits::default().max_translation_gap_mps
        );
        assert_eq!(ron(&config), ron(&parse(&ron(&config)).unwrap()));

        let config =
            parse("(version: 1, sensors: (imu: true), pipeline: OrbSlam((tracking_loss: (pose_gate: Enabled(())))))")
                .unwrap();
        assert_eq!(
            orb(&config).tracking_loss.pose_gate,
            PoseGate::Enabled(PoseGateLimits::default())
        );
    }

    fn config_display_mentions_gate(config: &SystemConfig) -> bool {
        config.to_string().contains("pose gate")
    }

    #[test]
    fn the_reseed_defaults_are_todays_behaviour_and_round_trip() {
        let defaults = crate::system::RecoveryConfig::default();
        assert!(!defaults.keep_inertial);
        assert_eq!(defaults.min_seed_points, 50);
        assert_eq!(defaults.max_tilt_deg, 10.0);
        assert_eq!(defaults.max_coast_speed_mps, 4.0);
        assert_eq!(defaults.runaway_window_sec, 10.0);
        assert_eq!(defaults.max_bridge_sec, 10.0);
        assert_eq!(defaults.bridge_covariance_scale, 100.0);

        let config = parse(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((tracking_loss: (reseed: (keep_inertial: true, min_seed_points: 10, max_tilt_deg: 7.5)))))",
        )
        .unwrap();
        let reseed = orb(&config).tracking_loss.reseed;
        assert_eq!(reseed.max_tilt_deg, 7.5);
        assert_eq!(
            orb(&parse(&ron(&config)).unwrap()).tracking_loss.reseed,
            reseed
        );
    }

    /// Card E's Orin 2-core preset, as `configs/lamaria-rig-imu-orin2.ron` sets it.
    const ORIN2_PRESET: &str = r#"
        (
            version: 1,
            sensors: ( cameras: Rig, imu: true ),
            pipeline: OrbSlam((
                frontend: Orb((
                    klt_kernel: WindowedU8,
                    flow_threads: 2,
                    klt_rig: AllCameras,
                    klt_window: 11,
                    klt_prediction: Enabled(max_level: 0, max_iter: 8),
                    klt_retry: CoarseLevel(max_level: 1),
                    forward_backward: Enabled(max_round_trip_px: 0.6, max_iter: 4, every_n_frames: 1),
                    track_refill: Enabled(per_camera: 48, min_spacing_px: 23.0, refill_below: 0.8, border_px: 9),
                    adopt_landmarks: true,
                    gyro_guess: RotatedRay,
                    flow_pose: AnalyticGn(max_iter: 3),
                    flow_bookkeeping: KeyframesOnly,
                    keyframe_orb: Mapping,
                )),
                mapping: ( execution: Asynchronous ),
            )),
        )
    "#;

    #[test]
    fn the_orin_presets_round_trip() {
        let parsed = parse(ORIN2_PRESET).unwrap();
        assert_eq!(orb(&parsed).frontend, orb(&orin_preset(2, 48)).frontend);
        for preset in [orin_preset(2, 48), orin_preset(3, 64)] {
            let text = ron(&preset);
            let reparsed = parse(&text).unwrap();
            assert_eq!(ron(&reparsed), text);
            assert_eq!(orb(&reparsed).frontend, orb(&preset).frontend);
        }
        assert_eq!(
            orb(&parse("(version: 1)").unwrap()).frontend,
            FrontendConfig::default()
        );
    }

    #[test]
    fn each_online_calibration_state_parses() {
        use crate::mapping::global_vi_ba::{
            ClockOffsetPrior, FocalSlicePrior, GyroIntrinsicsPrior, RotationSlicePrior,
        };
        let online = |states: &str| {
            parse(&format!(
                "(version: 1, sensors: (imu: true), pipeline: OrbSlam((mapping: (local_ba: (online_calibration: ({states}))))))"
            ))
            .map(|config| orb(&config).mapping.local_ba.online_calibration)
        };
        assert_eq!(online("").unwrap(), OnlineCalibrationConfig::default());
        assert_eq!(
            online("clock_offset: Some((sigma_ms: 2.0))")
                .unwrap()
                .clock_offset,
            Some(ClockOffsetPrior { sigma_ms: 2.0 })
        );
        assert_eq!(
            online("camera_rotation: Some((step_sigma_deg: 0.05, value_sigma_deg: 1.0))")
                .unwrap()
                .camera_rotation,
            Some(RotationSlicePrior {
                step_sigma_deg: 0.05,
                value_sigma_deg: 1.0
            })
        );
        assert_eq!(
            online("camera_focal: Some((step_sigma: 5e-4, value_sigma: 0.01))")
                .unwrap()
                .camera_focal,
            Some(FocalSlicePrior {
                step_sigma: 5e-4,
                value_sigma: 0.01
            })
        );
        assert_eq!(
            online("gyro_intrinsics: Some((sigma: 0.005))")
                .unwrap()
                .gyro_intrinsics,
            Some(GyroIntrinsicsPrior { sigma: 0.005 })
        );
        assert!(rejects_field(
            "(version: 1, sensors: (imu: true), pipeline: OrbSlam((mapping: (local_ba: (online_calibration: (time_slices: 8))))))",
            "time_slices"
        ));
        assert!(matches!(
            online("clock_offset: Some((sigma_ms: 0.0))"),
            Err(LoadError::Invalid(ConfigError::InvalidSetting { .. }))
        ));
        assert!(matches!(
            parse(
                "(version: 1, pipeline: OrbSlam((mapping: (local_ba: (online_calibration: (clock_offset: Some((sigma_ms: 2.0))))))))"
            ),
            Err(LoadError::Invalid(ConfigError::InertialKnobWithoutImu(_)))
        ));
    }
}

/// G12: lost keyframes need a confident IMU, the pose gate an inertial
/// prediction, and a kept inertial state an IMU; each is inert without one.
#[test]
fn knobs_that_read_the_imu_are_refused_without_it() {
    use crate::mapping::bundle_adjustment::LostKeyframes;
    use crate::tracking::pose_gate::{PoseGate, PoseGateLimits};
    let with = |imu: bool, set: &dyn Fn(&mut OrbSlamPipeline)| {
        let mut config = SystemConfig {
            sensors: sensors(CameraSelection::Stereo, imu),
            ..SystemConfig::default()
        };
        set(orb_mut(&mut config));
        config.validate()
    };
    type SetKnob<'a> = &'a dyn Fn(&mut OrbSlamPipeline);
    let knobs: [(&str, SetKnob); 3] = [
        ("mapping.local_ba.lost_keyframes", &|orb| {
            orb.mapping.local_ba.lost_keyframes = LostKeyframes::FixedInertial
        }),
        ("tracking_loss.pose_gate", &|orb| {
            orb.tracking_loss.pose_gate = PoseGate::Enabled(PoseGateLimits::default())
        }),
        ("tracking_loss.reseed.keep_inertial", &|orb| {
            orb.tracking_loss.reseed.keep_inertial = true
        }),
    ];
    for (knob, set) in knobs {
        assert_eq!(with(true, set), Ok(()), "{knob}");
        assert_eq!(
            with(false, set),
            Err(ConfigError::InertialKnobWithoutImu(knob)),
            "{knob}"
        );
    }
    assert_eq!(with(false, &|_| {}), Ok(()), "the defaults need no IMU");
}
