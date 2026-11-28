use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use kornia_sensors::{RigCamera, SensorRig};

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
        graph: LoopGraphSettings::default(),
        execution: LoopClosingExecution::default(),
    }
}

fn sensors(cameras: CameraSelection, imu: bool) -> SensorSelection {
    SensorSelection { cameras, imu }
}

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 458.0,
        fy: 457.0,
        cx: 367.0,
        cy: 248.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

const SIZE: ImageSize = ImageSize {
    width: 752,
    height: 480,
};

fn rig() -> SensorRig {
    SensorRig::mono(camera(), SIZE)
}

fn stereo_rig() -> SensorRig {
    SensorRig::rectified_stereo(camera(), SIZE, 0.11)
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
  loop closing: ORBvoc.txt (pose graph: w*I)
",
        ),
        (
            parallax,
            "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  tracking
  keyframes: every 3..=unbounded frames, ref ratio 0.6, parallax 10 px
  local mapping: asynchronous
",
        ),
    ] {
        assert_eq!(config.to_string(), expected);
    }
}

/// Every window-only flow knob on, at the values the flow tests use.
fn windowed_frontend() -> OrbFrontendConfig {
    OrbFrontendConfig {
        klt_kernel: LkKernel::WindowedU8,
        flow_threads: 3,
        klt_rig: KltRig::AllCameras,
        klt_window: 11,
        klt_retry: KltRetry::CoarseLevel { max_level: 1 },
        forward_backward: ForwardBackward::Enabled {
            max_round_trip_px: 0.6,
            max_iter: 4,
            every_n_frames: 1,
        },
        track_refill: TrackRefill::Enabled {
            per_camera: 64,
            min_spacing_px: 23.0,
            refill_below: 0.8,
            border_px: 9,
        },
        adopt_landmarks: true,
        gyro_guess: GyroGuess::RotatedRay,
        flow_pose: FlowPose::AnalyticGn { max_iter: 3 },
        ..OrbFrontendConfig::default()
    }
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
fn the_window_only_flow_knobs_default_off() {
    let frontend = OrbFrontendConfig::default();
    assert_eq!(frontend.flow_threads, 1);
    assert_eq!(frontend.klt_rig, KltRig::PrimaryOnly);
    assert_eq!(frontend.klt_window, 21);
    assert_eq!(frontend.klt_retry, KltRetry::Off);
    assert_eq!(frontend.forward_backward, ForwardBackward::Off);
    assert_eq!(frontend.track_refill, TrackRefill::Off);
    assert!(!frontend.adopt_landmarks);
    assert_eq!(frontend.gyro_guess, GyroGuess::Off);
    assert_eq!(frontend.flow_pose, FlowPose::RansacPnp);
    for (knob, default) in [
        (KltRig::default() == KltRig::PrimaryOnly, "klt_rig"),
        (KltRetry::default() == KltRetry::Off, "klt_retry"),
        (
            ForwardBackward::default() == ForwardBackward::Off,
            "forward_backward",
        ),
        (TrackRefill::default() == TrackRefill::Off, "track_refill"),
        (GyroGuess::default() == GyroGuess::Off, "gyro_guess"),
        (FlowPose::default() == FlowPose::RansacPnp, "flow_pose"),
    ] {
        assert!(knob, "{default}");
    }
    let preset = with_frontend(windowed_frontend(), CameraSelection::Rig);
    assert_eq!(preset.validate(), Ok(()));
    let header = preset.to_string();
    for line in [
        "  klt kernel: window-only fixed-point u8",
        "  flow threads: 3",
        "  klt rig: AllCameras",
        "  klt window: 11",
        "  flow adopts new landmarks",
        "  flow pose: AnalyticGn { max_iter: 3 }",
    ] {
        assert!(header.contains(line), "{line} in {header}");
    }
}

#[test]
fn flow_knob_ranges_are_checked() {
    let windowed = OrbFrontendConfig {
        klt_kernel: LkKernel::WindowedU8,
        ..OrbFrontendConfig::default()
    };
    let refill = |per_camera, min_spacing_px, refill_below| TrackRefill::Enabled {
        per_camera,
        min_spacing_px,
        refill_below,
        border_px: 9,
    };
    let round_trip = |max_round_trip_px, max_iter, every_n_frames| ForwardBackward::Enabled {
        max_round_trip_px,
        max_iter,
        every_n_frames,
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
                forward_backward: round_trip(0.0, 4, 1),
                ..windowed
            },
            "frontend.forward_backward.max_round_trip_px",
        ),
        (
            OrbFrontendConfig {
                forward_backward: round_trip(f32::NAN, 4, 1),
                ..windowed
            },
            "frontend.forward_backward.max_round_trip_px",
        ),
        (
            OrbFrontendConfig {
                forward_backward: round_trip(0.6, 0, 1),
                ..windowed
            },
            "frontend.forward_backward.max_iter",
        ),
        (
            OrbFrontendConfig {
                forward_backward: round_trip(0.6, 4, 0),
                ..windowed
            },
            "frontend.forward_backward.every_n_frames",
        ),
        (
            OrbFrontendConfig {
                track_refill: refill(0, 23.0, 0.8),
                ..windowed
            },
            "frontend.track_refill.per_camera",
        ),
        (
            OrbFrontendConfig {
                track_refill: refill(48, 0.0, 0.8),
                ..windowed
            },
            "frontend.track_refill.min_spacing_px",
        ),
        (
            OrbFrontendConfig {
                track_refill: refill(48, 23.0, 0.0),
                ..windowed
            },
            "frontend.track_refill.refill_below",
        ),
        (
            OrbFrontendConfig {
                track_refill: refill(48, 23.0, 1.5),
                ..windowed
            },
            "frontend.track_refill.refill_below",
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

/// The fixed-point pyramid pads every level by the window, so an odd window
/// past the bound must be refused before a system allocates by it; `2^53 + 1`
/// is odd and exact in f64.
#[test]
fn klt_window_is_bounded() {
    for klt_kernel in [LkKernel::FixedPointU8, LkKernel::WindowedU8] {
        let frontend = |klt_window| OrbFrontendConfig {
            klt_kernel,
            klt_window,
            ..OrbFrontendConfig::default()
        };
        let config = with_frontend(frontend(MAX_KLT_WINDOW), CameraSelection::Mono);
        assert_eq!(config.validate(), Ok(()));
        for klt_window in [MAX_KLT_WINDOW + 2, (1 << 53) + 1] {
            let config = with_frontend(frontend(klt_window), CameraSelection::Mono);
            assert_eq!(
                invalid_setting(&config).as_deref(),
                Some("frontend.klt_window")
            );
        }
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
    // Off a rig the knob would track the primary alone, silently.
    let all_cameras = OrbFrontendConfig {
        klt_rig: KltRig::AllCameras,
        ..OrbFrontendConfig::default()
    };
    for cameras in [CameraSelection::Mono, CameraSelection::Stereo] {
        assert_eq!(
            with_frontend(all_cameras, cameras).validate(),
            needs("klt_rig", "sensors.cameras Rig")
        );
    }
    assert_eq!(on_rig(all_cameras), Ok(()));
    // The full-pyramid kernels read the gyro on the predicted search only.
    let gyro = |klt_kernel, klt_prediction| OrbFrontendConfig {
        klt_kernel,
        klt_prediction,
        gyro_guess: GyroGuess::RotatedRay,
        ..OrbFrontendConfig::default()
    };
    let predicted = KltPrediction::Enabled {
        max_level: 1,
        max_iter: 10,
    };
    for kernel in [LkKernel::F32, LkKernel::FixedPointU8] {
        assert_eq!(
            on_rig(gyro(kernel, KltPrediction::Disabled)),
            needs("gyro_guess", "klt_prediction or klt_kernel WindowedU8")
        );
        assert_eq!(on_rig(gyro(kernel, predicted)), Ok(()));
    }
    assert_eq!(
        on_rig(gyro(LkKernel::WindowedU8, KltPrediction::Disabled)),
        Ok(())
    );
    assert_eq!(
        on_rig(OrbFrontendConfig {
            klt_window: 11,
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
            klt_kernel: LkKernel::FixedPointU8,
            ..OrbFrontendConfig::default()
        }),
        Ok(())
    );
    let fixed_point = OrbFrontendConfig {
        klt_kernel: LkKernel::FixedPointU8,
        ..OrbFrontendConfig::default()
    };
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
                forward_backward: windowed_frontend().forward_backward,
                ..fixed_point
            },
            "forward_backward",
        ),
        (
            OrbFrontendConfig {
                track_refill: windowed_frontend().track_refill,
                ..fixed_point
            },
            "track_refill",
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
}

#[test]
fn flow_threads_need_a_shard_layout_for_the_rig() {
    for (threads, cameras, ok) in [
        (1, 1, true),
        (1, 2, true),
        (2, 2, true),
        (3, 2, true),
        (4, 2, false),
        (2, 1, true),
        (3, 1, false),
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

/// Every optical-flow knob defaults to the behaviour before it existed, and
/// the default configuration prints the same header as before.
#[test]
fn frontend_defaults_reproduce_main() {
    let frontend = OrbFrontendConfig::default();
    assert_eq!(frontend.klt_kernel, LkKernel::default());
    assert_eq!(frontend.klt_kernel, LkKernel::F32);
    assert_eq!(frontend.klt_prediction, KltPrediction::Disabled);
    assert_eq!(frontend.klt_first, KltFirst::Disabled);
    assert_eq!(frontend.klt_first, KltFirst::default());
    assert_eq!(
        SystemConfig::default().to_string(),
        "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: asynchronous
"
    );

    let mut flow = SystemConfig::default();
    orb_mut(&mut flow).frontend = FrontendConfig::Orb(OrbFrontendConfig {
        klt_first: KltFirst::Enabled { min_tracks: 80 },
        klt_kernel: LkKernel::FixedPointU8,
        klt_prediction: KltPrediction::Enabled {
            max_level: 1,
            max_iter: 10,
        },
        ..OrbFrontendConfig::default()
    });
    assert_eq!(
        flow.to_string(),
        "OrbSlam pipeline (config version 1)
  frontend: ORB, 1000 keypoints
  klt kernel: fixed-point u8
  klt-first: ORB when a keyframe is due, the flow pose is refused or fewer than 80 tracks survive
  klt prediction: levels 0..=1, 10 iterations when trusted
  tracking
  keyframes: every 3..=8 frames, ref ratio 0.6
  local mapping: asynchronous
"
    );
    flow.sensors = sensors(CameraSelection::Rig, false);
    assert!(
        flow.to_string()
            .contains("  klt-first: inactive on a rig, ORB on every frame\n"),
        "{flow}"
    );
    assert!(flow.validate().is_ok());
    let FrontendConfig::Orb(frontend) = &mut orb_mut(&mut flow).frontend;
    frontend.klt_prediction = KltPrediction::Enabled {
        max_level: 1,
        max_iter: 0,
    };
    assert_eq!(
        invalid_setting(&flow).as_deref(),
        Some("frontend.klt_prediction.max_iter")
    );
    let FrontendConfig::Orb(frontend) = &mut orb_mut(&mut flow).frontend;
    frontend.klt_prediction = KltPrediction::Disabled;
    frontend.klt_first = KltFirst::Enabled { min_tracks: 0 };
    assert_eq!(
        invalid_setting(&flow).as_deref(),
        Some("frontend.klt_first.min_tracks")
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
fn loop_graph_information_is_range_checked() {
    use crate::loop_closure::OdometryRandomWalk;
    let with = |information| {
        with_loop_closing(
            LoopClosingMode::Enabled {
                vocabulary: PathBuf::from("ORBvoc.txt"),
                graph: LoopGraphSettings { information },
                execution: LoopClosingExecution::default(),
            },
            sensors(CameraSelection::Stereo, false),
        )
    };
    let drift = |edit: fn(&mut OdometryRandomWalk, &mut [f64; 3])| {
        let mut odometry = OdometryRandomWalk {
            translation_m2_per_m: 0.0145,
            rotation_rad2_per_m: 2.7e-6,
            floor_m: 0.001,
            floor_rad: 1e-4,
        };
        let mut loop_values = [0.10, 1.0_f64.to_radians(), 3.0];
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
    // Positive in f64, but zero or infinite where the solver reads them as f32.
    assert_eq!(
        drift(|_, values| values[2] = 1e-50),
        at("loop_cauchy_scale")
    );
    assert_eq!(drift(|_, values| values[0] = 1e-40), at("loop_sigma_m"));
    assert_eq!(
        drift(|walk, _| walk.floor_rad = 1e50),
        at("odometry.floor_rad")
    );
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
        AsynchronousReplay { frames: 2 }.resolve(LocalMappingMode::Asynchronous),
        AsynchronousReplay { frames: 2 }
    );
    assert_eq!(
        Synchronous.resolve(LocalMappingMode::Asynchronous),
        Synchronous
    );
}

#[test]
fn vocabulary_path_must_not_be_empty() {
    let config = with_loop_closing(
        LoopClosingMode::Enabled {
            vocabulary: PathBuf::new(),
            graph: LoopGraphSettings::default(),
            execution: LoopClosingExecution::default(),
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
    let stereo_imu = stereo_rig().with_imu(Pose3d::IDENTITY);

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

mod runtime_settings {
    use super::*;

    fn stereo_imu_rig() -> SensorRig {
        stereo_rig().with_imu(Pose3d::IDENTITY)
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
        let stereo = sensors(CameraSelection::Stereo, false);
        let selected = stereo.select_rig(stereo_imu_rig()).unwrap();
        let close_depth = |depth| {
            let mut config = SystemConfig {
                sensors: stereo,
                ..SystemConfig::default()
            };
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

    /// The selection, not the rig, decides the layout: a rectified pair run as
    /// mono has no baseline and no close depth.
    #[test]
    fn a_stereo_rig_run_as_mono_has_no_baseline() {
        let settings = SystemConfig::default().settings(&stereo_imu_rig());
        assert_eq!(settings.stereo_baseline_m, None);
        assert_eq!(settings.stereo_close_depth_m, None);
    }

    fn raw_fisheye_rig() -> SensorRig {
        let fisheye = kornia_3d::camera::Fisheye624Camera {
            f: 240.0,
            cx: 320.0,
            cy: 240.0,
            k: [-0.025, 0.096, -0.063, 0.0062, 0.0035, -0.00073],
            p: [0.0009, 0.00036],
            s: [-0.0014, -0.0001, -0.00046, -0.0001],
        };
        SensorRig::from_cameras(vec![
            kornia_sensors::RigCamera::primary(fisheye, SIZE),
            kornia_sensors::RigCamera::new(
                fisheye,
                SIZE,
                Pose3d::new(
                    kornia_algebra::Mat3F64::IDENTITY,
                    kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
                ),
            ),
        ])
        .unwrap()
    }

    /// The flow tracker reads raw pixels as pinhole pixels only when it
    /// flows the rig cameras; with the primary alone `klt_first` stays inert
    /// on a raw primary, so nothing needs refusing.
    #[test]
    fn klt_first_on_a_raw_rig_is_refused_only_with_rig_flow() {
        let rig_config = |klt_first, klt_rig| {
            let mut config = SystemConfig {
                sensors: sensors(CameraSelection::Rig, false),
                ..SystemConfig::default()
            };
            orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
                klt_first,
                klt_rig,
                ..OrbFrontendConfig::default()
            });
            config
        };
        let enabled = KltFirst::Enabled { min_tracks: 80 };
        let selected = sensors(CameraSelection::Rig, false)
            .select_rig(raw_fisheye_rig())
            .unwrap();
        assert!(
            crate::SlamSystem::build(
                rig_config(KltFirst::Disabled, KltRig::PrimaryOnly),
                raw_fisheye_rig()
            )
            .is_ok()
        );
        assert_eq!(
            rig_config(enabled, KltRig::PrimaryOnly).validate_raw_fisheye(&selected),
            Ok(())
        );
        assert_eq!(
            rig_config(enabled, KltRig::AllCameras).validate_raw_fisheye(&selected),
            Err(ConfigError::FrontendKnobNeeds {
                knob: "klt_first",
                needs: "pinhole rig images",
            })
        );
        assert_eq!(
            rig_config(enabled, KltRig::AllCameras).validate_raw_fisheye(&rig()),
            Ok(())
        );
    }

    #[test]
    fn a_rig_may_hold_raw_fisheye_cameras() {
        let fisheye = kornia_3d::camera::Fisheye624Camera {
            f: 240.0,
            cx: 320.0,
            cy: 240.0,
            k: [-0.025, 0.096, -0.063, 0.0062, 0.0035, -0.00073],
            p: [0.0009, 0.00036],
            s: [-0.0014, -0.0001, -0.00046, -0.0001],
        };
        let rig = SensorRig::from_cameras(vec![
            kornia_sensors::RigCamera::primary(fisheye, SIZE),
            kornia_sensors::RigCamera::new(
                fisheye,
                SIZE,
                Pose3d::new(
                    kornia_algebra::Mat3F64::IDENTITY,
                    kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
                ),
            ),
        ])
        .unwrap();
        assert_eq!(
            sensors(CameraSelection::Rig, false).validate_rig(&rig),
            Ok(())
        );
    }

    /// `klt_rig: AllCameras` lets a rig defer ORB to flow frames; the default
    /// keeps ORB on every rig frame.
    #[test]
    fn klt_first_reaches_a_rig_only_with_every_camera_tracked() {
        let two_cameras = SensorRig::from_cameras(vec![
            kornia_sensors::RigCamera::primary(camera(), SIZE),
            kornia_sensors::RigCamera::new(
                camera(),
                SIZE,
                Pose3d::new(
                    kornia_algebra::Mat3F64::IDENTITY,
                    kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
                ),
            ),
        ])
        .unwrap();
        let selected = sensors(CameraSelection::Rig, false)
            .select_rig(two_cameras)
            .unwrap();
        let klt_min_tracks = |klt_rig| {
            let mut config = SystemConfig {
                sensors: sensors(CameraSelection::Rig, false),
                ..SystemConfig::default()
            };
            orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
                klt_first: KltFirst::Enabled { min_tracks: 80 },
                klt_rig,
                ..OrbFrontendConfig::default()
            });
            config.settings(&selected).klt_min_tracks
        };
        assert_eq!(klt_min_tracks(KltRig::PrimaryOnly), None);
        assert_eq!(klt_min_tracks(KltRig::AllCameras), Some(80));
    }

    #[test]
    fn only_the_rig_selection_keeps_the_rig_cameras() {
        let two_cameras = SensorRig::from_cameras(vec![
            kornia_sensors::RigCamera::primary(camera(), SIZE),
            kornia_sensors::RigCamera::new(
                camera(),
                SIZE,
                Pose3d::new(
                    crate::rig::tests::yaw(30.0),
                    kornia_algebra::Vec3F64::new(0.1, 0.0, 0.0),
                ),
            ),
        ])
        .unwrap();
        let selected = sensors(CameraSelection::Rig, false)
            .select_rig(two_cameras.clone())
            .unwrap();
        assert_eq!(selected.cameras().len(), 2);
        let config = SystemConfig {
            sensors: sensors(CameraSelection::Rig, false),
            ..SystemConfig::default()
        };
        let settings = config.settings(&selected);
        assert!(settings.multi_camera);
        assert_eq!(settings.stereo_baseline_m, None);
        let mono = sensors(CameraSelection::Mono, false)
            .select_rig(two_cameras)
            .unwrap();
        assert_eq!(mono.cameras().len(), 1);
        assert!(!SystemConfig::default().settings(&mono).multi_camera);
        assert!(matches!(
            sensors(CameraSelection::Rig, false).select_rig(rig()),
            Err(ConfigError::MissingSensor(_))
        ));
    }

    /// Flow-only tracking and the predicted search need flow pixels in the
    /// frame's pinhole: a raw fisheye primary keeps both off.
    #[test]
    fn a_fisheye_primary_runs_orb_on_every_frame() {
        let mut config = SystemConfig {
            sensors: sensors(CameraSelection::Mono, false),
            ..SystemConfig::default()
        };
        orb_mut(&mut config).frontend = FrontendConfig::Orb(OrbFrontendConfig {
            klt_first: KltFirst::Enabled { min_tracks: 30 },
            klt_prediction: KltPrediction::Enabled {
                max_level: 1,
                max_iter: 10,
            },
            ..OrbFrontendConfig::default()
        });
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let pinhole = SensorRig::from_cameras(vec![RigCamera::primary(camera(), size)]).unwrap();
        let settings = config.settings(&pinhole);
        assert_eq!(settings.klt_min_tracks, Some(30));
        assert_eq!(settings.klt_prediction, Some((1, 10)));
        let fisheye = SensorRig::from_cameras(vec![RigCamera::primary(
            crate::rig::tests::aria_lens(),
            size,
        )])
        .unwrap();
        let settings = config.settings(&fisheye);
        assert_eq!(settings.klt_min_tracks, None);
        assert_eq!(settings.klt_prediction, None);
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

        let config = parse(
            "(version: 1, pipeline: OrbSlam((keyframes: (max_frames_between: 18446744073709551615, min_parallax_px: Some(10.0), max_interval_sec: Some(0.5)))))",
        )
        .unwrap();
        assert_eq!(orb(&config).keyframes.max_frames_between, usize::MAX);
        assert_eq!(orb(&config).keyframes.min_parallax_px, Some(10.0));
        assert_eq!(orb(&config).keyframes.max_interval_sec, Some(0.5));
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

    const FULL_DRIFT: &str = "Drift(odometry: (translation_m2_per_m: 0.0145, rotation_rad2_per_m: 2.7e-6, floor_m: 0.001, floor_rad: 1e-4), loop_sigma_m: 0.1, loop_sigma_rad: 0.0175, loop_cauchy_scale: 2.0)";

    fn with_graph(graph: &str) -> String {
        format!(
            "(version: 1, sensors: (cameras: Stereo), pipeline: OrbSlam((loop_closing: Enabled(vocabulary: \"v.txt\"{graph}))))"
        )
    }

    #[test]
    fn loop_graph_defaults_to_scalar_weights_and_drift_is_opt_in() {
        let base = Path::new("/configs");
        let default = SystemConfig::from_ron_str(&with_graph(""), base).unwrap();
        assert_eq!(
            orb(&default).loop_closing.graph(),
            Some(LoopGraphSettings {
                information: LoopGraphInformation::ScalarWeights,
            })
        );
        let drift = SystemConfig::from_ron_str(
            &with_graph(&format!(", graph: (information: {FULL_DRIFT})")),
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
        assert_eq!(odometry.translation_m2_per_m, 0.0145);
        assert_eq!(loop_sigma_m, 0.1);
        assert_eq!(loop_cauchy_scale, 2.0);
    }

    /// `Drift` has no defaults: every field is written out, and the sigmas are range-checked.
    #[test]
    fn drift_information_requires_every_field() {
        let base = Path::new("/configs");
        let parse = |information: &str| {
            SystemConfig::from_ron_str(
                &with_graph(&format!(", graph: (information: {information})")),
                base,
            )
        };
        match parse("Drift()") {
            Err(LoadError::Parse(error)) => {
                assert!(error.to_string().contains("odometry"), "{error}")
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
        match parse(&FULL_DRIFT.replace(", loop_cauchy_scale: 2.0", "")) {
            Err(LoadError::Parse(error)) => {
                assert!(error.to_string().contains("loop_cauchy_scale"), "{error}")
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
        assert!(parse(FULL_DRIFT).is_ok());
        assert!(matches!(
            parse(&FULL_DRIFT.replace("loop_sigma_m: 0.1", "loop_sigma_m: 0.0")),
            Err(LoadError::Invalid(ConfigError::InvalidSetting { .. }))
        ));
    }

    #[test]
    fn loop_closing_execution_is_optional_and_parses() {
        let base = Path::new("/configs");
        let parse = |execution: &str| SystemConfig::from_ron_str(&with_graph(execution), base);
        let execution = |config: &SystemConfig| match &orb(config).loop_closing {
            LoopClosingMode::Enabled { execution, .. } => *execution,
            LoopClosingMode::Disabled => panic!("loop closing is enabled"),
        };
        assert_eq!(
            execution(&parse("").unwrap()),
            LoopClosingExecution::Synchronous
        );
        assert_eq!(
            execution(&parse(", execution: FollowMapping").unwrap()),
            LoopClosingExecution::FollowMapping
        );
        assert_eq!(
            execution(&parse(", execution: AsynchronousReplay(frames: 4)").unwrap()),
            LoopClosingExecution::AsynchronousReplay { frames: 4 }
        );
        assert_eq!(
            execution(&parse(", execution: Asynchronous").unwrap()),
            LoopClosingExecution::Asynchronous
        );
        assert!(matches!(
            parse(", execution: Deferred(frames: 4)"),
            Err(LoadError::Parse(_))
        ));
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
                graph: LoopGraphSettings::default(),
                execution: LoopClosingExecution::default(),
            },
            sensors(CameraSelection::Stereo, true),
        );
        let settings = orb_mut(&mut config);
        settings.frontend = FrontendConfig::Orb(OrbFrontendConfig {
            n_keypoints: 2000,
            stereo_close_depth: StereoCloseDepth::Metres(4.0),
            klt_first: KltFirst::Enabled { min_tracks: 80 },
            klt_prediction: KltPrediction::Enabled {
                max_level: 1,
                max_iter: 10,
            },
            klt_kernel: LkKernel::WindowedU8,
            // Several flow threads and every camera need a rig; this one is
            // stereo.
            flow_threads: 1,
            klt_rig: KltRig::PrimaryOnly,
            ..windowed_frontend()
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
                klt_first: KltFirst::Enabled { min_tracks: 80 },
                klt_prediction: KltPrediction::Enabled {
                    max_level: 1,
                    max_iter: 10,
                },
                klt_kernel: LkKernel::WindowedU8,
                flow_threads: 1,
                klt_rig: KltRig::PrimaryOnly,
                ..windowed_frontend()
            })
        );
    }
}
