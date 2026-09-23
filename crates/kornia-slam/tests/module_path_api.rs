//! The pre-existing public module paths must keep resolving. Crate-root imports
//! are covered by `system_api.rs`; these are the paths that module ownership
//! moves would otherwise break.
#![allow(deprecated)]

use kornia_3d::camera::PinholeCamera;

// The runtime, reachable through `pipeline` as well as `system`.
use kornia_slam::pipeline::{
    LoopClosingConfig, LoopClosureEvent, PgoPipelineConfig, PipelineConfig, SlamConfig,
    SlamPipeline, SlamSystem,
};
// State and policy types, reachable through `system` as well as `tracking`.
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingLossRecoveryPolicy, TrackingResult,
    TrackingStatus,
};

#[test]
fn pipeline_module_path_constructs_the_runtime() {
    let camera = PinholeCamera {
        fx: 400.0,
        fy: 400.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    let config = SlamConfig {
        pgo: Some(LoopClosingConfig::default()),
        ..PipelineConfig::default()
    };

    let _system: SlamPipeline = SlamSystem::new(camera, config);
    let _pgo: PgoPipelineConfig = LoopClosingConfig::default();
    let _event: Option<LoopClosureEvent> = None;
}

#[test]
fn system_module_path_resolves_state_and_policy_types() {
    let state = SystemState::new();
    assert_eq!(state.mode, SystemMode::Bootstrap);

    let _policy = KeyframePolicy::default();
    let _recovery = TrackingLossRecoveryPolicy::default();

    let result = TrackingResult {
        pose_world_to_cam: state.pose_world_to_cam,
        status: TrackingStatus::Tracked,
    };
    assert_eq!(result.status, TrackingStatus::Tracked);
}

#[test]
fn tracking_module_path_resolves_the_same_types() {
    let _: kornia_slam::tracking::SystemState = SystemState::new();
    let _: kornia_slam::tracking::KeyframePolicy = KeyframePolicy::default();
}

#[test]
fn estimation_module_path_resolves_pose_estimation_and_flow() {
    // Owned by `tracking`, still reachable where they used to live.
    let _: kornia_slam::estimation::MapProjectionEstimator =
        kornia_slam::tracking::pose_estimation::MapProjectionEstimator::new(Default::default());
    let _cfg = kornia_slam::estimation::map_projection::MapProjectionConfig::default();
    let _pnp = kornia_slam::estimation::pnp::PnpConfig::default();
    let _flow: kornia_slam::estimation::optical_flow::TrackSet =
        kornia_slam::tracking::optical_flow::TrackSet::default();
    let _survivor = kornia_slam::estimation::SurvivorFilterConfig::default();
    let _estimate_is_reachable = |e: kornia_slam::estimation::Estimate| e.inliers;
}

#[test]
fn estimation_module_path_resolves_initialization() {
    // Owned by `initialization`, still reachable where they used to live.
    let _two_view = kornia_slam::estimation::two_view::TwoViewInitConfig::default();
    let _imu: kornia_slam::estimation::ImuInitConfig =
        kornia_slam::estimation::imu_init::ImuInitConfig {
            min_keyframes: 10,
            min_time_sec: 1.0,
            min_motion: 0.1,
        };
    let _factor_mod_is_reachable =
        std::marker::PhantomData::<kornia_slam::estimation::inertial_init_factor::KfConst>;
}

#[test]
fn map_module_path_resolves_under_both_spellings() {
    // Owned by `mapping`, still reachable at its original crate-root path.
    let _map: kornia_slam::map::Map = kornia_slam::mapping::Map::default();
    let _mode = kornia_slam::map::LocalMappingMode::Asynchronous;
    let _health = kornia_slam::initialization::InitialMapHealth::default();
}
