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
