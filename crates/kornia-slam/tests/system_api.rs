use kornia_3d::camera::PinholeCamera;
use kornia_slam::initialization::{
    ImuInitConfig, ImuInitRejectReason, ImuInitResult, ImuInitializer, KeyframeVelocity,
    TwoViewEstimate, TwoViewInitConfig,
};
use kornia_slam::{LoopClosingConfig, SlamConfig, SlamSystem};

fn test_camera() -> PinholeCamera {
    PinholeCamera {
        fx: 400.0,
        fy: 400.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

#[test]
fn slam_system_is_constructible_from_the_public_api() {
    let config = SlamConfig {
        pgo: Some(LoopClosingConfig::default()),
        ..SlamConfig::default()
    };

    let _system = SlamSystem::new(test_camera(), config);
}

#[test]
fn initialization_api_is_exposed_through_the_facade() {
    fn assert_public_type<T>() {}

    assert_public_type::<ImuInitRejectReason>();
    assert_public_type::<ImuInitResult>();
    assert_public_type::<KeyframeVelocity>();
    assert_public_type::<TwoViewEstimate>();

    let _initializer = ImuInitializer::new(ImuInitConfig {
        min_keyframes: 10,
        min_time_sec: 1.0,
        min_motion: 0.05,
        ..ImuInitConfig::default()
    });
    let _two_view_config = TwoViewInitConfig::default();
}

#[test]
#[allow(deprecated)]
fn legacy_pipeline_names_remain_source_compatible() {
    #[allow(deprecated)]
    use kornia_slam::{PgoPipelineConfig, PipelineConfig, SlamPipeline};

    let config = PipelineConfig {
        pgo: Some(PgoPipelineConfig::default()),
        ..PipelineConfig::default()
    };

    let _pipeline = SlamPipeline::new(test_camera(), config);
}
