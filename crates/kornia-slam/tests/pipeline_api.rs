use kornia_3d::camera::PinholeCamera;
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
