use kornia_3d::camera::PinholeCamera;
use kornia_slam::{PgoPipelineConfig, PipelineConfig, SlamPipeline};

#[test]
fn slam_pipeline_is_constructible_from_the_public_api() {
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
    let config = PipelineConfig {
        pgo: Some(PgoPipelineConfig::default()),
        ..PipelineConfig::default()
    };

    let _pipeline = SlamPipeline::new(camera, config);
}
