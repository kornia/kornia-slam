use kornia_3d::camera::PinholeCamera;
use kornia_slam::{LoopClosingConfig, SlamConfig, SlamSystem};

#[test]
fn slam_system_is_constructible_from_the_public_api() {
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
        ..SlamConfig::default()
    };

    let _system = SlamSystem::new(camera, config);
}
