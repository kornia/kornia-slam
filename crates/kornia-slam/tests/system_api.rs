use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_slam::{LoopClosingConfig, SensorRig, SlamConfig, SlamSystem};

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
    let camera = test_camera();
    let config = SlamConfig {
        pgo: Some(LoopClosingConfig::default()),
        ..SlamConfig::default()
    };

    let _system = SlamSystem::new(camera, config);
}

#[test]
fn system_accepts_an_explicit_sensor_rig() {
    let rig = SensorRig::new(test_camera()).with_imu(Pose3d::IDENTITY);
    assert_eq!(rig.camera_to_body(), Some(Pose3d::IDENTITY));

    let _system = SlamSystem::with_rig(rig, SlamConfig::default());
}
