use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::{Image, ImageSize};
use kornia_slam::pipeline::{CameraSelection, PipelineConfig, SensorSelection};
use kornia_slam::{SensorFrame, SensorRig, SlamSystem, TrackingStatus};

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
fn slam_system_builds_and_processes_through_the_public_api() {
    let config = PipelineConfig {
        sensors: SensorSelection {
            cameras: CameraSelection::Mono,
            imu: true,
        },
        ..PipelineConfig::default()
    };
    let rig = SensorRig::new(test_camera()).with_imu(Pose3d::IDENTITY);
    let mut system = SlamSystem::build(config, rig).unwrap();
    assert_eq!(system.rig().camera_to_body(), Some(Pose3d::IDENTITY));

    let image = Image::from_size_val(
        ImageSize {
            width: 640,
            height: 480,
        },
        0u8,
    )
    .unwrap();
    let result = system
        .process(SensorFrame {
            idx: 0,
            timestamp_sec: 0.0,
            image: &image,
            right_image: None,
            imu_samples: &[],
        })
        .unwrap();
    assert_eq!(result.status, TrackingStatus::Skipped);
    assert!(system.frontend_observation().keypoints_xy.is_empty());
}
