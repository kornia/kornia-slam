//! The `#[non_exhaustive]` types a downstream crate builds (cu-kornia-vio
//! does each of these) can be built from outside this crate: a struct
//! literal of one is E0639 here, so every value below goes through a
//! constructor, `Default` or a public field assignment.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuCalib;
use kornia_slam::KeyframePolicy as RootKeyframePolicy;
use kornia_slam::mapping::culling::KeyframeCullingConfig;
use kornia_slam::mapping::map::{StillNeighbour, StillSide};
use kornia_slam::mapping::stationary::{StationaryFactor, StationaryGyro};
use kornia_slam::mapping::wheel_odom::{WheelOdomFactor, WheelOdomMeasurement};
use kornia_slam::tracking::{KeyframeEvidence, KeyframePolicy};

fn calib() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.7e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9e-5,
        accel_bias_noise: 3.0e-3,
    }
}

#[test]
fn stationary_factor_with_every_field_set() {
    let gyro = StationaryGyro::new(Vec3F64::new(1e-3, -2e-3, 5e-4), 2.5);
    assert_eq!((gyro.mean.x, gyro.covered_s), (1e-3, 2.5));

    let mut f = StationaryFactor::new(3, 7, 12.0, calib());
    f.gyro = Some(gyro);
    f.translation_sigma_m = 0.003;
    f.rotation_sigma_rad = 0.004;
    f.velocity_sigma_m_s = 0.02;
    f.huber_chi2 = None;
    assert_eq!((f.from_idx, f.to_idx, f.span_s), (3, 7, 12.0));
    assert_eq!(f.gyro, Some(gyro));
    assert!(f.is_valid());
    assert_eq!(
        StationaryFactor::new(3, 7, 12.0, calib())
            .with_gyro(gyro)
            .gyro,
        Some(gyro)
    );
}

#[test]
fn wheel_measurement_and_factor_with_every_field_set() {
    let cov = [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-3]];
    let m = WheelOdomMeasurement::new([0.1, 0.0, 0.05], cov).with_out_of_plane_sigmas(0.02, 0.03);
    assert_eq!(
        (m.delta, m.covariance, m.z_sigma, m.tilt_sigma),
        ([0.1, 0.0, 0.05], cov, 0.02, 0.03)
    );
    let mut set = WheelOdomMeasurement::new([0.0; 3], cov);
    set.z_sigma = 0.02;
    set.tilt_sigma = 0.03;
    set.delta = m.delta;
    assert_eq!(set, m);

    let f = WheelOdomFactor::new(1, 2, m, Pose3d::IDENTITY, Pose3d::IDENTITY, Some(12.59));
    assert_eq!((f.from_idx, f.to_idx, f.huber_chi2), (1, 2, Some(12.59)));
    assert_eq!(f.measurement, m);
    assert!(f.is_valid());
}

#[test]
fn keyframe_evidence_and_culling_config_from_outside() {
    let e = KeyframeEvidence::new(10, Some(4), 120, 200).with_median_parallax_px(Some(7.5));
    assert_eq!(e.median_parallax_px, Some(7.5));
    let mut set = KeyframeEvidence::new(10, Some(4), 120, 200);
    set.median_parallax_px = Some(7.5);
    assert_eq!(set, e);
    // KeyframePolicy is exhaustive: functional update still works.
    let policy = KeyframePolicy {
        max_frames_between: usize::MAX,
        ..KeyframePolicy::default()
    };
    let _: RootKeyframePolicy = policy;
    let _ = policy.should_insert_with(&e);

    let mut cfg = KeyframeCullingConfig::default();
    cfg.min_keyframes = 4;
    cfg.max_removals = 1;
    cfg.protect_newest = 5;
    cfg.max_merged_imu_span_sec = 11.0;
    cfg.merge_imu_edges = false;
    let copy = cfg;
    assert_eq!(
        (copy.min_keyframes, copy.max_removals, copy.protect_newest),
        (4, 1, 5)
    );
}

#[test]
fn still_neighbour_from_outside() {
    let n = StillNeighbour::new(42, StillSide::After, 3.5);
    assert_eq!(
        (n.keyframe_idx, n.side, n.span_s),
        (42, StillSide::After, 3.5)
    );
    assert_eq!(n, StillNeighbour::new(42, StillSide::After, 3.5));
    assert_ne!(n, StillNeighbour::new(42, StillSide::Before, 3.5));
}
