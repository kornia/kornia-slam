//! The `#[non_exhaustive]` factor types a downstream crate builds (cu-kornia-vio
//! does each of these) can be built from outside this crate: a struct literal
//! of one is E0639 here, so every value below goes through a constructor or a
//! public field assignment.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_factors::stationary::{StationaryFactor, StationaryGyro};
use kornia_factors::wheel_odometry::{WheelOdomFactor, WheelOdomMeasurement};
use kornia_sensors::imu::ImuCalib;
use kornia_sensors::wheel::PlanarMotion;

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
    let motion = PlanarMotion {
        delta: [0.1, 0.0, 0.05],
        covariance: Mat3F64::from_diagonal(Vec3F64::new(1e-4, 1e-4, 1e-3)),
    };
    let m = WheelOdomMeasurement::new(motion).with_out_of_plane_sigmas(0.02, 0.03);
    assert_eq!((m.motion, m.z_sigma, m.tilt_sigma), (motion, 0.02, 0.03));
    assert_eq!(
        WheelOdomMeasurement::from(motion),
        WheelOdomMeasurement::new(motion)
    );
    let mut set = WheelOdomMeasurement::new(PlanarMotion::IDENTITY);
    set.z_sigma = 0.02;
    set.tilt_sigma = 0.03;
    set.motion = motion;
    assert_eq!(set, m);

    let f = WheelOdomFactor::new(1, 2, m, Pose3d::IDENTITY, Pose3d::IDENTITY, Some(12.59));
    assert_eq!((f.from_idx, f.to_idx, f.huber_chi2), (1, 2, Some(12.59)));
    assert_eq!(f.measurement, m);
    assert!(f.is_valid());
}
