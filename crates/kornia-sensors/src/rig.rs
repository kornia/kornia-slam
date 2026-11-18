//! Fixed calibration of a camera rig with optional stereo, IMU and wheel
//! odometry, independent of any estimated state.

use crate::imu::ImuCalib;
use crate::wheel::{DifferentialDrive, WheelNoise};
use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

/// Camera calibration plus optional stereo, IMU and wheel calibration.
///
/// `camera` is the geometry camera: keypoint coordinates, the stereo baseline
/// and the IMU and wheel extrinsics all refer to it. For rectified or pinhole images it
/// also describes the supplied images. With [`SensorRig::fisheye`] the images
/// are raw fisheye instead, and `camera` is a virtual pinhole that keypoints
/// are mapped into.
/// Estimated bias, gravity and poses are runtime state of the consumer.
#[derive(Debug, Clone)]
pub struct SensorRig {
    pub camera: PinholeCamera,
    /// Metric baseline (metres) of a rectified stereo pair whose left view is
    /// `camera`. `None` for a monocular rig.
    pub stereo_baseline_m: Option<f64>,
    /// `None` selects visual-only operation.
    pub imu: Option<ImuCalibration>,
    /// Kannala-Brandt model of raw fisheye images, in image pixels. `None`
    /// when the images are already in `camera`'s coordinates.
    pub fisheye: Option<FisheyeCamera>,
    /// `None` for a rig without wheel odometry.
    pub wheels: Option<WheelCalibration>,
    /// Every camera of a non-rectified multi-camera rig, the primary first;
    /// empty otherwise. Each one supplies its own images.
    pub cameras: Vec<RigCamera>,
}

/// One camera of a multi-camera rig (ORB-SLAM3's non-rectified stereo).
#[derive(Debug, Clone)]
pub struct RigCamera {
    /// Undistorted pinhole model of this camera's images.
    pub camera: PinholeCamera,
    pub size: ImageSize,
    /// `T_PC`: `X_primary = T_PC * X_camera`, where the primary camera is
    /// [`SensorRig::camera`]. Identity for the primary itself.
    pub primary_from_camera: Pose3d,
}

/// Ray through an undistorted pixel of `camera`, in the primary-camera frame:
/// `(origin, unit direction)`, origin = `primary_from_camera.translation`.
pub fn camera_ray(
    camera: &PinholeCamera,
    primary_from_camera: &Pose3d,
    pixel_undist: [f64; 2],
) -> (Vec3F64, Vec3F64) {
    let [u, v] = pixel_undist;
    let bearing = Vec3F64::new(
        (u - camera.cx) / camera.fx,
        (v - camera.cy) / camera.fy,
        1.0,
    );
    (
        primary_from_camera.translation,
        primary_from_camera.rotation * bearing.normalize(),
    )
}

impl RigCamera {
    /// [`camera_ray`] of this camera.
    pub fn ray(&self, pixel_undist: [f64; 2]) -> (Vec3F64, Vec3F64) {
        camera_ray(&self.camera, &self.primary_from_camera, pixel_undist)
    }
}

impl SensorRig {
    /// A visual-only rig. Add IMU calibration with [`SensorRig::with_imu`].
    pub fn new(camera: PinholeCamera) -> Self {
        Self {
            camera,
            stereo_baseline_m: None,
            imu: None,
            fisheye: None,
            wheels: None,
            cameras: Vec::new(),
        }
    }

    /// Declares a multi-camera rig. `cameras[0]` must be the primary camera,
    /// with an identity `primary_from_camera`.
    pub fn with_rig_cameras(mut self, cameras: Vec<RigCamera>) -> Self {
        self.cameras = cameras;
        self
    }

    /// Whether the rig has more than one non-rectified camera.
    pub fn is_multi_camera(&self) -> bool {
        self.cameras.len() > 1
    }

    /// Declares the images as a rectified stereo pair with the given baseline in metres.
    pub fn with_stereo_baseline(mut self, baseline_m: f64) -> Self {
        self.stereo_baseline_m = Some(baseline_m);
        self
    }

    /// Adds an IMU with the given camera-to-body extrinsic and the default
    /// noise parameters (see [`ImuCalibration::new`]).
    pub fn with_imu(mut self, camera_to_body: Pose3d) -> Self {
        self.imu = Some(ImuCalibration::new(camera_to_body));
        self
    }

    /// Adds wheel odometry with the given base-to-camera extrinsic,
    /// kinematics and the default noise (see [`WheelCalibration::new`]).
    pub fn with_wheels(mut self, base_to_camera: Pose3d, kinematics: DifferentialDrive) -> Self {
        self.wheels = Some(WheelCalibration::new(base_to_camera, kinematics));
        self
    }

    /// Declares the images as raw fisheye: features are extracted on them and
    /// their keypoints mapped through `fisheye` into the virtual pinhole `camera`.
    pub fn with_fisheye(mut self, fisheye: FisheyeCamera) -> Self {
        self.fisheye = Some(fisheye);
        self
    }

    /// Stereo `bf = fx * baseline` (pixel·metres), the constant in
    /// `depth = bf / disparity`. `None` for a monocular rig.
    pub fn stereo_bf(&self) -> Option<f64> {
        self.stereo_baseline_m.map(|b| self.camera.fx * b)
    }

    /// Camera-to-body transform `T_BC`, or `None` for a visual-only rig.
    pub fn camera_to_body(&self) -> Option<Pose3d> {
        self.imu.as_ref().map(|imu| imu.camera_to_body)
    }

    /// `T_BC` of rig camera `camera`: `camera_to_body ∘ primary_from_camera`.
    /// `None` without an IMU or past the last camera.
    pub fn camera_to_body_of(&self, camera: usize) -> Option<Pose3d> {
        let primary_to_body = self.camera_to_body()?;
        if self.cameras.is_empty() {
            return (camera == 0).then_some(primary_to_body);
        }
        let rig_camera = self.cameras.get(camera)?;
        Some(primary_to_body.compose(&rig_camera.primary_from_camera))
    }

    /// Noise parameters to preintegrate with; the defaults for a rig without IMU
    /// calibration.
    pub fn imu_noise(&self) -> ImuCalib {
        self.imu
            .as_ref()
            .map_or_else(default_imu_noise, |imu| imu.noise)
    }
}

/// IMU noise parameters and its rigid relationship to the camera.
#[derive(Debug, Clone, Copy)]
pub struct ImuCalibration {
    pub noise: ImuCalib,
    /// Camera-to-body transform `T_BC`: `X_body = T_BC * X_camera`.
    /// The body frame is the frame of the supplied IMU measurements.
    pub camera_to_body: Pose3d,
}

impl ImuCalibration {
    /// Uses the EuRoC ADIS16448 noise densities with the given extrinsic.
    /// Supply `noise` explicitly when calibration for the actual sensor is
    /// available.
    pub fn new(camera_to_body: Pose3d) -> Self {
        Self {
            noise: default_imu_noise(),
            camera_to_body,
        }
    }
}

/// Wheel-odometry model and the wheeled base's rigid relationship to the camera.
#[derive(Debug, Clone, Copy)]
pub struct WheelCalibration {
    pub kinematics: DifferentialDrive,
    pub noise: WheelNoise,
    /// Base-to-camera transform `T_CB`: `X_camera = T_CB * X_base`. The base
    /// frame is the REP-103 frame of [`crate::wheel`].
    pub base_to_camera: Pose3d,
}

impl WheelCalibration {
    /// Uses [`WheelNoise::default`]; supply `noise` explicitly once it is
    /// measured for the robot.
    pub fn new(base_to_camera: Pose3d, kinematics: DifferentialDrive) -> Self {
        Self {
            kinematics,
            noise: WheelNoise::default(),
            base_to_camera,
        }
    }
}

/// EuRoC MAV ADIS16448 noise densities and random walks (`imu0/sensor.yaml`).
fn default_imu_noise() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.6968e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9393e-5,
        accel_bias_noise: 3.0e-3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_rig_without_imu_is_visual_only() {
        let rig = SensorRig::new(test_camera());
        assert!(rig.imu.is_none());
        assert!(rig.camera_to_body().is_none());
    }

    #[test]
    fn stereo_bf_is_focal_times_baseline() {
        let rig = SensorRig::new(test_camera());
        assert!(rig.stereo_bf().is_none());

        let rig = rig.with_stereo_baseline(0.11);
        assert_eq!(rig.stereo_bf(), Some(rig.camera.fx * 0.11));
    }

    /// Camera 1 looks 70 deg to the side of the primary, 11 cm away.
    fn yawed_rig() -> SensorRig {
        let yaw = kornia_algebra::SO3F64::exp(Vec3F64::new(0.0, 70f64.to_radians(), 0.0)).matrix();
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        SensorRig::new(test_camera()).with_rig_cameras(vec![
            RigCamera {
                camera: test_camera(),
                size,
                primary_from_camera: Pose3d::IDENTITY,
            },
            RigCamera {
                camera: test_camera(),
                size,
                primary_from_camera: Pose3d::new(yaw, Vec3F64::new(0.11, 0.0, 0.0)),
            },
        ])
    }

    #[test]
    fn a_ray_projects_back_to_its_pixel() {
        let rig = yawed_rig();
        for rig_camera in &rig.cameras {
            let camera_from_primary = rig_camera.primary_from_camera.inverse();
            for i in 0..100 {
                let pixel = [(i % 10) as f64 * 64.0 + 3.5, (i / 10) as f64 * 48.0 + 2.5];
                let (origin, direction) = rig_camera.ray(pixel);
                assert!((direction.length() - 1.0).abs() < 1e-12);
                let p_camera = camera_from_primary.transform_point(&(origin + direction * 5.0));
                let back = rig_camera.camera.project_to_pixel(&p_camera, 0.1).unwrap();
                assert!(
                    (back.x - pixel[0]).abs() < 1e-9 && (back.y - pixel[1]).abs() < 1e-9,
                    "{pixel:?} came back as {back:?}"
                );
            }
        }
    }

    #[test]
    fn rays_of_one_camera_share_a_bit_equal_origin() {
        let rig = yawed_rig();
        assert_eq!(rig.cameras[0].ray([10.0, 20.0]).0, Vec3F64::ZERO);
        let (a, _) = rig.cameras[1].ray([10.0, 20.0]);
        let (b, _) = rig.cameras[1].ray([600.0, 400.0]);
        assert_eq!(
            a.to_array().map(f64::to_bits),
            b.to_array().map(f64::to_bits)
        );
    }

    /// G9: camera 1's extrinsic takes a camera-1 point to the body through the primary, checked
    /// on the point rather than against the implementation's own composition.
    #[test]
    fn each_rig_camera_has_its_body_extrinsic() {
        let camera_to_body = Pose3d::new(
            kornia_algebra::SO3F64::exp(Vec3F64::new(0.2, -0.4, 0.1)).matrix(),
            Vec3F64::new(0.0, 0.02, -0.01),
        );
        let rig = yawed_rig().with_imu(camera_to_body);
        assert_eq!(rig.camera_to_body_of(0), Some(camera_to_body));
        let x_camera = Vec3F64::new(0.3, -0.2, 2.5);
        let through_primary = camera_to_body.transform_point(
            &rig.cameras[1]
                .primary_from_camera
                .transform_point(&x_camera),
        );
        let x_body = rig.camera_to_body_of(1).unwrap().transform_point(&x_camera);
        assert!(
            (x_body - through_primary).length() < 1e-12,
            "{x_body:?} vs {through_primary:?}"
        );
        assert!(rig.camera_to_body_of(2).is_none());
        assert!(yawed_rig().camera_to_body_of(0).is_none());
        let mono = SensorRig::new(test_camera()).with_imu(camera_to_body);
        assert_eq!(mono.camera_to_body_of(0), Some(camera_to_body));
        assert!(mono.camera_to_body_of(1).is_none());
    }

    #[test]
    fn default_imu_noise_is_euroc() {
        for rig in [
            SensorRig::new(test_camera()),
            SensorRig::new(test_camera()).with_imu(Pose3d::IDENTITY),
        ] {
            let noise = rig.imu_noise();
            assert_eq!(noise.gyro_noise, 1.6968e-4);
            assert_eq!(noise.accel_noise, 2.0e-3);
            assert_eq!(noise.gyro_bias_noise, 1.9393e-5);
            assert_eq!(noise.accel_bias_noise, 3.0e-3);
        }
    }
}
