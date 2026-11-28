//! Fixed calibration of a camera rig with optional IMU and wheel odometry,
//! independent of any estimated state.

use crate::imu::ImuCalib;
use crate::wheel::{DifferentialDrive, WheelNoise};
use kornia_3d::camera::{Camera, CameraModel, PinholeCamera, Ray};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::ImageSize;

/// Largest [`Pose3d::rotation_defect`] a rig transform may have: the rounding
/// of printed calibration digits, not a scale.
pub(crate) const MAX_ROTATION_DEFECT: f64 = 1e-6;

/// One camera of a [`SensorRig`]: the model of its images and its pose in the
/// primary camera frame.
#[derive(Debug, Clone)]
pub struct RigCamera {
    /// Lens model of this camera's images, in image pixels.
    pub model: Camera,
    /// Size of this camera's images, in pixels.
    pub size: ImageSize,
    /// `T_PC`: `X_primary = T_PC * X_camera`. Identity for the primary.
    pub primary_from_camera: Pose3d,
}

impl RigCamera {
    /// The primary camera of a rig: `primary_from_camera` is the identity.
    pub fn primary(model: impl Into<Camera>, size: ImageSize) -> Self {
        Self::new(model, size, Pose3d::IDENTITY)
    }

    pub fn new(model: impl Into<Camera>, size: ImageSize, primary_from_camera: Pose3d) -> Self {
        Self {
            model: model.into(),
            size,
            primary_from_camera,
        }
    }

    /// Pinhole the system measures keypoints in: the model itself for a pinhole;
    /// for a fisheye model its focal lengths and principal point with no distortion.
    pub fn pinhole(&self) -> PinholeCamera {
        let (fx, fy, cx, cy) = match &self.model {
            Camera::Pinhole(camera) => return camera.clone(),
            Camera::Fisheye(camera) => (camera.fx, camera.fy, camera.cx, camera.cy),
            Camera::Fisheye624(camera) => (camera.f, camera.f, camera.cx, camera.cy),
        };
        PinholeCamera {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Whether the images are raw fisheye images.
    pub fn is_fisheye(&self) -> bool {
        !matches!(self.model, Camera::Pinhole(_))
    }

    /// `T_CP`: `X_camera = T_CP * X_primary`.
    pub fn camera_from_primary(&self) -> Pose3d {
        self.primary_from_camera.inverse()
    }

    /// Ray of a raw image pixel in the primary frame. Origin is
    /// `primary_from_camera.translation`, copied. `None` where the model
    /// images no ray (see [`CameraModel::unproject`]).
    pub fn ray(&self, pixel: Vec2F64) -> Option<Ray> {
        let bearing = self.model.unproject(&pixel)?;
        Some(Ray {
            origin: self.primary_from_camera.translation,
            direction: self.primary_from_camera.rotation * bearing,
        })
    }
}

/// A list of cameras that cannot form a [`SensorRig`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RigError {
    #[error("a rig needs at least one camera")]
    NoCameras,
    #[error("cameras[0] is the primary: its primary_from_camera must be the identity")]
    PrimaryNotIdentity,
    #[error("camera {camera}: the image size is zero")]
    EmptyImage { camera: usize },
    #[error("camera {camera}: primary_from_camera has a non-finite entry")]
    NotFinite { camera: usize },
    #[error("camera {camera}: rotation is not orthonormal (defect {defect})")]
    NotRigid { camera: usize, defect: f64 },
    #[error("camera {camera}: the rotation is a reflection")]
    Reflection { camera: usize },
}

/// The cameras of a rig plus optional IMU and wheel calibration.
///
/// `cameras()[0]` is the primary camera: the IMU and wheel extrinsics refer to
/// it. Each camera supplies its own images, in the coordinates of its model.
/// Estimated bias, gravity and poses are runtime state of the consumer.
#[derive(Debug, Clone)]
pub struct SensorRig {
    cameras: Vec<RigCamera>,
    /// `None` selects visual-only operation.
    pub imu: Option<ImuCalibration>,
    /// `None` for a rig without wheel odometry.
    pub wheels: Option<WheelCalibration>,
}

impl SensorRig {
    /// A visual-only one-camera rig. Add IMU calibration with [`SensorRig::with_imu`].
    pub fn mono(model: impl Into<Camera>, size: ImageSize) -> Self {
        Self::with_cameras(vec![RigCamera::primary(model, size)])
    }

    /// A rectified stereo pair: the right camera is `camera` too, `baseline_m`
    /// metres along the left camera's `x` axis.
    pub fn rectified_stereo(camera: PinholeCamera, size: ImageSize, baseline_m: f64) -> Self {
        let right = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(baseline_m, 0.0, 0.0));
        Self::with_cameras(vec![
            RigCamera::primary(camera.clone(), size),
            RigCamera::new(camera, size, right),
        ])
    }

    /// A visual-only rig of `cameras`, the primary first.
    ///
    /// # Errors
    ///
    /// No camera, a primary that is not at the identity, a zero image size, or
    /// a `primary_from_camera` that is not a finite proper rigid motion.
    pub fn from_cameras(cameras: Vec<RigCamera>) -> Result<Self, RigError> {
        let primary = cameras.first().ok_or(RigError::NoCameras)?;
        if primary.primary_from_camera != Pose3d::IDENTITY {
            return Err(RigError::PrimaryNotIdentity);
        }
        for (camera, rig_camera) in cameras.iter().enumerate() {
            if rig_camera.size.width == 0 || rig_camera.size.height == 0 {
                return Err(RigError::EmptyImage { camera });
            }
            check_rigid(&rig_camera.primary_from_camera, camera)?;
        }
        Ok(Self::with_cameras(cameras))
    }

    fn with_cameras(cameras: Vec<RigCamera>) -> Self {
        Self {
            cameras,
            imu: None,
            wheels: None,
        }
    }

    /// Every camera, the primary first; never empty.
    pub fn cameras(&self) -> &[RigCamera] {
        &self.cameras
    }

    pub fn primary(&self) -> &RigCamera {
        &self.cameras[0]
    }

    /// `T_BA`: maps points in camera `a` into camera `b`. `None` past the last camera.
    pub fn b_from_a(&self, a: usize, b: usize) -> Option<Pose3d> {
        let a = self.cameras.get(a)?;
        let b = self.cameras.get(b)?;
        Some(b.camera_from_primary().compose(&a.primary_from_camera))
    }

    /// Metric baseline (metres) when the rig is exactly a rectified stereo
    /// pair: two equal undistorted pinholes, the second translated along `+x`
    /// only. `None` otherwise.
    pub fn rectified_stereo_baseline_m(&self) -> Option<f64> {
        let [left, right] = self.cameras.as_slice() else {
            return None;
        };
        let (Camera::Pinhole(l), Camera::Pinhole(r)) = (&left.model, &right.model) else {
            return None;
        };
        let undistorted = |c: &PinholeCamera| [c.k1, c.k2, c.p1, c.p2] == [0.0; 4];
        let same = [l.fx, l.fy, l.cx, l.cy] == [r.fx, r.fy, r.cx, r.cy];
        let t = right.primary_from_camera.translation;
        let rectified = right.primary_from_camera.rotation == Mat3F64::IDENTITY
            && t.x > 0.0
            && t.y == 0.0
            && t.z == 0.0;
        (undistorted(l) && undistorted(r) && same && rectified).then_some(t.x)
    }

    /// Stereo `bf = fx * baseline` (pixel·metres), the constant in
    /// `depth = bf / disparity`. `None` unless the rig is a rectified pair.
    pub fn stereo_bf(&self) -> Option<f64> {
        self.rectified_stereo_baseline_m()
            .map(|b| self.primary().pinhole().fx * b)
    }

    /// The rig with only its primary camera.
    pub fn truncated_to_primary(mut self) -> Self {
        self.cameras.truncate(1);
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

    /// Primary camera-to-body transform `T_BC`, or `None` for a visual-only rig.
    pub fn camera_to_body(&self) -> Option<Pose3d> {
        self.imu.as_ref().map(|imu| imu.camera_to_body)
    }

    /// `T_BC` of rig camera `camera`: `camera_to_body ∘ primary_from_camera`.
    /// `None` without an IMU or past the last camera.
    pub fn camera_to_body_of(&self, camera: usize) -> Option<Pose3d> {
        let primary_to_body = self.camera_to_body()?;
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

/// Refuses a transform that is not a finite, proper rigid motion; `camera`
/// names it in the error.
fn check_rigid(pose: &Pose3d, camera: usize) -> Result<(), RigError> {
    let entries = pose.rotation.to_cols_array();
    let finite = entries
        .iter()
        .chain(&pose.translation.to_array())
        .all(|v| v.is_finite());
    if !finite {
        return Err(RigError::NotFinite { camera });
    }
    let defect = pose.rotation_defect();
    if defect > MAX_ROTATION_DEFECT {
        return Err(RigError::NotRigid { camera, defect });
    }
    // `RᵀR = I` holds for a reflection too, which would mirror every ray.
    if pose.rotation.determinant() <= 0.0 {
        return Err(RigError::Reflection { camera });
    }
    Ok(())
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
    use kornia_3d::camera::Fisheye624Camera;
    use kornia_algebra::SO3F64;

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

    /// EuRoC `cam0`.
    fn radtan() -> PinholeCamera {
        PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        }
    }

    /// An Aria SLAM camera.
    fn fisheye624() -> Fisheye624Camera {
        Fisheye624Camera {
            f: 241.6,
            cx: 322.9,
            cy: 240.4,
            k: [-0.025, 0.096, -0.063, 0.0062, 0.0035, -0.00073],
            p: [0.0009, 0.00036],
            s: [-0.0014, -0.0001, -0.00046, -0.0001],
        }
    }

    const SIZE: ImageSize = ImageSize {
        width: 640,
        height: 480,
    };

    /// Camera 1 looks 70 deg to the side of the primary, 11 cm away.
    fn yawed(model: impl Into<Camera> + Clone) -> SensorRig {
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 70f64.to_radians(), 0.0)).matrix();
        SensorRig::from_cameras(vec![
            RigCamera::primary(model.clone(), SIZE),
            RigCamera::new(model, SIZE, Pose3d::new(yaw, Vec3F64::new(0.11, 0.0, 0.0))),
        ])
        .unwrap()
    }

    #[test]
    fn mono_is_a_one_camera_rig() {
        let rig = SensorRig::mono(test_camera(), SIZE);
        assert_eq!(rig.cameras().len(), 1);
        assert_eq!(rig.primary().primary_from_camera, Pose3d::IDENTITY);
        assert_eq!(rig.primary().size, SIZE);
        let fields = |c: PinholeCamera| [c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2];
        assert_eq!(fields(rig.primary().pinhole()), fields(test_camera()));
        assert!(!rig.primary().is_fisheye());
        assert!(rig.rectified_stereo_baseline_m().is_none());
        assert!(rig.imu.is_none());
        assert!(rig.camera_to_body().is_none());
    }

    #[test]
    fn a_rig_needs_a_primary_camera_at_the_identity() {
        assert_eq!(
            SensorRig::from_cameras(Vec::new()).map(|_| ()),
            Err(RigError::NoCameras)
        );
        let moved = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.01, 0.0, 0.0));
        assert_eq!(
            SensorRig::from_cameras(vec![RigCamera::new(test_camera(), SIZE, moved)]).map(|_| ()),
            Err(RigError::PrimaryNotIdentity)
        );
        let empty = ImageSize {
            width: 0,
            height: 480,
        };
        assert_eq!(
            SensorRig::from_cameras(vec![RigCamera::primary(test_camera(), empty)]).map(|_| ()),
            Err(RigError::EmptyImage { camera: 0 })
        );
    }

    #[test]
    fn transforms_that_are_not_rigid_are_refused() {
        let with_second = |primary_from_camera: Pose3d| {
            SensorRig::from_cameras(vec![
                RigCamera::primary(test_camera(), SIZE),
                RigCamera::new(test_camera(), SIZE, primary_from_camera),
            ])
            .map(|_| ())
        };
        let t = Vec3F64::new(0.11, 0.0, 0.0);
        assert!(matches!(
            with_second(Pose3d::new(Mat3F64::IDENTITY * (1.0 + 5e-4), t)),
            Err(RigError::NotRigid { camera: 1, .. })
        ));
        let mirror = Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, -1.0));
        assert_eq!(
            with_second(Pose3d::new(mirror, t)),
            Err(RigError::Reflection { camera: 1 })
        );
        assert_eq!(
            with_second(Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(f64::NAN, 0.0, 0.0)
            )),
            Err(RigError::NotFinite { camera: 1 })
        );
        assert_eq!(with_second(Pose3d::new(Mat3F64::IDENTITY, t)), Ok(()));
    }

    #[test]
    fn stereo_is_derived_from_a_rectified_pair() {
        let pair = |right: PinholeCamera, primary_from_camera: Pose3d| {
            SensorRig::from_cameras(vec![
                RigCamera::primary(test_camera(), SIZE),
                RigCamera::new(right, SIZE, primary_from_camera),
            ])
            .unwrap()
            .rectified_stereo_baseline_m()
        };
        let along_x = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.11, 0.0, 0.0));
        assert_eq!(pair(test_camera(), along_x), Some(0.11));
        assert_eq!(
            SensorRig::rectified_stereo(test_camera(), SIZE, 0.11).rectified_stereo_baseline_m(),
            Some(0.11)
        );

        let distorted = PinholeCamera {
            k1: -1e-3,
            ..test_camera()
        };
        assert_eq!(pair(distorted, along_x), None);
        let rotated = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 1e-3, 0.0)).matrix(),
            along_x.translation,
        );
        assert_eq!(pair(test_camera(), rotated), None);
        let offset = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.11, 1e-4, 0.0));
        assert_eq!(pair(test_camera(), offset), None);
        assert_eq!(yawed(test_camera()).rectified_stereo_baseline_m(), None);
    }

    #[test]
    fn stereo_bf_is_focal_times_baseline() {
        assert!(SensorRig::mono(test_camera(), SIZE).stereo_bf().is_none());
        let rig = SensorRig::rectified_stereo(test_camera(), SIZE, 0.11);
        assert_eq!(rig.stereo_bf(), Some(test_camera().fx * 0.11));
    }

    #[test]
    fn a_ray_projects_back_to_its_pixel() {
        let models: [Camera; 3] = [test_camera().into(), radtan().into(), fisheye624().into()];
        for model in models {
            let rig = yawed(model);
            for rig_camera in rig.cameras() {
                let camera_from_primary = rig_camera.camera_from_primary();
                for i in 0..100 {
                    let pixel =
                        Vec2F64::new((i % 10) as f64 * 64.0 + 3.5, (i / 10) as f64 * 48.0 + 2.5);
                    let Some(ray) = rig_camera.ray(pixel) else {
                        continue;
                    };
                    assert!((ray.direction.length() - 1.0).abs() < 1e-12);
                    let p_camera =
                        camera_from_primary.transform_point(&(ray.origin + ray.direction * 5.0));
                    let back = rig_camera.model.project(&p_camera).unwrap();
                    assert!(
                        (back - pixel).length() < 1e-6,
                        "{pixel:?} came back as {back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rays_of_one_camera_share_a_bit_equal_origin() {
        let rig = yawed(test_camera());
        let ray = |c: usize, u: f64, v: f64| rig.cameras()[c].ray(Vec2F64::new(u, v)).unwrap();
        assert_eq!(ray(0, 10.0, 20.0).origin, Vec3F64::ZERO);
        let (a, b) = (ray(1, 10.0, 20.0).origin, ray(1, 600.0, 400.0).origin);
        assert_eq!(
            a.to_array().map(f64::to_bits),
            b.to_array().map(f64::to_bits)
        );
    }

    #[test]
    fn b_from_a_chains_through_the_primary() {
        let rig = yawed(test_camera());
        let p0 = Vec3F64::new(0.3, -0.2, 4.0);
        let p1 = rig.b_from_a(0, 1).unwrap().transform_point(&p0);
        let back = rig.b_from_a(1, 0).unwrap().transform_point(&p1);
        assert!((back - p0).length() < 1e-12);
        let camera_from_primary = rig.cameras()[1].camera_from_primary();
        assert!((camera_from_primary.transform_point(&p0) - p1).length() < 1e-12);
        assert!(rig.b_from_a(0, 2).is_none());
    }

    /// Camera 1's extrinsic takes a camera-1 point to the body through the primary, checked
    /// on the point rather than against the implementation's own composition.
    #[test]
    fn each_rig_camera_has_its_body_extrinsic() {
        let camera_to_body = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.2, -0.4, 0.1)).matrix(),
            Vec3F64::new(0.0, 0.02, -0.01),
        );
        let rig = yawed(test_camera()).with_imu(camera_to_body);
        assert_eq!(rig.camera_to_body_of(0), Some(camera_to_body));
        let x_camera = Vec3F64::new(0.3, -0.2, 2.5);
        let through_primary = camera_to_body.transform_point(
            &rig.cameras()[1]
                .primary_from_camera
                .transform_point(&x_camera),
        );
        let x_body = rig.camera_to_body_of(1).unwrap().transform_point(&x_camera);
        assert!(
            (x_body - through_primary).length() < 1e-12,
            "{x_body:?} vs {through_primary:?}"
        );
        assert!(rig.camera_to_body_of(2).is_none());
        assert!(yawed(test_camera()).camera_to_body_of(0).is_none());
        let mono = SensorRig::mono(test_camera(), SIZE).with_imu(camera_to_body);
        assert_eq!(mono.camera_to_body_of(0), Some(camera_to_body));
        assert!(mono.camera_to_body_of(1).is_none());
    }

    #[test]
    fn a_fisheye_primary_measures_in_its_undistorted_pinhole() {
        let rig = SensorRig::mono(fisheye624(), SIZE);
        assert!(rig.primary().is_fisheye());
        let pinhole = rig.primary().pinhole();
        assert_eq!(
            [pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy],
            [241.6, 241.6, 322.9, 240.4]
        );
        assert_eq!([pinhole.k1, pinhole.k2, pinhole.p1, pinhole.p2], [0.0; 4]);
        assert!(rig.truncated_to_primary().cameras().len() == 1);
    }

    #[test]
    fn default_imu_noise_is_euroc() {
        for rig in [
            SensorRig::mono(test_camera(), SIZE),
            SensorRig::mono(test_camera(), SIZE).with_imu(Pose3d::IDENTITY),
        ] {
            let noise = rig.imu_noise();
            assert_eq!(noise.gyro_noise, 1.6968e-4);
            assert_eq!(noise.accel_noise, 2.0e-3);
            assert_eq!(noise.gyro_bias_noise, 1.9393e-5);
            assert_eq!(noise.accel_bias_noise, 3.0e-3);
        }
    }
}
