//! A rig calibration file: exactly what [`SensorRig`] stores, so loading is a
//! copy and not a recomputation (ORB-SLAM3's `Stereo.T_c1_c2` plus
//! `IMU.T_b_c1`, `Settings.cc:342, 422`).

use kornia_3d::camera::{Camera, Fisheye624Camera, FisheyeCamera, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::ImageSize;

use crate::imu::ImuCalib;
use crate::rig::{ImuCalibration, MAX_ROTATION_DEFECT, RigCamera, RigError, SensorRig};

/// Cameras of a rig, its primary camera-to-body extrinsic and IMU noise.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCalibration {
    /// The primary first.
    pub cameras: Vec<RigCameraCalibration>,
    /// `T_BC` of the primary, row-major, last row `0 0 0 1`.
    pub camera_to_body: Option<[[f64; 4]; 4]>,
    pub imu_noise: Option<ImuCalib>,
}

/// One camera of a [`RigCalibration`]: the lens model of its raw images, their
/// size and `T_PC`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCameraCalibration {
    pub model: CameraModelCalibration,
    pub width: usize,
    pub height: usize,
    /// `T_PC`, row-major (ORB-SLAM3 `Stereo.T_c1_c2`); identity for `cameras[0]`.
    pub primary_from_camera: [[f64; 4]; 4],
}

/// Lens model of a [`RigCameraCalibration`], in image pixels.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CameraModelCalibration {
    /// Brown-Conrady radial-tangential; zero distortion for rectified images.
    Pinhole {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        #[serde(default)]
        k1: f64,
        #[serde(default)]
        k2: f64,
        #[serde(default)]
        p1: f64,
        #[serde(default)]
        p2: f64,
    },
    KannalaBrandt {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    },
    /// Aria radial-tangential thin-prism fisheye.
    Fisheye624 {
        f: f64,
        cx: f64,
        cy: f64,
        k: [f64; 6],
        p: [f64; 2],
        s: [f64; 4],
    },
}

/// A [`RigCalibration`] that does not describe a rig.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RigCalibrationError {
    #[error("a rig calibration needs at least one camera")]
    NoCameras,
    #[error("cameras[0] is the primary: its primary_from_camera must be the identity")]
    PrimaryNotIdentity,
    #[error("camera {camera}: rotation is not orthonormal (defect {defect})")]
    NotRigid { camera: usize, defect: f64 },
    #[error("camera {camera}: the last row of the transform is not 0 0 0 1")]
    NotHomogeneous { camera: usize },
    #[error("camera {camera}: the transform has a non-finite entry")]
    NotFinite { camera: usize },
    #[error("camera {camera}: the rotation is a reflection")]
    Reflection { camera: usize },
    #[error("camera {camera}: an intrinsic is not finite or a focal length is not positive")]
    InvalidIntrinsics { camera: usize },
    #[error("imu_noise: every density must be finite and non-negative")]
    InvalidImuNoise,
    #[error("camera_to_body: {0}")]
    CameraToBody(Box<RigCalibrationError>),
    #[error(transparent)]
    Rig(#[from] RigError),
}

impl RigCalibration {
    /// What `rig` stores, copied.
    pub fn from_sensor_rig(rig: &SensorRig) -> Self {
        Self {
            cameras: rig
                .cameras()
                .iter()
                .map(RigCameraCalibration::new)
                .collect(),
            camera_to_body: rig.camera_to_body().as_ref().map(matrix_of),
            imu_noise: rig.imu.map(|imu| imu.noise),
        }
    }

    /// The rig this file defines on its own: its cameras, plus an IMU when
    /// `camera_to_body` is given (default noise without `imu_noise`).
    pub fn to_sensor_rig(&self) -> Result<SensorRig, RigCalibrationError> {
        self.validate()?;
        let cameras = self
            .cameras
            .iter()
            .map(RigCameraCalibration::rig_camera)
            .collect();
        let mut rig = SensorRig::from_cameras(cameras)?;
        if let Some(camera_to_body) = &self.camera_to_body {
            let mut imu = ImuCalibration::new(pose_of(camera_to_body));
            if let Some(noise) = self.imu_noise {
                imu.noise = noise;
            }
            rig.imu = Some(imu);
        }
        Ok(rig)
    }

    fn validate(&self) -> Result<(), RigCalibrationError> {
        let Some(primary) = self.cameras.first() else {
            return Err(RigCalibrationError::NoCameras);
        };
        for (camera, calibration) in self.cameras.iter().enumerate() {
            check_rigid(&calibration.primary_from_camera, camera)?;
            if !calibration.model.is_valid() {
                return Err(RigCalibrationError::InvalidIntrinsics { camera });
            }
        }
        if let Some(noise) = &self.imu_noise {
            let densities = [
                noise.gyro_noise,
                noise.accel_noise,
                noise.gyro_bias_noise,
                noise.accel_bias_noise,
            ];
            if !densities.iter().all(|d| d.is_finite() && *d >= 0.0) {
                return Err(RigCalibrationError::InvalidImuNoise);
            }
        }
        // `camera_to_body` enters every IMU residual; it gets the same checks.
        if let Some(camera_to_body) = &self.camera_to_body {
            check_rigid(camera_to_body, 0)
                .map_err(|e| RigCalibrationError::CameraToBody(Box::new(e)))?;
        }
        if pose_of(&primary.primary_from_camera) != Pose3d::IDENTITY {
            return Err(RigCalibrationError::PrimaryNotIdentity);
        }
        Ok(())
    }
}

impl RigCameraCalibration {
    fn new(camera: &RigCamera) -> Self {
        Self {
            model: CameraModelCalibration::from(&camera.model),
            width: camera.size.width,
            height: camera.size.height,
            primary_from_camera: matrix_of(&camera.primary_from_camera),
        }
    }

    fn rig_camera(&self) -> RigCamera {
        let size = ImageSize {
            width: self.width,
            height: self.height,
        };
        RigCamera::new(
            self.model.camera(),
            size,
            pose_of(&self.primary_from_camera),
        )
    }
}

impl From<&Camera> for CameraModelCalibration {
    fn from(camera: &Camera) -> Self {
        match camera {
            Camera::Pinhole(c) => Self::Pinhole {
                fx: c.fx,
                fy: c.fy,
                cx: c.cx,
                cy: c.cy,
                k1: c.k1,
                k2: c.k2,
                p1: c.p1,
                p2: c.p2,
            },
            Camera::Fisheye(c) => Self::KannalaBrandt {
                fx: c.fx,
                fy: c.fy,
                cx: c.cx,
                cy: c.cy,
                k1: c.k1,
                k2: c.k2,
                k3: c.k3,
                k4: c.k4,
            },
            Camera::Fisheye624(c) => Self::Fisheye624 {
                f: c.f,
                cx: c.cx,
                cy: c.cy,
                k: c.k,
                p: c.p,
                s: c.s,
            },
        }
    }
}

impl CameraModelCalibration {
    /// Finite intrinsics and positive focal lengths; a zero focal would map
    /// every keypoint onto the principal point.
    fn is_valid(&self) -> bool {
        let focal = |f: f64| f.is_finite() && f > 0.0;
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite());
        match self {
            Self::Pinhole {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            } => focal(*fx) && focal(*fy) && finite(&[*cx, *cy, *k1, *k2, *p1, *p2]),
            Self::KannalaBrandt {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            } => focal(*fx) && focal(*fy) && finite(&[*cx, *cy, *k1, *k2, *k3, *k4]),
            Self::Fisheye624 { f, cx, cy, k, p, s } => {
                focal(*f) && finite(&[*cx, *cy]) && finite(k) && finite(p) && finite(s)
            }
        }
    }

    fn camera(&self) -> Camera {
        match *self {
            Self::Pinhole {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            } => PinholeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            }
            .into(),
            Self::KannalaBrandt {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            } => FisheyeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            }
            .into(),
            Self::Fisheye624 { f, cx, cy, k, p, s } => {
                Fisheye624Camera { f, cx, cy, k, p, s }.into()
            }
        }
    }
}

/// Row-major `[R t; 0 0 0 1]`; entries are copied, so the round trip is exact.
fn matrix_of(pose: &Pose3d) -> [[f64; 4]; 4] {
    let r = pose.rotation.to_cols_array();
    let t = pose.translation;
    [
        [r[0], r[3], r[6], t.x],
        [r[1], r[4], r[7], t.y],
        [r[2], r[5], r[8], t.z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Refuses a transform that is not a finite, homogeneous, proper rigid
/// motion; `camera` names it in the error.
fn check_rigid(transform: &[[f64; 4]; 4], camera: usize) -> Result<(), RigCalibrationError> {
    if transform[3] != [0.0, 0.0, 0.0, 1.0] {
        return Err(RigCalibrationError::NotHomogeneous { camera });
    }
    if transform.iter().flatten().any(|v| !v.is_finite()) {
        return Err(RigCalibrationError::NotFinite { camera });
    }
    let pose = pose_of(transform);
    let defect = pose.rotation_defect();
    if defect > MAX_ROTATION_DEFECT {
        return Err(RigCalibrationError::NotRigid { camera, defect });
    }
    // `RᵀR = I` holds for a reflection too, which would mirror every ray.
    if pose.rotation.determinant() <= 0.0 {
        return Err(RigCalibrationError::Reflection { camera });
    }
    Ok(())
}

fn pose_of(m: &[[f64; 4]; 4]) -> Pose3d {
    Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(m[0][0], m[1][0], m[2][0]),
            Vec3F64::new(m[0][1], m[1][1], m[2][1]),
            Vec3F64::new(m[0][2], m[1][2], m[2][2]),
        ),
        Vec3F64::new(m[0][3], m[1][3], m[2][3]),
    )
}

/// Rig cameras from per-camera body extrinsics `T_BC` (Kalibr, LaMAria, EuRoC):
/// `T_PC = T_BC[primary]⁻¹ ∘ T_BC[c]`, the primary first.
pub fn rig_cameras_from_body_extrinsics(
    cameras: &[(Camera, ImageSize, Pose3d)],
    primary: usize,
) -> Vec<RigCamera> {
    let Some((_, _, primary_to_body)) = cameras.get(primary) else {
        return Vec::new();
    };
    let primary_from_body = primary_to_body.inverse();
    std::iter::once(primary)
        .chain((0..cameras.len()).filter(|&c| c != primary))
        .map(|c| {
            let (model, size, camera_to_body) = &cameras[c];
            let primary_from_camera = if c == primary {
                Pose3d::IDENTITY
            } else {
                primary_from_body.compose(camera_to_body)
            };
            RigCamera::new(model.clone(), *size, primary_from_camera)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    fn camera(fx: f64) -> PinholeCamera {
        PinholeCamera {
            fx,
            fy: fx * 1.0001,
            cx: 319.7,
            cy: 241.3,
            k1: -0.0123,
            k2: 0.00045,
            p1: 1.5e-5,
            p2: -2.25e-5,
        }
    }

    const SIZE: ImageSize = ImageSize {
        width: 640,
        height: 480,
    };

    fn yaw_70() -> Pose3d {
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 70f64.to_radians(), 0.0)).matrix();
        Pose3d::new(yaw, Vec3F64::new(0.11, 0.0013, -0.0021))
    }

    fn imu() -> ImuCalibration {
        ImuCalibration {
            noise: ImuCalib {
                gyro_noise: 1.3e-4,
                accel_noise: 2.7e-3,
                gyro_bias_noise: 3.1e-6,
                accel_bias_noise: 4.3e-4,
            },
            camera_to_body: Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix(),
                Vec3F64::new(-0.0042, 0.0113, 0.0071),
            ),
        }
    }

    /// Two cameras, the second yawed 70 deg and 11 cm away, with an IMU.
    fn yawed_rig() -> SensorRig {
        let mut rig = SensorRig::from_cameras(vec![
            RigCamera::primary(camera(241.9), SIZE),
            RigCamera::new(camera(243.1), SIZE, yaw_70()),
        ])
        .unwrap();
        rig.imu = Some(imu());
        rig
    }

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

    fn kannala_brandt() -> FisheyeCamera {
        FisheyeCamera {
            fx: 461.64,
            fy: 459.72,
            cx: 732.95,
            cy: 720.54,
            k1: 0.0344,
            k2: -0.0216,
            k3: 0.0031,
            k4: -0.0005,
        }
    }

    fn model_params(model: &CameraModelCalibration) -> Vec<f64> {
        match model {
            CameraModelCalibration::Pinhole {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            } => vec![*fx, *fy, *cx, *cy, *k1, *k2, *p1, *p2],
            CameraModelCalibration::KannalaBrandt {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            } => vec![*fx, *fy, *cx, *cy, *k1, *k2, *k3, *k4],
            CameraModelCalibration::Fisheye624 { f, cx, cy, k, p, s } => [*f, *cx, *cy]
                .into_iter()
                .chain(k.iter().chain(p).chain(s).copied())
                .collect(),
        }
    }

    fn bits(calibration: &RigCalibration) -> Vec<u64> {
        let mut bits = Vec::new();
        for c in &calibration.cameras {
            bits.extend(model_params(&c.model).into_iter().map(f64::to_bits));
            bits.extend([c.width as u64, c.height as u64]);
            bits.extend(c.primary_from_camera.iter().flatten().map(|v| v.to_bits()));
        }
        if let Some(m) = &calibration.camera_to_body {
            bits.extend(m.iter().flatten().map(|v| v.to_bits()));
        }
        if let Some(n) = calibration.imu_noise {
            bits.extend(
                [
                    n.gyro_noise,
                    n.accel_noise,
                    n.gyro_bias_noise,
                    n.accel_bias_noise,
                ]
                .map(f64::to_bits),
            );
        }
        bits
    }

    fn round_trip(calibration: &RigCalibration) -> RigCalibration {
        let text =
            ron::ser::to_string_pretty(calibration, ron::ser::PrettyConfig::default()).unwrap();
        ron::from_str(&text).unwrap()
    }

    #[test]
    fn a_written_calibration_reads_back_bit_equal() {
        let written = RigCalibration::from_sensor_rig(&yawed_rig());
        let read = round_trip(&written);
        assert_eq!(bits(&read), bits(&written));

        let rig = read.to_sensor_rig().unwrap();
        let reread = RigCalibration::from_sensor_rig(&rig);
        assert_eq!(bits(&reread), bits(&written));
        assert_eq!(rig.primary().pinhole().fx.to_bits(), 241.9f64.to_bits());
        assert_eq!(rig.cameras()[1].size, SIZE);
    }

    #[test]
    fn fisheye_models_read_back_bit_equal() {
        let mut rig = SensorRig::from_cameras(vec![
            RigCamera::primary(kannala_brandt(), SIZE),
            RigCamera::new(fisheye624(), SIZE, yaw_70()),
        ])
        .unwrap();
        rig.imu = Some(imu());
        let written = RigCalibration::from_sensor_rig(&rig);
        assert!(matches!(
            written.cameras[0].model,
            CameraModelCalibration::KannalaBrandt { .. }
        ));
        assert!(matches!(
            written.cameras[1].model,
            CameraModelCalibration::Fisheye624 { .. }
        ));
        let read = round_trip(&written);
        assert_eq!(bits(&read), bits(&written));
        let reread = RigCalibration::from_sensor_rig(&read.to_sensor_rig().unwrap());
        assert_eq!(bits(&reread), bits(&written));
    }

    #[test]
    fn a_file_alone_defines_a_fisheye_rig() {
        let identity = matrix_of(&Pose3d::IDENTITY);
        let fisheye = |primary_from_camera| RigCameraCalibration {
            model: CameraModelCalibration::from(&Camera::from(fisheye624())),
            width: 640,
            height: 480,
            primary_from_camera,
        };
        let file = RigCalibration {
            cameras: vec![fisheye(identity), fisheye(matrix_of(&yaw_70()))],
            camera_to_body: None,
            imu_noise: None,
        };
        let rig = file.to_sensor_rig().unwrap();
        assert_eq!(rig.cameras().len(), 2);
        assert!(rig.cameras().iter().all(RigCamera::is_fisheye));
        assert_eq!(rig.cameras()[1].size, SIZE);
        assert_eq!(rig.cameras()[1].primary_from_camera, yaw_70());
        assert!(rig.imu.is_none());
        assert!(rig.rectified_stereo_baseline_m().is_none());
    }

    #[test]
    fn a_file_without_camera_to_body_defines_no_imu() {
        let mut calibration = RigCalibration::from_sensor_rig(&yawed_rig());
        let with_imu = calibration.to_sensor_rig().unwrap();
        assert_eq!(with_imu.imu_noise(), imu().noise);
        assert_eq!(with_imu.camera_to_body(), Some(imu().camera_to_body));
        calibration.camera_to_body = None;
        assert!(calibration.to_sensor_rig().unwrap().imu.is_none());
    }

    #[test]
    fn transforms_that_are_not_a_rig_are_rejected() {
        let good = RigCalibration::from_sensor_rig(&yawed_rig());
        let apply = |calibration: &RigCalibration| calibration.to_sensor_rig().map(|_| ());

        let mut primary_moved = good.clone();
        primary_moved.cameras[0].primary_from_camera[0][3] = 0.01;
        assert_eq!(
            apply(&primary_moved),
            Err(RigCalibrationError::PrimaryNotIdentity)
        );

        let scaled = |s: f64| {
            let mut c = good.clone();
            for row in &mut c.cameras[1].primary_from_camera[..3] {
                for v in &mut row[..3] {
                    *v *= s;
                }
            }
            c
        };
        // A scale 1 + e gives a defect of about 2e.
        assert!(matches!(
            apply(&scaled(1.0 + 5e-4)),
            Err(RigCalibrationError::NotRigid { camera: 1, .. })
        ));
        assert_eq!(apply(&scaled(1.0 + 5e-10)), Ok(()));

        let mut projective = good.clone();
        projective.cameras[1].primary_from_camera[3][2] = 1e-3;
        assert_eq!(
            apply(&projective),
            Err(RigCalibrationError::NotHomogeneous { camera: 1 })
        );

        // An orthonormal reflection and a NaN translation are not rigid either.
        let mut mirrored = good.clone();
        for v in &mut mirrored.cameras[1].primary_from_camera[2][..3] {
            *v = -*v;
        }
        assert_eq!(
            apply(&mirrored),
            Err(RigCalibrationError::Reflection { camera: 1 })
        );
        let mut not_finite = good.clone();
        not_finite.cameras[1].primary_from_camera[1][3] = f64::NAN;
        assert_eq!(
            apply(&not_finite),
            Err(RigCalibrationError::NotFinite { camera: 1 })
        );

        let mut empty_image = good.clone();
        empty_image.cameras[1].width = 0;
        assert_eq!(
            apply(&empty_image),
            Err(RigCalibrationError::Rig(RigError::EmptyImage { camera: 1 }))
        );

        let empty = RigCalibration {
            cameras: Vec::new(),
            ..good
        };
        assert_eq!(apply(&empty), Err(RigCalibrationError::NoCameras));
    }

    #[test]
    fn a_non_rigid_camera_to_body_is_refused() {
        let good = RigCalibration::from_sensor_rig(&yawed_rig());
        let apply = |calibration: &RigCalibration| calibration.to_sensor_rig().map(|_| ());
        assert_eq!(apply(&good), Ok(()));
        let refused = |edit: fn(&mut [[f64; 4]; 4])| {
            let mut c = good.clone();
            edit(c.camera_to_body.as_mut().unwrap());
            apply(&c).unwrap_err()
        };

        let mirrored = refused(|m| {
            for v in &mut m[2][..3] {
                *v = -*v;
            }
        });
        assert_eq!(
            mirrored,
            RigCalibrationError::CameraToBody(Box::new(RigCalibrationError::Reflection {
                camera: 0
            }))
        );
        let not_finite = refused(|m| m[0][3] = f64::NAN);
        assert_eq!(
            not_finite,
            RigCalibrationError::CameraToBody(Box::new(RigCalibrationError::NotFinite {
                camera: 0
            }))
        );
        let projective = refused(|m| m[3][0] = 1e-3);
        assert_eq!(
            projective,
            RigCalibrationError::CameraToBody(Box::new(RigCalibrationError::NotHomogeneous {
                camera: 0
            }))
        );
        let scaled = refused(|m| {
            for row in &mut m[..3] {
                for v in &mut row[..3] {
                    *v *= 1.0 + 5e-4;
                }
            }
        });
        assert!(matches!(
            scaled,
            RigCalibrationError::CameraToBody(inner)
                if matches!(*inner, RigCalibrationError::NotRigid { .. })
        ));
    }

    #[test]
    fn intrinsics_and_imu_noise_must_be_usable() {
        let good = RigCalibration::from_sensor_rig(&yawed_rig());
        let apply = |calibration: &RigCalibration| calibration.to_sensor_rig().map(|_| ());
        assert_eq!(apply(&good), Ok(()));

        let mut zero_focal = good.clone();
        if let CameraModelCalibration::Pinhole { fx, .. } = &mut zero_focal.cameras[1].model {
            *fx = 0.0;
        }
        assert_eq!(
            apply(&zero_focal),
            Err(RigCalibrationError::InvalidIntrinsics { camera: 1 })
        );
        let mut nan_distortion =
            RigCalibration::from_sensor_rig(&SensorRig::mono(fisheye624(), SIZE));
        if let CameraModelCalibration::Fisheye624 { s, .. } = &mut nan_distortion.cameras[0].model {
            s[3] = f64::NAN;
        }
        assert_eq!(
            apply(&nan_distortion),
            Err(RigCalibrationError::InvalidIntrinsics { camera: 0 })
        );

        let mut nan_noise = good.clone();
        if let Some(noise) = &mut nan_noise.imu_noise {
            noise.accel_bias_noise = f64::NAN;
        }
        assert_eq!(apply(&nan_noise), Err(RigCalibrationError::InvalidImuNoise));
        let mut negative_noise = good;
        if let Some(noise) = &mut negative_noise.imu_noise {
            noise.gyro_noise = -1e-4;
        }
        assert_eq!(
            apply(&negative_noise),
            Err(RigCalibrationError::InvalidImuNoise)
        );
    }

    #[test]
    fn body_extrinsics_compose_through_the_primary() {
        let t_bc = |w: Vec3F64, t: Vec3F64| Pose3d::new(SO3F64::exp(w).matrix(), t);
        let cameras = [
            (
                Camera::from(camera(240.0)),
                SIZE,
                t_bc(Vec3F64::new(0.1, 0.2, 0.3), Vec3F64::new(0.01, 0.0, 0.0)),
            ),
            (
                Camera::from(camera(250.0)),
                SIZE,
                t_bc(Vec3F64::new(-0.3, 1.1, 0.2), Vec3F64::new(-0.1, 0.02, 0.0)),
            ),
        ];
        let rig = rig_cameras_from_body_extrinsics(&cameras, 1);
        assert_eq!(rig.len(), 2);
        assert_eq!(rig[0].pinhole().fx, 250.0);
        assert_eq!(rig[0].primary_from_camera, Pose3d::IDENTITY);
        // Camera 0's point reaches the body the same way directly and through the primary.
        let x_camera = Vec3F64::new(0.3, -0.2, 2.5);
        let direct = cameras[0].2.transform_point(&x_camera);
        let through_primary = cameras[1]
            .2
            .transform_point(&rig[1].primary_from_camera.transform_point(&x_camera));
        assert!(
            (direct - through_primary).length() < 1e-12,
            "{direct:?} vs {through_primary:?}"
        );
        assert!(rig_cameras_from_body_extrinsics(&cameras, 2).is_empty());
    }
}
