//! A rig calibration file: exactly what [`SensorRig`] stores, so loading is a
//! copy and not a recomputation (ORB-SLAM3's `Stereo.T_c1_c2` plus
//! `IMU.T_b_c1`, `Settings.cc:342, 422`).

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::ImageSize;

use crate::imu::ImuCalib;
use crate::rig::{RigCamera, SensorRig};

/// Camera geometry of a rig, its primary camera-to-body extrinsic and IMU noise.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCalibration {
    /// The primary first.
    pub cameras: Vec<RigCameraCalibration>,
    /// `T_BC` of the primary, row-major, last row `0 0 0 1`.
    pub camera_to_body: Option<[[f64; 4]; 4]>,
    pub imu_noise: Option<ImuCalib>,
}

/// One camera of a [`RigCalibration`]: undistorted pinhole intrinsics and
/// `T_PC`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCameraCalibration {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
    pub width: usize,
    pub height: usize,
    /// `T_PC`, row-major (ORB-SLAM3 `Stereo.T_c1_c2`); identity for `cameras[0]`.
    pub primary_from_camera: [[f64; 4]; 4],
}

/// A [`RigCalibration`] that does not describe a rig.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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
}

/// Largest [`Pose3d::rotation_defect`] a written transform may have: the
/// rounding of printed digits, not a scale.
const MAX_ROTATION_DEFECT: f64 = 1e-6;

impl RigCalibration {
    /// What `rig` stores, copied.
    pub fn from_sensor_rig(rig: &SensorRig) -> Self {
        let cameras = if rig.cameras.is_empty() {
            // Outside a multi-camera rig the image size is not recorded.
            let size = ImageSize {
                width: 0,
                height: 0,
            };
            vec![RigCameraCalibration::new(
                &rig.camera,
                size,
                &Pose3d::IDENTITY,
            )]
        } else {
            rig.cameras
                .iter()
                .map(|c| RigCameraCalibration::new(&c.camera, c.size, &c.primary_from_camera))
                .collect()
        };
        Self {
            cameras,
            camera_to_body: rig.camera_to_body().as_ref().map(matrix_of),
            imu_noise: rig.imu.map(|imu| imu.noise),
        }
    }

    /// Replaces `rig`'s camera geometry, plus `camera_to_body` and the noise
    /// when `rig.imu` is `Some`. A rig without IMU stays without one: the IMU
    /// selection belongs to the system config.
    pub fn apply_to(&self, mut rig: SensorRig) -> Result<SensorRig, RigCalibrationError> {
        self.validate()?;
        // The file holds pinhole geometry only; a raw model stays with its
        // camera so raw images are never read as pinhole ones.
        let cameras: Vec<RigCamera> = self
            .cameras
            .iter()
            .enumerate()
            .map(|(c, calibration)| RigCamera {
                raw_model: rig.cameras.get(c).and_then(|camera| camera.raw_model),
                ..calibration.rig_camera()
            })
            .collect();
        let Some(primary) = cameras.first() else {
            return Err(RigCalibrationError::NoCameras);
        };
        rig.camera = primary.camera.clone();
        if cameras.len() > 1 || !rig.cameras.is_empty() {
            rig.cameras = cameras;
        }
        if let Some(imu) = rig.imu.as_mut() {
            if let Some(camera_to_body) = &self.camera_to_body {
                imu.camera_to_body = pose_of(camera_to_body);
            }
            if let Some(noise) = self.imu_noise {
                imu.noise = noise;
            }
        }
        Ok(rig)
    }

    fn validate(&self) -> Result<(), RigCalibrationError> {
        let Some(primary) = self.cameras.first() else {
            return Err(RigCalibrationError::NoCameras);
        };
        for (camera, calibration) in self.cameras.iter().enumerate() {
            let transform = &calibration.primary_from_camera;
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
        }
        if pose_of(&primary.primary_from_camera) != Pose3d::IDENTITY {
            return Err(RigCalibrationError::PrimaryNotIdentity);
        }
        Ok(())
    }
}

impl RigCameraCalibration {
    fn new(camera: &PinholeCamera, size: ImageSize, primary_from_camera: &Pose3d) -> Self {
        Self {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            k1: camera.k1,
            k2: camera.k2,
            p1: camera.p1,
            p2: camera.p2,
            width: size.width,
            height: size.height,
            primary_from_camera: matrix_of(primary_from_camera),
        }
    }

    fn rig_camera(&self) -> RigCamera {
        RigCamera {
            camera: PinholeCamera {
                fx: self.fx,
                fy: self.fy,
                cx: self.cx,
                cy: self.cy,
                k1: self.k1,
                k2: self.k2,
                p1: self.p1,
                p2: self.p2,
            },
            size: ImageSize {
                width: self.width,
                height: self.height,
            },
            primary_from_camera: pose_of(&self.primary_from_camera),
            raw_model: None,
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
    cameras: &[(PinholeCamera, ImageSize, Pose3d)],
    primary: usize,
) -> Vec<RigCamera> {
    let Some((_, _, primary_to_body)) = cameras.get(primary) else {
        return Vec::new();
    };
    let primary_from_body = primary_to_body.inverse();
    std::iter::once(primary)
        .chain((0..cameras.len()).filter(|&c| c != primary))
        .map(|c| {
            let (camera, size, camera_to_body) = &cameras[c];
            RigCamera {
                camera: camera.clone(),
                size: *size,
                primary_from_camera: if c == primary {
                    Pose3d::IDENTITY
                } else {
                    primary_from_body.compose(camera_to_body)
                },
                raw_model: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::ImuCalibration;
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

    /// Two cameras, the second yawed 70 deg and 11 cm away, with an IMU.
    fn yawed_rig() -> SensorRig {
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 70f64.to_radians(), 0.0)).matrix();
        let mut rig = SensorRig::new(camera(241.9)).with_rig_cameras(vec![
            RigCamera {
                camera: camera(241.9),
                size,
                primary_from_camera: Pose3d::IDENTITY,
                raw_model: None,
            },
            RigCamera {
                camera: camera(243.1),
                size,
                primary_from_camera: Pose3d::new(yaw, Vec3F64::new(0.11, 0.0013, -0.0021)),
                raw_model: None,
            },
        ]);
        rig.imu = Some(ImuCalibration {
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
        });
        rig
    }

    fn bits(calibration: &RigCalibration) -> Vec<u64> {
        let mut bits = Vec::new();
        for c in &calibration.cameras {
            bits.extend([c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2].map(f64::to_bits));
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

    #[test]
    fn a_written_calibration_reads_back_bit_equal() {
        let written = RigCalibration::from_sensor_rig(&yawed_rig());
        let text = ron::ser::to_string_pretty(&written, ron::ser::PrettyConfig::default()).unwrap();
        let read: RigCalibration = ron::from_str(&text).unwrap();
        assert_eq!(bits(&read), bits(&written));

        let blank = SensorRig::new(camera(100.0))
            .with_rig_cameras(yawed_rig().cameras)
            .with_imu(Pose3d::IDENTITY);
        let applied = read.apply_to(blank).unwrap();
        let reread = RigCalibration::from_sensor_rig(&applied);
        assert_eq!(bits(&reread), bits(&written));
        assert_eq!(applied.camera.fx.to_bits(), written.cameras[0].fx.to_bits());
    }

    #[test]
    fn a_calibration_keeps_each_cameras_raw_model() {
        let raw = crate::Fisheye624Camera {
            f: 241.6,
            cx: 322.9,
            cy: 240.4,
            k: [-0.025, 0.096, -0.063, 0.0062, 0.0035, -0.00073],
            p: [0.0009, 0.00036],
            s: [-0.0014, -0.0001, -0.00046, -0.0001],
        };
        let calibration = RigCalibration::from_sensor_rig(&yawed_rig());
        let mut rig = yawed_rig();
        rig.cameras[1].raw_model = Some(raw);
        let applied = calibration.apply_to(rig).unwrap();
        assert_eq!(applied.cameras[0].raw_model, None);
        assert_eq!(applied.cameras[1].raw_model, Some(raw));
    }

    #[test]
    fn a_rig_without_imu_stays_without_one() {
        let calibration = RigCalibration::from_sensor_rig(&yawed_rig());
        let mut rig = yawed_rig();
        rig.imu = None;
        assert!(calibration.apply_to(rig).unwrap().imu.is_none());
    }

    #[test]
    fn transforms_that_are_not_a_rig_are_rejected() {
        let good = RigCalibration::from_sensor_rig(&yawed_rig());
        let apply = |calibration: &RigCalibration| calibration.apply_to(yawed_rig()).map(|_| ());

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

        // G10: an orthonormal reflection and a NaN translation are not rigid either.
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

        let empty = RigCalibration {
            cameras: Vec::new(),
            ..good
        };
        assert_eq!(apply(&empty), Err(RigCalibrationError::NoCameras));
    }

    #[test]
    fn body_extrinsics_compose_through_the_primary() {
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let t_bc = |w: Vec3F64, t: Vec3F64| Pose3d::new(SO3F64::exp(w).matrix(), t);
        let cameras = [
            (
                camera(240.0),
                size,
                t_bc(Vec3F64::new(0.1, 0.2, 0.3), Vec3F64::new(0.01, 0.0, 0.0)),
            ),
            (
                camera(250.0),
                size,
                t_bc(Vec3F64::new(-0.3, 1.1, 0.2), Vec3F64::new(-0.1, 0.02, 0.0)),
            ),
        ];
        let rig = rig_cameras_from_body_extrinsics(&cameras, 1);
        assert_eq!(rig.len(), 2);
        assert_eq!(rig[0].camera.fx, 250.0);
        assert_eq!(rig[0].primary_from_camera, Pose3d::IDENTITY);
        // G9: camera 0's point reaches the body the same way directly and through the primary.
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
