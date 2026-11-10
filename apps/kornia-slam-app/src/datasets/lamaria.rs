//! Reader for LaMAria sequences in the pinhole ASL export
//! (`<seq>/aria/{cam0,cam1,imu0}`) with their `pinhole_calibrations/<seq>.json`.
//!
//! The export ships raw Aria IMU samples. When the factory calibration record
//! from the recording's `.vrs` header is supplied, samples are rectified as
//! `real = R⁻¹ (raw − b)` per sensor (Aria's linear IMU model; idea from
//! visloc-rs `docs/lamaria_imu_rectification.md`).

use std::path::{Path, PathBuf};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::ImuCalib;
use serde::Deserialize;

use super::euroc::{DatasetError, DatasetSample, ImuSample, read_camera_csv, read_imu_csv};

/// Factory IMU whose frame LaMAria's `imu0` stream and `T_b_s` extrinsics use.
const FACTORY_IMU_LABEL: &str = "imu-right";

/// One camera of the rig: an undistorted pinhole view of its own size.
#[derive(Debug, Clone)]
pub struct LamariaCamera {
    pub camera: PinholeCamera,
    pub width: usize,
    pub height: usize,
    /// `T_BC`: `X_imu = T_BC * X_cam`.
    pub camera_to_body: Pose3d,
}

/// Calibration of a sequence: both SLAM cameras and the `imu0` noise model.
#[derive(Debug, Clone)]
pub struct LamariaCalibration {
    pub cameras: [LamariaCamera; 2],
    pub imu_noise: ImuCalib,
}

/// `real = rectification_inv * (raw - bias)` for one IMU sensor.
#[derive(Debug, Clone, Copy)]
pub struct SensorRectification {
    pub rectification_inv: Mat3F64,
    pub bias: Vec3F64,
}

impl SensorRectification {
    pub fn apply(&self, raw: [f64; 3]) -> [f64; 3] {
        let v = self.rectification_inv * (Vec3F64::from_array(raw) - self.bias);
        [v.x, v.y, v.z]
    }
}

/// Aria factory rectification of the accelerometer and the gyroscope.
#[derive(Debug, Clone, Copy)]
pub struct ImuRectification {
    pub accel: SensorRectification,
    pub gyro: SensorRectification,
}

impl ImuRectification {
    pub fn apply(&self, sample: &ImuSample) -> ImuSample {
        ImuSample {
            timestamp_sec: sample.timestamp_sec,
            gyro: self.gyro.apply(sample.gyro),
            accel: self.accel.apply(sample.accel),
        }
    }
}

/// A LaMAria sequence: time-ordered `cam0`/`cam1` images and `imu0` samples.
#[derive(Debug, Clone)]
pub struct LamariaDataset {
    pub calibration: LamariaCalibration,
    /// `cam0` and `cam1` samples; the export writes them with equal timestamps.
    pub cameras: [Vec<DatasetSample>; 2],
    pub imu_samples: Vec<ImuSample>,
    pub imu_rectified: bool,
}

impl LamariaDataset {
    /// Opens `root/aria` with the calibration JSON at `calibration`. With
    /// `factory_calibration`, the IMU samples are rectified; `imu_time_offset_ns`
    /// is added to every IMU timestamp.
    pub fn open(
        root: &Path,
        calibration: &Path,
        factory_calibration: Option<&Path>,
        imu_time_offset_ns: i64,
    ) -> Result<Self, DatasetError> {
        let calibration = read_calibration(calibration)?;
        let aria = root.join("aria");
        let read_cam = |name: &str| {
            let dir = aria.join(name);
            read_camera_csv(&dir.join("data.csv"), &dir.join("data"))
        };
        let cameras = [read_cam("cam0")?, read_cam("cam1")?];
        if cameras[0].len() != cameras[1].len()
            || cameras[0]
                .iter()
                .zip(&cameras[1])
                .any(|(a, b)| (a.timestamp_sec - b.timestamp_sec).abs() > 1e-6)
        {
            return Err(DatasetError::Parse(
                "cam0 and cam1 timestamps differ".into(),
            ));
        }
        let mut imu_samples = read_imu_csv(&aria.join("imu0").join("data.csv"))?;
        let offset_sec = imu_time_offset_ns as f64 * 1e-9;
        for sample in &mut imu_samples {
            sample.timestamp_sec += offset_sec;
        }
        let imu_rectified = match factory_calibration {
            Some(path) => {
                let rectification = read_factory_rectification(path)?;
                for sample in &mut imu_samples {
                    *sample = rectification.apply(sample);
                }
                true
            }
            None => false,
        };
        Ok(Self {
            calibration,
            cameras,
            imu_samples,
            imu_rectified,
        })
    }

    pub fn len(&self) -> usize {
        self.cameras[0].len()
    }
}

#[derive(Debug, Deserialize)]
struct CalibrationJson {
    cam0: CameraJson,
    cam1: CameraJson,
    imu0: ImuJson,
}

#[derive(Debug, Deserialize)]
struct CameraJson {
    model: String,
    params: Vec<f64>,
    resolution: ResolutionJson,
    #[serde(rename = "T_b_s")]
    t_b_s: RigidJson,
}

#[derive(Debug, Deserialize)]
struct ResolutionJson {
    width: usize,
    height: usize,
}

/// A rigid transform with `qvec` in **xyzw** order (pycolmap `Rotation3d`).
#[derive(Debug, Deserialize)]
struct RigidJson {
    qvec: [f64; 4],
    tvec: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct ImuJson {
    acc_noise_density: f64,
    gyro_noise_density: f64,
    acc_bias_random_walk_sigma: f64,
    gyro_bias_random_walk_sigma: f64,
}

fn read_calibration(path: &Path) -> Result<LamariaCalibration, DatasetError> {
    let json: CalibrationJson = read_json(path)?;
    Ok(LamariaCalibration {
        cameras: [camera(json.cam0, path)?, camera(json.cam1, path)?],
        imu_noise: ImuCalib {
            gyro_noise: json.imu0.gyro_noise_density,
            accel_noise: json.imu0.acc_noise_density,
            gyro_bias_noise: json.imu0.gyro_bias_random_walk_sigma,
            accel_bias_noise: json.imu0.acc_bias_random_walk_sigma,
        },
    })
}

fn camera(json: CameraJson, path: &Path) -> Result<LamariaCamera, DatasetError> {
    let [fx, fy, cx, cy] = json.params.as_slice() else {
        return Err(DatasetError::Parse(format!(
            "expected 4 pinhole params in {}",
            path.display()
        )));
    };
    if json.model != "PINHOLE" {
        return Err(DatasetError::Parse(format!(
            "unsupported camera model '{}' in {}",
            json.model,
            path.display()
        )));
    }
    Ok(LamariaCamera {
        camera: PinholeCamera {
            fx: *fx,
            fy: *fy,
            cx: *cx,
            cy: *cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
        width: json.resolution.width,
        height: json.resolution.height,
        camera_to_body: rigid(&json.t_b_s),
    })
}

fn rigid(json: &RigidJson) -> Pose3d {
    let [x, y, z, w] = json.qvec;
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    let rotation = SO3F64::from_array([x / norm, y / norm, z / norm, w / norm]).matrix();
    Pose3d::new(rotation, Vec3F64::from_array(json.tvec))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FactoryJson {
    imu_calibrations: Vec<FactoryImuJson>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FactoryImuJson {
    label: String,
    accelerometer: FactorySensorJson,
    gyroscope: FactorySensorJson,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FactorySensorJson {
    bias: FactoryBiasJson,
    model: FactoryModelJson,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FactoryBiasJson {
    offset: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FactoryModelJson {
    rectification_matrix: [[f64; 3]; 3],
}

/// Reads the `imu-right` rectification from an Aria factory calibration record.
pub fn read_factory_rectification(path: &Path) -> Result<ImuRectification, DatasetError> {
    let json: FactoryJson = read_json(path)?;
    let imu = json
        .imu_calibrations
        .iter()
        .find(|imu| imu.label == FACTORY_IMU_LABEL)
        .ok_or_else(|| {
            DatasetError::Parse(format!(
                "no {FACTORY_IMU_LABEL} calibration in {}",
                path.display()
            ))
        })?;
    Ok(ImuRectification {
        accel: sensor_rectification(&imu.accelerometer, path)?,
        gyro: sensor_rectification(&imu.gyroscope, path)?,
    })
}

fn sensor_rectification(
    json: &FactorySensorJson,
    path: &Path,
) -> Result<SensorRectification, DatasetError> {
    let rows = json.model.rectification_matrix;
    let matrix = Mat3F64::from_cols(
        Vec3F64::new(rows[0][0], rows[1][0], rows[2][0]),
        Vec3F64::new(rows[0][1], rows[1][1], rows[2][1]),
        Vec3F64::new(rows[0][2], rows[1][2], rows[2][2]),
    );
    if matrix.determinant().abs() < 1e-6 {
        return Err(DatasetError::Parse(format!(
            "singular IMU rectification matrix in {}",
            path.display()
        )));
    }
    Ok(SensorRectification {
        rectification_inv: matrix.inverse(),
        bias: Vec3F64::from_array(json.bias.offset),
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DatasetError> {
    if !path.exists() {
        return Err(DatasetError::FileNotFound(PathBuf::from(path)));
    }
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| DatasetError::Parse(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle_deg(rotation: Mat3F64) -> f64 {
        ((rotation.x_axis.x + rotation.y_axis.y + rotation.z_axis.z - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    #[test]
    fn the_identity_qvec_is_the_identity_rotation() {
        let pose = rigid(&RigidJson {
            qvec: [0.0, 0.0, 0.0, 1.0],
            tvec: [1.0, 2.0, 3.0],
        });
        assert!(angle_deg(pose.rotation) < 1e-9);
        assert_eq!(pose.translation, Vec3F64::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn qvec_is_read_as_xyzw() {
        // 90 deg about z: x = y = 0, z = w = sqrt(1/2).
        let h = 0.5f64.sqrt();
        let pose = rigid(&RigidJson {
            qvec: [0.0, 0.0, h, h],
            tvec: [0.0; 3],
        });
        let x = pose.rotation * Vec3F64::new(1.0, 0.0, 0.0);
        assert!((x - Vec3F64::new(0.0, 1.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn the_aria_slam_cameras_are_about_75_degrees_apart() {
        // R_11_5cp pinhole calibration.
        let cam0 = rigid(&RigidJson {
            qvec: [
                0.7871966367212387,
                0.08407781810505843,
                -0.05361927390966591,
                0.6085863530369895,
            ],
            tvec: [0.0168, -0.1089, 0.0761],
        });
        let cam1 = rigid(&RigidJson {
            qvec: [
                0.9969920181758002,
                0.012669053334945501,
                -0.07641622358987671,
                0.0026403699746858265,
            ],
            tvec: [-0.0021, -0.0002, -0.0064],
        });
        let relative = cam0.rotation.transpose() * cam1.rotation;
        let angle = angle_deg(relative);
        assert!((65.0..85.0).contains(&angle), "cam0-cam1 angle {angle}");
    }

    #[test]
    fn rectification_inverts_the_factory_model() {
        // raw = R * real + b, so R⁻¹ (raw − b) recovers real.
        let r = Mat3F64::from_cols(
            Vec3F64::new(1.0025, -0.0024, -0.0041),
            Vec3F64::new(0.0022, 0.9991, 0.0034),
            Vec3F64::new(0.0001, -0.0031, 0.9997),
        );
        let b = Vec3F64::new(0.0038, 0.0021, -0.0014);
        let real = Vec3F64::new(0.3, -1.2, 9.7);
        let raw = r * real + b;
        let rect = SensorRectification {
            rectification_inv: r.inverse(),
            bias: b,
        };
        let got = Vec3F64::from_array(rect.apply([raw.x, raw.y, raw.z]));
        assert!((got - real).length() < 1e-12);
    }
}
