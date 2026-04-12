//! Dataset readers for visual odometry benchmarks.

use kornia_3d::camera::PinholeCamera;
use kornia_sensors::imu::{ImuCalib, ImuMeasurement};
use kornia_algebra::Vec3F64;
use serde::Deserialize;
use std::{fs::File, io::BufRead, io::BufReader, path::Path, path::PathBuf};

/// Error type used by dataset readers.
#[derive(thiserror::Error, Debug)]
pub enum DatasetError {
    /// Generic I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse failure with contextual message.
    #[error("parse error: {0}")]
    Parse(String),

    /// Referenced file does not exist.
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),
}

/// One dataset image sample.
#[derive(Debug, Clone)]
pub struct DatasetSample {
    /// Timestamp in seconds.
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Path to the image file.
    pub image_path: PathBuf,
}

/// One ground-truth pose from `state_groundtruth_estimate0/data.csv`.
#[derive(Debug, Clone, Copy)]
pub struct GroundTruthPose {
    /// Timestamp in seconds.
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Position x (meters).
    #[allow(dead_code)]
    pub tx: f64,
    /// Position y (meters).
    #[allow(dead_code)]
    pub ty: f64,
    /// Position z (meters).
    #[allow(dead_code)]
    pub tz: f64,
    /// Quaternion scalar part.
    #[allow(dead_code)]
    pub qw: f64,
    /// Quaternion x.
    #[allow(dead_code)]
    pub qx: f64,
    /// Quaternion y.
    #[allow(dead_code)]
    pub qy: f64,
    /// Quaternion z.
    #[allow(dead_code)]
    pub qz: f64,
}

/// One raw IMU measurement from `mav0/imu0/data.csv`.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    /// Timestamp in seconds.
    pub timestamp_sec: f64,
    /// Gyroscope [wx, wy, wz] in rad/s.
    pub gyro: [f64; 3],
    /// Accelerometer [ax, ay, az] in m/s².
    pub accel: [f64; 3],
}

impl ImuSample {
    /// Convert to a `kornia_sensors::imu::ImuMeasurement`.
    pub fn to_imu_measurement(self) -> ImuMeasurement {
        ImuMeasurement {
            timestamp: self.timestamp_sec,
            gyro: Vec3F64::new(self.gyro[0], self.gyro[1], self.gyro[2]),
            accel: Vec3F64::new(self.accel[0], self.accel[1], self.accel[2]),
        }
    }
}

/// EuRoC `cam0` calibration loaded from `sensor.yaml`.
#[derive(Debug, Clone, Copy)]
pub struct EurocCameraCalibration {
    /// Focal length in x.
    pub fx: f64,
    /// Focal length in y.
    pub fy: f64,
    /// Principal point x.
    pub cx: f64,
    /// Principal point y.
    pub cy: f64,
    /// Radial distortion coefficient.
    pub k1: f64,
    /// Radial distortion coefficient.
    pub k2: f64,
    /// Tangential distortion coefficient.
    pub p1: f64,
    /// Tangential distortion coefficient.
    pub p2: f64,
}

impl EurocCameraCalibration {
    /// Converts the parsed EuRoC calibration into a `PinholeCamera`.
    pub fn to_pinhole_camera(self) -> PinholeCamera {
        PinholeCamera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: self.p2,
        }
    }
}

#[derive(Debug, Deserialize)]
struct EurocSensorYaml {
    camera_model: String,
    distortion_model: Option<String>,
    intrinsics: Vec<f64>,
    distortion_coefficients: Vec<f64>,
}

/// IMU sensor.yaml from EuRoC `mav0/imu0/sensor.yaml`.
#[derive(Debug, Deserialize)]
struct EurocImuSensorYaml {
    gyroscope_noise_density: f64,
    accelerometer_noise_density: f64,
    gyroscope_random_walk: f64,
    accelerometer_random_walk: f64,
}

/// Reader for the EuRoC MAV dataset (ASL format).
///
/// Expects `<root>/mav0/cam0/data.csv` with nanosecond timestamps and
/// PNG images in `<root>/mav0/cam0/data/`.
#[derive(Debug, Clone)]
pub struct EurocDataset {
    /// Base directory of the extracted dataset.
    #[allow(dead_code)]
    pub root: std::path::PathBuf,
    /// Ordered camera samples.
    pub cam0_samples: Vec<DatasetSample>,
    /// Camera calibration for `cam0`.
    pub cam0_calibration: EurocCameraCalibration,
    /// Ground-truth poses (empty if GT file not present).
    #[allow(dead_code)]
    pub ground_truth: Vec<GroundTruthPose>,
    /// IMU samples from `imu0/data.csv` (empty if not present).
    pub imu0_samples: Vec<ImuSample>,
    /// IMU calibration from `imu0/sensor.yaml` (None if not present).
    pub imu0_calibration: Option<ImuCalib>,
}

impl EurocDataset {
    /// Opens the dataset from `<root>/mav0/cam0/data.csv`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let root = root.as_ref().to_path_buf();
        let csv = root.join("mav0").join("cam0").join("data.csv");
        let data_dir = root.join("mav0").join("cam0").join("data");
        let cam0_calibration = Self::load_cam0_calibration(&root)?;

        if !csv.exists() {
            return Err(DatasetError::FileNotFound(csv));
        }
        if !data_dir.exists() {
            return Err(DatasetError::FileNotFound(data_dir));
        }

        let file = File::open(&csv)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            // Skip header line (starts with '#').
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split(',');
            let ts_str = parts.next().ok_or_else(|| {
                DatasetError::Parse(format!("missing timestamp at line {}", line_idx + 1))
            })?;
            let file_str = parts.next().ok_or_else(|| {
                DatasetError::Parse(format!("missing filename at line {}", line_idx + 1))
            })?;

            let timestamp_ns = ts_str.trim().parse::<u64>().map_err(|e| {
                DatasetError::Parse(format!("invalid timestamp at line {}: {e}", line_idx + 1))
            })?;
            let timestamp_sec = timestamp_ns as f64 * 1e-9;

            samples.push(DatasetSample {
                timestamp_sec,
                image_path: data_dir.join(file_str.trim()),
            });
        }

        let ground_truth = Self::load_ground_truth(&root);
        let imu0_samples = Self::load_imu_data(&root);
        let imu0_calibration = Self::load_imu_calibration(&root);

        Ok(Self {
            root,
            cam0_samples: samples,
            cam0_calibration,
            ground_truth,
            imu0_samples,
            imu0_calibration,
        })
    }

    /// Returns ordered cam0 samples.
    pub fn samples(&self) -> &[DatasetSample] {
        &self.cam0_samples
    }

    /// Returns the `cam0` camera model.
    pub fn camera(&self) -> PinholeCamera {
        self.cam0_calibration.to_pinhole_camera()
    }

    /// Returns parsed ground-truth poses (possibly empty).
    #[allow(dead_code)]
    pub fn ground_truth(&self) -> &[GroundTruthPose] {
        &self.ground_truth
    }

    /// Returns IMU samples sorted by timestamp (possibly empty).
    pub fn imu_samples(&self) -> &[ImuSample] {
        &self.imu0_samples
    }

    /// Returns IMU calibration if `imu0/sensor.yaml` was found.
    pub fn imu_calib(&self) -> Option<ImuCalib> {
        self.imu0_calibration
    }

    /// Loads ground-truth poses from `mav0/state_groundtruth_estimate0/data.csv`.
    ///
    /// Returns an empty Vec if the file does not exist.
    fn load_ground_truth(root: &Path) -> Vec<GroundTruthPose> {
        let csv = root
            .join("mav0")
            .join("state_groundtruth_estimate0")
            .join("data.csv");
        let file = match File::open(&csv) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        let mut poses = Vec::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            // Columns: timestamp_ns, px, py, pz, qw, qx, qy, qz, ...
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 8 {
                continue;
            }
            let Ok(ts_ns) = cols[0].trim().parse::<u64>() else {
                continue;
            };
            let f = |i: usize| cols[i].trim().parse::<f64>();
            let (Ok(px), Ok(py), Ok(pz), Ok(qw), Ok(qx), Ok(qy), Ok(qz)) =
                (f(1), f(2), f(3), f(4), f(5), f(6), f(7))
            else {
                continue;
            };

            poses.push(GroundTruthPose {
                timestamp_sec: ts_ns as f64 * 1e-9,
                tx: px,
                ty: py,
                tz: pz,
                qw,
                qx,
                qy,
                qz,
            });
        }
        poses
    }

    /// Loads IMU measurements from `mav0/imu0/data.csv`.
    ///
    /// CSV format: `timestamp_ns, w_x, w_y, w_z, a_x, a_y, a_z`
    /// Returns an empty Vec if the file does not exist.
    fn load_imu_data(root: &Path) -> Vec<ImuSample> {
        let csv = root.join("mav0").join("imu0").join("data.csv");
        let file = match File::open(&csv) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 7 {
                continue;
            }
            let Ok(ts_ns) = cols[0].trim().parse::<u64>() else {
                continue;
            };
            let f = |i: usize| cols[i].trim().parse::<f64>();
            let (Ok(wx), Ok(wy), Ok(wz), Ok(ax), Ok(ay), Ok(az)) =
                (f(1), f(2), f(3), f(4), f(5), f(6))
            else {
                continue;
            };

            samples.push(ImuSample {
                timestamp_sec: ts_ns as f64 * 1e-9,
                gyro: [wx, wy, wz],
                accel: [ax, ay, az],
            });
        }
        samples
    }

    /// Loads IMU calibration from `mav0/imu0/sensor.yaml`.
    ///
    /// Returns `None` if the file does not exist or cannot be parsed.
    fn load_imu_calibration(root: &Path) -> Option<ImuCalib> {
        let sensor_yaml = root.join("mav0").join("imu0").join("sensor.yaml");
        let file = File::open(&sensor_yaml).ok()?;
        let sensor: EurocImuSensorYaml = serde_yaml::from_reader(file).ok()?;
        Some(ImuCalib {
            gyro_noise: sensor.gyroscope_noise_density,
            accel_noise: sensor.accelerometer_noise_density,
            gyro_bias_noise: sensor.gyroscope_random_walk,
            accel_bias_noise: sensor.accelerometer_random_walk,
        })
    }

    fn load_cam0_calibration(root: &Path) -> Result<EurocCameraCalibration, DatasetError> {
        let sensor_yaml = root.join("mav0").join("cam0").join("sensor.yaml");
        if !sensor_yaml.exists() {
            return Err(DatasetError::FileNotFound(sensor_yaml));
        }

        let file = File::open(&sensor_yaml)?;
        let sensor: EurocSensorYaml = serde_yaml::from_reader(file).map_err(|e| {
            DatasetError::Parse(format!(
                "invalid EuRoC cam0 calibration at {}: {e}",
                sensor_yaml.display()
            ))
        })?;

        if sensor.camera_model != "pinhole" {
            return Err(DatasetError::Parse(format!(
                "unsupported EuRoC camera_model '{}' at {}",
                sensor.camera_model,
                sensor_yaml.display()
            )));
        }

        if let Some(distortion_model) = sensor.distortion_model.as_deref()
            && distortion_model != "radial-tangential"
        {
            return Err(DatasetError::Parse(format!(
                "unsupported EuRoC distortion_model '{}' at {}",
                distortion_model,
                sensor_yaml.display()
            )));
        }

        let [fx, fy, cx, cy] = sensor.intrinsics.as_slice() else {
            return Err(DatasetError::Parse(format!(
                "expected 4 intrinsics in {}",
                sensor_yaml.display()
            )));
        };
        let [k1, k2, p1, p2] = sensor.distortion_coefficients.as_slice() else {
            return Err(DatasetError::Parse(format!(
                "expected 4 distortion coefficients in {}",
                sensor_yaml.display()
            )));
        };

        Ok(EurocCameraCalibration {
            fx: *fx,
            fy: *fy,
            cx: *cx,
            cy: *cy,
            k1: *k1,
            k2: *k2,
            p1: *p1,
            p2: *p2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "kornia-slam-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_minimal_euroc_tree(root: &Path, include_sensor_yaml: bool) {
        let cam0_dir = root.join("mav0").join("cam0");
        let data_dir = cam0_dir.join("data");
        fs::create_dir_all(&data_dir).unwrap();
        fs::write(
            cam0_dir.join("data.csv"),
            "#timestamp [ns],filename\n1403636579763555584,1403636579763555584.png\n",
        )
        .unwrap();
        fs::write(data_dir.join("1403636579763555584.png"), []).unwrap();

        if include_sensor_yaml {
            fs::write(
                cam0_dir.join("sensor.yaml"),
                "sensor_type: camera\ncamera_model: pinhole\nintrinsics: [458.654, 457.296, 367.215, 248.375]\ndistortion_model: radial-tangential\ndistortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]\n",
            )
            .unwrap();
        }
    }

    #[test]
    fn dataset_loads_cam0_calibration() {
        let dir = TestDir::new("euroc-calibration-ok");
        write_minimal_euroc_tree(dir.path(), true);

        let dataset = EurocDataset::open(dir.path()).unwrap();
        let camera = dataset.camera();

        assert_eq!(camera.fx, 458.654);
        assert_eq!(camera.fy, 457.296);
        assert_eq!(camera.cx, 367.215);
        assert_eq!(camera.cy, 248.375);
        assert_eq!(camera.k1, -0.28340811);
        assert_eq!(camera.k2, 0.07395907);
        assert_eq!(camera.p1, 0.00019359);
        assert_eq!(camera.p2, 1.76187114e-05);
    }

    #[test]
    fn dataset_open_fails_when_sensor_yaml_is_missing() {
        let dir = TestDir::new("euroc-calibration-missing");
        write_minimal_euroc_tree(dir.path(), false);

        let err = EurocDataset::open(dir.path()).unwrap_err();

        match err {
            DatasetError::FileNotFound(path) => {
                assert_eq!(
                    path,
                    dir.path().join("mav0").join("cam0").join("sensor.yaml")
                );
            }
            other => panic!("expected FileNotFound for sensor.yaml, got {other:?}"),
        }
    }

    #[test]
    fn dataset_open_fails_when_sensor_yaml_is_malformed() {
        let dir = TestDir::new("euroc-calibration-bad");
        write_minimal_euroc_tree(dir.path(), true);
        fs::write(
            dir.path()
                .join("mav0")
                .join("cam0")
                .join("sensor.yaml"),
            "camera_model: pinhole\nintrinsics: [458.654, 457.296, 367.215]\ndistortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]\n",
        )
        .unwrap();

        let err = EurocDataset::open(dir.path()).unwrap_err();

        match err {
            DatasetError::Parse(message) => {
                assert!(message.contains("expected 4 intrinsics"));
            }
            other => panic!("expected Parse for malformed sensor.yaml, got {other:?}"),
        }
    }
}
