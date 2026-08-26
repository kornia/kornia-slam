//! Dataset readers for visual odometry benchmarks.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::stereo::{CameraCalib, StereoError, StereoRectifier};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
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

/// One IMU sample from `imu0/data.csv`.
#[derive(Debug, Clone)]
pub struct ImuSample {
    /// Timestamp in seconds.
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Angular velocity in body frame (rad/s).
    pub gyro: [f64; 3],
    /// Linear acceleration in body frame (m/s^2).
    pub accel: [f64; 3],
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

/// EuRoC camera calibration loaded from a cam's `sensor.yaml`.
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
    /// Image width in pixels (`resolution[0]`).
    pub width: usize,
    /// Image height in pixels (`resolution[1]`).
    pub height: usize,
    /// Sensor-to-body transform `T_BS` (row-major 4x4): maps a point in the
    /// camera frame to the body frame, `X_body = T_BS * X_cam`.
    pub t_bs: [f64; 16],
}

impl EurocCameraCalibration {
    /// Camera-to-body extrinsic `T_BS` as rotation + translation
    /// (`X_body = R * X_cam + t`). On EuRoC the body frame is `imu0`.
    pub fn body_from_camera(&self) -> (Mat3F64, Vec3F64) {
        decompose_t_bs(&self.t_bs)
    }

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

/// EuRoC adapter for kornia-3d's Bouguet stereo rectifier
/// ([`kornia_3d::stereo`]). EuRoC ships raw cam0/cam1 images with independent
/// intrinsics, distortion, and a body-frame `T_BS` extrinsic each, so we adapt
/// that pair into the generic [`CameraCalib`] + relative-pose inputs the
/// rectifier expects.
///
/// Builds a [`StereoRectifier`] from the left (`cam0`) and right (`cam1`)
/// calibrations, deriving the relative pose left → right from their `T_BS`
/// body-frame extrinsics.
pub fn rectifier_from_euroc(
    left: &EurocCameraCalibration,
    right: &EurocCameraCalibration,
) -> Result<StereoRectifier, StereoError> {
    // Relative pose left -> right: X_right = R * X_left + t.
    let (r_l, t_l) = decompose_t_bs(&left.t_bs);
    let (r_r, t_r) = decompose_t_bs(&right.t_bs);
    let r_rt = r_r.transpose();
    let r_rel = r_rt * r_l;
    let t_rel = r_rt * (t_l - t_r);
    StereoRectifier::from_calib(
        &camera_calib_from_euroc(left),
        &camera_calib_from_euroc(right),
        r_rel,
        t_rel,
    )
}

/// Generic [`CameraCalib`] from an EuRoC per-camera calibration.
fn camera_calib_from_euroc(cam: &EurocCameraCalibration) -> CameraCalib {
    CameraCalib {
        width: cam.width,
        height: cam.height,
        fx: cam.fx,
        fy: cam.fy,
        cx: cam.cx,
        cy: cam.cy,
        distortion: PolynomialDistortion {
            k1: cam.k1,
            k2: cam.k2,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: cam.p1,
            p2: cam.p2,
        },
    }
}

/// Splits a row-major 4x4 `T_BS` into rotation (3x3) and translation (3).
fn decompose_t_bs(m: &[f64; 16]) -> (Mat3F64, Vec3F64) {
    let r = Mat3F64::from_cols(
        Vec3F64::new(m[0], m[4], m[8]),
        Vec3F64::new(m[1], m[5], m[9]),
        Vec3F64::new(m[2], m[6], m[10]),
    );
    let t = Vec3F64::new(m[3], m[7], m[11]);
    (r, t)
}

#[derive(Debug, Deserialize)]
struct TbsBlock {
    data: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct EurocSensorYaml {
    camera_model: String,
    distortion_model: Option<String>,
    intrinsics: Vec<f64>,
    distortion_coefficients: Vec<f64>,
    #[serde(default)]
    resolution: Vec<usize>,
    #[serde(rename = "T_BS")]
    t_bs: Option<TbsBlock>,
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
    /// Ordered left-camera (`cam0`) samples.
    pub left_samples: Vec<DatasetSample>,
    /// Left-camera (`cam0`) calibration.
    pub left_calibration: EurocCameraCalibration,
    /// Ordered right-camera (`cam1`) samples; empty if the dataset is monocular.
    pub right_samples: Vec<DatasetSample>,
    /// Right-camera (`cam1`) calibration; `None` if the dataset is monocular.
    pub right_calibration: Option<EurocCameraCalibration>,
    /// Ground-truth poses (empty if GT file not present).
    #[allow(dead_code)]
    pub ground_truth: Vec<GroundTruthPose>,
    /// IMU samples
    pub imu_samples: Vec<ImuSample>,
}

impl EurocDataset {
    /// Opens the dataset from `<root>/mav0/cam0/data.csv`, also loading `cam1`
    /// when present.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let root = root.as_ref().to_path_buf();
        let left_calibration = Self::load_camera_calibration(&root, "cam0")?;
        let left_samples = Self::load_camera_samples(&root, "cam0")?;

        // The right camera (cam1) is present only for stereo datasets.
        let right_dir = root.join("mav0").join("cam1");
        let (right_calibration, right_samples) = if right_dir.exists() {
            (
                Some(Self::load_camera_calibration(&root, "cam1")?),
                Self::load_camera_samples(&root, "cam1")?,
            )
        } else {
            (None, Vec::new())
        };

        let ground_truth = Self::load_ground_truth(&root);
        let imu_samples = Self::load_imu_samples(&root)?;

        Ok(Self {
            root,
            left_samples,
            left_calibration,
            right_samples,
            right_calibration,
            ground_truth,
            imu_samples,
        })
    }

    /// Loads the IMU samples (`imu0`); empty when the dataset ships none.
    fn load_imu_samples(root: &Path) -> Result<Vec<ImuSample>, DatasetError> {
        let csv = root.join("mav0").join("imu0").join("data.csv");
        if !csv.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&csv)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 7 {
                return Err(DatasetError::Parse(format!(
                    "invalid imu sample at line {}: expected 7 columns",
                    line_idx + 1
                )));
            }

            let ts_ns = cols[0].trim().parse::<u64>().map_err(|e| {
                DatasetError::Parse(format!("invalid timestamp at line {}: {e}", line_idx + 1))
            })?;
            let gyro = [
                cols[1].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid gyro x at line {}: {e}", line_idx + 1))
                })?,
                cols[2].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid gyro y at line {}: {e}", line_idx + 1))
                })?,
                cols[3].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid gyro z at line {}: {e}", line_idx + 1))
                })?,
            ];
            let accel = [
                cols[4].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid accel x at line {}: {e}", line_idx + 1))
                })?,
                cols[5].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid accel y at line {}: {e}", line_idx + 1))
                })?,
                cols[6].trim().parse::<f64>().map_err(|e| {
                    DatasetError::Parse(format!("invalid accel z at line {}: {e}", line_idx + 1))
                })?,
            ];

            samples.push(ImuSample {
                timestamp_sec: ts_ns as f64 * 1e-9,
                gyro,
                accel,
            });
        }

        Ok(samples)
    }

    /// Loads the ordered image samples for a given camera (`cam0` / `cam1`).
    fn load_camera_samples(root: &Path, cam: &str) -> Result<Vec<DatasetSample>, DatasetError> {
        let csv = root.join("mav0").join(cam).join("data.csv");
        let data_dir = root.join("mav0").join(cam).join("data");
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
        Ok(samples)
    }

    /// Returns ordered left-camera samples.
    pub fn samples(&self) -> &[DatasetSample] {
        &self.left_samples
    }

    /// Whether the dataset has a usable right camera (calibration + samples).
    pub fn is_stereo(&self) -> bool {
        self.right_calibration.is_some() && !self.right_samples.is_empty()
    }

    /// Whether the dataset has IMU samples
    pub fn has_imu(&self) -> bool {
        !self.imu_samples.is_empty()
    }

    /// Returns the left-camera model.
    pub fn camera(&self) -> PinholeCamera {
        self.left_calibration.to_pinhole_camera()
    }

    /// Returns parsed ground-truth poses (possibly empty).
    #[allow(dead_code)]
    pub fn ground_truth(&self) -> &[GroundTruthPose] {
        &self.ground_truth
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

    fn load_camera_calibration(
        root: &Path,
        cam: &str,
    ) -> Result<EurocCameraCalibration, DatasetError> {
        let sensor_yaml = root.join("mav0").join(cam).join("sensor.yaml");
        if !sensor_yaml.exists() {
            return Err(DatasetError::FileNotFound(sensor_yaml));
        }

        let file = File::open(&sensor_yaml)?;
        let sensor: EurocSensorYaml = serde_yaml::from_reader(file).map_err(|e| {
            DatasetError::Parse(format!(
                "invalid EuRoC {cam} calibration at {}: {e}",
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

        let (width, height) = match sensor.resolution.as_slice() {
            [w, h] => (*w, *h),
            _ => (0, 0),
        };

        let t_bs = match sensor.t_bs {
            Some(block) if block.data.len() == 16 => {
                let mut m = [0.0f64; 16];
                m.copy_from_slice(&block.data);
                m
            }
            // Identity fallback when T_BS is absent (monocular use).
            _ => {
                let mut m = [0.0f64; 16];
                m[0] = 1.0;
                m[5] = 1.0;
                m[10] = 1.0;
                m[15] = 1.0;
                m
            }
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
            width,
            height,
            t_bs,
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
    fn dataset_loads_left_calibration() {
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

    fn stereo_cam(t_bs: [f64; 16], cx: f64, cy: f64) -> EurocCameraCalibration {
        EurocCameraCalibration {
            fx: 458.0,
            fy: 457.0,
            cx,
            cy,
            k1: -0.28,
            k2: 0.07,
            p1: 0.0,
            p2: 0.0,
            width: 752,
            height: 480,
            t_bs,
        }
    }

    #[test]
    fn rectified_baseline_matches_mh01() {
        // Real MH_01_easy cam0/cam1 T_BS (row-major) and principal points.
        let t_bs0 = [
            0.0148655429818,
            -0.999880929698,
            0.00414029679422,
            -0.0216401454975,
            0.999557249008,
            0.0149672133247,
            0.025715529948,
            -0.064676986768,
            -0.0257744366974,
            0.00375618835797,
            0.999660727178,
            0.00981073058949,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let t_bs1 = [
            0.0125552670891,
            -0.999755099723,
            0.0182237714554,
            -0.0198435579556,
            0.999598781151,
            0.0130119051815,
            0.0251588363115,
            0.0453689425024,
            -0.0253898008918,
            0.0179005838253,
            0.999517347078,
            0.00786212447038,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let left = stereo_cam(t_bs0, 367.215, 248.375);
        let right = stereo_cam(t_bs1, 379.999, 255.238);
        let rect = rectifier_from_euroc(&left, &right).expect("valid MH_01 stereo calib");

        // EuRoC VI-sensor stereo baseline is ~0.11 m.
        assert!(
            (rect.baseline() - 0.11).abs() < 0.01,
            "baseline {} not ~0.11 m",
            rect.baseline()
        );
        assert!(rect.bf() > 0.0);
    }
}
