//! Dataset reader for the Hilti-Trimble SLAM Challenge 2026.
//!
//! The challenge extraction tools produce a EuRoC-like directory layout:
//!
//! ```text
//! <sequence>/
//!   cam0/data.csv
//!   cam0/data/<timestamp>.png
//!   cam1/data.csv          # optional
//!   cam1/data/<timestamp>.png
//!   imu0/data.csv          # optional
//! ```
//!
//! Camera calibration is loaded from a separate Kalibr YAML file.

use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuMeasurement;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use kornia_3d::camera::{FisheyeCamera, PinholeCamera};

use super::euroc::{DatasetError, DatasetSample, GroundTruthPose};

/// Kannala-Brandt fisheye calibration parsed from Kalibr YAML.
#[derive(Debug, Clone, Copy)]
pub struct KannalaBrandtCalibration {
    /// Focal length in x.
    pub fx: f64,
    /// Focal length in y.
    pub fy: f64,
    /// Principal point x.
    pub cx: f64,
    /// Principal point y.
    pub cy: f64,
    /// First KB distortion coefficient.
    pub k1: f64,
    /// Second KB distortion coefficient.
    pub k2: f64,
    /// Third KB distortion coefficient.
    pub k3: f64,
    /// Fourth KB distortion coefficient.
    pub k4: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl KannalaBrandtCalibration {
    /// Virtual pinhole camera that fisheye keypoints are undistorted into.
    ///
    /// Keeps the calibrated focal and principal point and drops distortion. ORB
    /// runs on the raw fisheye image; each keypoint is unprojected through the
    /// Kannala-Brandt model to a bearing and reprojected through this pinhole,
    /// so the existing pinhole geometry (PnP, triangulation, BA) stays valid.
    /// The focal only sets the pixel scale of the undistorted coordinates.
    pub fn to_undistorted_pinhole(self) -> PinholeCamera {
        PinholeCamera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Kannala-Brandt fisheye camera used to unproject raw keypoints to bearings.
    pub fn to_fisheye_camera(self) -> FisheyeCamera {
        FisheyeCamera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            k3: self.k3,
            k4: self.k4,
        }
    }
}

/// Reader for the Hilti-Trimble SLAM Challenge 2026 dataset.
#[derive(Debug, Clone)]
pub struct HiltiDataset {
    /// Base directory of the extracted sequence.
    #[allow(dead_code)]
    pub root: PathBuf,
    /// Ordered `cam0` image samples.
    pub cam0_samples: Vec<DatasetSample>,
    /// Ordered `cam1` image samples. Empty if `cam1` is absent.
    pub cam1_samples: Vec<DatasetSample>,
    /// Ordered IMU samples. Empty if `imu0` is absent.
    pub imu_samples: Vec<ImuMeasurement>,
    /// `cam0` Kannala-Brandt calibration.
    pub cam0_calibration: KannalaBrandtCalibration,
    /// `cam1` Kannala-Brandt calibration.
    pub cam1_calibration: KannalaBrandtCalibration,
    /// Transform from IMU to `cam0`.
    pub t_cam0_imu: [[f64; 4]; 4],
    /// Transform from IMU to `cam1`.
    pub t_cam1_imu: [[f64; 4]; 4],
    /// Ground-truth poses from `state_groundtruth_estimate0/data.csv`. Empty
    /// when the extraction ran without a `--gt-topic`.
    pub ground_truth: Vec<GroundTruthPose>,
}

#[derive(Debug, Deserialize)]
struct KalibrChain {
    cam0: KalibrCamera,
    cam1: KalibrCamera,
}

#[derive(Debug, Deserialize)]
struct KalibrCamera {
    camera_model: String,
    distortion_model: String,
    intrinsics: Vec<f64>,
    distortion_coeffs: Vec<f64>,
    resolution: Vec<u32>,
    #[serde(rename = "T_cam_imu")]
    t_cam_imu: Vec<Vec<f64>>,
}

impl HiltiDataset {
    /// Opens an extracted Hilti sequence and its Kalibr camera calibration.
    pub fn open(
        data_root: impl AsRef<Path>,
        calibration: impl AsRef<Path>,
    ) -> Result<Self, DatasetError> {
        let root = data_root.as_ref().to_path_buf();
        let calibration = calibration.as_ref().to_path_buf();

        let kalibr = read_kalibr_chain(&calibration)?;
        let cam0_calibration = kalibr.cam0.to_calibration("cam0", &calibration)?;
        let cam1_calibration = kalibr.cam1.to_calibration("cam1", &calibration)?;
        let t_cam0_imu = kalibr.cam0.t_cam_imu_matrix("cam0", &calibration)?;
        let t_cam1_imu = kalibr.cam1.t_cam_imu_matrix("cam1", &calibration)?;

        let cam0_samples = read_image_samples(&root, "cam0")?;
        let cam1_samples = read_optional_image_samples(&root, "cam1")?;
        let imu_samples = read_optional_imu_samples(&root)?;
        let ground_truth = read_optional_ground_truth(&root)?;

        Ok(Self {
            root,
            cam0_samples,
            cam1_samples,
            imu_samples,
            cam0_calibration,
            cam1_calibration,
            t_cam0_imu,
            t_cam1_imu,
            ground_truth,
        })
    }

    /// Returns ordered `cam0` samples.
    pub fn samples(&self) -> &[DatasetSample] {
        &self.cam0_samples
    }

    /// Returns the ground-truth poses (empty when none were extracted).
    pub fn ground_truth(&self) -> &[GroundTruthPose] {
        &self.ground_truth
    }

    /// True when a usable `cam1` stream is present.
    pub fn is_stereo(&self) -> bool {
        !self.cam1_samples.is_empty()
    }
}

impl KalibrCamera {
    fn to_calibration(
        &self,
        camera_name: &str,
        path: &Path,
    ) -> Result<KannalaBrandtCalibration, DatasetError> {
        let path_display = path.display();
        if self.camera_model != "pinhole" {
            let camera_model = &self.camera_model;
            return Err(DatasetError::Parse(format!(
                "unsupported {camera_name} camera_model '{camera_model}' at {path_display}"
            )));
        }

        if self.distortion_model != "equidistant" {
            let distortion_model = &self.distortion_model;
            return Err(DatasetError::Parse(format!(
                "unsupported {camera_name} distortion_model '{distortion_model}' at {path_display}"
            )));
        }

        let [fx, fy, cx, cy] = self.intrinsics.as_slice() else {
            return Err(DatasetError::Parse(format!(
                "expected 4 {camera_name} intrinsics in {path_display}"
            )));
        };
        let [k1, k2, k3, k4] = self.distortion_coeffs.as_slice() else {
            return Err(DatasetError::Parse(format!(
                "expected 4 {camera_name} distortion_coeffs in {path_display}"
            )));
        };
        let [width, height] = self.resolution.as_slice() else {
            return Err(DatasetError::Parse(format!(
                "expected 2 {camera_name} resolution values in {path_display}"
            )));
        };

        Ok(KannalaBrandtCalibration {
            fx: *fx,
            fy: *fy,
            cx: *cx,
            cy: *cy,
            k1: *k1,
            k2: *k2,
            k3: *k3,
            k4: *k4,
            width: *width,
            height: *height,
        })
    }

    fn t_cam_imu_matrix(
        &self,
        camera_name: &str,
        path: &Path,
    ) -> Result<[[f64; 4]; 4], DatasetError> {
        let path_display = path.display();
        if self.t_cam_imu.len() != 4 {
            return Err(DatasetError::Parse(format!(
                "expected 4 rows in {camera_name} T_cam_imu at {path_display}"
            )));
        }

        let mut matrix = [[0.0; 4]; 4];
        for (row_idx, row) in self.t_cam_imu.iter().enumerate() {
            if row.len() != 4 {
                return Err(DatasetError::Parse(format!(
                    "expected 4 columns in {camera_name} T_cam_imu row {row_idx} at {path_display}"
                )));
            }
            for (col_idx, value) in row.iter().enumerate() {
                matrix[row_idx][col_idx] = *value;
            }
        }

        Ok(matrix)
    }
}

fn read_kalibr_chain(path: &Path) -> Result<KalibrChain, DatasetError> {
    if !path.exists() {
        return Err(DatasetError::FileNotFound(path.to_path_buf()));
    }

    let path_display = path.display();
    // Hilti ships the Kalibr chain with an OpenCV FileStorage header
    // (`%YAML:1.0`), which is not a valid YAML directive and trips serde_yaml.
    // Drop any leading `%YAML…` line before parsing.
    let raw = std::fs::read_to_string(path)?;
    let cleaned: String = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("%YAML"))
        .collect::<Vec<_>>()
        .join("\n");
    serde_yaml::from_str(&cleaned).map_err(|e| {
        DatasetError::Parse(format!("invalid Kalibr calibration at {path_display}: {e}"))
    })
}

fn read_image_samples(root: &Path, camera_name: &str) -> Result<Vec<DatasetSample>, DatasetError> {
    let camera_dir = root.join(camera_name);
    let csv = camera_dir.join("data.csv");
    let data_dir = camera_dir.join("data");

    if !csv.exists() {
        return Err(DatasetError::FileNotFound(csv));
    }
    if !data_dir.exists() {
        return Err(DatasetError::FileNotFound(data_dir));
    }

    parse_image_csv(&csv, &data_dir)
}

fn read_optional_image_samples(
    root: &Path,
    camera_name: &str,
) -> Result<Vec<DatasetSample>, DatasetError> {
    let camera_dir = root.join(camera_name);
    let csv = camera_dir.join("data.csv");
    let data_dir = camera_dir.join("data");

    if !camera_dir.exists() && !csv.exists() && !data_dir.exists() {
        return Ok(Vec::new());
    }
    if !csv.exists() {
        return Ok(Vec::new());
    }
    if !data_dir.exists() {
        return Ok(Vec::new());
    }

    parse_image_csv(&csv, &data_dir)
}

fn parse_image_csv(csv: &Path, data_dir: &Path) -> Result<Vec<DatasetSample>, DatasetError> {
    let file = File::open(csv)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    let csv_path = csv.display();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let line_number = line_idx + 1;
        let mut columns = line.split(',');
        let timestamp = columns.next().ok_or_else(|| {
            DatasetError::Parse(format!("missing timestamp at {csv_path}:{line_number}"))
        })?;
        let filename = columns.next().ok_or_else(|| {
            DatasetError::Parse(format!("missing filename at {csv_path}:{line_number}"))
        })?;
        let timestamp_ns = timestamp.trim().parse::<u64>().map_err(|e| {
            DatasetError::Parse(format!(
                "invalid timestamp at {csv_path}:{line_number}: {e}"
            ))
        })?;

        samples.push(DatasetSample {
            timestamp_sec: timestamp_ns as f64 * 1e-9,
            image_path: data_dir.join(filename.trim()),
        });
    }

    samples.sort_by(|a, b| a.timestamp_sec.total_cmp(&b.timestamp_sec));
    Ok(samples)
}

fn read_optional_imu_samples(root: &Path) -> Result<Vec<ImuMeasurement>, DatasetError> {
    let imu_dir = root.join("imu0");
    let csv = imu_dir.join("data.csv");

    if !imu_dir.exists() || !csv.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(&csv)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    let csv_path = csv.display();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let line_number = line_idx + 1;
        let columns: Vec<_> = line.split(',').collect();
        if columns.len() < 7 {
            return Err(DatasetError::Parse(format!(
                "expected 7 IMU columns at {csv_path}:{line_number}"
            )));
        }

        let parse_f64 = |column_idx: usize| {
            let column_number = column_idx + 1;
            columns[column_idx].trim().parse::<f64>().map_err(|e| {
                DatasetError::Parse(format!(
                    "invalid IMU value at {csv_path}:{line_number} column {column_number}: {e}"
                ))
            })
        };
        let timestamp_ns = columns[0].trim().parse::<u64>().map_err(|e| {
            DatasetError::Parse(format!(
                "invalid IMU timestamp at {csv_path}:{line_number}: {e}"
            ))
        })?;

        samples.push(ImuMeasurement {
            timestamp: timestamp_ns as f64 * 1e-9,
            gyro: Vec3F64::new(parse_f64(1)?, parse_f64(2)?, parse_f64(3)?),
            accel: Vec3F64::new(parse_f64(4)?, parse_f64(5)?, parse_f64(6)?),
        });
    }

    samples.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
    Ok(samples)
}

/// Reads `state_groundtruth_estimate0/data.csv` (EuRoC ground-truth format)
/// when present. Returns an empty Vec when the directory or file is absent.
fn read_optional_ground_truth(root: &Path) -> Result<Vec<GroundTruthPose>, DatasetError> {
    let csv = root.join("state_groundtruth_estimate0").join("data.csv");
    if !csv.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(&csv)?;
    let reader = BufReader::new(file);
    let mut poses = Vec::new();
    let csv_path = csv.display();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let line_number = line_idx + 1;
        // Columns: timestamp_ns, px, py, pz, qw, qx, qy, qz, ...
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 8 {
            return Err(DatasetError::Parse(format!(
                "expected at least 8 ground-truth columns at {csv_path}:{line_number}"
            )));
        }
        let parse = |idx: usize| {
            cols[idx].trim().parse::<f64>().map_err(|e| {
                DatasetError::Parse(format!(
                    "invalid ground-truth value at {csv_path}:{line_number} column {}: {e}",
                    idx + 1
                ))
            })
        };
        let ts_ns = cols[0].trim().parse::<u64>().map_err(|e| {
            DatasetError::Parse(format!(
                "invalid ground-truth timestamp at {csv_path}:{line_number}: {e}"
            ))
        })?;

        poses.push(GroundTruthPose {
            timestamp_sec: ts_ns as f64 * 1e-9,
            tx: parse(1)?,
            ty: parse(2)?,
            tz: parse(3)?,
            qw: parse(4)?,
            qx: parse(5)?,
            qy: parse(6)?,
            qz: parse(7)?,
        });
    }

    poses.sort_by(|a, b| a.timestamp_sec.total_cmp(&b.timestamp_sec));
    Ok(poses)
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
            let pid = std::process::id();
            let path =
                std::env::temp_dir().join(format!("kornia-slam-hilti-{name}-{pid}-{unique}"));
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

    const KALIBR_YAML: &str = r#"
cam0:
  camera_model: pinhole
  distortion_model: equidistant
  intrinsics: [461.64, 459.72, 732.95, 720.54]
  distortion_coeffs: [0.0344, -0.0216, 0.0031, -0.0005]
  resolution: [1472, 1440]
  T_cam_imu:
    - [1.0, 0.0, 0.0, 0.01]
    - [0.0, 1.0, 0.0, 0.02]
    - [0.0, 0.0, 1.0, 0.03]
    - [0.0, 0.0, 0.0, 1.0]
cam1:
  camera_model: pinhole
  distortion_model: equidistant
  intrinsics: [462.0, 460.0, 731.0, 719.0]
  distortion_coeffs: [0.03, -0.02, 0.002, -0.0004]
  resolution: [1472, 1440]
  T_cam_imu:
    - [1.0, 0.0, 0.0, -0.10]
    - [0.0, 1.0, 0.0, 0.20]
    - [0.0, 0.0, 1.0, 0.30]
    - [0.0, 0.0, 0.0, 1.0]
"#;

    fn write_calibration(root: &Path) -> PathBuf {
        let calibration = root.join("kalibr_imucam_chain.yaml");
        fs::write(&calibration, KALIBR_YAML).unwrap();
        calibration
    }

    fn write_camera(root: &Path, camera_name: &str, csv: &str) {
        let camera_dir = root.join(camera_name);
        let data_dir = camera_dir.join("data");
        fs::create_dir_all(&data_dir).unwrap();
        fs::write(camera_dir.join("data.csv"), csv).unwrap();
        fs::write(data_dir.join("1000000000.png"), []).unwrap();
        fs::write(data_dir.join("2000000000.png"), []).unwrap();
    }

    fn write_imu(root: &Path, csv: &str) {
        let imu_dir = root.join("imu0");
        fs::create_dir_all(&imu_dir).unwrap();
        fs::write(imu_dir.join("data.csv"), csv).unwrap();
    }

    #[test]
    fn open_reads_cam0_samples_sorted_by_timestamp() {
        let dir = TestDir::new("cam0");
        let calibration = write_calibration(dir.path());
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n2000000000,2000000000.png\n1000000000,1000000000.png\n",
        );

        let dataset = HiltiDataset::open(dir.path(), calibration).unwrap();

        assert_eq!(dataset.cam0_samples.len(), 2);
        assert_eq!(dataset.cam0_samples[0].timestamp_sec, 1.0);
        assert_eq!(dataset.cam0_samples[1].timestamp_sec, 2.0);
        assert!(dataset.cam1_samples.is_empty());
        assert!(dataset.imu_samples.is_empty());
    }

    #[test]
    fn open_reads_optional_cam1_and_imu_samples() {
        let dir = TestDir::new("optional");
        let calibration = write_calibration(dir.path());
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );
        write_camera(
            dir.path(),
            "cam1",
            "#timestamp [ns],filename\n2000000000,2000000000.png\n",
        );
        write_imu(
            dir.path(),
            "#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n\
             2000000000,0.3,0.2,0.1,0.03,0.02,9.81\n\
             1000000000,0.1,0.2,0.3,0.01,0.02,9.80\n",
        );

        let dataset = HiltiDataset::open(dir.path(), calibration).unwrap();

        assert_eq!(dataset.cam1_samples.len(), 1);
        assert_eq!(dataset.cam1_samples[0].timestamp_sec, 2.0);
        assert_eq!(dataset.imu_samples.len(), 2);
        assert_eq!(dataset.imu_samples[0].timestamp, 1.0);
        assert_eq!(dataset.imu_samples[0].gyro, Vec3F64::new(0.1, 0.2, 0.3));
        assert_eq!(dataset.imu_samples[1].accel, Vec3F64::new(0.03, 0.02, 9.81));
    }

    #[test]
    fn open_parses_kalibr_calibration_and_extrinsics() {
        let dir = TestDir::new("calibration");
        let calibration = write_calibration(dir.path());
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );

        let dataset = HiltiDataset::open(dir.path(), calibration).unwrap();

        assert_eq!(dataset.cam0_calibration.fx, 461.64);
        assert_eq!(dataset.cam0_calibration.fy, 459.72);
        assert_eq!(dataset.cam0_calibration.cx, 732.95);
        assert_eq!(dataset.cam0_calibration.cy, 720.54);
        assert_eq!(dataset.cam0_calibration.k1, 0.0344);
        assert_eq!(dataset.cam0_calibration.k2, -0.0216);
        assert_eq!(dataset.cam0_calibration.k3, 0.0031);
        assert_eq!(dataset.cam0_calibration.k4, -0.0005);
        assert_eq!(dataset.cam0_calibration.width, 1472);
        assert_eq!(dataset.cam0_calibration.height, 1440);
        assert_eq!(dataset.cam1_calibration.fx, 462.0);
        assert_eq!(dataset.t_cam0_imu[0][3], 0.01);
        assert_eq!(dataset.t_cam1_imu[0][3], -0.10);
    }

    #[test]
    fn open_parses_opencv_filestorage_yaml_header() {
        // The real Hilti chain starts with an OpenCV `%YAML:1.0` header, which
        // is not valid YAML; the reader must strip it.
        let dir = TestDir::new("opencv-header");
        let calibration = dir.path().join("kalibr_imucam_chain.yaml");
        let with_header =
            format!("%YAML:1.0 # need to specify the file type at the top!\n{KALIBR_YAML}");
        fs::write(&calibration, with_header).unwrap();
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );

        let dataset = HiltiDataset::open(dir.path(), calibration).unwrap();
        assert_eq!(dataset.cam0_calibration.fx, 461.64);
    }

    #[test]
    fn open_fails_when_calibration_is_missing() {
        let dir = TestDir::new("missing-calibration");
        let calibration = dir.path().join("missing.yaml");
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );

        let err = HiltiDataset::open(dir.path(), &calibration).unwrap_err();

        match err {
            DatasetError::FileNotFound(path) => assert_eq!(path, calibration),
            other => panic!("expected FileNotFound, got {other:?}"),
        }
    }

    #[test]
    fn open_fails_when_cam0_csv_is_missing() {
        let dir = TestDir::new("missing-cam0");
        let calibration = write_calibration(dir.path());

        let err = HiltiDataset::open(dir.path(), calibration).unwrap_err();

        match err {
            DatasetError::FileNotFound(path) => {
                assert_eq!(path, dir.path().join("cam0").join("data.csv"));
            }
            other => panic!("expected FileNotFound, got {other:?}"),
        }
    }

    #[test]
    fn open_fails_on_non_equidistant_distortion() {
        let dir = TestDir::new("bad-distortion");
        let calibration = dir.path().join("kalibr_imucam_chain.yaml");
        fs::write(&calibration, KALIBR_YAML.replace("equidistant", "radtan")).unwrap();
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );

        let err = HiltiDataset::open(dir.path(), calibration).unwrap_err();

        match err {
            DatasetError::Parse(message) => assert!(message.contains("distortion_model")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn open_fails_on_malformed_optional_imu_csv() {
        let dir = TestDir::new("bad-imu");
        let calibration = write_calibration(dir.path());
        write_camera(
            dir.path(),
            "cam0",
            "#timestamp [ns],filename\n1000000000,1000000000.png\n",
        );
        write_imu(dir.path(), "#timestamp [ns],w_x\n1000000000,0.1\n");

        let err = HiltiDataset::open(dir.path(), calibration).unwrap_err();

        match err {
            DatasetError::Parse(message) => assert!(message.contains("expected 7 IMU columns")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }
}
