
use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{DatasetError, DatasetSample};

use crate::config::PipelineConfig;

use kornia_3d::camera::PinholeCamera;

/// Kitti `cam0` calibration loaded from `calib.txt`.
#[derive(Debug, Clone, Copy)]
pub struct KittiCameraCalibration {
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

impl KittiCameraCalibration{

    /// Converts the parsed Kitti calibration into a `PinholeCamera`.
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


/// PNG images in `<root>/image_0/`.
#[derive(Debug, Clone)]
pub struct KittiDataset {
    /// Base directory of the extracted dataset.
    #[allow(dead_code)]
    pub root: std::path::PathBuf,
    /// Ordered camera samples.
    pub cam0_samples: Vec<DatasetSample>,
    /// Camera calibration for `cam0`.
    pub cam0_calibration: KittiCameraCalibration,
}

impl KittiDataset{
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DatasetError>{

        let root = root.as_ref().to_path_buf();
        
        let image_dir = root.join("image_0");

        if !image_dir.is_dir(){
            return Err(DatasetError::FileNotFound(image_dir));
        }


        // load the camera
        let camera = Self::load_camera_from_calib(&root)?;
        // load timestamps
        let timestamp = Self::load_times(&root)?;
        // load sequesnces
        let samples = Self::load_image(&root, &timestamp)?;

        Ok(Self{
            root,
            cam0_samples: samples,
            cam0_calibration: camera,
        })
    }

    pub fn pipeline_config()-> PipelineConfig{

        let mut cfg = PipelineConfig::default();
        // Two-view bootstrap: allow lower parallax / inlier counts.
        cfg.two_view_init.acceptance_config.min_matches = 60;
        cfg.two_view_init.acceptance_config.min_inliers = 18;
        cfg.two_view_init.acceptance_config.min_triangulated = 24;
        cfg.two_view_init.triangulation_config.min_parallax_deg = 0.5;
        cfg.two_view_init.triangulation_config.max_reprojection_error = 5.0;

        // Tracking/PnP: be more tolerant when motion or illumination changes.
        cfg.map_projection.match_config.nn_ratio = 0.8;
        cfg.map_projection.match_config.th_low = 60;
        cfg.map_projection.projection.search_radius = 30.0;
        cfg.map_projection.projection.max_hamming = 64;
        cfg.map_projection.local_projection.search_radius = 42.0;
        cfg.map_projection.local_projection.max_hamming = 80;
        cfg.map_projection.pnp.final_reproj_threshold_px = 5.0;
        cfg.map_projection.pnp.min_inliers = 15;

        // Insert keyframes earlier so tracking remains anchored.
        cfg.keyframe_policy.min_frames_between = 1;
        cfg.keyframe_policy.max_frames_between = 5;
        cfg.keyframe_policy.ref_ratio = 0.9;

        cfg



    }

    fn load_camera_from_calib(root: &Path) -> Result<KittiCameraCalibration, DatasetError>{
        let calib_path = root.join("calib.txt");
        if !calib_path.exists(){
            return Err(DatasetError::FileNotFound(calib_path));
        }

        let calib_file = fs::read_to_string(&calib_path)?;

        let p0 = calib_file.lines().find_map(|line|{
            let line = line.trim();
            let (key, value) = line.split_once(":")?;
            if key.trim() != "P0"{
                return None;
            }
            let values: Vec<f64> = value
            .split_whitespace()
            .filter_map(|v| v.parse::<f64>().ok())
            .collect();
            if values.len() != 12 {
                return None;
            }
            Some(values)
        })
        .ok_or_else(||{
            DatasetError::Parse(format!(
                "Missing or Malformed P0 in {}",
                calib_path.display()
            ))
        })?;


        let fx = p0[0];
        let fy = p0[5];
        let cx = p0[2];
        let cy = p0[6];



        Ok(KittiCameraCalibration{
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        })
    }

    fn load_times(root: &Path) -> Result<Vec<f64>, DatasetError>{
        let times_file = root.join("times.txt");
        if !times_file.exists() {
            return Err(DatasetError::FileNotFound(times_file));
        }
        let times = fs::read_to_string(&times_file)?;
        let mut parsed_times = Vec::new();
        for (line_idx, line) in times.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let timestamp = trimmed.parse::<f64>().map_err(|_| {
                DatasetError::Parse(format!(
                    "Malformed timestamp at line {} in {}",
                    line_idx + 1,
                    times_file.display()
                ))
            })?;
            parsed_times.push(timestamp);
        }
        Ok(parsed_times)
    }

    fn load_image(root: &Path, timestamp: &[f64]) -> Result<Vec<DatasetSample>, DatasetError>{

        let mut paths: Vec<PathBuf> = Vec::new();
        let image_dir = root.join("image_0");
        for entry in fs::read_dir(&image_dir)?{
            let entry = entry?;
            let path = entry.path();
            if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("png")) 
            {
                paths.push(path);
            }
        }

        paths.sort();

        let mut samples = Vec::with_capacity(paths.len());
        for (i, image_path) in paths.into_iter().enumerate(){
            let timestamp_sec = timestamp.get(i).copied().unwrap_or(i as f64);
            samples.push(DatasetSample{
                timestamp_sec,
                image_path
            });
        }
        Ok(samples)

    }

    /// Returns ordered cam0 samples.
    pub fn samples(&self) -> &[DatasetSample] {
        &self.cam0_samples
    }

    /// Returns the `cam0` camera model.
    pub fn camera(&self) -> PinholeCamera {
        self.cam0_calibration.to_pinhole_camera()
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

    fn write_minimal_kitti_tree(
        root: &Path,
        include_calib: bool,
        calib_contents: Option<&str>,
        include_times: bool,
        times_contents: Option<&str>,
        include_images: bool,
    ) {
        if include_images {
            let image_dir = root.join("image_0");
            fs::create_dir_all(&image_dir).unwrap();
            fs::write(image_dir.join("000000.png"), []).unwrap();
            fs::write(image_dir.join("000001.png"), []).unwrap();
        }

        if include_calib {
            let contents = calib_contents.unwrap_or(
                "P0: 718.856 0 607.1928 0 0 718.856 185.2157 0 0 0 1 0\n",
            );
            fs::write(root.join("calib.txt"), contents).unwrap();
        }

        if include_times {
            let contents = times_contents.unwrap_or("0.0\n0.1\n");
            fs::write(root.join("times.txt"), contents).unwrap();
        }
    }

    #[test]
    fn dataset_open_succeeds_with_minimal_tree() {
        let dir = TestDir::new("kitti-ok");
        write_minimal_kitti_tree(dir.path(), true, None, true, None, true);

        let dataset = KittiDataset::open(dir.path()).unwrap();
        let camera = dataset.camera();
        let samples = dataset.samples();

        assert_eq!(camera.fx, 718.856);
        assert_eq!(camera.fy, 718.856);
        assert_eq!(camera.cx, 607.1928);
        assert_eq!(camera.cy, 185.2157);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].timestamp_sec, 0.0);
        assert_eq!(samples[1].timestamp_sec, 0.1);
        assert_eq!(
            samples[0].image_path,
            dir.path().join("image_0").join("000000.png")
        );
    }

    #[test]
    fn dataset_open_fails_when_calib_txt_is_missing() {
        let dir = TestDir::new("kitti-calib-missing");
        write_minimal_kitti_tree(dir.path(), false, None, true, None, true);

        let err = KittiDataset::open(dir.path()).unwrap_err();
        match err {
            DatasetError::FileNotFound(path) => {
                assert_eq!(path, dir.path().join("calib.txt"));
            }
            other => panic!("expected FileNotFound for calib.txt, got {other:?}"),
        }
    }

    #[test]
    fn dataset_open_fails_when_calib_txt_is_malformed() {
        let dir = TestDir::new("kitti-calib-bad");
        write_minimal_kitti_tree(
            dir.path(),
            true,
            Some("P0: 718.856 0 607.1928\n"),
            true,
            None,
            true,
        );

        let err = KittiDataset::open(dir.path()).unwrap_err();
        match err {
            DatasetError::Parse(message) => {
                assert!(message.contains("Missing or Malformed P0"));
            }
            other => panic!("expected Parse for malformed calib.txt, got {other:?}"),
        }
    }

    #[test]
    fn dataset_open_succeeds_when_times_txt_is_missing_and_falls_back_to_indices() {
        let dir = TestDir::new("kitti-times-missing");
        write_minimal_kitti_tree(dir.path(), true, None, false, None, true);

        let dataset = KittiDataset::open(dir.path()).unwrap();
        let samples = dataset.samples();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].timestamp_sec, 0.0);
        assert_eq!(samples[1].timestamp_sec, 1.0);
    }

    #[test]
    fn dataset_open_succeeds_with_malformed_times_txt_and_uses_available_timestamps() {
        let dir = TestDir::new("kitti-times-bad");
        write_minimal_kitti_tree(
            dir.path(),
            true,
            None,
            true,
            Some("not-a-number\n0.5\n"),
            true,
        );

        let dataset = KittiDataset::open(dir.path()).unwrap();
        let samples = dataset.samples();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].timestamp_sec, 0.5);
        assert_eq!(samples[1].timestamp_sec, 1.0);
    }

    #[test]
    fn dataset_open_fails_when_image_dir_is_missing() {
        let dir = TestDir::new("kitti-image-dir-missing");
        write_minimal_kitti_tree(dir.path(), true, None, true, None, false);

        let err = KittiDataset::open(dir.path()).unwrap_err();
        match err {
            DatasetError::FileNotFound(path) => {
                assert_eq!(path, dir.path().join("image_0"));
            }
            other => panic!("expected FileNotFound for image_0, got {other:?}"),
        }
    }
}