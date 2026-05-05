
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
        let timestamp = Self::load_times(&root);
        // load sequesnces
        // let samples = Self::load_image(&root, &timestamp)?;

        let samples = match Self::load_image(&root, &timestamp){
            Ok(s) => s,
            Err(DatasetError) => return Err(DatasetError::FileNotFound(root)),
        };


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

    fn load_times(root: &Path) -> Vec<f64>{
        let times_file = root.join("times.txt");
        
        let times = match fs::read_to_string(&times_file){
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        return times.lines().filter_map(|t| t.trim().parse::<f64>().ok()).collect();
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