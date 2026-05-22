//! `OrbSlam` Python wrapper — owns the SLAM pipeline + ORB detector + frame
//! counter, exposes `process(image) -> TrackingResult` and a small set of map
//! accessors.

use kornia_3d::pose::Pose3d;
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::features::OrbDetector;
use kornia_slam::system::TrackingStatus;
use kornia_slam::Frame;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::camera::PyPinholeCamera;
use crate::config::PipelineConfig;
use crate::pipeline::Pipeline;
use crate::result::{pose_to_matrix4x4, PyTrackingResult};

/// High-level monocular ORB-SLAM driver.
///
/// Construct with a camera; feed frames via `process(image)`; query the live
/// map via `map_points()` and the lightweight accessors.
#[pyclass(name = "OrbSlam", module = "kornia_slam._kornia_slam")]
pub struct PyOrbSlam {
    pipeline: Pipeline,
    detector: OrbDetector,
    next_idx: usize,
    last_pose: Pose3d,
    last_keyframe_idx: Option<usize>,
}

#[pymethods]
impl PyOrbSlam {
    #[new]
    #[pyo3(signature = (camera, n_keypoints=1000, enable_local_ba=true))]
    fn new(camera: PyPinholeCamera, n_keypoints: usize, enable_local_ba: bool) -> PyResult<Self> {
        if n_keypoints == 0 {
            return Err(PyValueError::new_err("n_keypoints must be > 0"));
        }
        let detector = OrbDetector {
            n_keypoints,
            ..Default::default()
        };
        let config = PipelineConfig {
            enable_local_ba,
            ..PipelineConfig::default()
        };
        let pipeline = Pipeline::new(camera.inner, config);
        Ok(Self {
            pipeline,
            detector,
            next_idx: 0,
            last_pose: Pose3d::IDENTITY,
            last_keyframe_idx: None,
        })
    }

    /// Process one image and return a `TrackingResult`.
    ///
    /// Accepts a NumPy array of shape `(H, W)` (gray) or `(H, W, 3)` (RGB),
    /// dtype `uint8`. Releases the GIL while the underlying SLAM work runs.
    fn process(&mut self, py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<PyTrackingResult> {
        // Build owned gray buffer + per-keypoint color buffer up front, before
        // releasing the GIL (numpy access requires the GIL).
        let (gray_bytes, image_size, color_source) = extract_image(image)?;

        let allocator = CpuAllocator;
        let gray_image: Image<u8, 1, CpuAllocator> = Image::new(image_size, gray_bytes, allocator)
            .map_err(|e| PyValueError::new_err(format!("failed to wrap gray image: {e}")))?;

        let frame_idx = self.next_idx;

        // GIL-released work: detect features, build Frame, run pipeline.
        let (result_pose, result_status) = py.allow_threads(|| {
            let features = match self.detector.detect_and_extract_u8(&gray_image) {
                Ok(f) => f,
                Err(_) => {
                    // Detector failed (extremely rare — typically only on
                    // pathological inputs). Surface as Skipped, keep pose.
                    return (self.last_pose, TrackingStatus::Skipped);
                }
            };

            let image_bytes = gray_image.as_slice();
            let keypoint_colors: Vec<[u8; 3]> = features
                .keypoints_xy
                .iter()
                .map(|kp| {
                    let x = (kp[0] as usize).min(image_size.width.saturating_sub(1));
                    let y = (kp[1] as usize).min(image_size.height.saturating_sub(1));
                    match &color_source {
                        ColorSource::Gray => {
                            let g = image_bytes[y * image_size.width + x];
                            [g, g, g]
                        }
                        ColorSource::Rgb(rgb) => {
                            let base = (y * image_size.width + x) * 3;
                            [rgb[base], rgb[base + 1], rgb[base + 2]]
                        }
                    }
                })
                .collect();

            let frame = Frame {
                idx: frame_idx,
                features,
                pose_world_to_cam: Pose3d::IDENTITY,
                image_size,
                keypoint_colors,
            };

            let tracking_result = self.pipeline.process_frame(frame);
            (tracking_result.pose_world_to_cam, tracking_result.status)
        });

        let result = kornia_slam::system::TrackingResult {
            pose_world_to_cam: result_pose,
            status: result_status,
        };

        self.last_pose = result_pose;
        self.last_keyframe_idx = self.pipeline.current_keyframe_idx();
        self.next_idx += 1;

        Ok(PyTrackingResult::new(&result, frame_idx))
    }

    /// Returns active (non-culled) map point positions in world frame as
    /// `np.ndarray[float64, (N, 3)]`. Always allocates a fresh array; never
    /// borrows the underlying Rust storage.
    fn map_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let active: Vec<f64> = self
            .pipeline
            .map_points()
            .iter()
            .filter(|mp| !mp.culled)
            .flat_map(|mp| [mp.position.x, mp.position.y, mp.position.z])
            .collect();
        let n = active.len() / 3;
        let arr = Array2::from_shape_vec((n, 3), active).unwrap();
        arr.into_pyarray_bound(py)
    }

    /// 4x4 world-to-camera pose of the most recently processed frame, as
    /// `np.ndarray[float64, (4, 4)]`. Identity before any frame has been
    /// processed.
    #[getter]
    fn current_pose<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let m = pose_to_matrix4x4(&self.last_pose);
        let flat: Vec<f64> = m.iter().flatten().copied().collect();
        let arr = Array2::from_shape_vec((4, 4), flat).unwrap();
        arr.into_pyarray_bound(py)
    }

    /// Number of active (non-culled) map points.
    #[getter]
    fn num_active_map_points(&self) -> usize {
        self.pipeline.num_active_map_points()
    }

    /// Frame index of the current reference keyframe, or `None` while still in
    /// bootstrap.
    #[getter]
    fn current_keyframe_idx(&self) -> Option<usize> {
        self.last_keyframe_idx
    }
}

enum ColorSource {
    Gray,
    Rgb(Vec<u8>),
}

fn extract_image(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, ImageSize, ColorSource)> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<u8>>() {
        let view = arr.as_array();
        let shape = view.shape();
        let (h, w) = (shape[0], shape[1]);
        if h == 0 || w == 0 {
            return Err(PyValueError::new_err("image has zero-sized dimension"));
        }
        let slice = arr
            .as_slice()
            .map_err(|_| PyValueError::new_err("image must be C-contiguous (row-major)"))?;
        let bytes = slice.to_vec();
        let size = ImageSize {
            width: w,
            height: h,
        };
        return Ok((bytes, size, ColorSource::Gray));
    }

    if let Ok(arr) = obj.extract::<PyReadonlyArray3<u8>>() {
        let view = arr.as_array();
        let shape = view.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        if c != 3 {
            return Err(PyValueError::new_err(format!(
                "RGB image must have 3 channels, got {c}"
            )));
        }
        if h == 0 || w == 0 {
            return Err(PyValueError::new_err("image has zero-sized dimension"));
        }
        let slice = arr
            .as_slice()
            .map_err(|_| PyValueError::new_err("image must be C-contiguous (row-major)"))?;
        let rgb = slice.to_vec();
        let mut gray = Vec::with_capacity(h * w);
        for px in rgb.chunks_exact(3) {
            let g = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            gray.push(g.round().clamp(0.0, 255.0) as u8);
        }
        let size = ImageSize {
            width: w,
            height: h,
        };
        return Ok((gray, size, ColorSource::Rgb(rgb)));
    }

    Err(PyValueError::new_err(
        "image must be a numpy uint8 array of shape (H, W) or (H, W, 3)",
    ))
}
