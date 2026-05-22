//! `TrackingResult` and `TrackingStatus` Python types.

use kornia_3d::pose::Pose3d;
use kornia_slam::system::{TrackingResult, TrackingStatus};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Status of one `process()` call. Mirrors `kornia_slam::TrackingStatus`.
#[pyclass(
    name = "TrackingStatus",
    eq,
    eq_int,
    module = "kornia_slam._kornia_slam"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyTrackingStatus {
    Tracked,
    Skipped,
    KeyframeAccepted,
}

impl From<TrackingStatus> for PyTrackingStatus {
    fn from(s: TrackingStatus) -> Self {
        match s {
            TrackingStatus::Tracked => PyTrackingStatus::Tracked,
            TrackingStatus::Skipped => PyTrackingStatus::Skipped,
            TrackingStatus::KeyframeAccepted => PyTrackingStatus::KeyframeAccepted,
        }
    }
}

#[pymethods]
impl PyTrackingStatus {
    fn __repr__(&self) -> String {
        match self {
            PyTrackingStatus::Tracked => "TrackingStatus.Tracked".into(),
            PyTrackingStatus::Skipped => "TrackingStatus.Skipped".into(),
            PyTrackingStatus::KeyframeAccepted => "TrackingStatus.KeyframeAccepted".into(),
        }
    }
}

/// Result of one `slam.process(image)` call.
///
/// Carries the post-frame world-to-camera pose, the tracking status, and the
/// frame index assigned internally by `OrbSlam`.
#[pyclass(name = "TrackingResult", frozen, module = "kornia_slam._kornia_slam")]
pub struct PyTrackingResult {
    pose: [[f64; 4]; 4],
    #[pyo3(get)]
    status: PyTrackingStatus,
    #[pyo3(get)]
    frame_idx: usize,
}

impl PyTrackingResult {
    pub fn new(result: &TrackingResult, frame_idx: usize) -> Self {
        Self {
            pose: pose_to_matrix4x4(&result.pose_world_to_cam),
            status: result.status.into(),
            frame_idx,
        }
    }
}

#[pymethods]
impl PyTrackingResult {
    /// 4x4 world-to-camera pose as `np.ndarray[float64, (4, 4)]`.
    #[getter]
    fn pose<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self.pose.iter().flatten().copied().collect();
        Array2::from_shape_vec((4, 4), flat)
            .unwrap()
            .into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackingResult(frame_idx={}, status={})",
            self.frame_idx,
            match self.status {
                PyTrackingStatus::Tracked => "Tracked",
                PyTrackingStatus::Skipped => "Skipped",
                PyTrackingStatus::KeyframeAccepted => "KeyframeAccepted",
            }
        )
    }
}

pub fn pose_to_matrix4x4(pose: &Pose3d) -> [[f64; 4]; 4] {
    let r = pose.rotation;
    let t = pose.translation;
    [
        [r.x_axis.x, r.y_axis.x, r.z_axis.x, t.x],
        [r.x_axis.y, r.y_axis.y, r.z_axis.y, t.y],
        [r.x_axis.z, r.y_axis.z, r.z_axis.z, t.z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
