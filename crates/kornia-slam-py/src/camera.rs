//! Python wrapper around `kornia_3d::camera::PinholeCamera`.
//!
//! Exposes the pinhole intrinsics + radial/tangential distortion as a
//! `#[pyclass]` so the Python user can build one cleanly and hand it to
//! `OrbSlam`. The underlying Rust `PinholeCamera` is plain-data (`fx, fy,
//! cx, cy, k1, k2, p1, p2`) — no width / height; image size is taken from the
//! NumPy array shape on each `process()` call.

use kornia_3d::camera::PinholeCamera;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Pinhole camera intrinsics with radial+tangential distortion.
///
/// Construct with `PinholeCamera(width, height, fx, fy, cx, cy, distortion=None)`.
/// `distortion`, if provided, must be a length-4 sequence `[k1, k2, p1, p2]`
/// matching the underlying `kornia_3d::camera::PinholeCamera` model. Length-5
/// inputs `[k1, k2, p1, p2, k3]` are accepted with k3 ignored (and a clear
/// `ValueError` raised if k3 is non-zero, so the user knows it's dropped).
#[pyclass(name = "PinholeCamera", frozen, module = "kornia_slam._kornia_slam")]
#[derive(Clone, Debug)]
pub struct PyPinholeCamera {
    pub(crate) inner: PinholeCamera,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

#[pymethods]
impl PyPinholeCamera {
    #[new]
    #[pyo3(signature = (width, height, fx, fy, cx, cy, distortion=None))]
    fn new(
        width: usize,
        height: usize,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        distortion: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be > 0"));
        }
        let (k1, k2, p1, p2) = parse_distortion(distortion)?;
        Ok(Self {
            inner: PinholeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            },
            width,
            height,
        })
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.height
    }

    #[getter]
    fn fx(&self) -> f64 {
        self.inner.fx
    }

    #[getter]
    fn fy(&self) -> f64 {
        self.inner.fy
    }

    #[getter]
    fn cx(&self) -> f64 {
        self.inner.cx
    }

    #[getter]
    fn cy(&self) -> f64 {
        self.inner.cy
    }

    #[getter]
    fn distortion(&self) -> [f64; 4] {
        [self.inner.k1, self.inner.k2, self.inner.p1, self.inner.p2]
    }

    /// Returns the 3x3 intrinsic matrix as `np.ndarray[float64, (3, 3)]`.
    fn intrinsic_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let m = [
            [self.inner.fx, 0.0, self.inner.cx],
            [0.0, self.inner.fy, self.inner.cy],
            [0.0, 0.0, 1.0],
        ];
        let flat: Vec<f64> = m.into_iter().flatten().collect();
        Array2::from_shape_vec((3, 3), flat)
            .unwrap()
            .into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "PinholeCamera(width={}, height={}, fx={}, fy={}, cx={}, cy={}, distortion=[{}, {}, {}, {}])",
            self.width,
            self.height,
            self.inner.fx,
            self.inner.fy,
            self.inner.cx,
            self.inner.cy,
            self.inner.k1,
            self.inner.k2,
            self.inner.p1,
            self.inner.p2,
        )
    }
}

fn parse_distortion(distortion: Option<&Bound<'_, PyList>>) -> PyResult<(f64, f64, f64, f64)> {
    let Some(list) = distortion else {
        return Ok((0.0, 0.0, 0.0, 0.0));
    };
    let n = list.len();
    if n != 4 && n != 5 {
        return Err(PyValueError::new_err(format!(
            "distortion must be None, length 4 [k1, k2, p1, p2], or length 5 [k1, k2, p1, p2, k3]; got length {n}"
        )));
    }
    let extract = |i: usize| -> PyResult<f64> {
        list.get_item(i)
            .and_then(|v| v.extract::<f64>())
            .map_err(|_| PyTypeError::new_err(format!("distortion[{i}] must be a float")))
    };
    let k1 = extract(0)?;
    let k2 = extract(1)?;
    let p1 = extract(2)?;
    let p2 = extract(3)?;
    if n == 5 {
        let k3 = extract(4)?;
        if k3 != 0.0 {
            return Err(PyValueError::new_err(
                "distortion[4] (k3) is not supported by the underlying PinholeCamera; pass length 4 or set k3 = 0.0",
            ));
        }
    }
    Ok((k1, k2, p1, p2))
}
