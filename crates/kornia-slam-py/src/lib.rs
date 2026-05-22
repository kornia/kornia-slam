//! PyO3 high-level bindings for kornia-slam ORB-SLAM.
//!
//! The Python-facing surface is:
//!
//! - `PinholeCamera` — intrinsics container
//! - `OrbSlam`       — the SLAM driver with `process(image)` + map accessors
//! - `TrackingResult` / `TrackingStatus` — return value + status enum
//!
//! See `docs/superpowers/specs/2026-05-22-pyo3-orb-slam-bindings-design.md`.

// pyo3 0.22's #[pymethods] macro expansion emits `.into()` on `PyResult<T>` ->
// `PyResult<T>`, which clippy flags as useless_conversion on rustc 1.93+. Until
// pyo3 fixes this upstream, suppress it crate-wide.
#![allow(clippy::useless_conversion)]

mod camera;
mod config;
mod pipeline;
mod result;
mod slam;

use pyo3::prelude::*;

#[pymodule]
fn _kornia_slam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<camera::PyPinholeCamera>()?;
    m.add_class::<result::PyTrackingStatus>()?;
    m.add_class::<result::PyTrackingResult>()?;
    m.add_class::<slam::PyOrbSlam>()?;
    Ok(())
}
