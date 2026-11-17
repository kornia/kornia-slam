//! Calibration states of the offline solve, each block present only when its
//! knob is on: per camera and time slice an IMU-camera rotation offset and a
//! focal scale (visloc's drift states), the symmetric gyro correction and one
//! IMU-camera clock offset (VINS-Fusion's `td`). OpenVINS keeps one flag per
//! state (`StateOptions.h:47-56`); so does this layout.

use kornia_algebra::Vec3F64;
use kornia_sensors::imu::GyroIntrinsics;

use super::{CalibrationEstimate, OfflineCalibrationConfig};

/// Column of every calibration state in the border of the reduced system.
#[derive(Debug, Clone)]
pub(super) struct Layout {
    cameras: usize,
    slices: usize,
    rotation: Option<usize>,
    focal: Option<usize>,
    gyro: Option<usize>,
    clock: Option<usize>,
    pub(super) len: usize,
}

impl Layout {
    pub(super) fn new(config: &OfflineCalibrationConfig, cameras: usize) -> Self {
        let slices = config.time_slices.max(1);
        let mut len = 0;
        let mut take = |on: bool, columns: usize| {
            on.then(|| {
                let at = len;
                len += columns;
                at
            })
        };
        let rotation = take(config.camera_rotation.is_some(), cameras * slices * 3);
        let focal = take(config.camera_focal.is_some(), cameras * slices);
        let gyro = take(config.gyro_intrinsics.is_some(), 6);
        let clock = take(config.clock_offset.is_some(), 1);
        Self {
            cameras,
            slices,
            rotation,
            focal,
            gyro,
            clock,
            len,
        }
    }

    pub(super) fn slices(&self) -> usize {
        self.slices
    }

    /// First of the three rotation-offset columns of `camera` in `slice`.
    pub(super) fn rotation(&self, camera: usize, slice: usize) -> Option<usize> {
        self.rotation
            .map(|at| at + (camera * self.slices + slice) * 3)
    }

    pub(super) fn focal(&self, camera: usize, slice: usize) -> Option<usize> {
        self.focal.map(|at| at + camera * self.slices + slice)
    }

    /// First of the six gyro columns: three scales, then shear xy, xz, yz.
    pub(super) fn gyro(&self) -> Option<usize> {
        self.gyro
    }

    pub(super) fn clock(&self) -> Option<usize> {
        self.clock
    }

    /// With any calibration state on, the IMU edges are re-preintegrated from
    /// the raw stream at every state (ORB-SLAM3 `Preintegrated::Reintegrate`,
    /// `ImuTypes.cc:168-175`) instead of corrected to first order.
    pub(super) fn reintegrates(&self) -> bool {
        self.len > 0
    }
}

/// Values of the calibration states, in [`Layout`] order.
#[derive(Debug, Clone)]
pub(super) struct Calibration {
    pub(super) values: Vec<f64>,
}

impl Calibration {
    pub(super) fn zero(layout: &Layout) -> Self {
        Self {
            values: vec![0.0; layout.len],
        }
    }

    pub(super) fn vec3(&self, column: usize) -> Vec3F64 {
        Vec3F64::new(
            self.values[column],
            self.values[column + 1],
            self.values[column + 2],
        )
    }

    pub(super) fn gyro(&self, layout: &Layout) -> GyroIntrinsics {
        layout
            .gyro()
            .map_or(GyroIntrinsics::IDENTITY, |at| GyroIntrinsics {
                scale: self.vec3(at),
                shear: self.vec3(at + 3),
            })
    }

    /// `td` (s): the camera stamp `t` is read on the IMU clock at `t - td`.
    pub(super) fn clock_offset(&self, layout: &Layout) -> f64 {
        layout.clock().map_or(0.0, |at| self.values[at])
    }

    pub(super) fn retracted(&self, step: &[f64]) -> Self {
        Self {
            values: self.values.iter().zip(step).map(|(v, d)| v + d).collect(),
        }
    }

    pub(super) fn estimate(&self, layout: &Layout) -> CalibrationEstimate {
        let per_slice = |column: &dyn Fn(usize, usize) -> Option<usize>| {
            (0..layout.cameras)
                .map(|camera| {
                    (0..layout.slices)
                        .filter_map(|slice| column(camera, slice))
                        .collect::<Vec<_>>()
                })
                .filter(|columns| !columns.is_empty())
                .collect::<Vec<_>>()
        };
        CalibrationEstimate {
            camera_rotation: per_slice(&|c, s| layout.rotation(c, s))
                .into_iter()
                .map(|columns| {
                    columns
                        .into_iter()
                        .map(|at| self.vec3(at).to_array())
                        .collect()
                })
                .collect(),
            camera_focal: per_slice(&|c, s| layout.focal(c, s))
                .into_iter()
                .map(|columns| columns.into_iter().map(|at| self.values[at]).collect())
                .collect(),
            gyro_intrinsics: layout.gyro().map(|_| self.gyro(layout)),
            clock_offset_sec: layout.clock().map(|_| self.clock_offset(layout)),
        }
    }
}

/// One linear prior row: whitened residual `Σ coefficient · value` over its
/// terms (every calibration prior is linear in the states).
pub(super) struct PriorRow {
    pub(super) residual: f64,
    pub(super) terms: Vec<(usize, f64)>,
}

impl PriorRow {
    fn new(values: &[f64], terms: Vec<(usize, f64)>) -> Self {
        let residual = terms.iter().map(|&(at, w)| w * values[at]).sum();
        Self { residual, terms }
    }
}

/// The value priors of every calibration state and the random walk between
/// consecutive slices of the drift states.
pub(super) fn prior_rows(
    layout: &Layout,
    config: &OfflineCalibrationConfig,
    calibration: &Calibration,
) -> Vec<PriorRow> {
    let v = &calibration.values;
    let mut rows = Vec::new();
    let mut drift = |column: &dyn Fn(usize, usize) -> Option<usize>,
                     width: usize,
                     value_sigma: f64,
                     step_sigma: f64| {
        for camera in 0..layout.cameras {
            for slice in 0..layout.slices {
                let Some(at) = column(camera, slice) else {
                    continue;
                };
                let previous = slice.checked_sub(1).and_then(|s| column(camera, s));
                for d in 0..width {
                    rows.push(PriorRow::new(v, vec![(at + d, 1.0 / value_sigma)]));
                    if let Some(prev) = previous {
                        let w = 1.0 / step_sigma;
                        rows.push(PriorRow::new(v, vec![(at + d, w), (prev + d, -w)]));
                    }
                }
            }
        }
    };
    if let Some(prior) = config.camera_rotation {
        drift(
            &|c, s| layout.rotation(c, s),
            3,
            prior.value_sigma_deg.to_radians(),
            prior.step_sigma_deg.to_radians(),
        );
    }
    if let Some(prior) = config.camera_focal {
        drift(
            &|c, s| layout.focal(c, s),
            1,
            prior.value_sigma,
            prior.step_sigma,
        );
    }
    if let (Some(prior), Some(at)) = (config.gyro_intrinsics, layout.gyro()) {
        rows.extend((0..6).map(|d| PriorRow::new(v, vec![(at + d, 1.0 / prior.sigma)])));
    }
    if let (Some(prior), Some(at)) = (config.clock_offset, layout.clock()) {
        rows.push(PriorRow::new(v, vec![(at, 1.0 / (prior.sigma_ms * 1e-3))]));
    }
    rows
}
