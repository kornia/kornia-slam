//! Calibration states shared by the offline solve ([`super::global_vi_ba`])
//! and the online local VI-BA, each block present only when its knob is on:
//! per camera and time slice an IMU-camera rotation offset and a focal scale
//! (visloc's drift states), the symmetric gyro correction and one IMU-camera
//! clock offset (VINS-Fusion's `td`). OpenVINS keeps one flag per state
//! (`StateOptions.h:47-56`); so does this layout.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::imu::{GyroIntrinsics, ImuMeasurement};

use super::global_vi_ba::{
    CalibrationEstimate, ClockOffsetPrior, GyroIntrinsicsPrior, OfflineCalibrationConfig,
};
use crate::system::OnlineCalibrationConfig;

/// Central-difference steps of the gyro-intrinsics and clock columns: the
/// clock step is a quarter of the 1 kHz sample spacing.
const GYRO_STEP: f64 = 1e-6;
const CLOCK_STEP_SEC: f64 = 0.25e-3;
/// Central-difference step of the primary camera's rotation columns, which
/// enter only the IMU residual (card G, D1).
pub(in crate::mapping) const ROTATION_STEP_RAD: f64 = 1e-6;

/// Column of every calibration state in the border of the reduced system.
#[derive(Debug, Clone)]
pub(in crate::mapping) struct Layout {
    cameras: usize,
    slices: usize,
    rotation: Option<usize>,
    focal: Option<usize>,
    gyro: Option<usize>,
    clock: Option<usize>,
    pub(in crate::mapping) len: usize,
}

impl Layout {
    pub(in crate::mapping) fn new(config: &OfflineCalibrationConfig, cameras: usize) -> Self {
        Self::with(
            cameras,
            config.time_slices.max(1),
            [
                config.camera_rotation.is_some(),
                config.camera_focal.is_some(),
                config.gyro_intrinsics.is_some(),
                config.clock_offset.is_some(),
            ],
        )
    }

    /// The online layout: one slice per camera. Camera 0's rotation columns
    /// hold the offset of the IMU-camera extrinsic (card G, D1); camera
    /// `k ≥ 1`'s that of its extrinsic to camera 0.
    pub(in crate::mapping) fn online(config: &OnlineCalibrationConfig, cameras: usize) -> Self {
        Self::with(
            cameras,
            1,
            [
                config.camera_rotation.is_some(),
                config.camera_focal.is_some(),
                config.gyro_intrinsics.is_some(),
                config.clock_offset.is_some(),
            ],
        )
    }

    /// Columns in the order rotation, focal, gyro, clock; `on` is that order.
    fn with(cameras: usize, slices: usize, on: [bool; 4]) -> Self {
        let mut len = 0;
        let mut take = |on: bool, columns: usize| {
            on.then(|| {
                let at = len;
                len += columns;
                at
            })
        };
        let rotation = take(on[0], cameras * slices * 3);
        let focal = take(on[1], cameras * slices);
        let gyro = take(on[2], 6);
        let clock = take(on[3], 1);
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

    pub(in crate::mapping) fn cameras(&self) -> usize {
        self.cameras
    }

    pub(in crate::mapping) fn slices(&self) -> usize {
        self.slices
    }

    /// First of the three rotation-offset columns of `camera` in `slice`.
    pub(in crate::mapping) fn rotation(&self, camera: usize, slice: usize) -> Option<usize> {
        self.rotation
            .map(|at| at + (camera * self.slices + slice) * 3)
    }

    pub(in crate::mapping) fn focal(&self, camera: usize, slice: usize) -> Option<usize> {
        self.focal.map(|at| at + camera * self.slices + slice)
    }

    /// First of the six gyro columns: three scales, then shear xy, xz, yz.
    pub(in crate::mapping) fn gyro(&self) -> Option<usize> {
        self.gyro
    }

    pub(in crate::mapping) fn clock(&self) -> Option<usize> {
        self.clock
    }

    /// With any calibration state on, the IMU edges are re-preintegrated from
    /// the raw stream at every state (ORB-SLAM3 `Preintegrated::Reintegrate`,
    /// `ImuTypes.cc:168-175`) instead of corrected to first order.
    pub(in crate::mapping) fn reintegrates(&self) -> bool {
        self.len > 0
    }

    /// Online only: the IMU edges are re-integrated only for the states that
    /// enter the preintegration, the gyro and the clock (card G, D2); the
    /// camera states keep the first-order bias correction.
    pub(in crate::mapping) fn imu_reintegrates(&self) -> bool {
        self.gyro.is_some() || self.clock.is_some()
    }
}

/// Values of the calibration states, in [`Layout`] order.
#[derive(Debug, Clone)]
pub(in crate::mapping) struct Calibration {
    pub(in crate::mapping) values: Vec<f64>,
}

impl Calibration {
    pub(in crate::mapping) fn zero(layout: &Layout) -> Self {
        Self {
            values: vec![0.0; layout.len],
        }
    }

    pub(in crate::mapping) fn vec3(&self, column: usize) -> Vec3F64 {
        Vec3F64::new(
            self.values[column],
            self.values[column + 1],
            self.values[column + 2],
        )
    }

    pub(in crate::mapping) fn gyro(&self, layout: &Layout) -> GyroIntrinsics {
        layout
            .gyro()
            .map_or(GyroIntrinsics::IDENTITY, |at| GyroIntrinsics {
                scale: self.vec3(at),
                shear: self.vec3(at + 3),
            })
    }

    /// `td` (s): the camera stamp `t` is read on the IMU clock at `t - td`.
    pub(in crate::mapping) fn clock_offset(&self, layout: &Layout) -> f64 {
        layout.clock().map_or(0.0, |at| self.values[at])
    }

    pub(in crate::mapping) fn retracted(&self, step: &[f64]) -> Self {
        Self {
            values: self.values.iter().zip(step).map(|(v, d)| v + d).collect(),
        }
    }

    pub(in crate::mapping) fn estimate(&self, layout: &Layout) -> CalibrationEstimate {
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
pub(in crate::mapping) struct PriorRow {
    pub(in crate::mapping) residual: f64,
    pub(in crate::mapping) terms: Vec<(usize, f64)>,
}

impl PriorRow {
    fn new(values: &[f64], terms: Vec<(usize, f64)>) -> Self {
        let residual = terms.iter().map(|&(at, w)| w * values[at]).sum();
        Self { residual, terms }
    }
}

/// The value priors of every calibration state and the random walk between
/// consecutive slices of the drift states.
pub(in crate::mapping) fn prior_rows(
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
    imu_value_rows(
        layout,
        config.gyro_intrinsics,
        config.clock_offset,
        v,
        &mut rows,
    );
    rows
}

/// The value priors of the gyro and clock states, which carry no random walk.
fn imu_value_rows(
    layout: &Layout,
    gyro: Option<GyroIntrinsicsPrior>,
    clock: Option<ClockOffsetPrior>,
    v: &[f64],
    rows: &mut Vec<PriorRow>,
) {
    if let (Some(prior), Some(at)) = (gyro, layout.gyro()) {
        rows.extend((0..6).map(|d| PriorRow::new(v, vec![(at + d, 1.0 / prior.sigma)])));
    }
    if let (Some(prior), Some(at)) = (clock, layout.clock()) {
        rows.push(PriorRow::new(v, vec![(at, 1.0 / (prior.sigma_ms * 1e-3))]));
    }
}

/// The online priors (card G, C2): per camera state a value row to the
/// factory value and a step row to `previous`, the estimate the solve
/// started from; the gyro and clock states a value row only, as offline.
/// The step rows make the estimate an exponential filter across solves, not
/// a marginal: consecutive windows share data.
pub(in crate::mapping) fn online_prior_rows(
    layout: &Layout,
    config: &OnlineCalibrationConfig,
    current: &Calibration,
    previous: &Calibration,
) -> Vec<PriorRow> {
    let v = &current.values;
    let mut rows = Vec::new();
    let mut per_camera = |column: &dyn Fn(usize) -> Option<usize>,
                          width: usize,
                          value_sigma: f64,
                          step_sigma: f64| {
        for at in (0..layout.cameras).filter_map(column) {
            for d in 0..width {
                rows.push(PriorRow::new(v, vec![(at + d, 1.0 / value_sigma)]));
                // `previous` is a constant of this solve, so the row has one term.
                let w = 1.0 / step_sigma;
                rows.push(PriorRow {
                    residual: w * (v[at + d] - previous.values[at + d]),
                    terms: vec![(at + d, w)],
                });
            }
        }
    };
    if let Some(prior) = config.camera_rotation {
        per_camera(
            &|c| layout.rotation(c, 0),
            3,
            prior.value_sigma_deg.to_radians(),
            prior.step_sigma_deg.to_radians(),
        );
    }
    if let Some(prior) = config.camera_focal {
        per_camera(
            &|c| layout.focal(c, 0),
            1,
            prior.value_sigma,
            prior.step_sigma,
        );
    }
    imu_value_rows(
        layout,
        config.gyro_intrinsics,
        config.clock_offset,
        v,
        &mut rows,
    );
    rows
}

/// `imu_t_bc · Exp(−θ)` (`X_body = T_bc · X_cam`): the IMU-camera extrinsic
/// with the primary camera's rotation offset `θ` applied about the camera
/// centre, so its translation is unchanged; `None` is the identity extrinsic.
/// The body pose is then `T_bw = imu_t_bc · Exp(−θ) · T_cw`, `T_cw` the
/// camera pose tracking estimates (card G, D1).
pub(in crate::mapping) fn imu_t_bc_with_offset(
    imu_t_bc: Option<&Pose3d>,
    theta: Vec3F64,
) -> Pose3d {
    let offset = Pose3d::new(SO3F64::exp(theta * -1.0).matrix(), Vec3F64::ZERO);
    imu_t_bc.unwrap_or(&Pose3d::IDENTITY).compose(&offset)
}

/// The IMU calibration columns of `layout` (gyro, then clock) and the
/// central-difference step of each.
pub(in crate::mapping) fn imu_calibration_columns(layout: &Layout) -> Vec<(usize, f64)> {
    let mut columns: Vec<(usize, f64)> = Vec::new();
    if let Some(at) = layout.gyro() {
        columns.extend((at..at + 6).map(|c| (c, GYRO_STEP)));
    }
    if let Some(at) = layout.clock() {
        columns.push((at, CLOCK_STEP_SEC));
    }
    columns
}

/// `samples` with the gyro correction applied to each reading.
pub(in crate::mapping) fn corrected_samples(
    samples: &[ImuMeasurement],
    gyro: &GyroIntrinsics,
) -> Vec<ImuMeasurement> {
    samples
        .iter()
        .map(|s| ImuMeasurement {
            gyro: gyro.apply(s.gyro),
            ..*s
        })
        .collect()
}
