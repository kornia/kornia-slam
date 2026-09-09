use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};

/// Preintegrated IMU measurements connecting two consecutive keyframes.
#[derive(Debug, Clone)]
pub struct ImuFactor {
    /// Index of the earlier keyframe (`Keyframe::frame.idx`).
    pub prev_kf_idx: usize,
    /// Index of the later keyframe.
    pub curr_kf_idx: usize,
    /// IMU deltas integrated over the interval between the two keyframes.
    pub preintegrated: PreintegratedImu,
    /// Raw measurements covering `[t0, t1]`, retained so this factor can be
    /// repropagated (see `PreintegratedImu::from_measurements`) once the
    /// bias it was linearized at has drifted too far from the current
    /// estimate for the first-order correction to stay valid.
    pub raw_samples: Vec<ImuMeasurement>,
    pub t0: f64,
    pub t1: f64,
}
