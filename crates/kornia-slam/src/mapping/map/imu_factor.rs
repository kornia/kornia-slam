use kornia_sensors::imu::{ImuBias, ImuMeasurement, PreintegratedImu};

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
    /// Seconds outside `[t0, t1]` that the bias link between the two
    /// keyframes also spans: still stretches with no motion, elided from the
    /// samples (an edge over the tail of a stop) or absorbed when a keyframe
    /// at the end of a stationary chain was removed. `preintegrated`'s bias
    /// covariance includes the walk over them, and every re-preintegration
    /// ([`Self::repreintegrate`]) adds it back. Zero for an ordinary edge.
    pub bias_walk_extra_s: f64,
}

impl ImuFactor {
    /// The edge's raw samples re-preintegrated over `[t0, t1]` at `bias`,
    /// with the bias walk widened by [`Self::bias_walk_extra_s`].
    pub fn repreintegrate(&self, bias: ImuBias) -> PreintegratedImu {
        let mut preintegrated = PreintegratedImu::from_measurements(
            bias,
            self.preintegrated.calib,
            &self.raw_samples,
            self.t0,
            self.t1,
        );
        preintegrated.widen_bias_walk(self.bias_walk_extra_s);
        preintegrated
    }
}
