use kornia_algebra::{Mat3F64, Mat6F64, Mat9F64, SMatF64, SO3F64, Vec3F64};

/// Standard gravity magnitude in m/s².
pub const GRAVITY_MAGNITUDE: f64 = 9.81;

/// A single IMU reading: angular velocity and linear acceleration in the body frame.
#[derive(Debug, Clone, Copy)]
pub struct ImuMeasurement {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Gyroscope reading in rad/s (body frame).
    pub gyro: Vec3F64,
    /// Accelerometer reading in m/s² (body frame).
    pub accel: Vec3F64,
}

/// Accelerometer and gyroscope biases.
#[derive(Debug, Clone, Copy)]
pub struct ImuBias {
    pub gyro: Vec3F64,
    pub accel: Vec3F64,
}

impl Default for ImuBias {
    fn default() -> Self {
        Self {
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        }
    }
}

/// IMU noise parameters from the sensor datasheet.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ImuCalib {
    /// Gyroscope white noise density (rad/s/√Hz).
    pub gyro_noise: f64,
    /// Accelerometer white noise density (m/s²/√Hz).
    pub accel_noise: f64,
    /// Gyroscope bias random walk (rad/s²/√Hz).
    pub gyro_bias_noise: f64,
    /// Accelerometer bias random walk (m/s³/√Hz).
    pub accel_bias_noise: f64,
}

/// How a preintegration turns samples into intervals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampleInterpolation {
    /// Each sample held over the interval that ends at its stamp, the last
    /// one to `t1` ([`PreintegratedImu::from_measurements`]).
    #[default]
    Hold,
    /// The mean of consecutive samples, the end samples interpolated to the
    /// frame stamps (ORB-SLAM3 `Tracking.cc:1684-1720`). The integrator
    /// stays Euler.
    Midpoint,
}

/// Symmetric gyro correction `ω ← (I + S) ω`: `scale` on the diagonal of `S`,
/// `shear` its off-diagonal entries (xy, xz, yz). Unitless.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroIntrinsics {
    pub scale: Vec3F64,
    pub shear: Vec3F64,
}

impl GyroIntrinsics {
    /// No correction: `S = 0`.
    pub const IDENTITY: Self = Self {
        scale: Vec3F64::ZERO,
        shear: Vec3F64::ZERO,
    };

    /// The corrected reading `(I + S) · gyro`, rad/s.
    pub fn apply(&self, gyro: Vec3F64) -> Vec3F64 {
        let (s, k) = (self.scale, self.shear);
        Vec3F64::new(
            (1.0 + s.x) * gyro.x + k.x * gyro.y + k.y * gyro.z,
            k.x * gyro.x + (1.0 + s.y) * gyro.y + k.z * gyro.z,
            k.y * gyro.x + k.z * gyro.y + (1.0 + s.z) * gyro.z,
        )
    }
}

/// Preintegrated IMU measurements between two camera frames.
///
/// Accumulates gyroscope and accelerometer readings into a relative
/// displacement (ΔR, Δv, Δp) on the manifold R × SO(3) × R³ × R³.
///
/// Also propagates covariance matrices tracking how uncertainty grows:
/// - 9×9 navigation covariance in tangent space [δrot(3), δvel(3), δpos(3)]
/// - 6×6 bias covariance [δbias_gyro(3), δbias_accel(3)] (decoupled, grows linearly)
///
/// Bias Jacobians (∂Δ/∂bias) are tracked alongside, so a small change in the
/// bias estimate can be applied to the deltas to first order without
/// re-integrating raw measurements (Forster et al. TRO 2017, eq. 70-71).
#[derive(Debug, Clone)]
pub struct PreintegratedImu {
    /// Accumulated rotation ∈ SO(3).
    pub delta_rotation: Mat3F64,
    /// Accumulated velocity ∈ R³.
    pub delta_velocity: Vec3F64,
    /// Accumulated position ∈ R³.
    pub delta_position: Vec3F64,
    /// Total elapsed time.
    pub dt: f64,
    /// Bias estimate used during integration.
    pub bias: ImuBias,
    /// Sensor noise parameters.
    pub calib: ImuCalib,
    /// 9×9 covariance in tangent space [rot, vel, pos].
    pub covariance: Mat9F64,
    /// 6×6 bias covariance [bias_gyro, bias_accel].
    /// Grows by σ²·dt each step (random walk).
    pub bias_covariance: Mat6F64,
    /// ∂ΔR/∂bias_gyro (so(3) tangent perturbation per unit gyro-bias change).
    pub d_rotation_d_bias_gyro: Mat3F64,
    /// ∂Δv/∂bias_gyro.
    pub d_velocity_d_bias_gyro: Mat3F64,
    /// ∂Δv/∂bias_accel.
    pub d_velocity_d_bias_accel: Mat3F64,
    /// ∂Δp/∂bias_gyro.
    pub d_position_d_bias_gyro: Mat3F64,
    /// ∂Δp/∂bias_accel.
    pub d_position_d_bias_accel: Mat3F64,
}

impl PreintegratedImu {
    pub fn new(bias: ImuBias, calib: ImuCalib) -> Self {
        Self {
            delta_rotation: Mat3F64::IDENTITY,
            delta_velocity: Vec3F64::ZERO,
            delta_position: Vec3F64::ZERO,
            dt: 0.0,
            bias,
            calib,
            covariance: Mat9F64::zeros(),
            bias_covariance: Mat6F64::zeros(),
            d_rotation_d_bias_gyro: Mat3F64::ZERO,
            d_velocity_d_bias_gyro: Mat3F64::ZERO,
            d_velocity_d_bias_accel: Mat3F64::ZERO,
            d_position_d_bias_gyro: Mat3F64::ZERO,
            d_position_d_bias_accel: Mat3F64::ZERO,
        }
    }

    /// Widens the bias random walk by `seconds` beyond the integrated span:
    /// for an edge whose bias link also covers time outside its samples (a
    /// still stretch elided before it), so `bias_covariance` is the walk over
    /// `dt + seconds`. Non-positive or non-finite `seconds` change nothing.
    pub fn widen_bias_walk(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0.0 {
            self.add_bias_walk(seconds);
        }
    }

    /// Bias random walk: gyro_bias_noise/accel_bias_noise are continuous-time
    /// densities [units/s/√Hz], so the walk over `seconds` has variance
    /// density²·seconds (inverse discretization of the per-sample /dt, since
    /// here the time scales up a rate rather than scaling down a reading).
    fn add_bias_walk(&mut self, seconds: f64) {
        let bg_var = self.calib.gyro_bias_noise * self.calib.gyro_bias_noise * seconds;
        let ba_var = self.calib.accel_bias_noise * self.calib.accel_bias_noise * seconds;
        for i in 0..3 {
            self.bias_covariance[(i, i)] += bg_var;
            self.bias_covariance[(i + 3, i + 3)] += ba_var;
        }
    }

    /// Builds a preintegrated factor from raw measurements covering `[t0, t1]`,
    /// linearized fresh at `bias` — i.e. full re-integration rather than the
    /// first-order `delta_*_with_bias` correction.
    ///
    /// That correction is only valid for small `Δbias` from the bias this
    /// factor was originally integrated at. A caller that keeps re-optimizing
    /// the same edge across many windows while bias is still moving (e.g. a
    /// sliding-window BA) needs to call this again once `Δbias` grows past a
    /// few sensor-noise widths, or the correction itself becomes the dominant
    /// source of residual — a purely numerical error that the optimizer can't
    /// tell apart from real signal, so it keeps pushing bias to explain it.
    pub fn from_measurements(
        bias: ImuBias,
        calib: ImuCalib,
        samples: &[ImuMeasurement],
        t0: f64,
        t1: f64,
    ) -> Self {
        let mut pre = Self::new(bias, calib);

        let mut sorted: Vec<&ImuMeasurement> = samples
            .iter()
            .filter(|m| m.timestamp >= t0 && m.timestamp <= t1)
            .collect();
        sorted.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));

        if sorted.is_empty() {
            return pre;
        }

        let mut last_t = t0;
        for sample in &sorted {
            let dt = sample.timestamp - last_t;
            if dt > 0.0 {
                pre.integrate(sample, dt);
                last_t = sample.timestamp;
            }
        }

        if last_t < t1
            && let Some(last_sample) = sorted.last()
        {
            pre.integrate(last_sample, t1 - last_t);
        }

        pre
    }

    /// [`Self::from_measurements`] with a choice of `interpolation`.
    /// `Midpoint` needs the samples just before `t0` and just after `t1` in
    /// `samples`; without the latter it holds the last sample to `t1`.
    pub fn from_measurements_with(
        bias: ImuBias,
        calib: ImuCalib,
        samples: &[ImuMeasurement],
        t0: f64,
        t1: f64,
        interpolation: SampleInterpolation,
    ) -> Self {
        match interpolation {
            SampleInterpolation::Hold => Self::from_measurements(bias, calib, samples, t0, t1),
            SampleInterpolation::Midpoint => {
                Self::from_measurements_midpoint(bias, calib, samples, t0, t1)
            }
        }
    }

    /// Trapezoids over the signal linear between samples: each interval
    /// between consecutive breakpoints (`t0`, the stamps inside, `t1`)
    /// integrates the mean of the signal at its two ends. A right Riemann sum
    /// instead lags by half a sample period (OpenVINS `Propagator.cpp:269-349`
    /// cuts and averages the same way).
    fn from_measurements_midpoint(
        bias: ImuBias,
        calib: ImuCalib,
        samples: &[ImuMeasurement],
        t0: f64,
        t1: f64,
    ) -> Self {
        let mut pre = Self::new(bias, calib);
        let mut sorted: Vec<ImuMeasurement> = samples
            .iter()
            .filter(|m| m.timestamp.is_finite())
            .copied()
            .collect();
        if sorted.is_empty() || t1 <= t0 {
            return pre;
        }
        sorted.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));

        let mut breakpoints = Vec::with_capacity(sorted.len() + 2);
        breakpoints.push(t0);
        breakpoints.extend(
            sorted
                .iter()
                .map(|m| m.timestamp)
                .filter(|&t| t > t0 && t < t1),
        );
        breakpoints.push(t1);
        for pair in breakpoints.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if end <= start {
                continue;
            }
            let (gyro_start, accel_start) = reading_at(&sorted, start);
            let (gyro_end, accel_end) = reading_at(&sorted, end);
            let mean = ImuMeasurement {
                timestamp: end,
                gyro: (gyro_start + gyro_end) * 0.5,
                accel: (accel_start + accel_end) * 0.5,
            };
            pre.integrate(&mean, end - start);
        }
        pre
    }

    /// ΔR re-expressed at a new bias estimate via the first-order correction
    /// `ΔR · Exp(∂ΔR/∂bg · δbg)`.
    pub fn delta_rotation_with_bias(&self, bias: &ImuBias) -> Mat3F64 {
        let dbg = bias.gyro - self.bias.gyro;
        let correction = SO3F64::exp(self.d_rotation_d_bias_gyro * dbg).matrix();
        self.delta_rotation * correction
    }

    /// Δv re-expressed at a new bias estimate (first order).
    pub fn delta_velocity_with_bias(&self, bias: &ImuBias) -> Vec3F64 {
        let dbg = bias.gyro - self.bias.gyro;
        let dba = bias.accel - self.bias.accel;
        self.delta_velocity + self.d_velocity_d_bias_gyro * dbg + self.d_velocity_d_bias_accel * dba
    }

    /// Δp re-expressed at a new bias estimate (first order).
    pub fn delta_position_with_bias(&self, bias: &ImuBias) -> Vec3F64 {
        let dbg = bias.gyro - self.bias.gyro;
        let dba = bias.accel - self.bias.accel;
        self.delta_position + self.d_position_d_bias_gyro * dbg + self.d_position_d_bias_accel * dba
    }

    /// Integrate a single IMU measurement over time step dt.
    ///
    /// Update order (each step uses values from before this call):
    /// 1. position:  Δp += Δv·dt + ½·ΔR·a·dt²
    /// 2. velocity:  Δv += ΔR·a·dt
    /// 3. rotation:  ΔR = ΔR · exp((ω - bg)·dt)
    ///
    /// Covariance propagation:
    ///   C' = A · C · Aᵀ + B · N · Bᵀ
    /// where A is the state transition Jacobian and B maps noise into the state.
    pub fn integrate(&mut self, measurement: &ImuMeasurement, dt: f64) {
        // Bias-correct
        let gyro = measurement.gyro - self.bias.gyro;
        let accel = measurement.accel - self.bias.accel;

        // Incremental rotation and its right Jacobian
        let omega_dt = gyro * dt;
        let d_rot = SO3F64::exp(omega_dt).matrix();
        let jr = SO3F64::right_jacobian(omega_dt);

        // Rotate acceleration by current ΔR: group action SO(3) × R³ → R³
        let rotated_accel = self.delta_rotation * accel;

        // Skew-symmetric matrix of unrotated acceleration (needed for A matrix Jacobians).
        // We use hat(a) pre-multiplied by ΔR, NOT hat(ΔR·a), following Forster et al. / ORB-SLAM3.
        let accel_hat = SO3F64::hat(accel);

        // --- Covariance propagation: C' = A·C·Aᵀ + B·N·Bᵀ ---

        // A is the 9×9 state transition Jacobian (how state perturbation evolves):
        //
        //     [ dRᵀ               0    0  ]
        // A = [ -ΔR·[a]×·dt       I    0  ]
        //     [ -½·ΔR·[a]×·dt²    I·dt  I ]
        //
        // where [a]× is hat(accel) in the body frame, pre-multiplied by ΔR.
        //
        // d_rotᵀ = dR⁻¹: this is the pullback operation from differential geometry.
        // It transports the rotation error from the tangent space at the old ΔR
        // back to so(3) (tangent space at identity) relative to the new ΔR.
        // For Lie groups, pullback is multiplication by the inverse, which for
        // rotation matrices is the transpose.
        let d_rot_t = d_rot.transpose();
        let dr_accel_hat = self.delta_rotation * accel_hat;
        let neg_dr_ah_dt = dr_accel_hat * -dt;
        let neg_dr_ah_half_dt2 = dr_accel_hat * (-0.5 * dt * dt);

        let mut a = Mat9F64::identity();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&d_rot_t.into());
        a.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&neg_dr_ah_dt.into());
        a.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&neg_dr_ah_half_dt2.into());
        a.fixed_view_mut::<3, 3>(6, 3).fill_diagonal(dt);

        // B is the 9×6 noise input matrix (how sensor noise enters the state):
        //
        //     [ J_r·dt   0     ]     gyro noise enters rotation via right Jacobian
        // B = [ 0        ΔR·dt ]     accel noise enters velocity
        //     [ 0        ½·ΔR·dt² ]  accel noise enters position
        let mut b = SMatF64::<9, 6>::zeros();
        b.fixed_view_mut::<3, 3>(0, 0).copy_from(&(jr * dt).into());
        b.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(self.delta_rotation * dt).into());
        b.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(self.delta_rotation * (0.5 * dt * dt)).into());

        // gyro_noise/accel_noise are continuous-time spectral densities [units/√Hz]
        // (datasheet convention), so the discrete per-sample variance is density²/dt
        // — same discretization ORB-SLAM3 uses when loading IMU calibration.
        let ng = self.calib.gyro_noise * self.calib.gyro_noise / dt;
        let na = self.calib.accel_noise * self.calib.accel_noise / dt;
        let mut n = Mat6F64::zeros();
        n.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(ng);
        n.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(na);

        self.covariance = (a * self.covariance) * a.transpose() + (b * n) * b.transpose();
        self.add_bias_walk(dt);

        // --- Bias Jacobian propagation (Forster et al. TRO 2017, eq. 70-71) ---
        // Measurements enter bias-corrected (ω - bg, a - ba), so each step's
        // sensitivity to the bias accumulates. Position/velocity rows must use
        // the pre-update ∂ΔR/∂bg, so the rotation row comes last.
        self.d_position_d_bias_gyro = self.d_position_d_bias_gyro
            + self.d_velocity_d_bias_gyro * dt
            + neg_dr_ah_half_dt2 * self.d_rotation_d_bias_gyro;
        self.d_position_d_bias_accel = self.d_position_d_bias_accel
            + self.d_velocity_d_bias_accel * dt
            - self.delta_rotation * (0.5 * dt * dt);
        self.d_velocity_d_bias_gyro += neg_dr_ah_dt * self.d_rotation_d_bias_gyro;
        self.d_velocity_d_bias_accel -= self.delta_rotation * dt;
        self.d_rotation_d_bias_gyro = d_rot_t * self.d_rotation_d_bias_gyro - jr * dt;

        // --- State update ---

        // 1. Position (uses old Δv and ΔR)
        self.delta_position =
            self.delta_position + self.delta_velocity * dt + rotated_accel * (0.5 * dt * dt);

        // 2. Velocity (uses old ΔR)
        self.delta_velocity += rotated_accel * dt;

        // 3. Rotation: compose on the manifold SO(3), then normalize to
        // prevent floating-point drift away from SO(3) over long windows.
        self.delta_rotation = normalize_rotation(&(self.delta_rotation * d_rot));

        self.dt += dt;
    }

    /// Predict the state at camera frame k+1 given state at frame k.
    ///
    /// Inputs (all in world frame):
    ///   - r_k:  rotation at frame k (SO(3) matrix)
    ///   - v_k:  velocity at frame k (m/s)
    ///   - p_k:  position at frame k (m)
    ///   - gravity: gravity vector in world frame, e.g. [0, 0, -9.81]
    ///
    /// Returns (r_k1, v_k1, p_k1) — predicted rotation, velocity, position.
    pub fn predict(
        &self,
        r_k: &Mat3F64,
        v_k: &Vec3F64,
        p_k: &Vec3F64,
        gravity: &Vec3F64,
    ) -> (Mat3F64, Vec3F64, Vec3F64) {
        let dt = self.dt;

        // Predicted rotation: R_{k+1} = R_k · ΔR
        let r_k1 = *r_k * self.delta_rotation;

        // Predicted velocity: v_{k+1} = v_k + g·dt + R_k · Δv
        let r_k_dv = *r_k * self.delta_velocity;
        let v_k1 = *v_k + *gravity * dt + r_k_dv;

        // Predicted position: p_{k+1} = p_k + v_k·dt + ½·g·dt² + R_k · Δp
        let r_k_dp = *r_k * self.delta_position;
        let p_k1 = *p_k + *v_k * dt + *gravity * (0.5 * dt * dt) + r_k_dp;

        (r_k1, v_k1, p_k1)
    }

    /// Integrate a batch of IMU measurements, computing dt from consecutive timestamps.
    pub fn integrate_batch(&mut self, measurements: &[ImuMeasurement]) {
        for i in 0..measurements.len().saturating_sub(1) {
            let dt = measurements[i + 1].timestamp - measurements[i].timestamp;
            self.integrate(&measurements[i], dt);
        }
    }
}

/// Gyro and accelerometer readings at `t` from time-sorted, non-empty
/// `sorted`: linear between the two samples that bracket it, held at the
/// first sample before them, and past the last sample extrapolated along the
/// last two (ORB-SLAM3 `Tracking.cc:1700-1712`), held when there is only one.
fn reading_at(sorted: &[ImuMeasurement], t: f64) -> (Vec3F64, Vec3F64) {
    let after = sorted.partition_point(|m| m.timestamp < t);
    let line = |a: &ImuMeasurement, b: &ImuMeasurement| {
        let f = (t - a.timestamp) / (b.timestamp - a.timestamp);
        (
            a.gyro + (b.gyro - a.gyro) * f,
            a.accel + (b.accel - a.accel) * f,
        )
    };
    match (after.checked_sub(1).map(|i| &sorted[i]), sorted.get(after)) {
        (Some(a), Some(b)) => line(a, b),
        (None, Some(b)) => (b.gyro, b.accel),
        (Some(b), None) => match after.checked_sub(2).map(|i| &sorted[i]) {
            Some(a) if b.timestamp > a.timestamp => line(a, b),
            _ => (b.gyro, b.accel),
        },
        (None, None) => (Vec3F64::ZERO, Vec3F64::ZERO),
    }
}

/// Re-project a matrix onto SO(3) via Gram-Schmidt orthonormalization.
/// Prevents floating-point drift from accumulating over many rotation compositions.
fn normalize_rotation(r: &Mat3F64) -> Mat3F64 {
    let cols = r.to_cols_array();
    let c0 = Vec3F64::new(cols[0], cols[1], cols[2]);
    let c1 = Vec3F64::new(cols[3], cols[4], cols[5]);

    let e0 = c0.normalize();
    let e1 = (c1 - e0 * c1.dot(e0)).normalize();
    let e2 = e0.cross(e1); // right-handed cross product ensures det = +1

    Mat3F64::from_cols(e0, e1, e2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec9F64;

    const EPS: f64 = 1e-8;

    #[test]
    fn gyro_intrinsics_apply_the_symmetric_correction() {
        let gyro = Vec3F64::new(0.3, -0.7, 1.1);
        assert_eq!(GyroIntrinsics::IDENTITY.apply(gyro), gyro);
        let intrinsics = GyroIntrinsics {
            scale: Vec3F64::new(0.01, -0.02, 0.03),
            shear: Vec3F64::new(0.001, -0.002, 0.004),
        };
        let out = intrinsics.apply(gyro);
        let expected = Vec3F64::new(
            1.01 * 0.3 + 0.001 * -0.7 - 0.002 * 1.1,
            0.001 * 0.3 + 0.98 * -0.7 + 0.004 * 1.1,
            -0.002 * 0.3 + 0.004 * -0.7 + 1.03 * 1.1,
        );
        assert!((out - expected).length() < 1e-15, "{out:?} vs {expected:?}");
    }

    fn default_calib() -> ImuCalib {
        // EuRoC V1 sensor parameters
        ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        }
    }

    #[test]
    fn widen_bias_walk_adds_the_walk_over_the_extra_time() {
        let calib = default_calib();
        let mut pre = PreintegratedImu::new(ImuBias::default(), calib);
        pre.widen_bias_walk(10.0);
        for i in 0..3 {
            let g = calib.gyro_bias_noise * calib.gyro_bias_noise * 10.0;
            let a = calib.accel_bias_noise * calib.accel_bias_noise * 10.0;
            assert!((pre.bias_covariance[(i, i)] - g).abs() <= g * 1e-12);
            assert!((pre.bias_covariance[(i + 3, i + 3)] - a).abs() <= a * 1e-12);
        }
        let diagonal = Mat6F64::from_diagonal(&pre.bias_covariance.diagonal());
        assert_eq!(pre.bias_covariance, diagonal);
        let before = pre.bias_covariance;
        for s in [0.0, -1.0, f64::NAN] {
            pre.widen_bias_walk(s);
        }
        assert_eq!(pre.bias_covariance, before);
    }

    #[test]
    fn zero_motion() {
        // Zero gyro, zero accel (freefall) — nothing should change.
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        };
        pre.integrate(&m, 0.01);

        assert!((pre.delta_velocity.x).abs() < EPS);
        assert!((pre.delta_velocity.y).abs() < EPS);
        assert!((pre.delta_velocity.z).abs() < EPS);
        assert!((pre.delta_position.x).abs() < EPS);
        assert!((pre.delta_position.y).abs() < EPS);
        assert!((pre.delta_position.z).abs() < EPS);
    }

    #[test]
    fn constant_accel() {
        // 1 m/s² along x for 1 second (100 steps of 0.01s).
        // Expected: v = 1 m/s, p = 0.5 m
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(1.0, 0.0, 0.0),
        };
        for _ in 0..100 {
            pre.integrate(&m, 0.01);
        }

        assert!((pre.delta_velocity.x - 1.0).abs() < 1e-6);
        assert!((pre.delta_position.x - 0.5).abs() < 1e-4);
        assert!((pre.dt - 1.0).abs() < EPS);
    }

    #[test]
    fn pure_rotation() {
        // Rotate 90 degrees around z-axis: omega_z = pi/2 for 1 second.
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::new(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            accel: Vec3F64::ZERO,
        };
        for _ in 0..1000 {
            pre.integrate(&m, 0.001);
        }

        // After 90 deg around z: x-axis -> y-axis
        let cols = pre.delta_rotation.to_cols_array();
        // col0 should be ~[0, 1, 0]
        assert!((cols[0] - 0.0).abs() < 1e-3);
        assert!((cols[1] - 1.0).abs() < 1e-3);
        assert!((cols[2] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn predict_stationary_with_gravity() {
        // IMU sitting still on a table. Accelerometer reads +9.81 upward
        // (resisting gravity). After integrating, predict should show
        // zero motion — gravity cancels the accelerometer reading.
        let gravity = Vec3F64::new(0.0, 0.0, -9.81);
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());

        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81), // accelerometer reads up
        };
        for _ in 0..100 {
            pre.integrate(&m, 0.01); // 1 second total
        }

        let r_k = Mat3F64::IDENTITY;
        let v_k = Vec3F64::ZERO;
        let p_k = Vec3F64::ZERO;

        let (r_k1, v_k1, p_k1) = pre.predict(&r_k, &v_k, &p_k, &gravity);

        // Rotation should stay identity
        let cols = r_k1.to_cols_array();
        let id = Mat3F64::IDENTITY.to_cols_array();
        for i in 0..9 {
            assert!((cols[i] - id[i]).abs() < 1e-6, "rotation changed at {i}");
        }

        // Velocity and position should be ~zero (gravity cancels accel)
        assert!((v_k1.x).abs() < 1e-6);
        assert!((v_k1.y).abs() < 1e-6);
        assert!((v_k1.z).abs() < 1e-4);
        assert!((p_k1.x).abs() < 1e-6);
        assert!((p_k1.y).abs() < 1e-6);
        assert!((p_k1.z).abs() < 1e-4);
    }

    #[test]
    fn integrate_batch_matches_sequential() {
        let measurements: Vec<ImuMeasurement> = (0..10)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * 0.005, // 200Hz
                gyro: Vec3F64::new(0.1, -0.05, 0.2),
                accel: Vec3F64::new(0.0, 0.0, 9.81),
            })
            .collect();

        // Batch
        let mut pre_batch = PreintegratedImu::new(ImuBias::default(), default_calib());
        pre_batch.integrate_batch(&measurements);

        // Sequential
        let mut pre_seq = PreintegratedImu::new(ImuBias::default(), default_calib());
        for i in 0..measurements.len() - 1 {
            let dt = measurements[i + 1].timestamp - measurements[i].timestamp;
            pre_seq.integrate(&measurements[i], dt);
        }

        // Should be identical
        assert!((pre_batch.dt - pre_seq.dt).abs() < EPS);
        assert!((pre_batch.delta_velocity.x - pre_seq.delta_velocity.x).abs() < EPS);
        assert!((pre_batch.delta_velocity.y - pre_seq.delta_velocity.y).abs() < EPS);
        assert!((pre_batch.delta_velocity.z - pre_seq.delta_velocity.z).abs() < EPS);
        assert!((pre_batch.delta_position.x - pre_seq.delta_position.x).abs() < EPS);
        assert!((pre_batch.delta_position.y - pre_seq.delta_position.y).abs() < EPS);
        assert!((pre_batch.delta_position.z - pre_seq.delta_position.z).abs() < EPS);
    }

    #[test]
    fn covariance_grows() {
        // Covariance should increase with each measurement.
        let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
        let m = ImuMeasurement {
            timestamp: 0.0,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::new(0.0, 0.0, 9.81),
        };

        // Diagonal should be zero initially
        for i in 0..9 {
            assert_eq!(pre.covariance[(i, i)], 0.0);
        }

        // After some integration, diagonal should be positive
        for _ in 0..100 {
            pre.integrate(&m, 0.01);
        }
        for i in 0..9 {
            assert!(
                pre.covariance[(i, i)] > 0.0,
                "diagonal {i} should be positive"
            );
        }

        // More integration = more uncertainty
        let cov_after_100 = pre.covariance;
        for _ in 0..100 {
            pre.integrate(&m, 0.01);
        }
        for i in 0..9 {
            assert!(
                pre.covariance[(i, i)] >= cov_after_100[(i, i)],
                "diagonal {i} should not decrease"
            );
        }
    }

    /// E14: past the last sample the midpoint signal goes on along the last
    /// two samples, as ORB-SLAM3 extrapolates (`Tracking.cc:1700-1712`). A
    /// gyro rate linear in time then integrates exactly to a frame stamp
    /// 2.5 ms after the last sample; holding the last sample lost 3.1e-5 rad.
    #[test]
    fn midpoint_extrapolates_past_the_last_sample() {
        let samples: Vec<ImuMeasurement> = (0..=20)
            .map(|k| {
                let t = k as f64 * 0.005;
                ImuMeasurement {
                    timestamp: t,
                    gyro: Vec3F64::new(0.0, 0.0, 1.0 + 10.0 * t),
                    accel: Vec3F64::ZERO,
                }
            })
            .collect();
        let t1 = 0.1025;
        let pre = PreintegratedImu::from_measurements_with(
            ImuBias::default(),
            default_calib(),
            &samples,
            0.0,
            t1,
            SampleInterpolation::Midpoint,
        );
        let angle = SO3F64::from_matrix(&pre.delta_rotation).log().z;
        let exact = t1 + 5.0 * t1 * t1;
        assert!((angle - exact).abs() < 1e-9, "{angle} vs {exact}");
    }

    /// Card A U1: ω = sin(2πt) about z at 200 Hz, frames every 50 ms offset
    /// 1.3 ms from the sample grid. Holding each sample lags by half a sample
    /// period (theory 7.8e-4 rad per interval); the midpoint is near exact
    /// (theory 4.1e-6 rad).
    #[test]
    fn midpoint_removes_the_half_sample_lag() {
        let omega = |t: f64| (2.0 * std::f64::consts::PI * t).sin();
        let samples: Vec<ImuMeasurement> = (0..1000)
            .map(|i| {
                let t = i as f64 * 0.005;
                ImuMeasurement {
                    timestamp: t,
                    gyro: Vec3F64::new(0.0, 0.0, omega(t)),
                    accel: Vec3F64::ZERO,
                }
            })
            .collect();
        let angle_error = |interpolation: SampleInterpolation, t0: f64, t1: f64| {
            let pre = PreintegratedImu::from_measurements_with(
                ImuBias::default(),
                default_calib(),
                &samples,
                t0,
                t1,
                interpolation,
            );
            let two_pi = 2.0 * std::f64::consts::PI;
            let exact = ((two_pi * t0).cos() - (two_pi * t1).cos()) / two_pi;
            (SO3F64::from_matrix(&pre.delta_rotation).log().z - exact).abs()
        };
        let worst = |interpolation| {
            (0..80)
                .map(|k| {
                    let t0 = k as f64 * 0.05 + 0.0013;
                    angle_error(interpolation, t0, t0 + 0.05)
                })
                .fold(0.0, f64::max)
        };
        let hold = worst(SampleInterpolation::Hold);
        let midpoint = worst(SampleInterpolation::Midpoint);
        assert!((7.0e-4..=8.5e-4).contains(&hold), "hold {hold:.3e}");
        assert!(midpoint <= 1e-5, "midpoint {midpoint:.3e}");
    }

    #[test]
    fn hold_through_from_measurements_with_is_from_measurements() {
        let samples: Vec<ImuMeasurement> = (0..20)
            .map(|i| ImuMeasurement {
                timestamp: i as f64 * 0.005,
                gyro: Vec3F64::new(0.1, -0.2 * i as f64, 0.3),
                accel: Vec3F64::new(0.0, 0.5, 9.81),
            })
            .collect();
        let direct = PreintegratedImu::from_measurements(
            ImuBias::default(),
            default_calib(),
            &samples,
            0.0013,
            0.0813,
        );
        let with = PreintegratedImu::from_measurements_with(
            ImuBias::default(),
            default_calib(),
            &samples,
            0.0013,
            0.0813,
            SampleInterpolation::Hold,
        );
        assert_eq!(direct.delta_rotation, with.delta_rotation);
        assert_eq!(direct.delta_velocity, with.delta_velocity);
        assert_eq!(direct.covariance, with.covariance);
    }

    /// The covariance is `Σ_k J_k · N · J_kᵀ`, with `J_k` the central-difference
    /// derivative of the final `[Log(ΔR̂ᵀ·ΔR), Δv − Δv̂, Δp − Δp̂]` by step `k`'s
    /// reading, so a transposed block of the transition matrix shows up.
    #[test]
    fn covariance_matches_linearized_noise_propagation() {
        let dt = 0.01;
        let calib = ImuCalib {
            gyro_noise: 0.05,
            accel_noise: 0.02,
            ..default_calib()
        };
        let readings: Vec<ImuMeasurement> = (0..20)
            .map(|k| {
                let t = k as f64 * dt;
                ImuMeasurement {
                    timestamp: t,
                    gyro: Vec3F64::new(0.8 + 2.0 * t, -0.5, 0.3 * (10.0 * t).sin()),
                    accel: Vec3F64::new(1.0, -2.0 + 5.0 * t, 9.81),
                }
            })
            .collect();
        let integrate = |readings: &[ImuMeasurement]| {
            let mut pre = PreintegratedImu::new(ImuBias::default(), calib);
            for m in readings {
                pre.integrate(m, dt);
            }
            pre
        };
        let nominal = integrate(&readings);
        let error = |pre: &PreintegratedImu| {
            let rot_err = nominal.delta_rotation.transpose() * pre.delta_rotation;
            let r = SO3F64::from_matrix(&rot_err).log();
            let v = pre.delta_velocity - nominal.delta_velocity;
            let p = pre.delta_position - nominal.delta_position;
            Vec9F64::from_column_slice(&[r.x, r.y, r.z, v.x, v.y, v.z, p.x, p.y, p.z])
        };

        let h = 1e-6;
        let variances = [
            calib.gyro_noise * calib.gyro_noise / dt,
            calib.accel_noise * calib.accel_noise / dt,
        ];
        let mut expected = Mat9F64::zeros();
        for k in 0..readings.len() {
            for axis in 0..6 {
                let shifted = |eps: f64| {
                    let mut perturbed = readings.clone();
                    let mut e = [0.0; 3];
                    e[axis % 3] = eps;
                    let e = Vec3F64::from_array(e);
                    if axis < 3 {
                        perturbed[k].gyro += e;
                    } else {
                        perturbed[k].accel += e;
                    }
                    error(&integrate(&perturbed))
                };
                let column = (shifted(h) - shifted(-h)) / (2.0 * h);
                expected += column * column.transpose() * variances[axis / 3];
            }
        }

        let got = nominal.covariance;
        for r in 0..9 {
            for c in 0..9 {
                let scale = (got[(r, r)] * got[(c, c)]).sqrt();
                assert!(
                    (got[(r, c)] - expected[(r, c)]).abs() < 1e-6 * scale,
                    "({r}, {c}): {} vs {}",
                    got[(r, c)],
                    expected[(r, c)]
                );
            }
        }
    }
}
