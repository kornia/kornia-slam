use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};

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

impl ImuMeasurement {
    /// The reading an ideal IMU produces for a known kinematic state.
    ///
    /// This is the **forward sensor model** — the inverse of what
    /// [`PreintegratedImu`] does, which is why it belongs here rather than in a
    /// consumer. Integrating a stream produced by this function must reproduce
    /// the motion it was generated from; that round trip is the sharpest
    /// available check on [`PreintegratedImu::integrate`], and
    /// `preintegration_inverts_the_forward_model` below runs it.
    ///
    /// ```text
    /// gyro  = ω_body            + b_g
    /// accel = R_wbᵀ · (a − g)   + b_a
    /// ```
    ///
    /// The accelerometer measures **specific force**, not acceleration: a body
    /// in free fall (`a == g`) reads zero, and one at rest reads `−R_wbᵀ·g`.
    ///
    /// # Arguments
    ///
    /// - `rotation_wb` — body-to-world rotation at `timestamp`.
    /// - `accel_world` — world-frame acceleration, **excluding** gravity.
    /// - `angular_velocity_body` — body-frame angular velocity.
    /// - `gravity` — world-frame gravity vector (e.g. `[0, 9.81, 0]` with
    ///   OpenCV Y-down axes).
    /// - `bias` — bias to add. Noise, if wanted, is the caller's business:
    ///   this function stays deterministic so it can be used in exact tests.
    pub fn simulate(
        timestamp: f64,
        rotation_wb: &Mat3F64,
        accel_world: Vec3F64,
        angular_velocity_body: Vec3F64,
        gravity: Vec3F64,
        bias: &ImuBias,
    ) -> Self {
        let r_bw = Mat3F64(*rotation_wb.transpose());
        Self {
            timestamp,
            gyro: angular_velocity_body + bias.gyro,
            accel: r_bw * (accel_world - gravity) + bias.accel,
        }
    }
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
#[derive(Debug, Clone, Copy)]
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
    /// 9×9 covariance in tangent space [rot, vel, pos], stored column-major.
    pub covariance: [f64; 81],
    /// 6×6 bias covariance [bias_gyro, bias_accel], stored column-major.
    /// Grows by σ²·dt each step (random walk).
    pub bias_covariance: [f64; 36],
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
            covariance: [0.0; 81],
            bias_covariance: [0.0; 36],
            d_rotation_d_bias_gyro: Mat3F64::ZERO,
            d_velocity_d_bias_gyro: Mat3F64::ZERO,
            d_velocity_d_bias_accel: Mat3F64::ZERO,
            d_position_d_bias_gyro: Mat3F64::ZERO,
            d_position_d_bias_accel: Mat3F64::ZERO,
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

    /// ΔR re-expressed at a new bias estimate via the first-order correction
    /// `ΔR · Exp(∂ΔR/∂bg · δbg)`.
    pub fn delta_rotation_with_bias(&self, bias: &ImuBias) -> Mat3F64 {
        let dbg = bias.gyro - self.bias.gyro;
        let correction = SO3F64::exp(self.d_rotation_d_bias_gyro * dbg).matrix();
        Mat3F64(self.delta_rotation.mul_mat3(&correction))
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
        let d_rot_t = Mat3F64(*d_rot.transpose());
        let dr_accel_hat = Mat3F64(self.delta_rotation.mul_mat3(&accel_hat));
        let neg_dr_ah_dt = mat3_scalar(&dr_accel_hat, -dt);
        let neg_dr_ah_half_dt2 = mat3_scalar(&dr_accel_hat, -0.5 * dt * dt);
        let i_dt = mat3_scalar(&Mat3F64::IDENTITY, dt);

        let a = block3x3_to_9x9([
            [d_rot_t, Mat3F64::ZERO, Mat3F64::ZERO],
            [neg_dr_ah_dt, Mat3F64::IDENTITY, Mat3F64::ZERO],
            [neg_dr_ah_half_dt2, i_dt, Mat3F64::IDENTITY],
        ]);

        // B is the 9×6 noise input matrix (how sensor noise enters the state):
        //
        //     [ J_r·dt   0     ]     gyro noise enters rotation via right Jacobian
        // B = [ 0        ΔR·dt ]     accel noise enters velocity
        //     [ 0        ½·ΔR·dt² ]  accel noise enters position
        let jr_dt = mat3_scalar(&jr, dt);
        let dr_dt = mat3_scalar(&self.delta_rotation, dt);
        let dr_half_dt2 = mat3_scalar(&self.delta_rotation, 0.5 * dt * dt);

        let b = block3x2_to_9x6([
            [jr_dt, Mat3F64::ZERO],
            [Mat3F64::ZERO, dr_dt],
            [Mat3F64::ZERO, dr_half_dt2],
        ]);

        // gyro_noise/accel_noise are continuous-time spectral densities [units/√Hz]
        // (datasheet convention), so the discrete per-sample variance is density²/dt
        // — same discretization ORB-SLAM3 uses when loading IMU calibration.
        let ng = self.calib.gyro_noise * self.calib.gyro_noise / dt;
        let na = self.calib.accel_noise * self.calib.accel_noise / dt;
        let n = diag_6x6(ng, na);

        // C' = A·C·Aᵀ + B·N·Bᵀ
        let ac = mat9_mul(&a, &self.covariance);
        let a_t = mat9_transpose(&a);
        let acat = mat9_mul(&ac, &a_t);

        let bn = mat9x6_mul_6x6(&b, &n);
        let b_t = mat6x9_transpose(&b);
        let bnbt = mat9x6_mul_6x9(&bn, &b_t);

        self.covariance = mat9_add(&acat, &bnbt);

        // Bias random walk: gyro_bias_noise/accel_bias_noise are continuous-time
        // densities [units/s/√Hz], so each step's bias increment has variance
        // density²·dt (inverse discretization of the /dt above, since here dt
        // scales up a rate rather than scaling down a per-sample reading).
        let bg_var = self.calib.gyro_bias_noise * self.calib.gyro_bias_noise * dt;
        let ba_var = self.calib.accel_bias_noise * self.calib.accel_bias_noise * dt;
        for i in 0..3 {
            self.bias_covariance[i + i * 6] += bg_var;
        }
        for i in 3..6 {
            self.bias_covariance[i + i * 6] += ba_var;
        }

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
        self.delta_rotation = normalize_rotation(&Mat3F64(self.delta_rotation.mul_mat3(&d_rot)));

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
        let r_k1 = Mat3F64(r_k.mul_mat3(&self.delta_rotation));

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

/// Re-project a matrix onto SO(3) via Gram-Schmidt orthonormalization.
/// Prevents floating-point drift from accumulating over many rotation compositions.
fn normalize_rotation(r: &Mat3F64) -> Mat3F64 {
    let cols = r.to_cols_array();
    // Extract columns (column-major: col0 = [0,1,2], col1 = [3,4,5], col2 = [6,7,8])
    let c0 = glam::DVec3::new(cols[0], cols[1], cols[2]);
    let c1 = glam::DVec3::new(cols[3], cols[4], cols[5]);

    // Gram-Schmidt: orthonormalize col0, col1, then cross for col2
    let e0 = c0.normalize();
    let e1 = (c1 - e0 * c1.dot(e0)).normalize();
    let e2 = e0.cross(e1); // right-handed cross product ensures det = +1

    Mat3F64::from_cols_array(&[e0.x, e0.y, e0.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z])
}

// --- Matrix helpers ---
// These operate on flat column-major arrays to avoid pulling in a full linear algebra crate.

/// Scalar multiply a Mat3F64.
fn mat3_scalar(m: &Mat3F64, s: f64) -> Mat3F64 {
    Mat3F64(m.mul_scalar(s))
}

/// Column-major index for any matrix with 9 rows (9×9 and 9×6 alike).
#[inline]
fn idx9(row: usize, col: usize) -> usize {
    row + col * 9
}

/// Build a 9×9 matrix from a 3×3 grid of Mat3F64 blocks.
fn block3x3_to_9x9(blocks: [[Mat3F64; 3]; 3]) -> [f64; 81] {
    let mut out = [0.0f64; 81];
    for br in 0..3 {
        for bc in 0..3 {
            let cols = blocks[br][bc].to_cols_array();
            for c in 0..3 {
                for r in 0..3 {
                    out[idx9(br * 3 + r, bc * 3 + c)] = cols[c * 3 + r];
                }
            }
        }
    }
    out
}

/// Build a 9×6 matrix from a 3×2 grid of Mat3F64 blocks.
fn block3x2_to_9x6(blocks: [[Mat3F64; 2]; 3]) -> [f64; 54] {
    let mut out = [0.0f64; 54];
    for br in 0..3 {
        for bc in 0..2 {
            let cols = blocks[br][bc].to_cols_array();
            for c in 0..3 {
                for r in 0..3 {
                    out[idx9(br * 3 + r, bc * 3 + c)] = cols[c * 3 + r];
                }
            }
        }
    }
    out
}

/// 6×6 diagonal noise matrix.
fn diag_6x6(gyro_var: f64, accel_var: f64) -> [f64; 36] {
    let mut out = [0.0f64; 36];
    for i in 0..3 {
        out[i + i * 6] = gyro_var;
    }
    for i in 3..6 {
        out[i + i * 6] = accel_var;
    }
    out
}

/// 9×9 multiply (column-major).
fn mat9_mul(a: &[f64; 81], b: &[f64; 81]) -> [f64; 81] {
    let mut out = [0.0f64; 81];
    for c in 0..9 {
        for r in 0..9 {
            let mut sum = 0.0;
            for k in 0..9 {
                sum += a[idx9(r, k)] * b[idx9(k, c)];
            }
            out[idx9(r, c)] = sum;
        }
    }
    out
}

/// 9×9 transpose (column-major).
fn mat9_transpose(a: &[f64; 81]) -> [f64; 81] {
    let mut out = [0.0f64; 81];
    for c in 0..9 {
        for r in 0..9 {
            out[idx9(r, c)] = a[idx9(c, r)];
        }
    }
    out
}

/// 9×9 add.
fn mat9_add(a: &[f64; 81], b: &[f64; 81]) -> [f64; 81] {
    let mut out = [0.0f64; 81];
    for i in 0..81 {
        out[i] = a[i] + b[i];
    }
    out
}

/// 9×6 × 6×6 -> 9×6 (column-major).
fn mat9x6_mul_6x6(a: &[f64; 54], b: &[f64; 36]) -> [f64; 54] {
    let mut out = [0.0f64; 54];
    for c in 0..6 {
        for r in 0..9 {
            let mut sum = 0.0;
            for k in 0..6 {
                sum += a[r + k * 9] * b[k + c * 6];
            }
            out[r + c * 9] = sum;
        }
    }
    out
}

/// Transpose 9×6 -> 6×9 (column-major).
fn mat6x9_transpose(a: &[f64; 54]) -> [f64; 54] {
    let mut out = [0.0f64; 54];
    for c in 0..6 {
        for r in 0..9 {
            // a[r,c] in 9×6 -> out[c,r] in 6×9
            out[c + r * 6] = a[r + c * 9];
        }
    }
    out
}

/// 9×6 × 6×9 -> 9×9 (column-major).
fn mat9x6_mul_6x9(a: &[f64; 54], b: &[f64; 54]) -> [f64; 81] {
    let mut out = [0.0f64; 81];
    for c in 0..9 {
        for r in 0..9 {
            let mut sum = 0.0;
            for k in 0..6 {
                sum += a[r + k * 9] * b[k + c * 6];
            }
            out[idx9(r, c)] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

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
            assert_eq!(pre.covariance[i + i * 9], 0.0);
        }

        // After some integration, diagonal should be positive
        for _ in 0..100 {
            pre.integrate(&m, 0.01);
        }
        for i in 0..9 {
            assert!(
                pre.covariance[i + i * 9] > 0.0,
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
                pre.covariance[i + i * 9] >= cov_after_100[i + i * 9],
                "diagonal {i} should not decrease"
            );
        }
    }

    /// Preintegration must invert the forward sensor model **exactly** on a
    /// closed-form trajectory: constant body-frame angular velocity and constant
    /// world-frame acceleration.
    ///
    /// ```text
    /// R_wb(t) = R0·Exp(ω·t)    v(t) = v0 + a·t    p(t) = p0 + v0·t + ½a·t²
    /// ⇒  ΔR = Exp(ω·Δt)   Δv = R_bw0·(a−g)·Δt   Δp = R_bw0·½(a−g)·Δt²
    /// ```
    ///
    /// Exactness here is not an accident, and it is the point of the test. At
    /// step `k` the integrator forms `ΔR_k · accel_k`, and since
    /// `ΔR_k = R_bw0·R_wb(t_k)` while `accel_k = R_wb(t_k)ᵀ·(a−g)`, the two
    /// rotations cancel to leave the constant `R_bw0·(a−g)`. Summing a constant
    /// has no discretization error, so the result is exact to machine precision
    /// at any rate.
    ///
    /// That cancellation is fragile in exactly the useful way: it only holds if
    /// the rotation handling is right. A transposed `ΔR`, a flipped gyro sign,
    /// or `hat(ΔR·a)` in place of `ΔR·hat(a)` all break it and show up as a
    /// large error rather than a subtle one. First-order convergence is checked
    /// separately by `preintegration_converges_at_first_order`, which uses a
    /// trajectory where the cancellation does not occur.
    #[test]
    fn preintegration_inverts_the_forward_model() {
        let gravity = Vec3F64::new(0.0, 9.81, 0.0);
        let omega = Vec3F64::new(0.3, -0.5, 0.2); // body-frame, constant
        let accel_world = Vec3F64::new(0.4, -0.25, 0.6); // excludes gravity
        let r0 = SO3F64::exp(Vec3F64::new(0.2, 0.1, -0.3));
        let total = 0.5;

        for rate_hz in [100.0f64, 400.0, 1600.0] {
            let dt = 1.0 / rate_hz;
            let n = (total / dt).round() as usize;

            let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
            for k in 0..n {
                let t = k as f64 * dt;
                let r_wb = r0 * SO3F64::exp(omega * t);
                let m = ImuMeasurement::simulate(
                    t,
                    &r_wb.matrix(),
                    accel_world,
                    omega,
                    gravity,
                    &ImuBias::default(),
                );
                pre.integrate(&m, dt);
            }

            let r_bw0 = Mat3F64(*r0.matrix().transpose());
            let net = accel_world - gravity;

            let rot = (SO3F64::exp(omega * total).inverse()
                * SO3F64::from_matrix(&pre.delta_rotation))
            .log()
            .length();
            let vel = (pre.delta_velocity - r_bw0 * net * total).length();
            let pos = (pre.delta_position - r_bw0 * net * (0.5 * total * total)).length();

            assert!(rot < 1e-12, "rotation error {rot} rad at {rate_hz} Hz");
            assert!(vel < 1e-10, "velocity error {vel} m/s at {rate_hz} Hz");
            assert!(pos < 1e-10, "position error {pos} m at {rate_hz} Hz");
        }
    }

    /// Preintegration error must fall at first order once the exact
    /// cancellation of `preintegration_inverts_the_forward_model` is broken.
    ///
    /// Adding a constant jerk makes world acceleration time-varying, so
    /// `ΔR_k · accel_k` is no longer constant and the rectangular rule in
    /// `integrate` — which advances `Δv += ΔR·a·dt` using `ΔR` from the start of
    /// each step — leaves a genuine `O(dt)` residual. Ground truth stays closed
    /// form:
    ///
    /// ```text
    /// a(t) = a0 + j·t   v(t) = v0 + a0·t + ½j·t²   p(t) = p0 + v0·t + ½a0·t² + ⅙j·t³
    /// ```
    ///
    /// Halving the timestep must halve the error. This is what discriminates a
    /// discretization residual from a modelling error: a wrong frame or sign
    /// gives a constant error that no rate increase removes.
    #[test]
    fn preintegration_converges_at_first_order() {
        let gravity = Vec3F64::new(0.0, 9.81, 0.0);
        let omega = Vec3F64::new(0.3, -0.5, 0.2);
        let a0 = Vec3F64::new(0.4, -0.25, 0.6);
        let jerk = Vec3F64::new(1.5, 2.0, -1.2);
        let r0 = SO3F64::exp(Vec3F64::new(0.2, 0.1, -0.3));
        let total = 0.5;

        let error_at = |rate_hz: f64| -> (f64, f64) {
            let dt = 1.0 / rate_hz;
            let n = (total / dt).round() as usize;

            let mut pre = PreintegratedImu::new(ImuBias::default(), default_calib());
            for k in 0..n {
                let t = k as f64 * dt;
                let r_wb = r0 * SO3F64::exp(omega * t);
                let m = ImuMeasurement::simulate(
                    t,
                    &r_wb.matrix(),
                    a0 + jerk * t,
                    omega,
                    gravity,
                    &ImuBias::default(),
                );
                pre.integrate(&m, dt);
            }

            let r_bw0 = Mat3F64(*r0.matrix().transpose());
            let (t1, t2, t3) = (total, total * total, total * total * total);

            // Δv = R_bw0·(v1 − v0 − g·Δt), with v1 − v0 = a0·Δt + ½j·Δt².
            let expected_dv = r_bw0 * (a0 * t1 + jerk * (0.5 * t2) - gravity * t1);
            // Δp = R_bw0·(p1 − p0 − v0·Δt − ½g·Δt²).
            let expected_dp = r_bw0 * (a0 * (0.5 * t2) + jerk * (t3 / 6.0) - gravity * (0.5 * t2));

            (
                (pre.delta_velocity - expected_dv).length(),
                (pre.delta_position - expected_dp).length(),
            )
        };

        let coarse = error_at(200.0);
        let fine = error_at(400.0);
        let finer = error_at(800.0);

        for (name, c, f, ff) in [
            ("velocity", coarse.0, fine.0, finer.0),
            ("position", coarse.1, fine.1, finer.1),
        ] {
            assert!(ff > 0.0, "{name} error vanished — trajectory is degenerate");
            let (r1, r2) = (c / f, f / ff);
            assert!(
                (1.6..2.6).contains(&r1),
                "{name} error ratio 200->400 Hz is {r1}, expected ~2 (first order)"
            );
            assert!(
                (1.6..2.6).contains(&r2),
                "{name} error ratio 400->800 Hz is {r2}, expected ~2 (first order)"
            );
        }
    }

    /// A level, stationary body reads `-g`, and bias adds straight through.
    #[test]
    fn forward_model_matches_hand_computed_readings() {
        let gravity = Vec3F64::new(0.0, 9.81, 0.0);
        let level = Mat3F64::IDENTITY;

        let at_rest = ImuMeasurement::simulate(
            0.0,
            &level,
            Vec3F64::ZERO,
            Vec3F64::ZERO,
            gravity,
            &ImuBias::default(),
        );
        assert!((at_rest.accel - Vec3F64::new(0.0, -9.81, 0.0)).length() < EPS);
        assert!(at_rest.gyro.length() < EPS);

        // Free fall: specific force is zero.
        let falling = ImuMeasurement::simulate(
            0.0,
            &level,
            gravity,
            Vec3F64::ZERO,
            gravity,
            &ImuBias::default(),
        );
        assert!(falling.accel.length() < EPS, "free fall should read zero");

        // Bias is additive in both channels.
        let bias = ImuBias {
            gyro: Vec3F64::new(0.01, -0.02, 0.03),
            accel: Vec3F64::new(0.1, 0.2, -0.3),
        };
        let biased =
            ImuMeasurement::simulate(0.0, &level, Vec3F64::ZERO, Vec3F64::ZERO, gravity, &bias);
        assert!((biased.accel - at_rest.accel - bias.accel).length() < EPS);
        assert!((biased.gyro - at_rest.gyro - bias.gyro).length() < EPS);
    }

    /// Rotating the body rotates the measured gravity vector into the body
    /// frame — the property a wrong transpose would silently invert.
    #[test]
    fn forward_model_expresses_gravity_in_the_body_frame() {
        let gravity = Vec3F64::new(0.0, 9.81, 0.0);
        // Roll 90 degrees about body x: world +Y (down) maps to body -Z.
        let r_wb = SO3F64::exp(Vec3F64::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
        let m = ImuMeasurement::simulate(
            0.0,
            &r_wb.matrix(),
            Vec3F64::ZERO,
            Vec3F64::ZERO,
            gravity,
            &ImuBias::default(),
        );
        // accel = R_bw·(0 − g) = R_wbᵀ·(−g)
        let expected = Mat3F64(*r_wb.matrix().transpose()) * -gravity;
        assert!(
            (m.accel - expected).length() < EPS,
            "got {:?}, expected {expected:?}",
            m.accel
        );
        // Concretely: −g along world +Y becomes +Z in the rolled body frame.
        assert!((m.accel.z - 9.81).abs() < 1e-6, "got {:?}", m.accel);
    }
}
