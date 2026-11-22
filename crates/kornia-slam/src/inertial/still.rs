//! Stillness from the IMU stream: the still run since a keyframe, judged in
//! chunks against the noise model.
//!
//! Ported from c95203b (cu-kornia-vio's parked policy), with its hand-set
//! ceilings replaced by statistical tests at one confidence level, after
//! OpenVINS `UpdaterZeroVelocity.cpp:150-152, :206-214`. Each chunk's mean is
//! tested for constancy against the run's first chunk, so no bias is needed;
//! the variance of a mean is the larger of the modelled and the measured
//! sample variance over the sample count, so vibration widens the test
//! instead of failing it. A chunk whose spread is above the model's band is
//! `Unknown`: not moving, but not vouching for rest either.

use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

use super::ImuWindowStats;

/// Length of the chunks a run is judged in, seconds (cu-kornia-vio `parked.rs:73`).
pub(crate) const RUN_CHUNK_SEC: f64 = 0.5;

/// Standard normal quantile of the 99.9 % one-sided confidence every
/// stillness test uses.
pub(crate) const Z_999: f64 = 3.09;

/// χ²₃ at 99.9 %: the constancy test of a 3-axis mean.
const CHI2_3_999: f64 = 16.27;

/// Fewest samples in a judged chunk.
const MIN_CHUNK_SAMPLES: usize = 10;

/// A gap longer than this many sample periods ends a run: nothing vouches for it.
const HOLE_PERIODS: f64 = 5.0;

/// What the IMU says about the run so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImuVote {
    /// No chunk judged yet.
    Pending,
    /// Every chunk constant and within the model's spread.
    Quiet,
    /// Constant, but some chunk's spread is above the model (vibration).
    Unknown,
    /// A test failed; the run is closed.
    Moving(ImuCloser),
}

/// The test that closed a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImuCloser {
    GyroMean,
    AccelMean,
    GyroView,
    Hole,
}

impl ImuCloser {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::GyroMean => "imu_gyro",
            Self::AccelMean => "imu_accel",
            Self::GyroView => "gyro_view",
            Self::Hole => "imu_hole",
        }
    }
}

/// The view check that needs the gyro bias: the rotation integrated since
/// the keyframe, bias removed, must stay under `max_rad`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GyroView {
    pub(crate) bias: Vec3F64,
    pub(crate) max_rad: f64,
}

/// A chunk's means and the per-axis variance of each mean.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ChunkMean {
    gyro: Vec3F64,
    accel: Vec3F64,
    gyro_var: f64,
    accel_var: f64,
    quiet: bool,
}

impl ChunkMean {
    /// Per-axis sample variance `max(density² · rate, measured)`, over `n` for
    /// the mean. `quiet` when both measured variances sit inside the 99.9 %
    /// band of the model's (3(n-1) degrees of freedom, normal approximation).
    fn new(stats: &ImuWindowStats, calib: &ImuCalib, rate_hz: f64) -> Self {
        let n = stats.samples as f64;
        let band = 1.0 + Z_999 * (2.0 / (3.0 * (n - 1.0))).sqrt();
        let axis_var = |rms_dev: f64, density: f64| {
            let model = density * density * rate_hz;
            let measured = rms_dev * rms_dev / 3.0;
            (model.max(measured) / n, measured <= model * band)
        };
        let (gyro_var, gyro_quiet) = axis_var(stats.gyro_rms_dev, calib.gyro_noise);
        let (accel_var, accel_quiet) = axis_var(stats.accel_rms_dev, calib.accel_noise);
        Self {
            gyro: stats.mean_gyro,
            accel: stats.mean_accel,
            gyro_var,
            accel_var,
            quiet: gyro_quiet && accel_quiet,
        }
    }

    /// The first failing constancy test against `reference`, with its statistic.
    fn moved_from(&self, reference: &Self) -> Option<(ImuCloser, f64)> {
        let stat = |a: Vec3F64, b: Vec3F64, va: f64, vb: f64| {
            let d = a - b;
            d.dot(d) / (va + vb)
        };
        let gyro = stat(self.gyro, reference.gyro, self.gyro_var, reference.gyro_var);
        let accel = stat(
            self.accel,
            reference.accel,
            self.accel_var,
            reference.accel_var,
        );
        if !within(gyro, CHI2_3_999) {
            Some((ImuCloser::GyroMean, gyro))
        } else if !within(accel, CHI2_3_999) {
            Some((ImuCloser::AccelMean, accel))
        } else {
            None
        }
    }
}

/// `x <= max`, false for NaN.
fn within(x: f64, max: f64) -> bool {
    x <= max
}

/// Mean sample rate of `samples`, Hz; `None` for fewer than two or no time.
pub(crate) fn sample_rate_hz(samples: &[ImuMeasurement]) -> Option<f64> {
    let (first, last) = (samples.first()?, samples.last()?);
    let span = last.timestamp - first.timestamp;
    (samples.len() >= 2 && span > 0.0).then(|| (samples.len() - 1) as f64 / span)
}

/// Whether consecutive samples are ever further apart than `max_gap_sec`.
pub(crate) fn has_hole(samples: &[ImuMeasurement], max_gap_sec: f64) -> bool {
    samples.windows(2).any(|w| {
        let gap = w[1].timestamp - w[0].timestamp;
        gap.is_nan() || gap > max_gap_sec
    })
}

/// The still stretch since one keyframe, verified chunk by chunk as samples
/// arrive. Once closed it stays closed; the next keyframe starts a new run.
#[derive(Debug, Clone)]
pub(crate) struct StillRun {
    /// `Frame::idx` of the keyframe the run starts at.
    pub(crate) kf_idx: usize,
    /// The samples are verified up to here.
    end_sec: f64,
    reference: Option<ChunkMean>,
    vote: ImuVote,
    /// `∫(ω - b̂_g) dt` over the judged chunks, rad; only with a known bias.
    rotation: Vec3F64,
    /// The statistic and threshold of the test that closed the run.
    closing: Option<(f64, f64)>,
}

impl StillRun {
    pub(crate) fn new(kf_idx: usize, start_sec: f64) -> Self {
        Self {
            kf_idx,
            end_sec: start_sec,
            reference: None,
            vote: ImuVote::Pending,
            rotation: Vec3F64::ZERO,
            closing: None,
        }
    }

    pub(crate) fn vote(&self) -> ImuVote {
        self.vote
    }

    pub(crate) fn is_open(&self) -> bool {
        !matches!(self.vote, ImuVote::Moving(_))
    }

    /// The newest stamp the samples were judged to; frozen once the run closes.
    pub(crate) fn end_sec(&self) -> f64 {
        self.end_sec
    }

    /// The statistic and threshold that closed the run, if a test did.
    pub(crate) fn closing(&self) -> Option<(f64, f64)> {
        self.closing
    }

    /// Whether the IMU vouches for rest at `t`: every chunk quiet and `t`
    /// within one chunk of the judged end.
    pub(crate) fn quiet_at(&self, t: f64) -> bool {
        self.vote == ImuVote::Quiet && t <= self.end_sec + RUN_CHUNK_SEC
    }

    /// Judges the samples after the judged end once a chunk's worth has
    /// arrived. `samples` must hold every buffered sample from there on, in
    /// time order. `view` is the gyro view check, when the bias is known.
    pub(crate) fn advance(
        &mut self,
        samples: &[ImuMeasurement],
        calib: &ImuCalib,
        view: Option<GyroView>,
    ) {
        if !self.is_open() {
            return;
        }
        let from = self.end_sec;
        let Some(newest) = samples.last().map(|m| m.timestamp) else {
            return;
        };
        if newest < from + RUN_CHUNK_SEC {
            return;
        }
        let start = samples.partition_point(|m| m.timestamp < from);
        let chunk = &samples[start..];
        match judge_chunk(chunk, from, calib) {
            Some(mean) => self.accept(mean, newest - from, view),
            None => self.close(ImuCloser::Hole, None),
        }
        self.end_sec = newest;
    }

    fn accept(&mut self, mean: ChunkMean, span: f64, view: Option<GyroView>) {
        let reference = *self.reference.get_or_insert(mean);
        if let Some((closer, stat)) = mean.moved_from(&reference) {
            return self.close(closer, Some((stat, CHI2_3_999)));
        }
        if let Some(view) = view {
            self.rotation += (mean.gyro - view.bias) * span;
            let rad = self.rotation.length();
            if !within(rad, view.max_rad) {
                return self.close(ImuCloser::GyroView, Some((rad, view.max_rad)));
            }
        }
        self.vote = match (self.vote, mean.quiet) {
            (ImuVote::Pending | ImuVote::Quiet, true) => ImuVote::Quiet,
            _ => ImuVote::Unknown,
        };
    }

    fn close(&mut self, closer: ImuCloser, closing: Option<(f64, f64)>) {
        self.vote = ImuVote::Moving(closer);
        self.closing = closing;
    }
}

/// A chunk's means, or `None` when it has too few samples or a hole (a gap
/// over [`HOLE_PERIODS`] sample periods, the start gap included).
fn judge_chunk(chunk: &[ImuMeasurement], from: f64, calib: &ImuCalib) -> Option<ChunkMean> {
    if chunk.len() < MIN_CHUNK_SAMPLES {
        return None;
    }
    let rate = sample_rate_hz(chunk)?;
    let max_gap = HOLE_PERIODS / rate;
    let start_gap = chunk.first().map_or(f64::INFINITY, |m| m.timestamp - from);
    if !within(start_gap, max_gap) || has_hole(chunk, max_gap) {
        return None;
    }
    let stats = ImuWindowStats::from_samples(chunk)?;
    Some(ChunkMean::new(&stats, calib, rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 200.0;
    const GRAVITY: f64 = 9.81;

    /// EuRoC's ADIS16448 densities.
    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1.6968e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9393e-5,
            accel_bias_noise: 3.0e-3,
        }
    }

    /// Deterministic standard normals (xorshift and Box-Muller).
    struct Normals(u64);

    impl Normals {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn next(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }

        fn vec(&mut self, sigma: f64) -> Vec3F64 {
            Vec3F64::new(self.next(), self.next(), self.next()) * sigma
        }
    }

    /// Model-noise samples at rest over `[from, to)` seconds: gravity on z, a
    /// gyro bias, plus `extra(t)` as (accel, gyro).
    fn imu(
        to: f64,
        gyro_bias: Vec3F64,
        extra: impl Fn(f64) -> (Vec3F64, Vec3F64),
    ) -> Vec<ImuMeasurement> {
        let c = calib();
        let mut normals = Normals(0x9e37_79b9_7f4a_7c15);
        let n = (to * RATE).round() as usize;
        (0..n)
            .map(|k| {
                let t = k as f64 / RATE;
                let (ea, eg) = extra(t);
                ImuMeasurement {
                    timestamp: t,
                    gyro: gyro_bias + normals.vec(c.gyro_noise * RATE.sqrt()) + eg,
                    accel: Vec3F64::new(0.0, 0.0, GRAVITY)
                        + normals.vec(c.accel_noise * RATE.sqrt())
                        + ea,
                }
            })
            .collect()
    }

    fn rest(_: f64) -> (Vec3F64, Vec3F64) {
        (Vec3F64::ZERO, Vec3F64::ZERO)
    }

    /// Feeds the samples in 0.05 s batches, as 20 Hz frames would.
    fn run_over(samples: &[ImuMeasurement], view: Option<GyroView>) -> StillRun {
        let mut run = StillRun::new(1, 0.0);
        for end in (10..=samples.len()).step_by(10) {
            run.advance(&samples[..end], &calib(), view);
        }
        run
    }

    #[test]
    fn model_noise_at_rest_is_quiet() {
        let run = run_over(&imu(10.0, Vec3F64::ZERO, rest), None);
        assert_eq!(run.vote(), ImuVote::Quiet);
        assert!(run.end_sec() > 9.4, "{}", run.end_sec());
        assert!(run.quiet_at(10.0));
    }

    #[test]
    fn vibration_widens_the_test_and_abstains() {
        let w = std::f64::consts::TAU * 38.0;
        let vibrate = |t: f64| {
            let s = (w * t).sin();
            (
                Vec3F64::new(0.179 * s, 0.0, 0.0),
                Vec3F64::new(0.0, 0.053 * s, 0.0),
            )
        };
        let run = run_over(&imu(10.0, Vec3F64::ZERO, vibrate), None);
        assert_eq!(run.vote(), ImuVote::Unknown);
        assert!(!run.quiet_at(5.0));
    }

    #[test]
    fn a_tilt_and_a_shove_are_moving() {
        let rate = 2f64.to_radians();
        // 2 deg about x over 1 s from t = 3 s: gravity swings in y-z.
        let tilt = |t: f64| {
            let angle = rate * (t - 3.0).clamp(0.0, 1.0);
            let on = (3.0..4.0).contains(&t);
            (
                Vec3F64::new(0.0, GRAVITY * angle.sin(), GRAVITY * (angle.cos() - 1.0)),
                Vec3F64::new(if on { rate } else { 0.0 }, 0.0, 0.0),
            )
        };
        let shove = |t: f64| {
            let on = (3.0..3.2).contains(&t);
            (
                Vec3F64::new(if on { 0.5 } else { 0.0 }, 0.0, 0.0),
                Vec3F64::ZERO,
            )
        };
        for (extra, closer) in [
            (
                &tilt as &dyn Fn(f64) -> (Vec3F64, Vec3F64),
                ImuCloser::GyroMean,
            ),
            (&shove, ImuCloser::AccelMean),
        ] {
            let run = run_over(&imu(10.0, Vec3F64::ZERO, extra), None);
            assert_eq!(run.vote(), ImuVote::Moving(closer));
            assert!((3.0..=4.6).contains(&run.end_sec()), "{}", run.end_sec());
        }
    }

    /// EuRoC MH_01 rests with a 0.082 rad/s gyro bias, which a hand-set mean
    /// ceiling (c95203b's 0.02 rad/s) took for a turn.
    #[test]
    fn an_unknown_bias_at_rest_is_not_moving() {
        let bias = Vec3F64::new(0.08, 0.0, 0.0);
        let run = run_over(&imu(10.0, bias, rest), None);
        assert_eq!(run.vote(), ImuVote::Quiet);
    }

    #[test]
    fn a_slow_turn_trips_the_view_check_once_the_bias_is_known() {
        let bias = Vec3F64::new(0.08, -0.01, 0.02);
        let turn = |_: f64| (Vec3F64::ZERO, Vec3F64::new(0.0, 0.0, 0.01));
        let samples = imu(10.0, bias, turn);
        assert!(
            run_over(&samples, None).is_open(),
            "constant: no view check"
        );
        let view = GyroView {
            bias,
            max_rad: 2.35 / 400.0,
        };
        let run = run_over(&samples, Some(view));
        assert_eq!(run.vote(), ImuVote::Moving(ImuCloser::GyroView));
        assert!(run.end_sec() < 1.6, "{}", run.end_sec());
        let (rad, max) = run.closing().expect("statistic recorded");
        assert!(rad > max);
        assert!(run_over(&imu(10.0, bias, rest), Some(view)).is_open());
    }

    #[test]
    fn a_sample_hole_closes_the_run() {
        let mut samples = imu(4.0, Vec3F64::ZERO, rest);
        let later = imu(8.0, Vec3F64::ZERO, rest);
        samples.extend(later.into_iter().filter(|m| m.timestamp >= 4.1));
        let run = run_over(&samples, None);
        assert_eq!(run.vote(), ImuVote::Moving(ImuCloser::Hole));
        assert!(!run.quiet_at(4.5));
    }
}
