//! Deterministic pseudo-random generation for the simulator.
//!
//! Every simulated quantity must be reproducible from a recorded seed — a
//! recovery test that fails on one run and passes on the next is worse than no
//! test, because it teaches you to ignore it.
//!
//! The generator itself is `rand`'s [`StdRng`] (ChaCha12), seeded via
//! [`SeedableRng::seed_from_u64`], which is reproducible across platforms and
//! across runs. That is the same pattern kornia-3d already uses in its RANSAC
//! drivers, so the simulator does not introduce a second notion of "seeded
//! randomness" into the stack.
//!
//! What this type adds on top is only the **distributions** the simulator needs
//! and `rand` does not carry in its core crate: a Gaussian and a uniform
//! direction on the sphere. Those are a dozen lines; `rand_distr` is not in the
//! workspace and is not worth adding for them.

use kornia_algebra::Vec3F64;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A seeded, reproducible random source.
///
/// The same seed always yields the same sequence, on every platform.
///
/// Deliberately not `Clone`: `StdRng` is not, and a cloned generator would
/// silently replay the same stream into two places — a way to make a "seeded"
/// run quietly non-independent.
#[derive(Debug)]
pub struct SimRng {
    inner: StdRng,
    /// Box-Muller produces two independent normals per call; the spare is kept
    /// here so no sample is wasted.
    spare_normal: Option<f64>,
}

impl SimRng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
            spare_normal: None,
        }
    }

    /// Raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.random()
    }

    /// Uniform sample in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.inner.random()
    }

    /// Uniform sample in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        self.inner.random_range(lo..hi)
    }

    /// Standard normal sample, `N(0, 1)`, via the polar Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        // Rejection-sample a point inside the unit disc. The polar form avoids
        // the sin/cos of the basic transform and is numerically better behaved.
        let (u, v, s) = loop {
            let u = self.uniform_range(-1.0, 1.0);
            let v = self.uniform_range(-1.0, 1.0);
            let s = u * u + v * v;
            // s == 0 would divide by zero below; s >= 1 is outside the disc.
            if s > 0.0 && s < 1.0 {
                break (u, v, s);
            }
        };
        let factor = (-2.0 * s.ln() / s).sqrt();
        self.spare_normal = Some(v * factor);
        u * factor
    }

    /// Zero-mean normal sample with the given standard deviation.
    pub fn normal_scaled(&mut self, std_dev: f64) -> f64 {
        self.normal() * std_dev
    }

    /// Isotropic zero-mean Gaussian vector with per-axis standard deviation
    /// `std_dev`.
    pub fn normal_vec3(&mut self, std_dev: f64) -> Vec3F64 {
        Vec3F64::new(
            self.normal_scaled(std_dev),
            self.normal_scaled(std_dev),
            self.normal_scaled(std_dev),
        )
    }

    /// A direction sampled uniformly over the unit sphere.
    ///
    /// Uses the cylindrical-projection method (Archimedes): sampling `z`
    /// uniformly in `[-1, 1]` and the azimuth uniformly gives equal-area
    /// coverage. Naively sampling spherical angles would instead bunch samples
    /// at the poles and quietly bias the landmark distribution.
    pub fn unit_vec3(&mut self) -> Vec3F64 {
        let z = self.uniform_range(-1.0, 1.0);
        let azimuth = self.uniform_range(0.0, std::f64::consts::TAU);
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3F64::new(r * azimuth.cos(), r * azimuth.sin(), z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = SimRng::new(42);
        let mut b = SimRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SimRng::new(1);
        let mut b = SimRng::new(2);
        // Small seeds must not produce correlated streams — `seed_from_u64`
        // runs the seed through a hash before keying the generator.
        let differing = (0..100).filter(|_| a.next_u64() != b.next_u64()).count();
        assert_eq!(differing, 100);
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = SimRng::new(7);
        for _ in 0..10_000 {
            let x = rng.uniform();
            assert!((0.0..1.0).contains(&x), "uniform out of range: {x}");
        }
    }

    #[test]
    fn normal_matches_requested_moments() {
        let mut rng = SimRng::new(11);
        let n = 200_000;
        let std_dev = 2.5;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal_scaled(std_dev)).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;

        // Standard error of the mean is std/sqrt(n) ~= 0.0056; 4 sigma is a
        // generous but non-vacuous bound for a deterministic seed.
        assert!(mean.abs() < 0.03, "mean {mean} too far from 0");
        assert!(
            (variance.sqrt() - std_dev).abs() < 0.03,
            "std {} too far from {std_dev}",
            variance.sqrt()
        );
    }

    #[test]
    fn unit_vec3_is_normalized_and_unbiased() {
        let mut rng = SimRng::new(13);
        let n = 50_000;
        let mut sum = Vec3F64::ZERO;
        for _ in 0..n {
            let v = rng.unit_vec3();
            assert!((v.length() - 1.0).abs() < 1e-12, "not unit length");
            sum += v;
        }
        // A uniform sphere distribution has zero mean; a pole-biased sampler
        // would show a clear z drift here.
        let mean = sum * (1.0 / n as f64);
        assert!(mean.length() < 0.02, "directional bias: {mean:?}");
    }
}
