//! Deterministic pseudo-random generation for the simulator.
//!
//! Every simulated quantity must be reproducible from a recorded seed — a
//! recovery test that fails on one run and passes on the next is worse than no
//! test, because it teaches you to ignore it.
//!
//! This is a self-contained xoshiro256++ rather than a `rand` dependency.
//! kornia-slam has no RNG dependency today, and the simulator needs exactly
//! three primitives (uniform, Gaussian, unit vector); pulling in `rand` plus
//! `rand_distr` to get them would be a poor trade. The generator is a standard,
//! well-tested design, not an invention — see the reference implementation by
//! Blackman and Vigna (<https://prng.di.unimi.it/>).

use kornia_algebra::Vec3F64;

/// A seeded, reproducible random source.
///
/// The same seed always yields the same sequence, on every platform: the state
/// update is pure integer arithmetic and the float conversion is an exact
/// power-of-two scaling, so there is no dependence on the host's floating-point
/// rounding mode or on libm.
#[derive(Debug, Clone)]
pub struct SimRng {
    state: [u64; 4],
    /// Box-Muller produces two independent normals per call; the spare is kept
    /// here so no sample is wasted.
    spare_normal: Option<f64>,
}

impl SimRng {
    /// Creates a generator from a seed.
    ///
    /// The seed is expanded through SplitMix64, so even low-entropy seeds
    /// (`0`, `1`, `2`, …) produce well-separated streams — which matters
    /// because test seeds are almost always small integers.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    /// Raw 64-bit output (xoshiro256++).
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform sample in `[0, 1)`.
    ///
    /// Takes the top 53 bits — the full mantissa width of an `f64` — so every
    /// representable value in the range is reachable with uniform probability.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform sample in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
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
        // Small seeds must not produce correlated streams; this is what the
        // SplitMix64 expansion in `new` is for.
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
