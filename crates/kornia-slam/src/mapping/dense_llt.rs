//! Dense Cholesky factor and solve of the local VI-BA's reduced system, on an
//! explicit [`faer::Par`].
//!
//! faer's `Mat::llt` and `Llt::solve` always read the process-wide
//! parallelism. `DenseLlt` calls the same low-level routines with the same
//! inputs (a zero-upper copy of the lower triangle, right-hand sides held in a
//! `Mat`), so with [`DenseParallelism::Global`] the result is faer's bit for
//! bit. It also keeps the factor, its scratch and the solve column across
//! iterations instead of allocating them each time.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{LltError, cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::cholesky::llt::solve::{solve_in_place_scratch, solve_in_place_with_conj};
use faer::{Conj, Mat, Par};

/// Threads the local VI-BA factors and solves its reduced system on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DenseParallelism {
    /// faer's process-wide setting: every core unless the process changed it.
    /// The offline global VI-BA sets it to sequential, so after one has run
    /// in this process `Global` is sequential too.
    #[default]
    Global,
    /// The calling thread only.
    Sequential,
}

impl DenseParallelism {
    pub(crate) fn par(self) -> Par {
        match self {
            Self::Global => faer::get_global_parallelism(),
            Self::Sequential => Par::Seq,
        }
    }
}

/// A reused lower Cholesky factor of a `dim`×`dim` matrix.
pub(crate) struct DenseLlt {
    par: Par,
    l: Mat<f64>,
    column: Mat<f64>,
    scratch: MemBuffer,
}

impl DenseLlt {
    pub(crate) fn new(dim: usize, par: Par) -> Self {
        Self {
            par,
            l: Mat::zeros(dim, dim),
            column: Mat::zeros(dim, 1),
            scratch: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                dim,
                par,
                Default::default(),
            )),
        }
    }

    /// Factors the lower triangle of `a`. After an `Err` the factor is
    /// garbage: factor again before any solve.
    pub(crate) fn factor(&mut self, a: &Mat<f64>) -> Result<(), LltError> {
        // As `Llt::new`: a zeroed matrix holding only the lower triangle.
        self.l.fill(0.0);
        self.l.copy_from_triangular_lower(a);
        cholesky_in_place(
            self.l.as_mut(),
            Default::default(),
            self.par,
            MemStack::new(&mut self.scratch),
            Default::default(),
        )?;
        Ok(())
    }

    /// Overwrites `rhs` with `A⁻¹ rhs`.
    pub(crate) fn solve_in_place(&self, rhs: &mut Mat<f64>) {
        solve(&self.l, self.par, rhs);
    }

    /// `A⁻¹ b`, in a reused column. It stays a `Mat`: faer's kernels can
    /// depend on alignment, and faer's own solve runs on a `Mat`.
    pub(crate) fn solve_column(&mut self, b: &[f64]) -> &[f64] {
        self.column.col_as_slice_mut(0).copy_from_slice(b);
        solve(&self.l, self.par, &mut self.column);
        self.column.col_as_slice(0)
    }
}

/// As faer's `SolveCore for Llt`, on `par`.
fn solve(l: &Mat<f64>, par: Par, rhs: &mut Mat<f64>) {
    // Zero-size: no heap allocation.
    let mut mem = MemBuffer::new(solve_in_place_scratch::<f64>(l.nrows(), rhs.ncols(), par));
    solve_in_place_with_conj(
        l.as_ref(),
        Conj::No,
        rhs.as_mut(),
        par,
        MemStack::new(&mut mem),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Side;
    use faer::prelude::Solve;

    const N: usize = 150;

    /// `BᵀB + N·I` from a fixed LCG: symmetric positive definite.
    fn spd() -> Mat<f64> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let b = Mat::<f64>::from_fn(N, N, |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        let mut a = b.transpose() * &b;
        for i in 0..N {
            a[(i, i)] += N as f64;
        }
        a
    }

    fn rhs_column() -> Mat<f64> {
        Mat::from_fn(N, 1, |i, _| (i as f64 * 0.37).sin())
    }

    fn unit_block() -> Mat<f64> {
        let mut unit = Mat::zeros(N, 15);
        for c in 0..15 {
            unit[(60 + c, c)] = 1.0;
        }
        unit
    }

    fn bits(m: &Mat<f64>) -> Vec<u64> {
        (0..m.ncols())
            .flat_map(|c| m.col_as_slice(c).iter().map(|v| v.to_bits()))
            .collect()
    }

    fn explicit(a: &Mat<f64>, par: Par) -> (Vec<u64>, Vec<u64>) {
        let mut dense = DenseLlt::new(N, par);
        dense.factor(a).unwrap();
        let column = dense.solve_column(rhs_column().col_as_slice(0)).to_vec();
        let mut unit = unit_block();
        dense.solve_in_place(&mut unit);
        (column.iter().map(|v| v.to_bits()).collect(), bits(&unit))
    }

    #[test]
    fn the_explicit_solve_is_faers_solve_bit_for_bit() {
        let a = spd();
        let chol = a.llt(Side::Lower).unwrap();
        let faer_column = bits(&chol.solve(&rhs_column()));
        let faer_unit = bits(&chol.solve(&unit_block()));
        let (column, unit) = explicit(&a, faer::get_global_parallelism());
        assert_eq!(column, faer_column);
        assert_eq!(unit, faer_unit);
    }

    #[test]
    fn sequential_and_threaded_factor_to_the_same_bits() {
        let a = spd();
        assert_eq!(explicit(&a, Par::Seq), explicit(&a, Par::rayon(0)));
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_is_refused() {
        let a = spd();
        let mut indefinite = a.clone();
        indefinite[(N / 2, N / 2)] = -1.0;
        let mut dense = DenseLlt::new(N, Par::Seq);
        assert!(dense.factor(&indefinite).is_err());
        dense.factor(&a).unwrap();
        let column: Vec<u64> = dense
            .solve_column(rhs_column().col_as_slice(0))
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(column, explicit(&a, Par::Seq).0);
    }
}
