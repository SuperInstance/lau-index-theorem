//! Analytic index: dim(ker D) - dim(coker D).
//!
//! The analytic index of an elliptic operator D is:
//!   ind_a(D) = dim(ker D) - dim(coker D) = dim(ker D) - dim(ker D*)

use nalgebra::{DMatrix, DVector};
use crate::types::EllipticOperator;

/// Compute the kernel of a matrix via SVD.
/// Returns basis vectors for the kernel (null space).
pub fn kernel_basis(m: &DMatrix<f64>, tol: f64) -> Vec<DVector<f64>> {
    let svd = m.clone().svd(true, true);

    let mut basis = Vec::new();
    if let Some(v_t) = &svd.v_t {
        let singular_values = &svd.singular_values;
        for j in 0..m.ncols() {
            let sv = if j < singular_values.len() { singular_values[j] } else { 0.0 };
            if sv < tol {
                let col = v_t.row(j);
                basis.push(DVector::from_vec(col.iter().cloned().collect()));
            }
        }
    }
    basis
}

/// Compute the cokernel of a matrix (kernel of its adjoint).
pub fn cokernel_basis(m: &DMatrix<f64>, tol: f64) -> Vec<DVector<f64>> {
    kernel_basis(&m.transpose(), tol)
}

/// Dimension of the kernel.
pub fn dim_kernel(m: &DMatrix<f64>, tol: f64) -> usize {
    m.ncols() - m.rank(tol)
}

/// Dimension of the cokernel.
pub fn dim_cokernel(m: &DMatrix<f64>, tol: f64) -> usize {
    m.nrows() - m.rank(tol)
}

/// Compute the analytic index: dim(ker D) - dim(coker D).
pub fn analytic_index(op: &EllipticOperator, tol: f64) -> i64 {
    let m = op.matrix();
    let dk = dim_kernel(&m, tol) as i64;
    let dc = dim_cokernel(&m, tol) as i64;
    dk - dc
}

/// Compute the analytic index directly from a matrix.
pub fn analytic_index_matrix(m: &DMatrix<f64>, tol: f64) -> i64 {
    let dk = dim_kernel(m, tol) as i64;
    let dc = dim_cokernel(m, tol) as i64;
    dk - dc
}

/// A matrix operator is Fredholm iff it has finite-dimensional kernel and cokernel.
pub fn is_fredholm(op: &EllipticOperator, tol: f64) -> bool {
    let m = op.matrix();
    let dk = dim_kernel(&m, tol);
    let dc = dim_cokernel(&m, tol);
    dk < 1000 && dc < 1000 // Finite-dimensional check
}

/// Compute the Fredholm index for a matrix.
pub fn fredholm_index(m: &DMatrix<f64>, tol: f64) -> i64 {
    analytic_index_matrix(m, tol)
}

/// Trace of the identity on the kernel projection.
/// tr(id)|_{ker D} = dim(ker D)
pub fn trace_identity_kernel(op: &EllipticOperator, tol: f64) -> f64 {
    let m = op.matrix();
    dim_kernel(&m, tol) as f64
}

/// Construct the projection onto the kernel.
pub fn kernel_projection(m: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let basis = kernel_basis(m, tol);
    if basis.is_empty() {
        return DMatrix::zeros(m.ncols(), m.ncols());
    }
    let n = m.ncols();
    let mut proj = DMatrix::zeros(n, n);
    for v in &basis {
        for i in 0..n {
            for j in 0..n {
                proj[(i, j)] += v[i] * v[j];
            }
        }
    }
    proj
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kernel_identity_empty() {
        let m = DMatrix::identity(3, 3);
        let ker = kernel_basis(&m, 1e-8);
        assert!(ker.is_empty());
    }

    #[test]
    fn test_kernel_zero_full() {
        let m = DMatrix::zeros(3, 3);
        let ker = kernel_basis(&m, 1e-8);
        assert_eq!(ker.len(), 3);
    }

    #[test]
    fn test_kernel_rank_one() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let ker = kernel_basis(&m, 1e-8);
        assert_eq!(ker.len(), 2);
    }

    #[test]
    fn test_cokernel_identity_empty() {
        let m = DMatrix::identity(3, 3);
        let coker = cokernel_basis(&m, 1e-8);
        assert!(coker.is_empty());
    }

    #[test]
    fn test_analytic_index_square_invertible() {
        let m = DMatrix::identity(3, 3);
        let idx = analytic_index_matrix(&m, 1e-8);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_analytic_index_zero_matrix() {
        let m = DMatrix::zeros(3, 3);
        let idx = analytic_index_matrix(&m, 1e-8);
        assert_eq!(idx, 0); // dim ker = 3, dim coker = 3, index = 0
    }

    #[test]
    fn test_analytic_index_tall_matrix() {
        // 3×2, rank 2 → ker = 0, coker = 1, index = -1
        let m = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        let idx = analytic_index_matrix(&m, 1e-8);
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_analytic_index_wide_matrix() {
        // 2×3, rank 2 → ker = 1, coker = 0, index = 1
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        let idx = analytic_index_matrix(&m, 1e-8);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_is_fredholm() {
        let op = EllipticOperator::identity(3);
        assert!(is_fredholm(&op, 1e-8));
    }

    #[test]
    fn test_trace_identity_kernel() {
        let m = DMatrix::identity(3, 3);
        let op = EllipticOperator::new(m);
        let tr = trace_identity_kernel(&op, 1e-8);
        assert_abs_diff_eq!(tr, 0.0);
    }

    #[test]
    fn test_trace_identity_kernel_zero_op() {
        let m = DMatrix::zeros(3, 3);
        let op = EllipticOperator::new(m);
        let tr = trace_identity_kernel(&op, 1e-8);
        assert_abs_diff_eq!(tr, 3.0);
    }

    #[test]
    fn test_kernel_projection_is_idempotent() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let p = kernel_projection(&m, 1e-8);
        let p2 = &p * &p;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(p[(i, j)], p2[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_kernel_projection_trace_equals_dim() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let p = kernel_projection(&m, 1e-8);
        let tr: f64 = (0..3).map(|i| p[(i, i)]).sum();
        assert_abs_diff_eq!(tr, 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_analytic_index_operator() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);
        let op = EllipticOperator::new(m);
        assert_eq!(analytic_index(&op, 1e-8), 1);
    }
}
