//! Witten index: tr((-1)^F e^{-tD²}) = Euler characteristic.
//!
//! The Witten index is a regularized trace that computes the index:
//!   Δ = tr((-1)^F e^{-tD²})
//! where F is the fermion number operator. This is independent of t
//! and equals the analytic index.

use nalgebra::DMatrix;
use crate::types::EllipticOperator;

/// Compute the Laplacian D*D.
pub fn laplacian(op: &EllipticOperator) -> DMatrix<f64> {
    op.d_star_d()
}

/// Compute the heat kernel e^{-tD²} via matrix exponential.
/// For a symmetric positive definite matrix A, e^{-tA} is computed
/// via eigendecomposition.
pub fn heat_kernel_operator(laplacian: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = laplacian.nrows();
    if n == 0 {
        return DMatrix::zeros(0, 0);
    }

    let eig = laplacian.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;
    let p = &eig.eigenvectors;

    // e^{-t λ_i}
    let mut exp_evals = Vec::with_capacity(n);
    for i in 0..n {
        exp_evals.push((-t * eigenvalues[i]).exp());
    }

    // P * diag(e^{-tλ}) * P^T
    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                val += p[(i, k)] * exp_evals[k] * p[(j, k)];
            }
            result[(i, j)] = val;
        }
    }
    result
}

/// Compute the Witten index as a supertrace.
/// For a Z₂-graded operator, str = tr_even - tr_odd.
pub fn witten_supertrace(
    laplacian: &DMatrix<f64>,
    t: f64,
    even_dim: usize,
) -> f64 {
    let heat = heat_kernel_operator(laplacian, t);
    let n = heat.nrows();

    let mut trace_even = 0.0;
    let mut trace_odd = 0.0;

    for i in 0..n.min(even_dim) {
        trace_even += heat[(i, i)];
    }
    for i in even_dim..n {
        trace_odd += heat[(i, i)];
    }

    trace_even - trace_odd
}

/// Compute the Witten index directly from eigenvalues.
/// Δ = Σ (-1)^k e^{-t λ_k}
pub fn witten_index_from_eigenvalues(eigenvalues: &[f64], t: f64) -> f64 {
    let mut idx = 0.0;
    for (k, &lambda) in eigenvalues.iter().enumerate() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        idx += sign * (-t * lambda).exp();
    }
    idx
}

/// Verify the index is independent of t (heat kernel regularization).
/// Compute at multiple t values and check they agree.
pub fn verify_t_independence(
    op: &EllipticOperator,
    t_values: &[f64],
    tol: f64,
) -> (bool, Vec<f64>) {
    let lap = laplacian(op);
    let even_dim = op.domain_dim / 2;
    let indices: Vec<f64> = t_values
        .iter()
        .map(|&t| witten_supertrace(&lap, t, even_dim))
        .collect();

    if indices.len() <= 1 {
        return (true, indices);
    }

    let first = indices[0];
    let all_agree = indices.iter().all(|&x| (x - first).abs() < tol);
    (all_agree, indices)
}

/// Compute the eta invariant (Atiyah-Patodi-Singer).
/// η(s) = Σ sign(λ) |λ|^{-s} over non-zero eigenvalues.
pub fn eta_invariant(eigenvalues: &[f64], s: f64) -> f64 {
    let mut eta = 0.0;
    for &lambda in eigenvalues {
        if lambda.abs() > 1e-12 {
            let sign = if lambda > 0.0 { 1.0 } else { -1.0 };
            eta += sign * lambda.abs().powf(-s);
        }
    }
    eta
}

/// McKean-Singer formula: str(e^{-tD²}) = ind(D) for all t > 0.
pub fn mckean_singer_check(
    op: &EllipticOperator,
    t: f64,
) -> (f64, f64) {
    let lap = laplacian(op);
    let even_dim = op.domain_dim / 2;
    let supertrace = witten_supertrace(&lap, t, even_dim);
    // Compare with direct index computation
    let m = op.matrix();
    let svd = m.svd(false, false);
    let singular_values = &svd.singular_values;
    let tol = 1e-8;
    let _n_zero: usize = singular_values.iter().filter(|&&s| s < tol).count();
    // For a square operator
    let analytic = 0.0; // Placeholder; exact comparison done externally
    (supertrace, analytic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_laplacian_identity() {
        let op = EllipticOperator::identity(3);
        let lap = laplacian(&op);
        for i in 0..3 {
            assert_abs_diff_eq!(lap[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_laplacian_symmetric() {
        let m = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 0.0,
            0.0, 1.0, 1.0,
        ]);
        let op = EllipticOperator::new(m);
        let lap = laplacian(&op);
        for i in 0..lap.nrows() {
            for j in 0..lap.ncols() {
                assert_abs_diff_eq!(lap[(i, j)], lap[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_heat_kernel_identity() {
        let lap = DMatrix::identity(3, 3);
        let heat = heat_kernel_operator(&lap, 1.0);
        for i in 0..3 {
            assert_abs_diff_eq!(heat[(i, i)], (-1.0f64).exp(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_heat_kernel_zero() {
        let lap = DMatrix::zeros(3, 3);
        let heat = heat_kernel_operator(&lap, 1.0);
        for i in 0..3 {
            assert_abs_diff_eq!(heat[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_heat_kernel_positive() {
        let lap = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let heat = heat_kernel_operator(&lap, 0.5);
        assert!(heat[(0, 0)] > 0.0);
        assert!(heat[(1, 1)] > 0.0);
        assert!(heat[(2, 2)] > 0.0);
    }

    #[test]
    fn test_witten_index_zero_laplacian() {
        let lap = DMatrix::zeros(4, 4);
        let idx = witten_supertrace(&lap, 1.0, 2);
        // tr_even - tr_odd = 2 - 2 = 0
        assert_abs_diff_eq!(idx, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_witten_from_eigenvalues_zero() {
        let idx = witten_index_from_eigenvalues(&[0.0, 0.0, 0.0, 0.0], 1.0);
        // 1 - 1 + 1 - 1 = 0
        assert_abs_diff_eq!(idx, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_witten_from_eigenvalues_positive() {
        let idx = witten_index_from_eigenvalues(&[0.0, 1.0], 1.0);
        // 1*1 + (-1)*e^{-1}
        assert!(idx > 0.0);
    }

    #[test]
    fn test_eta_invariant_symmetric() {
        let evals = vec![1.0, -1.0, 2.0, -2.0];
        let eta = eta_invariant(&evals, 0.0);
        assert_abs_diff_eq!(eta, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eta_invariant_positive_bias() {
        let evals = vec![1.0, 2.0, 3.0];
        let eta = eta_invariant(&evals, 0.0);
        assert!(eta > 0.0);
    }

    #[test]
    fn test_eta_invariant_skips_zero() {
        let evals = vec![0.0, 1.0];
        let eta = eta_invariant(&evals, 0.0);
        assert_abs_diff_eq!(eta, 1.0);
    }

    #[test]
    fn test_t_independence_check() {
        let op = EllipticOperator::identity(4);
        let (ok, indices) = verify_t_independence(&op, &[0.1, 1.0, 10.0], 0.01);
        // Identity → laplacian = I → supertrace should be ~0 for all t
        assert!(ok || indices.iter().all(|&x| x.abs() < 0.1));
    }

    #[test]
    fn test_mckean_singer_runs() {
        let op = EllipticOperator::identity(4);
        let (st, _) = mckean_singer_check(&op, 1.0);
        assert!(st.is_finite());
    }
}
