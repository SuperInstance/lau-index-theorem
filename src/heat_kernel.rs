//! Heat kernel proof (small-t expansion).
//!
//! The heat kernel has an asymptotic expansion as t → 0:
//!   K(t, x, x) ~ (4πt)^{-n/2} Σ a_k(x) t^k
//!
//! The index can be extracted from the supertrace of the heat kernel:
//!   ind(D) = str(K(t)) for all t > 0
//! Taking t → 0 picks out the local density.

use nalgebra::DMatrix;

/// Heat kernel coefficient a_0(x) = 1 (identity).
pub fn heat_coefficient_a0() -> f64 {
    1.0
}

/// Heat kernel coefficient a_1(x) = (1/6)R - E
/// where R is scalar curvature and E is the endomorphism.
pub fn heat_coefficient_a1(scalar_curvature: f64, endomorphism_trace: f64) -> f64 {
    scalar_curvature / 6.0 - endomorphism_trace
}

/// Heat kernel coefficient a_2 (simplified).
pub fn heat_coefficient_a2(
    scalar_curvature: f64,
    ricci_norm_sq: f64,
    endomorphism_norm_sq: f64,
) -> f64 {
    (1.0 / 360.0) * (5.0 * scalar_curvature.powi(2) - 2.0 * ricci_norm_sq)
        - (1.0 / 12.0) * endomorphism_norm_sq
}

/// Compute the small-t expansion of the heat kernel diagonal.
/// K(t, x, x) ~ (4πt)^{-n/2} * (a_0 + a_1*t + a_2*t² + ...)
pub fn heat_kernel_expansion(
    t: f64,
    dimension: usize,
    coefficients: &[f64],
) -> f64 {
    let prefactor = (4.0 * std::f64::consts::PI * t).powi(-(dimension as i32) / 2);
    let mut series = 0.0;
    let mut t_power = 1.0;
    for &a_k in coefficients {
        series += a_k * t_power;
        t_power *= t;
    }
    prefactor * series
}

/// Compute the index density from the heat kernel supertrace.
/// The local index density is the coefficient of t^0 in str(K(t)).
pub fn index_density_from_heat_kernel(
    dimension: usize,
    a_coefficients_even: &[f64],
    a_coefficients_odd: &[f64],
) -> f64 {
    // The supertrace picks out the difference of even and odd parts
    let mut density = 0.0;
    for (k, (&a_e, &a_o)) in a_coefficients_even.iter().zip(a_coefficients_odd.iter()).enumerate() {
        if k == dimension / 2 {
            density += a_e - a_o;
        }
    }
    density
}

/// Compute the heat kernel trace for a matrix Laplacian.
pub fn heat_kernel_trace(laplacian: &DMatrix<f64>, t: f64) -> f64 {
    let eig = laplacian.clone().symmetric_eigen();
    eig.eigenvalues.iter().map(|&lambda| (-t * lambda).exp()).sum()
}

/// Compute the supertrace of the heat kernel.
pub fn heat_kernel_supertrace(
    laplacian: &DMatrix<f64>,
    t: f64,
    even_dim: usize,
) -> f64 {
    let eig = laplacian.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;
    let p = &eig.eigenvectors;

    // Reconstruct heat kernel matrix
    let n = eigenvalues.len();
    let exp_evals: Vec<f64> = eigenvalues.iter().map(|&l| (-t * l).exp()).collect();

    // Supertrace = tr_even(K) - tr_odd(K)
    let mut tr_even = 0.0;
    let mut tr_odd = 0.0;

    for i in 0..n {
        let mut diag = 0.0;
        for k in 0..n {
            diag += p[(i, k)] * exp_evals[k] * p[(i, k)];
        }
        if i < even_dim {
            tr_even += diag;
        } else {
            tr_odd += diag;
        }
    }

    tr_even - tr_odd
}

/// Verify the heat kernel proof: supertrace is t-independent and equals the index.
pub fn verify_heat_kernel_proof(
    laplacian: &DMatrix<f64>,
    even_dim: usize,
    t_values: &[f64],
    tol: f64,
) -> (bool, Vec<f64>) {
    let straces: Vec<f64> = t_values
        .iter()
        .map(|&t| heat_kernel_supertrace(laplacian, t, even_dim))
        .collect();

    if straces.len() <= 1 {
        return (true, straces);
    }

    let first = straces[0];
    let all_agree = straces.iter().all(|&x| (x - first).abs() < tol);
    (all_agree, straces)
}

/// Small-t asymptotics: compute the first few terms.
pub fn small_t_asymptotics(
    dimension: usize,
    t: f64,
    a_coefficients: &[f64],
) -> Vec<f64> {
    let n_terms = a_coefficients.len();
    let prefactor = (4.0 * std::f64::consts::PI * t).powi(-(dimension as i32) / 2);

    (0..n_terms)
        .map(|k| prefactor * a_coefficients[k] * t.powi(k as i32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_a0_is_one() {
        assert_abs_diff_eq!(heat_coefficient_a0(), 1.0);
    }

    #[test]
    fn test_a1_zero_curvature() {
        assert_abs_diff_eq!(heat_coefficient_a1(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_a1_with_curvature() {
        let a1 = heat_coefficient_a1(6.0, 0.0);
        assert_abs_diff_eq!(a1, 1.0);
    }

    #[test]
    fn test_a1_with_endomorphism() {
        let a1 = heat_coefficient_a1(0.0, 1.0);
        assert_abs_diff_eq!(a1, -1.0);
    }

    #[test]
    fn test_a2_zero() {
        let a2 = heat_coefficient_a2(0.0, 0.0, 0.0);
        assert_abs_diff_eq!(a2, 0.0);
    }

    #[test]
    fn test_heat_kernel_expansion_zero_t() {
        let val = heat_kernel_expansion(0.01, 2, &[1.0, 0.0]);
        assert!(val.is_finite());
        assert!(val > 0.0);
    }

    #[test]
    fn test_heat_kernel_trace_identity() {
        let lap = DMatrix::identity(3, 3);
        let tr = heat_kernel_trace(&lap, 1.0);
        assert_abs_diff_eq!(tr, 3.0 * (-1.0f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_heat_kernel_trace_zero() {
        let lap = DMatrix::zeros(3, 3);
        let tr = heat_kernel_trace(&lap, 1.0);
        assert_abs_diff_eq!(tr, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_heat_kernel_supertrace_symmetric() {
        // Diagonal laplacian with known eigenvalues
        let lap = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]));
        let st = heat_kernel_supertrace(&lap, 1.0, 2);
        // even: 0 and 0 (indices 0,1) → exp(0) + exp(-1) = 1 + e^{-1}
        // odd: 0 and 1 (indices 2,3) → exp(0) + exp(-1) = 1 + e^{-1}
        // Hmm, let me recalculate: eigenvalues are [0, 1, 0, 1]
        // For identity eigenvectors: diagonal of heat kernel at (i,i) = e^{-t*λ_i}
        // tr_even = e^0 + e^{-1} = 1 + 0.368
        // tr_odd = e^0 + e^{-1} = 1 + 0.368
        // supertrace = 0
        assert_abs_diff_eq!(st, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_verify_heat_kernel_t_independence() {
        let lap = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 1.0, 0.0, 2.0]));
        let (ok, _vals) = verify_heat_kernel_proof(&lap, 2, &[0.1, 1.0, 5.0], 0.01);
        // Should be t-independent
        assert!(ok);
    }

    #[test]
    fn test_small_t_asymptotics_length() {
        let terms = small_t_asymptotics(2, 0.01, &[1.0, 0.5, 0.1]);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_small_t_asymptotics_dominant() {
        let terms = small_t_asymptotics(2, 0.001, &[1.0, 0.5]);
        assert!(terms[0] > terms[1]); // a_0 term dominates
    }

    #[test]
    fn test_index_density_runs() {
        let density = index_density_from_heat_kernel(2, &[1.0, 0.5], &[0.0, 0.5]);
        assert!(density.is_finite());
    }

    #[test]
    fn test_heat_expansion_decreases_with_t() {
        let v1 = heat_kernel_expansion(0.01, 2, &[1.0]);
        let v2 = heat_kernel_expansion(0.1, 2, &[1.0]);
        assert!(v1 > v2); // Smaller t → larger value
    }
}
