//! Policy bundle with Chern classes.
//!
//! A "policy bundle" is a vector bundle whose sections are policy functions.
//! The Chern classes of this bundle carry topological information about
//! the space of policies.

use crate::types::VectorBundle;

/// Compute the total Chern class of a Whitney sum E ⊕ F.
/// c(E ⊕ F) = c(E) · c(F) (cup product).
pub fn whitney_sum_chern(e: &VectorBundle, f: &VectorBundle) -> VectorBundle {
    let rank = e.rank + f.rank;
    let base_dim = e.base_dim.max(f.base_dim);
    let mut chern = vec![0.0; rank];

    // Convolution of Chern classes
    let _n = e.chern_classes.len().min(f.chern_classes.len());
    for i in 0..chern.len() {
        for j in 0..=i.min(e.chern_classes.len() - 1) {
            let k = i - j;
            if k < f.chern_classes.len() {
                chern[i] += e.chern_classes[j] * f.chern_classes[k];
            }
        }
    }

    VectorBundle { rank, base_dim, chern_classes: chern }
}

/// Compute Chern classes of a tensor product bundle.
/// c(E ⊗ F) = Π_i Π_j (1 + x_i + y_j) where x_i, y_j are Chern roots.
pub fn tensor_product_chern(e_chern_roots: &[f64], f_chern_roots: &[f64]) -> Vec<f64> {
    let total_rank = e_chern_roots.len() * f_chern_roots.len();
    // Simplified: use only first Chern class
    let c1_e: f64 = e_chern_roots.iter().sum();
    let c1_f: f64 = f_chern_roots.iter().sum();

    let mut chern = vec![0.0; total_rank];
    if !chern.is_empty() {
        chern[0] = c1_e * f_chern_roots.len() as f64 + c1_f * e_chern_roots.len() as f64;
    }
    chern
}

/// Compute the first Chern class from a curvature 2-form matrix.
/// c₁ = (i/2π) tr(F) where F is the curvature.
pub fn first_chern_from_curvature(curvature: &nalgebra::DMatrix<f64>) -> f64 {
    let n = curvature.nrows().min(curvature.ncols());
    let mut trace = 0.0;
    for i in 0..n {
        trace += curvature[(i, i)];
    }
    trace / (2.0 * std::f64::consts::PI)
}

/// Compute the k-th Chern class from a connection matrix.
/// c_k = coefficient of t^k in det(I + tF/2πi).
pub fn chern_class_from_connection(
    curvature: &nalgebra::DMatrix<f64>,
    k: usize,
) -> f64 {
    let n = curvature.nrows().min(curvature.ncols());
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }

    // For k=1: (1/2π) tr(F)
    if k == 1 {
        return first_chern_from_curvature(curvature);
    }

    // For higher k: use elementary symmetric polynomials of eigenvalues
    let eig = curvature.clone().symmetric_eigen();
    let evals: Vec<f64> = eig.eigenvalues.iter().map(|&x| x / (2.0 * std::f64::consts::PI)).collect();

    // k-th elementary symmetric polynomial
    elementary_symmetric(&evals, k)
}

/// Compute the k-th elementary symmetric polynomial.
fn elementary_symmetric(values: &[f64], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if values.is_empty() || k > values.len() {
        return 0.0;
    }

    // Use Newton's identity recursively
    let _n = values.len();
    let mut dp = vec![0.0; k + 1];
    dp[0] = 1.0;

    for &v in values {
        for j in (1..=k).rev() {
            dp[j] += v * dp[j - 1];
        }
    }
    dp[k]
}

/// Compute the Chern character from Chern classes.
/// ch = rank + c₁ + (c₁² - 2c₂)/2 + ...
pub fn chern_character_from_classes(rank: usize, chern_classes: &[f64]) -> Vec<f64> {
    let mut ch = vec![rank as f64];
    if !chern_classes.is_empty() {
        ch.push(chern_classes[0]); // c₁
    }
    if chern_classes.len() >= 2 {
        let c1_sq = chern_classes[0].powi(2);
        ch.push((c1_sq - 2.0 * chern_classes[1]) / 2.0); // (c₁² - 2c₂)/2
    }
    ch
}

/// Policy bundle: a vector bundle whose sections are agent policies.
pub struct PolicyBundle {
    pub bundle: VectorBundle,
    /// Number of agents (fiber = policy space dimension).
    pub n_agents: usize,
}

impl PolicyBundle {
    pub fn new(n_agents: usize, policy_dim: usize, base_dim: usize) -> Self {
        Self {
            bundle: VectorBundle::new(policy_dim, base_dim),
            n_agents,
        }
    }

    /// The index of the policy bundle = rank (policy dimension).
    pub fn policy_index(&self) -> usize {
        self.bundle.rank
    }

    /// Characteristic number = agent dimension (from the index theorem).
    pub fn characteristic_number(&self) -> f64 {
        self.bundle.euler_class() * self.n_agents as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_whitney_sum_rank() {
        let e = VectorBundle::new(2, 4);
        let f = VectorBundle::new(3, 4);
        let sum = whitney_sum_chern(&e, &f);
        assert_eq!(sum.rank, 5);
    }

    #[test]
    fn test_whitney_sum_trivial() {
        let e = VectorBundle::trivial(1, 2);
        let f = VectorBundle::trivial(1, 2);
        let sum = whitney_sum_chern(&e, &f);
        assert_eq!(sum.rank, 2);
    }

    #[test]
    fn test_tensor_product_chern_length() {
        let chern = tensor_product_chern(&[1.0], &[1.0]);
        assert_eq!(chern.len(), 1);
    }

    #[test]
    fn test_first_chern_zero_curvature() {
        let curv = nalgebra::DMatrix::zeros(3, 3);
        assert_abs_diff_eq!(first_chern_from_curvature(&curv), 0.0);
    }

    #[test]
    fn test_first_chern_identity() {
        let curv = nalgebra::DMatrix::identity(3, 3) * std::f64::consts::PI * 2.0;
        assert_abs_diff_eq!(first_chern_from_curvature(&curv), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chern_class_k0() {
        let curv = nalgebra::DMatrix::identity(3, 3);
        assert_abs_diff_eq!(chern_class_from_connection(&curv, 0), 1.0);
    }

    #[test]
    fn test_chern_class_k1() {
        let curv = nalgebra::DMatrix::identity(2, 2);
        let c1 = chern_class_from_connection(&curv, 1);
        assert!(c1.is_finite());
    }

    #[test]
    fn test_elementary_symmetric_k0() {
        assert_abs_diff_eq!(elementary_symmetric(&[1.0, 2.0, 3.0], 0), 1.0);
    }

    #[test]
    fn test_elementary_symmetric_k1() {
        assert_abs_diff_eq!(elementary_symmetric(&[1.0, 2.0, 3.0], 1), 6.0);
    }

    #[test]
    fn test_elementary_symmetric_k2() {
        assert_abs_diff_eq!(elementary_symmetric(&[1.0, 2.0, 3.0], 2), 11.0);
    }

    #[test]
    fn test_chern_character_from_classes_rank() {
        let ch = chern_character_from_classes(3, &[1.0]);
        assert_abs_diff_eq!(ch[0], 3.0);
    }

    #[test]
    fn test_policy_bundle_index() {
        let pb = PolicyBundle::new(10, 3, 4);
        assert_eq!(pb.policy_index(), 3);
    }

    #[test]
    fn test_policy_bundle_characteristic_number() {
        let pb = PolicyBundle::new(5, 2, 2);
        let cn = pb.characteristic_number();
        assert!(cn.is_finite());
    }
}
