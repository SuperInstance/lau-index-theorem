//! Characteristic number = agent dimension.
//!
//! The characteristic number of a vector bundle over a closed manifold
//! is obtained by integrating the top characteristic class over the manifold.
//! In the index theorem context, this equals the agent dimension.

use crate::types::VectorBundle;
use crate::policy_bundle::{chern_class_from_connection, first_chern_from_curvature};

/// Compute the Euler number (top Chern number) of a bundle.
pub fn euler_number(bundle: &VectorBundle) -> f64 {
    if bundle.chern_classes.is_empty() {
        return 0.0;
    }
    // The Euler number is the integral of the top Chern class
    bundle.chern_classes.iter().last().copied().unwrap_or(0.0)
}

/// Characteristic number for agent dimension.
/// In the index theorem: agent_dim = ind(D) = ∫ ch(E) · td(TM)
pub fn agent_dimension(
    bundle_rank: usize,
    todd_genus: f64,
    chern_character: f64,
) -> f64 {
    todd_genus * chern_character * bundle_rank as f64
}

/// Compute the Pontryagin number from Pontryagin classes.
/// p_k = (-1)^k c_{2k}(E ⊗ ℝℂ)
pub fn pontryagin_number(pontryagin_classes: &[f64], manifold_dim: usize) -> f64 {
    if manifold_dim % 4 != 0 {
        return 0.0;
    }
    let top_k = manifold_dim / 4;
    if top_k == 0 || top_k > pontryagin_classes.len() {
        return 0.0;
    }
    pontryagin_classes[top_k - 1]
}

/// Compute the Stiefel-Whitney number (mod 2 version).
pub fn stiefel_whitney_number(w_classes: &[u8], manifold_dim: usize) -> u8 {
    if manifold_dim == 0 {
        return 1;
    }
    let top_k = manifold_dim;
    if top_k > w_classes.len() {
        return 0;
    }
    w_classes[top_k - 1] % 2
}

/// Hirzebruch signature theorem: signature = L-genus.
pub fn hirzebruch_signature(pontyagin_roots: &[f64]) -> f64 {
    let mut sig = 1.0;
    for &x in pontyagin_roots {
        let x2 = x * x;
        if x2.abs() < 1e-12 {
            sig *= 1.0;
        } else {
            sig *= x2 / x2.sinh();
        }
    }
    sig
}

/// Riemann-Roch for curves: dim H⁰(D) - dim H¹(D) = deg(D) + 1 - g.
pub fn riemann_roch_curve(degree: i64, genus: usize) -> i64 {
    degree + 1 - genus as i64
}

/// Characteristic number equals the index for Dirac-type operators.
/// Verify: ch(S±) · td(TM) = Â(TM) · ch(E) → ind = ∫ ...
pub fn verify_characteristic_equals_index(
    a_hat_genus: f64,
    chern_char: f64,
    analytic_index: i64,
    tolerance: f64,
) -> bool {
    let top_idx = a_hat_genus * chern_char;
    (top_idx - analytic_index as f64).abs() < tolerance
}

/// Compute the genus from the characteristic number.
/// g = (2 - χ) / 2 for surfaces.
pub fn genus_from_euler(euler_char: i64) -> i64 {
    (2 - euler_char) / 2
}

/// Multiplicative sequence for characteristic classes.
pub fn multiplicative_sequence(classes: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n > classes.len() {
        return 0.0;
    }
    classes[n - 1]
}

/// Compute the Hirzebruch χ_y-genus.
pub fn hirzebruch_chi_y_genus(betti: &[usize], y: f64) -> f64 {
    betti.iter()
        .enumerate()
        .map(|(k, &b)| b as f64 * y.powi(k as i32))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_euler_number_trivial() {
        let b = VectorBundle::trivial(1, 2);
        assert_abs_diff_eq!(euler_number(&b), 0.0);
    }

    #[test]
    fn test_euler_number_with_class() {
        let b = VectorBundle { rank: 2, base_dim: 4, chern_classes: vec![0.0, 2.0] };
        assert_abs_diff_eq!(euler_number(&b), 2.0);
    }

    #[test]
    fn test_agent_dimension_basic() {
        let dim = agent_dimension(2, 1.0, 1.0);
        assert_abs_diff_eq!(dim, 2.0);
    }

    #[test]
    fn test_agent_dimension_zero_genus() {
        let dim = agent_dimension(3, 0.0, 1.0);
        assert_abs_diff_eq!(dim, 0.0);
    }

    #[test]
    fn test_pontryagin_number_dim4() {
        let pn = pontryagin_number(&[3.0], 4);
        assert_abs_diff_eq!(pn, 3.0);
    }

    #[test]
    fn test_pontryagin_number_wrong_dim() {
        let pn = pontryagin_number(&[1.0], 3);
        assert_abs_diff_eq!(pn, 0.0);
    }

    #[test]
    fn test_stiefel_whitney_number_dim0() {
        assert_eq!(stiefel_whitney_number(&[], 0), 1);
    }

    #[test]
    fn test_stiefel_whitney_number_dim2() {
        assert_eq!(stiefel_whitney_number(&[1], 1), 1);
    }

    #[test]
    fn test_hirzebruch_signature_empty() {
        assert_abs_diff_eq!(hirzebruch_signature(&[]), 1.0);
    }

    #[test]
    fn test_hirzebruch_signature_zero() {
        assert_abs_diff_eq!(hirzebruch_signature(&[0.0]), 1.0);
    }

    #[test]
    fn test_riemann_roch_degree_zero() {
        // P^1: degree 0, genus 0 → dim H^0 - dim H^1 = 1
        assert_eq!(riemann_roch_curve(0, 0), 1);
    }

    #[test]
    fn test_riemann_roch_genus_1() {
        assert_eq!(riemann_roch_curve(0, 1), 0);
    }

    #[test]
    fn test_riemann_roch_positive_degree() {
        assert_eq!(riemann_roch_curve(3, 0), 4);
    }

    #[test]
    fn test_verify_characteristic_equals_index_exact() {
        assert!(verify_characteristic_equals_index(1.0, 1.0, 1, 0.5));
    }

    #[test]
    fn test_verify_characteristic_equals_index_wrong() {
        assert!(!verify_characteristic_equals_index(1.0, 1.0, 5, 0.5));
    }

    #[test]
    fn test_genus_from_euler_sphere() {
        assert_eq!(genus_from_euler(2), 0); // S^2
    }

    #[test]
    fn test_genus_from_euler_torus() {
        assert_eq!(genus_from_euler(0), 1); // T^2
    }

    #[test]
    fn test_multiplicative_sequence_k0() {
        assert_abs_diff_eq!(multiplicative_sequence(&[1.0, 2.0], 0), 1.0);
    }

    #[test]
    fn test_hirzebruch_chi_y_genus_sphere() {
        // S^2: b = [1, 0, 1]
        let chi = hirzebruch_chi_y_genus(&[1, 0, 1], 1.0);
        assert_abs_diff_eq!(chi, 2.0);
    }

    #[test]
    fn test_hirzebruch_chi_y_genus_point() {
        let chi = hirzebruch_chi_y_genus(&[1], 0.5);
        assert_abs_diff_eq!(chi, 1.0);
    }
}
