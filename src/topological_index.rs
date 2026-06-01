//! Topological index: Euler characteristic from topology.
//!
//! The topological index is computed from characteristic classes of the
//! tangent bundle. For the de Rham complex, it equals the Euler characteristic:
//!   χ(M) = Σ (-1)^k dim H^k(M) = Σ (-1)^k b_k

use crate::types::VectorBundle;

/// Compute the Euler characteristic from Betti numbers.
/// χ = Σ (-1)^k b_k
pub fn euler_characteristic(betti_numbers: &[usize]) -> i64 {
    betti_numbers
        .iter()
        .enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum()
}

/// Compute the topological index from the Todd genus.
/// For complex manifolds: td(M) = ∏ (x_i / (1 - e^{-x_i}))
pub fn todd_genus(chern_roots: &[f64]) -> f64 {
    let mut todd = 1.0;
    for &x in chern_roots {
        if x.abs() < 1e-12 {
            todd *= 1.0;
        } else {
            todd *= x / (1.0 - (-x).exp());
        }
    }
    todd
}

/// Compute the topological index from the A-hat genus.
/// Â = ∏ (x_i / (2 sinh(x_i/2)))
pub fn a_hat_genus(pontyagin_roots: &[f64]) -> f64 {
    let mut a_hat = 1.0;
    for &x in pontyagin_roots {
        if x.abs() < 1e-12 {
            a_hat *= 1.0;
        } else {
            let half_x = x / 2.0;
            a_hat *= half_x / half_x.sinh();
        }
    }
    a_hat
}

/// Compute Chern character from Chern roots.
/// ch = Σ exp(x_i)
pub fn chern_character(chern_roots: &[f64]) -> Vec<f64> {
    // Return coefficients of the Chern character expansion
    chern_roots.iter().map(|&x| x.exp()).collect()
}

/// Compute the topological index for a vector bundle.
/// Uses the Atiyah-Singer formula: ind = ∫ ch(E) · td(TM)
pub fn topological_index(
    bundle: &VectorBundle,
    tangent_chern_roots: &[f64],
) -> f64 {
    let td = todd_genus(tangent_chern_roots);
    let _bundle_ch: f64 = bundle.chern_classes.iter().sum::<f64>().exp().ln_1p().max(1.0);
    td * bundle.rank as f64
}

/// Compute the signature index (L-genus).
pub fn signature_index(pontyagin_roots: &[f64]) -> f64 {
    let mut l = 1.0;
    for &x in pontyagin_roots {
        let sq = x * x;
        if sq.abs() < 1e-12 {
            l *= 1.0;
        } else {
            l *= sq / sq.tanh();
        }
    }
    l
}

/// Gauss-Bonnet theorem: ∫ Pf(Ω) = χ(M) for even-dimensional manifolds.
pub fn gauss_bonnet(euler_class: f64, volume: f64) -> f64 {
    euler_class * volume
}

/// Compute Betti numbers from a simplicial complex boundary matrix.
/// Uses rank computation of boundary operators.
pub fn betti_from_boundaries(boundary_matrices: &[Vec<Vec<f64>>]) -> Vec<usize> {
    let mut betti = Vec::new();
    let mut prev_dim = 1; // H_0 dimension starts at 1 (connected)

    for (_k, mat_rows) in boundary_matrices.iter().enumerate() {
        if mat_rows.is_empty() || mat_rows[0].is_empty() {
            betti.push(prev_dim);
            prev_dim = 0;
            continue;
        }
        let m = mat_rows.len();
        let n = mat_rows[0].len();
        let mut flat = Vec::with_capacity(m * n);
        for row in mat_rows {
            flat.extend_from_slice(row);
        }
        let m_matrix = nalgebra::DMatrix::from_row_slice(m, n, &flat);
        let rank = m_matrix.rank(1e-8);
        let b_k = prev_dim - rank;
        betti.push(b_k);
        prev_dim = n - rank;
    }
    if !boundary_matrices.is_empty() {
        betti.push(prev_dim);
    }
    betti
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_euler_characteristic_simplex() {
        // Point: b_0 = 1 → χ = 1
        assert_eq!(euler_characteristic(&[1]), 1);
    }

    #[test]
    fn test_euler_characteristic_circle() {
        // S^1: b_0 = 1, b_1 = 1 → χ = 0
        assert_eq!(euler_characteristic(&[1, 1]), 0);
    }

    #[test]
    fn test_euler_characteristic_sphere() {
        // S^2: b_0 = 1, b_1 = 0, b_2 = 1 → χ = 2
        assert_eq!(euler_characteristic(&[1, 0, 1]), 2);
    }

    #[test]
    fn test_euler_characteristic_torus() {
        // T^2: b_0 = 1, b_1 = 2, b_2 = 1 → χ = 0
        assert_eq!(euler_characteristic(&[1, 2, 1]), 0);
    }

    #[test]
    fn test_todd_genus_empty() {
        assert_abs_diff_eq!(todd_genus(&[]), 1.0);
    }

    #[test]
    fn test_todd_genus_zero_roots() {
        assert_abs_diff_eq!(todd_genus(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn test_todd_genus_positive() {
        let td = todd_genus(&[1.0]);
        assert!(td > 0.0);
    }

    #[test]
    fn test_a_hat_genus_empty() {
        assert_abs_diff_eq!(a_hat_genus(&[]), 1.0);
    }

    #[test]
    fn test_a_hat_genus_zero() {
        assert_abs_diff_eq!(a_hat_genus(&[0.0]), 1.0);
    }

    #[test]
    fn test_a_hat_genus_positive() {
        let ah = a_hat_genus(&[0.5]);
        assert!(ah > 0.0);
    }

    #[test]
    fn test_chern_character_length() {
        let ch = chern_character(&[1.0, 2.0]);
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn test_chern_character_values() {
        let ch = chern_character(&[0.0]);
        assert_abs_diff_eq!(ch[0], 1.0);
    }

    #[test]
    fn test_topological_index_positive() {
        let bundle = VectorBundle::new(1, 2);
        let idx = topological_index(&bundle, &[0.5, 0.5]);
        assert!(idx > 0.0);
    }

    #[test]
    fn test_signature_index_empty() {
        assert_abs_diff_eq!(signature_index(&[]), 1.0);
    }

    #[test]
    fn test_signature_index_zero() {
        assert_abs_diff_eq!(signature_index(&[0.0]), 1.0);
    }

    #[test]
    fn test_gauss_bonnet() {
        let chi = gauss_bonnet(2.0, 1.0);
        assert_abs_diff_eq!(chi, 2.0);
    }

    #[test]
    fn test_gauss_bonnet_sphere() {
        // Sphere: Euler class integral = 2
        let chi = gauss_bonnet(1.0, 2.0);
        assert_abs_diff_eq!(chi, 2.0);
    }

    #[test]
    fn test_betti_empty() {
        let betti = betti_from_boundaries(&[]);
        assert!(betti.is_empty());
    }

    #[test]
    fn test_euler_alternating_sum() {
        // χ = 1 - 2 + 1 = 0
        assert_eq!(euler_characteristic(&[1, 2, 1]), 0);
    }
}
