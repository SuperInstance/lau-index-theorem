//! Common types for the index theorem.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// An elliptic operator represented as a matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EllipticOperator {
    /// The matrix data stored as row-major vec for serde compatibility.
    pub data: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Dimension of the domain.
    pub domain_dim: usize,
    /// Dimension of the codomain.
    pub codomain_dim: usize,
}

impl EllipticOperator {
    pub fn new(matrix: DMatrix<f64>) -> Self {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let data = matrix.iter().cloned().collect();
        Self { data, rows, cols, domain_dim: cols, codomain_dim: rows }
    }

    /// Get the matrix representation.
    pub fn matrix(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.rows, self.cols, &self.data)
    }

    /// Zero operator of given dimensions.
    pub fn zero(m: usize, n: usize) -> Self {
        Self { data: vec![0.0; m * n], rows: m, cols: n, domain_dim: n, codomain_dim: m }
    }

    /// Identity operator.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        Self { data, rows: n, cols: n, domain_dim: n, codomain_dim: n }
    }

    /// Apply the operator to a vector.
    pub fn apply(&self, v: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        &self.matrix() * v
    }

    /// Compute the adjoint D*.
    pub fn adjoint(&self) -> Self {
        Self::new(self.matrix().transpose())
    }

    /// Compute D*D (self-adjoint, positive semi-definite).
    pub fn d_star_d(&self) -> DMatrix<f64> {
        let m = self.matrix();
        &m.transpose() * &m
    }

    /// Compute DD*.
    pub fn d_d_star(&self) -> DMatrix<f64> {
        let m = self.matrix();
        &m * &m.transpose()
    }
}

/// A vector bundle (represented by its fiber dimension and base manifold dimension).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorBundle {
    /// Fiber dimension (rank).
    pub rank: usize,
    /// Base manifold dimension.
    pub base_dim: usize,
    /// Chern classes (as cohomology degrees).
    pub chern_classes: Vec<f64>,
}

impl VectorBundle {
    pub fn new(rank: usize, base_dim: usize) -> Self {
        Self {
            rank,
            base_dim,
            chern_classes: vec![0.0; rank],
        }
    }

    pub fn trivial(rank: usize, base_dim: usize) -> Self {
        Self {
            rank,
            base_dim,
            chern_classes: vec![0.0; rank],
        }
    }

    /// Total Chern class: c = 1 + c₁ + c₂ + ...
    pub fn total_chern_class(&self) -> Vec<f64> {
        let mut tc = vec![1.0];
        tc.extend_from_slice(&self.chern_classes);
        tc
    }

    /// Euler class (top Chern class).
    pub fn euler_class(&self) -> f64 {
        if self.chern_classes.is_empty() {
            0.0
        } else {
            self.chern_classes[self.chern_classes.len() - 1]
        }
    }
}

/// Result of index computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexResult {
    pub analytic_index: i64,
    pub topological_index: i64,
    pub witten_index: f64,
    pub indices_agree: bool,
}
