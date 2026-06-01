//! # lau-index-theorem
//!
//! Agent Index Theorem (Atiyah-Singer) — tr(id) is the baby case.
//!
//! Core components:
//! - Analytic index: dim(ker D) - dim(coker D)
//! - Topological index: Euler characteristic from topology
//! - Witten index: tr((-1)^F e^{-tD²}) = Euler characteristic
//! - Policy bundle with Chern classes
//! - Heat kernel proof (small-t expansion)
//! - Characteristic number = agent dimension

pub mod analytic_index;
pub mod topological_index;
pub mod witten_index;
pub mod policy_bundle;
pub mod heat_kernel;
pub mod characteristic;
pub mod types;

pub use types::*;
