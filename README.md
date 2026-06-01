# lau-index-theorem

**Agent Index Theorem (Atiyah-Singer) — `tr(id)` is the baby case.**

A Rust implementation of the Atiyah-Singer Index Theorem, one of the deepest results in mathematics. The index theorem connects analysis (solutions of differential equations) to topology (shape of spaces). This crate provides both sides of the equation — the analytic index and the topological index — and the machinery to prove they're equal.

## What This Does

| Module | What it computes | The theorem connection |
|---|---|---|
| **Analytic index** | `dim(ker D) − dim(coker D)` | Solutions of D |
| **Topological index** | Euler characteristic, Todd/Â genus | Topology of the base |
| **Witten index** | `tr((−1)^F e^{−tD²})` | Heat kernel proof |
| **Heat kernel** | Small-t expansion, supertrace | The bridge: analysis ↔ topology |
| **Policy bundle** | Chern classes, characteristic classes | Vector bundles of agent policies |
| **Characteristic** | Euler number, Pontryagin, Riemann-Roch | Topological invariants = agent dimension |

## Key Idea

The Atiyah-Singer Index Theorem states:

```
Analytic index = Topological index
dim(ker D) − dim(ker D*) = ∫ ch(E) · td(TM)
```

The left side counts solutions of an elliptic operator `D` (analytic). The right side integrates characteristic classes over the manifold (topological). The theorem says these are **always equal**.

The "baby case" mentioned in the tagline: when `D = id`, the index is `tr(id) = dim` — the identity operator's index is just the dimension. Everything else is a vast generalization of counting.

This crate frames the index theorem in terms of **agent systems**: the "policy bundle" is a vector bundle whose sections are agent policies, and the "characteristic number" equals the agent dimension.

## Install

```toml
[dependencies]
lau-index-theorem = "0.1.0"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-index-theorem.git
cd lau-index-theorem
cargo test
```

### Dependencies

- `nalgebra` 0.33 — linear algebra (SVD, eigendecomposition)
- `serde` / `serde_json` — serialization

## Quick Start

```rust
use lau_index_theorem::{
    types::EllipticOperator,
    analytic_index, topological_index, witten_index,
};

// Create an elliptic operator (matrix)
let op = EllipticOperator::identity(4);

// Analytic index: dim(ker D) - dim(coker D)
let a_idx = analytic_index::analytic_index(&op, 1e-8);
// For identity: ker = 0, coker = 0, index = 0

// Topological index from Betti numbers (S²: b₀=1, b₁=0, b₂=1)
let euler = topological_index::euler_characteristic(&[1, 0, 1]);
// χ(S²) = 1 - 0 + 1 = 2

// Witten index: tr((-1)^F e^{-tD²})
let lap = witten_index::laplacian(&op);
let w_idx = witten_index::witten_supertrace(&lap, 1.0, 2);
```

## API Reference

### `EllipticOperator` — Matrix representation of D

```rust
let op = EllipticOperator::new(matrix);     // from DMatrix<f64>
let op = EllipticOperator::identity(4);      // 4×4 identity
let m = op.matrix();                          // DMatrix<f64>
let adj = op.adjoint();                       // D*
let dsd = op.d_star_d();                      // D*D (Laplacian)
let dds = op.d_d_star();                      // DD*
```

### `VectorBundle` — Fiber bundle data

```rust
let bundle = VectorBundle::new(rank: 2, base_dim: 4);
let tc = bundle.total_chern_class();          // [1, c₁, c₂]
let euler = bundle.euler_class();             // top Chern class
```

### Analytic Index

```rust
use lau_index_theorem::analytic_index::*;

let ker = kernel_basis(&matrix, tol);         // null space vectors
let coker = cokernel_basis(&matrix, tol);     // kernel of D^T
let dk = dim_kernel(&matrix, tol);
let dc = dim_cokernel(&matrix, tol);
let idx = analytic_index_matrix(&matrix, tol); // dk - dc as i64
let fredholm = is_fredholm(&op, tol);
let proj = kernel_projection(&matrix, tol);   // projection onto ker
```

### Topological Index

```rust
use lau_index_theorem::topological_index::*;

let chi = euler_characteristic(&[1, 0, 1]);   // χ(S²) = 2
let td = todd_genus(&[0.5, 0.5]);             // Todd class
let ah = a_hat_genus(&[0.1]);                  // Â genus
let sig = signature_index(&[0.5]);             // L-genus
let gb = gauss_bonnet(euler_class, volume);    // Gauss-Bonnet
```

### Witten Index

```rust
use lau_index_theorem::witten_index::*;

let heat = heat_kernel_operator(&laplacian, t);  // e^{-tD²}
let idx = witten_supertrace(&lap, t, even_dim);  // tr₊ - tr₋
let idx2 = witten_index_from_eigenvalues(&evals, t);
let eta = eta_invariant(&evals, s);               // APS invariant
let (ok, vals) = verify_t_independence(&op, &[0.1, 1.0, 5.0], tol);
```

### Heat Kernel

```rust
use lau_index_theorem::heat_kernel::*;

let a0 = heat_coefficient_a0();                                     // = 1
let a1 = heat_coefficient_a1(scalar_curvature, endo_trace);        // R/6 - E
let a2 = heat_coefficient_a2(R, ricci_sq, endo_sq);                // (5R²-2|Ric|²)/360 - |E|²/12
let val = heat_kernel_expansion(t, dimension, &coeffs);
let tr = heat_kernel_trace(&laplacian, t);                          // Σ e^{-tλᵢ}
let st = heat_kernel_supertrace(&lap, t, even_dim);
let (ok, vals) = verify_heat_kernel_proof(&lap, even_dim, &[0.1, 1.0], tol);
```

### Characteristic Numbers

```rust
use lau_index_theorem::characteristic::*;

let e = euler_number(&bundle);                      // top Chern number
let dim = agent_dimension(bundle_rank, todd, chern); // ind = Â·ch·rank
let pn = pontryagin_number(&classes, manifold_dim);
let sw = stiefel_whitney_number(&w_classes, dim);
let sig = hirzebruch_signature(&pont_roots);
let rr = riemann_roch_curve(degree, genus);          // dim H⁰ - dim H¹
let g = genus_from_euler(euler_char);                 // g = (2-χ)/2
let chi_y = hirzebruch_chi_y_genus(&betti, y);
```

### Policy Bundle

```rust
use lau_index_theorem::policy_bundle::*;

let pb = PolicyBundle::new(n_agents: 10, policy_dim: 3, base_dim: 4);
let idx = pb.policy_index();                        // = policy_dim
let cn = pb.characteristic_number();                 // Euler class × agents

let sum = whitney_sum_chern(&e, &f);                // c(E⊕F) = c(E)·c(F)
let c1 = first_chern_from_curvature(&curvature);    // tr(F)/2π
let ck = chern_class_from_connection(&curv, k);     // elementary symmetric poly
let ch = chern_character_from_classes(rank, &classes);
```

## How It Works

### Analytic Side

The analytic index is computed via SVD of the matrix representation:
- **Kernel dimension** = number of singular values below tolerance
- **Cokernel dimension** = nrows − rank
- **Index** = dim(ker D) − dim(coker D) = dim(ker D) − dim(ker D*)

The kernel projection is constructed from the right singular vectors corresponding to zero singular values.

### Topological Side

Topological invariants are computed from characteristic classes:
- **Euler characteristic**: alternating sum of Betti numbers
- **Todd genus**: `Π xᵢ/(1−e^{−xᵢ})` from Chern roots
- **Â genus**: `Π (xᵢ/2)/sinh(xᵢ/2)` from Pontryagin roots
- **Chern classes**: elementary symmetric polynomials of curvature eigenvalues

### The Bridge: Heat Kernel

The heat kernel proof works by:
1. Computing `str(e^{−tD²}) = tr_even − tr_odd` (supertrace)
2. Showing this is **independent of t** (for all t > 0)
3. As `t → 0`: picks out local density (index density)
4. As `t → ∞`: picks out the global index (dim ker − dim coker)

The McKean-Singer formula makes this precise: `ind(D) = str(e^{−tD²})` for all t.

## The Math

### The Atiyah-Singer Index Theorem

For an elliptic operator `D` on a compact manifold `M`:

```
ind(D) = ∫_M ch(E) · td(TM) = ∫_M Â(TM) · ch(V)
```

Special cases:
- **Gauss-Bonnet**: `χ(M) = ∫ Pf(Ω)` — Euler characteristic from curvature
- **Hirzebruch signature**: `σ(M) = ∫ L(TM)` — signature from Pontryagin classes
- **Riemann-Roch**: `dim H⁰(D) − dim H¹(D) = deg(D) + 1 − g` — for curves

### Analytic Index

```
ind_a(D) = dim ker(D) − dim ker(D*) = dim ker(D) − dim coker(D)
```

This is a **Fredholm index**: finite-dimensional for elliptic operators, stable under compact perturbations.

### Heat Kernel Asymptotics

```
K(t, x, x) ~ (4πt)^{−n/2} · (a₀ + a₁t + a₂t² + ...)
```

Where:
- `a₀ = 1`
- `a₁ = R/6 − E` (scalar curvature minus endomorphism)
- `a₂ = (5R² − 2|Ric|²)/360 − |E|²/12`

### Witten Index

```
Δ = tr((−1)^F e^{−tD²}) = Σₖ (−1)^k e^{−tλₖ}
```

Independent of t (heat kernel regularization). Equals the analytic index by McKean-Singer.

### Eta Invariant

```
η(s) = Σ sign(λᵢ)|λᵢ|^{−s}
```

Appears in the Atiyah-Patodi-Singer boundary correction term.

## Test Coverage

**93 tests** covering:
- Analytic index: kernel, cokernel, Fredholm, kernel projection, trace identity (16 tests)
- Topological index: Euler characteristic (sphere, torus, circle), Todd genus, Â genus, Chern character, Gauss-Bonnet, Betti numbers (18 tests)
- Witten index: Laplacian, heat kernel, supertrace, eta invariant, t-independence (14 tests)
- Heat kernel: coefficients, expansion, trace, supertrace, small-t asymptotics (14 tests)
- Characteristic: Euler number, Pontryagin, Stiefel-Whitney, Hirzebruch signature, Riemann-Roch, genus (21 tests)
- Policy bundle: Whitney sum, tensor product, Chern classes, Chern character (14 tests)
- Types: operators, bundles, index results (embedded in above)

## License

MIT
