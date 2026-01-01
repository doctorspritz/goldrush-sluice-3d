# Fluid Simulation Research Documentation

Research compiled on: 2025-12-20

## Table of Contents
1. [glam Crate - Vector Operations](#glam-crate---vector-operations)
2. [Rust Numerical Computing Patterns](#rust-numerical-computing-patterns)
3. [Rust Fluid Simulation Libraries](#rust-fluid-simulation-libraries)
4. [Iterative Solvers - Gauss-Seidel & Jacobi](#iterative-solvers---gauss-seidel--jacobi)
5. [Code Examples and Implementations](#code-examples-and-implementations)
6. [References](#references)

---

## 1. glam Crate - Vector Operations

### Summary
`glam` is a simple and fast 3D math library for games and graphics, with extensive SIMD support and a focus on performance without sacrificing API simplicity.

**Current Version**: 0.30.9
**Repository**: https://github.com/bitshifter/glam-rs
**Documentation**: https://docs.rs/glam/latest/glam/

### Key Concepts

#### Design Philosophy
- No generics and minimal traits in the public API for simplicity
- SIMD optimization through 128-bit vector types on supported platforms
- All external dependencies are optional
- Aims for 100% test coverage
- Outperforms similar Rust libraries for common operations (validated via mathbench project)

#### SIMD Support
- **SSE2**: Enabled by default on x86_64
- **NEON**: Default on aarch64 targets
- **SIMD128**: Available on wasm32
- **Portable SIMD**: Experimental via `core-simd` feature (nightly-only)

### Vec2 Methods Relevant to Collision Detection

#### Distance Calculations
```rust
// Euclidean distance between two points
fn distance(self, rhs: Self) -> f32

// Squared distance (faster, avoids sqrt)
fn distance_squared(self, rhs: Self) -> f32
```

#### Length Operations
```rust
// Length of vector
fn length(self) -> f32

// Squared length (faster than length())
fn length_squared(self) -> f32

// Reciprocal of length (1.0 / length())
fn length_recip(self) -> f32

// Check if vector is normalized (length ~= 1.0)
fn is_normalized(self) -> bool  // Precision threshold ~1e-4
```

#### Normalization
```rust
// Basic normalization (panics if zero/near-zero when glam_assert enabled)
fn normalize(self) -> Self

// Safe normalization - returns Option
fn try_normalize(self) -> Option<Self>

// Safe with fallback value
fn normalize_or(self, fallback: Self) -> Self

// Safe with zero fallback
fn normalize_or_zero(self) -> Self

// Returns both normalized vector and original length
fn normalize_and_length(self) -> (Self, f32)
```

#### Dot Product
```rust
// Standard dot product
fn dot(self, rhs: Self) -> f32

// Broadcast dot product to all components
fn dot_into_vec(self, rhs: Self) -> Self
```

#### Cross Product (2D - Perpendicular Operations)
```rust
// Perpendicular dot product (wedge product, 2D cross product, determinant)
fn perp_dot(self, rhs: Self) -> f32

// Returns Vec2 rotated by 90 degrees
fn perp(self) -> Self
```

#### Projection and Rejection
```rust
// Vector projection of self onto rhs (rhs must be non-zero length)
fn project_onto(self, rhs: Self) -> Self

// Vector projection (rhs must be normalized)
fn project_onto_normalized(self, rhs: Self) -> Self

// Vector rejection from rhs
fn reject_from(self, rhs: Self) -> Self

// Vector rejection (rhs must be normalized)
fn reject_from_normalized(self, rhs: Self) -> Self
```

#### Interpolation and Rotation
```rust
// Linear interpolation (midpoint equivalent: lerp(b, 0.5))
fn lerp(self, rhs: Self, t: f32) -> Self

// Midpoint between vectors (slightly faster than lerp(b, 0.5))
fn midpoint(self, rhs: Self) -> Self

// Angle between vectors
fn angle_between(self, rhs: Self) -> f32

// Rotate towards rhs up to max_angle (radians)
fn rotate_towards(self, rhs: Self, max_angle: f32) -> Self

// Create Vec2 from angle and rotate
// Example: Vec2::from_angle(PI).rotate(Vec2::Y) returns -Vec2::Y
fn rotate(self, angle: f32) -> Self
```

#### Reflection and Refraction
```rust
// Reflection vector for incident and surface normal
fn reflect(self, normal: Self) -> Self

// Refraction with index of refraction ratio
// Returns zero vector on total internal reflection
fn refract(self, normal: Self, eta: f32) -> Self
```

#### Fused Multiply-Add
```rust
// Computes (self * a) + b with only one rounding error
fn mul_add(self, a: Self, b: Self) -> Self
```

### Performance Notes
- Vec2 uses SIMD on supported platforms
- `length_squared()` is faster than `length()` (avoids sqrt)
- `distance_squared()` is faster than `distance()` (avoids sqrt)
- Precise lerp algorithm implemented in recent versions
- SIMD on wasm32 untested for performance

---

## 2. Rust Numerical Computing Patterns

### Overview of Rust Scientific Computing Libraries

#### nalgebra
- General-purpose linear algebra library
- Supports both statically-sized and dynamically-sized matrices
- Transformations and comprehensive matrix operations
- **Repository**: https://github.com/dimforge/nalgebra
- **Last updated**: February 2025

#### nalgebra-sparse
- Extends nalgebra with sparse matrix formats
- Supports CSR, CSC, and COO formats
- Designed for iterative solvers (matrix-vector products)

**Common Pattern**:
```rust
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use nalgebra::{DMatrix, DVector};

// 1. Construct matrix using COO format
// 2. Convert to CSR/CSC for solver
// 3. Implement iterative method using matrix-vector products
// 4. Use dense vectors for RHS and solution
```

**Typical Workflow**:
1. Use COO for construction
2. Convert to CSR/CSC for solver or matrix-vector products
3. For iterative solvers, CSR/CSC formats are suitable

**Current Limitations**:
- Limited availability of sparse system solvers
- Limited support for complex numbers
- Focus on correctness and API design over performance
- Performance enhancements planned incrementally

**Documentation**: https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/

#### ndarray & ndarray-linalg
- n-dimensional array data structure
- `ndarray-linalg` provides LAPACK-based linear algebra
- **Direct solvers** (LU decomposition) rather than iterative
- Factorize trait for solving multiple systems with same A matrix

**Installation Example**:
```toml
[dependencies]
ndarray = "0.14"
ndarray-linalg = { version = "0.13", features = ["openblas-static"] }
```

**Key Methods**:
- Solve trait for `A * x = b`
- Support for Hermitian/symmetric problems
- Matrix inversion, multiplication, decomposition

**Note**: Primarily provides direct solvers, not iterative methods.

**Documentation**: https://docs.rs/ndarray-linalg/latest/

#### sprs
- Sparse linear algebra library
- CsMat (compressed sparse matrix) and CsVec (sparse vector)
- Simple sparse Cholesky decomposition (LGPL license required)
- Sparse triangular solves with dense RHS

**Documentation**: https://docs.rs/sprs

#### matrixmultiply
- General matrix multiplication for f32 and f64
- Detects and uses AVX or SSE2 on x86 platforms
- Microkernel strategy for parallelization
- Multithreading support

#### scirust
- Pure Rust implementation (no C library integration)
- Takes advantage of Rust language features:
  - Generic programming
  - Immutable by default
  - Type traits
  - Iterators
- No planned integration with BLAS/LAPACK

**Documentation**: http://indigits.github.io/scirust/doc/scirust/index.html

### Iterative Solver Implementation Patterns

For implementing custom iterative solvers in Rust:

1. **Matrix Storage**: Use sparse formats (CSR/CSC) for large systems
2. **Vector Operations**: Use dense vectors from nalgebra or ndarray
3. **Matrix-Vector Product**: Core operation for iterative methods
4. **Convergence Checking**: Implement residual norm calculations
5. **Preconditioning**: Consider incomplete factorizations

---

## 3. Rust Fluid Simulation Libraries

### salva - Particle-Based (SPH) Fluid Simulation

**Repository**: https://github.com/dimforge/salva
**Documentation**: https://salva.rs
**API Docs**:
- 2D: https://docs.rs/salva2d
- 3D: https://docs.rs/salva3d

#### Features
- 2D and 3D fluid simulation with shared codebase
- Smoothed Particle Hydrodynamics (SPH) methods

#### Pressure Solvers
- **DFSPH** (Divergence-Free Smoothed Particle Hydrodynamics)
- **IISPH** (Implicit Incompressible SPH)

#### Viscosity Models
- DFSPH viscosity
- Artificial viscosity
- XSPH viscosity

#### Additional Physics
- Surface tension (WCSPH, He et al. 2014, Akinci et al. 2013)
- Elasticity (Becker et al. 2009)
- Multiphase fluids with varying densities and viscosities

#### Integration
- Uses nalgebra for vector/matrix math
- Optional integration with Rapier physics engine
- Two-way coupling with rigid bodies, multibodies, deformable bodies

**License**: Apache-2.0

### seanlth/Fluid-Solver - Grid-Based Eulerian Solver

**Repository**: https://github.com/seanlth/Fluid-Solver

#### Method
- **Chorin projection method** on staggered grid
- Decouples velocity and pressure in momentum equation

#### Grid Structure
- **Staggered (MAC) grid** to prevent checkerboarding
- Marker-particle visualization (glium, lodepng libraries)

#### Pressure Solve
- **Jacobi relaxation** iterative linear solver
- Implemented in three backends:
  - Pure Rust
  - C
  - OpenCL

#### Advection Options
- Upwind scheme
- Semi-Lagrangian scheme

#### Interpolation Methods
- Linear
- Cubic
- Catmull-Rom
- Hermite

#### Time Integration
- Euler
- Bogacki-Shampine
- Runge-Kutta 4

**Key Insight**: Modular design allows experimentation with different algorithmic combinations.

### mwalczyk/ponyo - Semi-Lagrangian Fluid Solver

**Repository**: https://github.com/mwalczyk/ponyo

#### Approach
- **Semi-Lagrangian** advection
- Based on Robert Bridson's "Fluid Simulation for Computer Graphics" (2nd Edition)

#### Grid Structure
- **Staggered marker-and-cell (MAC) grid**
- Second-order Runge-Kutta interpolation during backward particle trace

#### Pressure Solve
- **Gauss-Seidel method**

**Learning Resources**: Credits "Incremental Fluids" by Benedikt Bitterli for understanding projection method.

### msakuta/cfd-wasm - CFD in WebAssembly

**Repository**: https://github.com/msakuta/cfd-wasm

#### Features
- Computational Fluid Dynamics in WebAssembly with Rust
- "More stable solver" that won't diverge with extreme parameters

#### Gauss-Seidel Parameters
- **Diffusion iterations** (default: 4)
- **Projection iterations** (default: 20)

**Guidelines**:
- Increasing values improves accuracy but requires more computation
- Lower values acceptable for diffusion
- Projection better kept high for "nice swirly fluid behavior"

**Theoretical Foundations**: References Stam paper on stable fluid dynamics

### Wumpf/blub - 3D WebGPU Fluid Simulation

**Repository**: https://github.com/Wumpf/blub

#### Approach
- 3D experiments using WebGPU-rs
- Hybrid Lagrangian/Eulerian (PIC/FLIP/APIC)

#### Pressure Solving
- **Secondary pressure solver** using fluid density instead of divergence
- Improves simulation quality for large timesteps
- Typically runs simulation/solver at 120Hz

### gabyx/RsFluid - Eulerian Learning Project

**Repository**: https://github.com/gabyx/RsFluid

#### Purpose
- Eulerian fluid simulation written to learn Rust

#### Parallel Implementation
- `solve_incompressibility` splits `grid.cells` into parts
- Uses iterator chains ending with `PosStencilMut<Cell>` stencils

### AudranDoublet/opr - State-of-the-art SPH

**Repository**: https://github.com/AudranDoublet/opr

#### Features
- State-of-the-art (2020) SPH implementation
- **Divergence Free SPH (DFSPH)** pressure solver
- Designed for easy implementation of additional solvers

### Other Notable Projects

- **kugi83/fluid_simulation**: Simple 2D fluid simulation with Rust and Bevy
- **miguelggcc/CFD-SIMPLE-Rust**: CFD solver using SIMPLE algorithm with collocated grid
- **AkinAguda/fluid-simulation-rust**: Rewrite of fluid simulation entirely in Rust

---

## 4. Iterative Solvers - Gauss-Seidel & Jacobi

### Gauss-Seidel Method

#### Overview
- Classical iterative method for solving linear systems `Ax = b`
- Uses previously computed components within same iteration
- Typically faster convergence than Jacobi method
- Also known as Liebmann method or method of successive displacement

**Wikipedia**: https://en.wikipedia.org/wiki/Gauss–Seidel_method

#### Key Characteristics
- **Memory Efficient**: Only one storage vector required (elements overwritten)
- **Sequential**: Hard to parallelize (each variable depends on previously updated values)
- **Convergence**: Guaranteed for strictly diagonally dominant matrices
- **Performance**: Can reduce iterations by factor of 2 compared to Jacobi

#### Algorithm Pattern
During iteration, use most recently calculated values when available (unlike Jacobi which always uses previous iteration values).

#### Parallelization Challenges
- Updates are sequential in nature
- Each variable depends on previously updated variables from same iteration
- Restricts straightforward parallelization

#### Solutions for Parallelization
- **Red-Black Coloring**: Color grid such that no red element depends on black (and vice versa)
- **Multi-Colored Gauss-Seidel (MCGS)**: For unstructured grids, GPU implementations

### Jacobi Method

#### Overview
- Intuitive iterative solution method
- Values at grid points replaced by weighted averages
- Easier to parallelize than Gauss-Seidel
- Typically slower convergence than Gauss-Seidel

#### Key Characteristics
- **Parallelizable**: All updates use values from previous iteration
- **Simpler**: Straightforward implementation
- **Convergence**: Guaranteed for strictly diagonally dominant matrices
- **Performance**: Typically requires more iterations than Gauss-Seidel

### Comparison: Jacobi vs. Gauss-Seidel

| Aspect | Jacobi | Gauss-Seidel |
|--------|--------|--------------|
| Memory | Two vectors required | One vector (in-place) |
| Parallelization | Easy | Difficult |
| Convergence Rate | Slower | Faster (~2x) |
| Implementation | Simpler | Slightly more complex |
| GPU Suitability | Better | Requires coloring schemes |

### Alternative Methods

#### Preconditioned Conjugate Gradient (PCG)
- Recommended by Robert Bridson for faster convergence
- Modified Incomplete Cholesky (MIC) level-zero preconditioning
- "Significantly faster convergence rates" than Jacobi/Gauss-Seidel
- Public domain implementation available from Bridson

**Reference**: Bridson's "Fluid Simulation for Computer Graphics" [Bri15]

#### Other Iterative Methods
- **GMRES**: Generalized Minimal Residual
- **BiCGSTAB**: Biconjugate Gradient Stabilized
- **Multigrid Methods**: Geometric and algebraic

### Recent Research (2025)

**GPU-Based Gauss-Seidel for CFD**:
- Stephen Thomas (Lehigh University) and Pasqua D'Ambra
- Modified Forward Gauss-Seidel for communication-avoiding Krylov techniques
- Matches accuracy of traditional orthogonalization
- Scales to 64 GPUs with 700M+ unknowns
- Typically 20-30 FGS sweeps for convergence
- Validated on AMD MI-series GPUs

**Source**: https://quantumzeitgeist.com/700-inexact-gauss-seidel-coarse-solvers-preserve-scalability-gpus/

### Application to Fluid Simulation

#### Pressure Projection
Gauss-Seidel commonly used for solving pressure Poisson equation:
```
∇²p = ∇·u*
```
where `u*` is intermediate velocity field.

#### Grid-Based Implementation
For 2D grid with spacing `h`:
```
p[i,j] = (p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - h²·b[i,j]) / 4
```

#### Staggered Grid Considerations
- Pressure at cell centers
- Velocities at cell faces
- Prevents checkerboard patterns
- Requires careful index handling

---

## 5. Code Examples and Implementations

### Grid-Based Fluid Simulation Pattern

Based on seanlth/Fluid-Solver and other implementations:

```rust
// Typical structure for grid-based solver

struct FluidGrid {
    width: usize,
    height: usize,
    // Staggered grid velocities
    u: Vec<f32>,  // horizontal velocity (at vertical faces)
    v: Vec<f32>,  // vertical velocity (at horizontal faces)
    // Cell-centered pressure
    p: Vec<f32>,
    // Divergence for pressure solve
    div: Vec<f32>,
}

impl FluidGrid {
    // Pressure projection step with Jacobi/Gauss-Seidel
    fn solve_pressure(&mut self, iterations: usize) {
        for _ in 0..iterations {
            for j in 1..self.height-1 {
                for i in 1..self.width-1 {
                    let idx = i + j * self.width;
                    // Gauss-Seidel uses updated values immediately
                    self.p[idx] = (
                        self.p[idx-1] + self.p[idx+1] +
                        self.p[idx-self.width] + self.p[idx+self.width] -
                        self.div[idx]
                    ) * 0.25;
                }
            }
        }
    }

    // Apply pressure gradient to make velocity divergence-free
    fn apply_pressure_gradient(&mut self) {
        for j in 1..self.height-1 {
            for i in 1..self.width-1 {
                let idx = i + j * self.width;
                self.u[idx] -= (self.p[idx+1] - self.p[idx]);
                self.v[idx] -= (self.p[idx+self.width] - self.p[idx]);
            }
        }
    }
}
```

### Using glam for Collision Detection

```rust
use glam::Vec2;

// Circle-circle collision
fn circles_colliding(pos1: Vec2, r1: f32, pos2: Vec2, r2: f32) -> bool {
    let dist_sq = pos1.distance_squared(pos2);
    let radii_sum = r1 + r2;
    dist_sq < radii_sum * radii_sum
}

// Point-in-circle test
fn point_in_circle(point: Vec2, center: Vec2, radius: f32) -> bool {
    point.distance_squared(center) < radius * radius
}

// Closest point on line segment
fn closest_point_on_segment(point: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let ap = point - a;
    let t = ap.dot(ab) / ab.length_squared();
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

// Reflect velocity off surface
fn reflect_velocity(velocity: Vec2, normal: Vec2) -> Vec2 {
    velocity.reflect(normal.normalize())
}

// Separate overlapping circles
fn separate_circles(pos1: Vec2, r1: f32, pos2: Vec2, r2: f32) -> (Vec2, Vec2) {
    let delta = pos2 - pos1;
    let dist = delta.length();
    if dist < r1 + r2 {
        let normal = delta.normalize_or_zero();
        let separation = (r1 + r2 - dist) * 0.5;
        let offset = normal * separation;
        (pos1 - offset, pos2 + offset)
    } else {
        (pos1, pos2)
    }
}
```

### Sparse Matrix with nalgebra-sparse

```rust
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use nalgebra::DVector;

// Example: Building and solving with sparse matrix
fn example_sparse_system() {
    // 1. Build matrix in COO format
    let mut coo = CooMatrix::<f64>::new(100, 100);

    // Add entries (row, col, value)
    for i in 0..100 {
        coo.push(i, i, 4.0);  // Diagonal
        if i > 0 {
            coo.push(i, i-1, -1.0);  // Sub-diagonal
        }
        if i < 99 {
            coo.push(i, i+1, -1.0);  // Super-diagonal
        }
    }

    // 2. Convert to CSR for efficient matrix-vector products
    let csr = CsrMatrix::from(&coo);

    // 3. Right-hand side
    let b = DVector::from_element(100, 1.0);

    // 4. Initial guess
    let mut x = DVector::zeros(100);

    // 5. Implement iterative solver (e.g., Gauss-Seidel)
    // (Manual implementation required - not built into nalgebra-sparse)
}
```

### Red-Black Gauss-Seidel for Parallelization

```rust
// Parallel Gauss-Seidel using red-black coloring
fn solve_pressure_parallel(&mut self, iterations: usize) {
    for _ in 0..iterations {
        // Red cells (i+j even)
        for j in 1..self.height-1 {
            for i in 1..self.width-1 {
                if (i + j) % 2 == 0 {
                    let idx = i + j * self.width;
                    self.p[idx] = (
                        self.p[idx-1] + self.p[idx+1] +
                        self.p[idx-self.width] + self.p[idx+self.width] -
                        self.div[idx]
                    ) * 0.25;
                }
            }
        }

        // Black cells (i+j odd)
        for j in 1..self.height-1 {
            for i in 1..self.width-1 {
                if (i + j) % 2 == 1 {
                    let idx = i + j * self.width;
                    self.p[idx] = (
                        self.p[idx-1] + self.p[idx+1] +
                        self.p[idx-self.width] + self.p[idx+self.width] -
                        self.div[idx]
                    ) * 0.25;
                }
            }
        }
    }
}
```

---

## 6. References

### Documentation & Libraries

#### glam
- Official Docs: https://docs.rs/glam/latest/glam/
- GitHub: https://github.com/bitshifter/glam-rs
- Changelog: https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md
- Vec2 API: https://docs.rs/glam/latest/glam/f32/struct.Vec2.html
- Crates.io: https://crates.io/crates/glam/

#### Rust Linear Algebra Libraries
- nalgebra: https://github.com/dimforge/nalgebra
- nalgebra-sparse: https://docs.rs/nalgebra-sparse/latest/
- ndarray-linalg: https://docs.rs/ndarray-linalg/latest/
- sprs: https://docs.rs/sprs
- scirust: http://indigits.github.io/scirust/doc/scirust/index.html

#### Scientific Computing in Rust
- Are We Learning Yet: https://www.arewelearningyet.com/scientific-computing/
- Rust Internals Discussion: https://internals.rust-lang.org/t/rust-and-numeric-computation/20425

### Fluid Simulation Projects

#### Grid-Based Solvers
- seanlth/Fluid-Solver: https://github.com/seanlth/Fluid-Solver
- mwalczyk/ponyo: https://github.com/mwalczyk/ponyo
- msakuta/cfd-wasm: https://github.com/msakuta/cfd-wasm
- gabyx/RsFluid: https://github.com/gabyx/RsFluid
- Wumpf/blub: https://github.com/Wumpf/blub
- miguelggcc/CFD-SIMPLE-Rust: https://github.com/miguelggcc/CFD-SIMPLE-Rust

#### Particle-Based Solvers
- dimforge/salva: https://github.com/dimforge/salva
  - Documentation: https://salva.rs
  - 2D API: https://docs.rs/salva2d
  - 3D API: https://docs.rs/salva3d
- AudranDoublet/opr: https://github.com/AudranDoublet/opr

#### Other Projects
- kugi83/fluid_simulation: https://github.com/kugi83/fluid_simulation
- AkinAguda/fluid-simulation-rust: https://github.com/AkinAguda/fluid-simulation-rust

#### GitHub Topics
- Fluid Simulation: https://github.com/topics/fluid-simulation?l=rust&o=desc&s=stars
- Fluid Solver: https://github.com/topics/fluid-solver

### Collision Detection with glam
- bam3d: https://github.com/DallasC/bam3d
- impacted: https://github.com/jcornaz/impacted
- Barry: https://github.com/Jondolf/barry
- glamour: https://github.com/simonask/glamour

### Iterative Solvers

#### General Resources
- Gauss-Seidel Method (Wikipedia): https://en.wikipedia.org/wiki/Gauss–Seidel_method
- Iterative Solvers (ACME Lab): https://acme.byu.edu/00000180-6957-d2d1-ade4-6b77210d0001/iterative-solvers
- Jacobi and Gauss-Seidel Explanation: https://erkaman.github.io/posts/jacobi_and_gauss_seidel.html
- GPU Gauss-Seidel Research: https://quantumzeitgeist.com/700-inexact-gauss-seidel-coarse-solvers-preserve-scalability-gpus/

#### GitHub Implementations
- Gauss-Seidel Topic: https://github.com/topics/gauss-seidel
- nuhferjc/gauss-seidel: https://github.com/nuhferjc/gauss-seidel
- Parallel Implementation: https://github.com/rpandya1990/Gauss-seidel-Parallel-Implementation

### Books & Academic Resources

#### Robert Bridson - Fluid Simulation for Computer Graphics
- Publisher Page: https://www.routledge.com/Fluid-Simulation-for-Computer-Graphics/Bridson/p/book/9781482232837
- Author's Page: https://www.cs.ubc.ca/~rbridson/fluidsimulation/
- Google Books: https://books.google.com/books/about/Fluid_Simulation_for_Computer_Graphics.html?id=1-LqBgAAQBAJ
- Key Topics: Pressure projection, PCG solver, MICCG(0) preconditioning

#### Other Academic Resources
- Staggered Grid Wiki: https://www.cfd-online.com/Wiki/Staggered_grid
- MAC Grid Paper: Application of Projection Method and Staggered Grid to Incompressible Navier-Stokes
- Iteration Methods: https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html

### Additional Resources

#### Blog Posts & Tutorials
- Introducing glam and mathbench: https://bitshifter.github.io/2019/07/10/introducing-glam-and-mathbench/
- Grids in Rust: https://blog.adamchalmers.com/grids-1/
- Solving Sparse Matrix Systems in Rust: https://medium.com/software-makes-hardware/solving-sparse-matrix-systems-in-rust-5e978ed07bc3
- Linear Algebra for Rust: https://numerical-web.net/wordpress/?p=111

#### Rust Cookbook
- Linear Algebra: https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html

---

## Key Takeaways for Implementation

### For Vector Operations (glam)
1. Use `distance_squared()` for comparisons (avoid sqrt when possible)
2. Use `normalize_or_zero()` for safe normalization in collision handling
3. Use `perp_dot()` for 2D cross product calculations
4. Take advantage of SIMD optimizations (enabled by default on most platforms)
5. Use `reflect()` for bouncing particles off surfaces

### For Iterative Solvers
1. Gauss-Seidel converges ~2x faster than Jacobi but harder to parallelize
2. Use red-black coloring for parallel Gauss-Seidel on grids
3. Consider PCG with incomplete Cholesky preconditioning for better convergence
4. 20-30 iterations often sufficient for visual quality in real-time simulations
5. Higher iteration counts for projection yield better incompressibility

### For Fluid Simulation
1. Staggered (MAC) grids prevent pressure-velocity checkerboarding
2. Chorin projection method decouples velocity and pressure calculation
3. Semi-Lagrangian advection provides stability for large timesteps
4. nalgebra-sparse good for building custom solvers with sparse matrices
5. salva library offers production-ready SPH if particle-based approach preferred

### For Collision Detection
1. Spatial partitioning essential for many-particle systems
2. Use `length_squared()` for distance comparisons
3. Broad phase (grid/tree) + narrow phase (exact tests) pattern
4. Consider using existing collision libraries (Barry, impacted) if complex shapes needed
5. glam's simple API makes it easy to implement custom collision logic

---

**Research compiled by**: Claude Code (Sonnet 4.5)
**Date**: December 20, 2025
**Purpose**: Documentation research for Rust fluid simulation implementation with focus on glam vector operations, numerical computing patterns, and Gauss-Seidel solvers.
