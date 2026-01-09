# Phase 0: Scaffolding Instructions

This phase creates the module structure so parallel agents can work without conflicts.

---

## Pre-Flight Checklist

```bash
# 1. Ensure clean working directory
git status  # Should show no uncommitted changes or handle them

# 2. Create backup branch
git checkout main
git pull origin main
git checkout -b backup/pre-module-split-2025-12-29
git push origin backup/pre-module-split-2025-12-29
git checkout main

# 3. CRITICAL: Verify regression tests pass BEFORE any changes
cargo test -p sim --test refactor_regression

# If regression tests fail, DO NOT PROCEED
# Fix any issues or regenerate golden values first

# 4. Verify full test suite passes
cargo test -p sim
```

### Regression Test Baseline

The regression test suite (`refactor_regression.rs`) captures golden values that must remain unchanged throughout the refactor:

| Test | What It Guards | Critical For |
|------|----------------|--------------|
| `regression_p2g_transfer` | P2G velocity transfer | transfer.rs |
| `regression_pressure_solve` | Pressure solver convergence | pressure.rs |
| `regression_flip_cycle` | Full P2G→G2P roundtrip | transfer.rs |
| `regression_extrapolation` | Velocity extrapolation | extrapolation.rs |
| `regression_vorticity` | Vorticity computation | vorticity.rs |
| `regression_sdf_computation` | SDF distance field | sdf.rs |
| `regression_interpolation_kernels` | B-spline weights sum to 1.0 | interp.rs |
| `regression_full_step` | Complete update() pipeline | ALL |
| `regression_sediment_settling` | Sediment deposition | sediment.rs |
| `regression_stability_100_steps` | No NaN/Inf over time | ALL |

**Every agent must run `cargo test -p sim --test refactor_regression` after each change.**

---

## Step 1: Create grid/ Module Directory

```bash
# Create directory
mkdir -p crates/sim/src/grid

# Move grid.rs to grid/mod.rs
mv crates/sim/src/grid.rs crates/sim/src/grid/mod.rs
```

## Step 2: Create Empty grid/ Submodule Files

Create these files with minimal content:

### grid/interp.rs
```rust
//! Interpolation helpers for APIC/FLIP transfer.

use glam::Vec2;

// Move these functions from grid/mod.rs:
// - quadratic_bspline_1d
// - quadratic_bspline
// - apic_d_inverse
```

### grid/cell_types.rs
```rust
//! Cell type definitions and accessors.

use super::Grid;
use crate::particle::ParticleMaterial;

// Move these from grid/mod.rs:
// - CellType enum
// - DepositedCell struct
// - MultigridLevel struct
// - Cell index methods
```

### grid/sdf.rs
```rust
//! Signed Distance Field computation and sampling.

use super::Grid;
use glam::Vec2;

// Move these from grid/mod.rs:
// - compute_sdf
// - sample_sdf
// - sdf_gradient
// - bed height methods
```

### grid/velocity.rs
```rust
//! Velocity field operations.

use super::Grid;
use glam::Vec2;

// Move these from grid/mod.rs:
// - sample_velocity
// - sample_velocity_bspline
// - apply_gravity
// - apply_viscosity
// - enforce_boundary_conditions
```

### grid/pressure.rs
```rust
//! Pressure projection and multigrid solver.

use super::Grid;

// Move these from grid/mod.rs:
// - compute_divergence
// - solve_pressure
// - solve_pressure_multigrid
// - apply_pressure_gradient
// - All mg_* methods
```

### grid/extrapolation.rs
```rust
//! Velocity extrapolation into air cells.

use super::Grid;

// Move these from grid/mod.rs:
// - extrapolate_velocities
// - mark_fluid_faces_known
// - extrapolate_u_layer
// - extrapolate_v_layer
```

### grid/vorticity.rs
```rust
//! Vorticity computation and confinement.

use super::Grid;

// Move these from grid/mod.rs:
// - compute_vorticity
// - compute_enstrophy
// - apply_vorticity_confinement
// - apply_vorticity_confinement_with_piles
```

## Step 3: Update grid/mod.rs

Add module declarations at the top of grid/mod.rs:

```rust
//! MAC grid for fluid simulation.

// Submodules
mod cell_types;
mod extrapolation;
mod interp;
mod pressure;
mod sdf;
mod velocity;
mod vorticity;

// Re-exports for backwards compatibility
pub use cell_types::{CellType, DepositedCell, MultigridLevel};
pub use interp::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d};

// ... rest of existing code ...
```

---

## Step 4: Create flip/ Module Directory

```bash
# Create directory
mkdir -p crates/sim/src/flip

# Move flip.rs to flip/mod.rs
mv crates/sim/src/flip.rs crates/sim/src/flip/mod.rs
```

## Step 5: Create Empty flip/ Submodule Files

### flip/spawning.rs
```rust
//! Particle spawning methods.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - is_spawn_safe
// - spawn_water
// - spawn_sand
// - spawn_magnetite
// - spawn_gold
```

### flip/diagnostics.rs
```rust
//! Profiling and diagnostic methods.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - update_profiled
// - update_with_diagnostics
// - run_isolated_flip_cycle
// - compute_kinetic_energy
// - compute_cfl
// - initialize_taylor_green
// - initialize_solid_rotation
```

### flip/transfer.rs
```rust
//! Particle-grid transfer (P2G, G2P).

use super::FlipSimulation;
use crate::grid::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d};

// Move these from flip/mod.rs:
// - particles_to_grid
// - store_old_velocities
// - grid_to_particles
```

### flip/advection.rs
```rust
//! Particle advection and spatial hashing.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - advect_particles
// - build_spatial_hash
// - compute_neighbor_counts
// - push_particles_apart
```

### flip/sediment.rs
```rust
//! Sediment transport physics.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - apply_sediment_forces
// - apply_dem_settling
// - deposit_settled_sediment
// - entrain_deposited_sediment
// - collapse_deposited_sediment
```

### flip/pile.rs
```rust
//! Pile heightfield and particle state management.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - update_particle_states
// - compute_pile_heightfield
// - enforce_pile_constraints
```

### flip/pressure.rs
```rust
//! Two-way pressure coupling.

use super::FlipSimulation;

// Move these from flip/mod.rs:
// - apply_pressure_gradient_two_way
// - apply_porosity_drag
```

## Step 6: Update flip/mod.rs

Add module declarations at the top of flip/mod.rs:

```rust
//! APIC fluid simulation with sediment transport.

// Submodules
mod advection;
mod diagnostics;
mod pile;
mod pressure;
mod sediment;
mod spawning;
mod transfer;

// ... rest of existing code ...
```

---

## Step 7: Verify Compilation

```bash
# Build
cargo build -p sim

# Should compile with no errors
# Warnings about unused modules are OK at this stage

# Run tests
cargo test -p sim
```

---

## Step 8: Commit Scaffolding

```bash
git add crates/sim/src/grid/ crates/sim/src/flip/
git commit -m "refactor(sim): scaffold grid/ and flip/ module directories

Create module structure for parallel refactoring:
- grid/mod.rs with empty submodule declarations
- flip/mod.rs with empty submodule declarations
- Empty stub files for each planned submodule

No behavior changes. Methods will be moved in subsequent PRs."
```

---

## Step 9: Create Worktrees for Agents

```bash
# From repository root
cd ..

# Create worktrees for each agent
git worktree add goldrush-refactor-G1 -b refactor/grid-cell-types-G1
git worktree add goldrush-refactor-G2 -b refactor/grid-velocity-G2
git worktree add goldrush-refactor-G3 -b refactor/grid-pressure-G3
git worktree add goldrush-refactor-F1 -b refactor/flip-spawning-F1
git worktree add goldrush-refactor-F2 -b refactor/flip-diagnostics-F2
git worktree add goldrush-refactor-F3 -b refactor/flip-advection-F3
git worktree add goldrush-refactor-F4 -b refactor/flip-transfer-F4
git worktree add goldrush-refactor-F5 -b refactor/flip-sediment-F5
```

---

## Completion Criteria

- [ ] `crates/sim/src/grid/mod.rs` exists (moved from grid.rs)
- [ ] `crates/sim/src/flip/mod.rs` exists (moved from flip.rs)
- [ ] All grid submodule files created (7 files)
- [ ] All flip submodule files created (7 files)
- [ ] Module declarations added to both mod.rs files
- [ ] `cargo build -p sim` passes
- [ ] `cargo test -p sim` passes
- [ ] Scaffolding committed
- [ ] Backup tag created: `git tag refactor-phase0-complete`
- [ ] Worktrees created for agents

---

## Next Steps

After scaffolding is complete:

1. Update STATUS.md to mark Phase 0 complete
2. Signal to agents that Phase 1 can begin
3. Agents claim their modules in STATUS.md
4. Each agent works in their worktree following PARALLEL_PROTOCOL.md
